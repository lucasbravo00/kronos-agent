"""Shared fakes and synthetic data for the step-2 tests (a helper module, not a test module).

Imported as ``import step2_fakes``: pytest puts ``tests/`` on ``sys.path`` and never collects this
file (its name does not match ``test_*.py``). Nothing here touches the network or the real data
root. Three families:

1. :class:`FakeBingX`, a handler for ``httpx.MockTransport`` with the BingX real-market behaviour
   verified live (CLAUDE.md, "Exchange facts"): ``/openApi/swap/v2/server/time`` and
   ``/openApi/swap/v3/quote/klines`` for any interval; kline objects with ``open, close, high,
   low, volume`` strings and an integer ``time``, NEWEST FIRST; a candle is served when
   ``startTime <= open < endTime`` (``startTime`` inclusive at an exact boundary; an ``endTime``
   exactly at a candle's open excludes that candle while ``boundary - 1`` returns it, D-041) and
   its open is at or before the server time, so the current unclosed candle comes first once it
   opened and candles opening after the server time are never returned (tests may supply them);
   ``startTime`` without ``endTime`` gives an empty list, as does a range without candles;
   ``limit`` (capped at 1000) keeps the NEWEST candles of the range. The server time is a
   mutable attribute (optionally advancing by ``tick_ms`` per request), gaps and a history start
   can be declared per interval, a ``mutator`` can change what one call returns (to make two
   fetches differ), and every request is logged. Use it through a real
   :class:`~kronos_agent.exchange.public_market.PublicMarketClient` (:func:`bingx_client`).
2. :class:`FakeBinanceVision`, a handler serving in-memory kline zips and their ``.CHECKSUM``
   files built by :func:`kline_csv`, :func:`make_zip` and :func:`checksum_text` (header line or
   none, millisecond or microsecond times), ``404`` for any other path, with a request log.
3. Synthetic candles with exchange-like decimal strings: :func:`synthetic_candles` (BingX-like
   :class:`Candle` lists) and :func:`binance_frame` (``BINANCE_COLUMNS`` string frames with the
   mixed price formats of real 2020 archives), conversions between the two forms, and store
   writers that go through :mod:`kronos_agent.data.store` under a temporary root.

Host names: the BingX host is never written here (``pm.PUBLIC_HOST`` is used); the Binance Vision
host is public and named in :data:`BINANCE_HOST`.
"""

from __future__ import annotations

import calendar
import hashlib
import io
import zipfile
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from kronos_agent.data.store import BASE_COLUMNS, BINANCE_COLUMNS, StoreId, write_store
from kronos_agent.data.timeutil import to_ms
from kronos_agent.exchange import public_market as pm
from kronos_agent.exchange.public_market import Candle, PublicMarketClient

HOUR_MS = 3_600_000
MINUTE_MS = 60_000
MAX_KLINES_LIMIT = 1000
TEST_PROVENANCE: dict[str, Any] = {"kind": "synthetic test candles"}


def utc_ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """Epoch milliseconds of a UTC wall-clock time."""
    return to_ms(datetime(year, month, day, hour, minute, tzinfo=UTC))


class FakeClock:
    """Monotonic clock for ``PublicMarketClient``: advances only through ``sleep`` (no waiting)."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# -- synthetic candles -----------------------------------------------------------------------------


def _fixed(value: int, decimals: int) -> str:
    """Plain decimal text of the non-negative integer ``value`` scaled by ``10**-decimals``."""
    if value < 0:
        raise ValueError("synthetic values are non-negative")
    if decimals == 0:
        return str(value)
    scale = 10**decimals
    return f"{value // scale}.{value % scale:0{decimals}d}"


def synthetic_candles(
    start_ms: int,
    count: int,
    *,
    interval: str = "1h",
    seed: int = 0,
    start_price: float = 60_000.0,
) -> list[Candle]:
    """``count`` BingX-like candles opening at ``start_ms, start_ms + step, ...``.

    Prices are plain decimals with one decimal (``"60123.4"``), volumes with four
    (``"339.1763"``); ``high >= max(open, close)``, ``low <= min(open, close)``, every price
    positive (a log-normal walk). The same arguments always give the same candles; tests that
    need consistent overlapping ranges generate once and slice.
    """
    step = pm.interval_ms(interval)
    if start_ms % step:
        raise ValueError(f"start_ms must be aligned to the {interval} step")
    if count < 0:
        raise ValueError("count must be >= 0")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.002, size=count)
    close = np.rint(start_price * 10.0 * np.exp(np.cumsum(returns))).astype(np.int64)
    open_ = np.concatenate(([round(start_price * 10.0)], close[:-1]))
    up = np.rint(np.abs(rng.normal(0.0, 0.001, size=count)) * start_price * 10.0).astype(np.int64)
    down = np.rint(np.abs(rng.normal(0.0, 0.001, size=count)) * start_price * 10.0).astype(np.int64)
    high = np.maximum(open_, close) + up
    low = np.maximum(np.minimum(open_, close) - down, 1)
    volume = rng.integers(1, 5_000_000, size=count)  # ten-thousandths: 0.0001 .. 499.9999
    return [
        Candle(
            start_ms + index * step,
            _fixed(int(open_[index]), 1),
            _fixed(int(high[index]), 1),
            _fixed(int(low[index]), 1),
            _fixed(int(close[index]), 1),
            _fixed(int(volume[index]), 4),
        )
        for index in range(count)
    ]


def _price_text(cents: int, whole: bool) -> str:
    return str(cents // 100) if whole else _fixed(cents, 2)


def binance_frame(
    start_ms: int,
    count: int,
    *,
    interval: str = "1m",
    seed: int = 0,
    start_price: float = 7_200.0,
    whole_dollar_every: int = 5,
    zero_volume_every: int = 97,
) -> pd.DataFrame:
    """A ``BINANCE_COLUMNS`` string frame of ``count`` Binance-like candles from ``start_ms``.

    Prices have two decimals with trailing zeros kept (``"7171.10"``), and every
    ``whole_dollar_every``-th row is in whole dollars without a decimal point (``"7225"``), as in
    real 2020 archives (high rounded up, low down, so the OHLC relations still hold). Volumes
    have three decimals; every ``zero_volume_every``-th row (offset 3) has zero volume and zero
    trades. Quote volumes are exact ``volume x mean(o, h, l, c)`` rounded to five decimals;
    taker-buy volumes are a fraction of the volume. ``open_ms`` is int64, everything else ``str``.
    """
    step = pm.interval_ms(interval)
    if start_ms % step:
        raise ValueError(f"start_ms must be aligned to the {interval} step")
    if count < 0:
        raise ValueError("count must be >= 0")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.0015, size=count)
    close = np.rint(start_price * 100.0 * np.exp(np.cumsum(returns))).astype(np.int64)
    open_ = np.concatenate(([round(start_price * 100.0)], close[:-1]))
    up = np.rint(np.abs(rng.normal(0.0, 0.0008, size=count)) * start_price * 100.0).astype(np.int64)
    down = np.rint(np.abs(rng.normal(0.0, 0.0008, size=count)) * start_price * 100.0)
    high = np.maximum(open_, close) + up
    low = np.maximum(np.minimum(open_, close) - down.astype(np.int64), 100)
    volume = rng.integers(0, 2_000_000, size=count)  # thousandths: 0.000 .. 1999.999
    trades = rng.integers(1, 3_000, size=count)
    taker_share = rng.random(size=count)
    columns: dict[str, list[Any]] = {name: [] for name in BINANCE_COLUMNS}
    quantum = Decimal("0.00001")
    for index in range(count):
        o, h, lo, c = int(open_[index]), int(high[index]), int(low[index]), int(close[index])
        whole = whole_dollar_every > 0 and index % whole_dollar_every == 0
        if whole:
            o, c = 100 * round(o / 100), 100 * round(c / 100)
            h, lo = 100 * -(-h // 100), max(100, 100 * (lo // 100))
        prices = [_price_text(value, whole) for value in (o, h, lo, c)]
        vol = int(volume[index])
        count_ = int(trades[index])
        if zero_volume_every > 0 and index % zero_volume_every == 3:
            vol, count_ = 0, 0
        taker = int(vol * taker_share[index])
        mean_price = sum(Decimal(text) for text in prices) / 4
        volume_text, taker_text = _fixed(vol, 3), _fixed(taker, 3)
        quote = (Decimal(volume_text) * mean_price).quantize(quantum, rounding=ROUND_HALF_EVEN)
        taker_quote = (Decimal(taker_text) * mean_price).quantize(quantum, rounding=ROUND_HALF_EVEN)
        row = [
            start_ms + index * step,
            *prices,
            volume_text,
            format(quote, "f"),
            str(count_),
            taker_text,
            format(taker_quote, "f"),
        ]
        for name, value in zip(BINANCE_COLUMNS, row, strict=True):
            columns[name].append(value)
    frame = pd.DataFrame({name: pd.Series(columns[name], dtype=object) for name in BINANCE_COLUMNS})
    frame["open_ms"] = frame["open_ms"].astype(np.int64)
    return frame


def candles_to_frame(candles: Sequence[Candle]) -> pd.DataFrame:
    """``BASE_COLUMNS`` string frame (``open_ms`` int64, values ``str``) of ``candles``."""
    data: dict[str, Any] = {"open_ms": np.array([c.open_ms for c in candles], dtype=np.int64)}
    for name in BASE_COLUMNS[1:]:
        data[name] = pd.Series([getattr(c, name) for c in candles], dtype=object)
    return pd.DataFrame(data)


def frame_to_candles(frame: pd.DataFrame) -> list[Candle]:
    """Candles of a string frame with at least ``BASE_COLUMNS`` (BingX or Binance)."""
    return [
        Candle(int(open_ms), o, h, lo, c, v)
        for open_ms, o, h, lo, c, v in zip(*(frame[name] for name in BASE_COLUMNS), strict=True)
    ]


# -- stores under a temporary root -----------------------------------------------------------------


def write_bingx_store(
    root: Path,
    candles: Sequence[Candle],
    *,
    interval: str = "1h",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``candles/bingx/<interval>`` under ``root`` through ``data.store``; return its meta."""
    return write_store(
        StoreId("bingx", interval),
        candles_to_frame(candles),
        provenance=dict(provenance or TEST_PROVENANCE),
        root=root,
    )


def write_binance_store(
    root: Path,
    frame: pd.DataFrame,
    *,
    source: str = "binance",
    interval: str = "1h",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``candles/<source>/<interval>`` from a ``BINANCE_COLUMNS`` string frame."""
    return write_store(
        StoreId(source, interval),
        frame,
        provenance=dict(provenance or TEST_PROVENANCE),
        root=root,
    )


# -- BingX real-market fake ------------------------------------------------------------------------


def ok(data: Any) -> httpx.Response:
    """A BingX success envelope."""
    return httpx.Response(200, json={"code": 0, "msg": "", "data": data})


def api_error(code: int, msg: str = "error") -> httpx.Response:
    """A BingX error envelope (``code != 0`` is an error, never "no data")."""
    return httpx.Response(200, json={"code": code, "msg": msg})


def kline_object(candle: Candle) -> dict[str, Any]:
    """The verified klines v3 item: six keys, string values, integer ``time``."""
    return {
        "open": candle.open,
        "close": candle.close,
        "high": candle.high,
        "low": candle.low,
        "volume": candle.volume,
        "time": candle.open_ms,
    }


@dataclass(frozen=True)
class KlinesCall:
    """One klines request answered by :class:`FakeBingX` (``index`` counts from 0)."""

    index: int
    interval: str
    symbol: str
    start_ms: int | None
    end_ms: int | None
    limit: int
    server_time_ms: int


Mutator = Callable[[KlinesCall, list[Candle]], Sequence[Candle]]


class FakeBingX:
    """BingX real-market handler for ``httpx.MockTransport`` (see the module docstring).

    ``candles`` maps an interval to the candles the exchange holds (future ones allowed: they
    are served only once the server time reaches their open). ``history_start_ms`` and
    ``missing`` (per interval) hide candles before a start and individual opens. ``mutator``
    receives the :class:`KlinesCall` and the rows about to be served (newest first) and returns
    the rows to serve. ``overrides`` (path -> handler) and :meth:`queue` (one-shot responses per
    path) inject failures. ``tick_ms`` advances the server time after every request.
    """

    def __init__(
        self,
        server_time_ms: int,
        candles: Mapping[str, Iterable[Candle]] | None = None,
        *,
        symbol: str = "BTC-USDT",
        history_start_ms: Mapping[str, int] | None = None,
        missing: Mapping[str, Iterable[int]] | None = None,
        tick_ms: int = 0,
        default_limit: int = 500,
    ) -> None:
        self.server_time_ms = int(server_time_ms)
        self.symbol = symbol
        self.history_start_ms: dict[str, int] = dict(history_start_ms or {})
        self.missing: dict[str, set[int]] = {k: set(v) for k, v in (missing or {}).items()}
        self.tick_ms = tick_ms
        self.default_limit = default_limit
        self.mutator: Mutator | None = None
        self.requests: list[httpx.Request] = []
        self.klines_calls: list[KlinesCall] = []
        self.overrides: dict[str, Callable[[httpx.Request], httpx.Response]] = {}
        self.queued: dict[str, list[httpx.Response]] = {}
        self._series: dict[str, dict[int, Candle]] = {}
        self._keys: dict[str, list[int]] = {}
        for interval, series in (candles or {}).items():
            self.set_candles(interval, series)

    # -- data ------------------------------------------------------------------------------------

    def set_candles(self, interval: str, candles: Iterable[Candle]) -> None:
        """Replace every candle of ``interval``; duplicates are a test bug and raise."""
        series: dict[int, Candle] = {}
        for candle in candles:
            if candle.open_ms in series:
                raise ValueError(f"duplicate candle at {candle.open_ms} in the fake")
            series[candle.open_ms] = candle
        self._series[interval] = series
        self._keys[interval] = sorted(series)

    def put(self, interval: str, *candles: Candle) -> None:
        """Insert or replace candles of ``interval`` by open time."""
        series = self._series.setdefault(interval, {})
        for candle in candles:
            series[candle.open_ms] = candle
        self._keys[interval] = sorted(series)

    def candle(self, interval: str, open_ms: int) -> Candle:
        return self._series[interval][open_ms]

    def advance(self, ms: int) -> None:
        self.server_time_ms += ms

    def queue(self, path: str, *responses: httpx.Response) -> None:
        """Serve ``responses`` (in order) to the next requests of ``path``."""
        self.queued.setdefault(path, []).extend(responses)

    def served(
        self,
        interval: str,
        start_ms: int | None,
        end_ms: int | None,
        limit: int = MAX_KLINES_LIMIT,
    ) -> list[Candle]:
        """What a klines request returns now, NEWEST first (before any ``mutator``).

        The candles opening at or after ``start_ms`` (and the history start), strictly before
        ``end_ms`` (the ``endTime``: one exactly at a candle's open excludes it, D-041; ``None``
        when the request has none) and at or before the server time (the forming candle
        included). The two upper bounds stay separate: the server time never excludes a candle
        opening exactly at it.
        """
        keys = self._keys.get(interval, [])
        lower = start_ms if start_ms is not None else (keys[0] if keys else 0)
        lower = max(lower, self.history_start_ms.get(interval, lower))
        stop = bisect_right(keys, self.server_time_ms)
        if end_ms is not None:
            stop = min(stop, bisect_left(keys, end_ms))
        series = self._series[interval] if keys else {}
        hidden = self.missing.get(interval, set())
        rows = [series[key] for key in keys[bisect_left(keys, lower) : stop] if key not in hidden]
        rows.reverse()
        return rows[: min(limit, MAX_KLINES_LIMIT)]

    # -- request log -------------------------------------------------------------------------------

    def klines_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == pm.KLINES_PATH]

    def server_time_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == pm.SERVER_TIME_PATH]

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    # -- handler -----------------------------------------------------------------------------------

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.scheme != "https" or request.url.raw_host != pm.PUBLIC_HOST.encode("ascii"):
            raise AssertionError("the BingX fake answers only the pinned public host")
        if request.method != "GET":
            raise AssertionError(f"the public client only sends GET, got {request.method}")
        self.requests.append(request)
        try:
            path = request.url.path
            pending = self.queued.get(path)
            if pending:
                return pending.pop(0)
            if path in self.overrides:
                return self.overrides[path](request)
            if path == pm.SERVER_TIME_PATH:
                return ok({"serverTime": self.server_time_ms})
            if path == pm.KLINES_PATH:
                return self._klines(request)
            return httpx.Response(404, text="not found")
        finally:
            self.server_time_ms += self.tick_ms

    def _klines(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        symbol = params.get("symbol", "")
        interval = params.get("interval", "")
        if symbol != self.symbol:
            return api_error(109400, f"symbol {symbol!r} does not exist")
        if interval not in pm.INTERVAL_MS:
            return api_error(109400, f"interval {interval!r} is invalid")
        limit = min(int(params.get("limit", self.default_limit)), MAX_KLINES_LIMIT)
        start = params.get("startTime")
        end = params.get("endTime")
        start_ms = None if start is None else int(start)
        end_ms = None if end is None else int(end)
        call = KlinesCall(
            index=len(self.klines_calls),
            interval=interval,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
            server_time_ms=self.server_time_ms,
        )
        self.klines_calls.append(call)
        if start_ms is not None and end_ms is None:
            rows: list[Candle] = []  # verified: startTime alone returns an empty list
        else:
            rows = self.served(interval, start_ms, end_ms, limit)
        if self.mutator is not None:
            rows = list(self.mutator(call, rows))
        return ok([kline_object(candle) for candle in rows])


def alter_on_calls(calls: Iterable[int], open_ms: int, **changes: str) -> Mutator:
    """Mutator replacing fields of the candle at ``open_ms`` on the given klines call indexes."""
    targets = set(calls)

    def mutate(call: KlinesCall, rows: list[Candle]) -> list[Candle]:
        if call.index not in targets:
            return rows
        return [replace(c, **changes) if c.open_ms == open_ms else c for c in rows]

    return mutate


def vary_each_call(open_ms: int, field: str = "volume") -> Mutator:
    """Mutator adding ``call.index + 1`` to one field of the candle at ``open_ms`` on every call
    (a forming candle whose values change between fetches)."""

    def mutate(call: KlinesCall, rows: list[Candle]) -> list[Candle]:
        out: list[Candle] = []
        for candle in rows:
            if candle.open_ms == open_ms:
                value = Decimal(getattr(candle, field)) + call.index + 1
                candle = replace(candle, **{field: format(value, "f")})
            out.append(candle)
        return out

    return mutate


def chain(*mutators: Mutator) -> Mutator:
    """Apply several mutators in order."""

    def mutate(call: KlinesCall, rows: list[Candle]) -> list[Candle]:
        for mutator in mutators:
            rows = list(mutator(call, rows))
        return rows

    return mutate


def for_interval(interval: str, mutator: Mutator) -> Mutator:
    """Apply ``mutator`` to the klines calls of ``interval`` only (the opens of a coarser
    interval are opens of the finer ones too)."""

    def mutate(call: KlinesCall, rows: list[Candle]) -> list[Candle]:
        return list(mutator(call, rows)) if call.interval == interval else rows

    return mutate


def bingx_client(fake: FakeBingX, **kwargs: Any) -> tuple[PublicMarketClient, FakeClock]:
    """A real :class:`PublicMarketClient` on ``httpx.MockTransport(fake)``, pacing off by default.

    Sleeps (rate-limit backoff, pacing when enabled) advance the returned :class:`FakeClock`
    instead of waiting.
    """
    clock = FakeClock()
    options: dict[str, Any] = {"min_interval_s": 0.0, **kwargs}
    client = PublicMarketClient(
        transport=httpx.MockTransport(fake), sleep=clock.sleep, clock=clock, **options
    )
    return client, clock


# -- Binance Vision fake ---------------------------------------------------------------------------


BINANCE_HOST = "data.binance.vision"
BINANCE_SYMBOL = "BTCUSDT"
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
ZIP_DATE_TIME = (2020, 1, 1, 0, 0, 0)


def archive_name(interval: str, period: str) -> str:
    """``BTCUSDT-<interval>-<period>.zip`` (period ``YYYY-MM`` or ``YYYY-MM-DD``)."""
    return f"{BINANCE_SYMBOL}-{interval}-{period}.zip"


def archive_url_path(kind: str, interval: str, period: str) -> str:
    """``/data/futures/um/<kind>/klines/BTCUSDT/<interval>/BTCUSDT-<interval>-<period>.zip``."""
    if kind not in ("monthly", "daily"):
        raise ValueError(f"kind must be monthly or daily, got {kind!r}")
    name = archive_name(interval, period)
    return f"/data/futures/um/{kind}/klines/{BINANCE_SYMBOL}/{interval}/{name}"


def period_bounds_ms(kind: str, period: str) -> tuple[int, int]:
    """``[start_ms, end_ms)`` of a monthly (``YYYY-MM``) or daily (``YYYY-MM-DD``) period."""
    if kind == "monthly":
        year, month = (int(part) for part in period.split("-"))
        days = calendar.monthrange(year, month)[1]
        start = datetime(year, month, 1, tzinfo=UTC)
        return to_ms(start), to_ms(start + timedelta(days=days))
    if kind == "daily":
        day = date.fromisoformat(period)
        start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        return to_ms(start), to_ms(start + timedelta(days=1))
    raise ValueError(f"kind must be monthly or daily, got {kind!r}")


def period_frame(
    kind: str, period: str, *, interval: str = "1m", seed: int = 0, start_price: float = 7_200.0
) -> pd.DataFrame:
    """:func:`binance_frame` covering a whole monthly or daily period."""
    start_ms, end_ms = period_bounds_ms(kind, period)
    step = pm.interval_ms(interval)
    return binance_frame(
        start_ms, (end_ms - start_ms) // step, interval=interval, seed=seed, start_price=start_price
    )


def kline_csv(
    frame: pd.DataFrame,
    *,
    interval: str,
    header: bool = True,
    microseconds: bool = False,
    line_ending: str = "\n",
) -> bytes:
    """Binance kline CSV of a ``BINANCE_COLUMNS`` string frame.

    Twelve fields per line (:data:`KLINE_COLUMNS`), ``close_time = open_time + interval - 1``
    in the file's unit (milliseconds, or microseconds when ``microseconds``), ``ignore = 0``;
    the header line is optional (2020 files have none).
    """
    step = pm.interval_ms(interval)
    scale = 1000 if microseconds else 1
    lines = [",".join(KLINE_COLUMNS)] if header else []
    for row in frame.itertuples(index=False):
        open_time = int(row.open_ms) * scale
        lines.append(
            ",".join(
                [
                    str(open_time),
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                    str(open_time + step * scale - 1),
                    row.quote_volume,
                    row.count,
                    row.taker_buy_volume,
                    row.taker_buy_quote_volume,
                    "0",
                ]
            )
        )
    return (line_ending.join(lines) + line_ending).encode("ascii") if lines else b""


def make_zip(member_name: str, payload: bytes) -> bytes:
    """A deterministic one-member deflated zip (fixed member time)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo(member_name, date_time=ZIP_DATE_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, payload)
    return buffer.getvalue()


def checksum_text(zip_bytes: bytes, zip_name: str) -> bytes:
    """``<sha256 hex>  <zip name>\\n``, the ``.CHECKSUM`` file format."""
    return f"{hashlib.sha256(zip_bytes).hexdigest()}  {zip_name}\n".encode("ascii")


class FakeBinanceVision:
    """data.binance.vision handler for ``httpx.MockTransport``: in-memory files, 404 otherwise.

    ``files`` maps URL paths to bytes; :meth:`add_archive` builds a kline zip and its
    ``.CHECKSUM``. ``overrides`` (path -> handler) and :meth:`queue` (one-shot responses) inject
    failures such as 5xx or timeouts. Every request is logged; anything but GET gets 405.
    """

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []
        self.overrides: dict[str, Callable[[httpx.Request], httpx.Response]] = {}
        self.queued: dict[str, list[httpx.Response]] = {}

    def add_file(self, url_path: str, content: bytes) -> None:
        self.files[url_path] = bytes(content)

    def remove(self, url_path: str) -> None:
        self.files.pop(url_path, None)

    def queue(self, path: str, *responses: httpx.Response) -> None:
        self.queued.setdefault(path, []).extend(responses)

    def add_archive(
        self,
        kind: str,
        interval: str,
        period: str,
        frame: pd.DataFrame,
        *,
        header: bool = True,
        microseconds: bool = False,
        line_ending: str = "\n",
        checksum: bool | str | bytes = True,
        member_name: str | None = None,
    ) -> bytes:
        """Serve ``frame`` as the archive of ``(kind, interval, period)``; return the zip bytes.

        ``checksum=True`` serves the correct ``.CHECKSUM``, ``False`` none (404), and a string or
        bytes value serves that content verbatim (a mismatch or a malformed file).
        """
        name = archive_name(interval, period)
        member = member_name or name.removesuffix(".zip") + ".csv"
        payload = kline_csv(
            frame,
            interval=interval,
            header=header,
            microseconds=microseconds,
            line_ending=line_ending,
        )
        zip_bytes = make_zip(member, payload)
        path = archive_url_path(kind, interval, period)
        self.files[path] = zip_bytes
        checksum_path = path + ".CHECKSUM"
        if checksum is True:
            self.files[checksum_path] = checksum_text(zip_bytes, name)
        elif checksum is False:
            self.files.pop(checksum_path, None)
        else:
            self.files[checksum_path] = (
                checksum.encode("ascii") if isinstance(checksum, str) else bytes(checksum)
            )
        return zip_bytes

    def requested_paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.scheme != "https" or url.raw_host != BINANCE_HOST.encode("ascii") or url.port:
            raise AssertionError("the Binance Vision fake answers only https://" + BINANCE_HOST)
        self.requests.append(request)
        if request.method != "GET":
            return httpx.Response(405, text="method not allowed")
        path = url.path
        pending = self.queued.get(path)
        if pending:
            return pending.pop(0)
        if path in self.overrides:
            return self.overrides[path](request)
        content = self.files.get(path)
        if content is None:
            return httpx.Response(404, text="<Error><Code>NoSuchKey</Code></Error>")
        media = "application/zip" if path.endswith(".zip") else "text/plain"
        return httpx.Response(200, content=content, headers={"content-type": media})
