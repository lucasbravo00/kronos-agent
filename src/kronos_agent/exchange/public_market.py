"""Real-market data from BingX perpetual futures: public GET only, no credentials, host pinned.

This client talks exclusively to ``https://open-api.bingx.com`` (SPEC rule 3: signals and volume
indicators always use REAL market candles). It carries no keys, has no ``headers`` parameter and
no host parameter, and a private transport wrapper refuses any request that is not plain HTTPS to
that host on the default port, so nothing built here can ever reach the demo host.

Facts verified live on 2026-10-29 (see ``CLAUDE.md``, "Exchange facts"): klines v3 returns
objects ``{open, close, high, low, volume, time}`` with string prices, NEWEST FIRST, the current
unclosed candle included first; ``startTime`` without ``endTime`` returns an empty list; ``limit``
is capped at 1000 anchored at ``endTime``; ``code != 0`` is an error (100410 = rate limit), never
"no data". BingX's own skills file documents 11-element arrays for this endpoint; that shape is
rejected by the schema guard on purpose.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Self

import httpx

from ..hashing import canonical_json, sha256_bytes

PUBLIC_HOST = "open-api.bingx.com"

SERVER_TIME_PATH = "/openApi/swap/v2/server/time"
KLINES_PATH = "/openApi/swap/v3/quote/klines"
CONTRACTS_PATH = "/openApi/swap/v2/quote/contracts"
PREMIUM_INDEX_PATH = "/openApi/swap/v2/quote/premiumIndex"
FUNDING_RATE_PATH = "/openApi/swap/v2/quote/fundingRate"
ALLOWED_PATHS: frozenset[str] = frozenset(
    {SERVER_TIME_PATH, KLINES_PATH, CONTRACTS_PATH, PREMIUM_INDEX_PATH, FUNDING_RATE_PATH}
)

INTERVAL_MS: Mapping[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

MAX_KLINES_PER_REQUEST = 1000
MAX_FUNDING_HISTORY = 1000
RATE_LIMIT_CODE = 100410
RATE_LIMIT_BACKOFF_S: tuple[float, ...] = (10.0, 30.0, 90.0)
TRANSIENT_BACKOFF_S = 1.0
CANDLE_KEYS: frozenset[str] = frozenset({"open", "close", "high", "low", "volume", "time"})
PRICE_KEYS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
FUNDING_ITEM_KEYS: frozenset[str] = frozenset({"symbol", "fundingRate", "fundingTime", "markPrice"})
USER_AGENT = "kronos-agent/0.1 (public market data)"


class PublicMarketError(Exception):
    """Base class of every error raised by this module."""


class HostNotAllowedError(PublicMarketError):
    """A request was about to leave for anything but https://open-api.bingx.com (default port)."""


class UnexpectedRedirect(PublicMarketError):
    """The server answered 3xx; redirects are never followed."""


class BingXBannedError(PublicMarketError):
    """HTTP 403: the firewall banned this client; retrying would make it worse."""


class BingXRateLimited(PublicMarketError):
    """HTTP 429 or API code 100410 persisted after every backoff step."""


class BingXUnavailableError(PublicMarketError):
    """HTTP 5xx or a transport timeout persisted after ``max_retries`` retries."""

    def __init__(self, path: str, status: int | None, detail: str):
        self.path = path
        self.status = status
        self.detail = detail
        where = f"HTTP {status}" if status is not None else "transport error"
        super().__init__(f"{path}: {where} ({detail})")


class BingXApiError(PublicMarketError):
    """The API answered ``code != 0`` (or a non-2xx status) for ``path``; never an empty result."""

    def __init__(self, code: int | None, msg: str, path: str, *, status: int = 200):
        self.code = code
        self.msg = msg
        self.path = path
        self.status = status
        super().__init__(f"{path}: HTTP {status}, code {code}: {msg}")


class BingXSchemaError(PublicMarketError):
    """The response does not have the verified shape (objects with six fields, ``code`` int...)."""


class DataGapError(PublicMarketError):
    """Consecutive candles are not exactly one interval apart (gap or duplicate)."""


class CandleMismatchError(PublicMarketError):
    """Two identical fetches of the same closed window returned different candles."""


@dataclass(frozen=True, slots=True)
class Candle:
    """One kline with the exchange's raw decimal strings (no float round-trip)."""

    open_ms: int
    open: str
    high: str
    low: str
    close: str
    volume: str

    def close_ms(self, interval_ms: int) -> int:
        """Close time in epoch milliseconds for a candle of ``interval_ms``."""
        return self.open_ms + interval_ms

    def as_floats(self) -> tuple[float, float, float, float, float]:
        """``(open, high, low, close, volume)`` as floats, for numeric work."""
        return (
            float(self.open),
            float(self.high),
            float(self.low),
            float(self.close),
            float(self.volume),
        )


def canonical_bytes(candles: Sequence[Candle]) -> bytes:
    """Deterministic encoding ``[[open_ms, open, high, low, close, volume], ...]`` (SPEC section 5)."""
    rows = [[c.open_ms, c.open, c.high, c.low, c.close, c.volume] for c in candles]
    return canonical_json(rows)


def window_sha256(candles: Sequence[Candle]) -> str:
    """sha256 hex of :func:`canonical_bytes`; recorded with every decision window."""
    return sha256_bytes(canonical_bytes(candles))


def interval_ms(interval: str) -> int:
    """Milliseconds of a whitelisted interval string; ``ValueError`` for anything else."""
    try:
        return INTERVAL_MS[interval]
    except (KeyError, TypeError):
        raise ValueError(
            f"unsupported interval {interval!r}; expected one of {list(INTERVAL_MS)}"
        ) from None


def _check_public_url(url: httpx.URL) -> None:
    """Refuse anything but plain HTTPS to the public host on the default port, no userinfo.

    The host is compared in its canonical ASCII form (``raw_host``); ``url.host`` is never read
    because it IDNA-decodes and raises ``idna`` errors for an invalid A-label.
    """
    host = url.raw_host.decode("ascii")  # ASCII by construction in httpx
    if url.scheme != "https" or host != "open-api.bingx.com" or url.port is not None:
        raise HostNotAllowedError(
            f"refusing request to {url.scheme!r}://{host!r} (port {url.port}); "
            "the public client only talks to https://open-api.bingx.com"
        )
    if url.userinfo != b"":
        raise HostNotAllowedError("refusing request URL with userinfo")


class _PublicPinnedTransport(httpx.BaseTransport):
    """Last gate before the wire: every request is checked against the pinned public host."""

    def __init__(self, inner: httpx.BaseTransport):
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        _check_public_url(request.url)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def _json_object(response: httpx.Response, path: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BingXSchemaError(f"{path}: response body is not JSON") from exc
    if not isinstance(payload, dict):
        raise BingXSchemaError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    return payload


def _api_code(payload: Mapping[str, Any], path: str) -> int:
    code = payload.get("code")
    if isinstance(code, bool) or not isinstance(code, int):
        raise BingXSchemaError(f"{path}: missing or non-integer 'code' in response")
    return code


def _data_field(payload: Mapping[str, Any], path: str) -> Any:
    if "data" not in payload:
        raise BingXSchemaError(f"{path}: response has no 'data' field")
    return payload["data"]


def _list_of_dicts(data: Any, path: str) -> list[dict[str, Any]]:
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise BingXSchemaError(f"{path}: expected 'data' to be a list of objects")
    return data


def _int_field(value: Any, path: str, key: str) -> int:
    if isinstance(value, bool):
        raise BingXSchemaError(f"{path}: field {key!r} is a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise BingXSchemaError(f"{path}: field {key!r} is not an integer")


def _decimal_string(value: Any, path: str, key: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise BingXSchemaError(f"{path}: field {key!r} must be a decimal string")
    text = str(value)
    try:
        float(text)
    except ValueError:
        raise BingXSchemaError(f"{path}: field {key!r} is not numeric") from None
    return text


def _parse_candles(data: Any, path: str) -> list[Candle]:
    """Schema guard: a list of six-key objects. Arrays (the shape BingX's skills file documents) fail."""
    if not isinstance(data, list):
        raise BingXSchemaError(f"{path}: expected 'data' to be a list")
    candles: list[Candle] = []
    for item in data:
        if not isinstance(item, dict):
            raise BingXSchemaError(
                f"{path}: kline items must be objects with keys {sorted(CANDLE_KEYS)}, "
                f"got {type(item).__name__}"
            )
        if set(item) != CANDLE_KEYS:
            raise BingXSchemaError(
                f"{path}: kline object keys {sorted(item)} != expected {sorted(CANDLE_KEYS)}"
            )
        strings = {key: _decimal_string(item[key], path, key) for key in PRICE_KEYS}
        candles.append(Candle(open_ms=_int_field(item["time"], path, "time"), **strings))
    return candles


def _check_contiguous(candles: Sequence[Candle], step_ms: int) -> None:
    for previous, current in pairwise(candles):
        delta = current.open_ms - previous.open_ms
        if delta != step_ms:
            kind = "duplicate" if delta == 0 else "gap"
            raise DataGapError(
                f"{kind} between candles at {previous.open_ms} and {current.open_ms} "
                f"(delta {delta} ms, expected {step_ms} ms)"
            )


def _require_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol must be a non-empty string such as 'BTC-USDT'")
    return symbol


def _require_ms(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer of epoch milliseconds")
    return value


class PublicMarketClient:
    """Paced, retrying, schema-checked GETs to the public BingX perpetual endpoints.

    ``transport`` (any ``httpx.BaseTransport``), ``sleep`` and ``clock`` (monotonic seconds) are
    injectable so unit tests run without network and without waiting. There is deliberately no
    host, URL or headers parameter.
    """

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
        min_interval_s: float = 2.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive seconds")
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be >= 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        inner = (
            transport if transport is not None else httpx.HTTPTransport(trust_env=False, retries=0)
        )
        self._transport = _PublicPinnedTransport(inner)
        self._http = httpx.Client(
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": USER_AGENT},
        )
        self._min_interval_s = float(min_interval_s)
        self._max_retries = int(max_retries)
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __repr__(self) -> str:
        return (
            f"PublicMarketClient(host={PUBLIC_HOST!r}, min_interval_s={self._min_interval_s}, "
            f"max_retries={self._max_retries})"
        )

    # -- transport -------------------------------------------------------------------------------

    def _pace(self) -> None:
        """Keep at least ``min_interval_s`` between request starts (<= 0.5 req/s by default)."""
        now = self._clock()
        if self._last_request_at is not None:
            wait = self._min_interval_s - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_request_at = now

    def _request_json(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One paced GET with retries; returns the JSON object once ``code == 0``."""
        if path not in ALLOWED_PATHS:
            raise ValueError(
                f"path {path!r} is not in the public allow-list {sorted(ALLOWED_PATHS)}"
            )
        url = httpx.URL(scheme="https", host=PUBLIC_HOST, path=path, params=dict(params or {}))
        rate_limit_backoffs = iter(RATE_LIMIT_BACKOFF_S)
        transient_left = self._max_retries
        attempts = 0
        while True:
            attempts += 1
            self._pace()
            try:
                response = self._http.get(url)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if transient_left == 0:
                    raise BingXUnavailableError(
                        path, None, f"{type(exc).__name__} after {attempts} attempts"
                    ) from exc
                transient_left -= 1
                self._sleep(TRANSIENT_BACKOFF_S)
                continue
            except (UnicodeError, httpx.InvalidURL):
                # httpx builds the never-followed redirect request itself: a 3xx Location it
                # cannot parse (an invalid A-label, no host) fails outside the HTTPError tree
                raise UnexpectedRedirect(
                    f"{path}: redirect with an unparseable Location refused"
                ) from None
            status = response.status_code
            if 300 <= status < 400:
                raise UnexpectedRedirect(f"{path}: HTTP {status} redirect refused (never followed)")
            if status == 403:
                raise BingXBannedError(f"{path}: HTTP 403 (firewall ban); not retried")
            if status >= 500:
                if transient_left == 0:
                    raise BingXUnavailableError(path, status, f"after {attempts} attempts")
                transient_left -= 1
                self._sleep(TRANSIENT_BACKOFF_S)
                continue
            if status == 200:
                payload = _json_object(response, path)
                code = _api_code(payload, path)
                if code == 0:
                    return payload
                if code != RATE_LIMIT_CODE:
                    raise BingXApiError(
                        code, str(payload.get("msg", ""))[:200], path, status=status
                    )
            elif status != 429:
                raise BingXApiError(None, response.text[:200], path, status=status)
            # HTTP 429 or code 100410: 10 s -> 30 s -> 90 s, then give up.
            backoff = next(rate_limit_backoffs, None)
            if backoff is None:
                raise BingXRateLimited(
                    f"{path}: rate limited (HTTP 429 / code {RATE_LIMIT_CODE}) after {attempts} attempts"
                )
            self._sleep(backoff)

    # -- endpoints -------------------------------------------------------------------------------

    def server_time_ms(self) -> int:
        """``GET /openApi/swap/v2/server/time`` -> ``data.serverTime`` in epoch milliseconds."""
        payload = self._request_json(SERVER_TIME_PATH)
        data = _data_field(payload, SERVER_TIME_PATH)
        if not isinstance(data, dict):
            raise BingXSchemaError(f"{SERVER_TIME_PATH}: expected 'data' to be an object")
        return _int_field(data.get("serverTime"), SERVER_TIME_PATH, "serverTime")

    def klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = MAX_KLINES_PER_REQUEST,
    ) -> list[Candle]:
        """One page of klines, ASCENDING by open time (the API returns newest first).

        Both bounds are mandatory: ``startTime`` alone returns an empty list on BingX. ``limit``
        is capped at 1000, anchored at ``end_ms``. The current unclosed candle is NOT removed here;
        use :meth:`closed_klines` for decision windows.
        """
        _require_symbol(symbol)
        interval_ms(interval)
        start_ms = _require_ms(start_ms, "start_ms")
        end_ms = _require_ms(end_ms, "end_ms")
        if start_ms > end_ms:
            raise ValueError("start_ms must be <= end_ms (both bounds are required by BingX)")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_KLINES_PER_REQUEST
        ):
            raise ValueError(f"limit must be an integer in [1, {MAX_KLINES_PER_REQUEST}]")
        payload = self._request_json(
            KLINES_PATH,
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            },
        )
        candles = _parse_candles(_data_field(payload, KLINES_PATH), KLINES_PATH)
        return sorted(candles, key=lambda c: c.open_ms)

    def closed_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        server_time_ms: int | None = None,
    ) -> list[Candle]:
        """Every CLOSED candle with open time in ``[start_ms, end_ms]``, ascending and contiguous.

        Pages backwards from ``end_ms`` (``endTime = oldest_returned - 1``) until ``start_ms`` is
        covered or a page comes back empty, drops candles whose close is after the server time,
        and raises :class:`DataGapError` on gaps or duplicates. If the history does not reach
        ``start_ms`` the result simply starts later; callers check ``candles[0].open_ms``.

        Each seam assumes BingX compares ``endTime`` with open times and does not round it up to
        the next boundary, so the next page's newest candle is the one before the previous page's
        oldest (observed live on 2026-10-31; the two-page network test checks it). A seam
        semantic that repeats or skips a candle raises :class:`DataGapError` and one that ignores
        ``endTime`` raises :class:`BingXSchemaError`; nothing is deduplicated or filled. Bounds
        on an exact boundary likewise assume inclusive ``startTime``/``endTime``: an exclusive
        one only trims that edge candle (an exclusive ``startTime`` also costs one empty page).
        """
        _require_symbol(symbol)
        step = interval_ms(interval)
        start_ms = _require_ms(start_ms, "start_ms")
        end_ms = _require_ms(end_ms, "end_ms")
        if start_ms > end_ms:
            raise ValueError("start_ms must be <= end_ms")
        if server_time_ms is None:
            server_time_ms = self.server_time_ms()
        else:
            server_time_ms = _require_ms(server_time_ms, "server_time_ms")
        collected: list[Candle] = []
        page_end = end_ms
        while page_end >= start_ms:
            page = self.klines(symbol, interval, start_ms, page_end, MAX_KLINES_PER_REQUEST)
            if not page:
                break
            oldest = page[0].open_ms
            if oldest > page_end:
                raise BingXSchemaError(
                    f"{KLINES_PATH}: page ignored endTime={page_end} (oldest open {oldest}); "
                    "refusing to loop"
                )
            collected.extend(page)
            if oldest < start_ms + step:
                break
            page_end = oldest - 1
        closed = sorted(
            (
                c
                for c in collected
                if start_ms <= c.open_ms <= end_ms and c.close_ms(step) <= server_time_ms
            ),
            key=lambda c: c.open_ms,
        )
        _check_contiguous(closed, step)
        return closed

    def fetch_twice(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        server_time_ms: int | None = None,
    ) -> list[Candle]:
        """Two identical :meth:`closed_klines` fetches; :class:`CandleMismatchError` if they differ.

        The server time is read once so both fetches drop the same unclosed candle (SPEC section 5:
        "download twice until they match"; the live loop does not trade on a mismatch).
        """
        if server_time_ms is None:
            server_time_ms = self.server_time_ms()
        first = self.closed_klines(
            symbol, interval, start_ms, end_ms, server_time_ms=server_time_ms
        )
        second = self.closed_klines(
            symbol, interval, start_ms, end_ms, server_time_ms=server_time_ms
        )
        if canonical_bytes(first) != canonical_bytes(second):
            raise CandleMismatchError(
                f"two fetches of {symbol} {interval} [{start_ms}, {end_ms}] differ: "
                f"{len(first)} vs {len(second)} candles, sha256 {window_sha256(first)[:12]} vs "
                f"{window_sha256(second)[:12]}"
            )
        return first

    canonical_bytes = staticmethod(canonical_bytes)
    window_sha256 = staticmethod(window_sha256)

    def contract(self, symbol: str) -> dict[str, Any]:
        """``GET /openApi/swap/v2/quote/contracts`` entry for ``symbol`` (precisions, fees, minimums)."""
        _require_symbol(symbol)
        payload = self._request_json(CONTRACTS_PATH, {"symbol": symbol})
        for item in _list_of_dicts(_data_field(payload, CONTRACTS_PATH), CONTRACTS_PATH):
            if item.get("symbol") == symbol:
                return item
        raise BingXSchemaError(f"{CONTRACTS_PATH}: no contract for {symbol!r} in response")

    def premium_index(self, symbol: str) -> dict[str, Any]:
        """``GET /openApi/swap/v2/quote/premiumIndex`` for ``symbol`` (mark price, funding interval)."""
        _require_symbol(symbol)
        payload = self._request_json(PREMIUM_INDEX_PATH, {"symbol": symbol})
        data = _data_field(payload, PREMIUM_INDEX_PATH)
        if isinstance(data, dict):
            return data
        for item in _list_of_dicts(data, PREMIUM_INDEX_PATH):
            if item.get("symbol") == symbol:
                return item
        raise BingXSchemaError(f"{PREMIUM_INDEX_PATH}: no premium index for {symbol!r} in response")

    def funding_rate_history(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """``GET /openApi/swap/v2/quote/fundingRate``: items ``{symbol, fundingRate, fundingTime, markPrice}``."""
        _require_symbol(symbol)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_FUNDING_HISTORY
        ):
            raise ValueError(f"limit must be an integer in [1, {MAX_FUNDING_HISTORY}]")
        payload = self._request_json(FUNDING_RATE_PATH, {"symbol": symbol, "limit": limit})
        items = _list_of_dicts(_data_field(payload, FUNDING_RATE_PATH), FUNDING_RATE_PATH)
        for item in items:
            missing = FUNDING_ITEM_KEYS - set(item)
            if missing:
                raise BingXSchemaError(f"{FUNDING_RATE_PATH}: funding item lacks {sorted(missing)}")
        return items


__all__ = [
    "ALLOWED_PATHS",
    "INTERVAL_MS",
    "PUBLIC_HOST",
    "BingXApiError",
    "BingXBannedError",
    "BingXRateLimited",
    "BingXSchemaError",
    "BingXUnavailableError",
    "Candle",
    "CandleMismatchError",
    "DataGapError",
    "HostNotAllowedError",
    "PublicMarketClient",
    "PublicMarketError",
    "UnexpectedRedirect",
    "canonical_bytes",
    "interval_ms",
    "window_sha256",
]
