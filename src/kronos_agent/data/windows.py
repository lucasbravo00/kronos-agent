"""Decision windows: the exact candles, hashes and float context Kronos reads for one hour ``D``.

One builder, :func:`build_window`, makes every window from any candle sequence: it keeps exactly
the hourly candles opening in ``[D - max_context h, D)``, refuses a missing one
(:class:`WindowGapError`) and ignores everything else, later candles and the unclosed one
included. The store path (:class:`StoreWindowSource`, a 1h study frame behind the sealed guard)
and the live path (:func:`live_decision_window`, two fetches of the real market that must match)
both go through it, so the same candles give the same ``sha256``, ``inputs_sha256`` and context
floats bit for bit (``docs/DATA.md``, "Decision windows").

``sha256`` is :func:`kronos_agent.exchange.public_market.window_sha256` of the candles;
``inputs_sha256`` also covers the ``amount`` policy and, under ``binance_quote_volume``, the
quote-volume strings. Floats come from ``float()`` of the stored strings. Timestamps are tz-naive
``datetime64[ns]`` meaning UTC, labelled by candle OPEN time: the context ends at ``D - 1h`` and
the targets start at ``D``.

``amount`` policies (question 9, default ``volume_x_mean_ohlc``): ``volume_x_mean_ohlc`` adds the
predictor's own proxy (:func:`kronos_agent.kronos.features.add_amount_proxy`, bit-identical to the
implicit one); ``binance_quote_volume`` uses Binance's quote volume (Binance stores only);
``zeros`` passes zero ``volume`` and ``amount``, which is what the predictor does when volume is
missing. A window from a Binance store carries the quote volumes whatever the policy (they are
part of what the source holds); only ``binance_quote_volume`` reads them.

:func:`save_window` publishes a window's inputs once, under a name that holds its source, context
length and hour: the candles, their ``sha256`` and the Binance quote volumes. How the model reads
them (the ``amount`` policy, ``inputs_sha256``) and how they were fetched (the live fetch pairs)
belong to each fan, whose metadata records them, so one saved window serves every policy and
every retry of its hour.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exchange.public_market import (
    Candle,
    CandleMismatchError,
    DataGapError,
    PublicMarketClient,
    window_sha256,
)
from ..hashing import canonical_json, sha256_bytes
from ..kronos.features import add_amount_proxy, validate_window
from ..paths import data_dir
from .fsutil import publish_new
from .partition import SEALED_START_MS, SealedAccess, require_accessible
from .store import DECIMAL_RE, SOURCE_SYMBOLS, StoreId, float_values, read_meta, read_study_frame
from .timeutil import HOUR_MS, hour_key, parse_utc_hour, to_ms, utc_text

DEFAULT_MAX_CONTEXT = 512
LIVE_MIN_DELAY_MS = 90_000
LIVE_MAX_DELAY_MS = 900_000
MAX_FETCH_PAIRS = 3
AMOUNT_POLICIES: tuple[str, ...] = ("volume_x_mean_ohlc", "binance_quote_volume", "zeros")
# Version 2: the saved file holds the inputs only (no amount policy, inputs_sha256 or fetch pairs).
WINDOW_SCHEMA_VERSION = 2
WINDOWS_DIR = "windows"
LIVE_SOURCE = "bingx"
WINDOW_INTERVAL = "1h"
CONTEXT_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")

_PLAIN_DECIMAL = re.compile(DECIMAL_RE)


class WindowError(ValueError):
    """A decision window that cannot be built as requested."""


class WindowGapError(WindowError):
    """A candle the window needs is missing (never filled)."""


class LiveTimingError(WindowError):
    """The live window was requested before ``D + 90 s`` or at/after ``D + 15 min``."""


class LiveMismatchError(WindowError):
    """Every pair of live fetches disagreed (SPEC section 11: no trading on a mismatch)."""


def naive_timestamps(open_ms: Sequence[int] | np.ndarray) -> pd.Series:
    """Tz-naive ``datetime64[ns]`` Series (UTC by convention) of epoch milliseconds, RangeIndex."""
    values = np.asarray(open_ms, dtype=np.int64)
    return pd.Series(pd.to_datetime(values, unit="ms")).astype("datetime64[ns]")


@dataclass(frozen=True, eq=False)
class DecisionWindow:
    """The window of one decision hour: candles as strings, their hashes and the float context.

    ``candles`` open at ``D - max_context h, ..., D - 1h``; ``context`` holds the six Kronos
    channels as float64 (one row per candle), ``x_timestamps`` their open times and
    ``y_timestamps`` the ``horizon`` target opens ``D, ..., D + (horizon - 1) h``. ``symbol`` is
    the source's own symbol; seeds use the project symbol instead (``docs/DATA.md``, "Forecasts").
    ``quote_volumes`` holds Binance's quote-volume string of every candle when the window was
    built with them (always from a Binance store; never for BingX); only the
    ``binance_quote_volume`` policy reads them. ``fetch_pairs`` is the number of fetch pairs the
    live path needed (``None`` for stores).
    """

    source: str
    symbol: str
    decision_hour: datetime
    max_context: int
    horizon: int
    amount_policy: str
    candles: tuple[Candle, ...]
    quote_volumes: tuple[str, ...] | None
    sha256: str
    inputs_sha256: str
    fetch_pairs: int | None
    context: pd.DataFrame
    x_timestamps: pd.Series
    y_timestamps: pd.Series

    @property
    def last_close(self) -> float:
        """``float()`` of the close of the candle opening at ``D - 1h``."""
        return float(self.candles[-1].close)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WindowError(f"{name} must be a positive integer, got {value!r}")
    return value


def _check_policy(amount_policy: str, quote_volumes: Sequence[str] | None, rows: int) -> None:
    if amount_policy not in AMOUNT_POLICIES:
        raise WindowError(f"amount_policy {amount_policy!r} not in {list(AMOUNT_POLICIES)}")
    if quote_volumes is None:
        if amount_policy == "binance_quote_volume":
            raise WindowError("binance_quote_volume needs one quote-volume string per candle")
        return
    if len(quote_volumes) != rows:
        raise WindowError("quote volumes must hold one string per candle")
    for value in quote_volumes:
        if not isinstance(value, str) or not _PLAIN_DECIMAL.fullmatch(value):
            raise WindowError("quote volumes must be plain decimal strings")


def context_frame(
    candles: Sequence[Candle], amount_policy: str, quote_volumes: Sequence[str] | None = None
) -> pd.DataFrame:
    """The six Kronos channels (float64, RangeIndex) of ``candles`` under ``amount_policy``.

    ``quote_volumes`` (one plain decimal string per candle, in order) is required by
    ``binance_quote_volume``; the other policies check it when given and do not read it.
    """
    _check_policy(amount_policy, quote_volumes, len(candles))
    frame = pd.DataFrame(
        {
            "open": float_values(c.open for c in candles),
            "high": float_values(c.high for c in candles),
            "low": float_values(c.low for c in candles),
            "close": float_values(c.close for c in candles),
            "volume": float_values(c.volume for c in candles),
        }
    )
    if amount_policy == "volume_x_mean_ohlc":
        frame = add_amount_proxy(frame)
    elif amount_policy == "binance_quote_volume":
        frame["amount"] = float_values(quote_volumes or ())
    else:
        frame["volume"] = np.zeros(len(frame), dtype=np.float64)
        frame["amount"] = np.zeros(len(frame), dtype=np.float64)
    return frame[list(CONTEXT_COLUMNS)].reset_index(drop=True)


def inputs_sha256(
    candles: Sequence[Candle], amount_policy: str, quote_volumes: Sequence[str] | None = None
) -> str:
    """sha256 of every string the model reads plus the ``amount`` policy.

    ``sha256(canonical_json({"rows": [[open_ms, o, h, l, c, v], ...], "amount_policy": ...}))``,
    with ``"quote_volumes": [...]`` added only under ``binance_quote_volume`` (the only policy
    that reads them; under the others they are checked when given and leave the hash unchanged).
    """
    _check_policy(amount_policy, quote_volumes, len(candles))
    document: dict[str, Any] = {
        "rows": [[c.open_ms, c.open, c.high, c.low, c.close, c.volume] for c in candles],
        "amount_policy": amount_policy,
    }
    if amount_policy == "binance_quote_volume":
        document["quote_volumes"] = list(quote_volumes or ())
    return sha256_bytes(canonical_json(document))


def _check_candle_strings(candle: Candle) -> None:
    for name in ("open", "high", "low", "close", "volume"):
        value = getattr(candle, name)
        if not isinstance(value, str) or not _PLAIN_DECIMAL.fullmatch(value):
            raise WindowError(
                f"candle {utc_text(candle.open_ms)}: {name} is not a plain decimal string"
            )


def build_window(
    candles: Sequence[Candle],
    decision_hour: datetime,
    *,
    source: str,
    horizon: int,
    max_context: int = DEFAULT_MAX_CONTEXT,
    amount_policy: str = "volume_x_mean_ohlc",
    quote_volumes: Mapping[int, str] | None = None,
    fetch_pairs: int | None = None,
) -> DecisionWindow:
    """The window of ``decision_hour`` from any candle sequence (the one builder).

    Keeps exactly the hourly candles opening in ``[D - max_context h, D)``, in order; every
    other candle (earlier ones, the candle opening at ``D``, later ones) is ignored. A missing
    candle raises :class:`WindowGapError` naming the missing ranges; two candles with the same
    open inside the window raise :class:`WindowError` (nothing is deduplicated).
    ``quote_volumes`` maps ``open_ms`` to Binance's quote-volume string (Binance sources only):
    required under ``binance_quote_volume``, kept with the window under every policy (a saved
    window records them) and read by the model only under ``binance_quote_volume``.
    """
    hour = parse_utc_hour(decision_hour)
    if source not in SOURCE_SYMBOLS:
        raise WindowError(f"unknown source {source!r}; expected one of {list(SOURCE_SYMBOLS)}")
    horizon = _positive_int(horizon, "horizon")
    max_context = _positive_int(max_context, "max_context")
    if amount_policy not in AMOUNT_POLICIES:
        raise WindowError(f"amount_policy {amount_policy!r} not in {list(AMOUNT_POLICIES)}")
    if amount_policy == "binance_quote_volume" and not source.startswith("binance"):
        raise WindowError(f"binance_quote_volume needs a Binance store, not {source!r}")
    if quote_volumes is not None and not source.startswith("binance"):
        raise WindowError(f"quote volumes exist only in Binance stores, not in {source!r}")
    if fetch_pairs is not None:
        fetch_pairs = _positive_int(fetch_pairs, "fetch_pairs")
    decision_ms = to_ms(hour)
    start_ms = decision_ms - max_context * HOUR_MS
    selected: dict[int, Candle] = {}
    for candle in candles:
        if not isinstance(candle, Candle):
            raise WindowError(f"expected Candle objects, got {type(candle).__name__}")
        if start_ms <= candle.open_ms < decision_ms:
            if candle.open_ms in selected:
                raise WindowError(f"two candles open at {utc_text(candle.open_ms)}")
            selected[candle.open_ms] = candle
    expected = range(start_ms, decision_ms, HOUR_MS)
    missing = [open_ms for open_ms in expected if open_ms not in selected]
    if missing:
        raise WindowGapError(
            f"{source} window of {utc_text(decision_ms)} misses {len(missing)} of {max_context} "
            f"candles: {_ranges_text(missing)}"
        )
    if len(selected) != max_context:
        raise WindowError(f"{source} window holds candles off the hourly grid")
    ordered = tuple(selected[open_ms] for open_ms in expected)
    for candle in ordered:
        _check_candle_strings(candle)
    quotes: tuple[str, ...] | None = None
    if amount_policy == "binance_quote_volume" and quote_volumes is None:
        raise WindowError("binance_quote_volume needs the quote volumes of the window")
    if quote_volumes is not None:
        absent = [c.open_ms for c in ordered if c.open_ms not in quote_volumes]
        if absent:
            raise WindowGapError(
                f"{source} window of {utc_text(decision_ms)} misses quote volumes: "
                f"{_ranges_text(absent)}"
            )
        quotes = tuple(quote_volumes[c.open_ms] for c in ordered)
    context = context_frame(ordered, amount_policy, quotes)
    x_timestamps = naive_timestamps([c.open_ms for c in ordered])
    y_timestamps = naive_timestamps([decision_ms + k * HOUR_MS for k in range(horizon)])
    try:
        validate_window(
            context,
            x_timestamps,
            max_context=max_context,
            horizon=horizon,
            y_timestamps=y_timestamps,
        )
    except ValueError as exc:
        raise WindowError(
            f"{source} window of {utc_text(decision_ms)} is not valid: {exc}"
        ) from exc
    return DecisionWindow(
        source=source,
        symbol=SOURCE_SYMBOLS[source],
        decision_hour=hour,
        max_context=max_context,
        horizon=horizon,
        amount_policy=amount_policy,
        candles=ordered,
        quote_volumes=quotes,
        sha256=window_sha256(ordered),
        inputs_sha256=inputs_sha256(ordered, amount_policy, quotes),
        fetch_pairs=fetch_pairs,
        context=context,
        x_timestamps=x_timestamps,
        y_timestamps=y_timestamps,
    )


def _ranges_text(open_times: Sequence[int]) -> str:
    """``[a, b)`` UTC ranges of consecutive missing hourly opens (at most five shown)."""
    ranges: list[list[int]] = []
    for open_ms in open_times:
        if ranges and open_ms == ranges[-1][1]:
            ranges[-1][1] = open_ms + HOUR_MS
        else:
            ranges.append([open_ms, open_ms + HOUR_MS])
    shown = ", ".join(f"[{utc_text(a)}, {utc_text(b)})" for a, b in ranges[:5])
    return shown + (f" and {len(ranges) - 5} more ranges" if len(ranges) > 5 else "")


class StoreWindowSource:
    """Windows from ``candles/<source>/1h``, loaded once as a STRING study frame.

    The frame is read through :func:`kronos_agent.data.store.read_study_frame`, so the sealed
    guard applies at load time; the default range is the whole store up to the sealed start
    (the whole store with a ``SealedAccess``). :meth:`window` checks the decision hour with
    :func:`kronos_agent.data.partition.require_accessible` first.
    """

    def __init__(
        self,
        source: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        sealed_access: SealedAccess | None = None,
        root: Path | None = None,
    ) -> None:
        if source not in SOURCE_SYMBOLS:
            raise WindowError(f"unknown source {source!r}; expected one of {list(SOURCE_SYMBOLS)}")
        self.source = source
        self.store_id = StoreId(source, WINDOW_INTERVAL)
        start = 0 if start_ms is None else start_ms
        if end_ms is not None:
            end = end_ms
        elif sealed_access is None:
            end = SEALED_START_MS
        else:
            last = read_meta(self.store_id, root).get("last_open_ms")
            end = SEALED_START_MS if last is None else max(SEALED_START_MS, last + HOUR_MS)
        self._binance = source.startswith("binance")
        columns = ["open", "high", "low", "close", "volume"]
        if self._binance:
            columns.append("quote_volume")
        frame = read_study_frame(
            self.store_id,
            start_ms=start,
            end_ms=end,
            numeric=False,
            usecols=columns,
            sealed_access=sealed_access,
            root=root,
        )
        # the range was validated by the sealed guard of read_study_frame (integers, start <= end)
        start, end = int(start), int(end)
        if frame.empty:
            raise WindowError(
                f"no {source} 1h candles in [{utc_text(start)}, {utc_text(end)}) of the store"
            )
        self.meta: dict[str, Any] = read_meta(self.store_id, root)
        self.start_ms = start
        self.end_ms = end
        self._sealed_access = sealed_access
        self._open_ms = frame["open_ms"].to_numpy(dtype=np.int64)
        self._values = {
            name: frame[name].to_numpy(dtype=object)
            for name in ("open", "high", "low", "close", "volume")
        }
        self._quotes = frame["quote_volume"].to_numpy(dtype=object) if self._binance else None
        self.first_open_ms = int(self._open_ms[0])
        self.last_open_ms = int(self._open_ms[-1])

    def _slice(self, start_ms: int, end_ms: int) -> slice:
        if start_ms > end_ms:
            raise WindowError(f"start_ms {start_ms} must be <= end_ms {end_ms}")
        lo = int(np.searchsorted(self._open_ms, start_ms, side="left"))
        hi = int(np.searchsorted(self._open_ms, end_ms, side="left"))
        return slice(lo, hi)

    def candles_between(self, start_ms: int, end_ms: int) -> list[Candle]:
        """Loaded candles opening in ``[start_ms, end_ms)``, ascending."""
        rows = self._slice(start_ms, end_ms)
        values = self._values
        return [
            Candle(int(open_ms), str(o), str(h), str(lo), str(c), str(v))
            for open_ms, o, h, lo, c, v in zip(
                self._open_ms[rows],
                values["open"][rows],
                values["high"][rows],
                values["low"][rows],
                values["close"][rows],
                values["volume"][rows],
                strict=True,
            )
        ]

    def quote_volumes_between(self, start_ms: int, end_ms: int) -> dict[int, str]:
        """Binance quote-volume strings by ``open_ms`` in ``[start_ms, end_ms)`` (Binance only)."""
        if self._quotes is None:
            raise WindowError(f"{self.source} has no quote volume (Binance stores only)")
        rows = self._slice(start_ms, end_ms)
        return {
            int(open_ms): str(value)
            for open_ms, value in zip(self._open_ms[rows], self._quotes[rows], strict=True)
        }

    def window(
        self,
        decision_hour: datetime,
        *,
        horizon: int,
        max_context: int = DEFAULT_MAX_CONTEXT,
        amount_policy: str = "volume_x_mean_ohlc",
        sealed_access: SealedAccess | None = None,
    ) -> DecisionWindow:
        """The window of ``decision_hour`` from the loaded candles.

        A sealed hour needs a ``SealedAccess`` (this call's, else the one the source was loaded
        with). A window reaching outside the loaded range raises :class:`WindowError`; a candle
        missing inside it raises :class:`WindowGapError`. A Binance window carries the quote
        volumes of its candles under every ``amount_policy``.
        """
        hour = parse_utc_hour(decision_hour)
        access = sealed_access if sealed_access is not None else self._sealed_access
        require_accessible(hour, access)
        max_context = _positive_int(max_context, "max_context")
        decision_ms = to_ms(hour)
        start_ms = decision_ms - max_context * HOUR_MS
        if start_ms < self.start_ms or decision_ms > self.end_ms:
            raise WindowError(
                f"window [{utc_text(start_ms)}, {utc_text(decision_ms)}) is outside the loaded "
                f"range [{utc_text(self.start_ms)}, {utc_text(self.end_ms)})"
            )
        quotes = self.quote_volumes_between(start_ms, decision_ms) if self._binance else None
        return build_window(
            self.candles_between(start_ms, decision_ms),
            hour,
            source=self.source,
            horizon=horizon,
            max_context=max_context,
            amount_policy=amount_policy,
            quote_volumes=quotes,
        )


def live_decision_window(
    client: PublicMarketClient,
    decision_hour: datetime,
    *,
    horizon: int,
    max_context: int = DEFAULT_MAX_CONTEXT,
    amount_policy: str = "volume_x_mean_ohlc",
    symbol: str = "BTC-USDT",
) -> DecisionWindow:
    """The live window of ``decision_hour`` from the real market, fetched twice until they match.

    Reads the server time itself and refuses (:class:`LiveTimingError`) unless
    ``D + 90 s <= server time < D + 15 min`` (SPEC sections 5 and 11), so it can never fetch
    history. Each pair is ``client.fetch_twice`` over ``[D - max_context h, D - 1 ms]`` (an end
    off the candle grid: the candles opening before ``D``, whatever the exchange does with an
    exact boundary) at a freshly read server time; a differing pair is repeated,
    at most :data:`MAX_FETCH_PAIRS` pairs and only while the server time is before
    ``D + 15 min``, then :class:`LiveMismatchError`.
    """
    hour = parse_utc_hour(decision_hour)
    if symbol != SOURCE_SYMBOLS[LIVE_SOURCE]:
        raise WindowError(f"the live window is {SOURCE_SYMBOLS[LIVE_SOURCE]}, not {symbol!r}")
    if amount_policy not in AMOUNT_POLICIES:
        raise WindowError(f"amount_policy {amount_policy!r} not in {list(AMOUNT_POLICIES)}")
    if amount_policy == "binance_quote_volume":
        raise WindowError("binance_quote_volume is not available on the live (real-market) source")
    _positive_int(horizon, "horizon")
    max_context = _positive_int(max_context, "max_context")
    decision_ms = to_ms(hour)
    first_ms = decision_ms - max_context * HOUR_MS
    deadline_ms = decision_ms + LIVE_MAX_DELAY_MS
    mismatches: list[str] = []
    for pair in range(1, MAX_FETCH_PAIRS + 1):
        server_ms = client.server_time_ms()
        if pair == 1 and not decision_ms + LIVE_MIN_DELAY_MS <= server_ms < deadline_ms:
            earliest = utc_text(decision_ms + LIVE_MIN_DELAY_MS)
            raise LiveTimingError(
                f"server time {utc_text(server_ms)} is outside [{earliest}, "
                f"{utc_text(deadline_ms)}) for the decision at {utc_text(decision_ms)}"
            )
        if pair > 1 and server_ms >= deadline_ms:
            raise LiveMismatchError(
                f"{len(mismatches)} fetch pair(s) of the {utc_text(decision_ms)} window differed "
                f"and the server time {utc_text(server_ms)} reached D + 15 min: {mismatches[-1]}"
            )
        try:
            candles = client.fetch_twice(
                symbol, WINDOW_INTERVAL, first_ms, decision_ms - 1, server_time_ms=server_ms
            )
        except CandleMismatchError as exc:
            mismatches.append(str(exc))
            continue
        except DataGapError as exc:
            raise WindowGapError(f"live window of {utc_text(decision_ms)}: {exc}") from exc
        return build_window(
            candles,
            hour,
            source=LIVE_SOURCE,
            horizon=horizon,
            max_context=max_context,
            amount_policy=amount_policy,
            fetch_pairs=pair,
        )
    raise LiveMismatchError(
        f"all {MAX_FETCH_PAIRS} fetch pairs of the {utc_text(decision_ms)} window differed: "
        f"{mismatches[-1]}"
    )


def window_path(
    source: str,
    decision_hour: datetime,
    *,
    max_context: int = DEFAULT_MAX_CONTEXT,
    root: Path | None = None,
) -> Path:
    """``windows/<source>/context-<max_context>/<YYYY-MM-DDTHH>Z.json`` under the data root."""
    if source not in SOURCE_SYMBOLS:
        raise WindowError(f"unknown source {source!r}")
    max_context = _positive_int(max_context, "max_context")
    base = Path(root) if root is not None else data_dir()
    return (
        base / WINDOWS_DIR / source / f"context-{max_context}" / f"{hour_key(decision_hour)}.json"
    )


def _window_document(window: DecisionWindow) -> dict[str, Any]:
    """The saved inputs of ``window``: nothing that depends on the amount policy or the fetch."""
    return {
        "schema_version": WINDOW_SCHEMA_VERSION,
        "source": window.source,
        "symbol": window.symbol,
        "decision_hour": utc_text(to_ms(window.decision_hour)),
        "decision_ms": to_ms(window.decision_hour),
        "max_context": window.max_context,
        "sha256": window.sha256,
        "candles": [[c.open_ms, c.open, c.high, c.low, c.close, c.volume] for c in window.candles],
        "quote_volumes": None if window.quote_volumes is None else list(window.quote_volumes),
    }


def save_window(window: DecisionWindow, *, root: Path | None = None) -> Path:
    """Publish the inputs of a decision's window (:func:`window_path`); never overwrite.

    The file is canonical JSON of what the window reads from its source: the candles as strings,
    their ``sha256`` and, for a Binance window, the quote volumes (a Binance window without them
    is refused, so every saved Binance window can serve every ``amount`` policy). The ``amount``
    policy, ``inputs_sha256``, the horizon and the live fetch pairs are not part of it: the fan's
    metadata records them. If the file exists it must hold identical bytes, and is then kept;
    different bytes (other candles or quote volumes for the same hour) raise
    :class:`WindowError`.
    """
    if not isinstance(window, DecisionWindow):
        raise TypeError(f"expected a DecisionWindow, got {type(window).__name__}")
    if window.source.startswith("binance") and window.quote_volumes is None:
        raise WindowError(
            f"a {window.source} window is saved with its quote volumes; build it from the store "
            "(StoreWindowSource)"
        )
    path = window_path(
        window.source, window.decision_hour, max_context=window.max_context, root=root
    )
    data = canonical_json(_window_document(window)) + b"\n"
    try:
        publish_new(path, data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise WindowError(
                f"a different window is already saved for {window.source} "
                f"{hour_key(window.decision_hour)} ({path.parent.name}/{path.name}: other candles "
                "or quote volumes); it is never overwritten"
            ) from None
    return path


def load_window(path: Path) -> tuple[list[Candle], dict[str, Any]]:
    """Candles and metadata of a saved window; the candle hash is recomputed and must match.

    The metadata is every other field of the file (source, ``D``, context length, ``sha256``,
    ``quote_volumes``); the quote volumes must be ``None`` or one plain decimal string per candle.
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="ascii"))
    except (OSError, ValueError) as exc:
        raise WindowError(f"cannot read saved window {path.name}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != WINDOW_SCHEMA_VERSION:
        raise WindowError(f"{path.name} is not a saved decision window")
    try:
        candles = [
            Candle(int(row[0]), row[1], row[2], row[3], row[4], row[5])
            for row in document["candles"]
        ]
        for candle in candles:
            _check_candle_strings(candle)
        if len(candles) != document["max_context"]:
            raise WindowError(
                f"{path.name}: {len(candles)} candles for a context of {document['max_context']}"
            )
        if window_sha256(candles) != document["sha256"]:
            raise WindowError(f"{path.name}: candles do not match the recorded sha256")
        quotes = document["quote_volumes"]
        if (quotes is not None) != document["source"].startswith("binance"):
            raise WindowError(f"{path.name}: quote volumes belong to Binance windows, and to all")
        if quotes is not None and (
            not isinstance(quotes, list)
            or len(quotes) != len(candles)
            or not all(isinstance(q, str) and _PLAIN_DECIMAL.fullmatch(q) for q in quotes)
        ):
            raise WindowError(f"{path.name}: quote volumes are not one plain decimal per candle")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise WindowError(f"{path.name} is malformed: {exc!r}") from exc
    metadata = {key: value for key, value in document.items() if key != "candles"}
    return candles, metadata


__all__ = [
    "AMOUNT_POLICIES",
    "CONTEXT_COLUMNS",
    "DEFAULT_MAX_CONTEXT",
    "LIVE_MAX_DELAY_MS",
    "LIVE_MIN_DELAY_MS",
    "MAX_FETCH_PAIRS",
    "WINDOWS_DIR",
    "WINDOW_SCHEMA_VERSION",
    "DecisionWindow",
    "LiveMismatchError",
    "LiveTimingError",
    "StoreWindowSource",
    "WindowError",
    "WindowGapError",
    "build_window",
    "context_frame",
    "inputs_sha256",
    "live_decision_window",
    "load_window",
    "naive_timestamps",
    "save_window",
    "window_path",
]
