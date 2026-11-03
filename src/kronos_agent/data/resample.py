"""Exact resampling of Binance 1m candles to 5m, 15m and 1h, and the check against Binance's 1h.

The stores keep the exchange's decimal strings verbatim, so a resampled candle is built from the
strings and never from floats (``docs/DATA.md``, "Resampling and its check"):

- ``open`` is the first minute's string and ``close`` the last minute's string;
- ``high`` / ``low`` are the string of the largest / smallest value compared as decimals, the
  first occurrence on ties (``"7225"`` and ``"7225.00"`` are equal; the earlier one is kept), so
  ``"100.1"`` beats ``"100.09999999999999999"`` although both are the same float;
- ``volume``, ``quote_volume``, ``taker_buy_volume`` and ``taker_buy_quote_volume`` are exact
  decimal sums in a ``decimal.localcontext`` with ``prec = 60`` and the ``Inexact`` and
  ``Rounded`` traps on (a sum that would need rounding raises instead of rounding), written with
  ``format(total, "f")``, which keeps the largest number of decimals of the summands;
- ``count`` is the integer sum.

A bucket is labelled by its open, covers ``[open, open + step)`` and is kept only when every
minute of it is present; an incomplete bucket is a gap, listed with its minute count (nothing is
ever filled). The builder (:mod:`kronos_agent.data.binance_vision`) resamples archive by archive:
archive periods are whole UTC days or months, so a bucket never crosses an archive boundary.

The check (:func:`compare_with_official`) compares a resampled 1h frame with Binance's own 1h
archives hour by hour, as decimals: ``open``, ``high``, ``low``, ``close``, ``volume``, ``count``
and ``taker_buy_volume`` must be equal; the two quote volumes may differ by at most
``sum(0.5 * 10^-d_i) + 0.5 * 10^-d_official``, the rounding of the 60 minute values and of the
official value (``d`` = digits after the decimal point of each string), which
:func:`quote_rounding` computes per bucket. Hours at or after the sealed start contribute counts
and open times only, never values or differences.
"""

from __future__ import annotations

import contextlib
import decimal
import functools
import re
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from ..exchange.public_market import interval_ms
from .store import BINANCE_COLUMNS, DECIMAL_RE, StoreError, StoreId, validate_frame
from .timeutil import MINUTE_MS, utc_text

TARGET_INTERVALS: tuple[str, ...] = ("5m", "15m", "1h")
SUM_PRECISION = 60
SUM_COLUMNS: tuple[str, ...] = (
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume",
)
EXACT_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "count",
    "taker_buy_volume",
)
TOLERANCE_COLUMNS: tuple[str, ...] = ("quote_volume", "taker_buy_quote_volume")
ROUNDING_COLUMNS: tuple[str, ...] = tuple(f"{name}_rounding" for name in TOLERANCE_COLUMNS)
MAX_MISMATCH_SAMPLES = 20

_DECIMAL = re.compile(DECIMAL_RE)
_ZERO = Decimal(0)
_MINUTE_STORE = StoreId("binance", "1m")
_RESAMPLED_1H = StoreId("binance", "1h")
_OFFICIAL_1H = StoreId("binance_official", "1h")


class ResampleError(ValueError):
    """Input that cannot be resampled or compared exactly as specified."""


@contextlib.contextmanager
def exact_decimal_context() -> Iterator[decimal.Context]:
    """A local decimal context in which every operation is exact or raises.

    ``prec = SUM_PRECISION`` (60 significant digits, far above any candle value) with the
    ``Inexact``, ``Rounded`` and ``InvalidOperation`` traps on: a result that would need more
    digits raises ``decimal.Inexact`` / ``decimal.Rounded`` instead of being rounded silently.
    """
    with decimal.localcontext() as context:
        context.prec = SUM_PRECISION
        context.rounding = decimal.ROUND_HALF_EVEN
        context.traps[decimal.Inexact] = True
        context.traps[decimal.Rounded] = True
        context.traps[decimal.InvalidOperation] = True
        yield context


def _sum_exact(values: Sequence[Decimal]) -> Decimal:
    """Exact sum inside :func:`exact_decimal_context` (callers hold the context)."""
    try:
        return sum(values, _ZERO)
    except (decimal.Inexact, decimal.Rounded) as exc:
        raise ResampleError(
            f"an exact decimal sum needs more than {SUM_PRECISION} significant digits"
        ) from exc


def decimal_sum(values: Sequence[str]) -> str:
    """Exact sum of plain decimal strings, written with ``format(total, "f")``.

    Every value must match the store's plain-decimal pattern (``"1e5"``, ``" 5"``, ``"nan"`` are
    refused: ``Decimal`` would accept them). The sum keeps the largest number of decimals of the
    summands (``"1.10" + "2.2" = "3.30"``, ``"0.000" + "0" = "0.000"``) and never rounds: a sum
    needing more than 60 significant digits raises :class:`ResampleError`. An empty sequence is
    refused (a bucket always has minutes).
    """
    if isinstance(values, (str, bytes)):
        raise TypeError("values must be a sequence of decimal strings, not a single string")
    texts = list(values)
    if not texts:
        raise ResampleError("cannot sum an empty sequence of values")
    _require_decimal_strings(texts, "values")
    with exact_decimal_context():
        total = _sum_exact([Decimal(text) for text in texts])
    return format(total, "f")


def _target_step(target: str) -> int:
    if target not in TARGET_INTERVALS:
        raise ValueError(f"unknown target interval {target!r}; expected one of {TARGET_INTERVALS}")
    return interval_ms(target)


def _buckets(open_ms: np.ndarray, step_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(bucket_open_ms, first_row, rows)`` for every bucket holding at least one minute.

    ``open_ms`` is strictly increasing and minute-aligned, so the rows of one bucket are
    contiguous and a bucket with ``step / 1 min`` rows holds every one of its minutes.
    """
    labels = open_ms - open_ms % step_ms
    if labels.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), change.astype(np.int64)))
    counts = np.diff(np.concatenate((starts, np.array([labels.size], dtype=np.int64))))
    return labels[starts].astype(np.int64), starts, counts


def _validated_minutes(frame_1m: pd.DataFrame) -> None:
    try:
        validate_frame(frame_1m, _MINUTE_STORE)
    except StoreError as exc:
        raise ResampleError(f"not a binance/1m string frame: {exc}") from exc


def resample_many(
    frame_1m: pd.DataFrame, targets: Sequence[str] = TARGET_INTERVALS, *, validate: bool = True
) -> dict[str, tuple[pd.DataFrame, list[tuple[int, int]]]]:
    """Resample one 1m string frame to several targets: ``{target: (frame, incomplete)}``.

    The decimal conversion of the minute values is done once for every target (the builder
    resamples each archive to 5m, 15m and 1h). ``frame`` holds the complete buckets as a
    ``BINANCE_COLUMNS`` string frame; ``incomplete`` lists ``(bucket_open_ms, minutes_present)``
    for every bucket holding some but not all of its minutes. ``validate=False`` skips the
    store-format check of ``frame_1m`` (only for frames already validated by the caller).
    """
    steps = {target: _target_step(target) for target in targets}
    if validate:
        _validated_minutes(frame_1m)
    open_ms = frame_1m["open_ms"].to_numpy(dtype=np.int64)
    strings = {name: frame_1m[name].tolist() for name in BINANCE_COLUMNS[1:]}
    highs = [Decimal(text) for text in strings["high"]]
    lows = [Decimal(text) for text in strings["low"]]
    sums = {name: [Decimal(text) for text in strings[name]] for name in SUM_COLUMNS}
    trades = [int(text) for text in strings["count"]]
    results: dict[str, tuple[pd.DataFrame, list[tuple[int, int]]]] = {}
    with exact_decimal_context():
        for target, step in steps.items():
            minutes = step // MINUTE_MS
            labels, starts, counts = _buckets(open_ms, step)
            complete = np.flatnonzero(counts == minutes)
            rows: dict[str, list[str]] = {name: [] for name in BINANCE_COLUMNS[1:]}
            for bucket in complete:
                first = int(starts[bucket])
                stop = first + minutes
                span = range(first, stop)
                rows["open"].append(strings["open"][first])
                rows["high"].append(strings["high"][max(span, key=highs.__getitem__)])
                rows["low"].append(strings["low"][min(span, key=lows.__getitem__)])
                rows["close"].append(strings["close"][stop - 1])
                for name in SUM_COLUMNS:
                    rows[name].append(format(_sum_exact(sums[name][first:stop]), "f"))
                rows["count"].append(str(sum(trades[first:stop])))
            data: dict[str, Any] = {"open_ms": labels[complete].astype(np.int64)}
            for name in BINANCE_COLUMNS[1:]:
                data[name] = pd.Series(rows[name], dtype=object)
            incomplete = [
                (int(labels[bucket]), int(counts[bucket]))
                for bucket in np.flatnonzero(counts < minutes)
            ]
            results[target] = (pd.DataFrame(data, columns=list(BINANCE_COLUMNS)), incomplete)
    return results


def resample_exact(frame_1m: pd.DataFrame, target: str) -> pd.DataFrame:
    """The complete ``target`` buckets of a ``binance/1m`` string frame (see the module docstring).

    Returns a ``BINANCE_COLUMNS`` string frame labelled by bucket open; buckets missing any
    minute are left out (:func:`incomplete_buckets` lists them).
    """
    return resample_many(frame_1m, (target,))[target][0]


def incomplete_buckets(frame_1m: pd.DataFrame, target: str) -> list[tuple[int, int]]:
    """``(bucket_open_ms, minutes_present)`` of every bucket holding some but not all minutes.

    A bucket with no minute at all is simply absent (a gap of the 1m store itself).
    """
    step = _target_step(target)
    _validated_minutes(frame_1m)
    labels, _, counts = _buckets(frame_1m["open_ms"].to_numpy(dtype=np.int64), step)
    minutes = step // MINUTE_MS
    return [(int(labels[b]), int(counts[b])) for b in np.flatnonzero(counts < minutes)]


def decimals(text: str) -> int:
    """Digits after the decimal point of a plain decimal string (``"7225"`` -> 0)."""
    point = text.find(".")
    return 0 if point < 0 else len(text) - point - 1


@functools.cache
def _half_unit_for(places: int) -> Decimal:
    return Decimal((0, (5,), -(places + 1)))


def half_unit(text: str) -> Decimal:
    """``0.5 * 10^-d`` for a string with ``d`` decimals: its rounding error bound, exactly."""
    return _half_unit_for(decimals(text))


def _require_decimal_strings(texts: Sequence[Any], column: str) -> None:
    """Every value a Python ``str`` matching the plain-decimal pattern (first offender named)."""
    for text in texts:
        if not isinstance(text, str):
            raise ResampleError(f"{column}: {text!r} is not a string")
    matches = list(map(_DECIMAL.fullmatch, texts))
    if None in matches:
        bad = texts[matches.index(None)]
        raise ResampleError(f"{column}: {bad!r} is not a plain decimal string ({DECIMAL_RE})")


def quote_rounding(frame_1m: pd.DataFrame, target: str) -> pd.DataFrame:
    """Per bucket: its minute count and the rounding bound of each quote-volume sum.

    ``frame_1m`` needs ``open_ms`` and the two quote-volume columns (the check reads only those
    from the 1m store). For every bucket holding at least one minute, returns ``open_ms`` (the
    bucket open), ``minutes`` and, for ``quote_volume`` and ``taker_buy_quote_volume``,
    ``sum(0.5 * 10^-d_i)`` over the minute values as a plain decimal string
    (:data:`ROUNDING_COLUMNS`).
    """
    step = _target_step(target)
    missing = [c for c in ("open_ms", *TOLERANCE_COLUMNS) if c not in frame_1m.columns]
    if missing:
        raise ResampleError(f"quote_rounding needs the columns {missing}")
    open_ms = frame_1m["open_ms"]
    if open_ms.dtype != np.int64:
        raise ResampleError(f"open_ms must be int64, got {open_ms.dtype}")
    values = open_ms.to_numpy()
    if values.size and ((values % MINUTE_MS).any() or (np.diff(values) <= 0).any()):
        raise ResampleError("open_ms must be minute-aligned and strictly increasing")
    labels, starts, counts = _buckets(values, step)
    data: dict[str, Any] = {"open_ms": labels, "minutes": counts.astype(np.int64)}
    with exact_decimal_context():
        for column, output in zip(TOLERANCE_COLUMNS, ROUNDING_COLUMNS, strict=True):
            texts = frame_1m[column].tolist()
            _require_decimal_strings(texts, column)
            units = [half_unit(text) for text in texts]
            data[output] = pd.Series(
                [
                    format(_sum_exact(units[int(first) : int(first) + int(count)]), "f")
                    for first, count in zip(starts, counts, strict=True)
                ],
                dtype=object,
            )
    return pd.DataFrame(data, columns=["open_ms", "minutes", *ROUNDING_COLUMNS])


def _require_compare_inputs(resampled: pd.DataFrame, official: pd.DataFrame) -> None:
    if not isinstance(resampled, pd.DataFrame) or not isinstance(official, pd.DataFrame):
        raise ResampleError("resampled and official must be pandas DataFrames")
    expected = [*BINANCE_COLUMNS, *ROUNDING_COLUMNS]
    if list(resampled.columns) != expected:
        raise ResampleError(
            f"resampled must have the columns {expected} (the 1h store plus the rounding bound "
            f"of each quote-volume sum, see quote_rounding), got {list(resampled.columns)}"
        )
    try:
        validate_frame(resampled[list(BINANCE_COLUMNS)], _RESAMPLED_1H)
        validate_frame(official, _OFFICIAL_1H)
    except StoreError as exc:
        raise ResampleError(f"not a 1h string frame: {exc}") from exc
    for column in ROUNDING_COLUMNS:
        _require_decimal_strings(resampled[column].tolist(), column)


def compare_with_official(
    resampled: pd.DataFrame, official: pd.DataFrame, *, sealed_start_ms: int
) -> dict[str, Any]:
    """Compare a resampled 1h frame with Binance's own 1h candles, hour by hour, as decimals.

    ``resampled`` is a ``binance/1h`` string frame followed by :data:`ROUNDING_COLUMNS` (from
    :func:`quote_rounding` of the minutes it was built from); ``official`` a
    ``binance_official/1h`` string frame. Every hour present in both is compared:
    :data:`EXACT_COLUMNS` must be equal, :data:`TOLERANCE_COLUMNS` may differ by at most
    ``rounding + 0.5 * 10^-d_official``. Returns per-column counts, the largest absolute
    difference (and, for the quote volumes, the largest ratio to the tolerance), capped samples
    of mismatches with their values and the hours present on one side only.

    Hours opening at or after ``sealed_start_ms`` are counted (compared, mismatching) and their
    mismatches are listed by open time and column only: no value or difference of a sealed hour
    ever enters the result.
    """
    if isinstance(sealed_start_ms, bool) or not isinstance(sealed_start_ms, (int, np.integer)):
        raise ResampleError(f"sealed_start_ms must be an integer, got {sealed_start_ms!r}")
    sealed_start = int(sealed_start_ms)
    _require_compare_inputs(resampled, official)
    r_open = resampled["open_ms"].to_numpy(dtype=np.int64)
    o_open = official["open_ms"].to_numpy(dtype=np.int64)
    common, r_rows, o_rows = np.intersect1d(r_open, o_open, assume_unique=True, return_indices=True)
    sealed = common >= sealed_start
    columns: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    sealed_mismatches: list[dict[str, Any]] = []
    exact_mismatches = 0
    tolerance_mismatches = 0
    with exact_decimal_context():
        for column in (*EXACT_COLUMNS, *TOLERANCE_COLUMNS):
            tolerant = column in TOLERANCE_COLUMNS
            ours = resampled[column].to_numpy(dtype=object)[r_rows]
            theirs = official[column].to_numpy(dtype=object)[o_rows]
            rounding = (
                resampled[f"{column}_rounding"].to_numpy(dtype=object)[r_rows] if tolerant else None
            )
            mismatches = mismatches_sealed = 0
            largest = _ZERO
            largest_at: int | None = None
            ratio_max = 0.0
            ratio_at: int | None = None
            column_samples = column_sealed = 0
            for row in range(common.size):
                open_ms = int(common[row])
                try:
                    difference = Decimal(ours[row]) - Decimal(theirs[row])
                except (decimal.Inexact, decimal.Rounded) as exc:
                    raise ResampleError(
                        f"{column} at {utc_text(open_ms)}: difference needs more than "
                        f"{SUM_PRECISION} digits"
                    ) from exc
                size = abs(difference)
                tolerance: Decimal | None = None
                if rounding is not None:
                    tolerance = Decimal(rounding[row]) + half_unit(theirs[row])
                    bad = size > tolerance
                else:
                    bad = size != 0
                if sealed[row]:
                    if bad:
                        mismatches_sealed += 1
                        if column_sealed < MAX_MISMATCH_SAMPLES:
                            column_sealed += 1
                            sealed_mismatches.append(
                                {"open_ms": open_ms, "hour": utc_text(open_ms), "column": column}
                            )
                    continue
                if size > largest:
                    largest, largest_at = size, open_ms
                if tolerance is not None:
                    ratio = float(size) / float(tolerance)
                    if ratio > ratio_max:
                        ratio_max, ratio_at = ratio, open_ms
                if bad:
                    mismatches += 1
                    if column_samples < MAX_MISMATCH_SAMPLES:
                        column_samples += 1
                        samples.append(
                            {
                                "open_ms": open_ms,
                                "hour": utc_text(open_ms),
                                "column": column,
                                "resampled": ours[row],
                                "official": theirs[row],
                                "difference": format(difference, "f"),
                                "tolerance": None if tolerance is None else format(tolerance, "f"),
                            }
                        )
            stats: dict[str, Any] = {
                "rule": "tolerance" if tolerant else "equal",
                "compared": int(common.size),
                "mismatches": mismatches + mismatches_sealed,
                "mismatches_before_sealed": mismatches,
                "mismatches_sealed": mismatches_sealed,
                "max_abs_difference": format(largest, "f"),
                "max_abs_difference_open_ms": largest_at,
            }
            if tolerant:
                stats["max_ratio_to_tolerance"] = ratio_max
                stats["max_ratio_open_ms"] = ratio_at
                tolerance_mismatches += mismatches + mismatches_sealed
            else:
                exact_mismatches += mismatches + mismatches_sealed
            columns[column] = stats
    return {
        "sealed_start_ms": sealed_start,
        "hours_resampled": int(r_open.size),
        "hours_official": int(o_open.size),
        "hours_compared": int(common.size),
        "hours_compared_before_sealed": int((~sealed).sum()),
        "hours_compared_sealed": int(sealed.sum()),
        "only_resampled": [int(v) for v in np.setdiff1d(r_open, o_open, assume_unique=True)],
        "only_official": [int(v) for v in np.setdiff1d(o_open, r_open, assume_unique=True)],
        "columns": columns,
        "exact_mismatches": exact_mismatches,
        "tolerance_mismatches": tolerance_mismatches,
        "mismatch_samples": samples,
        "sealed_mismatches": sealed_mismatches,
        "max_mismatch_samples_per_column": MAX_MISMATCH_SAMPLES,
    }


__all__ = [
    "EXACT_COLUMNS",
    "MAX_MISMATCH_SAMPLES",
    "ROUNDING_COLUMNS",
    "SUM_COLUMNS",
    "SUM_PRECISION",
    "TARGET_INTERVALS",
    "TOLERANCE_COLUMNS",
    "ResampleError",
    "compare_with_official",
    "decimal_sum",
    "decimals",
    "exact_decimal_context",
    "half_unit",
    "incomplete_buckets",
    "quote_rounding",
    "resample_exact",
    "resample_many",
]
