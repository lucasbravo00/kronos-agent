"""Block-bootstrap intervals for every statistic a step-2 report prints (``docs/DATA.md``).

Each statistic gets a 95 % percentile interval from 1000 resamples of whole blocks drawn with
replacement. Blocks are calendar weeks starting Monday 00:00 UTC for hourly series, or groups of
weeks (4 for the 61-hour Kronos sample, whose contexts overlap for about 3 weeks). The numpy seed
of a statistic is the first 4 bytes of ``sha256("kronos-agent/bootstrap/v1/<key>")`` with
``key = "<study>/<statistic>"``, so every interval is reproducible from its key alone.

Statistics aggregate per-row values that the caller builds BEFORE resampling (returns, price
differences, agreement flags): a resample concatenates whole blocks in draw order and the
statistic only aggregates, so block seams never create values. Paired comparisons pass both
columns in one call and the same blocks are drawn for both.

When the data cannot support an interval the point estimate comes alone with a reason: fewer
than :data:`MIN_BLOCKS` blocks, more than 1 % of resamples non-finite, or a degenerate bootstrap
distribution. A share of exactly 0 or 1 is the exception: it gets a Wilson interval at the
number of blocks. For shares, ``deff = var_bootstrap / (p (1 - p) / n)`` and
``n_eff = n / deff`` are reported.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

WEEK_MS = 604_800_000
MONDAY_OFFSET_MS = 345_600_000  # 1970-01-05T00:00Z, the first Monday after the epoch
DEFAULT_RESAMPLES = 1000
MIN_BLOCKS = 20
MAX_FAILED_SHARE = 0.01
SEED_PREFIX = "kronos-agent/bootstrap/v1/"

METHOD_PERCENTILE = "percentile_block_bootstrap"
METHOD_WILSON = "wilson_blocks"
METHOD_POINT_ONLY = "point_only"


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with its interval (or the reason it has none).

    ``n`` rows in ``n_blocks`` blocks; ``seed`` is :func:`bootstrap_seed` of ``key``;
    ``failed_resamples`` counts non-finite resample statistics; ``method`` is
    ``percentile_block_bootstrap``, ``wilson_blocks`` or ``point_only``; ``deff``/``n_eff`` are
    set for shares with ``0 < p < 1``.
    """

    key: str
    point: float | None
    low: float | None
    high: float | None
    n: int
    n_blocks: int
    n_resamples: int
    confidence: float
    seed: int
    failed_resamples: int
    method: str
    n_eff: float | None
    deff: float | None
    reason: str | None

    @property
    def has_interval(self) -> bool:
        return self.low is not None and self.high is not None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_seed(key: str) -> int:
    """``int.from_bytes(sha256(b"kronos-agent/bootstrap/v1/" + key)[:4], "big")``."""
    if not isinstance(key, str) or not key:
        raise ValueError("a bootstrap key is a non-empty string '<study>/<statistic>'")
    digest = hashlib.sha256((SEED_PREFIX + key).encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big")


def week_blocks(open_ms: np.ndarray, weeks: int = 1) -> np.ndarray:
    """Block id of each epoch-millisecond time: ``((ms - MONDAY_OFFSET_MS) // WEEK_MS) // weeks``.

    Week ``0`` starts on Monday 1970-01-05T00:00Z; ``weeks = 4`` groups four calendar weeks on a
    fixed grid. Floor division keeps times before the offset in negative blocks.
    """
    if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks < 1:
        raise ValueError(f"weeks must be a positive integer, got {weeks!r}")
    values = np.asarray(open_ms, dtype=np.int64)
    return ((values - MONDAY_OFFSET_MS) // WEEK_MS) // weeks


def quantile_levels(confidence: float) -> tuple[float, float]:
    """``((1 - confidence) / 2, (1 + confidence) / 2)`` rounded to 12 decimals.

    ``1 - 0.95`` is ``0.050000000000000044`` in binary; the rounding makes a 95 % interval use
    exactly the 0.025 and 0.975 quantiles.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    return round((1.0 - confidence) / 2.0, 12), round((1.0 + confidence) / 2.0, 12)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def wilson_interval(successes: float, n: float, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval of ``successes / n`` at ``confidence`` (clipped to ``[0, 1]``)."""
    if not n > 0:
        raise ValueError(f"n must be positive, got {n!r}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must lie in [0, n], got {successes!r} of {n!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    z = NormalDist().inv_cdf(quantile_levels(confidence)[1])
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    # at p = 0 (p = 1) the lower (upper) bound is exactly 0 (1); rounding would leave ~1e-17
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def _all_equal(values: np.ndarray) -> bool:
    """True when every value is the same up to a few ulps (means of a constant column that
    differ only by summation rounding count as one value)."""
    low, high = float(values.min()), float(values.max())
    scale = max(abs(low), abs(high), np.finfo(np.float64).tiny)
    return high - low <= 4.0 * np.finfo(np.float64).eps * scale


def _block_rows(blocks: np.ndarray) -> list[np.ndarray]:
    """Row indices of every block (blocks in ascending id order, rows in original order)."""
    ids, inverse = np.unique(blocks, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse, minlength=len(ids)))[:-1]
    return np.split(order, bounds)


def block_bootstrap(
    columns: Sequence[np.ndarray],
    blocks: np.ndarray,
    statistic: Callable[..., float],
    *,
    key: str,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    min_blocks: int = MIN_BLOCKS,
    share: bool = False,
) -> BootstrapResult:
    """Percentile block-bootstrap interval of ``statistic(*columns)``.

    ``columns`` are equally long 1-D arrays of per-row values and ``blocks`` the block id of
    each row. Each resample draws ``n_blocks`` block ids with replacement (one
    ``rng.integers(0, n_blocks, size=(n_resamples, n_blocks))`` call on
    ``np.random.default_rng(bootstrap_seed(key))``) and concatenates the rows of the drawn
    blocks in draw order, each block's rows in their original order. Non-finite statistics are
    failed resamples. The interval is ``np.quantile(..., method="linear")`` at the levels of
    :func:`quantile_levels` (exactly 0.025 and 0.975 at 95 %).

    Point only, with ``reason`` set: fewer than ``min_blocks`` blocks; more than
    :data:`MAX_FAILED_SHARE` failed resamples; all resample statistics equal up to a few ulps
    (degenerate), except that ``share=True`` with a point of exactly 0 or 1 gets a Wilson
    interval at ``n = n_blocks`` (method ``wilson_blocks``). ``share=True`` also sets ``deff``
    and ``n_eff`` when ``0 < point < 1``.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError(f"n_resamples must be a positive integer, got {n_resamples!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    if isinstance(min_blocks, bool) or not isinstance(min_blocks, int) or min_blocks < 1:
        raise ValueError(f"min_blocks must be a positive integer, got {min_blocks!r}")
    arrays = [np.asarray(column) for column in columns]
    if not arrays:
        raise ValueError("block_bootstrap needs at least one column")
    block_ids = np.asarray(blocks)
    n = int(block_ids.shape[0]) if block_ids.ndim == 1 else -1
    if block_ids.ndim != 1 or any(a.ndim != 1 or a.shape[0] != n for a in arrays):
        raise ValueError("columns and blocks must be 1-D arrays of the same length")
    seed = bootstrap_seed(key)

    def result(
        point: float | None,
        *,
        n_blocks: int,
        low: float | None = None,
        high: float | None = None,
        failed: int = 0,
        method: str = METHOD_POINT_ONLY,
        reason: str | None = None,
        n_eff: float | None = None,
        deff: float | None = None,
    ) -> BootstrapResult:
        return BootstrapResult(
            key=key,
            point=point,
            low=low,
            high=high,
            n=n,
            n_blocks=n_blocks,
            n_resamples=n_resamples,
            confidence=float(confidence),
            seed=seed,
            failed_resamples=failed,
            method=method,
            n_eff=n_eff,
            deff=deff,
            reason=reason,
        )

    with np.errstate(all="ignore"):
        point = _finite_or_none(statistic(*arrays))
    rows_by_block = _block_rows(block_ids) if n else []
    n_blocks = len(rows_by_block)
    if point is None:
        return result(None, n_blocks=n_blocks, reason="the statistic is undefined on the sample")
    if n_blocks < min_blocks:
        return result(
            point,
            n_blocks=n_blocks,
            reason=f"fewer than {min_blocks} blocks ({n_blocks}): point estimate only",
        )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    values = np.empty(n_resamples, dtype=np.float64)
    with np.errstate(all="ignore"):
        for index, drawn in enumerate(draws):
            rows = np.concatenate([rows_by_block[block] for block in drawn])
            values[index] = float(statistic(*(array[rows] for array in arrays)))
    finite = np.isfinite(values)
    failed = int(n_resamples - finite.sum())
    if failed > MAX_FAILED_SHARE * n_resamples:
        return result(
            point,
            n_blocks=n_blocks,
            failed=failed,
            reason=(
                f"{failed} of {n_resamples} resamples gave a non-finite statistic "
                f"(more than {MAX_FAILED_SHARE:.0%}): point estimate only"
            ),
        )
    valid = values[finite]
    if share and point in (0.0, 1.0):
        low, high = wilson_interval(point * n_blocks, n_blocks, confidence)
        return result(
            point,
            n_blocks=n_blocks,
            low=low,
            high=high,
            failed=failed,
            method=METHOD_WILSON,
            reason=f"share of exactly {point:g}: Wilson interval at n = {n_blocks} blocks",
        )
    if _all_equal(valid):
        return result(
            point,
            n_blocks=n_blocks,
            failed=failed,
            reason="degenerate bootstrap distribution (every resample equal): no interval",
        )
    lower_level, upper_level = quantile_levels(confidence)
    low = float(np.quantile(valid, lower_level, method="linear"))
    high = float(np.quantile(valid, upper_level, method="linear"))
    n_eff = deff = None
    if share and 0.0 < point < 1.0:
        variance = float(np.var(valid, ddof=1))
        deff = variance / (point * (1.0 - point) / n)
        n_eff = n / deff if deff > 0 else None
    return result(
        point,
        n_blocks=n_blocks,
        low=low,
        high=high,
        failed=failed,
        method=METHOD_PERCENTILE,
        n_eff=n_eff,
        deff=deff,
    )


def autocorrelation(x: np.ndarray, lags: Sequence[int]) -> dict[int, float]:
    """Sample autocorrelation ``r_k = sum((x_t - m)(x_{t+k} - m)) / sum((x_t - m)^2)``.

    The usual (biased) ACF estimator with the overall mean ``m``; ``nan`` when ``x`` is
    constant. Lags are integers in ``[0, len(x))``.
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    centred = values - values.mean() if values.size else values
    denominator = float(np.dot(centred, centred))
    out: dict[int, float] = {}
    for lag in lags:
        if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)):
            raise ValueError(  # noqa: TRY004 - ValueError is this module's contract
                f"lags must be integers, got {lag!r}"
            )
        k = int(lag)
        if not 0 <= k < values.size:
            raise ValueError(f"lag {k} must lie in [0, {values.size})")
        if denominator == 0.0:
            out[k] = math.nan
            continue
        out[k] = float(np.dot(centred[: values.size - k], centred[k:]) / denominator)
    return out


__all__ = [
    "DEFAULT_RESAMPLES",
    "MAX_FAILED_SHARE",
    "METHOD_PERCENTILE",
    "METHOD_POINT_ONLY",
    "METHOD_WILSON",
    "MIN_BLOCKS",
    "MONDAY_OFFSET_MS",
    "WEEK_MS",
    "BootstrapResult",
    "autocorrelation",
    "block_bootstrap",
    "bootstrap_seed",
    "quantile_levels",
    "week_blocks",
    "wilson_interval",
]
