"""Binance vs BingX real-market mismatch (``docs/DATA.md``, study 3; SPEC section 5).

Kronos was pretrained on Binance and will be retrained on Binance, but everything is evaluated and
operated on the BingX real market. Before anything is trained, this study measures how far the
two venues' 1h candles differ, on every hour present in both sources from 2024-04-27T10:00Z (the
start of the BingX real-market history) to the sealed start. These are contemporaneous candles of
two venues: no validation outcome is involved.

Sections, each statistic with a named comparator and a bootstrap seed key, all recorded in
:func:`study_config` before the first run (SPEC rule 7):

- **alignment**: the correlation of the 1h log returns at lags -2..+2 h; the report fails
  (:class:`AlignmentError`) unless lag 0 is the largest, i.e. unless both stores label a candle by
  the same open time.
- **returns**: Pearson and Spearman correlation, ``1 - rho``, std and MAD of
  ``r_bingx - r_binance`` in bp and their ratio to ``std(r_binance)``, where both hours ``t`` and
  ``t - 1h`` exist in both sources; comparator: the lag +-1 h correlations.
- **prices**: ``(bingx - binance) / binance`` in bp for open, high, low and close, and the share
  of hours whose strings are equal as decimals; comparators: one 0.1 USDT tick in bp, the median
  ``|1h return|`` in bp and SPEC section 5's 0.05 bp demo-vs-real median (external).
- **volume**: Spearman correlation of the hourly volumes (comparator: lag +-1 h) and the median
  BingX/Binance ratio (descriptive: the units may differ).
- **amount** (question 9): ``e_t = proxy_t / quote_volume_t - 1`` on ``binance/1h`` and, at the
  Kronos sample hours, ``dz`` between the proxy and the quote volume after the predictor's own
  normalisation, against the same ``dz`` for the volume channel between the two venues.
- **kronos** (``mismatch --kronos``, behind gate 3): signal agreement between the two sources
  with shared seeds, against the agreement of two seeds on the same BingX window, with the
  pre-registered readings (both directions of each paired difference are named; only an
  interval containing 0 is "not distinguishable") and the minimum detectable difference.

Two steps, so that nothing that can fail the report waits for approved compute:
:func:`prepare_study` reads both stores and computes every section without Kronos (the
alignment check included), writing nothing; :func:`write_study` adds the Kronos part, when there
is one, and writes the report, the series and the figures. :func:`build_report` runs both.

Data access: candles only through :func:`kronos_agent.data.store.read_study_frame`, as STRING
frames (``numeric=False``): equality is decided on the exchange's decimal strings and floats come
from ``float()``; every range stops at the sealed start. Each store's ``csv_sha256`` is recorded
as read (its job lock held shared around the reads). Reports only through
:mod:`kronos_agent.report.writer`. This module never runs a fan: the Kronos part takes the
:class:`~kronos_agent.kronos.batch.FanSummary` list of the batch the command ran, or of the fans
it found stored, with that invocation's gate-3 estimate; the gate-3 record of every batch that
computed fans is appended to ``reports/data/mismatch.gate3.jsonl`` before its first fan
(:func:`gate3_record`, :func:`append_gate3_record`), and the report lists every record of the
forecast configuration of its fans.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from ..eval.bootstrap import (
    DEFAULT_RESAMPLES,
    MAX_FAILED_SHARE,
    METHOD_POINT_ONLY,
    MIN_BLOCKS,
    autocorrelation,
    block_bootstrap,
    bootstrap_seed,
    week_blocks,
)
from ..exchange.public_market import Candle
from ..kronos.batch import (
    GATE3_APPROVAL_TOLERANCE,
    GATE3_CPU_SECONDS,
    FanSummary,
    FanTask,
    require_gate3,
)
from ..kronos.decision import DEFAULT_STREAM
from ..kronos.features import FEATURE_COLUMNS, add_amount_proxy
from ..params import load_params
from ..paths import data_dir
from ..report.writer import (
    RECORDS_SUFFIX,
    STUDIES_SUBDIR,
    append_record,
    read_records,
    record_config,
    report_header,
    write_report,
    write_study_series,
)
from ..seeds import SEED_SCHEME
from .forecast_store import SEED_SYMBOL, ForecastStoreError, fan_relpath
from .partition import (
    KRONOS_FIRST_DECISION,
    PARTITION,
    SEALED_START_MS,
    VALIDATION_END,
    require_accessible,
)
from .store import StoreId, float_values, read_meta, read_study_frame, store_read_lock
from .timeutil import HOUR_MS, from_ms, hour_range, parse_utc_hour, to_ms, utc_text
from .windows import DEFAULT_MAX_CONTEXT, WindowError, build_window

STEM = "mismatch"
OVERLAP_START = datetime(2024, 4, 27, 10, tzinfo=UTC)
OVERLAP_START_MS = to_ms(OVERLAP_START)
SOURCES: tuple[str, ...] = ("binance", "bingx")
BINANCE_READ_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "quote_volume")
BINGX_READ_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close")
BP = 10_000.0
RESAMPLES = DEFAULT_RESAMPLES
CONFIDENCE = 0.95
ALIGNMENT_LAGS: tuple[int, ...] = (-2, -1, 0, 1, 2)
VOLUME_LAGS: tuple[int, ...] = (-1, 1)
TICK_USDT = 0.1
DEMO_VS_REAL_MEDIAN_BP = 0.05
OWNER_SEED_FLOOR = 0.88
CONTEXT_HOURS = DEFAULT_MAX_CONTEXT
# the predictor's normalisation at the pinned commit (KronosPredictor.predict / predict_batch)
Z_CLIP = 5.0
Z_EPSILON = 1e-5
VOLUME_INDEX = FEATURE_COLUMNS.index("volume")
AMOUNT_INDEX = FEATURE_COLUMNS.index("amount")
KRONOS_SAMPLE_START = KRONOS_FIRST_DECISION
KRONOS_SAMPLE_END = VALIDATION_END
KRONOS_STEP_HOURS = 61  # coprime with 24: every hour of the day is sampled
KRONOS_FALLBACK_STEP_HOURS = 97
KRONOS_BLOCK_WEEKS = 4
STREAM_FORECAST = DEFAULT_STREAM
STREAM_FLOOR = "mismatch-floor"
FAN_ROLES: dict[tuple[str, str], str] = {
    ("binance", STREAM_FORECAST): "binance",
    ("bingx", STREAM_FORECAST): "bingx",
    ("bingx", STREAM_FLOOR): "bingx_floor",
}
AGREEMENT_PAIRS: dict[str, tuple[str, str]] = {
    "x": ("binance", "bingx"),
    "f": ("bingx", "bingx_floor"),
    "xd": ("binance", "bingx_floor"),
}
AUTOCORRELATION_LAGS: tuple[int, ...] = tuple(range(1, 9))
POWER = 0.80
Z_975 = NormalDist().inv_cdf(0.975)
Z_POWER = NormalDist().inv_cdf(POWER)
PLANNED_MDD_TEXT = {
    "primary": "about 9 points at 291 hours (docs/DATA.md)",
    "fallback": "about 11 points at 183 hours (docs/DATA.md)",
}
# Pre-registered readings: each paired difference names both directions of an interval that
# excludes 0; only an interval containing 0 (or no interval) is "not distinguishable".
READING_DATA_EFFECT = "the data effect is smaller than seed noise"  # A_x - A_f above 0
READING_DATA_EFFECT_LARGER = "the data effect is larger than seed noise"  # A_x - A_f below 0
READING_SOURCES_SHIFT = "the sources shift the signal beyond seed noise"  # A_xd - A_f below 0
READING_SOURCES_ABOVE_FLOOR = (  # A_xd - A_f above 0
    "the cross-source pair agrees more often than the two-seed floor"
)
READING_NOT_DISTINGUISHABLE = "not distinguishable"
KRONOS_NOT_RUN = (
    "not run: not every fan of a pre-registered sample is stored under the current forecast "
    "config; `kronos-agent mismatch --kronos` computes them behind gate 3"
)
SERIES_STEMS: dict[str, str] = {
    "hours": "mismatch_hours",
    "amount": "mismatch_amount",
    "amount_windows": "mismatch_amount_windows",
    "kronos": "mismatch_kronos",
}
AGREEMENT_FIGURE = f"{STEM}_agreement.png"
# The invocation's gate-3 data: its measured estimate (None: the part was rebuilt from stored
# fans, nothing measured), the CPU-hours given with --approved-by-owner, its dropped tasks.
GATE3_KEYS = frozenset({"estimate", "approved_cpu_hours", "dropped"})
# Every batch that computes fans appends one record to reports/data/<GATE3_RECORDS>.jsonl first.
GATE3_RECORDS = f"{STEM}.gate3"
GATE3_RECORD_SCHEMA_VERSION = 1
GATE3_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "study",
        "given_at",
        "design",
        "config_id",
        "approved_cpu_hours",
        "measured_cpu_hours",
        "ratio",
        "tolerance",
        "estimate",
        "dropped",
    }
)
DESIGNS: tuple[str, ...] = ("primary", "fallback")
WEEK_BLOCKS = "calendar weeks starting Monday 00:00 UTC"
KRONOS_BLOCKS = "groups of 4 calendar weeks (Monday 00:00 UTC grid)"


class MismatchError(ValueError):
    """Inputs that are inconsistent, or that leave nothing to measure."""


class AlignmentError(MismatchError):
    """Lag 0 is not the largest return correlation: the two stores are not aligned (the report
    fails, ``docs/DATA.md`` study 3)."""


# -- the recorded statistics ----------------------------------------------------------------------


@dataclass(frozen=True)
class StatSpec:
    """One statistic of the study: its section, definition, comparator and bootstrap blocks.

    ``kind`` names the aggregation; descriptive kinds (``"... (descriptive)"``) get no interval.
    The seed key is ``mismatch/<section>/<name>``.
    """

    section: str
    name: str
    kind: str
    definition: str
    comparator: str
    blocks: str = WEEK_BLOCKS

    @property
    def key(self) -> str:
        return f"{STEM}/{self.section}/{self.name}"

    @property
    def descriptive(self) -> bool:
        return self.kind.endswith("(descriptive)")

    def as_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "name": self.name,
            "kind": self.kind,
            "definition": self.definition,
            "comparator": self.comparator,
            "blocks": None if self.descriptive else self.blocks,
            "seed_key": None if self.descriptive else self.key,
        }


def _lag_name(lag: int) -> str:
    if lag == 0:
        return "lag_0"
    return f"lag_{'plus' if lag > 0 else 'minus'}_{abs(lag)}"


def _lag_text(lag: int) -> str:
    return "t" if lag == 0 else f"t {'+' if lag > 0 else '-'} {abs(lag)}h"


_LAG_COMPARATOR = (
    "lag_0: the aligned correlation (returns/pearson), which must be the largest of the five "
    "lags or the report fails"
)
_LAG0_COMPARATOR = (
    "lag_minus_2 .. lag_plus_2: the same correlation out of alignment; lag 0 must be larger "
    "than every one of them or the report fails"
)
_RETURN_COMPARATOR = (
    "alignment/lag_minus_1 and alignment/lag_plus_1: the correlation of the same returns one hour "
    "out of alignment"
)
_DIFF_COMPARATOR = "binance_std_bp: the standard deviation of r_binance itself"
_RATIO_COMPARATOR = "1: a venue difference as large as the hourly return itself"
_PRICE_COMPARATOR = (
    "tick_bp_median, abs_return_bp_median and SPEC 0.05 bp: one 0.1 USDT tick in bp of the Binance "
    "close, the median |1h return| of Binance in bp, and SPEC section 5's demo-vs-real median "
    "close difference of 0.05 bp (external reference)"
)
_EQUAL_COMPARATOR = (
    "SPEC section 5 (external reference): since 2024-07-26 no 1h candle of the demo account "
    "equals the real one"
)
_VOLUME_COMPARATOR = (
    "spearman_lag_minus_1 and spearman_lag_plus_1: the same correlation one hour out of alignment"
)
_E_COMPARATOR = (
    "0 bp: an exact proxy; its effect on the model input is dz_* against dz_volume_* (the "
    "cross-venue difference of the volume channel)"
)
_DZ_COMPARATOR = (
    "dz_volume_*: the same dz for the volume channel between the Binance and BingX windows at "
    "the same hours"
)
_DZ_SELF = "the comparator of the dz_proxy statistics"
_AGREEMENT_FLOOR = "a_f_share: the agreement of two seeds on the same BingX window"
_FLOOR_EXTERNAL = "the owner's two-seed floor of 88 % (SPEC section 5, external reference)"
_PAIRED_RULE = (
    "0, with the pre-registered reading and the minimum detectable difference at 80 % power"
)
_P_UP_COMPARATOR = "p_up_f_*: the floor pair (the same BingX window under two seeds)"


def _alignment_specs() -> tuple[StatSpec, ...]:
    specs = []
    for lag in ALIGNMENT_LAGS:
        kind = "pearson (point only: the interval is returns/pearson)" if lag == 0 else "pearson"
        specs.append(
            StatSpec(
                "alignment",
                _lag_name(lag),
                kind,
                f"Pearson correlation of r_binance(t) and r_bingx({_lag_text(lag)}) over the hours "
                "whose two returns exist (blocks by t)",
                _LAG0_COMPARATOR if lag == 0 else _LAG_COMPARATOR,
            )
        )
    return tuple(specs)


RETURN_SPECS: tuple[StatSpec, ...] = (
    StatSpec(
        "returns",
        "pearson",
        "pearson",
        "Pearson correlation of the 1h log returns ln(close_t / close_(t-1h)) of the two sources, "
        "where both hours t and t - 1h exist in both sources (equals alignment lag 0)",
        _RETURN_COMPARATOR,
    ),
    StatSpec(
        "returns",
        "spearman",
        "spearman",
        "Spearman correlation (Pearson of average ranks) of the same returns",
        _RETURN_COMPARATOR,
    ),
    StatSpec(
        "returns",
        "diff_std_bp",
        "std",
        "sample standard deviation (ddof 1) of d = r_bingx - r_binance, in bp",
        _DIFF_COMPARATOR,
    ),
    StatSpec(
        "returns",
        "diff_mad_bp",
        "mad",
        "median absolute deviation of d from its median (unscaled), in bp",
        _DIFF_COMPARATOR,
    ),
    StatSpec(
        "returns",
        "binance_std_bp",
        "std",
        "sample standard deviation (ddof 1) of r_binance, in bp",
        "the comparator of diff_std_bp and diff_mad_bp (the scale of the hourly return)",
    ),
    StatSpec(
        "returns",
        "diff_std_ratio",
        "ratio",
        "std(d) / std(r_binance), both columns resampled together",
        _RATIO_COMPARATOR,
    ),
    StatSpec(
        "returns",
        "diff_mad_ratio",
        "ratio",
        "MAD(d) / std(r_binance), both columns resampled together",
        _RATIO_COMPARATOR,
    ),
)


def _price_specs() -> tuple[StatSpec, ...]:
    specs: list[StatSpec] = []
    for field in PRICE_FIELDS:
        text = f"d_{field} = (bingx {field} - binance {field}) / binance {field} x 10000 per hour"
        specs += [
            StatSpec(
                "prices", f"{field}_signed_bp_mean", "mean", f"mean of {text}", _PRICE_COMPARATOR
            ),
            StatSpec(
                "prices",
                f"{field}_signed_bp_median",
                "median",
                f"median of {text}",
                _PRICE_COMPARATOR,
            ),
            StatSpec(
                "prices", f"{field}_abs_bp_mean", "mean", f"mean of |d_{field}|", _PRICE_COMPARATOR
            ),
            StatSpec(
                "prices",
                f"{field}_abs_bp_median",
                "median",
                f"median of |d_{field}|",
                _PRICE_COMPARATOR,
            ),
            StatSpec(
                "prices",
                f"{field}_equal_share",
                "share",
                f"share of hours whose {field} strings are equal as decimals",
                _EQUAL_COMPARATOR,
            ),
        ]
        for label in ("p90", "p99", "max"):
            specs.append(
                StatSpec(
                    "prices",
                    f"{field}_abs_bp_{label}",
                    f"{label} (descriptive)",
                    f"{label} of |d_{field}|",
                    _PRICE_COMPARATOR,
                )
            )
    specs += [
        StatSpec(
            "prices",
            "abs_return_bp_median",
            "median",
            "median |close_t / close_(t-1h) - 1| of Binance in bp over the returns hours",
            "the comparator of the price differences (the typical hourly move)",
        ),
        StatSpec(
            "prices",
            "tick_bp_median",
            "median (descriptive)",
            "median over the hours of 0.1 USDT / binance close x 10000 (one price tick in bp)",
            "the comparator of the price differences (the price grid)",
        ),
    ]
    return tuple(specs)


VOLUME_SPECS: tuple[StatSpec, ...] = (
    StatSpec(
        "volume",
        "spearman",
        "spearman",
        "Spearman correlation of the hourly volumes of the two sources at the same hour",
        _VOLUME_COMPARATOR,
    ),
    StatSpec(
        "volume",
        "spearman_lag_minus_1",
        "spearman",
        "Spearman correlation of volume_binance(t) and volume_bingx(t - 1h)",
        "the comparator of volume/spearman",
    ),
    StatSpec(
        "volume",
        "spearman_lag_plus_1",
        "spearman",
        "Spearman correlation of volume_binance(t) and volume_bingx(t + 1h)",
        "the comparator of volume/spearman",
    ),
    StatSpec(
        "volume",
        "median_ratio",
        "median (descriptive)",
        "median of volume_bingx / volume_binance over the hours with a positive Binance volume "
        "(the units may differ)",
        "none: descriptive, the two venues may count volume in different units",
    ),
)


AMOUNT_SPECS: tuple[StatSpec, ...] = (
    StatSpec(
        "amount",
        "e_bp_mean",
        "mean",
        "mean of e_t = proxy_t / quote_volume_t - 1 in bp, proxy = volume x mean(open, high, low, "
        "close) as the predictor computes it, over the Binance 1h hours of the study range with a "
        "positive quote volume (signed)",
        _E_COMPARATOR,
    ),
    StatSpec("amount", "e_bp_median", "median", "median of e_t in bp (signed)", _E_COMPARATOR),
    *(
        StatSpec(
            "amount",
            f"abs_e_bp_{label}",
            f"{label} (descriptive)",
            f"{label} of |e_t| in bp",
            _E_COMPARATOR,
        )
        for label in ("median", "p90", "p99", "max")
    ),
    StatSpec(
        "amount",
        "dz_proxy_median_mean",
        "mean",
        "mean over the Kronos sample hours of the median over the 512 context rows of "
        "|z(proxy) - z(quote_volume)|, z = the predictor's normalisation of the amount channel",
        _DZ_COMPARATOR,
        KRONOS_BLOCKS,
    ),
    StatSpec(
        "amount",
        "dz_proxy_max_mean",
        "mean",
        "mean over the Kronos sample hours of the max over the 512 rows of the same",
        _DZ_COMPARATOR,
        KRONOS_BLOCKS,
    ),
    StatSpec(
        "amount",
        "dz_volume_median_mean",
        "mean",
        "the same median for the volume channel between the Binance and BingX windows of D",
        _DZ_SELF,
        KRONOS_BLOCKS,
    ),
    StatSpec(
        "amount",
        "dz_volume_max_mean",
        "mean",
        "the same max for the volume channel between the Binance and BingX windows of D",
        _DZ_SELF,
        KRONOS_BLOCKS,
    ),
    StatSpec(
        "amount",
        "dz_median_difference",
        "paired mean difference",
        "dz_proxy median minus dz_volume median at the same hours (both columns resampled "
        "together)",
        "0: the proxy moves the model input as much as the venue does",
        KRONOS_BLOCKS,
    ),
    StatSpec(
        "amount",
        "dz_max_difference",
        "paired mean difference",
        "dz_proxy max minus dz_volume max at the same hours (both columns resampled together)",
        "0: the proxy moves the model input as much as the venue does",
        KRONOS_BLOCKS,
    ),
    StatSpec(
        "amount",
        "dz_proxy_max_max",
        "max (descriptive)",
        "the largest per-hour max of |z(proxy) - z(quote_volume)|",
        _DZ_COMPARATOR,
    ),
    StatSpec(
        "amount",
        "dz_volume_max_max",
        "max (descriptive)",
        "the largest per-hour max of the volume-channel dz",
        _DZ_SELF,
    ),
)


def _kronos_specs() -> tuple[StatSpec, ...]:
    specs = [
        StatSpec(
            "kronos",
            "a_x_share",
            "share",
            "share of D whose binance/forecast and bingx/forecast signals agree (shared seeds: "
            "the data effect only)",
            _AGREEMENT_FLOOR,
            KRONOS_BLOCKS,
        ),
        StatSpec(
            "kronos",
            "a_f_share",
            "share",
            "share of D whose bingx/forecast and bingx/mismatch-floor signals agree (the same "
            "window, other seeds: the seed effect only)",
            _FLOOR_EXTERNAL,
            KRONOS_BLOCKS,
        ),
        StatSpec(
            "kronos",
            "a_xd_share",
            "share",
            "share of D whose binance/forecast and bingx/mismatch-floor signals agree (data and "
            "seed)",
            _AGREEMENT_FLOOR,
            KRONOS_BLOCKS,
        ),
        StatSpec(
            "kronos",
            "a_x_minus_a_f",
            "paired share difference",
            "A_x - A_f on the same hours (one bootstrap call, both columns)",
            _PAIRED_RULE,
            KRONOS_BLOCKS,
        ),
        StatSpec(
            "kronos",
            "a_xd_minus_a_f",
            "paired share difference",
            "A_xd - A_f on the same hours (one bootstrap call, both columns)",
            _PAIRED_RULE,
            KRONOS_BLOCKS,
        ),
    ]
    for pair, (first, second) in AGREEMENT_PAIRS.items():
        comparator = _P_UP_COMPARATOR if pair != "f" else "the comparator of the other p_up pairs"
        specs += [
            StatSpec(
                "kronos",
                f"p_up_{pair}_correlation",
                "pearson",
                f"Pearson correlation of p_up of {first} and {second} over the hours",
                comparator,
                KRONOS_BLOCKS,
            ),
            StatSpec(
                "kronos",
                f"p_up_{pair}_abs_delta_mean",
                "mean",
                f"mean |p_up({first}) - p_up({second})|",
                comparator,
                KRONOS_BLOCKS,
            ),
        ]
    return tuple(specs)


SECTION_SPECS: dict[str, tuple[StatSpec, ...]] = {
    "alignment": _alignment_specs(),
    "returns": RETURN_SPECS,
    "prices": _price_specs(),
    "volume": VOLUME_SPECS,
    "amount": AMOUNT_SPECS,
    "kronos": _kronos_specs(),
}
SECTIONS: tuple[str, ...] = tuple(SECTION_SPECS)


# -- small helpers --------------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    """``float(value)`` when finite, else ``None`` (reports never hold NaN)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _median(values: np.ndarray) -> float:
    return float(np.median(values))


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else math.nan


def _mad(values: np.ndarray) -> float:
    return float(np.median(np.abs(values - np.median(values))))


def _difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(first) - np.mean(second))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two equally long arrays (``nan`` when one is constant or n < 2).

    Sums use numpy's pairwise summation (no BLAS), so the result does not depend on the machine's
    BLAS build; it is clipped to ``[-1, 1]`` against rounding.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise MismatchError("pearson needs two one-dimensional arrays of the same length")
    if a.size < 2:
        return math.nan
    am = a - np.mean(a)
    bm = b - np.mean(b)
    sxx = float(np.sum(am * am))
    syy = float(np.sum(bm * bm))
    if not (sxx > 0.0 and syy > 0.0):
        return math.nan
    r = float(np.sum(am * bm)) / math.sqrt(sxx * syy)
    return max(-1.0, min(1.0, r))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (``scipy.stats.rankdata`` default)."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if not n:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ordered = x[order]
    starts_group = np.empty(n, dtype=bool)
    starts_group[0] = True
    starts_group[1:] = ordered[1:] != ordered[:-1]
    group = np.cumsum(starts_group) - 1
    starts = np.flatnonzero(starts_group)
    ends = np.append(starts[1:], n)
    mean_rank = (starts + ends - 1) / 2.0 + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = mean_rank[group]
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation: :func:`pearson` of :func:`average_ranks`."""
    return pearson(average_ranks(x), average_ranks(y))


def predictor_z(context: pd.DataFrame) -> np.ndarray:
    """The pinned predictor's normalisation of a context: ``(K, 6)`` float32, channels in
    :data:`~kronos_agent.kronos.features.FEATURE_COLUMNS` order.

    Exactly ``KronosPredictor.predict``/``predict_batch`` at the vendored commit: the six channels
    as float32, per-column mean and population std over every row of the window,
    ``(x - mean) / (std + 1e-5)``, clipped at +-5.
    """
    if not isinstance(context, pd.DataFrame):
        raise MismatchError(f"expected a context DataFrame, got {type(context).__name__}")
    x = context[list(FEATURE_COLUMNS)].values.astype(np.float32)
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    z = (x - x_mean) / (x_std + Z_EPSILON)
    return np.clip(z, -Z_CLIP, Z_CLIP)


_DESCRIPTIVE_FUNCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "median": _median,
    "p90": lambda values: float(np.quantile(values, 0.90, method="linear")),
    "p99": lambda values: float(np.quantile(values, 0.99, method="linear")),
    "max": lambda values: float(np.max(values)),
}


def _ranges(hours_ms: Sequence[int] | np.ndarray, step_ms: int = HOUR_MS) -> list[list[str]]:
    """Merged ``[start, end)`` UTC text ranges of sorted open times on a ``step_ms`` grid."""
    merged: list[list[int]] = []
    for value in np.asarray(hours_ms, dtype=np.int64).tolist():
        if merged and merged[-1][1] == value:
            merged[-1][1] = value + step_ms
        else:
            merged.append([value, value + step_ms])
    return [[utc_text(a), utc_text(b)] for a, b in merged]


def _exclusion(reason: str, hours_ms: np.ndarray) -> dict[str, Any]:
    return {"reason": reason, "hours": int(hours_ms.size), "ranges": _ranges(hours_ms)}


def _hour_values(hours: Iterable[datetime | str]) -> np.ndarray:
    """Decision hours as epoch milliseconds, strictly increasing, none in the sealed segment."""
    if isinstance(hours, (str, datetime)):
        raise MismatchError("hours must be an iterable of decision hours")
    try:
        values = [to_ms(parse_utc_hour(hour)) for hour in hours]
    except ValueError as exc:
        raise MismatchError(str(exc)) from None
    if not values:
        raise MismatchError("no decision hour to study")
    array = np.array(values, dtype=np.int64)
    if (np.diff(array) <= 0).any():
        raise MismatchError("decision hours must be strictly increasing")
    require_accessible(from_ms(int(array[-1])))
    return array


def _lag_pairs(opens: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indexes ``(i, j)`` with ``opens[j] == opens[i] + lag h`` (``opens`` sorted)."""
    if not opens.size:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty.copy()
    target = opens + lag * HOUR_MS
    position = np.searchsorted(opens, target, side="left")
    clipped = np.minimum(position, opens.size - 1)
    valid = (position < opens.size) & (opens[clipped] == target)
    return np.flatnonzero(valid), position[valid]


# -- bootstrap bookkeeping ------------------------------------------------------------------------


def _point_only(key: str, point: float | None, n: int, n_blocks: int, reason: str) -> dict:
    """A result with the keys of ``BootstrapResult.as_dict()`` and no interval."""
    point = _finite(point)
    if point is None and reason is None:
        reason = "the statistic is undefined on the sample"
    return {
        "key": key,
        "point": point,
        "low": None,
        "high": None,
        "n": int(n),
        "n_blocks": int(n_blocks),
        "n_resamples": 0,
        "confidence": CONFIDENCE,
        "seed": bootstrap_seed(key),
        "failed_resamples": 0,
        "method": METHOD_POINT_ONLY,
        "n_eff": None,
        "deff": None,
        "reason": reason,
    }


def _has_interval(result: Mapping[str, Any]) -> bool:
    return result.get("low") is not None and result.get("high") is not None


class _Section:
    """Results of one section, checked against the recorded :class:`StatSpec` table."""

    def __init__(self, section: str) -> None:
        self.section = section
        self.specs = {spec.name: spec for spec in SECTION_SPECS[section]}
        self.statistics: dict[str, dict[str, Any]] = {}
        self.descriptive: dict[str, float | None] = {}

    def _spec(self, name: str, *, descriptive: bool) -> StatSpec:
        spec = self.specs[name]  # KeyError: a statistic that was not recorded
        if spec.descriptive != descriptive:
            raise MismatchError(f"{spec.key} is recorded as {spec.kind}")
        return spec

    def add(
        self,
        name: str,
        columns: Sequence[np.ndarray],
        blocks: np.ndarray,
        statistic: Callable[..., float],
        *,
        share: bool = False,
    ) -> dict[str, Any]:
        """Bootstrap ``statistic(*columns)`` with the blocks given (one call: paired columns)."""
        spec = self._spec(name, descriptive=False)
        arrays = [np.asarray(column, dtype=np.float64) for column in columns]
        if not arrays[0].size:
            result = _point_only(spec.key, None, 0, 0, "no rows in this sample")
        else:
            result = block_bootstrap(
                arrays,
                np.asarray(blocks),
                statistic,
                key=spec.key,
                n_resamples=RESAMPLES,
                confidence=CONFIDENCE,
                share=share,
            ).as_dict()
        self.statistics[name] = result
        return result

    def point_only(
        self, name: str, point: float | None, n: int, n_blocks: int, reason: str
    ) -> dict:
        spec = self._spec(name, descriptive=False)
        result = _point_only(spec.key, point, n, n_blocks, reason)
        self.statistics[name] = result
        return result

    def describe(self, name: str, values: np.ndarray) -> float | None:
        spec = self._spec(name, descriptive=True)
        array = np.asarray(values, dtype=np.float64)
        label = spec.kind.split(" ")[0]
        value = _finite(_DESCRIPTIVE_FUNCTIONS[label](array)) if array.size else None
        self.descriptive[name] = value
        return value

    def body(self) -> dict[str, Any]:
        names = [*self.statistics, *self.descriptive]
        return {
            "statistics": self.statistics,
            "descriptive": self.descriptive,
            "comparators": {name: self.specs[name].comparator for name in names},
            "seeds": {f"{self.section}/{name}": self.specs[name].key for name in self.statistics},
            "statistics_with_intervals": sum(_has_interval(r) for r in self.statistics.values()),
        }


# -- inputs ---------------------------------------------------------------------------------------


def _check_open_ms(frame: pd.DataFrame, label: str) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame):
        raise MismatchError(f"{label}: expected a DataFrame, got {type(frame).__name__}")
    if "open_ms" not in frame.columns or not pd.api.types.is_integer_dtype(frame["open_ms"].dtype):
        raise MismatchError(f"{label}: needs an integer open_ms column")
    opens = frame["open_ms"].to_numpy(dtype=np.int64)
    if opens.size and ((np.diff(opens) <= 0).any() or (opens % HOUR_MS).any()):
        raise MismatchError(f"{label}: open_ms must be exact hours in strictly increasing order")
    return opens


def _string_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise MismatchError(f"{label}: missing columns {missing}")
    for name in columns:
        if len(frame) and pd.api.types.infer_dtype(frame[name], skipna=False) != "string":
            raise MismatchError(
                f"{label}: column {name} must hold the exchange's decimal strings (a "
                "read_study_frame(numeric=False) frame)"
            )


def _decimal_equal(first: Sequence[str], second: Sequence[str]) -> np.ndarray:
    try:
        return np.array(
            [Decimal(a) == Decimal(b) for a, b in zip(first, second, strict=True)], dtype=bool
        )
    except InvalidOperation as exc:
        raise MismatchError(f"a price is not a decimal string: {exc}") from None


def joined_hours(binance_1h: pd.DataFrame, bingx_1h: pd.DataFrame) -> pd.DataFrame:
    """Every hour present in both sources, values as ``float()`` and equality as decimals.

    ``binance_1h`` and ``bingx_1h`` are STRING study frames (``read_study_frame(numeric=False)``)
    already limited to the study range: Binance with ``open, high, low, close, volume,
    quote_volume``, BingX with ``open, high, low, close, volume``. The result has one row per
    hour present in both (an inner join on ``open_ms``, ascending, RangeIndex): ``open_ms``,
    ``binance_<column>`` and ``bingx_<column>`` as float64, and ``equal_<price>`` (the two strings
    equal as decimals, so ``"60000.10"`` equals ``"60000.1"``) for open, high, low and close.
    Nothing is filled: an hour missing in one source is not in the result. Prices must be > 0.
    """
    open_b = _check_open_ms(binance_1h, "binance 1h frame")
    open_x = _check_open_ms(bingx_1h, "bingx 1h frame")
    _string_columns(binance_1h, BINANCE_READ_COLUMNS, "binance 1h frame")
    _string_columns(bingx_1h, BINGX_READ_COLUMNS, "bingx 1h frame")
    common, index_b, index_x = np.intersect1d(
        open_b, open_x, assume_unique=True, return_indices=True
    )
    data: dict[str, Any] = {"open_ms": common.astype(np.int64)}
    for name in BINANCE_READ_COLUMNS:
        data[f"binance_{name}"] = float_values(binance_1h[name].to_numpy(dtype=object)[index_b])
    for name in BINGX_READ_COLUMNS:
        data[f"bingx_{name}"] = float_values(bingx_1h[name].to_numpy(dtype=object)[index_x])
    for name in PRICE_FIELDS:
        data[f"equal_{name}"] = _decimal_equal(
            binance_1h[name].to_numpy(dtype=object)[index_b],
            bingx_1h[name].to_numpy(dtype=object)[index_x],
        )
    joined = pd.DataFrame(data)
    prices = joined[[f"{s}_{p}" for s in SOURCES for p in PRICE_FIELDS]].to_numpy()
    if prices.size and not (prices > 0).all():
        raise MismatchError("prices must be positive")
    return joined


def _returns_table(joined: pd.DataFrame) -> pd.DataFrame:
    """1h returns at every joined hour ``t`` whose ``t - 1h`` is also joined (both sources)."""
    opens = joined["open_ms"].to_numpy(dtype=np.int64)
    later = np.flatnonzero(np.diff(opens) == HOUR_MS) + 1
    earlier = later - 1
    close_b = joined["binance_close"].to_numpy(dtype=np.float64)
    close_x = joined["bingx_close"].to_numpy(dtype=np.float64)
    ratio_b = close_b[later] / close_b[earlier]
    return pd.DataFrame(
        {
            "open_ms": opens[later],
            "r_binance": np.log(ratio_b),
            "r_bingx": np.log(close_x[later] / close_x[earlier]),
            "simple_binance": ratio_b - 1.0,
        }
    )


def _require_rows(joined: pd.DataFrame) -> None:
    required = ["open_ms", *(f"binance_{c}" for c in BINANCE_READ_COLUMNS)]
    required += [f"bingx_{c}" for c in BINGX_READ_COLUMNS]
    required += [f"equal_{p}" for p in PRICE_FIELDS]
    if not isinstance(joined, pd.DataFrame):
        raise MismatchError(f"expected the joined_hours DataFrame, got {type(joined).__name__}")
    missing = [name for name in required if name not in joined.columns]
    if missing:
        raise MismatchError(f"the joined frame lacks {missing}: build it with joined_hours")
    if joined.empty:
        raise MismatchError("no hour is present in both sources: nothing to measure")
    _check_open_ms(joined, "joined frame")


# -- alignment, returns, prices, volume -----------------------------------------------------------


def alignment(joined: pd.DataFrame) -> dict[str, Any]:
    """The correlation of the 1h log returns at lags -2..+2 h; fails unless lag 0 is the largest.

    Lag ``k`` pairs ``r_binance(t)`` with ``r_bingx(t + k h)`` (``k > 0``: BingX later), over the
    hours where both returns exist (a return exists at ``t`` when ``t`` and ``t - 1h`` are present
    in both sources); blocks are the calendar weeks of ``t``. Lags +-1 and +-2 get intervals;
    lag 0 is ``returns/pearson`` (same rows), whose interval is reported there.
    :class:`AlignmentError` unless the lag-0 correlation is strictly larger than every other one.
    """
    _require_rows(joined)
    table = _returns_table(joined)
    if len(table) < 3:
        raise MismatchError(f"{len(table)} hourly returns: too few to check the alignment")
    opens = table["open_ms"].to_numpy(dtype=np.int64)
    r_b = table["r_binance"].to_numpy()
    r_x = table["r_bingx"].to_numpy()
    section = _Section("alignment")
    points: dict[int, float | None] = {}
    pairs: dict[str, int] = {}
    for lag in ALIGNMENT_LAGS:
        first, second = _lag_pairs(opens, lag)
        name = _lag_name(lag)
        x, y = r_b[first], r_x[second]
        blocks = week_blocks(opens[first])
        if lag == 0:
            result = section.point_only(
                name,
                pearson(x, y) if x.size else None,
                x.size,
                int(np.unique(blocks).size),
                "lag 0 is returns/pearson (the same rows): its interval is reported there",
            )
        else:
            result = section.add(name, [x, y], blocks, pearson)
        points[lag] = result["point"]
        pairs[name] = int(x.size)
    aligned = points[0]
    others = {lag: value for lag, value in points.items() if lag != 0}
    largest = aligned is not None and all(v is None or aligned > v for v in others.values())
    if not largest:
        shown = ", ".join(f"{_lag_name(lag)} {value}" for lag, value in points.items())
        raise AlignmentError(
            f"the 1h return correlation is not largest at lag 0 ({shown}): the two stores are "
            "not aligned on candle open times; the mismatch report fails (docs/DATA.md study 3)"
        )
    return {
        **section.body(),
        "convention": "lag k pairs r_binance(t) with r_bingx(t + k h); k > 0: BingX later",
        "lag0_is_largest": True,
        "pairs": pairs,
    }


def return_statistics(joined: pd.DataFrame) -> dict[str, Any]:
    """Correlation and difference of the 1h log returns where ``t`` and ``t - 1h`` exist in both
    sources: Pearson, Spearman, ``1 - rho`` (derived), std and MAD of ``r_bingx - r_binance`` in
    bp and their ratio to ``std(r_binance)``; blocks are the calendar weeks of ``t``."""
    _require_rows(joined)
    table = _returns_table(joined)
    opens = table["open_ms"].to_numpy(dtype=np.int64)
    r_b = table["r_binance"].to_numpy()
    r_x = table["r_bingx"].to_numpy()
    blocks = week_blocks(opens)
    section = _Section("returns")
    rho = section.add("pearson", [r_b, r_x], blocks, pearson)
    rank_rho = section.add("spearman", [r_b, r_x], blocks, spearman)
    diff_bp = (r_x - r_b) * BP
    binance_bp = r_b * BP
    section.add("diff_std_bp", [diff_bp], blocks, _std)
    section.add("diff_mad_bp", [diff_bp], blocks, _mad)
    section.add("binance_std_bp", [binance_bp], blocks, _std)
    section.add("diff_std_ratio", [diff_bp, binance_bp], blocks, lambda d, r: _std(d) / _std(r))
    section.add("diff_mad_ratio", [diff_bp, binance_bp], blocks, lambda d, r: _mad(d) / _std(r))
    lost = np.setdiff1d(joined["open_ms"].to_numpy(dtype=np.int64), opens)
    return {
        **section.body(),
        "derived": {
            "one_minus_pearson": _one_minus(rho),
            "one_minus_spearman": _one_minus(rank_rho),
        },
        "samples": {
            "joined_hours": len(joined),
            "used": len(table),
            "lost": len(joined) - len(table),
            "lost_reason": "the hour t - 1h is missing in a source (no return at t)",
            "lost_ranges": _ranges(lost),
        },
    }


def _one_minus(result: Mapping[str, Any]) -> dict[str, Any]:
    """``1 - rho`` from rho's result: the interval is the transform of rho's (not a new
    statistic, not counted among the intervals)."""
    point, low, high = result.get("point"), result.get("low"), result.get("high")
    return {
        "derived_from": result.get("key"),
        "point": None if point is None else 1.0 - point,
        "low": None if high is None else 1.0 - high,
        "high": None if low is None else 1.0 - low,
        "note": "1 - rho; its interval is the transform of rho's interval",
    }


def price_differences(joined: pd.DataFrame) -> dict[str, Any]:
    """``(bingx - binance) / binance`` in bp for open, high, low and close at the same hour.

    Per price: signed mean and median, mean and median of the absolute value (intervals), p90,
    p99 and max of the absolute value (descriptive) and the share of hours whose strings are
    equal as decimals (a share interval). Comparators: one 0.1 USDT tick in bp (median over the
    hours), the median ``|1h return|`` of Binance in bp (interval) and SPEC section 5's 0.05 bp
    demo-vs-real median (external reference). Blocks: calendar weeks of the hour.
    """
    _require_rows(joined)
    opens = joined["open_ms"].to_numpy(dtype=np.int64)
    blocks = week_blocks(opens)
    section = _Section("prices")
    for field in PRICE_FIELDS:
        binance = joined[f"binance_{field}"].to_numpy(dtype=np.float64)
        bingx = joined[f"bingx_{field}"].to_numpy(dtype=np.float64)
        signed = (bingx - binance) / binance * BP
        absolute = np.abs(signed)
        section.add(f"{field}_signed_bp_mean", [signed], blocks, _mean)
        section.add(f"{field}_signed_bp_median", [signed], blocks, _median)
        section.add(f"{field}_abs_bp_mean", [absolute], blocks, _mean)
        section.add(f"{field}_abs_bp_median", [absolute], blocks, _median)
        equal = joined[f"equal_{field}"].to_numpy(dtype=bool).astype(np.float64)
        section.add(f"{field}_equal_share", [equal], blocks, _mean, share=True)
        for label in ("p90", "p99", "max"):
            section.describe(f"{field}_abs_bp_{label}", absolute)
    table = _returns_table(joined)
    moves = np.abs(table["simple_binance"].to_numpy()) * BP
    section.add(
        "abs_return_bp_median",
        [moves],
        week_blocks(table["open_ms"].to_numpy(dtype=np.int64)),
        _median,
    )
    close = joined["binance_close"].to_numpy(dtype=np.float64)
    section.describe("tick_bp_median", TICK_USDT / close * BP)
    return {
        **section.body(),
        "external_references": {"demo_vs_real_median_bp": DEMO_VS_REAL_MEDIAN_BP},
        "samples": {"hours": len(joined), "returns_hours": len(table)},
    }


def volume_statistics(joined: pd.DataFrame) -> dict[str, Any]:
    """Spearman correlation of the hourly volumes at the same hour, against lag +-1 h, and the
    median BingX/Binance volume ratio (descriptive: the units may differ)."""
    _require_rows(joined)
    opens = joined["open_ms"].to_numpy(dtype=np.int64)
    volume_b = joined["binance_volume"].to_numpy(dtype=np.float64)
    volume_x = joined["bingx_volume"].to_numpy(dtype=np.float64)
    section = _Section("volume")
    section.add("spearman", [volume_b, volume_x], week_blocks(opens), spearman)
    for lag in VOLUME_LAGS:
        first, second = _lag_pairs(opens, lag)
        section.add(
            f"spearman_{_lag_name(lag)}",
            [volume_b[first], volume_x[second]],
            week_blocks(opens[first]),
            spearman,
        )
    positive = volume_b > 0
    section.describe("median_ratio", volume_x[positive] / volume_b[positive])
    zero = opens[~positive]
    return {
        **section.body(),
        "samples": {
            "hours": len(joined),
            "ratio_hours": int(positive.sum()),
            "zero_binance_volume_hours": int(zero.size),
            "ratio_exclusions": (
                [_exclusion("zero Binance volume (median_ratio undefined)", zero)]
                if zero.size
                else []
            ),
        },
    }


# -- the amount proxy -----------------------------------------------------------------------------


def _candles(frame: pd.DataFrame) -> list[Candle]:
    values = [frame[name].to_numpy(dtype=object) for name in BINGX_READ_COLUMNS]
    return [
        Candle(int(open_ms), str(o), str(h), str(lo), str(c), str(v))
        for open_ms, o, h, lo, c, v in zip(
            frame["open_ms"].to_numpy(dtype=np.int64), *values, strict=True
        )
    ]


def amount_proxy(
    binance_1h: pd.DataFrame, *, sample_hours: Iterable[datetime | str], bingx_1h: pd.DataFrame
) -> dict[str, Any]:
    """Question 9: how far the ``amount`` proxy is from Binance's quote volume, and what it does
    to the model input.

    ``binance_1h``/``bingx_1h`` are STRING 1h study frames (Binance with ``quote_volume``).
    ``e_t = proxy_t / quote_volume_t - 1`` in bp over the Binance hours of the study range
    (``[2024-04-27T10:00Z, sealed start)``) with a positive quote volume, ``proxy = volume x
    mean(open, high, low, close)`` evaluated as the predictor does
    (:func:`~kronos_agent.kronos.features.add_amount_proxy`); blocks: calendar weeks. At every
    ``sample_hours`` ``D`` whose 512-candle windows are complete in both sources:
    ``dz = |z(proxy) - z(quote_volume)|`` over the 512 rows with :func:`predictor_z` (median and
    max per hour), and the comparator ``dz`` of the volume channel between the Binance and BingX
    windows; blocks: 4 calendar weeks (the contexts overlap for about 3 weeks). Returns the
    section body plus ``series`` (per hour) and ``window_series`` (per sample hour).
    """
    open_b = _check_open_ms(binance_1h, "binance 1h frame")
    open_x = _check_open_ms(bingx_1h, "bingx 1h frame")
    _string_columns(binance_1h, BINANCE_READ_COLUMNS, "binance 1h frame")
    _string_columns(bingx_1h, BINGX_READ_COLUMNS, "bingx 1h frame")
    requested = _hour_values(sample_hours)
    section = _Section("amount")

    in_range = (open_b >= OVERLAP_START_MS) & (open_b < SEALED_START_MS)
    floats = pd.DataFrame(
        {
            name: float_values(binance_1h[name].to_numpy(dtype=object)[in_range])
            for name in BINGX_READ_COLUMNS
        }
    )
    proxy = add_amount_proxy(floats)["amount"].to_numpy(dtype=np.float64)
    quote = float_values(binance_1h["quote_volume"].to_numpy(dtype=object)[in_range])
    hours_ms = open_b[in_range]
    positive = quote > 0
    e_bp = (proxy[positive] / quote[positive] - 1.0) * BP
    e_blocks = week_blocks(hours_ms[positive])
    section.add("e_bp_mean", [e_bp], e_blocks, _mean)
    section.add("e_bp_median", [e_bp], e_blocks, _median)
    for label in ("median", "p90", "p99", "max"):
        section.describe(f"abs_e_bp_{label}", np.abs(e_bp))
    hour_series = pd.DataFrame(
        {
            "open_ms": hours_ms,
            "proxy": proxy,
            "quote_volume": quote,
            "e_bp": np.where(positive, (proxy / np.where(positive, quote, 1.0) - 1.0) * BP, np.nan),
        }
    )

    candles_b = _candles(binance_1h)
    candles_x = _candles(bingx_1h)
    quotes = dict(
        zip(
            open_b.tolist(), binance_1h["quote_volume"].to_numpy(dtype=object).tolist(), strict=True
        )
    )
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    span = CONTEXT_HOURS * HOUR_MS
    for decision_ms in requested.tolist():
        lo_b, hi_b = np.searchsorted(open_b, [decision_ms - span, decision_ms])
        lo_x, hi_x = np.searchsorted(open_x, [decision_ms - span, decision_ms])
        missing = [
            title
            for title, count in (("Binance", hi_b - lo_b), ("BingX real market", hi_x - lo_x))
            if count != CONTEXT_HOURS
        ]
        if missing:
            skipped.append(
                {
                    "decision_hour": utc_text(decision_ms),
                    "reason": f"incomplete 512-candle window in {' and '.join(missing)}",
                }
            )
            continue
        hour = from_ms(decision_ms)
        window_b = candles_b[lo_b:hi_b]
        try:
            proxied = build_window(window_b, hour, source="binance", horizon=1)
            quoted = build_window(
                window_b,
                hour,
                source="binance",
                horizon=1,
                amount_policy="binance_quote_volume",
                quote_volumes={c.open_ms: str(quotes[c.open_ms]) for c in window_b},
            )
            bingx = build_window(candles_x[lo_x:hi_x], hour, source="bingx", horizon=1)
        except WindowError as exc:
            raise MismatchError(f"decision hour {utc_text(decision_ms)}: {exc}") from exc
        z_proxy = predictor_z(proxied.context)
        z_quote = predictor_z(quoted.context)
        z_bingx = predictor_z(bingx.context)
        dz_proxy = np.abs(z_proxy[:, AMOUNT_INDEX] - z_quote[:, AMOUNT_INDEX]).astype(np.float64)
        dz_volume = np.abs(z_proxy[:, VOLUME_INDEX] - z_bingx[:, VOLUME_INDEX]).astype(np.float64)
        records.append(
            {
                "decision_ms": decision_ms,
                "dz_proxy_median": float(np.median(dz_proxy)),
                "dz_proxy_max": float(np.max(dz_proxy)),
                "dz_volume_median": float(np.median(dz_volume)),
                "dz_volume_max": float(np.max(dz_volume)),
            }
        )
    window_series = pd.DataFrame.from_records(
        records,
        columns=[
            "decision_ms",
            "dz_proxy_median",
            "dz_proxy_max",
            "dz_volume_median",
            "dz_volume_max",
        ],
    ).astype({"decision_ms": np.int64})
    decision = window_series["decision_ms"].to_numpy(dtype=np.int64)
    blocks = week_blocks(decision, weeks=KRONOS_BLOCK_WEEKS)

    def column(name: str) -> np.ndarray:
        return window_series[name].to_numpy(dtype=np.float64)

    section.add("dz_proxy_median_mean", [column("dz_proxy_median")], blocks, _mean)
    section.add("dz_proxy_max_mean", [column("dz_proxy_max")], blocks, _mean)
    section.add("dz_volume_median_mean", [column("dz_volume_median")], blocks, _mean)
    section.add("dz_volume_max_mean", [column("dz_volume_max")], blocks, _mean)
    section.add(
        "dz_median_difference",
        [column("dz_proxy_median"), column("dz_volume_median")],
        blocks,
        _difference,
    )
    section.add(
        "dz_max_difference", [column("dz_proxy_max"), column("dz_volume_max")], blocks, _difference
    )
    section.describe("dz_proxy_max_max", column("dz_proxy_max"))
    section.describe("dz_volume_max_max", column("dz_volume_max"))
    zero = hours_ms[~positive]
    return {
        **section.body(),
        "normalisation": "the pinned predictor's: float32, per-window mean and population std, "
        "(x - mean) / (std + 1e-5), clipped at +-5",
        "samples": {
            "hours": {
                "range": [utc_text(OVERLAP_START_MS), utc_text(SEALED_START_MS)],
                "binance_hours": int(hours_ms.size),
                "used": int(positive.sum()),
                "exclusions": [_exclusion("zero quote volume (e_t undefined)", zero)]
                if zero.size
                else [],
            },
            "sample_hours": {
                "requested": int(requested.size),
                "used": len(window_series),
                "skipped": len(skipped),
                "skipped_hours": skipped,
            },
        },
        "series": hour_series,
        "window_series": window_series,
    }


# -- the Kronos signal agreement ------------------------------------------------------------------


def kronos_sample_hours() -> list[datetime]:
    """``D_k = 2024-07-01T00:00Z + 61 k h`` while ``D_k < 2026-07-10T08:00Z`` (291 hours)."""
    return hour_range(KRONOS_SAMPLE_START, KRONOS_SAMPLE_END, KRONOS_STEP_HOURS)


def kronos_fallback_hours() -> list[datetime]:
    """The fallback sample if the owner declines the compute: a 97-hour step (183 hours)."""
    return hour_range(KRONOS_SAMPLE_START, KRONOS_SAMPLE_END, KRONOS_FALLBACK_STEP_HOURS)


def kronos_tasks(hours: Iterable[datetime | str]) -> list[FanTask]:
    """The three fans of every hour: ``binance/forecast``, ``bingx/forecast`` (the same seeds)
    and ``bingx/mismatch-floor`` (other seeds), in that order, hour by hour."""
    tasks: list[FanTask] = []
    for decision_ms in _hour_values(hours).tolist():
        hour = from_ms(decision_ms)
        tasks.extend(FanTask(source, hour, stream) for source, stream in FAN_ROLES)
    return tasks


def _interval_position(result: Mapping[str, Any]) -> str:
    if not _has_interval(result):
        return "no interval"
    if result["low"] > 0.0:
        return "entirely above 0"
    if result["high"] < 0.0:
        return "entirely below 0"
    return "contains 0"


def minimum_detectable_difference(result: Mapping[str, Any]) -> dict[str, Any]:
    """The difference detectable at 80 % power and two-sided 5 % from a 95 % interval:
    ``(z_0.975 + z_0.80) x SE`` with ``SE = (high - low) / (2 z_0.975)`` (the block-bootstrap
    standard error read from the interval). ``value`` is ``None`` without an interval."""
    formula = "(z_0.975 + z_0.80) x (high - low) / (2 z_0.975) of the 95 % interval"
    if not _has_interval(result):
        return {"value": None, "standard_error": None, "formula": formula, "reason": "no interval"}
    standard_error = (float(result["high"]) - float(result["low"])) / (2.0 * Z_975)
    return {
        "value": (Z_975 + Z_POWER) * standard_error,
        "standard_error": standard_error,
        "formula": formula,
        "reason": None,
    }


def _reading(result: Mapping[str, Any], *, above: str, below: str) -> dict:
    """The pre-registered reading of a paired difference: ``above`` or ``below`` for an interval
    entirely on that side of 0, "not distinguishable" only for an interval containing 0 or no
    interval; always with the minimum detectable difference."""
    position = _interval_position(result)
    text = {"entirely above 0": above, "entirely below 0": below}.get(
        position, READING_NOT_DISTINGUISHABLE
    )
    return {
        "reading": text,
        "interval_position": position,
        "minimum_detectable_difference": minimum_detectable_difference(result),
    }


def _summary_hour(summary: FanSummary) -> int:
    try:
        return to_ms(parse_utc_hour(summary.decision_hour))
    except ValueError as exc:
        raise MismatchError(f"fan summary with an invalid decision hour: {exc}") from None


def _summary_config_id(summary: FanSummary, decision_ms: int) -> str:
    parts = str(summary.relpath).split("/")
    config = parts[2] if len(parts) == 5 else ""
    try:
        expected = fan_relpath(summary.source, config, summary.stream, from_ms(decision_ms))
    except (ForecastStoreError, ValueError) as exc:
        raise MismatchError(
            f"fan {summary.relpath!r} is not in the forecast layout: {exc}"
        ) from None
    if expected != summary.relpath:
        raise MismatchError(f"fan {summary.relpath!r} is not where its summary says ({expected})")
    return config


def _design(hours_ms: set[int]) -> str:
    primary = {to_ms(hour) for hour in kronos_sample_hours()}
    fallback = {to_ms(hour) for hour in kronos_fallback_hours()}
    if hours_ms <= primary:
        return "primary"
    if hours_ms <= fallback:
        return "fallback"
    outside = sorted(hours_ms - primary)
    raise MismatchError(
        f"fan hours outside both pre-registered samples (first: {utc_text(outside[0])}); the "
        "Kronos agreement uses kronos_sample_hours() or kronos_fallback_hours()"
    )


def kronos_agreement(summaries: Iterable[FanSummary]) -> dict[str, Any]:
    """Signal agreement of the three fans of every sample hour, against the two-seed floor.

    ``summaries`` are the :class:`~kronos_agent.kronos.batch.FanSummary` items of one batch of
    :func:`kronos_tasks`: one forecast configuration (the ``config_id`` of every file), hours of
    one pre-registered sample (``primary`` 61-hour or ``fallback`` 97-hour step), each fan once.
    An hour enters only with its three fans (both windows were complete); the floor fan must read
    the same BingX window as the forecast fan. Per hour ``A_x = agree(binance, bingx)``,
    ``A_f = agree(bingx, bingx-floor)``, ``A_xd = agree(binance, bingx-floor)``; the primary
    statistics are the paired differences ``A_x - A_f`` and ``A_xd - A_f`` (one bootstrap call
    each, blocks of 4 calendar weeks) with the pre-registered readings and the minimum detectable
    difference; ``p_up`` correlation and mean ``|delta p_up|`` of the three pairs (the floor pair
    is the comparator); lag 1-8 autocorrelation of the agreement flags; the owner's 88 % as an
    external reference.
    """
    if isinstance(summaries, FanSummary):
        raise MismatchError("summaries must be an iterable of FanSummary")
    by_hour: dict[int, dict[str, FanSummary]] = {}
    config_ids: set[str] = set()
    computed = reused = 0
    elapsed = elapsed_all = 0.0
    for summary in summaries:
        if not isinstance(summary, FanSummary):
            raise MismatchError(f"expected FanSummary items, got {type(summary).__name__}")
        role = FAN_ROLES.get((summary.source, summary.stream))
        if role is None:
            raise MismatchError(
                f"unexpected fan {summary.source}/{summary.stream}: the study uses "
                f"{[f'{s}/{t}' for s, t in FAN_ROLES]}"
            )
        decision_ms = _summary_hour(summary)
        config_ids.add(_summary_config_id(summary, decision_ms))
        slot = by_hour.setdefault(decision_ms, {})
        if role in slot:
            raise MismatchError(f"two {role} fans for {utc_text(decision_ms)}")
        slot[role] = summary
        elapsed_all += float(summary.elapsed_s)
        if summary.computed:
            computed += 1
            elapsed += float(summary.elapsed_s)
        else:
            reused += 1
    if not by_hour:
        raise MismatchError("no fan summary: nothing to measure")
    if len(config_ids) != 1:
        raise MismatchError(
            f"the fans come from several forecast configurations {sorted(config_ids)}"
        )
    design = _design(set(by_hour))
    eligible = [
        to_ms(h)
        for h in (kronos_sample_hours() if design == "primary" else kronos_fallback_hours())
    ]
    used: list[int] = []
    excluded: list[dict[str, Any]] = []
    roles = list(FAN_ROLES.values())
    for decision_ms in eligible:
        slot = by_hour.get(decision_ms, {})
        missing = [role for role in roles if role not in slot]
        if missing:
            excluded.append(
                {
                    "decision_hour": utc_text(decision_ms),
                    "reason": "no fan trio (a window incomplete in a source; see the batch's "
                    f"dropped list): missing {missing}",
                }
            )
            continue
        bingx, floor = slot["bingx"], slot["bingx_floor"]
        if (bingx.window_sha256, bingx.inputs_sha256) != (floor.window_sha256, floor.inputs_sha256):
            raise MismatchError(
                f"{utc_text(decision_ms)}: the floor fan does not read the BingX forecast window"
            )
        used.append(decision_ms)
    if not used:
        raise MismatchError("no sample hour has its three fans: nothing to measure")
    hours = np.array(used, dtype=np.int64)
    signal = {
        role: np.array([by_hour[h][role].signal for h in used], dtype=np.int64) for role in roles
    }
    p_up = {
        role: np.array([by_hour[h][role].p_up for h in used], dtype=np.float64) for role in roles
    }
    flags = {pair: signal[a] == signal[b] for pair, (a, b) in AGREEMENT_PAIRS.items()}
    blocks = week_blocks(hours, weeks=KRONOS_BLOCK_WEEKS)
    section = _Section("kronos")
    for pair in AGREEMENT_PAIRS:
        section.add(f"a_{pair}_share", [flags[pair].astype(np.float64)], blocks, _mean, share=True)
    floor_flags = flags["f"].astype(np.float64)
    data_effect = section.add(
        "a_x_minus_a_f", [flags["x"].astype(np.float64), floor_flags], blocks, _difference
    )
    sources_shift = section.add(
        "a_xd_minus_a_f", [flags["xd"].astype(np.float64), floor_flags], blocks, _difference
    )
    for pair, (a, b) in AGREEMENT_PAIRS.items():
        section.add(f"p_up_{pair}_correlation", [p_up[a], p_up[b]], blocks, pearson)
        section.add(f"p_up_{pair}_abs_delta_mean", [np.abs(p_up[a] - p_up[b])], blocks, _mean)
    lags = [lag for lag in AUTOCORRELATION_LAGS if lag < hours.size]
    autocorrelations: dict[str, dict[str, Any]] = {}
    for pair, values in flags.items():
        raw = autocorrelation(values.astype(np.float64), lags)
        autocorrelations[f"a_{pair}"] = {
            "values": {str(lag): _finite(value) for lag, value in raw.items()},
            "reason": None
            if all(math.isfinite(v) for v in raw.values())
            else "undefined: the flags are constant",
        }
    series = pd.DataFrame(
        {
            "decision_ms": hours,
            **{f"signal_{role}": signal[role] for role in roles},
            **{f"p_up_{role}": p_up[role] for role in roles},
            **{f"agree_{pair}": flags[pair] for pair in AGREEMENT_PAIRS},
        }
    )
    return {
        **section.body(),
        "design": design,
        "readings": {
            "a_x_minus_a_f": {
                "rule": f"interval entirely above 0: '{READING_DATA_EFFECT}'; entirely below 0: "
                f"'{READING_DATA_EFFECT_LARGER}'; containing 0 or no interval: "
                f"'{READING_NOT_DISTINGUISHABLE}'",
                **_reading(
                    data_effect, above=READING_DATA_EFFECT, below=READING_DATA_EFFECT_LARGER
                ),
            },
            "a_xd_minus_a_f": {
                "rule": f"interval entirely below 0: '{READING_SOURCES_SHIFT}' (SPEC section 5: "
                f"say so); entirely above 0: '{READING_SOURCES_ABOVE_FLOOR}'; containing 0 or no "
                f"interval: '{READING_NOT_DISTINGUISHABLE}'",
                **_reading(
                    sources_shift, above=READING_SOURCES_ABOVE_FLOOR, below=READING_SOURCES_SHIFT
                ),
            },
            "planned_mdd": PLANNED_MDD_TEXT[design],
        },
        "autocorrelation": autocorrelations,
        "owner_reference": OWNER_SEED_FLOOR,
        "zero_signals": {role: int((signal[role] == 0).sum()) for role in roles},
        "fans": {
            "summaries": computed + reused,
            "computed": computed,
            "reused": reused,
            "elapsed_s_computed": elapsed,
            "elapsed_s_all": elapsed_all,
            "config_id": next(iter(config_ids)),
        },
        "samples": {
            "design": design,
            "eligible": len(eligible),
            "used": len(used),
            "excluded": len(excluded),
            "excluded_hours": excluded,
        },
        "series": series,
    }


# -- gate 3: the invocation and the records -------------------------------------------------------


def _gate3_ratio(estimate: Any, approved_cpu_hours: Any, label: str) -> float | None:
    """:func:`~kronos_agent.kronos.batch.require_gate3` of a recorded estimate; ``Gate3Refused``
    propagates, a malformed estimate or figure is a :class:`MismatchError`."""
    if not isinstance(estimate, Mapping):
        raise MismatchError(f"{label}: the estimate must be the mapping of kronos.batch.estimate")
    try:
        return require_gate3(estimate, approved_cpu_hours=approved_cpu_hours)
    except ValueError as exc:
        raise MismatchError(f"{label}: {exc}") from None


def gate3_record(
    *,
    design: str,
    estimate: Mapping[str, Any],
    approved_cpu_hours: float | None,
    dropped: Iterable[Mapping[str, Any]],
    given_at: datetime,
) -> dict[str, Any]:
    """The record a ``mismatch --kronos`` batch appends before its first fan (D-033, D-040).

    ``estimate`` is the batch's measured :func:`kronos_agent.kronos.batch.estimate` (its pending
    fans, seconds per fan, threads, CPU-seconds and ``config_id``), ``approved_cpu_hours`` the
    figure given with ``--approved-by-owner`` (``None`` without it), ``dropped`` the tasks
    dropped for an incomplete window and ``given_at`` the command's start, when the flag was
    given. ``ratio`` is the measured CPU-hours over the approved figure when the batch needed the
    approval. Refused like the batch itself (``Gate3Refused``): above the limit without an
    approved figure, or more than
    :data:`~kronos_agent.kronos.batch.GATE3_APPROVAL_TOLERANCE` times it.
    """
    if design not in DESIGNS:
        raise MismatchError(f"design {design!r} is not one of {list(DESIGNS)}")
    if not isinstance(given_at, datetime) or given_at.tzinfo is None:
        raise MismatchError("given_at must be an aware datetime: the command's start")
    ratio = _gate3_ratio(estimate, approved_cpu_hours, "gate-3 record")
    return {
        "schema_version": GATE3_RECORD_SCHEMA_VERSION,
        "study": STEM,
        "given_at": given_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design": design,
        "config_id": estimate.get("config_id"),
        "approved_cpu_hours": approved_cpu_hours,
        "measured_cpu_hours": float(estimate["cpu_seconds"]) / 3600.0,
        "ratio": ratio,
        "tolerance": GATE3_APPROVAL_TOLERANCE,
        "estimate": dict(estimate),
        "dropped": [dict(item) for item in dropped],
    }


def _checked_record(record: Any, label: str) -> dict[str, Any]:
    """A gate-3 record of this study, its estimate checked again with the current tolerance."""
    if not isinstance(record, Mapping) or set(record) != GATE3_RECORD_KEYS:
        raise MismatchError(
            f"{label} is not a gate-3 record (expected {sorted(GATE3_RECORD_KEYS)})"
        )
    if (
        record["schema_version"] != GATE3_RECORD_SCHEMA_VERSION
        or record["study"] != STEM
        or record["design"] not in DESIGNS
        or not isinstance(record["config_id"], str)
        or not isinstance(record["dropped"], list)
    ):
        raise MismatchError(f"{label} is not a gate-3 record of the {STEM} study")
    try:
        datetime.strptime(record["given_at"], "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        raise MismatchError(f"{label}: given_at {record['given_at']!r} is not a UTC time") from None
    ratio = _gate3_ratio(record["estimate"], record["approved_cpu_hours"], label)
    if record["estimate"].get("config_id") != record["config_id"] or record["ratio"] != ratio:
        raise MismatchError(f"{label} does not agree with its own estimate")
    return dict(record)


def append_gate3_record(record: Mapping[str, Any], *, directory: Path | None = None) -> Path:
    """Append a :func:`gate3_record` to ``<directory>/mismatch.gate3.jsonl`` (default
    ``reports/data``) through the report writer: append-only (the earlier records are kept byte
    for byte) and refused like a report (personal data, absolute paths, host names)."""
    checked = _checked_record(record, "the new gate-3 record")
    return append_record(GATE3_RECORDS, checked, directory=directory)


def gate3_records(config_id: str, *, directory: Path | None = None) -> list[dict[str, Any]]:
    """The records of ``mismatch.gate3.jsonl`` whose ``config_id`` is ``config_id``, in the order
    they were appended; every record of the file is checked again (a malformed one, or one whose
    estimate gate 3 would refuse now, fails the report)."""
    name = f"{GATE3_RECORDS}{RECORDS_SUFFIX}"
    records = read_records(GATE3_RECORDS, directory=directory)
    checked = [_checked_record(item, f"{name}:{line}") for line, item in enumerate(records, 1)]
    return [record for record in checked if record["config_id"] == config_id]


def _gate3_invocation(gate3: Any, summaries: Sequence[FanSummary]) -> dict[str, Any]:
    """This invocation's gate-3 data, refused like its batch would be (``Gate3Refused``)."""
    if not isinstance(gate3, Mapping):
        raise MismatchError(
            "the Kronos part needs its gate-3 record: {'estimate': <kronos.batch.estimate>, or "
            "None when it is rebuilt from stored fans, 'approved_cpu_hours': <CPU-hours> or None, "
            "'dropped': <PreparedBatch.dropped>}"
        )
    unknown = sorted(set(gate3) - GATE3_KEYS)
    if unknown:
        raise MismatchError(f"unknown gate-3 keys {unknown}; expected {sorted(GATE3_KEYS)}")
    estimate = gate3.get("estimate")
    approved = gate3.get("approved_cpu_hours")
    ratio: float | None = None
    if estimate is None:
        if approved is not None:
            raise MismatchError("an approved figure without an estimate: nothing was measured")
    else:
        ratio = _gate3_ratio(estimate, approved, "gate3")
        tasks = estimate.get("tasks")
        if tasks is not None and tasks != len(summaries):
            raise MismatchError(
                f"the estimate counted {tasks} fans but {len(summaries)} summaries were given"
            )
    return {
        "estimate": None if estimate is None else dict(estimate),
        "approved_cpu_hours": approved,
        "ratio": ratio,
        "rebuilt_from_stored_fans": estimate is None,
        "dropped": [dict(item) for item in gate3.get("dropped") or ()],
    }


# -- the recorded definition ----------------------------------------------------------------------


def _partition_config() -> dict[str, list[str | None]]:
    return {
        segment.name: [
            utc_text(segment.start_ms),
            None if segment.end_ms is None else utc_text(segment.end_ms),
        ]
        for segment in PARTITION
    }


def study_config() -> dict[str, Any]:
    """The recorded definition of ``reports/data/mismatch`` (SPEC rule 7), Kronos part included.

    The fan configuration (model, horizon, sampling, trajectories, threads, ``amount`` policy,
    seed salt) comes from ``params.yaml``: fans of another configuration make another study.
    """
    kronos = load_params().kronos
    return {
        "study": STEM,
        "title": "Binance vs BingX real-market mismatch",
        "design": "docs/DATA.md, Studies and reports, study 3",
        "stores": {source: StoreId(source, "1h").relpath for source in SOURCES},
        "range": {
            "from": utc_text(OVERLAP_START_MS),
            "to": utc_text(SEALED_START_MS),
            "rule": "every hour present in both sources from the start of the BingX real-market "
            "history to the sealed start; candles read through read_study_frame as decimal "
            "strings, opening before the sealed start; contemporaneous candles of two venues, "
            "no validation outcome is involved",
            "partition": _partition_config(),
        },
        "definitions": {
            "return": "r_t = ln(close_t / close_(t-1h)) where t and t - 1h exist in both sources",
            "floats": "float() of the stored decimal strings",
            "equal_as_decimals": "Decimal(binance string) == Decimal(bingx string)",
            "amount_proxy": "volume x mean(open, high, low, close), the predictor's expression",
            "predictor_normalisation": "float32 channels, per-window mean and population std over "
            "the 512 rows, (x - mean) / (std + 1e-5), clipped at +-5 (KronosPredictor at the "
            "vendored commit)",
        },
        "statistics": [spec.as_dict() for specs in SECTION_SPECS.values() for spec in specs],
        "derived": {
            "returns/one_minus_pearson": "1 - pearson; its interval is the transform of pearson's",
            "returns/one_minus_spearman": "1 - spearman; its interval is the transform of "
            "spearman's",
        },
        "external_references": {
            "demo_vs_real_median_bp": DEMO_VS_REAL_MEDIAN_BP,
            "owner_two_seed_floor": OWNER_SEED_FLOOR,
            "tick_usdt": TICK_USDT,
        },
        "reading_rules": [
            (
                "alignment: the report fails unless the lag-0 return correlation is larger "
                "than the correlation at every other lag in -2..+2 h"
            ),
            (
                "kronos: an interval of kronos/a_x_minus_a_f entirely above 0 reads "
                f"'{READING_DATA_EFFECT}', entirely below 0 '{READING_DATA_EFFECT_LARGER}'; an "
                f"interval containing 0, or no interval, '{READING_NOT_DISTINGUISHABLE}'"
            ),
            (
                "kronos: an interval of kronos/a_xd_minus_a_f entirely below 0 reads "
                f"'{READING_SOURCES_SHIFT}' (SPEC section 5: say so), entirely above 0 "
                f"'{READING_SOURCES_ABOVE_FLOOR}'; an interval containing 0, or no interval, "
                f"'{READING_NOT_DISTINGUISHABLE}'"
            ),
            (
                "every reading is printed with the minimum detectable difference: "
                "(z_0.975 + z_0.80) x the standard error read from the 95 % interval, "
                "(high - low) / (2 z_0.975)"
            ),
        ],
        "kronos": {
            "status_rule": "computed by `mismatch --kronos` behind gate 3; a run without --kronos "
            "rebuilds it from the stored fans when every fan of a pre-registered sample (the "
            "primary one first, then the fallback) is stored under the current forecast config, "
            "and otherwise reports that it has not run",
            "sample": {
                "rule": f"D_k = {utc_text(to_ms(KRONOS_SAMPLE_START))} + {KRONOS_STEP_HOURS} k h "
                f"while D_k < {utc_text(to_ms(KRONOS_SAMPLE_END))} (every hour of the day)",
                "step_hours": KRONOS_STEP_HOURS,
                "count": len(kronos_sample_hours()),
                "planned": "873 fans, about 1.5 CPU-hours (3.15 s per fan measured on 2026-11-01)",
                "planned_mdd": PLANNED_MDD_TEXT["primary"],
            },
            "fallback": {
                "rule": "the same with a 97-hour step, only if the owner declines the compute",
                "step_hours": KRONOS_FALLBACK_STEP_HOURS,
                "count": len(kronos_fallback_hours()),
                "planned": "549 fans, about 0.96 CPU-hours",
                "planned_mdd": PLANNED_MDD_TEXT["fallback"],
            },
            "hours_used": "an hour enters only with its three fans (both windows complete)",
            "fans": [{"source": s, "stream": t, "role": r} for (s, t), r in FAN_ROLES.items()],
            "fan_configuration": {
                "model": kronos.model,
                "max_context": kronos.max_context,
                "horizon_n": kronos.horizon_n,
                "temperature": kronos.temperature,
                "top_p": kronos.top_p,
                "top_k": kronos.top_k,
                "trajectories": kronos.trajectories,
                "threads": kronos.threads,
                "device": kronos.device,
                "amount_policy": kronos.amount_policy,
                "seed_salt": kronos.seed_salt,
                "seed_scheme": SEED_SCHEME,
                "seed_symbol": SEED_SYMBOL,
            },
            "signal": "sign of mean_k(close[k, N]) / close[D - 1h] - 1 at N = horizon_n; p_up = "
            "share of trajectories whose close at N is strictly above close[D - 1h]",
            "agreement": "agree(a, b) = the two signals are equal",
            "autocorrelation_lags": list(AUTOCORRELATION_LAGS),
            "blocks": KRONOS_BLOCKS,
            "gate3": {
                "cpu_seconds_limit": GATE3_CPU_SECONDS,
                "approval_tolerance": GATE3_APPROVAL_TOLERANCE,
                "rule": "the estimate (cpu_seconds = pending fans x seconds per fan x threads) is "
                "measured before any fan; above the limit the batch runs only with "
                "--approved-by-owner CPU_HOURS, the figure the owner approved, and only when the "
                "measured CPU-hours are at most the tolerance times it; every batch that "
                "computes fans appends its record (approved figure, estimate, config_id, design, "
                f"dropped tasks, the command's start) to {GATE3_RECORDS}{RECORDS_SUFFIX} beside "
                "the report before its first fan, and the report lists every record of the "
                "forecast config of its fans",
            },
        },
        "intervals": {
            "method": "95 % percentile block bootstrap (kronos_agent.eval.bootstrap)",
            "resamples": RESAMPLES,
            "confidence": CONFIDENCE,
            "blocks": f"{WEEK_BLOCKS} for hourly series; {KRONOS_BLOCKS} for the 61-hour "
            "Kronos sample (also the dz of the amount section)",
            "min_blocks": MIN_BLOCKS,
            "max_failed_share": MAX_FAILED_SHARE,
            "seed": "numpy seed = first 4 bytes of sha256('kronos-agent/bootstrap/v1/' + key), "
            "key = 'mismatch/<section>/<statistic>'",
            "paired": "paired comparisons resample both columns in one call",
            "shares": "a share of exactly 0 or 1 gets a Wilson interval at the number of blocks; "
            "deff and n_eff are reported for shares",
            "quantiles": "medians get an interval unless their bootstrap distribution is a single "
            "value; p90, p99 and max are descriptive",
        },
        "series": {
            name: f"studies/{stem}.csv (relative to the data root)"
            for name, stem in SERIES_STEMS.items()
        },
        "figures": [
            f"figures/{STEM}_return_scatter.png: r_bingx against r_binance in bp",
            f"figures/{STEM}_price_differences.png: histograms of the signed price differences",
            (
                f"figures/{STEM}_agreement.png: A_x, A_f, A_xd and the paired differences with "
                "intervals (Kronos part only)"
            ),
        ],
        "tests_consumed": 0,
    }


# -- the report -----------------------------------------------------------------------------------


def _read_sources(root: Path | None) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Both 1h stores over the study range as STRING frames, and each store as it was read.

    The job locks of both sources are held shared around the reads (a ``data`` run cannot rebuild
    either store meanwhile, and one running now refuses the read with ``StoreBusyError``); each
    store's meta is read before its frame and again after it, so the ``csv_sha256`` recorded is
    the one of the bytes parsed, never the one of a store rebuilt after the study read it.
    """
    usecols = {"binance": BINANCE_READ_COLUMNS, "bingx": BINGX_READ_COLUMNS}
    frames: dict[str, pd.DataFrame] = {}
    stores: list[dict[str, Any]] = []
    with contextlib.ExitStack() as locks:
        for source in SOURCES:
            locks.enter_context(store_read_lock(StoreId(source, "1h"), root))
        for source in SOURCES:
            store_id = StoreId(source, "1h")
            meta = read_meta(store_id, root)
            frame = read_study_frame(
                store_id,
                start_ms=OVERLAP_START_MS,
                end_ms=SEALED_START_MS,
                numeric=False,
                usecols=list(usecols[source]),
                root=root,
            )
            if read_meta(store_id, root)["csv_sha256"] != meta["csv_sha256"]:
                raise MismatchError(
                    f"{store_id.relpath} was rebuilt while the study read it; run it again"
                )
            frames[source] = frame
            stores.append(
                {
                    "relpath": store_id.relpath,
                    "csv_sha256": meta["csv_sha256"],
                    "read_from": utc_text(OVERLAP_START_MS),
                    "read_to": utc_text(SEALED_START_MS),
                    "rows_read": len(frame),
                }
            )
    return frames, stores


def _presence(open_b: np.ndarray, open_x: np.ndarray) -> dict[str, Any]:
    grid = np.arange(OVERLAP_START_MS, SEALED_START_MS, HOUR_MS, dtype=np.int64)
    in_b = np.isin(grid, open_b)
    in_x = np.isin(grid, open_x)
    exclusions = [
        _exclusion(reason, grid[mask])
        for reason, mask in (
            ("no Binance candle (BingX real market present)", ~in_b & in_x),
            ("no BingX real-market candle (Binance present)", in_b & ~in_x),
            ("no candle in either source", ~in_b & ~in_x),
        )
        if mask.any()
    ]
    used = int((in_b & in_x).sum())
    return {
        "range": [utc_text(OVERLAP_START_MS), utc_text(SEALED_START_MS)],
        "eligible": int(grid.size),
        "used": used,
        "excluded": int(grid.size) - used,
        "exclusions": exclusions,
    }


def _hours_series(joined: pd.DataFrame) -> pd.DataFrame:
    table = _returns_table(joined)
    opens = joined["open_ms"].to_numpy(dtype=np.int64)
    position = np.searchsorted(opens, table["open_ms"].to_numpy(dtype=np.int64))
    r_binance = np.full(opens.size, np.nan)
    r_bingx = np.full(opens.size, np.nan)
    r_binance[position] = table["r_binance"].to_numpy()
    r_bingx[position] = table["r_bingx"].to_numpy()
    data: dict[str, Any] = {
        "open_ms": opens,
        "binance_close": joined["binance_close"].to_numpy(),
        "bingx_close": joined["bingx_close"].to_numpy(),
        "r_binance": r_binance,
        "r_bingx": r_bingx,
    }
    for field in PRICE_FIELDS:
        binance = joined[f"binance_{field}"].to_numpy(dtype=np.float64)
        bingx = joined[f"bingx_{field}"].to_numpy(dtype=np.float64)
        data[f"{field}_diff_bp"] = (bingx - binance) / binance * BP
        data[f"equal_{field}"] = joined[f"equal_{field}"].to_numpy(dtype=bool)
    data["binance_volume"] = joined["binance_volume"].to_numpy()
    data["bingx_volume"] = joined["bingx_volume"].to_numpy()
    return pd.DataFrame(data)


@dataclass(frozen=True)
class PreparedStudy:
    """Every part of study 3 that needs no fan, computed and checked, nothing written yet.

    ``sections`` holds alignment, returns, prices, volume and amount (their per-hour series moved
    to ``series_frames``), ``stores`` each store as it was read and ``hours`` the hours present
    in both sources with the exclusions; ``root`` is the data root the candles came from.
    """

    root: Path | None
    sections: Mapping[str, Mapping[str, Any]]
    series_frames: Mapping[str, pd.DataFrame]
    stores: tuple[dict[str, Any], ...]
    hours: Mapping[str, Any]


def prepare_study(*, root: Path | None = None, directory: Path | None = None) -> PreparedStudy:
    """Read both 1h stores and compute every section without Kronos; write nothing.

    The definition (:func:`study_config`) is recorded, or verified, under ``directory`` (default
    ``reports/data``) before any candle is read: a changed definition raises
    ``ConfigChangedError``. Candles: ``candles/binance/1h`` and ``candles/bingx/1h`` from
    2024-04-27T10:00Z to the sealed start, through ``read_study_frame``. Fails
    (:class:`AlignmentError`) unless lag 0 is the largest return correlation, and with
    :class:`MismatchError` when nothing can be measured: the command runs this before any Kronos
    work, so approved compute is never spent on a report that then fails.
    """
    record_config(STEM, study_config(), directory=directory)
    frames, stores = _read_sources(root)
    joined = joined_hours(frames["binance"], frames["bingx"])
    if joined.empty:
        raise MismatchError("no hour is present in both sources: nothing to measure")
    sections: dict[str, dict[str, Any]] = {
        "alignment": alignment(joined),
        "returns": return_statistics(joined),
        "prices": price_differences(joined),
        "volume": volume_statistics(joined),
    }
    sections["amount"] = amount_proxy(
        frames["binance"], sample_hours=kronos_sample_hours(), bingx_1h=frames["bingx"]
    )
    series_frames = {
        "hours": _hours_series(joined),
        "amount": sections["amount"].pop("series"),
        "amount_windows": sections["amount"].pop("window_series"),
    }
    hours = _presence(
        frames["binance"]["open_ms"].to_numpy(dtype=np.int64),
        frames["bingx"]["open_ms"].to_numpy(dtype=np.int64),
    )
    return PreparedStudy(
        root=root, sections=sections, series_frames=series_frames, stores=tuple(stores), hours=hours
    )


def _summary_list(
    kronos_summaries: Sequence[FanSummary] | None,
    gate3: Mapping[str, Any] | None,
    kronos_reason: str | None,
) -> list[FanSummary] | None:
    if gate3 is not None and kronos_summaries is None:
        raise MismatchError("a gate-3 record without fan summaries: pass both or neither")
    if kronos_summaries is None:
        if kronos_reason is not None and not isinstance(kronos_reason, str):
            raise MismatchError("kronos_reason must be a text")
        return None
    if kronos_reason is not None:
        raise MismatchError("kronos_reason says why the Kronos part did not run: no summaries")
    if isinstance(kronos_summaries, FanSummary):
        raise MismatchError("kronos_summaries must be a sequence of FanSummary")
    return list(kronos_summaries)


def build_report(
    *,
    root: Path | None = None,
    kronos_summaries: Sequence[FanSummary] | None = None,
    gate3: Mapping[str, Any] | None = None,
    kronos_reason: str | None = None,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Run study 3 and write it: :func:`prepare_study`, then :func:`write_study`.

    The Kronos inputs are checked first, so a missing or refused gate-3 record fails before any
    candle is read.
    """
    summaries = _summary_list(kronos_summaries, gate3, kronos_reason)
    if summaries is not None:
        _gate3_invocation(gate3, summaries)
    study = prepare_study(root=root, directory=directory)
    return write_study(
        study,
        kronos_summaries=summaries,
        gate3=gate3,
        kronos_reason=kronos_reason,
        directory=directory,
    )


def _remove_kronos_files(root: Path | None, directory: Path | None) -> None:
    """Remove the Kronos part's figure and series that a report without that part does not list
    (D-037: the files beside a report are the ones it lists, with their sha256)."""
    from ..report.figures import figures_dir

    base = Path(root) if root is not None else data_dir()
    for path in (
        figures_dir(directory) / AGREEMENT_FIGURE,
        base / STUDIES_SUBDIR / f"{SERIES_STEMS['kronos']}.csv",
    ):
        path.unlink(missing_ok=True)


def write_study(
    study: PreparedStudy,
    *,
    kronos_summaries: Sequence[FanSummary] | None = None,
    gate3: Mapping[str, Any] | None = None,
    kronos_reason: str | None = None,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Write study 3: ``reports/data/mismatch.json`` and ``.md``, the figures and the per-hour
    series (``data/studies/mismatch_*.csv``); returns the written payload.

    ``kronos_summaries`` (the summaries of a batch of :func:`kronos_tasks`, or of the fans found
    stored) adds the Kronos signal agreement and needs ``gate3``: ``{"estimate":
    <kronos.batch.estimate>, or None when the part is rebuilt from stored fans,
    "approved_cpu_hours": <the figure given with --approved-by-owner> or None, "dropped":
    <PreparedBatch.dropped>}``, refused like the batch (``Gate3Refused``). The report lists, with
    it, every record of ``mismatch.gate3.jsonl`` whose forecast config is the one of the fans
    (:func:`gate3_records`). Without summaries the report says the part has not run
    (``kronos_reason``, else :data:`KRONOS_NOT_RUN`), and the Kronos figure and series of an
    earlier report are removed, so the files beside the report are the ones it lists.
    """
    if not isinstance(study, PreparedStudy):
        raise TypeError(f"expected a PreparedStudy, got {type(study).__name__}")
    summaries = _summary_list(kronos_summaries, gate3, kronos_reason)
    config = study_config()
    record_config(STEM, config, directory=directory)
    root = study.root
    sections: dict[str, dict[str, Any]] = {
        name: dict(body) for name, body in study.sections.items()
    }
    series_frames = dict(study.series_frames)
    if summaries is not None:
        invocation = _gate3_invocation(gate3, summaries)
        kronos = kronos_agreement(summaries)
        series_frames["kronos"] = kronos.pop("series")
        fans = kronos["fans"]
        estimate = invocation["estimate"]
        if estimate is not None and estimate.get("config_id") not in (None, fans["config_id"]):
            raise MismatchError(
                f"the estimate is of forecast config {estimate.get('config_id')}, the fans of "
                f"{fans['config_id']}"
            )
        threads = config["kronos"]["fan_configuration"]["threads"]
        sections["kronos"] = {
            "status": "run",
            "gate3": {
                "invocation": invocation,
                "records": gate3_records(fans["config_id"], directory=directory),
                "records_file": f"{GATE3_RECORDS}{RECORDS_SUFFIX}",
                "fans_cpu_seconds": fans["elapsed_s_all"] * threads,
                "cpu_seconds_limit": GATE3_CPU_SECONDS,
                "approval_tolerance": GATE3_APPROVAL_TOLERANCE,
            },
            **kronos,
        }
    else:
        sections["kronos"] = {"status": "not_run", "reason": kronos_reason or KRONOS_NOT_RUN}
    series = {
        name: write_study_series(SERIES_STEMS[name], frame, root=root)
        for name, frame in series_frames.items()
    }
    seeds: dict[str, Any] = {"bootstrap": {}}
    comparators: dict[str, str] = {}
    with_intervals = 0
    for section, body in sections.items():
        seeds["bootstrap"].update(body.get("seeds", {}))
        for name, text in body.get("comparators", {}).items():
            comparators[f"{section}/{name}"] = text
        with_intervals += body.get("statistics_with_intervals", 0)
    seeds["kronos"] = (
        {
            "seed_scheme": SEED_SCHEME,
            "seed_symbol": SEED_SYMBOL,
            "seed_salt": config["kronos"]["fan_configuration"]["seed_salt"],
            "streams": {role: f"{s}/{t}" for (s, t), role in FAN_ROLES.items()},
        }
        if summaries is not None
        else None
    )
    samples = {
        "hours": dict(study.hours),
        "returns": sections["returns"]["samples"],
        "volume": sections["volume"]["samples"],
        "amount": sections["amount"]["samples"],
        "kronos": sections["kronos"].get("samples", {"status": "not_run"}),
    }
    payload_body: dict[str, Any] = {
        "sections": sections,
        "series": series,
        "data": {"stores": [dict(store) for store in study.stores]},
        "overlap_start": utc_text(OVERLAP_START_MS),
    }
    payload_body["figures"] = mismatch_figures(
        {**payload_body, "study": STEM}, series_frames, directory=directory
    )
    header = report_header(
        STEM,
        config,
        statistics_with_intervals=with_intervals,
        seeds=seeds,
        samples=samples,
        comparators=comparators,
        directory=directory,
    )
    payload = {**payload_body, **header}
    write_report(STEM, payload, render_markdown(payload), directory=directory)
    if summaries is None:
        _remove_kronos_files(root, directory)
    return payload


# -- Markdown -------------------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _interval_text(result: Mapping[str, Any]) -> str:
    if not _has_interval(result):
        return "n/a"
    return f"[{_fmt(result['low'])}, {_fmt(result['high'])}]"


def _note(result: Mapping[str, Any]) -> str:
    parts = [str(result.get("method", ""))]
    if result.get("n_eff") is not None:
        parts.append(f"n_eff {_fmt(result['n_eff'])}")
    if result.get("reason"):
        parts.append(str(result["reason"]))
    return "; ".join(part for part in parts if part).replace("|", "/")


def _short(text: str) -> str:
    return text.split(":")[0].replace("|", "/")


def _statistics_table(body: Mapping[str, Any]) -> list[str]:
    comparators = body.get("comparators") or {}
    lines = [
        "| statistic | point | 95 % interval | n | blocks | method / note | comparator |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in (body.get("statistics") or {}).items():
        lines.append(
            f"| {name} | {_fmt(result.get('point'))} | {_interval_text(result)} | "
            f"{_fmt(result.get('n'))} | {_fmt(result.get('n_blocks'))} | {_note(result)} | "
            f"{_short(comparators.get(name, ''))} |"
        )
    descriptive = body.get("descriptive") or {}
    if descriptive:
        text = ", ".join(f"{name} {_fmt(value)}" for name, value in descriptive.items())
        lines += ["", f"Descriptive (no interval): {text}."]
    return lines


def _header_lines(report: Mapping[str, Any]) -> list[str]:
    code = report.get("code") or {}
    return [
        (
            f"- Study: `{report.get('study')}`; config sha256 `{report.get('config_sha256')}`; "
            f"docs/DATA.md sha256 `{report.get('data_md_sha256')}`"
        ),
        (
            f"- Code: commit `{code.get('commit')}`, dirty {_fmt(code.get('dirty'))}, source "
            f"tree sha256 `{code.get('source_tree_sha256')}`"
        ),
        f"- Generated at {report.get('generated_at')}",
        (
            f"- Tests consumed: {_fmt(report.get('tests_consumed'))} (this study measures the "
            "data, not a strategy, and looks at no trade outcome)"
        ),
        (
            f"- Statistics with intervals: {_fmt(report.get('statistics_with_intervals'))}; "
            "expected to miss their true value by chance at 95 %: "
            f"{_fmt(report.get('expected_chance_misses'))}"
        ),
        (
            f"- Intervals: 95 % percentile block bootstrap, {RESAMPLES} resamples; blocks: "
            f"{WEEK_BLOCKS} (hourly series), {KRONOS_BLOCKS} (Kronos sample hours); seed key "
            "`mismatch/<section>/<statistic>` (listed in the JSON report)"
        ),
    ]


def _exclusion_lines(exclusions: Iterable[Mapping[str, Any]], limit: int = 5) -> list[str]:
    lines: list[str] = []
    for item in exclusions:
        ranges = item.get("ranges") or []
        shown = ", ".join(f"[{a}, {b})" for a, b in ranges[:limit])
        more = f" and {len(ranges) - limit} more ranges (JSON)" if len(ranges) > limit else ""
        lines.append(
            f"- Excluded ({item.get('reason')}): {_fmt(item.get('hours'))} hours: {shown}{more}"
        )
    return lines


def _skipped_lines(skipped: Iterable[Mapping[str, Any]], limit: int = 5) -> list[str]:
    """Skipped decision hours grouped by reason, the first ``limit`` of each listed."""
    by_reason: dict[str, list[str]] = {}
    for item in skipped:
        by_reason.setdefault(str(item.get("reason")), []).append(str(item.get("decision_hour")))
    lines = []
    for reason, hours in by_reason.items():
        shown = ", ".join(hours[:limit])
        more = f" and {len(hours) - limit} more (JSON)" if len(hours) > limit else ""
        lines.append(f"- Skipped ({reason}): {len(hours)} sample hours: {shown}{more}")
    return lines


def _lost_lines(item: Mapping[str, Any]) -> list[str]:
    """The ranges of a sample's lost hours (``lost``, ``lost_reason``, ``lost_ranges``)."""
    if not item.get("lost"):
        return []
    return _exclusion_lines(
        [
            {
                "reason": item.get("lost_reason"),
                "hours": item.get("lost"),
                "ranges": item.get("lost_ranges"),
            }
        ]
    )


def _estimate_text(estimate: Mapping[str, Any]) -> str:
    return (
        f"{_fmt(estimate.get('cpu_seconds'))} CPU-seconds ({_fmt(estimate.get('pending'))} pending "
        f"fans x {_fmt(estimate.get('seconds_per_fan'))} s x {_fmt(estimate.get('threads'))} "
        "threads)"
    )


def _gate3_lines(gate3: Mapping[str, Any]) -> list[str]:
    """The invocation's estimate, then every gate-3 record of the fans' forecast config."""
    invocation = gate3.get("invocation") or {}
    estimate = invocation.get("estimate")
    if estimate is None:
        lines = ["This run rebuilt the part from the stored fans: no fan pending, none computed."]
    else:
        lines = [f"This run: gate-3 estimate {_estimate_text(estimate)}."]
    records = gate3.get("records") or []
    lines += [
        "",
        (
            f"Gate-3 records of these fans' forecast config ({gate3.get('records_file')} beside "
            f"this report, appended before each batch's first fan): {len(records)}"
        ),
    ]
    for record in records:
        approved = record.get("approved_cpu_hours")
        if record.get("ratio") is not None:
            approval = (
                f"the owner approved {_fmt(approved)} CPU-hours (--approved-by-owner given at "
                f"{record.get('given_at')}); measured {_fmt(record.get('measured_cpu_hours'))} "
                f"CPU-hours, {_fmt(record.get('ratio'))} times the approved figure (at most "
                f"{_fmt(record.get('tolerance'))})"
            )
        else:
            approval = (
                f"within the limit, no approval needed (run started at {record.get('given_at')})"
            )
        lines.append(
            f"- {record.get('design')} sample: {approval}; estimate "
            f"{_estimate_text(record.get('estimate') or {})}; "
            f"{len(record.get('dropped') or [])} tasks dropped"
        )
    if not records:
        lines.append("- none: no recorded batch of this study computed fans of this config")
    total = gate3.get("fans_cpu_seconds")
    limit = gate3.get("cpu_seconds_limit")
    approved_any = any(record.get("ratio") is not None for record in records)
    lines.append(
        f"- The fans took {_fmt(total)} CPU-seconds of model time in all (gate-3 limit "
        f"{_fmt(limit)} per batch)"
        + (
            "; above the limit, and no record carries the owner's approval"
            if total is not None and limit is not None and total > limit and not approved_any
            else ""
        )
    )
    return lines


def _kronos_lines(kronos: Mapping[str, Any]) -> list[str]:
    if kronos.get("status") != "run":
        return [f"Kronos signal agreement: {kronos.get('reason', KRONOS_NOT_RUN)}."]
    samples = kronos.get("samples") or {}
    fans = kronos.get("fans") or {}
    lines = [
        (
            f"Design `{kronos.get('design')}`: {_fmt(samples.get('used'))} of "
            f"{_fmt(samples.get('eligible'))} sample hours with the three fans; "
            f"{_fmt(fans.get('summaries'))} fans ({_fmt(fans.get('computed'))} computed, "
            f"{_fmt(fans.get('reused'))} reused), forecast config `{fans.get('config_id')}`."
        ),
        "",
        *_gate3_lines(kronos.get("gate3") or {}),
        "",
        *_statistics_table(kronos),
        "",
        f"External reference: the owner's two-seed floor of {_fmt(OWNER_SEED_FLOOR)}.",
    ]
    readings = kronos.get("readings") or {}
    for name in ("a_x_minus_a_f", "a_xd_minus_a_f"):
        reading = readings.get(name) or {}
        mdd = (reading.get("minimum_detectable_difference") or {}).get("value")
        lines.append(
            f"- Reading of {name} (pre-registered): **{reading.get('reading')}** (interval "
            f"{reading.get('interval_position')}; minimum detectable difference {_fmt(mdd)}; "
            f"planned {readings.get('planned_mdd')})"
        )
    for name, item in (kronos.get("autocorrelation") or {}).items():
        values = ", ".join(f"lag {lag} {_fmt(v)}" for lag, v in (item.get("values") or {}).items())
        lines.append(f"- Autocorrelation of the {name} flags: {values}")
    for item in samples.get("excluded_hours") or []:
        lines.append(f"- Excluded {item.get('decision_hour')}: {item.get('reason')}")
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Markdown of the mismatch report: header, samples, every section, readings, files."""
    sections = report.get("sections") or {}
    samples = report.get("samples") or {}
    hours = samples.get("hours") or {}
    lines = [
        "# Binance vs BingX real-market mismatch",
        "",
        (
            "Study 3 of docs/DATA.md (SPEC section 5): every hour present in both sources from "
            f"{report.get('overlap_start')} to the sealed start. Contemporaneous candles of two "
            "venues; no validation outcome is involved."
        ),
        "",
        *_header_lines(report),
        "",
        "## Samples",
        "",
        (
            f"- Hours [{', '.join(hours.get('range') or [])}): eligible "
            f"{_fmt(hours.get('eligible'))}, used {_fmt(hours.get('used'))}, excluded "
            f"{_fmt(hours.get('excluded'))}"
        ),
        *_exclusion_lines(hours.get("exclusions") or []),
    ]
    returns = samples.get("returns") or {}
    lines.append(
        f"- Returns: {_fmt(returns.get('used'))} hours, lost {_fmt(returns.get('lost'))} "
        f"({returns.get('lost_reason')})"
    )
    lines += _lost_lines(returns)
    volume = samples.get("volume") or {}
    lines.append(
        f"- Volume ratio (descriptive): {_fmt(volume.get('ratio_hours'))} of "
        f"{_fmt(volume.get('hours'))} hours"
    )
    lines += _exclusion_lines(volume.get("ratio_exclusions") or [])
    amount = samples.get("amount") or {}
    amount_hours = amount.get("hours") or {}
    sample_hours = amount.get("sample_hours") or {}
    lines.append(
        f"- Amount proxy e_t: eligible {_fmt(amount_hours.get('binance_hours'))} Binance hours, "
        f"used {_fmt(amount_hours.get('used'))}"
    )
    lines += _exclusion_lines(amount_hours.get("exclusions") or [])
    lines.append(
        f"- Amount proxy dz: Kronos sample hours used {_fmt(sample_hours.get('used'))} of "
        f"{_fmt(sample_hours.get('requested'))}"
    )
    lines += _skipped_lines(sample_hours.get("skipped_hours") or [])
    titles = {
        "alignment": "Alignment (1h log returns at lags -2..+2 h)",
        "returns": "Returns",
        "prices": "Prices: (BingX - Binance) / Binance in bp",
        "volume": "Volume",
        "amount": "amount proxy (question 9)",
    }
    for name, title in titles.items():
        body = sections.get(name) or {}
        lines += ["", f"## {title}", "", *_statistics_table(body)]
        if name == "alignment":
            lines += ["", f"Lag 0 is the largest: {_fmt(body.get('lag0_is_largest'))}."]
        if name == "returns":
            for key, item in (body.get("derived") or {}).items():
                lines.append(
                    f"- {key}: {_fmt(item.get('point'))} "
                    f"[{_fmt(item.get('low'))}, {_fmt(item.get('high'))}] (transform of "
                    f"{item.get('derived_from')})"
                )
        if name == "prices":
            external = body.get("external_references") or {}
            lines.append(
                f"- External reference: SPEC section 5 demo-vs-real median "
                f"{_fmt(external.get('demo_vs_real_median_bp'))} bp."
            )
    lines += ["", "## Kronos signal agreement", "", *_kronos_lines(sections.get("kronos") or {})]
    lines += ["", "## Comparators", ""]
    lines += [f"- {name}: {text}" for name, text in (report.get("comparators") or {}).items()]
    lines += ["", "## Series, figures and data", ""]
    for name, item in (report.get("series") or {}).items():
        lines.append(
            f"- Series {name} `{item.get('relpath')}` (relative to the data root): "
            f"{_fmt(item.get('rows'))} rows, sha256 `{item.get('sha256')}`"
        )
    for figure in report.get("figures") or []:
        lines.append(
            f"- ![{figure.get('name')}]({figure.get('relpath')}) sha256 `{figure.get('sha256')}`"
        )
    for store in (report.get("data") or {}).get("stores") or []:
        lines.append(
            f"- Store `{store.get('relpath')}` csv sha256 `{store.get('csv_sha256')}`; read "
            f"[{store.get('read_from')}, {store.get('read_to')}), {_fmt(store.get('rows_read'))} "
            "rows"
        )
    return "\n".join(lines) + "\n"


# -- figures --------------------------------------------------------------------------------------


def mismatch_figures(
    report: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    *,
    directory: Path | None = None,
) -> list[dict[str, str]]:
    """The study's figures under ``<report directory>/figures`` (identical bytes on every render).

    ``frames`` are the per-hour series :func:`build_report` writes: ``hours`` (the joined hours
    with returns and price differences) and, when the Kronos part ran, ``kronos``; the agreement
    figure reads its intervals from ``report``. Returns ``[{name, relpath, sha256}]``.
    """
    from ..report.figures import figures_dir, new_figure, save_png

    target = figures_dir(directory)
    out: list[dict[str, str]] = []
    hours = frames["hours"]
    sections = report.get("sections") or {}

    figure, axis = new_figure()
    r_b = hours["r_binance"].to_numpy(dtype=np.float64) * BP
    r_x = hours["r_bingx"].to_numpy(dtype=np.float64) * BP
    ok = np.isfinite(r_b) & np.isfinite(r_x)
    if ok.any():
        limit = float(np.quantile(np.abs(np.concatenate([r_b[ok], r_x[ok]])), 0.999)) or 1.0
        axis.scatter(r_b[ok], r_x[ok], s=3.0, alpha=0.3, color="tab:blue", linewidths=0.0)
        axis.plot([-limit, limit], [-limit, limit], color="tab:red", linewidth=0.8, label="equal")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.legend(loc="upper left")
    else:
        axis.text(0.5, 0.5, "no return", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("Binance 1h log return (bp)")
    axis.set_ylabel("BingX real-market 1h log return (bp)")
    axis.set_title(f"1h log returns at the same hour (n = {int(ok.sum())}; axes at p99.9 of |r|)")
    out.append(
        {"name": "return_scatter", **save_png(figure, target / f"{STEM}_return_scatter.png")}
    )

    figure, axes = new_figure(2, 2)
    values = {f: hours[f"{f}_diff_bp"].to_numpy(dtype=np.float64) for f in PRICE_FIELDS}
    pooled = np.abs(np.concatenate(list(values.values())))
    limit = float(np.quantile(pooled, 0.995)) if pooled.size else 0.0
    limit = limit if limit > 0 else 1.0
    edges = np.linspace(-limit, limit, 81)
    for axis, field in zip(np.asarray(axes).ravel(), PRICE_FIELDS, strict=True):
        data = values[field]
        inside = np.abs(data) <= limit
        counts, _ = np.histogram(data[inside], bins=edges)
        axis.stairs(counts, edges, fill=True, color="tab:blue", alpha=0.6)
        axis.axvline(0.0, color="black", linewidth=0.6)
        axis.set_title(f"{field} (bp; {int((~inside).sum())} hours outside the axis)")
    figure.suptitle("(BingX - Binance) / Binance at the same hour, bp")
    out.append(
        {"name": "price_differences", **save_png(figure, target / f"{STEM}_price_differences.png")}
    )

    kronos = sections.get("kronos") or {}
    if kronos.get("status") == "run" and frames.get("kronos") is not None:
        statistics = kronos.get("statistics") or {}
        figure, axes = new_figure(1, 2)
        for axis, names, title in (
            (axes[0], ("a_x_share", "a_f_share", "a_xd_share"), "agreement shares"),
            (axes[1], ("a_x_minus_a_f", "a_xd_minus_a_f"), "paired differences"),
        ):
            for position, name in enumerate(names):
                result = statistics.get(name) or {}
                point = result.get("point")
                if point is None:
                    continue
                if _has_interval(result):
                    error = [[max(0.0, point - result["low"])], [max(0.0, result["high"] - point)]]
                    axis.errorbar(
                        [position], [point], yerr=error, fmt="o", capsize=4, color="tab:blue"
                    )
                else:
                    axis.plot([position], [point], "o", color="tab:blue")
            axis.set_xticks(range(len(names)))
            axis.set_xticklabels(names)
            axis.set_xlim(-0.5, len(names) - 0.5)
            axis.set_title(title)
        axes[0].axhline(
            OWNER_SEED_FLOOR, color="tab:gray", linestyle="--", linewidth=0.8, label="owner 88 %"
        )
        axes[0].legend(loc="lower right")
        axes[1].axhline(0.0, color="black", linewidth=0.6)
        figure.suptitle("Kronos signal agreement with 95 % intervals")
        out.append({"name": "agreement", **save_png(figure, target / AGREEMENT_FIGURE)})
    return out


__all__ = [
    "AGREEMENT_FIGURE",
    "AGREEMENT_PAIRS",
    "ALIGNMENT_LAGS",
    "DESIGNS",
    "FAN_ROLES",
    "GATE3_RECORDS",
    "KRONOS_FALLBACK_STEP_HOURS",
    "KRONOS_NOT_RUN",
    "KRONOS_STEP_HOURS",
    "OVERLAP_START",
    "OVERLAP_START_MS",
    "OWNER_SEED_FLOOR",
    "READING_DATA_EFFECT",
    "READING_DATA_EFFECT_LARGER",
    "READING_NOT_DISTINGUISHABLE",
    "READING_SOURCES_ABOVE_FLOOR",
    "READING_SOURCES_SHIFT",
    "SECTION_SPECS",
    "SERIES_STEMS",
    "STEM",
    "STREAM_FLOOR",
    "STREAM_FORECAST",
    "AlignmentError",
    "MismatchError",
    "PreparedStudy",
    "StatSpec",
    "alignment",
    "amount_proxy",
    "append_gate3_record",
    "average_ranks",
    "build_report",
    "gate3_record",
    "gate3_records",
    "joined_hours",
    "kronos_agreement",
    "kronos_fallback_hours",
    "kronos_sample_hours",
    "kronos_tasks",
    "minimum_detectable_difference",
    "mismatch_figures",
    "pearson",
    "predictor_z",
    "prepare_study",
    "price_differences",
    "render_markdown",
    "return_statistics",
    "spearman",
    "study_config",
    "volume_statistics",
    "write_study",
]
