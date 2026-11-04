"""The tokenizer diagnostic of the retraining (``docs/RETRAINING.md`` section 4; SPEC 9.1):
``kronos-agent finetune tokenizer-diag --part training|validation``.

A recorded study (D-034): descriptive, 0 tests, the shipped Kronos-Tokenizer-base frozen in eval
mode on the CPU at the governed 2 threads, deterministic, under ``torch.no_grad()`` (the quantiser
is a sign function and ``decode`` is deterministic, so no seed is involved). It never changes
recipe R1: a flag can only motivate the separate, later tokenizer test of SPEC 9.2.

Samples (decision hours on ``G25``; the definition is recorded before any candle is read):

- Part A, ``training``: Binance 1h, ``D`` from 2020-01-22T08:00Z (the first full context) to
  ``TRAINING_END - 48 h``. Each window holds its 512 context candles ``[D - 512 h, D)`` and the
  next 49 ``[D, D + 49 h)``, so that the template's ``W = 561`` can be formed (the design's "next
  48 rows" plus the target row; interfaces D9). Every candle read opens before
  2024-06-09T16:00Z, the training segment's candle boundary.
- Part B, ``validation``: Binance and BingX 1h, ``D`` in ``[VALIDATION_START, VALIDATION_END)``,
  context only (inputs only). It runs after the pre-registration bundle (the command checks it),
  recorded in a ledger that may authorize reading the data root (:func:`prepare_diag` checks
  ``ledger.binding_problem``), and reads Part A's report for its s1 and s2 counts.
- Reference: 100 synthetic GBM windows (``features.synthetic_window(512, seed=s)``), featureless.

Windows come from ``data.windows.StoreWindowSource`` (``read_study_frame`` behind the sealed
guard) and their floats from ``data.windows.context_frame`` under ``volume_x_mean_ohlc``, the
inference functions; a window with a missing candle is dropped and listed with its ranges.

Per window, ``x`` float32 in ``FEATURE_COLUMNS`` order and ``z = (x - mean) / (std + 1e-5)``
(population std) with statistics reduced on a Fortran-ordered float32 copy of the rows they
average over, the layout of the predictor's and the vendored dataset's ``DataFrame.values``
(:func:`normalise_rows`; ``finetune.nll.normalise`` does the same for 516 rows): ``inference``
over the 512 context rows (K:620-631); ``r1`` over 513 rows (FB:122-127, R1's ``W = 513``);
``template`` over 561 rows; ``context_only`` over the 512 context rows applied to 513 rows.
``zc = clip(z, -5, 5)``; ``(s1, s2) = tokenizer.encode(zc, half=True)``; ``zr =
tokenizer.decode([s1, s2], half=True)`` and ``xr = zr * (std + 1e-5) + mean`` (float32, K:657);
``z_pre`` the s1-only reconstruction of the forward pass (K:98-106); the bit margin from the
quantiser's L2-normalised latent (M:246). Code sizes come from the tokenizer, never hard-coded.

Statistics, flags and intervals are design section 4's (and interfaces 10.3): clip saturation,
reconstruction (RMSE in normalised space, the close error in bp, the return resolution ``rho``,
for the full codes and for s1 alone), codebook usage (at matched row counts where sizes matter),
the bit margin, the token mismatch of Part A (the shares of context rows whose s1 token, s2
token, or either differs; ``context_only`` is a positive control that must give exactly 0),
Jensen-Shannon divergences. The flag of D-048 (5), "more than 5 % of context tokens changing",
reads the share of rows with either token changed (:data:`TOKEN_MISMATCH_STATISTIC`): every row
carries one s1 and one s2 token, and that share bounds the s1 share, the s2 share and the share
of the 1,024 context tokens alike, so the owner is told whichever of those readings passes 5 %.
Shares, means and medians get a 95 % percentile block bootstrap over 4-week blocks of ``D``
(``eval.bootstrap``, 1,000 resamples, keys ``tokenizer-<part>/<set>/<statistic>``); only the
per-window 0/1 indicators are shares there (:data:`SHARE_STATISTICS`, with ``deff`` and ``n_eff``
against windows and a Wilson interval for a share of exactly 0 or 1), the per-window fractions
(saturation of context entries, bit margins, token mismatches) are means; p90, p99, maxima,
comparators and codebook figures are descriptive. The reference has no decision hour
(one block): point estimates only. Flags are pre-registered and descriptive: no automatic
action.

Report: ``reports/finetune/tokenizer-<part>.json`` and ``.md`` through the report layer (the
definition recorded once as ``tokenizer-<part>.config.json``; a changed definition is refused),
with two figures under ``figures/``. No path enters it.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..data.partition import (
    MAX_CONTEXT,
    SEALED_START,
    TRAINING_END,
    TRAINING_START,
    VALIDATION_END,
    VALIDATION_START,
)
from ..data.timeutil import HOUR_MS, from_ms, to_ms, utc_text
from ..hashing import sha256_file
from ..kronos.features import FEATURE_COLUMNS
from ..params import Params
from ..paths import REPO_ROOT, reports_dir
from .folds import grid_decisions_ms
from .nll import NORMALISATION_EPSILON

PARTS: tuple[str, ...] = ("training", "validation")
STEMS: Mapping[str, str] = MappingProxyType(
    {"training": "tokenizer-training", "validation": "tokenizer-validation"}
)
GRID = "G25"
CONTEXT = 512
NEXT_ROWS = 49  # 512 + 48 + 1 = 561 rows, the template's W (interfaces D9)
PART_A_FIRST = datetime(
    2020, 1, 22, 8, tzinfo=UTC
)  # TRAINING_START + 512 h: the first full context
PART_A_LAST = TRAINING_END - timedelta(hours=48)
REFERENCE_SEEDS: tuple[int, ...] = tuple(range(100))
BLOCK_WEEKS = 4
BIT_MARGINS: tuple[float, ...] = (1e-3, 1e-2)
LAST_ROWS = 24
BATCH_WINDOWS = 32
# pre-registered, descriptive, no automatic action (design section 4, "Flags"); point estimates
FLAGS: Mapping[str, float] = MappingProxyType(
    {
        "price_saturation_context": 0.001,
        "price_saturation_next": 0.01,
        "volume_amount_saturation": 0.01,
        "rho": 0.25,
        "s1_normalised_entropy": 0.5,
        "s1_unused_share": 0.5,
        "validation_perplexity_ratio": 0.5,
        "r1_token_mismatch": 0.05,
    }
)
FLAG_RULES: Mapping[str, str] = MappingProxyType(
    {
        "price_saturation_context": "a price channel's share of context entries with |z| > 5 "
        "above 0.1 %",
        "price_saturation_next": "a price channel's share of next rows (row 512, inference "
        "statistics) beyond +-5 above 1 % (Part A)",
        "volume_amount_saturation": "the volume or amount share of context entries with |z| > 5 "
        "above 1 %",
        "rho": "the return resolution rho above 0.25",
        "s1_normalised_entropy": "the s1 normalised entropy below 0.5",
        "s1_unused_share": "more than half the s1 codes unused",
        "validation_perplexity_ratio": "the validation s1 perplexity below half Part A's (Part B)",
        "r1_token_mismatch": "R1's context-token mismatch above 5 % (Part A): the share of the "
        "512 context rows whose s1 or s2 token differs from the inference encoding, which "
        "bounds the s1 share, the s2 share and their mean; the owner is told before the "
        "pre-registration bundle (D-048 (5))",
    }
)
# the flag of D-048 (5) reads this statistic: every context row carries one s1 and one s2 token,
# so the rows with either token changed are the widest reading of "context tokens changing"
TOKEN_MISMATCH_STATISTIC = "mismatch_r1_any"
# the statistics whose interval treats each window as one 0/1 row (eval.bootstrap share=True):
# per-window indicators only; the per-window fractions (saturation_context, bit_margin, mismatch)
# are means of fractions
SHARE_STATISTICS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "training": ("saturation_windows_<channel>", "saturation_next_<channel>"),
        "validation": ("saturation_windows_<channel>",),
    }
)
# the columns of a variant's per-window mismatch shares: s1, s2 and either (any), overall and over
# the last 24 context rows
MISMATCH_COLUMNS: tuple[str, ...] = ("s1", "s1_last24", "s2", "s2_last24", "any", "any_last24")

SOURCES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"training": ("binance",), "validation": ("binance", "bingx")}
)
REFERENCE = "reference"
AMOUNT_POLICY = "volume_x_mean_ohlc"
CLIP = 5.0
DIAG_THREADS = 2
# variant: (rows the statistics average over, rows encoded); the first 512 tokens are compared
VARIANTS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "inference": (CONTEXT, CONTEXT),
        "r1": (CONTEXT + 1, CONTEXT + 1),
        "template": (CONTEXT + NEXT_ROWS, CONTEXT + NEXT_ROWS),
        "context_only": (CONTEXT, CONTEXT + 1),
    }
)
MISMATCH_VARIANTS: tuple[str, ...] = ("r1", "template", "context_only")
CONTROL_VARIANT = "context_only"  # a positive control: exactly 0 mismatching tokens
PRICE_CHANNELS: tuple[str, ...] = FEATURE_COLUMNS[:4]
VOLUME_CHANNELS: tuple[str, ...] = FEATURE_COLUMNS[4:]
CLOSE_INDEX = FEATURE_COLUMNS.index("close")
RESAMPLES = 1000
CONFIDENCE = 0.95
BASIS_POINTS = 1e4
TICK_USDT = 0.1
SATURATION_COMPARATOR = math.erfc(CLIP / math.sqrt(2.0))  # P(|N(0,1)| > 5) = 5.733e-7
COVERAGE_PERCENTS: tuple[int, ...] = (50, 90, 99)
DEAD_BIT_BOUNDS: tuple[float, float] = (0.01, 0.99)
GATE3_CPU_SECONDS = 3600  # SPEC section 17 (kronos.batch.GATE3_CPU_SECONDS)
# the pooled-row statistics with an interval per set (close error median and rho, full and s1)
POOLED_STATISTICS = 4
PROBE_RESAMPLES = 3
SCHEMA_VERSION = 1
FINETUNE_REPORTS_SUBDIR = "finetune"
RETRAINING_MD_RELPATH = "docs/RETRAINING.md"
TRAINING_CANDLE_END = VALIDATION_START - timedelta(hours=MAX_CONTEXT)  # 2024-06-09T16:00Z
VALIDATION_CANDLE_END = SEALED_START - timedelta(hours=MAX_CONTEXT)  # 2026-07-10T16:00Z
_MAX_DROPPED_SHOWN = 5


class TokenizerDiagError(ValueError):
    """The diagnostic cannot run as defined, or its inputs are inconsistent (exit 1)."""


class TokenizerDiagRefused(PermissionError):
    """The diagnostic's measured estimate passes gate 3 (exit 2); it has no approval flag."""


@dataclass(frozen=True)
class TokenizerSteps:
    """The heavy parts, injectable (:func:`real_steps`; tests pass stand-ins):
    ``load_tokenizer()`` returns the tokenizer, ``window_source(source, start_ms=, end_ms=,
    root=)`` a ``StoreWindowSource``-like object and ``clock()`` seconds."""

    load_tokenizer: Callable[[], Any]  # load_models("small").tokenizer
    window_source: Callable[..., Any]  # data.windows.StoreWindowSource
    clock: Callable[[], float]  # time.perf_counter


def _load_shipped_tokenizer() -> Any:
    """The verified shipped Kronos-Tokenizer-base (``load_models("small")``, eval mode, CPU);
    the upstream Hub mixin's stdout line is captured."""
    from ..kronos.weights import load_models

    with contextlib.redirect_stdout(io.StringIO()):
        return load_models("small").tokenizer


def real_steps() -> TokenizerSteps:
    """The real steps (imported here, after the runtime is configured)."""
    from ..data.windows import StoreWindowSource

    return TokenizerSteps(
        load_tokenizer=_load_shipped_tokenizer,
        window_source=StoreWindowSource,
        clock=time.perf_counter,
    )


# -- the definition ---------------------------------------------------------------------------


def _check_part(part: object) -> str:
    if not isinstance(part, str) or part not in PARTS:
        raise TokenizerDiagError(f"part must be one of {list(PARTS)}, got {part!r}")
    return part


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def decisions_ms(part: str) -> tuple[int, ...]:
    """The ``G25`` decision hours of ``part`` (1,534 for ``training``, 710 for ``validation``)."""
    if _check_part(part) == "training":
        return grid_decisions_ms(to_ms(PART_A_FIRST), to_ms(PART_A_LAST) + HOUR_MS, GRID)
    return grid_decisions_ms(to_ms(VALIDATION_START), to_ms(VALIDATION_END), GRID)


def _source_range(part: str, decisions: Sequence[int]) -> tuple[int, int]:
    """``[start_ms, end_ms)`` of the candles a part reads (interfaces 10.1), checked against the
    segment's candle boundaries."""
    if not decisions:
        raise TokenizerDiagError(f"{STEMS[part]} has no decision hour")
    if part == "training":
        start = to_ms(TRAINING_START)
        end = int(decisions[-1]) + NEXT_ROWS * HOUR_MS
        first_context = int(decisions[0]) - CONTEXT * HOUR_MS
        if first_context < start or end > to_ms(TRAINING_CANDLE_END):
            raise TokenizerDiagError(
                f"Part A reads candles in [{utc_text(first_context)}, {utc_text(end)}): "
                f"outside the training segment's candles [{utc_text(start)}, "
                f"{utc_text(to_ms(TRAINING_CANDLE_END))})"
            )
        return start, end
    start = to_ms(VALIDATION_START) - CONTEXT * HOUR_MS
    end = to_ms(VALIDATION_END)
    if int(decisions[0]) - CONTEXT * HOUR_MS < start or int(decisions[-1]) > end:
        raise TokenizerDiagError("Part B decision hours lie outside the validation segment")
    if end > to_ms(VALIDATION_CANDLE_END):
        raise TokenizerDiagError("Part B would read past the validation segment's candles")
    return start, end


def _tokenizer_identity() -> dict[str, Any]:
    """The registered shipped tokenizer (``manifests/weights.json``): what ``load_models``
    verifies before it loads it."""
    from ..kronos.weights import repos_for_model

    repo = repos_for_model("small")[1]
    return {
        "model": "small",
        "repo_id": repo.repo_id,
        "revision": repo.revision,
        "sha256": {item.name: item.sha256 for item in repo.files if item.sha256 is not None},
    }


def _statistic_definitions(part: str) -> dict[str, str]:
    definitions = {
        "saturation_context_<channel>": "share of context entries with |z| > 5 (inference "
        "statistics); mean of the per-window shares (equal rows per window)",
        "saturation_windows_<channel>": "share of windows with at least one context entry with "
        "|z| > 5",
        "max_abs_z_<channel>": "max |z| over context entries (descriptive)",
        "rmse_<channel>, rmse_last24_<channel>": "sqrt of the mean of (zr - zc)^2 over every "
        "context row and over the last 24 (normalised space)",
        "close_error_median_abs_bp": "median over windows and rows of |1e4 (xr - x) / x| of the "
        "close (x the float32 context); p90, p99 and max descriptive",
        "rho": "median |er| / median |r| with r_j = log(close_j / close_(j-1)), j = 1..511, and "
        "er_j the same on xr minus r_j (returns of a non-positive reconstructed close dropped "
        "and counted); a ratio computed per resample",
        "pre_<statistic>": "the same for z_pre, the reconstruction from the s1 bits alone",
        "bit_margin_<t>": "share of the L2-normalised latent coordinates with |z| < t",
        "codebook": "distinct s1, s2 codes and (s1, s2) pairs, also at matched row counts (M = "
        "the smallest compared set's rows, rows in time order at np.linspace(0, n - 1, "
        "M).round()); entropy (natural log), perplexity, normalised entropy (entropy / log(2 ** "
        "bits)); the fewest s1 codes covering 50, 90 and 99 % of rows; per-bit share of ones "
        "(dead below 1 % or above 99 %); all descriptive",
        "comparators": "median |1h return| in bp, one 0.1 USDT tick in bp (median over windows "
        "of 1e4 * 0.1 / last close), P(|N(0,1)| > 5) = 5.733e-7",
    }
    if part == "training":
        definitions.update(
            {
                "saturation_next_<channel>": "share of windows whose row 512 (inference "
                "statistics) lies beyond +-5",
                "mismatch_<variant>_<s1|s2|any>[_last24]": "per window, the share of the 512 "
                "context rows (and of the last 24) whose s1 token, s2 token, or either of the "
                "two (any: the context tokens of D-048 (5)) under the variant's normalisation "
                "differs from the inference encoding; the window encodes all its rows (the "
                "encoder is causal, M:345-350), the first 512 positions compared; context_only "
                "must give exactly 0 (a positive control)",
                "js_first_second_half": "Jensen-Shannon divergence (log base 2) of the s1 "
                "marginals of the first and the second half of the windows (comparator)",
            }
        )
    else:
        definitions.update(
            {
                "js_training_validation_<source>": "Jensen-Shannon divergence (log base 2) of "
                "Part A's s1 marginal and the validation source's",
                "js_binance_bingx": "Jensen-Shannon divergence (log base 2) of the two "
                "validation s1 marginals",
                "perplexity_ratio_<source>": "the validation s1 perplexity over Part A's",
            }
        )
    return definitions


def diag_plan(part: str) -> dict[str, Any]:
    """The recorded definition of ``part``: decision rule, windows, normalisations, statistics,
    flags and intervals. Constants only (no data, no params version, no code hash)."""
    part = _check_part(part)
    decisions = decisions_ms(part)
    start, end = _source_range(part, decisions)
    training = part == "training"
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study": STEMS[part],
        "design": f"{RETRAINING_MD_RELPATH} section 4 (SPEC 9.1)",
        "part": part,
        "segment": part,
        "tests": 0,
        "decisions": {
            "grid": GRID,
            "grid_rule": "hours whose count since 2020-01-01T00:00Z is divisible by 25",
            "rule": (
                f"D from {utc_text(to_ms(PART_A_FIRST))} (the first full context) to "
                f"{utc_text(to_ms(PART_A_LAST))} (TRAINING_END - 48 h) inclusive"
                if training
                else f"D in [{utc_text(to_ms(VALIDATION_START))}, "
                f"{utc_text(to_ms(VALIDATION_END))})"
            ),
            "first": utc_text(decisions[0]),
            "last": utc_text(decisions[-1]),
            "planned": len(decisions),
        },
        "sources": {
            source: {
                "store": f"candles/{source}/1h",
                "read": "data.windows.StoreWindowSource (read_study_frame, the sealed guard)",
                "start": utc_text(start),
                "end": utc_text(end),
            }
            for source in SOURCES[part]
        },
        "window": {
            "context_rows": CONTEXT,
            "context": "[D - 512 h, D): StoreWindowSource.window(D, horizon=1)",
            "next_rows": NEXT_ROWS if training else 0,
            "next": "[D, D + 49 h): StoreWindowSource.candles_between, 49 candles required"
            if training
            else None,
            "amount_policy": AMOUNT_POLICY,
            "floats": "data.windows.context_frame (the inference functions)",
            "columns": list(FEATURE_COLUMNS),
            "incomplete": "a window with a missing candle is dropped and listed with its ranges",
        },
        "reference": {
            "windows": len(REFERENCE_SEEDS),
            "generator": "kronos.features.synthetic_window(512, seed=s)",
            "seeds": [REFERENCE_SEEDS[0], REFERENCE_SEEDS[-1]],
            "rows": CONTEXT,
            "blocks": "one (no decision hour): point estimates only",
        },
        "tokenizer": _tokenizer_identity(),
        "compute": {
            "device": "cpu",
            "threads": DIAG_THREADS,
            "deterministic": True,
            "no_grad": True,
            "eval_mode": True,
            "batch_windows": BATCH_WINDOWS,
        },
        "normalisation": {
            "formula": "(x - mean) / (std + 1e-5), population std, x float32; the statistics "
            "reduced on a Fortran-ordered float32 copy of the rows they average over",
            "clip": CLIP,
            "variants": {
                name: {"statistics_rows": stats, "rows": rows}
                for name, (stats, rows) in VARIANTS.items()
                if training or name == "inference"
            },
        },
        "encoding": {
            "tokens": "tokenizer.encode(zc, half=True)",
            "reconstruction": "tokenizer.decode([s1, s2], half=True); xr = zr * (std + 1e-5) "
            "+ mean with the inference statistics, float32 (K:657)",
            "z_pre": "post_quant_embed_pre of the s1 bits of indices_to_bits, every decoder "
            "layer, head (the forward pass's z_pre, K:98-106)",
            "bit_margin": "torch.nn.functional.normalize(quant_embed(encoder(embed(zc))), "
            "dim=-1) (M:246)",
            "code_sizes": "2 ** tokenizer.s1_bits and 2 ** tokenizer.s2_bits",
        },
        "statistics": _statistic_definitions(part),
        "last_rows": LAST_ROWS,
        "bit_margins": list(BIT_MARGINS),
        "coverage_percents": list(COVERAGE_PERCENTS),
        "dead_bit_bounds": list(DEAD_BIT_BOUNDS),
        "flags": {
            name: {"threshold": threshold, "rule": FLAG_RULES[name]}
            for name, threshold in FLAGS.items()
            if not (training and name == "validation_perplexity_ratio")
            and not (not training and name in ("price_saturation_next", "r1_token_mismatch"))
        },
        "token_mismatch_statistic": TOKEN_MISMATCH_STATISTIC if training else None,
        "flag_reading": "point estimates of the real sources against the thresholds; "
        "descriptive, no automatic action",
        "intervals": {
            "method": "95 % percentile block bootstrap (eval.bootstrap.block_bootstrap)",
            "resamples": RESAMPLES,
            "confidence": CONFIDENCE,
            "blocks": f"week_blocks(D_ms, weeks={BLOCK_WEEKS})",
            "key": f"{STEMS[part]}/<set>/<statistic>",
            "shares": {
                "statistics": list(SHARE_STATISTICS[part]),
                "rule": "share=True only for these per-window 0/1 indicators (deff and n_eff "
                "against windows; a share of exactly 0 or 1 gets a Wilson interval at the block "
                "count); every other statistic, the per-window fractions of entries, coordinates "
                "or rows included (saturation_context, bit_margin, mismatch), is a mean with "
                "share=False",
            },
            "descriptive": "p90, p99, maxima, comparators, codebook statistics, divergences",
        },
        "comparators": {
            "saturation": f"P(|N(0,1)| > 5) = {SATURATION_COMPARATOR:.4g}",
            "reconstruction": "the median |1h return| and one 0.1 USDT tick, in bp",
            "reference": "100 synthetic GBM windows (featureless)",
        },
    }
    if not training:
        plan["inputs"] = (
            f"Part A's report ({STEMS['training']}.json): its s1 and s2 counts and s1 perplexity"
        )
    return plan


def _report_dir(directory: Path | None) -> Path:
    return Path(directory) if directory is not None else reports_dir() / FINETUNE_REPORTS_SUBDIR


def record_diag_config(part: str, *, directory: Path | None = None) -> str:
    """Record (or verify) the definition of ``part`` as ``<stem>.config.json``; its sha256.
    A changed definition is refused (``ConfigChangedError``)."""
    from ..report.writer import record_config

    part = _check_part(part)
    return record_config(STEMS[part], diag_plan(part), directory=_report_dir(directory))


# -- normalisation ----------------------------------------------------------------------------


def _standardise(values: np.ndarray, *, stats_rows: int) -> tuple[np.ndarray, ...]:
    """``(z, mean, std)`` of a ``(n, rows, 6)`` array: ``z`` float32 before the clip, ``mean``
    and ``std`` float32 ``(n, 6)`` over each window's first ``stats_rows`` rows."""
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[2] != len(FEATURE_COLUMNS):
        raise TokenizerDiagError(
            f"values must have shape (n, rows, {len(FEATURE_COLUMNS)}), got {array.shape}"
        )
    rows = array.shape[1]
    if not _is_int(stats_rows) or not 1 <= stats_rows <= rows:
        raise TokenizerDiagError(
            f"stats_rows must be an integer in [1, {rows}], got {stats_rows!r}"
        )
    z = np.empty(array.shape, dtype=np.float32)
    means = np.empty((array.shape[0], array.shape[2]), dtype=np.float32)
    stds = np.empty_like(means)
    for index in range(array.shape[0]):
        x = array[index].astype(np.float32)
        statistics = np.asfortranarray(x[:stats_rows])
        mean, std = np.mean(statistics, axis=0), np.std(statistics, axis=0)
        z[index] = (x - mean) / (std + NORMALISATION_EPSILON)
        means[index], stds[index] = mean, std
    return z, means, stds


def normalise_rows(values: np.ndarray, *, stats_rows: int, clip: float = 5.0) -> np.ndarray:
    """float32 ``(rows, 6)`` or ``(n, rows, 6)``: each window normalised with the mean and
    population std of its first ``stats_rows`` rows, then clipped at ``+-clip``.

    Per window exactly ``finetune.nll.normalise`` generalised to ``stats_rows``: ``x =
    values.astype(float32)``; the statistics reduced on a Fortran-ordered copy (the layout of the
    predictor's and the vendored dataset's ``DataFrame.values``; a C-ordered float32 reduction
    differs in the last bits); ``(x - mean) / (std + 1e-5)`` on every row.
    """
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not clip > 0:
        raise TokenizerDiagError(f"clip must be a positive number, got {clip!r}")
    array = np.asarray(values)
    single = array.ndim == 2
    z, _, _ = _standardise(array[None] if single else array, stats_rows=stats_rows)
    clipped = np.clip(z, -clip, clip)
    return clipped[0] if single else clipped


# -- windows ----------------------------------------------------------------------------------


def _read_only(array: np.ndarray) -> np.ndarray:
    copy = np.array(array, copy=True)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True, eq=False)
class DiagWindows:
    """One set's kept windows: ``decision_ms`` int64 ``(n,)`` ascending (the reference repeats
    one hour), ``values`` float64 ``(n, rows, 6)`` (512 rows, or 561 with Part A's next rows),
    ``dropped`` the windows a missing candle removed (``{"decision", "missing": [[start, end),
    ...]}``). Stored as read-only copies."""

    source: str
    decision_ms: np.ndarray
    values: np.ndarray
    dropped: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        decisions = self.decision_ms
        values = self.values
        if not isinstance(decisions, np.ndarray) or decisions.dtype != np.int64:
            raise TokenizerDiagError("decision_ms must be an int64 numpy array")
        if decisions.ndim != 1 or decisions.size < 1 or np.any(np.diff(decisions) < 0):
            raise TokenizerDiagError("decision_ms must be one-dimensional, not empty, ascending")
        if not isinstance(values, np.ndarray) or values.dtype != np.float64:
            raise TokenizerDiagError("values must be a float64 numpy array")
        rows = (CONTEXT, CONTEXT + NEXT_ROWS)
        if (
            values.ndim != 3
            or values.shape[0] != decisions.size
            or values.shape[1] not in rows
            or values.shape[2] != len(FEATURE_COLUMNS)
        ):
            raise TokenizerDiagError(
                f"values must have shape ({decisions.size}, 512 or 561, "
                f"{len(FEATURE_COLUMNS)}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise TokenizerDiagError("values must be finite")
        object.__setattr__(self, "decision_ms", _read_only(decisions))
        object.__setattr__(self, "values", _read_only(values))
        object.__setattr__(self, "dropped", tuple(dict(item) for item in self.dropped))

    @property
    def windows(self) -> int:
        return int(self.decision_ms.size)


@dataclass(frozen=True, eq=False)
class PreparedDiag:
    """The windows of a part, built and checked, nothing encoded yet."""

    part: str
    config_sha256: str
    sources: Mapping[str, DiagWindows]  # "binance" (and "bingx" in Part B)
    reference: DiagWindows  # source "reference", the 100 GBM windows
    training_report: Mapping[str, Any] | None  # Part B: Part A's report as written


def _ranges(open_times: Sequence[int]) -> list[list[str]]:
    """``[start, end)`` UTC texts of consecutive missing hourly opens."""
    ranges: list[list[int]] = []
    for open_ms in sorted(open_times):
        if ranges and ranges[-1][1] == open_ms:
            ranges[-1][1] = open_ms + HOUR_MS
        else:
            ranges.append([open_ms, open_ms + HOUR_MS])
    return [[utc_text(start), utc_text(end)] for start, end in ranges]


def _missing(present: Sequence[Any], start_ms: int, end_ms: int) -> list[int]:
    have = {candle.open_ms for candle in present}
    return [open_ms for open_ms in range(start_ms, end_ms, HOUR_MS) if open_ms not in have]


def _read_windows(
    part: str,
    source: str,
    decisions: Sequence[int],
    steps: TokenizerSteps,
    root: Path | None,
) -> DiagWindows:
    from ..data.windows import WindowGapError, context_frame

    start_ms, end_ms = _source_range(part, decisions)
    store = steps.window_source(source, start_ms=start_ms, end_ms=end_ms, root=root)
    kept: list[int] = []
    values: list[np.ndarray] = []
    dropped: list[dict[str, Any]] = []
    for decision in decisions:
        decision = int(decision)
        missing: list[int] = []
        window = None
        try:
            window = store.window(
                from_ms(decision), horizon=1, max_context=CONTEXT, amount_policy=AMOUNT_POLICY
            )
        except WindowGapError:
            first = decision - CONTEXT * HOUR_MS
            missing += _missing(store.candles_between(first, decision), first, decision)
        following: list[Any] = []
        if part == "training":
            end = decision + NEXT_ROWS * HOUR_MS
            following = list(store.candles_between(decision, end))
            if len(following) != NEXT_ROWS:
                missing += _missing(following, decision, end)
        if missing or window is None:
            dropped.append(
                {
                    "decision": utc_text(decision),
                    "decision_ms": decision,
                    "missing_candles": len(missing),
                    "missing": _ranges(missing),
                }
            )
            continue
        block = window.context.to_numpy(dtype=np.float64)
        if following:
            rows = context_frame(following, AMOUNT_POLICY).to_numpy(dtype=np.float64)
            block = np.concatenate([block, rows], axis=0)
        kept.append(decision)
        values.append(block)
    if not kept:
        raise TokenizerDiagError(
            f"{STEMS[part]}: no {source} window of the {len(decisions)} planned has every candle"
        )
    return DiagWindows(
        source=source,
        decision_ms=np.asarray(kept, dtype=np.int64),
        values=np.stack(values),
        dropped=tuple(dropped),
    )


def _reference_windows() -> DiagWindows:
    """The 100 synthetic GBM windows (context only), dated one hour after their last candle."""
    from ..kronos.features import synthetic_window

    values: list[np.ndarray] = []
    decisions: list[int] = []
    for seed in REFERENCE_SEEDS:
        frame, stamps, _ = synthetic_window(CONTEXT, seed=seed)
        values.append(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64))
        decisions.append(int(stamps.iloc[-1].value // 1_000_000) + HOUR_MS)
    return DiagWindows(
        source=REFERENCE,
        decision_ms=np.asarray(decisions, dtype=np.int64),
        values=np.stack(values),
        dropped=(),
    )


def _check_params(params: Params) -> None:
    from ..runtime import require_configured

    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    kronos = params.kronos
    if kronos.model != "small" or kronos.max_context != CONTEXT:
        raise TokenizerDiagError(
            f"the diagnostic reads Kronos-small's tokenizer with {CONTEXT} context rows, "
            f"params.yaml has {kronos.model!r} with {kronos.max_context}"
        )
    info = require_configured()
    if kronos.threads != DIAG_THREADS or info.threads != kronos.threads:
        raise TokenizerDiagError(
            f"the diagnostic runs {DIAG_THREADS} threads: params.yaml has {kronos.threads}, the "
            f"runtime {info.threads}"
        )
    if not info.deterministic:
        raise TokenizerDiagError("the diagnostic runs with deterministic algorithms on")


def _training_report(directory: Path) -> dict[str, Any]:
    """Part A's report as written, checked against its recorded definition (Part B needs its s1
    and s2 counts)."""
    stem = STEMS["training"]
    path = directory / f"{stem}.json"
    shown = f"{FINETUNE_REPORTS_SUBDIR}/{path.name}"
    try:
        report = json.loads(path.read_text(encoding="ascii"))
    except FileNotFoundError:
        raise TokenizerDiagError(
            f"Part B needs Part A's report ({shown}) for its s1 and s2 counts: run `kronos-agent "
            "finetune tokenizer-diag --part training` first"
        ) from None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TokenizerDiagError(f"{shown} cannot be read: {type(exc).__name__}") from None
    if not isinstance(report, dict) or report.get("study") != stem:
        raise TokenizerDiagError(f"{shown} is not Part A's report")
    if report.get("part") != "training":
        raise TokenizerDiagError(f"{shown} is not Part A's report")
    for key in ("s1_counts", "s2_counts"):
        counts = report.get(key)
        if (
            not isinstance(counts, list)
            or not counts
            or not all(_is_int(value) and value >= 0 for value in counts)
            or sum(counts) < 1
        ):
            raise TokenizerDiagError(f"{shown}: {key} must be a list of non-negative counts")
    config = directory / f"{stem}.config.json"
    if not config.is_file() or sha256_file(config) != report.get("config_sha256"):
        raise TokenizerDiagError(
            f"{shown} was not written under the recorded Part A definition ({config.name})"
        )
    return report


def prepare_diag(
    part: str,
    *,
    params: Params,
    steps: TokenizerSteps,
    root: Path | None = None,
    directory: Path | None = None,
    ledger_root: Path | None = None,
) -> PreparedDiag:
    """Record (or verify) the definition of ``part`` first (``ConfigChangedError`` for a changed
    one), then read Part A's report (Part B) and build every window: each source through
    ``steps.window_source`` under the data ``root``, and the reference. ``directory`` is the
    report directory (``reports/finetune``). Part B reads validation inputs, which wait for the
    pre-registration bundle: before anything, the ledger at ``ledger_root`` (default
    ``paths.ledger_dir()``) must be one that may authorize reading the data ``root``
    (``ledger.require_binding``, ``LedgerRefused`` otherwise: a relocated reports root never
    admits the repository's validation inputs; the command checks the bundle itself)."""
    part = _check_part(part)
    _check_params(params)
    if part == "validation":
        from .. import ledger

        ledger.require_binding(root=ledger_root, data_root=root)
    directory = _report_dir(directory)
    config_sha256 = record_diag_config(part, directory=directory)
    training_report = _training_report(directory) if part == "validation" else None
    decisions = decisions_ms(part)
    sources = {
        source: _read_windows(part, source, decisions, steps, root) for source in SOURCES[part]
    }
    return PreparedDiag(
        part=part,
        config_sha256=config_sha256,
        sources=MappingProxyType(sources),
        reference=_reference_windows(),
        training_report=training_report,
    )


# -- encoding ---------------------------------------------------------------------------------


def _check_tokenizer(tokenizer: Any) -> tuple[int, int]:
    """``(s1_bits, s2_bits)`` of a tokenizer in eval mode on the CPU."""
    import torch

    if not isinstance(tokenizer, torch.nn.Module):
        raise TokenizerDiagError(f"the tokenizer must be a torch module, got {type(tokenizer)}")
    if any(module.training for module in tokenizer.modules()):
        raise TokenizerDiagError("the tokenizer must be in eval mode (every submodule)")
    if any(
        tensor.device.type != "cpu" for tensor in (*tokenizer.parameters(), *tokenizer.buffers())
    ):
        raise TokenizerDiagError("the tokenizer must be on the CPU")
    s1_bits, s2_bits = getattr(tokenizer, "s1_bits", None), getattr(tokenizer, "s2_bits", None)
    if not (_is_int(s1_bits) and _is_int(s2_bits) and s1_bits >= 1 and s1_bits == s2_bits):
        # decode(half=True) splits the codebook in two equal halves (K:126-131)
        raise TokenizerDiagError(f"the tokenizer needs s1_bits == s2_bits >= 1, got {s1_bits}")
    return int(s1_bits), int(s2_bits)


def _encode(tokenizer: Any, zc: np.ndarray) -> tuple[Any, Any]:
    import torch

    return tokenizer.encode(torch.from_numpy(np.ascontiguousarray(zc, dtype=np.float32)), half=True)


def _z_pre(tokenizer: Any, s1: Any, s2: Any) -> Any:
    """The reconstruction from the s1 bits alone, as the forward pass computes ``z_pre``."""
    bits = tokenizer.indices_to_bits([s1, s2], half=True)
    hidden = tokenizer.post_quant_embed_pre(bits[..., : tokenizer.s1_bits])
    for layer in tokenizer.decoder:
        hidden = layer(hidden)
    return tokenizer.head(hidden)


def _latent(tokenizer: Any, x: Any) -> Any:
    """The quantiser's input after its L2 normalisation (``encode`` up to M:246)."""
    import torch

    hidden = tokenizer.embed(x)
    for layer in tokenizer.encoder:
        hidden = layer(hidden)
    return torch.nn.functional.normalize(tokenizer.quant_embed(hidden), dim=-1)


def _returns(closes: np.ndarray) -> np.ndarray:
    """``log(close_j / close_(j-1))``, j = 1..511, NaN where a close is not positive."""
    previous, current = closes[:, :-1], closes[:, 1:]
    valid = (previous > 0) & (current > 0)
    out = np.full(current.shape, np.nan)
    with np.errstate(all="ignore"):
        out[valid] = np.log(current[valid] / previous[valid])
    return out


def _reconstruction(
    name: str,
    recon: np.ndarray,
    zc: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    x_close: np.ndarray,
    r: np.ndarray,
) -> dict[str, np.ndarray]:
    error = recon.astype(np.float64) - zc.astype(np.float64)
    squares = error * error
    xr = recon * (std[:, None, :] + NORMALISATION_EPSILON) + mean[:, None, :]  # float32, K:657
    xr_close = xr[:, :, CLOSE_INDEX].astype(np.float64)
    return {
        f"{name}_mse": squares.mean(axis=1),
        f"{name}_mse_last": squares[:, -LAST_ROWS:].mean(axis=1),
        f"{name}_close_error": BASIS_POINTS * (xr_close - x_close) / x_close,
        f"{name}_er": _returns(xr_close) - r,
    }


def _diagnose_batch(tokenizer: Any, values: np.ndarray) -> dict[str, np.ndarray]:
    """Per-window arrays of one batch: saturation counts, tokens, reconstruction errors, bit
    margins and, for Part A windows (561 rows), the next row's z and the token mismatches."""
    import torch

    context = values[:, :CONTEXT]
    z, mean, std = _standardise(context, stats_rows=CONTEXT)
    zc = np.clip(z, -CLIP, CLIP)
    abs_z = np.abs(z)
    with torch.no_grad():
        tensor = torch.from_numpy(zc)
        s1, s2 = tokenizer.encode(tensor, half=True)
        zr = tokenizer.decode([s1, s2], half=True).numpy()
        z_pre = _z_pre(tokenizer, s1, s2).numpy()
        latent = np.abs(_latent(tokenizer, tensor).numpy())
    s1_codes, s2_codes = s1.numpy().astype(np.int64), s2.numpy().astype(np.int64)
    x_close = context.astype(np.float32)[:, :, CLOSE_INDEX].astype(np.float64)
    r = _returns(x_close)
    out: dict[str, np.ndarray] = {
        "saturated": (abs_z > CLIP).sum(axis=1),
        "max_abs_z": abs_z.max(axis=1),
        "s1": s1_codes,
        "s2": s2_codes,
        "margins": np.stack(
            [(latent < margin).mean(axis=(1, 2)) for margin in BIT_MARGINS], axis=1
        ),
        "last_close": x_close[:, -1].copy(),
        "r": r,
    }
    out.update(_reconstruction("full", zr, zc, mean, std, x_close, r))
    out.update(_reconstruction("pre", z_pre, zc, mean, std, x_close, r))
    if values.shape[1] > CONTEXT:
        out.update(_mismatches(tokenizer, values, s1_codes, s2_codes))
    return out


def _mismatches(
    tokenizer: Any, values: np.ndarray, s1: np.ndarray, s2: np.ndarray
) -> dict[str, np.ndarray]:
    """Row 512's z (inference statistics) and, per variant, the per-window shares of the 512
    context positions (and of the last 24) whose s1 token, s2 token, or either of the two
    differs from the inference encoding: the columns of :data:`MISMATCH_COLUMNS`."""
    out: dict[str, np.ndarray] = {}
    for variant in MISMATCH_VARIANTS:
        stats_rows, rows = VARIANTS[variant]
        z, _, _ = _standardise(values[:, :rows], stats_rows=stats_rows)
        if variant == CONTROL_VARIANT:
            out["next_z"] = z[:, CONTEXT].copy()
        v1, v2 = _encode(tokenizer, np.clip(z, -CLIP, CLIP))
        d1 = np.asarray(v1)[:, :CONTEXT] != s1
        d2 = np.asarray(v2)[:, :CONTEXT] != s2
        either = d1 | d2
        out[f"mismatch_{variant}"] = np.stack(
            [
                d1.mean(axis=1),
                d1[:, -LAST_ROWS:].mean(axis=1),
                d2.mean(axis=1),
                d2[:, -LAST_ROWS:].mean(axis=1),
                either.mean(axis=1),
                either[:, -LAST_ROWS:].mean(axis=1),
            ],
            axis=1,
        )
    return out


def _diagnose(tokenizer: Any, windows: DiagWindows) -> dict[str, np.ndarray]:
    import torch

    parts: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for start in range(0, windows.windows, BATCH_WINDOWS):
            parts.append(_diagnose_batch(tokenizer, windows.values[start : start + BATCH_WINDOWS]))
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}


# -- the estimate -----------------------------------------------------------------------------


def _ratio_of_medians(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(np.median(numerator) / np.median(denominator))


def _statistics_seconds(prepared: PreparedDiag, clock: Callable[[], float]) -> float:
    """Measured wall seconds of the pooled-row intervals: :data:`PROBE_RESAMPLES` resamples of
    the dearest one (rho: two columns, two medians) at the size of the largest set with enough
    blocks, scaled to :data:`RESAMPLES` and to :data:`POOLED_STATISTICS` per such set."""
    from ..eval.bootstrap import MIN_BLOCKS, block_bootstrap, week_blocks

    sets = [
        windows
        for windows in prepared.sources.values()
        if np.unique(week_blocks(windows.decision_ms, weeks=BLOCK_WEEKS)).size >= MIN_BLOCKS
    ]
    if not sets:
        return 0.0
    largest = max(sets, key=lambda windows: windows.windows)
    blocks = np.repeat(week_blocks(largest.decision_ms, weeks=BLOCK_WEEKS), CONTEXT - 1)
    generator = np.random.default_rng(0)
    columns = [np.abs(generator.standard_normal(blocks.size)) for _ in range(2)]
    started = clock()
    block_bootstrap(
        columns,
        blocks,
        _ratio_of_medians,
        key=f"{STEMS[prepared.part]}/estimate",
        n_resamples=PROBE_RESAMPLES,
        min_blocks=1,
    )
    seconds = clock() - started
    if not math.isfinite(seconds) or seconds < 0:
        raise TokenizerDiagError(f"the timed bootstrap took {seconds!r} s: the clock is not usable")
    return seconds / PROBE_RESAMPLES * RESAMPLES * POOLED_STATISTICS * len(sets)


def estimate_diag(prepared: PreparedDiag, *, steps: TokenizerSteps) -> dict[str, Any]:
    """The measured estimate: one batch of :data:`BATCH_WINDOWS` windows of the first source
    encoded as the run encodes them, timed after an untimed warm-up, for every window of the
    part (the reference included), plus the measured cost of the pooled-row intervals; CPU
    seconds are wall x the runtime's threads (D-033)."""
    from ..runtime import require_configured

    if not isinstance(prepared, PreparedDiag):
        raise TypeError(f"expected a PreparedDiag, got {type(prepared).__name__}")
    tokenizer = steps.load_tokenizer()
    _check_tokenizer(tokenizer)
    first = next(iter(prepared.sources.values()))
    sample = first.values[:BATCH_WINDOWS]
    _diagnose(tokenizer, _sample(first, sample))  # warm-up, untimed
    started = steps.clock()
    _diagnose(tokenizer, _sample(first, sample))
    seconds = steps.clock() - started
    if not math.isfinite(seconds) or seconds <= 0:
        raise TokenizerDiagError(f"the timed batch took {seconds!r} s: the clock is not usable")
    per_window = seconds / sample.shape[0]
    windows = sum(item.windows for item in prepared.sources.values()) + prepared.reference.windows
    statistics_seconds = _statistics_seconds(prepared, steps.clock)
    threads = require_configured().threads
    cpu_seconds = (windows * per_window + statistics_seconds) * threads
    return {
        "study": STEMS[prepared.part],
        "windows": windows,
        "batch_windows": int(sample.shape[0]),
        "seconds_per_window": per_window,
        "encoding_seconds": windows * per_window,
        "statistics_seconds": statistics_seconds,
        "threads": threads,
        "cpu_seconds": cpu_seconds,
        "cpu_hours": cpu_seconds / 3600.0,
        "gate3_cpu_seconds": GATE3_CPU_SECONDS,
        "needs_owner_approval": cpu_seconds > GATE3_CPU_SECONDS,
    }


def _sample(windows: DiagWindows, values: np.ndarray) -> DiagWindows:
    return DiagWindows(
        source=windows.source,
        decision_ms=windows.decision_ms[: values.shape[0]].copy(),
        values=values.copy(),
        dropped=(),
    )


def require_within_gate3(estimate: Mapping[str, Any]) -> None:
    """Refuse (:class:`TokenizerDiagRefused`) an estimate above :data:`GATE3_CPU_SECONDS`
    (SPEC section 17, gate 3): the diagnostic has no approval flag."""
    if not isinstance(estimate, Mapping):
        raise TypeError(f"estimate must be a mapping, got {type(estimate).__name__}")
    cpu_seconds = estimate.get("cpu_seconds")
    if (
        isinstance(cpu_seconds, bool)
        or not isinstance(cpu_seconds, (int, float))
        or not math.isfinite(cpu_seconds)
        or cpu_seconds < 0
    ):
        raise TokenizerDiagError(
            f"estimate cpu_seconds must be a finite number >= 0, got {cpu_seconds!r}"
        )
    if cpu_seconds > GATE3_CPU_SECONDS:
        raise TokenizerDiagRefused(
            f"the tokenizer diagnostic needs about {cpu_seconds:.0f} CPU-seconds, above the "
            f"{GATE3_CPU_SECONDS} of gate 3 (SPEC section 17): it has no approval flag; show "
            "the estimate to the owner"
        )


# -- statistics -------------------------------------------------------------------------------


def _mean(column: np.ndarray) -> float:
    return float(np.mean(column))


def _root_mean(column: np.ndarray) -> float:
    return float(np.sqrt(np.mean(column)))


def _median(column: np.ndarray) -> float:
    return float(np.median(column))


def _point_only(key: str, reason: str) -> dict[str, Any]:
    from ..eval.bootstrap import METHOD_POINT_ONLY, bootstrap_seed

    return {
        "key": key,
        "point": None,
        "low": None,
        "high": None,
        "n": 0,
        "n_blocks": 0,
        "n_resamples": RESAMPLES,
        "confidence": CONFIDENCE,
        "seed": bootstrap_seed(key),
        "failed_resamples": 0,
        "method": METHOD_POINT_ONLY,
        "n_eff": None,
        "deff": None,
        "reason": reason,
    }


def _bootstrap(
    key: str,
    columns: Sequence[np.ndarray],
    blocks: np.ndarray,
    statistic: Callable[..., float],
    *,
    share: bool = False,
) -> dict[str, Any]:
    from ..eval.bootstrap import block_bootstrap

    arrays = [np.asarray(column, dtype=np.float64) for column in columns]
    if not arrays[0].size:
        return _point_only(key, "no rows in this sample")
    return block_bootstrap(
        arrays,
        np.asarray(blocks),
        statistic,
        key=key,
        n_resamples=RESAMPLES,
        confidence=CONFIDENCE,
        share=share,
    ).as_dict()


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def matched_indices(rows: int, matched: int) -> np.ndarray:
    """``np.linspace(0, rows - 1, matched).round()`` as int64: ``matched`` rows of ``rows`` in
    time order, evenly spread (the codebook's matched row counts)."""
    if not _is_int(rows) or not _is_int(matched) or not 1 <= matched <= rows:
        raise TokenizerDiagError(f"need 1 <= matched <= rows, got {matched!r} of {rows!r}")
    return np.linspace(0, rows - 1, matched).round().astype(np.int64)


def codebook_statistics(counts: np.ndarray, bits: int) -> dict[str, Any]:
    """Usage of a codebook of ``2 ** bits`` codes from its ``counts``: distinct codes, unused
    share, entropy (natural log), perplexity, normalised entropy (over ``log(2 ** bits)``), the
    fewest codes covering 50, 90 and 99 % of rows, the per-bit share of ones (bit ``b`` of code
    ``c`` is ``(c >> b) & 1``, as ``BSQuantizer.bits_to_indices`` weighs it) and the dead bits
    (share below 1 % or above 99 %)."""
    array = np.asarray(counts)
    if not _is_int(bits) or bits < 1:
        raise TokenizerDiagError(f"bits must be a positive integer, got {bits!r}")
    if array.ndim != 1 or array.size != 2**bits or not np.issubdtype(array.dtype, np.integer):
        raise TokenizerDiagError(f"counts must be {2**bits} integers, got shape {array.shape}")
    array = array.astype(np.int64)
    total = int(array.sum())
    if (array < 0).any() or total < 1:
        raise TokenizerDiagError("counts must be non-negative with at least one row")
    probabilities = array[array > 0] / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    ordered = np.cumsum(np.sort(array)[::-1])
    coverage = {
        f"{percent}%": int(np.searchsorted(ordered * 100, percent * total, side="left")) + 1
        for percent in COVERAGE_PERCENTS
    }
    codes = np.arange(array.size, dtype=np.int64)
    ones = [float(array[((codes >> bit) & 1) == 1].sum() / total) for bit in range(bits)]
    low, high = DEAD_BIT_BOUNDS
    distinct = int(np.count_nonzero(array))
    return {
        "codes": int(array.size),
        "rows": total,
        "distinct": distinct,
        "unused_share": 1.0 - distinct / array.size,
        "entropy": entropy,
        "perplexity": math.exp(entropy),
        "normalised_entropy": entropy / (bits * math.log(2.0)),
        "coverage": coverage,
        "bit_ones": ones,
        "dead_bits": [bit for bit, share in enumerate(ones) if share < low or share > high],
    }


def js_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Jensen-Shannon divergence, log base 2 (0 to 1), of two count vectors of one codebook."""
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64)
    if p.ndim != 1 or p.shape != q.shape or (p < 0).any() or (q < 0).any():
        raise TokenizerDiagError("JS divergence needs two non-negative count vectors of one size")
    if not p.sum() > 0 or not q.sum() > 0:
        raise TokenizerDiagError("JS divergence needs at least one row on each side")
    p, q = p / p.sum(), q / q.sum()
    middle = (p + q) / 2.0

    def kl(a: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / middle[mask])))

    return 0.5 * kl(p) + 0.5 * kl(q)


def _counts(codes: np.ndarray, bits: int, label: str) -> np.ndarray:
    flat = np.asarray(codes, dtype=np.int64).ravel()
    if flat.size and (flat.min() < 0 or flat.max() >= 2**bits):
        raise TokenizerDiagError(f"{label} codes outside [0, 2 ** {bits})")
    return np.bincount(flat, minlength=2**bits)


def _codebook(arrays: Mapping[str, np.ndarray], bits: int, matched: int) -> dict[str, Any]:
    s1 = arrays["s1"].ravel()
    s2 = arrays["s2"].ravel()
    pairs = s1 * 2**bits + s2
    index = matched_indices(int(s1.size), matched)
    sections: dict[str, Any] = {"rows": int(s1.size), "matched_rows": int(matched)}
    for name, codes in (("s1", s1), ("s2", s2)):
        statistics = codebook_statistics(_counts(codes, bits, name), bits)
        statistics["distinct_matched"] = int(np.unique(codes[index]).size)
        sections[name] = statistics
    sections["pairs"] = {
        "codes": 2 ** (2 * bits),
        "distinct": int(np.unique(pairs).size),
        "distinct_matched": int(np.unique(pairs[index]).size),
    }
    return sections


def _set_results(
    part: str,
    name: str,
    windows: DiagWindows,
    arrays: Mapping[str, np.ndarray],
    *,
    bits: int,
    matched: int,
) -> dict[str, Any]:
    """Statistics (with their intervals), descriptive values and codebook usage of one set."""
    from ..eval.bootstrap import week_blocks

    prefix = f"{STEMS[part]}/{name}"
    blocks = week_blocks(windows.decision_ms, weeks=BLOCK_WEEKS)
    statistics: dict[str, dict[str, Any]] = {}
    descriptive: dict[str, Any] = {}

    def add(
        statistic: str,
        columns: Sequence[np.ndarray],
        function: Callable[..., float],
        *,
        share: bool = False,
        row_blocks: np.ndarray | None = None,
    ) -> None:
        statistics[statistic] = _bootstrap(
            f"{prefix}/{statistic}",
            columns,
            blocks if row_blocks is None else row_blocks,
            function,
            share=share,
        )

    for column, channel in enumerate(FEATURE_COLUMNS):
        saturated = arrays["saturated"][:, column]
        # a per-window fraction of 512 entries: a mean of fractions, not a share of windows
        add(f"saturation_context_{channel}", [saturated / CONTEXT], _mean)
        add(f"saturation_windows_{channel}", [(saturated > 0).astype(float)], _mean, share=True)
        descriptive[f"max_abs_z_{channel}"] = float(arrays["max_abs_z"][:, column].max())
        if "next_z" in arrays:
            next_abs = np.abs(arrays["next_z"][:, column])
            add(f"saturation_next_{channel}", [(next_abs > CLIP).astype(float)], _mean, share=True)
            descriptive[f"max_abs_z_next_{channel}"] = float(next_abs.max())
    row_blocks = np.repeat(blocks, CONTEXT)
    r = arrays["r"]
    for label, key in (("", "full"), ("pre_", "pre")):
        for column, channel in enumerate(FEATURE_COLUMNS):
            add(f"{label}rmse_{channel}", [arrays[f"{key}_mse"][:, column]], _root_mean)
            add(f"{label}rmse_last24_{channel}", [arrays[f"{key}_mse_last"][:, column]], _root_mean)
        errors = np.abs(arrays[f"{key}_close_error"])
        add(f"{label}close_error_median_abs_bp", [errors.ravel()], _median, row_blocks=row_blocks)
        for quantile, text in ((0.90, "p90"), (0.99, "p99")):
            descriptive[f"{label}close_error_{text}_abs_bp"] = float(np.quantile(errors, quantile))
        descriptive[f"{label}close_error_max_abs_bp"] = float(errors.max())
        er = arrays[f"{key}_er"]
        valid = np.isfinite(er) & np.isfinite(r)
        return_blocks = np.broadcast_to(blocks[:, None], er.shape)[valid]
        add(
            f"{label}rho",
            [np.abs(er[valid]), np.abs(r[valid])],
            _ratio_of_medians,
            row_blocks=return_blocks,
        )
        descriptive[f"{label}returns_dropped"] = int(er.size - int(valid.sum()))
    for column, margin in enumerate(BIT_MARGINS):  # per-window fractions of coordinates
        add(f"bit_margin_{margin:g}", [arrays["margins"][:, column]], _mean)
    mismatch: dict[str, Any] = {}
    for variant in MISMATCH_VARIANTS:
        key = f"mismatch_{variant}"
        if key not in arrays:
            continue
        for column, suffix in enumerate(MISMATCH_COLUMNS):
            statistic = f"{key}_{suffix}"
            values = arrays[key][:, column]
            if variant == CONTROL_VARIANT:
                descriptive[statistic] = float(values.mean())
            else:  # per-window fractions of the 512 (or 24) rows
                add(statistic, [values], _mean)
        mismatch[variant] = {
            "statistics_rows": VARIANTS[variant][0],
            "rows": VARIANTS[variant][1],
        }
    finite_returns = r[np.isfinite(r)]
    descriptive["median_abs_return_bp"] = (
        _finite(BASIS_POINTS * np.median(np.abs(finite_returns))) if finite_returns.size else None
    )
    descriptive["tick_bp"] = _finite(np.median(BASIS_POINTS * TICK_USDT / arrays["last_close"]))
    return {
        "windows": windows.windows,
        "blocks": int(np.unique(blocks).size),
        "statistics": statistics,
        "descriptive": descriptive,
        "codebook": _codebook(arrays, bits, matched),
        "mismatch_variants": mismatch,
    }


def _check_control(arrays: Mapping[str, np.ndarray]) -> None:
    """The ``context_only`` positive control: not one token may differ (a TokenizerDiagError
    otherwise: the comparison of the other variants would not mean what it says)."""
    control = arrays.get(f"mismatch_{CONTROL_VARIANT}")
    if control is not None and np.any(control != 0):
        raise TokenizerDiagError(
            "the context-only positive control changed tokens of the 512 context rows (mean "
            f"share {float(control[:, 0].mean()):.3g} s1): the encoder's prefix is not "
            "reproduced, so the token mismatch cannot be read"
        )


def _divergences(
    part: str,
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    bits: int,
    training_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if part == "training":
        s1 = arrays["binance"]["s1"]
        half = s1.shape[0] // 2
        if half < 1:
            return {"js_first_second_half": None}
        return {
            "js_first_second_half": js_divergence(
                _counts(s1[:half], bits, "s1"), _counts(s1[half:], bits, "s1")
            )
        }
    if training_report is None:
        raise TokenizerDiagError("Part B needs Part A's report (prepare_diag reads it)")
    training = np.asarray(training_report["s1_counts"], dtype=np.int64)
    out: dict[str, Any] = {}
    for source in SOURCES[part]:
        out[f"js_training_validation_{source}"] = js_divergence(
            training, _counts(arrays[source]["s1"], bits, "s1")
        )
    out["js_binance_bingx"] = js_divergence(
        _counts(arrays["binance"]["s1"], bits, "s1"), _counts(arrays["bingx"]["s1"], bits, "s1")
    )
    out["js_first_second_half_part_a"] = (training_report.get("divergences") or {}).get(
        "js_first_second_half"
    )
    return out


def _flag(
    flags: list[dict[str, Any]],
    name: str,
    source: str,
    statistic: str,
    value: Any,
    raised: bool,
    *,
    note: str = "",
) -> None:
    if value is None or not raised:
        return
    threshold = FLAGS[name]
    flags.append(
        {
            "flag": name,
            "source": source,
            "statistic": statistic,
            "value": float(value),
            "threshold": threshold,
            "rule": FLAG_RULES[name],
            "message": f"{source}: {FLAG_RULES[name]} ({statistic} = {float(value):.4g}, "
            f"threshold {threshold:g}{note})",
        }
    )


def _raise_flags(
    part: str, results: Mapping[str, Mapping[str, Any]], training_perplexity: float | None
) -> list[dict[str, Any]]:
    """The pre-registered flags raised by the real sources' point estimates."""
    flags: list[dict[str, Any]] = []
    for source in SOURCES[part]:
        result = results[source]
        statistics = result["statistics"]

        def point(statistic: str, statistics: Mapping[str, Any] = statistics) -> Any:
            return (statistics.get(statistic) or {}).get("point")

        for channel in PRICE_CHANNELS:
            statistic = f"saturation_context_{channel}"
            value = point(statistic)
            _flag(
                flags,
                "price_saturation_context",
                source,
                statistic,
                value,
                value is not None and value > FLAGS["price_saturation_context"],
            )
            if part == "training":
                statistic = f"saturation_next_{channel}"
                value = point(statistic)
                _flag(
                    flags,
                    "price_saturation_next",
                    source,
                    statistic,
                    value,
                    value is not None and value > FLAGS["price_saturation_next"],
                )
        for channel in VOLUME_CHANNELS:
            statistic = f"saturation_context_{channel}"
            value = point(statistic)
            _flag(
                flags,
                "volume_amount_saturation",
                source,
                statistic,
                value,
                value is not None and value > FLAGS["volume_amount_saturation"],
            )
        value = point("rho")
        _flag(flags, "rho", source, "rho", value, value is not None and value > FLAGS["rho"])
        s1 = result["codebook"]["s1"]
        _flag(
            flags,
            "s1_normalised_entropy",
            source,
            "s1 normalised_entropy",
            s1["normalised_entropy"],
            s1["normalised_entropy"] < FLAGS["s1_normalised_entropy"],
        )
        _flag(
            flags,
            "s1_unused_share",
            source,
            "s1 unused_share",
            s1["unused_share"],
            s1["unused_share"] > FLAGS["s1_unused_share"],
        )
        if part == "validation" and training_perplexity:
            ratio = s1["perplexity"] / training_perplexity
            _flag(
                flags,
                "validation_perplexity_ratio",
                source,
                "s1 perplexity / Part A's",
                ratio,
                ratio < FLAGS["validation_perplexity_ratio"],
            )
        if part == "training":
            value = point(TOKEN_MISMATCH_STATISTIC)
            shares = [
                f"{token} {point(f'mismatch_r1_{token}'):.4g}"
                for token in ("s1", "s2")
                if point(f"mismatch_r1_{token}") is not None
            ]
            _flag(
                flags,
                "r1_token_mismatch",
                source,
                TOKEN_MISMATCH_STATISTIC,
                value,
                value is not None and value > FLAGS["r1_token_mismatch"],
                note="".join(f"; {share}" for share in shares),
            )
    return flags


# -- figures ----------------------------------------------------------------------------------


def _figures(
    part: str,
    results: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, np.ndarray],
    directory: Path,
) -> list[dict[str, str]]:
    """``figures/<stem>-saturation.png`` and ``-codebook.png`` (identical bytes on every render)."""
    from matplotlib.ticker import FormatStrFormatter, NullFormatter

    from ..report.figures import figures_dir, new_figure, save_png

    stem = STEMS[part]
    target = figures_dir(directory)
    names = [*SOURCES[part], REFERENCE]
    panels = [("saturation_context", "context entries"), ("saturation_windows", "windows")]
    if part == "training":
        panels.append(("saturation_next", "next rows (row 512)"))
    figure, axes = new_figure(1, len(panels))
    width = 0.8 / len(names)
    positions = np.arange(len(FEATURE_COLUMNS), dtype=np.float64)
    for axis, (prefix, title) in zip(np.atleast_1d(axes), panels, strict=True):
        for index, name in enumerate(names):
            statistics = results[name]["statistics"]
            entries = [statistics.get(f"{prefix}_{channel}") for channel in FEATURE_COLUMNS]
            if all(entry is None for entry in entries):
                continue
            heights = [100.0 * ((entry or {}).get("point") or 0.0) for entry in entries]
            offset = positions + (index - (len(names) - 1) / 2.0) * width
            axis.bar(offset, heights, width=width, label=name)
        axis.set_xticks(positions)
        axis.set_xticklabels(list(FEATURE_COLUMNS), rotation=45)
        axis.set_ylabel("percent with |z| > 5")
        axis.set_title(title)
    np.atleast_1d(axes)[0].legend(loc="upper left")
    figure.suptitle(
        "Clip saturation, point estimates (intervals in the report); N(0,1) gives "
        f"{100 * SATURATION_COMPARATOR:.1e} %"
    )
    out = [{"name": "saturation", **save_png(figure, target / f"{stem}-saturation.png")}]

    figure, axes = new_figure(1, 2)
    for name in names:
        share = np.sort(counts[name])[::-1] / max(1, int(counts[name].sum()))
        positive = share[share > 0]
        axes[0].plot(np.arange(1, positive.size + 1), positive, label=name)
        codebook = results[name]["codebook"]
        ones = [*codebook["s1"]["bit_ones"], *codebook["s2"]["bit_ones"]]
        axes[1].plot(np.arange(len(ones)), ones, marker="o", linestyle="none", label=name)
    axes[0].set_yscale("log")
    # plain tick labels: the default log formatter renders through mathtext
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.0e"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel("s1 code rank")
    axes[0].set_ylabel("share of rows")
    axes[0].set_title("s1 code usage (rank-frequency)")
    axes[0].legend(loc="upper right")
    for bound in DEAD_BIT_BOUNDS:
        axes[1].axhline(bound, color="tab:gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("bit (s1 bits first, then s2)")
    axes[1].set_ylabel("share of ones")
    axes[1].set_title("share of ones per bit (dead: < 1 % or > 99 %)")
    axes[1].legend(loc="center right")
    figure.suptitle(f"Codebook usage, {stem}")
    out.append({"name": "codebook", **save_png(figure, target / f"{stem}-codebook.png")})
    return out


# -- the run ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagResult:
    """What a run wrote and what the command prints and records in the ledger."""

    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    details: dict[str, Any]  # for the ledger's exploration entry
    flags: tuple[dict[str, Any], ...]
    summary_lines: tuple[str, ...]


def _retraining_md_sha256() -> str | None:
    path = REPO_ROOT / RETRAINING_MD_RELPATH
    return sha256_file(path) if path.is_file() else None


def _windows_record(part: str, prepared: PreparedDiag) -> dict[str, Any]:
    planned = len(decisions_ms(part))
    record: dict[str, Any] = {
        source: {
            "planned": planned,
            "kept": windows.windows,
            "dropped": len(windows.dropped),
            "dropped_windows": list(windows.dropped),
            "first": utc_text(int(windows.decision_ms[0])),
            "last": utc_text(int(windows.decision_ms[-1])),
        }
        for source, windows in prepared.sources.items()
    }
    record[REFERENCE] = {
        "planned": len(REFERENCE_SEEDS),
        "kept": prepared.reference.windows,
        "dropped": 0,
        "dropped_windows": [],
    }
    return record


def _has_interval(result: Mapping[str, Any]) -> bool:
    return result.get("low") is not None and result.get("high") is not None


def run_diag(
    prepared: PreparedDiag,
    *,
    params: Params,
    steps: TokenizerSteps,
    directory: Path | None = None,
    progress: Callable[[str], None] = print,
) -> DiagResult:
    """Encode every window of ``prepared`` with ``steps.load_tokenizer()``, compute the
    statistics, intervals and flags of design section 4 and write the report (its definition
    verified again: a changed one is refused)."""
    import torch

    from ..report.writer import report_header, write_report

    if not isinstance(prepared, PreparedDiag):
        raise TypeError(f"expected a PreparedDiag, got {type(prepared).__name__}")
    part = _check_part(prepared.part)
    _check_params(params)
    directory = _report_dir(directory)
    stem = STEMS[part]
    if record_diag_config(part, directory=directory) != prepared.config_sha256:
        raise TokenizerDiagError(f"{stem}: the windows were prepared under another definition")
    tokenizer = steps.load_tokenizer()
    bits, _ = _check_tokenizer(tokenizer)
    training = prepared.training_report
    training_perplexity: float | None = None
    if part == "validation":
        if training is None:
            raise TokenizerDiagError("Part B needs Part A's report (prepare_diag reads it)")
        for key in ("s1_counts", "s2_counts"):
            if len(training[key]) != 2**bits:
                raise TokenizerDiagError(
                    f"Part A's {key} hold {len(training[key])} codes, the tokenizer {2**bits}"
                )
        training_perplexity = codebook_statistics(np.asarray(training["s1_counts"]), bits)[
            "perplexity"
        ]
    sets: dict[str, DiagWindows] = {**prepared.sources, REFERENCE: prepared.reference}
    started = steps.clock()
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, windows in sets.items():
        progress(f"{stem}: encoding {windows.windows} {name} windows")
        arrays[name] = _diagnose(tokenizer, windows)
        _check_control(arrays[name])
    matched = min(int(item["s1"].size) for item in arrays.values())
    progress(f"{stem}: statistics and {RESAMPLES}-resample intervals")
    results = {
        name: _set_results(part, name, windows, arrays[name], bits=bits, matched=matched)
        for name, windows in sets.items()
    }
    counts = {name: _counts(arrays[name]["s1"], bits, "s1") for name in sets}
    divergences = _divergences(part, arrays, bits, training)
    if part == "validation":
        for source in SOURCES[part]:
            ratio = results[source]["codebook"]["s1"]["perplexity"] / training_perplexity
            divergences[f"perplexity_ratio_{source}"] = ratio
    flags = _raise_flags(part, results, training_perplexity)
    seconds = steps.clock() - started
    with_intervals = sum(
        _has_interval(result) for item in results.values() for result in item["statistics"].values()
    )
    plan = diag_plan(part)
    header = report_header(
        stem,
        plan,
        statistics_with_intervals=with_intervals,
        seeds=_seeds(part),
        samples=_samples(results),
        comparators=_comparators(),
        directory=directory,
    )
    if header["config_sha256"] != prepared.config_sha256:
        raise TokenizerDiagError(
            f"{stem}: the recorded definition changed after the windows were prepared"
        )
    body: dict[str, Any] = {
        "retraining_md_sha256": _retraining_md_sha256(),
        "part": part,
        "params_version": params.version,
        "tokenizer": {**_tokenizer_identity(), "s1_bits": bits, "s2_bits": bits},
        "windows": _windows_record(part, prepared),
        "results": results,
        "divergences": divergences,
        "flags": flags,
        "flag_rules": plan["flags"],
        "figures": _figures(part, results, counts, directory),
        "runtime": {
            "threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "seconds": seconds,
            "batch_windows": BATCH_WINDOWS,
        },
    }
    if part == "training":
        body["s1_counts"] = counts["binance"].tolist()
        body["s2_counts"] = _counts(arrays["binance"]["s2"], bits, "s2").tolist()
    else:
        body["part_a"] = {
            "config_sha256": training.get("config_sha256") if training else None,
            "s1_perplexity": training_perplexity,
        }
    payload = {**body, **header}
    json_path, markdown_path = write_report(stem, payload, _markdown(payload), directory=directory)
    details = _details(part, prepared, payload, json_path, flags)
    return DiagResult(
        report=payload,
        json_path=json_path,
        markdown_path=markdown_path,
        details=details,
        flags=tuple(flags),
        summary_lines=tuple(summary_lines(payload)),
    )


def _seeds(part: str) -> dict[str, Any]:
    return {
        "bootstrap": "numpy seed of each interval: the first 4 bytes of sha256('kronos-agent/"
        f"bootstrap/v1/' + key), key {STEMS[part]}/<set>/<statistic>",
        "reference": f"features.synthetic_window seeds {REFERENCE_SEEDS[0]}..{REFERENCE_SEEDS[-1]}",
        "tokenizer": "none: the quantiser is a sign function and decode is deterministic",
    }


def _samples(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "windows": item["windows"],
            "rows": item["codebook"]["rows"],
            "blocks": item["blocks"],
        }
        for name, item in results.items()
    }


def _comparators() -> dict[str, Any]:
    return {
        "saturation": f"P(|N(0,1)| > 5) = {SATURATION_COMPARATOR:.4g}",
        "close_error": "the median |1h return| and one 0.1 USDT tick, in bp (per set)",
        "reference": "100 synthetic GBM windows (featureless)",
        "divergence": "Part A's first half against its second half",
        "perplexity": "Part A's s1 perplexity (Part B)",
    }


def _details(
    part: str,
    prepared: PreparedDiag,
    payload: Mapping[str, Any],
    json_path: Path,
    flags: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """What the ledger's exploration entry records (interfaces 10.3)."""
    details: dict[str, Any] = {
        "part": part,
        "report_sha256": sha256_file(json_path),
        "config_sha256": prepared.config_sha256,
        "windows_planned": len(decisions_ms(part)),
        "windows_kept": {source: windows.windows for source, windows in prepared.sources.items()},
        "statistics_with_intervals": payload["statistics_with_intervals"],
        "flags_raised": sorted({f"{flag['flag']}/{flag['source']}" for flag in flags}),
    }
    if part == "training":
        statistics = payload["results"]["binance"]["statistics"]
        for token in ("any", "s1", "s2"):
            details[f"r1_{token}_mismatch_share"] = (
                statistics.get(f"mismatch_r1_{token}") or {}
            ).get("point")
    return details


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    """What the command prints after a run (no path)."""
    part = report.get("part")
    stem = report.get("study")
    windows = report.get("windows") or {}
    kept = ", ".join(
        f"{name} {item.get('kept')}/{item.get('planned')}" for name, item in windows.items()
    )
    lines = [
        (
            f"{stem}: windows kept {kept}; {report.get('statistics_with_intervals')} statistics "
            f"with intervals (about {report.get('expected_chance_misses')} expected to miss)"
        )
    ]
    for name, item in (report.get("results") or {}).items():
        statistics = item.get("statistics") or {}
        s1 = (item.get("codebook") or {}).get("s1") or {}
        close = _interval_text(statistics.get("close_error_median_abs_bp"))
        lines.append(
            f"{name}: close median |e| {close} bp; rho {_interval_text(statistics.get('rho'))}; "
            "s1 perplexity "
            f"{_number(s1.get('perplexity'))} (normalised entropy "
            f"{_number(s1.get('normalised_entropy'))}, {s1.get('distinct')} codes used)"
        )
        if part == "training" and TOKEN_MISMATCH_STATISTIC in statistics:
            descriptive = item.get("descriptive") or {}
            lines.append(
                f"{name}: token mismatch of the context rows (s1 or s2): r1 "
                f"{_interval_text(statistics.get('mismatch_r1_any'))} (s1 "
                f"{_interval_text(statistics.get('mismatch_r1_s1'))}, s2 "
                f"{_interval_text(statistics.get('mismatch_r1_s2'))}), template "
                f"{_interval_text(statistics.get('mismatch_template_any'))}, context-only "
                f"{_number(descriptive.get('mismatch_context_only_any'))} (control)"
            )
    flags = report.get("flags") or []
    lines.append(f"flags raised: {len(flags)} (descriptive, no automatic action)")
    return lines


# -- Markdown ---------------------------------------------------------------------------------


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _interval_text(result: Mapping[str, Any] | None) -> str:
    if not result:
        return "n/a"
    point = _number(result.get("point"))
    if _has_interval(result):
        return f"{point} [{_number(result['low'])}, {_number(result['high'])}]"
    return point


def _cell(text: Any) -> str:
    return str(text).replace("|", "/")


def _statistics_table(item: Mapping[str, Any]) -> list[str]:
    lines = [
        "| statistic | point | 95 % interval | n | blocks | method / note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in (item.get("statistics") or {}).items():
        interval = (
            f"[{_number(result['low'])}, {_number(result['high'])}]"
            if _has_interval(result)
            else "n/a"
        )
        note = "; ".join(str(part) for part in (result.get("method"), result.get("reason")) if part)
        lines.append(
            f"| {name} | {_number(result.get('point'))} | {interval} | "
            f"{_number(result.get('n'))} | {_number(result.get('n_blocks'))} | {_cell(note)} |"
        )
    descriptive = item.get("descriptive") or {}
    if descriptive:
        text = ", ".join(f"{name} {_number(value)}" for name, value in descriptive.items())
        lines += ["", f"Descriptive (no interval): {text}."]
    return lines


def _codebook_table(results: Mapping[str, Mapping[str, Any]]) -> list[str]:
    header = (
        "| set | codes | rows | matched rows | s1 distinct (matched) | s2 distinct (matched) | "
        "pairs distinct (matched) | s1 entropy | s1 perplexity | s1 normalised | s1 unused | "
        "s1 codes for 50/90/99 % | s2 perplexity | dead bits s1 / s2 |"
    )
    lines = [header, "|" + " --- |" * 14]
    for name, item in results.items():
        book = item["codebook"]
        s1, s2, pairs = book["s1"], book["s2"], book["pairs"]
        coverage = "/".join(str(s1["coverage"][f"{p}%"]) for p in COVERAGE_PERCENTS)
        lines.append(
            f"| {name} | {s1['codes']} | {book['rows']} | {book['matched_rows']} | "
            f"{s1['distinct']} ({s1['distinct_matched']}) | "
            f"{s2['distinct']} ({s2['distinct_matched']}) | "
            f"{pairs['distinct']} ({pairs['distinct_matched']}) | {_number(s1['entropy'])} | "
            f"{_number(s1['perplexity'])} | {_number(s1['normalised_entropy'])} | "
            f"{_number(s1['unused_share'])} | {coverage} | {_number(s2['perplexity'])} | "
            f"{s1['dead_bits'] or '-'} / {s2['dead_bits'] or '-'} |"
        )
    return lines


def _markdown(payload: Mapping[str, Any]) -> str:
    part = payload["part"]
    stem = payload["study"]
    title = (
        "Part A: training segment (Binance), before registration"
        if part == "training"
        else "Part B: validation segment (Binance and BingX real market), inputs only"
    )
    code = payload.get("code") or {}
    tokenizer = payload.get("tokenizer") or {}
    runtime = payload.get("runtime") or {}
    hashes = ", ".join(
        f"{name} `{value}`" for name, value in (tokenizer.get("sha256") or {}).items()
    )
    misses = _number(payload.get("expected_chance_misses"))
    lines = [
        f"# Tokenizer diagnostic, {title}",
        "",
        (
            "Design: docs/RETRAINING.md section 4 (SPEC 9.1). A descriptive study with 0 tests: "
            "the tokenizer stays frozen and the clip stays 5; a flag only motivates the later, "
            "separate tokenizer test of SPEC 9.2. No automatic action follows from it."
        ),
        "",
        (
            f"- Study `{stem}`; config sha256 `{payload.get('config_sha256')}`; "
            f"docs/RETRAINING.md sha256 `{payload.get('retraining_md_sha256')}`; docs/DATA.md "
            f"sha256 `{payload.get('data_md_sha256')}`"
        ),
        (
            f"- Code: commit `{code.get('commit')}`, dirty {_number(code.get('dirty'))}, source "
            f"tree sha256 `{code.get('source_tree_sha256')}`; params version "
            f"`{payload.get('params_version')}`"
        ),
        (
            f"- Tokenizer `{tokenizer.get('repo_id')}` (files sha256 {hashes}), s1 "
            f"{tokenizer.get('s1_bits')} bits, s2 {tokenizer.get('s2_bits')} bits"
        ),
        (
            f"- Runtime: CPU, {runtime.get('threads')} threads, deterministic "
            f"{_number(runtime.get('deterministic'))}, torch {runtime.get('torch_version')}, "
            f"{_number(runtime.get('seconds'))} s"
        ),
        (
            f"- Generated at {payload.get('generated_at')}; tests consumed "
            f"{payload.get('tests_consumed')}"
        ),
        (
            f"- Statistics with intervals: {payload.get('statistics_with_intervals')}; expected "
            f"to miss their true value by chance at 95 %: {misses}"
        ),
        (
            f"- Intervals: 95 % percentile block bootstrap, {RESAMPLES} resamples, blocks of "
            f"{BLOCK_WEEKS} calendar weeks of D; seed key `{stem}/<set>/<statistic>`. p90, p99, "
            "maxima, comparators, codebook statistics and divergences are descriptive. The "
            "synthetic reference has no decision hour (one block): point estimates only."
        ),
        "",
        "## Windows",
        "",
        "| set | planned | kept | dropped |",
        "| --- | --- | --- | --- |",
    ]
    windows = payload.get("windows") or {}
    for name, item in windows.items():
        lines.append(f"| {name} | {item['planned']} | {item['kept']} | {item['dropped']} |")
    for name, item in windows.items():
        dropped = item.get("dropped_windows") or []
        for entry in dropped[:_MAX_DROPPED_SHOWN]:
            ranges = ", ".join(f"[{a}, {b})" for a, b in entry["missing"][:_MAX_DROPPED_SHOWN])
            lines.append(f"- {name} {entry['decision']} dropped: missing {ranges}")
        if len(dropped) > _MAX_DROPPED_SHOWN:
            lines.append(f"- {name}: {len(dropped) - _MAX_DROPPED_SHOWN} more dropped (JSON)")
    lines += ["", "## Flags (pre-registered, descriptive)", ""]
    flags = payload.get("flags") or []
    if flags:
        lines += [f"- WARNING {flag['flag']}: {flag['message']}" for flag in flags]
    else:
        lines.append("- None raised.")
    lines += ["", "Rules (point estimates of the real sources):", ""]
    lines += [
        f"- `{name}` ({rule['threshold']:g}): {rule['rule']}"
        for name, rule in (payload.get("flag_rules") or {}).items()
    ]
    for name, item in (payload.get("results") or {}).items():
        lines += ["", f"## Statistics: {name}", ""]
        lines += _statistics_table(item)
    lines += ["", "## Codebook usage (descriptive)", ""]
    lines += _codebook_table(payload.get("results") or {})
    divergences = payload.get("divergences") or {}
    if divergences:
        lines += [""]
        lines += [f"- {name}: {_number(value)}" for name, value in divergences.items()]
    lines += ["", "## Figures", ""]
    for figure in payload.get("figures") or []:
        lines.append(f"![{figure['name']}]({figure['relpath']}) (sha256 `{figure['sha256']}`)")
    return "\n".join(lines) + "\n"


__all__ = [
    "BATCH_WINDOWS",
    "BIT_MARGINS",
    "BLOCK_WEEKS",
    "CONTEXT",
    "FLAGS",
    "FLAG_RULES",
    "GRID",
    "LAST_ROWS",
    "MISMATCH_COLUMNS",
    "NEXT_ROWS",
    "PARTS",
    "PART_A_FIRST",
    "PART_A_LAST",
    "REFERENCE_SEEDS",
    "STEMS",
    "TOKEN_MISMATCH_STATISTIC",
    "DiagResult",
    "DiagWindows",
    "PreparedDiag",
    "TokenizerDiagError",
    "TokenizerDiagRefused",
    "TokenizerSteps",
    "codebook_statistics",
    "decisions_ms",
    "diag_plan",
    "estimate_diag",
    "js_divergence",
    "matched_indices",
    "normalise_rows",
    "prepare_diag",
    "real_steps",
    "record_diag_config",
    "require_within_gate3",
    "run_diag",
    "summary_lines",
]
