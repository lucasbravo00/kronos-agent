"""The pilot check of the retraining (``docs/RETRAINING.md`` 6.2 steps 3-4):
``kronos-agent finetune pilot-check``.

A recorded study with 0 tests on training-segment data: the pilot's own inner validation. On the
``G8`` hours of the pilot fold's inner validation (168 for ``q-2024-07``), with Binance 1h
windows and the governed fan configuration (``N = 4``, 20 trajectories, CPU, 2 threads), three
fan batches run under the named model-period opt-in ``pilot-check`` (shipped forecasts before
the cutoff, the pilot's inside its own inner validation; recorded in the report): the pilot
checkpoint and the shipped weights on stream ``forecast`` (common random numbers) and the
shipped weights on ``forecast-aa`` (the two-seed floor). Decisions any batch drops are dropped
from every statistic and listed.

Statistics, point estimates only (168 decisions touch 2 four-week blocks, fewer than the 20 an
interval needs, D-035):

- ``a_same``: the share of decisions where the pilot and the shipped model give the same signal
  under common seeds; ``a_floor`` the same for the two shipped streams; the mean ``|delta
  p_up|`` of pilot and shipped;
- the KL divergence of the two models' first-step s1 distributions: teacher-forced logits
  (``finetune.nll.teacher_forced_logits``, the shipped tokenizer, eval mode, CPU) of the pilot
  export's NLL4-TF inputs restricted to these decisions, ``log_softmax`` in float64 at T = 1
  without top-p; the mean over decisions of ``sum p_pilot (log p_pilot - log p_shipped)``
  (primary), the reverse and the maximum (descriptive);
- ``d(D) = CRPS_pilot - CRPS_shipped`` at ``h = 4`` (fair CRPS of the 20 paths' ``1e4 log(close
  / c_ref)`` against the realised close at ``D + 3 h``, ``c_ref`` the close of ``D - 1 h``, both
  read from the NLL4-TF inputs and ``c_ref`` checked against every fan): its SD and lag-1..6
  autocorrelation, projected (approximate) to the minimum detectable difference of the
  walk-forward's 4,380 decisions (:func:`mdd_projection`); its mean is descriptive, and no skill
  is read from it;
- the relative weight change of every floating tensor of the model's ``state_dict``;
- the pilot's NLL4-TF curve from its fold record (``runner_result.json``, whose sha256 the
  registry entry records), shown because fold 1 is not blinded (design 6.3).

Reading (pre-registered, design 6.2 step 4): ``a_same >= 0.98`` (included) is ``NO-GO``: the
fine-tune barely changed the model, the work stops and the owner chooses; otherwise ``GO``.

Report ``reports/finetune/pilot-check-<recipe8>.json`` and ``.md`` (the definition recorded once
as ``pilot-check-<recipe8>.config.json``) and the figure of the NLL4-TF curve. Gate 3 and the
ledger entries are the command's (interfaces 14.2): :func:`estimate_check`, then
``kronos.batch.require_gate3``, then the approval, then :func:`run_check`. Not here: the compute
control and the gate-3 ask for the remaining folds (design 6.2 step 5).

Spend control (design 3.5): :func:`run_check` charges the run's CPU-seconds (D-033: the larger of
the CPU time of this process and its children and the wall time times the governed threads) from
its start, and stops once they pass its limit (:func:`spend_limit`: 1.25 times the CPU-hours the
owner approved, or gate 3's 3,600 CPU-seconds without a figure), checked after every fan, before
each batch and before each teacher-forced pass: :class:`PilotCheckStopped` (a refusal, exit 2);
the fans stored so far are kept, and a rerun reuses them.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data.partition import MAX_CONTEXT, MODEL_PERIOD_PILOT_CHECK
from ..data.timeutil import HOUR_MS, from_ms, parse_utc_hour, to_ms, utc_text
from ..hashing import sha256_file
from ..kronos.features import FEATURE_COLUMNS
from ..params import Params
from ..paths import REPO_ROOT, data_dir, reports_dir
from .folds import Fold, grid_decisions_ms
from .recipe import Recipe, recipe8, recipe_id
from .registry import CheckpointRef, checkpoint_id_for

if TYPE_CHECKING:
    from ..data.forecast_store import StoredFan
    from ..kronos.batch import FanSummary, PreparedBatch
    from ..kronos.weights import LoadedModels
    from .export_csv import ExportRecord
    from .nll import NllInputs, TeacherForced

STEM_PREFIX = "pilot-check"
GRID = "G8"
SOURCE = "binance"
STREAM_SAME = "forecast"
STREAM_FLOOR = "forecast-aa"
HORIZON = 4
TRAJECTORIES = 20
GO_NO_GO_THRESHOLD = 0.98
AUTOCORRELATION_LAGS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
PROJECTED_DECISIONS = 4380
PROJECTED_BLOCKS = 27
KL_SECONDS_PER_DECISION_PRIOR = 0.16  # M8's 0.155 s for 4 steps, per model and decision

BATCHES: tuple[str, ...] = ("pilot", "shipped", "floor")
BATCH_STREAMS: Mapping[str, str] = MappingProxyType(
    {"pilot": STREAM_SAME, "shipped": STREAM_SAME, "floor": STREAM_FLOOR}
)
PROJECTED_PHASES = 2  # the two 8-hour phases of G4; the second counted as adding nothing
Z_SUM = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)  # 2.801585...
READING_GO = "GO"
READING_NO_GO = "NO-GO"
NO_GO_OPTIONS = (
    "the fine-tune barely changed the model; the work stops and the owner chooses: run anyway "
    "as a measured null; the GPU with the template budget; or lr 4e-5"
)
ROLE_TEXT = (
    "the named model-period opt-in 'pilot-check' (docs/RETRAINING.md 6.2): the shipped weights "
    "forecast hours before their cutoff and the pilot hours inside its own inner validation, "
    "training-segment hours where no skill is read"
)
GATE3_CPU_SECONDS = 3600  # SPEC section 17 (kronos.batch.GATE3_CPU_SECONDS)
APPROVAL_TOLERANCE = 1.25  # D-040, D-046 (kronos.batch.GATE3_APPROVAL_TOLERANCE)
BASIS_POINTS = 1e4
CLOSE_INDEX = FEATURE_COLUMNS.index("close")
SCHEMA_VERSION = 1
FINETUNE_REPORTS_SUBDIR = "finetune"
RETRAINING_MD_RELPATH = "docs/RETRAINING.md"
INTERVALS_REASON = (
    "the decisions touch 2 four-week blocks, fewer than the 20 an interval needs (D-035): "
    "point estimates only"
)


class PilotCheckError(ValueError):
    """The pilot check cannot run as defined, or its inputs are inconsistent (exit 1)."""


class PilotCheckStopped(PermissionError):
    """The run's measured spend passed its limit (design 3.5): it stopped between fans, the fans
    stored so far kept (exit 2)."""


def _process_cpu_seconds() -> float:
    """CPU-seconds of this process (every thread) and of its reaped children."""
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


@dataclass(frozen=True)
class PilotInputs:
    """The pilot, as ``walkforward.resolve_pilot`` finds it: its recipe and fold, its checkpoint
    named with the sha256 of its registry line (from the ledger's fit entry), its fold record and
    that record's sha256, and its verified export."""

    recipe: Recipe
    fold: Fold
    checkpoint: CheckpointRef
    fit_record_path: Path
    fit_record_sha256: str
    export: ExportRecord


@dataclass(frozen=True)
class PilotCheckSteps:
    """The heavy parts, injectable (:func:`real_steps`; tests pass stand-ins). Each is called as
    its real counterpart: ``prepare_tasks(tasks, params=, root=, checkpoint=, period_role=)``,
    ``measure_seconds_per_fan(params=, checkpoint=)``, ``run_fan_batch(prepared, workers=,
    params=, root=, progress=)`` (``progress(done, total, summary)`` after each fan),
    ``load_models(name, checkpoint=)``, ``load_fan(path)`` and ``teacher_forced_logits(tokenizer,
    model, inputs)``; ``clock()`` and ``cpu_seconds()`` measure the spend."""

    prepare_tasks: Callable[..., PreparedBatch]  # kronos.batch.prepare_tasks
    measure_seconds_per_fan: Callable[..., float]  # kronos.batch.measure_seconds_per_fan
    run_fan_batch: Callable[..., list[FanSummary]]  # kronos.batch.run_fan_batch
    load_models: Callable[..., LoadedModels]  # kronos.weights.load_models
    load_fan: Callable[[Path], StoredFan]  # data.forecast_store.load_fan
    teacher_forced_logits: Callable[..., TeacherForced]  # finetune.nll.teacher_forced_logits
    clock: Callable[[], float] = time.monotonic  # wall seconds
    cpu_seconds: Callable[[], float] = _process_cpu_seconds  # this process and its children


def _load_models_quietly(name: str, *, checkpoint: CheckpointRef | None = None) -> LoadedModels:
    """``load_models`` with the upstream Hub mixin's stdout line captured."""
    from ..kronos.weights import load_models

    with contextlib.redirect_stdout(io.StringIO()):
        return load_models(name, checkpoint=checkpoint)


def real_steps() -> PilotCheckSteps:
    """The real steps (imported here, after the runtime is configured)."""
    from ..data.forecast_store import load_fan
    from ..kronos import batch
    from .nll import teacher_forced_logits

    return PilotCheckSteps(
        prepare_tasks=batch.prepare_tasks,
        measure_seconds_per_fan=batch.measure_seconds_per_fan,
        run_fan_batch=batch.run_fan_batch,
        load_models=_load_models_quietly,
        load_fan=load_fan,
        teacher_forced_logits=teacher_forced_logits,
        clock=time.monotonic,
        cpu_seconds=_process_cpu_seconds,
    )


# -- the definition ---------------------------------------------------------------------------


def stem_for(recipe: Recipe) -> str:
    """``pilot-check-<recipe8>``: the study of the pilot of ``recipe``."""
    if not isinstance(recipe, Recipe):
        raise TypeError(f"expected a Recipe, got {type(recipe).__name__}")
    return f"{STEM_PREFIX}-{recipe8(recipe)}"


def check_decisions_ms(fold: Fold) -> tuple[int, ...]:
    """The ``G8`` hours of the fold's inner validation ``[inner start, CSV end)``; every decision's
    512 context candles and :data:`HORIZON` targets must open inside the fold's CSV range
    (:class:`PilotCheckError` otherwise)."""
    if not isinstance(fold, Fold):
        raise TypeError(f"expected a Fold, got {type(fold).__name__}")
    decisions = grid_decisions_ms(fold.inner_validation_start_ms, fold.csv_end_ms, GRID)
    if not decisions:
        raise PilotCheckError(f"fold {fold.run_label}: no {GRID} hour in its inner validation")
    for decision in decisions:
        first = decision - MAX_CONTEXT * HOUR_MS
        last = decision + (HORIZON - 1) * HOUR_MS
        if first < fold.csv_start_ms or last >= fold.csv_end_ms:
            raise PilotCheckError(
                f"fold {fold.run_label}: the decision {utc_text(decision)} reads candles in "
                f"[{utc_text(first)}, {utc_text(last + HOUR_MS)}), outside its CSV range "
                f"[{utc_text(fold.csv_start_ms)}, {utc_text(fold.csv_end_ms)})"
            )
    return decisions


def _statistic_definitions() -> dict[str, str]:
    return {
        "a_same": "share of decisions where signal(pilot, forecast) == signal(shipped, forecast)",
        "a_floor": "share of decisions where signal(shipped, forecast) == signal(shipped, "
        "forecast-aa)",
        "mean_abs_delta_p_up": "mean |p_up(pilot) - p_up(shipped)|, stream forecast",
        "kl_pilot_shipped": "mean over decisions of sum p_pilot (log p_pilot - log p_shipped) "
        "of the step-0 s1 distributions (teacher-forced logits, the shipped tokenizer, "
        "log_softmax in float64, T = 1, no top-p); the primary KL",
        "kl_shipped_pilot": "the reverse (descriptive)",
        "kl_pilot_shipped_max": "the largest per-decision kl_pilot_shipped (descriptive)",
        "d": "d(D) = CRPS_pilot - CRPS_shipped at h = 4, stream forecast, fair CRPS (K = 20) of "
        "x_k = 1e4 log(close_k(4) / c_ref) against y = 1e4 log(close(D + 3 h) / c_ref), in bp; "
        "c_ref and the realised close are rows 511 and 515 of the NLL4-TF inputs; a fan with a "
        "close <= 0 drops its decision from d (counted)",
        "d_mean_bp": "descriptive only: no skill is read from it",
        "d_sd_bp": "the SD of d (ddof 1)",
        "rho_k": "eval.bootstrap.autocorrelation(d, lags 1..6)",
        "relative_weight_change": "sqrt(sum_t ||theta_pilot,t - theta_0,t||^2) / sqrt(sum_t "
        "||theta_0,t||^2) over every floating tensor of model.state_dict(), float64; with the "
        "share of tensors changed and max |delta|",
        "nll4_tf_curve": "per pseudo-epoch NLL4-TF, training loss and vendored validation loss "
        "of the fold record (epoch 0 = the shipped weights), the selected pseudo-epoch and the "
        "stop reason",
    }


def _projection_definition() -> dict[str, Any]:
    return {
        "label": "approximate",
        "decisions": PROJECTED_DECISIONS,
        "phases": PROJECTED_PHASES,
        "blocks": PROJECTED_BLOCKS,
        "lags": list(AUTOCORRELATION_LAGS),
        "inflation": "max(1, 1 + 2 * sum_k rho_k)",
        "n_eff": "decisions / phases / inflation: the 8-hour autocorrelation applied to one of "
        "the two 8-hour phases of G4, the second phase counted as adding no information; the "
        "27 four-week blocks not modelled",
        "se": "d_sd_bp / sqrt(n_eff)",
        "mdd_bp": "(z_0.975 + z_0.80) * se",
        "z_sum": Z_SUM,
        "mdd_relative": "mdd_bp / mean CRPS_shipped",
    }


def check_plan(recipe: Recipe, fold: Fold) -> dict[str, Any]:
    """The recorded definition of the pilot check of ``recipe`` on ``fold``: the decision rule,
    streams, role, statistics and their formulas, the reading, the projection, the fan horizon
    and trajectories and the expected checkpoint id. No params version, no code hash."""
    decisions = check_decisions_ms(fold)
    return {
        "schema_version": SCHEMA_VERSION,
        "study": stem_for(recipe),
        "design": f"{RETRAINING_MD_RELPATH} 6.2 steps 3-4",
        "tests": 0,
        "recipe": {"name": recipe.name, "recipe_id": recipe_id(recipe)},
        "fold": {
            "label": fold.label,
            "run_label": fold.run_label,
            "layout": fold.layout,
            "csv_start": utc_text(fold.csv_start_ms),
            "csv_end": utc_text(fold.csv_end_ms),
            "inner_validation_start": utc_text(fold.inner_validation_start_ms),
        },
        "checkpoint_id": checkpoint_id_for(recipe8(recipe), fold.run_label),
        "decisions": {
            "grid": GRID,
            "rule": "the G8 hours (D.hour % 8 == 0) of the fold's inner validation [inner start, "
            "CSV end); each decision's 512 context candles and 4 targets open inside the CSV "
            "range",
            "first": utc_text(decisions[0]),
            "last": utc_text(decisions[-1]),
            "planned": len(decisions),
            "dropped": "a decision any batch drops, or missing from the NLL4-TF inputs, is "
            "dropped from every statistic and listed",
        },
        "source": {
            "store": f"candles/{SOURCE}/1h",
            "windows": "kronos.batch.prepare_tasks (data.windows.StoreWindowSource)",
        },
        "role": {"period_role": MODEL_PERIOD_PILOT_CHECK, "text": ROLE_TEXT},
        "fans": {
            "pilot": {"model": "the pilot's registered checkpoint", "stream": STREAM_SAME},
            "shipped": {"model": "Kronos-small as shipped", "stream": STREAM_SAME},
            "floor": {"model": "Kronos-small as shipped", "stream": STREAM_FLOOR},
            "horizon": HORIZON,
            "trajectories": TRAJECTORIES,
            "workers": 1,
            "seeds": "kronos_agent.seeds of D, the stream and the project symbol (pilot and "
            "shipped share the forecast stream: common random numbers)",
        },
        "nll_inputs": "the pilot export's nll_inputs.npz (its sha256 verified) restricted to "
        "the decisions",
        "statistics": _statistic_definitions(),
        "intervals": INTERVALS_REASON,
        "reading": {
            "statistic": "a_same",
            "threshold": GO_NO_GO_THRESHOLD,
            "rule": f"{READING_NO_GO} when a_same >= {GO_NO_GO_THRESHOLD:g} (included), "
            f"otherwise {READING_GO}",
            "no_go": NO_GO_OPTIONS,
            "source": "SPEC 12's live-against-replay success level for 'the same system'",
        },
        "projection": _projection_definition(),
    }


def _report_dir(directory: Path | None) -> Path:
    return Path(directory) if directory is not None else reports_dir() / FINETUNE_REPORTS_SUBDIR


def record_check_config(recipe: Recipe, fold: Fold, *, directory: Path | None = None) -> str:
    """Record (or verify) :func:`check_plan` as ``<stem>.config.json``; its sha256. Allowed before
    the pilot exists; a changed definition is refused (``ConfigChangedError``)."""
    from ..report.writer import record_config

    return record_config(
        stem_for(recipe), check_plan(recipe, fold), directory=_report_dir(directory)
    )


# -- preparation ------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PreparedCheck:
    """The three prepared batches (``pilot``, ``shipped``, ``floor``) over the kept decisions,
    the NLL4-TF inputs of those decisions and what was dropped."""

    inputs: PilotInputs
    config_sha256: str
    batches: Mapping[str, PreparedBatch]
    nll_inputs: NllInputs
    decisions_ms: tuple[int, ...]
    dropped: tuple[dict[str, Any], ...]


def _check_params(params: Params) -> None:
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    kronos = params.kronos
    if kronos.horizon_n != HORIZON or kronos.trajectories != TRAJECTORIES:
        raise PilotCheckError(
            f"the pilot check is defined for fans of {HORIZON} steps and {TRAJECTORIES} "
            f"trajectories; params.yaml has {kronos.horizon_n} and {kronos.trajectories}"
        )


def _check_inputs(inputs: PilotInputs) -> None:
    if not isinstance(inputs, PilotInputs):
        raise TypeError(f"expected PilotInputs, got {type(inputs).__name__}")
    if not isinstance(inputs.recipe, Recipe) or not isinstance(inputs.fold, Fold):
        raise TypeError("PilotInputs needs a Recipe and a Fold")
    if not isinstance(inputs.checkpoint, CheckpointRef):
        raise TypeError("PilotInputs needs a CheckpointRef")
    expected = checkpoint_id_for(recipe8(inputs.recipe), inputs.fold.run_label)
    if inputs.checkpoint.checkpoint_id != expected:
        raise PilotCheckError(
            f"the pilot of fold {inputs.fold.run_label} is {expected}, not "
            f"{inputs.checkpoint.checkpoint_id}"
        )
    if inputs.checkpoint.line_sha256 is None:
        raise PilotCheckError(
            "the pilot's checkpoint is named with the sha256 of its registry line (the ledger's "
            "fit entry records it)"
        )


def _verify_inputs(inputs: PilotInputs) -> Any:
    """The fold record still hashes to its sha256, the registry entry still is the recorded line
    and describes this pilot, and the export verifies and is the entry's; the entry."""
    from .export_csv import read_export
    from .registry import find_entry

    _check_inputs(inputs)
    try:
        actual = sha256_file(inputs.fit_record_path)
    except OSError as exc:
        raise PilotCheckError(
            f"cannot read the pilot's fold record ({type(exc).__name__})"
        ) from None
    if actual != inputs.fit_record_sha256:
        raise PilotCheckError(
            f"the pilot's fold record hashes to {actual[:12]}..., not to the recorded "
            f"{inputs.fit_record_sha256[:12]}...: it changed after it was recorded"
        )
    entry = find_entry(inputs.checkpoint)  # RegistryChainError: the line changed
    document = entry.document
    recipe, fold, export = inputs.recipe, inputs.fold, inputs.export
    expected = {
        "kind": "fold",
        "recipe_id": recipe_id(recipe),
        "fold": fold.label,
        "replicate": fold.replicate,
        "data_end_ms": fold.csv_end_ms,
        "fit_record_sha256": inputs.fit_record_sha256,
        "export_content_sha256": export.content_sha256,
        "nll_inputs_sha256": export.nll_inputs_sha256,
    }
    differing = [key for key, value in expected.items() if document.get(key) != value]
    if differing:
        raise PilotCheckError(
            f"the registry entry of {entry.checkpoint_id} does not describe this pilot: "
            f"{', '.join(differing)} differ"
        )
    verified = read_export(export.directory)
    if (
        verified.content_sha256 != export.content_sha256
        or verified.nll_inputs_sha256 != export.nll_inputs_sha256
    ):
        raise PilotCheckError(f"the export of fold {fold.label} changed since it was resolved")
    content = verified.content
    if content.get("fold") != fold.label or content.get("recipe_id") != recipe_id(recipe):
        raise PilotCheckError(f"the export is not fold {fold.label} of recipe {recipe.name}")
    return entry


def _prepare_batches(
    decisions: Sequence[int],
    inputs: PilotInputs,
    *,
    params: Params,
    steps: PilotCheckSteps,
    root: Path | None,
) -> dict[str, PreparedBatch]:
    from ..kronos.batch import FanTask

    hours = [from_ms(decision) for decision in decisions]
    batches: dict[str, PreparedBatch] = {}
    for name in BATCHES:
        tasks = [FanTask(SOURCE, hour, BATCH_STREAMS[name]) for hour in hours]
        batches[name] = steps.prepare_tasks(
            tasks,
            params=params,
            root=root,
            checkpoint=inputs.checkpoint if name == "pilot" else None,
            period_role=MODEL_PERIOD_PILOT_CHECK,
        )
        _check_batch(name, batches[name], inputs)
    return batches


def _check_batch(name: str, batch: PreparedBatch, inputs: PilotInputs) -> None:
    """A prepared batch is the one this check asked for: the role, the model and the stream."""
    if batch.period_role != MODEL_PERIOD_PILOT_CHECK:
        raise PilotCheckError(f"the {name} batch runs the role {batch.period_role!r}")
    if batch.evaluation_length not in (None, HORIZON):
        raise PilotCheckError(f"the {name} batch has fans of {batch.evaluation_length} steps")
    checkpoint = batch.checkpoint
    if name == "pilot":
        wanted = inputs.checkpoint
        if (
            checkpoint is None
            or checkpoint.checkpoint_id != wanted.checkpoint_id
            or checkpoint.line_sha256 != wanted.line_sha256
        ):
            raise PilotCheckError(
                "the pilot batch is not prepared with the pilot's checkpoint and its registry line"
            )
    elif checkpoint is not None:
        raise PilotCheckError(f"the {name} batch runs {checkpoint.checkpoint_id}, not the shipped")
    streams = {task.stream for task in batch.tasks}
    if streams - {BATCH_STREAMS[name]} or any(task.source != SOURCE for task in batch.tasks):
        raise PilotCheckError(f"the {name} batch holds other tasks than {SOURCE} {streams}")


def _kept_ms(batch: PreparedBatch) -> list[int]:
    return [to_ms(task.decision_hour) for task in batch.tasks]


def _note_dropped(dropped: dict[int, dict[str, Any]], name: str, batch: PreparedBatch) -> None:
    """One entry per dropped decision: the batches that dropped it and the missing ranges."""
    for item in batch.dropped:
        decision = int(item["decision_ms"])
        entry = dropped.setdefault(
            decision,
            {
                "decision": utc_text(decision),
                "decision_ms": decision,
                "reason": "a missing candle in its window",
                "batches": [],
                "missing": item.get("missing_utc"),
            },
        )
        if name not in entry["batches"]:
            entry["batches"].append(name)


def prepare_check(
    inputs: PilotInputs,
    *,
    params: Params,
    steps: PilotCheckSteps,
    root: Path | None = None,
    directory: Path | None = None,
) -> PreparedCheck:
    """Record (or verify) the definition; verify the fold record's sha256, the registry line and
    the export; check params (``horizon_n`` 4, 20 trajectories); restrict the export's NLL4-TF
    inputs to the planned decisions; prepare the three batches (``root`` is the data root,
    ``directory`` the report directory). A decision missing from the NLL4-TF inputs or dropped
    by any batch is left out of all three batches and listed once."""
    from .nll import NllInputs, load_nll_inputs

    _check_inputs(inputs)
    config_sha256 = record_check_config(inputs.recipe, inputs.fold, directory=directory)
    _check_params(params)
    _verify_inputs(inputs)
    planned = check_decisions_ms(inputs.fold)
    export = inputs.export
    nll_all = load_nll_inputs(export.nll_inputs_path, expected_sha256=export.nll_inputs_sha256)
    available = {int(value) for value in nll_all.decision_ms}
    dropped: dict[int, dict[str, Any]] = {
        decision: {
            "decision": utc_text(decision),
            "decision_ms": decision,
            "reason": "not in the export's NLL4-TF inputs (a missing candle)",
            "batches": [],
            "missing": None,
        }
        for decision in planned
        if decision not in available
    }
    wanted = [decision for decision in planned if decision in available]
    if not wanted:
        raise PilotCheckError(f"no pilot-check decision of fold {inputs.fold.label} has inputs")
    batches = _prepare_batches(wanted, inputs, params=params, steps=steps, root=root)
    kept = set(wanted)
    for name, batch in batches.items():
        kept &= set(_kept_ms(batch))
        _note_dropped(dropped, name, batch)
    kept_ms = tuple(sorted(kept))
    if not kept_ms:
        raise PilotCheckError(f"every pilot-check decision of fold {inputs.fold.label} dropped")
    if any(_kept_ms(batch) != list(kept_ms) for batch in batches.values()):
        # a decision dropped by one batch leaves the others: prepare all three on the rest
        batches = _prepare_batches(kept_ms, inputs, params=params, steps=steps, root=root)
        if any(_kept_ms(batch) != list(kept_ms) for batch in batches.values()):
            raise PilotCheckError("the three batches keep different decisions")
    index = np.searchsorted(nll_all.decision_ms, np.asarray(kept_ms, dtype=np.int64))
    nll = NllInputs(
        decision_ms=nll_all.decision_ms[index].copy(), values=nll_all.values[index].copy()
    )
    return PreparedCheck(
        inputs=inputs,
        config_sha256=config_sha256,
        batches=MappingProxyType(batches),
        nll_inputs=nll,
        decisions_ms=kept_ms,
        dropped=tuple(dropped[decision] for decision in sorted(dropped)),
    )


# -- the estimate -----------------------------------------------------------------------------


def estimate_check(
    prepared: PreparedCheck, *, params: Params, steps: PilotCheckSteps
) -> dict[str, Any]:
    """The measured estimate: one fan of the pilot timed after a warm-up
    (``steps.measure_seconds_per_fan``), the pending fans of the three batches at that rate
    (``kronos.batch.estimate``, CPU-seconds = wall x threads) and the two teacher-forced passes
    at :data:`KL_SECONDS_PER_DECISION_PRIOR` per model and decision."""
    from ..kronos.batch import estimate as batch_estimate

    if not isinstance(prepared, PreparedCheck):
        raise TypeError(f"expected a PreparedCheck, got {type(prepared).__name__}")
    _check_params(params)
    seconds = steps.measure_seconds_per_fan(
        params=params, checkpoint=prepared.batches["pilot"].checkpoint
    )
    threads = params.kronos.threads
    per_batch = {
        name: batch_estimate(batch, seconds_per_fan=seconds, threads=threads, workers=1)
        for name, batch in prepared.batches.items()
    }
    fans_cpu_seconds = float(sum(item["cpu_seconds"] for item in per_batch.values()))
    decisions = len(prepared.decisions_ms)
    kl_cpu_seconds = 2 * decisions * KL_SECONDS_PER_DECISION_PRIOR * threads
    cpu_seconds = fans_cpu_seconds + kl_cpu_seconds
    return {
        "study": stem_for(prepared.inputs.recipe),
        "decisions": decisions,
        "seconds_per_fan": float(seconds),
        "threads": threads,
        "workers": 1,
        "batches": per_batch,
        "pending": int(sum(item["pending"] for item in per_batch.values())),
        "fans_cpu_seconds": fans_cpu_seconds,
        "kl_cpu_seconds": kl_cpu_seconds,
        "kl_seconds_per_decision_prior": KL_SECONDS_PER_DECISION_PRIOR,
        "cpu_seconds": cpu_seconds,
        "cpu_hours": cpu_seconds / 3600.0,
        "gate3_cpu_seconds": GATE3_CPU_SECONDS,
        "needs_owner_approval": cpu_seconds > GATE3_CPU_SECONDS,
    }


def spend_limit(approved_cpu_hours: float | None) -> float:
    """The CPU-seconds a run may spend before it stops (design 3.5): :data:`APPROVAL_TOLERANCE`
    times the CPU-hours the owner approved, or :data:`GATE3_CPU_SECONDS` without a figure (the
    estimate was below gate 3)."""
    if approved_cpu_hours is None:
        return float(GATE3_CPU_SECONDS)
    if (
        isinstance(approved_cpu_hours, bool)
        or not isinstance(approved_cpu_hours, (int, float))
        or not math.isfinite(approved_cpu_hours)
        or approved_cpu_hours <= 0
    ):
        raise ValueError(
            f"the approved CPU-hours must be a finite number > 0, got {approved_cpu_hours!r}"
        )
    return APPROVAL_TOLERANCE * float(approved_cpu_hours) * 3600.0


class _Spend:
    """A run's CPU-seconds since it started, charged as D-033 counts CPU work (the larger of the
    measured CPU time and the wall time times the threads), against its limit."""

    def __init__(
        self, *, stem: str, limit: float, approved: float | None, threads: int, steps: Any
    ) -> None:
        self.stem = stem
        self.limit = limit
        self.approved = approved
        self.threads = threads
        self.steps = steps
        self.wall0 = float(steps.clock())
        self.cpu0 = float(steps.cpu_seconds())

    def spent(self) -> tuple[float, float, float]:
        """``(charged, cpu, wall)`` seconds since the start."""
        wall = max(0.0, float(self.steps.clock()) - self.wall0)
        cpu = max(0.0, float(self.steps.cpu_seconds()) - self.cpu0)
        return max(cpu, wall * self.threads), cpu, wall

    def check(self, where: str) -> None:
        """Stop (:class:`PilotCheckStopped`) once the charged spend passes the limit."""
        charged, _, _ = self.spent()
        if charged <= self.limit:
            return
        if self.approved is None:
            rule = f"gate 3's {GATE3_CPU_SECONDS} CPU-seconds, without an approved figure"
        else:
            rule = f"{APPROVAL_TOLERANCE:g} x the {self.approved:g} CPU-hours the owner approved"
        raise PilotCheckStopped(
            f"{self.stem}: the run's spend, {charged:.0f} CPU-seconds, passed its limit of "
            f"{self.limit:.0f} ({rule}) {where}: it stopped there (design 3.5); the fans stored "
            "so far are kept and a rerun reuses them; show the owner the spend before a rerun"
        )

    def record(self) -> dict[str, Any]:
        charged, cpu, wall = self.spent()
        return {
            "cpu_seconds": charged,
            "measured_cpu_seconds": cpu,
            "wall_seconds": wall,
            "threads": self.threads,
            "limit_cpu_seconds": self.limit,
            "approved_cpu_hours": self.approved,
            "rule": "charged: the larger of the measured CPU time and wall x threads (D-033); "
            f"stopped once past the limit, {APPROVAL_TOLERANCE:g} x the approved CPU-hours or "
            f"{GATE3_CPU_SECONDS} CPU-seconds without a figure (design 3.5)",
        }


# -- statistics -------------------------------------------------------------------------------


def go_no_go(a_same: float) -> str:
    """The pre-registered reading: :data:`READING_NO_GO` when ``a_same >= 0.98`` (0.98 included),
    :data:`READING_GO` otherwise."""
    if (
        isinstance(a_same, bool)
        or not isinstance(a_same, (int, float, np.floating))
        or not math.isfinite(a_same)
        or not 0.0 <= a_same <= 1.0
    ):
        raise PilotCheckError(f"a_same must be a share in [0, 1], got {a_same!r}")
    return READING_NO_GO if a_same >= GO_NO_GO_THRESHOLD else READING_GO


def mdd_projection(d_bp: np.ndarray) -> dict[str, Any]:
    """The approximate projection of the per-decision CRPS differences ``d_bp`` (bp) to the
    minimum detectable difference of the walk-forward (design 6.2 step 3, interfaces D10):
    ``inflation = max(1, 1 + 2 sum_k rho_k)`` over lags 1..6, ``n_eff = 4380 / 2 /
    inflation``, ``se = sd / sqrt(n_eff)``, ``mdd = (z_0.975 + z_0.80) se``. A constant series
    has no autocorrelation: inflation 1."""
    from ..eval.bootstrap import autocorrelation

    values = np.asarray(d_bp, dtype=np.float64)
    if values.ndim != 1 or values.size <= max(AUTOCORRELATION_LAGS) + 1:
        raise PilotCheckError(
            f"the projection needs more than {max(AUTOCORRELATION_LAGS) + 1} differences, got "
            f"{values.size}"
        )
    if not np.isfinite(values).all():
        raise PilotCheckError("the CRPS differences must be finite")
    sd = float(np.std(values, ddof=1))
    rho = autocorrelation(values, AUTOCORRELATION_LAGS)
    finite = all(math.isfinite(value) for value in rho.values())
    rho_sum = float(sum(rho.values())) if finite else None
    inflation = max(1.0, 1.0 + 2.0 * rho_sum) if rho_sum is not None else 1.0
    n_eff = PROJECTED_DECISIONS / PROJECTED_PHASES / inflation
    se = sd / math.sqrt(n_eff)
    return {
        "label": "approximate",
        "d_sd_bp": sd,
        "rho": {str(lag): (value if math.isfinite(value) else None) for lag, value in rho.items()},
        "rho_sum": rho_sum,
        "inflation": inflation,
        "n_eff": n_eff,
        "se_bp": se,
        "z_sum": Z_SUM,
        "mdd_bp": Z_SUM * se,
        "decisions": PROJECTED_DECISIONS,
        "phases": PROJECTED_PHASES,
        "blocks": PROJECTED_BLOCKS,
        "assumptions": "the 8-hour autocorrelation applied to one of the two 8-hour phases of "
        "G4, the second phase counted as adding no information, the 27 four-week blocks not "
        "modelled",
    }


def _float64(value: Any) -> np.ndarray | None:
    """A float64 numpy copy of a floating tensor or array, ``None`` for any other dtype."""
    if hasattr(value, "detach"):
        import torch

        if not value.is_floating_point():
            return None
        return value.detach().cpu().to(torch.float64).numpy()  # to the CPU first (MPS fact)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        return None
    return array.astype(np.float64)


def relative_weight_change(state: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """``sqrt(sum_t ||state_t - base_t||^2) / sqrt(sum_t ||base_t||^2)`` over every floating
    tensor of two ``state_dict``s of one model (float64), with the share of tensors changed and
    the largest ``|delta|``. Both must hold the same names and shapes."""
    if set(state) != set(base):
        raise PilotCheckError("the two state_dicts hold different tensors")
    delta_squares = base_squares = max_abs = 0.0
    tensors = changed = 0
    for name in sorted(base):
        after, before = _float64(state[name]), _float64(base[name])
        if after is None or before is None:
            if (after is None) != (before is None):
                raise PilotCheckError(f"tensor {name} is floating in one state_dict only")
            continue
        if after.shape != before.shape:
            raise PilotCheckError(f"tensor {name} has shapes {after.shape} and {before.shape}")
        delta = after - before
        if not (np.isfinite(after).all() and np.isfinite(before).all()):
            raise PilotCheckError(f"tensor {name} is not finite")
        tensors += 1
        delta_squares += float(np.sum(delta * delta))
        base_squares += float(np.sum(before * before))
        if np.any(delta != 0):
            changed += 1
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
    if tensors == 0 or base_squares <= 0:
        raise PilotCheckError("no floating tensor with a non-zero norm to compare")
    return {
        "relative": math.sqrt(delta_squares) / math.sqrt(base_squares),
        "delta_norm": math.sqrt(delta_squares),
        "base_norm": math.sqrt(base_squares),
        "tensors": tensors,
        "changed_tensors": changed,
        "changed_share": changed / tensors,
        "max_abs_delta": max_abs,
    }


def _first_step_kl(pilot: TeacherForced, shipped: TeacherForced) -> dict[str, Any]:
    """KL divergences of the two models' step-0 s1 distributions, in nats (float64)."""
    import torch

    if not torch.equal(pilot.s1_targets, shipped.s1_targets):
        raise PilotCheckError("the two teacher-forced passes scored different tokens")
    log_p = torch.log_softmax(pilot.s1_logits[:, 0, :].to(torch.float64), dim=-1)
    log_q = torch.log_softmax(shipped.s1_logits[:, 0, :].to(torch.float64), dim=-1)
    forward = (log_p.exp() * (log_p - log_q)).sum(dim=-1).numpy()
    reverse = (log_q.exp() * (log_q - log_p)).sum(dim=-1).numpy()
    if not (np.isfinite(forward).all() and np.isfinite(reverse).all()):
        raise PilotCheckError("a KL divergence is not finite")
    return {
        "kl_pilot_shipped": float(forward.mean()),
        "kl_shipped_pilot": float(reverse.mean()),
        "kl_pilot_shipped_max": float(forward.max()),
        "decisions": int(forward.size),
    }


def _by_decision(
    name: str, summaries: Sequence[FanSummary], decisions: Sequence[int]
) -> list[FanSummary]:
    """The batch's summaries in decision order, checked against the batch's stream and source."""
    found: dict[int, FanSummary] = {}
    for summary in summaries:
        if summary.source != SOURCE or summary.stream != BATCH_STREAMS[name]:
            raise PilotCheckError(
                f"the {name} batch returned a fan of {summary.source} {summary.stream}"
            )
        found[to_ms(parse_utc_hour(summary.decision_hour))] = summary
    missing = [decision for decision in decisions if decision not in found]
    if missing or len(found) != len(decisions):
        raise PilotCheckError(
            f"the {name} batch returned {len(found)} fans for {len(decisions)} decisions"
        )
    return [found[decision] for decision in decisions]


def _fan_closes(
    name: str,
    summaries: Sequence[FanSummary],
    decisions: Sequence[int],
    *,
    load_fan: Callable[[Path], StoredFan],
    base: Path,
) -> np.ndarray:
    """``(n, 20)`` closes at the last step of each stored fan."""
    closes = np.empty((len(summaries), TRAJECTORIES), dtype=np.float64)
    shape = (TRAJECTORIES, HORIZON, len(FEATURE_COLUMNS))
    for index, (summary, decision) in enumerate(zip(summaries, decisions, strict=True)):
        stored = load_fan(base / summary.relpath)
        paths = np.asarray(stored.paths, dtype=np.float64)
        meta = stored.meta
        if paths.shape != shape:
            raise PilotCheckError(f"the {name} fan of {utc_text(decision)} is not {shape}")
        if meta.get("decision_ms") != decision or meta.get("stream") != BATCH_STREAMS[name]:
            raise PilotCheckError(
                f"{summary.relpath} is not the {name} fan of {utc_text(decision)}"
            )
        closes[index] = paths[:, HORIZON - 1, CLOSE_INDEX]
    return closes


def _crps(
    pilot: np.ndarray, shipped: np.ndarray, c_ref: np.ndarray, realised: np.ndarray
) -> dict[str, Any]:
    """Fair CRPS of both models at ``h = 4`` and their per-decision difference ``d`` (bp)."""
    from ..eval.crps import fair_crps

    if not ((c_ref > 0).all() and (realised > 0).all()):
        raise PilotCheckError("a reference or realised close of the NLL4-TF inputs is not positive")
    valid = (pilot > 0).all(axis=1) & (shipped > 0).all(axis=1)
    if int(valid.sum()) <= max(AUTOCORRELATION_LAGS) + 1:
        raise PilotCheckError(
            f"only {int(valid.sum())} decisions have two valid fans (a close <= 0 makes a fan "
            "invalid): too few for the CRPS differences"
        )
    y = BASIS_POINTS * np.log(realised[valid] / c_ref[valid])
    x_pilot = BASIS_POINTS * np.log(pilot[valid] / c_ref[valid, None])
    x_shipped = BASIS_POINTS * np.log(shipped[valid] / c_ref[valid, None])
    crps_pilot = np.asarray(fair_crps(x_pilot, y), dtype=np.float64)
    crps_shipped = np.asarray(fair_crps(x_shipped, y), dtype=np.float64)
    return {
        "valid": valid,
        "crps_pilot": crps_pilot,
        "crps_shipped": crps_shipped,
        "d": crps_pilot - crps_shipped,
    }


def _check_c_ref(name: str, summaries: Sequence[FanSummary], c_ref: np.ndarray) -> None:
    """Every fan's ``last_close`` equals the NLL4-TF inputs' row 511 close, bit for bit."""
    for summary, value in zip(summaries, c_ref, strict=True):
        if float(summary.last_close) != float(value):
            raise PilotCheckError(
                f"misaligned inputs: the {name} fan of {summary.decision_hour} has last close "
                f"{summary.last_close!r}, the NLL4-TF inputs {float(value)!r}"
            )


def _read_fit_record(inputs: PilotInputs) -> dict[str, Any]:
    """The pilot's fold record (``runner_result.json``), its sha256 checked again."""
    try:
        data = Path(inputs.fit_record_path).read_bytes()
    except OSError as exc:
        raise PilotCheckError(
            f"cannot read the pilot's fold record ({type(exc).__name__})"
        ) from None
    from ..hashing import sha256_bytes

    if sha256_bytes(data) != inputs.fit_record_sha256:
        raise PilotCheckError("the pilot's fold record changed after it was recorded")
    try:
        record = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        raise PilotCheckError("the pilot's fold record is not ASCII JSON") from None
    if not isinstance(record, dict) or record.get("status") != "completed":
        raise PilotCheckError("the pilot's fold record is not a completed run")
    return record


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _nll4_tf_curve(record: Mapping[str, Any]) -> dict[str, Any]:
    """The NLL4-TF curve of the fold record: epoch 0 is the shipped weights (never selected)."""
    epochs = record.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise PilotCheckError("the pilot's fold record has no pseudo-epoch")
    curve: list[dict[str, Any]] = []
    for entry in epochs:
        if not isinstance(entry, Mapping) or isinstance(entry.get("pseudo_epoch"), bool):
            raise PilotCheckError("a pseudo-epoch of the fold record is malformed")
        epoch = entry.get("pseudo_epoch")
        if not isinstance(epoch, int) or epoch < 0:
            raise PilotCheckError("a pseudo-epoch of the fold record is malformed")
        curve.append(
            {
                "pseudo_epoch": epoch,
                "nll4_tf": _number_or_none(entry.get("nll4_tf")),
                "train_loss": _number_or_none(entry.get("train_loss")),
                "vendored_val_loss": _number_or_none(entry.get("vendored_val_loss")),
            }
        )
    selected = record.get("selected_pseudo_epoch")
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
        raise PilotCheckError("the pilot's fold record selected no pseudo-epoch")
    return {
        "epochs": curve,
        "selected_pseudo_epoch": selected,
        "stop_reason": record.get("stop_reason"),
        "run_id": record.get("run_id"),
        "epoch_0": "the shipped weights (a diagnostic, never selected)",
    }


def _check_models(shipped: LoadedModels, pilot: LoadedModels, checkpoint: CheckpointRef) -> None:
    """The two loads are the shipped weights and the pilot, with the same tokenizer."""
    from ..kronos.weights import repos_for_model

    if getattr(shipped, "checkpoint", None) is not None:
        raise PilotCheckError("the shipped models are a checkpoint")
    if getattr(pilot, "checkpoint", None) != checkpoint.checkpoint_id:
        raise PilotCheckError(f"the pilot models are not {checkpoint.checkpoint_id}")
    prefix = f"{repos_for_model('small')[1].repo_id}/"

    def tokenizer_hashes(models: LoadedModels) -> dict[str, str]:
        return {
            key: value
            for key, value in dict(models.weights_sha256).items()
            if key.startswith(prefix)
        }

    if not tokenizer_hashes(shipped) or tokenizer_hashes(shipped) != tokenizer_hashes(pilot):
        raise PilotCheckError("the pilot was not trained with the shipped tokenizer")


# -- the run ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """What a run wrote and what the command prints and records in the ledger."""

    report: dict[str, Any]
    json_path: Path
    markdown_path: Path
    details: dict[str, Any]  # for the ledger's exploration entry
    reading: str  # GO or NO-GO
    summary_lines: tuple[str, ...]


def _retraining_md_sha256() -> str | None:
    path = REPO_ROOT / RETRAINING_MD_RELPATH
    return sha256_file(path) if path.is_file() else None


def _agreement(summaries: Mapping[str, Sequence[FanSummary]]) -> tuple[float, float, float]:
    """``(a_same, a_floor, mean_abs_delta_p_up)`` of the three batches' fans, decision by
    decision: the pilot's and the shipped signals under common seeds, the two shipped streams'
    signals, and ``|p_up(pilot) - p_up(shipped)|`` on stream ``forecast``."""
    signals = {name: np.array([s.signal for s in summaries[name]]) for name in BATCHES}
    p_up = {name: np.array([s.p_up for s in summaries[name]], dtype=np.float64) for name in BATCHES}
    if len({values.size for values in signals.values()}) != 1 or not signals["pilot"].size:
        raise PilotCheckError("the three batches hold different numbers of fans")
    return (
        float(np.mean(signals["pilot"] == signals["shipped"])),
        float(np.mean(signals["shipped"] == signals["floor"])),
        float(np.mean(np.abs(p_up["pilot"] - p_up["shipped"]))),
    )


def run_check(
    prepared: PreparedCheck,
    *,
    params: Params,
    steps: PilotCheckSteps,
    root: Path | None = None,
    directory: Path | None = None,
    progress: Callable[[str], None] = print,
    approved_cpu_hours: float | None = None,
) -> CheckResult:
    """Run the three fan batches (one worker each), compute the statistics of design 6.2 step 3
    and the reading of step 4, and write the report. The fold record, the registry line and the
    export are verified again first; ``root`` is the data root. ``approved_cpu_hours`` is the
    figure the owner approved (``None`` below gate 3): the run stops once its spend passes
    :func:`spend_limit` of it (:class:`PilotCheckStopped`; module docstring)."""
    from ..report.writer import report_header, write_report

    if not isinstance(prepared, PreparedCheck):
        raise TypeError(f"expected a PreparedCheck, got {type(prepared).__name__}")
    inputs = prepared.inputs
    recipe, fold = inputs.recipe, inputs.fold
    stem = stem_for(recipe)
    directory = _report_dir(directory)
    _check_params(params)
    spend = _Spend(
        stem=stem,
        limit=spend_limit(approved_cpu_hours),
        approved=None if approved_cpu_hours is None else float(approved_cpu_hours),
        threads=params.kronos.threads,
        steps=steps,
    )
    if record_check_config(recipe, fold, directory=directory) != prepared.config_sha256:
        raise PilotCheckError(f"{stem}: the batches were prepared under another definition")
    entry = _verify_inputs(inputs)
    record = _read_fit_record(inputs)
    curve = _nll4_tf_curve(record)
    decisions = list(prepared.decisions_ms)
    values = prepared.nll_inputs.values
    if prepared.nll_inputs.decision_ms.tolist() != decisions:
        raise PilotCheckError("the NLL4-TF inputs do not hold the prepared decisions")
    c_ref = values[:, MAX_CONTEXT - 1, CLOSE_INDEX].astype(np.float64)
    realised = values[:, MAX_CONTEXT + HORIZON - 1, CLOSE_INDEX].astype(np.float64)

    summaries: dict[str, list[FanSummary]] = {}
    for name in BATCHES:
        batch = prepared.batches[name]
        spend.check(f"before the {name} fans")
        progress(
            f"{stem}: {name} fans ({len(batch.tasks)} decisions, stream {BATCH_STREAMS[name]}, "
            f"{batch.pending} to compute)"
        )

        def after_fan(done: int, total: int, summary: Any, name: str = name) -> None:
            spend.check(f"after {done} of the {total} {name} fans")

        found = steps.run_fan_batch(batch, workers=1, params=params, root=root, progress=after_fan)
        summaries[name] = _by_decision(name, found, decisions)
        _check_c_ref(name, summaries[name], c_ref)
    a_same, a_floor, mean_abs_delta_p_up = _agreement(summaries)

    base = Path(root) if root is not None else data_dir()
    closes = {
        name: _fan_closes(name, summaries[name], decisions, load_fan=steps.load_fan, base=base)
        for name in ("pilot", "shipped")
    }
    crps = _crps(closes["pilot"], closes["shipped"], c_ref, realised)
    d = crps["d"]
    projection = mdd_projection(d)
    mean_shipped = float(np.mean(crps["crps_shipped"]))
    projection["mdd_relative"] = projection["mdd_bp"] / mean_shipped if mean_shipped > 0 else None

    spend.check("before the teacher-forced passes")
    progress(f"{stem}: teacher-forced first-step distributions of both models")
    shipped_models = steps.load_models("small")
    pilot_models = steps.load_models("small", checkpoint=prepared.batches["pilot"].checkpoint)
    _check_models(shipped_models, pilot_models, inputs.checkpoint)
    tokenizer = shipped_models.tokenizer
    forced_shipped = steps.teacher_forced_logits(
        tokenizer, shipped_models.model, prepared.nll_inputs
    )
    spend.check("between the two teacher-forced passes")
    forced_pilot = steps.teacher_forced_logits(tokenizer, pilot_models.model, prepared.nll_inputs)
    kl = _first_step_kl(forced_pilot, forced_shipped)
    weights = relative_weight_change(
        pilot_models.model.state_dict(), shipped_models.model.state_dict()
    )
    reading = go_no_go(a_same)

    statistics = {
        "a_same": a_same,
        "a_floor": a_floor,
        "mean_abs_delta_p_up": mean_abs_delta_p_up,
        **{key: value for key, value in kl.items() if key != "decisions"},
        "crps": {
            "decisions": int(d.size),
            "dropped_invalid_fans": int((~crps["valid"]).sum()),
            "mean_crps_pilot_bp": float(np.mean(crps["crps_pilot"])),
            "mean_crps_shipped_bp": mean_shipped,
            "d_mean_bp": float(np.mean(d)),
            "d_mean_note": "descriptive only: no skill is read from it (training-segment hours "
            "inside the pilot's own inner validation)",
            "d_sd_bp": projection["d_sd_bp"],
            "rho": projection["rho"],
        },
        "projection": projection,
        "relative_weight_change": weights,
    }
    checkpoint_files = {item.name: item.sha256 for item in entry.files}
    spent = spend.record()
    body: dict[str, Any] = {
        "retraining_md_sha256": _retraining_md_sha256(),
        "params_version": params.version,
        "spend": spent,
        "recipe": {
            "name": recipe.name,
            "recipe_id": recipe_id(recipe),
            "max_epochs": recipe.max_epochs,
        },
        "fold": {
            "label": fold.label,
            "run_label": fold.run_label,
            "csv_start": utc_text(fold.csv_start_ms),
            "csv_end": utc_text(fold.csv_end_ms),
            "inner_validation_start": utc_text(fold.inner_validation_start_ms),
        },
        "checkpoint": {
            "checkpoint_id": entry.checkpoint_id,
            "registry_line_sha256": entry.line_sha256,
            "files_sha256": checkpoint_files,
            "data_end": utc_text(entry.data_end_ms),
        },
        "fit_record_sha256": inputs.fit_record_sha256,
        "export_content_sha256": inputs.export.content_sha256,
        "role": {"period_role": MODEL_PERIOD_PILOT_CHECK, "text": ROLE_TEXT},
        "decisions": {
            "planned": len(check_decisions_ms(fold)),
            "kept": len(decisions),
            "first": utc_text(decisions[0]),
            "last": utc_text(decisions[-1]),
            "dropped": list(prepared.dropped),
        },
        "fans": {
            name: {
                "config_id": prepared.batches[name].config_id,
                "stream": BATCH_STREAMS[name],
                "fans": len(summaries[name]),
                "computed": sum(1 for summary in summaries[name] if summary.computed),
                "checkpoint_id": None
                if prepared.batches[name].checkpoint is None
                else prepared.batches[name].checkpoint.checkpoint_id,
            }
            for name in BATCHES
        },
        "statistics": statistics,
        "statistics_with_intervals_reason": INTERVALS_REASON,
        "nll4_tf_curve": curve,
        "reading": {
            "value": reading,
            "a_same": a_same,
            "threshold": GO_NO_GO_THRESHOLD,
            "rule": f"{READING_NO_GO} when a_same >= {GO_NO_GO_THRESHOLD:g} (included)",
            "no_go": NO_GO_OPTIONS if reading == READING_NO_GO else None,
        },
    }
    header = report_header(
        stem,
        check_plan(recipe, fold),
        statistics_with_intervals=0,
        seeds=_seeds(),
        samples={
            "decisions_planned": body["decisions"]["planned"],
            "decisions_kept": len(decisions),
            "crps_decisions": int(d.size),
            "fans": sum(len(items) for items in summaries.values()),
        },
        comparators=_comparators(),
        directory=directory,
    )
    if header["config_sha256"] != prepared.config_sha256:
        raise PilotCheckError(f"{stem}: the recorded definition changed after the preparation")
    body["figures"] = _figures(stem, curve, directory)
    payload = {**body, **header}
    json_path, markdown_path = write_report(stem, payload, _markdown(payload), directory=directory)
    details = {
        "reading": reading,
        "a_same": a_same,
        "a_floor": a_floor,
        "mean_abs_delta_p_up": mean_abs_delta_p_up,
        "kl_pilot_shipped": kl["kl_pilot_shipped"],
        "relative_weight_change": weights["relative"],
        "d_sd_bp": projection["d_sd_bp"],
        "mdd_bp": projection["mdd_bp"],
        "decisions": len(decisions),
        "report_sha256": sha256_file(json_path),
        "config_sha256": prepared.config_sha256,
        "checkpoint_id": entry.checkpoint_id,
        "registry_line_sha256": entry.line_sha256,
        "cpu_seconds": spent["cpu_seconds"],
        "limit_cpu_seconds": spent["limit_cpu_seconds"],
    }
    return CheckResult(
        report=payload,
        json_path=json_path,
        markdown_path=markdown_path,
        details=details,
        reading=reading,
        summary_lines=tuple(summary_lines(payload)),
    )


def _seeds() -> dict[str, Any]:
    return {
        "fans": "kronos_agent.seeds of D, the stream (forecast, forecast-aa) and the project "
        "symbol BTC-USDT; pilot and shipped share the forecast stream (common random numbers)",
        "kl": "none: teacher forcing draws no random number",
        "training": "the pilot's training seeds are in its registry entry",
    }


def _comparators() -> dict[str, Any]:
    return {
        "a_floor": "the agreement of two shipped seed streams: the seed-noise floor of a_same",
        "a_same_threshold": f"{GO_NO_GO_THRESHOLD:g}, SPEC 12's live-against-replay success level",
        "owner_floor": "SPEC 9's 88 % agreement (external reference)",
        "kl": "0 for identical models",
        "projection": f"{PROJECTED_DECISIONS} G4 decisions of the walk-forward in "
        f"{PROJECTED_BLOCKS} four-week blocks",
    }


def _figures(stem: str, curve: Mapping[str, Any], directory: Path) -> list[dict[str, str]]:
    """``figures/<stem>-nll4tf.png``: the NLL4-TF curve (epoch 0 = shipped, the selected
    pseudo-epoch marked) and the training and vendored validation losses (diagnostics)."""
    from ..report.figures import figures_dir, new_figure, save_png

    epochs = curve["epochs"]
    figure, axes = new_figure(1, 2)
    points = [(e["pseudo_epoch"], e["nll4_tf"]) for e in epochs if e["nll4_tf"] is not None]
    if points:
        x, y = zip(*points, strict=True)
        axes[0].plot(x, y, marker="o", color="tab:blue", label="NLL4-TF")
    shipped = [e["nll4_tf"] for e in epochs if e["pseudo_epoch"] == 0 and e["nll4_tf"] is not None]
    if shipped:
        axes[0].axhline(
            shipped[0], color="tab:gray", linestyle="--", linewidth=0.8, label="epoch 0 (shipped)"
        )
    selected = curve["selected_pseudo_epoch"]
    chosen = [e["nll4_tf"] for e in epochs if e["pseudo_epoch"] == selected]
    if chosen and chosen[0] is not None:
        axes[0].plot(
            [selected],
            [chosen[0]],
            marker="*",
            markersize=12,
            color="tab:red",
            linestyle="none",
            label=f"selected ({selected})",
        )
    ticks = sorted({e["pseudo_epoch"] for e in epochs})
    axes[0].set_xticks(ticks)
    axes[0].set_xlabel("pseudo-epoch")
    axes[0].set_ylabel("nats per candle")
    axes[0].set_title("NLL4-TF, inner validation")
    axes[0].legend(loc="best")
    for key, label in (("train_loss", "training loss"), ("vendored_val_loss", "vendored val")):
        series = [(e["pseudo_epoch"], e[key]) for e in epochs if e[key] is not None]
        if series:
            x, y = zip(*series, strict=True)
            axes[1].plot(x, y, marker="o", label=label)
    axes[1].set_xticks(ticks)
    axes[1].set_xlabel("pseudo-epoch")
    axes[1].set_title("diagnostics (never used for selection)")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(loc="best")
    figure.suptitle(f"Pilot fold record, {stem} (shown: fold 1 is not blinded)")
    target = figures_dir(directory) / f"{stem}-nll4tf.png"
    return [{"name": "nll4tf", **save_png(figure, target)}]


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    """What the command prints after a run (no path); the reading is printed by the command."""
    statistics = report.get("statistics") or {}
    crps = statistics.get("crps") or {}
    projection = statistics.get("projection") or {}
    weights = statistics.get("relative_weight_change") or {}
    decisions = report.get("decisions") or {}
    curve = report.get("nll4_tf_curve") or {}
    epochs = curve.get("epochs") or []
    shown = ", ".join(f"{e['pseudo_epoch']}: {_number(e['nll4_tf'])}" for e in epochs)
    relative = projection.get("mdd_relative")
    share = "" if relative is None else f" ({100 * relative:.3g} % of the mean shipped CRPS)"
    return [
        (
            f"{report.get('study')}: {decisions.get('kept')} of {decisions.get('planned')} {GRID} "
            f"decisions of fold {(report.get('fold') or {}).get('run_label')}, 3 fans each "
            "(point estimates only)"
        ),
        (
            f"A_same {_number(statistics.get('a_same'))} (pilot and shipped, common seeds); "
            f"A_floor {_number(statistics.get('a_floor'))} (two shipped streams); mean |delta "
            f"p_up| {_number(statistics.get('mean_abs_delta_p_up'))}"
        ),
        (
            f"KL(pilot || shipped) of the first-step s1 distributions "
            f"{_number(statistics.get('kl_pilot_shipped'))} nats (reverse "
            f"{_number(statistics.get('kl_shipped_pilot'))}, max "
            f"{_number(statistics.get('kl_pilot_shipped_max'))})"
        ),
        (
            f"relative weight change {_number(weights.get('relative'))} "
            f"({weights.get('changed_tensors')} of {weights.get('tensors')} tensors changed, max "
            f"|delta| {_number(weights.get('max_abs_delta'))})"
        ),
        (
            f"d = CRPS_pilot - CRPS_shipped at h = 4 over {crps.get('decisions')} decisions: sd "
            f"{_number(crps.get('d_sd_bp'))} bp, mean {_number(crps.get('d_mean_bp'))} bp "
            "(descriptive: no skill is read from it)"
        ),
        (
            "projected minimum detectable difference (approximate) "
            f"{_number(projection.get('mdd_bp'))} bp for {PROJECTED_DECISIONS} decisions" + share
        ),
        (
            f"NLL4-TF by pseudo-epoch (0 = shipped) {shown}; selected "
            f"{curve.get('selected_pseudo_epoch')} ({curve.get('stop_reason')})"
        ),
        _spend_text(report.get("spend") or {}),
    ]


def _spend_text(spend: Mapping[str, Any]) -> str:
    approved = spend.get("approved_cpu_hours")
    figure = "no approved figure" if approved is None else f"{approved:g} CPU-hours approved"
    return (
        f"spend {_number(spend.get('cpu_seconds'))} CPU-seconds (wall "
        f"{_number(spend.get('wall_seconds'))} s) against a limit of "
        f"{_number(spend.get('limit_cpu_seconds'))} ({figure})"
    )


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


def _markdown(payload: Mapping[str, Any]) -> str:
    statistics = payload["statistics"]
    crps = statistics["crps"]
    projection = statistics["projection"]
    weights = statistics["relative_weight_change"]
    reading = payload["reading"]
    curve = payload["nll4_tf_curve"]
    code = payload.get("code") or {}
    checkpoint = payload["checkpoint"]
    decisions = payload["decisions"]
    share = projection.get("mdd_relative")
    relative = "n/a" if share is None else f"{100 * share:.3g} %"
    lines = [
        f"# Pilot check, {payload['study']}",
        "",
        (
            "Design: docs/RETRAINING.md 6.2 steps 3-4. A study with 0 tests on training-segment "
            f"hours, under {ROLE_TEXT}."
        ),
        "",
        f"## Reading: {reading['value']}",
        "",
        (
            f"A_same = {_number(reading['a_same'])} against the pre-registered threshold "
            f"{_number(reading['threshold'])} ({reading['rule']})."
        ),
    ]
    if reading.get("no_go"):
        lines += ["", f"NO-GO: {reading['no_go']}."]
    lines += [
        "",
        "## Provenance",
        "",
        (
            f"- Study `{payload['study']}`; config sha256 `{payload.get('config_sha256')}`; "
            f"docs/RETRAINING.md sha256 `{payload.get('retraining_md_sha256')}`"
        ),
        (
            f"- Code: commit `{code.get('commit')}`, dirty {_number(code.get('dirty'))}, source "
            f"tree sha256 `{code.get('source_tree_sha256')}`; params version "
            f"`{payload.get('params_version')}`"
        ),
        (
            f"- Recipe {payload['recipe']['name']} `{payload['recipe']['recipe_id']}` (max_epochs "
            f"{payload['recipe']['max_epochs']}); fold {payload['fold']['run_label']}, CSV "
            f"[{payload['fold']['csv_start']}, {payload['fold']['csv_end']}), inner validation "
            f"from {payload['fold']['inner_validation_start']}"
        ),
        f"- Checkpoint `{checkpoint['checkpoint_id']}`, registry line sha256 "
        f"`{checkpoint['registry_line_sha256']}`, data end {checkpoint['data_end']}; files "
        + ", ".join(f"{name} `{value}`" for name, value in checkpoint["files_sha256"].items()),
        (
            f"- Fold record sha256 `{payload['fit_record_sha256']}`; export content sha256 "
            f"`{payload['export_content_sha256']}`"
        ),
        (
            f"- Generated at {payload.get('generated_at')}; tests consumed "
            f"{payload.get('tests_consumed')}; statistics with intervals 0: "
            f"{payload['statistics_with_intervals_reason']}"
        ),
        f"- Run: {_spend_text(payload.get('spend') or {})} (design 3.5)",
        "",
        "## Decisions and fans",
        "",
        (
            f"{decisions['kept']} of {decisions['planned']} {GRID} decisions kept, "
            f"{decisions['first']} .. {decisions['last']}; {len(decisions['dropped'])} dropped."
        ),
        "",
        "| batch | model | stream | config id | fans | computed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, fan in payload["fans"].items():
        model = fan["checkpoint_id"] or "Kronos-small as shipped"
        lines.append(
            f"| {name} | {model} | {fan['stream']} | `{fan['config_id']}` | {fan['fans']} | "
            f"{fan['computed']} |"
        )
    for item in decisions["dropped"][:10]:
        lines.append(f"- dropped {item['decision']}: {item['reason']}")
    lines += [
        "",
        "## Statistics (point estimates)",
        "",
        "| statistic | value | note |",
        "| --- | --- | --- |",
        f"| A_same | {_number(statistics['a_same'])} | pilot and shipped, stream forecast |",
        f"| A_floor | {_number(statistics['a_floor'])} | two shipped streams (seed floor) |",
        f"| mean abs delta p_up | {_number(statistics['mean_abs_delta_p_up'])} | stream forecast |",
        f"| KL(pilot, shipped) | {_number(statistics['kl_pilot_shipped'])} | primary, nats |",
        f"| KL(shipped, pilot) | {_number(statistics['kl_shipped_pilot'])} | descriptive |",
        f"| KL max | {_number(statistics['kl_pilot_shipped_max'])} | descriptive |",
        (
            f"| relative weight change | {_number(weights['relative'])} | "
            f"{weights['changed_tensors']} of {weights['tensors']} tensors changed, max abs delta "
            f"{_number(weights['max_abs_delta'])} |"
        ),
        f"| mean CRPS pilot (bp) | {_number(crps['mean_crps_pilot_bp'])} | h = 4 |",
        f"| mean CRPS shipped (bp) | {_number(crps['mean_crps_shipped_bp'])} | h = 4 |",
        f"| d mean (bp) | {_number(crps['d_mean_bp'])} | descriptive only: no skill is read |",
        f"| d sd (bp) | {_number(crps['d_sd_bp'])} | ddof 1, {crps['decisions']} decisions |",
        "",
        "Autocorrelation of d (lags 1..6): "
        + ", ".join(f"{lag}: {_number(value)}" for lag, value in crps["rho"].items())
        + ".",
        "",
        (
            f"Projection (approximate): inflation {_number(projection['inflation'])}, n_eff "
            f"{_number(projection['n_eff'])}, se {_number(projection['se_bp'])} bp, minimum "
            f"detectable difference {_number(projection['mdd_bp'])} bp ({relative} of the mean "
            f"shipped CRPS) for {projection['decisions']} decisions; "
            f"{projection['assumptions']}."
        ),
        "",
        "## NLL4-TF curve (fold 1 is shown, design 6.3)",
        "",
        "| pseudo-epoch | NLL4-TF | training loss | vendored validation loss |",
        "| --- | --- | --- | --- |",
    ]
    for item in curve["epochs"]:
        lines.append(
            f"| {item['pseudo_epoch']} | {_number(item['nll4_tf'])} | "
            f"{_number(item['train_loss'])} | {_number(item['vendored_val_loss'])} |"
        )
    lines += [
        "",
        (
            f"Selected pseudo-epoch {curve['selected_pseudo_epoch']} ({curve['stop_reason']}); "
            f"epoch 0 is {curve['epoch_0']}."
        ),
        "",
        "## Figures",
        "",
    ]
    for figure in payload.get("figures") or []:
        lines.append(f"![{figure['name']}]({figure['relpath']}) (sha256 `{figure['sha256']}`)")
    return "\n".join(lines) + "\n"


__all__ = [
    "APPROVAL_TOLERANCE",
    "AUTOCORRELATION_LAGS",
    "BATCHES",
    "GATE3_CPU_SECONDS",
    "GO_NO_GO_THRESHOLD",
    "GRID",
    "HORIZON",
    "KL_SECONDS_PER_DECISION_PRIOR",
    "NO_GO_OPTIONS",
    "PROJECTED_BLOCKS",
    "PROJECTED_DECISIONS",
    "READING_GO",
    "READING_NO_GO",
    "SOURCE",
    "STEM_PREFIX",
    "STREAM_FLOOR",
    "STREAM_SAME",
    "TRAJECTORIES",
    "CheckResult",
    "PilotCheckError",
    "PilotCheckSteps",
    "PilotCheckStopped",
    "PilotInputs",
    "PreparedCheck",
    "check_decisions_ms",
    "check_plan",
    "estimate_check",
    "go_no_go",
    "mdd_projection",
    "prepare_check",
    "real_steps",
    "record_check_config",
    "relative_weight_change",
    "run_check",
    "spend_limit",
    "stem_for",
    "summary_lines",
]
