"""The per-fold training path of the walk-forward: ``kronos-agent finetune fold`` (step 6;
``docs/RETRAINING.md`` 3.5, 6.1 to 6.3 and 8.1 to 8.3). Design (c) gives this module the fold
loop, blinding, the failure policy, resumption and the projection-based spend control; this build
holds what one fold needs, and the loop over folds 2 onwards comes with the walk-forward's own
gate-3 ask.

Layout. The measured worst case of the 24 monthly folds (about 116 h) passes 72 h, so the
walk-forward runs the 8 quarterly folds over the same test decisions (D-048 (1), D-056): a
``wf-`` label is refused, and so is ``final-`` (the model to date has its own later command).
The pilot is fold 1, :data:`PILOT_LABEL`: its CSV ends at 2024-06-09T16:00Z, the last candle a
training-segment sample may read, so it reads training-segment data only (asserted), is shown
(design 6.3) and needs no pre-registration bundle. Every other fold, replicates and sealed folds
included, trains only once the effective recipe's bundle is in the experiment ledger (design
7.1), and is blinded: its command prints only ``FitOutcome.printable()``, the checkpoint id and
the first 12 hex of its weights' sha256, its ledger entries hold no loss, NLL value or selected
pseudo-epoch, and its fold record (``runner_result.json``) is never opened on success (its
sha256 comes from the registry entry, which the training parent computed). A fold that reads
sealed candles needs ``--order``: the owner's recorded order whose candidate is the T-B test of
its recipe, read from a ledger that may open the run's data root (``ledger.binding_problem``: a
relocated reports root never opens the repository's data; gate 6).

Estimate (design 8.1, on the measurement step's record ``reports/finetune/timing-2.json``, which
must be fresh: same device, torch, platform and training code): ``wall = theta x steps x t_step
+ (E_max + 1) x (t_nll + t_val) + t_setup``, with ``steps = E_max x pseudo_epoch_steps``,
``t_step`` and ``theta`` the record's choice (``theta`` 1.0 when it has none, on the CPU),
``t_nll`` M8's seconds per decision times the fold's NLL4-TF decisions, ``t_val`` the vendored
validation diagnostic priced as :data:`VALIDATION_FORWARD_SHARE` of a training step per batch of
its windows, and ``t_setup`` :data:`SETUP_SECONDS_PRIOR`; ``t_val`` and ``t_setup`` are priors the
measurement step did not measure. Planned counts are used (gaps only reduce them), so it is a
worst case. Gate 3 is counted in the chosen device's hours: MPS wall-hours, or CPU-hours (wall x
threads) on the CPU; an MPS fold's CPU side is recorded beside it, not gated.

Gate 3 and spend (design 3.5, 8.3; D-040, D-046). The first compute unit of an invocation above
:data:`GATE3_HOURS` needs ``--approved-by-owner HOURS``, the figure the owner approved in chat,
and its estimate may be at most :data:`APPROVAL_TOLERANCE` times it; both refusals come before
anything is written. The approval is recorded in the ledger (``gate3_approval``) before the
first unit, and covers only what this invocation runs: nothing starts that could take its spend
past 1.25 times the figure (:data:`GATE3_HOURS` without one). Before every child the spend so
far plus the unit's estimate must fit that cap, and the child's deadline is the smaller of
:data:`DEADLINE_FACTOR` times its estimate (design 3.5: a fold past twice its worst case is
stopped) and what the cap leaves. Spend is the wall clock around the child on MPS and the CPU
time of this process and its children on the CPU.

The pilot rule (design 3.2, 6.2; pre-registered as :data:`PILOT_RULE`, recorded in the pilot's
``planned`` entry before its result exists): if the pilot's selected pseudo-epoch is its last,
``E_max`` doubles once for every fold and fold 1 is refit from the shipped weights. The doubled
recipe keeps the name R1 but has its own id, export, seeds and checkpoint id. The refit runs in
the same invocation when the cap admits it; otherwise the command refuses it after printing the
pilot, and the next invocation starts at the refit with the figure the owner approves for it.

Stages, read from the fine-tune registry and the ledger: ``train`` (the fold is not registered),
``bookkeeping`` (registered, but its ``fit`` entry, or the pilot's ``done`` entry, is missing: a
run killed between registration and the ledger; recorded without compute, its ``fit`` entry
naming the approval and the attempt the run had, read from the entries recorded before it, never
this invocation's approval), ``refit`` (the pilot rule fired and the doubled recipe's pilot is
not registered) and ``done`` (the effective recipe's pilot is registered and recorded). A
registered fold other than the pilot is never trained again (design 3.5).

Failures (design 6.3). ``finetune/train.py`` is part of the training-code hash the timing record
is tied to, so it is used as it is: when ``train.fit`` raises ``ChildFailed`` or
``ChildMemoryAbort``, the run's status is read from the one run directory the call created (its
``runner_result.json`` status, ``no_result`` without the file, ``killed`` when there is no such
directory, several, or an unreadable result). A ``fit`` entry records it. ``failed`` and
``no_result`` are retried once with the same seeds when the cap admits the retry; a second
failure, a memory abort (not a fold failure: the owner is told), a time abort or a killed child
is :class:`FoldFailed` (exit 1), after its entries. Any other exception once the child started
(SIGTERM through ``SystemExit``, an interrupt) records the run's status, then propagates.

Printed lines carry run ids and checkpoint ids, never a path or the child's log. Nothing here
imports torch at module level (the real ``mps_available`` imports it lazily).
"""

from __future__ import annotations

import json
import math
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .. import ledger
from ..data.fsutil import exclusive_lock, lock_is_held
from ..data.partition import MAX_CONTEXT, VALIDATION_START, PartitionError
from ..data.timeutil import HOUR_MS, to_ms, utc_text
from ..hashing import canonical_json, sha256_bytes
from ..params import Params
from ..paths import default_label, reports_dir
from ..runtime import RuntimeConfigError, require_configured
from . import registry, runner, timing
from .folds import FOLD_LABEL_RE, Fold, fold_for
from .recipe import R1, Recipe, recipe8, recipe_id, training_seeds
from .registry import CheckpointRef, RegistryEntry, checkpoint_id_for
from .timing import TimingError
from .train import (
    ChildFailed,
    ChildMemoryAbort,
    Compute,
    FitOutcome,
    RunRequest,
    Wrappers,
    training_code_sha256,
)

if TYPE_CHECKING:
    from ..data.partition import SealedAccess
    from .export_csv import ExportRecord

WALK_FORWARD_LAYOUT = "quarterly"  # D-056: the D-048 (1) rule (monthly worst case ~116 h > 72 h)
PILOT_LABEL = "q-2024-07"  # first quarterly fold: CSV ends 2024-06-09T16:00Z
PILOT_RULE_FACTOR = 2  # E_max doubled once, 6 -> 12 (design 3.2, 6.2)
GATE3_HOURS = 1.0
APPROVAL_TOLERANCE = 1.25  # D-040, D-046
DEADLINE_FACTOR = 2.0  # design 3.5: past twice its worst case a fold is stopped
CHILD_TIMEOUT_MARGIN_SECONDS = 120.0
SETUP_SECONDS_PRIOR = 300.0  # start-up, loads, export check, registration (not measured)
VALIDATION_FORWARD_SHARE = 1 / 3  # a forward-only batch ~ a third of a training step (prior)
RETRY_STATUSES: tuple[str, ...] = ("failed", "no_result")  # design 6.3: NaN or inf loss, crash
FOLD_LOCK_NAME = "fold.lock"  # <fine-tune root>/.scratch/fold.lock
COMMAND = "finetune fold"
PILOT_RULE: Mapping[str, str] = MappingProxyType(
    {
        "condition": "selected_pseudo_epoch == max_epochs",
        "action": "max_epochs x 2 for every fold; fold 1 refit from the shipped weights",
        "applies": "once",
    }
)
PILOT_ACTIVITY = "pilot-fold"  # the ledger's exploration activity of the pilot
RULE_FIRED = "fired"
RULE_NOT_FIRED = "not fired"
RULE_ALREADY_DOUBLED = "already doubled"
STAGES: tuple[str, ...] = ("train", "bookkeeping", "refit", "done")
UNITS: Mapping[str, str] = MappingProxyType({"mps": "mps_hours", "cpu": "cpu_hours"})
UNIT_TEXT: Mapping[str, str] = MappingProxyType(
    {"mps_hours": "MPS-hours", "cpu_hours": "CPU-hours"}
)
MAX_ATTEMPTS = 2  # one retry with the same seeds (design 6.3)
IDLE_REMINDER = (
    "run it when the machine is otherwise idle, on power, lid open, under caffeinate -i -s"
)
# every status a fit entry may record (the ledger's fit statuses)
FAILED_RUN_STATUSES: tuple[str, ...] = ("failed", "memory_abort", "time_abort", "killed")
_SECONDS_PER_HOUR = 3600.0
_REFUSED_LAYOUTS = {
    "wf": "the walk-forward runs the quarterly layout (D-048 (1), D-056): fold labels are "
    "q-YYYY-MM, the pilot is q-2024-07",
    "final": "the model to date has its own command, built after the walk-forward",
}
_MAX_STATE_STEPS = 6  # a pilot invocation passes bookkeeping, train, refit and done at most


class FoldRefused(PermissionError):
    """A governance rule refuses the fold (exit 2): the layout, gate 3, the spend cap, the
    pre-registration bundle, a registered fold, the owner's order, a timing run in progress, a
    state that moved since the estimate was shown."""


class FoldFailed(RuntimeError):
    """A fold whose training failed for good, after its ``fit`` entries were recorded (a second
    failure, a memory abort, a time abort, a killed child), or whose recorded result no longer
    verifies (exit 1)."""


# -- the steps ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSteps:
    """The heavy work, injectable (:func:`real_steps` gives the real ones; tests pass stand-ins).
    Each is called as its real counterpart: ``export_fold(fold, recipe, root=, sealed_access=)``,
    ``fit(request, root=, timeout_seconds=, register=True)``, ``load_timing(directory=)``,
    ``mps_available()``, ``clock()`` (seconds) and ``children_cpu_seconds()``."""

    export_fold: Callable[..., ExportRecord]  # finetune.export_csv.export_fold
    fit: Callable[..., FitOutcome]  # finetune.train.fit
    load_timing: Callable[..., dict[str, Any]]  # finetune.timing.load_timing
    mps_available: Callable[[], bool]  # torch.backends.mps.is_available
    clock: Callable[[], float]  # time.monotonic
    children_cpu_seconds: Callable[[], float]  # getrusage(RUSAGE_CHILDREN) user + system


def _mps_available() -> bool:
    import torch

    return bool(torch.backends.mps.is_available())


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def real_steps() -> FoldSteps:
    """The real steps (imported here, after the runtime is configured)."""
    from . import export_csv, train

    return FoldSteps(
        export_fold=export_csv.export_fold,
        fit=train.fit,
        load_timing=timing.load_timing,
        mps_available=_mps_available,
        clock=time.monotonic,
        children_cpu_seconds=_children_cpu_seconds,
    )


# -- the estimate -------------------------------------------------------------------------------


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value: object, name: str) -> float:
    if not _finite(value) or not value > 0:
        raise TimingError(
            f"the timing record has no finite {name} > 0 (got {value!r}): measure again"
        )
    return float(value)


def _section(record: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = record
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    if not isinstance(value, Mapping):
        raise TimingError(f"the timing record has no {'.'.join(keys)}: measure again")
    return value


def _choice(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"a timing record is a mapping, got {type(record).__name__}")
    choice = _section(record, "choice")
    device = choice.get("device")
    if device is None:
        raise FoldRefused(
            "the timing record decided no device ("
            + "; ".join(str(reason) for reason in choice.get("reasons") or ())
            + "): the owner decides with it before any fold trains (design 3.4)"
        )
    if device not in UNITS:
        raise TimingError(f"the timing record chose an unknown device {device!r}")
    return choice


@dataclass(frozen=True)
class FitEstimate:
    """The estimate of one fit (design 8.1 with the timing record's measurements); seconds."""

    recipe_id: str
    max_epochs: int
    steps: int  # max_epochs x pseudo_epoch_steps
    device: str
    threads: int
    unit: str  # "mps_hours" | "cpu_hours"
    seconds_per_step: float  # the record's choice
    theta: float  # the record's choice; 1.0 when it has none (CPU)
    nll_decisions: int  # NLL4-TF decisions (planned: 336)
    val_windows: int  # vendored validation windows (planned: 832)
    t_nll: float  # M8 seconds_per_decision x nll_decisions
    t_val: float  # ceil(val_windows / batch) x seconds_per_step x VALIDATION_FORWARD_SHARE
    t_setup: float  # SETUP_SECONDS_PRIOR
    wall_seconds: float  # theta x steps x t_step + (max_epochs + 1) x (t_nll + t_val) + t_setup
    device_hours: float  # MPS: wall / 3600; CPU: wall x threads / 3600
    cpu_side_seconds: float  # MPS: the CPU side, informational; CPU: wall x threads
    timing_record_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_estimate(
    recipe: Recipe,
    fold: Fold,
    record: Mapping[str, Any],
    *,
    nll_decisions: int | None = None,
    val_windows: int | None = None,
    record_sha256: str | None = None,
) -> FitEstimate:
    """The estimate of fitting ``recipe`` on ``fold`` from the timing ``record`` (module
    docstring). ``nll_decisions`` defaults to the fold's planned NLL4-TF decisions and
    ``val_windows`` to ``inner_validation_hours - window_rows + 1``: a worst case. On MPS the CPU
    side is ``(E_max + 1) x t_nll x threads + (M4 cpu_seconds / M4 seconds) x theta x steps x
    t_step + t_setup``. ``record_sha256`` is the sha256 of the record's file
    (:func:`plan_fold` passes it); without it, the sha256 of the record's canonical JSON.
    :class:`FoldRefused` when the record decided no device or the recipe has no step budget;
    ``TimingError`` for a record without the numbers."""
    if not isinstance(recipe, Recipe):
        raise TypeError(f"expected a Recipe, got {type(recipe).__name__}")
    if not isinstance(fold, Fold):
        raise TypeError(f"expected a Fold, got {type(fold).__name__}")
    choice = _choice(record)
    device = str(choice["device"])
    threads = choice.get("threads")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise TimingError(f"the timing record has no thread count for its choice: {threads!r}")
    t_step = _positive(choice.get("seconds_per_step"), "choice.seconds_per_step")
    theta = choice.get("theta")
    if theta is None:
        theta = 1.0
    elif not _finite(theta) or theta < 1.0:
        raise TimingError(f"the timing record's theta must be a finite number >= 1, got {theta!r}")
    steps = recipe.onecycle_total_steps
    if steps is None:
        raise FoldRefused(
            f"recipe {recipe.name} has no pseudo-epoch budget: its fit has no step count to "
            "estimate (design 3.2)"
        )
    measurements = _section(record, "measurements")
    per_decision = _positive(
        _section(measurements, "M8").get("seconds_per_decision"), "M8 seconds_per_decision"
    )
    decisions = len(fold.nll_decisions_ms()) if nll_decisions is None else nll_decisions
    windows = (
        recipe.inner_validation_hours - recipe.window_rows + 1
        if val_windows is None
        else val_windows
    )
    for name, value in (("nll_decisions", decisions), ("val_windows", windows)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
    t_nll = per_decision * decisions
    t_val = math.ceil(windows / recipe.batch_size) * t_step * VALIDATION_FORWARD_SHARE
    t_setup = SETUP_SECONDS_PRIOR
    training = float(theta) * steps * t_step
    evaluations = recipe.max_epochs + 1
    wall = training + evaluations * (t_nll + t_val) + t_setup
    if device == "mps":
        device_hours = wall / _SECONDS_PER_HOUR
        m4 = _section(measurements, "M4")
        rate = _positive(m4.get("cpu_seconds"), "M4 cpu_seconds") / _positive(
            m4.get("seconds"), "M4 seconds"
        )
        cpu_side = evaluations * t_nll * threads + rate * training + t_setup
    else:
        device_hours = wall * threads / _SECONDS_PER_HOUR
        cpu_side = wall * threads
    if record_sha256 is None:
        record_sha256 = sha256_bytes(canonical_json(record))
    return FitEstimate(
        recipe_id=recipe_id(recipe),
        max_epochs=recipe.max_epochs,
        steps=steps,
        device=device,
        threads=threads,
        unit=UNITS[device],
        seconds_per_step=t_step,
        theta=float(theta),
        nll_decisions=decisions,
        val_windows=windows,
        t_nll=t_nll,
        t_val=t_val,
        t_setup=t_setup,
        wall_seconds=wall,
        device_hours=device_hours,
        cpu_side_seconds=cpu_side,
        timing_record_sha256=record_sha256,
    )


def compute_choice(record: Mapping[str, Any]) -> Compute:
    """The fold's ``train.Compute`` from the timing record's choice: device, threads,
    deterministic mode, activation checkpointing, the MPS watermark ratio (MPS only) and the
    memory guard on. :class:`FoldRefused` when the record decided no device, or found no way to
    run the recipe batch (neither as is nor with activation checkpointing: batch 16 is a recipe
    change, the owner's)."""
    choice = _choice(record)
    checkpointing = choice.get("activation_checkpointing")
    fits = choice.get("batch_32_fits")
    if not isinstance(checkpointing, bool) or not isinstance(fits, bool):
        raise TimingError(
            "the timing record's choice lacks batch_32_fits or activation_checkpointing"
        )
    if not fits and not checkpointing:
        raise FoldRefused(
            "the timing record found no way to run batch 32 (it does not fit as is and "
            "activation checkpointing did not pass): batch 16 is a recipe change the owner "
            "decides (design (a))"
        )
    deterministic = choice.get("deterministic")
    if not isinstance(deterministic, bool):
        raise TimingError("the timing record's choice lacks deterministic")
    device = str(choice["device"])
    return Compute(
        device=device,
        threads=choice.get("threads"),  # Compute refuses a non-positive or non-integer count
        deterministic=deterministic,
        activation_checkpointing=checkpointing,
        mps_watermark_ratio=choice.get("mps_watermark_ratio") if device == "mps" else None,
        memory_guard=True,
    )


# -- the pilot and the effective recipe ---------------------------------------------------------


def is_pilot(fold: Fold) -> bool:
    """``fold`` is the pilot: :data:`PILOT_LABEL`, replicate 0."""
    return fold.label == PILOT_LABEL and fold.replicate == 0


def doubled_recipe() -> Recipe:
    """R1 with ``max_epochs`` doubled once (the pilot rule): another id, the same name."""
    return replace(R1, max_epochs=R1.max_epochs * PILOT_RULE_FACTOR)


def pilot_rule(recipe: Recipe, selected_pseudo_epoch: int) -> str:
    """The pilot rule's outcome: ``"already doubled"`` for a recipe whose ``max_epochs`` is not
    R1's (it never doubles again), ``"fired"`` when the selected pseudo-epoch is the last one,
    ``"not fired"`` otherwise."""
    if recipe.max_epochs != R1.max_epochs:
        return RULE_ALREADY_DOUBLED
    if selected_pseudo_epoch == recipe.max_epochs:
        return RULE_FIRED
    return RULE_NOT_FIRED


def _payload(entry: ledger.LedgerEntry) -> Mapping[str, Any]:
    return entry.payload


def _is_pilot_done(entry: ledger.LedgerEntry, identifier: str, subject: str) -> bool:
    payload = _payload(entry)
    return (
        entry.kind == "exploration"
        and payload.get("activity") == PILOT_ACTIVITY
        and payload.get("stage") == "done"
        and payload.get("recipe_id") == identifier
        and payload.get("subject") == subject
    )


def _details(entry: ledger.LedgerEntry) -> Mapping[str, Any]:
    details = _payload(entry).get("details")
    return details if isinstance(details, Mapping) else {}


def effective_recipe(
    *, entries: Sequence[ledger.LedgerEntry] | None = None, ledger_root: Path | None = None
) -> Recipe:
    """R1, or :func:`doubled_recipe` once the ledger holds the pilot's ``done`` entry of R1 with
    ``details.pilot_rule == "fired"`` (``entries`` are read from the ledger when not given)."""
    if entries is None:
        entries = ledger.read_ledger(ledger_root)
    base = recipe_id(R1)
    for entry in entries:
        if _is_pilot_done(entry, base, PILOT_LABEL) and _details(entry).get("pilot_rule") == (
            RULE_FIRED
        ):
            return doubled_recipe()
    return R1


# -- the state of a fold ------------------------------------------------------------------------


def _registered_fit(
    entries: Sequence[ledger.LedgerEntry], checkpoint_id: str
) -> ledger.LedgerEntry | None:
    found = [
        entry
        for entry in entries
        if entry.kind == "fit"
        and _payload(entry).get("checkpoint_id") == checkpoint_id
        and _payload(entry).get("status") == "registered"
    ]
    return found[-1] if found else None


def _pilot_done(
    entries: Sequence[ledger.LedgerEntry], recipe: Recipe, fold: Fold
) -> ledger.LedgerEntry | None:
    identifier = recipe_id(recipe)
    found = [entry for entry in entries if _is_pilot_done(entry, identifier, fold.run_label)]
    return found[-1] if found else None


@dataclass(frozen=True)
class _Records:
    """What the registry and the ledger hold for one recipe's checkpoint of a fold."""

    recipe: Recipe
    checkpoint_id: str
    entry: RegistryEntry | None  # verified against the fit entry's line when there is one
    fit: ledger.LedgerEntry | None  # its registered fit entry
    done: ledger.LedgerEntry | None  # the pilot's done exploration entry

    @property
    def recorded(self) -> bool:
        return self.fit is not None and self.done is not None


def _records(
    recipe: Recipe,
    fold: Fold,
    *,
    entries: Sequence[ledger.LedgerEntry],
    registered: Sequence[RegistryEntry],
    ft_root: Path | None,
    pilot: bool,
) -> _Records:
    checkpoint_id = checkpoint_id_for(recipe8(recipe), fold.run_label)
    fit = _registered_fit(entries, checkpoint_id)
    if fit is not None:
        # the line recorded in the ledger must still be the registry's (registry 3.6)
        line = _payload(fit).get("registry_line_sha256")
        entry: RegistryEntry | None = registry.find_entry(
            CheckpointRef(checkpoint_id, root=ft_root, line_sha256=line)
        )
    else:
        entry = next((item for item in registered if item.checkpoint_id == checkpoint_id), None)
    done = _pilot_done(entries, recipe, fold) if pilot else None
    if entry is None and done is not None:
        raise registry.CheckpointNotFound(
            f"the ledger records the pilot of {checkpoint_id} as done (entry #{done.seq}), but "
            "the fine-tune registry does not hold it: the registry is never repaired"
        )
    return _Records(recipe, checkpoint_id, entry, fit, done)


@dataclass(frozen=True)
class _State:
    """Where a fold stands: the stage, the records it concerns, and the next compute unit."""

    stage: str
    records: _Records
    rule: str | None  # the pilot rule: recorded (done), or read from the fold record
    compute: Recipe | None  # the recipe of the next compute unit, if any


def _fold_record_path(entry: RegistryEntry, ft_root: Path | None) -> Path:
    run_id = entry.document.get("run_id")
    if not isinstance(run_id, str):
        raise FoldFailed(f"the registry entry of {entry.checkpoint_id} names no run")
    return registry.runs_dir(ft_root) / run_id / runner.RESULT_JSON


def _read_fold_record(entry: RegistryEntry, ft_root: Path | None) -> tuple[dict[str, Any], str]:
    """The pilot's fold record as written by its run, checked against the sha256 its registry
    entry recorded (:class:`FoldFailed` otherwise); the parsed document and the sha256. Only
    the pilot's record is ever read here (blinding, design 6.3)."""
    path = _fold_record_path(entry, ft_root)
    shown = f"{path.parent.name}/{path.name}"
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise FoldFailed(f"the fold record {shown} cannot be read ({exc.strerror})") from None
    digest = sha256_bytes(data)
    if digest != entry.document.get("fit_record_sha256"):
        raise FoldFailed(
            f"the fold record {shown} no longer hashes to the fit_record_sha256 of its registry "
            f"entry ({digest[:12]}..., recorded {str(entry.document.get('fit_record_sha256'))[:12]}"
            "...): it changed after registration"
        )
    try:
        document = json.loads(data)
    except ValueError:
        raise FoldFailed(f"the fold record {shown} is not JSON") from None
    if not isinstance(document, dict):
        raise FoldFailed(f"the fold record {shown} is not a JSON object")
    return document, digest


def _selected_pseudo_epoch(document: Mapping[str, Any], recipe: Recipe) -> int:
    selected = document.get("selected_pseudo_epoch")
    if (
        not isinstance(selected, int)
        or isinstance(selected, bool)
        or not 1 <= selected <= recipe.max_epochs
    ):
        raise FoldFailed(
            f"the pilot's fold record has no selected pseudo-epoch in 1..{recipe.max_epochs} "
            f"(got {selected!r})"
        )
    return selected


def _pilot_state(
    fold: Fold,
    *,
    entries: Sequence[ledger.LedgerEntry],
    registered: Sequence[RegistryEntry],
    ft_root: Path | None,
) -> _State:
    base = _records(R1, fold, entries=entries, registered=registered, ft_root=ft_root, pilot=True)
    doubled = _records(
        doubled_recipe(), fold, entries=entries, registered=registered, ft_root=ft_root, pilot=True
    )
    if base.entry is None:
        return _State("train", base, None, R1)
    if not base.recorded:
        document, _ = _read_fold_record(base.entry, ft_root)
        rule = pilot_rule(R1, _selected_pseudo_epoch(document, R1))
        refit = rule == RULE_FIRED and doubled.entry is None
        return _State("bookkeeping", base, rule, doubled.recipe if refit else None)
    assert base.done is not None
    rule = _details(base.done).get("pilot_rule")
    if rule != RULE_FIRED:
        return _State("done", base, str(rule), None)
    if doubled.entry is None:
        return _State("refit", doubled, None, doubled.recipe)
    if not doubled.recorded:
        return _State("bookkeeping", doubled, RULE_ALREADY_DOUBLED, None)
    assert doubled.done is not None
    return _State("done", doubled, str(_details(doubled.done).get("pilot_rule")), None)


def _fold_state(
    fold: Fold,
    recipe: Recipe,
    *,
    entries: Sequence[ledger.LedgerEntry],
    ft_root: Path | None,
) -> _State:
    """The stage of ``fold`` (the module docstring); ``recipe`` is the effective one."""
    registered = registry.read_registry(ft_root)
    if is_pilot(fold):
        return _pilot_state(fold, entries=entries, registered=registered, ft_root=ft_root)
    records = _records(
        recipe, fold, entries=entries, registered=registered, ft_root=ft_root, pilot=False
    )
    if records.entry is None:
        return _State("train", records, None, recipe)
    if records.fit is None:
        return _State("bookkeeping", records, None, None)
    raise FoldRefused(
        f"fold {fold.run_label} is registered as {records.checkpoint_id} (fit entry "
        f"#{records.fit.seq}): a registered fold is never trained again (design 3.5)"
    )


# -- the plan -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldPlan:
    """What ``finetune fold`` would do now, and its estimate (nothing written)."""

    fold: Fold
    recipe: Recipe  # the effective recipe when planned (R1, or R1 doubled by the pilot rule)
    is_pilot: bool
    blinded: bool
    order_id: str | None
    compute: Compute
    record_sha256: str  # the timing record's file
    stage: str  # "train" | "bookkeeping" | "refit" | "done"
    first: FitEstimate | None  # the unit this invocation starts with
    refit: FitEstimate | None  # pilot only: the E_max-12 unit, if the rule fires
    state_lines: tuple[str, ...] = ()  # what the stage concerns (printed with the estimate)
    history: tuple[str, ...] = ()  # the earlier fit entries of the run label and their spend


def _timing_directory(reports_root: Path | None) -> Path:
    base = Path(reports_root) if reports_root is not None else reports_dir()
    return base / timing.FINETUNE_REPORTS_SUBDIR


def _record_file_sha256(record: Mapping[str, Any], directory: Path) -> str:
    """The sha256 of the timing record's file, which must parse to ``record`` (as loaded)."""
    path = directory / f"{timing.TIMING_STEM}.json"
    shown = f"{timing.FINETUNE_REPORTS_SUBDIR}/{path.name}"
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TimingError(f"{shown} cannot be read ({exc.strerror})") from None
    try:
        parsed = json.loads(data)
    except ValueError:
        raise TimingError(f"{shown} is not JSON") from None
    if parsed != record:
        raise TimingError(f"{shown} changed while it was read: run the command again")
    return sha256_bytes(data)


def _check_label(label: object) -> None:
    """Refuse a ``wf-`` or ``final-`` label (the layout, the model to date); ``fold_for`` judges
    every other label."""
    if not isinstance(label, str):
        raise TypeError(f"a fold label is a string, got {type(label).__name__}")
    match = FOLD_LABEL_RE.fullmatch(label)
    if match is None:
        return
    kind = "final" if match["fyear"] is not None else match["kind"]
    if kind in _REFUSED_LAYOUTS:
        raise FoldRefused(f"{label}: {_REFUSED_LAYOUTS[kind]}")


def _check_order(
    order_id: str | None,
    fold: Fold,
    recipe: Recipe,
    *,
    ledger_root: Path | None,
    data_root: Path | None,
) -> None:
    """``--order`` only with a sealed fold, and required by one that reads sealed candles; the
    ledger the order is read from must be one that may open the data root (``LedgerRefused``,
    ``ledger.binding_problem``), and the order's candidate the T-B test of the fold's recipe
    (gate 6)."""
    if order_id is not None and fold.layout != "sealed":
        raise FoldRefused(
            f"--order belongs to a sealed fold (s-YYYY-MM); {fold.label} reads no sealed candle"
        )
    if order_id is None:
        if fold.needs_sealed_access:
            raise FoldRefused(
                f"fold {fold.label} reads sealed candles: it needs the owner's recorded order "
                "(--order ID, `kronos-agent ledger record-order`; SPEC section 17, gate 6)"
            )
        return
    ledger.require_binding(root=ledger_root, data_root=data_root)
    order = ledger.find_order(order_id, root=ledger_root)
    expected = ledger.id_for_test("T-B", recipe_id(recipe))
    candidate = _payload(order).get("candidate")
    if candidate != expected:
        raise FoldRefused(
            f"order {order_id} opens the sealed segment for candidate {candidate}, not for "
            f"{expected}, the T-B test of recipe {recipe.name} ({recipe8(recipe)}) (gate 6)"
        )


def _history(fold: Fold, entries: Sequence[ledger.LedgerEntry]) -> tuple[str, ...]:
    fits = [
        entry
        for entry in entries
        if entry.kind == "fit"
        and _payload(entry).get("fold") == fold.label
        and _payload(entry).get("replicate") == fold.replicate
    ]
    if not fits:
        return ()
    lines = [f"earlier fit entries of {fold.run_label} in the ledger: {len(fits)}"]
    wall = cpu = 0.0
    for entry in fits:
        payload = _payload(entry)
        wall += float(payload.get("wall_seconds") or 0.0)
        cpu += float(payload.get("cpu_seconds") or 0.0)
        checkpoint = payload.get("checkpoint_id") or "no checkpoint"
        lines.append(
            f"  #{entry.seq} {entry.ts}: recipe {str(payload.get('recipe_id'))[:8]} attempt "
            f"{payload.get('attempt')} {payload.get('status')} ({checkpoint}) on "
            f"{payload.get('device')}: {float(payload.get('wall_seconds') or 0.0) / 3600:.3f} "
            f"wall-hours, {float(payload.get('cpu_seconds') or 0.0) / 3600:.3f} CPU-hours"
        )
    lines.append(f"  together: {wall / 3600:.3f} wall-hours, {cpu / 3600:.3f} CPU-hours")
    return tuple(lines)


def _state_lines(state: _State, fold: Fold) -> tuple[str, ...]:
    records = state.records
    name = f"{records.recipe.name} ({recipe8(records.recipe)}, E_max {records.recipe.max_epochs})"
    if state.stage == "train":
        return (f"stage train: {fold.run_label} of recipe {name} is not registered",)
    if state.stage == "refit":
        return (
            (
                f"stage refit: the pilot rule fired; the pilot of {name} is not registered: "
                "fold 1 is refit from the shipped weights"
            ),
        )
    if state.stage == "bookkeeping":
        missing = [
            what
            for what, entry in (("fit", records.fit), ("pilot done", records.done))
            if entry is None and (what == "fit" or is_pilot(fold))
        ]
        line = (
            f"stage bookkeeping: {records.checkpoint_id} is registered but its ledger "
            f"{' and '.join(missing)} entr{'ies are' if len(missing) > 1 else 'y is'} missing: "
            "recorded from the registry"
            + (" and the fold record" if is_pilot(fold) else "")
            + " without any compute"
        )
        lines = [line]
        if state.rule is not None and is_pilot(fold):
            lines.append(f"pilot rule (from the fold record): {state.rule}")
        if state.compute is not None:
            lines.append("then the refit of fold 1 at the doubled E_max")
        return tuple(lines)
    assert records.fit is not None and records.entry is not None
    return (
        (
            f"stage done: the pilot {records.checkpoint_id} of recipe {name} is registered "
            f"(registry line {records.entry.line_sha256[:12]}...) and recorded (fit entry "
            f"#{records.fit.seq}); pilot rule: {state.rule}; nothing to run (next: finetune "
            "pilot-check)"
        ),
    )


def plan_fold(
    label: str,
    *,
    params: Params,
    replicate: int = 0,
    order_id: str | None = None,
    steps: FoldSteps,
    ft_root: Path | None = None,
    data_root: Path | None = None,
    ledger_root: Path | None = None,
    reports_root: Path | None = None,
) -> FoldPlan:
    """Plan ``finetune fold --fold label`` (nothing written).

    Refused (:class:`FoldRefused`) before anything else: a ``wf-`` or ``final-`` label; then
    ``fold_for`` (its ``FoldError`` for a label it rejects, a quarterly replicate among them);
    ``--order`` with a fold that is not sealed, or missing for one that reads sealed candles, or
    read from a ledger that may not open ``data_root`` (``ledger.LedgerRefused``), or naming
    another candidate; a fold other than the pilot without the effective recipe's
    pre-registration bundle; a timing record that decided no device or no way to run batch 32,
    measured another batch, is stale (``timing.StaleTimingError``), or chose MPS where MPS is not
    available; a registered fold other than the pilot. ``data_root`` is where the run will
    export the fold (nothing is read from it here).
    """
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    if not isinstance(steps, FoldSteps):
        raise TypeError(f"expected FoldSteps, got {type(steps).__name__}")
    info = require_configured()
    if info.threads != params.kronos.threads:
        raise RuntimeConfigError(
            f"the runtime runs {info.threads} threads, params.yaml {params.kronos.threads}: the "
            "fold command runs the governed thread count"
        )
    _check_label(label)
    entries = ledger.read_ledger(ledger_root)
    recipe = effective_recipe(entries=entries)
    fold = fold_for(label, recipe, replicate=replicate)
    pilot = is_pilot(fold)
    if pilot and fold.csv_end_ms > to_ms(VALIDATION_START) - MAX_CONTEXT * HOUR_MS:
        raise PartitionError(
            f"the pilot's CSV ends at {utc_text(fold.csv_end_ms)}, after the training segment's "
            "candle boundary 2024-06-09T16:00Z"
        )
    _check_order(order_id, fold, recipe, ledger_root=ledger_root, data_root=data_root)
    if not pilot and ledger.preregistration_for(recipe_id(recipe), entries=entries) is None:
        raise FoldRefused(
            f"fold {fold.run_label} trains after the pre-registration bundle (design 7.1): the "
            f"ledger has no preregistration entry for recipe {recipe.name} ({recipe8(recipe)})"
        )
    directory = _timing_directory(reports_root)
    record = steps.load_timing(directory=directory)
    compute = compute_choice(record)
    batch = _section(record, "choice").get("batch_size")
    if batch != recipe.batch_size:
        raise FoldRefused(
            f"the timing record measured batch {batch!r}; recipe {recipe.name} trains batch "
            f"{recipe.batch_size}: measure again (`kronos-agent finetune timing`)"
        )
    timing.require_fresh_timing(
        record,
        device=compute.device,
        torch_version=info.torch_version,
        platform=default_label(),
        training_code_sha256=training_code_sha256(),
    )
    if compute.device == "mps" and not steps.mps_available():
        raise FoldRefused(
            "the timing record chose MPS, which is not available in this process: run the fold "
            "on the machine that was measured"
        )
    record_sha256 = _record_file_sha256(record, directory)
    state = _fold_state(fold, recipe, entries=entries, ft_root=ft_root)

    def estimate(unit_recipe: Recipe) -> FitEstimate:
        return fit_estimate(unit_recipe, fold, record, record_sha256=record_sha256)

    first = estimate(state.compute) if state.compute is not None else None
    refit = None
    if pilot and state.stage == "train":
        refit = estimate(doubled_recipe())
    return FoldPlan(
        fold=fold,
        recipe=recipe,
        is_pilot=pilot,
        blinded=not pilot,
        order_id=order_id,
        compute=compute,
        record_sha256=record_sha256,
        stage=state.stage,
        first=first,
        refit=refit,
        state_lines=_state_lines(state, fold),
        history=_history(fold, entries),
    )


def _hours(value: float, unit: str) -> str:
    return f"{value:.3f} {UNIT_TEXT[unit]}"


def _estimate_line(prefix: str, estimate: FitEstimate) -> str:
    evaluations = estimate.max_epochs + 1
    return (
        f"{prefix} (recipe {estimate.recipe_id[:8]}, E_max {estimate.max_epochs}, "
        f"{estimate.steps} steps): theta {estimate.theta:.3f} x {estimate.steps} x "
        f"{estimate.seconds_per_step:.3f} s + {evaluations} x (NLL4-TF {estimate.t_nll:.1f} s "
        f"for {estimate.nll_decisions} decisions + validation {estimate.t_val:.1f} s for "
        f"{estimate.val_windows} windows) + setup {estimate.t_setup:.0f} s = "
        f"{estimate.wall_seconds:.1f} s = {_hours(estimate.device_hours, estimate.unit)}"
        + (
            f" (CPU side about {estimate.cpu_side_seconds / 3600:.2f} CPU-hours, recorded, not "
            "gated)"
            if estimate.device == "mps"
            else ""
        )
    )


def estimate_lines(plan: FoldPlan) -> list[str]:
    """What ``finetune fold`` prints before it runs anything: the fold, the compute, the stage,
    every term of the estimate, the refit if the pilot rule fires, gate 3 and the earlier fit
    entries of the run label."""
    fold, compute = plan.fold, plan.compute
    role = (
        "the pilot (fold 1; training-segment data only, shown: design 6.3)"
        if plan.is_pilot
        else "blinded (design 6.3)"
    )
    lines = [
        f"finetune fold {fold.run_label}: {fold.layout} layout (D-056), {role}; recipe "
        f"{plan.recipe.name} ({recipe8(plan.recipe)}, E_max {plan.recipe.max_epochs})"
        + (f"; order {plan.order_id}" if plan.order_id else ""),
        f"compute: {compute.device.upper()}, {compute.threads} threads, batch "
        f"{plan.recipe.batch_size}, activation checkpointing "
        f"{'on' if compute.activation_checkpointing else 'off'}, deterministic "
        f"{'on' if compute.deterministic else 'off'}"
        + (
            f", MPS watermark ratio {compute.mps_watermark_ratio or 1.0:g}"
            if compute.device == "mps"
            else ""
        )
        + f"; timing record {timing.TIMING_STEM} (sha256 {plan.record_sha256[:12]}...)",
        *plan.state_lines,
    ]
    if plan.first is not None:
        lines.append(_estimate_line("estimate", plan.first))
        lines.append(
            f"the validation term ({VALIDATION_FORWARD_SHARE:.3f} of a step per batch) and "
            f"the setup ({SETUP_SECONDS_PRIOR:.0f} s) are priors the measurement step did not "
            "measure; planned counts, so a worst case"
        )
    if plan.refit is not None:
        lines.append(
            _estimate_line(
                "if the pilot rule fires (selected pseudo-epoch = E_max), the refit", plan.refit
            )
        )
        assert plan.first is not None
        lines.append(
            "pilot and refit together: "
            + _hours(plan.first.device_hours + plan.refit.device_hours, plan.first.unit)
        )
    if plan.first is not None:
        hours, unit = plan.first.device_hours, plan.first.unit
        if hours > GATE3_HOURS:
            lines.append(
                f"gate 3: {_hours(hours, unit)} is above {GATE3_HOURS:g} hour: run it with "
                "--approved-by-owner HOURS, the figure the owner approves in chat; the estimate "
                f"may be at most {APPROVAL_TOLERANCE:g} times it, and nothing starts that could "
                f"take this run's spend past {APPROVAL_TOLERANCE:g} times it"
            )
        else:
            lines.append(
                f"gate 3: {_hours(hours, unit)}, not above {GATE3_HOURS:g} hour: no approval "
                f"needed; without one this run may spend up to {GATE3_HOURS:g} hour"
            )
        lines.append(IDLE_REMINDER)
    lines.extend(plan.history)
    return lines


# -- running ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitOutcome:
    """One attempt of a compute unit (a ``fit`` entry)."""

    recipe_id: str
    run_label: str
    attempt: int
    status: str
    checkpoint_id: str | None
    registry_line_sha256: str | None
    fit_seq: int | None
    shown: Mapping[str, Any] | None  # pilot only: the done entry's details


@dataclass(frozen=True)
class FoldOutcome:
    """What one invocation did; ``lines`` is everything it printed (through ``progress``)."""

    units: tuple[UnitOutcome, ...]
    spent_hours: float
    unit: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class PilotRecord:
    """The effective recipe's registered and recorded pilot (:func:`resolve_pilot`)."""

    recipe: Recipe
    fold: Fold
    checkpoint: CheckpointRef  # root=ft_root, line_sha256 from the ledger's fit entry
    registry_entry: RegistryEntry
    fit_record_path: Path
    fit_record_sha256: str
    export: ExportRecord  # read_export(fold_export_dir(recipe8, PILOT_LABEL))


def _gate3_cap(plan: FoldPlan, approved_hours: float | None) -> float:
    """The cap of this invocation's spend, in the first unit's hours; :class:`FoldRefused`
    (nothing written) when the first unit passes gate 3 without a figure, or 1.25 times the
    figure."""
    if approved_hours is not None and (not _finite(approved_hours) or not approved_hours > 0):
        raise ValueError(f"the approved hours must be a finite number > 0, got {approved_hours!r}")
    cap = GATE3_HOURS if approved_hours is None else APPROVAL_TOLERANCE * approved_hours
    first = plan.first
    if first is None:
        return cap
    hours, unit = first.device_hours, first.unit
    if approved_hours is None:
        if hours > GATE3_HOURS:
            raise FoldRefused(
                f"fold {plan.fold.run_label} needs about {_hours(hours, unit)}, above the "
                f"{GATE3_HOURS:g} hour of gate 3 (SPEC section 17): show the estimate to the "
                "owner and rerun with --approved-by-owner HOURS, the figure they approve "
                "(nothing ran)"
            )
        return cap
    ratio = hours / approved_hours
    if ratio > APPROVAL_TOLERANCE:
        raise FoldRefused(
            f"the measured estimate {_hours(hours, unit)} is {ratio:.2f} times the "
            f"{approved_hours:g} {UNIT_TEXT[unit]} the owner approved, above the tolerance of "
            f"{APPROVAL_TOLERANCE:g} (D-040, D-046): show the owner the estimate (nothing ran)"
        )
    return cap


def _same_state(plan: FoldPlan, state: _State) -> None:
    expected = plan.first.recipe_id if plan.first is not None else None
    found = recipe_id(state.compute) if state.compute is not None else None
    if state.stage != plan.stage or found != expected:
        raise FoldRefused(
            f"fold {plan.fold.run_label} moved from stage {plan.stage} to {state.stage} since its "
            "estimate was made (another run?): run the command again to see the new estimate"
        )


def _number(value: object) -> float | None:
    return float(value) if _finite(value) else None


def _pilot_details(
    document: Mapping[str, Any], recipe: Recipe, *, rule: str, selected: int
) -> dict[str, Any]:
    """The pilot's ``done`` details read from its fold record (the pilot is shown)."""
    epochs = []
    for item in document.get("epochs") or ():
        if not isinstance(item, Mapping):
            continue
        epoch = item.get("pseudo_epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            continue
        epochs.append(
            {
                "pseudo_epoch": epoch,
                "nll4_tf": _number(item.get("nll4_tf")),
                "train_loss": _number(item.get("train_loss")),
                "vendored_val_loss": _number(item.get("vendored_val_loss")),
            }
        )
    step_seconds = document.get("step_seconds")
    median = step_seconds.get("median") if isinstance(step_seconds, Mapping) else None
    stop = document.get("stop_reason")
    return {
        "epochs": epochs,
        "selected_pseudo_epoch": selected,
        "stop_reason": stop if isinstance(stop, str) else None,
        "max_epochs": recipe.max_epochs,
        "pilot_rule": rule,
        "wall_seconds": _number(document.get("wall_seconds")),
        "median_step_seconds": _number(median),
    }


def _value_text(value: object) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _curve_lines(fold: Fold, recipe: Recipe, details: Mapping[str, Any]) -> list[str]:
    lines = [
        (
            f"pilot {fold.run_label}, recipe {recipe.name} ({recipe8(recipe)}, E_max "
            f"{recipe.max_epochs}): NLL4-TF per pseudo-epoch in nats per candle (pseudo-epoch 0 "
            "is the shipped weights, never selected; training-segment data, shown: design 6.3)"
        )
    ]
    for epoch in details["epochs"]:
        lines.append(
            f"  pseudo-epoch {epoch['pseudo_epoch']}: NLL4-TF {_value_text(epoch['nll4_tf'])}, "
            f"training loss {_value_text(epoch['train_loss'])}, vendored validation loss "
            f"{_value_text(epoch['vendored_val_loss'])}"
        )
    lines.append(
        f"selected pseudo-epoch {details['selected_pseudo_epoch']} of {details['max_epochs']} "
        f"(stop: {details['stop_reason']}); pilot rule ({PILOT_RULE['condition']}, "
        f"{PILOT_RULE['applies']}): {details['pilot_rule']}"
    )
    return lines


def _moment_text(moment: datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ`` of an aware datetime."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class _FoldRun:
    """One invocation of ``finetune fold`` under the fold lock (:func:`run_fold`)."""

    def __init__(
        self,
        plan: FoldPlan,
        *,
        cap: float,
        steps: FoldSteps,
        ft_root: Path | None,
        data_root: Path | None,
        ledger_root: Path | None,
        say: Callable[[str], None],
    ) -> None:
        self.plan = plan
        self.fold = plan.fold
        self.cap = cap
        self.steps = steps
        self.ft_root = ft_root
        self.data_root = data_root
        self.ledger_root = ledger_root
        self.say = say
        self.unit = UNITS[plan.compute.device]
        self.spent = 0.0
        self.approval_seq: int | None = None
        self.approved_hours: float | None = None
        self.units: list[UnitOutcome] = []
        self.progressed = False
        estimates = [item for item in (plan.first, plan.refit) if item is not None]
        self.estimates = {item.recipe_id: item for item in estimates}

    # -- the ledger ----------------------------------------------------------------------------

    def entries(self) -> tuple[ledger.LedgerEntry, ...]:
        return ledger.read_ledger(self.ledger_root)

    def state(self) -> _State:
        entries = self.entries()
        return _fold_state(self.fold, self.plan.recipe, entries=entries, ft_root=self.ft_root)

    def record_fit(
        self,
        recipe: Recipe,
        *,
        status: str,
        attempt: int,
        run_id: str | None,
        entry: RegistryEntry | None,
        fit_record_sha256: str | None,
        wall_seconds: float,
        cpu_seconds: float,
        approval_seq: int | None,
    ) -> ledger.LedgerEntry:
        return ledger.record_fit(
            checkpoint_id=entry.checkpoint_id if entry is not None else None,
            registry_line_sha256=entry.line_sha256 if entry is not None else None,
            recipe_id=recipe_id(recipe),
            fold=self.fold.label,
            replicate=self.fold.replicate,
            run_id=run_id,
            status=status,
            attempt=attempt,
            blinded=self.plan.blinded,
            fit_record_sha256=fit_record_sha256,
            device=self.plan.compute.device,
            threads=self.plan.compute.threads,
            wall_seconds=max(0.0, float(wall_seconds)),
            cpu_seconds=max(0.0, float(cpu_seconds)),
            approval_seq=approval_seq,
            order_id=self.plan.order_id,
            root=self.ledger_root,
        )

    def record_failure(
        self,
        recipe: Recipe,
        *,
        status: str,
        attempt: int,
        run_id: str | None,
        wall: float,
        cpu: float,
    ) -> ledger.LedgerEntry:
        fit = self.record_fit(
            recipe,
            status=status,
            attempt=attempt,
            run_id=run_id,
            entry=None,
            fit_record_sha256=None,
            wall_seconds=wall,
            cpu_seconds=cpu,
            approval_seq=self.approval_seq,
        )
        self.units.append(
            UnitOutcome(
                recipe_id(recipe), self.fold.run_label, attempt, status, None, None, fit.seq, None
            )
        )
        return fit

    # -- bookkeeping ---------------------------------------------------------------------------

    def bookkeeping(self, state: _State) -> None:
        """Record the missing entries of a registered fold from its registry entry (and, for
        the pilot, its fold record), without compute."""
        records = state.records
        entry = records.entry
        assert entry is not None
        if self.plan.is_pilot:
            record, digest = _read_fold_record(entry, self.ft_root)
            wall = _number(record.get("wall_seconds")) or 0.0
            cpu = _number(record.get("cpu_seconds")) or 0.0
        else:  # blinded: the fold record is never opened, so no spend is read from it
            record, digest = {}, str(entry.document.get("fit_record_sha256"))
            wall = cpu = 0.0
        fit = records.fit
        if fit is None:
            approval_seq, attempt = self.registered_unit(records.recipe)
            fit = self.record_fit(
                records.recipe,
                status="registered",
                attempt=attempt,
                run_id=str(entry.document.get("run_id")),
                entry=entry,
                fit_record_sha256=digest,
                wall_seconds=wall,
                cpu_seconds=cpu,
                approval_seq=approval_seq,
            )
            self.say(
                f"bookkeeping: fit entry #{fit.seq} (line {fit.line_sha256[:12]}...) records "
                f"{entry.checkpoint_id}, registered by run {entry.document.get('run_id')}"
            )
        if self.plan.is_pilot and records.done is None:
            self.record_pilot_done(records.recipe, entry, record, digest, fit_seq=fit.seq)
        self.progressed = True

    def registered_unit(self, recipe: Recipe) -> tuple[int | None, int]:
        """``(approval_seq, attempt)`` of a registered run of ``recipe`` whose ``fit`` entry is
        missing (killed between its registration and the ledger), from what was recorded before
        it, never from this invocation: the unit's anchor is the pilot's latest ``planned`` entry
        of ``recipe`` (its details name the approval the unit ran under) or, for another fold,
        the latest ``gate3_approval`` of its run label recorded before this invocation's own; the
        run was the retry (attempt 2) when the last ``fit`` entry of ``recipe`` and the run
        label after the anchor is a retried failure of attempt 1."""
        entries = self.entries()
        identifier = recipe_id(recipe)
        label = self.fold.run_label
        anchor, approval_seq = 0, None
        if self.plan.is_pilot:
            planned = [
                item
                for item in entries
                if item.kind == "exploration"
                and _payload(item).get("activity") == PILOT_ACTIVITY
                and _payload(item).get("stage") == "planned"
                and _payload(item).get("recipe_id") == identifier
                and _payload(item).get("subject") == label
            ]
            if planned:
                anchor = planned[-1].seq
                value = _details(planned[-1]).get("approval_seq")
                if isinstance(value, int) and not isinstance(value, bool):
                    approval_seq = value
        else:
            approvals = [
                item.seq
                for item in entries
                if item.kind == "gate3_approval"
                and _payload(item).get("command") == COMMAND
                and _payload(item).get("subject") == label
                and (self.approval_seq is None or item.seq < self.approval_seq)
            ]
            if approvals:
                anchor = approval_seq = approvals[-1]
        fits = [
            _payload(item)
            for item in entries
            if item.seq > anchor
            and item.kind == "fit"
            and _payload(item).get("recipe_id") == identifier
            and _payload(item).get("fold") == self.fold.label
            and _payload(item).get("replicate") == self.fold.replicate
        ]
        retried = bool(fits) and (
            fits[-1].get("status") in RETRY_STATUSES and fits[-1].get("attempt") == 1
        )
        return approval_seq, 2 if retried else 1

    def record_pilot_done(
        self,
        recipe: Recipe,
        entry: RegistryEntry,
        record: Mapping[str, Any],
        digest: str,
        *,
        fit_seq: int,
    ) -> dict[str, Any]:
        """The pilot's ``done`` entry from its fold record, with the pilot rule's outcome; its
        curve is printed (the pilot is shown)."""
        selected = _selected_pseudo_epoch(record, recipe)
        rule = pilot_rule(recipe, selected)
        details = _pilot_details(record, recipe, rule=rule, selected=selected)
        details.update(
            checkpoint_id=entry.checkpoint_id,
            registry_line_sha256=entry.line_sha256,
            fit_record_sha256=digest,
            run_id=str(entry.document.get("run_id")),
        )
        done = ledger.record_exploration(
            activity=PILOT_ACTIVITY,
            subject=self.fold.run_label,
            stage="done",
            recipe_id=recipe_id(recipe),
            inputs=["training"],
            outcomes=["training"],
            details=details,
            root=self.ledger_root,
        )
        for line in _curve_lines(self.fold, recipe, details):
            self.say(line)
        self.say(
            f"ledger: pilot done entry #{done.seq} (line {done.line_sha256[:12]}...) after fit "
            f"entry #{fit_seq}"
        )
        return details

    # -- a compute unit ------------------------------------------------------------------------

    def deadline(self, estimate: FitEstimate) -> float | None:
        """The next child's deadline in seconds, or ``None`` when the cap does not admit it."""
        if self.spent + estimate.device_hours > self.cap:
            return None
        divisor = 1.0 if self.plan.compute.device == "mps" else float(self.plan.compute.threads)
        left = (self.cap - self.spent) * _SECONDS_PER_HOUR / divisor
        return min(DEADLINE_FACTOR * estimate.wall_seconds, left)

    def over_cap(self, estimate: FitEstimate, what: str) -> str:
        if self.approved_hours is None:
            limit = f"the {GATE3_HOURS:g} hour gate 3 allows without an approval"
        else:
            limit = (
                f"{APPROVAL_TOLERANCE:g} x the approved {self.approved_hours:g} "
                f"{UNIT_TEXT[self.unit]}"
            )
        return (
            f"{what} needs about {_hours(estimate.device_hours, estimate.unit)}; with "
            f"{_hours(self.spent, self.unit)} spent in this run that passes {limit}"
        )

    def sealed_access(self, recipe: Recipe) -> SealedAccess | None:
        """The access a fold that reads sealed candles exports with: the owner's order, read from
        this run's ledger, which must be one that may open this run's data root (gate 6)."""
        if not self.fold.needs_sealed_access:
            return None
        assert self.plan.order_id is not None  # plan_fold refuses a sealed fold without it
        _check_order(
            self.plan.order_id,
            self.fold,
            recipe,
            ledger_root=self.ledger_root,
            data_root=self.data_root,
        )
        from ..data.partition import sealed_access

        return sealed_access(
            self.plan.order_id, ledger_root=self.ledger_root, data_root=self.data_root
        )

    def compute_unit(self, recipe: Recipe, *, refit: bool) -> None:
        """Export the fold, then train ``recipe`` on it with the retry policy and the ledger
        entries of the module docstring."""
        estimate = self.estimates.get(recipe_id(recipe))
        if estimate is None:
            raise FoldRefused(
                f"no estimate of recipe {recipe8(recipe)} was shown for this run: run the command "
                "again to see it"
            )
        deadline = self.deadline(estimate)
        if deadline is None:
            if refit:
                raise FoldRefused(
                    self.over_cap(
                        estimate, f"the pilot rule fired; the refit at E_max {recipe.max_epochs}"
                    )
                    + f": show the owner; rerun `finetune fold --fold {self.fold.run_label} "
                    "--approved-by-owner H` with the figure they approve for the refit"
                )
            raise FoldRefused(
                self.over_cap(estimate, f"fold {self.fold.run_label}")
                + ": nothing more starts (what ran stays recorded); show the owner"
            )
        export = self.steps.export_fold(
            self.fold, recipe, root=self.data_root, sealed_access=self.sealed_access(recipe)
        )
        self.say(
            f"export {self.fold.label} of recipe {recipe8(recipe)}: {export.rows} rows, content "
            f"sha256 {export.content_sha256[:12]}..."
        )
        if self.plan.is_pilot:
            self.planned(recipe, estimate, export)
        request = RunRequest(
            mode="train",
            recipe=recipe,
            fold=self.fold,
            export=export,
            compute=self.plan.compute,
            wrappers=Wrappers(),
            deadline_seconds=deadline,
            blinded=self.plan.blinded,
        )
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt > 1:
                deadline = self.deadline(estimate)
                if deadline is None:
                    raise FoldFailed(
                        self.over_cap(estimate, f"the retry of {self.fold.run_label}")
                        + ": the fold failed and its retry does not start; tell the owner "
                        "(design 6.3)"
                    )
                request = replace(request, deadline_seconds=deadline)
            if self.train(request, recipe, attempt=attempt):
                return

    def planned(self, recipe: Recipe, estimate: FitEstimate, export: ExportRecord) -> None:
        """The pilot's ``planned`` entry: what runs, and the pilot rule before its result."""
        identifier = recipe_id(recipe)
        seeds = training_seeds(identifier, self.fold.label, self.fold.replicate)
        details = {
            "max_epochs": recipe.max_epochs,
            "onecycle_total_steps": recipe.onecycle_total_steps,
            "seeds_digest": seeds.digest,
            "compute": asdict(self.plan.compute),
            "export_content_sha256": export.content_sha256,
            "timing_record": {"stem": timing.TIMING_STEM, "sha256": self.plan.record_sha256},
            "training_code_sha256": training_code_sha256(),
            "estimate_hours": estimate.device_hours,
            "unit": estimate.unit,
            "approval_seq": self.approval_seq,
            "pilot_rule_definition": dict(PILOT_RULE),
        }
        entry = ledger.record_exploration(
            activity=PILOT_ACTIVITY,
            subject=self.fold.run_label,
            stage="planned",
            recipe_id=identifier,
            inputs=["training"],
            outcomes=["training"],
            details=details,
            root=self.ledger_root,
        )
        self.say(
            f"ledger: pilot planned entry #{entry.seq} (line {entry.line_sha256[:12]}...), the "
            "pilot rule recorded before the pilot's result exists"
        )

    def run_dirs(self) -> set[str]:
        parent = registry.runs_dir(self.ft_root)
        prefix = f"{self.fold.run_label}-"
        try:
            return {
                path.name
                for path in parent.iterdir()
                if path.name.startswith(prefix) and path.is_dir()
            }
        except FileNotFoundError:
            return set()

    def failed_run(self, before: set[str]) -> tuple[str, str | None]:
        """``(status, run_id)`` of the run directory the call created: its record's status
        (only that is read), ``no_result`` without a record, ``killed`` without exactly one new
        directory or with an unreadable record; a run that completed but was not registered
        counts as ``failed``."""
        created = sorted(self.run_dirs() - before)
        if len(created) != 1:
            return "killed", None
        run_id = created[0]
        path = registry.runs_dir(self.ft_root) / run_id / runner.RESULT_JSON
        try:
            document = json.loads(path.read_bytes())
        except FileNotFoundError:
            return "no_result", run_id
        except (OSError, ValueError):
            return "killed", run_id
        status = document.get("status") if isinstance(document, dict) else None
        if status == "completed":
            return "failed", run_id
        if status not in FAILED_RUN_STATUSES:
            return "killed", run_id
        return str(status), run_id

    def charge(self, clock0: float, cpu0: float, own0: float) -> tuple[float, float]:
        """``(wall, cpu)`` seconds since the marks; the device's hours are added to the spend."""
        wall = max(0.0, self.steps.clock() - clock0)
        cpu = max(0.0, self.steps.children_cpu_seconds() - cpu0)
        cpu += max(0.0, time.process_time() - own0)
        self.spent += (wall if self.plan.compute.device == "mps" else cpu) / _SECONDS_PER_HOUR
        return wall, cpu

    def train(self, request: RunRequest, recipe: Recipe, *, attempt: int) -> bool:
        """One child; True once registered, False after a failure that is retried."""
        deadline = request.deadline_seconds
        assert deadline is not None
        self.say(
            f"training {self.fold.run_label} (recipe {recipe8(recipe)}, attempt {attempt} of "
            f"{MAX_ATTEMPTS}) on {self.plan.compute.device.upper()}: deadline {deadline:.0f} s"
        )
        self.progressed = True
        before = self.run_dirs()
        clock0 = self.steps.clock()
        cpu0 = self.steps.children_cpu_seconds()
        own0 = time.process_time()
        try:
            outcome = self.steps.fit(
                request,
                root=self.ft_root,
                timeout_seconds=deadline + CHILD_TIMEOUT_MARGIN_SECONDS,
                register=True,
            )
        except (ChildFailed, ChildMemoryAbort) as exc:
            wall, cpu = self.charge(clock0, cpu0, own0)
            status, run_id = self.failed_run(before)
            fit = self.record_failure(
                recipe, status=status, attempt=attempt, run_id=run_id, wall=wall, cpu=cpu
            )
            what = (
                f"fold {self.fold.run_label} attempt {attempt} (run {run_id or 'unknown'}) "
                f"ended {status} (fit entry #{fit.seq})"
            )
            if status in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                self.say(f"{what}: retried once with the same seeds (design 6.3)")
                return False
            detail = "" if self.plan.blinded else f": {exc}"
            if status == "memory_abort":
                raise FoldFailed(
                    f"{what}{detail}; a memory abort is not a fold failure (design 3.5): tell "
                    "the owner; the fold restarts from scratch once the machine has the memory"
                ) from None
            again = "after its retry" if attempt > 1 else "and is not retried"
            raise FoldFailed(f"{what} {again}{detail}: tell the owner (design 6.3)") from None
        except BaseException:
            wall, cpu = self.charge(clock0, cpu0, own0)
            if self.run_dirs() - before:  # the child started: its status is recorded first
                status, run_id = self.failed_run(before)
                try:
                    fit = self.record_failure(
                        recipe, status=status, attempt=attempt, run_id=run_id, wall=wall, cpu=cpu
                    )
                    self.say(
                        f"fold {self.fold.run_label} attempt {attempt} (run {run_id or 'unknown'}) "
                        f"stopped: {status} (fit entry #{fit.seq})"
                    )
                except Exception as error:  # noqa: BLE001 - the original exception goes on
                    self.say(
                        "WARNING: the stopped run could not be recorded in the ledger "
                        f"({type(error).__name__}: {error})"
                    )
            raise
        wall, cpu = self.charge(clock0, cpu0, own0)
        self.registered(outcome, recipe, attempt=attempt, wall=wall, cpu=cpu)
        return True

    def registered(
        self, outcome: FitOutcome, recipe: Recipe, *, attempt: int, wall: float, cpu: float
    ) -> None:
        """The ``fit`` entry of a registered run; the pilot's ``done`` entry and curve, or a
        blinded fold's printable outcome."""
        entry = outcome.checkpoint
        if entry is None:
            raise FoldFailed(f"run {outcome.run_id} registered no checkpoint")
        if self.plan.is_pilot:
            record, digest = _read_fold_record(entry, self.ft_root)
        else:  # blinded: the registry entry's hash of the record; the record is never opened
            record, digest = {}, str(entry.document.get("fit_record_sha256"))
        fit = self.record_fit(
            recipe,
            status="registered",
            attempt=attempt,
            run_id=outcome.run_id,
            entry=entry,
            fit_record_sha256=digest,
            wall_seconds=wall,
            cpu_seconds=cpu,
            approval_seq=self.approval_seq,
        )
        weights = next(
            (item.sha256 for item in entry.files if item.name == "model.safetensors"), ""
        )
        shown = None
        if self.plan.is_pilot:
            self.say(
                f"registered {entry.checkpoint_id} (registry line {entry.line_sha256[:12]}..., "
                f"model.safetensors {weights[:12]}...): fit entry #{fit.seq}, "
                f"{wall / 3600:.3f} wall-hours"
            )
            shown = MappingProxyType(
                self.record_pilot_done(recipe, entry, record, digest, fit_seq=fit.seq)
            )
        else:
            printable = outcome.printable()
            self.say(
                f"fold {self.fold.run_label}: run {printable['run_id']} {printable['status']}, "
                f"finite {printable['finite']}, wall {float(printable['wall_seconds']):.0f} s, "
                f"checkpoint {entry.checkpoint_id} (model.safetensors {weights[:12]}...); fit "
                f"entry #{fit.seq}"
            )
        self.units.append(
            UnitOutcome(
                recipe_id(recipe),
                self.fold.run_label,
                attempt,
                "registered",
                entry.checkpoint_id,
                entry.line_sha256,
                fit.seq,
                shown,
            )
        )

    # -- the invocation ------------------------------------------------------------------------

    def run(self, state: _State) -> None:
        """Work through the stages from ``state`` (read under the lock) until done, or the
        single unit of a fold other than the pilot."""
        if not self.plan.is_pilot:
            if state.stage == "bookkeeping":
                self.bookkeeping(state)
            else:
                assert state.compute is not None
                self.compute_unit(state.compute, refit=False)
            return
        for _ in range(_MAX_STATE_STEPS):
            if state.stage == "done":
                if not self.progressed:
                    self.say("nothing to run: the pilot is registered and recorded")
                else:
                    for line in _state_lines(state, self.fold):
                        self.say(line)
                return
            if state.stage == "bookkeeping":
                self.bookkeeping(state)
            else:
                assert state.compute is not None
                self.compute_unit(state.compute, refit=state.stage == "refit")
            state = self.state()
        raise FoldFailed(f"fold {self.fold.run_label}: its stages did not settle (a bug)")


def run_fold(
    plan: FoldPlan,
    *,
    approved_hours: float | None,
    given_at: datetime,
    steps: FoldSteps,
    ft_root: Path | None = None,
    data_root: Path | None = None,
    ledger_root: Path | None = None,
    progress: Callable[[str], None] = print,
) -> FoldOutcome:
    """Run the plan (module docstring): gate 3 on the first compute unit, before anything is
    written; the fold lock (``<fine-tune root>/.scratch/fold.lock``, a held one is
    ``LockBusyError``) and no timing run in progress; the state read again under the lock (a
    state that moved since the plan is refused); the approval entry (with a figure); then the
    bookkeeping and the compute units the cap admits. Every line goes to ``progress`` as it
    happens and into the outcome's ``lines``."""
    if not isinstance(plan, FoldPlan):
        raise TypeError(f"expected a FoldPlan, got {type(plan).__name__}")
    if not isinstance(steps, FoldSteps):
        raise TypeError(f"expected FoldSteps, got {type(steps).__name__}")
    if not isinstance(given_at, datetime) or given_at.tzinfo is None:
        raise ValueError("given_at must be an aware datetime (the command's start)")
    lines: list[str] = []

    def say(line: str) -> None:
        lines.append(line)
        progress(line)

    cap = _gate3_cap(plan, approved_hours)
    scratch = registry.scratch_dir(ft_root)
    with exclusive_lock(scratch / FOLD_LOCK_NAME):
        if lock_is_held(scratch / timing.LOCK_NAME):
            raise FoldRefused(
                "a timing run (`finetune timing`) is in progress on this fine-tune root: run the "
                "fold after it"
            )
        state = _fold_state(
            plan.fold, plan.recipe, entries=ledger.read_ledger(ledger_root), ft_root=ft_root
        )
        _same_state(plan, state)
        run = _FoldRun(
            plan,
            cap=cap,
            steps=steps,
            ft_root=ft_root,
            data_root=data_root,
            ledger_root=ledger_root,
            say=say,
        )
        first = plan.first
        if first is not None and approved_hours is not None:
            approval = ledger.record_gate3_approval(
                command=COMMAND,
                subject=plan.fold.run_label,
                unit=first.unit,
                approved=float(approved_hours),
                estimate=first.device_hours,
                given_at=given_at,
                estimate_detail={
                    "first": first.as_dict(),
                    "refit": plan.refit.as_dict() if plan.refit is not None else None,
                },
                root=ledger_root,
            )
            run.approval_seq = approval.seq
            run.approved_hours = float(approved_hours)
            say(
                f"gate 3: the owner approved {approved_hours:g} {UNIT_TEXT[first.unit]} for an "
                f"estimate of {_hours(first.device_hours, first.unit)} (ratio "
                f"{first.device_hours / approved_hours:.2f}; --approved-by-owner given at "
                f"{_moment_text(given_at)}): approval entry #{approval.seq} (line "
                f"{approval.line_sha256[:12]}...); this run may spend up to "
                f"{_hours(cap, first.unit)}"
            )
        elif first is not None:
            say(
                f"gate 3: {_hours(first.device_hours, first.unit)}, not above {GATE3_HOURS:g} "
                f"hour: no approval needed; this run may spend up to {_hours(cap, first.unit)}"
            )
        elif approved_hours is not None:
            say("nothing to compute: the --approved-by-owner figure is not used or recorded")
        run.run(state)
    return FoldOutcome(
        units=tuple(run.units), spent_hours=run.spent, unit=run.unit, lines=tuple(lines)
    )


# -- the pilot for the pilot check ----------------------------------------------------------------


def resolve_pilot(
    *,
    ft_root: Path | None = None,
    data_root: Path | None = None,
    ledger_root: Path | None = None,
) -> PilotRecord:
    """The effective recipe's registered pilot, for ``finetune pilot-check`` (design 6.2): its
    registered ``fit`` entry and the registry entry at the line it recorded
    (``registry.find_entry``: a changed line is a ``RegistryChainError``), its ``done``
    exploration entry with a rule outcome other than ``fired``, the fold record hashing to the
    recorded ``fit_record_sha256`` and the export whose content sha256 the registry entry
    recorded. Anything missing is :class:`FoldRefused` (train the pilot first, or refit it);
    anything changed is :class:`FoldFailed` (or the registry's and the exporter's own errors)."""
    from .export_csv import fold_export_dir, read_export

    entries = ledger.read_ledger(ledger_root)
    recipe = effective_recipe(entries=entries)
    fold = fold_for(PILOT_LABEL, recipe)
    checkpoint_id = checkpoint_id_for(recipe8(recipe), fold.run_label)
    doubled = recipe.max_epochs != R1.max_epochs
    fit = _registered_fit(entries, checkpoint_id)
    if fit is None:
        if doubled:
            raise FoldRefused(
                f"the pilot rule fired: refit fold 1 first (finetune fold --fold {PILOT_LABEL} "
                f"--approved-by-owner HOURS): {checkpoint_id} has no registered fit entry"
            )
        raise FoldRefused(
            f"train the pilot first: finetune fold --fold {PILOT_LABEL} ({checkpoint_id} has no "
            "registered fit entry in the ledger)"
        )
    done = _pilot_done(entries, recipe, fold)
    if done is None:
        raise FoldRefused(
            f"the pilot {checkpoint_id} has no done entry in the ledger: run finetune fold --fold "
            f"{PILOT_LABEL}, which records it"
        )
    if _details(done).get("pilot_rule") == RULE_FIRED:
        raise FoldRefused(
            f"the pilot rule fired: refit fold 1 first (finetune fold --fold {PILOT_LABEL})"
        )
    payload = _payload(fit)
    reference = CheckpointRef(
        checkpoint_id, root=ft_root, line_sha256=payload.get("registry_line_sha256")
    )
    entry = registry.find_entry(reference)
    record_path = _fold_record_path(entry, ft_root)
    _, digest = _read_fold_record(entry, ft_root)
    if digest != payload.get("fit_record_sha256"):
        raise FoldFailed(
            f"the pilot's fold record hashes to {digest[:12]}..., not to the fit_record_sha256 "
            f"of its fit entry #{fit.seq}"
        )
    export = read_export(fold_export_dir(recipe8(recipe), PILOT_LABEL, root=data_root))
    if export.content_sha256 != entry.document.get("export_content_sha256"):
        raise FoldFailed(
            f"the pilot's export ({PILOT_LABEL}, recipe {recipe8(recipe)}) has content sha256 "
            f"{export.content_sha256[:12]}..., not the one its registry entry recorded"
        )
    return PilotRecord(
        recipe=recipe,
        fold=fold,
        checkpoint=reference,
        registry_entry=entry,
        fit_record_path=record_path,
        fit_record_sha256=digest,
        export=export,
    )


__all__ = [
    "APPROVAL_TOLERANCE",
    "CHILD_TIMEOUT_MARGIN_SECONDS",
    "COMMAND",
    "DEADLINE_FACTOR",
    "FOLD_LOCK_NAME",
    "GATE3_HOURS",
    "PILOT_LABEL",
    "PILOT_RULE",
    "PILOT_RULE_FACTOR",
    "RETRY_STATUSES",
    "SETUP_SECONDS_PRIOR",
    "VALIDATION_FORWARD_SHARE",
    "WALK_FORWARD_LAYOUT",
    "FitEstimate",
    "FoldFailed",
    "FoldOutcome",
    "FoldPlan",
    "FoldRefused",
    "FoldSteps",
    "PilotRecord",
    "UnitOutcome",
    "compute_choice",
    "doubled_recipe",
    "effective_recipe",
    "estimate_lines",
    "fit_estimate",
    "is_pilot",
    "pilot_rule",
    "plan_fold",
    "real_steps",
    "resolve_pilot",
    "run_fold",
]
