"""Batches of Kronos fans: windows built in the parent, spawn-started workers, gate 3.

A batch turns :class:`FanTask` items (source, decision hour, stream) into stored fans
(:mod:`kronos_agent.data.forecast_store`) and one :class:`FanSummary` per task, in task order
(``docs/DATA.md``, "Batches" and "Gate 3"):

1. :func:`prepare_tasks` runs in the parent process. It checks the runtime thread count and the
   embargo, refuses decision hours before the Kronos cutoff or in the sealed segment (a batch
   never holds a ``SealedAccess``), builds every task's window from the 1h stores (one study
   read per source, over the hours the tasks need), drops and lists the hours whose window is
   incomplete with the missing ranges, checks every fan already stored for a kept task, and
   counts the fans still to compute (``pending``) under the configuration the run will produce.
2. :func:`measure_seconds_per_fan` times one fan after an untimed warm-up on a synthetic window
   at the governed thread count; :func:`estimate` gives ``cpu_seconds = pending x seconds per fan
   x threads`` and ``wall_seconds = cpu_seconds / (threads x workers)``; :func:`require_gate3`
   refuses more than :data:`GATE3_CPU_SECONDS` unless the owner approved a figure in CPU-hours
   and the measured estimate is at most :data:`GATE3_APPROVAL_TOLERANCE` times it (SPEC section
   17, D-040): the approval names the estimate it covers, so a re-measured estimate well above
   the one shown goes back to the owner before any fan runs.
3. :func:`run_fan_batch` reuses every stored fan after
   :func:`~kronos_agent.data.forecast_store.check_reusable` (never recomputed, never
   overwritten), refuses to compute more fans than the estimate counted, and forecasts the rest:
   in this process for one worker, otherwise in spawn-started worker processes, at most
   ``os.cpu_count() // threads`` of them and only with at least 4 GiB of RAM. Each worker runs
   :func:`~kronos_agent.runtime.configure_runtime` with the governed thread count, then
   :func:`~kronos_agent.kronos.vendored.import_kronos` (decision D-014: a spawn-started process
   verifies and imports the vendored tree itself), loads the models once and checks that they
   give the configuration the parent prepared. The parent passes its loaded ``Params`` and the
   windows it built; workers return fans, and the parent checks and stores them.

A fan depends only on its window, stream and configuration: the seeds come from the decision
hour and the stream, :func:`~kronos_agent.kronos.forecast.forecast_fan` reseeds ``random``,
``numpy`` and ``torch`` right before sampling, and every process runs the same thread count, so
neither the number of workers nor the order of the tasks changes a fan (tested bit for bit).

Step 6 (``docs/RETRAINING.md``, 3.6 and 5.3): a batch is prepared with an optional evaluation
length ``L`` (fans of ``L`` steps, their own ``config_id``), a fine-tuned checkpoint of the
fine-tune registry (one batch runs one checkpoint; its hashes give the configuration and its data
end the model period; the batch keeps the sha256 of the registry line it read, so its loaders
refuse an entry changed after the preparation) and a model-period role, and
:func:`run_fan_batch` runs it exactly as it was prepared. :func:`time_fan_workers` times
spawn-started workers on the synthetic window (nothing stored), and :func:`identity_control`
sends the shipped weights through the whole checkpoint path (``save_pretrained``, a scratch
registry, ``load_models(checkpoint=...)`` in a spawn-started worker) and compares the fan with
the shipped model's, bit for bit.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import io
import math
import multiprocessing
import os
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..data.forecast_store import (
    ForecastStoreError,
    StoredFan,
    check_reusable,
    check_stream,
    config_id,
    fan_relpath,
    forecast_config,
    load_fan,
    paths_sha256,
    save_fan,
)
from ..data.fsutil import ensure_directory
from ..data.partition import (
    KRONOS_FIRST_DECISION_MS,
    MODEL_PERIOD_EVALUATION,
    MODEL_PERIOD_ROLES,
    MODEL_PERIOD_TIMING,
    PartitionError,
    check_embargo,
    require_accessible,
    require_model_period,
    sample_fits,
)
from ..data.store import SOURCE_SYMBOLS
from ..data.timeutil import HOUR_MS, parse_utc_hour, to_ms, utc_text
from ..data.windows import (
    DecisionWindow,
    StoreWindowSource,
    WindowError,
    WindowGapError,
    build_window,
)
from ..exchange.public_market import Candle
from ..finetune import registry
from ..finetune.registry import CheckpointRef
from ..hashing import sha256_file
from ..manifests import ManifestMismatch
from ..models import ModelSpec, spec_for
from ..params import KronosParams, Params, load_params
from ..paths import data_dir, default_label
from ..provenance import code_version
from ..runtime import configure_runtime, require_configured
from .decision import (
    DEFAULT_STREAM,
    evaluation_params,
    fan_mean_close,
    fan_p_up,
    fan_signal,
    forecast_window,
)
from .features import synthetic_window
from .forecast import Fan
from .vendored import (
    VendoredTreeError,
    import_kronos,
    source_manifest_path,
    verify_vendored_tree,
)
from .weights import (
    LoadedModels,
    WeightsError,
    checkpoint_dir_problems,
    load_models,
    repos_for_model,
    verify_model,
    verify_repo,
)

GATE3_CPU_SECONDS = 3600
# D-040: a measured estimate more than 25 % above the CPU-hours the owner approved goes back to
# the owner before any fan runs; exactly 1.25 times the approved figure still passes.
GATE3_APPROVAL_TOLERANCE = 1.25
MIN_RAM_BYTES_FOR_WORKERS = 4 * 2**30
START_METHOD = "spawn"
# the synthetic timing window: one hour after the end of features.synthetic_window's default grid
TIMING_DECISION_HOUR = datetime(2026, 10, 29, 17, tzinfo=UTC)
TIMING_SOURCE = "binance"  # a Binance-shaped window accepts every amount policy
TIMING_WARMUP_STREAM = "timing-warmup"
TIMING_STREAM = "timing"
# str.format template of the streams of time_fan_workers: timing-w0, timing-w1, ...
TIMING_WORKER_STREAM = "timing-w{index}"
# the shipped weights registered as a checkpoint by identity_control (a scratch registry)
IDENTITY_CHECKPOINT_ID = "ks-identity-shipped"
# how long time_fan_workers waits for its workers to load the models and run their warm-up
TIMING_STARTUP_TIMEOUT_S = 900.0
_STARTUP_POLL_S = 0.05

Progress = Callable[[int, int, "FanSummary"], None]

# the timing clock is a module attribute so that tests can drive it
_clock: Callable[[], float] = time.perf_counter
# state of a spawn-started worker, set once by _init_worker
_WORKER: dict[str, Any] = {}


class BatchError(RuntimeError):
    """A batch that cannot run as prepared: duplicate tasks, another configuration, workers."""


class Gate3Refused(PermissionError):
    """More than one CPU-hour without the owner's approved figure or well above it, or more fans
    than were estimated (SPEC section 17, gate 3)."""


@dataclass(frozen=True)
class FanTask:
    """One fan to compute: a source's window of ``decision_hour`` and a seed ``stream``.

    The hour is normalised to an aware UTC ``datetime`` (strings in a
    :func:`~kronos_agent.data.timeutil.parse_utc_hour` form are accepted); the stream must be a
    plain label (it names a seed family and a directory).
    """

    source: str
    decision_hour: datetime
    stream: str = DEFAULT_STREAM

    def __post_init__(self) -> None:
        if self.source not in SOURCE_SYMBOLS:
            raise ValueError(
                f"unknown source {self.source!r}; expected one of {list(SOURCE_SYMBOLS)}"
            )
        object.__setattr__(self, "decision_hour", parse_utc_hour(self.decision_hour))
        try:
            check_stream(self.stream)
        except ForecastStoreError as exc:
            raise ValueError(str(exc)) from None


@dataclass(frozen=True)
class FanSummary:
    """What a study needs of one fan; ``computed`` is False when a stored fan was reused.

    ``decision_hour`` is UTC text (``2024-07-01T00:00Z``), ``relpath`` the fan file relative to
    the data root, ``elapsed_s`` the model time recorded with the fan.
    """

    source: str
    decision_hour: str
    stream: str
    relpath: str
    window_sha256: str
    inputs_sha256: str
    signal: int
    p_up: float
    last_close: float
    mean_close_n: float
    elapsed_s: float
    computed: bool


@dataclass(frozen=True)
class PreparedBatch:
    """The kept tasks, the dropped ones (with their missing ranges) and the pending count.

    Built by :func:`prepare_tasks`, which also records the window of every kept task (in task
    order), the forecast configuration the pending count was computed for and how the fans are
    made (``evaluation_length``, ``checkpoint``, ``period_role``), so that :func:`run_fan_batch`
    forecasts exactly the windows that were estimated, as they were prepared.
    """

    tasks: tuple[FanTask, ...]
    dropped: tuple[dict[str, Any], ...]
    pending: int
    windows: tuple[DecisionWindow, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
    config_id: str = ""
    evaluation_length: int | None = None
    checkpoint: CheckpointRef | None = None
    period_role: str = MODEL_PERIOD_EVALUATION


@dataclass(frozen=True)
class _RegistryModels:
    """What :func:`~kronos_agent.data.forecast_store.forecast_config` reads of loaded models."""

    spec: ModelSpec
    weights_sha256: dict[str, str]
    device: str


# -- configuration ---------------------------------------------------------------------------------


def _check_checkpoint(checkpoint: object) -> CheckpointRef | None:
    if checkpoint is not None and not isinstance(checkpoint, CheckpointRef):
        raise TypeError(
            f"checkpoint must be a CheckpointRef or None, got {type(checkpoint).__name__}"
        )
    return checkpoint


def registered_weights_sha256(
    model: str, *, checkpoint: CheckpointRef | None = None
) -> dict[str, str]:
    """The ``weights_sha256`` that ``load_models(model, checkpoint=checkpoint)`` records, read
    from the registries alone.

    Nothing is loaded or hashed: the parent process uses it to know the configuration of a batch
    before (or without) loading the models; every loader checks it against its loaded models. A
    checkpoint's are its entry's (``RegistryEntry.weights_sha256``); an entry of another model
    is a :class:`~kronos_agent.kronos.weights.WeightsError`.
    """
    if _check_checkpoint(checkpoint) is None:
        return {
            f"{repo.repo_id}/{item.name}": item.sha256
            for repo in repos_for_model(model)
            for item in repo.files
            if item.sha256 is not None
        }
    entry = registry.find_entry(checkpoint)
    if entry.model != model:
        raise WeightsError(
            f"checkpoint {entry.checkpoint_id} is a {entry.model!r} model, not {model!r}"
        )
    return entry.weights_sha256()


def registry_config(
    kronos: KronosParams, *, checkpoint: CheckpointRef | None = None
) -> dict[str, Any]:
    """:func:`~kronos_agent.data.forecast_store.forecast_config` of the registered weights (the
    shipped ones, or the checkpoint's)."""
    models = _RegistryModels(
        spec=spec_for(kronos.model),
        weights_sha256=registered_weights_sha256(kronos.model, checkpoint=checkpoint),
        device=kronos.device,
    )
    return forecast_config(models, kronos)


def _check_role(period_role: object) -> str:
    if not isinstance(period_role, str) or period_role not in MODEL_PERIOD_ROLES:
        raise ValueError(
            f"period_role must be one of {list(MODEL_PERIOD_ROLES)}, got {period_role!r}"
        )
    return period_role


def _check_models_checkpoint(models: LoadedModels, checkpoint: CheckpointRef | None) -> None:
    """Injected or loaded models must be the batch's: the shipped weights or its checkpoint."""
    expected = None if checkpoint is None else checkpoint.checkpoint_id
    actual = getattr(models, "checkpoint", None)
    if actual != expected:
        raise BatchError(
            f"the models are {'the shipped weights' if actual is None else actual}, the batch "
            f"runs {'the shipped weights' if expected is None else expected}"
        )


def max_workers(threads: int) -> int:
    """At most ``os.cpu_count() // threads`` worker processes (at least one)."""
    threads = _positive_int(threads, "threads")
    return max(1, (os.cpu_count() or 1) // threads)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_threads(kronos: KronosParams) -> None:
    info = require_configured()
    if info.threads != kronos.threads:
        raise BatchError(
            f"this process runs {info.threads} threads but params.yaml fixes {kronos.threads}: "
            "fans would not be comparable"
        )


# -- preparation -----------------------------------------------------------------------------------


def _task_list(tasks: Iterable[FanTask]) -> list[FanTask]:
    if isinstance(tasks, (str, bytes, FanTask)):
        raise TypeError("tasks must be an iterable of FanTask")
    task_list = list(tasks)
    seen: set[FanTask] = set()
    for task in task_list:
        if not isinstance(task, FanTask):
            raise TypeError(f"expected FanTask items, got {type(task).__name__}")
        if task in seen:
            raise BatchError(
                f"duplicate task {task.source} {utc_text(to_ms(task.decision_hour))} "
                f"{task.stream}: each fan is computed and stored once"
            )
        seen.add(task)
    return task_list


def _ranges(open_times: Sequence[int]) -> list[list[int]]:
    """Half-open ``[first, end)`` ranges of consecutive hourly opens."""
    ranges: list[list[int]] = []
    for open_ms in open_times:
        if ranges and ranges[-1][1] == open_ms:
            ranges[-1][1] = open_ms + HOUR_MS
        else:
            ranges.append([open_ms, open_ms + HOUR_MS])
    return ranges


def _dropped_record(
    source: str, hour: datetime, max_context: int, present: Iterable[int], reason: str
) -> dict[str, Any]:
    decision_ms = to_ms(hour)
    have = set(present)
    missing = [
        open_ms
        for open_ms in range(decision_ms - max_context * HOUR_MS, decision_ms, HOUR_MS)
        if open_ms not in have
    ]
    ranges = _ranges(missing)
    return {
        "source": source,
        "decision_hour": utc_text(decision_ms),
        "decision_ms": decision_ms,
        "reason": reason,
        "missing_candles": len(missing),
        "missing": ranges,
        "missing_utc": [[utc_text(start), utc_text(end)] for start, end in ranges],
    }


def _build_windows(
    task_list: Sequence[FanTask], kronos: KronosParams, root: Path | None
) -> tuple[dict[tuple[str, datetime], DecisionWindow], dict[tuple[str, datetime], dict[str, Any]]]:
    """Windows of every (source, hour) the tasks need; incomplete ones as dropped records."""
    hours_by_source: dict[str, set[datetime]] = {}
    for task in task_list:
        hours_by_source.setdefault(task.source, set()).add(task.decision_hour)
    span_ms = kronos.max_context * HOUR_MS
    windows: dict[tuple[str, datetime], DecisionWindow] = {}
    dropped: dict[tuple[str, datetime], dict[str, Any]] = {}
    for source in sorted(hours_by_source):
        hours = sorted(hours_by_source[source])
        try:
            store = StoreWindowSource(
                source,
                start_ms=to_ms(hours[0]) - span_ms,
                end_ms=to_ms(hours[-1]),
                root=root,
            )
        except WindowError as exc:  # not a single candle in the range the tasks need
            for hour in hours:
                dropped[(source, hour)] = _dropped_record(
                    source, hour, kronos.max_context, (), str(exc)
                )
            continue
        for hour in hours:
            try:
                windows[(source, hour)] = store.window(
                    hour,
                    horizon=kronos.horizon_n,
                    max_context=kronos.max_context,
                    amount_policy=kronos.amount_policy,
                )
            except WindowGapError as exc:
                decision_ms = to_ms(hour)
                present = [
                    candle.open_ms
                    for candle in store.candles_between(decision_ms - span_ms, decision_ms)
                ]
                dropped[(source, hour)] = _dropped_record(
                    source, hour, kronos.max_context, present, str(exc)
                )
    return windows, dropped


def _load_reusable(
    path: Path, relpath: str, task: FanTask, window: DecisionWindow, expected_config_id: str
) -> StoredFan:
    """The stored fan at ``path`` if it may be reused for ``task``; ForecastStoreError if not."""
    stored = load_fan(path)
    if stored.meta["config_id"] != expected_config_id:
        raise ForecastStoreError(
            f"{relpath} records forecast config {stored.meta['config_id']}, not "
            f"{expected_config_id}; the file is never overwritten"
        )
    check_reusable(stored, window, stream=task.stream)
    return stored


def prepare_tasks(
    tasks: Iterable[FanTask],
    *,
    params: Params,
    root: Path | None = None,
    evaluation_length: int | None = None,
    checkpoint: CheckpointRef | None = None,
    period_role: str = MODEL_PERIOD_EVALUATION,
) -> PreparedBatch:
    """Build every task's window in this process; drop the incomplete ones; count pending fans.

    ``evaluation_length`` makes fans of that many steps (windows built with that horizon, the
    configuration :func:`~kronos_agent.kronos.decision.evaluation_params` gives),
    ``checkpoint`` names the fine-tuned checkpoint that makes them (the shipped weights when
    ``None``) and ``period_role`` the model-period rule
    (:func:`~kronos_agent.data.partition.require_model_period`); all three are recorded in the
    result, the checkpoint with the ``line_sha256`` of the registry entry read here (the one it
    carried, if any, is verified first), so that every later loader of the batch refuses an
    entry changed since. Refused for the whole batch: another runtime thread count
    (:class:`BatchError`), a context plus the governed ``N`` beyond the embargo, an evaluation
    length above 24 or a task whose ``L`` targets reach past the horizon purge, and a decision
    hour outside the model period (``PartitionError``), a sealed hour (``SealedPartitionError``:
    a batch never holds a ``SealedAccess``), duplicate tasks (:class:`BatchError`), an unknown
    checkpoint or one whose entry changed since its line was recorded (``RegistryError``), a
    missing store, and a stored fan that cannot be reused (``ForecastStoreError``). A task whose
    window misses candles is dropped and listed with the missing ``[start, end)`` ranges.
    ``pending`` counts the kept tasks without a stored fan under :func:`registry_config` (the
    configuration the run produces).
    """
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    kronos = params.kronos
    measured = evaluation_params(params, evaluation_length).kronos
    _check_role(period_role)
    _check_checkpoint(checkpoint)
    _require_threads(kronos)
    check_embargo(kronos.max_context, kronos.horizon_n)
    task_list = _task_list(tasks)
    data_end_ms = None
    if checkpoint is not None:
        entry = registry.find_entry(checkpoint)
        data_end_ms = entry.data_end_ms
        # the data end checked here must stay the one the fans are made with: the hash chain
        # cannot protect the registry's last line, the recorded line sha256 does
        checkpoint = replace(checkpoint, line_sha256=entry.line_sha256)
    for task in task_list:
        require_model_period(task.decision_hour, role=period_role, data_end_ms=data_end_ms)
        require_accessible(task.decision_hour)
        if evaluation_length is not None and not sample_fits(
            task.decision_hour, measured.horizon_n, max_context=kronos.max_context
        ):
            decision_ms = to_ms(task.decision_hour)
            raise PartitionError(
                f"the {task.source} sample of {utc_text(decision_ms)} with an evaluation length "
                f"of {measured.horizon_n} reads targets up to "
                f"{utc_text(decision_ms + (measured.horizon_n - 1) * HOUR_MS)}, past the horizon "
                "purge of the partition: the batch is refused (samples are purged, never "
                "shortened)"
            )
    config = registry_config(measured, checkpoint=checkpoint)
    batch_config_id = config_id(config)
    base = Path(root) if root is not None else data_dir()
    windows, dropped_by_key = _build_windows(task_list, measured, root)
    kept: list[FanTask] = []
    kept_windows: list[DecisionWindow] = []
    dropped: list[dict[str, Any]] = []
    pending = 0
    for task in task_list:
        key = (task.source, task.decision_hour)
        if key in dropped_by_key:
            dropped.append({**copy.deepcopy(dropped_by_key[key]), "stream": task.stream})
            continue
        window = windows[key]
        kept.append(task)
        kept_windows.append(window)
        relpath = fan_relpath(task.source, batch_config_id, task.stream, task.decision_hour)
        if (base / relpath).exists():
            _load_reusable(base / relpath, relpath, task, window, batch_config_id)
        else:
            pending += 1
    return PreparedBatch(
        tasks=tuple(kept),
        dropped=tuple(dropped),
        pending=pending,
        windows=tuple(kept_windows),
        config=config,
        config_id=batch_config_id,
        evaluation_length=None if evaluation_length is None else measured.horizon_n,
        checkpoint=checkpoint,
        period_role=period_role,
    )


# -- gate 3 ----------------------------------------------------------------------------------------


def timing_window(params: Params, *, evaluation_length: int | None = None) -> DecisionWindow:
    """A synthetic decision window shaped by ``params`` (context, horizon, ``amount`` policy).

    Candles of :func:`~kronos_agent.kronos.features.synthetic_window` as plain decimal strings,
    opening in the ``max_context`` hours before :data:`TIMING_DECISION_HOUR`; no market data.
    The horizon is ``evaluation_length`` when given, the governed ``N`` otherwise.
    """
    kronos = params.kronos
    horizon = evaluation_params(params, evaluation_length).kronos.horizon_n
    frame, _, _ = synthetic_window(kronos.max_context)
    decision_ms = to_ms(TIMING_DECISION_HOUR)
    opens = [
        decision_ms - (kronos.max_context - index) * HOUR_MS for index in range(kronos.max_context)
    ]
    candles = [
        Candle(open_ms, f"{o:.2f}", f"{h:.2f}", f"{lo:.2f}", f"{c:.2f}", f"{v:.4f}")
        for open_ms, o, h, lo, c, v in zip(
            opens,
            frame["open"],
            frame["high"],
            frame["low"],
            frame["close"],
            frame["volume"],
            strict=True,
        )
    ]
    quotes = None
    if kronos.amount_policy == "binance_quote_volume":
        quotes = {
            open_ms: f"{amount:.4f}" for open_ms, amount in zip(opens, frame["amount"], strict=True)
        }
    return build_window(
        candles,
        TIMING_DECISION_HOUR,
        source=TIMING_SOURCE,
        horizon=horizon,
        max_context=kronos.max_context,
        amount_policy=kronos.amount_policy,
        quote_volumes=quotes,
    )


def _load_models_quietly(model: str, *, checkpoint: CheckpointRef | None = None) -> LoadedModels:
    """``load_models`` with the upstream Hub mixin's stdout line captured (nothing is printed)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return load_models(model, checkpoint=checkpoint)


def measure_seconds_per_fan(
    *,
    params: Params,
    root: Path | None = None,
    models: LoadedModels | None = None,
    evaluation_length: int | None = None,
    checkpoint: CheckpointRef | None = None,
    repeats: int = 1,
) -> float:
    """Wall seconds of one fan, the median of ``repeats`` fans timed after an untimed warm-up.

    Every fan goes through :func:`~kronos_agent.kronos.decision.forecast_window` on
    :func:`timing_window` (streams ``timing-warmup`` and ``timing``, the ``"timing"`` model-period
    role) in this process, which must run the governed thread count, with ``evaluation_length``
    steps (the governed ``N`` when ``None``). ``models`` defaults to
    ``load_models(params.kronos.model, checkpoint=checkpoint)``; with a ``checkpoint``, injected
    models must be that checkpoint's. ``root`` is accepted for the uniform signature of the batch
    entry points: the synthetic window reads nothing under it.
    """
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    repeats = _positive_int(repeats, "repeats")
    _check_checkpoint(checkpoint)
    _require_threads(params.kronos)
    window = timing_window(params, evaluation_length=evaluation_length)
    if models is None:
        models = _load_models_quietly(params.kronos.model, checkpoint=checkpoint)
    elif checkpoint is not None:
        _check_models_checkpoint(models, checkpoint)
    fan = functools.partial(
        forecast_window,
        models,
        window,
        params,
        evaluation_length=evaluation_length,
        period_role=MODEL_PERIOD_TIMING,
    )
    fan(stream=TIMING_WARMUP_STREAM)
    timed: list[float] = []
    for _ in range(repeats):
        started = _clock()
        fan(stream=TIMING_STREAM)
        seconds = _clock() - started
        if not math.isfinite(seconds) or seconds <= 0:
            raise BatchError(f"the timed fan took {seconds!r} s: the clock is not usable")
        timed.append(float(seconds))
    return float(statistics.median(timed))


def estimate(
    prepared: PreparedBatch, *, seconds_per_fan: float, threads: int, workers: int
) -> dict[str, Any]:
    """CPU and wall seconds of the pending fans of ``prepared`` (fans already stored are free).

    ``cpu_seconds = pending x seconds_per_fan x threads`` and
    ``wall_seconds = cpu_seconds / (threads x workers)``, with ``workers`` capped as
    :func:`run_fan_batch` caps it (:func:`max_workers`).
    """
    if not isinstance(prepared, PreparedBatch):
        raise TypeError(f"expected a PreparedBatch, got {type(prepared).__name__}")
    if (
        isinstance(seconds_per_fan, bool)
        or not isinstance(seconds_per_fan, (int, float))
        or not math.isfinite(seconds_per_fan)
        or seconds_per_fan <= 0
    ):
        raise ValueError(f"seconds_per_fan must be a finite number > 0, got {seconds_per_fan!r}")
    threads = _positive_int(threads, "threads")
    requested = _positive_int(workers, "workers")
    pending = prepared.pending
    if (
        isinstance(pending, bool)
        or not isinstance(pending, int)
        or not (0 <= pending <= len(prepared.tasks))
    ):
        raise ValueError(
            f"pending must be an integer in [0, {len(prepared.tasks)}], got {pending!r}"
        )
    count = min(requested, max_workers(threads))
    cpu_seconds = pending * float(seconds_per_fan) * threads
    return {
        "tasks": len(prepared.tasks),
        "dropped": len(prepared.dropped),
        "stored": len(prepared.tasks) - pending,
        "pending": pending,
        "seconds_per_fan": float(seconds_per_fan),
        "threads": threads,
        "workers_requested": requested,
        "workers": count,
        "cpu_seconds": cpu_seconds,
        "cpu_hours": cpu_seconds / 3600.0,
        "wall_seconds": cpu_seconds / (threads * count),
        "gate3_cpu_seconds": GATE3_CPU_SECONDS,
        "needs_owner_approval": cpu_seconds > GATE3_CPU_SECONDS,
        "config_id": prepared.config_id,
    }


def require_gate3(estimate: Mapping[str, Any], *, approved_cpu_hours: float | None) -> float | None:
    """Gate 3 for a measured estimate; returns measured / approved CPU-hours, or ``None``.

    Up to :data:`GATE3_CPU_SECONDS` CPU-seconds (exactly 3600 included) the batch needs no
    approval: ``None`` is returned and ``approved_cpu_hours`` is not compared. Above it the batch
    is refused (:class:`Gate3Refused`) unless ``approved_cpu_hours``, the CPU-hours the owner
    approved in chat for the estimate shown, is given and the measured CPU-hours are at most
    :data:`GATE3_APPROVAL_TOLERANCE` times it (exactly the tolerance passes); the ratio is then
    returned. A malformed estimate (missing, non-finite or negative ``cpu_seconds``) or approved
    figure (not a finite number > 0) is a ``ValueError``.
    """
    if not isinstance(estimate, Mapping):
        raise TypeError(f"estimate must be a mapping, got {type(estimate).__name__}")
    cpu_seconds = estimate.get("cpu_seconds")
    if (
        isinstance(cpu_seconds, bool)
        or not isinstance(cpu_seconds, (int, float))
        or not math.isfinite(cpu_seconds)
        or cpu_seconds < 0
    ):
        raise ValueError(f"estimate cpu_seconds must be a finite number >= 0, got {cpu_seconds!r}")
    if approved_cpu_hours is not None and (
        isinstance(approved_cpu_hours, bool)
        or not isinstance(approved_cpu_hours, (int, float))
        or not math.isfinite(approved_cpu_hours)
        or approved_cpu_hours <= 0
    ):
        raise ValueError(
            f"the approved CPU-hours must be a finite number > 0, got {approved_cpu_hours!r}"
        )
    if cpu_seconds <= GATE3_CPU_SECONDS:
        return None
    measured = cpu_seconds / 3600.0
    if approved_cpu_hours is None:
        raise Gate3Refused(
            f"the batch needs about {cpu_seconds:.0f} CPU-seconds ({measured:.2f} CPU-hours), "
            f"above the {GATE3_CPU_SECONDS} of gate 3 (SPEC section 17): show the estimate to "
            "the owner and rerun with --approved-by-owner CPU_HOURS, the figure they approve"
        )
    ratio = measured / float(approved_cpu_hours)
    if ratio > GATE3_APPROVAL_TOLERANCE:
        raise Gate3Refused(
            f"gate 3: the measured {measured:.2f} CPU-hours ({cpu_seconds:.0f} CPU-seconds) are "
            f"{ratio:.2f} times the {approved_cpu_hours:g} CPU-hours the owner approved, more "
            f"than {GATE3_APPROVAL_TOLERANCE:g} times (D-040): show the new estimate to the owner "
            "before any fan runs"
        )
    return ratio


# -- running ---------------------------------------------------------------------------------------


def _physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _require_ram_for_workers() -> None:
    total = _physical_memory_bytes()
    if total is None or total < MIN_RAM_BYTES_FOR_WORKERS:
        shown = "unknown" if total is None else f"{total / 2**30:.1f} GiB"
        raise BatchError(
            f"more than one worker needs at least {MIN_RAM_BYTES_FOR_WORKERS / 2**30:.0f} GiB of "
            f"RAM (each worker loads its own models); this machine has {shown}: use one worker"
        )


def _check_parent_prerequisites(model: str, checkpoint: CheckpointRef | None = None) -> None:
    """Fail in the parent, with a readable error, before any worker is spawned: the weights a
    worker loads (the shipped model and tokenizer, or the tokenizer and the checkpoint's files)
    and the vendored tree."""
    if checkpoint is None:
        weights = verify_model(model)
        label = repr(model)
    else:
        weights = verify_repo(repos_for_model(model)[1])
        weights.problems.extend(registry.checkpoint_problems(checkpoint))
        label = f"checkpoint {checkpoint.checkpoint_id}"
    if not weights.ok:
        raise WeightsError(
            f"weights for {label} not verified: {ManifestMismatch(weights.problems)}"
        )
    tree = verify_vendored_tree(strict=True)
    if not tree.ok:
        raise VendoredTreeError(str(ManifestMismatch(tree.problems)))


def _check_models_config(models: LoadedModels, kronos: KronosParams, expected: str) -> None:
    actual = config_id(forecast_config(models, kronos))
    if actual != expected:
        raise BatchError(
            f"the loaded models give forecast config {actual}, the batch was prepared for "
            f"{expected}"
        )


def _init_worker(
    params: Params,
    expected_config_id: str,
    evaluation_length: int | None = None,
    checkpoint: CheckpointRef | None = None,
    period_role: str = MODEL_PERIOD_EVALUATION,
) -> None:
    """Start-up of a spawn-started worker: runtime, vendored import (D-014), models loaded once
    (the shipped weights or ``checkpoint``) and checked against the batch's configuration; the
    evaluation length and model-period role are kept for every fan of the worker."""
    configure_runtime(params.kronos.threads)
    import_kronos()
    models = _load_models_quietly(params.kronos.model, checkpoint=checkpoint)
    _check_models_config(
        models, evaluation_params(params, evaluation_length).kronos, expected_config_id
    )
    _WORKER.clear()
    _WORKER.update(
        params=params,
        models=models,
        evaluation_length=evaluation_length,
        period_role=period_role,
    )


def _worker_forecast(window: DecisionWindow, stream: str) -> Fan:
    if not _WORKER:
        raise BatchError("worker used before _init_worker ran")
    return forecast_window(
        _WORKER["models"],
        window,
        _WORKER["params"],
        stream=stream,
        evaluation_length=_WORKER["evaluation_length"],
        period_role=_WORKER["period_role"],
    )


def _summary(
    task: FanTask,
    window: DecisionWindow,
    paths: np.ndarray,
    elapsed_s: float,
    relpath: str,
    *,
    computed: bool,
) -> FanSummary:
    return FanSummary(
        source=task.source,
        decision_hour=utc_text(to_ms(task.decision_hour)),
        stream=task.stream,
        relpath=relpath,
        window_sha256=window.sha256,
        inputs_sha256=window.inputs_sha256,
        signal=fan_signal(paths, window.last_close),
        p_up=fan_p_up(paths, window.last_close),
        last_close=window.last_close,
        mean_close_n=fan_mean_close(paths),
        elapsed_s=float(elapsed_s),
        computed=computed,
    )


class _Collector:
    """Results in task order, stored fans checked and new fans saved, progress reported."""

    def __init__(
        self,
        prepared: PreparedBatch,
        *,
        params: Params,
        config: Mapping[str, Any],
        batch_config_id: str,
        base: Path,
        progress: Progress | None,
    ) -> None:
        self.prepared = prepared
        self.params = params
        self.config = config
        self.config_id = batch_config_id
        self.base = base
        self.progress = progress
        self.results: list[FanSummary | None] = [None] * len(prepared.tasks)
        self.done = 0

    def relpath(self, index: int) -> str:
        task = self.prepared.tasks[index]
        return fan_relpath(task.source, self.config_id, task.stream, task.decision_hour)

    def _record(self, index: int, summary: FanSummary) -> None:
        self.results[index] = summary
        self.done += 1
        if self.progress is not None:
            self.progress(self.done, len(self.results), summary)

    def classify(self) -> tuple[list[int], list[tuple[int, FanSummary]]]:
        """Indexes left to compute, and the summaries of the stored fans that may be reused
        (each checked with check_reusable and its config id; nothing is recorded yet)."""
        pending: list[int] = []
        reused: list[tuple[int, FanSummary]] = []
        for index, (task, window) in enumerate(
            zip(self.prepared.tasks, self.prepared.windows, strict=True)
        ):
            relpath = self.relpath(index)
            path = self.base / relpath
            if not path.exists():
                pending.append(index)
                continue
            stored = _load_reusable(path, relpath, task, window, self.config_id)
            elapsed = stored.meta["elapsed_s"]
            reused.append(
                (index, _summary(task, window, stored.paths, elapsed, relpath, computed=False))
            )
        return pending, reused

    def reuse(self, reused: Sequence[tuple[int, FanSummary]]) -> None:
        for index, summary in reused:
            self._record(index, summary)

    def store(self, index: int, fan: Fan) -> None:
        task = self.prepared.tasks[index]
        window = self.prepared.windows[index]
        relpath = self.relpath(index)
        save_fan(
            self.base / relpath,
            fan,
            window,
            stream=task.stream,
            config=self.config,
            params_version=self.params.version,
        )
        self._record(
            index, _summary(task, window, fan.paths, fan.elapsed_s, relpath, computed=True)
        )

    def summaries(self) -> list[FanSummary]:
        missing = [index for index, summary in enumerate(self.results) if summary is None]
        if missing:
            raise BatchError(f"{len(missing)} tasks of the batch have no fan")
        return [summary for summary in self.results if summary is not None]


def _run_in_process(
    collector: _Collector,
    pending: Sequence[int],
    *,
    params: Params,
    models: LoadedModels | None,
) -> None:
    prepared = collector.prepared
    if models is None:
        models = _load_models_quietly(params.kronos.model, checkpoint=prepared.checkpoint)
        _check_models_config(
            models,
            evaluation_params(params, prepared.evaluation_length).kronos,
            collector.config_id,
        )
    for index in pending:
        task = prepared.tasks[index]
        window = prepared.windows[index]
        fan = forecast_window(
            models,
            window,
            params,
            stream=task.stream,
            evaluation_length=prepared.evaluation_length,
            period_role=prepared.period_role,
        )
        collector.store(index, fan)


def _run_in_workers(
    collector: _Collector, pending: Sequence[int], *, params: Params, workers: int
) -> None:
    prepared = collector.prepared
    _check_parent_prerequisites(params.kronos.model, prepared.checkpoint)
    executor = ProcessPoolExecutor(
        max_workers=min(workers, len(pending)),
        mp_context=multiprocessing.get_context(START_METHOD),
        initializer=_init_worker,
        initargs=(
            params,
            collector.config_id,
            prepared.evaluation_length,
            prepared.checkpoint,
            prepared.period_role,
        ),
    )
    try:
        futures: dict[Future[Fan], int] = {
            executor.submit(
                _worker_forecast,
                collector.prepared.windows[index],
                collector.prepared.tasks[index].stream,
            ): index
            for index in pending
        }
        for future in as_completed(futures):
            collector.store(futures[future], future.result())
    except BrokenProcessPool as exc:
        raise BatchError(
            f"a worker process failed to start or died ({exc}); its traceback is on stderr"
        ) from exc
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def run_fan_batch(
    prepared: PreparedBatch,
    *,
    workers: int = 1,
    params: Params | None = None,
    root: Path | None = None,
    models: LoadedModels | None = None,
    progress: Progress | None = None,
) -> list[FanSummary]:
    """Every fan of ``prepared``, stored under the data root; one summary per task, in order.

    The fans are made as ``prepared`` says (its evaluation length, checkpoint and model-period
    role), never as this call's arguments would. Stored fans are checked
    (:func:`~kronos_agent.data.forecast_store.check_reusable`, the config id) and loaded, never
    recomputed. The rest are computed in this process when the worker count (capped at
    :func:`max_workers`) is one, otherwise in that many spawn-started workers
    (:class:`BatchError` below 4 GiB of RAM); ``models`` (injected models, tests) runs in this
    process only and must be the batch's weights (the shipped ones or its checkpoint).
    :class:`Gate3Refused` when more fans are pending than ``prepared`` counted (the estimate is
    stale). ``progress(done, total, summary)`` is called after each task, stored ones first,
    then in completion order. ``params`` defaults to ``params.yaml``.
    """
    if not isinstance(prepared, PreparedBatch):
        raise TypeError(f"expected a PreparedBatch, got {type(prepared).__name__}")
    if params is None:
        params = load_params()
    elif not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    kronos = params.kronos
    requested = _positive_int(workers, "workers")
    if len(prepared.windows) != len(prepared.tasks) or not prepared.config_id:
        raise BatchError("run a batch built by prepare_tasks: it carries the windows and config")
    for task, window in zip(prepared.tasks, prepared.windows, strict=True):
        if window.source != task.source or window.decision_hour != task.decision_hour:
            raise BatchError(f"the prepared window of {task} belongs to another task")
    _check_role(prepared.period_role)
    _check_checkpoint(prepared.checkpoint)
    measured = evaluation_params(params, prepared.evaluation_length).kronos
    _require_threads(kronos)
    count = min(requested, max_workers(kronos.threads))
    if models is not None and count > 1:
        raise BatchError("injected models run in this process only: pass workers=1")
    if models is not None:
        _check_models_checkpoint(models, prepared.checkpoint)
    if count > 1:
        _require_ram_for_workers()
    if models is not None:
        config = forecast_config(models, measured)
    else:
        config = registry_config(measured, checkpoint=prepared.checkpoint)
    batch_config_id = config_id(config)
    if batch_config_id != prepared.config_id:
        raise BatchError(
            f"the batch was prepared for forecast config {prepared.config_id}, this run would "
            f"produce {batch_config_id}: prepare it again"
        )
    collector = _Collector(
        prepared,
        params=params,
        config=config,
        batch_config_id=batch_config_id,
        base=Path(root) if root is not None else data_dir(),
        progress=progress,
    )
    pending, reused = collector.classify()
    if len(pending) > prepared.pending:
        raise Gate3Refused(
            f"{len(pending)} fans are pending but the estimate counted {prepared.pending}: "
            "prepare and estimate the batch again"
        )
    collector.reuse(reused)
    if pending:
        if count == 1:
            _run_in_process(collector, pending, params=params, models=models)
        else:
            _run_in_workers(collector, pending, params=params, workers=count)
    return collector.summaries()


# -- measurement-step helpers (docs/RETRAINING.md, section (a): M6, M7 and M9) ---------------------


def _init_timing_worker(
    params: Params,
    expected_config_id: str,
    evaluation_length: int | None,
    checkpoint: CheckpointRef | None,
    window: DecisionWindow,
    barrier: Any,
) -> None:
    """A batch worker of :func:`time_fan_workers`: initialised as every batch worker (role
    ``"timing"``), one warm-up fan, then the start barrier it shares with the parent."""
    _init_worker(params, expected_config_id, evaluation_length, checkpoint, MODEL_PERIOD_TIMING)
    _worker_forecast(window, TIMING_WARMUP_STREAM)
    barrier.wait(timeout=TIMING_STARTUP_TIMEOUT_S)


def _await_workers(barrier: Any, count: int, futures: Sequence[Future[Fan]]) -> None:
    """Return when the ``count`` workers wait at ``barrier`` (it is then released for all).

    A fan that ends before every worker is ready means a worker failed to start (its pool is
    broken): the barrier is aborted and the failure raised; so is a start-up longer than
    :data:`TIMING_STARTUP_TIMEOUT_S`.
    """
    deadline = time.monotonic() + TIMING_STARTUP_TIMEOUT_S
    while barrier.n_waiting < count:
        done = [future for future in futures if future.done()]
        if done:
            barrier.abort()
            done[0].result()  # the worker's failure (a broken pool) is raised here
            raise BatchError("a timing fan ended before every worker had started")
        if time.monotonic() > deadline:
            barrier.abort()
            raise BatchError(
                f"the timing workers did not start within {TIMING_STARTUP_TIMEOUT_S:.0f} s"
            )
        time.sleep(_STARTUP_POLL_S)
    barrier.wait(timeout=TIMING_STARTUP_TIMEOUT_S)


def time_fan_workers(
    params: Params,
    *,
    workers: int,
    fans: int,
    evaluation_length: int | None = None,
    checkpoint: CheckpointRef | None = None,
) -> dict[str, Any]:
    """Wall time of ``fans`` synthetic fans spread over spawn-started batch workers (M6, M7).

    ``workers`` is capped at :func:`max_workers` (the returned ``workers`` is the count that
    ran); more than one needs 4 GiB of RAM (:class:`BatchError`); ``fans`` must be at least that
    count. Each worker starts as a batch worker of ``checkpoint`` (the shipped weights for
    ``None``) with the ``"timing"`` model-period role and runs one warm-up fan
    (``timing-warmup``); once every worker is ready the clock starts and the ``fans`` fans of
    :func:`timing_window` with ``evaluation_length`` steps (streams ``timing-w0``,
    ``timing-w1``, ...) are spread over them as a batch spreads its tasks. Nothing is stored.
    ``startup_seconds`` is the time until every worker was ready, ``wall_seconds`` the timed
    part; ``seconds_per_fan = wall x workers / fans``, ``cpu_seconds = wall x workers x
    threads``.
    """
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    requested = _positive_int(workers, "workers")
    fans = _positive_int(fans, "fans")
    _check_checkpoint(checkpoint)
    kronos = params.kronos
    _require_threads(kronos)
    count = min(requested, max_workers(kronos.threads))
    if fans < count:
        raise ValueError(f"fans ({fans}) must be at least the number of workers ({count})")
    if count > 1:
        _require_ram_for_workers()
    window = timing_window(params, evaluation_length=evaluation_length)
    measured = evaluation_params(params, evaluation_length).kronos
    expected = config_id(registry_config(measured, checkpoint=checkpoint))
    _check_parent_prerequisites(kronos.model, checkpoint)
    context = multiprocessing.get_context(START_METHOD)
    barrier = context.Barrier(count + 1)
    started = _clock()
    executor = ProcessPoolExecutor(
        max_workers=count,
        mp_context=context,
        initializer=_init_timing_worker,
        initargs=(params, expected, evaluation_length, checkpoint, window, barrier),
    )
    try:
        # every submission spawns a worker until there are `count` (none is idle before the
        # barrier), so all of them start, load the models and run their warm-up
        futures = [
            executor.submit(_worker_forecast, window, TIMING_WORKER_STREAM.format(index=index))
            for index in range(fans)
        ]
        _await_workers(barrier, count, futures)
        ready = _clock()
        for future in as_completed(futures):
            future.result()
        wall = _clock() - ready
    except BrokenProcessPool as exc:
        raise BatchError(
            f"a timing worker failed to start or died ({exc}); its traceback is on stderr"
        ) from exc
    finally:
        barrier.abort()  # releases a worker still waiting after a failure; no effect otherwise
        executor.shutdown(wait=True, cancel_futures=True)
    if not math.isfinite(wall) or wall <= 0:
        raise BatchError(f"the timed fans took {wall!r} s: the clock is not usable")
    return {
        "workers": count,
        "fans": fans,
        "evaluation_length": measured.horizon_n,
        "startup_seconds": float(ready - started),
        "wall_seconds": float(wall),
        "seconds_per_fan": wall * count / fans,
        "fans_per_second": fans / wall,
        "cpu_seconds": wall * count * kronos.threads,
    }


def _identity_payload(params: Params) -> dict[str, Any]:
    """The registry payload (kind ``"identity"``) of the shipped weights saved again."""
    kronos = params.kronos
    spec = spec_for(kronos.model)
    model_repo, tokenizer_repo = repos_for_model(kronos.model)
    runtime = require_configured()

    def repository(repo: Any) -> dict[str, Any]:
        return {"repo_id": repo.repo_id, "sha256": {item.name: item.sha256 for item in repo.files}}

    return {
        "kind": "identity",
        "checkpoint_id": IDENTITY_CHECKPOINT_ID,
        "model": kronos.model,
        "base": repository(model_repo),
        "tokenizer": repository(tokenizer_repo),
        "expected_config": dict(model_repo.expected_config or {}),
        "state_dict_numel": spec.state_dict_numel,
        "data_end_ms": KRONOS_FIRST_DECISION_MS,
        "data_end": utc_text(KRONOS_FIRST_DECISION_MS),
        **{key: None for key in (*registry.FOLD_ONLY_KEYS, *registry.FOLD_OPTIONAL_KEYS)},
        "device": kronos.device,
        "threads": kronos.threads,
        "deterministic": runtime.deterministic,
        "activation_checkpointing": False,  # nothing is trained
        "torch_version": runtime.torch_version,
        "platform": default_label(),
        "code": code_version(),
        "vendored_manifest_sha256": sha256_file(source_manifest_path()),
    }


def _fan_in_one_worker(
    params: Params,
    expected_config_id: str,
    checkpoint: CheckpointRef,
    window: DecisionWindow,
    stream: str,
) -> Fan:
    """One fan of ``window`` by a spawn-started batch worker of ``checkpoint`` (role timing)."""
    _check_parent_prerequisites(params.kronos.model, checkpoint)
    executor = ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context(START_METHOD),
        initializer=_init_worker,
        initargs=(params, expected_config_id, None, checkpoint, MODEL_PERIOD_TIMING),
    )
    try:
        return executor.submit(_worker_forecast, window, stream).result()
    except BrokenProcessPool as exc:
        raise BatchError(
            f"the worker failed to start or died ({exc}); its traceback is on stderr"
        ) from exc
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def identity_control(
    params: Params, *, scratch_root: Path, models: LoadedModels | None = None
) -> dict[str, Any]:
    """The shipped weights through the whole checkpoint path; their fan must not change (M9).

    ``scratch_root`` must not exist yet. The shipped models (``models``, or loaded) are saved
    with ``save_pretrained`` into ``<scratch_root>/source``, registered as
    :data:`IDENTITY_CHECKPOINT_ID` (kind ``"identity"``, data end July 2024) in the scratch
    registry ``<scratch_root>/registry`` after
    :func:`~kronos_agent.kronos.weights.checkpoint_dir_problems`, and loaded back with
    ``load_models(checkpoint=...)`` in one spawn-started batch worker. The fan of
    :func:`timing_window` (stream ``timing``, role ``"timing"``) of the shipped models in this
    process and the one of the checkpoint in the worker must be identical (``identical``: same
    dtype, shape and bytes), while the two forecast configurations differ (their weight keys do;
    equal ids are a :class:`BatchError`).
    """
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    started = _clock()
    kronos = params.kronos
    _require_threads(kronos)
    scratch = Path(scratch_root)
    if scratch.exists() or scratch.is_symlink():
        raise BatchError(
            f"the identity control needs a new scratch directory; {scratch.name} exists"
        )
    if models is None:
        models = _load_models_quietly(kronos.model)
    else:
        _check_models_checkpoint(models, None)
    ensure_directory(scratch)
    source = scratch / "source"
    with contextlib.redirect_stdout(io.StringIO()):
        models.model.save_pretrained(source)
    entry = registry.install_checkpoint(
        source,
        payload=_identity_payload(params),
        check=functools.partial(checkpoint_dir_problems, name=kronos.model),
        root=scratch / "registry",
    )
    window = timing_window(params)
    shipped = forecast_window(
        models, window, params, stream=TIMING_STREAM, period_role=MODEL_PERIOD_TIMING
    )
    shipped_config_id = config_id(forecast_config(models, kronos))
    reference = CheckpointRef(
        entry.checkpoint_id, root=scratch / "registry", line_sha256=entry.line_sha256
    )
    checkpoint_config_id = config_id(registry_config(kronos, checkpoint=reference))
    if checkpoint_config_id == shipped_config_id:
        raise BatchError(
            f"the checkpoint and the shipped weights share forecast config {shipped_config_id}: "
            "their weight keys must differ"
        )
    loaded = _fan_in_one_worker(params, checkpoint_config_id, reference, window, TIMING_STREAM)
    identical = (
        shipped.paths.dtype == loaded.paths.dtype
        and shipped.paths.shape == loaded.paths.shape
        and shipped.paths.tobytes() == loaded.paths.tobytes()
    )
    return {
        "identical": bool(identical),
        "checkpoint_id": entry.checkpoint_id,
        "shipped_config_id": shipped_config_id,
        "checkpoint_config_id": checkpoint_config_id,
        "paths_sha256": {
            "shipped": paths_sha256(shipped.paths),
            "checkpoint": paths_sha256(loaded.paths),
        },
        "files": [
            {"name": item.name, "size": item.size, "sha256": item.sha256} for item in entry.files
        ],
        "seconds": float(_clock() - started),
    }


__all__ = [
    "GATE3_APPROVAL_TOLERANCE",
    "GATE3_CPU_SECONDS",
    "IDENTITY_CHECKPOINT_ID",
    "MIN_RAM_BYTES_FOR_WORKERS",
    "START_METHOD",
    "TIMING_DECISION_HOUR",
    "TIMING_STARTUP_TIMEOUT_S",
    "TIMING_STREAM",
    "TIMING_WARMUP_STREAM",
    "TIMING_WORKER_STREAM",
    "BatchError",
    "FanSummary",
    "FanTask",
    "Gate3Refused",
    "PreparedBatch",
    "estimate",
    "identity_control",
    "max_workers",
    "measure_seconds_per_fan",
    "prepare_tasks",
    "registered_weights_sha256",
    "registry_config",
    "require_gate3",
    "run_fan_batch",
    "time_fan_workers",
    "timing_window",
]
