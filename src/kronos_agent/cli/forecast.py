"""``kronos-agent forecast``: the Kronos fan of one decision hour, or a batch of hours (gate 3).

One hour: ``forecast --source binance|bingx --hour D [--save]`` builds the window of ``D`` from
``candles/<source>/1h`` through the study read path (a sealed hour is refused), runs one fan
with the governed parameters of ``params.yaml`` in this process and prints its signal (the sign
of ``mean_k close[k, N] / close[D - 1h] - 1``) and ``p_up``. ``--save`` first saves the window's
inputs under ``windows/<source>/`` (an identical saved window is kept, a different one refused
before any model is loaded), then stores the fan under
``forecasts/<source>/<config_id>/forecast/``; a fan already stored there is checked and reused,
never recomputed or overwritten.

A batch: ``forecast --source S --from D1 --to D2 [--step-hours H] [--workers W] [--estimate]
[--approved-by-owner CPU_HOURS]`` covers the decision hours ``D1, D1 + H, ...`` before ``D2``
(half-open). :func:`run_batch` builds every window first (incomplete ones are dropped and
listed), measures the seconds per fan after a warm-up, prints the gate-3 estimate (``cpu_seconds
= pending fans x seconds per fan x threads``; ``wall_seconds = cpu_seconds / (threads x
workers)``) and stops there with ``--estimate``. Above 3600 CPU-seconds the batch is refused
(exit 2) unless ``--approved-by-owner`` gives the CPU-hours the owner approved in chat for the
estimate shown (SPEC section 17, gate 3) and the measured CPU-hours are at most
:data:`~kronos_agent.kronos.batch.GATE3_APPROVAL_TOLERANCE` times that figure (D-040); a run
within it prints both figures, their ratio and the command's start, the time the flag was given.
Fans are stored as they are computed; the ``mismatch`` command runs its Kronos part through the
same :func:`run_batch`.

:func:`run` configures the runtime (the governed thread count) before anything imports torch.
The weights are loaded by :func:`_load_models`, which tests replace.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..data.partition import (
    PartitionError,
    SealedPartitionError,
    check_embargo,
    require_accessible,
    require_kronos_period,
)
from ..data.timeutil import HOUR_MS, hour_range, to_ms, utc_text
from ..params import Params, ParamsError, load_params
from . import (
    EXIT_FAILURE,
    EXIT_OK,
    ExceptionTypes,
    data_root,
    positive_int,
    positive_number,
    run_guarded,
    utc_hour,
)

if TYPE_CHECKING:
    from ..data.windows import DecisionWindow
    from ..kronos.batch import FanSummary, FanTask, PreparedBatch
    from ..kronos.weights import LoadedModels

SOURCES: tuple[str, ...] = ("binance", "bingx")
MAX_LISTED = 10


@dataclass(frozen=True)
class BatchRun:
    """A prepared batch, its gate-3 estimate and its summaries (``None`` with ``--estimate``).

    ``approved_cpu_hours`` is the figure given with ``--approved-by-owner`` (``None`` without
    it) and ``ratio`` the measured CPU-hours over it when the batch needed the approval
    (``None`` otherwise); ``started_at`` is the command's start, when the flag was given.
    """

    prepared: PreparedBatch
    estimate: dict[str, Any]
    summaries: tuple[FanSummary, ...] | None
    approved_cpu_hours: float | None = None
    ratio: float | None = None
    started_at: datetime | None = None


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "forecast", help="Kronos fan of one decision hour, or a gate-3 batch of decision hours"
    )
    parser.add_argument("--source", choices=SOURCES, required=True)
    when = parser.add_mutually_exclusive_group(required=True)
    when.add_argument(
        "--hour", type=utc_hour, metavar="D", help="one decision hour (YYYY-MM-DDTHH:00Z)"
    )
    when.add_argument(
        "--from",
        dest="start",
        type=utc_hour,
        metavar="D1",
        help="first decision hour of a batch (inclusive)",
    )
    parser.add_argument(
        "--to", dest="end", type=utc_hour, metavar="D2", help="end of the batch (exclusive)"
    )
    parser.add_argument(
        "--step-hours",
        type=positive_int,
        default=None,
        metavar="H",
        help="hours between the batch's decision hours (default 1)",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        metavar="N",
        help="spawn-started worker processes (default 1; at most cpu_count // threads; more than "
        "one needs 4 GiB of RAM, each loads its own models)",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="measure and print the gate-3 estimate of the batch, run none of its fans",
    )
    parser.add_argument(
        "--approved-by-owner",
        type=positive_number,
        default=None,
        metavar="CPU_HOURS",
        help="the CPU-hours the owner approved in chat for this batch's estimate (gate 3); a "
        "measured estimate well above it is refused",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="with --hour: save the window's inputs, then store the fan, under the data root",
    )
    parser.set_defaults(usage_error=parser.error)


def _check_usage(args: argparse.Namespace) -> None:
    """Options that belong to the other mode, or a batch without its end, are usage errors."""
    if args.hour is not None:
        batch_only = [
            flag
            for flag, value in (
                ("--to", args.end),
                ("--step-hours", args.step_hours),
                ("--workers", args.workers),
                ("--estimate", args.estimate),
                ("--approved-by-owner", args.approved_by_owner),
            )
            if value is not None and value is not False
        ]
        if batch_only:
            args.usage_error(f"{', '.join(batch_only)} belong to a batch (--from/--to), not --hour")
        return
    if args.end is None:
        args.usage_error("a batch needs --to (the end hour, exclusive)")
    if args.end <= args.start:
        args.usage_error("--to must be later than --from")
    if args.save:
        args.usage_error("--save belongs to --hour: a batch always stores its fans")


def configure_for(params: Params) -> None:
    """Fix the governed thread count before anything imports torch (``kronos_agent.runtime``)."""
    from ..runtime import configure_runtime

    configure_runtime(params.kronos.threads)


def kronos_errors() -> tuple[ExceptionTypes, ExceptionTypes]:
    """``(refusals, failures)`` of the commands that run Kronos fans (imported lazily).

    A store read while a ``data`` run holds its job lock is refused (``StoreBusyError`` is a
    ``LockBusyError``): a busy lock is a refusal, not a failure (D-039).
    """
    from ..data.forecast_store import ForecastStoreError
    from ..data.fsutil import LockBusyError
    from ..data.store import StoreError
    from ..data.windows import WindowError
    from ..kronos.batch import BatchError, Gate3Refused
    from ..kronos.decision import DecisionError
    from ..kronos.vendored import VendoredTreeError
    from ..kronos.weights import WeightsError
    from ..manifests import ManifestMismatch
    from ..runtime import RuntimeConfigError

    refusals: ExceptionTypes = (Gate3Refused, SealedPartitionError, PartitionError, LockBusyError)
    failures: ExceptionTypes = (
        WindowError,
        StoreError,
        WeightsError,
        VendoredTreeError,
        ManifestMismatch,
        DecisionError,
        BatchError,
        ForecastStoreError,
        RuntimeConfigError,
        ParamsError,
    )
    return refusals, failures


def load_and_configure() -> Params | None:
    """``params.yaml`` loaded and the runtime configured with its thread count, before anything
    imports numpy or torch; ``None`` after printing the ``ERROR:`` line."""
    from ..runtime import RuntimeConfigError

    try:
        params = load_params()
        configure_for(params)
    except (ParamsError, RuntimeConfigError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return None
    return params


def now() -> datetime:
    """The UTC clock of the commands that run fans (a module function, so tests replace it)."""
    return datetime.now(UTC)


def run(args: argparse.Namespace) -> int:
    started_at = now()  # the command's start: when a gate-3 approval was given
    _check_usage(args)
    params = load_and_configure()
    if params is None:
        return EXIT_FAILURE
    refusals, failures = kronos_errors()
    if args.hour is not None:
        return run_guarded(lambda: _one_hour(args, params), refusals=refusals, failures=failures)
    return run_guarded(
        lambda: _batch(args, params, started_at), refusals=refusals, failures=failures
    )


# -- models and printing ---------------------------------------------------------------------------


def _load_models(params: Params) -> LoadedModels:
    """The governed model and its tokenizer, verified and loaded on CPU in eval mode.

    The upstream Hub mixin prints a line while loading; it is captured, nothing is printed.
    """
    from ..kronos.weights import load_models

    with contextlib.redirect_stdout(io.StringIO()):
        return load_models(params.kronos.model)


def _say(message: str) -> None:
    print(message, flush=True)


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:MAX_LISTED])
    return shown + (f" and {len(items) - MAX_LISTED} more" if len(items) > MAX_LISTED else "")


def print_dropped(dropped: Sequence[dict[str, Any]], total: int) -> None:
    """The tasks whose window misses candles, with the missing ranges (at most 10 lines)."""
    if not dropped:
        return
    _say(f"dropped {len(dropped)} of {total} tasks: their window misses candles")
    for item in dropped[:MAX_LISTED]:
        ranges = _listed([f"[{start}, {end})" for start, end in item.get("missing_utc") or []])
        _say(
            f"  {item['source']} {item['decision_hour']} {item.get('stream')}: "
            f"{item['missing_candles']} missing candles {ranges}"
        )
    if len(dropped) > MAX_LISTED:
        _say(f"  and {len(dropped) - MAX_LISTED} more")


def print_estimate(estimate: dict[str, Any]) -> None:
    approval = (
        "above the limit: needs the CPU-hours the owner approves (--approved-by-owner CPU_HOURS)"
        if estimate["needs_owner_approval"]
        else "within the limit"
    )
    _say(
        f"gate 3 estimate: {estimate['tasks']} fans ({estimate['stored']} stored, "
        f"{estimate['pending']} pending) x {estimate['seconds_per_fan']:.3f} s per fan x "
        f"{estimate['threads']} threads = {estimate['cpu_seconds']:.0f} CPU-seconds "
        f"({estimate['cpu_hours']:.2f} CPU-hours); about {estimate['wall_seconds']:.0f} s of wall "
        f"time with {estimate['workers']} worker(s); limit {estimate['gate3_cpu_seconds']} "
        f"CPU-seconds: {approval}; forecast config {estimate['config_id']}"
    )


def _progress(done: int, total: int, summary: FanSummary) -> None:
    how = f"computed in {summary.elapsed_s:.2f} s" if summary.computed else "stored fan reused"
    _say(
        f"[{done}/{total}] {summary.source} {summary.decision_hour} {summary.stream}: signal "
        f"{summary.signal:+d}, p_up {summary.p_up:.2f} ({how})"
    )


def _print_window(window: DecisionWindow) -> None:
    first = window.candles[0].open_ms
    _say(
        f"{window.source} window of {utc_text(to_ms(window.decision_hour))}: "
        f"{len(window.candles)} candles from {utc_text(first)}; sha256 {window.sha256}; "
        f"inputs sha256 {window.inputs_sha256}; amount policy {window.amount_policy}"
    )


def _print_signal(*, last_close: float, mean_close: float, signal: int, p_up: float) -> None:
    _say(
        f"close at D - 1h {last_close:.2f}; mean close at N {mean_close:.2f}; signal "
        f"{signal:+d}; p_up {p_up:.3f}"
    )


def _approval_line(run: BatchRun) -> str | None:
    """The line about ``--approved-by-owner`` once gate 3 has passed (``None``: no flag given)."""
    from ..kronos.batch import GATE3_APPROVAL_TOLERANCE

    if run.approved_cpu_hours is None:
        return None
    given = run.started_at.strftime("%Y-%m-%dT%H:%M:%SZ") if run.started_at else "the start"
    if run.ratio is None:
        return (
            f"gate 3: within the limit, no approval needed (--approved-by-owner "
            f"{run.approved_cpu_hours:g} CPU-hours given at {given} is not used)"
        )
    return (
        f"gate 3: --approved-by-owner {run.approved_cpu_hours:g} CPU-hours given at {given}; "
        f"measured {run.estimate['cpu_hours']:.2f} CPU-hours, {run.ratio:.2f} times the approved "
        f"figure (at most {GATE3_APPROVAL_TOLERANCE:g}): the owner's go-ahead covers this estimate"
    )


# -- the batch flow (also used by the mismatch command) --------------------------------------------


def run_batch(
    tasks: Iterable[FanTask],
    *,
    params: Params,
    workers: int,
    estimate_only: bool,
    approved_cpu_hours: float | None,
    started_at: datetime | None = None,
    before_fans: Callable[[BatchRun], None] | None = None,
) -> BatchRun:
    """Prepare, measure, estimate and (unless ``estimate_only``) run a batch behind gate 3.

    The runtime must already run the governed thread count (:func:`configure_for`). The models
    loaded for the timing run the fans in this process when one worker is used; with more
    workers each spawn-started worker loads its own and the parent's are released first.
    ``Gate3Refused`` above :data:`~kronos_agent.kronos.batch.GATE3_CPU_SECONDS` without
    ``approved_cpu_hours`` (the CPU-hours the owner approved) or when the measured CPU-hours
    are more than :data:`~kronos_agent.kronos.batch.GATE3_APPROVAL_TOLERANCE` times it, before
    any fan runs. ``before_fans`` is called once gate 3 has passed and before the first fan,
    only when some fan is pending (the mismatch command records its approval there).
    """
    from ..kronos import batch

    root = data_root()
    task_list = list(tasks)
    prepared = batch.prepare_tasks(task_list, params=params, root=root)
    print_dropped(prepared.dropped, len(task_list))
    models: LoadedModels | None = _load_models(params)
    seconds = batch.measure_seconds_per_fan(params=params, root=root, models=models)
    estimate = batch.estimate(
        prepared, seconds_per_fan=seconds, threads=params.kronos.threads, workers=workers
    )
    print_estimate(estimate)
    if estimate_only:
        return BatchRun(prepared=prepared, estimate=estimate, summaries=None)
    ratio = batch.require_gate3(estimate, approved_cpu_hours=approved_cpu_hours)
    planned = BatchRun(
        prepared=prepared,
        estimate=estimate,
        summaries=None,
        approved_cpu_hours=approved_cpu_hours,
        ratio=ratio,
        started_at=started_at,
    )
    approval = _approval_line(planned)
    if approval is not None:
        _say(approval)
    if before_fans is not None and prepared.pending:
        before_fans(planned)
    if estimate["workers"] > 1:
        models = None  # every worker loads its own; the parent's copy is not needed
    summaries = batch.run_fan_batch(
        prepared, workers=workers, params=params, root=root, models=models, progress=_progress
    )
    computed = sum(summary.computed for summary in summaries)
    _say(
        f"{len(summaries)} fans: {computed} computed, {len(summaries) - computed} stored fans "
        f"reused; under forecasts/<source>/{prepared.config_id}/ of the data root"
    )
    return BatchRun(
        prepared=prepared,
        estimate=estimate,
        summaries=tuple(summaries),
        approved_cpu_hours=approved_cpu_hours,
        ratio=ratio,
        started_at=started_at,
    )


# -- actions ---------------------------------------------------------------------------------------


def _batch(args: argparse.Namespace, params: Params, started_at: datetime) -> int:
    from ..kronos.batch import FanTask

    hours = hour_range(args.start, args.end, args.step_hours or 1)
    _say(
        f"{args.source} batch: {len(hours)} decision hours from {utc_text(to_ms(hours[0]))} "
        f"every {args.step_hours or 1} h before {utc_text(to_ms(args.end))}"
    )
    run_batch(
        [FanTask(args.source, hour) for hour in hours],
        params=params,
        workers=args.workers or 1,
        estimate_only=args.estimate,
        approved_cpu_hours=args.approved_by_owner,
        started_at=started_at,
    )
    return EXIT_OK


def _one_hour(args: argparse.Namespace, params: Params) -> int:
    kronos = params.kronos
    hour = args.hour
    check_embargo(kronos.max_context, kronos.horizon_n)
    require_kronos_period(hour)
    require_accessible(hour)
    if args.save:
        return _one_hour_saved(args, params)
    from ..data.windows import StoreWindowSource
    from ..kronos.decision import fan_mean_close, fan_p_up, fan_signal, forecast_window

    decision_ms = to_ms(hour)
    store = StoreWindowSource(
        args.source,
        start_ms=decision_ms - kronos.max_context * HOUR_MS,
        end_ms=decision_ms,
        root=data_root(),
    )
    window = store.window(
        hour,
        horizon=kronos.horizon_n,
        max_context=kronos.max_context,
        amount_policy=kronos.amount_policy,
    )
    _print_window(window)
    fan = forecast_window(_load_models(params), window, params)
    _say(
        f"seeds {fan.seeds.canonical} (torch {fan.seeds.torch_seed}, numpy "
        f"{fan.seeds.numpy_seed}); model {fan.model_name}, {fan.trajectories} trajectories x "
        f"{fan.horizon} candles, {fan.threads} threads, torch {fan.torch_version}, "
        f"{fan.elapsed_s:.2f} s"
    )
    _print_signal(
        last_close=window.last_close,
        mean_close=fan_mean_close(fan.paths),
        signal=fan_signal(fan.paths, window.last_close),
        p_up=fan_p_up(fan.paths, window.last_close),
    )
    return EXIT_OK


def _one_hour_saved(args: argparse.Namespace, params: Params) -> int:
    """``--save``: the window's inputs are saved (or found identical) before any model is loaded
    or any fan stored, so a different saved window never leaves a fan behind an error."""
    from ..data.forecast_store import load_fan
    from ..data.windows import save_window
    from ..kronos.batch import FanTask, prepare_tasks, run_fan_batch

    root = data_root()
    prepared = prepare_tasks([FanTask(args.source, args.hour)], params=params, root=root)
    if prepared.dropped:
        (dropped,) = prepared.dropped
        print(f"ERROR: {dropped['reason']}")
        return EXIT_FAILURE
    (window,) = prepared.windows
    _print_window(window)
    window_path = save_window(window, root=root)
    models = _load_models(params) if prepared.pending else None
    (summary,) = run_fan_batch(prepared, workers=1, params=params, root=root, models=models)
    seeds = load_fan(root / summary.relpath).meta["seeds"]
    _say(f"seeds {seeds['canonical']} (torch {seeds['torch_seed']}, numpy {seeds['numpy_seed']})")
    _print_signal(
        last_close=summary.last_close,
        mean_close=summary.mean_close_n,
        signal=summary.signal,
        p_up=summary.p_up,
    )
    how = "computed and stored" if summary.computed else "stored fan reused"
    _say(
        f"fan {summary.relpath} ({how}); window {window_path.relative_to(root).as_posix()} "
        "(under the data root)"
    )
    return EXIT_OK
