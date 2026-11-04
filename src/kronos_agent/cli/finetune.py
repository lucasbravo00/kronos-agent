"""``kronos-agent finetune <subcommand>``: retraining Kronos-small for BTC (step 6, SPEC section 9;
``docs/RETRAINING.md``).

- ``timing [--estimate] [--approved-by-owner CPU_HOURS]``: the measurement step of design
  section (a) (phase P0; synthetic data only, 0 tests). It prints its prior estimate first and
  refuses to start (exit 2) above its budget: its own 3,000 CPU-seconds and 45 GPU-minutes,
  below gate 3, or with ``--approved-by-owner`` the CPU-hours the owner approved in chat for this
  measurement (gate 3, D-055): ``CPU_HOURS x 3600`` CPU-seconds, a ceiling with no tolerance
  (the estimate must fit, and the run stops at, that ceiling minus a reserve of 600
  CPU-seconds for the unit running then), the GPU budget unchanged. Then it measures M1 to M9
  (training step time and memory per device, thread count and batch; activation checkpointing;
  the MPS preflight before any MPS timing; sustained MPS throughput; fan speed, worker scaling
  and contention; NLL4-TF time; the identity control of the checkpoint path), stops when the
  measured spend passes its stop limit (the step's own limit, or the approved ceiling minus the
  reserve), and writes ``reports/finetune/timing-2.json`` and ``.md`` with the compute choice
  and the approval (the approved CPU-hours, the command's start as the time the flag was given,
  the estimate; the first run's ``timing.*`` hold another plan and stay as they are: D-054). The
  budget is not part of the recorded plan, so the flag never makes it a changed study.
  ``--estimate`` prints the estimate against the budget and runs nothing. Run it when the
  machine is otherwise idle, on power, lid open, under ``caffeinate -i -s``.
- ``fold --fold LABEL [--replicate K] [--estimate] [--approved-by-owner HOURS] [--order ID]``:
  one fold of the quarterly walk-forward (``kronos_agent.finetune.walkforward``; D-056); the pilot
  is ``--fold q-2024-07`` (training-segment data only, shown, 0 tests). It prints the fold, the
  compute the timing record chose, the stage and every term of the estimate (MPS wall-hours, or
  CPU-hours on the CPU), the refit if the pilot rule fires, and the earlier fit entries of the run
  label. ``--estimate`` stops there. Above one hour it needs ``--approved-by-owner HOURS``, the
  figure the owner approved in chat (gate 3), and refuses an estimate more than 1.25 times it;
  the approval goes into the experiment ledger before the first unit, and nothing starts that
  could take the run's spend past 1.25 times it. Other folds than the pilot train only after the
  pre-registration bundle (design 7.1), recorded in a ledger that may authorize reading the
  data root, blinded; a fold that reads sealed candles needs ``--order`` (gate 6). A training
  child still running on this machine (a parent killed with SIGKILL leaves its child training)
  is a refusal. Run it when the machine is otherwise idle, on power, lid open, under
  ``caffeinate -i -s``, started so that ending the terminal session cannot kill it.
- ``pilot-check [--record-config] [--estimate] [--approved-by-owner CPU_HOURS]``: the pilot
  check of design 6.2 (``kronos_agent.finetune.pilot_check``; 504 fans and the KL, 0 tests) on
  the registered pilot; ``--record-config`` records its definition only (allowed before the
  pilot exists); its estimate is printed first and gate 3 applies as for the batch commands
  (D-046), the approval recorded in the ledger before the first fan; the run stops between fans
  once its spend passes 1.25 times the approved figure, or 3,600 CPU-seconds without one
  (design 3.5; exit 2, the stored fans kept, the stopped run and its spend recorded in the
  ledger); a rerun counts what the stopped runs spent since the check's last done entry, in
  its estimate, its gate 3 and its limit; the reading is GO or NO-GO (both exit 0).
- ``tokenizer-diag --part training|validation [--record-config]``: the tokenizer diagnostic of
  design section 4 (``kronos_agent.finetune.tokenizer_diag``; 0 tests). Part B (``validation``)
  runs only after the pre-registration bundle, recorded in a ledger that may authorize reading
  the data root (``ledger.binding_problem``: a relocated reports root never reads the
  repository's validation inputs); each raised flag is printed as a ``WARNING:``.

Every subcommand but ``timing`` first verifies every pre-registration bundle the experiment
ledger records (design 7.1: a changed bundle is a refusal) and records what it ran in the
ledger (``kronos_agent.ledger``).

Exit codes (D-039): 0 done; 1 failure (``ERROR:``): a measurement that failed (an M9 mismatch, a
crashed training child, a measurement error), reported after the report is written, or a run
that could not start (unverified weights, a broken vendored tree); a fold whose training failed
for good (after its ledger entries); a broken ledger or registry; 2 refusal (``REFUSED:``): the
estimate above the budget (nothing ran), a run the budget stopped (after the report is written),
a changed timing plan, a concurrent timing run; gate 3 without ``--approved-by-owner`` or well
above that figure, the spend cap (a pilot check its spend stopped), a stale timing record, a
missing or changed pre-registration bundle, the layout, a registered fold, the sealed segment (an
order from a ledger that may not open the data root among them), a changed study config, a busy
lock.
argparse usage errors exit 2 as well (a bare ``--approved-by-owner``, or a figure that is not a
finite number > 0, as for the batch commands of D-046; ``--record-config`` with ``--estimate``
or ``--approved-by-owner``); an unexpected exception keeps its traceback.

At load this module imports only what ``kronos_agent.cli`` allows; :func:`run` configures the
runtime with the governed thread count before anything imports torch, then imports the
subcommand's module. A later subcommand adds its ``(register, run)`` pair to :data:`SUBCOMMANDS`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..paths import reports_dir
from . import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_REFUSED,
    ExceptionTypes,
    display_path,
    positive_int,
    positive_number,
    run_guarded,
)

if TYPE_CHECKING:
    from ..params import Params

# reports/finetune/; kronos_agent.finetune.timing.FINETUNE_REPORTS_SUBDIR names the same folder
FINETUNE_REPORTS_SUBDIR = "finetune"
# argparse choices are literals so that --help imports nothing heavy; a test asserts that they
# equal kronos_agent.finetune.tokenizer_diag.PARTS
TOKENIZER_DIAG_PARTS: tuple[str, ...] = ("training", "validation")
PILOT_CHECK_COMMAND = "finetune pilot-check"  # the ledger's gate3_approval command
# the flag of the R1 token mismatch (a key of tokenizer_diag.FLAGS) adds this to its WARNING line
MISMATCH_FLAG = "r1_token_mismatch"
MISMATCH_FLAG_TEXT = "tell the owner before the pre-registration bundle (D-048 (5))"


class _Exclusive(argparse.Action):
    """An option that refuses the options named in ``excludes`` when they came before it on the
    command line; each of them refuses it in turn, so either order is a usage error (exit 2).
    ``flag`` makes it a ``store_true`` switch. (argparse's mutually exclusive groups cannot say
    that ``--record-config`` excludes two options that go together.)"""

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        excludes: tuple[str, ...] = (),
        flag: bool = False,
        **kwargs,
    ) -> None:
        if flag:
            kwargs.update(nargs=0, default=False)
        super().__init__(option_strings, dest, **kwargs)
        self.excludes = tuple(excludes)
        self.flag = flag

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        for option in self.excludes:
            given = getattr(namespace, option.lstrip("-").replace("-", "_"), None)
            if given is not None and given is not False:
                raise argparse.ArgumentError(self, f"not allowed with argument {option}")
        setattr(namespace, self.dest, True if self.flag else values)


def _say(message: str) -> None:
    print(message, flush=True)


def _now() -> datetime:
    """The UTC clock of the command (a module function, so tests replace it)."""
    return datetime.now(UTC)


# -- timing ----------------------------------------------------------------------------------------


def _register_timing(actions) -> None:
    parser = actions.add_parser(
        "timing",
        help="the measurement step (design section (a)): step times, memory, MPS preflight, "
        "fan speed, NLL4-TF time, identity control; writes reports/finetune/timing-2",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="print the prior-based estimate against the budget and run nothing",
    )
    parser.add_argument(
        "--approved-by-owner",
        type=positive_number,
        default=None,
        metavar="CPU_HOURS",
        help="the CPU-hours the owner approved in chat for this measurement (gate 3): the CPU "
        "budget becomes CPU_HOURS x 3600 CPU-seconds instead of 3,000, a ceiling the spend never "
        "passes (the estimate must fit, and the run stops at, 600 CPU-seconds less); the GPU "
        "budget stays 45 minutes",
    )


def _timing_errors() -> tuple[ExceptionTypes, ExceptionTypes]:
    """``(refusals, failures)`` of ``finetune timing`` (imported lazily)."""
    from ..data.fsutil import LockBusyError
    from ..data.partition import PartitionError, SealedPartitionError
    from ..finetune.export_csv import ExportError, ExportRefused
    from ..finetune.folds import FoldError
    from ..finetune.nll import NllError
    from ..finetune.recipe import RecipeError
    from ..finetune.registry import RegistryError
    from ..finetune.timing import StaleTimingError, TimingBudgetExceeded, TimingError
    from ..finetune.train import TrainError
    from ..kronos.batch import BatchError
    from ..kronos.vendored import VendoredTreeError
    from ..kronos.weights import WeightsError
    from ..manifests import ManifestMismatch
    from ..params import ParamsError
    from ..provenance import ProvenanceError
    from ..report.writer import ConfigChangedError, ReportError
    from ..runtime import RuntimeConfigError

    refusals: ExceptionTypes = (
        TimingBudgetExceeded,
        StaleTimingError,
        ConfigChangedError,
        LockBusyError,
        ExportRefused,
        SealedPartitionError,
        PartitionError,
    )
    failures: ExceptionTypes = (
        TimingError,
        ReportError,
        ParamsError,
        RuntimeConfigError,
        WeightsError,
        VendoredTreeError,
        ManifestMismatch,
        TrainError,
        BatchError,
        ExportError,
        RecipeError,
        FoldError,
        NllError,
        RegistryError,
        ProvenanceError,
    )
    return refusals, failures


def _timing(args: argparse.Namespace) -> int:
    from .forecast import load_and_configure

    started_at = _now()  # the command's start: when --approved-by-owner was given
    params = load_and_configure()  # the governed thread count, before anything imports torch
    if params is None:
        return EXIT_FAILURE
    refusals, failures = _timing_errors()
    return run_guarded(
        lambda: _timing_run(args, params, started_at), refusals=refusals, failures=failures
    )


def _timing_run(args: argparse.Namespace, params: Params, started_at: datetime) -> int:
    from ..finetune import timing

    if args.estimate:
        budget = timing.timing_budget(args.approved_by_owner, given_at=started_at)
        for line in timing.estimate_lines(timing.timing_estimate(), budget):
            _say(line)
        _say("order: " + "; ".join(timing.UNIT_ORDER))
        _say("--estimate: nothing ran")
        return EXIT_OK
    directory = reports_dir() / FINETUNE_REPORTS_SUBDIR
    report = timing.run_timing(
        params=params,
        directory=directory,
        progress=_say,
        approved_cpu_hours=args.approved_by_owner,
        given_at=started_at,
    )
    for line in timing.summary_lines(report):
        _say(line)
    json_path = directory / f"{timing.TIMING_STEM}.json"
    _say(f"wrote {display_path(json_path)} and {display_path(json_path.with_suffix('.md'))}")
    status = report.get("status")
    if status == "failed":
        _say("ERROR: " + "; ".join(report.get("failures") or ["the measurement step failed"]))
        return EXIT_FAILURE
    if status == "stopped":
        reason = (report.get("spend") or {}).get("stop_reason")
        _say(
            f"REFUSED: the measurement step stopped at its budget ({reason}); the report lists "
            "what ran and what did not"
        )
        return EXIT_REFUSED
    return EXIT_OK


# -- what the step-6 subcommands share ----------------------------------------------------------


def _ledger_errors() -> tuple[ExceptionTypes, ExceptionTypes]:
    """``(refusals, failures)`` common to ``fold``, ``pilot-check`` and ``tokenizer-diag``."""
    from ..data.fsutil import LockBusyError
    from ..data.partition import PartitionError, SealedPartitionError
    from ..finetune.export_csv import ExportError, ExportRefused
    from ..finetune.folds import FoldError
    from ..finetune.recipe import RecipeError
    from ..finetune.registry import RegistryError
    from ..finetune.timing import StaleTimingError, TimingError
    from ..finetune.train import TrainError
    from ..finetune.walkforward import FoldFailed, FoldRefused
    from ..kronos.vendored import VendoredTreeError
    from ..kronos.weights import WeightsError
    from ..ledger import LedgerError, LedgerRefused
    from ..manifests import ManifestMismatch
    from ..params import ParamsError
    from ..provenance import ProvenanceError
    from ..report.writer import ConfigChangedError, ReportError
    from ..runtime import RuntimeConfigError

    refusals: ExceptionTypes = (
        FoldRefused,
        LedgerRefused,  # BundleChanged and OrderNotFound are LedgerRefused
        StaleTimingError,
        LockBusyError,  # StoreBusyError is a LockBusyError
        SealedPartitionError,
        PartitionError,
        ExportRefused,
        ConfigChangedError,
    )
    failures: ExceptionTypes = (
        FoldFailed,
        TrainError,
        LedgerError,
        RegistryError,
        ExportError,
        FoldError,
        RecipeError,
        TimingError,
        WeightsError,
        VendoredTreeError,
        ManifestMismatch,
        ParamsError,
        RuntimeConfigError,
        ProvenanceError,
        ReportError,
    )
    return refusals, failures


def _guarded(
    action: Callable[[], int], *, refusals: ExceptionTypes = (), failures: ExceptionTypes = ()
) -> int:
    common_refusals, common_failures = _ledger_errors()
    return run_guarded(
        action, refusals=(*common_refusals, *refusals), failures=(*common_failures, *failures)
    )


def _verify_bundles() -> None:
    """Every pre-registration bundle the ledger records still hashes to its entry (design 7.1;
    ``ledger.BundleChanged``, a refusal, otherwise)."""
    from .. import ledger

    verified = ledger.verify_bundles()
    if verified:
        _say(f"ledger: {len(verified)} pre-registration bundle(s) verified")


def _approval_option(parser, *, metavar: str, help: str, excludes: tuple[str, ...] = ()) -> None:
    parser.add_argument(
        "--approved-by-owner",
        action=_Exclusive,
        excludes=excludes,
        type=positive_number,
        default=None,
        metavar=metavar,
        help=help,
    )


# -- fold ------------------------------------------------------------------------------------------


def _register_fold(actions) -> None:
    parser = actions.add_parser(
        "fold",
        help="train one fold of the quarterly walk-forward (the pilot is --fold q-2024-07): its "
        "estimate first, gate 3, the experiment ledger",
    )
    parser.add_argument(
        "--fold",
        required=True,
        metavar="LABEL",
        help="the fold label, q-YYYY-MM (the quarterly layout, D-056); the pilot is q-2024-07",
    )
    parser.add_argument(
        "--replicate",
        type=positive_int,
        default=None,
        metavar="K",
        help="a seed replicate of the fold (D-048 (4)); without it, the fold's own seed (no "
        "quarterly fold has replicates until their mapping is decided)",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="print the plan and its measured estimate and run nothing (no export, no ledger "
        "entry, no run directory)",
    )
    _approval_option(
        parser,
        metavar="HOURS",
        help="the hours the owner approved in chat for this fold (gate 3): MPS wall-hours when "
        "the timing record chose MPS, CPU-hours on the CPU; the measured estimate may be at most "
        "1.25 times it, and nothing starts that could take this run's spend past 1.25 times it",
    )
    parser.add_argument(
        "--order",
        default=None,
        metavar="ID",
        help="the owner's order recorded in the experiment ledger (a sealed fold that reads "
        "sealed candles needs it; gate 6)",
    )


def _fold(args: argparse.Namespace) -> int:
    from .forecast import load_and_configure

    started_at = _now()  # the command's start: when --approved-by-owner was given
    params = load_and_configure()  # the governed thread count, before anything imports torch
    if params is None:
        return EXIT_FAILURE
    return _guarded(lambda: _fold_run(args, params, started_at))


def _fold_run(args: argparse.Namespace, params: Params, started_at: datetime) -> int:
    from ..finetune import walkforward

    _verify_bundles()
    steps = walkforward.real_steps()
    plan = walkforward.plan_fold(
        args.fold,
        params=params,
        replicate=args.replicate or 0,
        order_id=args.order,
        steps=steps,
    )
    for line in walkforward.estimate_lines(plan):
        _say(line)
    if args.estimate:
        _say("--estimate: nothing ran")
        return EXIT_OK
    outcome = walkforward.run_fold(
        plan,
        approved_hours=args.approved_by_owner,
        given_at=started_at,
        steps=steps,
        progress=_say,
    )
    registered = [unit.checkpoint_id for unit in outcome.units if unit.checkpoint_id]
    _say(
        f"finetune fold {plan.fold.run_label}: done; this run spent "
        f"{outcome.spent_hours:.3f} {walkforward.UNIT_TEXT[outcome.unit]}"
        + (f"; registered {', '.join(registered)}" if registered else "")
    )
    return EXIT_OK


# -- pilot-check -----------------------------------------------------------------------------------


def _register_pilot_check(actions) -> None:
    parser = actions.add_parser(
        "pilot-check",
        help="the pilot check of design 6.2 on the registered pilot (504 fans, the KL, the "
        "weight change; 0 tests): its estimate first, gate 3, the reading GO or NO-GO",
    )
    parser.add_argument(
        "--record-config",
        action=_Exclusive,
        flag=True,
        excludes=("--estimate", "--approved-by-owner"),
        help="record the pilot check's definition only (allowed before the pilot exists)",
    )
    parser.add_argument(
        "--estimate",
        action=_Exclusive,
        flag=True,
        excludes=("--record-config",),
        help="print the measured estimate and run nothing",
    )
    _approval_option(
        parser,
        metavar="CPU_HOURS",
        excludes=("--record-config",),
        help="the CPU-hours the owner approved in chat for this check (gate 3); a measured "
        "estimate more than 1.25 times it is refused before any fan",
    )


def _pilot_check(args: argparse.Namespace) -> int:
    from .forecast import load_and_configure

    started_at = _now()  # the command's start: when --approved-by-owner was given
    params = load_and_configure()  # the governed thread count, before anything imports torch
    if params is None:
        return EXIT_FAILURE
    from ..data.forecast_store import ForecastStoreError
    from ..data.windows import WindowError
    from ..finetune.nll import NllError
    from ..finetune.pilot_check import PilotCheckError, PilotCheckStopped
    from ..kronos.batch import BatchError, Gate3Refused
    from ..kronos.decision import DecisionError

    return _guarded(
        lambda: _pilot_check_run(args, params, started_at),
        refusals=(Gate3Refused, PilotCheckStopped),
        failures=(
            PilotCheckError,
            BatchError,
            ForecastStoreError,
            NllError,
            DecisionError,
            WindowError,
        ),
    )


def _pilot_check_run(args: argparse.Namespace, params: Params, started_at: datetime) -> int:
    from .. import ledger
    from ..finetune import pilot_check, walkforward
    from ..finetune.folds import fold_for
    from ..finetune.recipe import recipe_id
    from ..kronos.batch import require_gate3

    _verify_bundles()
    recipe = walkforward.effective_recipe()
    fold = fold_for(walkforward.PILOT_LABEL, recipe)
    stem = pilot_check.stem_for(recipe)
    if args.record_config:
        digest = pilot_check.record_check_config(recipe, fold)
        _say(f"pilot check {stem}: definition {stem}.config.json recorded (sha256 {digest})")
        return EXIT_OK
    record = walkforward.resolve_pilot()
    inputs = pilot_check.PilotInputs(
        recipe=record.recipe,
        fold=record.fold,
        checkpoint=record.checkpoint,
        fit_record_path=record.fit_record_path,
        fit_record_sha256=record.fit_record_sha256,
        export=record.export,
    )
    steps = pilot_check.real_steps()
    prepared = pilot_check.prepare_check(inputs, params=params, steps=steps)
    estimate = pilot_check.estimate_check(prepared, params=params, steps=steps)
    # what the stopped runs of this check spent counts against its approval (design 3.5)
    prior = pilot_check.prior_spend(stem, entries=ledger.read_ledger())
    total = float(estimate["cpu_seconds"]) + prior["cpu_seconds"]
    shown = {
        **estimate,
        "pending_cpu_seconds": float(estimate["cpu_seconds"]),
        "prior_cpu_seconds": prior["cpu_seconds"],
        "prior_entries": prior["entries"],
        "cpu_seconds": total,
        "cpu_hours": total / 3600.0,
        "needs_owner_approval": total > float(estimate.get("gate3_cpu_seconds", 3600)),
    }
    _say(
        f"pilot check {stem} of {record.checkpoint.checkpoint_id}: estimate about "
        f"{float(estimate['cpu_seconds']):.0f} CPU-seconds ({float(estimate['cpu_hours']):.2f} "
        f"CPU-hours) for {estimate['pending']} pending fans and the KL"
        + (
            f", plus the {prior['cpu_seconds']:.0f} CPU-seconds its stopped runs spent (ledger "
            f"entr{'ies' if len(prior['entries']) > 1 else 'y'} "
            + ", ".join(f"#{seq}" for seq in prior["entries"])
            + f"): {total:.0f} CPU-seconds ({total / 3600:.2f} CPU-hours) in all"
            if prior["entries"]
            else ""
        )
        + (
            "; above gate 3 (3600 CPU-seconds): it needs --approved-by-owner CPU_HOURS"
            if shown["needs_owner_approval"]
            else ""
        )
    )
    if args.estimate:
        _say("--estimate: nothing ran")
        return EXIT_OK
    require_gate3(shown, approved_cpu_hours=args.approved_by_owner)
    approval_seq = None
    if args.approved_by_owner is not None:
        approval = ledger.record_gate3_approval(
            command=PILOT_CHECK_COMMAND,
            subject=stem,
            unit="cpu_hours",
            approved=args.approved_by_owner,
            estimate=float(shown["cpu_hours"]),
            given_at=started_at,
            estimate_detail=dict(shown),
        )
        approval_seq = approval.seq
        _say(
            f"gate 3: the owner approved {args.approved_by_owner:g} CPU-hours (--approved-by-owner "
            f"given at {started_at.strftime('%Y-%m-%dT%H:%M:%SZ')}): approval entry "
            f"#{approval.seq} (line {approval.line_sha256[:12]}...)"
        )
    _say(
        f"spend control: the run stops once it passes "
        f"{pilot_check.spend_limit(args.approved_by_owner):.0f} CPU-seconds (design 3.5)"
        + (
            f", counted from the {prior['cpu_seconds']:.0f} its stopped runs spent"
            if prior["entries"]
            else ""
        )
        + "; the fans stored by then are kept"
    )
    try:
        result = pilot_check.run_check(
            prepared,
            params=params,
            steps=steps,
            progress=_say,
            approved_cpu_hours=args.approved_by_owner,
            prior_cpu_seconds=prior["cpu_seconds"],
        )
    except pilot_check.PilotCheckStopped as stopped:
        details = {**stopped.spend, "approval_seq": approval_seq}
        spent = float(details.get("run_cpu_seconds") or 0.0)
        try:
            entry = ledger.record_exploration(
                activity=pilot_check.LEDGER_ACTIVITY,
                subject=stem,
                stage=pilot_check.STOPPED_STAGE,
                recipe_id=recipe_id(recipe),
                inputs=["training"],
                outcomes=["training"],
                details=details,
            )
        except Exception as error:  # noqa: BLE001 - the stop goes on as the refusal it is
            _say(
                f"WARNING: the stopped run could not be recorded in the ledger "
                f"({type(error).__name__}: {error}): show the owner its {spent:.0f} CPU-seconds"
            )
        else:
            _say(
                f"ledger: stopped entry #{entry.seq} (line {entry.line_sha256[:12]}...) records "
                f"the {spent:.0f} CPU-seconds this run spent; the next run of {stem} counts them "
                "against its approval"
            )
        raise
    entry = ledger.record_exploration(
        activity=pilot_check.LEDGER_ACTIVITY,
        subject=stem,
        stage="done",
        recipe_id=recipe_id(recipe),
        inputs=["training"],
        outcomes=["training"],
        details=result.details,
    )
    for line in result.summary_lines:
        _say(line)
    _say(
        f"wrote {display_path(result.json_path)} and {display_path(result.markdown_path)}; "
        f"ledger: exploration entry #{entry.seq} (line {entry.line_sha256[:12]}...)"
    )
    if result.reading == pilot_check.READING_GO:
        _say(f"READING: {result.reading}")
    else:  # a reading is a result: exit 0 (the walk-forward refuses to start on NO-GO)
        _say(f"READING: {result.reading}: {pilot_check.NO_GO_OPTIONS}")
    return EXIT_OK


# -- tokenizer-diag --------------------------------------------------------------------------------


def _register_tokenizer_diag(actions) -> None:
    parser = actions.add_parser(
        "tokenizer-diag",
        help="the tokenizer diagnostic of design section 4 (0 tests): Part A on training data, "
        "Part B on validation inputs after the pre-registration bundle",
    )
    parser.add_argument(
        "--part",
        required=True,
        choices=TOKENIZER_DIAG_PARTS,
        help="training (Part A: Binance training segment, the token mismatch of D-048 (5)) or "
        "validation (Part B: Binance and BingX inputs, after the pre-registration bundle)",
    )
    parser.add_argument(
        "--record-config",
        action="store_true",
        help="record the part's definition only (Part B's belongs in the bundle, design 7.1)",
    )


def _tokenizer_diag(args: argparse.Namespace) -> int:
    from .forecast import load_and_configure

    params = load_and_configure()  # the governed thread count, before anything imports torch
    if params is None:
        return EXIT_FAILURE
    from ..data.store import StoreError
    from ..data.windows import WindowError
    from ..finetune.tokenizer_diag import TokenizerDiagError, TokenizerDiagRefused

    return _guarded(
        lambda: _tokenizer_diag_run(args, params),
        refusals=(TokenizerDiagRefused,),
        failures=(TokenizerDiagError, WindowError, StoreError),
    )


def _flag_line(flag: Mapping[str, Any]) -> str:
    """The ``WARNING:`` line of one raised flag (``tokenizer_diag``'s flag record)."""
    line = f"WARNING: {flag.get('flag')}: {flag.get('message')}"
    if flag.get("flag") == MISMATCH_FLAG:
        line += f"; {MISMATCH_FLAG_TEXT}"
    return line


def _tokenizer_diag_run(args: argparse.Namespace, params: Params) -> int:
    from .. import ledger
    from ..finetune import tokenizer_diag

    _verify_bundles()
    part = args.part
    stem = tokenizer_diag.STEMS[part]
    if args.record_config:
        digest = tokenizer_diag.record_diag_config(part)
        _say(
            f"tokenizer diagnostic {stem}: definition {stem}.config.json recorded (sha256 {digest})"
        )
        return EXIT_OK
    if part == "validation":
        # only a ledger that may authorize reading this data root admits Part B with its bundle
        # (ledger.binding_problem; hard rule 7): a bundle in a scratch ledger never reads the
        # repository's validation inputs
        ledger.require_binding()
        if not ledger.entries_of("preregistration"):
            raise ledger.LedgerRefused(
                "Part B runs after the pre-registration bundle (design 4, 7.1): the experiment "
                "ledger has no preregistration entry"
            )
    steps = tokenizer_diag.real_steps()
    prepared = tokenizer_diag.prepare_diag(part, params=params, steps=steps)
    estimate = tokenizer_diag.estimate_diag(prepared, steps=steps)
    _say(
        f"tokenizer diagnostic {stem}: estimate about {float(estimate['cpu_seconds']):.0f} "
        f"CPU-seconds for {estimate['windows']} windows "
        f"({float(estimate['seconds_per_window']):.3f} s each); gate 3 at "
        f"{estimate['gate3_cpu_seconds']} CPU-seconds"
    )
    tokenizer_diag.require_within_gate3(estimate)
    result = tokenizer_diag.run_diag(prepared, params=params, steps=steps, progress=_say)
    entry = ledger.record_exploration(
        activity="tokenizer-diag",
        subject=stem,
        stage="done",
        recipe_id=None,
        inputs=[part],
        outcomes=["training"] if part == "training" else [],
        details=result.details,
    )
    for line in result.summary_lines:
        _say(line)
    for flag in result.flags:
        _say(_flag_line(flag))
    _say(
        f"wrote {display_path(result.json_path)} and {display_path(result.markdown_path)}; "
        f"ledger: exploration entry #{entry.seq} (line {entry.line_sha256[:12]}...)"
    )
    return EXIT_OK


# -- the command -----------------------------------------------------------------------------------

SUBCOMMANDS: dict[str, tuple[Callable[..., None], Callable[[argparse.Namespace], int]]] = {
    "timing": (_register_timing, _timing),
    "fold": (_register_fold, _fold),
    "pilot-check": (_register_pilot_check, _pilot_check),
    "tokenizer-diag": (_register_tokenizer_diag, _tokenizer_diag),
}


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "finetune",
        help="retraining Kronos-small for BTC (step 6): timing (the measurement step), fold, "
        "pilot-check, tokenizer-diag",
    )
    actions = parser.add_subparsers(dest="finetune_command", required=True)
    for add, _ in SUBCOMMANDS.values():
        add(actions)


def run(args: argparse.Namespace) -> int:
    _, handler = SUBCOMMANDS[args.finetune_command]
    return handler(args)
