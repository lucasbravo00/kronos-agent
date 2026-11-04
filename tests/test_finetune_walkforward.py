"""The per-fold training path, ``kronos_agent.finetune.walkforward`` (``docs/RETRAINING.md`` 3.5,
6.1 to 6.3 and 8.1 to 8.3): the estimate and the compute choice on the timing record, its
freshness, the layout, bundle and order rules (and the ledger's binding to the data root),
gate 3, the spend cap and the deadline each figure gives, the pilot rule and its refit, the
stages (a run registered under another ledger is never adopted), the failure policy (time
aborts, signals, a run registered before an error), a training child left running, blinding
(T28) and the pilot's resolution.

Everything runs on the stand-ins of ``fold_fakes`` in the test's own roots: a timing record with
the measurements of ``timing-2.json``, a tiny synthetic export, a stand-in training child that
writes a fold record and registers a stand-in checkpoint, a stand-in process list and a stand-in
clock. Nothing trains, exports real candles or times anything (D-055 (6)); one test reads the
real timing record through a read-only view of the reports root, and one scans the real process
list for a sleeping process that carries a training child's command line.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import finetune_fakes as ff
import fold_fakes
import pytest

from kronos_agent import budget, ledger, paths
from kronos_agent.data.fsutil import LockBusyError, exclusive_lock
from kronos_agent.data.partition import MAX_CONTEXT, VALIDATION_START, SealedAccess
from kronos_agent.data.timeutil import HOUR_MS, to_ms
from kronos_agent.finetune import export_csv, registry, runner, timing, train
from kronos_agent.finetune import walkforward as wf
from kronos_agent.finetune.folds import FoldError, fold_for, quarterly_folds
from kronos_agent.finetune.recipe import R1, recipe8, recipe_id, training_seeds
from kronos_agent.finetune.timing import StaleTimingError, TimingError
from kronos_agent.hashing import sha256_file
from kronos_agent.params import load_params
from kronos_agent.paths import reports_dir

PARAMS = load_params()
GIVEN_AT = datetime(2026, 11, 3, 9, 30, tzinfo=UTC)
DOUBLED = wf.doubled_recipe()
R1_ID = recipe_id(R1)
DOUBLED_ID = recipe_id(DOUBLED)
PILOT_ID = "ks-btc-0ee0d123-q-2024-07"
REFIT_ID = "ks-btc-56e3d2f9-q-2024-07"
FOLD2 = "q-2024-10"


@pytest.fixture
def world() -> fold_fakes.World:
    """The stand-ins, and a fresh timing record with timing-2's measurements in the test's
    reports root."""
    fold_fakes.write_timing_record()
    return fold_fakes.World()


def plan(world: fold_fakes.World, label: str = wf.PILOT_LABEL, **kwargs) -> wf.FoldPlan:
    return wf.plan_fold(label, params=PARAMS, steps=world.steps(), **kwargs)


def run(world: fold_fakes.World, fold_plan: wf.FoldPlan, approved: float | None = None):
    return wf.run_fold(
        fold_plan,
        approved_hours=approved,
        given_at=GIVEN_AT,
        steps=world.steps(),
        progress=lambda line: None,
    )


def kinds() -> list[str]:
    return fold_fakes.ledger_kinds()


def entries(kind: str) -> list[ledger.LedgerEntry]:
    return list(ledger.entries_of(kind))


def run_dirs() -> list[str]:
    parent = registry.runs_dir()
    return sorted(path.name for path in parent.iterdir()) if parent.is_dir() else []


def pilot_done(rule_epoch: int = 3, *, approved: float = 10.0, world=None) -> fold_fakes.World:
    """A pilot of R1 trained and recorded with ``selected_pseudo_epoch = rule_epoch``."""
    world = world or fold_fakes.World()
    stop = "max_epochs" if rule_epoch == R1.max_epochs else "early_stop"
    world.script.append(fold_fakes.Script(selected=rule_epoch, stop_reason=stop))
    run(world, plan(world), approved)
    return world


# -- the steps and the constants -------------------------------------------------------------------


def test_the_real_steps_are_the_real_functions():
    steps = wf.real_steps()
    assert steps.export_fold is export_csv.export_fold
    assert steps.fit is train.fit
    assert steps.load_timing is timing.load_timing
    assert isinstance(steps.mps_available(), bool)
    assert isinstance(steps.children_cpu_seconds(), float) and steps.clock() > 0


def test_the_constants_agree_with_the_ledger_the_folds_and_the_partition():
    assert wf.APPROVAL_TOLERANCE == ledger.APPROVAL_TOLERANCE == 1.25
    assert wf.GATE3_HOURS == ledger.GATE3_HOURS == 1.0
    assert wf.COMMAND in ledger.APPROVAL_COMMANDS
    assert wf.PILOT_ACTIVITY in ledger.EXPLORATION_ACTIVITIES
    assert set(wf.RETRY_STATUSES) | set(wf.FAILED_RUN_STATUSES) | {"no_result", "registered"} == (
        set(ledger.FIT_STATUSES)
    )
    assert set(wf.UNITS.values()) == set(ledger.APPROVAL_UNITS)
    assert wf.MAX_ATTEMPTS == max(ledger.FIT_ATTEMPTS)
    pilot = quarterly_folds(R1)[0]
    assert pilot.label == wf.PILOT_LABEL and wf.WALK_FORWARD_LAYOUT == pilot.layout
    assert pilot.csv_end_ms == to_ms(VALIDATION_START) - MAX_CONTEXT * HOUR_MS
    assert wf.is_pilot(pilot) and not wf.is_pilot(quarterly_folds(R1)[1])
    assert json.loads(json.dumps(dict(wf.PILOT_RULE))) == dict(wf.PILOT_RULE)


# -- the estimate ----------------------------------------------------------------------------------


def test_the_fit_estimate_terms_on_a_stand_in_record():
    record = fold_fakes.timing_record(seconds_per_step=2.0, theta=1.5)
    fold = fold_for(wf.PILOT_LABEL, R1)
    estimate = wf.fit_estimate(R1, fold, record, record_sha256="a" * 64)
    assert (estimate.recipe_id, estimate.max_epochs, estimate.steps) == (R1_ID, 6, 3072)
    assert (estimate.device, estimate.threads, estimate.unit) == ("mps", 2, "mps_hours")
    assert (estimate.nll_decisions, estimate.val_windows) == (336, 832)
    assert estimate.t_nll == pytest.approx(336 * fold_fakes.M8_SECONDS_PER_DECISION)
    assert estimate.t_val == pytest.approx(26 * 2.0 / 3)  # ceil(832 / 32) batches
    assert estimate.t_setup == wf.SETUP_SECONDS_PRIOR == 300.0
    training = 1.5 * 3072 * 2.0
    wall = training + 7 * (estimate.t_nll + estimate.t_val) + 300.0
    assert estimate.wall_seconds == pytest.approx(wall)
    assert estimate.device_hours == pytest.approx(wall / 3600)
    rate = fold_fakes.M4_CPU_SECONDS / fold_fakes.M4_SECONDS
    assert estimate.cpu_side_seconds == pytest.approx(
        7 * estimate.t_nll * 2 + rate * training + 300
    )
    assert estimate.timing_record_sha256 == "a" * 64
    assert estimate.as_dict()["wall_seconds"] == estimate.wall_seconds
    smaller = wf.fit_estimate(R1, fold, record, nll_decisions=168, val_windows=64)
    assert smaller.t_nll == pytest.approx(168 * fold_fakes.M8_SECONDS_PER_DECISION)
    assert smaller.t_val == pytest.approx(2 * 2.0 / 3)
    assert len(smaller.timing_record_sha256) == 64  # the record's canonical JSON


def test_on_the_cpu_the_estimate_is_in_cpu_hours_and_theta_defaults_to_one():
    record = fold_fakes.timing_record(
        device="cpu", threads=2, theta=None, mps_watermark_ratio=None, seconds_per_step=1.0
    )
    estimate = wf.fit_estimate(R1, fold_for(wf.PILOT_LABEL, R1), record)
    assert (estimate.unit, estimate.theta) == ("cpu_hours", 1.0)
    assert estimate.device_hours == pytest.approx(estimate.wall_seconds * 2 / 3600)
    assert estimate.cpu_side_seconds == pytest.approx(estimate.wall_seconds * 2)


def test_a_record_without_its_numbers_is_refused():
    fold = fold_for(wf.PILOT_LABEL, R1)
    record = fold_fakes.timing_record()
    del record["measurements"]["M8"]
    with pytest.raises(TimingError, match="M8"):
        wf.fit_estimate(R1, fold, record)
    with pytest.raises(TimingError, match="theta"):
        wf.fit_estimate(R1, fold, fold_fakes.timing_record(theta=0.5))
    with pytest.raises(TimingError, match="seconds_per_step"):
        wf.fit_estimate(R1, fold, fold_fakes.timing_record(seconds_per_step=None))
    with pytest.raises(wf.FoldRefused, match="decided no device"):
        wf.fit_estimate(R1, fold, fold_fakes.timing_record(device=None))
    template = replace(R1, budget="template", pseudo_epoch_steps=None)
    with pytest.raises(wf.FoldRefused, match="no pseudo-epoch budget"):
        wf.fit_estimate(template, fold, fold_fakes.timing_record())


@pytest.mark.real_roots("reports")
def test_the_estimate_of_the_real_timing_record():
    """INTERFACES-3a section 1: on timing-2.json the pilot at E_max 6 is 17,801.8 s = 4.945
    MPS-hours and the refit at E_max 12 35,204.3 s = 9.779 (CPU side about 0.57 and 1.03)."""
    record = timing.load_timing(directory=reports_dir() / timing.FINETUNE_REPORTS_SUBDIR)
    fold = fold_for(wf.PILOT_LABEL, R1)
    pilot = wf.fit_estimate(R1, fold, record)
    refit = wf.fit_estimate(DOUBLED, fold, record)
    assert pilot.wall_seconds == pytest.approx(17_801.8, abs=1.0)
    assert refit.wall_seconds == pytest.approx(35_204.3, abs=1.0)
    assert round(pilot.device_hours, 3) == 4.945 and round(refit.device_hours, 3) == 9.779
    assert round(pilot.cpu_side_seconds / 3600, 2) == 0.57
    assert round(refit.cpu_side_seconds / 3600, 2) == 1.03
    assert (pilot.unit, refit.recipe_id[:8]) == ("mps_hours", "56e3d2f9")


# -- the compute choice and the record's freshness ------------------------------------------------


def test_the_compute_choice_follows_the_timing_record():
    compute = wf.compute_choice(fold_fakes.timing_record())
    assert compute == train.Compute(
        device="mps",
        threads=2,
        deterministic=True,
        activation_checkpointing=True,
        mps_watermark_ratio=1.0,
        memory_guard=True,
    )
    cpu = wf.compute_choice(
        fold_fakes.timing_record(device="cpu", threads=8, mps_watermark_ratio=None, theta=None)
    )
    assert (cpu.device, cpu.threads, cpu.mps_watermark_ratio, cpu.memory_guard) == (
        "cpu",
        8,
        None,
        True,
    )
    with pytest.raises(wf.FoldRefused, match="decided no device"):
        wf.compute_choice(fold_fakes.timing_record(device=None, threads=None))
    neither = fold_fakes.timing_record(batch_32_fits=False, activation_checkpointing=False)
    with pytest.raises(wf.FoldRefused, match="batch 16 is a recipe change"):
        wf.compute_choice(neither)


@pytest.mark.parametrize(
    "where, value, message",
    [
        ("environment.torch_version", "2.0.0", "torch"),
        ("environment.platform", "linux-x86_64", "platform"),
        ("code.training_code_sha256", "f" * 64, "training code"),
    ],
)
def test_a_stale_timing_record_is_refused(world, where, value, message):
    record = fold_fakes.timing_record()
    section, key = where.split(".")
    record[section][key] = value
    fold_fakes.write_timing_record(record)
    with pytest.raises(StaleTimingError, match=message):
        plan(world)


def test_mps_chosen_but_unavailable_and_another_batch_are_refused(world):
    world.mps = False
    with pytest.raises(wf.FoldRefused, match="MPS, which is not available"):
        plan(world)
    world.mps = True
    fold_fakes.write_timing_record(batch_size=16)
    with pytest.raises(wf.FoldRefused, match="measured batch 16"):
        plan(world)


def test_a_missing_timing_record_is_a_failure(world):
    (fold_fakes.timing_directory() / f"{timing.TIMING_STEM}.json").unlink()
    with pytest.raises(TimingError, match="no timing record"):
        plan(world)


# -- labels, orders and the bundle -----------------------------------------------------------------


@pytest.mark.parametrize(
    "label, replicate, error, message",
    [
        ("wf-2024-07", 0, wf.FoldRefused, "quarterly layout"),
        ("wf-2024-07", 1, wf.FoldRefused, "quarterly layout"),
        ("wf-2025-06", 2, wf.FoldRefused, "quarterly layout"),
        ("final-2026-08-01", 0, wf.FoldRefused, "model to date"),
        ("q-2024-07", 1, FoldError, "replicates"),  # quarterly replicates: INTERFACES-3a 16
        ("q-2024-08", 0, FoldError, "not a planned quarterly fold"),
        ("synthetic-x", 0, FoldError, "synthetic"),
        ("Q-2024-07", 0, FoldError, "is not wf-YYYY-MM"),
    ],
)
def test_labels_outside_the_quarterly_layout_are_refused(world, label, replicate, error, message):
    with pytest.raises(error, match=message):
        plan(world, label, replicate=replicate)
    assert not ledger.ledger_path().exists() and world.exports == []


def test_the_order_belongs_to_sealed_folds_only(world):
    with pytest.raises(wf.FoldRefused, match="--order belongs to a sealed fold"):
        plan(world, order_id="order-0001")
    with pytest.raises(wf.FoldRefused, match="needs the owner's recorded order"):
        plan(world, "s-2026-09")
    with pytest.raises(ledger.OrderNotFound):
        plan(world, "s-2026-08", order_id="order-0001")


def test_folds_after_the_pilot_train_after_the_bundle(world):
    with pytest.raises(wf.FoldRefused, match="after the pre-registration bundle"):
        plan(world, FOLD2)
    fold_fakes.write_bundle(R1)
    fold2 = plan(world, FOLD2)
    assert (fold2.stage, fold2.is_pilot, fold2.blinded, fold2.refit) == ("train", False, True, None)
    assert fold2.first is not None and fold2.first.recipe_id == R1_ID
    pilot = plan(world)
    assert (pilot.is_pilot, pilot.blinded, pilot.stage) == (True, False, "train")
    assert pilot.first.recipe_id == R1_ID and pilot.refit.recipe_id == DOUBLED_ID


def test_the_plan_and_its_lines_write_nothing(world):
    pilot = plan(world)
    lines = wf.estimate_lines(pilot)
    text = "\n".join(lines)
    assert "q-2024-07" in text and "4.945 MPS-hours" in text and "9.779 MPS-hours" in text
    assert "pilot and refit together: 14.724 MPS-hours" in text
    assert "17801.8 s" in text and "35204.3 s" in text and "timing-2 (sha256" in text
    assert "--approved-by-owner HOURS" in text and wf.IDLE_REMINDER in text
    assert "priors the measurement step did not measure" in text
    assert "/" + "Users" not in text and "~/" not in text
    assert not ledger.ledger_path().exists()
    assert world.exports == [] and world.fits == [] and run_dirs() == []
    assert pilot.record_sha256 == sha256_file(
        fold_fakes.timing_directory() / f"{timing.TIMING_STEM}.json"
    )


# -- gate 3 ----------------------------------------------------------------------------------------


def test_gate3_without_a_figure_is_refused_before_anything_is_written(world):
    with pytest.raises(wf.FoldRefused, match="above the 1 hour of gate 3") as excinfo:
        run(world, plan(world))
    assert "4.945 MPS-hours" in str(excinfo.value) and "nothing ran" in str(excinfo.value)
    assert not ledger.ledger_path().exists()
    assert world.exports == [] and world.fits == [] and run_dirs() == []


def test_gate3_refuses_an_estimate_above_the_tolerance(world):
    with pytest.raises(wf.FoldRefused, match="1.268 times the 3.9 MPS-hours") as excinfo:
        run(world, plan(world), 3.9)
    assert "tolerance of 1.25" in str(excinfo.value)
    assert "a figure H of at least 3.96 MPS-hours admits it" in str(excinfo.value)
    with pytest.raises(wf.FoldRefused, match="1.254 times the 3.943 MPS-hours"):
        run(world, plan(world), 3.943)  # a ratio of 1.2541 reads as 1.25 with 2 decimals
    assert not ledger.ledger_path().exists() and world.fits == []
    with pytest.raises(ValueError, match="finite number > 0"):
        run(world, plan(world), float("nan"))


def test_the_pilot_records_approval_planned_fit_and_done_in_that_order(world):
    world.script.append(fold_fakes.Script(selected=3, seconds=fold_fakes.PILOT_SECONDS))
    fold_plan = plan(world)
    outcome = run(world, fold_plan, 10.0)
    assert kinds() == ["gate3_approval", "exploration", "fit", "exploration"]
    (call,) = world.fits
    assert call["ledger_kinds"] == ["gate3_approval", "exploration"]  # before the child
    request = call["request"]
    assert request.mode == "train" and request.recipe == R1 and not request.blinded
    assert request.wrappers == train.Wrappers() and request.compute == fold_plan.compute
    deadline = min(2 * fold_plan.first.wall_seconds, 12.5 * 3600)
    assert request.deadline_seconds == pytest.approx(deadline)
    assert call["timeout_seconds"] == pytest.approx(deadline + wf.CHILD_TIMEOUT_MARGIN_SECONDS)
    assert call["register"] is True
    (approval,) = entries("gate3_approval")
    payload = approval.payload
    assert (payload["command"], payload["subject"], payload["unit"]) == (
        "finetune fold",
        "q-2024-07",
        "mps_hours",
    )
    assert payload["approved"] == 10.0 and payload["given_at"] == "2026-11-03T09:30:00Z"
    assert payload["estimate"] == pytest.approx(4.945, abs=1e-3)
    assert payload["estimate_detail"]["first"]["recipe_id"] == R1_ID
    assert payload["estimate_detail"]["refit"]["recipe_id"] == DOUBLED_ID
    planned, done = entries("exploration")
    assert planned.payload["stage"] == "planned" and done.payload["stage"] == "done"
    assert planned.payload["activity"] == done.payload["activity"] == "pilot-fold"
    assert planned.payload["inputs"] == planned.payload["outcomes"] == ["training"]
    details = planned.payload["details"]
    assert details["pilot_rule_definition"] == dict(wf.PILOT_RULE)
    assert details["approval_seq"] == approval.seq and details["max_epochs"] == 6
    assert details["onecycle_total_steps"] == 3072
    assert details["seeds_digest"] == training_seeds(R1_ID, "q-2024-07", 0).digest
    assert details["timing_record"]["sha256"] == fold_plan.record_sha256
    assert details["training_code_sha256"] == train.training_code_sha256()
    assert details["export_content_sha256"] == request.export.content_sha256
    (fit,) = entries("fit")
    entry = registry.find_entry(registry.CheckpointRef(PILOT_ID))
    run_dir = registry.runs_dir() / entry.document["run_id"]
    assert fit.payload["checkpoint_id"] == PILOT_ID
    assert fit.payload["registry_line_sha256"] == entry.line_sha256
    assert fit.payload["fit_record_sha256"] == sha256_file(run_dir / runner.RESULT_JSON)
    assert (fit.payload["status"], fit.payload["attempt"], fit.payload["blinded"]) == (
        "registered",
        1,
        False,
    )
    assert fit.payload["approval_seq"] == approval.seq and fit.payload["device"] == "mps"
    assert fit.payload["wall_seconds"] == pytest.approx(fold_fakes.PILOT_SECONDS)
    shown = done.payload["details"]
    assert (shown["selected_pseudo_epoch"], shown["pilot_rule"]) == (3, "not fired")
    assert shown["checkpoint_id"] == PILOT_ID and shown["registry_line_sha256"] == entry.line_sha256
    assert shown["epochs"][0] == {
        "pseudo_epoch": 0,
        "nll4_tf": fold_fakes.SHIPPED_NLL,
        "train_loss": None,
        "vendored_val_loss": None,
    }
    assert shown["median_step_seconds"] == fold_fakes.STEP_MEDIAN
    assert shown["stop_reason"] == "early_stop" and shown["max_epochs"] == 6
    (unit,) = outcome.units
    assert (unit.status, unit.checkpoint_id, unit.fit_seq) == ("registered", PILOT_ID, fit.seq)
    assert unit.shown["pilot_rule"] == "not fired"
    assert outcome.spent_hours == pytest.approx(fold_fakes.PILOT_SECONDS / 3600)
    text = "\n".join(outcome.lines)
    assert f"pseudo-epoch 0: NLL4-TF {fold_fakes.SHIPPED_NLL:.6f}" in text
    assert "selected pseudo-epoch 3 of 6 (stop: early_stop)" in text and "not fired" in text
    assert "approval entry #1" in text and "/" + "Users" not in text
    after = plan(world)
    assert (after.stage, after.first, after.refit) == ("done", None, None)
    assert wf.effective_recipe() == R1


def test_a_unit_below_one_hour_needs_no_figure_and_its_cpu_deadline_is_per_thread(world):
    fold_fakes.write_timing_record(
        device="cpu", threads=2, theta=None, mps_watermark_ratio=None, seconds_per_step=0.1
    )
    world.script.append(fold_fakes.Script(seconds=500.0, cpu_seconds=900.0))
    fold_plan = plan(world)
    assert fold_plan.first.unit == "cpu_hours" and fold_plan.first.device_hours < 1.0
    outcome = run(world, fold_plan)
    assert "gate3_approval" not in kinds()
    (call,) = world.fits
    expected = min(2 * fold_plan.first.wall_seconds, 1.0 * 3600 / 2)  # CPU-hours per thread
    assert call["request"].deadline_seconds == pytest.approx(expected)
    (fit,) = entries("fit")
    assert fit.payload["approval_seq"] is None and fit.payload["device"] == "cpu"
    assert outcome.unit == "cpu_hours"
    assert outcome.spent_hours == pytest.approx(900.0 / 3600, abs=0.01)  # CPU time, not wall


# -- the pilot rule --------------------------------------------------------------------------------


def test_the_pilot_rule_and_the_effective_recipe():
    assert wf.pilot_rule(R1, 6) == "fired" and wf.pilot_rule(R1, 5) == "not fired"
    assert wf.pilot_rule(DOUBLED, 12) == wf.pilot_rule(DOUBLED, 3) == "already doubled"
    assert (DOUBLED.name, DOUBLED.max_epochs, DOUBLED.onecycle_total_steps) == ("R1", 12, 6144)
    assert DOUBLED_ID.startswith("56e3d2f9b0a342ce") and replace(DOUBLED, max_epochs=6) == R1
    assert wf.effective_recipe(entries=()) == R1


def test_the_rule_fires_and_the_refit_past_the_cap_is_refused_after_the_pilot(world):
    world.script.append(
        fold_fakes.Script(selected=6, stop_reason="max_epochs", seconds=fold_fakes.PILOT_SECONDS)
    )
    lines: list[str] = []
    with pytest.raises(wf.FoldRefused) as excinfo:
        wf.run_fold(
            plan(world),
            approved_hours=10.0,
            given_at=GIVEN_AT,
            steps=world.steps(),
            progress=lines.append,
        )
    message = str(excinfo.value)
    assert message.startswith("the pilot rule fired; the refit at E_max 12 needs about 9.779")
    assert "with 4.945 MPS-hours spent in this run that passes 1.25 x the approved 10" in message
    assert "rerun `finetune fold --fold q-2024-07 --approved-by-owner H`" in message
    assert kinds() == ["gate3_approval", "exploration", "fit", "exploration"]
    assert len(world.fits) == 1 and any("pilot rule" in line and "fired" in line for line in lines)
    assert entries("exploration")[-1].payload["details"]["pilot_rule"] == "fired"
    assert wf.effective_recipe() == DOUBLED
    refit_plan = plan(world)
    assert (refit_plan.stage, refit_plan.recipe, refit_plan.refit) == ("refit", DOUBLED, None)
    assert refit_plan.first.recipe_id == DOUBLED_ID
    assert refit_plan.first.device_hours == pytest.approx(9.779, abs=1e-3)
    world.script.append(fold_fakes.Script(selected=8, seconds=fold_fakes.REFIT_SECONDS))
    run(world, refit_plan, 8.0)  # 9.779 / 8 = 1.22
    assert [call["request"].recipe for call in world.fits] == [R1, DOUBLED]  # R1 never again
    assert world.fits[1]["request"].recipe.max_epochs == 12
    entry = registry.find_entry(registry.CheckpointRef(REFIT_ID))
    assert entry.document["recipe_id"] == DOUBLED_ID
    done = entries("exploration")[-1].payload
    assert (done["recipe_id"], done["details"]["pilot_rule"]) == (DOUBLED_ID, "already doubled")
    assert plan(world).stage == "done" and wf.effective_recipe() == DOUBLED


def test_the_refit_runs_in_the_same_invocation_when_the_cap_admits_it(world):
    world.script += [
        fold_fakes.Script(selected=6, stop_reason="max_epochs", seconds=fold_fakes.PILOT_SECONDS),
        fold_fakes.Script(selected=5, seconds=fold_fakes.REFIT_SECONDS),
    ]
    outcome = run(world, plan(world), 12.0)
    assert [call["request"].recipe for call in world.fits] == [R1, DOUBLED]
    assert [unit.checkpoint_id for unit in outcome.units] == [PILOT_ID, REFIT_ID]
    assert kinds() == [
        "gate3_approval",
        "exploration",
        "fit",
        "exploration",
        "exploration",
        "fit",
        "exploration",
    ]
    approval = entries("gate3_approval")[0]
    assert {entry.payload["approval_seq"] for entry in entries("fit")} == {approval.seq}
    left = (15.0 - fold_fakes.PILOT_SECONDS / 3600) * 3600
    refit_deadline = world.fits[1]["request"].deadline_seconds
    assert refit_deadline == pytest.approx(min(2 * fold_fakes.REFIT_SECONDS, left), rel=1e-6)
    assert outcome.spent_hours == pytest.approx(
        (fold_fakes.PILOT_SECONDS + fold_fakes.REFIT_SECONDS) / 3600
    )
    assert wf.pilot_rule(DOUBLED, 12) == "already doubled"  # the rule never fires twice


# -- stages ----------------------------------------------------------------------------------------


def register_without_entries(
    world: fold_fakes.World, monkeypatch, label: str, *, selected: int
) -> str:
    """A fold registered by a run of the command killed between its registration and its ``fit``
    entry: what the command records before a child starts (the approval and, for the pilot, its
    planned entry) is in the ledger, the child's own entries are not."""
    stop = "max_epochs" if selected == R1.max_epochs else "early_stop"
    world.script.append(fold_fakes.Script(selected=selected, stop_reason=stop))
    with monkeypatch.context() as patch:
        kill_at_the_registered_fit_entry(patch)
        with pytest.raises(KeyboardInterrupt):
            run(world, plan(world, label), 10.0)
    return registry.read_registry()[-1].checkpoint_id


def test_a_registered_pilot_without_its_entries_is_recorded_without_compute(world, monkeypatch):
    register_without_entries(world, monkeypatch, wf.PILOT_LABEL, selected=4)
    assert kinds() == ["gate3_approval", "exploration"]  # the approval and the planned entry
    (approval,) = entries("gate3_approval")
    fold_plan = plan(world)
    assert (fold_plan.stage, fold_plan.first) == ("bookkeeping", None)
    assert any("bookkeeping" in line for line in wf.estimate_lines(fold_plan))
    outcome = run(world, fold_plan)
    assert len(world.fits) == 1 and outcome.units == ()  # no compute after the killed run
    assert kinds() == ["gate3_approval", "exploration", "fit", "exploration"]
    fit, done = entries("fit")[0], entries("exploration")[-1]
    assert fit.payload["checkpoint_id"] == PILOT_ID and fit.payload["attempt"] == 1
    assert fit.payload["approval_seq"] == approval.seq  # the planned entry names it
    assert done.payload["details"]["pilot_rule"] == "not fired"
    assert plan(world).stage == "done"


def test_bookkeeping_that_fires_the_rule_then_needs_the_refit_figure(world, monkeypatch):
    register_without_entries(world, monkeypatch, wf.PILOT_LABEL, selected=6)
    fold_plan = plan(world)
    assert fold_plan.stage == "bookkeeping" and fold_plan.first.recipe_id == DOUBLED_ID
    before = fold_fakes.ledger_text()
    with pytest.raises(wf.FoldRefused, match="above the 1 hour of gate 3"):
        run(world, fold_plan)
    assert fold_fakes.ledger_text() == before
    world.script.append(fold_fakes.Script(selected=2, seconds=600.0))
    run(world, fold_plan, 8.0)
    assert kinds() == [
        "gate3_approval",
        "exploration",
        "gate3_approval",
        "fit",
        "exploration",
        "exploration",
        "fit",
        "exploration",
    ]
    assert [call["request"].recipe for call in world.fits] == [R1, DOUBLED]
    assert plan(world).stage == "done"
    pilot_approval, refit_approval = entries("gate3_approval")
    pilot_fit, refit_fit = (entry.payload for entry in entries("fit"))
    assert pilot_fit["recipe_id"] == R1_ID and pilot_fit["approval_seq"] == pilot_approval.seq
    assert refit_fit["recipe_id"] == DOUBLED_ID and refit_fit["approval_seq"] == refit_approval.seq


def kill_at_the_registered_fit_entry(monkeypatch) -> None:
    """The parent dies between a run's registration and its ``fit`` entry (the entries of
    failed attempts are still recorded)."""
    record_fit = ledger.record_fit

    def killed(**kwargs):
        if kwargs["status"] == "registered":
            raise KeyboardInterrupt("killed between the registration and the fit entry")
        return record_fit(**kwargs)

    monkeypatch.setattr(ledger, "record_fit", killed)


def test_bookkeeping_of_a_killed_pilot_names_its_own_approval_and_attempt(world, monkeypatch):
    """The pilot fails once, its retry registers and selects its last pseudo-epoch, and the
    parent dies before the fit entry. The next invocation records that fit entry with the
    approval the pilot ran under (its planned entry's) and attempt 2, then refits under its own
    new approval: never the refit's approval for the pilot."""
    world.script += [
        fold_fakes.Script(status="failed", seconds=120.0),
        fold_fakes.Script(selected=6, stop_reason="max_epochs", seconds=17_000.0),
    ]
    with monkeypatch.context() as patch:
        kill_at_the_registered_fit_entry(patch)
        with pytest.raises(KeyboardInterrupt):
            run(world, plan(world), 10.0)
    assert kinds() == ["gate3_approval", "exploration", "fit"]  # approval, planned, attempt 1
    (pilot_approval,) = entries("gate3_approval")
    fold_plan = plan(world)
    assert fold_plan.stage == "bookkeeping" and fold_plan.first.recipe_id == DOUBLED_ID
    world.script.append(fold_fakes.Script(selected=4, seconds=30_000.0))
    run(world, fold_plan, 8.0)
    refit_approval = entries("gate3_approval")[-1]
    assert refit_approval.seq > pilot_approval.seq
    registered = {
        entry.payload["recipe_id"]: entry.payload
        for entry in entries("fit")
        if entry.payload["status"] == "registered"
    }
    assert (registered[R1_ID]["approval_seq"], registered[R1_ID]["attempt"]) == (
        pilot_approval.seq,
        2,
    )
    assert (registered[DOUBLED_ID]["approval_seq"], registered[DOUBLED_ID]["attempt"]) == (
        refit_approval.seq,
        1,
    )
    assert plan(world).stage == "done"


def test_bookkeeping_of_a_killed_blinded_retry_records_attempt_2(world, monkeypatch):
    fold_fakes.write_bundle(R1)
    world.script += [fold_fakes.Script(status="failed", seconds=120.0), fold_fakes.Script()]
    with monkeypatch.context() as patch:
        kill_at_the_registered_fit_entry(patch)
        with pytest.raises(KeyboardInterrupt):
            run(world, plan(world, FOLD2), 10.0)
    (approval,) = entries("gate3_approval")
    fold_plan = plan(world, FOLD2)
    assert (fold_plan.stage, fold_plan.first) == ("bookkeeping", None)
    run(world, fold_plan, 10.0)  # nothing to compute: the figure is not recorded
    failed, registered = (entry.payload for entry in entries("fit"))
    assert (failed["status"], failed["attempt"]) == ("failed", 1)
    assert (registered["status"], registered["attempt"]) == ("registered", 2)
    assert registered["approval_seq"] == approval.seq and len(entries("gate3_approval")) == 1


def test_a_done_pilot_runs_nothing(world):
    pilot_done(world=world)
    fold_plan = plan(world)
    assert fold_plan.stage == "done"
    before = fold_fakes.ledger_text()
    outcome = run(world, fold_plan, 10.0)
    assert outcome.units == () and len(world.fits) == 1
    assert fold_fakes.ledger_text() == before  # no approval for nothing
    assert any("nothing to run" in line for line in outcome.lines)


def test_a_registered_fold_after_the_pilot_is_never_trained_again(world, monkeypatch):
    fold_fakes.write_bundle(R1)
    register_without_entries(world, monkeypatch, FOLD2, selected=3)
    (approval,) = entries("gate3_approval")
    fold_plan = plan(world, FOLD2)
    assert (fold_plan.stage, fold_plan.first) == ("bookkeeping", None)
    assert fold_plan.recorded is not None and fold_plan.recorded.recipe_id == R1_ID
    run(world, fold_plan)
    (fit,) = entries("fit")
    entry = registry.find_entry(registry.CheckpointRef(fit.payload["checkpoint_id"]))
    assert fit.payload["blinded"] is True and fit.payload["wall_seconds"] == 0.0
    assert fit.payload["fit_record_sha256"] == entry.document["fit_record_sha256"]
    assert (fit.payload["approval_seq"], fit.payload["attempt"]) == (approval.seq, 1)
    with pytest.raises(wf.FoldRefused, match="never trained again"):
        plan(world, FOLD2)


# -- a run registered under another ledger --------------------------------------------------------


def other_reports_root(monkeypatch, tmp_path) -> None:
    """Another reports root, with its own (empty) ledger and the same timing record, over the
    test's data and weights roots: a relocated reports root over this fine-tune root."""
    name = f"{timing.TIMING_STEM}.json"
    record = (fold_fakes.timing_directory() / name).read_bytes()
    other = tmp_path / "reports-b"
    (other / timing.FINETUNE_REPORTS_SUBDIR).mkdir(parents=True)
    (other / timing.FINETUNE_REPORTS_SUBDIR / name).write_bytes(record)
    monkeypatch.setenv(paths.REPORTS_DIR_ENV, str(other))
    assert ledger.read_ledger() == ()


def test_a_pilot_registered_under_another_ledger_is_never_adopted(world, monkeypatch, tmp_path):
    """The reviewer's case: the pilot approved, planned, trained and recorded under ledger A;
    ledger B, over the same fine-tune root, sees a registered pilot without its entries. B never
    planned it, so B refuses it (at plan time, nothing recorded) instead of adopting it without
    its planned entry and its gate-3 approval."""
    pilot_done(world=world)
    other_reports_root(monkeypatch, tmp_path)
    with pytest.raises(wf.FoldRefused, match="registered under another ledger") as excinfo:
        plan(world)
    message = str(excinfo.value)
    assert message.startswith(f"{PILOT_ID} was registered by run q-2024-07-")
    assert "never planned (no pilot planned entry of its recipe) before the run started" in message
    assert not ledger.ledger_path().exists() and len(world.fits) == 1
    with pytest.raises(wf.FoldRefused, match="train the pilot first"):
        wf.resolve_pilot()  # and the pilot check finds no pilot under B


def test_a_blinded_fold_registered_under_another_ledger_is_never_adopted(
    world, monkeypatch, tmp_path
):
    fold_fakes.write_bundle(R1)
    world.script.append(fold_fakes.Script())
    run(world, plan(world, FOLD2), 10.0)  # ledger A: the bundle, the approval, the fit
    other_reports_root(monkeypatch, tmp_path)
    fold_fakes.write_bundle(R1)  # B's own bundle, recorded after A's run started
    with pytest.raises(wf.FoldRefused, match="registered under another ledger"):
        plan(world, FOLD2)
    assert fold_fakes.ledger_kinds() == ["preregistration"] and len(world.fits) == 1


def test_the_trace_of_a_registered_run(world, monkeypatch, tmp_path):
    """``_registered_unit_trace`` on a blinded fold killed after its registration: the approval
    before its start is its trace; without it (above gate 3) or without the bundle before its
    start, the run is another ledger's; a run this invocation started is its own."""
    fold_fakes.write_bundle(R1)
    register_without_entries(world, monkeypatch, FOLD2, selected=3)
    bundle, approval = ledger.read_ledger()
    entry = registry.find_entry(registry.CheckpointRef("ks-btc-0ee0d123-q-2024-10"))
    fold = fold_for(FOLD2, R1)
    trace = wf._registered_unit_trace
    assert trace(fold, R1, entry, (bundle, approval), above_gate3=True) == (approval.seq, 1)
    with pytest.raises(wf.FoldRefused, match="never approved .no gate-3 approval"):
        trace(fold, R1, entry, (bundle,), above_gate3=True)
    assert trace(fold, R1, entry, (bundle,), above_gate3=False) == (None, 1)
    with pytest.raises(wf.FoldRefused, match="never pre-registered .no bundle of its recipe"):
        trace(fold, R1, entry, (approval,), above_gate3=True)
    run_id = entry.document["run_id"]
    own = wf._Invocation(anchor=2, approval_seq=approval.seq, runs=frozenset({run_id}))
    assert trace(fold, R1, entry, (), above_gate3=True, invocation=own) == (approval.seq, 1)
    later = wf._Invocation(anchor=0)  # entries after the invocation's anchor are not a trace
    with pytest.raises(wf.FoldRefused, match="never pre-registered"):
        trace(fold, R1, entry, (bundle, approval), above_gate3=True, invocation=later)
    failed = ledger.record_fit(  # a failure entry naming the registered run itself is ignored
        **{
            **lf_fit(fold, run_id),
            "status": "failed",
            "attempt": 1,
            "approval_seq": approval.seq,
        }
    )
    assert trace(fold, R1, entry, (bundle, approval, failed), above_gate3=True) == (
        approval.seq,
        1,
    )
    weird = replace(entry, document={**entry.document, "run_id": "q-2024-10-yesterday"})
    with pytest.raises(wf.FoldFailed, match="its start cannot be read"):
        trace(fold, R1, weird, (bundle, approval), above_gate3=True)
    # a bundle and an approval recorded after the run started (another ledger's) are no trace
    late = tmp_path / "late-ledger"
    after = wf._run_start(run_id, fold.run_label) + timedelta(hours=1)
    later_bundle = ledger.record_preregistration(  # the bundle file is the test's
        bundle_relpath=bundle.payload["bundle_relpath"],
        recipe_id=R1_ID,
        tests=["T-A", "T-B"],
        forecasting_code_sha256=bundle.payload["forecasting_code_sha256"],
        training_code_sha256=bundle.payload["training_code_sha256"],
        root=late,
        now=after,
    )
    later_approval = ledger.record_gate3_approval(
        command=wf.COMMAND,
        subject=fold.run_label,
        unit="mps_hours",
        approved=10.0,
        estimate=4.945,
        given_at=after,
        estimate_detail={},
        root=late,
        now=after,
    )
    with pytest.raises(wf.FoldRefused, match="never pre-registered"):
        trace(fold, R1, entry, (later_bundle, later_approval), above_gate3=True)


def lf_fit(fold, run_id: str) -> dict:
    """Keyword arguments of a ``ledger.record_fit`` of ``fold`` without a checkpoint."""
    return {
        "checkpoint_id": None,
        "registry_line_sha256": None,
        "recipe_id": R1_ID,
        "fold": fold.label,
        "replicate": fold.replicate,
        "run_id": run_id,
        "status": "failed",
        "attempt": 1,
        "blinded": True,
        "fit_record_sha256": None,
        "device": "mps",
        "threads": 2,
        "wall_seconds": 1.0,
        "cpu_seconds": 1.0,
        "approval_seq": None,
        "order_id": None,
    }


def test_a_state_that_moved_since_the_plan_is_refused(world):
    stale = plan(world)
    pilot_done(world=world)
    with pytest.raises(wf.FoldRefused, match="moved from stage train to done"):
        run(world, stale, 10.0)


def test_the_fold_lock_and_a_timing_run_are_refused(world):
    scratch = registry.scratch_dir()
    with exclusive_lock(scratch / wf.FOLD_LOCK_NAME), pytest.raises(LockBusyError):
        run(world, plan(world), 10.0)
    with exclusive_lock(scratch / timing.LOCK_NAME), pytest.raises(wf.FoldRefused, match="timing"):
        run(world, plan(world), 10.0)
    assert world.fits == [] and not ledger.ledger_path().exists()


# -- failures --------------------------------------------------------------------------------------


def test_a_failed_attempt_is_retried_once_with_the_same_seeds(world):
    world.script += [fold_fakes.Script(status="failed", seconds=120.0), fold_fakes.Script()]
    outcome = run(world, plan(world), 4.0)  # the cap, 5 MPS-hours, shapes both deadlines
    assert [unit.status for unit in outcome.units] == ["failed", "registered"]
    first, second = (call["request"] for call in world.fits)
    seeds = [
        training_seeds(recipe_id(item.recipe), item.fold.label, item.fold.replicate)
        for item in (first, second)
    ]
    assert seeds[0] == seeds[1] and first.export == second.export
    assert first.deadline_seconds == pytest.approx(5.0 * 3600)
    assert second.deadline_seconds == pytest.approx(5.0 * 3600 - 120.0)  # what the cap leaves
    failed, registered = entries("fit")
    assert (failed.payload["status"], failed.payload["attempt"]) == ("failed", 1)
    assert failed.payload["checkpoint_id"] is None and failed.payload["run_id"] is not None
    assert failed.payload["wall_seconds"] == 120.0
    assert (registered.payload["status"], registered.payload["attempt"]) == ("registered", 2)
    assert kinds() == ["gate3_approval", "exploration", "fit", "fit", "exploration"]


def test_a_second_failure_is_a_fold_failure_after_its_entries(world):
    world.script += [fold_fakes.Script(status="failed")] * 2
    with pytest.raises(wf.FoldFailed, match="after its retry"):
        run(world, plan(world), 10.0)
    assert [entry.payload["status"] for entry in entries("fit")] == ["failed", "failed"]
    assert [entry.payload["attempt"] for entry in entries("fit")] == [1, 2]
    assert kinds() == ["gate3_approval", "exploration", "fit", "fit"]


def test_a_child_without_a_result_is_retried(world):
    world.script += [fold_fakes.Script(status="no_result"), fold_fakes.Script()]
    outcome = run(world, plan(world), 10.0)
    assert [unit.status for unit in outcome.units] == ["no_result", "registered"]
    assert entries("fit")[0].payload["run_id"] is not None


@pytest.mark.parametrize(
    "status, message",
    [
        ("memory_abort", "not a fold failure"),
        ("time_abort", "is not retried"),
        ("killed", "is not retried"),
    ],
)
def test_memory_and_time_aborts_and_killed_children_are_not_retried(world, status, message):
    world.script.append(fold_fakes.Script(status=status))
    with pytest.raises(wf.FoldFailed, match=message):
        run(world, plan(world), 10.0)
    (fit,) = entries("fit")
    assert fit.payload["status"] == status and len(world.fits) == 1
    assert (fit.payload["run_id"] is None) is (status == "killed")


def test_the_retry_does_not_start_when_the_cap_does_not_admit_it(world):
    world.script.append(fold_fakes.Script(status="failed", seconds=1800.0))
    with pytest.raises(wf.FoldFailed, match="retry does not start"):
        run(world, plan(world), 4.0)  # cap 5.0: 0.5 spent + 4.945 > 5
    assert len(world.fits) == 1 and [e.payload["status"] for e in entries("fit")] == ["failed"]


def test_an_interrupted_child_is_recorded_before_the_interrupt_goes_on(world):
    world.script.append(fold_fakes.Script(interrupt=True))
    with pytest.raises(KeyboardInterrupt):
        run(world, plan(world), 10.0)
    (fit,) = entries("fit")
    assert fit.payload["status"] == "failed" and fit.payload["checkpoint_id"] is None
    # the parent's interrupt stops the child with SIGTERM, which the child records as SystemExit
    world.script.append(fold_fakes.Script(interrupt=True, error_type="SystemExit"))
    with pytest.raises(KeyboardInterrupt):
        run(world, plan(world), 10.0)
    assert entries("fit")[-1].payload["status"] == "killed"


@pytest.mark.parametrize("handled", [True, False])
def test_a_child_stopped_by_the_parents_timeout_is_a_time_abort_never_retried(world, handled):
    """``train.fit``'s timeout (the deadline + 120 s) sends SIGTERM: a child that handles it
    records status failed with error SystemExit; one still alive 30 s later is killed and writes
    no record. Either way the attempt reached its deadline: a time abort (design 3.5), not a
    failure to retry, whatever the record says."""
    status = "failed" if handled else "no_result"
    world.script.append(fold_fakes.Script(status=status, error_type="SystemExit", timeout=True))
    with pytest.raises(wf.FoldFailed, match="ran to its deadline of 35604 s") as excinfo:
        run(world, plan(world), 12.0)
    assert "is not retried: tell the owner" in str(excinfo.value)
    (fit,) = entries("fit")
    assert (fit.payload["status"], fit.payload["attempt"]) == ("time_abort", 1)
    assert fit.payload["run_id"] is not None and len(world.fits) == 1
    expected = 35_604.0 + wf.CHILD_TIMEOUT_MARGIN_SECONDS + train.KILL_GRACE_SECONDS
    assert fit.payload["wall_seconds"] == pytest.approx(expected, abs=1.0)


def test_a_child_a_signal_stopped_before_its_deadline_is_killed_never_retried(world):
    world.script.append(fold_fakes.Script(status="failed", error_type="SystemExit"))
    with pytest.raises(wf.FoldFailed, match="ended killed .fit entry #3. and is not retried"):
        run(world, plan(world), 10.0)
    assert [entry.payload["status"] for entry in entries("fit")] == ["killed"]
    assert len(world.fits) == 1


@pytest.mark.parametrize("label", [wf.PILOT_LABEL, FOLD2])
def test_a_run_registered_before_an_interrupt_is_never_recorded_as_a_failure(world, label):
    """``train.fit`` registers the checkpoint, then removes the run's selected/ directory: an
    interrupt there reaches the fold command with the run registered. No failure entry is
    recorded: its ``fit`` entry is recorded at once from the registry, registered, attempt 1,
    under this invocation's approval and with the spend measured around the child (a blinded
    fold's included: the next invocation could not measure it). The next invocation records the
    pilot's done entry, and finds a blinded fold registered and recorded."""
    if label == FOLD2:
        fold_fakes.write_bundle(R1)
    world.script.append(
        fold_fakes.Script(after_registration=KeyboardInterrupt("cleaning up"), seconds=7_200.0)
    )
    lines: list[str] = []
    with pytest.raises(KeyboardInterrupt):
        wf.run_fold(
            plan(world, label),
            approved_hours=10.0,
            given_at=GIVEN_AT,
            steps=world.steps(),
            progress=lines.append,
        )
    (approval,) = entries("gate3_approval")
    (fit,) = entries("fit")  # no failed fit entry
    assert (fit.payload["status"], fit.payload["attempt"]) == ("registered", 1)
    assert fit.payload["approval_seq"] == approval.seq
    assert fit.payload["wall_seconds"] == pytest.approx(7_200.0)
    entry = registry.find_entry(registry.CheckpointRef(fit.payload["checkpoint_id"]))
    assert fit.payload["fit_record_sha256"] == entry.document["fit_record_sha256"]
    assert fit.payload["registry_line_sha256"] == entry.line_sha256
    assert any(
        "registered its checkpoint before it was stopped: no failure is recorded; fit entry "
        f"#{fit.seq} records it from the registry" in line
        for line in lines
    )
    if label == FOLD2:
        assert kinds() == ["preregistration", "gate3_approval", "fit"]
        with pytest.raises(wf.FoldRefused, match="never trained again"):
            plan(world, label)
        return
    assert kinds() == ["gate3_approval", "exploration", "fit"]  # the approval, planned, the fit
    fold_plan = plan(world)
    assert fold_plan.stage == "bookkeeping" and fold_plan.first is None
    run(world, fold_plan)  # no compute: no figure needed
    assert kinds() == ["gate3_approval", "exploration", "fit", "exploration"]
    assert entries("exploration")[-1].payload["stage"] == "done"
    assert len(world.fits) == 1 and plan(world).stage == "done"


@pytest.mark.parametrize("label", [wf.PILOT_LABEL, FOLD2])
def test_a_run_registered_before_an_error_is_recorded_from_the_registry_at_once(world, label):
    """The same after an error: the fit entry from the registry with the measured spend (the
    blinded fold's too, whose record is never opened), then the pilot's done entry."""
    if label == FOLD2:
        fold_fakes.write_bundle(R1)
    error = train.ChildFailed("an error after the registration")
    world.script.append(fold_fakes.Script(after_registration=error, seconds=7_200.0))
    outcome = run(world, plan(world, label), 10.0)
    approval = entries("gate3_approval")[0]
    (fit,) = entries("fit")
    assert (fit.payload["status"], fit.payload["attempt"]) == ("registered", 1)
    assert fit.payload["approval_seq"] == approval.seq and len(world.fits) == 1
    assert fit.payload["wall_seconds"] == pytest.approx(7_200.0)
    assert outcome.spent_hours == pytest.approx(2.0)
    assert [(unit.status, unit.fit_seq) for unit in outcome.units] == [("registered", fit.seq)]
    if label == wf.PILOT_LABEL:
        assert kinds() == ["gate3_approval", "exploration", "fit", "exploration"]
    else:
        assert kinds() == ["preregistration", "gate3_approval", "fit"]


# -- a training child left running ----------------------------------------------------------------


def orphan(label: str = "q-2024-07-20261103T120000Z") -> tuple[int, str]:
    """A live training child of this fine-tune root, as the process list shows it."""
    return 4242, str(registry.runs_dir() / label)


def test_a_training_child_left_running_refuses_the_estimate_and_the_run(world, capsys):
    """A parent killed with SIGKILL leaves its child training (train.start_run puts it in its own
    session): the plan is refused naming the pid and the run, and so is a run planned before."""
    stale = plan(world)
    world.running.append(orphan())
    with pytest.raises(wf.FoldRefused, match="a training child is still running") as excinfo:
        plan(world)
    message = str(excinfo.value)
    assert "pid 4242, run q-2024-07-20261103T120000Z (this fine-tune root)" in message
    assert "nothing was started" in message and "/" + "Users" not in message
    with pytest.raises(wf.FoldRefused, match="pid 4242"):
        run(world, stale, 10.0)
    assert not ledger.ledger_path().exists() and world.fits == []
    world.running[:] = [(77, "/elsewhere/.runs/synthetic-x-20261103T120000Z"), (78, "")]
    with pytest.raises(wf.FoldRefused) as excinfo:
        plan(world)
    assert "pid 77, run synthetic-x-20261103T120000Z (another root)" in str(excinfo.value)
    assert "pid 78 (no run directory on its command line)" in str(excinfo.value)


def test_a_training_child_that_appears_during_the_run_stops_the_next_child(world):
    world.script.append(fold_fakes.Script(status="failed", seconds=120.0))
    original = world.fit

    def fit(request, **kwargs):  # another child starts while the first attempt runs
        world.running.append(orphan("q-2024-07-20261103T130000Z"))
        return original(request, **kwargs)

    world.fit = fit
    with pytest.raises(wf.FoldRefused, match="pid 4242, run q-2024-07-20261103T130000Z"):
        run(world, plan(world), 10.0)
    assert len(world.fits) == 1  # the retry did not start
    assert [entry.payload["status"] for entry in entries("fit")] == ["failed"]


def test_the_process_list_is_parsed_for_training_children():
    runs = "/x/weights/finetuned/.runs"
    text = "\n".join(
        [
            "    1 /sbin/launchd",
            f"  501 /venv/bin/python3 -B -P -m kronos_agent.finetune.runner --run-dir {runs}/q-a",
            f"  502 python -B -c import time -m kronos_agent.finetune.runner --run-dir {runs}/q b",
            "  503 uv run kronos-agent finetune fold --fold q-2024-07",
            "  504 python -m kronos_agent.finetune.runner",
            "  505 python -m kronos_agent.finetune.runnerX --run-dir /y",
            "  bad line",
            "",
        ]
    )
    assert wf.parse_training_children(text) == (
        (501, f"{runs}/q-a"),
        (502, f"{runs}/q b"),
        (504, ""),
    )


def test_the_real_process_list_shows_a_live_training_child(tmp_path):
    """A live process with a training child's command line (it only sleeps) is found by the real
    ``/bin/ps`` scan, with its run directory; this process is never one."""
    import subprocess
    import sys

    run_dir = tmp_path / ".runs" / "q-2024-07-20261103T120000Z"
    command = [sys.executable, "-B", "-c", "import time; time.sleep(60)"]
    command += ["-m", train.CHILD_MODULE, "--run-dir", str(run_dir)]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        found = wf.running_training_children()
        assert (process.pid, str(run_dir)) in found
        assert all(pid != os.getpid() for pid, _ in found)
    finally:
        process.kill()
        process.wait()
    assert (process.pid, str(run_dir)) not in wf.running_training_children()
    assert wf.real_steps().running_children is wf.running_training_children


def test_a_process_list_that_cannot_be_read_starts_nothing(monkeypatch):
    import subprocess

    def failing(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "ps: error")

    monkeypatch.setattr(wf.subprocess, "run", failing)
    with pytest.raises(train.TrainError, match="ps exit 1"):
        wf.running_training_children()

    def missing(*args, **kwargs):
        raise FileNotFoundError("no ps")

    monkeypatch.setattr(wf.subprocess, "run", missing)
    with pytest.raises(train.TrainError, match="FileNotFoundError"):
        wf.running_training_children()


# -- the figure and the deadline it gives -----------------------------------------------------------


def test_the_figure_text_states_the_deadline_each_figure_gives():
    """The refit (9.779 MPS-hours, 35,204.3 s): 7.83 is the smallest figure the 1.25 rule and
    the cap accept, and leaves a deadline 0.1 % above the worst case; twice the estimate (design
    3.5) needs 15.65. In the pilot's own invocation, after the pilot's 4.945, the refit starts
    only from 11.78 and keeps twice its estimate only from 19.61."""
    record = fold_fakes.timing_record()
    fold = fold_for(wf.PILOT_LABEL, R1)
    pilot, refit = wf.fit_estimate(R1, fold, record), wf.fit_estimate(DOUBLED, fold, record)
    text = wf.figure_text(refit)
    assert text.startswith("a figure H of at least 7.83 MPS-hours admits it; its deadline is ")
    assert "the smaller of 70409 s (twice the estimate, design 3.5) and 1.25 x H x 3600 s" in text
    assert (
        "35235 s at 7.83, 0.1% above its worst case of 35204 s; twice the estimate from 15.65"
        in (text)
    )
    assert wf.unit_deadline(refit, cap=1.25 * 7.83, spent=0.0) == pytest.approx(35_235.0)
    assert wf.unit_deadline(refit, cap=1.25 * 7.82, spent=0.0) is None
    assert wf.unit_deadline(refit, cap=1.25 * 15.65, spent=0.0) == pytest.approx(70_408.6, abs=0.1)
    both = wf.figure_text(refit, spent=pilot.device_hours)
    assert both.startswith("a figure H of at least 11.78 MPS-hours admits it after the 4.945")
    assert "(1.25 x H - 4.945) x 3600 s" in both and "twice the estimate from 19.61" in both
    assert wf.figure_text(pilot).startswith("a figure H of at least 3.96 MPS-hours admits it")
    cpu = wf.fit_estimate(
        R1,
        fold,
        fold_fakes.timing_record(device="cpu", threads=2, theta=None, mps_watermark_ratio=None),
    )
    assert "1.25 x H x 3600 s / 2 threads" in wf.figure_text(cpu)
    # the figure of the run that prints it: the deadline it would give, or that it is too small
    assert wf.figure_text(refit, approved=10.0).endswith(
        "; H = 10, this run's figure, would give it 45000 s, 27.8% above its worst case"
    )
    assert wf.figure_text(refit, approved=7.82).endswith(
        "; H = 7.82, this run's figure, does not admit it"
    )
    # a quotient a hair above a hundredth, where the rounding of the hundredths lands below it:
    # the next hundredth, which the cap and the 1.25 rule admit (never an assertion error)
    hours = 9.775000000004
    edge = replace(refit, device_hours=hours, wall_seconds=hours * 3600)
    assert wf.unit_deadline(edge, cap=1.25 * 7.82, spent=0.0) is None
    assert wf.figure_text(edge).startswith("a figure H of at least 7.83 MPS-hours admits it")
    assert wf.figure_text(edge, spent=1.0).startswith("a figure H of at least 8.63 MPS-hours")


def test_the_refit_refusal_and_the_estimate_lines_state_the_figures(world):
    lines = wf.estimate_lines(plan(world))
    text = "\n".join(lines)
    assert "the figure for q-2024-07: a figure H of at least 3.96 MPS-hours admits it" in text
    assert "the figure for the refit, if the pilot rule fires: in this invocation, a figure H " in (
        text
    )
    assert "of at least 11.78 MPS-hours admits it after the 4.945 MPS-hours spent" in text
    assert "own figure: a figure H of at least 7.83 MPS-hours admits it" in text
    world.script.append(
        fold_fakes.Script(selected=6, stop_reason="max_epochs", seconds=fold_fakes.PILOT_SECONDS)
    )
    with pytest.raises(wf.FoldRefused) as excinfo:
        run(world, plan(world), 10.0)
    message = str(excinfo.value)
    assert "with the figure they approve for the refit: a figure H of at least 7.83" in message
    assert "35235 s at 7.83, 0.1% above its worst case of 35204 s" in message
    assert "twice the estimate from 15.65" in message
    # what the figure this run had would give the refit in its own invocation
    assert message.endswith(
        "; H = 10, this run's figure, would give it 45000 s, 27.8% above its worst case"
    )
    refit_lines = "\n".join(wf.estimate_lines(plan(world)))
    assert "stage refit" in refit_lines
    assert "the figure for q-2024-07: a figure H of at least 7.83 MPS-hours" in refit_lines


def test_an_error_before_the_child_starts_records_no_fit(world, monkeypatch):
    def refuse(*args, **kwargs):
        raise registry.CheckpointExists("registered already (nothing was started)")

    monkeypatch.setattr(world, "fit", refuse)
    with pytest.raises(registry.CheckpointExists):
        run(world, plan(world), 10.0)
    assert "fit" not in kinds()


# -- blinding (T28) --------------------------------------------------------------------------------


def test_t28_a_fold_after_the_pilot_shows_no_loss_and_never_opens_its_record(world):
    fold_fakes.write_bundle(R1)
    world.script.append(fold_fakes.Script(selected=4, lock_record=True))
    lines: list[str] = []
    estimate = wf.estimate_lines(plan(world, FOLD2))
    try:
        outcome = wf.run_fold(
            plan(world, FOLD2),
            approved_hours=10.0,
            given_at=GIVEN_AT,
            steps=world.steps(),
            progress=lines.append,
        )
    finally:
        records = list(registry.runs_dir().glob("*/runner_result.json"))
        for record in records:
            with pytest.raises(PermissionError):  # the stand-in: any open would have failed
                record.read_bytes()
            fold_fakes.unlock(record)
    assert len(records) == 1
    shown = "\n".join([*estimate, *lines, *outcome.lines]) + fold_fakes.ledger_text()
    for value in fold_fakes.DISTINCTIVE:
        assert value not in shown, value
    assert "selected" not in "\n".join(lines) and "NLL4-TF 3." not in shown
    assert kinds() == ["preregistration", "gate3_approval", "fit"]  # no exploration entry
    (fit,) = entries("fit")
    entry = registry.find_entry(registry.CheckpointRef(fit.payload["checkpoint_id"]))
    assert fit.payload["blinded"] is True and world.fits[0]["request"].blinded is True
    assert fit.payload["fit_record_sha256"] == entry.document["fit_record_sha256"]
    (unit,) = outcome.units
    assert unit.shown is None and unit.checkpoint_id == "ks-btc-0ee0d123-q-2024-10"
    weights = next(item.sha256 for item in entry.files if item.name == "model.safetensors")
    assert any(weights[:12] in line and "finite True" in line for line in lines)


# -- sealed folds ----------------------------------------------------------------------------------


def record_r1_order() -> str:
    """The bundle of R1, its T-A and T-B readings one ISO week apart and the owner's order for
    the T-B candidate, dated before the current time (the fold's own entries follow at the
    current time; the ledger's times never decrease); returns the order id."""
    monday = datetime(2026, 10, 5, 12, tzinfo=UTC)  # 2026-W41
    fold_fakes.write_bundle(R1, now=monday)
    budget.spend_test("T-A", recipe_id=R1_ID, now=monday)
    budget.spend_test("T-B", recipe_id=R1_ID, now=monday + timedelta(days=7))
    order = ledger.record_order(
        scope="open_sealed",
        candidate=ledger.id_for_test("T-B", R1_ID),
        text="open it once",
        now=monday + timedelta(days=7),
    )
    return order.payload["order_id"]


def test_a_sealed_fold_reads_through_the_owners_recorded_order(world):
    order_id = record_r1_order()
    sealed = plan(world, "s-2026-09", order_id=order_id)
    assert sealed.fold.needs_sealed_access and sealed.stage == "train"
    run(world, sealed, 10.0)
    access = world.exports[-1]["access"]
    assert isinstance(access, SealedAccess) and access.order_id == order_id
    (fit,) = entries("fit")
    assert fit.payload["order_id"] == order_id and fit.payload["blinded"] is True


@pytest.mark.skipif(
    not (paths.REPO_ROOT / "data").is_dir(), reason="the repository has no data/ to protect"
)
def test_a_sealed_fold_never_opens_the_repositorys_data_on_a_relocated_ledgers_order(
    world, monkeypatch
):
    """The reviewer's case: the reports root (and the ledger with it) relocated, the data root
    left at the repository's data/: the order recorded in the relocated ledger is refused at
    plan time, so neither --estimate nor a run gets a plan, and no export starts. Only the
    metadata of the repository's data/ is read."""
    order_id = record_r1_order()
    monkeypatch.delenv(paths.DATA_DIR_ENV)
    for label in ("s-2026-09", "s-2026-08"):
        with pytest.raises(ledger.LedgerRefused, match="not the repository's reports/ledger"):
            plan(world, label, order_id=order_id)
    assert world.exports == [] and world.fits == []


@pytest.fixture
def repository(tmp_path, monkeypatch):
    """A stand-in repository whose ``data/`` is the data root (as an unset
    KRONOS_AGENT_DATA_DIR makes the real one), the ledger staying the test's own: a relocated
    reports root over the repository's data (``ledger.REPO_ROOT`` points at the stand-in)."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "reports" / "ledger").mkdir(parents=True)
    monkeypatch.setattr(ledger, "REPO_ROOT", repo)
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(repo / "data"))
    return repo


def test_a_blinded_fold_never_trains_on_the_repositorys_data_under_a_relocated_ledger(
    world, repository, monkeypatch, tmp_path
):
    """The reviewer's case: a bundle recorded in a relocated (scratch) ledger, the data root left
    at the repository's data/. Every fold after the pilot reads validation dates (q-2024-10's
    CSV ends 2024-09-09T16:00Z), so the bundle admits it only from a ledger that may authorize
    reading that data root: refused at plan time and, for a plan made on other roots, under the
    fold lock; the pilot (training-segment data) needs no authorization."""
    fold_fakes.write_bundle(R1)
    with pytest.raises(ledger.LedgerRefused, match="not the repository's reports/ledger") as info:
        plan(world, FOLD2)
    assert "a pre-registration bundle recorded in another ledger" in str(info.value)
    assert plan(world).stage == "train"  # the pilot reads training-segment data only
    own = tmp_path / "own-data"
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(own))  # a data root of its own: planned
    fold2 = plan(world, FOLD2)
    with pytest.raises(ledger.LedgerRefused, match="not the repository's reports/ledger"):
        wf.run_fold(
            fold2,
            approved_hours=10.0,
            given_at=GIVEN_AT,
            steps=world.steps(),
            data_root=repository / "data",
            progress=lambda line: None,
        )
    assert kinds() == ["preregistration"] and world.exports == [] and world.fits == []
    world.script.append(fold_fakes.Script())
    run(world, fold2, 10.0)
    assert kinds() == ["preregistration", "gate3_approval", "fit"]
    assert export_csv.fold_export_dir(recipe8(R1), FOLD2, root=own).is_dir()
    assert not any((repository / "data").iterdir())  # nothing exported to the repository's


# -- the pilot for the pilot check -----------------------------------------------------------------


def test_resolve_pilot_finds_the_recorded_pilot(world):
    with pytest.raises(wf.FoldRefused, match="train the pilot first"):
        wf.resolve_pilot()
    pilot_done(world=world)
    record = wf.resolve_pilot()
    (fit,) = entries("fit")
    assert (record.recipe, record.fold.label) == (R1, "q-2024-07")
    assert record.checkpoint.checkpoint_id == PILOT_ID
    assert record.checkpoint.line_sha256 == fit.payload["registry_line_sha256"]
    assert record.registry_entry.line_sha256 == fit.payload["registry_line_sha256"]
    assert record.fit_record_sha256 == fit.payload["fit_record_sha256"]
    assert sha256_file(record.fit_record_path) == record.fit_record_sha256
    assert record.export.content_sha256 == record.registry_entry.document["export_content_sha256"]
    assert record.export.directory == export_csv.fold_export_dir(recipe8(R1), "q-2024-07")


def test_resolve_pilot_refuses_a_changed_registry_line(world):
    pilot_done(world=world)
    entry = registry.find_entry(registry.CheckpointRef(PILOT_ID))
    ff.edit_last_line(registry.finetuned_dir(), **ff.earlier_data_end(entry, 24))
    with pytest.raises(registry.RegistryChainError, match="changed after it was recorded"):
        wf.resolve_pilot()


def test_resolve_pilot_refuses_a_changed_fold_record(world):
    pilot_done(world=world)
    record = wf.resolve_pilot().fit_record_path
    record.write_bytes(record.read_bytes() + b" ")
    with pytest.raises(wf.FoldFailed, match="no longer hashes"):
        wf.resolve_pilot()


def test_resolve_pilot_refuses_while_the_rule_fired_without_the_refit(world):
    world.script.append(
        fold_fakes.Script(selected=6, stop_reason="max_epochs", seconds=fold_fakes.PILOT_SECONDS)
    )
    with pytest.raises(wf.FoldRefused):
        run(world, plan(world), 10.0)
    with pytest.raises(wf.FoldRefused, match="the pilot rule fired: refit fold 1 first"):
        wf.resolve_pilot()
