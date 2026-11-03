"""Model periods at the batch layer (test T6, model part; ``docs/RETRAINING.md`` 1.4, 3.6, 5.3).

``prepare_tasks`` refuses, for the whole batch, a decision hour the model may have seen: the
shipped weights before 2024-07-01, a fine-tuned checkpoint before the later of that and the end
of its training data (``data_end_ms``, read from its registry entry). The named opt-in roles are
recorded in the prepared batch, and so is the checkpoint, with the sha256 of the registry line
read: an entry changed afterwards (an earlier data end, which the hash chain cannot see on the
registry's last line) is refused when the batch runs. ``run_fan_batch`` and its spawn-started
workers forecast with the checkpoint, length and role the batch was prepared with, never others.

The checkpoints are stand-in registry entries in a temporary fine-tune root (no weights:
preparing reads only the entry), the fans come from stand-in models that report an entry's
hashes and data end (``finetune_fakes.walk_models``), and the BingX store is synthetic; no worker
process is started. ``forecast_window``'s own model-period checks are in ``test_decision.py``,
the ``require_model_period`` table in ``test_partition.py``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import finetune_fakes as ff
import pytest
import step2_fakes as fakes

from kronos_agent.data.forecast_store import config_id, load_fan
from kronos_agent.data.partition import KRONOS_FIRST_DECISION, PartitionError
from kronos_agent.data.timeutil import to_ms, utc_text
from kronos_agent.finetune import registry
from kronos_agent.finetune.registry import CheckpointRef, RegistryChainError
from kronos_agent.kronos import batch
from kronos_agent.kronos.batch import (
    BatchError,
    FanTask,
    prepare_tasks,
    registry_config,
    run_fan_batch,
)
from kronos_agent.params import load_params

HOUR = timedelta(hours=1)
PILOT_END = datetime(2024, 6, 9, 16, tzinfo=UTC)  # wf-2024-07's CSV end, before the cutoff
LATER_END = datetime(2025, 3, 10, 16, tzinfo=UTC)  # wf-2025-04's CSV end
IN_SAMPLE = datetime(2025, 2, 20, 8, tzinfo=UTC)  # inside the later checkpoint's training data
PILOT_ID = "ks-btc-0123abcd-wf-2024-07"
LATER_ID = "ks-btc-0123abcd-wf-2025-04"
STORE_FIRST = datetime(2024, 6, 9, tzinfo=UTC)
PARAMS = load_params()
KRONOS = PARAMS.kronos


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """BingX 1h candles from 2024-06-09T00:00Z to the later checkpoint's data end (excluded)."""
    root = tmp_path_factory.mktemp("store")
    count = int((LATER_END - STORE_FIRST) / HOUR)
    fakes.write_bingx_store(root, fakes.synthetic_candles(to_ms(STORE_FIRST), count, seed=21))
    return root


@pytest.fixture
def world(tmp_path, store):
    """Two stand-in checkpoints in a temporary fine-tune root (the later one is its last line)."""
    ft = tmp_path / "ft"
    pilot = ff.install_stand_in(ft, PILOT_ID, data_end_ms=to_ms(PILOT_END))
    later = ff.install_stand_in(ft, LATER_ID, data_end_ms=to_ms(LATER_END))
    return SimpleNamespace(
        data=tmp_path / "data",
        store=store,
        ft=ft,
        pilot=pilot,
        later=later,
        pilot_ref=CheckpointRef(PILOT_ID, ft),
        later_ref=CheckpointRef(LATER_ID, ft),
    )


def prepare(world, hours, *, checkpoint=None, role="evaluation", length=None):
    return prepare_tasks(
        [FanTask("bingx", hour) for hour in hours],
        params=PARAMS,
        root=world.store,
        evaluation_length=length,
        checkpoint=checkpoint,
        period_role=role,
    )


def test_a_checkpoint_batch_starts_at_the_end_of_its_training_data(world):
    with pytest.raises(PartitionError, match="the first hour this checkpoint may forecast") as info:
        prepare(world, [LATER_END - HOUR], checkpoint=world.later_ref)
    assert f"trained on candles before {utc_text(to_ms(LATER_END))}" in str(info.value)
    with pytest.raises(PartitionError):  # one hour too early refuses the whole batch
        prepare(world, [LATER_END, LATER_END - HOUR], checkpoint=world.later_ref)
    prepared = prepare(world, [LATER_END], checkpoint=world.later_ref)
    assert [task.decision_hour for task in prepared.tasks] == [LATER_END]
    assert prepared.pending == 1 and prepared.dropped == ()
    # the batch keeps the entry it read: id, root and line sha256
    assert prepared.checkpoint == CheckpointRef(
        LATER_ID, world.ft, line_sha256=world.later.line_sha256
    )
    assert (prepared.period_role, prepared.evaluation_length) == ("evaluation", None)
    assert prepared.config_id == config_id(registry_config(KRONOS, checkpoint=world.later_ref))
    assert prepared.config["weights_sha256"] == world.later.weights_sha256()
    shipped = prepare(world, [LATER_END - HOUR])  # the shipped weights may forecast that hour
    assert shipped.checkpoint is None
    assert shipped.config_id == config_id(registry_config(KRONOS)) != prepared.config_id


def test_a_checkpoint_inherits_the_cutoff_of_the_shipped_weights(world):
    """The pilot checkpoint's data ends before July 2024, but it starts from the shipped weights,
    which saw Binance up to June 2024."""
    for checkpoint, words in ((None, "June 2024"), (world.pilot_ref, "2024-07-01T00:00Z, the")):
        with pytest.raises(PartitionError) as info:
            prepare(world, [KRONOS_FIRST_DECISION - HOUR], checkpoint=checkpoint)
        assert words in str(info.value)
        prepared = prepare(world, [KRONOS_FIRST_DECISION], checkpoint=checkpoint)
        assert [task.decision_hour for task in prepared.tasks] == [KRONOS_FIRST_DECISION]


def test_the_opt_in_roles_are_checked_and_recorded_in_the_batch(world):
    with pytest.raises(PartitionError, match="the first hour this checkpoint may forecast"):
        prepare(world, [IN_SAMPLE], checkpoint=world.later_ref)  # the default "evaluation"
    for role in ("pilot-check", "timing"):
        prepared = prepare(world, [IN_SAMPLE], checkpoint=world.later_ref, role=role)
        assert prepared.period_role == role and len(prepared.tasks) == 1
    with pytest.raises(PartitionError, match="reads the shipped weights only"):
        prepare(world, [IN_SAMPLE], checkpoint=world.later_ref, role="cutoff-check")
    before_cutoff = prepare(world, [KRONOS_FIRST_DECISION - HOUR], role="cutoff-check")
    assert before_cutoff.period_role == "cutoff-check" and before_cutoff.checkpoint is None
    for role in ("pilot", "", None):
        with pytest.raises(ValueError, match="period_role must be one of"):
            prepare(world, [LATER_END], role=role)


def test_run_fan_batch_forecasts_as_the_batch_was_prepared(world):
    """The role, the checkpoint and the models come from the prepared batch: the in-sample hour
    of a pilot-check batch is forecast, the same batch under the default role is refused, and
    models of the shipped weights or of another checkpoint are refused."""
    prepared = prepare(world, [IN_SAMPLE], checkpoint=world.later_ref, role="pilot-check")
    predictor = ff.WalkPredictor()
    models = ff.walk_models(world.later, predictor=predictor)
    evaluation = replace(prepared, period_role="evaluation")
    with pytest.raises(PartitionError, match="the first hour this checkpoint may forecast"):
        run_fan_batch(evaluation, params=PARAMS, root=world.data, models=models)
    assert predictor.calls == [] and not world.data.exists()
    for others, shown in (
        (ff.walk_models(), "the shipped weights"),
        (ff.walk_models(world.pilot), PILOT_ID),
    ):
        with pytest.raises(BatchError, match=f"the models are {shown}, the batch runs {LATER_ID}"):
            run_fan_batch(prepared, params=PARAMS, root=world.data, models=others)
    with pytest.raises(BatchError, match=f"the models are {LATER_ID}, the batch runs the shipped"):
        run_fan_batch(
            replace(prepared, checkpoint=None), params=PARAMS, root=world.data, models=models
        )
    (summary,) = run_fan_batch(prepared, params=PARAMS, root=world.data, models=models)
    assert summary.computed and len(predictor.calls) == 1
    stored = load_fan(world.data / summary.relpath)
    assert stored.meta["config_id"] == prepared.config_id
    assert stored.meta["weights_sha256"] == world.later.weights_sha256()
    assert stored.meta["horizon"] == KRONOS.horizon_n


class PoolReached(Exception):
    pass


def test_the_workers_get_the_prepared_checkpoint_length_and_role(world, monkeypatch):
    prepared = prepare(
        world,
        [IN_SAMPLE, IN_SAMPLE + 7 * HOUR],
        checkpoint=world.later_ref,
        role="pilot-check",
        length=24,
    )
    seen: dict = {}
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(batch, "_physical_memory_bytes", lambda: 8 * 2**30)

    def parent_checks(model, checkpoint=None):
        seen["parent"] = (model, checkpoint)

    def pool(**kwargs):
        seen["pool"] = kwargs
        raise PoolReached

    monkeypatch.setattr(batch, "_check_parent_prerequisites", parent_checks)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", pool)
    with pytest.raises(PoolReached):
        run_fan_batch(prepared, workers=2, params=PARAMS, root=world.data)
    assert seen["parent"] == ("small", prepared.checkpoint)
    assert seen["pool"]["initializer"] is batch._init_worker
    assert seen["pool"]["initargs"] == (
        PARAMS,
        prepared.config_id,
        24,
        prepared.checkpoint,
        "pilot-check",
    )
    assert prepared.checkpoint.line_sha256 == world.later.line_sha256


def test_a_worker_forecasts_as_it_was_initialised(world, monkeypatch):
    """``_init_worker`` (run here, in this process, with the model loader stubbed) loads the
    batch's checkpoint, checks the configuration of the prepared length, and its fans have that
    length and that role."""
    prepared = prepare(
        world, [IN_SAMPLE], checkpoint=world.later_ref, role="pilot-check", length=24
    )
    predictor = ff.WalkPredictor()
    loads = []

    def load(model, *, checkpoint=None):
        loads.append((model, checkpoint))
        return ff.walk_models(world.later, predictor=predictor)

    monkeypatch.setattr(batch, "_load_models_quietly", load)
    monkeypatch.setattr(batch, "_WORKER", {})
    batch._init_worker(PARAMS, prepared.config_id, 24, prepared.checkpoint, "pilot-check")
    assert loads == [("small", prepared.checkpoint)]
    assert (batch._WORKER["evaluation_length"], batch._WORKER["period_role"]) == (24, "pilot-check")
    fan = batch._worker_forecast(prepared.windows[0], "forecast")
    assert fan.paths.shape == (KRONOS.trajectories, 24, 6)
    assert predictor.calls[-1]["pred_len"] == 24
    governed = config_id(registry_config(KRONOS, checkpoint=world.later_ref))
    with pytest.raises(BatchError, match="the batch was prepared for"):
        batch._init_worker(PARAMS, governed, 24, prepared.checkpoint, "pilot-check")


def test_an_entry_changed_after_the_preparation_is_refused(world, monkeypatch):
    """An earlier data end written into the registry's last line keeps the chain valid; the
    batch, which kept the line it was prepared with, refuses it before any model is loaded."""
    prepared = prepare(world, [LATER_END], checkpoint=world.later_ref)
    ff.edit_last_line(world.ft, **ff.earlier_data_end(world.later, 24))
    assert registry.registry_status(world.ft)["chain_ok"] is True

    def never(*args, **kwargs):
        raise AssertionError("no model may be loaded for a changed entry")

    monkeypatch.setattr(batch, "load_models", never)
    with pytest.raises(RegistryChainError, match="changed after it was recorded"):
        run_fan_batch(prepared, params=PARAMS, root=world.data)
    recorded = CheckpointRef(LATER_ID, world.ft, line_sha256=world.later.line_sha256)
    with pytest.raises(RegistryChainError, match="changed after it was recorded"):
        prepare(world, [LATER_END], checkpoint=recorded)
    # without the recorded line, the edited entry would open hours of its own training data
    assert prepare(world, [LATER_END - HOUR], checkpoint=world.later_ref).tasks
    assert not world.data.exists()
