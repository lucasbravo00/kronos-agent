"""Evaluation lengths at the batch layer (test T23; ``docs/RETRAINING.md`` 5.3).

A batch prepared with an evaluation length ``L`` builds windows of ``L`` targets and counts its
pending fans under the configuration with ``horizon_n = L``: its own ``config_id``, every other
key and value of the governed configuration unchanged. Lengths outside 1..24 are refused, the
embargo still bounds the governed ``N``, and a task whose ``L`` targets reach past the horizon
purge refuses the whole batch (samples are purged, never shortened: the last 24-step validation
decision is 2026-07-09T16:00Z). ``run_fan_batch`` stores fans of the prepared length under that
configuration, never another, and ``timing_window`` and ``measure_seconds_per_fan`` time the
length asked for. ``forecast_window``'s own length checks are in ``test_decision.py``.

Stores are synthetic and fans come from a stand-in model (``finetune_fakes.walk_models``),
except in the test marked ``weights``: a real Kronos-small 24-step batch records
``horizon_n = 24`` and stores its fans under its own ``config_id``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import finetune_fakes as ff
import numpy as np
import pytest
import step2_fakes as fakes

from kronos_agent.data.forecast_store import (
    CONFIG_KEYS,
    config_id,
    fan_relpath,
    forecast_config,
    load_fan,
)
from kronos_agent.data.partition import PartitionError
from kronos_agent.data.timeutil import HOUR_MS, to_ms
from kronos_agent.kronos import batch
from kronos_agent.kronos.batch import (
    BatchError,
    FanTask,
    measure_seconds_per_fan,
    prepare_tasks,
    registry_config,
    run_fan_batch,
    timing_window,
)
from kronos_agent.kronos.decision import evaluation_params
from kronos_agent.params import load_params

HOUR = timedelta(hours=1)
D1 = datetime(2025, 3, 3, 12, tzinfo=UTC)
BINGX = fakes.synthetic_candles(to_ms(D1) - 600 * HOUR_MS, 700, seed=3)
TASKS = (
    FanTask("bingx", D1),
    FanTask("bingx", D1, "mismatch-floor"),
    FanTask("bingx", D1 + 7 * HOUR),
)
# the last validation decision whose 24 targets fit the horizon purge (docs/DATA.md)
PURGE_LAST = datetime(2026, 7, 9, 16, tzinfo=UTC)
PURGE_CANDLES = fakes.synthetic_candles(to_ms(PURGE_LAST) - 600 * HOUR_MS, 601, seed=11)
PARAMS = load_params()
KRONOS = PARAMS.kronos
SMALL_PARAMS = replace(PARAMS, kronos=replace(KRONOS, trajectories=4))


@pytest.fixture
def root(tmp_path):
    fakes.write_bingx_store(tmp_path, BINGX)
    return tmp_path


def test_a_batch_of_the_evaluation_length_has_its_own_configuration(root):
    governed = prepare_tasks(TASKS, params=PARAMS, root=root)
    long = prepare_tasks(TASKS, params=PARAMS, root=root, evaluation_length=24)
    assert (long.evaluation_length, governed.evaluation_length) == (24, None)
    assert [window.horizon for window in long.windows] == [24] * len(TASKS)
    assert [window.horizon for window in governed.windows] == [KRONOS.horizon_n] * len(TASKS)
    for short, longer in zip(governed.windows, long.windows, strict=True):
        assert (short.sha256, short.inputs_sha256) == (longer.sha256, longer.inputs_sha256)
    assert long.config == registry_config(evaluation_params(PARAMS, 24).kronos)
    assert (long.config["horizon_n"], governed.config["horizon_n"]) == (24, KRONOS.horizon_n)
    assert tuple(long.config) == tuple(governed.config) == CONFIG_KEYS
    assert {key for key in CONFIG_KEYS if long.config[key] != governed.config[key]} == {"horizon_n"}
    assert long.config_id != governed.config_id
    assert (long.pending, governed.pending) == (len(TASKS), len(TASKS))
    same = prepare_tasks(TASKS, params=PARAMS, root=root, evaluation_length=KRONOS.horizon_n)
    assert same.config_id == governed.config_id  # L = N is the governed configuration


def test_lengths_outside_1_to_24_refuse_the_batch(root):
    with pytest.raises(PartitionError, match="EVAL_LENGTH_MAX = 24"):
        prepare_tasks(TASKS, params=PARAMS, root=root, evaluation_length=25)
    for bad in (0, -1, True, 4.0, "24"):
        with pytest.raises(ValueError, match="an evaluation length must be a positive integer"):
            prepare_tasks(TASKS, params=PARAMS, root=root, evaluation_length=bad)
    nine = replace(PARAMS, kronos=replace(KRONOS, horizon_n=9))  # 512 + 9 > 520
    with pytest.raises(PartitionError, match="embargo"):  # the governed N, whatever the length
        prepare_tasks(TASKS, params=nine, root=root, evaluation_length=4)


def test_the_horizon_purge_refuses_the_whole_batch(tmp_path):
    fakes.write_bingx_store(tmp_path, PURGE_CANDLES)
    last = prepare_tasks(
        [FanTask("bingx", PURGE_LAST)], params=PARAMS, root=tmp_path, evaluation_length=24
    )
    assert [window.horizon for window in last.windows] == [24]
    later = PURGE_LAST + HOUR
    with pytest.raises(PartitionError, match="past the horizon purge") as info:
        prepare_tasks(
            [FanTask("bingx", PURGE_LAST), FanTask("bingx", later)],
            params=PARAMS,
            root=tmp_path,
            evaluation_length=24,
        )
    assert (
        "the bingx sample of 2026-07-09T17:00Z with an evaluation length of 24 reads targets up "
        "to 2026-07-10T16:00Z" in str(info.value)
    )
    assert "never shortened" in str(info.value)
    shorter = prepare_tasks(
        [FanTask("bingx", later)], params=PARAMS, root=tmp_path, evaluation_length=23
    )
    assert [window.horizon for window in shorter.windows] == [23]
    governed = prepare_tasks([FanTask("bingx", later)], params=PARAMS, root=tmp_path)
    assert [window.horizon for window in governed.windows] == [KRONOS.horizon_n]


def test_run_fan_batch_stores_fans_of_the_prepared_length(root):
    predictor = ff.WalkPredictor()
    models = ff.walk_models(predictor=predictor)
    long = prepare_tasks(TASKS, params=PARAMS, root=root, evaluation_length=24)
    summaries = run_fan_batch(long, params=PARAMS, root=root, models=models)
    assert [call["pred_len"] for call in predictor.calls] == [24] * len(TASKS)
    for task, summary in zip(TASKS, summaries, strict=True):
        assert summary.relpath == fan_relpath(
            task.source, long.config_id, task.stream, task.decision_hour
        )
        stored = load_fan(root / summary.relpath)
        assert stored.paths.shape == (KRONOS.trajectories, 24, 6)
        assert (stored.meta["horizon"], stored.meta["config"]["horizon_n"]) == (24, 24)
        assert stored.meta["config_id"] == long.config_id
    assert prepare_tasks(TASKS, params=PARAMS, root=root, evaluation_length=24).pending == 0
    governed = prepare_tasks(TASKS, params=PARAMS, root=root)
    assert governed.pending == len(TASKS)  # the 24-step fans are not the governed ones
    stored_governed = run_fan_batch(governed, params=PARAMS, root=root, models=ff.walk_models())
    assert {s.relpath for s in stored_governed}.isdisjoint({s.relpath for s in summaries})
    for changed in (replace(long, evaluation_length=None), replace(long, evaluation_length=8)):
        with pytest.raises(BatchError, match="the batch was prepared for forecast config"):
            run_fan_batch(changed, params=PARAMS, root=root, models=models)


def test_timing_takes_the_length_it_is_asked_for(monkeypatch):
    window = timing_window(PARAMS, evaluation_length=24)
    assert (window.horizon, window.decision_hour) == (24, batch.TIMING_DECISION_HOUR)
    assert timing_window(PARAMS).horizon == KRONOS.horizon_n
    clock = {"now": 100.0}
    monkeypatch.setattr(batch, "_clock", lambda: clock["now"])

    class Clocked(ff.WalkPredictor):
        def predict_batch(self, *args, **kwargs):
            clock["now"] += 2.0
            return super().predict_batch(*args, **kwargs)

    predictor = Clocked()
    seconds = measure_seconds_per_fan(
        params=PARAMS, models=ff.walk_models(predictor=predictor), evaluation_length=24, repeats=3
    )
    assert seconds == 2.0
    assert [call["pred_len"] for call in predictor.calls] == [24] * 4  # a warm-up and 3 timed


@pytest.mark.weights
def test_a_real_24_step_batch_records_its_length_and_its_own_config(loaded_small, root):
    """Real Kronos-small fans (4 trajectories) of a 24-step batch, in this process."""
    tasks = TASKS[:1]
    long = prepare_tasks(tasks, params=SMALL_PARAMS, root=root, evaluation_length=24)
    expected = forecast_config(loaded_small, evaluation_params(SMALL_PARAMS, 24).kronos)
    assert long.config == expected and long.config_id == config_id(expected)
    (summary,) = run_fan_batch(long, params=SMALL_PARAMS, root=root, models=loaded_small)
    stored = load_fan(root / summary.relpath)
    assert stored.meta["config"]["horizon_n"] == 24 and stored.meta["horizon"] == 24
    assert stored.meta["config_id"] == long.config_id
    assert long.config_id != config_id(forecast_config(loaded_small, SMALL_PARAMS.kronos))
    assert stored.paths.shape == (4, 24, 6) and np.isfinite(stored.paths).all()
    assert summary.relpath == fan_relpath("bingx", long.config_id, "forecast", D1)
