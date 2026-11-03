"""Fan batches: windows built in the parent, dropped incomplete hours, gate 3, the workers.

Stores are synthetic (``step2_fakes``) under ``tmp_path``. The tests without weights inject a
stand-in model (``models=``): a random walk drawn from numpy's global generator, which
``seed_everything`` seeds before every fan, reporting the registered weight hashes of
Kronos-small so that the batch configuration is the real one. The tests marked ``weights`` run
real Kronos-small fans (4 trajectories instead of the governed 20: the properties tested do not
depend on the fan size) through ``forecast_window`` in this process, a one-worker batch and a
two-worker spawn batch, each with another task order.
"""

from __future__ import annotations

import ast
import math
import multiprocessing
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes

from kronos_agent.data.forecast_store import (
    ForecastStoreError,
    check_reusable,
    config_id,
    fan_relpath,
    forecast_config,
    load_fan,
    save_fan,
)
from kronos_agent.data.partition import (
    KRONOS_FIRST_DECISION,
    SEALED_START,
    PartitionError,
    SealedPartitionError,
)
from kronos_agent.data.store import StoreMissingError
from kronos_agent.data.timeutil import from_ms, to_ms, utc_text
from kronos_agent.data.windows import build_window
from kronos_agent.kronos import batch
from kronos_agent.kronos.batch import (
    GATE3_APPROVAL_TOLERANCE,
    GATE3_CPU_SECONDS,
    TIMING_DECISION_HOUR,
    TIMING_STREAM,
    BatchError,
    FanTask,
    Gate3Refused,
    PreparedBatch,
    estimate,
    max_workers,
    measure_seconds_per_fan,
    prepare_tasks,
    registered_weights_sha256,
    registry_config,
    require_gate3,
    run_fan_batch,
    timing_window,
)
from kronos_agent.kronos.decision import (
    fan_mean_close,
    fan_p_up,
    fan_signal,
    forecast_window,
)
from kronos_agent.kronos.features import FEATURE_COLUMNS
from kronos_agent.kronos.weights import LoadedModels
from kronos_agent.models import spec_for
from kronos_agent.params import load_params
from kronos_agent.paths import REPO_ROOT

HOUR_MS = 3_600_000
D1 = datetime(2025, 3, 3, 12, tzinfo=UTC)
D1_MS = to_ms(D1)
D2 = D1 + timedelta(hours=7)
FIRST_MS = D1_MS - 600 * HOUR_MS
COUNT = 700
BINGX = fakes.synthetic_candles(FIRST_MS, COUNT, seed=3)
BINANCE_FRAME = fakes.binance_frame(FIRST_MS, COUNT, interval="1h", seed=4)
PARAMS = load_params()
KRONOS = PARAMS.kronos
SMALL_PARAMS = replace(PARAMS, kronos=replace(KRONOS, trajectories=4))
TASKS = (
    FanTask("bingx", D1),
    FanTask("binance", D1),
    FanTask("bingx", D1, "mismatch-floor"),
    FanTask("bingx", D2),
)


class WalkPredictor:
    """``predict_batch`` stand-in: a seeded random walk from the last close of each context."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict_batch(
        self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p, **_
    ):
        self.calls.append({"batch": len(df_list), "rows": len(df_list[0]), "pred_len": pred_len})
        frames = []
        for df, y in zip(df_list, y_timestamp_list, strict=True):
            last = float(df["close"].iloc[-1])
            close = last * np.exp(np.cumsum(np.random.normal(0.0, 0.01, size=pred_len)))
            values = np.column_stack(
                [close, close * 1.001, close * 0.999, close, np.ones(pred_len), close]
            )
            frames.append(pd.DataFrame(values, columns=list(FEATURE_COLUMNS), index=y))
        return frames


class FailingPredictor(WalkPredictor):
    """Fails on its call number ``fail_on`` (counted from 0)."""

    def __init__(self, fail_on: int) -> None:
        super().__init__()
        self.fail_on = fail_on

    def predict_batch(self, *args, **kwargs):
        if len(self.calls) == self.fail_on:
            raise RuntimeError("model failure")
        return super().predict_batch(*args, **kwargs)


class ClockedPredictor(WalkPredictor):
    """Advances a fake clock by the next of ``steps`` seconds on every call."""

    def __init__(self, clock: dict, steps) -> None:
        super().__init__()
        self.clock = clock
        self.steps = list(steps)

    def predict_batch(self, *args, **kwargs):
        self.clock["now"] += self.steps.pop(0)
        return super().predict_batch(*args, **kwargs)


def fake_models(predictor=None, weights=None) -> LoadedModels:
    return LoadedModels(
        spec=spec_for("small"),
        tokenizer=None,
        model=None,
        predictor=predictor if predictor is not None else WalkPredictor(),
        weights_sha256=dict(registered_weights_sha256("small") if weights is None else weights),
        load_seconds=0.0,
    )


def write_stores(root, *, bingx=BINGX, binance=BINANCE_FRAME):
    fakes.write_bingx_store(root, bingx)
    fakes.write_binance_store(root, binance)
    return root


def key_of(task: FanTask) -> tuple[str, str, str]:
    return task.source, utc_text(to_ms(task.decision_hour)), task.stream


def keys_of(summaries) -> list[tuple[str, str, str]]:
    return [(s.source, s.decision_hour, s.stream) for s in summaries]


def npz_files(root) -> list:
    return sorted((root / "forecasts").rglob("*.npz")) if (root / "forecasts").exists() else []


@pytest.fixture
def root(tmp_path):
    return write_stores(tmp_path)


# -- tasks and configuration -----------------------------------------------------------------------


def test_fan_task_normalises_its_hour_and_validates():
    task = FanTask("bingx", "2025-03-03T12:00Z")
    assert task.decision_hour == D1 and task.stream == "forecast"
    assert task == FanTask("bingx", D1) and hash(task) == hash(FanTask("bingx", D1))
    assert FanTask("bingx", D1, "mismatch-floor") != task
    with pytest.raises(ValueError, match="source"):
        FanTask("demo", D1)
    with pytest.raises(ValueError, match="stream"):
        FanTask("bingx", D1, "a/b")
    with pytest.raises(ValueError):
        FanTask("bingx", D1 + timedelta(minutes=30))
    with pytest.raises(ValueError):
        FanTask("bingx", datetime(2025, 3, 3, 12))  # naive


def test_the_registry_gives_the_configuration_of_the_loaded_models():
    weights = registered_weights_sha256("small")
    assert set(weights) == {
        "NeoQuasar/Kronos-small/config.json",
        "NeoQuasar/Kronos-small/model.safetensors",
        "NeoQuasar/Kronos-Tokenizer-base/config.json",
        "NeoQuasar/Kronos-Tokenizer-base/model.safetensors",
    }
    assert all(len(value) == 64 for value in weights.values())
    assert registry_config(KRONOS) == forecast_config(fake_models(), KRONOS)


# -- preparation -----------------------------------------------------------------------------------


def test_prepare_tasks_builds_every_window_in_task_order(root):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    assert prepared.tasks == TASKS
    assert prepared.dropped == ()
    assert prepared.pending == 4
    assert prepared.config == registry_config(KRONOS)
    assert prepared.config_id == config_id(registry_config(KRONOS))
    for task, window in zip(prepared.tasks, prepared.windows, strict=True):
        assert (window.source, window.decision_hour) == (task.source, task.decision_hour)
        assert (window.max_context, window.horizon) == (512, KRONOS.horizon_n)
        assert window.amount_policy == KRONOS.amount_policy
    bingx_d1 = build_window(BINGX, D1, source="bingx", horizon=KRONOS.horizon_n)
    assert prepared.windows[0].sha256 == bingx_d1.sha256
    assert prepared.windows[0].inputs_sha256 == bingx_d1.inputs_sha256
    assert prepared.windows[2] is prepared.windows[0]  # the floor stream reads the same window
    binance_d1 = build_window(
        fakes.frame_to_candles(BINANCE_FRAME), D1, source="binance", horizon=KRONOS.horizon_n
    )
    assert prepared.windows[1].sha256 == binance_d1.sha256
    assert prepare_tasks([], params=PARAMS, root=root).tasks == ()


def test_prepare_tasks_drops_incomplete_windows_with_the_missing_ranges(tmp_path):
    gap = D1_MS + 20 * HOUR_MS
    write_stores(tmp_path, bingx=[c for c in BINGX if c.open_ms != gap])
    before_store = FIRST_MS + 511 * HOUR_MS  # its context starts one hour before the store
    tasks = [
        FanTask("bingx", from_ms(before_store)),
        FanTask("bingx", from_ms(FIRST_MS + 512 * HOUR_MS)),  # the first complete window
        FanTask("bingx", from_ms(gap)),  # the missing candle opens at D: never needed
        FanTask("bingx", from_ms(gap + HOUR_MS)),  # the missing candle opens at D - 1h
        FanTask("bingx", from_ms(gap + HOUR_MS), "mismatch-floor"),
        FanTask("binance", from_ms(gap + HOUR_MS)),  # the other source is complete
    ]
    prepared = prepare_tasks(tasks, params=PARAMS, root=tmp_path)
    assert prepared.tasks == (tasks[1], tasks[2], tasks[5])
    assert prepared.pending == 3
    assert [w.decision_hour for w in prepared.windows] == [t.decision_hour for t in prepared.tasks]
    first, second, third = prepared.dropped
    for record in prepared.dropped:
        assert "misses 1 of 512 candles" in record.pop("reason")
    assert first == {
        "source": "bingx",
        "decision_hour": utc_text(before_store),
        "decision_ms": before_store,
        "stream": "forecast",
        "missing_candles": 1,
        "missing": [[FIRST_MS - HOUR_MS, FIRST_MS]],
        "missing_utc": [[utc_text(FIRST_MS - HOUR_MS), utc_text(FIRST_MS)]],
    }
    assert second == {
        "source": "bingx",
        "decision_hour": utc_text(gap + HOUR_MS),
        "decision_ms": gap + HOUR_MS,
        "stream": "forecast",
        "missing_candles": 1,
        "missing": [[gap, gap + HOUR_MS]],
        "missing_utc": [[utc_text(gap), utc_text(gap + HOUR_MS)]],
    }
    assert third == {**second, "stream": "mismatch-floor"}


def test_a_source_without_any_candle_in_range_drops_its_tasks(tmp_path):
    write_stores(tmp_path, binance=BINANCE_FRAME.iloc[:50].reset_index(drop=True))
    tasks = [FanTask("binance", D1), FanTask("bingx", D1)]
    prepared = prepare_tasks(tasks, params=PARAMS, root=tmp_path)
    assert prepared.tasks == (tasks[1],)
    (dropped,) = prepared.dropped
    assert dropped["missing_candles"] == 512
    assert dropped["missing"] == [[D1_MS - 512 * HOUR_MS, D1_MS]]
    assert "no binance 1h candles" in dropped["reason"]


@pytest.mark.parametrize(
    "case, error",
    [
        ("before the Kronos cutoff", PartitionError),
        ("sealed hour", SealedPartitionError),
        ("embargo", PartitionError),
        ("duplicate task", BatchError),
        ("threads", BatchError),
        ("not a task", TypeError),
        ("missing store", StoreMissingError),
    ],
)
def test_prepare_tasks_refuses_whole_batches(root, case, error):
    tasks, params, where = list(TASKS), PARAMS, root
    if case == "before the Kronos cutoff":
        tasks.append(FanTask("bingx", KRONOS_FIRST_DECISION - timedelta(hours=1)))
    elif case == "sealed hour":
        tasks.append(FanTask("bingx", SEALED_START + timedelta(hours=3)))
    elif case == "embargo":
        params = replace(PARAMS, kronos=replace(KRONOS, horizon_n=9))
    elif case == "duplicate task":
        tasks.append(FanTask("bingx", "2025-03-03T12Z"))
    elif case == "threads":
        params = replace(PARAMS, kronos=replace(KRONOS, threads=4))
    elif case == "not a task":
        tasks.append(("bingx", D1))
    elif case == "missing store":
        where = root / "empty"
    with pytest.raises(error):
        prepare_tasks(tasks, params=params, root=where)


def test_pending_counts_only_the_fans_not_stored_yet(root):
    run_fan_batch(
        prepare_tasks(TASKS[:1], params=PARAMS, root=root),
        params=PARAMS,
        root=root,
        models=fake_models(),
    )
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    assert (len(prepared.tasks), prepared.pending) == (4, 3)


def test_the_batch_path_reads_candles_only_through_the_sealed_guard():
    """docs/DATA.md: window sources read candles only through read_study_frame; read_store
    (integrity work) never feeds a fan. Checked on the source of every module of the path."""
    found: dict[str, set[str]] = {}
    for relpath in (
        "kronos/batch.py",
        "kronos/decision.py",
        "data/forecast_store.py",
        "data/windows.py",
    ):
        tree = ast.parse((REPO_ROOT / "src/kronos_agent" / relpath).read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name)
        found[relpath] = names
        assert "read_store" not in names, relpath
    assert "read_study_frame" in found["data/windows.py"]
    assert "StoreWindowSource" in found["kronos/batch.py"]


# -- gate 3 ----------------------------------------------------------------------------------------


def test_estimate_formula_and_the_worker_cap(root, monkeypatch):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    result = estimate(prepared, seconds_per_fan=3.0, threads=2, workers=2)
    assert result["cpu_seconds"] == 4 * 3.0 * 2
    assert result["wall_seconds"] == 24.0 / (2 * 2)
    assert result["cpu_hours"] == 24.0 / 3600
    assert (result["tasks"], result["dropped"], result["stored"], result["pending"]) == (4, 0, 0, 4)
    assert (result["workers"], result["workers_requested"], result["threads"]) == (2, 2, 2)
    assert result["seconds_per_fan"] == 3.0
    assert result["needs_owner_approval"] is False
    assert result["gate3_cpu_seconds"] == GATE3_CPU_SECONDS == 3600
    assert result["config_id"] == prepared.config_id
    assert (max_workers(2), max_workers(1), max_workers(20)) == (5, 10, 1)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)  # 3 // 2: one worker at most
    capped = estimate(prepared, seconds_per_fan=3.0, threads=2, workers=4)
    assert (capped["workers"], capped["workers_requested"]) == (1, 4)
    assert capped["wall_seconds"] == 12.0
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert max_workers(2) == 1


def test_gate3_refusal_and_approval_with_a_stubbed_seconds_per_fan(root):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)  # 4 pending fans
    at_limit = estimate(prepared, seconds_per_fan=450.0, threads=2, workers=1)
    assert at_limit["cpu_seconds"] == 3600.0 and at_limit["needs_owner_approval"] is False
    assert require_gate3(at_limit, approved_cpu_hours=None) is None  # one CPU-hour passes
    assert require_gate3(at_limit, approved_cpu_hours=0.1) is None  # no approval needed
    above = estimate(prepared, seconds_per_fan=450.5, threads=2, workers=1)
    assert above["needs_owner_approval"] is True
    with pytest.raises(Gate3Refused, match="approved-by-owner CPU_HOURS"):
        require_gate3(above, approved_cpu_hours=None)
    assert require_gate3(above, approved_cpu_hours=1.0) == pytest.approx(3604.0 / 3600)
    assert issubclass(Gate3Refused, PermissionError)


def test_gate3_tolerance_of_the_approved_figure():
    """D-040: the measured CPU-hours may be at most 1.25 times the figure the owner approved;
    exactly 1.25 passes, anything above is refused with both figures and their ratio."""
    assert GATE3_APPROVAL_TOLERANCE == 1.25
    at_tolerance = {"cpu_seconds": 4500.0}  # 1.25 CPU-hours, exactly 1.25 x 1.0
    assert require_gate3(at_tolerance, approved_cpu_hours=1.0) == 1.25
    above = {"cpu_seconds": 4500.01}
    with pytest.raises(Gate3Refused) as refused:
        require_gate3(above, approved_cpu_hours=1.0)
    message = str(refused.value)
    assert "measured 1.25 CPU-hours" in message and "the 1 CPU-hours the owner approved" in message
    assert "1.25 times" in message and "more than 1.25 times" in message
    # D-040's case: 5,500 CPU-seconds shown and approved, 7,857 measured by the approved run
    shown = 873 * 3.15 * 2 / 3600
    with pytest.raises(Gate3Refused, match="1.43 times"):
        require_gate3({"cpu_seconds": 7857.0}, approved_cpu_hours=shown)
    assert require_gate3({"cpu_seconds": 873 * 3.15 * 2}, approved_cpu_hours=shown) == (
        pytest.approx(1.0)
    )
    assert require_gate3({"cpu_seconds": 873 * 3.3 * 2}, approved_cpu_hours=shown) < 1.25
    with pytest.raises(Gate3Refused):  # a hundred times the shown estimate
        require_gate3({"cpu_seconds": 550_000.0}, approved_cpu_hours=shown)


@pytest.mark.parametrize("figure", [0, 0.0, -1.0, float("nan"), float("inf"), True, "1.5"])
def test_require_gate3_refuses_a_malformed_approved_figure(figure):
    for cpu_seconds in (10.0, 10_000.0):  # whether or not the batch needs the approval
        with pytest.raises(ValueError, match="approved CPU-hours"):
            require_gate3({"cpu_seconds": cpu_seconds}, approved_cpu_hours=figure)


def test_gate3_on_the_planned_kronos_agreement_batch():
    hours = [KRONOS_FIRST_DECISION + timedelta(hours=61 * k) for k in range(291)]
    tasks = tuple(
        FanTask(source, hour, stream)
        for hour in hours
        for source, stream in (
            ("binance", "forecast"),
            ("bingx", "forecast"),
            ("bingx", "mismatch-floor"),
        )
    )
    full = estimate(
        PreparedBatch(tasks=tasks, dropped=(), pending=873),
        seconds_per_fan=3.15,
        threads=2,
        workers=4,
    )
    assert full["cpu_seconds"] == pytest.approx(873 * 3.15 * 2)  # about 1.5 CPU-hours
    with pytest.raises(Gate3Refused):
        require_gate3(full, approved_cpu_hours=None)
    assert require_gate3(full, approved_cpu_hours=1.53) == pytest.approx(full["cpu_hours"] / 1.53)
    fallback = estimate(
        PreparedBatch(tasks=tasks[:549], dropped=(), pending=549),
        seconds_per_fan=3.15,
        threads=2,
        workers=1,
    )
    assert fallback["cpu_seconds"] == pytest.approx(3458.7)  # under one CPU-hour
    assert require_gate3(fallback, approved_cpu_hours=None) is None
    resumed = estimate(
        PreparedBatch(tasks=tasks, dropped=(), pending=600),  # 273 fans already stored
        seconds_per_fan=3.15,
        threads=2,
        workers=1,
    )
    assert resumed["stored"] == 273 and resumed["cpu_seconds"] == pytest.approx(3780.0)


@pytest.mark.parametrize("cpu_seconds", [float("nan"), float("inf"), -1.0, True, "100", None])
def test_require_gate3_refuses_a_malformed_estimate(cpu_seconds):
    with pytest.raises(ValueError):
        require_gate3({"cpu_seconds": cpu_seconds}, approved_cpu_hours=100.0)


def test_estimate_argument_checks(root):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    for bad in (0, 0.0, -1.0, float("nan"), float("inf"), True, "3"):
        with pytest.raises(ValueError):
            estimate(prepared, seconds_per_fan=bad, threads=2, workers=1)
    with pytest.raises(ValueError):
        estimate(prepared, seconds_per_fan=1.0, threads=0, workers=1)
    with pytest.raises(ValueError):
        estimate(prepared, seconds_per_fan=1.0, threads=2, workers=0)
    with pytest.raises(ValueError):
        estimate(replace(prepared, pending=5), seconds_per_fan=1.0, threads=2, workers=1)
    with pytest.raises(TypeError):
        estimate({"pending": 1}, seconds_per_fan=1.0, threads=2, workers=1)
    with pytest.raises(ValueError):
        require_gate3({}, approved_cpu_hours=1.0)


@pytest.mark.parametrize("policy", ["volume_x_mean_ohlc", "zeros", "binance_quote_volume"])
def test_the_timing_window_follows_the_params(policy):
    params = replace(PARAMS, kronos=replace(KRONOS, amount_policy=policy, horizon_n=8))
    window = timing_window(params)
    assert window.decision_hour == TIMING_DECISION_HOUR
    assert (window.max_context, window.horizon, window.amount_policy) == (512, 8, policy)
    assert window.candles[-1].open_ms == to_ms(TIMING_DECISION_HOUR) - HOUR_MS
    fan = forecast_window(fake_models(), window, params, stream=TIMING_STREAM)
    assert fan.paths.shape == (KRONOS.trajectories, 8, 6)


def test_measure_seconds_per_fan_times_one_fan_after_a_warm_up(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(batch, "_clock", lambda: clock["now"])
    predictor = ClockedPredictor(clock, [7.0, 2.5])  # warm-up 7 s (untimed), then 2.5 s
    seconds = measure_seconds_per_fan(params=PARAMS, models=fake_models(predictor))
    assert seconds == 2.5
    assert len(predictor.calls) == 2
    for call in predictor.calls:
        assert call == {"batch": KRONOS.trajectories, "rows": 512, "pred_len": KRONOS.horizon_n}
    stuck = ClockedPredictor(clock, [1.0, 0.0])
    with pytest.raises(BatchError, match="clock"):
        measure_seconds_per_fan(params=PARAMS, models=fake_models(stuck))
    four_threads = replace(PARAMS, kronos=replace(KRONOS, threads=4))
    with pytest.raises(BatchError, match="threads"):
        measure_seconds_per_fan(params=four_threads, models=fake_models())


# -- running with injected models ------------------------------------------------------------------


def test_run_fan_batch_stores_and_summarises_in_task_order(root):
    predictor = WalkPredictor()
    models = fake_models(predictor)
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    seen = []
    summaries = run_fan_batch(
        prepared,
        params=PARAMS,
        root=root,
        models=models,
        progress=lambda done, total, summary: seen.append((done, total, summary)),
    )
    assert keys_of(summaries) == [key_of(task) for task in TASKS]
    assert [(done, total) for done, total, _ in seen] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert [summary for _, _, summary in seen] == summaries
    assert len(predictor.calls) == 4 and len(npz_files(root)) == 4
    for task, window, summary in zip(prepared.tasks, prepared.windows, summaries, strict=True):
        assert summary.relpath == fan_relpath(
            task.source, prepared.config_id, task.stream, task.decision_hour
        )
        stored = load_fan(root / summary.relpath)
        check_reusable(stored, window, stream=task.stream)
        assert summary.computed is True
        assert (summary.window_sha256, summary.inputs_sha256) == (
            window.sha256,
            window.inputs_sha256,
        )
        assert summary.last_close == window.last_close
        assert summary.signal == fan_signal(stored.paths, window.last_close)
        assert summary.p_up == fan_p_up(stored.paths, window.last_close)
        assert summary.mean_close_n == fan_mean_close(stored.paths)
        assert summary.elapsed_s == stored.meta["elapsed_s"]
        assert stored.meta["params_version"] == PARAMS.version
        recomputed = forecast_window(models, window, PARAMS, stream=task.stream)
        assert recomputed.paths.tobytes() == stored.paths.tobytes()  # reload == recompute
    by_key = {key_of(task): load_fan(root / s.relpath) for task, s in zip(TASKS, summaries)}
    binance_seeds = by_key[("binance", "2025-03-03T12:00Z", "forecast")].meta["seeds"]
    bingx_seeds = by_key[("bingx", "2025-03-03T12:00Z", "forecast")].meta["seeds"]
    floor_seeds = by_key[("bingx", "2025-03-03T12:00Z", "mismatch-floor")].meta["seeds"]
    assert binance_seeds == bingx_seeds != floor_seeds
    calls = len(predictor.calls)
    again = prepare_tasks(TASKS, params=PARAMS, root=root)
    assert again.pending == 0
    reused = run_fan_batch(again, params=PARAMS, root=root, models=models)
    assert len(predictor.calls) == calls  # stored fans are never recomputed
    assert [s.computed for s in reused] == [False] * 4
    assert [replace(s, computed=True) for s in reused] == summaries


def test_task_order_never_changes_a_fan(tmp_path):
    forward_root = write_stores(tmp_path / "forward")
    backward_root = write_stores(tmp_path / "backward")
    forward = run_fan_batch(
        prepare_tasks(TASKS, params=PARAMS, root=forward_root),
        params=PARAMS,
        root=forward_root,
        models=fake_models(),
    )
    backward = run_fan_batch(
        prepare_tasks(TASKS[::-1], params=PARAMS, root=backward_root),
        params=PARAMS,
        root=backward_root,
        models=fake_models(),
    )
    assert keys_of(backward) == keys_of(forward)[::-1]
    for summary in forward:
        assert (
            load_fan(forward_root / summary.relpath).paths.tobytes()
            == load_fan(backward_root / summary.relpath).paths.tobytes()
        )
    assert {s.relpath: replace(s, elapsed_s=0.0) for s in forward} == {
        s.relpath: replace(s, elapsed_s=0.0) for s in backward
    }


def test_a_failed_batch_keeps_its_stored_fans_and_resumes(root, tmp_path):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    with pytest.raises(RuntimeError, match="model failure"):
        run_fan_batch(prepared, params=PARAMS, root=root, models=fake_models(FailingPredictor(2)))
    assert len(npz_files(root)) == 2
    resumed_batch = prepare_tasks(TASKS, params=PARAMS, root=root)
    assert resumed_batch.pending == 2
    resumed = run_fan_batch(resumed_batch, params=PARAMS, root=root, models=fake_models())
    assert [s.computed for s in resumed] == [False, False, True, True]
    clean_root = write_stores(tmp_path / "clean")
    clean = run_fan_batch(
        prepare_tasks(TASKS, params=PARAMS, root=clean_root),
        params=PARAMS,
        root=clean_root,
        models=fake_models(),
    )
    for a, b in zip(resumed, clean, strict=True):
        assert load_fan(root / a.relpath).paths.tobytes() == (
            load_fan(clean_root / b.relpath).paths.tobytes()
        )


def test_a_stale_estimate_is_refused(root):
    run_fan_batch(
        prepare_tasks(TASKS[:1], params=PARAMS, root=root),
        params=PARAMS,
        root=root,
        models=fake_models(),
    )
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    assert prepared.pending == 3
    (stored,) = npz_files(root)
    stored.unlink()
    predictor = WalkPredictor()
    with pytest.raises(Gate3Refused, match="4 fans are pending but the estimate counted 3"):
        run_fan_batch(prepared, params=PARAMS, root=root, models=fake_models(predictor))
    assert predictor.calls == [] and npz_files(root) == []


def test_a_batch_runs_only_under_the_configuration_it_was_prepared_for(root):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    more = replace(PARAMS, kronos=replace(KRONOS, trajectories=21))
    with pytest.raises(BatchError, match="prepared for forecast config"):
        run_fan_batch(prepared, params=more, root=root, models=fake_models())
    other_weights = fake_models(weights={"NeoQuasar/Kronos-small/model.safetensors": "0" * 64})
    with pytest.raises(BatchError, match="prepared for forecast config"):
        run_fan_batch(prepared, params=PARAMS, root=root, models=other_weights)
    assert npz_files(root) == []


def test_a_stored_fan_of_another_window_is_refused_and_kept(root):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    task = prepared.tasks[0]
    altered = [replace(c, close="1.0") if c.open_ms == D1_MS - 3 * HOUR_MS else c for c in BINGX]
    other = build_window(altered, D1, source="bingx", horizon=KRONOS.horizon_n)
    path = root / fan_relpath(task.source, prepared.config_id, task.stream, task.decision_hour)
    save_fan(
        path,
        forecast_window(fake_models(), other, PARAMS),
        other,
        stream="forecast",
        config=prepared.config,
        params_version=PARAMS.version,
    )
    before = path.read_bytes()
    predictor = WalkPredictor()
    with pytest.raises(ForecastStoreError, match="window sha256") as info:
        run_fan_batch(prepared, params=PARAMS, root=root, models=fake_models(predictor))
    assert "2025-03-03T12Z.npz" in str(info.value)
    assert path.read_bytes() == before and predictor.calls == []
    with pytest.raises(ForecastStoreError, match="never overwritten"):
        prepare_tasks(TASKS, params=PARAMS, root=root)


def test_injected_models_run_in_this_process_only(root, monkeypatch):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    with pytest.raises(BatchError, match="this process only"):
        run_fan_batch(prepared, workers=2, params=PARAMS, root=root, models=fake_models())
    monkeypatch.setattr(os, "cpu_count", lambda: 2)  # at most one worker: this process
    summaries = run_fan_batch(prepared, workers=4, params=PARAMS, root=root, models=fake_models())
    assert [s.computed for s in summaries] == [True] * 4


class _PoolReached(Exception):
    pass


def _no_pool(*args, **kwargs):
    raise _PoolReached


@pytest.mark.parametrize(
    "memory, refused", [(2 * 2**30, True), (4 * 2**30 - 1, True), (None, True), (4 * 2**30, False)]
)
def test_more_than_one_worker_needs_four_gib_of_ram(root, monkeypatch, memory, refused):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(batch, "_physical_memory_bytes", lambda: memory)
    monkeypatch.setattr(batch, "_check_parent_prerequisites", lambda model, checkpoint=None: None)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", _no_pool)
    if refused:
        with pytest.raises(BatchError, match="4 GiB"):
            run_fan_batch(prepared, workers=2, params=PARAMS, root=root)
    else:
        with pytest.raises(_PoolReached):
            run_fan_batch(prepared, workers=2, params=PARAMS, root=root)
    assert npz_files(root) == []


def test_run_fan_batch_needs_a_batch_built_by_prepare_tasks(root):
    prepared = prepare_tasks(TASKS, params=PARAMS, root=root)
    bare = PreparedBatch(prepared.tasks, prepared.dropped, prepared.pending)
    with pytest.raises(BatchError, match="prepare_tasks"):
        run_fan_batch(bare, params=PARAMS, root=root, models=fake_models())
    swapped = replace(prepared, windows=prepared.windows[::-1])
    with pytest.raises(BatchError, match="another task"):
        run_fan_batch(swapped, params=PARAMS, root=root, models=fake_models())
    four_threads = replace(PARAMS, kronos=replace(KRONOS, threads=4))
    with pytest.raises(BatchError, match="threads"):
        run_fan_batch(prepared, params=four_threads, root=root, models=fake_models())
    with pytest.raises(ValueError):
        run_fan_batch(prepared, workers=0, params=PARAMS, root=root, models=fake_models())
    with pytest.raises(TypeError):
        run_fan_batch(list(prepared.tasks), params=PARAMS, root=root, models=fake_models())
    assert npz_files(root) == []


# -- real Kronos-small fans ------------------------------------------------------------------------

SPAWN_SKIP_REASON = "two spawn workers need os.cpu_count() >= 4 and at least 4 GiB of RAM"


def _spawn_possible() -> bool:
    memory = batch._physical_memory_bytes()
    return max_workers(2) >= 2 and memory is not None and memory >= batch.MIN_RAM_BYTES_FOR_WORKERS


@pytest.fixture(scope="module")
def real_batches(loaded_small, tmp_path_factory):
    """Real fans of TASKS: in-process forecast_window, a one-worker batch with the tasks reversed
    and a two-worker spawn batch with the tasks shuffled, each under its own data root."""
    params = SMALL_PARAMS
    reference_root = write_stores(tmp_path_factory.mktemp("reference"))
    prepared = prepare_tasks(TASKS, params=params, root=reference_root)
    reference = {
        key_of(task): forecast_window(loaded_small, window, params, stream=task.stream)
        for task, window in zip(prepared.tasks, prepared.windows, strict=True)
    }
    one_root = write_stores(tmp_path_factory.mktemp("one_worker"))
    backward = TASKS[::-1]
    one = run_fan_batch(
        prepare_tasks(backward, params=params, root=one_root),
        workers=1,
        params=params,
        root=one_root,
    )
    two = None
    children: list[int] = []
    shuffled = (TASKS[2], TASKS[0], TASKS[3], TASKS[1])
    two_root = write_stores(tmp_path_factory.mktemp("two_workers"))
    if _spawn_possible():
        two = run_fan_batch(
            prepare_tasks(shuffled, params=params, root=two_root),
            workers=2,
            params=params,
            root=two_root,
            progress=lambda done, total, s: children.append(len(multiprocessing.active_children())),
        )
    return SimpleNamespace(
        params=params,
        reference=reference,
        one=(one_root, backward, one),
        two=(two_root, shuffled, two),
        children=children,
    )


@pytest.mark.weights
def test_one_worker_and_in_process_forecasts_give_identical_fans(real_batches):
    root, order, summaries = real_batches.one
    assert keys_of(summaries) == [key_of(task) for task in order]
    for task, summary in zip(order, summaries, strict=True):
        fan = real_batches.reference[key_of(task)]
        stored = load_fan(root / summary.relpath)
        assert stored.paths.tobytes() == fan.paths.tobytes()  # reload == recompute, bit for bit
        assert stored.meta["seeds"]["canonical"] == fan.seeds.canonical
        assert summary.computed and summary.signal == fan_signal(fan.paths, summary.last_close)
    seeds = {
        key_of(task): load_fan(root / s.relpath).meta["seeds"] for task, s in zip(order, summaries)
    }
    assert (
        seeds[("binance", "2025-03-03T12:00Z", "forecast")]
        == seeds[("bingx", "2025-03-03T12:00Z", "forecast")]
    )
    assert (
        seeds[("bingx", "2025-03-03T12:00Z", "mismatch-floor")]
        != seeds[("bingx", "2025-03-03T12:00Z", "forecast")]
    )


@pytest.mark.weights
def test_two_spawn_workers_give_the_fans_of_one_worker_and_of_in_process(real_batches):
    root, order, summaries = real_batches.two
    if summaries is None:
        pytest.skip(SPAWN_SKIP_REASON)
    assert max(real_batches.children) >= 2  # the fans came from spawn-started worker processes
    assert keys_of(summaries) == [key_of(task) for task in order]
    for task, summary in zip(order, summaries, strict=True):
        stored = load_fan(root / summary.relpath)
        assert stored.paths.tobytes() == real_batches.reference[key_of(task)].paths.tobytes()
        assert summary.computed
    _, _, one = real_batches.one
    assert {s.relpath: replace(s, elapsed_s=0.0) for s in summaries} == {
        s.relpath: replace(s, elapsed_s=0.0) for s in one
    }


@pytest.mark.weights
def test_stored_real_fans_are_reused_never_recomputed(real_batches, monkeypatch):
    root, order, first = real_batches.one
    prepared = prepare_tasks(order, params=real_batches.params, root=root)
    assert prepared.pending == 0

    def no_models(*args, **kwargs):
        raise AssertionError("nothing is pending: no model may be loaded")

    monkeypatch.setattr(batch, "load_models", no_models)
    again = run_fan_batch(prepared, workers=2, params=real_batches.params, root=root)
    assert [s.computed for s in again] == [False] * len(order)
    assert [replace(s, computed=True) for s in again] == first


@pytest.mark.weights
def test_a_real_stored_fan_is_refused_for_a_changed_window(real_batches):
    root, order, summaries = real_batches.one
    (summary,) = [s for task, s in zip(order, summaries) if key_of(task) == key_of(TASKS[0])]
    stored = load_fan(root / summary.relpath)
    check_reusable(
        stored, build_window(BINGX, D1, source="bingx", horizon=KRONOS.horizon_n), stream="forecast"
    )
    changed = [
        replace(c, volume="0.0001") if c.open_ms == D1_MS - 200 * HOUR_MS else c for c in BINGX
    ]
    with pytest.raises(ForecastStoreError, match="window sha256"):
        check_reusable(
            stored,
            build_window(changed, D1, source="bingx", horizon=KRONOS.horizon_n),
            stream="forecast",
        )


@pytest.mark.weights
def test_the_registry_configuration_is_the_one_of_the_loaded_models(loaded_small):
    assert registered_weights_sha256("small") == loaded_small.weights_sha256
    assert forecast_config(loaded_small, KRONOS) == registry_config(KRONOS)


@pytest.mark.weights
def test_measure_seconds_per_fan_with_kronos_small(loaded_small):
    seconds = measure_seconds_per_fan(params=SMALL_PARAMS, models=loaded_small)
    assert math.isfinite(seconds) and seconds > 0
