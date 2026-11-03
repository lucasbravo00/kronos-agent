"""The pilot check (``docs/RETRAINING.md`` 6.2 steps 3-4; interfaces 11 and 12).

The pilot fold is the real ``q-2024-07`` of R1: a synthetic Binance 1h store covers its inner
validation's windows, a stand-in export holds the NLL4-TF inputs of its 336 ``G4`` decisions built
from the same candles (``data.windows.context_frame``), its checkpoint is a stand-in registry entry
(``finetune_fakes.install_stand_in``) whose line, fold record and export hashes are this pilot's,
and its fold record is a written ``runner_result.json``. Fans come from the real batch layer with
stand-in models (``finetune_fakes.walk_models``) or from stand-in fan summaries; the KL and the
weight change run on tiny random models. Nothing loads Kronos-small, no real fan or training runs.
The tests that compute the fair CRPS need ``kronos_agent.eval.crps`` (interfaces 9); the spend
control's tests (design 3.5) run on stand-in fans and a stand-in clock, with the CRPS oracle
standing in for that module while it is not built.
"""

from __future__ import annotations

import json
import math
import sys
import time
import types
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import finetune_fakes as ff
import numpy as np
import pytest
import step2_fakes as fakes
import torch

from kronos_agent.data.forecast_store import load_fan
from kronos_agent.data.partition import MODEL_PERIOD_PILOT_CHECK
from kronos_agent.data.timeutil import HOUR_MS, to_ms, utc_text
from kronos_agent.data.windows import context_frame
from kronos_agent.finetune import pilot_check as pc
from kronos_agent.finetune.export_csv import EXPORT_SCHEMA_VERSION, nll_inputs_bytes, read_export
from kronos_agent.finetune.folds import fold_for
from kronos_agent.finetune.nll import NllInputs, teacher_forced_logits
from kronos_agent.finetune.recipe import R1, recipe8, recipe_id
from kronos_agent.finetune.registry import CheckpointRef, RegistryChainError, checkpoint_id_for
from kronos_agent.hashing import canonical_json, sha256_bytes, sha256_file
from kronos_agent.kronos import batch
from kronos_agent.kronos.batch import FanSummary
from kronos_agent.kronos.weights import LoadedModels
from kronos_agent.params import load_params
from kronos_agent.paths import REPO_ROOT
from kronos_agent.report.writer import ConfigChangedError

PARAMS = load_params()
FOLD = fold_for("q-2024-07", R1)
CHECKPOINT_ID = checkpoint_id_for(recipe8(R1), FOLD.run_label)
CLOSE = 3
RUN_ID = "q-2024-07-20261103T000000Z"
CURVE = [
    {"pseudo_epoch": 0, "nll4_tf": 5.10, "train_loss": None, "vendored_val_loss": None},
    {"pseudo_epoch": 1, "nll4_tf": 5.00, "train_loss": 4.0, "vendored_val_loss": 4.2},
    {"pseudo_epoch": 2, "nll4_tf": 4.95, "train_loss": 3.9, "vendored_val_loss": 4.1},
    {"pseudo_epoch": 3, "nll4_tf": 4.97, "train_loss": 3.8, "vendored_val_loss": 4.15},
    {"pseudo_epoch": 4, "nll4_tf": 4.99, "train_loss": 3.7, "vendored_val_loss": 4.2},
]


def crps_oracle(ensemble, observed):
    """The fair ensemble CRPS by its double sum (design 5.1)."""
    x = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    k = x.shape[1]
    first = np.abs(x - y[:, None]).mean(axis=1)
    pairs = np.abs(x[:, :, None] - x[:, None, :]).sum(axis=(1, 2))
    return first - pairs / (2 * k * (k - 1))


def needs_crps():
    pytest.importorskip("kronos_agent.eval.crps", reason="the fair CRPS of interfaces 9")


@pytest.fixture(scope="module")
def market():
    """Binance-like 1h candles from the first context of the fold's NLL4-TF decisions to its CSV
    end, and the 516-row inference floats of each of those 336 G4 decisions."""
    g4 = FOLD.nll_decisions_ms()
    start = g4[0] - 512 * HOUR_MS
    frame = fakes.binance_frame(
        start, (FOLD.csv_end_ms - start) // HOUR_MS, interval="1h", seed=5, start_price=60_000.0
    )
    candles = fakes.frame_to_candles(frame)
    opens = np.array([candle.open_ms for candle in candles], dtype=np.int64)
    return SimpleNamespace(frame=frame, candles=candles, opens=opens, g4=g4)


def nll_values(market, decisions, *, shift_hours=0):
    """The 516 inference rows of each decision (shifted by ``shift_hours`` for a misaligned set)."""
    rows = []
    for decision in decisions:
        first = decision + (shift_hours - 512) * HOUR_MS
        lo = int(np.searchsorted(market.opens, first))
        rows.append(
            context_frame(market.candles[lo : lo + 516], "volume_x_mean_ohlc").to_numpy(np.float64)
        )
    return np.stack(rows)


def write_export(directory, decisions, values):
    """A stand-in export of the pilot fold: its NLL4-TF inputs and an export.json that
    ``read_export`` verifies (the CSV is a placeholder; the pilot check never reads it)."""
    directory.mkdir(parents=True, exist_ok=True)
    csv = b"timestamps,open,high,low,close,volume,amount\n"
    npz = nll_inputs_bytes(np.asarray(decisions, dtype=np.int64), values, grid_hours=4)
    (directory / "train.csv").write_bytes(csv)
    (directory / "nll_inputs.npz").write_bytes(npz)
    content = {
        "fold": FOLD.label,
        "recipe_id": recipe_id(R1),
        "rows": 2,
        "n_fit": 1,
        "n_val": 1,
        "train_ratio": 0.75,
        "gap_aware": False,
        "fit_windows_kept": 1,
        "val_windows_kept": 1,
        "csv": {"sha256": sha256_bytes(csv), "size": len(csv)},
        "nll_inputs": {"sha256": sha256_bytes(npz), "size": len(npz)},
        "nll": {"decision_ms": [int(d) for d in decisions]},
    }
    document = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "content": content,
        "content_sha256": sha256_bytes(canonical_json(content)),
        "provenance": {"store": None, "code": {}, "created_at": "2026-11-03T00:00:00Z"},
    }
    (directory / "export.json").write_text(json.dumps(document), encoding="ascii")
    return read_export(directory)


@pytest.fixture
def pilot(tmp_path, market):
    return make_pilot(tmp_path, market)


def make_pilot(tmp_path, market, *, decisions=None, values=None, missing=()):
    """The pilot's store, export, fold record and registered stand-in checkpoint under
    ``tmp_path``; ``missing`` removes candles from the store (never from the export)."""
    data = tmp_path / "data"
    frame = market.frame
    if missing:
        frame = frame[~frame["open_ms"].isin(set(missing))].reset_index(drop=True)
    fakes.write_binance_store(data, frame)
    decisions = market.g4 if decisions is None else decisions
    values = nll_values(market, decisions) if values is None else values
    export = write_export(tmp_path / "export", decisions, values)
    ft = tmp_path / "ft"
    run_dir = ft / ".runs" / RUN_ID
    run_dir.mkdir(parents=True)
    record = {
        "status": "completed",
        "run_id": RUN_ID,
        "stop_reason": "early_stop",
        "selected_pseudo_epoch": 2,
        "epochs": CURVE,
    }
    record_path = run_dir / "runner_result.json"
    record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="ascii")
    inner = FOLD.inner_validation_start_ms
    entry = ff.install_stand_in(
        ft,
        CHECKPOINT_ID,
        data_end_ms=FOLD.csv_end_ms,
        recipe_id=recipe_id(R1),
        fold=FOLD.label,
        fit_range_ms=[FOLD.csv_start_ms, inner],
        inner_validation_range_ms=[inner, FOLD.csv_end_ms],
        test_start_ms=FOLD.test_start_ms,
        test_end_ms=FOLD.test_end_ms,
        fit_record_sha256=sha256_file(record_path),
        export_content_sha256=export.content_sha256,
        nll_inputs_sha256=export.nll_inputs_sha256,
        run_id=RUN_ID,
    )
    inputs = pc.PilotInputs(
        recipe=R1,
        fold=FOLD,
        checkpoint=CheckpointRef(CHECKPOINT_ID, ft, line_sha256=entry.line_sha256),
        fit_record_path=record_path,
        fit_record_sha256=sha256_file(record_path),
        export=export,
    )
    return SimpleNamespace(
        data=data, reports=tmp_path / "reports", ft=ft, entry=entry, inputs=inputs, values=values
    )


class DriftPredictor(ff.WalkPredictor):
    """A walk with an upward drift: another model's fans under the same seeds."""

    def predict_batch(
        self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p, **_
    ):
        frames = super().predict_batch(
            df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p
        )
        drift = np.exp(0.004 * np.arange(1, pred_len + 1))
        return [frame.mul(drift, axis=0) for frame in frames]


@pytest.fixture(scope="module")
def tiny():
    tokenizer, shipped = ff.build_tiny_models(0)
    _, other = ff.build_tiny_models(1)
    return SimpleNamespace(
        tokenizer=ff.eval_copy(tokenizer), shipped=ff.eval_copy(shipped), other=ff.eval_copy(other)
    )


def steps_for(world, tiny, *, pilot_predictor=None, same_model=False, **overrides):
    """Real batch preparation and fans with walk models, tiny models for the KL."""
    shipped_walk = ff.walk_models()
    pilot_walk = ff.walk_models(world.entry, predictor=pilot_predictor)
    calls: list[dict] = []

    def measure(**kwargs):
        calls.append(kwargs)
        return 3.0625

    def run_fan_batch(prepared, *, workers, params, root, progress=None):
        models = pilot_walk if prepared.checkpoint is not None else shipped_walk
        return batch.run_fan_batch(
            prepared, workers=workers, params=params, root=root, models=models, progress=progress
        )

    def load_models(name, checkpoint=None):
        walk = pilot_walk if checkpoint is not None else shipped_walk
        model = tiny.shipped if checkpoint is None or same_model else tiny.other
        return LoadedModels(
            spec=walk.spec,
            tokenizer=tiny.tokenizer,
            model=model,
            predictor=None,
            weights_sha256=dict(walk.weights_sha256),
            load_seconds=0.0,
            checkpoint=walk.checkpoint,
            data_end_ms=walk.data_end_ms,
        )

    values = {
        "prepare_tasks": batch.prepare_tasks,
        "measure_seconds_per_fan": measure,
        "run_fan_batch": run_fan_batch,
        "load_models": load_models,
        "load_fan": load_fan,
        "teacher_forced_logits": teacher_forced_logits,
    }
    values.update(overrides)
    steps = pc.PilotCheckSteps(**values)
    return steps, calls


def prepare(world, steps, **kwargs):
    return pc.prepare_check(
        world.inputs, params=PARAMS, steps=steps, root=world.data, directory=world.reports, **kwargs
    )


# -- the definition ---------------------------------------------------------------------------


def test_the_check_decisions_are_the_168_g8_hours_of_the_inner_validation():
    decisions = pc.check_decisions_ms(FOLD)
    assert len(decisions) == 168
    assert (utc_text(decisions[0]), utc_text(decisions[-1])) == (
        "2024-04-14T16:00Z",
        "2024-06-09T08:00Z",
    )
    assert all(datetime.fromtimestamp(d / 1000, UTC).hour % 8 == 0 for d in decisions)
    assert set(decisions) <= set(FOLD.nll_decisions_ms())
    late = replace(FOLD, csv_end_ms=to_ms(datetime(2024, 6, 9, 10, tzinfo=UTC)))
    with pytest.raises(pc.PilotCheckError, match="outside its CSV range"):
        pc.check_decisions_ms(late)
    early = replace(FOLD, csv_start_ms=to_ms(datetime(2024, 4, 1, tzinfo=UTC)))
    with pytest.raises(pc.PilotCheckError, match="outside its CSV range"):
        pc.check_decisions_ms(early)


def test_the_stem_and_the_plan(tmp_path):
    assert pc.stem_for(R1) == "pilot-check-0ee0d123"
    doubled = replace(R1, max_epochs=12)
    assert pc.stem_for(doubled) == "pilot-check-56e3d2f9"
    plan = pc.check_plan(R1, FOLD)
    assert plan["checkpoint_id"] == "ks-btc-0ee0d123-q-2024-07"
    assert plan["fans"]["horizon"] == 4 and plan["fans"]["trajectories"] == 20
    assert {name: plan["fans"][name]["stream"] for name in pc.BATCHES} == {
        "pilot": "forecast",
        "shipped": "forecast",
        "floor": "forecast-aa",
    }
    assert plan["role"]["period_role"] == MODEL_PERIOD_PILOT_CHECK
    assert plan["decisions"]["planned"] == 168 and plan["reading"]["threshold"] == 0.98
    assert plan["projection"]["decisions"] == 4380 and plan["projection"]["blocks"] == 27
    assert plan["tests"] == 0
    assert "params_version" not in json.dumps(plan) and "source_tree" not in json.dumps(plan)
    assert pc.check_plan(doubled, fold_for("q-2024-07", doubled))["checkpoint_id"] == (
        "ks-btc-56e3d2f9-q-2024-07"
    )
    digest = pc.record_check_config(R1, FOLD, directory=tmp_path)
    assert pc.record_check_config(R1, FOLD, directory=tmp_path) == digest
    assert sha256_file(tmp_path / "pilot-check-0ee0d123.config.json") == digest
    path = tmp_path / "pilot-check-0ee0d123.config.json"
    path.write_bytes(path.read_bytes().replace(b'"threshold":0.98', b'"threshold":0.97'))
    with pytest.raises(ConfigChangedError):
        pc.record_check_config(R1, FOLD, directory=tmp_path)


def test_the_reading_of_a_same():
    assert pc.go_no_go(0.979) == "GO"
    assert pc.go_no_go(0.98) == "NO-GO"  # 0.98 included
    assert pc.go_no_go(1.0) == "NO-GO"
    assert pc.go_no_go(0) == "GO"
    for bad in (1.01, -0.1, float("nan"), True, "0.99"):
        with pytest.raises(pc.PilotCheckError):
            pc.go_no_go(bad)


def test_the_projection_on_a_series_with_known_autocorrelation():
    # runs of 8 equal values with alternating signs: r_k = (160 - 39 k) / 160 for k = 1..6
    series = np.repeat(np.tile([1.0, -1.0], 10), 8)
    projection = pc.mdd_projection(series)
    expected_rho = {str(k): (160 - 39 * k) / 160 for k in range(1, 7)}
    assert projection["rho"] == pytest.approx(expected_rho)
    assert projection["inflation"] == pytest.approx(1 + 2 * 141 / 160)
    assert projection["n_eff"] == pytest.approx(4380 / 2 / (1 + 2 * 141 / 160))
    assert projection["d_sd_bp"] == pytest.approx(math.sqrt(160 / 159))
    assert projection["z_sum"] == pytest.approx(2.8015852181)
    se = math.sqrt(160 / 159) / math.sqrt(projection["n_eff"])
    assert projection["se_bp"] == pytest.approx(se)
    assert projection["mdd_bp"] == pytest.approx(2.8015852181 * se)
    assert projection["label"] == "approximate"
    # alternating signs: a negative sum, clipped to an inflation of 1
    alternating = pc.mdd_projection(np.tile([1.0, -1.0], 84))
    assert alternating["rho_sum"] < 0 and alternating["inflation"] == 1.0
    assert alternating["n_eff"] == 2190.0
    constant = pc.mdd_projection(np.full(20, 2.5))
    assert constant["rho"] == {str(k): None for k in range(1, 7)} and constant["inflation"] == 1.0
    assert (constant["d_sd_bp"], constant["mdd_bp"]) == (0.0, 0.0)
    for bad in (np.ones(7), np.array([1.0, np.nan] * 10)):
        with pytest.raises(pc.PilotCheckError):
            pc.mdd_projection(bad)


def test_the_relative_weight_change(tiny):
    same = pc.relative_weight_change(tiny.shipped.state_dict(), tiny.shipped.state_dict())
    assert (same["relative"], same["changed_tensors"], same["max_abs_delta"]) == (0.0, 0, 0.0)
    assert same["tensors"] > 0
    other = pc.relative_weight_change(tiny.other.state_dict(), tiny.shipped.state_dict())
    assert other["relative"] > 0 and other["changed_share"] > 0
    base = {"a": torch.tensor([3.0, 4.0]), "steps": torch.tensor([7]), "b": np.zeros(2)}
    state = {"a": torch.tensor([3.0, 5.0]), "steps": torch.tensor([9]), "b": np.zeros(2)}
    known = pc.relative_weight_change(state, base)
    assert known["relative"] == pytest.approx(0.2)  # ||(0, 1)|| / ||(3, 4)||
    assert (known["tensors"], known["changed_tensors"], known["max_abs_delta"]) == (2, 1, 1.0)
    with pytest.raises(pc.PilotCheckError, match="different tensors"):
        pc.relative_weight_change({"a": base["a"]}, base)
    with pytest.raises(pc.PilotCheckError, match="shapes"):
        pc.relative_weight_change({**base, "a": torch.zeros(3)}, base)
    with pytest.raises(pc.PilotCheckError, match="floating in one"):
        pc.relative_weight_change({**base, "a": torch.tensor([3, 4])}, base)


def test_the_first_step_kl(tiny, market):
    decisions = market.g4[:3]
    inputs = NllInputs(np.asarray(decisions, dtype=np.int64), nll_values(market, decisions))
    shipped = teacher_forced_logits(tiny.tokenizer, tiny.shipped, inputs)
    again = teacher_forced_logits(tiny.tokenizer, tiny.shipped, inputs)
    other = teacher_forced_logits(tiny.tokenizer, tiny.other, inputs)
    zero = pc._first_step_kl(again, shipped)
    assert (zero["kl_pilot_shipped"], zero["kl_shipped_pilot"], zero["kl_pilot_shipped_max"]) == (
        0.0,
        0.0,
        0.0,
    )
    kl = pc._first_step_kl(other, shipped)
    assert kl["kl_pilot_shipped"] > 0 and kl["kl_shipped_pilot"] > 0
    assert kl["kl_pilot_shipped_max"] >= kl["kl_pilot_shipped"] and kl["decisions"] == 3
    log_p = torch.log_softmax(other.s1_logits[:, 0].double(), -1)
    log_q = torch.log_softmax(shipped.s1_logits[:, 0].double(), -1)
    expected = (log_p.exp() * (log_p - log_q)).sum(-1).mean().item()
    assert kl["kl_pilot_shipped"] == pytest.approx(expected)
    shifted = replace(other, s1_targets=other.s1_targets + 1)
    with pytest.raises(pc.PilotCheckError, match="different tokens"):
        pc._first_step_kl(shifted, shipped)


def test_the_agreement_statistics_of_known_signals_and_p_up():
    def fans(signals, p_up):
        return [
            FanSummary(
                "binance",
                "2024-04-14T16:00Z",
                "s",
                "x",
                "0" * 64,
                "0" * 64,
                s,
                p,
                1.0,
                1.0,
                0.0,
                True,
            )
            for s, p in zip(signals, p_up, strict=True)
        ]

    shipped = [1, -1, 1, 1, -1, 0, 1, -1]
    pilot = [1, -1, -1, 1, -1, 0, 1, 1]  # 6 of 8 agree
    floor = [-1, -1, 1, 1, 1, 0, 1, -1]  # 6 of 8 agree with shipped
    p_shipped = [0.5, 0.2, 0.7, 0.6, 0.3, 0.5, 0.9, 0.1]
    p_pilot = [0.55, 0.2, 0.4, 0.6, 0.3, 0.45, 0.9, 0.6]
    summaries = {
        "pilot": fans(pilot, p_pilot),
        "shipped": fans(shipped, p_shipped),
        "floor": fans(floor, p_shipped),
    }
    a_same, a_floor, delta = pc._agreement(summaries)
    assert (a_same, a_floor) == (0.75, 0.75)
    assert delta == pytest.approx((0.05 + 0.3 + 0.05 + 0.5) / 8)
    with pytest.raises(pc.PilotCheckError):
        pc._agreement({**summaries, "floor": summaries["floor"][:7]})


def test_the_fold_record_curve_is_read_and_checked():
    record = {"selected_pseudo_epoch": 2, "stop_reason": "early_stop", "epochs": CURVE}
    curve = pc._nll4_tf_curve(record)
    assert [e["pseudo_epoch"] for e in curve["epochs"]] == [0, 1, 2, 3, 4]
    assert curve["epochs"][0]["nll4_tf"] == 5.10 and curve["epochs"][0]["train_loss"] is None
    assert (curve["selected_pseudo_epoch"], curve["stop_reason"]) == (2, "early_stop")
    for bad in (
        {**record, "epochs": []},
        {**record, "selected_pseudo_epoch": None},
        {**record, "selected_pseudo_epoch": 0},
        {**record, "epochs": [{"pseudo_epoch": "1"}]},
    ):
        with pytest.raises(pc.PilotCheckError):
            pc._nll4_tf_curve(bad)


# -- preparation ------------------------------------------------------------------------------


def test_prepared_batches_carry_the_role_the_pilot_line_and_the_three_streams(pilot, tiny):
    steps, _ = steps_for(pilot, tiny)
    prepared = prepare(pilot, steps)
    decisions = pc.check_decisions_ms(FOLD)
    assert prepared.decisions_ms == decisions and prepared.dropped == ()
    assert set(prepared.batches) == {"pilot", "shipped", "floor"}
    for name, stream in (("pilot", "forecast"), ("shipped", "forecast"), ("floor", "forecast-aa")):
        prepared_batch = prepared.batches[name]
        assert prepared_batch.period_role == MODEL_PERIOD_PILOT_CHECK
        assert {task.stream for task in prepared_batch.tasks} == {stream}
        assert [to_ms(task.decision_hour) for task in prepared_batch.tasks] == list(decisions)
        assert prepared_batch.pending == 168 and prepared_batch.evaluation_length is None
    assert prepared.batches["pilot"].checkpoint == CheckpointRef(
        CHECKPOINT_ID, pilot.ft, line_sha256=pilot.entry.line_sha256
    )
    assert prepared.batches["shipped"].checkpoint is None
    assert prepared.batches["floor"].checkpoint is None
    assert prepared.batches["pilot"].config_id != prepared.batches["shipped"].config_id
    # the NLL4-TF inputs are the export's restricted to the G8 decisions
    index = [list(FOLD.nll_decisions_ms()).index(d) for d in decisions]
    assert prepared.nll_inputs.decision_ms.tolist() == list(decisions)
    assert np.array_equal(prepared.nll_inputs.values, pilot.values[index])
    assert prepared.config_sha256 == sha256_file(pilot.reports / "pilot-check-0ee0d123.config.json")


def test_decisions_missing_from_the_inputs_or_a_window_are_dropped_everywhere(
    tmp_path, market, tiny
):
    decisions = pc.check_decisions_ms(FOLD)
    without = [d for d in market.g4 if d != decisions[3]]  # an NLL4-TF decision lost to a gap
    gap = decisions[100] - 2 * HOUR_MS  # a candle in the windows of decisions 100 onwards
    world = make_pilot(tmp_path, market, decisions=without, missing=(gap,))
    steps, _ = steps_for(world, tiny)
    prepared = prepare(world, steps)
    lost = {decisions[3]} | {d for d in decisions if d - 512 * HOUR_MS <= gap < d}
    assert set(prepared.decisions_ms) == set(decisions) - lost
    for prepared_batch in prepared.batches.values():
        assert [to_ms(t.decision_hour) for t in prepared_batch.tasks] == list(prepared.decisions_ms)
    listed = {item["decision_ms"]: item for item in prepared.dropped}
    assert set(listed) == lost  # each dropped decision listed once
    assert (
        listed[decisions[3]]["batches"] == [] and "NLL4-TF inputs" in listed[decisions[3]]["reason"]
    )
    gap_entry = listed[decisions[100]]
    assert gap_entry["batches"] == ["pilot", "shipped", "floor"]
    assert gap_entry["missing"] == [[utc_text(gap), utc_text(gap + HOUR_MS)]]
    assert prepared.nll_inputs.decision_ms.tolist() == list(prepared.decisions_ms)


def test_a_changed_fold_record_or_registry_line_is_refused(tmp_path, market, tiny):
    world = make_pilot(tmp_path, market)
    steps, _ = steps_for(world, tiny)
    record = world.inputs.fit_record_path
    original = record.read_bytes()
    record.write_bytes(
        original.replace(b'"selected_pseudo_epoch": 2', b'"selected_pseudo_epoch": 3')
    )
    with pytest.raises(pc.PilotCheckError, match="fold record"):
        prepare(world, steps)
    record.write_bytes(original)
    prepare(world, steps)
    ff.edit_last_line(world.ft, threads=3)
    with pytest.raises(RegistryChainError, match="changed after it was recorded"):
        prepare(world, steps)


def test_inputs_and_params_are_checked(pilot, tiny):
    steps, _ = steps_for(pilot, tiny)
    inputs = pilot.inputs
    for bad in (
        replace(inputs, checkpoint=CheckpointRef(CHECKPOINT_ID, pilot.ft)),
        replace(inputs, checkpoint=CheckpointRef("ks-btc-0ee0d123-q-2024-10", pilot.ft, "0" * 64)),
        replace(inputs, fit_record_sha256="0" * 64),
        replace(inputs, fold=replace(FOLD, replicate=1)),
    ):
        with pytest.raises(pc.PilotCheckError):
            pc.prepare_check(
                bad, params=PARAMS, steps=steps, root=pilot.data, directory=pilot.reports
            )
    for kronos in (replace(PARAMS.kronos, horizon_n=8), replace(PARAMS.kronos, trajectories=30)):
        other = replace(PARAMS, kronos=kronos)
        with pytest.raises(pc.PilotCheckError, match="4 steps and 20"):
            pc.prepare_check(
                inputs, params=other, steps=steps, root=pilot.data, directory=pilot.reports
            )


def test_an_entry_that_does_not_describe_the_pilot_is_refused(tmp_path, market, tiny):
    world = make_pilot(tmp_path, market)
    other = make_pilot(tmp_path / "other", market, decisions=market.g4[:200])
    inputs = replace(world.inputs, export=other.inputs.export)  # another export than the entry's
    steps, _ = steps_for(world, tiny)
    with pytest.raises(pc.PilotCheckError, match="export_content_sha256, nll_inputs_sha256"):
        pc.prepare_check(
            inputs, params=PARAMS, steps=steps, root=world.data, directory=world.reports
        )


# -- the estimate -----------------------------------------------------------------------------


def test_the_estimate_arithmetic(pilot, tiny):
    steps, calls = steps_for(pilot, tiny)
    prepared = prepare(pilot, steps)
    estimate = pc.estimate_check(prepared, params=PARAMS, steps=steps)
    assert calls == [{"params": PARAMS, "checkpoint": prepared.batches["pilot"].checkpoint}]
    assert calls[0]["checkpoint"].line_sha256 == pilot.entry.line_sha256
    assert estimate["seconds_per_fan"] == 3.0625 and estimate["pending"] == 504
    assert estimate["fans_cpu_seconds"] == pytest.approx(504 * 3.0625 * 2)
    assert estimate["kl_cpu_seconds"] == pytest.approx(2 * 168 * 0.16 * 2)
    assert estimate["cpu_seconds"] == pytest.approx(3087.0 + 107.52)
    assert estimate["cpu_hours"] == pytest.approx((3087.0 + 107.52) / 3600)
    assert estimate["needs_owner_approval"] is False and estimate["gate3_cpu_seconds"] == 3600
    assert {name: e["pending"] for name, e in estimate["batches"].items()} == {
        "pilot": 168,
        "shipped": 168,
        "floor": 168,
    }
    batch.require_gate3(estimate, approved_cpu_hours=None)
    slow = replace(steps, measure_seconds_per_fan=lambda **kwargs: 3.5)
    dear = pc.estimate_check(prepared, params=PARAMS, steps=slow)
    assert dear["cpu_seconds"] == pytest.approx(504 * 3.5 * 2 + 107.52)
    assert dear["needs_owner_approval"] is True
    with pytest.raises(batch.Gate3Refused):
        batch.require_gate3(dear, approved_cpu_hours=None)
    json.dumps(estimate)  # the approval entry records it


# -- runs -------------------------------------------------------------------------------------


def test_a_run_with_identical_fans_reads_no_go_and_writes_its_report(pilot, tiny):
    needs_crps()
    steps, _ = steps_for(pilot, tiny)
    prepared = prepare(pilot, steps)
    lines: list[str] = []
    result = pc.run_check(
        prepared,
        params=PARAMS,
        steps=steps,
        root=pilot.data,
        directory=pilot.reports,
        progress=lines.append,
    )
    statistics = result.report["statistics"]
    assert result.reading == "NO-GO" and statistics["a_same"] == 1.0
    assert statistics["mean_abs_delta_p_up"] == 0.0 and statistics["crps"]["d_sd_bp"] == 0.0
    assert 0.0 < statistics["a_floor"] < 1.0
    assert (
        statistics["kl_pilot_shipped"] > 0 and statistics["relative_weight_change"]["relative"] > 0
    )
    report = result.report
    assert report["study"] == "pilot-check-0ee0d123" and report["tests_consumed"] == 0
    assert (
        report["statistics_with_intervals"] == 0
        and "fewer than the 20" in (report["statistics_with_intervals_reason"])
    )
    assert report["checkpoint"]["registry_line_sha256"] == pilot.entry.line_sha256
    assert report["checkpoint"]["data_end"] == "2024-06-09T16:00Z"
    assert report["fit_record_sha256"] == pilot.inputs.fit_record_sha256
    assert report["role"]["period_role"] == "pilot-check"
    assert report["decisions"] == {**report["decisions"], "planned": 168, "kept": 168}
    assert {name: fan["fans"] for name, fan in report["fans"].items()} == {
        "pilot": 168,
        "shipped": 168,
        "floor": 168,
    }
    assert report["nll4_tf_curve"]["selected_pseudo_epoch"] == 2
    assert [e["nll4_tf"] for e in report["nll4_tf_curve"]["epochs"]] == [
        e["nll4_tf"] for e in CURVE
    ]
    assert report["reading"]["no_go"] == pc.NO_GO_OPTIONS
    figure = report["figures"][0]
    assert figure["relpath"] == "figures/pilot-check-0ee0d123-nll4tf.png"
    assert sha256_file(pilot.reports / figure["relpath"]) == figure["sha256"]
    text = result.json_path.read_text("ascii") + result.markdown_path.read_text("ascii")
    assert str(pilot.data.parent) not in text and str(REPO_ROOT) not in text
    assert "## Reading: NO-GO" in result.markdown_path.read_text("ascii")
    details = result.details
    assert set(details) == {
        "reading",
        "a_same",
        "a_floor",
        "mean_abs_delta_p_up",
        "kl_pilot_shipped",
        "relative_weight_change",
        "d_sd_bp",
        "mdd_bp",
        "decisions",
        "report_sha256",
        "config_sha256",
        "checkpoint_id",
        "registry_line_sha256",
    }
    assert details["report_sha256"] == sha256_file(result.json_path)
    assert details["checkpoint_id"] == CHECKPOINT_ID and details["decisions"] == 168
    canonical_json(details)  # the ledger's exploration entry records it
    assert any("pilot fans" in line for line in lines)
    assert result.summary_lines[0].startswith("pilot-check-0ee0d123: 168 of 168 G8 decisions")
    # the same inputs give the same figure (the fans are reused from the store)
    again = pc.run_check(
        prepare(pilot, steps),
        params=PARAMS,
        steps=steps,
        root=pilot.data,
        directory=pilot.reports,
        progress=lines.append,
    )
    assert again.report["figures"] == report["figures"]
    assert again.report["fans"]["pilot"]["computed"] == 0


def test_a_run_with_another_pilot_reads_go(pilot, tiny):
    needs_crps()
    steps, _ = steps_for(pilot, tiny, pilot_predictor=DriftPredictor())
    result = pc.run_check(
        prepare(pilot, steps),
        params=PARAMS,
        steps=steps,
        root=pilot.data,
        directory=pilot.reports,
        progress=lambda line: None,
    )
    statistics = result.report["statistics"]
    assert result.reading == "GO" and statistics["a_same"] < 0.98
    assert statistics["crps"]["d_sd_bp"] > 0 and statistics["projection"]["mdd_bp"] > 0
    assert result.report["reading"]["no_go"] is None
    assert result.report["fans"]["pilot"]["checkpoint_id"] == CHECKPOINT_ID


def summaries_for(prepared, *, signals, p_up, closes, last_close):
    """Stand-in fans: summaries with the given signals and p_up, and stored paths whose last-step
    closes are ``closes[name][i]`` (20 values)."""
    fans: dict[str, SimpleNamespace] = {}
    found: dict[str, list[FanSummary]] = {}
    for name, prepared_batch in prepared.batches.items():
        found[name] = []
        for index, task in enumerate(prepared_batch.tasks):
            relpath = f"forecasts/binance/{name}/{task.stream}/{index}.npz"
            paths = np.ones((20, 4, 6)) * last_close[index]
            paths[:, 3, CLOSE] = closes[name][index]
            decision = to_ms(task.decision_hour)
            meta = {"decision_ms": decision, "stream": task.stream}
            fans[relpath] = SimpleNamespace(paths=paths, meta=meta)
            found[name].append(
                FanSummary(
                    source="binance",
                    decision_hour=utc_text(decision),
                    stream=task.stream,
                    relpath=relpath,
                    window_sha256="0" * 64,
                    inputs_sha256="0" * 64,
                    signal=int(signals[name][index]),
                    p_up=float(p_up[name][index]),
                    last_close=float(last_close[index]),
                    mean_close_n=float(np.mean(closes[name][index])),
                    elapsed_s=0.0,
                    computed=True,
                )
            )
    return found, fans


def test_the_statistics_on_stand_in_fans_with_known_signals_and_p_up(pilot, tiny):
    needs_crps()
    steps, _ = steps_for(pilot, tiny, same_model=True)
    prepared = prepare(pilot, steps)
    n = len(prepared.decisions_ms)
    values = prepared.nll_inputs.values
    last_close = values[:, 511, CLOSE]
    realised = values[:, 515, CLOSE]
    generator = np.random.default_rng(3)
    signals = {"shipped": generator.choice([-1, 1], size=n)}
    signals["pilot"] = signals["shipped"].copy()
    signals["pilot"][:10] *= -1  # 10 disagreements: A_same = 158 / 168
    signals["floor"] = signals["shipped"].copy()
    signals["floor"][-20:] *= -1  # A_floor = 148 / 168
    p_up = {"shipped": generator.uniform(0.2, 0.7, size=n)}
    p_up["pilot"] = p_up["shipped"] + 0.05
    p_up["floor"] = p_up["shipped"]
    closes = {
        name: last_close[:, None] * np.exp(generator.normal(0.0, 0.01, size=(n, 20)))
        for name in pc.BATCHES
    }
    found, fans = summaries_for(
        prepared, signals=signals, p_up=p_up, closes=closes, last_close=last_close
    )
    steps = replace(
        steps,
        run_fan_batch=lambda prepared_batch, **kwargs: found[
            next(name for name, b in prepared.batches.items() if b is prepared_batch)
        ],
        load_fan=lambda path: fans[Path(path).relative_to(pilot.data).as_posix()],
    )
    result = pc.run_check(
        prepared,
        params=PARAMS,
        steps=steps,
        root=pilot.data,
        directory=pilot.reports,
        progress=lambda line: None,
    )
    statistics = result.report["statistics"]
    assert statistics["a_same"] == pytest.approx(158 / 168)
    assert statistics["a_floor"] == pytest.approx(148 / 168)
    assert statistics["mean_abs_delta_p_up"] == pytest.approx(0.05)
    assert result.reading == "GO"  # 158 / 168 = 0.9405 < 0.98
    y = 1e4 * np.log(realised / last_close)
    d = crps_oracle(1e4 * np.log(closes["pilot"] / last_close[:, None]), y) - crps_oracle(
        1e4 * np.log(closes["shipped"] / last_close[:, None]), y
    )
    crps = statistics["crps"]
    assert crps["d_mean_bp"] == pytest.approx(float(np.mean(d)))
    assert crps["d_sd_bp"] == pytest.approx(float(np.std(d, ddof=1)))
    projection = pc.mdd_projection(d)
    assert statistics["projection"]["mdd_bp"] == pytest.approx(projection["mdd_bp"])
    assert statistics["kl_pilot_shipped"] == 0.0  # the same tiny model twice
    assert statistics["relative_weight_change"]["relative"] == 0.0


def test_the_c_ref_cross_check_refuses_a_shifted_input(tmp_path, market, tiny):
    shifted = nll_values(market, market.g4, shift_hours=-1)  # rows one hour early
    world = make_pilot(tmp_path, market, values=shifted)
    steps, _ = steps_for(world, tiny)
    prepared = prepare(world, steps)

    def run_fan_batch(prepared_batch, **kwargs):  # the true last closes of each window
        last = {to_ms(w.decision_hour): w.last_close for w in prepared_batch.windows}
        return [
            FanSummary(
                "binance",
                utc_text(to_ms(t.decision_hour)),
                t.stream,
                "x.npz",
                "0" * 64,
                "0" * 64,
                1,
                0.5,
                last[to_ms(t.decision_hour)],
                1.0,
                0.0,
                True,
            )
            for t in prepared_batch.tasks
        ]

    steps = replace(steps, run_fan_batch=run_fan_batch)
    with pytest.raises(pc.PilotCheckError, match="misaligned inputs"):
        pc.run_check(prepared, params=PARAMS, steps=steps, root=world.data, directory=world.reports)


def test_a_definition_or_record_changed_after_the_preparation_stops_the_run_before_any_fan(
    pilot, tiny
):
    steps, _ = steps_for(pilot, tiny)
    prepared = prepare(pilot, steps)

    def no_fans(*args, **kwargs):
        raise AssertionError("a fan batch ran")

    steps = replace(steps, run_fan_batch=no_fans)
    config = pilot.reports / "pilot-check-0ee0d123.config.json"
    original = config.read_bytes()
    config.write_bytes(original.replace(b'"workers":1', b'"workers":2'))
    with pytest.raises(ConfigChangedError):
        pc.run_check(prepared, params=PARAMS, steps=steps, root=pilot.data, directory=pilot.reports)
    config.write_bytes(original)
    record = pilot.inputs.fit_record_path
    record.write_bytes(record.read_bytes() + b"\n")
    with pytest.raises(pc.PilotCheckError, match="fold record"):
        pc.run_check(prepared, params=PARAMS, steps=steps, root=pilot.data, directory=pilot.reports)
    assert not (pilot.reports / "figures").exists()


def test_a_batch_that_is_not_the_one_asked_for_is_refused(pilot, tiny):
    steps, _ = steps_for(pilot, tiny)

    def evaluation_role(tasks, **kwargs):
        prepared_batch = batch.prepare_tasks(tasks, **kwargs)
        return replace(prepared_batch, period_role="timing")

    with pytest.raises(pc.PilotCheckError, match="role"):
        prepare(pilot, replace(steps, prepare_tasks=evaluation_role))

    def without_line(tasks, **kwargs):
        prepared_batch = batch.prepare_tasks(tasks, **kwargs)
        if prepared_batch.checkpoint is None:
            return prepared_batch
        return replace(
            prepared_batch, checkpoint=replace(prepared_batch.checkpoint, line_sha256=None)
        )

    with pytest.raises(pc.PilotCheckError, match="registry line"):
        prepare(pilot, replace(steps, prepare_tasks=without_line))


def test_real_steps_are_the_batch_layer_the_store_and_nll():
    steps = pc.real_steps()
    assert steps.prepare_tasks is batch.prepare_tasks
    assert steps.measure_seconds_per_fan is batch.measure_seconds_per_fan
    assert steps.run_fan_batch is batch.run_fan_batch
    assert steps.load_fan is load_fan
    assert steps.teacher_forced_logits is teacher_forced_logits
    assert steps.load_models is pc._load_models_quietly
    assert steps.clock is time.monotonic and steps.cpu_seconds is pc._process_cpu_seconds
    assert isinstance(steps.cpu_seconds(), float) and steps.cpu_seconds() > 0


# -- spend control (design 3.5) -----------------------------------------------------------------


def test_the_spend_limit():
    assert pc.spend_limit(None) == pc.GATE3_CPU_SECONDS == batch.GATE3_CPU_SECONDS == 3600
    assert pc.APPROVAL_TOLERANCE == batch.GATE3_APPROVAL_TOLERANCE == 1.25
    assert pc.spend_limit(1.5) == 6750.0 and pc.spend_limit(1) == 4500.0
    for bad in (0, -1.0, float("nan"), float("inf"), True, "1.5"):
        with pytest.raises(ValueError, match="finite number > 0"):
            pc.spend_limit(bad)


@pytest.fixture
def crps_stub(monkeypatch):
    """``kronos_agent.eval.crps`` as the fair CRPS of the oracle while the module is not built
    (interfaces 9): these tests check the spend control, not the CRPS."""
    try:
        import kronos_agent.eval.crps  # noqa: F401
    except ImportError:
        module = types.ModuleType("kronos_agent.eval.crps")
        module.fair_crps = crps_oracle
        monkeypatch.setitem(sys.modules, "kronos_agent.eval.crps", module)


class Meter:
    """A stand-in clock and CPU counter, advanced by the stand-in fans and passes; ``stored``
    counts the fans each batch stored before the run stopped (the batch layer stores a fan before
    it reports it to ``progress``)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.cpu = 50.0
        self.stored = {name: 0 for name in pc.BATCHES}

    def clock(self) -> float:
        return self.now

    def cpu_seconds(self) -> float:
        return self.cpu


def metered(world, tiny, *, fan=(0.0, 0.0), kl=(0.0, 0.0)):
    """The prepared check and steps whose stand-in fans (identical in the three batches) and
    teacher-forced passes advance a :class:`Meter` by ``(wall, cpu)`` seconds each."""
    steps, _ = steps_for(world, tiny, same_model=True)
    prepared = prepare(world, steps)
    n = len(prepared.decisions_ms)
    last_close = prepared.nll_inputs.values[:, 511, CLOSE]
    ones = {name: np.ones(n) for name in pc.BATCHES}
    closes = {name: last_close[:, None] * np.ones((n, 20)) for name in pc.BATCHES}
    found, fans = summaries_for(
        prepared, signals=ones, p_up=ones, closes=closes, last_close=last_close
    )
    meter = Meter()

    def run_fan_batch(prepared_batch, *, workers, params, root, progress=None):
        name = next(name for name, item in prepared.batches.items() if item is prepared_batch)
        for done, summary in enumerate(found[name], 1):
            meter.now += fan[0]
            meter.cpu += fan[1]
            meter.stored[name] += 1
            if progress is not None:
                progress(done, len(found[name]), summary)
        return found[name]

    def teacher_forced(tokenizer, model, inputs):
        meter.now += kl[0]
        meter.cpu += kl[1]
        return teacher_forced_logits(tokenizer, model, inputs)

    steps = replace(
        steps,
        run_fan_batch=run_fan_batch,
        load_fan=lambda path: fans[Path(path).relative_to(world.data).as_posix()],
        teacher_forced_logits=teacher_forced,
        clock=meter.clock,
        cpu_seconds=meter.cpu_seconds,
    )
    return prepared, steps, meter


def run_metered(world, prepared, steps, approved=None):
    return pc.run_check(
        prepared,
        params=PARAMS,
        steps=steps,
        root=world.data,
        directory=world.reports,
        progress=lambda line: None,
        approved_cpu_hours=approved,
    )


def test_without_a_figure_the_run_stops_between_fans_past_gate_3(pilot, tiny):
    """100 s of wall per fan at the governed 2 threads is charged 200 CPU-seconds (D-033): the
    19th fan passes 3,600 CPU-seconds, and the run stops before the 20th. The 19 fans stay
    stored, no other batch starts, nothing is written."""
    prepared, steps, meter = metered(pilot, tiny, fan=(100.0, 1.0))
    with pytest.raises(pc.PilotCheckStopped) as info:
        run_metered(pilot, prepared, steps)
    message = str(info.value)
    assert "passed its limit of 3600 (gate 3's 3600 CPU-seconds, without an approved" in message
    assert "spend, 3800 CPU-seconds" in message and "after 19 of the 168 pilot fans" in message
    assert "the fans stored so far are kept" in message
    assert meter.stored == {"pilot": 19, "shipped": 0, "floor": 0}
    assert not (pilot.reports / "pilot-check-0ee0d123.json").exists()
    assert not (pilot.reports / "figures").exists()
    assert issubclass(pc.PilotCheckStopped, PermissionError)  # a refusal (exit 2)


def test_the_measured_cpu_counts_when_it_passes_wall_times_threads(pilot, tiny):
    prepared, steps, meter = metered(pilot, tiny, fan=(0.0, 250.0))
    with pytest.raises(pc.PilotCheckStopped, match="after 15 of the 168 pilot fans"):
        run_metered(pilot, prepared, steps)
    assert meter.stored["pilot"] == 15


def test_an_approved_figure_sets_the_limit_and_a_run_within_it_completes(pilot, tiny, crps_stub):
    """5 s per fan (10 CPU-seconds) and 100 s per teacher-forced pass: 5,440 CPU-seconds in
    all. Under 1.5 approved CPU-hours (a limit of 6,750) the run completes and records its
    spend; without a figure (3,600) it stops at the 25th fan of the third batch."""
    prepared, steps, meter = metered(pilot, tiny, fan=(5.0, 1.0), kl=(100.0, 1.0))
    result = run_metered(pilot, prepared, steps, approved=1.5)
    spend = result.report["spend"]
    assert spend["cpu_seconds"] == pytest.approx(5440.0) and spend["wall_seconds"] == 2720.0
    assert (spend["limit_cpu_seconds"], spend["approved_cpu_hours"]) == (6750.0, 1.5)
    assert result.details["cpu_seconds"] == pytest.approx(5440.0)
    assert result.summary_lines[-1].startswith("spend 5440 CPU-seconds (wall 2720 s)")
    assert "6750 (1.5 CPU-hours approved)" in result.summary_lines[-1]
    assert "- Run: spend 5440 CPU-seconds" in result.markdown_path.read_text(encoding="ascii")
    prepared, steps, meter = metered(pilot, tiny, fan=(5.0, 1.0), kl=(100.0, 1.0))
    with pytest.raises(pc.PilotCheckStopped, match="after 25 of the 168 floor fans"):
        run_metered(pilot, prepared, steps)
    assert meter.stored == {"pilot": 168, "shipped": 168, "floor": 25}


def test_the_spend_is_checked_between_the_teacher_forced_passes(pilot, tiny, crps_stub):
    prepared, steps, _ = metered(pilot, tiny, kl=(2000.0, 1.0))  # 4,000 CPU-seconds a pass
    with pytest.raises(pc.PilotCheckStopped, match="between the two teacher-forced passes"):
        run_metered(pilot, prepared, steps)
    assert not (pilot.reports / "pilot-check-0ee0d123.json").exists()
