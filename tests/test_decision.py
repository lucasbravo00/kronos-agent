"""Forecast of a decision window: governed-parameter checks, project-symbol seeds, the signal.

Most tests run a stand-in model: a ``LoadedModels`` whose predictor draws a random walk from the
last context close with numpy's global generator, which ``seed_everything`` seeds before every
fan (so seeds, determinism and the inputs passed to the model are observable), or a predictor
that must never be reached. Models in the state ``load_models`` leaves Kronos in are imitated with
small torch modules (eval mode, no gradients) shared with a stand-in predictor that carries the
clip, context length and device of a ``KronosPredictor``. The tests marked ``weights`` run real
Kronos-small fans with 4 trajectories instead of the governed 20 (the properties tested do not
depend on the fan size).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes
import torch

from kronos_agent.data.forecast_store import CONFIG_KEYS, config_id, forecast_config
from kronos_agent.data.partition import KRONOS_FIRST_DECISION, PartitionError
from kronos_agent.data.timeutil import to_ms, utc_text
from kronos_agent.data.windows import build_window, naive_timestamps
from kronos_agent.kronos.decision import (
    CLOSE_INDEX,
    DecisionError,
    evaluation_params,
    fan_mean_close,
    fan_p_up,
    fan_signal,
    forecast_window,
)
from kronos_agent.kronos.features import FEATURE_COLUMNS
from kronos_agent.kronos.weights import PREDICTOR_CLIP, LoadedModels
from kronos_agent.models import spec_for
from kronos_agent.params import load_params
from kronos_agent.seeds import seed_for_decision

HOUR_MS = 3_600_000
D = datetime(2025, 3, 3, 12, tzinfo=UTC)  # inside validation, after the Kronos cutoff
D_MS = to_ms(D)
FIRST_MS = D_MS - 600 * HOUR_MS
BINGX = fakes.synthetic_candles(FIRST_MS, 700, seed=3)
BINANCE_FRAME = fakes.binance_frame(FIRST_MS, 700, interval="1h", seed=4)
BINANCE = fakes.frame_to_candles(BINANCE_FRAME)
PARAMS = load_params()
HORIZON = PARAMS.kronos.horizon_n
SMALL_PARAMS = replace(PARAMS, kronos=replace(PARAMS.kronos, trajectories=4))


class WalkPredictor:
    """``predict_batch`` stand-in: a seeded random walk from the last close of each context."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        T,
        top_k,
        top_p,
        sample_count,
        verbose,
    ):
        self.calls.append(
            {
                "batch": len(df_list),
                "rows": len(df_list[0]),
                "columns": list(df_list[0].columns),
                "pred_len": pred_len,
                "last_close": float(df_list[0]["close"].iloc[-1]),
                "x_first": x_timestamp_list[0].iloc[0],
                "x_last": x_timestamp_list[0].iloc[-1],
                "y": list(y_timestamp_list[0]),
                "sampling": (T, top_k, top_p, sample_count),
            }
        )
        frames = []
        for df, y in zip(df_list, y_timestamp_list, strict=True):
            last = float(df["close"].iloc[-1])
            close = last * np.exp(np.cumsum(np.random.normal(0.0, 0.01, size=pred_len)))
            ones = np.ones(pred_len)
            values = np.column_stack([close, close * 1.001, close * 0.999, close, ones, close])
            frames.append(pd.DataFrame(values, columns=list(FEATURE_COLUMNS), index=y))
        return frames


class UnreachablePredictor:
    def predict_batch(self, *args, **kwargs):
        raise AssertionError("the model must not run for a refused window")


def fake_models(predictor=None, *, device: str = "cpu") -> LoadedModels:
    return LoadedModels(
        spec=spec_for("small"),
        tokenizer=None,
        model=None,
        predictor=predictor if predictor is not None else WalkPredictor(),
        weights_sha256={"fake/model.safetensors": "0" * 64},
        load_seconds=0.0,
        device=device,
    )


class BuiltPredictor(WalkPredictor):
    """The stand-in walk, carrying what ``load_models`` builds a ``KronosPredictor`` with."""

    def __init__(self, tokenizer, model, *, clip=PREDICTOR_CLIP, max_context=512, device="cpu"):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.clip = clip
        self.max_context = max_context
        self.device = device


class UnreachableBuiltPredictor(BuiltPredictor):
    def predict_batch(self, *args, **kwargs):
        raise AssertionError("the model must not run for refused models")


def frozen_module() -> torch.nn.Module:
    """A small torch module in the state ``load_models`` leaves Kronos in: eval, no gradients."""
    module = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Dropout(0.1))
    return module.eval().requires_grad_(False)


def built_models(predictor_class=BuiltPredictor, **predictor_changes) -> LoadedModels:
    """Stand-in models whose tokenizer and model are torch modules shared with the predictor."""
    tokenizer, model = frozen_module(), frozen_module()
    predictor = predictor_class(tokenizer, model, **predictor_changes)
    return replace(fake_models(), tokenizer=tokenizer, model=model, predictor=predictor)


def window_of(candles=BINGX, hour=D, *, source="bingx", **kwargs):
    kwargs.setdefault("horizon", HORIZON)
    return build_window(candles, hour, source=source, **kwargs)


def make_paths(closes_by_step, *, other=1.0):
    """``(K, N, 6)`` paths: close ``closes_by_step[k][n]``, every other channel ``other``."""
    closes = np.asarray(closes_by_step, dtype=np.float64)
    paths = np.full((*closes.shape, len(FEATURE_COLUMNS)), other, dtype=np.float64)
    paths[:, :, CLOSE_INDEX] = closes
    return paths


# -- the signal of a fan ---------------------------------------------------------------------------


def test_close_index_is_the_close_channel():
    assert FEATURE_COLUMNS[CLOSE_INDEX] == "close"
    assert CLOSE_INDEX == 3


def test_signal_is_the_sign_of_the_mean_close_at_the_last_step():
    # last step: mean 97.67 < 100 although two of three trajectories end above 100 (a median or a
    # majority vote would say up); the first step is far above 100 and the other channels are 200
    # (reading the wrong step or channel would say up)
    paths = make_paths(
        [[110.0, 105.0, 90.0], [110.0, 105.0, 101.0], [110.0, 105.0, 102.0]], other=200.0
    )
    assert fan_signal(paths, 100.0) == -1
    assert fan_p_up(paths, 100.0) == pytest.approx(2 / 3)
    assert fan_mean_close(paths) == pytest.approx((90.0 + 101.0 + 102.0) / 3)
    up = make_paths([[90.0, 99.0, 103.0], [90.0, 99.0, 99.0]])
    assert fan_signal(up, 100.0) == 1  # mean 101
    assert fan_p_up(up, 100.0) == 0.5


def test_signal_zero_and_p_up_strictly_above():
    flat = make_paths([[99.0], [101.0]])  # mean exactly 100
    assert fan_signal(flat, 100.0) == 0
    assert fan_p_up(flat, 100.0) == 0.5
    ties = make_paths([[100.0], [100.0], [101.0]])
    assert fan_p_up(ties, 100.0) == pytest.approx(1 / 3)  # equal is not above
    assert fan_signal(ties, 100.0) == 1
    assert isinstance(fan_signal(ties, 100.0), int)
    assert isinstance(fan_p_up(ties, 100.0), float)
    assert fan_p_up(make_paths([[50.0], [60.0]]), 100.0) == 0.0
    assert fan_p_up(make_paths([[150.0], [160.0]]), np.float64(100.0)) == 1.0


@pytest.mark.parametrize(
    "paths, last_close",
    [
        (np.ones((2, 4)), 100.0),  # not three-dimensional
        (np.ones((2, 4, 5)), 100.0),  # five channels
        (np.ones((0, 4, 6)), 100.0),  # no trajectory
        (np.ones((2, 0, 6)), 100.0),  # no step
        (make_paths([[np.nan], [1.0]]), 100.0),
        (make_paths([[np.inf], [1.0]]), 100.0),
        (np.ones((2, 4, 6), dtype=np.int64), 100.0),
        ([[[1.0] * 6]], 100.0),  # a list, not an array
        (make_paths([[1.0]]), 0.0),
        (make_paths([[1.0]]), -5.0),
        (make_paths([[1.0]]), float("nan")),
        (make_paths([[1.0]]), True),
        (make_paths([[1.0]]), "100.0"),
    ],
)
def test_signal_inputs_are_validated(paths, last_close):
    with pytest.raises(DecisionError):
        fan_signal(paths, last_close)
    with pytest.raises(DecisionError):
        fan_p_up(paths, last_close)


# -- forecast_window with a stand-in model ---------------------------------------------------------


def test_seeds_use_the_project_symbol_for_every_source():
    models = fake_models()
    bingx = window_of(BINGX, source="bingx")
    binance = window_of(BINANCE, source="binance")
    assert (bingx.symbol, binance.symbol) == ("BTC-USDT", "BTCUSDT")
    fan_bingx = forecast_window(models, bingx, PARAMS)
    fan_binance = forecast_window(models, binance, PARAMS)
    expected = seed_for_decision(
        D, stream="forecast", symbol="BTC-USDT", salt=PARAMS.kronos.seed_salt
    )
    assert fan_bingx.seeds == fan_binance.seeds == expected
    assert fan_binance.seeds.canonical.startswith(
        "kronos-agent/seed/v1/forecast/BTC-USDT/2025-03-03T12:00:00Z"
    )
    assert "BTCUSDT" not in fan_binance.seeds.canonical
    assert not np.array_equal(fan_bingx.paths, fan_binance.paths)  # same seeds, other data


def test_another_stream_gives_other_seeds_and_another_fan():
    models = fake_models()
    window = window_of()
    forecast = forecast_window(models, window, PARAMS)
    floor = forecast_window(models, window, PARAMS, stream="mismatch-floor")
    assert floor.seeds != forecast.seeds
    assert floor.seeds.canonical.split("/")[3] == "mismatch-floor"
    assert floor.seeds.torch_seed != forecast.seeds.torch_seed
    assert floor.seeds.numpy_seed != forecast.seeds.numpy_seed
    assert not np.array_equal(floor.paths, forecast.paths)


def test_the_model_receives_exactly_the_window():
    predictor = WalkPredictor()
    window = window_of()
    fan = forecast_window(fake_models(predictor), window, PARAMS)
    (call,) = predictor.calls
    kronos = PARAMS.kronos
    assert call["batch"] == kronos.trajectories
    assert call["rows"] == kronos.max_context == 512
    assert call["columns"] == list(FEATURE_COLUMNS)
    assert call["pred_len"] == kronos.horizon_n
    assert call["sampling"] == (kronos.temperature, kronos.top_k, kronos.top_p, 1)
    assert call["last_close"] == window.last_close == float(BINGX[599].close)  # opens at D - 1h
    assert call["x_first"] == pd.Timestamp(2025, 3, 3, 12) - pd.Timedelta(hours=512)
    assert call["x_last"] == pd.Timestamp(2025, 3, 3, 11)
    assert call["y"] == [pd.Timestamp(2025, 3, 3, 12 + k) for k in range(kronos.horizon_n)]
    assert fan.paths.shape == (kronos.trajectories, kronos.horizon_n, 6)
    assert fan.y_timestamps.equals(window.y_timestamps)


def test_fans_do_not_depend_on_what_ran_before():
    models = fake_models()
    first = forecast_window(models, window_of(), PARAMS)
    forecast_window(models, window_of(BINANCE, source="binance"), PARAMS, stream="other")
    np.random.normal(size=1000)  # disturb the global generator
    again = forecast_window(models, window_of(), PARAMS)
    assert again.seeds == first.seeds
    assert np.array_equal(again.paths, first.paths)


def test_candles_after_d_never_change_the_fan():
    models = fake_models()
    reference = forecast_window(models, window_of(), PARAMS)
    altered = [replace(c, close="1.0", high="2.0") if c.open_ms >= D_MS else c for c in BINGX]
    assert np.array_equal(
        forecast_window(models, window_of(altered), PARAMS).paths, reference.paths
    )
    changed = [replace(c, close="1.0") if c.open_ms == D_MS - HOUR_MS else c for c in BINGX]
    assert not np.array_equal(
        forecast_window(models, window_of(changed), PARAMS).paths, reference.paths
    )


def _refusal_case(case: str):
    """(models, window, params) for one mismatch between the window and the governed params."""
    models = fake_models(UnreachablePredictor())
    window = window_of()
    params = PARAMS
    if case == "model":
        params = replace(PARAMS, kronos=replace(PARAMS.kronos, model="base"))
    elif case == "device":
        models = fake_models(UnreachablePredictor(), device="mps")
    elif case == "max_context":
        window = window_of(max_context=256)
    elif case == "horizon":
        window = window_of(horizon=HORIZON + 1)
    elif case == "amount_policy":
        window = window_of(amount_policy="zeros")
    else:
        raise AssertionError(case)
    return models, window, params


@pytest.mark.parametrize("case", ["model", "device", "max_context", "horizon", "amount_policy"])
def test_windows_that_do_not_match_the_governed_parameters_are_refused(case):
    models, window, params = _refusal_case(case)
    with pytest.raises(DecisionError, match="governed parameters"):
        forecast_window(models, window, params)


def _unlike_loaded(case: str) -> LoadedModels:
    """Models that would not forecast as ``load_models`` leaves them; the model is unreachable."""
    if case == "clip":
        return built_models(UnreachableBuiltPredictor, clip=3)
    if case == "predictor context":
        return built_models(UnreachableBuiltPredictor, max_context=2048)
    if case == "predictor device":
        return built_models(UnreachableBuiltPredictor, device="mps")
    models = built_models(UnreachableBuiltPredictor)
    if case == "model in training mode":
        models.model.train()
    elif case == "one tokenizer submodule in training mode":
        models.tokenizer[1].train()  # the dropout layer only
    elif case == "gradients enabled":
        models.tokenizer[0].bias.requires_grad_(True)
    elif case == "the predictor's own tokenizer":
        models.predictor.tokenizer = frozen_module().train()
    else:
        raise AssertionError(case)
    return models


@pytest.mark.parametrize(
    "case, named",
    [
        ("clip", "the predictor's clip is 3, load_models builds it with 5"),
        ("predictor context", "the predictor's max_context is 2048, load_models builds it with"),
        ("predictor device", "the predictor's device is 'mps', load_models builds it with"),
        ("model in training mode", "the model is in training mode"),
        ("one tokenizer submodule in training mode", "the tokenizer is in training mode"),
        ("gradients enabled", "the tokenizer has parameters with gradients enabled"),
        ("the predictor's own tokenizer", "the predictor's tokenizer is in training mode"),
    ],
)
def test_models_that_would_not_forecast_as_loaded_are_refused_before_the_model_runs(case, named):
    with pytest.raises(DecisionError, match="would not forecast as load_models") as info:
        forecast_window(_unlike_loaded(case), window_of(), PARAMS)
    assert named in str(info.value)


def test_models_as_load_models_leaves_them_forecast():
    models = built_models()
    fan = forecast_window(models, window_of(), PARAMS)
    assert len(models.predictor.calls) == 1
    stand_in = forecast_window(fake_models(), window_of(), PARAMS)
    assert fan.paths.tobytes() == stand_in.paths.tobytes()


def _inconsistent(case: str):
    window = window_of()
    if case == "sha256":
        return replace(window, sha256="0" * 64)
    if case == "inputs_sha256":
        return replace(window, inputs_sha256="0" * 64)
    if case == "context":
        context = window.context.copy()
        context.loc[10, "close"] = context.loc[10, "close"] + 1.0
        return replace(window, context=context)
    if case == "candles":
        shifted = window_of(hour=D + timedelta(hours=1))
        return replace(window, candles=shifted.candles)
    if case == "x_timestamps":
        return replace(window, x_timestamps=window.x_timestamps + pd.Timedelta(hours=1))
    if case == "y_timestamps":
        return replace(window, y_timestamps=window.y_timestamps + pd.Timedelta(hours=1))
    if case == "y_first_is_d_minus_1h":
        later = naive_timestamps([D_MS - HOUR_MS + k * HOUR_MS for k in range(HORIZON)])
        return replace(window, y_timestamps=later)
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "sha256",
        "inputs_sha256",
        "context",
        "candles",
        "x_timestamps",
        "y_timestamps",
        "y_first_is_d_minus_1h",
    ],
)
def test_inconsistent_windows_are_refused_before_the_model_runs(case):
    with pytest.raises(DecisionError):
        forecast_window(fake_models(UnreachablePredictor()), _inconsistent(case), PARAMS)


def test_the_embargo_bounds_the_horizon():
    for horizon, allowed in ((8, True), (9, False)):  # 512 + 8 = 520 h is the embargo
        params = replace(PARAMS, kronos=replace(PARAMS.kronos, horizon_n=horizon))
        window = window_of(horizon=horizon)
        if allowed:
            fan = forecast_window(fake_models(), window, params)
            assert fan.paths.shape[1] == 8
        else:
            with pytest.raises(PartitionError, match="embargo"):
                forecast_window(fake_models(UnreachablePredictor()), window, params)


def test_the_kronos_cutoff_refuses_hours_the_weights_may_have_seen():
    first = to_ms(KRONOS_FIRST_DECISION) - 600 * HOUR_MS
    candles = fakes.synthetic_candles(first, 700, seed=9)
    before = window_of(candles, KRONOS_FIRST_DECISION - timedelta(hours=1))
    with pytest.raises(PartitionError, match="June 2024"):
        forecast_window(fake_models(UnreachablePredictor()), before, PARAMS)
    allowed = forecast_window(fake_models(), before, PARAMS, allow_pretraining_period=True)
    assert allowed.paths.shape == (PARAMS.kronos.trajectories, HORIZON, 6)
    at_cutoff = window_of(candles, KRONOS_FIRST_DECISION)
    assert forecast_window(fake_models(), at_cutoff, PARAMS).seeds.canonical.endswith(
        "2024-07-01T00:00:00Z"
    )


def test_the_runtime_thread_count_must_be_the_governed_one():
    params = replace(PARAMS, kronos=replace(PARAMS.kronos, threads=4))
    with pytest.raises(ValueError, match="threads"):
        forecast_window(fake_models(UnreachablePredictor()), window_of(), params)


def test_argument_types():
    with pytest.raises(TypeError):
        forecast_window(fake_models(), window_of().context, PARAMS)
    with pytest.raises(TypeError):
        forecast_window(fake_models(), window_of(), PARAMS.kronos)
    with pytest.raises(TypeError):
        evaluation_params(PARAMS.kronos, 24)


# -- evaluation length (T23, docs/RETRAINING.md 5.3) -----------------------------------------------

# the last validation decision whose 24 targets fit the horizon purge (docs/DATA.md)
PURGE_LAST = datetime(2026, 7, 9, 16, tzinfo=UTC)
PURGE_CANDLES = fakes.synthetic_candles(to_ms(PURGE_LAST) - 600 * HOUR_MS, 700, seed=11)


def test_evaluation_params_replace_only_the_horizon():
    assert evaluation_params(PARAMS, None) is PARAMS
    long = evaluation_params(PARAMS, 24)
    assert long.kronos.horizon_n == 24 and long.version == PARAMS.version
    assert replace(long.kronos, horizon_n=PARAMS.kronos.horizon_n) == PARAMS.kronos
    assert replace(long, kronos=PARAMS.kronos) == PARAMS
    with pytest.raises(PartitionError, match="EVAL_LENGTH_MAX"):
        evaluation_params(PARAMS, 25)
    for bad in (0, True, 4.0):
        with pytest.raises(ValueError):
            evaluation_params(PARAMS, bad)


def test_a_fan_of_the_evaluation_length():
    predictor = WalkPredictor()
    window = window_of(horizon=24)
    fan = forecast_window(fake_models(predictor), window, PARAMS, evaluation_length=24)
    (call,) = predictor.calls
    assert call["pred_len"] == 24 and call["rows"] == 512
    assert call["y"] == [pd.Timestamp(2025, 3, 3, 12) + pd.Timedelta(hours=k) for k in range(24)]
    assert fan.paths.shape == (PARAMS.kronos.trajectories, 24, 6)
    # the same seeds as the N-step fan of the hour and stream: only the length differs
    assert fan.seeds == forecast_window(fake_models(), window_of(), PARAMS).seeds


def test_the_window_must_have_the_evaluation_length():
    models = fake_models(UnreachablePredictor())
    with pytest.raises(DecisionError, match="window horizon 4 != params horizon_n 24"):
        forecast_window(models, window_of(), PARAMS, evaluation_length=24)
    with pytest.raises(DecisionError, match="window horizon 24 != params horizon_n 4"):
        forecast_window(models, window_of(horizon=24), PARAMS)
    with pytest.raises(PartitionError, match="EVAL_LENGTH_MAX"):
        forecast_window(models, window_of(horizon=25), PARAMS, evaluation_length=25)


def test_the_embargo_still_bounds_the_governed_n():
    """T23: ``check_embargo`` sees N, never the evaluation length."""
    assert 512 + 24 > 520  # a 24-step fan would not fit the embargo as an N
    fan = forecast_window(fake_models(), window_of(horizon=24), PARAMS, evaluation_length=24)
    assert fan.paths.shape[1] == 24
    nine = replace(PARAMS, kronos=replace(PARAMS.kronos, horizon_n=9))  # 512 + 9 > 520
    with pytest.raises(PartitionError, match="embargo"):
        forecast_window(fake_models(UnreachablePredictor()), window_of(), nine, evaluation_length=4)


def test_samples_past_the_horizon_purge_are_refused_never_shortened():
    """T23: with 24 steps the last validation decision is 2026-07-09T16:00Z."""
    window = window_of(PURGE_CANDLES, PURGE_LAST, horizon=24)
    fan = forecast_window(fake_models(), window, PARAMS, evaluation_length=24)
    assert fan.paths.shape[1] == 24
    later = window_of(PURGE_CANDLES, PURGE_LAST + timedelta(hours=1), horizon=24)
    with pytest.raises(PartitionError, match="horizon purge") as info:
        forecast_window(fake_models(UnreachablePredictor()), later, PARAMS, evaluation_length=24)
    assert "2026-07-09T17:00Z with an evaluation length of 24" in str(info.value)
    assert "2026-07-10T16:00Z" in str(info.value)  # the last target it would read
    # the governed N fits there: the purge applies to the length asked for
    governed = window_of(PURGE_CANDLES, PURGE_LAST + timedelta(hours=1))
    assert forecast_window(fake_models(), governed, PARAMS).paths.shape[1] == HORIZON


def test_a_fan_of_the_evaluation_length_has_its_own_configuration():
    """T23: a 24-step fan records ``horizon_n = 24`` and gets its own ``config_id``; the
    governed configuration keeps its keys and values (only the code hash moved with this
    change)."""
    models = fake_models()
    governed = forecast_config(models, PARAMS.kronos)
    long = forecast_config(models, evaluation_params(PARAMS, 24).kronos)
    assert tuple(governed) == tuple(long) == CONFIG_KEYS
    assert governed["horizon_n"] == PARAMS.kronos.horizon_n == 4 and long["horizon_n"] == 24
    assert {key for key in CONFIG_KEYS if governed[key] != long[key]} == {"horizon_n"}
    assert config_id(long) != config_id(governed)
    kronos = PARAMS.kronos
    assert {key: governed[key] for key in CONFIG_KEYS[2:13]} == {
        "device": "cpu",
        "max_context": 512,
        "predictor_clip": PREDICTOR_CLIP,
        "horizon_n": 4,
        "temperature": kronos.temperature,
        "top_p": kronos.top_p,
        "top_k": 0,
        "trajectories": kronos.trajectories,
        "threads": 2,
        "amount_policy": "volume_x_mean_ohlc",
        "seed_salt": "",
    }


# -- model periods (T6, docs/RETRAINING.md 1.4) ----------------------------------------------------

WF_2025_02_END = datetime(2025, 1, 10, 16, tzinfo=UTC)  # the end of a later fold's training data
PERIOD_CANDLES = fakes.synthetic_candles(
    to_ms(KRONOS_FIRST_DECISION) - 2000 * HOUR_MS, 7000, seed=13
)


def checkpoint_models(data_end: datetime, predictor=None) -> LoadedModels:
    return replace(
        fake_models(predictor),
        checkpoint="ks-btc-0123abcd-wf-2025-02",
        data_end_ms=to_ms(data_end),
    )


def test_a_checkpoint_never_forecasts_before_its_data_end():
    before = window_of(PERIOD_CANDLES, WF_2025_02_END - timedelta(hours=1))
    with pytest.raises(PartitionError, match="the first hour this checkpoint may forecast") as info:
        forecast_window(checkpoint_models(WF_2025_02_END, UnreachablePredictor()), before, PARAMS)
    assert f"before {utc_text(to_ms(WF_2025_02_END))}" in str(info.value)
    forecast_window(fake_models(), before, PARAMS)  # the shipped weights may forecast it
    at_end = window_of(PERIOD_CANDLES, WF_2025_02_END)
    assert forecast_window(checkpoint_models(WF_2025_02_END), at_end, PARAMS).paths.shape[1] == 4


def test_a_checkpoint_inherits_the_cutoff_of_the_shipped_weights():
    pilot_end = datetime(2024, 6, 9, 16, tzinfo=UTC)
    before = window_of(PERIOD_CANDLES, KRONOS_FIRST_DECISION - timedelta(hours=1))
    with pytest.raises(PartitionError, match="2024-07-01T00:00Z, the first hour"):
        forecast_window(checkpoint_models(pilot_end, UnreachablePredictor()), before, PARAMS)
    allowed = forecast_window(
        checkpoint_models(pilot_end), before, PARAMS, allow_pretraining_period=True
    )
    assert allowed.paths.shape == (PARAMS.kronos.trajectories, HORIZON, 6)
    at_cutoff = window_of(PERIOD_CANDLES, KRONOS_FIRST_DECISION)
    forecast_window(checkpoint_models(pilot_end), at_cutoff, PARAMS)


def test_the_opt_in_roles_of_the_model_period():
    pilot_end = datetime(2024, 6, 9, 16, tzinfo=UTC)
    inner = window_of(PERIOD_CANDLES, datetime(2024, 5, 1, 8, tzinfo=UTC))  # pilot-check hours
    for models in (fake_models(), checkpoint_models(pilot_end)):
        forecast_window(models, inner, PARAMS, period_role="pilot-check")
        forecast_window(models, inner, PARAMS, period_role="timing")
    forecast_window(fake_models(), inner, PARAMS, period_role="cutoff-check")
    with pytest.raises(PartitionError, match="shipped weights only"):
        forecast_window(
            checkpoint_models(pilot_end, UnreachablePredictor()),
            inner,
            PARAMS,
            period_role="cutoff-check",
        )
    with pytest.raises(PartitionError, match="June 2024"):
        forecast_window(fake_models(UnreachablePredictor()), inner, PARAMS)  # "evaluation"
    with pytest.raises(ValueError, match="role must be one of"):
        forecast_window(fake_models(UnreachablePredictor()), inner, PARAMS, period_role="pilot")


# -- real Kronos-small fans ------------------------------------------------------------------------


@pytest.mark.weights
def test_real_fans_share_seeds_across_sources_and_differ_by_stream(loaded_small):
    bingx = window_of(BINGX, source="bingx")
    binance = window_of(BINANCE, source="binance")
    fan_bingx = forecast_window(loaded_small, bingx, SMALL_PARAMS)
    fan_binance = forecast_window(loaded_small, binance, SMALL_PARAMS)
    floor = forecast_window(loaded_small, bingx, SMALL_PARAMS, stream="mismatch-floor")
    assert fan_bingx.seeds.canonical == fan_binance.seeds.canonical
    assert fan_bingx.seeds == seed_for_decision(
        D, symbol="BTC-USDT", salt=SMALL_PARAMS.kronos.seed_salt
    )
    assert floor.seeds.canonical != fan_bingx.seeds.canonical
    for fan in (fan_bingx, fan_binance, floor):
        assert fan.paths.shape == (4, HORIZON, 6)
        assert np.isfinite(fan.paths).all()
        assert fan.model_name == "small" and fan.threads == 2
    assert not np.array_equal(fan_bingx.paths, fan_binance.paths)  # the data differ
    assert not np.array_equal(fan_bingx.paths, floor.paths)  # the seeds differ
    assert fan_signal(fan_bingx.paths, bingx.last_close) in (-1, 0, 1)
    assert 0.0 <= fan_p_up(fan_bingx.paths, bingx.last_close) <= 1.0
