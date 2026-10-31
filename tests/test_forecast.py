"""Fan of trajectories: window validation, determinism, the amount proxy and the torch facts.

Validation tests run against a stand-in model whose predictor must never be reached; the tests
marked ``weights`` run real fans with Kronos-small.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kronos_agent.kronos.features import (
    FEATURE_COLUMNS,
    add_amount_proxy,
    future_timestamps,
    synthetic_window,
    upstream_fixture_frame,
    upstream_fixture_window,
    validate_window,
)
from kronos_agent.kronos.forecast import (
    ForecastRequest,
    fan_from_params,
    forecast_fan,
    validate_request,
)
from kronos_agent.models import spec_for
from kronos_agent.params import load_params
from kronos_agent.runtime import require_configured
from kronos_agent.seeds import seed_for_decision

DECISION_HOUR = datetime(2026, 10, 29, 16, tzinfo=UTC)
OTHER_HOUR = datetime(2026, 10, 29, 17, tzinfo=UTC)
TRAJECTORIES = 20
HORIZON = 8
MAX_CONTEXT = 512


class _UnreachablePredictor:
    """Stand-in predictor: reaching it means validation did not stop the call."""

    max_context = MAX_CONTEXT

    def predict_batch(self, *args, **kwargs):
        raise AssertionError("predict_batch must not be reached when validation fails")

    def predict(self, *args, **kwargs):
        raise AssertionError("predict must never be used for a fan")


FAKE_MODELS = SimpleNamespace(
    spec=spec_for("small"), predictor=_UnreachablePredictor(), weights_sha256={}, device="cpu"
)


@pytest.fixture(scope="module")
def window():
    return upstream_fixture_window(MAX_CONTEXT)


def make_request(
    df: pd.DataFrame,
    x: pd.Series,
    step: pd.Timedelta,
    *,
    horizon: int = HORIZON,
    trajectories: int = TRAJECTORIES,
    decision_hour: datetime = DECISION_HOUR,
    y: pd.Series | None = None,
    **sampling,
) -> ForecastRequest:
    y = future_timestamps(x, step, horizon) if y is None else y
    return ForecastRequest(
        df, x, y, horizon, trajectories, seed_for_decision(decision_hour), **sampling
    )


# --- features -----------------------------------------------------------------------------------


def test_add_amount_proxy_matches_the_upstream_rule_and_copies():
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [10.0, 20.0],
        }
    )
    out = add_amount_proxy(df)
    assert "amount" not in df.columns
    assert list(out.columns) == list(FEATURE_COLUMNS)
    upstream_rule = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    assert np.array_equal(out["amount"].to_numpy(), upstream_rule.to_numpy())
    assert out["amount"].tolist() == [12.5, 45.0]
    again = add_amount_proxy(out)
    assert again.equals(out)
    assert again is not out
    with pytest.raises(ValueError, match="missing columns"):
        add_amount_proxy(df.drop(columns=["volume"]))


def test_synthetic_window_is_deterministic_valid_and_positive():
    df, x, step = synthetic_window(MAX_CONTEXT)
    df_again, x_again, _ = synthetic_window(MAX_CONTEXT)
    assert df.equals(df_again)
    assert x.equals(x_again)
    assert list(df.columns) == list(FEATURE_COLUMNS)
    assert (df["volume"] > 0).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert step == pd.Timedelta("1h")
    assert x.iloc[-1] == pd.Timestamp("2026-10-29 16:00:00")
    assert x.dt.tz is None
    validate_window(
        df,
        x,
        max_context=MAX_CONTEXT,
        horizon=HORIZON,
        y_timestamps=future_timestamps(x, step, HORIZON),
    )
    other, _, _ = synthetic_window(MAX_CONTEXT, seed=1)
    assert not other.equals(df)
    with pytest.raises(ValueError, match="tz-naive"):
        synthetic_window(8, end=pd.Timestamp("2026-10-29 16:00:00", tz="UTC"))


def test_upstream_fixture_window_regrids_the_session_timestamps():
    raw = upstream_fixture_frame()
    df, x, step = upstream_fixture_window(MAX_CONTEXT)
    assert step == pd.Timedelta(minutes=5)
    assert df.equals(raw.iloc[:MAX_CONTEXT][list(FEATURE_COLUMNS)].reset_index(drop=True))
    assert x.iloc[0] == raw["timestamps"].iloc[0]
    assert (x.diff().iloc[1:] == step).all()
    raw_stamps = raw["timestamps"].iloc[:MAX_CONTEXT].reset_index(drop=True)
    assert (raw_stamps.diff().iloc[1:] != step).any()  # the fixture itself has session gaps
    with pytest.raises(ValueError, match="constant step"):
        validate_window(df, raw_stamps, max_context=MAX_CONTEXT)
    validate_window(df, x, max_context=MAX_CONTEXT)
    with pytest.raises(ValueError):
        upstream_fixture_window(len(raw) + 1)


def test_future_timestamps_continue_the_step(window):
    _df, x, step = window
    y = future_timestamps(x, step, 3)
    assert y.tolist() == [x.iloc[-1] + step * k for k in (1, 2, 3)]
    assert y.dt.tz is None
    with pytest.raises(ValueError):
        future_timestamps(x, step, 0)


def test_validate_window_horizon_and_y_rules(window):
    df, x, step = window
    y = future_timestamps(x, step, HORIZON)
    validate_window(df, x, max_context=MAX_CONTEXT)
    validate_window(df, x, max_context=MAX_CONTEXT, horizon=HORIZON)
    validate_window(df, x, max_context=MAX_CONTEXT, horizon=HORIZON, y_timestamps=y)
    with pytest.raises(ValueError, match="horizon"):
        validate_window(df, x, max_context=MAX_CONTEXT, horizon=0)
    with pytest.raises(ValueError, match="horizon is required"):
        validate_window(df, x, max_context=MAX_CONTEXT, y_timestamps=y)
    with pytest.raises(ValueError, match="exactly 256 rows"):
        validate_window(df, x, max_context=256)


# --- request validation (the stand-in predictor must never be reached) ------------------------

INVALID_WINDOWS = [
    "511_rows",
    "513_rows",
    "nan_in_close",
    "inf_in_volume",
    "unsorted_timestamps",
    "timestamp_gap",
    "tz_aware_timestamps",
    "datetime_index_not_series",
    "y_too_short",
    "y_too_long",
    "y_not_continuing_the_step",
    "y_gap_inside",
    "missing_volume_column",
    "non_numeric_column",
]


def _invalid_window(case: str, df: pd.DataFrame, x: pd.Series, step: pd.Timedelta):
    y = future_timestamps(x, step, HORIZON)
    if case == "511_rows":
        return df.iloc[:511], x.iloc[:511], y
    if case == "513_rows":
        longer_df = pd.concat([df, df.iloc[-1:]], ignore_index=True)
        longer_x = pd.concat([x, pd.Series([x.iloc[-1] + step])], ignore_index=True)
        return longer_df, longer_x, future_timestamps(longer_x, step, HORIZON)
    if case == "nan_in_close":
        bad = df.copy()
        bad.loc[10, "close"] = np.nan
        return bad, x, y
    if case == "inf_in_volume":
        bad = df.copy()
        bad.loc[10, "volume"] = np.inf
        return bad, x, y
    if case == "unsorted_timestamps":
        bad = x.copy()
        bad.iloc[5], bad.iloc[6] = x.iloc[6], x.iloc[5]
        return df, bad, y
    if case == "timestamp_gap":
        bad = x.copy()
        bad.iloc[100:] = bad.iloc[100:] + step
        return df, bad, future_timestamps(bad, step, HORIZON)
    if case == "tz_aware_timestamps":
        return df, x.dt.tz_localize("UTC"), y
    if case == "datetime_index_not_series":
        return df, pd.DatetimeIndex(x), y
    if case == "y_too_short":
        return df, x, y.iloc[: HORIZON - 1]
    if case == "y_too_long":
        return df, x, future_timestamps(x, step, HORIZON + 1)
    if case == "y_not_continuing_the_step":
        return df, x, y + step
    if case == "y_gap_inside":
        bad = y.copy()
        bad.iloc[4:] = bad.iloc[4:] + step
        return df, x, bad
    if case == "missing_volume_column":
        return df.drop(columns=["volume"]), x, y
    if case == "non_numeric_column":
        bad = df.copy()
        bad["close"] = bad["close"].astype(str)
        return bad, x, y
    raise AssertionError(case)


@pytest.mark.parametrize("case", INVALID_WINDOWS)
def test_invalid_windows_raise_value_error_before_the_model_runs(window, case):
    df, x, step = window
    bad_df, bad_x, bad_y = _invalid_window(case, df, x, step)
    request = ForecastRequest(bad_df, bad_x, bad_y, HORIZON, 2, seed_for_decision(DECISION_HOUR))
    with pytest.raises(ValueError):
        forecast_fan(FAKE_MODELS, request)


@pytest.mark.parametrize(
    "sampling",
    [
        {"top_k": 1, "top_p": 0.9},
        {"top_k": 5, "top_p": 0.5},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"temperature": 0.0},
    ],
)
def test_invalid_sampling_parameters_rejected(window, sampling):
    df, x, step = window
    with pytest.raises(ValueError):
        forecast_fan(FAKE_MODELS, make_request(df, x, step, trajectories=2, **sampling))


def test_top_k_zero_with_nucleus_and_greedy_top_k_pass_validation(window):
    df, x, step = window
    validate_request(FAKE_MODELS, make_request(df, x, step, top_k=0, top_p=0.9))
    validate_request(FAKE_MODELS, make_request(df, x, step, top_k=1, top_p=1.0))


def test_trajectories_and_horizon_must_be_positive_integers(window):
    df, x, step = window
    with pytest.raises(ValueError, match="trajectories"):
        forecast_fan(FAKE_MODELS, make_request(df, x, step, trajectories=0))
    with pytest.raises(ValueError, match="horizon"):
        forecast_fan(
            FAKE_MODELS,
            make_request(df, x, step, horizon=0, y=pd.Series([], dtype="datetime64[ns]")),
        )


def test_fan_from_params_checks_model_and_threads_before_the_model_runs(window):
    df, x, step = window
    params = load_params().kronos
    y = future_timestamps(x, step, params.horizon_n)
    with pytest.raises(ValueError, match="params.model"):
        fan_from_params(FAKE_MODELS, df, x, y, replace(params, model="base"), DECISION_HOUR)
    with pytest.raises(ValueError, match="threads"):
        fan_from_params(FAKE_MODELS, df, x, y, replace(params, threads=4), DECISION_HOUR)


class _RecordingPredictor:
    """Fake predict_batch that records the call and returns constant frames."""

    def __init__(self):
        self.observed: dict = {}

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
        import torch

        self.observed.update(
            threads=torch.get_num_threads(),
            initial_seed=torch.initial_seed(),
            grad_enabled=torch.is_grad_enabled(),
            batch=len(df_list),
            pred_len=pred_len,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
            columns=list(df_list[0].columns),
            rows=len(df_list[0]),
            y_len=len(y_timestamp_list[0]),
            x_len=len(x_timestamp_list[0]),
        )
        return [
            pd.DataFrame(
                np.full((pred_len, 6), float(i)),
                columns=list(FEATURE_COLUMNS),
                index=y_timestamp_list[i],
            )
            for i in range(len(df_list))
        ]


def test_fan_seeds_torch_and_calls_predict_batch_with_the_contract_arguments(window):
    import torch

    df, x, step = window
    predictor = _RecordingPredictor()
    models = SimpleNamespace(
        spec=spec_for("small"), predictor=predictor, weights_sha256={"a": "b"}, device="cpu"
    )
    request = make_request(df, x, step, horizon=3, trajectories=4)
    fan = forecast_fan(models, request)
    observed = predictor.observed
    assert observed["threads"] == require_configured().threads == fan.threads == 2
    assert observed["initial_seed"] == request.seeds.torch_seed
    assert observed["grad_enabled"] is False
    assert observed["batch"] == 4
    assert observed["pred_len"] == 3
    assert observed["T"] == 1.0 and observed["top_k"] == 0 and observed["top_p"] == 0.9
    assert observed["sample_count"] == 1
    assert observed["verbose"] is False
    assert observed["columns"] == list(FEATURE_COLUMNS)
    assert observed["rows"] == observed["x_len"] == MAX_CONTEXT
    assert observed["y_len"] == 3
    assert fan.paths.shape == (4, 3, 6)
    assert fan.paths.dtype == np.float64
    assert fan.paths[2].min() == fan.paths[2].max() == 2.0
    assert fan.columns == FEATURE_COLUMNS
    assert fan.model_name == "small"
    assert fan.weights_sha256 == {"a": "b"}
    assert fan.torch_version == torch.__version__
    assert fan.y_timestamps.equals(request.y_timestamps)
    assert fan.elapsed_s >= 0


# --- real fans with Kronos-small ----------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_fan(loaded_small, window):
    df, x, step = window
    started = time.perf_counter()
    fan = forecast_fan(loaded_small, make_request(df, x, step))
    return fan, time.perf_counter() - started


@pytest.mark.weights
def test_fan_shape_dtype_and_finite(reference_fan, window):
    fan, _ = reference_fan
    _df, x, step = window
    assert fan.paths.shape == (TRAJECTORIES, HORIZON, 6)
    assert fan.paths.dtype == np.float64
    assert np.isfinite(fan.paths).all()
    assert fan.columns == FEATURE_COLUMNS
    assert fan.trajectories == TRAJECTORIES and fan.horizon == HORIZON
    assert fan.y_timestamps.equals(future_timestamps(x, step, HORIZON))


@pytest.mark.weights
def test_fan_records_its_provenance(reference_fan, loaded_small):
    import torch

    fan, _ = reference_fan
    assert fan.model_name == "small"
    assert fan.weights_sha256 == loaded_small.weights_sha256
    assert fan.threads == 2
    assert fan.torch_version == torch.__version__
    assert fan.seeds.canonical == "kronos-agent/seed/v1/forecast/BTC-USDT/2026-10-29T16:00:00Z"
    assert fan.seeds.torch_seed == 6649841357966433750
    assert fan.seeds.numpy_seed == 1035604865
    assert fan.elapsed_s > 0


@pytest.mark.weights
def test_fan_wall_time_is_recorded_and_bounds_the_model_time(
    reference_fan, record_testsuite_property
):
    # No wall-clock budget: timing is the job of ``kronos-agent bench`` (SPEC section 3). The
    # 20 x 8 wall time is recorded as a suite property (shown with ``--junitxml``, xunit2-safe).
    fan, wall_seconds = reference_fan
    record_testsuite_property("fan_20x8_wall_seconds", round(wall_seconds, 3))
    assert 0 < fan.elapsed_s <= wall_seconds


@pytest.mark.weights
def test_same_seeds_give_a_bit_identical_fan(reference_fan, loaded_small, window):
    fan, _ = reference_fan
    df, x, step = window
    again = forecast_fan(loaded_small, make_request(df, x, step))
    assert again.seeds == fan.seeds
    assert np.array_equal(fan.paths, again.paths)


@pytest.mark.weights
def test_different_seeds_give_a_different_fan(reference_fan, loaded_small, window):
    fan, _ = reference_fan
    df, x, step = window
    other = forecast_fan(loaded_small, make_request(df, x, step, decision_hour=OTHER_HOUR))
    assert other.seeds != fan.seeds
    assert other.paths.shape == fan.paths.shape
    assert not np.array_equal(fan.paths, other.paths)


@pytest.mark.weights
def test_explicit_amount_proxy_equals_the_implicit_upstream_proxy(loaded_small, window):
    df, x, step = window
    without_amount = df.drop(columns=["amount"])
    implicit = forecast_fan(
        loaded_small, make_request(without_amount, x, step, horizon=4, trajectories=5)
    )
    explicit = forecast_fan(
        loaded_small,
        make_request(add_amount_proxy(without_amount), x, step, horizon=4, trajectories=5),
    )
    assert np.array_equal(implicit.paths, explicit.paths)


@pytest.mark.weights
def test_fan_uses_predict_batch_not_predict(loaded_small, window, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("predict must not be used for a fan")

    monkeypatch.setattr(loaded_small.predictor, "predict", boom)
    df, x, step = window
    fan = forecast_fan(loaded_small, make_request(df, x, step, horizon=2, trajectories=2))
    assert fan.paths.shape == (2, 2, 6)


@pytest.mark.weights
def test_predict_with_sample_count_returns_one_mean_path(loaded_small, window):
    import torch

    df, x, step = window
    y = future_timestamps(x, step, 2)
    with torch.no_grad():
        frame = loaded_small.predictor.predict(
            df, x, y, pred_len=2, T=1.0, top_k=0, top_p=0.9, sample_count=3, verbose=False
        )
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (2, 6)
    assert list(frame.columns) == list(FEATURE_COLUMNS)


@pytest.mark.weights
def test_fan_from_params_uses_the_governed_parameters(loaded_small, window):
    df, x, step = window
    params = load_params().kronos
    y = future_timestamps(x, step, params.horizon_n)
    fan = fan_from_params(loaded_small, df, x, y, params, DECISION_HOUR)
    assert fan.paths.shape == (params.trajectories, params.horizon_n, 6)
    assert np.isfinite(fan.paths).all()
    assert fan.seeds == seed_for_decision(DECISION_HOUR, salt=params.seed_salt)
    assert fan.threads == params.threads
