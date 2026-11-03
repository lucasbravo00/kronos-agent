"""Kronos fan of one decision window, and the fan's directional signal (``docs/DATA.md``).

:func:`forecast_window` is the one place where a :class:`~kronos_agent.data.windows.DecisionWindow`
becomes a fan. Before the model runs it checks the window against the governed parameters of
``params.yaml`` (model, device, ``max_context``, ``horizon_n``, ``amount_policy``), refuses models
that would not forecast as :func:`~kronos_agent.kronos.weights.load_models` leaves them (a
predictor with another clip, context length or device, a tokenizer or model in training mode or
with gradients enabled: :func:`~kronos_agent.kronos.weights.inference_problems`), checks that
the window is internally consistent (its candles, hashes, float context and timestamps all
describe the same ``[D - max_context h, D)`` candles, with targets from ``D``), refuses a
configuration whose context plus trading horizon the partition's embargo does not cover
(:func:`kronos_agent.data.partition.check_embargo`) and a decision hour the model may have seen
(:func:`kronos_agent.data.partition.require_model_period`: the shipped weights before July 2024,
a fine-tuned checkpoint also before the end of its training data). Then it calls
:func:`kronos_agent.kronos.forecast.fan_from_params` with ``symbol=params.symbol``, the project
symbol, never the window's own: a Binance window (``BTCUSDT``) and a BingX window (``BTC-USDT``)
of the same hour and stream get the same seeds, so two fans differ only by their data.

An evaluation length ``L`` (``docs/RETRAINING.md``, 5.3; at most 24 candles) makes a fan of ``L``
steps instead of the governed ``N``: the governed parameters with ``horizon_n = L``
(:func:`evaluation_params`) are what the window, the fan and its forecast configuration are
checked against, so an ``L``-step fan records ``horizon_n = L`` and gets its own ``config_id``;
the embargo still bounds the governed ``N``, and a sample whose ``L`` targets reach past the
horizon purge is refused, never shortened.

The sealed partition is not checked here: it is guarded where candles are read (the store path
needs a ``SealedAccess``), and every live decision is, by date, in the sealed segment.

The signal of a fan is the directional control of SPEC section 8 at the live horizon: the sign
of ``mean_k(close[k, N]) / close[D - 1h] - 1``, computed literally in float64; ``p_up`` is the
share of trajectories whose close at ``N`` is strictly above ``close[D - 1h]``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..data.partition import (
    MODEL_PERIOD_EVALUATION,
    PartitionError,
    check_embargo,
    check_evaluation_length,
    require_model_period,
    sample_fits,
)
from ..data.timeutil import HOUR_MS, to_ms, utc_text
from ..data.windows import DecisionWindow, context_frame, inputs_sha256, naive_timestamps
from ..exchange.public_market import window_sha256
from ..params import Params
from .features import FEATURE_COLUMNS
from .forecast import Fan, fan_from_params
from .weights import LoadedModels, inference_problems

CLOSE_INDEX = FEATURE_COLUMNS.index("close")
DEFAULT_STREAM = "forecast"


class DecisionError(ValueError):
    """A window or models that do not match the governed parameters or would not forecast as
    loaded, or an invalid fan for a signal."""


def _check_against_params(models: LoadedModels, window: DecisionWindow, params: Params) -> None:
    kronos = params.kronos
    problems: list[str] = []
    if models.spec.name != kronos.model:
        problems.append(f"loaded model {models.spec.name!r} != params model {kronos.model!r}")
    if models.device != kronos.device:
        problems.append(f"models on device {models.device!r} != params device {kronos.device!r}")
    if window.max_context != kronos.max_context:
        problems.append(f"window max_context {window.max_context} != params {kronos.max_context}")
    if models.spec.max_context != kronos.max_context:
        problems.append(
            f"model {models.spec.name!r} reads {models.spec.max_context} candles, "
            f"params max_context is {kronos.max_context}"
        )
    if window.horizon != kronos.horizon_n:
        problems.append(f"window horizon {window.horizon} != params horizon_n {kronos.horizon_n}")
    if window.amount_policy != kronos.amount_policy:
        problems.append(
            f"window amount_policy {window.amount_policy!r} != params {kronos.amount_policy!r}"
        )
    if problems:
        raise DecisionError(
            f"{window.source} window of {utc_text(to_ms(window.decision_hour))} does not match "
            "the governed parameters: " + "; ".join(problems)
        )


def _check_inference_state(models: LoadedModels) -> None:
    """The models must forecast as ``load_models`` leaves them: the predictor's clip, context
    length and device, every module in eval mode without gradients."""
    problems = inference_problems(models)
    if problems:
        raise DecisionError(
            "the models would not forecast as load_models leaves them: " + "; ".join(problems)
        )


def _check_window_consistency(window: DecisionWindow) -> None:
    """The window's candles, hashes, context and timestamps must describe the same inputs."""
    label = f"{window.source} window of {utc_text(to_ms(window.decision_hour))}"
    decision_ms = to_ms(window.decision_hour)
    expected_opens = list(range(decision_ms - window.max_context * HOUR_MS, decision_ms, HOUR_MS))
    if [candle.open_ms for candle in window.candles] != expected_opens:
        raise DecisionError(f"{label}: its candles are not the hours [D - max_context h, D)")
    if window_sha256(window.candles) != window.sha256:
        raise DecisionError(f"{label}: sha256 does not match its candles")
    if (
        inputs_sha256(window.candles, window.amount_policy, window.quote_volumes)
        != window.inputs_sha256
    ):
        raise DecisionError(f"{label}: inputs_sha256 does not match its candles and policy")
    expected = context_frame(window.candles, window.amount_policy, window.quote_volumes)
    context = window.context
    if (
        list(context.columns) != list(expected.columns)
        or context.shape != expected.shape
        or context.to_numpy(dtype=np.float64).tobytes()
        != expected.to_numpy(dtype=np.float64).tobytes()
    ):
        raise DecisionError(f"{label}: the float context is not the one of its candles")
    if not window.x_timestamps.equals(naive_timestamps(expected_opens)):
        raise DecisionError(f"{label}: x_timestamps are not the open times of its candles")
    targets = [decision_ms + step * HOUR_MS for step in range(window.horizon)]
    if not window.y_timestamps.equals(naive_timestamps(targets)):
        raise DecisionError(
            f"{label}: y_timestamps must be the {window.horizon} hourly opens starting at D"
        )


def evaluation_params(params: Params, evaluation_length: int | None) -> Params:
    """``params`` for fans of ``evaluation_length`` steps: ``horizon_n`` replaced by the length.

    ``None`` returns ``params`` itself (the governed ``N``). A length is checked by
    :func:`~kronos_agent.data.partition.check_evaluation_length` (1 to 24 candles);
    ``params.version`` is kept, so the fan records the governed parameters it was made under.
    """
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    if evaluation_length is None:
        return params
    length = check_evaluation_length(evaluation_length)
    return replace(params, kronos=replace(params.kronos, horizon_n=length))


def forecast_window(
    models: LoadedModels,
    window: DecisionWindow,
    params: Params,
    *,
    stream: str = DEFAULT_STREAM,
    evaluation_length: int | None = None,
    period_role: str = MODEL_PERIOD_EVALUATION,
    allow_pretraining_period: bool = False,
) -> Fan:
    """The fan of ``window`` with the governed parameters, seeded from ``D`` and ``stream``.

    With ``evaluation_length`` ``L``, the fan has ``L`` steps and everything is checked against
    :func:`evaluation_params` (the window must have been built with horizon ``L``); without it,
    against the governed ``N``. Refused before the model runs: a window or model that does not
    match those parameters (model, device, ``max_context``, ``horizon_n``, ``amount_policy``),
    models that would not forecast as ``load_models`` leaves them (another predictor clip,
    context length or device, a module in training mode or with gradients enabled) or a window
    that is internally inconsistent (:class:`DecisionError`); a context plus the governed ``N``
    beyond the embargo, a length above 24 or a sample of ``L`` targets past the horizon purge,
    and a decision hour outside the model period of ``period_role`` (the shipped weights from
    July 2024, a checkpoint from the later of that and its ``data_end_ms``) unless
    ``allow_pretraining_period`` (tests only)
    (:class:`~kronos_agent.data.partition.PartitionError`). The runtime thread count is checked
    by :func:`~kronos_agent.kronos.forecast.fan_from_params` (``ValueError``). Seeds use
    ``params.symbol`` for every source.
    """
    if not isinstance(window, DecisionWindow):
        raise TypeError(f"expected a DecisionWindow, got {type(window).__name__}")
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    effective = evaluation_params(params, evaluation_length)
    _check_against_params(models, window, effective)
    _check_inference_state(models)
    check_embargo(window.max_context, params.kronos.horizon_n)
    if evaluation_length is not None and not sample_fits(
        window.decision_hour, effective.kronos.horizon_n, max_context=window.max_context
    ):
        decision_ms = to_ms(window.decision_hour)
        last_ms = decision_ms + (effective.kronos.horizon_n - 1) * HOUR_MS
        raise PartitionError(
            f"the sample of {utc_text(decision_ms)} with an evaluation length of "
            f"{effective.kronos.horizon_n} reads targets up to {utc_text(last_ms)}, past the "
            "horizon purge of the partition (sample_fits): it is refused, never shortened"
        )
    require_model_period(
        window.decision_hour,
        role=period_role,
        data_end_ms=getattr(models, "data_end_ms", None),
        allow_pretraining_period=allow_pretraining_period,
    )
    _check_window_consistency(window)
    return fan_from_params(
        models,
        window.context,
        window.x_timestamps,
        window.y_timestamps,
        effective.kronos,
        window.decision_hour,
        stream=stream,
        symbol=params.symbol,
    )


def _horizon_closes(paths: np.ndarray) -> np.ndarray:
    """``close[k, N]`` of every trajectory: the close channel at the fan's last step."""
    if not isinstance(paths, np.ndarray):
        raise DecisionError(f"paths must be a numpy array, got {type(paths).__name__}")
    if paths.ndim != 3 or paths.shape[2] != len(FEATURE_COLUMNS):
        raise DecisionError(
            f"paths must have shape (trajectories, horizon, {len(FEATURE_COLUMNS)}), "
            f"got {paths.shape}"
        )
    if paths.shape[0] < 1 or paths.shape[1] < 1:
        raise DecisionError(f"paths must hold at least one trajectory and one step: {paths.shape}")
    if paths.dtype.kind != "f":
        raise DecisionError(f"paths must hold floats, got {paths.dtype}")
    closes = np.asarray(paths[:, -1, CLOSE_INDEX], dtype=np.float64)
    if not np.isfinite(closes).all():
        raise DecisionError("the fan's closes at the horizon contain NaN or infinite values")
    return closes


def _reference_close(last_close: float) -> float:
    if isinstance(last_close, bool) or not isinstance(last_close, (int, float, np.floating)):
        raise DecisionError(f"last_close must be a number, got {last_close!r}")
    reference = float(last_close)
    if not math.isfinite(reference) or reference <= 0:
        raise DecisionError(f"last_close must be a finite positive price, got {last_close!r}")
    return reference


def fan_mean_close(paths: np.ndarray) -> float:
    """``mean_k(close[k, N])``: the mean close of the trajectories at the fan's last step."""
    return float(np.mean(_horizon_closes(paths)))


def fan_signal(paths: np.ndarray, last_close: float) -> int:
    """``-1``, ``0`` or ``1``: the sign of ``mean_k(close[k, N]) / close[D - 1h] - 1``.

    The MEAN over trajectories (not the median, and not a majority vote: that is
    :func:`fan_p_up`) at the LAST step of the fan; ``last_close`` is ``float()`` of the close of
    the candle opening at ``D - 1h`` (``DecisionWindow.last_close``).
    """
    reference = _reference_close(last_close)
    ratio = fan_mean_close(paths) / reference - 1.0
    return 1 if ratio > 0 else -1 if ratio < 0 else 0


def fan_p_up(paths: np.ndarray, last_close: float) -> float:
    """Share of trajectories whose close at the last step is strictly above ``last_close``."""
    reference = _reference_close(last_close)
    closes = _horizon_closes(paths)
    return float(np.count_nonzero(closes > reference)) / float(closes.size)


__all__ = [
    "CLOSE_INDEX",
    "DEFAULT_STREAM",
    "DecisionError",
    "evaluation_params",
    "fan_mean_close",
    "fan_p_up",
    "fan_signal",
    "forecast_window",
]
