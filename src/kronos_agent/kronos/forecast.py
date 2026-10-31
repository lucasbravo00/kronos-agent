"""Fan of Kronos trajectories built with ``predict_batch`` (SPEC section 3, decision D-006).

At the pinned commit ``KronosPredictor.predict(sample_count=K)`` returns the MEAN of K sampled
paths, i.e. a single path. The fan of K independent trajectories is
``predict_batch([context] * K, [x_ts] * K, [y_ts] * K, pred_len=N, sample_count=1)``: one
batched pass in which every row samples on its own. Upstream has no seed argument, so
``random``, ``numpy`` and ``torch`` are seeded from the decision hour immediately before the
call, and every fan records its seeds, thread count, torch version, model name, weight hashes
and timing. ``torch`` is imported lazily: ``kronos_agent.runtime.configure_runtime()`` must have
run before the first forecast.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..params import KronosParams
from ..runtime import require_configured
from ..seeds import Seeds, seed_everything, seed_for_decision
from .features import FEATURE_COLUMNS, feature_columns_present, validate_window
from .weights import LoadedModels


@dataclass(frozen=True)
class ForecastRequest:
    """Everything one fan needs; validated by :func:`validate_request` before the model runs."""

    context: pd.DataFrame
    x_timestamps: pd.Series
    y_timestamps: pd.Series
    horizon: int
    trajectories: int
    seeds: Seeds
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0


@dataclass
class Fan:
    """``trajectories`` sampled paths of ``horizon`` candles plus the provenance of the run.

    ``paths`` has shape ``(trajectories, horizon, 6)``, dtype float64, channels in
    :data:`FEATURE_COLUMNS` order (``columns``), one row per entry of ``y_timestamps``.
    """

    paths: np.ndarray
    columns: tuple[str, ...]
    y_timestamps: pd.Series
    seeds: Seeds
    model_name: str
    weights_sha256: dict[str, str]
    threads: int
    torch_version: str
    elapsed_s: float

    @property
    def trajectories(self) -> int:
        return int(self.paths.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.paths.shape[1])


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def validate_request(models: LoadedModels, request: ForecastRequest) -> None:
    """Raise ``ValueError`` unless ``request`` can be forecast with ``models`` as configured.

    Sampling parameters: ``temperature > 0``, ``0 < top_p <= 1``, ``top_k >= 0`` and never
    ``top_k > 0`` together with ``top_p < 1`` (the pinned ``top_k_top_p_filtering`` returns
    after the top-k branch, silently ignoring ``top_p``). The window is checked with
    :func:`kronos_agent.kronos.features.validate_window` for ``models.spec.max_context`` rows.
    """
    _positive_int(request.horizon, "horizon")
    _positive_int(request.trajectories, "trajectories")
    if not isinstance(request.seeds, Seeds):
        raise ValueError(  # noqa: TRY004 - ValueError is this module's contract
            "seeds must be a kronos_agent.seeds.Seeds instance"
        )
    if not request.temperature > 0:
        raise ValueError(f"temperature must be > 0, got {request.temperature!r}")
    if not 0 < request.top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {request.top_p!r}")
    if not isinstance(request.top_k, int) or isinstance(request.top_k, bool) or request.top_k < 0:
        raise ValueError(f"top_k must be an integer >= 0, got {request.top_k!r}")
    if request.top_k > 0 and request.top_p < 1.0:
        raise ValueError(
            "top_k > 0 disables top_p in the pinned Kronos code; use top_k=0 with top_p<1 or "
            "top_k>0 with top_p=1.0"
        )
    validate_window(
        request.context,
        request.x_timestamps,
        max_context=models.spec.max_context,
        horizon=request.horizon,
        y_timestamps=request.y_timestamps,
    )


def forecast_fan(models: LoadedModels, request: ForecastRequest) -> Fan:
    """Sample ``request.trajectories`` independent paths with one ``predict_batch`` call.

    The context keeps the channels it has among :data:`FEATURE_COLUMNS`: when ``amount`` is
    absent the predictor builds the proxy itself (identical to
    :func:`kronos_agent.kronos.features.add_amount_proxy`). Seeds are applied with
    :func:`kronos_agent.seeds.seed_everything` immediately before the ``torch.no_grad()`` block.
    """
    validate_request(models, request)
    columns = feature_columns_present(request.context)
    context = request.context[columns].reset_index(drop=True)
    x_timestamps = request.x_timestamps.reset_index(drop=True)
    y_timestamps = request.y_timestamps.reset_index(drop=True)
    count = request.trajectories
    require_configured()
    import torch

    started = time.perf_counter()
    seed_everything(request.seeds)
    with torch.no_grad():
        frames = models.predictor.predict_batch(
            [context] * count,
            [x_timestamps] * count,
            [y_timestamps] * count,
            pred_len=request.horizon,
            T=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            sample_count=1,
            verbose=False,
        )
    elapsed = time.perf_counter() - started
    if len(frames) != count:
        raise RuntimeError(f"predict_batch returned {len(frames)} frames for {count} trajectories")
    paths = np.stack([frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64) for frame in frames])
    expected_shape = (count, request.horizon, len(FEATURE_COLUMNS))
    if paths.shape != expected_shape:
        raise RuntimeError(f"fan has shape {paths.shape}, expected {expected_shape}")
    return Fan(
        paths=paths,
        columns=FEATURE_COLUMNS,
        y_timestamps=y_timestamps,
        seeds=request.seeds,
        model_name=models.spec.name,
        weights_sha256=dict(models.weights_sha256),
        threads=torch.get_num_threads(),
        torch_version=torch.__version__,
        elapsed_s=elapsed,
    )


def fan_from_params(
    models: LoadedModels,
    context: pd.DataFrame,
    x_timestamps: pd.Series,
    y_timestamps: pd.Series,
    params: KronosParams,
    decision_hour: datetime,
    *,
    stream: str = "forecast",
    symbol: str = "BTC-USDT",
) -> Fan:
    """Fan for one decision hour with the governed parameters of ``params.yaml``.

    Seeds come from :func:`kronos_agent.seeds.seed_for_decision` with ``params.seed_salt``;
    horizon, trajectories, temperature, ``top_p`` and ``top_k`` come from ``params``. The loaded
    model must be ``params.model`` and the process must run with ``params.threads`` threads,
    otherwise the forecast would not be comparable and ``ValueError`` is raised.
    """
    if models.spec.name != params.model:
        raise ValueError(
            f"loaded model {models.spec.name!r} differs from params.model {params.model!r}"
        )
    info = require_configured()
    if info.threads != params.threads:
        raise ValueError(
            f"runtime configured with {info.threads} threads but params.yaml fixes {params.threads}"
        )
    seeds = seed_for_decision(decision_hour, stream=stream, symbol=symbol, salt=params.seed_salt)
    request = ForecastRequest(
        context=context,
        x_timestamps=x_timestamps,
        y_timestamps=y_timestamps,
        horizon=params.horizon_n,
        trajectories=params.trajectories,
        seeds=seeds,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
    )
    return forecast_fan(models, request)


__all__ = ["Fan", "ForecastRequest", "fan_from_params", "forecast_fan", "validate_request"]
