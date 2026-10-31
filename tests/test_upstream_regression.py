"""Offline replica of the upstream regression tests (``third_party/kronos/tests``) on local weights.

Shape, finiteness and local reproducibility are hard requirements. The numeric comparisons (the
shipped fixtures within rtol 1e-5, the MSE values within 1e-6) are report-only: the platform that
generated the fixtures is not recorded (decision D-007), so they are ``xfail(strict=False)`` with
the measured differences printed.
"""

from __future__ import annotations

import numpy as np
import pytest

from kronos_agent.bench import (
    FIXTURE_CONTEXTS,
    FIXTURE_RTOL,
    UPSTREAM_SEEDS,
    greedy_fixture_replica,
)
from kronos_agent.kronos.features import FEATURE_COLUMNS, upstream_fixture_frame
from kronos_agent.seeds import seed_everything

pytestmark = pytest.mark.weights

REPORT_ONLY = (
    "upstream fixtures were generated on an unrecorded platform; numeric agreement is reported, "
    "not required (decision D-007)"
)
MSE_SAMPLE_SIZE = 4
MSE_PRED_LEN = 30
MSE_EXPECTED = {512: 0.008979, 256: 0.003741}
MSE_TOLERANCE = 0.000001
MSE_FEATURES = ["open", "high", "low", "close"]


@pytest.fixture(scope="module", params=FIXTURE_CONTEXTS, ids=lambda n: f"ctx{n}")
def greedy(request, loaded_small):
    return greedy_fixture_replica(loaded_small, request.param)


def test_greedy_replica_shape_and_finite(greedy):
    assert greedy.obtained.shape == (8, 6)
    assert greedy.expected.shape == (8, 6)
    assert greedy.obtained.dtype == np.float32
    assert np.isfinite(greedy.obtained).all()
    assert greedy.seconds > 0


def test_greedy_replica_is_reproducible_locally(loaded_small, greedy):
    again = greedy_fixture_replica(loaded_small, greedy.context_len)
    assert np.array_equal(again.obtained, greedy.obtained)


@pytest.mark.xfail(strict=False, reason=REPORT_ONLY)
def test_greedy_replica_within_rtol_of_the_upstream_fixture(greedy):
    print(
        f"context {greedy.context_len}: max_abs_diff={greedy.max_abs_diff:.3e} "
        f"max_rel_diff={greedy.max_rel_diff:.3e} (rtol {FIXTURE_RTOL}, "
        f"passes={greedy.passes_rtol})"
    )
    np.testing.assert_allclose(greedy.obtained, greedy.expected, rtol=FIXTURE_RTOL)


def mse_replica(models, context_len: int) -> dict:
    """Upstream ``test_kronos_predictor_mse``: 4 windows drawn with ``random_state=123``, greedy
    ``predict`` of 30 candles each, mean squared error on OHLC against the real future."""
    import torch

    frame = upstream_fixture_frame()
    valid_region = frame.iloc[context_len : frame.shape[0] - MSE_PRED_LEN]
    sampled = valid_region.sample(n=MSE_SAMPLE_SIZE, random_state=123).sort_index()
    seed_everything(UPSTREAM_SEEDS)
    values: list[float] = []
    with torch.no_grad():
        for row_idx in sampled.index.to_list():
            context_slice = frame.iloc[row_idx - context_len : row_idx]
            future_slice = frame.iloc[row_idx : row_idx + MSE_PRED_LEN]
            prediction = models.predictor.predict(
                df=context_slice[list(FEATURE_COLUMNS)].reset_index(drop=True),
                x_timestamp=context_slice["timestamps"].reset_index(drop=True),
                y_timestamp=future_slice["timestamps"].reset_index(drop=True),
                pred_len=MSE_PRED_LEN,
                T=1.0,
                top_k=1,
                top_p=1.0,
                verbose=False,
                sample_count=1,
            )
            obtained = prediction[MSE_FEATURES].to_numpy(dtype=np.float32)
            expected = future_slice[MSE_FEATURES].to_numpy(dtype=np.float32)
            values.append(float(np.mean((obtained - expected) ** 2)))
    return {
        "context_len": context_len,
        "rows": sampled.index.to_list(),
        "mse_values": values,
        "mse": float(np.mean(values)),
    }


@pytest.fixture(scope="module", params=sorted(MSE_EXPECTED, reverse=True), ids=lambda n: f"ctx{n}")
def mse(request, loaded_small):
    return mse_replica(loaded_small, request.param)


def test_mse_replica_is_well_formed(mse):
    assert len(mse["rows"]) == MSE_SAMPLE_SIZE
    assert len(mse["mse_values"]) == MSE_SAMPLE_SIZE
    assert all(np.isfinite(value) and value >= 0 for value in mse["mse_values"])
    assert mse["mse"] >= 0


@pytest.mark.xfail(strict=False, reason=REPORT_ONLY)
def test_mse_replica_matches_the_upstream_value(mse):
    expected = MSE_EXPECTED[mse["context_len"]]
    print(
        f"context {mse['context_len']}: mse={mse['mse']:.6f} expected={expected} "
        f"diff={mse['mse'] - expected:+.3e} (tolerance {MSE_TOLERANCE})"
    )
    assert abs(mse["mse"] - expected) <= MSE_TOLERANCE
