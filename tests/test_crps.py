"""The scores of ``docs/RETRAINING.md`` 5.1 (``kronos_agent.eval.crps``; T24 in part).

The fair ensemble CRPS against its double sum, in float64 and in exact rational arithmetic on the
same floats; its invariances, ties, degenerate ensembles and refusals; its unbiasedness, which
the plain ensemble CRPS lacks. The closed form of the no-skill reference ``N(0, s^2)`` against
the CRPS integral and against the fair CRPS of a large Gaussian sample; symmetry, scaling and
broadcasting. The lazy reference ``|y|``. Seeded numpy generators only; no data, weights or
reports root is read.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from kronos_agent import provenance
from kronos_agent.eval.crps import MIN_MEMBERS, fair_crps, gaussian_crps, lazy_crps
from kronos_agent.finetune.train import TRAINING_CODE_FILES
from kronos_agent.paths import REPO_ROOT

UNIT_ROUNDOFF = 2.0**-53
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def double_sum(ensemble, observed):
    """The fair CRPS by its definition (design 5.1), in float64: ``(1/K) sum_i |x_i - y| -
    (1 / (2 K (K - 1))) sum_{i != j} |x_i - x_j|``."""
    x = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    k = x.shape[1]
    first = np.abs(x - y[:, None]).sum(axis=1) / k
    pairs = np.abs(x[:, :, None] - x[:, None, :]).sum(axis=(1, 2))
    return first - pairs / (2 * k * (k - 1))


def exact_double_sum(ensemble, observed) -> list[Fraction]:
    """The same double sum in exact rational arithmetic on the float64 inputs."""
    values = []
    rows = np.asarray(ensemble, dtype=np.float64).tolist()
    targets = np.asarray(observed, dtype=np.float64).tolist()
    for row, target in zip(rows, targets, strict=True):
        members = [Fraction(member) for member in row]
        y = Fraction(target)
        k = len(members)
        first = sum(abs(member - y) for member in members) / k
        pairs = sum(abs(a - b) for a in members for b in members)
        values.append(first - pairs / (2 * k * (k - 1)))
    return values


def assert_within_roundoff(result: np.ndarray, exact: list[Fraction], terms: int) -> None:
    """Each value within ``terms`` unit roundoffs of the exact one, relatively: so exactly 0
    where the exact value is 0."""
    for value, truth in zip(result.tolist(), exact, strict=True):
        assert abs(Fraction(value) - truth) <= Fraction(terms, 2**53) * truth, (value, truth)


def random_fans(k: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` fans of ``k`` members in bp (a per-row centre and spread) and realised values."""
    rng = np.random.default_rng(seed)
    centre = rng.normal(0.0, 20.0, size=(n, 1))
    spread = rng.uniform(5.0, 80.0, size=(n, 1))
    ensemble = centre + spread * rng.standard_normal((n, k))
    observed = rng.normal(0.0, 60.0, size=n)
    return ensemble, observed


def crps_integral(y: float, s: float, points: int = 20_001) -> float:
    """``int (F(t) - 1{t >= y})^2 dt`` of ``F = N(0, s^2)``, by Simpson's rule on each side of
    ``y`` (where the integrand jumps), from 40 s below ``min(y, 0)`` to 40 s above ``max(y, 0)``;
    the tails use ``erfc`` so that ``F`` and ``1 - F`` keep their relative precision."""
    erfc = np.vectorize(math.erfc, otypes=[np.float64])
    scale = s * math.sqrt(2.0)

    def simpson(low: float, high: float, integrand) -> float:
        t = np.linspace(low, high, points)
        v = integrand(t)
        step = (high - low) / (points - 1)
        return float(step / 3 * (v[0] + v[-1] + 4 * v[1:-1:2].sum() + 2 * v[2:-1:2].sum()))

    below = simpson(min(y, 0.0) - 40 * s, y, lambda t: (0.5 * erfc(-t / scale)) ** 2)
    above = simpson(y, max(y, 0.0) + 40 * s, lambda t: (0.5 * erfc(t / scale)) ** 2)
    return below + above


# -- fair ensemble CRPS -----------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 20])
def test_the_fair_crps_equals_its_double_sum(k):
    ensemble, observed = random_fans(k, 120, seed=k)
    result = fair_crps(ensemble, observed)
    assert result.dtype == np.float64 and result.shape == (120,)
    np.testing.assert_allclose(result, double_sum(ensemble, observed), rtol=1e-12, atol=1e-10)
    # the sorted form adds terms >= 0 only: within (K + 3) unit roundoffs of the exact value,
    # exactly 0 where the exact value is 0, never negative
    assert_within_roundoff(result, exact_double_sum(ensemble, observed), k + 3)
    assert (result >= 0).all()


def test_the_fair_crps_does_not_depend_on_the_order_of_the_members_or_the_layout():
    ensemble, observed = random_fans(20, 200, seed=7)
    result = fair_crps(ensemble, observed)
    rng = np.random.default_rng(8)
    for same in (
        rng.permuted(ensemble, axis=1),
        ensemble[:, ::-1],
        np.sort(ensemble, axis=1),
        np.asfortranarray(ensemble),
    ):
        assert np.array_equal(fair_crps(same, observed), result)


@pytest.mark.parametrize("k", [2, 3, 20])
def test_a_degenerate_ensemble_scores_its_absolute_error(k):
    member = np.array([-37.25, 0.0, 12.5, 1e-3, 80.0, -4.0])
    observed = np.array([10.0, 0.0, 12.5, -5.0, 79.9, -4.0 - 2.0**-40])
    ensemble = np.repeat(member[:, None], k, axis=1)
    result = fair_crps(ensemble, observed)
    np.testing.assert_allclose(result, np.abs(member - observed), rtol=(k + 3) * UNIT_ROUNDOFF)
    assert result[1] == 0.0 and result[2] == 0.0  # every member at the realised value


def test_ties_between_members_and_with_the_realised_value():
    rng = np.random.default_rng(11)
    ensemble = rng.integers(-3, 4, size=(200, 20)).astype(np.float64)  # 7 values: many ties
    observed = rng.integers(-3, 4, size=200).astype(np.float64)  # most often a member's value
    assert_within_roundoff(fair_crps(ensemble, observed), exact_double_sum(ensemble, observed), 23)
    # by hand: (0, 0, 1, 1) at 0.5 is 1/6; (1, 1, 5, 5) at 1 is 2/3; (1, 1, 1, 5) at 1 is 0
    tied = fair_crps(
        [[0.0, 1.0, 0.0, 1.0], [5.0, 1.0, 5.0, 1.0], [1.0, 5.0, 1.0, 1.0]], [0.5, 1.0, 1.0]
    )
    assert tied.tolist() == pytest.approx([1 / 6, 2 / 3, 0.0], rel=1e-15, abs=0.0)


def test_at_most_one_member_on_each_side_of_the_realised_value_scores_exactly_zero():
    # the fair CRPS is the mean over pairs of members of the distance from y to the pair's span
    pairs = fair_crps([[-1.0, 1.0], [-3.0, 0.5], [2.0, 2.0], [7.0, -7.0]], [0.0, 0.5, 2.0, 0.0])
    assert pairs.tolist() == [0.0, 0.0, 0.0, 0.0]
    triples = fair_crps([[-1.0, 0.0, 1.0], [0.0, 0.0, 5.0]], [0.0, 0.0])
    assert triples.tolist() == [0.0, 0.0]
    quadruples = fair_crps([[1.0, 1.0, 1.0, 5.0], [-2.0, 3.0, 3.0, 9.0]], [1.0, 3.0])
    assert quadruples.tolist() == [0.0, 0.0]


def test_the_fair_crps_takes_integers_float32_and_lists_and_leaves_its_inputs_alone():
    ensemble = np.array([[3, -1, 2], [0, 0, 7]], dtype=np.int64)
    observed = np.array([1, 7], dtype=np.int64)
    # by hand: (3, -1, 2) at 1 is 5/3 - 16/12 = 1/3; (0, 0, 7) at 7 is 14/3 - 28/12 = 7/3
    for x, y in (
        (ensemble, observed),
        (ensemble.astype(np.float32), observed.astype(np.float32)),
        (ensemble.tolist(), observed.tolist()),
    ):
        result = fair_crps(x, y)
        assert result.dtype == np.float64 and result.shape == (2,)
        assert result.tolist() == pytest.approx([1 / 3, 7 / 3], rel=1e-15, abs=0.0)
    unsorted, values = random_fans(5, 4, seed=3)
    before = (unsorted.copy(), values.copy())
    fair_crps(unsorted, values)
    assert np.array_equal(unsorted, before[0]) and np.array_equal(values, before[1])
    empty = fair_crps(np.zeros((0, 20)), np.zeros(0))
    assert empty.shape == (0,) and empty.dtype == np.float64


@pytest.mark.parametrize(
    ("ensemble", "observed"),
    [
        (np.zeros(5), np.zeros(5)),  # one fan as a vector
        (np.zeros((5, 3, 1)), np.zeros(5)),
        (np.zeros((5, 3)), np.zeros((5, 1))),
        (np.zeros((5, 3)), np.zeros(4)),
        (np.zeros((5, 3)), np.float64(0.0)),
        (np.zeros((1, 3)), 0.0),
    ],
)
def test_the_fair_crps_refuses_another_shape(ensemble, observed):
    with pytest.raises(ValueError, match=r"must be \(n, K\) and observed \(n,\)"):
        fair_crps(ensemble, observed)


@pytest.mark.parametrize("k", [0, 1])
def test_the_fair_crps_refuses_fewer_than_two_members(k):
    assert MIN_MEMBERS == 2
    with pytest.raises(ValueError, match=f"K >= 2 members per row, got {k}"):
        fair_crps(np.zeros((4, k)), np.zeros(4))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_the_fair_crps_refuses_a_value_that_is_not_finite(bad):
    ensemble = np.zeros((3, 4))
    ensemble[1, 2] = bad
    with pytest.raises(ValueError, match="the ensemble holds a value that is not finite"):
        fair_crps(ensemble, np.zeros(3))
    observed = np.zeros(3)
    observed[2] = bad
    with pytest.raises(ValueError, match="observed holds a value that is not finite"):
        fair_crps(np.zeros((3, 4)), observed)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 3), dtype=bool),
        np.ones((2, 3), dtype=complex),
        np.full((2, 3), "1.5"),
        np.full((2, 3), None, dtype=object),
    ],
)
def test_the_fair_crps_refuses_values_that_are_not_real_numbers(bad):
    with pytest.raises(ValueError, match="the ensemble must hold real numbers"):
        fair_crps(bad, np.zeros(2))
    with pytest.raises(ValueError, match="observed must hold real numbers"):
        fair_crps(np.zeros((2, 3)), bad[:, 0])


@pytest.mark.parametrize("k", [2, 3, 20])
def test_the_fair_crps_is_unbiased_where_the_plain_ensemble_crps_is_not(k):
    """Members drawn from ``N(0, s^2)``: over 100,000 fans the mean fair score is the Gaussian
    CRPS within 4 standard errors, while the plain ensemble CRPS (``2 K^2`` in place of
    ``2 K (K - 1)``) is higher by ``E|X - X'| / (2 K) = s / (K sqrt(pi))`` on average."""
    s, y, n = 30.0, 10.0, 100_000
    ensemble = np.random.default_rng(100 + k).normal(0.0, s, size=(n, k))
    observed = np.full(n, y)
    fair = fair_crps(ensemble, observed)
    truth = float(gaussian_crps(y, s))
    fair_se = fair.std(ddof=1) / math.sqrt(n)
    assert abs(fair.mean() - truth) <= 4 * fair_se
    members = np.sort(ensemble, axis=1)
    pairs = 2 * (members * (2 * np.arange(k) - (k - 1))).sum(axis=1)  # sum_{i != j} |x_i - x_j|
    plain = np.abs(ensemble - observed[:, None]).mean(axis=1) - pairs / (2 * k * k)
    bias = s / (k * math.sqrt(math.pi))
    assert abs(plain.mean() - truth - bias) <= 4 * plain.std(ddof=1) / math.sqrt(n)
    assert plain.mean() - truth > 10 * fair_se


# -- Gaussian (no-skill) reference ------------------------------------------------------------


@pytest.mark.parametrize(
    ("y", "s"),
    [
        (0.0, 1.0),
        (0.3, 1.0),
        (-2.0, 1.0),
        (5.0, 2.0),
        (-40.0, 30.0),
        (150.0, 25.0),
        (1e-3, 50.0),
        (-500.0, 10.0),
    ],
)
def test_the_gaussian_crps_equals_the_crps_integral(y, s):
    value = gaussian_crps(np.array([y]), np.array([s]))
    assert value.dtype == np.float64 and value.shape == (1,)
    assert float(value[0]) == pytest.approx(crps_integral(y, s), rel=1e-10, abs=0.0)


def test_the_gaussian_crps_equals_the_fair_crps_of_a_large_gaussian_sample():
    """50 fans of 2,000 members drawn from ``N(0, s^2)`` per realised value: their mean fair
    score is the closed form within 4 standard errors and 1 %."""
    s = 30.0
    rng = np.random.default_rng(20261103)
    for y in (-90.0, -15.0, 0.0, 7.5, 60.0):
        sample = fair_crps(rng.normal(0.0, s, size=(50, 2000)), np.full(50, y))
        truth = float(gaussian_crps(y, s))
        error = abs(sample.mean() - truth)
        assert error <= 4 * sample.std(ddof=1) / math.sqrt(sample.size) and error <= 0.01 * truth


def test_the_gaussian_crps_at_zero_and_far_out():
    s = np.array([1.0, 0.25, 40.0])
    np.testing.assert_allclose(
        gaussian_crps(np.zeros(3), s), s * (math.sqrt(2.0) - 1.0) * INV_SQRT_PI, rtol=1e-15
    )
    far = np.array([-400.0, 1e4, 2e3])  # 40 s and more away: erf is 1, the density 0
    assert gaussian_crps(far, np.array([10.0, 10.0, 1.0])).tolist() == pytest.approx(
        (np.abs(far) - np.array([10.0, 10.0, 1.0]) * INV_SQRT_PI).tolist(), rel=1e-15, abs=0.0
    )
    # a vanishing sigma leaves the lazy score
    y = np.array([-3.5, 0.0, 2.25])
    np.testing.assert_allclose(gaussian_crps(y, 1e-12), lazy_crps(y), rtol=1e-12, atol=1e-12)
    # past the float range (a = 1e300, a = inf for a subnormal sigma, exp(-a^2 / 2) = 0) the
    # limits hold whatever error state the caller set
    with np.errstate(all="raise"):
        tail = gaussian_crps(np.array([-1.0, 1.0, 2e3]), np.array([1e-300, 1e-310, 1.0]))
    assert tail.tolist() == pytest.approx([1.0, 1.0, 2e3 - INV_SQRT_PI], rel=1e-15, abs=0.0)


def test_the_gaussian_crps_is_symmetric_in_y_and_scales_with_its_inputs():
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 50.0, size=500)
    s = rng.uniform(1.0, 80.0, size=500)
    value = gaussian_crps(y, s)
    assert np.array_equal(gaussian_crps(-y, s), value)  # computed from |y|: bit for bit
    for factor in (1e-4, 0.5, 3.0, 1e4):
        np.testing.assert_allclose(
            gaussian_crps(factor * y, factor * s), factor * value, rtol=1e-13
        )
    # lower is better: the score grows with |y| at a fixed sigma, from its minimum at y = 0
    grid = np.linspace(0.0, 200.0, 2001)
    assert (np.diff(gaussian_crps(grid, 25.0)) > 0).all()


def test_the_gaussian_crps_broadcasts_sigma_to_the_shape_of_observed():
    rng = np.random.default_rng(9)
    y = rng.normal(0.0, 30.0, size=(6, 4))
    s_1 = rng.uniform(5.0, 50.0, size=6)
    s_h = s_1[:, None] * np.sqrt(np.arange(1, 5))  # s_h = s_1 sqrt(h) (design 5.1)
    value = gaussian_crps(y, s_h)
    assert value.shape == (6, 4) and value.dtype == np.float64
    one_by_one = [[float(gaussian_crps(y[i, j], s_h[i, j])) for j in range(4)] for i in range(6)]
    np.testing.assert_allclose(value, one_by_one, rtol=1e-14)
    per_row = gaussian_crps(y, s_1[:, None])
    assert np.array_equal(per_row, gaussian_crps(y, np.repeat(s_1[:, None], 4, axis=1)))
    assert np.array_equal(gaussian_crps(y[:, 0], 7.0), gaussian_crps(y[:, 0], np.full(6, 7.0)))
    assert gaussian_crps(1.0, 2.0).shape == ()
    with pytest.raises(ValueError, match="does not broadcast"):
        gaussian_crps(y[:, 0], s_1[:, None])  # (6,) against (6, 1) would score a (6, 6) table
    with pytest.raises(ValueError, match="does not broadcast"):
        gaussian_crps(y, np.ones(5))


@pytest.mark.parametrize("sigma", [0.0, -1.0, math.nan, math.inf, [2.0, 0.0], [1.0, -0.5]])
def test_the_gaussian_crps_refuses_a_sigma_that_is_not_finite_and_positive(sigma):
    with pytest.raises(ValueError, match="sigma"):
        gaussian_crps(np.array([1.0, 2.0]), sigma)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_the_gaussian_crps_refuses_an_observed_value_that_is_not_finite(bad):
    with pytest.raises(ValueError, match="observed holds a value that is not finite"):
        gaussian_crps(np.array([1.0, bad]), 3.0)


def test_the_gaussian_crps_refuses_values_that_are_not_real_numbers():
    with pytest.raises(ValueError, match="observed must hold real numbers"):
        gaussian_crps(np.array([True, False]), 3.0)
    with pytest.raises(ValueError, match="sigma must hold real numbers"):
        gaussian_crps(np.array([1.0, 2.0]), np.array([1 + 0j, 2 + 0j]))


# -- lazy reference -------------------------------------------------------------------------


def test_the_lazy_reference_scores_the_absolute_realised_value():
    observed = np.array([-3.5, 0.0, -0.0, 2.25, 1e4])
    value = lazy_crps(observed)
    assert value.dtype == np.float64 and value.tolist() == [3.5, 0.0, 0.0, 2.25, 1e4]
    assert not np.signbit(value).any()
    # the point forecast 0 is the degenerate ensemble at 0
    np.testing.assert_allclose(fair_crps(np.zeros((5, 20)), observed), value, rtol=1e-14)
    assert lazy_crps([[-1, 2], [3, -4]]).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert lazy_crps(np.array([-2], dtype=np.int64)).dtype == np.float64
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="observed holds a value that is not finite"):
            lazy_crps([1.0, bad])


# -- code hashes ----------------------------------------------------------------------------


def test_the_scores_are_in_neither_code_hash():
    """An edit of the scores changes no forecast ``config_id`` (D-047) and leaves the timing
    record fresh (D-055): ``eval/crps.py`` is in neither code hash."""
    relpath = "src/kronos_agent/eval/crps.py"
    assert (REPO_ROOT / relpath).is_file()
    assert relpath not in provenance.FORECASTING_FILES
    assert relpath not in TRAINING_CODE_FILES
