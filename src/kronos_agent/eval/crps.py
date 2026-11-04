"""Scores of the evaluation (``docs/RETRAINING.md`` 5.1): the fair ensemble CRPS of a fan, the
closed-form CRPS of the no-skill reference ``N(0, s^2)`` and the lazy reference's ``|y|``.

Every score is in the unit of its inputs (the evaluation passes basis points: ``x_k(h) = 1e4
log(close_k(h) / c_ref)`` and ``y(h) = 1e4 log(close(D + (h - 1) h) / c_ref)``), comes back as
float64 and is lower for a better forecast. Inputs are checked, never repaired: a wrong shape, a
value that is not a finite real number, or a ``sigma`` that is not > 0, is a ``ValueError``.

The fair ensemble CRPS of ``K >= 2`` members ``x_1 .. x_K`` at the realised ``y`` (Ferro 2014):

    CRPS = (1/K) sum_i |x_i - y| - (1 / (2 K (K - 1))) sum_{i != j} |x_i - x_j|

estimates the CRPS of the distribution the members are drawn from without bias (the plain
ensemble CRPS divides the second sum by ``2 K^2`` and favours a fan that is too narrow). For a
pair of members, ``|x_i - y| + |x_j - y| - |x_i - x_j|`` is twice the distance from ``y`` to the
interval the pair spans, so the score is the mean of that distance over the ``K (K - 1) / 2``
pairs, which is never negative. With the members sorted, ``x_(0) <= .. <= x_(K-1)``, a member
below ``y`` is the end nearer to ``y`` of the ``k`` pairs it forms with the members before it, and
a member above ``y`` of the ``K - 1 - k`` pairs it forms with the members after it:

    CRPS = (2 / (K (K - 1))) sum_k w_k |x_(k) - y|
    w_k = k where x_(k) < y, K - 1 - k where x_(k) > y (a member equal to y adds 0)

:func:`fair_crps` computes that form: O(n K log K) for the sort, and a sum of terms >= 0, so no
two large sums cancel, the score is never negative, a degenerate ensemble gives ``|x - y|`` up to
rounding and an ensemble with at most one member on each side of ``y`` gives exactly 0. The
sorted members are the same for any order of the members, so the score is too, bit for bit.

The spread recalibration and the order-statistic coverage of design (c) come with the
evaluation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MIN_MEMBERS = 2
_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # 2 phi(0)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
_erf = np.vectorize(math.erf, otypes=[np.float64])


def _real_array(name: str, value: Any) -> np.ndarray:
    """``value`` as a float64 array (not a copy when it already is one); refused unless it holds
    finite real numbers (integers and floats; booleans, complex numbers, strings and objects are
    refused)."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must hold real numbers, got dtype {array.dtype}")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} holds a value that is not finite (NaN or inf)")
    return array


def fair_crps(ensemble: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fair ensemble CRPS per row (design 5.1; module docstring), float64 of shape ``(n,)``:
    ``ensemble`` is ``(n, K)`` with ``K >= 2`` members per row and ``observed`` is ``(n,)``, in
    the same unit (bp). ``ValueError`` for another shape, ``K < 2`` or a value that is not a
    finite real number. The inputs are not modified."""
    x = _real_array("the ensemble", ensemble)
    y = _real_array("observed", observed)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(
            f"the ensemble must be (n, K) and observed (n,), got {x.shape} and {y.shape}"
        )
    k = x.shape[1]
    if k < MIN_MEMBERS:
        raise ValueError(f"the fair CRPS needs K >= {MIN_MEMBERS} members per row, got {k}")
    members = np.sort(np.ascontiguousarray(x), axis=1)  # a C-ordered copy: one summation order
    target = y[:, None]
    rank = np.arange(k, dtype=np.float64)
    weights = np.where(members < target, rank, (k - 1) - rank)
    return (weights * np.abs(members - target)).sum(axis=1) / (k * (k - 1) // 2)


def gaussian_crps(observed: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """CRPS of the no-skill reference ``N(0, s^2)`` at ``y`` (design 5.1), float64 in the unit of
    the inputs: ``s [z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)]`` with ``z = y / s``.

    Evaluated as ``|y| erf(a / sqrt(2)) + s (sqrt(2 / pi) exp(-a^2 / 2) - 1 / sqrt(pi))`` with
    ``a = |y| / s``, the same value (``z (2 Phi(z) - 1) = a erf(a / sqrt(2))``) computed from
    ``|y|``, so it is symmetric in ``y`` bit for bit. ``sigma`` (the design's ``s_h``) broadcasts
    to the shape of ``observed`` (one value, one per row as ``(n, 1)``, one per column); the
    result has the shape of ``observed``. ``ValueError`` unless every value is a finite real
    number and every ``sigma`` is > 0, or for a ``sigma`` that does not broadcast to
    ``observed``."""
    y = _real_array("observed", observed)
    s = _real_array("sigma", sigma)
    if not (s > 0).all():
        raise ValueError("sigma must be > 0 everywhere")
    try:
        shape = np.broadcast_shapes(y.shape, s.shape)
    except ValueError:
        shape = None
    if shape != y.shape:
        raise ValueError(f"sigma of shape {s.shape} does not broadcast to observed {y.shape}")
    s = np.broadcast_to(s, y.shape)
    magnitude = np.abs(y)
    # far in the tail a, a^2 or exp(-a^2 / 2) leave the float range: their limits (erf 1, the
    # density 0) are the right values, whatever error state the caller set
    with np.errstate(over="ignore", under="ignore"):
        a = magnitude / s
        two_phi = _SQRT_2_OVER_PI * np.exp(-0.5 * a * a)  # 2 phi(a)
        value = magnitude * _erf(a / _SQRT_2) + s * (two_phi - _INV_SQRT_PI)
    return np.asarray(value, dtype=np.float64)


def lazy_crps(observed: np.ndarray) -> np.ndarray:
    """CRPS of the lazy reference (design 5.1; SPEC 7: next = last), the point forecast 0:
    ``|y|``, float64 of the shape of ``observed``. ``ValueError`` for a value that is not a
    finite real number."""
    return np.asarray(np.abs(_real_array("observed", observed)))


__all__ = ["MIN_MEMBERS", "fair_crps", "gaussian_crps", "lazy_crps"]
