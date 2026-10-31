"""Seeds derived from the decision hour: test vectors, independence, ranges and rejections."""

from __future__ import annotations

import dataclasses
import hashlib
import random
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from kronos_agent.seeds import (
    SEED_SCHEME,
    TORCH_SEED_MASK,
    decision_hour_key,
    seed_everything,
    seed_for_decision,
)

H16 = datetime(2026, 10, 29, 16, tzinfo=UTC)
H17 = datetime(2026, 10, 29, 17, tzinfo=UTC)


def test_vector_16h():
    seeds = seed_for_decision(H16)
    assert seeds.canonical == f"{SEED_SCHEME}/forecast/BTC-USDT/2026-10-29T16:00:00Z"
    assert seeds.torch_seed == 6649841357966433750
    assert seeds.numpy_seed == 1035604865
    assert len(seeds.digest) == 64
    assert seeds.digest.startswith("dc48fb7b") and seeds.digest.endswith("fa1c")


def test_vector_17h():
    seeds = seed_for_decision(H17)
    assert seeds.canonical == f"{SEED_SCHEME}/forecast/BTC-USDT/2026-10-29T17:00:00Z"
    assert seeds.torch_seed == 9104675984868344003
    assert seeds.numpy_seed == 3824972400
    assert seeds.digest.startswith("fe5a4e91") and seeds.digest.endswith("3e56")


def test_derivation_is_the_documented_sha256_layout():
    seeds = seed_for_decision(H16, stream="control", symbol="BTC-USDT", salt="v2")
    assert seeds.canonical == f"{SEED_SCHEME}/control/BTC-USDT/2026-10-29T16:00:00Z/v2"
    digest = hashlib.sha256(seeds.canonical.encode("ascii")).digest()
    assert seeds.digest == digest.hex()
    assert seeds.torch_seed == int.from_bytes(digest[:8], "big") & TORCH_SEED_MASK
    assert seeds.numpy_seed == int.from_bytes(digest[8:12], "big")


def test_distinct_inputs_give_distinct_seeds():
    base = seed_for_decision(H16)
    variants = [
        seed_for_decision(H17),
        seed_for_decision(H16, stream="control"),
        seed_for_decision(H16, stream="bootstrap"),
        seed_for_decision(H16, symbol="ETH-USDT"),
        seed_for_decision(H16, salt="x"),
        seed_for_decision(H16, salt="y"),
    ]
    pairs = {(seeds.torch_seed, seeds.numpy_seed) for seeds in [base, *variants]}
    assert len(pairs) == len(variants) + 1
    assert seed_for_decision(H16) == base  # deterministic


def test_ranges_over_two_days_of_hours():
    for hour in range(48):
        seeds = seed_for_decision(H16 + timedelta(hours=hour))
        assert 0 <= seeds.torch_seed < 2**63
        assert 0 <= seeds.numpy_seed < 2**32


@pytest.mark.parametrize(
    "ts",
    [
        datetime(2026, 10, 29, 16),  # naive
        datetime(2026, 10, 29, 16, tzinfo=timezone(timedelta(hours=-3))),  # not UTC
        datetime(2026, 10, 29, 16, 30, tzinfo=UTC),  # minutes
        datetime(2026, 10, 29, 16, 0, 1, tzinfo=UTC),  # seconds
        datetime(2026, 10, 29, 16, 0, 0, 1, tzinfo=UTC),  # microseconds
    ],
    ids=["naive", "non-utc", "minute", "second", "microsecond"],
)
def test_invalid_decision_hours_are_rejected(ts):
    with pytest.raises(ValueError):
        decision_hour_key(ts)
    with pytest.raises(ValueError):
        seed_for_decision(ts)


def test_utc_equivalents_are_accepted():
    assert decision_hour_key(H16) == "2026-10-29T16:00:00Z"
    fixed_zero = datetime(2026, 10, 29, 16, tzinfo=timezone(timedelta(0)))
    assert seed_for_decision(fixed_zero) == seed_for_decision(H16)
    local = datetime(2026, 10, 29, 13, tzinfo=timezone(timedelta(hours=-3)))
    assert seed_for_decision(local.astimezone(UTC)) == seed_for_decision(H16)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stream": "fore/cast"},
        {"symbol": "BTC/USDT"},
        {"salt": "a/b"},
        {"stream": "fore\ncast"},
        {"stream": ""},
        {"symbol": ""},
    ],
    ids=[
        "slash-stream",
        "slash-symbol",
        "slash-salt",
        "newline-stream",
        "empty-stream",
        "empty-symbol",
    ],
)
def test_invalid_components_are_rejected(kwargs):
    with pytest.raises(ValueError):
        seed_for_decision(H16, **kwargs)


def test_seeds_are_frozen():
    seeds = seed_for_decision(H16)
    with pytest.raises(dataclasses.FrozenInstanceError):
        seeds.torch_seed = 1  # type: ignore[misc]


def test_seed_everything_is_reproducible():
    import torch

    def draw() -> tuple[list[float], list[float], float]:
        return torch.rand(4).tolist(), np.random.rand(4).tolist(), random.random()

    seeds = seed_for_decision(H16)
    seed_everything(seeds)
    first = draw()
    seed_everything(seeds)
    assert draw() == first
    seed_everything(seed_for_decision(H17))
    assert draw() != first
