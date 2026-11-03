"""Block bootstrap: seed vectors, Monday weeks, the documented resampling, point-only rules,
Wilson intervals for shares of 0 or 1, deff/n_eff, paired columns and seam safety."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from kronos_agent.eval.bootstrap import (
    DEFAULT_RESAMPLES,
    MAX_FAILED_SHARE,
    MIN_BLOCKS,
    MONDAY_OFFSET_MS,
    WEEK_MS,
    BootstrapResult,
    autocorrelation,
    block_bootstrap,
    bootstrap_seed,
    week_blocks,
    wilson_interval,
)

HOUR_MS = 3_600_000
MONDAY_MS = 1_719_792_000_000  # 2024-07-01T00:00Z, a Monday


def hourly_open_ms(weeks: int) -> np.ndarray:
    return MONDAY_MS + HOUR_MS * np.arange(weeks * 168, dtype=np.int64)


def documented_resamples(columns, blocks, statistic, key, n_resamples=DEFAULT_RESAMPLES):
    """The resampling written out from the docstring, independently of the implementation."""
    ids = np.unique(blocks)
    rows = [np.flatnonzero(blocks == block) for block in ids]
    rng = np.random.default_rng(bootstrap_seed(key))
    draws = rng.integers(0, len(ids), size=(n_resamples, len(ids)))
    values = []
    for drawn in draws:
        index = np.concatenate([rows[block] for block in drawn])
        values.append(statistic(*(np.asarray(column)[index] for column in columns)))
    return np.array(values), draws, rows


def test_seed_vectors():
    assert bootstrap_seed("mismatch/close_abs_median") == 1500629563
    assert bootstrap_seed("test/statistic") == 2274430041
    assert bootstrap_seed("poc_bingx/dpoc_mean_bp") == 1181880839
    digest = hashlib.sha256(b"kronos-agent/bootstrap/v1/a/b").digest()
    assert bootstrap_seed("a/b") == int.from_bytes(digest[:4], "big")
    with pytest.raises(ValueError):
        bootstrap_seed("")


def test_week_blocks_start_on_monday_utc():
    assert MONDAY_OFFSET_MS == 345_600_000 and WEEK_MS == 7 * 24 * HOUR_MS
    assert week_blocks(np.array([MONDAY_OFFSET_MS, MONDAY_OFFSET_MS - 1, 0])).tolist() == [
        0,
        -1,
        -1,
    ]
    times = np.array([MONDAY_MS - 1, MONDAY_MS, MONDAY_MS + WEEK_MS - 1, MONDAY_MS + WEEK_MS])
    assert week_blocks(times).tolist() == [2842, 2843, 2843, 2844]
    weeks = MONDAY_MS + WEEK_MS * np.arange(8, dtype=np.int64)
    assert week_blocks(weeks, weeks=4).tolist() == [710, 711, 711, 711, 711, 712, 712, 712]
    with pytest.raises(ValueError):
        week_blocks(times, weeks=0)


def test_interval_follows_the_documented_resampling():
    rng = np.random.default_rng(11)
    open_ms = hourly_open_ms(30)
    values = rng.normal(size=open_ms.size)
    blocks = week_blocks(open_ms)
    result = block_bootstrap([values], blocks, np.mean, key="unit/mean")
    resampled, _, _ = documented_resamples([values], blocks, np.mean, "unit/mean")
    assert result.low == float(np.quantile(resampled, 0.025, method="linear"))
    assert result.high == float(np.quantile(resampled, 0.975, method="linear"))
    assert result.point == float(np.mean(values))
    assert result.method == "percentile_block_bootstrap" and result.reason is None
    assert (result.n, result.n_blocks, result.n_resamples) == (values.size, 30, 1000)
    assert result.seed == bootstrap_seed("unit/mean") and result.failed_resamples == 0
    assert result.low < result.point < result.high and result.has_interval
    assert result.deff is None and result.n_eff is None  # only shares get them


def test_bootstrap_is_deterministic_per_key():
    rng = np.random.default_rng(1)
    open_ms = hourly_open_ms(25)
    values = rng.normal(size=open_ms.size)
    blocks = week_blocks(open_ms)
    first = block_bootstrap([values], blocks, np.mean, key="unit/mean")
    assert block_bootstrap([values], blocks, np.mean, key="unit/mean") == first
    other = block_bootstrap([values], blocks, np.mean, key="unit/other")
    assert other.point == first.point and (other.low, other.high) != (first.low, first.high)
    assert first.as_dict()["key"] == "unit/mean" and set(first.as_dict()) == {
        "key",
        "point",
        "low",
        "high",
        "n",
        "n_blocks",
        "n_resamples",
        "confidence",
        "seed",
        "failed_resamples",
        "method",
        "n_eff",
        "deff",
        "reason",
    }


def test_fewer_than_min_blocks_gives_the_point_only():
    assert MIN_BLOCKS == 20
    values = np.random.default_rng(2).normal(size=19 * 168)
    blocks = week_blocks(hourly_open_ms(19))
    result = block_bootstrap([values], blocks, np.mean, key="unit/mean")
    assert result.method == "point_only" and result.low is None and result.high is None
    assert "fewer than 20 blocks (19)" in result.reason
    assert result.point == float(np.mean(values))
    values = np.random.default_rng(2).normal(size=20 * 168)
    assert block_bootstrap(
        [values], week_blocks(hourly_open_ms(20)), np.mean, key="unit/mean"
    ).has_interval


def test_failed_resamples_are_counted_and_more_than_one_percent_drop_the_interval():
    assert MAX_FAILED_SHARE == 0.01
    weeks = 30
    blocks = week_blocks(hourly_open_ms(weeks))
    marker = np.zeros(blocks.size)
    marker[0] = 1.0  # one marked row in the first block
    noise = np.random.default_rng(3).normal(size=blocks.size)

    def statistic(threshold):
        def mean_unless_marked(mark, value):
            return math.nan if mark.sum() >= threshold else float(np.mean(value))

        return mean_unless_marked

    for threshold in (3, 6):
        stat = statistic(threshold)
        resampled, draws, _ = documented_resamples([marker, noise], blocks, stat, "unit/fail")
        expected_failures = int(np.sum((draws == 0).sum(axis=1) >= threshold))
        result = block_bootstrap([marker, noise], blocks, stat, key="unit/fail")
        assert result.failed_resamples == expected_failures
        if expected_failures > 10:
            assert result.method == "point_only" and "non-finite" in result.reason
        else:
            assert result.has_interval
            finite = resampled[np.isfinite(resampled)]
            assert result.low == float(np.quantile(finite, 0.025, method="linear"))
    assert int(np.sum((draws == 0).sum(axis=1) >= 3)) > 10  # threshold 3 exercised the refusal


def test_a_degenerate_distribution_has_no_interval():
    blocks = week_blocks(hourly_open_ms(24))
    constant = np.full(blocks.size, 7.0)
    result = block_bootstrap([constant], blocks, np.mean, key="unit/constant")
    assert result.point == 7.0 and not result.has_interval and "degenerate" in result.reason
    # unequal block sizes: means of 0.1 differ by summation rounding only, still one value
    uneven = np.concatenate([blocks[:5], blocks[200:]])
    tenths = np.full(uneven.size, 0.1)
    result = block_bootstrap([tenths], uneven, np.mean, key="unit/tenths")
    assert not result.has_interval and "degenerate" in result.reason
    median = block_bootstrap(
        [np.repeat([1.0, 2.0, 1.0], [2000, 16, 2016])], blocks, np.median, key="unit/median"
    )
    assert median.point == 1.0 and not median.has_interval


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_shares_of_exactly_zero_or_one_get_a_wilson_interval_at_the_block_count(value):
    blocks = week_blocks(hourly_open_ms(30))
    flags = np.full(blocks.size, value)
    result = block_bootstrap([flags], blocks, np.mean, key="unit/share", share=True)
    assert result.method == "wilson_blocks" and result.point == value
    assert (result.low, result.high) == wilson_interval(value * 30, 30)
    assert (result.low == 0.0) if value == 0.0 else (result.high == 1.0)
    assert "n = 30 blocks" in result.reason
    not_share = block_bootstrap([flags], blocks, np.mean, key="unit/share")
    assert not not_share.has_interval and "degenerate" in not_share.reason
    few = block_bootstrap([flags[: 10 * 168]], blocks[: 10 * 168], np.mean, key="u/s", share=True)
    assert few.method == "point_only"


def test_share_design_effect_and_effective_size():
    weeks = 40
    blocks = week_blocks(hourly_open_ms(weeks))
    rng = np.random.default_rng(4)
    clustered = np.repeat(rng.integers(0, 2, size=weeks), 168).astype(float)  # whole weeks agree
    result = block_bootstrap([clustered], blocks, np.mean, key="unit/clustered", share=True)
    p, n = result.point, clustered.size
    resampled, _, _ = documented_resamples([clustered], blocks, np.mean, "unit/clustered")
    assert result.deff == pytest.approx(np.var(resampled, ddof=1) / (p * (1 - p) / n), rel=1e-12)
    assert result.deff > 50 and result.n_eff == pytest.approx(n / result.deff)
    independent = rng.integers(0, 2, size=blocks.size).astype(float)
    result = block_bootstrap([independent], blocks, np.mean, key="unit/independent", share=True)
    assert 0.5 < result.deff < 2.0


def test_paired_columns_are_resampled_with_the_same_blocks():
    blocks = week_blocks(hourly_open_ms(26))
    first = np.random.default_rng(5).integers(0, 100, size=blocks.size).astype(float)
    second = first + 0.5

    def paired_difference(a, b):
        return float(np.mean(b - a))

    result = block_bootstrap([first, second], blocks, paired_difference, key="unit/paired")
    assert result.point == 0.5 and "degenerate" in result.reason  # every resample paired rows
    unpaired = block_bootstrap(
        [first, np.random.default_rng(6).permutation(second)], blocks, paired_difference, key="u/p"
    )
    assert unpaired.has_interval


def test_resamples_are_whole_blocks_in_draw_order():
    open_ms = hourly_open_ms(22)[::5]  # irregular block sizes (33 or 34 rows)
    blocks = week_blocks(open_ms)
    index = np.arange(open_ms.size, dtype=float)
    seen: list[np.ndarray] = []

    def capture(rows):
        seen.append(rows.copy())
        return float(rows.mean())

    block_bootstrap([index], blocks, capture, key="unit/seams", n_resamples=50)
    assert np.array_equal(seen[0], index)  # the point estimate sees the data as given
    _, draws, rows_by_block = documented_resamples([index], blocks, np.mean, "unit/seams", 50)
    for resample, drawn in zip(seen[1:], draws, strict=True):
        expected = np.concatenate([rows_by_block[block] for block in drawn]).astype(float)
        assert np.array_equal(resample, expected)
        assert set(resample.tolist()) <= set(index.tolist())  # no value created at a seam


def test_wilson_interval_reference_values():
    # published 95 % Wilson score intervals: 0/10 [0, 0.2775], 5/10 [0.2366, 0.7634]
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=5e-5)
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=5e-5) and high == pytest.approx(0.7634, abs=5e-5)
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and low == pytest.approx(0.7225, abs=5e-5)
    assert wilson_interval(0, 30)[0] == 0.0  # exact, not a rounding residue
    for args in ((1, 0), (11, 10), (-1, 10)):
        with pytest.raises(ValueError):
            wilson_interval(*args)
    with pytest.raises(ValueError):
        wilson_interval(1, 10, confidence=1.0)


def test_undefined_point_and_argument_checks():
    blocks = week_blocks(hourly_open_ms(21))
    result = block_bootstrap([np.zeros(blocks.size)], blocks, lambda x: math.nan, key="u/nan")
    assert result.point is None and "undefined" in result.reason
    assert isinstance(result, BootstrapResult)
    with pytest.raises(ValueError):
        block_bootstrap([np.zeros(3)], np.zeros(4), np.mean, key="u/len")
    with pytest.raises(ValueError):
        block_bootstrap([], np.zeros(4), np.mean, key="u/none")
    with pytest.raises(ValueError):
        block_bootstrap([np.zeros(4)], np.zeros(4), np.mean, key="u/c", confidence=1.0)
    with pytest.raises(ValueError):
        block_bootstrap([np.zeros(4)], np.zeros(4), np.mean, key="u/r", n_resamples=0)


def test_autocorrelation():
    alternating = np.array([1.0, -1.0] * 50)
    acf = autocorrelation(alternating, [0, 1, 2])
    assert acf[0] == 1.0
    assert acf[1] == pytest.approx(-99 / 100)
    assert acf[2] == pytest.approx(98 / 100)
    assert math.isnan(autocorrelation(np.ones(10), [1])[1])
    with pytest.raises(ValueError):
        autocorrelation(alternating, [100])
    with pytest.raises(ValueError):
        autocorrelation(alternating, [-1])
