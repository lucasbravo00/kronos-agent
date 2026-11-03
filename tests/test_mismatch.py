"""Binance vs BingX mismatch (study 3): every statistic against an independent computation, the
alignment failure, the pre-registered readings (both directions named), the Kronos agreement from
fan summaries, the gate-3 records, the recorded definition and the report runner in its two steps
(temporary data and report roots, no network)."""

from __future__ import annotations

import ast
import itertools
import json
import math
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes

from kronos_agent.data import mismatch as mm
from kronos_agent.data.forecast_store import fan_relpath
from kronos_agent.data.fsutil import exclusive_lock
from kronos_agent.data.partition import (
    SEALED_START,
    SEALED_START_MS,
    VALIDATION_END,
    SealedPartitionError,
)
from kronos_agent.data.store import (
    BINANCE_COLUMNS,
    StoreBusyError,
    StoreId,
    float_values,
    read_meta,
)
from kronos_agent.data.timeutil import HOUR_MS, from_ms, parse_utc_hour, to_ms, utc_text
from kronos_agent.eval.bootstrap import autocorrelation, bootstrap_seed
from kronos_agent.exchange.public_market import PUBLIC_HOST, Candle
from kronos_agent.hashing import canonical_json, sha256_file
from kronos_agent.kronos.batch import FanSummary, FanTask, Gate3Refused
from kronos_agent.kronos.features import FEATURE_COLUMNS, add_amount_proxy
from kronos_agent.params import load_params
from kronos_agent.paths import REPO_ROOT, vendored_kronos_dir
from kronos_agent.report.writer import REQUIRED_KEYS, ConfigChangedError, ReportError

BP = 10_000.0
START = mm.OVERLAP_START_MS
TICK = Decimal("0.1")
CONFIG_ID = "0123456789abcdef"
EXCHANGE_DOMAIN = PUBLIC_HOST.split(".", 1)[1]  # no host literal outside exchange/
VECTOR = "465655238.20240235"  # 17 significant digits: pandas' default parser is one ulp off


# -- synthetic sources ----------------------------------------------------------------------------


def bingx_like(frame: pd.DataFrame, *, seed: int, factor: str = "4.5", scale: str = "1") -> list:
    """BingX real-market candles from a Binance string frame: prices rounded to the 0.1 tick
    with a few ticks of noise (times ``scale``), volumes ``factor`` times Binance's."""
    rng = np.random.default_rng(seed)
    candles = []
    for row in frame.itertuples(index=False):
        o, h, lo, c = (
            Decimal(x) * Decimal(scale) for x in (row.open, row.high, row.low, row.close)
        )
        noise = [Decimal(int(v)) * TICK for v in rng.integers(-3, 4, size=4)]
        o2 = (o + noise[0]).quantize(TICK)
        c2 = (c + noise[1]).quantize(TICK)
        h2 = max((h + abs(noise[2])).quantize(TICK), o2, c2)
        l2 = min((lo - abs(noise[3])).quantize(TICK), o2, c2)
        volume = (Decimal(row.volume) * Decimal(factor)).quantize(Decimal("0.0001"))
        candles.append(
            Candle(
                int(row.open_ms),
                format(o2, "f"),
                format(h2, "f"),
                format(l2, "f"),
                format(c2, "f"),
                format(volume, "f"),
            )
        )
    return candles


def sources(count: int, *, seed: int = 5, start_ms: int = START, **kwargs):
    """A Binance 1h string frame and BingX-like candles of the same hours."""
    frame = fakes.binance_frame(start_ms, count, interval="1h", seed=seed, start_price=60_000.0)
    return frame, bingx_like(frame, seed=seed + 1, **kwargs)


def price_frames(closes_b, closes_x, *, start_ms: int = START, seed: int = 0):
    """String frames whose closes are given (Binance two decimals, BingX one decimal)."""
    rng = np.random.default_rng(seed)
    volumes = rng.integers(1, 2_000_000, size=len(closes_b))
    rows = {name: [] for name in BINANCE_COLUMNS}
    candles = []
    for index, (cb, cx) in enumerate(zip(closes_b, closes_x, strict=True)):
        open_ms = start_ms + index * HOUR_MS
        ob = closes_b[index - 1] if index else cb
        ox = closes_x[index - 1] if index else cx
        prices_b = [f"{v:.2f}" for v in (ob, max(ob, cb) * 1.0005, min(ob, cb) * 0.9995, cb)]
        prices_x = [f"{v:.1f}" for v in (ox, max(ox, cx) * 1.0005, min(ox, cx) * 0.9995, cx)]
        volume = int(volumes[index])
        volume_text = f"{volume // 1000}.{volume % 1000:03d}"
        mean = sum(Decimal(p) for p in prices_b) / 4
        quote = (Decimal(volume_text) * mean).quantize(Decimal("0.00001"), ROUND_HALF_EVEN)
        values = [open_ms, *prices_b, volume_text, format(quote, "f"), "7", "0.000", "0.00000"]
        for name, value in zip(BINANCE_COLUMNS, values, strict=True):
            rows[name].append(value)
        candles.append(Candle(open_ms, *prices_x, f"{volume * 4.5 / 1000:.4f}"))
    frame = pd.DataFrame({name: pd.Series(rows[name], dtype=object) for name in BINANCE_COLUMNS})
    frame["open_ms"] = frame["open_ms"].astype(np.int64)
    return frame, candles


def bingx_frame(candles) -> pd.DataFrame:
    return fakes.candles_to_frame(candles)


def walk(count: int, seed: int, sigma: float = 0.003) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(np.cumsum(rng.normal(0.0, sigma, size=count)))


# -- joined hours ---------------------------------------------------------------------------------


def test_joined_hours_is_an_inner_join_with_decimal_equality():
    frame, candles = sources(12)
    frame = frame.copy()
    frame.loc[2, "close"] = "60000.10"
    candles[2] = replace(candles[2], close="60000.1", volume=VECTOR)  # equal as decimals
    frame.loc[3, "open"] = "60000.10"
    candles[3] = replace(candles[3], open="60000.2")  # not equal
    binance = frame.drop(index=[5]).reset_index(drop=True)
    bingx = bingx_frame([c for c in candles if c.open_ms != START + 8 * HOUR_MS])
    joined = mm.joined_hours(binance, bingx)
    expected = [START + k * HOUR_MS for k in range(12) if k not in (5, 8)]
    assert joined["open_ms"].tolist() == expected
    assert joined["open_ms"].dtype == np.int64
    row2 = joined[joined["open_ms"] == START + 2 * HOUR_MS].iloc[0]
    assert bool(row2["equal_close"]) and row2["bingx_volume"] == float(VECTOR)
    assert np.float64(row2["bingx_volume"]).view(np.int64) == np.float64(float(VECTOR)).view(
        np.int64
    )
    row3 = joined[joined["open_ms"] == START + 3 * HOUR_MS].iloc[0]
    assert not bool(row3["equal_open"])
    by_open = {c.open_ms: c for c in candles}
    for row in joined.itertuples(index=False):
        source = frame[frame["open_ms"] == row.open_ms].iloc[0]
        assert row.binance_quote_volume == float(source["quote_volume"])
        assert row.bingx_high == float(by_open[row.open_ms].high)
        equal = Decimal(source["low"]) == Decimal(by_open[row.open_ms].low)
        assert bool(row.equal_low) == equal


def test_joined_hours_refuses_numeric_and_malformed_frames():
    frame, candles = sources(6)
    bingx = bingx_frame(candles)
    numeric = bingx.copy()
    for name in ("open", "high", "low", "close", "volume"):
        numeric[name] = float_values(numeric[name])
    with pytest.raises(mm.MismatchError, match="decimal strings"):
        mm.joined_hours(frame, numeric)
    with pytest.raises(mm.MismatchError, match="quote_volume"):
        mm.joined_hours(frame.drop(columns=["quote_volume"]), bingx)
    with pytest.raises(mm.MismatchError, match="increasing"):
        mm.joined_hours(frame.iloc[::-1].reset_index(drop=True), bingx)
    zero = frame.copy()
    zero.loc[1, "low"] = "0"
    with pytest.raises(mm.MismatchError, match="positive"):
        mm.joined_hours(zero, bingx)
    with pytest.raises(mm.MismatchError, match="no hour"):
        mm.return_statistics(mm.joined_hours(frame.iloc[:0], bingx))


# -- returns and alignment ------------------------------------------------------------------------


def test_returns_use_only_hours_whose_previous_hour_is_in_both_sources():
    closes = 60_000.0 * walk(40, seed=1)
    frame, candles = price_frames(closes, closes * (1.0 + 0.0001 * walk(40, seed=2)))
    binance = frame[frame["open_ms"] != START + 10 * HOUR_MS].reset_index(drop=True)
    bingx = bingx_frame([c for c in candles if c.open_ms != START + 20 * HOUR_MS])
    joined = mm.joined_hours(binance, bingx)
    result = mm.return_statistics(joined)
    close_b = dict(zip(joined["open_ms"], joined["binance_close"], strict=True))
    close_x = dict(zip(joined["open_ms"], joined["bingx_close"], strict=True))
    hours = [t for t in close_b if t - HOUR_MS in close_b]
    assert len(hours) == 40 - 2 - 3  # two missing hours; t = 0, 11 and 21 have no previous
    r_b = np.array([math.log(close_b[t] / close_b[t - HOUR_MS]) for t in hours])
    r_x = np.array([math.log(close_x[t] / close_x[t - HOUR_MS]) for t in hours])
    stats = result["statistics"]
    assert stats["pearson"]["point"] == pytest.approx(np.corrcoef(r_b, r_x)[0, 1], rel=1e-12)
    ranked = pd.Series(r_b).rank().corr(pd.Series(r_x).rank())
    assert stats["spearman"]["point"] == pytest.approx(ranked, rel=1e-12)
    d = (r_x - r_b) * BP
    assert stats["diff_std_bp"]["point"] == pytest.approx(np.std(d, ddof=1), rel=1e-12)
    mad = np.median(np.abs(d - np.median(d)))
    assert stats["diff_mad_bp"]["point"] == pytest.approx(mad, rel=1e-12)
    ratio = np.std(d, ddof=1) / np.std(r_b * BP, ddof=1)
    assert stats["diff_std_ratio"]["point"] == pytest.approx(ratio, rel=1e-12)
    assert stats["diff_mad_ratio"]["point"] == pytest.approx(
        mad / np.std(r_b * BP, ddof=1), rel=1e-12
    )
    assert result["samples"] == {
        "joined_hours": 38,
        "used": 35,
        "lost": 3,
        "lost_reason": "the hour t - 1h is missing in a source (no return at t)",
        "lost_ranges": [  # every exclusion carries its ranges (docs/DATA.md)
            [utc_text(START + k * HOUR_MS), utc_text(START + (k + 1) * HOUR_MS)]
            for k in (0, 11, 21)
        ],
    }
    derived = result["derived"]["one_minus_pearson"]
    assert derived["point"] == 1.0 - stats["pearson"]["point"]
    assert derived["derived_from"] == "mismatch/returns/pearson"
    aligned = mm.alignment(joined)
    assert aligned["statistics"]["lag_0"]["point"] == stats["pearson"]["point"]  # same rows
    assert aligned["statistics"]["lag_0"]["method"] == "point_only"


def test_alignment_lag_convention_and_failure_on_a_shifted_store():
    count = 24 * 7 * 3
    rng = np.random.default_rng(4)
    r_b = rng.normal(0.0, 0.004, size=count)
    later = np.concatenate(([0.0], r_b[:-1]))
    r_x = 0.8 * r_b + 0.5 * later  # BingX partly one hour later
    frame, candles = price_frames(
        60_000.0 * np.exp(np.cumsum(r_b)), 60_000.0 * np.exp(np.cumsum(r_x)), seed=3
    )
    result = mm.alignment(mm.joined_hours(frame, bingx_frame(candles)))
    stats = result["statistics"]
    assert result["lag0_is_largest"] is True
    assert stats["lag_plus_1"]["point"] > 0.3 > abs(stats["lag_minus_1"]["point"])
    assert stats["lag_0"]["point"] > stats["lag_plus_1"]["point"]
    assert set(stats) == {"lag_minus_2", "lag_minus_1", "lag_0", "lag_plus_1", "lag_plus_2"}
    assert result["pairs"]["lag_plus_1"] == result["pairs"]["lag_0"] - 1
    for name in ("lag_minus_2", "lag_minus_1", "lag_plus_1", "lag_plus_2"):
        assert stats[name]["key"] == f"mismatch/alignment/{name}"
        assert stats[name]["seed"] == bootstrap_seed(stats[name]["key"])
    # a store labelled one hour late (close-time labels) or early: the report fails
    closes = 60_000.0 * np.exp(np.cumsum(r_b))
    aligned_frame, aligned = price_frames(closes, closes, seed=3)
    assert mm.alignment(mm.joined_hours(aligned_frame, bingx_frame(aligned)))["lag0_is_largest"]
    for shift in (1, -1):
        moved = [replace(c, open_ms=c.open_ms + shift * HOUR_MS) for c in aligned]
        with pytest.raises(mm.AlignmentError, match="not largest at lag 0"):
            mm.alignment(mm.joined_hours(aligned_frame, bingx_frame(moved)))
    with pytest.raises(mm.MismatchError, match="too few"):
        mm.alignment(mm.joined_hours(frame.iloc[:3], bingx_frame(candles)))


def test_rank_and_correlation_helpers():
    assert mm.average_ranks(np.array([10.0, 20.0, 20.0, 5.0])).tolist() == [2.0, 3.5, 3.5, 1.0]
    assert mm.average_ranks(np.array([1.0, 1.0, 1.0])).tolist() == [2.0, 2.0, 2.0]
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert mm.pearson(x, x) == 1.0 and mm.pearson(x, -x) == -1.0
    assert math.isnan(mm.pearson(x, np.ones(4))) and math.isnan(mm.pearson(x[:1], x[:1]))
    assert mm.spearman(x, x**3) == 1.0  # monotone: rank correlation 1
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    b[:10] = a[:10]
    expected = pd.Series(a).rank().corr(pd.Series(b).rank())
    assert mm.spearman(a, b) == pytest.approx(expected, rel=1e-12)


# -- prices and volume ----------------------------------------------------------------------------


def test_price_differences_against_an_independent_computation():
    frame, candles = sources(24 * 7 * 2, seed=8)
    frame = frame.copy()
    frame.loc[0, "close"] = "60000.10"
    candles[0] = replace(candles[0], close="60000.1")
    joined = mm.joined_hours(frame, bingx_frame(candles))
    result = mm.price_differences(joined)
    stats, described = result["statistics"], result["descriptive"]
    by_open = {c.open_ms: c for c in candles}
    for field in mm.PRICE_FIELDS:
        b = np.array([float(v) for v in frame[field]])
        x = np.array([float(getattr(by_open[t], field)) for t in frame["open_ms"]])
        signed = (x - b) / b * BP
        assert stats[f"{field}_signed_bp_mean"]["point"] == pytest.approx(
            np.mean(signed), rel=1e-12
        )
        assert stats[f"{field}_signed_bp_median"]["point"] == pytest.approx(np.median(signed))
        assert stats[f"{field}_abs_bp_mean"]["point"] == pytest.approx(np.mean(np.abs(signed)))
        assert described[f"{field}_abs_bp_p90"] == pytest.approx(np.quantile(np.abs(signed), 0.9))
        assert described[f"{field}_abs_bp_max"] == pytest.approx(np.max(np.abs(signed)))
        equal = [
            Decimal(a) == Decimal(getattr(by_open[t], field))
            for a, t in zip(frame[field], frame["open_ms"], strict=True)
        ]
        assert stats[f"{field}_equal_share"]["point"] == np.mean(equal)
    assert stats["close_equal_share"]["point"] >= 1 / len(frame)  # "60000.10" == "60000.1"
    closes = np.array([float(v) for v in frame["close"]])
    moves = np.abs(closes[1:] / closes[:-1] - 1.0) * BP
    assert stats["abs_return_bp_median"]["point"] == pytest.approx(np.median(moves))
    assert described["tick_bp_median"] == pytest.approx(np.median(0.1 / closes * BP))
    assert result["external_references"] == {"demo_vs_real_median_bp": 0.05}
    assert stats["close_abs_bp_mean"]["method"] == "point_only"  # two weeks: too few blocks
    assert "fewer than 20 blocks" in stats["close_abs_bp_mean"]["reason"]
    assert set(result["comparators"]) == {*stats, *described}


def test_volume_statistics_lags_and_ratio():
    frame, candles = sources(300, seed=9)
    joined = mm.joined_hours(frame, bingx_frame(candles))
    result = mm.volume_statistics(joined)
    stats = result["statistics"]
    vb = joined["binance_volume"].to_numpy()
    vx = joined["bingx_volume"].to_numpy()
    expected = pd.Series(vb).rank().corr(pd.Series(vx).rank())
    assert stats["spearman"]["point"] == pytest.approx(expected, rel=1e-12)
    later = pd.Series(vb[:-1]).rank().corr(pd.Series(vx[1:]).rank())
    assert stats["spearman_lag_plus_1"]["point"] == pytest.approx(later, rel=1e-12)
    assert stats["spearman_lag_plus_1"]["n"] == stats["spearman_lag_minus_1"]["n"] == 299
    zero = vb == 0
    assert zero.sum() >= 3  # binance_frame has zero-volume hours
    assert result["descriptive"]["median_ratio"] == pytest.approx(np.median(vx[~zero] / vb[~zero]))
    assert result["samples"]["zero_binance_volume_hours"] == int(zero.sum())
    (exclusion,) = result["samples"]["ratio_exclusions"]  # the ranges of the excluded hours
    assert exclusion["hours"] == int(zero.sum())
    covered = [
        hour
        for start, end in exclusion["ranges"]
        for hour in range(to_ms(parse_utc_hour(start)), to_ms(parse_utc_hour(end)), HOUR_MS)
    ]
    assert covered == joined["open_ms"].to_numpy()[zero].tolist()


# -- the amount proxy -----------------------------------------------------------------------------


def predictor_lines() -> str:
    return (vendored_kronos_dir() / "model" / "kronos.py").read_text(encoding="utf-8")


def test_predictor_z_is_the_pinned_normalisation():
    source = predictor_lines()  # the lines predictor_z reproduces, at the vendored commit
    assert "x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)" in (
        source
    )
    assert "x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)" in source
    assert "x = (x - x_mean) / (x_std + 1e-5)" in source
    assert "x = np.clip(x, -self.clip, self.clip)" in source
    rng = np.random.default_rng(3)
    context = pd.DataFrame(rng.normal(100.0, 5.0, size=(512, 6)), columns=list(FEATURE_COLUMNS))
    context.loc[7, "amount"] = 1e6  # an outlier beyond +5 sigma
    x = context[list(FEATURE_COLUMNS)].values.astype(np.float32)
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    expected = np.clip((x - x_mean) / (x_std + 1e-5), -5, 5)
    z = mm.predictor_z(context)
    assert z.dtype == np.float32 and z.shape == (512, 6)
    assert z.tobytes() == expected.tobytes()
    assert z[7, mm.AMOUNT_INDEX] == 5.0


def test_amount_proxy_error_and_dz():
    count = 24 * 7 * 6
    frame, candles = sources(count, seed=13)
    hours = [from_ms(START + k * HOUR_MS) for k in (100, 511, 700, 900)]
    result = mm.amount_proxy(frame, sample_hours=hours, bingx_1h=bingx_frame(candles))
    floats = pd.DataFrame(
        {n: float_values(frame[n]) for n in ("open", "high", "low", "close", "volume")}
    )
    proxy = (floats["volume"] * floats[["open", "high", "low", "close"]].mean(axis=1)).to_numpy()
    quote = float_values(frame["quote_volume"])
    positive = quote > 0
    e_bp = (proxy[positive] / quote[positive] - 1.0) * BP
    stats = result["statistics"]
    assert stats["e_bp_mean"]["point"] == pytest.approx(np.mean(e_bp), rel=1e-12)
    assert stats["e_bp_median"]["point"] == pytest.approx(np.median(e_bp), rel=1e-12)
    assert result["descriptive"]["abs_e_bp_max"] == pytest.approx(np.max(np.abs(e_bp)))
    zero = frame["open_ms"].to_numpy()[~positive]
    assert zero.size >= 5  # binance_frame's zero-volume hours
    hours_samples = result["samples"]["hours"]
    assert hours_samples["used"] == int(positive.sum())
    assert hours_samples["exclusions"][0]["hours"] == zero.size
    sample = result["samples"]["sample_hours"]
    assert sample["used"] == 2 and sample["skipped"] == 2  # 100 and 511 lack 512 hours
    assert sample["skipped_hours"][0]["reason"].startswith(
        "incomplete 512-candle window in Binance"
    )
    window = result["window_series"]
    assert window["decision_ms"].tolist() == [START + 700 * HOUR_MS, START + 900 * HOUR_MS]
    # an independent dz at D = START + 700h
    d_ms = START + 700 * HOUR_MS
    rows = (floats.index >= 700 - 512) & (floats.index < 700)
    base = floats[rows].reset_index(drop=True)
    with_proxy = add_amount_proxy(base)
    with_quote = base.assign(amount=quote[rows])
    bingx = pd.DataFrame(
        {
            n: float_values([getattr(c, n) for c in candles[700 - 512 : 700]])
            for n in ("open", "high", "low", "close", "volume")
        }
    )
    bingx = add_amount_proxy(bingx)

    def z(context):
        x = context[list(FEATURE_COLUMNS)].values.astype(np.float32)
        return np.clip((x - x.mean(axis=0)) / (x.std(axis=0) + 1e-5), -5, 5)

    dz_proxy = np.abs(z(with_proxy)[:, 5] - z(with_quote)[:, 5]).astype(np.float64)
    dz_volume = np.abs(z(with_proxy)[:, 4] - z(bingx)[:, 4]).astype(np.float64)
    first = window[window["decision_ms"] == d_ms].iloc[0]
    assert first["dz_proxy_median"] == np.median(dz_proxy)
    assert first["dz_proxy_max"] == np.max(dz_proxy)
    assert first["dz_volume_median"] == np.median(dz_volume)
    assert first["dz_volume_max"] == np.max(dz_volume)
    assert stats["dz_median_difference"]["point"] == pytest.approx(
        window["dz_proxy_median"].mean() - window["dz_volume_median"].mean()
    )


def test_dz_is_zero_when_the_quote_volume_is_the_proxy():
    frame, _ = sources(24 * 7 * 4, seed=14)
    frame = frame.copy()
    floats = pd.DataFrame(
        {n: float_values(frame[n]) for n in ("open", "high", "low", "close", "volume")}
    )
    proxy = add_amount_proxy(floats)["amount"].to_numpy()
    frame["quote_volume"] = [format(Decimal(repr(float(v))), "f") for v in proxy]
    same = bingx_frame(fakes.frame_to_candles(frame))
    hours = [from_ms(START + k * HOUR_MS) for k in (520, 600)]
    result = mm.amount_proxy(frame, sample_hours=hours, bingx_1h=same)
    window = result["window_series"]
    assert (window[["dz_proxy_median", "dz_proxy_max"]] == 0.0).all().all()
    assert (window[["dz_volume_median", "dz_volume_max"]] == 0.0).all().all()
    positive = proxy > 0
    assert result["statistics"]["e_bp_mean"]["point"] == 0.0
    assert result["samples"]["hours"]["used"] == int(positive.sum())


# -- the Kronos agreement -------------------------------------------------------------------------


def test_kronos_samples_and_tasks():
    hours = mm.kronos_sample_hours()
    assert len(hours) == 291
    assert utc_text(to_ms(hours[0])) == "2024-07-01T00:00Z"
    assert hours[-1] < VALIDATION_END <= hours[-1] + timedelta(hours=61)
    assert {b - a for a, b in itertools.pairwise(hours)} == {timedelta(hours=61)}
    assert {hour.hour for hour in hours} == set(range(24))
    fallback = mm.kronos_fallback_hours()
    assert len(fallback) == 183 and fallback[1] - fallback[0] == timedelta(hours=97)
    assert fallback[-1] < VALIDATION_END <= fallback[-1] + timedelta(hours=97)
    tasks = mm.kronos_tasks(hours[:2])
    assert tasks == [
        FanTask("binance", hours[0], "forecast"),
        FanTask("bingx", hours[0], "forecast"),
        FanTask("bingx", hours[0], "mismatch-floor"),
        FanTask("binance", hours[1], "forecast"),
        FanTask("bingx", hours[1], "forecast"),
        FanTask("bingx", hours[1], "mismatch-floor"),
    ]
    assert len(mm.kronos_tasks(hours)) == 873
    with pytest.raises(SealedPartitionError):
        mm.kronos_tasks([SEALED_START])
    with pytest.raises(mm.MismatchError, match="increasing"):
        mm.kronos_tasks(list(reversed(hours[:3])))


def summary(
    source, stream, hour, *, signal, p_up=0.5, window=None, config=CONFIG_ID, computed=True
):
    window = window or ("a" * 64 if source == "bingx" else "b" * 64)
    return FanSummary(
        source=source,
        decision_hour=utc_text(to_ms(hour)),
        stream=stream,
        relpath=fan_relpath(source, config, stream, hour),
        window_sha256=window,
        inputs_sha256=window,
        signal=signal,
        p_up=p_up,
        last_close=60_000.0,
        mean_close_n=60_000.0 + signal,
        elapsed_s=3.0,
        computed=computed,
    )


def trio_summaries(hours, signals, p_ups=None):
    """Three summaries per hour from ``signals[k] = (binance, bingx, floor)``."""
    items = []
    for index, (hour, (s_b, s_x, s_f)) in enumerate(zip(hours, signals, strict=True)):
        p = p_ups[index] if p_ups is not None else (0.5, 0.5, 0.5)
        items += [
            summary("binance", "forecast", hour, signal=s_b, p_up=p[0]),
            summary("bingx", "forecast", hour, signal=s_x, p_up=p[1]),
            summary("bingx", "mismatch-floor", hour, signal=s_f, p_up=p[2]),
        ]
    return items


def shifted_signals(count, *, binance_every, floor_every, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        base = int(rng.choice([-1, 1]))
        s_b = -base if binance_every and k % binance_every == 0 else base
        s_f = -base if floor_every and k % floor_every == 0 else base
        out.append((s_b, base, s_f))
    return out


def test_kronos_agreement_statistics_and_the_data_effect_reading():
    hours = mm.kronos_sample_hours()
    signals = shifted_signals(len(hours), binance_every=0, floor_every=5)
    rng = np.random.default_rng(1)
    p_ups = [tuple(rng.random(3)) for _ in hours]
    result = mm.kronos_agreement(trio_summaries(hours, signals, p_ups))
    stats = result["statistics"]
    flags_f = np.array([s[1] == s[2] for s in signals])
    assert stats["a_x_share"]["point"] == 1.0 and stats["a_x_share"]["method"] == "wilson_blocks"
    assert stats["a_f_share"]["point"] == np.mean(flags_f)
    assert stats["a_x_minus_a_f"]["point"] == pytest.approx(1.0 - np.mean(flags_f))
    assert stats["a_x_minus_a_f"]["low"] > 0.0
    readings = result["readings"]
    assert readings["a_x_minus_a_f"]["reading"] == mm.READING_DATA_EFFECT
    assert readings["a_x_minus_a_f"]["interval_position"] == "entirely above 0"
    z = NormalDist().inv_cdf(0.975)
    difference = stats["a_x_minus_a_f"]
    expected = (z + NormalDist().inv_cdf(0.8)) * (difference["high"] - difference["low"]) / (2 * z)
    mdd = readings["a_x_minus_a_f"]["minimum_detectable_difference"]
    assert mdd["value"] == pytest.approx(expected) and mdd["reason"] is None
    # A_xd equals A_f in every resample: no interval, not distinguishable
    assert stats["a_xd_minus_a_f"]["point"] == 0.0 and stats["a_xd_minus_a_f"]["low"] is None
    assert readings["a_xd_minus_a_f"]["reading"] == mm.READING_NOT_DISTINGUISHABLE
    assert readings["a_xd_minus_a_f"]["minimum_detectable_difference"]["value"] is None
    p = np.array(p_ups)
    assert stats["p_up_x_correlation"]["point"] == pytest.approx(
        np.corrcoef(p[:, 0], p[:, 1])[0, 1], rel=1e-12
    )
    assert stats["p_up_f_abs_delta_mean"]["point"] == pytest.approx(
        np.mean(np.abs(p[:, 1] - p[:, 2]))
    )
    expected_acf = autocorrelation(flags_f.astype(float), range(1, 9))
    assert result["autocorrelation"]["a_f"]["values"] == {
        str(k): pytest.approx(v) for k, v in expected_acf.items()
    }
    assert result["autocorrelation"]["a_x"]["values"]["1"] is None  # constant flags
    assert result["owner_reference"] == 0.88 and result["design"] == "primary"
    assert result["samples"]["eligible"] == 291 and result["samples"]["used"] == 291
    assert result["fans"]["summaries"] == 873 and result["fans"]["config_id"] == CONFIG_ID
    assert stats["a_f_share"]["key"] == "mismatch/kronos/a_f_share"
    blocks = stats["a_f_share"]["n_blocks"]
    assert 20 <= blocks <= 28  # groups of 4 calendar weeks over two years
    series = result["series"]
    assert series["agree_f"].tolist() == flags_f.tolist()
    assert series["signal_binance"].tolist() == [s[0] for s in signals]


def test_kronos_agreement_sources_shift_reading_and_exclusions():
    hours = mm.kronos_sample_hours()
    signals = shifted_signals(len(hours), binance_every=3, floor_every=20, seed=2)
    items = trio_summaries(hours, signals)
    dropped = utc_text(to_ms(hours[10]))
    items = [s for s in items if not (s.decision_hour == dropped and s.stream == "mismatch-floor")]
    result = mm.kronos_agreement(items)
    readings, stats = result["readings"], result["statistics"]
    assert stats["a_xd_minus_a_f"]["high"] < 0.0
    assert readings["a_xd_minus_a_f"]["reading"] == mm.READING_SOURCES_SHIFT
    assert stats["a_x_minus_a_f"]["high"] < 0.0  # the data effect beats seed noise: named
    assert readings["a_x_minus_a_f"]["reading"] == mm.READING_DATA_EFFECT_LARGER
    assert readings["a_x_minus_a_f"]["interval_position"] == "entirely below 0"
    assert result["samples"]["used"] == 290 and result["samples"]["excluded"] == 1
    assert result["samples"]["excluded_hours"][0]["decision_hour"] == dropped
    assert "bingx_floor" in result["samples"]["excluded_hours"][0]["reason"]
    kept = [s for k, s in enumerate(signals) if k != 10]
    a_x = np.array([s[0] == s[1] for s in kept])
    a_f = np.array([s[1] == s[2] for s in kept])
    assert stats["a_x_minus_a_f"]["point"] == pytest.approx(a_x.mean() - a_f.mean())


def interval(low, high, point=0.0) -> dict:
    return {"point": point, "low": low, "high": high}


@pytest.mark.parametrize(
    "result, position",
    [
        (interval(0.01, 0.2), "entirely above 0"),
        (interval(-0.2, -0.01), "entirely below 0"),
        (interval(-0.1, 0.1), "contains 0"),
        (interval(0.0, 0.1), "contains 0"),  # touching 0 does not exclude it
        (interval(None, None), "no interval"),
    ],
)
def test_readings_name_both_directions_and_never_call_an_excluded_zero_indistinguishable(
    result, position
):
    pairs = {
        "a_x_minus_a_f": (mm.READING_DATA_EFFECT, mm.READING_DATA_EFFECT_LARGER),
        "a_xd_minus_a_f": (mm.READING_SOURCES_ABOVE_FLOOR, mm.READING_SOURCES_SHIFT),
    }
    for above, below in pairs.values():
        reading = mm._reading(result, above=above, below=below)
        assert reading["interval_position"] == position
        expected = {"entirely above 0": above, "entirely below 0": below}.get(
            position, mm.READING_NOT_DISTINGUISHABLE
        )
        assert reading["reading"] == expected
    assert len({text for pair in pairs.values() for text in pair}) == 4


def test_kronos_agreement_names_an_above_floor_cross_source_agreement():
    """A_xd above A_f (the cross-source pair agrees more than the two-seed floor) is named, not
    called "not distinguishable"."""
    hours = mm.kronos_sample_hours()
    signals = shifted_signals(len(hours), binance_every=0, floor_every=3, seed=4)
    # Binance agrees with the floor wherever BingX's forecast does not
    signals = [(s_f, s_x, s_f) for _s_b, s_x, s_f in signals]
    result = mm.kronos_agreement(trio_summaries(hours, signals))
    stats, readings = result["statistics"], result["readings"]
    assert stats["a_xd_minus_a_f"]["low"] > 0.0
    assert readings["a_xd_minus_a_f"]["reading"] == mm.READING_SOURCES_ABOVE_FLOOR
    assert readings["a_xd_minus_a_f"]["interval_position"] == "entirely above 0"
    # A_x equals A_f hour by hour here: no interval, the one case left "not distinguishable"
    assert stats["a_x_minus_a_f"]["low"] is None
    assert readings["a_x_minus_a_f"]["reading"] == mm.READING_NOT_DISTINGUISHABLE


def test_kronos_agreement_fallback_design_and_refusals():
    fallback = mm.kronos_fallback_hours()
    result = mm.kronos_agreement(trio_summaries(fallback[:30], [(1, 1, 1)] * 30))
    assert result["design"] == "fallback" and result["samples"]["eligible"] == 183
    assert result["readings"]["planned_mdd"].startswith("about 11 points")
    hours = mm.kronos_sample_hours()[:3]
    good = trio_summaries(hours, [(1, 1, -1)] * 3)
    with pytest.raises(mm.MismatchError, match="several forecast configurations"):
        mm.kronos_agreement(
            [
                *good[:-1],
                replace(
                    good[-1],
                    relpath=fan_relpath("bingx", "fedcba9876543210", "mismatch-floor", hours[-1]),
                ),
            ]
        )
    with pytest.raises(mm.MismatchError, match="two bingx fans"):
        mm.kronos_agreement([*good, good[1]])
    with pytest.raises(mm.MismatchError, match="unexpected fan"):
        mm.kronos_agreement([*good, summary("binance", "mismatch-floor", hours[0], signal=1)])
    with pytest.raises(mm.MismatchError, match="does not read the BingX forecast window"):
        mm.kronos_agreement([*good[:-1], replace(good[-1], window_sha256="c" * 64)])
    with pytest.raises(mm.MismatchError, match="outside both"):
        mm.kronos_agreement(trio_summaries([hours[0] + timedelta(hours=1)], [(1, 1, 1)]))
    with pytest.raises(mm.MismatchError, match="not where its summary says"):
        mm.kronos_agreement([*good[:-1], replace(good[-1], relpath=good[0].relpath)])
    with pytest.raises(mm.MismatchError, match="nothing to measure"):
        mm.kronos_agreement([])
    with pytest.raises(mm.MismatchError, match="FanSummary"):
        mm.kronos_agreement([{"source": "bingx"}])


# -- the recorded definition ----------------------------------------------------------------------


def test_study_config_records_every_statistic_rule_and_fan_setting():
    config = mm.study_config()
    assert canonical_json(config) == canonical_json(mm.study_config())
    text = canonical_json(config).decode("ascii")
    assert str(REPO_ROOT) not in text and str(Path.home()) not in text
    assert EXCHANGE_DOMAIN not in text
    assert config["study"] == "mismatch" and config["tests_consumed"] == 0
    assert config["intervals"]["resamples"] == 1000 and config["intervals"]["confidence"] == 0.95
    assert config["range"] == {
        "from": "2024-04-27T10:00Z",
        "to": "2026-08-01T00:00Z",
        "rule": config["range"]["rule"],
        "partition": config["range"]["partition"],
    }
    keys = [item["seed_key"] for item in config["statistics"] if item["seed_key"]]
    assert len(keys) == len(set(keys))
    assert mm.SECTIONS == ("alignment", "returns", "prices", "volume", "amount", "kronos")
    assert {key.split("/")[1] for key in keys} == set(mm.SECTIONS)
    for item in config["statistics"]:
        assert item["comparator"], item["name"]
    rules = " ".join(config["reading_rules"])
    for sentence in (
        mm.READING_DATA_EFFECT,
        mm.READING_DATA_EFFECT_LARGER,
        mm.READING_SOURCES_SHIFT,
        mm.READING_SOURCES_ABOVE_FLOOR,
    ):
        assert sentence in rules
    assert "containing 0, or no interval" in rules
    gate3 = config["kronos"]["gate3"]
    assert gate3["approval_tolerance"] == 1.25 and "mismatch.gate3.jsonl" in gate3["rule"]
    assert "--approved-by-owner CPU_HOURS" in gate3["rule"]
    assert "rebuilds it from the stored fans" in config["kronos"]["status_rule"]
    # the volume profile and its POC left the study (D-042): nothing of them is recorded
    assert re.search(r"(?i)\bpoc\b|profile", text) is None
    kronos = load_params().kronos
    fans = config["kronos"]["fan_configuration"]
    assert (fans["model"], fans["horizon_n"], fans["trajectories"]) == (
        kronos.model,
        kronos.horizon_n,
        kronos.trajectories,
    )
    assert config["kronos"]["sample"]["count"] == 291
    assert config["kronos"]["fallback"]["count"] == 183
    assert config["kronos"]["gate3"]["cpu_seconds_limit"] == 3600


def test_mismatch_never_calls_read_store():
    tree = ast.parse((REPO_ROOT / "src/kronos_agent/data/mismatch.py").read_text())
    names = {
        getattr(node, "id", None) or getattr(node, "attr", None) or getattr(node, "name", None)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute, ast.alias))
    }
    assert "read_store" not in names
    assert "read_study_frame" in names


# -- the report -----------------------------------------------------------------------------------


def write_sources(root: Path, weeks: int = 22, **kwargs) -> tuple[pd.DataFrame, list]:
    frame, candles = sources(24 * 7 * weeks, **kwargs)
    fakes.write_binance_store(root, frame)
    fakes.write_bingx_store(root, candles)
    return frame, candles


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(mm, "RESAMPLES", 200)  # recorded as such in the test's config


def gate3_invocation(
    count: int, *, approved: float | None = 2.0, cpu_seconds: float | None = None
) -> dict:
    """The gate-3 data of a ``mismatch --kronos`` run: its estimate and the approved figure."""
    seconds = count * 3.15 * 2 if cpu_seconds is None else cpu_seconds
    return {
        "estimate": {
            "tasks": count,
            "dropped": 0,
            "stored": 0,
            "pending": count,
            "seconds_per_fan": 3.15,
            "threads": 2,
            "workers": 4,
            "cpu_seconds": seconds,
            "cpu_hours": seconds / 3600,
            "wall_seconds": seconds / 8,
            "config_id": CONFIG_ID,
        },
        "approved_cpu_hours": approved,
        "dropped": [],
    }


GIVEN_AT = datetime(2026, 11, 2, 10, 5, 7, tzinfo=UTC)


def approved_record(*, figure: float = 1.6, design: str = "primary", **estimate) -> dict:
    """The record an approved 873-fan batch appends before its first fan."""
    return mm.gate3_record(
        design=design,
        estimate={**gate3_invocation(873)["estimate"], **estimate},
        approved_cpu_hours=figure,
        dropped=[],
        given_at=GIVEN_AT,
    )


def test_build_report_writes_everything_and_reproduces_it(tmp_path, fast, monkeypatch):
    root, reports = tmp_path / "data", tmp_path / "reports"
    write_sources(root)
    calls = []
    real_read = mm.read_study_frame

    def recording_read(store_id, **kwargs):
        calls.append((store_id.relpath, kwargs["start_ms"], kwargs["end_ms"], kwargs["numeric"]))
        return real_read(store_id, **kwargs)

    monkeypatch.setattr(mm, "read_study_frame", recording_read)
    payload = mm.build_report(root=root, directory=reports)
    assert calls == [
        ("candles/binance/1h.csv", START, SEALED_START_MS, False),
        ("candles/bingx/1h.csv", START, SEALED_START_MS, False),
    ]
    document = json.loads((reports / "mismatch.json").read_text())
    assert set(REQUIRED_KEYS) <= set(document)
    assert document["tests_consumed"] == 0 and document["study"] == "mismatch"
    config_path = reports / "mismatch.config.json"
    assert config_path.read_bytes() == canonical_json(mm.study_config())
    assert document["config_sha256"] == sha256_file(config_path)
    hours = document["samples"]["hours"]
    assert hours["eligible"] == (SEALED_START_MS - START) // HOUR_MS
    assert hours["used"] == 24 * 7 * 22
    assert hours["exclusions"][0]["reason"] == "no candle in either source"
    sections = document["sections"]
    assert set(sections) == set(mm.SECTIONS)  # no POC section (D-042)
    assert set(document["samples"]) == {"hours", "returns", "volume", "amount", "kronos"}
    assert sections["kronos"]["status"] == "not_run"
    assert document["seeds"]["kronos"] is None
    computed = set(document["seeds"]["bootstrap"].values())
    recorded = {
        item["seed_key"]
        for item in json.loads(config_path.read_text())["statistics"]
        if item["seed_key"] and not item["seed_key"].startswith("mismatch/kronos/")
    }
    assert computed == recorded
    intervals = sum(
        r.get("low") is not None and r.get("high") is not None
        for body in sections.values()
        for r in (body.get("statistics") or {}).values()
    )
    assert document["statistics_with_intervals"] == intervals > 30
    assert document["expected_chance_misses"] == pytest.approx(intervals * 0.05)
    assert set(document["series"]) == {"hours", "amount", "amount_windows"}
    for name, item in document["series"].items():
        assert item["relpath"] == f"studies/{mm.SERIES_STEMS[name]}.csv"
        assert sha256_file(root / item["relpath"]) == item["sha256"]
    figures = {item["name"]: item for item in document["figures"]}
    assert set(figures) == {"return_scatter", "price_differences"}
    assert sorted(path.name for path in (reports / "figures").iterdir()) == [
        "mismatch_price_differences.png",
        "mismatch_return_scatter.png",
    ]
    for item in figures.values():
        assert sha256_file(reports / item["relpath"]) == item["sha256"]
    assert [s["relpath"] for s in document["data"]["stores"]] == [
        "candles/binance/1h.csv",
        "candles/bingx/1h.csv",
    ]
    markdown = (reports / "mismatch.md").read_text()
    assert markdown.isascii() and "Kronos signal agreement: not run" in markdown
    titles = [line for line in markdown.splitlines() if line.startswith("## ")]
    assert titles == [
        "## Samples",
        "## Alignment (1h log returns at lags -2..+2 h)",
        "## Returns",
        "## Prices: (BingX - Binance) / Binance in bp",
        "## Volume",
        "## amount proxy (question 9)",
        "## Kronos signal agreement",
        "## Comparators",
        "## Series, figures and data",
    ]
    assert "Study 3 of docs/DATA.md" in markdown
    first_hour = f"[{utc_text(START)}, {utc_text(START + HOUR_MS)})"
    assert f"(no return at t)): 1 hours: {first_hour}" in markdown  # lost hours with ranges
    assert "- Excluded (zero Binance volume (median_ratio undefined)):" in markdown
    # the amount section's exclusions, with their ranges and reasons (mismatch-core-4)
    amount = document["samples"]["amount"]
    assert (
        f"- Amount proxy e_t: eligible {amount['hours']['binance_hours']} Binance hours, used "
        f"{amount['hours']['used']}"
    ) in markdown
    (zero,) = amount["hours"]["exclusions"]
    first_zero = f"[{zero['ranges'][0][0]}, {zero['ranges'][0][1]})"
    assert (
        f"- Excluded (zero quote volume (e_t undefined)): {zero['hours']} hours: {first_zero}"
    ) in markdown
    skipped = amount["sample_hours"]["skipped_hours"]
    assert len(skipped) == 255  # the data stop after 22 weeks
    for reason in {item["reason"] for item in skipped}:
        hours_of = [item["decision_hour"] for item in skipped if item["reason"] == reason]
        assert f"- Skipped ({reason}): {len(hours_of)} sample hours: {hours_of[0]}, " in markdown
    assert "incomplete 512-candle window in" in markdown and "more (JSON)" in markdown
    for text in (markdown, (reports / "mismatch.json").read_text()):
        assert str(tmp_path) not in text and EXCHANGE_DOMAIN not in text
        assert re.search(r"(?i)\bpoc\b|profile", text) is None
    hours_series = pd.read_csv(root / document["series"]["hours"]["relpath"])
    assert hours_series["open_ms"].tolist() == [START + k * HOUR_MS for k in range(24 * 7 * 22)]
    assert payload["config_sha256"] == document["config_sha256"]

    series_bytes = {n: (root / i["relpath"]).read_bytes() for n, i in document["series"].items()}
    figure_bytes = {n: (reports / i["relpath"]).read_bytes() for n, i in figures.items()}
    again = mm.build_report(root=root, directory=reports)
    assert again["sections"] == payload["sections"]
    assert {n: (root / i["relpath"]).read_bytes() for n, i in again["series"].items()} == (
        series_bytes
    )
    assert {
        item["name"]: (reports / item["relpath"]).read_bytes() for item in again["figures"]
    } == figure_bytes


def test_build_report_with_the_kronos_part(tmp_path, fast):
    root, reports = tmp_path / "data", tmp_path / "reports"
    write_sources(root)
    hours = mm.kronos_sample_hours()
    items = trio_summaries(hours, shifted_signals(len(hours), binance_every=3, floor_every=20))
    with pytest.raises(mm.MismatchError, match="gate-3 record"):
        mm.build_report(root=root, directory=reports, kronos_summaries=items)
    for figure in (None, 1.0):  # no approved figure; 1.53 measured CPU-hours = 1.53 x 1.0
        with pytest.raises(Gate3Refused):
            mm.build_report(
                root=root,
                directory=reports,
                kronos_summaries=items,
                gate3=gate3_invocation(len(items), approved=figure),
            )
    with pytest.raises(mm.MismatchError, match="counted"):
        mm.build_report(
            root=root, directory=reports, kronos_summaries=items[:-3], gate3=gate3_invocation(873)
        )
    assert not (reports / "mismatch.json").exists()
    record = approved_record()  # appended by the approved batch before its first fan
    other_config = "fedcba9876543210"
    mm.append_gate3_record(approved_record(config_id=other_config), directory=reports)
    mm.append_gate3_record(record, directory=reports)
    payload = mm.build_report(
        root=root,
        directory=reports,
        kronos_summaries=items,
        gate3=gate3_invocation(len(items), approved=1.6),
    )
    kronos = payload["sections"]["kronos"]
    assert kronos["status"] == "run" and kronos["design"] == "primary"
    gate3 = kronos["gate3"]
    assert gate3["records"] == [record]  # every record of these fans' config, no other
    assert gate3["records_file"] == "mismatch.gate3.jsonl"
    assert (gate3["cpu_seconds_limit"], gate3["approval_tolerance"]) == (3600, 1.25)
    invocation = gate3["invocation"]
    assert (
        invocation["approved_cpu_hours"] == 1.6 and invocation["rebuilt_from_stored_fans"] is False
    )
    assert invocation["estimate"]["cpu_seconds"] == pytest.approx(873 * 3.15 * 2)
    assert invocation["ratio"] == pytest.approx(873 * 3.15 * 2 / 3600 / 1.6)
    assert gate3["fans_cpu_seconds"] == pytest.approx(873 * 3.0 * 2)  # elapsed 3 s, 2 threads
    assert payload["seeds"]["kronos"]["streams"] == {
        "binance": "binance/forecast",
        "bingx": "bingx/forecast",
        "bingx_floor": "bingx/mismatch-floor",
    }
    assert "kronos/a_xd_minus_a_f" in payload["seeds"]["bootstrap"]
    assert payload["samples"]["kronos"]["used"] == 291
    assert {item["name"] for item in payload["figures"]} >= {"agreement"}
    markdown = (reports / "mismatch.md").read_text()
    assert f"**{mm.READING_SOURCES_SHIFT}**" in markdown
    assert f"**{mm.READING_DATA_EFFECT_LARGER}**" in markdown  # A_x - A_f below 0
    for line in markdown.splitlines():  # an interval excluding 0 is never "not distinguishable"
        if "not distinguishable" in line:
            assert "entirely above 0" not in line and "entirely below 0" not in line
    assert "Gate-3 records of these fans' forecast config (mismatch.gate3.jsonl" in markdown
    assert (
        "- primary sample: the owner approved 1.6 CPU-hours (--approved-by-owner given at "
        "2026-11-02T10:05:07Z); measured 1.528 CPU-hours, 0.9548 times the approved figure (at "
        "most 1.25)"
    ) in markdown
    assert "This run: gate-3 estimate 5500 CPU-seconds (873 pending fans x 3.15 s x 2" in markdown
    assert (root / "studies/mismatch_kronos.csv").exists()

    # the part rebuilt from the stored fans: nothing measured, the computing record still listed
    stored = [replace(item, computed=False) for item in items]
    payload = mm.build_report(
        root=root,
        directory=reports,
        kronos_summaries=stored,
        gate3={"estimate": None, "dropped": []},
    )
    gate3 = payload["sections"]["kronos"]["gate3"]
    assert gate3["records"] == [record] and gate3["invocation"]["rebuilt_from_stored_fans"]
    markdown = (reports / "mismatch.md").read_text()
    assert "This run rebuilt the part from the stored fans: no fan pending" in markdown
    assert "the owner approved 1.6 CPU-hours" in markdown
    with pytest.raises(mm.MismatchError, match="without an estimate"):
        mm.build_report(
            root=root,
            directory=reports,
            kronos_summaries=stored,
            gate3={"estimate": None, "approved_cpu_hours": 1.6, "dropped": []},
        )
    # an approval below the gate-3 limit is not needed
    small = items[:9]
    payload = mm.build_report(
        root=root,
        directory=reports,
        kronos_summaries=small,
        gate3=gate3_invocation(9, approved=None, cpu_seconds=60.0),
    )
    assert payload["sections"]["kronos"]["samples"]["used"] == 3
    assert payload["sections"]["kronos"]["gate3"]["invocation"]["ratio"] is None


def test_gate3_records_are_append_only_and_checked_again(tmp_path):
    reports = tmp_path / "reports"
    record = approved_record()
    assert record["given_at"] == "2026-11-02T10:05:07Z" and record["design"] == "primary"
    assert record["config_id"] == CONFIG_ID and record["approved_cpu_hours"] == 1.6
    assert record["measured_cpu_hours"] == pytest.approx(873 * 3.15 * 2 / 3600)
    assert record["ratio"] == pytest.approx(record["measured_cpu_hours"] / 1.6)
    assert record["tolerance"] == 1.25 and record["estimate"]["pending"] == 873
    path = mm.append_gate3_record(record, directory=reports)
    assert path == reports / "mismatch.gate3.jsonl"
    first = path.read_bytes()
    assert first == canonical_json(record) + b"\n"
    resumed = mm.gate3_record(  # a resume of 10 fans below the limit: no approval needed
        design="primary",
        estimate={**record["estimate"], "pending": 10, "cpu_seconds": 63.0, "cpu_hours": 0.0175},
        approved_cpu_hours=None,
        dropped=[{"source": "bingx", "decision_hour": "2024-07-01T00:00Z"}],
        given_at=GIVEN_AT + timedelta(hours=1),
    )
    assert resumed["ratio"] is None and resumed["approved_cpu_hours"] is None
    mm.append_gate3_record(resumed, directory=reports)
    assert path.read_bytes().startswith(first)  # the earlier record is kept byte for byte
    assert mm.gate3_records(CONFIG_ID, directory=reports) == [record, resumed]
    assert mm.gate3_records("fedcba9876543210", directory=reports) == []
    assert mm.gate3_records(CONFIG_ID, directory=tmp_path / "none") == []
    # refused like the batch: above the limit without a figure, or more than 1.25 times it
    estimate = record["estimate"]
    for figure, match in ((None, "approved-by-owner"), (1.0, "1.53 times")):
        with pytest.raises(Gate3Refused, match=match):
            mm.gate3_record(
                design="primary",
                estimate=estimate,
                approved_cpu_hours=figure,
                dropped=[],
                given_at=GIVEN_AT,
            )
    with pytest.raises(mm.MismatchError, match="design"):
        approved_record(design="tertiary")
    with pytest.raises(mm.MismatchError, match="aware"):
        mm.gate3_record(
            design="primary",
            estimate=estimate,
            approved_cpu_hours=1.6,
            dropped=[],
            given_at=datetime(2026, 11, 2, 10),
        )
    # a record altered on disk is checked again with the current tolerance
    tampered = {**record, "approved_cpu_hours": 1.0, "ratio": record["measured_cpu_hours"]}
    path.write_bytes(first + canonical_json(tampered) + b"\n")
    with pytest.raises(Gate3Refused):
        mm.gate3_records(CONFIG_ID, directory=reports)
    path.write_bytes(first + canonical_json({**record, "ratio": 0.5}) + b"\n")
    with pytest.raises(mm.MismatchError, match="does not agree"):
        mm.gate3_records(CONFIG_ID, directory=reports)
    path.write_bytes(first + b'{"study": "mismatch"}\n')
    with pytest.raises(mm.MismatchError, match="not a gate-3 record"):
        mm.gate3_records(CONFIG_ID, directory=reports)
    path.write_bytes(first + b'{"study": ')
    with pytest.raises(ReportError, match="torn"):
        mm.gate3_records(CONFIG_ID, directory=reports)
    with pytest.raises(ReportError, match="torn"):  # never appended to, never repaired
        mm.append_gate3_record(record, directory=reports)
    assert path.read_bytes() == first + b'{"study": '


def test_a_report_without_the_kronos_part_lists_every_file_beside_it(tmp_path, fast):
    """D-037: a report without the Kronos part removes that part's figure and series of an
    earlier report, so every file beside it is one it lists with its sha256."""
    root, reports = tmp_path / "data", tmp_path / "reports"
    write_sources(root)
    hours = mm.kronos_sample_hours()
    items = trio_summaries(hours, shifted_signals(len(hours), binance_every=3, floor_every=20))
    mm.build_report(
        root=root, directory=reports, kronos_summaries=items, gate3=gate3_invocation(len(items))
    )
    assert (reports / "figures" / mm.AGREEMENT_FIGURE).is_file()
    assert (root / "studies" / "mismatch_kronos.csv").is_file()
    reason = "not run: primary sample 300 of 873 fans stored (unit test)"
    payload = mm.build_report(root=root, directory=reports, kronos_reason=reason)
    assert payload["sections"]["kronos"] == {"status": "not_run", "reason": reason}
    assert f"Kronos signal agreement: {reason}." in (reports / "mismatch.md").read_text()
    assert {f"figures/{p.name}" for p in (reports / "figures").iterdir()} == {
        item["relpath"] for item in payload["figures"]
    }
    assert {f"studies/{p.name}" for p in (root / "studies").iterdir()} == {
        item["relpath"] for item in payload["series"].values()
    }
    payload = mm.build_report(root=root, directory=reports)
    assert payload["sections"]["kronos"]["reason"] == mm.KRONOS_NOT_RUN
    for claim in ("go-ahead", "step 2", "approval"):  # it describes this run only
        assert claim not in mm.KRONOS_NOT_RUN
    with pytest.raises(mm.MismatchError, match="did not run"):
        mm.build_report(
            root=root,
            directory=reports,
            kronos_summaries=items,
            gate3=gate3_invocation(len(items)),
            kronos_reason=reason,
        )


def test_the_report_records_each_store_as_it_was_read(tmp_path, fast, monkeypatch):
    """rules-3: a store rebuilt while the study computes does not become the one recorded."""
    root, reports = tmp_path / "data", tmp_path / "reports"
    _frame, candles = write_sources(root)
    read_sha = read_meta(StoreId("bingx", "1h"), root)["csv_sha256"]
    real_amount = mm.amount_proxy

    def amount_then_rebuild(*args, **kwargs):
        result = real_amount(*args, **kwargs)
        fakes.write_bingx_store(root, candles[24:])  # a rebuild while the sections are computed
        return result

    monkeypatch.setattr(mm, "amount_proxy", amount_then_rebuild)
    payload = mm.build_report(root=root, directory=reports)
    bingx = payload["data"]["stores"][1]
    assert bingx["relpath"] == "candles/bingx/1h.csv" and bingx["rows_read"] == len(candles)
    assert bingx["csv_sha256"] == read_sha
    assert read_meta(StoreId("bingx", "1h"), root)["csv_sha256"] != read_sha
    monkeypatch.setattr(mm, "amount_proxy", real_amount)

    real_read = mm.read_study_frame

    def read_then_rebuild(store_id, **kwargs):
        frame = real_read(store_id, **kwargs)
        if store_id.source == "bingx":
            fakes.write_bingx_store(root, candles[48:])  # replaced between the read and its meta
        return frame

    monkeypatch.setattr(mm, "read_study_frame", read_then_rebuild)
    with pytest.raises(mm.MismatchError, match="rebuilt while the study read it"):
        mm.prepare_study(root=root, directory=reports)
    monkeypatch.setattr(mm, "read_study_frame", real_read)
    # a data run that writes the BingX stores holds their job lock: the study read is refused
    with exclusive_lock(root / "raw" / "bingx" / ".lock"), pytest.raises(StoreBusyError):
        mm.prepare_study(root=root, directory=reports)


def test_prepare_study_computes_every_section_without_kronos_and_writes_nothing(tmp_path, fast):
    root, reports = tmp_path / "data", tmp_path / "reports"
    write_sources(root)
    study = mm.prepare_study(root=root, directory=reports)
    assert set(study.sections) == {"alignment", "returns", "prices", "volume", "amount"}
    assert study.sections["alignment"]["lag0_is_largest"] is True
    assert set(study.series_frames) == {"hours", "amount", "amount_windows"}
    assert [store["relpath"] for store in study.stores] == [
        "candles/binance/1h.csv",
        "candles/bingx/1h.csv",
    ]
    assert sorted(p.name for p in reports.iterdir()) == ["mismatch.config.json"]
    assert not (root / "studies").exists()
    payload = mm.write_study(study, directory=reports)
    assert payload["sections"]["kronos"]["status"] == "not_run"
    assert (reports / "mismatch.json").is_file() and (root / "studies").is_dir()
    with pytest.raises(TypeError):
        mm.write_study({"sections": {}}, directory=reports)


def test_build_report_fails_on_misaligned_stores_and_changed_configs(tmp_path, fast, monkeypatch):
    root, reports = tmp_path / "data", tmp_path / "reports"
    frame, candles = sources(24 * 7 * 3, seed=21)
    fakes.write_binance_store(root, frame)
    fakes.write_bingx_store(root, [replace(c, open_ms=c.open_ms + HOUR_MS) for c in candles])
    with pytest.raises(mm.AlignmentError):
        mm.build_report(root=root, directory=reports)
    assert not (reports / "mismatch.json").exists()
    assert (reports / "mismatch.config.json").exists()  # recorded before the run
    (reports / "mismatch.config.json").write_bytes(canonical_json({"study": "mismatch", "x": 1}))

    def no_read(*args, **kwargs):
        raise AssertionError("candles read before the config check")

    monkeypatch.setattr(mm, "read_study_frame", no_read)
    with pytest.raises(ConfigChangedError):
        mm.build_report(root=root, directory=reports)


def test_sealed_decision_hours_are_refused():
    frame, candles = sources(1200, start_ms=SEALED_START_MS - 600 * HOUR_MS, seed=2)
    bingx = bingx_frame(candles)
    assert (frame["open_ms"] >= SEALED_START_MS).any()
    last_open = datetime(2026, 7, 31, 23, tzinfo=UTC)  # the last hour before the seal
    result = mm.amount_proxy(frame, sample_hours=[last_open], bingx_1h=bingx)
    assert result["window_series"]["decision_ms"].tolist() == [to_ms(last_open)]
    assert result["samples"]["hours"]["binance_hours"] == 600  # e_t stops at the sealed start
    assert result["series"]["open_ms"].max() == SEALED_START_MS - HOUR_MS
    for hour in (SEALED_START, SEALED_START + timedelta(hours=5)):
        with pytest.raises(SealedPartitionError):
            mm.amount_proxy(frame, sample_hours=[last_open, hour], bingx_1h=bingx)
        with pytest.raises(SealedPartitionError):
            mm.kronos_tasks([hour])
    assert StoreId("bingx", "1h").relpath == "candles/bingx/1h.csv"
