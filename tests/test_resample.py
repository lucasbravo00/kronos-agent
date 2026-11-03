"""Exact resampling of 1m candles and the comparison with Binance's own 1h candles.

Every expected value here is written out by hand or computed with ``Decimal`` on the strings, so a
float implementation, a bucket shifted by one minute or an inclusive bucket end fails.
"""

from __future__ import annotations

import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes

from kronos_agent.data.partition import SEALED_START_MS
from kronos_agent.data.resample import (
    ROUNDING_COLUMNS,
    TARGET_INTERVALS,
    ResampleError,
    compare_with_official,
    decimal_sum,
    decimals,
    half_unit,
    incomplete_buckets,
    quote_rounding,
    resample_exact,
    resample_many,
)
from kronos_agent.data.store import BINANCE_COLUMNS, StoreId, validate_frame

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
T0 = fakes.utc_ms(2021, 3, 1)
BASE_ROW = {
    "open": "100.0",
    "high": "101.0",
    "low": "99.0",
    "close": "100.5",
    "volume": "1.000",
    "quote_volume": "100.00000",
    "count": "10",
    "taker_buy_volume": "0.500",
    "taker_buy_quote_volume": "50.00000",
}


def minutes(
    start_ms: int,
    count: int,
    changes: dict[int, dict[str, str]] | None = None,
    *,
    skip: set[int] = frozenset(),
) -> pd.DataFrame:
    """``count`` one-minute rows from ``start_ms`` with :data:`BASE_ROW` strings.

    ``changes`` maps a row index to the fields it replaces; ``skip`` leaves rows out (missing
    minutes). Row ``i`` opens at ``start_ms + i`` minutes.
    """
    changes = changes or {}
    data: dict[str, list] = {name: [] for name in BINANCE_COLUMNS}
    for index in range(count):
        if index in skip:
            continue
        row = {**BASE_ROW, **changes.get(index, {})}
        data["open_ms"].append(start_ms + index * MINUTE_MS)
        for name in BINANCE_COLUMNS[1:]:
            data[name].append(row[name])
    frame = pd.DataFrame({name: pd.Series(data[name], dtype=object) for name in BINANCE_COLUMNS})
    frame["open_ms"] = frame["open_ms"].astype(np.int64)
    return frame


def column(frame: pd.DataFrame, name: str) -> list:
    return frame[name].tolist()


def with_rounding(frame_1m: pd.DataFrame) -> pd.DataFrame:
    """The resampled 1h frame followed by the rounding columns of its quote volumes."""
    hourly = resample_exact(frame_1m, "1h")
    rounding = quote_rounding(frame_1m, "1h")
    rounding = rounding[rounding["minutes"] == 60][["open_ms", *ROUNDING_COLUMNS]]
    return hourly.merge(rounding, on="open_ms", how="left", validate="one_to_one")


def replaced(frame: pd.DataFrame, row: int, **values: str) -> pd.DataFrame:
    out = frame.copy()
    for name, value in values.items():
        series = out[name].to_numpy(dtype=object).copy()
        series[row] = value
        out[name] = pd.Series(series, dtype=object)
    return out


# -- exact decimal sums ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["0.1", "0.2"], "0.3"),  # float: 0.30000000000000004
        (["1.10", "2.2"], "3.30"),  # the largest scale of the summands is kept
        (["0.000", "0.000"], "0.000"),  # a zero sum keeps its decimals
        (["0", "0.000"], "0.000"),
        (["0"], "0"),
        (["7225", "7171.10"], "14396.10"),
        (["0.00000001", "0.00000002"], "0.00000003"),  # never "3E-8"
        (["0.1"] * 60, "6.0"),
        # more than 28 significant digits: the default context would round both
        (["12345678901234567890.123456789", "0.000000001"], "12345678901234567890.123456790"),
        (["99999999999999999999999999999.9", "0.1"], "100000000000000000000000000000.0"),
        (["-1.5", "1.5"], "0.0"),
    ],
)
def test_decimal_sum_is_exact(values, expected):
    assert decimal_sum(values) == expected


def test_decimal_sum_raises_instead_of_rounding():
    with pytest.raises(ResampleError, match="60 significant digits"):
        decimal_sum(["1" + "0" * 60])  # 61 digits: rounding would only drop a zero
    with pytest.raises(ResampleError, match="60 significant digits"):
        decimal_sum(["1" + "0" * 40, "0." + "0" * 20 + "1"])
    assert decimal_sum(["1" + "0" * 59]) == "1" + "0" * 59  # 60 digits fit


@pytest.mark.parametrize("bad", ["1e5", " 5", "5 ", "+5", "nan", "Infinity", "", ".5", "5.", 5])
def test_decimal_sum_refuses_anything_but_plain_decimal_strings(bad):
    with pytest.raises(ResampleError):
        decimal_sum(["1.0", bad])


def test_decimal_sum_refuses_empty_input_and_a_bare_string():
    with pytest.raises(ResampleError, match="empty"):
        decimal_sum([])
    with pytest.raises(TypeError):
        decimal_sum("123")


def test_decimals_and_half_unit():
    assert [decimals(text) for text in ("7225", "7171.10", "0.00001", "-3.5")] == [0, 2, 5, 1]
    assert half_unit("7225") == Decimal("0.5")
    assert half_unit("7171.10") == Decimal("0.005")
    assert half_unit("1.23456") == Decimal("0.000005")


# -- buckets ---------------------------------------------------------------------------------------


def test_five_minute_buckets_take_first_open_last_close_and_exact_sums():
    changes = {
        0: {"open": "100.0", "close": "100.2", "volume": "0.1", "count": "1"},
        1: {"open": "100.2", "close": "100.4", "volume": "0.2", "count": "2"},
        2: {"open": "100.4", "close": "100.6", "volume": "0.30", "count": "3"},
        3: {"open": "100.6", "close": "100.8", "volume": "1", "count": "4"},
        4: {"open": "100.8", "close": "101.0", "volume": "2.000", "count": "5"},
        5: {"open": "200.0", "close": "200.5"},
    }
    frame = minutes(T0, 10, changes)
    result = resample_exact(frame, "5m")
    assert list(result.columns) == list(BINANCE_COLUMNS)
    assert result["open_ms"].tolist() == [T0, T0 + 5 * MINUTE_MS]
    assert result["open_ms"].dtype == np.int64
    assert column(result, "open") == ["100.0", "200.0"]
    assert column(result, "close") == ["101.0", "100.5"]
    assert column(result, "volume") == ["3.600", "5.000"]
    assert column(result, "count") == ["15", "50"]
    assert column(result, "quote_volume") == ["500.00000", "500.00000"]
    assert column(result, "taker_buy_volume") == ["2.500", "2.500"]
    assert all(isinstance(value, str) for value in column(result, "volume"))
    validate_frame(result, StoreId("binance", "5m"))


def test_buckets_are_half_open_and_labelled_by_their_open():
    # minutes 00:01 .. 00:10 (00:00 missing): [00:00, 00:05) holds 4 minutes, [00:05, 00:10)
    # all 5, [00:10, 00:15) only 1. An implementation with buckets (t, t + 5 min] would make a
    # complete bucket of 00:01 .. 00:05.
    frame = minutes(T0, 11, skip={0})
    result = resample_exact(frame, "5m")
    assert result["open_ms"].tolist() == [T0 + 5 * MINUTE_MS]
    assert incomplete_buckets(frame, "5m") == [(T0, 4), (T0 + 10 * MINUTE_MS, 1)]


def test_a_missing_minute_drops_its_bucket_for_every_target_and_is_listed():
    frame = minutes(T0, 120, skip={67})  # 01:07 missing
    results = resample_many(frame)
    assert set(results) == set(TARGET_INTERVALS)
    five, five_missing = results["5m"]
    assert len(five) == 23 and T0 + 65 * MINUTE_MS not in five["open_ms"].tolist()
    assert five_missing == [(T0 + 65 * MINUTE_MS, 4)]
    quarter, quarter_missing = results["15m"]
    assert len(quarter) == 7 and quarter_missing == [(T0 + 60 * MINUTE_MS, 14)]
    hour, hour_missing = results["1h"]
    assert hour["open_ms"].tolist() == [T0]
    assert hour_missing == [(T0 + HOUR_MS, 59)]
    for target in TARGET_INTERVALS:
        assert results[target][0].equals(resample_exact(frame, target))
        assert results[target][1] == incomplete_buckets(frame, target)


def test_high_and_low_compare_as_decimals_not_floats():
    # equal as floats, different as decimals: the decimal maximum / minimum must win
    assert float("100.09999999999999999") == float("100.1")
    assert float("98.00000000000000001") == float("98.0")
    changes = {
        0: {"high": "100.09999999999999999", "low": "98.00000000000000001"},
        1: {"high": "100.1", "low": "98.0"},
        2: {"high": "99", "low": "99"},
        3: {"high": "99", "low": "99"},
        4: {"high": "99", "low": "99"},
    }
    result = resample_exact(minutes(T0, 5, changes), "5m")
    assert column(result, "high") == ["100.1"]
    assert column(result, "low") == ["98.0"]


@pytest.mark.parametrize(
    ("highs", "expected_high", "lows", "expected_low"),
    [
        (["7225", "7225.00"], "7225", ["7100.0", "7100"], "7100.0"),
        (["7225.00", "7225"], "7225.00", ["7100", "7100.0"], "7100"),
    ],
)
def test_ties_keep_the_first_occurrence_string(highs, expected_high, lows, expected_low):
    changes = {
        index: {
            "open": "7150",
            "close": "7160",
            "high": highs[index] if index < 2 else "7200",
            "low": lows[index] if index < 2 else "7150",
        }
        for index in range(5)
    }
    result = resample_exact(minutes(T0, 5, changes), "5m")
    assert column(result, "high") == [expected_high]
    assert column(result, "low") == [expected_low]


def test_resampling_mixed_price_formats_matches_a_decimal_reference():
    frame = fakes.binance_frame(T0, 240, interval="1m", seed=7)  # "7225" and "7171.10" mixed
    for target, size in (("5m", 5), ("15m", 15), ("1h", 60)):
        result = resample_exact(frame, target)
        assert len(result) == 240 // size
        for bucket, row in enumerate(result.itertuples(index=False)):
            part = frame.iloc[bucket * size : (bucket + 1) * size]
            highs = part["high"].tolist()
            lows = part["low"].tolist()
            expected_high = highs[[Decimal(v) for v in highs].index(max(map(Decimal, highs)))]
            expected_low = lows[[Decimal(v) for v in lows].index(min(map(Decimal, lows)))]
            assert row.open_ms == int(part["open_ms"].iloc[0])
            assert row.open == part["open"].iloc[0]
            assert row.close == part["close"].iloc[-1]
            assert row.high == expected_high and row.low == expected_low
            for name in ("volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"):
                total = sum((Decimal(v) for v in part[name]), Decimal(0))
                assert getattr(row, name) == format(total, "f")
            assert row.count == str(sum(int(v) for v in part["count"]))


def test_zero_volume_buckets_keep_their_decimals():
    zero = {"volume": "0.000", "taker_buy_volume": "0.000", "count": "0"}
    result = resample_exact(minutes(T0, 5, dict.fromkeys(range(5), zero)), "5m")
    assert column(result, "volume") == ["0.000"]
    assert column(result, "taker_buy_volume") == ["0.000"]
    assert column(result, "count") == ["0"]


def test_empty_frame_and_bad_input():
    empty = minutes(T0, 0)
    assert resample_exact(empty, "1h").empty
    assert list(resample_exact(empty, "1h").columns) == list(BINANCE_COLUMNS)
    assert incomplete_buckets(empty, "15m") == []
    frame = minutes(T0, 5)
    for target in ("1m", "4h", "1d"):
        with pytest.raises(ValueError, match="target"):
            resample_exact(frame, target)
    with pytest.raises(ResampleError):
        resample_exact(frame.assign(volume=1.0), "5m")  # floats are not the store's strings
    with pytest.raises(ResampleError):
        resample_exact(frame.assign(open_ms=frame["open_ms"] + 1), "5m")  # misaligned
    with pytest.raises(ResampleError):
        resample_exact(replaced(frame, 2, volume="1e3"), "5m")
    with pytest.raises(ResampleError):
        resample_exact(frame[list(BINANCE_COLUMNS[:-1])], "5m")


# -- rounding bounds and the official comparison ---------------------------------------------------


def test_quote_rounding_sums_half_units_of_each_minute():
    changes = {0: {"quote_volume": "1.0", "taker_buy_quote_volume": "7"}}
    frame = minutes(T0, 61, changes)
    rounding = quote_rounding(frame, "1h")
    assert rounding["open_ms"].tolist() == [T0, T0 + HOUR_MS]
    assert rounding["minutes"].tolist() == [60, 1]
    # 59 x 0.000005 + 0.05 and 59 x 0.000005 + 0.5
    assert rounding["quote_volume_rounding"].tolist() == ["0.050295", "0.000005"]
    assert rounding["taker_buy_quote_volume_rounding"].tolist() == ["0.500295", "0.000005"]
    only_quotes = frame[["open_ms", "quote_volume", "taker_buy_quote_volume"]]
    assert quote_rounding(only_quotes, "1h").equals(rounding)
    with pytest.raises(ResampleError):
        quote_rounding(frame[["open_ms", "quote_volume"]], "1h")


def tolerance_frames(official_quote: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 59 minutes with two decimals and one with one decimal: rounding = 59 x 0.005 + 0.05 = 0.345;
    # the sum is "60.00"; an official value with two decimals adds 0.005: tolerance 0.350.
    changes = {i: {"quote_volume": "1.00"} for i in range(59)}
    changes[59] = {"quote_volume": "1.0"}
    frame = minutes(T0, 60, changes)
    resampled = with_rounding(frame)
    assert resampled["quote_volume"].tolist() == ["60.00"]
    official = replaced(resampled[list(BINANCE_COLUMNS)], 0, quote_volume=official_quote)
    return resampled, official


@pytest.mark.parametrize(
    ("official_quote", "mismatch"),
    [
        ("60.35", False),  # exactly at the tolerance: within (<=)
        ("59.65", False),
        ("60.36", True),
        ("59.64", True),
        ("60.3", False),
    ],
)
def test_quote_volume_tolerance_uses_every_summand_and_the_official_rounding(
    official_quote, mismatch
):
    # A tolerance from the sum's decimals only (60 x 0.005 + 0.005 = 0.305), one without the
    # official half unit (0.345) or a strict comparison would all flag 60.35.
    resampled, official = tolerance_frames(official_quote)
    result = compare_with_official(resampled, official, sealed_start_ms=SEALED_START_MS)
    stats = result["columns"]["quote_volume"]
    assert stats["mismatches"] == int(mismatch)
    assert result["tolerance_mismatches"] == int(mismatch)
    assert result["exact_mismatches"] == 0
    difference = abs(Decimal(official_quote) - Decimal("60.00"))
    assert stats["max_abs_difference"] == format(difference, "f")
    if mismatch:
        (sample,) = result["mismatch_samples"]
        assert sample["column"] == "quote_volume" and sample["tolerance"] == "0.350"
        assert sample["resampled"] == "60.00" and sample["official"] == official_quote


def test_exact_columns_compare_as_decimals():
    frame = minutes(T0, 120, {0: {"open": "7225"}, 60: {"open": "7300.0"}})
    resampled = with_rounding(frame)
    official = resampled[list(BINANCE_COLUMNS)].copy()
    official = replaced(official, 0, open="7225.00")  # equal as a decimal
    official = replaced(official, 1, open="7300.1", count="601")
    result = compare_with_official(resampled, official, sealed_start_ms=SEALED_START_MS)
    assert result["hours_compared"] == 2
    assert result["columns"]["open"]["mismatches"] == 1
    assert result["columns"]["open"]["max_abs_difference"] == "0.1"
    assert result["columns"]["open"]["max_abs_difference_open_ms"] == T0 + HOUR_MS
    assert result["columns"]["count"]["mismatches"] == 1
    assert result["exact_mismatches"] == 2
    columns = {sample["column"] for sample in result["mismatch_samples"]}
    assert columns == {"open", "count"}


def test_hours_on_one_side_only():
    frame = minutes(T0, 180)
    resampled = with_rounding(frame)
    official = resampled[list(BINANCE_COLUMNS)].iloc[[1, 2]].reset_index(drop=True)
    resampled = resampled.iloc[[0, 1]].reset_index(drop=True)
    result = compare_with_official(resampled, official, sealed_start_ms=SEALED_START_MS)
    assert result["hours_compared"] == 1
    assert result["only_resampled"] == [T0]
    assert result["only_official"] == [T0 + 2 * HOUR_MS]
    assert result["exact_mismatches"] == result["tolerance_mismatches"] == 0


def test_sealed_hours_give_counts_and_open_times_but_never_values():
    start = SEALED_START_MS - HOUR_MS  # one hour before the sealed start, one at it
    frame = minutes(start, 120)
    resampled = with_rounding(frame)
    official = resampled[list(BINANCE_COLUMNS)].copy()
    official = replaced(official, 0, close="12345.6789")  # before the sealed start
    official = replaced(official, 1, close="98765.4321", quote_volume="55555.55555")  # sealed
    result = compare_with_official(resampled, official, sealed_start_ms=SEALED_START_MS)
    text = json.dumps(result)
    assert "12345.6789" in text  # a pre-sealed mismatch is shown with its values
    for secret in ("98765.4321", "55555.55555", "98664.9321", "49555.55555"):
        assert secret not in text  # neither the sealed values nor their differences
    assert result["hours_compared_sealed"] == 1 and result["hours_compared_before_sealed"] == 1
    assert result["columns"]["close"]["mismatches_sealed"] == 1
    assert result["columns"]["close"]["mismatches_before_sealed"] == 1
    assert result["columns"]["close"]["max_abs_difference_open_ms"] == start
    assert result["columns"]["quote_volume"]["mismatches_sealed"] == 1
    assert result["columns"]["quote_volume"]["max_abs_difference"] == "0"
    assert {(item["open_ms"], item["column"]) for item in result["sealed_mismatches"]} == {
        (SEALED_START_MS, "close"),
        (SEALED_START_MS, "quote_volume"),
    }
    assert all(set(item) == {"open_ms", "hour", "column"} for item in result["sealed_mismatches"])


def test_compare_requires_the_rounding_columns_and_store_frames():
    frame = minutes(T0, 60)
    resampled = with_rounding(frame)
    official = resampled[list(BINANCE_COLUMNS)]
    with pytest.raises(ResampleError, match="rounding"):
        compare_with_official(official, official, sealed_start_ms=SEALED_START_MS)
    with pytest.raises(ResampleError):
        compare_with_official(resampled, official.assign(close=1.0), sealed_start_ms=0)
    with pytest.raises(ResampleError):
        compare_with_official(resampled, official, sealed_start_ms=1.5)
