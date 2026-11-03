"""Decision windows: one builder, exact bounds, store window == live window, live timing and pairs.

The live path runs through a real ``PublicMarketClient`` on the BingX-shaped mock transport of
``step2_fakes``: the exchange holds the unclosed candle at ``D`` and candles in the future, and
the window must ignore both.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes

from kronos_agent.data import partition as pt
from kronos_agent.data.partition import SEALED_START, SEALED_START_MS, SealedPartitionError
from kronos_agent.data.timeutil import to_ms
from kronos_agent.data.windows import (
    AMOUNT_POLICIES,
    LIVE_MAX_DELAY_MS,
    LIVE_MIN_DELAY_MS,
    MAX_FETCH_PAIRS,
    WINDOW_SCHEMA_VERSION,
    DecisionWindow,
    LiveMismatchError,
    LiveTimingError,
    StoreWindowSource,
    WindowError,
    WindowGapError,
    build_window,
    context_frame,
    inputs_sha256,
    live_decision_window,
    load_window,
    naive_timestamps,
    save_window,
    window_path,
)
from kronos_agent.exchange import public_market as pm
from kronos_agent.exchange.public_market import Candle, window_sha256
from kronos_agent.hashing import canonical_json, sha256_bytes
from kronos_agent.kronos.features import FEATURE_COLUMNS, add_amount_proxy

HOUR_MS = 3_600_000
CONTEXT = 512
HORIZON = 4
D = datetime(2026, 3, 2, 12, tzinfo=UTC)  # inside validation
D_MS = to_ms(D)
FIRST_MS = D_MS - 520 * HOUR_MS
# 17 significant digits: pandas' default parser reads it one ulp off
VECTOR = "465655238.20240235"
CONTEXT_OPENS = list(range(D_MS - CONTEXT * HOUR_MS, D_MS, HOUR_MS))


def make_candles() -> list[Candle]:
    """600 hourly candles: 520 before ``D``, the one opening at ``D`` and 79 later ones."""
    candles = fakes.synthetic_candles(FIRST_MS, 600, seed=7)
    candles[519] = replace(candles[519], volume=VECTOR)  # the candle opening at D - 1h
    return candles


CANDLES = make_candles()
BY_OPEN = {c.open_ms: c for c in CANDLES}


def assert_same_window(a: DecisionWindow, b: DecisionWindow) -> None:
    assert a.sha256 == b.sha256
    assert a.inputs_sha256 == b.inputs_sha256
    assert a.candles == b.candles
    assert list(a.context.columns) == list(b.context.columns)
    assert a.context.to_numpy().tobytes() == b.context.to_numpy().tobytes()  # bit for bit
    assert a.x_timestamps.equals(b.x_timestamps)
    assert a.y_timestamps.equals(b.y_timestamps)
    assert a.last_close == b.last_close


def live_setup(server_time_ms: int = D_MS + LIVE_MIN_DELAY_MS, **kwargs):
    fake = fakes.FakeBingX(server_time_ms, {"1h": CANDLES}, **kwargs)
    client, _clock = fakes.bingx_client(fake)
    return fake, client


@pytest.fixture
def bingx_root(tmp_path):
    fakes.write_bingx_store(tmp_path, CANDLES)
    return tmp_path


# -- building blocks -------------------------------------------------------------------------------


def test_naive_timestamps_are_tz_naive_nanoseconds():
    series = naive_timestamps([D_MS, D_MS + HOUR_MS])
    assert series.dtype == np.dtype("datetime64[ns]")
    assert isinstance(series.index, pd.RangeIndex)
    assert series.tolist() == [pd.Timestamp(2026, 3, 2, 12), pd.Timestamp(2026, 3, 2, 13)]


def test_build_window_keeps_exactly_the_context_hours():
    window = build_window(CANDLES, D, source="bingx", horizon=HORIZON)
    assert [c.open_ms for c in window.candles] == CONTEXT_OPENS
    assert window.candles[0].open_ms == D_MS - CONTEXT * HOUR_MS
    assert window.candles[-1].open_ms == D_MS - HOUR_MS
    assert window.candles == tuple(BY_OPEN[open_ms] for open_ms in CONTEXT_OPENS)
    assert window.sha256 == window_sha256([BY_OPEN[open_ms] for open_ms in CONTEXT_OPENS])
    assert (window.source, window.symbol, window.decision_hour) == ("bingx", "BTC-USDT", D)
    assert (window.max_context, window.horizon) == (CONTEXT, HORIZON)
    assert window.amount_policy == "volume_x_mean_ohlc"
    assert window.fetch_pairs is None and window.quote_volumes is None
    assert window.x_timestamps.iloc[0] == pd.Timestamp(
        D - CONTEXT * timedelta(hours=1)
    ).tz_localize(None)
    assert window.x_timestamps.iloc[-1] == pd.Timestamp(2026, 3, 2, 11)
    assert window.y_timestamps.tolist() == [pd.Timestamp(2026, 3, 2, 12 + k) for k in range(4)]
    assert window.context.shape == (CONTEXT, 6)
    assert tuple(window.context.columns) == FEATURE_COLUMNS
    assert (window.context.dtypes == np.float64).all()
    assert window.context["close"].tolist() == [float(BY_OPEN[t].close) for t in CONTEXT_OPENS]
    assert window.last_close == float(BY_OPEN[D_MS - HOUR_MS].close)
    shuffled = list(CANDLES)
    random.Random(3).shuffle(shuffled)
    assert_same_window(build_window(shuffled, D, source="bingx", horizon=HORIZON), window)


def test_the_unclosed_and_later_candles_never_change_the_window():
    reference = build_window(CANDLES, D, source="bingx", horizon=HORIZON)
    altered = [
        replace(c, close="1.0", volume="999999.9999") if c.open_ms >= D_MS else c for c in CANDLES
    ]
    extra = [*altered, Candle(D_MS + 1000 * HOUR_MS, "1.0", "2.0", "0.5", "1.5", "3.0")]
    assert_same_window(build_window(extra, D, source="bingx", horizon=HORIZON), reference)
    before = [c for c in CANDLES if c.open_ms >= D_MS - CONTEXT * HOUR_MS]
    assert_same_window(build_window(before, D, source="bingx", horizon=HORIZON), reference)


@pytest.mark.parametrize(
    "missing_open, refused",
    [
        (D_MS - CONTEXT * HOUR_MS, True),  # the first context candle
        (D_MS - HOUR_MS, True),  # the last context candle
        (D_MS - 200 * HOUR_MS, True),
        (D_MS - (CONTEXT + 1) * HOUR_MS, False),  # one before the window
        (D_MS, False),  # the unclosed candle is never needed
    ],
)
def test_build_window_refuses_exactly_the_missing_context_candles(missing_open, refused):
    candles = [c for c in CANDLES if c.open_ms != missing_open]
    if refused:
        with pytest.raises(WindowGapError, match="misses 1 of 512"):
            build_window(candles, D, source="bingx", horizon=HORIZON)
    else:
        build_window(candles, D, source="bingx", horizon=HORIZON)


def test_build_window_refuses_duplicates_and_non_plain_strings():
    duplicate = [*CANDLES, replace(BY_OPEN[D_MS - 7 * HOUR_MS], volume="1.0")]
    with pytest.raises(WindowError, match="two candles") as info:
        build_window(duplicate, D, source="bingx", horizon=HORIZON)
    assert not isinstance(info.value, WindowGapError)
    duplicate_outside = [*CANDLES, replace(BY_OPEN[D_MS], volume="1.0")]
    build_window(duplicate_outside, D, source="bingx", horizon=HORIZON)
    scientific = [
        replace(c, high="6.0e4") if c.open_ms == D_MS - 2 * HOUR_MS else c for c in CANDLES
    ]
    with pytest.raises(WindowError, match="plain decimal"):
        build_window(scientific, D, source="bingx", horizon=HORIZON)


def test_build_window_argument_checks():
    with pytest.raises(WindowError):
        build_window(CANDLES, D, source="demo", horizon=HORIZON)
    with pytest.raises(WindowError):
        build_window(CANDLES, D, source="bingx", horizon=0)
    with pytest.raises(WindowError):
        build_window(CANDLES, D, source="bingx", horizon=HORIZON, amount_policy="volume")
    with pytest.raises(WindowError, match="Binance"):
        build_window(
            CANDLES, D, source="bingx", horizon=HORIZON, amount_policy="binance_quote_volume"
        )
    with pytest.raises(WindowError, match="only in Binance stores"):
        build_window(CANDLES, D, source="bingx", horizon=HORIZON, quote_volumes={D_MS: "1.0"})
    with pytest.raises(ValueError):
        build_window(CANDLES, D + timedelta(minutes=30), source="bingx", horizon=HORIZON)
    window = build_window(CANDLES, D, source="bingx", horizon=8, max_context=100)
    assert len(window.candles) == 100 and len(window.y_timestamps) == 8


def test_context_frame_amount_policies():
    candles = [BY_OPEN[open_ms] for open_ms in CONTEXT_OPENS]
    proxy = context_frame(candles, "volume_x_mean_ohlc")
    base = proxy.drop(columns=["amount"])
    expected_amount = add_amount_proxy(base)["amount"].to_numpy()
    assert proxy["amount"].to_numpy().tobytes() == expected_amount.tobytes()
    assert proxy["volume"].tolist() == [float(c.volume) for c in candles]
    zeros = context_frame(candles, "zeros")
    assert (zeros["volume"] == 0.0).all() and (zeros["amount"] == 0.0).all()
    assert zeros["close"].equals(proxy["close"])
    quotes = [f"{index}.25" for index in range(len(candles))]
    quoted = context_frame(candles, "binance_quote_volume", quotes)
    assert quoted["amount"].tolist() == [float(q) for q in quotes]
    assert quoted["volume"].equals(proxy["volume"])
    with pytest.raises(WindowError):
        context_frame(candles, "binance_quote_volume")
    with pytest.raises(WindowError):
        context_frame(candles, "binance_quote_volume", quotes[:-1])
    with pytest.raises(WindowError):
        context_frame(candles, "binance_quote_volume", ["1e3", *quotes[1:]])
    # the other policies check quote volumes when given and never read them
    assert context_frame(candles, "zeros", quotes).equals(zeros)
    assert context_frame(candles, "volume_x_mean_ohlc", quotes).equals(proxy)
    with pytest.raises(WindowError):
        context_frame(candles, "zeros", quotes[:-1])
    with pytest.raises(WindowError):
        context_frame(candles, "volume_x_mean_ohlc", ["1e3", *quotes[1:]])
    assert set(AMOUNT_POLICIES) == {"volume_x_mean_ohlc", "binance_quote_volume", "zeros"}


def test_inputs_sha256_covers_the_policy_and_the_quote_volumes():
    candles = [BY_OPEN[open_ms] for open_ms in CONTEXT_OPENS[-3:]]
    rows = [[c.open_ms, c.open, c.high, c.low, c.close, c.volume] for c in candles]
    expected = sha256_bytes(canonical_json({"rows": rows, "amount_policy": "volume_x_mean_ohlc"}))
    assert inputs_sha256(candles, "volume_x_mean_ohlc") == expected
    assert inputs_sha256(candles, "zeros") != expected
    quotes = ["10.5", "11.5", "12.5"]
    quoted = inputs_sha256(candles, "binance_quote_volume", quotes)
    assert quoted == sha256_bytes(
        canonical_json(
            {"rows": rows, "amount_policy": "binance_quote_volume", "quote_volumes": quotes}
        )
    )
    assert inputs_sha256(candles, "binance_quote_volume", ["10.5", "11.5", "12.6"]) != quoted
    # quote volumes the policy does not read leave its hash unchanged
    assert inputs_sha256(candles, "volume_x_mean_ohlc", quotes) == expected
    window = build_window(CANDLES, D, source="bingx", horizon=HORIZON)
    assert window.inputs_sha256 != window.sha256


# -- the store path --------------------------------------------------------------------------------


def test_store_window_equals_the_built_window(bingx_root):
    source = StoreWindowSource("bingx", root=bingx_root)
    assert source.first_open_ms == FIRST_MS and source.last_open_ms == FIRST_MS + 599 * HOUR_MS
    assert source.meta["rows"] == 600
    window = source.window(D, horizon=HORIZON)
    assert_same_window(window, build_window(CANDLES, D, source="bingx", horizon=HORIZON))
    assert [c.open_ms for c in source.candles_between(D_MS - 2 * HOUR_MS, D_MS)] == [
        D_MS - 2 * HOUR_MS,
        D_MS - HOUR_MS,
    ]
    assert source.candles_between(D_MS, D_MS) == []
    with pytest.raises(WindowError, match="quote volume"):
        source.quote_volumes_between(FIRST_MS, D_MS)
    with pytest.raises(WindowGapError):  # the context starts before the store
        source.window(D - timedelta(hours=10), horizon=HORIZON)


def test_the_seventeen_digit_vector_reaches_the_context_exactly(bingx_root):
    window = StoreWindowSource("bingx", root=bingx_root).window(D, horizon=HORIZON)
    assert window.candles[-1].volume == VECTOR
    volume = window.context["volume"].to_numpy()[-1]
    assert volume == float(VECTOR)
    assert np.float64(volume).view(np.int64) == np.float64(float(VECTOR)).view(np.int64)


def test_store_window_source_respects_its_loaded_range(bingx_root):
    narrow = StoreWindowSource("bingx", start_ms=D_MS - 100 * HOUR_MS, root=bingx_root)
    with pytest.raises(WindowError, match="outside the loaded range"):
        narrow.window(D, horizon=HORIZON)
    assert len(narrow.window(D, horizon=HORIZON, max_context=100).candles) == 100
    with pytest.raises(WindowError):
        StoreWindowSource("bingx", start_ms=0, end_ms=FIRST_MS, root=bingx_root)  # empty range


def test_store_window_source_never_loads_sealed_candles(tmp_path, monkeypatch):
    first = SEALED_START_MS - 600 * HOUR_MS
    fakes.write_bingx_store(tmp_path, fakes.synthetic_candles(first, 650, seed=2))
    source = StoreWindowSource("bingx", root=tmp_path)
    assert source.last_open_ms == SEALED_START_MS - HOUR_MS
    assert source.candles_between(SEALED_START_MS, SEALED_START_MS + 10 * HOUR_MS) == []
    last_hour = SEALED_START - timedelta(hours=1)
    assert (
        source.window(last_hour, horizon=HORIZON).candles[-1].open_ms
        == SEALED_START_MS - 2 * HOUR_MS
    )
    with pytest.raises(SealedPartitionError):
        source.window(SEALED_START, horizon=HORIZON)
    with pytest.raises(SealedPartitionError):
        StoreWindowSource("bingx", end_ms=SEALED_START_MS + HOUR_MS, root=tmp_path)
    with pytest.raises(SealedPartitionError):  # refused before the (absent) store is opened
        StoreWindowSource("bingx", end_ms=SEALED_START_MS + HOUR_MS, root=tmp_path / "absent")
    with pytest.raises(ValueError):
        StoreWindowSource("bingx", start_ms=float(first), root=tmp_path)
    monkeypatch.setattr(pt, "_verify_order", lambda order_id: None)
    access = pt.sealed_access("order-unit")
    opened = StoreWindowSource("bingx", sealed_access=access, root=tmp_path)
    assert opened.last_open_ms == SEALED_START_MS + 49 * HOUR_MS
    assert (
        opened.window(SEALED_START, horizon=HORIZON).candles[-1].open_ms
        == SEALED_START_MS - HOUR_MS
    )


BINANCE_FRAME = fakes.binance_frame(FIRST_MS, 600, interval="1h", seed=4)


@pytest.fixture
def binance_root(tmp_path):
    fakes.write_binance_store(tmp_path, BINANCE_FRAME)
    return tmp_path


def test_binance_window_with_the_quote_volume_policy(binance_root):
    source = StoreWindowSource("binance", root=binance_root)
    quoted = source.window(D, horizon=HORIZON, amount_policy="binance_quote_volume")
    proxy = source.window(D, horizon=HORIZON)
    frame = BINANCE_FRAME
    rows = frame[(frame["open_ms"] >= D_MS - CONTEXT * HOUR_MS) & (frame["open_ms"] < D_MS)]
    assert quoted.symbol == "BTCUSDT"
    assert quoted.quote_volumes == tuple(rows["quote_volume"])
    assert quoted.context["amount"].tolist() == [float(q) for q in rows["quote_volume"]]
    assert quoted.sha256 == proxy.sha256  # the same candles
    assert quoted.inputs_sha256 != proxy.inputs_sha256  # not the same model inputs
    assert source.quote_volumes_between(D_MS - HOUR_MS, D_MS) == {
        D_MS - HOUR_MS: rows["quote_volume"].iloc[-1]
    }


@pytest.mark.parametrize("policy", ["volume_x_mean_ohlc", "zeros"])
def test_a_binance_store_window_carries_its_quote_volumes_under_every_policy(binance_root, policy):
    """The quote volumes are part of what a Binance store holds (a saved window records them);
    a policy that does not read them gets the window, hashes and floats it had without them."""
    window = StoreWindowSource("binance", root=binance_root).window(
        D, horizon=HORIZON, amount_policy=policy
    )
    frame = BINANCE_FRAME
    rows = frame[(frame["open_ms"] >= D_MS - CONTEXT * HOUR_MS) & (frame["open_ms"] < D_MS)]
    assert window.quote_volumes == tuple(rows["quote_volume"])
    bare = build_window(
        fakes.frame_to_candles(frame), D, source="binance", horizon=HORIZON, amount_policy=policy
    )
    assert bare.quote_volumes is None
    assert_same_window(window, bare)
    assert window.inputs_sha256 == inputs_sha256(window.candles, policy)


# -- the live path ---------------------------------------------------------------------------------


def test_live_window_equals_the_store_window(bingx_root):
    fake, client = live_setup()
    live = live_decision_window(client, D, horizon=HORIZON)
    store_window = StoreWindowSource("bingx", root=bingx_root).window(D, horizon=HORIZON)
    assert_same_window(live, store_window)
    assert live.fetch_pairs == 1
    assert live.context["volume"].to_numpy()[-1] == float(VECTOR)
    assert fake.paths() == [pm.SERVER_TIME_PATH, pm.KLINES_PATH, pm.KLINES_PATH]
    for call in fake.klines_calls:
        assert (call.interval, call.symbol, call.limit) == ("1h", "BTC-USDT", 1000)
        assert call.start_ms == D_MS - CONTEXT * HOUR_MS
        assert call.end_ms == D_MS - 1  # inclusive bound one millisecond before D


def test_live_window_ignores_the_unclosed_and_future_candles(bingx_root):
    fake, client = live_setup(D_MS + 10 * 60_000)
    fake.mutator = fakes.vary_each_call(D_MS)  # the forming candle changes on every call
    assert fake.served("1h", D_MS, D_MS + 100 * HOUR_MS)[0].open_ms == D_MS  # held, not closed
    assert len(fake.served("1h", None, D_MS + 100 * HOUR_MS, 1000)) == 521  # no future candle
    live = live_decision_window(client, D, horizon=HORIZON)
    assert live.fetch_pairs == 1
    assert_same_window(live, StoreWindowSource("bingx", root=bingx_root).window(D, horizon=HORIZON))


@pytest.mark.parametrize(
    "offset_ms, allowed",
    [
        (-1, False),
        (0, False),
        (LIVE_MIN_DELAY_MS - 1, False),  # before the 90 s settle delay
        (LIVE_MIN_DELAY_MS, True),
        (LIVE_MAX_DELAY_MS - 1, True),
        (LIVE_MAX_DELAY_MS, False),  # 15 minutes late: no trading
        (HOUR_MS, False),
    ],
)
def test_live_window_timing(offset_ms, allowed):
    fake, client = live_setup(D_MS + offset_ms)
    if allowed:
        assert live_decision_window(client, D, horizon=HORIZON).fetch_pairs == 1
    else:
        with pytest.raises(LiveTimingError):
            live_decision_window(client, D, horizon=HORIZON)
        assert fake.paths() == [pm.SERVER_TIME_PATH]  # nothing fetched


def test_live_window_repeats_a_differing_pair(bingx_root):
    fake, client = live_setup()
    fake.mutator = fakes.alter_on_calls({1}, D_MS - 5 * HOUR_MS, volume="1.0")
    live = live_decision_window(client, D, horizon=HORIZON)
    assert live.fetch_pairs == 2
    assert fake.paths() == [pm.SERVER_TIME_PATH, pm.KLINES_PATH, pm.KLINES_PATH] * 2
    assert_same_window(live, StoreWindowSource("bingx", root=bingx_root).window(D, horizon=HORIZON))


def test_live_window_gives_up_after_three_differing_pairs():
    fake, client = live_setup()
    fake.mutator = fakes.vary_each_call(D_MS - 3 * HOUR_MS)  # a closed candle that keeps changing
    with pytest.raises(LiveMismatchError, match="all 3 fetch pairs"):
        live_decision_window(client, D, horizon=HORIZON)
    assert MAX_FETCH_PAIRS == 3
    assert fake.paths() == [pm.SERVER_TIME_PATH, pm.KLINES_PATH, pm.KLINES_PATH] * 3


def test_live_window_stops_retrying_at_the_deadline():
    fake, client = live_setup(D_MS + LIVE_MAX_DELAY_MS - 1000, tick_ms=1000)
    fake.mutator = fakes.vary_each_call(D_MS - 3 * HOUR_MS)
    with pytest.raises(LiveMismatchError, match="15 min"):
        live_decision_window(client, D, horizon=HORIZON)
    assert fake.paths() == [
        pm.SERVER_TIME_PATH,
        pm.KLINES_PATH,
        pm.KLINES_PATH,
        pm.SERVER_TIME_PATH,
    ]


def test_live_window_gaps_are_window_gaps():
    _fake, client = live_setup(missing={"1h": {D_MS - 100 * HOUR_MS}})
    with pytest.raises(WindowGapError, match="gap"):
        live_decision_window(client, D, horizon=HORIZON)
    _fake, client = live_setup(history_start_ms={"1h": D_MS - 300 * HOUR_MS})
    with pytest.raises(WindowGapError, match="misses 212 of 512"):
        live_decision_window(client, D, horizon=HORIZON)


def test_live_window_refuses_another_symbol_and_the_quote_policy():
    fake, client = live_setup()
    with pytest.raises(WindowError):
        live_decision_window(client, D, horizon=HORIZON, symbol="ETH-USDT")
    with pytest.raises(WindowError):
        live_decision_window(client, D, horizon=HORIZON, amount_policy="binance_quote_volume")
    with pytest.raises(WindowError):
        live_decision_window(client, D, horizon=HORIZON, amount_policy="quote")
    with pytest.raises(WindowError):
        live_decision_window(client, D, horizon=0)
    assert fake.requests == []


# -- saved windows ---------------------------------------------------------------------------------


SAVED_KEYS = {
    "schema_version",
    "source",
    "symbol",
    "decision_hour",
    "decision_ms",
    "max_context",
    "sha256",
    "candles",
    "quote_volumes",
}


def test_save_and_load_a_window(tmp_path):
    fake, client = live_setup()
    fake.mutator = fakes.alter_on_calls({1}, D_MS - 5 * HOUR_MS, volume="1.0")
    window = live_decision_window(client, D, horizon=HORIZON)
    assert window.fetch_pairs == 2
    path = save_window(window, root=tmp_path)
    assert path == window_path("bingx", D, root=tmp_path)
    assert path == tmp_path / "windows" / "bingx" / "context-512" / "2026-03-02T12Z.json"
    data = path.read_bytes()
    document = json.loads(data)
    assert data == canonical_json(document) + b"\n"
    # the inputs only: neither the amount policy, inputs_sha256 nor the fetch pairs
    assert set(document) == SAVED_KEYS and document["schema_version"] == WINDOW_SCHEMA_VERSION == 2
    assert document["sha256"] == window.sha256 and document["quote_volumes"] is None
    assert document["decision_hour"] == "2026-03-02T12:00Z" and document["max_context"] == CONTEXT
    assert save_window(window, root=tmp_path) == path  # an identical window is kept
    assert path.read_bytes() == data
    candles, metadata = load_window(path)
    assert tuple(candles) == window.candles
    assert metadata["sha256"] == window.sha256 and "candles" not in metadata
    # the same candles under another policy are the same saved window
    zeros = build_window(CANDLES, D, source="bingx", horizon=HORIZON, amount_policy="zeros")
    assert save_window(zeros, root=tmp_path) == path
    assert path.read_bytes() == data
    # other candles for the same hour are a different window: refused, never overwritten
    altered = [replace(c, close="1.0") if c.open_ms == D_MS - 7 * HOUR_MS else c for c in CANDLES]
    other = build_window(altered, D, source="bingx", horizon=HORIZON)
    with pytest.raises(WindowError, match="never overwritten"):
        save_window(other, root=tmp_path)
    assert path.read_bytes() == data
    # another context length is another window of the hour, beside the first
    short = build_window(CANDLES, D, source="bingx", horizon=HORIZON, max_context=100)
    assert save_window(short, root=tmp_path) == path.parent.parent / "context-100" / path.name
    document["candles"][7][4] = "1.0"
    path.write_bytes(canonical_json(document) + b"\n")
    with pytest.raises(WindowError, match="sha256"):
        load_window(path)


def test_a_live_retry_with_other_fetch_pairs_keeps_the_saved_window(tmp_path):
    """A step-8 retry of the same decision whose fetches matched at once (one pair) after a first
    attempt needed two: the same inputs, so the saved window is kept, not refused."""
    fake, client = live_setup()
    fake.mutator = fakes.alter_on_calls({1}, D_MS - 5 * HOUR_MS, volume="1.0")
    first = live_decision_window(client, D, horizon=HORIZON)
    _fake, retry_client = live_setup()
    retry = live_decision_window(retry_client, D, horizon=HORIZON)
    assert (first.fetch_pairs, retry.fetch_pairs) == (2, 1)
    assert (first.sha256, first.inputs_sha256) == (retry.sha256, retry.inputs_sha256)
    path = save_window(first, root=tmp_path)
    data = path.read_bytes()
    assert save_window(retry, root=tmp_path) == path
    assert path.read_bytes() == data


def test_one_saved_binance_window_serves_every_amount_policy(binance_root, tmp_path):
    source = StoreWindowSource("binance", root=binance_root)
    windows = {
        policy: source.window(D, horizon=HORIZON, amount_policy=policy)
        for policy in AMOUNT_POLICIES
    }
    assert len({window.inputs_sha256 for window in windows.values()}) == 3
    target = tmp_path / "saved"
    path = save_window(windows["volume_x_mean_ohlc"], root=target)
    data = path.read_bytes()
    for window in windows.values():
        assert save_window(window, root=target) == path
    assert path.read_bytes() == data
    candles, metadata = load_window(path)
    assert tuple(candles) == windows["zeros"].candles
    quotes = metadata["quote_volumes"]
    assert tuple(quotes) == windows["binance_quote_volume"].quote_volumes
    # the saved inputs give back every policy's inputs_sha256
    for policy, window in windows.items():
        read = quotes if policy == "binance_quote_volume" else None
        assert inputs_sha256(candles, policy, read) == window.inputs_sha256
    # a different quote volume for the same hour is a different window
    changed = dict(source.quote_volumes_between(D_MS - CONTEXT * HOUR_MS, D_MS))
    changed[D_MS - HOUR_MS] = "1.5"
    other = build_window(
        list(windows["zeros"].candles), D, source="binance", horizon=HORIZON, quote_volumes=changed
    )
    with pytest.raises(WindowError, match="never overwritten"):
        save_window(other, root=target)
    # a Binance window built without its quote volumes is never saved
    bare = build_window(list(windows["zeros"].candles), D, source="binance", horizon=HORIZON)
    with pytest.raises(WindowError, match="quote volumes"):
        save_window(bare, root=tmp_path / "bare")
    assert not (tmp_path / "bare").exists()
    # a tampered quote volume or a file of the earlier schema is not a saved window
    document = json.loads(data)
    document["quote_volumes"][3] = "1e3"
    path.write_bytes(canonical_json(document) + b"\n")
    with pytest.raises(WindowError, match="quote volumes"):
        load_window(path)
    document = json.loads(data)
    path.write_bytes(canonical_json({**document, "quote_volumes": None}) + b"\n")
    with pytest.raises(WindowError, match="belong to Binance windows"):
        load_window(path)
    path.write_bytes(canonical_json({**document, "schema_version": 1}) + b"\n")
    with pytest.raises(WindowError, match="not a saved decision window"):
        load_window(path)


# -- the BingX-shaped fake itself (later stages rely on these behaviours) --------------------------


def raw_klines(fake: fakes.FakeBingX, **params) -> list[dict]:
    url = f"https://{pm.PUBLIC_HOST}{pm.KLINES_PATH}"
    response = fake(httpx.Request("GET", url, params={"symbol": "BTC-USDT", **params}))
    payload = response.json()
    assert payload["code"] == 0, payload
    return payload["data"]


def test_the_bingx_fake_reproduces_the_verified_klines_behaviour():
    fake = fakes.FakeBingX(D_MS + 10 * 60_000, {"1h": CANDLES})
    rows = raw_klines(fake, interval="1h", startTime=D_MS - 3 * HOUR_MS, endTime=D_MS + 5 * HOUR_MS)
    assert [row["time"] for row in rows] == [D_MS - k * HOUR_MS for k in range(4)]  # newest first
    assert set(rows[0]) == {"open", "close", "high", "low", "volume", "time"}
    assert isinstance(rows[0]["close"], str) and isinstance(rows[0]["time"], int)
    # an endTime exactly at a candle's open excludes that candle, boundary - 1 ms keeps it (D-041)
    exact = raw_klines(fake, interval="1h", startTime=D_MS - 2 * HOUR_MS, endTime=D_MS - HOUR_MS)
    assert [row["time"] for row in exact] == [D_MS - 2 * HOUR_MS]
    seam = raw_klines(fake, interval="1h", startTime=D_MS - 2 * HOUR_MS, endTime=D_MS - 1)
    assert [row["time"] for row in seam] == [D_MS - HOUR_MS, D_MS - 2 * HOUR_MS]
    newest = raw_klines(fake, interval="1h", startTime=FIRST_MS, endTime=D_MS, limit=3)
    assert [row["time"] for row in newest] == [D_MS - k * HOUR_MS for k in (1, 2, 3)]
    assert raw_klines(fake, interval="1h", startTime=FIRST_MS) == []  # startTime alone
    assert raw_klines(fake, interval="1h", startTime=0, endTime=FIRST_MS - 1) == []
    assert raw_klines(fake, interval="5m", startTime=0, endTime=D_MS) == []
    fake.history_start_ms["1h"] = D_MS - HOUR_MS
    fake.missing["1h"] = {D_MS - 5 * HOUR_MS}
    tail = raw_klines(fake, interval="1h", startTime=FIRST_MS, endTime=D_MS + 1)
    assert [row["time"] for row in tail] == [D_MS, D_MS - HOUR_MS]  # the forming candle first
    assert len(fake.klines_calls) == 8 and fake.klines_calls[3].limit == 3
    error = fake(httpx.Request("GET", f"https://{pm.PUBLIC_HOST}{pm.KLINES_PATH}?symbol=ETH-USDT"))
    assert error.json()["code"] != 0
    assert fake(httpx.Request("GET", f"https://{pm.PUBLIC_HOST}/openApi/other")).status_code == 404
    fake.advance(HOUR_MS)
    opened = raw_klines(fake, interval="1h", startTime=D_MS, endTime=D_MS + HOUR_MS + 1)
    assert opened[0]["time"] == D_MS + HOUR_MS  # served once the server time reaches its open
    exact = raw_klines(fake, interval="1h", startTime=D_MS, endTime=D_MS + HOUR_MS)
    assert exact[0]["time"] == D_MS
