"""BingX real-market backfill: grid, request bounds, settle rule, fetch pairs, archive, walk, merge.

Every exchange answer comes from a real ``PublicMarketClient`` on the BingX-shaped mock transport
of ``step2_fakes`` (newest first, ``startTime`` inclusive, an ``endTime`` exactly at a candle's
open excluding that candle as observed live in D-041, the forming candle served once it opened,
candles after the server time never served). The 1h grid of these tests: chunk ``K = 480`` opens
at 2024-10-04T00:00Z and holds 1000 hourly slots; the exchange's history starts 250 h into it,
and the first run's server time leaves 400 settled candles in chunk ``K + 3``.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import step2_fakes as fakes

from kronos_agent.data import bingx_history as bh
from kronos_agent.data.bingx_history import (
    CHUNK_CANDLES,
    HISTORY_SCHEMA_VERSION,
    INTERVALS,
    MAX_FETCH_PAIRS,
    RETENTION_DAYS,
    REVISION_SCHEMA_VERSION,
    SETTLE_MS,
    SIDECAR_SCHEMA_VERSION,
    STOP_AFTER_EMPTY_CHUNKS,
    BackfillError,
    ChunkFetch,
    backfill,
    chunk_bounds,
    chunk_index,
    chunk_paths,
    chunk_span_ms,
    fetch_chunk,
    history_relpath,
    merge_chunks,
    read_chunk,
    read_history,
    read_revisions,
    request_bounds,
    retention_warnings,
    revision_relpath,
    settled,
)
from kronos_agent.data.fsutil import LockBusyError, exclusive_lock
from kronos_agent.data.store import StoreId, read_meta, read_store
from kronos_agent.data.timeutil import utc_text
from kronos_agent.exchange import public_market as pm
from kronos_agent.exchange.public_market import BingXApiError, Candle, PublicMarketClient
from kronos_agent.hashing import sha256_bytes

HOUR_MS = 3_600_000
MINUTE_MS = 60_000
DAY_MS = 86_400_000
SPAN = CHUNK_CANDLES * HOUR_MS
K = 480
HEADER = b"open_ms,open,high,low,close,volume\n"
# the step-2 vector: 17 significant digits that pandas' default CSV parser reads one ulp off
VECTOR = "465655238.20240235"


def start_of(k: int) -> int:
    """Open time of the first slot of 1h chunk ``k``."""
    return k * SPAN


HISTORY_START = start_of(K) + 250 * HOUR_MS
SERVER = start_of(K + 3) + 400 * HOUR_MS + 30 * MINUTE_MS
END_OF_DATA = start_of(K + 6)
SPECIAL_OPEN = start_of(K + 1) + 321 * HOUR_MS


def make_candles() -> list[Candle]:
    candles = fakes.synthetic_candles(
        HISTORY_START, (END_OF_DATA - HISTORY_START) // HOUR_MS, seed=11
    )
    index = (SPECIAL_OPEN - HISTORY_START) // HOUR_MS
    special = candles[index]
    # strings a float round trip would rewrite: a high with 22 significant digits and a close
    # with a trailing zero (same values, so the OHLC relations still hold), plus the vector
    candles[index] = replace(
        special,
        high=special.high + "00000000000000001",
        close=special.close + "0",
        volume=VECTOR,
    )
    return candles


CANDLES = make_candles()
BY_OPEN = {c.open_ms: c for c in CANDLES}
# history from the start of chunk K - 1, for the empty-chunk walks
LONG_CANDLES = fakes.synthetic_candles(
    start_of(K - 1), (END_OF_DATA - start_of(K - 1)) // HOUR_MS, seed=12
)


def make_fake(
    server_time_ms: int = SERVER,
    *,
    candles: list[Candle] = CANDLES,
    missing: set[int] | None = None,
    tick_ms: int = 0,
) -> fakes.FakeBingX:
    return fakes.FakeBingX(
        server_time_ms,
        {"1h": candles},
        missing=None if missing is None else {"1h": missing},
        tick_ms=tick_ms,
    )


def client_for(fake: fakes.FakeBingX) -> PublicMarketClient:
    client, _ = fakes.bingx_client(fake)
    return client


def chunk_calls(fake: fakes.FakeBingX, since: int = 0) -> list[int]:
    """Chunk index of every klines request from the ``since``-th one on."""
    return [call.start_ms // SPAN for call in fake.klines_calls[since:]]


def pairs_of(*chunks: int) -> list[int]:
    return [k for k in chunks for _ in range(2)]


def archive_dir(root: Path, interval: str = "1h") -> Path:
    return root / "raw" / "bingx" / interval


def settled_candles(
    k: int,
    server_time_ms: int,
    candles: list[Candle] = CANDLES,
    missing: frozenset[int] | set[int] = frozenset(),
) -> list[Candle]:
    """The candles chunk ``k`` must hold at ``server_time_ms``."""
    return [
        c
        for c in candles
        if start_of(k) <= c.open_ms < start_of(k + 1)
        and c.open_ms + HOUR_MS + SETTLE_MS <= server_time_ms
        and c.open_ms not in missing
    ]


def file_bytes(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(archive_dir(root).iterdir())}


@pytest.fixture(scope="module")
def pristine(tmp_path_factory) -> Path:
    """The archive of a first run at ``SERVER`` (built once, copied by ``archive_root``)."""
    root = tmp_path_factory.mktemp("bingx-archive")
    backfill(client_for(make_fake()), "1h", root=root)
    return root


@pytest.fixture
def archive_root(pristine, tmp_path) -> Path:
    root = tmp_path / "data"
    shutil.copytree(pristine, root)
    return root


# -- grid, request bounds, settle rule ------------------------------------------------------------


@pytest.mark.parametrize(
    ("interval", "span"),
    [("1h", 3_600_000_000), ("15m", 900_000_000), ("5m", 300_000_000), ("1m", 60_000_000)],
)
def test_the_chunk_grid_is_epoch_aligned_and_half_open(interval, span):
    step = pm.interval_ms(interval)
    assert INTERVALS == ("1h", "15m", "5m", "1m")
    assert chunk_span_ms(interval) == span == CHUNK_CANDLES * step
    assert chunk_bounds(0, interval) == (0, span)
    assert chunk_bounds(K, interval) == (K * span, (K + 1) * span)
    assert chunk_index(K * span, interval) == K
    assert chunk_index((K + 1) * span - step, interval) == K  # the 1000th slot
    assert chunk_index((K + 1) * span - 1, interval) == K
    assert chunk_index((K + 1) * span, interval) == K + 1  # the end opens the next chunk


@pytest.mark.parametrize(
    "call",
    [
        lambda: chunk_span_ms("2h"),  # the client knows 2h; the archive does not
        lambda: chunk_span_ms("1H"),
        lambda: chunk_bounds(-1, "1h"),
        lambda: chunk_bounds(True, "1h"),
        lambda: chunk_bounds(1.0, "1h"),
        lambda: chunk_index(-1, "1h"),
        lambda: request_bounds(K, "1h", -5),
        lambda: request_bounds(K, "1h", float(SERVER)),
    ],
)
def test_grid_functions_refuse_bad_arguments(call):
    with pytest.raises(ValueError):
        call()


def test_a_closed_chunk_is_requested_up_to_one_millisecond_before_the_next_chunk():
    start, end = chunk_bounds(K, "1h")
    # not ``end`` (an exact boundary) and not ``end - 1h`` (the 1000th candle's own open)
    for server_time_ms in (SERVER, end, end + 1, end + SETTLE_MS):
        assert request_bounds(K, "1h", server_time_ms) == (start, end - 1)


@pytest.mark.parametrize(
    ("offset", "end_offset"),
    [
        (400 * HOUR_MS + 30 * MINUTE_MS, 400 * HOUR_MS + 30 * MINUTE_MS),  # the server time
        (400 * HOUR_MS, 400 * HOUR_MS - 1),  # on a candle boundary: one millisecond earlier
        (HOUR_MS, HOUR_MS - 1),
        (1, 1),
        (SPAN - 1, SPAN - 1),
    ],
)
def test_the_newest_chunk_is_requested_up_to_the_server_time(offset, end_offset):
    assert request_bounds(K, "1h", start_of(K) + offset) == (start_of(K), start_of(K) + end_offset)


def test_end_time_is_never_an_exact_boundary_nor_in_the_future():
    rng = random.Random(5)
    for interval in INTERVALS:
        step = pm.interval_ms(interval)
        span = chunk_span_ms(interval)
        for _ in range(400):
            index = rng.randrange(400, 600)
            start = index * span
            offset = rng.choice(
                [rng.randrange(1, 3 * span), rng.randrange(1, 1100) * step, span, span + SETTLE_MS]
            )
            first, last = request_bounds(index, interval, start + offset)
            assert first == start
            assert start <= last < start + span
            assert last <= start + offset
            assert last % step != 0


def test_a_chunk_that_has_not_started_has_nothing_to_request():
    for server_time_ms in (start_of(K), start_of(K) - 1):
        with pytest.raises(ValueError, match="nothing to request"):
            request_bounds(K, "1h", server_time_ms)


def test_a_candle_is_settled_ninety_seconds_after_its_close():
    assert SETTLE_MS == 90_000
    close = start_of(K) + HOUR_MS
    assert settled(start_of(K), HOUR_MS, close + 90_000)
    assert not settled(start_of(K), HOUR_MS, close + 89_999)
    assert not settled(start_of(K), HOUR_MS, close + 30_000)
    assert not settled(start_of(K), HOUR_MS, close)


# -- fetching one chunk ---------------------------------------------------------------------------


def test_a_closed_chunk_is_one_page_of_its_1000_candles_fetched_twice():
    fake = make_fake()
    fetch = fetch_chunk(client_for(fake), "1h", K + 1, server_time_ms=SERVER)
    assert isinstance(fetch, ChunkFetch)
    assert (fetch.pairs, fetch.requests, fetch.complete) == (1, 2, True)
    assert fetch.candles == settled_candles(K + 1, SERVER)
    assert [c.open_ms for c in fetch.candles] == list(
        range(start_of(K + 1), start_of(K + 2), HOUR_MS)
    )
    sent = [(c.symbol, c.interval, c.start_ms, c.end_ms, c.limit) for c in fake.klines_calls]
    assert sent == [("BTC-USDT", "1h", start_of(K + 1), start_of(K + 2) - 1, 1000)] * 2


def test_a_candle_closed_thirty_seconds_ago_is_dropped_and_its_chunk_stays_incomplete():
    last_open = start_of(K + 2) - HOUR_MS
    server_time_ms = last_open + HOUR_MS + 30_000
    fake = make_fake(server_time_ms)
    fetch = fetch_chunk(client_for(fake), "1h", K + 1, server_time_ms=server_time_ms)
    assert fetch.complete is False
    assert [c.open_ms for c in fetch.candles] == list(range(start_of(K + 1), last_open, HOUR_MS))
    assert fake.klines_calls[0].end_ms == start_of(K + 2) - 1  # the page did hold it


@pytest.mark.parametrize(("after_close", "kept"), [(89_999, False), (90_000, True)])
def test_the_settle_boundary_is_exact(after_close, kept):
    last_open = start_of(K + 2) - HOUR_MS
    server_time_ms = last_open + HOUR_MS + after_close
    fake = make_fake(server_time_ms)
    fetch = fetch_chunk(client_for(fake), "1h", K + 1, server_time_ms=server_time_ms)
    assert len(fetch.candles) == (1000 if kept else 999)
    assert (fetch.candles[-1].open_ms == last_open) is kept
    assert fetch.complete is kept


@pytest.mark.parametrize(("altered_calls", "pairs"), [((), 1), ((0,), 2), ((1,), 2), ((1, 3), 3)])
def test_a_differing_pair_is_fetched_again(altered_calls, pairs):
    fake = make_fake()
    fake.mutator = fakes.alter_on_calls(altered_calls, start_of(K + 1) + 500 * HOUR_MS, volume="1")
    fetch = fetch_chunk(client_for(fake), "1h", K + 1, server_time_ms=SERVER)
    assert (fetch.pairs, fetch.requests) == (pairs, 2 * pairs)
    assert fetch.candles == settled_candles(K + 1, SERVER)  # the unaltered values


def test_a_chunk_whose_three_pairs_differ_is_refused_naming_it():
    assert MAX_FETCH_PAIRS == 3
    fake = make_fake()
    fake.mutator = fakes.alter_on_calls((1, 3, 5), start_of(K + 1) + 500 * HOUR_MS, volume="1")
    label = re.escape(f"1h chunk {utc_text(start_of(K + 1))}: all 3 fetch pairs differed")
    with pytest.raises(BackfillError, match=label):
        fetch_chunk(client_for(fake), "1h", K + 1, server_time_ms=SERVER)
    assert len(fake.klines_calls) == 6


def test_unsettled_candles_are_dropped_before_the_two_fetches_are_compared():
    # the candle that closed 30 s ago and the forming one change on every request
    base = start_of(K + 3)
    server_time_ms = base + 400 * HOUR_MS + 30_000
    fake = make_fake(server_time_ms)
    fake.mutator = fakes.chain(
        fakes.vary_each_call(base + 399 * HOUR_MS), fakes.vary_each_call(base + 400 * HOUR_MS)
    )
    fetch = fetch_chunk(client_for(fake), "1h", K + 3, server_time_ms=server_time_ms)
    assert (fetch.pairs, fetch.complete) == (1, False)
    assert [c.open_ms for c in fetch.candles] == list(range(base, base + 399 * HOUR_MS, HOUR_MS))
    assert all(call.end_ms == server_time_ms for call in fake.klines_calls)


def _added(open_ms: int) -> fakes.Mutator:
    def mutate(call: fakes.KlinesCall, rows: list[Candle]) -> list[Candle]:
        return [*rows, Candle(open_ms, "60000.0", "60001.0", "59999.0", "60000.5", "1.0")]

    return mutate


def _value(field: str, text: str) -> fakes.Mutator:
    def mutate(call: fakes.KlinesCall, rows: list[Candle]) -> list[Candle]:
        return [replace(rows[0], **{field: text}), *rows[1:]]

    return mutate


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (_added(start_of(K + 2)), "outside the request"),  # the next chunk's first candle
        (_added(start_of(K + 1) - HOUR_MS), "outside the request"),
        (_added(start_of(K + 1) + 30 * MINUTE_MS), "off the interval grid"),
        (lambda call, rows: [*rows, rows[7]], "two candles open"),
        (_value("volume", "1e-05"), "not a plain decimal"),
        (_value("close", "NaN"), "not a plain decimal"),
        (_value("high", "Infinity"), "not a plain decimal"),
        (_value("open", " 60000.5"), "not a plain decimal"),
        (_value("low", "59_999.0"), "not a plain decimal"),
        (_value("low", "+59999.0"), "not a plain decimal"),
        (_value("volume", ".5"), "not a plain decimal"),
    ],
)
def test_a_page_a_chunk_cannot_hold_is_refused(mutator, message):
    fake = make_fake()
    fake.mutator = mutator
    with pytest.raises(BackfillError, match=message):
        fetch_chunk(client_for(fake), "1h", K + 1, server_time_ms=SERVER)


# -- the first run --------------------------------------------------------------------------------


def test_the_first_run_walks_back_from_the_newest_settled_candle_to_three_empty_chunks(tmp_path):
    fake = make_fake(tick_ms=1_000)  # the exchange clock moves; the run's server time does not
    summary = backfill(client_for(fake), "1h", root=tmp_path)
    assert STOP_AFTER_EMPTY_CHUNKS == 3
    assert chunk_calls(fake) == pairs_of(K + 3, K + 2, K + 1, K, K - 1, K - 2, K - 3)
    assert len(fake.server_time_requests()) == 1
    ends = {call.start_ms: call.end_ms for call in fake.klines_calls}
    assert ends == {start_of(K + 3): SERVER} | {
        start_of(k): start_of(k + 1) - 1 for k in range(K - 3, K + 3)
    }
    assert all(call.limit == 1000 for call in fake.klines_calls)
    assert summary["stop_reason"] == "empty_chunks"
    assert summary["history_start_ms"] == HISTORY_START
    assert summary["history_start_utc"] == utc_text(HISTORY_START)
    assert summary["server_time_ms"] == SERVER
    assert summary["new_chunks"] == [start_of(k) for k in range(K, K + 4)]
    assert summary["empty_chunks"] == [start_of(k) for k in range(K - 3, K)]
    assert summary["gap_chunks"] == [] and summary["refreshed_chunks"] == []
    assert (summary["chunks_fetched"], summary["chunks_skipped"]) == (7, 0)
    assert (summary["klines_requests"], summary["mismatch_retries"]) == (14, 0)
    assert summary["new_candles"] == 750 + 1000 + 1000 + 400
    assert summary["archive"] == {
        "chunks": 4,
        "candles": 3150,
        "first_open_ms": HISTORY_START,
        "last_open_ms": start_of(K + 3) + 399 * HOUR_MS,
        "incomplete_chunks": [start_of(K + 3)],
    }
    json.dumps(summary, allow_nan=False)
    assert sorted(file_bytes(tmp_path)) == sorted(
        f"{start_of(k)}.{suffix}" for k in range(K, K + 4) for suffix in ("csv", "json")
    )
    for k in range(K, K + 4):
        candles, _ = read_chunk("1h", k, root=tmp_path)
        assert candles == settled_candles(k, SERVER)


def test_the_sidecar_is_the_commit_marker_of_its_chunk(archive_root):
    csv_path, json_path = chunk_paths("1h", K + 3, root=archive_root)
    data = csv_path.read_bytes()
    sidecar = json.loads(json_path.read_text())
    _, read_back = read_chunk("1h", K + 3, root=archive_root)
    assert read_back == sidecar
    assert data.startswith(HEADER) and data.endswith(b"\n")
    assert sidecar["csv_sha256"] == sha256_bytes(data) and sidecar["csv_size"] == len(data)
    assert sidecar["csv_relpath"] == f"raw/bingx/1h/{start_of(K + 3)}.csv"
    assert sidecar["request"] == {"startTime": start_of(K + 3), "endTime": SERVER, "limit": 1000}
    assert (sidecar["server_time_ms"], sidecar["settle_ms"]) == (SERVER, 90_000)
    assert (sidecar["complete"], sidecar["rows"], sidecar["settled_slots"]) == (False, 400, 400)
    assert (sidecar["requests"], sidecar["pairs"], sidecar["mismatch_retries"]) == (2, 1, 0)
    assert (sidecar["chunk_start_ms"], sidecar["chunk_end_ms"]) == chunk_bounds(K + 3, "1h")
    assert sidecar["first_open_ms"] == start_of(K + 3)
    assert sidecar["last_open_ms"] == start_of(K + 3) + 399 * HOUR_MS
    assert sidecar["gaps"] == []
    assert sidecar["schema_version"] == SIDECAR_SCHEMA_VERSION
    assert sidecar["archive_only"] == {"rows": 0, "ranges": []}
    assert set(sidecar["code"]) == {"commit", "dirty", "source_tree_sha256"}
    oldest = json.loads(chunk_paths("1h", K, root=archive_root)[1].read_text())
    assert (oldest["complete"], oldest["rows"], oldest["first_open_ms"]) == (
        True,
        750,
        HISTORY_START,
    )
    assert oldest["request"]["endTime"] == start_of(K + 1) - 1


def test_exchange_strings_are_archived_and_merged_verbatim(archive_root):
    special = BY_OPEN[SPECIAL_OPEN]
    assert special.volume == VECTOR
    # what a float round trip would write instead
    assert repr(float(special.high)) != special.high
    assert repr(float(special.close)) != special.close
    row = f"{SPECIAL_OPEN},{special.open},{special.high},{special.low},{special.close},{VECTOR}\n"
    assert row.encode() in chunk_paths("1h", K + 1, root=archive_root)[0].read_bytes()
    merge_chunks("1h", root=archive_root)
    frame = read_store(StoreId("bingx", "1h"), root=archive_root)
    stored = frame[frame["open_ms"] == SPECIAL_OPEN].iloc[0]
    assert (stored["high"], stored["close"], stored["volume"]) == (
        special.high,
        special.close,
        VECTOR,
    )


def test_missing_candles_are_recorded_gaps_never_filled(tmp_path):
    first_gap = start_of(K + 1) + 10 * HOUR_MS
    missing = {first_gap, first_gap + HOUR_MS, start_of(K + 2)}
    fake = make_fake(missing=missing)
    backfill(client_for(fake), "1h", root=tmp_path)
    candles, sidecar = read_chunk("1h", K + 1, root=tmp_path)
    assert candles == settled_candles(K + 1, SERVER, missing=missing)
    assert (sidecar["rows"], sidecar["complete"]) == (998, True)
    assert sidecar["gaps"] == [[first_gap, first_gap + 2 * HOUR_MS]]
    _, sidecar = read_chunk("1h", K + 2, root=tmp_path)
    # completeness is a matter of time, not of rows
    assert (sidecar["rows"], sidecar["complete"]) == (999, True)
    assert sidecar["first_open_ms"] == start_of(K + 2) + HOUR_MS
    meta = merge_chunks("1h", root=tmp_path)
    assert meta["gaps"] == [
        [first_gap, first_gap + 2 * HOUR_MS],
        [start_of(K + 2), start_of(K + 2) + HOUR_MS],
    ]
    assert meta["missing_candles"] == 3 and meta["rows"] == 3150 - 3


@pytest.mark.parametrize("empty", [1, 2, 3])
def test_empty_chunks_inside_the_history(tmp_path, empty):
    hidden = {c.open_ms for c in LONG_CANDLES if start_of(K) <= c.open_ms < start_of(K + empty)}
    fake = make_fake(candles=LONG_CANDLES, missing=hidden)
    summary = backfill(client_for(fake), "1h", root=tmp_path)
    holes = [start_of(K + i) for i in range(empty)]
    if empty < STOP_AFTER_EMPTY_CHUNKS:
        # a gap: the walk goes on, finds chunk K - 1 and the true history start behind it
        walked = [*range(K + 3, K - 1, -1), K - 1, K - 2, K - 3, K - 4]
        assert chunk_calls(fake) == pairs_of(*walked)
        assert summary["gap_chunks"] == holes
        assert summary["history_start_ms"] == start_of(K - 1)
        meta = merge_chunks("1h", root=tmp_path)
        assert meta["gaps"] == [[start_of(K), start_of(K + empty)]]
    else:
        # three consecutive empty chunks end the walk, even with history behind them
        assert chunk_calls(fake) == pairs_of(K + 3, K + 2, K + 1, K)
        assert summary["gap_chunks"] == []
        assert summary["history_start_ms"] == start_of(K + 3)
    assert summary["stop_reason"] == "empty_chunks"
    names = file_bytes(tmp_path)
    assert not any(f"{start}.csv" in names or f"{start}.json" in names for start in holes)


def test_a_later_run_counts_only_the_empty_chunks_below_the_archive(tmp_path):
    hidden = {c.open_ms for c in LONG_CANDLES if start_of(K) <= c.open_ms < start_of(K + 2)}
    fake = make_fake(candles=LONG_CANDLES, missing=hidden)
    backfill(client_for(fake), "1h", root=tmp_path)
    calls = len(fake.klines_calls)
    summary = backfill(client_for(fake), "1h", root=tmp_path)
    # the gap chunks are never archived, so they are fetched again, but they lie between archived
    # chunks: only K - 2, K - 3 and K - 4 count towards the stop
    assert chunk_calls(fake, calls) == pairs_of(K + 3, K + 1, K, K - 2, K - 3, K - 4)
    assert summary["gap_chunks"] == [start_of(K), start_of(K + 1)]
    assert summary["empty_chunks"] == [start_of(k) for k in (K - 4, K - 3, K - 2, K, K + 1)]
    assert summary["history_start_ms"] == start_of(K - 1)


def test_an_interrupted_commit_leaves_a_csv_without_sidecar_that_stops_the_next_run(
    tmp_path, monkeypatch
):
    real_write = bh.atomic_write_bytes

    def failing_write(path: Path, data: bytes) -> None:
        if path.suffix == ".json":
            raise OSError("disk full")
        real_write(path, data)

    monkeypatch.setattr(bh, "atomic_write_bytes", failing_write)
    with pytest.raises(OSError, match="disk full"):
        backfill(client_for(make_fake()), "1h", root=tmp_path)
    assert sorted(file_bytes(tmp_path)) == [f"{start_of(K + 3)}.csv"]  # the CSV, then the marker
    monkeypatch.setattr(bh, "atomic_write_bytes", real_write)
    fake = make_fake()
    with pytest.raises(BackfillError, match="has no sidecar"):
        backfill(client_for(fake), "1h", root=tmp_path)
    assert fake.requests == []


def test_a_run_with_nothing_new_rewrites_only_the_incomplete_chunks_sidecar(
    archive_root, monkeypatch
):
    written: list[str] = []
    real_write = bh.atomic_write_bytes

    def recording_write(path: Path, data: bytes) -> None:
        written.append(path.name)
        real_write(path, data)

    monkeypatch.setattr(bh, "atomic_write_bytes", recording_write)
    before = file_bytes(archive_root)
    summary = backfill(client_for(make_fake()), "1h", root=archive_root)
    assert written == [f"{start_of(K + 3)}.json"]
    assert (summary["new_candles"], summary["refreshed_chunks"]) == (0, [start_of(K + 3)])
    after = file_bytes(archive_root)
    assert all(after[name] == data for name, data in before.items() if name.endswith(".csv"))


def test_an_api_error_is_never_an_empty_chunk(tmp_path):
    fake = make_fake()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("startTime") == str(start_of(K + 1)):
            return fakes.api_error(109400, "invalid parameters")
        return fake(request)

    clock = fakes.FakeClock()
    client = PublicMarketClient(
        transport=httpx.MockTransport(handler), min_interval_s=0.0, sleep=clock.sleep, clock=clock
    )
    with pytest.raises(BingXApiError):
        backfill(client, "1h", root=tmp_path)
    # the chunks archived before the error stay; nothing else was written
    assert sorted(file_bytes(tmp_path)) == sorted(
        f"{start_of(k)}.{suffix}" for k in (K + 2, K + 3) for suffix in ("csv", "json")
    )


def test_mismatch_retries_are_recorded(tmp_path):
    fake = make_fake()
    fake.mutator = fakes.alter_on_calls((1,), start_of(K + 3) + 5 * HOUR_MS, volume="1")
    summary = backfill(client_for(fake), "1h", root=tmp_path)
    _, sidecar = read_chunk("1h", K + 3, root=tmp_path)
    assert (sidecar["pairs"], sidecar["requests"], sidecar["mismatch_retries"]) == (2, 4, 1)
    assert (summary["mismatch_retries"], summary["klines_requests"]) == (1, 16)


def test_a_failed_run_names_the_chunk_releases_the_lock_and_can_be_resumed(tmp_path):
    fake = make_fake()
    # requests 0 and 1 are chunk K + 3; requests 2 to 7 are the three pairs of chunk K + 2
    fake.mutator = fakes.alter_on_calls((3, 5, 7), start_of(K + 2) + 5 * HOUR_MS, volume="1")
    with pytest.raises(BackfillError, match=re.escape(f"1h chunk {utc_text(start_of(K + 2))}")):
        backfill(client_for(fake), "1h", root=tmp_path)
    assert sorted(file_bytes(tmp_path)) == [f"{start_of(K + 3)}.csv", f"{start_of(K + 3)}.json"]
    fake.mutator = None
    calls = len(fake.klines_calls)
    summary = backfill(client_for(fake), "1h", root=tmp_path)
    assert chunk_calls(fake, calls) == pairs_of(K + 3, K + 2, K + 1, K, K - 1, K - 2, K - 3)
    assert summary["history_start_ms"] == HISTORY_START


# -- later runs -----------------------------------------------------------------------------------


def test_a_later_run_skips_complete_chunks_and_extends_the_incomplete_one(archive_root):
    before = file_bytes(archive_root)
    newest = f"{start_of(K + 3)}.csv"
    fake = make_fake(SERVER + 100 * HOUR_MS)  # still inside chunk K + 3
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert chunk_calls(fake) == pairs_of(K + 3, K - 1, K - 2, K - 3)  # K .. K + 2: no request
    assert summary["chunks_skipped"] == 3
    assert (summary["new_chunks"], summary["refreshed_chunks"]) == ([], [start_of(K + 3)])
    assert summary["new_candles"] == 100
    after = file_bytes(archive_root)
    assert after[newest].startswith(before[newest]) and after[newest] != before[newest]
    for name in before:
        if not name.startswith(str(start_of(K + 3))):
            assert after[name] == before[name]  # complete chunks are never rewritten
    candles, sidecar = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, SERVER + 100 * HOUR_MS)
    assert (sidecar["rows"], sidecar["server_time_ms"]) == (500, SERVER + 100 * HOUR_MS)


def test_the_incomplete_chunk_completes_and_is_then_skipped(archive_root):
    run = make_fake(start_of(K + 4) + 30_000)  # its last candle closed 30 s ago
    backfill(client_for(run), "1h", root=archive_root)
    _, sidecar = read_chunk("1h", K + 3, root=archive_root)
    assert (sidecar["rows"], sidecar["complete"]) == (999, False)
    run = make_fake(start_of(K + 4) + 90_000)
    summary = backfill(client_for(run), "1h", root=archive_root)
    assert chunk_calls(run) == pairs_of(K + 3, K - 1, K - 2, K - 3)
    _, sidecar = read_chunk("1h", K + 3, root=archive_root)
    assert (sidecar["rows"], sidecar["complete"]) == (1000, True)
    assert summary["archive"]["incomplete_chunks"] == []
    run = make_fake(start_of(K + 4) + HOUR_MS + 90_000)  # the first candle of K + 4 settled
    summary = backfill(client_for(run), "1h", root=archive_root)
    assert chunk_calls(run) == pairs_of(K + 4, K - 1, K - 2, K - 3)
    assert summary["new_chunks"] == [start_of(K + 4)]
    assert summary["archive"]["incomplete_chunks"] == [start_of(K + 4)]


def revise(
    fake: fakes.FakeBingX, open_ms: int, interval: str = "1h", digit: str = "1"
) -> tuple[Candle, Candle]:
    """Make the exchange serve another volume string for the candle at ``open_ms``."""
    archived = fake.candle(interval, open_ms)
    revised = replace(archived, volume=archived.volume + digit)
    fake.put(interval, revised)
    return archived, revised


def row_of(candle: Candle) -> str:
    """The chunk CSV row of ``candle`` without its line end."""
    c = candle
    return f"{c.open_ms},{c.open},{c.high},{c.low},{c.close},{c.volume}"


def revision_entry(
    archived: Candle,
    refetched: Candle,
    first_ms: int,
    last_ms: int | None = None,
    interval: str = "1h",
) -> dict[str, Any]:
    """The record entry of a revision first observed at ``first_ms`` (and last at ``last_ms``)."""
    span = CHUNK_CANDLES * pm.interval_ms(interval)
    start = archived.open_ms // span * span
    last_ms = first_ms if last_ms is None else last_ms
    return {
        "chunk_start_ms": start,
        "chunk_start_utc": utc_text(start),
        "csv_relpath": f"raw/bingx/{interval}/{start}.csv",
        "open_ms": archived.open_ms,
        "open_utc": utc_text(archived.open_ms),
        "archived": row_of(archived),
        "refetched": row_of(refetched),
        "first_server_time_ms": first_ms,
        "first_server_time_utc": utc_text(first_ms),
        "last_server_time_ms": last_ms,
        "last_server_time_utc": utc_text(last_ms),
    }


def revisions_warning(interval: str, seen: list[tuple[int, int, int]]) -> str:
    """The status warning listing revisions given as ``(open, first seen, last seen)``."""
    items = []
    for open_ms, first_ms, last_ms in seen:
        when = utc_text(first_ms)
        if last_ms != first_ms:
            when += f" to {utc_text(last_ms)}"
        items.append(f"{utc_text(open_ms)} (seen {when})")
    return (
        f"bingx {interval}: {len(seen)} revision(s) recorded in raw/bingx/revision-{interval}.json "
        "(archived candles the exchange later served with other values; the archive and the store "
        f"keep the first-seen rows): {', '.join(items)}; report them to the owner"
    )


def observed_warning(interval: str, start_ms: int, listed: str, revised: int, new: int) -> str:
    """The warning a backfill sends once it has recorded the revisions of the chunk at
    ``start_ms`` (``listed``: their open times as printed)."""
    return (
        f"bingx {interval} chunk {utc_text(start_ms)}: the exchange served {revised} archived "
        f"candle(s) with other values ({new} of them not recorded before): {listed}; the archive "
        f"and the store keep the first-seen rows, and raw/bingx/revision-{interval}.json records "
        "every revision; report them to the owner"
    )


def warnings_of(root: Path, interval: str, now_ms: int) -> list[str]:
    return [w for w in retention_warnings(root, now_ms=now_ms) if w.startswith(f"bingx {interval}")]


def chunk_state(interval: str, k: int, root: Path) -> tuple[int, bool, int]:
    """``(rows, complete, archive-only rows)`` of an archived chunk."""
    _, sidecar = read_chunk(interval, k, root=root)
    return sidecar["rows"], sidecar["complete"], sidecar["archive_only"]["rows"]


def held_warning(interval: str, start_ms: int, server_time_ms: int, kept: int) -> str:
    """The status warning about a chunk kept incomplete for candles the exchange stopped
    serving."""
    return (
        f"bingx {interval}: raw/bingx/{interval}/{start_ms}.csv is incomplete: its last refetch "
        f"({utc_text(server_time_ms)}) no longer returned {kept} archived candle(s) (kept, sidecar "
        "archive_only), not all explained by the exchange's retention, so every backfill fetches "
        "the chunk again; report this to the owner"
    )


REVISED = start_of(K + 3) + 123 * HOUR_MS  # archived by the first run, in the newest chunk


def test_a_revised_candle_keeps_its_first_seen_row_and_the_backfill_goes_on(archive_root, caplog):
    # the owner's decision (D-045): what a live decision would have used stays in the archive,
    # the revision is recorded, and nothing stops
    before = file_bytes(archive_root)
    newest = f"{start_of(K + 3)}.csv"
    server = SERVER + 10 * HOUR_MS
    fake = make_fake(server)
    archived, revised = revise(fake, REVISED)
    assert archived == BY_OPEN[REVISED]
    with caplog.at_level("WARNING", logger=bh.__name__):
        summary = backfill(client_for(fake), "1h", root=archive_root)
    # without a warn callback, the warning goes to the module's logger
    assert caplog.messages == [
        observed_warning("1h", start_of(K + 3), utc_text(REVISED), revised=1, new=1)
    ]
    assert chunk_calls(fake) == pairs_of(K + 3, K - 1, K - 2, K - 3)  # the usual walk
    assert (summary["stop_reason"], summary["history_start_ms"]) == ("empty_chunks", HISTORY_START)
    assert (summary["refreshed_chunks"], summary["new_candles"]) == ([start_of(K + 3)], 10)
    assert (summary["revised_candles"], summary["new_revisions"]) == (1, 1)
    assert summary["revised_opens"] == [REVISED]
    assert summary["revision_relpath"] == revision_relpath("1h") == "raw/bingx/revision-1h.json"
    # nothing archived is overwritten: the archived rows first, byte for byte, then the new ones
    after = file_bytes(archive_root)
    assert after[newest].startswith(before[newest])
    for name in before:
        if not name.startswith(str(start_of(K + 3))):
            assert after[name] == before[name]
    candles, sidecar = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, server)  # the first-seen values
    assert archived in candles and revised not in candles
    assert chunk_state("1h", K + 3, archive_root) == (410, False, 0)
    assert sidecar["server_time_ms"] == server
    record = json.loads((archive_root / revision_relpath("1h")).read_text())
    assert {key: record[key] for key in record if key != "code"} == {
        "schema_version": REVISION_SCHEMA_VERSION,
        "source": "bingx",
        "symbol": "BTC-USDT",
        "interval": "1h",
        "columns": "open_ms,open,high,low,close,volume",
        "revisions": [revision_entry(archived, revised, server)],
    }
    assert set(record["code"]) == {"commit", "dirty", "source_tree_sha256"}
    assert read_revisions("1h", root=archive_root) == record["revisions"]
    # the status lists it (open time and observations; the rows stay in the record)
    assert warnings_of(archive_root, "1h", server) == [
        revisions_warning("1h", [(REVISED, server, server)])
    ]


def test_a_revision_seen_again_is_recorded_once(archive_root):
    path = archive_root / revision_relpath("1h")
    fake = make_fake(SERVER + 10 * HOUR_MS)
    archived, first = revise(fake, REVISED)
    seen = [fake.server_time_ms]
    backfill(client_for(fake), "1h", root=archive_root)
    # the next run refetches the incomplete chunk and observes the same pair of rows again
    fake.advance(HOUR_MS)
    seen.append(fake.server_time_ms)
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert (summary["revised_candles"], summary["new_revisions"]) == (1, 0)
    assert read_revisions("1h", root=archive_root) == [revision_entry(archived, first, *seen)]
    # another value for the same open is another revision; the archived row never changes
    _, second = revise(fake, REVISED, digit="2")
    fake.advance(HOUR_MS)
    seen.append(fake.server_time_ms)
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert (summary["revised_candles"], summary["new_revisions"]) == (1, 1)
    # the chunk completes by the usual rule while the second value is still served
    fake.server_time_ms = start_of(K + 4) + SETTLE_MS
    seen.append(fake.server_time_ms)
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert (summary["revised_candles"], summary["new_revisions"]) == (1, 0)
    assert chunk_state("1h", K + 3, archive_root) == (1000, True, 0)
    candles, _ = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, fake.server_time_ms)  # the first-seen rows
    assert read_revisions("1h", root=archive_root) == [
        revision_entry(archived, first, seen[0], seen[1]),
        revision_entry(archived, second, seen[2], seen[3]),
    ]
    # complete: never fetched again, so the record stays as it is
    data = path.read_bytes()
    fake.advance(2 * HOUR_MS)
    calls = len(fake.klines_calls)
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert start_of(K + 3) not in [call.start_ms for call in fake.klines_calls[calls:]]
    assert (summary["revised_candles"], summary["revised_opens"]) == (0, [])
    assert path.read_bytes() == data
    assert warnings_of(archive_root, "1h", fake.server_time_ms) == [
        revisions_warning("1h", [(REVISED, seen[0], seen[1]), (REVISED, seen[2], seen[3])])
    ]


def test_revisions_in_the_newest_chunk_and_in_an_older_one(archive_root):
    # a bounded run archives the new chunk K + 4 and leaves the incomplete K + 3 behind it
    bounded = make_fake(start_of(K + 4) + 5 * HOUR_MS + 30 * MINUTE_MS)
    backfill(client_for(bounded), "1h", root=archive_root, max_chunks=1)
    assert chunk_state("1h", K + 3, archive_root) == (400, False, 0)
    assert chunk_state("1h", K + 4, archive_root) == (5, False, 0)
    before = file_bytes(archive_root)
    older, newer = start_of(K + 3) + 7 * HOUR_MS, start_of(K + 4) + 2 * HOUR_MS
    server = start_of(K + 4) + 20 * HOUR_MS + 30 * MINUTE_MS
    fake = make_fake(server)
    pairs = [revise(fake, older), revise(fake, newer)]
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert chunk_calls(fake) == pairs_of(K + 4, K + 3, K - 1, K - 2, K - 3)
    assert summary["refreshed_chunks"] == [start_of(K + 3), start_of(K + 4)]
    assert (summary["revised_candles"], summary["new_revisions"]) == (2, 2)
    assert summary["revised_opens"] == [older, newer]
    assert summary["new_candles"] == 600 + 15
    after = file_bytes(archive_root)
    for k in (K + 3, K + 4):
        name = f"{start_of(k)}.csv"
        assert after[name].startswith(before[name])
        assert read_chunk("1h", k, root=archive_root)[0] == settled_candles(k, server)
    assert chunk_state("1h", K + 3, archive_root) == (1000, True, 0)  # completes as usual
    assert chunk_state("1h", K + 4, archive_root) == (20, False, 0)
    assert read_revisions("1h", root=archive_root) == [
        revision_entry(archived, refetched, server) for archived, refetched in pairs
    ]
    assert warnings_of(archive_root, "1h", server) == [
        revisions_warning("1h", [(older, server, server), (newer, server, server)])
    ]


def test_the_status_names_ten_revisions_and_counts_the_rest(archive_root):
    server = SERVER + 10 * HOUR_MS
    fake = make_fake(server)
    opens = [start_of(K + 3) + hours * HOUR_MS for hours in range(300, 312)]
    for open_ms in opens:
        revise(fake, open_ms)
    warnings: list[str] = []
    summary = backfill(client_for(fake), "1h", root=archive_root, warn=warnings.append)
    assert (summary["revised_candles"], summary["revised_opens"]) == (12, opens)
    listed = ", ".join(utc_text(open_ms) for open_ms in opens[:10]) + " and 2 more"
    assert warnings == [observed_warning("1h", start_of(K + 3), listed, revised=12, new=12)]
    (warning,) = warnings_of(archive_root, "1h", server)
    listed = revisions_warning("1h", [(open_ms, server, server) for open_ms in opens[:10]])
    head, tail = listed.split("; report them")
    assert warning == (head + " and 2 more; report them" + tail).replace(
        "10 revision(s)", "12 revision(s)"
    )
    assert utc_text(opens[10]) not in warning


def test_the_store_is_rebuilt_from_the_first_seen_rows(archive_root):
    fake = make_fake(SERVER + 10 * HOUR_MS)
    archived, revised = revise(fake, REVISED)
    backfill(client_for(fake), "1h", root=archive_root)
    meta = merge_chunks("1h", root=archive_root)
    store_id = StoreId("bingx", "1h")
    data = store_id.csv_path(archive_root).read_bytes()
    assert f"\n{row_of(archived)}\n".encode() in data
    assert row_of(revised).encode() not in data
    frame = read_store(store_id, root=archive_root)
    assert fakes.frame_to_candles(frame[frame["open_ms"] == REVISED]) == [archived]
    assert fakes.frame_to_candles(frame) == [
        c for c in CANDLES if c.open_ms < start_of(K + 3) + 410 * HOUR_MS
    ]
    assert (meta["rows"], meta["gaps"]) == (3160, [])


class Killed(BaseException):
    """A kill inside a write: nothing on the backfill's path catches it."""


@pytest.mark.parametrize(
    ("server_time_ms", "killed_at", "after"),
    [
        # the refetch completes the chunk (600 new candles): killed at its CSV
        pytest.param(start_of(K + 4) + SETTLE_MS, ".csv", (1000, True, 0), id="csv"),
        # no candle settled since the archive's run, so the CSV is unchanged: killed at its sidecar
        pytest.param(SERVER + 10 * MINUTE_MS, ".json", (400, False, 0), id="sidecar"),
    ],
)
def test_a_revision_is_recorded_before_its_chunk_is_written(
    archive_root, monkeypatch, server_time_ms, killed_at, after
):
    # a complete chunk is never fetched again: a revision recorded after its chunk's commit
    # marker would be lost for good by a kill between the two (D-045, recorded durably)
    before = file_bytes(archive_root)
    fake = make_fake(server_time_ms)
    archived, revised = revise(fake, REVISED)
    killed = chunk_paths("1h", K + 3, root=archive_root)[0].with_suffix(killed_at)
    real_write = bh.atomic_write_bytes

    def killing_write(path: Path, data: bytes) -> None:
        if path == killed:
            raise Killed
        real_write(path, data)

    monkeypatch.setattr(bh, "atomic_write_bytes", killing_write)
    warnings: list[str] = []
    with pytest.raises(Killed):
        backfill(client_for(fake), "1h", root=archive_root, warn=warnings.append)
    assert read_revisions("1h", root=archive_root) == [
        revision_entry(archived, revised, server_time_ms)
    ]
    # reported as soon as it was recorded, before the chunk's write
    assert warnings == [
        observed_warning("1h", start_of(K + 3), utc_text(REVISED), revised=1, new=1)
    ]
    assert file_bytes(archive_root) == before  # the chunk as archived: fetched again next run
    monkeypatch.setattr(bh, "atomic_write_bytes", real_write)
    fake.advance(10 * MINUTE_MS)
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert (summary["revised_candles"], summary["new_revisions"]) == (1, 0)
    # one entry: only its last observation moved
    assert read_revisions("1h", root=archive_root) == [
        revision_entry(archived, revised, server_time_ms, fake.server_time_ms)
    ]
    assert chunk_state("1h", K + 3, archive_root) == after
    candles, _ = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, fake.server_time_ms)  # the first-seen rows


def test_a_chunk_fetched_again_after_its_files_were_removed_keeps_its_first_seen_rows(
    archive_root, monkeypatch
):
    # a kill between the CSV and the sidecar of a refresh leaves a CSV its sidecar does not
    # describe; removing both files, the repair the error names, fetches the chunk again, and
    # its recorded revisions give back the first-seen rows (D-045)
    fake = make_fake(SERVER + 10 * HOUR_MS)
    archived, revised = revise(fake, REVISED)
    first = fake.server_time_ms
    csv_path, json_path = chunk_paths("1h", K + 3, root=archive_root)
    real_write = bh.atomic_write_bytes

    def killing_write(path: Path, data: bytes) -> None:
        if path == json_path:
            raise Killed
        real_write(path, data)

    monkeypatch.setattr(bh, "atomic_write_bytes", killing_write)
    with pytest.raises(Killed):
        backfill(client_for(fake), "1h", root=archive_root)
    monkeypatch.setattr(bh, "atomic_write_bytes", real_write)
    torn = (
        r"\.csv: size or sha256 differs from its sidecar .* those it serves with a revision "
        r"recorded in raw/bingx/revision-1h\.json are archived with their recorded first-seen rows"
    )
    fake.advance(HOUR_MS)
    requests = len(fake.requests)
    with pytest.raises(BackfillError, match=torn):
        backfill(client_for(fake), "1h", root=archive_root)
    assert len(fake.requests) == requests
    with pytest.raises(BackfillError, match=torn):
        merge_chunks("1h", root=archive_root)
    csv_path.unlink()
    json_path.unlink()
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert summary["new_chunks"] == [start_of(K + 3)]
    assert (summary["revised_candles"], summary["new_revisions"]) == (1, 0)
    candles, _ = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, fake.server_time_ms)  # the first-seen rows
    assert archived in candles and revised not in candles
    assert read_revisions("1h", root=archive_root) == [
        revision_entry(archived, revised, first, fake.server_time_ms)
    ]
    merge_chunks("1h", root=archive_root)
    frame = read_store(StoreId("bingx", "1h"), root=archive_root)
    assert fakes.frame_to_candles(frame[frame["open_ms"] == REVISED]) == [archived]
    assert warnings_of(archive_root, "1h", fake.server_time_ms) == [
        revisions_warning("1h", [(REVISED, first, fake.server_time_ms)])
    ]


@pytest.mark.parametrize(
    ("served", "counts"),
    [("first_seen", (0, 0)), ("recorded", (1, 0)), ("another", (1, 1))],
)
def test_a_chunk_fetched_again_takes_the_first_seen_rows_of_its_revisions(
    archive_root, served, counts
):
    fake = make_fake(SERVER + 10 * HOUR_MS)
    archived, recorded = revise(fake, REVISED)
    first = fake.server_time_ms
    backfill(client_for(fake), "1h", root=archive_root)
    for path in chunk_paths("1h", K + 3, root=archive_root):
        path.unlink()
    another = replace(archived, volume=archived.volume + "2")
    fake.put("1h", {"first_seen": archived, "recorded": recorded, "another": another}[served])
    fake.advance(HOUR_MS)
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert (summary["revised_candles"], summary["new_revisions"]) == counts
    assert (summary["new_chunks"], summary["refreshed_chunks"]) == ([start_of(K + 3)], [])
    assert (summary["new_candles"], summary["kept_from_archive"]) == (411, 0)
    candles, _ = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, fake.server_time_ms)  # the first-seen rows
    last = fake.server_time_ms if served == "recorded" else first
    expected = [revision_entry(archived, recorded, first, last)]
    if served == "another":  # another value of the first-seen row is another revision
        expected.append(revision_entry(archived, another, fake.server_time_ms))
    assert read_revisions("1h", root=archive_root) == expected


@pytest.fixture
def revised_root(archive_root) -> Path:
    """``archive_root`` after a run that recorded one revision."""
    fake = make_fake(SERVER + 10 * HOUR_MS)
    revise(fake, REVISED)
    backfill(client_for(fake), "1h", root=archive_root)
    return archive_root


def _revision_times(first_ms: int, last_ms: int) -> Callable[[dict[str, Any]], None]:
    def edit(record: dict[str, Any]) -> None:
        record["revisions"][0].update(
            first_server_time_ms=first_ms,
            first_server_time_utc=utc_text(first_ms),
            last_server_time_ms=last_ms,
            last_server_time_utc=utc_text(last_ms),
        )

    return edit


def _revision_rows(**rows: str) -> Callable[[dict[str, Any]], None]:
    def edit(record: dict[str, Any]) -> None:
        entry = record["revisions"][0]
        entry.update({key: text.format(**entry) for key, text in rows.items()})

    return edit


RECORD_CORRUPTIONS: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
    ("unknown schema", lambda r: r.update(schema_version=3)),
    ("boolean schema", lambda r: r.update(schema_version=True)),
    ("other columns", lambda r: r.update(columns="open_ms,close")),
    ("other interval", lambda r: r.update(interval="5m")),
    ("no revisions", lambda r: r.update(revisions=[])),
    ("repeated revision", lambda r: r["revisions"].append(dict(r["revisions"][0]))),
    ("unknown key", lambda r: r["revisions"][0].update(note="x")),
    ("boolean open", lambda r: r["revisions"][0].update(open_ms=True)),
    ("another chunk", lambda r: r["revisions"][0].update(chunk_start_ms=start_of(K + 2))),
    ("open text", lambda r: r["revisions"][0].update(open_utc="2024-01-01T00:00Z")),
    ("the same rows", _revision_rows(refetched="{archived}")),
    ("rows of two opens", _revision_rows(refetched="1{refetched}")),
    ("leading zero", _revision_rows(archived="0{archived}")),
    ("exponent", _revision_rows(archived="{archived}e5")),
    ("first after last", _revision_times(SERVER + 12 * HOUR_MS, SERVER + 11 * HOUR_MS)),
    ("seen before it settled", _revision_times(REVISED + HOUR_MS, SERVER + 10 * HOUR_MS)),
]


@pytest.mark.parametrize(
    "corrupt", [pytest.param(corrupt, id=name) for name, corrupt in RECORD_CORRUPTIONS]
)
def test_a_revision_record_that_is_not_one_is_reported(revised_root, corrupt):
    path = revised_root / revision_relpath("1h")
    record = json.loads(path.read_text())
    corrupt(record)
    path.write_text(json.dumps(record))
    with pytest.raises(BackfillError, match="does not describe revised 1h candles"):
        read_revisions("1h", root=revised_root)
    fake = make_fake(SERVER + 11 * HOUR_MS)
    with pytest.raises(BackfillError, match="does not describe revised 1h candles"):
        backfill(client_for(fake), "1h", root=revised_root)
    assert fake.requests == []  # verified before any request, not even the server time
    warning = next(w for w in warnings_of(revised_root, "1h", SERVER) if "revision" in w)
    assert warning.startswith(
        "bingx 1h: raw/bingx/revision-1h.json does not describe revised 1h candles"
    )
    # the store is rebuilt from the chunks alone: a record that cannot be read never blocks it
    assert merge_chunks("1h", root=revised_root)["rows"] == 3160


def test_a_revision_record_that_is_not_json_is_reported(revised_root):
    (revised_root / revision_relpath("1h")).write_text("{")
    with pytest.raises(BackfillError, match="not a valid revision record"):
        read_revisions("1h", root=revised_root)
    assert read_revisions("5m", root=revised_root) == []  # no record: no revision
    assert merge_chunks("1h", root=revised_root)["rows"] == 3160


def test_a_schema_1_record_is_read_as_its_one_revision(archive_root):
    # what a backfill that stopped at a revision wrote before the owner's decision (D-047)
    fake = make_fake(SERVER + 10 * HOUR_MS)
    archived, revised = revise(fake, REVISED)
    stopped = SERVER + 5 * HOUR_MS
    fields = ("open", "high", "low", "close", "volume")
    record = {
        "schema_version": 1,
        "source": "bingx",
        "symbol": "BTC-USDT",
        "interval": "1h",
        "chunk_start_ms": start_of(K + 3),
        "chunk_start_utc": utc_text(start_of(K + 3)),
        "csv_relpath": f"raw/bingx/1h/{start_of(K + 3)}.csv",
        "revised_candles": 1,
        "first_open_ms": REVISED,
        "first_open_utc": utc_text(REVISED),
        "archived": {name: getattr(archived, name) for name in fields},
        "served": {name: getattr(revised, name) for name in fields},
        "server_time_ms": stopped,
        "server_time_utc": utc_text(stopped),
        "code": {},
    }
    path = archive_root / revision_relpath("1h")
    path.write_text(json.dumps(record))
    assert read_revisions("1h", root=archive_root) == [revision_entry(archived, revised, stopped)]
    assert warnings_of(archive_root, "1h", SERVER + 10 * HOUR_MS) == [
        revisions_warning("1h", [(REVISED, stopped, stopped)])
    ]
    # the next run observes it again and writes the record in the current schema
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert (summary["revised_candles"], summary["new_revisions"]) == (1, 0)
    rewritten = json.loads(path.read_text())
    assert rewritten["schema_version"] == REVISION_SCHEMA_VERSION == 2
    assert rewritten["revisions"] == [
        revision_entry(archived, revised, stopped, fake.server_time_ms)
    ]
    for broken in ({"served": {"volume": "1"}}, {"chunk_start_ms": start_of(K + 2)}):
        path.write_text(json.dumps(record | broken))
        with pytest.raises(BackfillError, match="does not describe revised 1h candles"):
            read_revisions("1h", root=archive_root)


def test_an_archived_candle_the_refetch_no_longer_returns_is_kept_and_listed(archive_root):
    before = file_bytes(archive_root)
    newest = f"{start_of(K + 3)}.csv"
    target = start_of(K + 3) + 123 * HOUR_MS
    fake = make_fake(SERVER + 10 * HOUR_MS, missing={target})  # no longer served, mid-chunk
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert summary["refreshed_chunks"] == [start_of(K + 3)]
    assert (summary["new_candles"], summary["kept_from_archive"]) == (10, 1)
    assert summary["kept_inside_served_range"] == 1  # not retention: the owner should look
    after = file_bytes(archive_root)
    assert after[newest].startswith(before[newest])  # archived rows first, byte for byte
    candles, sidecar = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, SERVER + 10 * HOUR_MS)  # the candle is still there
    assert sidecar["schema_version"] == SIDECAR_SCHEMA_VERSION == 2
    assert sidecar["archive_only"] == {"rows": 1, "ranges": [[target, target + HOUR_MS]]}
    assert (sidecar["rows"], sidecar["gaps"]) == (410, [])
    meta = merge_chunks("1h", root=archive_root)
    assert [c["archive_only_rows"] for c in meta["provenance"]["chunks"]] == [0, 0, 0, 1]
    assert (meta["rows"], meta["gaps"]) == (3160, [])
    # a later refetch that serves it again leaves nothing archive-only
    summary = backfill(client_for(make_fake(SERVER + 11 * HOUR_MS)), "1h", root=archive_root)
    assert (summary["new_candles"], summary["kept_from_archive"]) == (1, 0)
    assert read_chunk("1h", K + 3, root=archive_root)[1]["archive_only"] == {
        "rows": 0,
        "ranges": [],
    }


def test_a_1h_chunk_whose_older_candles_vanish_stays_incomplete_until_they_are_served(
    archive_root,
):
    # tests-docs-3: the 1h history is fixed, so archived 1h candles the exchange stops serving
    # are never retention: the chunk keeps them and stays incomplete instead of completing
    before = file_bytes(archive_root)
    newest = f"{start_of(K + 3)}.csv"
    server = start_of(K + 4) + 600 * HOUR_MS + 30 * MINUTE_MS
    fake = make_fake(server)
    fake.history_start_ms["1h"] = start_of(K + 3) + 100 * HOUR_MS
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert chunk_calls(fake) == pairs_of(K + 4, K + 3, K - 1, K - 2, K - 3)
    assert (summary["new_chunks"], summary["refreshed_chunks"]) == (
        [start_of(K + 4)],
        [start_of(K + 3)],
    )
    assert summary["new_candles"] == 600 + 600
    assert (summary["kept_from_archive"], summary["kept_inside_served_range"]) == (100, 100)
    assert summary["history_start_ms"] == HISTORY_START  # the recorded start is unchanged
    after = file_bytes(archive_root)
    assert after[newest].startswith(before[newest])
    for name in before:
        if not name.startswith(str(start_of(K + 3))):
            assert after[name] == before[name]
    candles, sidecar = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, server)  # 100 from the archive, 900 served
    assert (sidecar["rows"], sidecar["complete"], sidecar["gaps"]) == (1000, False, [])
    vanished = [start_of(K + 3), start_of(K + 3) + 100 * HOUR_MS]
    assert sidecar["archive_only"] == {"rows": 100, "ranges": [vanished]}
    assert warnings_of(archive_root, "1h", server) == [
        held_warning("1h", start_of(K + 3), server, 100)
    ]
    # the next run fetches it again; once the exchange serves them, the chunk completes
    calls = len(fake.klines_calls)
    fake.advance(HOUR_MS)
    fake.history_start_ms["1h"] = HISTORY_START
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert chunk_calls(fake, calls) == pairs_of(K + 4, K + 3, K - 1, K - 2, K - 3)
    assert (summary["kept_from_archive"], summary["kept_inside_served_range"]) == (0, 0)
    assert chunk_state("1h", K + 3, archive_root) == (1000, True, 0)
    assert warnings_of(archive_root, "1h", fake.server_time_ms) == []


def test_an_empty_refetch_of_a_closed_1h_chunk_keeps_it_incomplete_until_it_is_served(
    archive_root,
):
    # tests-docs-3: one run gets code 0 with an empty list for chunk K + 3 after its end
    before = file_bytes(archive_root)
    server = start_of(K + 4) + 5 * HOUR_MS + 30 * MINUTE_MS
    fake = make_fake(server)
    fake.mutator = lambda call, rows: [] if call.start_ms == start_of(K + 3) else rows
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert chunk_calls(fake) == pairs_of(K + 4, K + 3, K - 1, K - 2, K - 3)
    assert summary["refreshed_chunks"] == [start_of(K + 3)]
    assert (summary["kept_from_archive"], summary["kept_inside_served_range"]) == (400, 400)
    assert file_bytes(archive_root)[f"{start_of(K + 3)}.csv"] == before[f"{start_of(K + 3)}.csv"]
    assert chunk_state("1h", K + 3, archive_root) == (400, False, 400)
    assert warnings_of(archive_root, "1h", server) == [
        held_warning("1h", start_of(K + 3), server, 400)
    ]
    # the next run, with normal answers, fills the chunk: no permanent gap
    fake.mutator = None
    fake.advance(HOUR_MS)
    calls = len(fake.klines_calls)
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert chunk_calls(fake, calls) == pairs_of(K + 4, K + 3, K - 1, K - 2, K - 3)
    assert summary["new_candles"] == 600 + 1
    candles, _ = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, fake.server_time_ms)
    assert chunk_state("1h", K + 3, archive_root) == (1000, True, 0)
    assert merge_chunks("1h", root=archive_root)["gaps"] == []
    assert warnings_of(archive_root, "1h", fake.server_time_ms) == []


# A 5m world: 5m has a rolling retention (about 48 days observed). Chunk K5 opens at
# 2024-10-04T00:00Z; the exchange's history starts 100 slots into it; the first run leaves 400
# settled candles in chunk K5 + 1.
STEP_5M = 300_000
SPAN_5M = CHUNK_CANDLES * STEP_5M
K5 = 5760
RETENTION_5M_MS = RETENTION_DAYS["5m"] * DAY_MS


def start_5m(k: int) -> int:
    return k * SPAN_5M


def rolled(slots: int) -> int:
    """Open time ``slots`` five-minute slots after the start of 5m chunk K5 + 1."""
    return start_5m(K5 + 1) + slots * STEP_5M


FIVE_START = start_5m(K5) + 100 * STEP_5M
FIVE_CANDLES = fakes.synthetic_candles(FIVE_START, 17_000, interval="5m", seed=13)
FIVE_SERVER = start_5m(K5 + 1) + 400 * STEP_5M + 2 * MINUTE_MS


def five_minute_fake(server_time_ms: int, history_start_ms: int = FIVE_START) -> fakes.FakeBingX:
    return fakes.FakeBingX(
        server_time_ms, {"5m": FIVE_CANDLES}, history_start_ms={"5m": history_start_ms}
    )


@pytest.fixture
def five_minute_root(tmp_path) -> Path:
    root = tmp_path / "data5m"
    summary = backfill(client_for(five_minute_fake(FIVE_SERVER)), "5m", root=root)
    assert summary["archive"]["incomplete_chunks"] == [start_5m(K5 + 1)]
    assert read_chunk("5m", K5 + 1, root=root)[1]["rows"] == 400
    return root


def test_an_empty_refetch_of_a_young_5m_chunk_keeps_it_incomplete(five_minute_root):
    # tests-docs-3: 3.5 days old, well inside the 48-day retention: an empty answer is no roll-off
    root = five_minute_root
    server = start_5m(K5 + 2) + 2 * HOUR_MS
    fake = five_minute_fake(server)
    fake.mutator = lambda call, rows: [] if call.start_ms == start_5m(K5 + 1) else rows
    summary = backfill(client_for(fake), "5m", root=root)
    assert summary["refreshed_chunks"] == [start_5m(K5 + 1)]
    assert (summary["kept_from_archive"], summary["kept_inside_served_range"]) == (400, 400)
    assert chunk_state("5m", K5 + 1, root) == (400, False, 400)
    assert warnings_of(root, "5m", server) == [held_warning("5m", start_5m(K5 + 1), server, 400)]
    fake.mutator = None
    fake.advance(STEP_5M)
    summary = backfill(client_for(fake), "5m", root=root)
    assert summary["refreshed_chunks"] == [start_5m(K5 + 1), start_5m(K5 + 2)]
    assert chunk_state("5m", K5 + 1, root) == (1000, True, 0)
    assert merge_chunks("5m", root=root)["gaps"] == []
    assert warnings_of(root, "5m", fake.server_time_ms) == []


def test_candles_rolled_off_the_retention_are_kept_and_the_chunk_completes(five_minute_root):
    # the run missed for 50 days: the exchange's 48-day retention cut falls inside chunk K5 + 1,
    # after its archived candles, which rolled off with everything older
    root = five_minute_root
    newest = archive_dir(root, "5m") / f"{start_5m(K5 + 1)}.csv"
    before = newest.read_bytes()
    server = start_5m(K5 + 1) + 50 * DAY_MS
    cut = server - RETENTION_5M_MS
    assert cut == start_5m(K5 + 1) + 576 * STEP_5M
    fake = five_minute_fake(server, history_start_ms=cut)
    summary = backfill(client_for(fake), "5m", root=root)
    assert summary["refreshed_chunks"] == [start_5m(K5 + 1)]
    assert (summary["kept_from_archive"], summary["kept_inside_served_range"]) == (400, 0)
    assert newest.read_bytes().startswith(before)
    _, sidecar = read_chunk("5m", K5 + 1, root=root)
    assert (sidecar["rows"], sidecar["complete"]) == (400 + 424, True)
    rolled_off = [start_5m(K5 + 1), start_5m(K5 + 1) + 400 * STEP_5M]
    assert sidecar["archive_only"] == {"rows": 400, "ranges": [rolled_off]}
    # what rolled off before it was ever archived is a gap, never filled
    assert sidecar["gaps"] == [[rolled_off[1], cut]]
    assert warnings_of(root, "5m", server) == []
    # complete now: never fetched again
    calls = len(fake.klines_calls)
    fake.advance(STEP_5M)
    summary = backfill(client_for(fake), "5m", root=root)
    assert start_5m(K5 + 1) not in [call.start_ms for call in fake.klines_calls[calls:]]
    assert summary["kept_from_archive"] == 0


def test_a_chunk_whose_refetch_returns_nothing_keeps_its_archive(five_minute_root):
    # the run missed for 50 days after the chunk's end: all of K5 + 1 is older than the 48-day
    # retention, so the exchange answers its refetch with nothing
    root = five_minute_root
    newest = archive_dir(root, "5m") / f"{start_5m(K5 + 1)}.csv"
    before = newest.read_bytes()
    server = start_5m(K5 + 2) + 50 * DAY_MS
    fake = five_minute_fake(server, history_start_ms=server - RETENTION_5M_MS)
    summary = backfill(client_for(fake), "5m", root=root)
    assert start_5m(K5 + 1) in summary["refreshed_chunks"]
    assert (summary["kept_from_archive"], summary["kept_inside_served_range"]) == (400, 0)
    assert newest.read_bytes() == before
    candles, sidecar = read_chunk("5m", K5 + 1, root=root)
    assert candles == [c for c in FIVE_CANDLES if start_5m(K5 + 1) <= c.open_ms < rolled(400)]
    assert (sidecar["rows"], sidecar["complete"]) == (400, True)
    assert sidecar["archive_only"] == {"rows": 400, "ranges": [[start_5m(K5 + 1), rolled(400)]]}
    meta = merge_chunks("5m", root=root)
    # what rolled off before it was ever archived is a gap, never filled
    assert meta["gaps"][0] == [rolled(400), server - RETENTION_5M_MS]
    calls = len(fake.klines_calls)
    backfill(client_for(fake), "5m", root=root)
    assert start_5m(K5 + 1) not in [call.start_ms for call in fake.klines_calls[calls:]]


def test_a_revision_beside_candles_that_rolled_off_the_retention(five_minute_root):
    # the run missed for 48 days: the 5m retention cut falls inside the archived candles of
    # chunk K5 + 1, and one of the candles still served comes back with another volume
    root = five_minute_root
    newest = archive_dir(root, "5m") / f"{start_5m(K5 + 1)}.csv"
    before = newest.read_bytes()
    server = rolled(200) + RETENTION_5M_MS
    fake = five_minute_fake(server, history_start_ms=rolled(200))
    archived, revised = revise(fake, rolled(300), interval="5m")
    summary = backfill(client_for(fake), "5m", root=root)
    assert start_5m(K5 + 1) in summary["refreshed_chunks"]
    assert (summary["kept_from_archive"], summary["kept_inside_served_range"]) == (200, 0)
    assert (summary["revised_candles"], summary["new_revisions"]) == (1, 1)
    assert summary["stop_reason"] == "empty_chunks"
    assert newest.read_bytes().startswith(before)
    candles, sidecar = read_chunk("5m", K5 + 1, root=root)
    assert candles == [c for c in FIVE_CANDLES if start_5m(K5 + 1) <= c.open_ms < start_5m(K5 + 2)]
    assert archived in candles and revised not in candles
    # completion follows the retention rule: the rolled-off candles explain the archive-only ones
    assert (sidecar["rows"], sidecar["complete"]) == (1000, True)
    assert sidecar["archive_only"] == {"rows": 200, "ranges": [[rolled(0), rolled(200)]]}
    assert read_revisions("5m", root=root) == [
        revision_entry(archived, revised, server, interval="5m")
    ]
    assert warnings_of(root, "5m", server) == [
        revisions_warning("5m", [(rolled(300), server, server)])
    ]


def test_a_revision_beside_1h_candles_that_vanished_keeps_the_chunk_incomplete(archive_root):
    # 1h candles the exchange stops serving are never retention: the chunk stays incomplete for
    # them, not for the revision, which is recorded while the run goes on
    before = file_bytes(archive_root)
    target = start_of(K + 3) + 200 * HOUR_MS
    server = start_of(K + 4) + 600 * HOUR_MS + 30 * MINUTE_MS
    fake = make_fake(server)
    fake.history_start_ms["1h"] = start_of(K + 3) + 100 * HOUR_MS
    archived, revised = revise(fake, target)
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert chunk_calls(fake) == pairs_of(K + 4, K + 3, K - 1, K - 2, K - 3)
    assert (summary["kept_from_archive"], summary["kept_inside_served_range"]) == (100, 100)
    assert (summary["revised_candles"], summary["new_revisions"]) == (1, 1)
    assert summary["new_chunks"] == [start_of(K + 4)]
    after = file_bytes(archive_root)
    name = f"{start_of(K + 3)}.csv"
    assert after[name].startswith(before[name])
    candles, sidecar = read_chunk("1h", K + 3, root=archive_root)
    assert candles == settled_candles(K + 3, server)
    assert (sidecar["rows"], sidecar["complete"], sidecar["archive_only"]["rows"]) == (
        1000,
        False,
        100,
    )
    assert read_revisions("1h", root=archive_root) == [revision_entry(archived, revised, server)]
    assert warnings_of(archive_root, "1h", server) == [
        revisions_warning("1h", [(target, server, server)]),
        held_warning("1h", start_of(K + 3), server, 100),
    ]


def test_sidecars_of_schema_1_are_read_and_refreshed_as_schema_2(archive_root):
    for k in (K + 1, K + 3):
        json_path = chunk_paths("1h", k, root=archive_root)[1]
        sidecar = json.loads(json_path.read_text())
        del sidecar["archive_only"]
        json_path.write_text(json.dumps(sidecar | {"schema_version": 1}))
    candles, sidecar = read_chunk("1h", K + 1, root=archive_root)
    assert sidecar["schema_version"] == 1 and candles == settled_candles(K + 1, SERVER)
    assert (
        merge_chunks("1h", root=archive_root)["provenance"]["chunks"][1]["archive_only_rows"] == 0
    )
    backfill(client_for(make_fake(SERVER + HOUR_MS)), "1h", root=archive_root)
    assert read_chunk("1h", K + 1, root=archive_root)[1]["schema_version"] == 1  # complete: kept
    _, refreshed = read_chunk("1h", K + 3, root=archive_root)
    assert refreshed["schema_version"] == 2 and refreshed["archive_only"]["rows"] == 0


def test_archive_only_candles_must_be_in_the_chunk(archive_root):
    json_path = chunk_paths("1h", K, root=archive_root)[1]
    sidecar = json.loads(json_path.read_text())
    unheld = [start_of(K), start_of(K) + HOUR_MS]  # chunk K holds candles from 250 h into it
    sidecar["archive_only"] = {"rows": 1, "ranges": [unheld]}
    json_path.write_text(json.dumps(sidecar))
    with pytest.raises(BackfillError, match="rows do not match the count and range"):
        read_chunk("1h", K, root=archive_root)


def test_max_chunks_limits_a_run_and_the_next_run_resumes_the_walk(tmp_path):
    fake = make_fake()
    summary = backfill(client_for(fake), "1h", root=tmp_path, max_chunks=2)
    assert chunk_calls(fake) == pairs_of(K + 3, K + 2)
    assert (summary["stop_reason"], summary["history_start_ms"]) == ("max_chunks", None)
    calls = len(fake.klines_calls)
    summary = backfill(client_for(fake), "1h", root=tmp_path)
    assert chunk_calls(fake, calls) == pairs_of(K + 3, K + 1, K, K - 1, K - 2, K - 3)
    assert summary["new_chunks"] == [start_of(K), start_of(K + 1)]
    assert summary["refreshed_chunks"] == [start_of(K + 3)]
    assert summary["history_start_ms"] == HISTORY_START


# -- the history start ----------------------------------------------------------------------------


def history_of(root: Path) -> dict[str, Any]:
    return merge_chunks("1h", root=root)["provenance"]["history_start"]


def test_the_history_start_a_walk_establishes_is_recorded_once(tmp_path):
    fake = make_fake()
    backfill(client_for(fake), "1h", root=tmp_path)
    assert history_relpath("1h") == "raw/bingx/history-1h.json"
    path = tmp_path / history_relpath("1h")
    record = read_history("1h", root=tmp_path)
    assert record == json.loads(path.read_text())
    assert {key: record[key] for key in record if key != "code"} == {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "source": "bingx",
        "symbol": "BTC-USDT",
        "interval": "1h",
        "stop_reason": "empty_chunks",
        "history_start_ms": HISTORY_START,
        "history_start_utc": utc_text(HISTORY_START),
        "oldest_chunk_start_ms": start_of(K),
        "server_time_ms": SERVER,
        "server_time_utc": utc_text(SERVER),
    }
    data = path.read_bytes()
    fake.advance(HOUR_MS)
    assert backfill(client_for(fake), "1h", root=tmp_path)["history_start_ms"] == HISTORY_START
    assert path.read_bytes() == data  # the same start: the record keeps the run that found it
    assert history_of(tmp_path) == {
        "established": True,
        "history_start_ms": HISTORY_START,
        "history_start_utc": utc_text(HISTORY_START),
        "stop_reason": "empty_chunks",
        "server_time_ms": SERVER,
        "relpath": "raw/bingx/history-1h.json",
        "reason": None,
    }


def test_a_bounded_run_leaves_the_history_start_open_and_never_fakes_it(tmp_path):
    fake = make_fake()
    backfill(client_for(fake), "1h", root=tmp_path, max_chunks=2)
    assert read_history("1h", root=tmp_path) is None
    history = history_of(tmp_path)
    assert (history["established"], history["history_start_ms"]) == (False, None)
    assert history["reason"].startswith("no backfill has established the start")
    backfill(client_for(fake), "1h", root=tmp_path)  # the whole walk establishes it
    assert history_of(tmp_path)["history_start_ms"] == HISTORY_START
    fake.advance(1000 * HOUR_MS)  # later, a bounded run archives newer chunks only
    summary = backfill(client_for(fake), "1h", root=tmp_path, max_chunks=1)
    assert (summary["stop_reason"], summary["history_start_ms"]) == ("max_chunks", None)
    assert history_of(tmp_path)["established"] is True
    assert read_history("1h", root=tmp_path)["server_time_ms"] == SERVER


def test_a_start_inside_the_oldest_chunk_is_established_without_refetching_it(tmp_path):
    fake = make_fake()
    backfill(client_for(fake), "1h", root=tmp_path, max_chunks=4)  # K + 3 .. K, then stops
    assert read_history("1h", root=tmp_path) is None
    calls = len(fake.klines_calls)
    summary = backfill(client_for(fake), "1h", root=tmp_path)
    assert chunk_calls(fake, calls) == pairs_of(K + 3, K - 1, K - 2, K - 3)
    # chunk K's first candle opens 250 h into it: its fetch found nothing older in the chunk
    assert summary["history_start_ms"] == HISTORY_START
    assert read_history("1h", root=tmp_path)["history_start_ms"] == HISTORY_START


def test_empty_chunks_below_a_bounded_archive_do_not_establish_a_start(tmp_path):
    # a bounded run archives K + 3 and K + 2, whose first candle opens at its chunk's start; by
    # the next whole walk, everything older rolled off the exchange's retention
    fake = make_fake()
    backfill(client_for(fake), "1h", root=tmp_path, max_chunks=2)
    fake.history_start_ms["1h"] = start_of(K + 2) + 10 * HOUR_MS
    fake.advance(100 * HOUR_MS)
    calls = len(fake.klines_calls)
    summary = backfill(client_for(fake), "1h", root=tmp_path)
    assert chunk_calls(fake, calls) == pairs_of(K + 3, K + 1, K, K - 1)
    assert summary["stop_reason"] == "empty_chunks"
    assert summary["history_start_ms"] is None  # the start was never seen
    assert read_history("1h", root=tmp_path) is None
    assert history_of(tmp_path)["established"] is False


def test_a_server_clock_that_went_backwards_is_refused(archive_root):
    fake = make_fake(SERVER - 1)
    with pytest.raises(BackfillError, match="went backwards"):
        backfill(client_for(fake), "1h", root=archive_root)
    assert fake.klines_calls == []


# -- archive integrity ----------------------------------------------------------------------------


def _paths(root: Path, k: int = K + 1) -> tuple[Path, Path]:
    return chunk_paths("1h", k, root=root)


def _edit_sidecar(root: Path, **changes: Any) -> None:
    json_path = _paths(root)[1]
    sidecar = json.loads(json_path.read_text())
    json_path.write_text(json.dumps(sidecar | changes))


def _rewrite_csv(root: Path, data: bytes, **changes: Any) -> None:
    """Replace chunk K + 1's CSV and keep its sidecar consistent with the new bytes."""
    _paths(root)[0].write_bytes(data)
    _edit_sidecar(root, csv_sha256=sha256_bytes(data), csv_size=len(data), **changes)


def _flip_a_digit(root: Path) -> None:
    csv_path = _paths(root)[0]
    data = bytearray(csv_path.read_bytes())
    data[-3] = ord("1") if data[-3] != ord("1") else ord("2")
    csv_path.write_bytes(bytes(data))


def _row_outside(root: Path) -> None:
    data = _paths(root)[0].read_bytes()
    _rewrite_csv(root, data + f"{start_of(K + 2)},1.0,1.0,1.0,1.0,1.0\n".encode(), rows=1001)


def _row_repeated(root: Path) -> None:
    lines = _paths(root)[0].read_bytes().splitlines(keepends=True)
    _rewrite_csv(root, b"".join([*lines[:3], lines[2], *lines[3:]]), rows=1001)


def _row_with_exponent(root: Path) -> None:
    data = _paths(root)[0].read_bytes().replace(b"," + VECTOR.encode(), b",4.6e8")
    _rewrite_csv(root, data)


def _without_archive_only(root: Path) -> None:
    json_path = _paths(root)[1]
    sidecar = json.loads(json_path.read_text())
    del sidecar["archive_only"]
    json_path.write_text(json.dumps(sidecar))


def _edit_history(root: Path, **changes: Any) -> None:
    path = root / history_relpath("1h")
    path.write_text(json.dumps(json.loads(path.read_text()) | changes))


OFF_GRID = start_of(K + 1) + 30 * MINUTE_MS


Corruption = list[tuple[str, Callable[[Path], None], str]]
# Files, sidecars and hashes: the backfill checks the whole archive before its first request.
FILE_CORRUPTIONS: Corruption = [
    ("no sidecar", lambda r: _paths(r)[1].unlink(), r"\.csv has no sidecar"),
    ("no csv", lambda r: _paths(r)[0].unlink(), r"\.json has no CSV"),
    ("flipped digit", _flip_a_digit, "size or sha256 differs from its sidecar"),
    ("truncated csv", lambda r: _paths(r)[0].write_bytes(_paths(r)[0].read_bytes()[:-1]), "sha256"),
    ("bad json", lambda r: _paths(r)[1].write_text("{"), "not a valid sidecar"),
    (
        "sidecar of another chunk",
        lambda r: _paths(r)[1].write_bytes(_paths(r, K + 2)[1].read_bytes()),
        "does not describe this chunk",
    ),
    ("boolean schema", lambda r: _edit_sidecar(r, schema_version=True), "does not describe"),
    ("unknown schema", lambda r: _edit_sidecar(r, schema_version=3), "does not describe"),
    ("boolean rows", lambda r: _edit_sidecar(r, rows=True), "malformed"),
    ("empty chunk", lambda r: _edit_sidecar(r, rows=0), "never holds an empty chunk"),
    (
        "archive_only count",
        lambda r: _edit_sidecar(r, archive_only={"rows": 1, "ranges": []}),
        "archive_only is missing or malformed",
    ),
    (
        "archive_only off the grid",
        lambda r: _edit_sidecar(r, archive_only={"rows": 1, "ranges": [[OFF_GRID, OFF_GRID + 1]]}),
        "archive_only is missing or malformed",
    ),
    ("archive_only missing", _without_archive_only, "archive_only is missing or malformed"),
    (
        "bad history record",
        lambda r: (r / history_relpath("1h")).write_text("{"),
        "not a valid history record",
    ),
    (
        "history record of another chunk",
        lambda r: _edit_history(r, oldest_chunk_start_ms=start_of(K + 1)),
        "does not describe the 1h history start",
    ),
    (
        "history record of another interval",
        lambda r: _edit_history(r, interval="5m"),
        "does not describe the 1h history start",
    ),
    ("stray file", lambda r: (archive_dir(r) / "notes.txt").write_text("x"), "unexpected entry"),
    ("leading zero", lambda r: (archive_dir(r) / "0480.csv").write_bytes(b""), "unexpected entry"),
    ("off the grid", lambda r: (archive_dir(r) / "123.csv").write_bytes(b""), "chunk grid"),
    ("directory", lambda r: (archive_dir(r) / f"{start_of(K - 9)}.csv").mkdir(), "unexpected"),
]
# Rows under a matching hash (a writer bug or a deliberate edit): every read parses strictly.
CONTENT_CORRUPTIONS: Corruption = [
    ("row outside the chunk", _row_outside, "outside the chunk or off the grid"),
    ("repeated row", _row_repeated, "not strictly increasing"),
    ("exponent", _row_with_exponent, "malformed row"),
    ("wrong row count", lambda r: _edit_sidecar(r, rows=999), "do not match the count"),
]


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [pytest.param(corrupt, message, id=name) for name, corrupt, message in FILE_CORRUPTIONS],
)
def test_an_inconsistent_archive_stops_backfill_and_merge_before_anything_else(
    archive_root, corrupt, message
):
    corrupt(archive_root)
    fake = make_fake(SERVER + HOUR_MS)
    with pytest.raises(BackfillError, match=message):
        backfill(client_for(fake), "1h", root=archive_root)
    assert fake.requests == []  # not even the server time
    with pytest.raises(BackfillError, match=message):
        merge_chunks("1h", root=archive_root)
    assert not StoreId("bingx", "1h").meta_path(archive_root).exists()


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [pytest.param(corrupt, message, id=name) for name, corrupt, message in CONTENT_CORRUPTIONS],
)
def test_archived_rows_are_parsed_strictly_on_every_read(archive_root, corrupt, message):
    corrupt(archive_root)
    with pytest.raises(BackfillError, match=message):
        read_chunk("1h", K + 1, root=archive_root)
    with pytest.raises(BackfillError, match=message):
        merge_chunks("1h", root=archive_root)
    assert not StoreId("bingx", "1h").meta_path(archive_root).exists()


def test_hidden_temporary_files_are_ignored(archive_root):
    (archive_dir(archive_root) / f".{start_of(K + 1)}.csv.x1y2.tmp").write_bytes(b"partial")
    (archive_dir(archive_root) / ".DS_Store").write_bytes(b"\0")
    fake = make_fake()
    summary = backfill(client_for(fake), "1h", root=archive_root)
    assert summary["chunks_skipped"] == 3
    assert merge_chunks("1h", root=archive_root)["rows"] == 3150


def test_taking_the_lock_removes_the_temporary_files_killed_runs_left(
    archive_root, tmp_path, caplog
):
    leftovers = {  # what mkstemp names a chunk, record and store write of a killed run
        f"raw/bingx/1h/.{start_of(K + 3)}.csv.abcdefgh.tmp": b"partial chunk",
        "raw/bingx/.history-1h.json.k2_9x0q1.tmp": b"{",
        "candles/bingx/.1h.csv.zz11yy22.tmp": b"x" * 1000,
    }
    untouched = {
        "raw/bingx/1h/.DS_Store": b"\0",
        f"raw/bingx/1h/.{start_of(K + 1)}.csv.x1y2.tmp": b"not a mkstemp name",
        "candles/binance/.1m.csv.abcdefgh.tmp": b"written under the Binance lock",
    }
    for relpath, data in {**leftovers, **untouched}.items():
        path = archive_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not ours")
    (archive_root / "candles/bingx/.1h.meta.json.abcdefgh.tmp").symlink_to(outside)
    before = file_bytes(archive_root)
    with exclusive_lock(archive_root / "raw" / "bingx" / ".lock"), pytest.raises(LockBusyError):
        backfill(client_for(make_fake()), "1h", root=archive_root)  # a live run may own them
    assert all((archive_root / relpath).exists() for relpath in leftovers)
    summary = backfill(client_for(make_fake()), "1h", root=archive_root)
    assert sorted(summary["removed_temporaries"], key=lambda item: item["relpath"]) == [
        {"relpath": relpath, "bytes": len(data)} for relpath, data in sorted(leftovers.items())
    ]
    assert not any((archive_root / relpath).exists() for relpath in leftovers)
    assert all((archive_root / relpath).read_bytes() == data for relpath, data in untouched.items())
    assert (archive_root / "candles/bingx/.1h.meta.json.abcdefgh.tmp").is_symlink()
    assert outside.read_bytes() == b"not ours"
    after = file_bytes(archive_root)
    assert all(after[name] == data for name, data in before.items() if name.endswith(".csv"))
    assert (archive_root / "raw" / "bingx" / ".lock").exists()  # never deleted
    (archive_root / "candles/bingx/.1h.meta.json.qwertyui.tmp").write_bytes(b"{}")
    with caplog.at_level("WARNING", logger=bh.__name__):
        merge_chunks("1h", root=archive_root)
    assert not (archive_root / "candles/bingx/.1h.meta.json.qwertyui.tmp").exists()
    assert "removed 1 temporary files of killed runs (2 bytes)" in caplog.text


def test_read_chunk_of_a_chunk_never_archived(tmp_path):
    with pytest.raises(BackfillError, match="is not archived"):
        read_chunk("1h", K, root=tmp_path)


def test_chunk_paths_live_under_the_data_root(tmp_path, monkeypatch):
    csv_path, json_path = chunk_paths("15m", 7, root=tmp_path)
    assert csv_path == tmp_path / "raw" / "bingx" / "15m" / f"{7 * 900_000_000}.csv"
    assert json_path == csv_path.with_suffix(".json")
    monkeypatch.setenv("KRONOS_AGENT_DATA_DIR", str(tmp_path))
    default_csv, _ = chunk_paths("1h", K)
    assert default_csv.resolve() == (archive_dir(tmp_path) / f"{start_of(K)}.csv").resolve()


# -- merge, lock, arguments, retention ------------------------------------------------------------


def test_merge_builds_the_store_from_every_chunk_incomplete_ones_included(archive_root):
    meta = merge_chunks("1h", root=archive_root)
    store_id = StoreId("bingx", "1h")
    assert meta == read_meta(store_id, archive_root)
    assert (meta["rows"], meta["first_open_ms"]) == (3150, HISTORY_START)
    assert meta["last_open_ms"] == start_of(K + 3) + 399 * HOUR_MS
    assert (meta["gaps"], meta["missing_candles"]) == ([], 0)
    provenance = meta["provenance"]
    assert provenance["incomplete_chunks"] == [f"raw/bingx/1h/{start_of(K + 3)}.csv"]
    assert [c["chunk_start_ms"] for c in provenance["chunks"]] == [
        start_of(k) for k in range(K, K + 4)
    ]
    assert [c["complete"] for c in provenance["chunks"]] == [True, True, True, False]
    assert [c["rows"] for c in provenance["chunks"]] == [750, 1000, 1000, 400]
    assert [c["archive_only_rows"] for c in provenance["chunks"]] == [0, 0, 0, 0]
    assert provenance["history_start"]["established"] is True
    assert provenance["history_start"]["history_start_ms"] == meta["first_open_ms"]
    for chunk in provenance["chunks"]:
        data = (archive_root / chunk["relpath"]).read_bytes()
        assert chunk["csv_sha256"] == sha256_bytes(data)
    # the store is the chunk rows concatenated, byte for byte
    bodies = [_paths(archive_root, k)[0].read_bytes()[len(HEADER) :] for k in range(K, K + 4)]
    assert store_id.csv_path(archive_root).read_bytes() == HEADER + b"".join(bodies)
    frame = read_store(store_id, root=archive_root)
    expected = [c for c in CANDLES if c.open_ms <= start_of(K + 3) + 399 * HOUR_MS]
    assert fakes.frame_to_candles(frame) == expected


def test_merge_refuses_an_empty_archive(tmp_path):
    with pytest.raises(BackfillError, match="no archived 1h chunks"):
        merge_chunks("1h", root=tmp_path)
    assert not StoreId("bingx", "1h").meta_path(tmp_path).exists()


def test_backfill_and_merge_refuse_to_run_while_the_archive_is_locked(archive_root):
    fake = make_fake()
    with exclusive_lock(archive_root / "raw" / "bingx" / ".lock"):
        with pytest.raises(LockBusyError):
            backfill(client_for(fake), "1h", root=archive_root)
        with pytest.raises(LockBusyError):
            merge_chunks("1h", root=archive_root)
    assert fake.requests == []
    assert not StoreId("bingx", "1h").meta_path(archive_root).exists()
    backfill(client_for(fake), "1h", root=archive_root)  # released: runs
    merge_chunks("1h", root=archive_root)


@pytest.mark.parametrize(
    ("interval", "kwargs"),
    [
        ("2h", {}),
        ("1d", {}),
        ("1h", {"symbol": "ETH-USDT"}),
        ("1h", {"max_chunks": 0}),
        ("1h", {"max_chunks": True}),
        ("1h", {"max_chunks": 1.5}),
    ],
)
def test_backfill_refuses_bad_arguments(tmp_path, interval, kwargs):
    fake = make_fake()
    with pytest.raises(ValueError):
        backfill(client_for(fake), interval, root=tmp_path, **kwargs)
    assert fake.requests == []


def test_retention_warnings(tmp_path):
    assert RETENTION_DAYS == {"1m": 450, "5m": 48}
    assert retention_warnings(tmp_path, now_ms=SERVER) == [
        (
            f"bingx {interval}: no archived chunks; the real market keeps about {days} days of "
            f"{interval} candles, so run `kronos-agent data bingx --interval {interval}` before "
            "they roll off"
        )
        for interval, days in (("1m", 450), ("5m", 48))
    ]
    step = 300_000
    span = CHUNK_CANDLES * step
    first = 5760 * span + 100 * step  # 5m chunk 5760 opens at 2024-10-04T00:00Z
    candles = fakes.synthetic_candles(first, 1600, interval="5m", seed=3)
    server_time_ms = 5761 * span + 300 * step + 2 * MINUTE_MS
    fake = fakes.FakeBingX(server_time_ms, {"5m": candles})
    summary = backfill(client_for(fake), "5m", root=tmp_path)
    newest = summary["archive"]["last_open_ms"]
    assert newest == server_time_ms - 2 * MINUTE_MS - step  # closed 2 min before the server time
    third = 16 * DAY_MS
    assert [w for w in retention_warnings(tmp_path, now_ms=newest + third) if "5m" in w] == []
    late = [w for w in retention_warnings(tmp_path, now_ms=newest + third + 1) if "5m" in w]
    assert late == [
        (
            f"bingx 5m: the newest archived candle ({utc_text(newest)}) opened 16.0 days ago, "
            "more than a third of the observed 48-day retention; run "
            "`kronos-agent data bingx --interval 5m`"
        )
    ]
    chunk_paths("5m", 5760, root=tmp_path)[1].unlink()
    broken = [w for w in retention_warnings(tmp_path, now_ms=newest) if "5m" in w]
    assert len(broken) == 1 and "cannot be read" in broken[0] and "has no sidecar" in broken[0]


def test_module_constants_match_the_design():
    constants = (CHUNK_CANDLES, SETTLE_MS, MAX_FETCH_PAIRS, STOP_AFTER_EMPTY_CHUNKS)
    assert constants == (1000, 90_000, 3, 3)
    assert (SIDECAR_SCHEMA_VERSION, bh.SIDECAR_SCHEMA_VERSIONS, HISTORY_SCHEMA_VERSION) == (
        2,
        (1, 2),
        1,
    )
    assert (REVISION_SCHEMA_VERSION, bh.REVISION_SCHEMA_VERSIONS) == (2, (1, 2))
    assert bh.LISTED_REVISIONS == 10
    assert bh.LOCK_RELPATH == "raw/bingx/.lock" and bh.ARCHIVE_RELDIR == "raw/bingx"


# -- the exchange fake: the klines bounds observed live (D-041) ------------------------------------


class InclusiveEndBingX(fakes.FakeBingX):
    """The semantic D-041 ruled out: an ``endTime`` exactly at a candle's open includes it."""

    def served(self, interval, start_ms, end_ms, limit=fakes.MAX_KLINES_LIMIT):
        return super().served(interval, start_ms, None if end_ms is None else end_ms + 1, limit)


def opens_of(candles: list[Candle]) -> list[int]:
    return [c.open_ms for c in candles]


def test_the_fake_serves_the_klines_bounds_observed_live():
    client = client_for(make_fake())
    start, end = chunk_bounds(K + 1, "1h")
    last_open = end - HOUR_MS
    # D-041 on a whole closed chunk: endTime = next chunk start - 1 ms returns its 1000 candles,
    # endTime exactly at the last candle's open only the first 999
    assert opens_of(client.klines("BTC-USDT", "1h", start, end - 1)) == list(
        range(start, end, HOUR_MS)
    )
    assert opens_of(client.klines("BTC-USDT", "1h", start, last_open)) == list(
        range(start, last_open, HOUR_MS)
    )
    assert opens_of(client.klines("BTC-USDT", "1h", start, start + HOUR_MS)) == [start]
    # the forming candle (SERVER is 30 min into it) is served once it opened, unless endTime is
    # exactly its open; nothing after the server time is ever served
    forming = start_of(K + 3) + 400 * HOUR_MS
    assert opens_of(client.klines("BTC-USDT", "1h", forming - HOUR_MS, SERVER)) == [
        forming - HOUR_MS,
        forming,
    ]
    assert opens_of(client.klines("BTC-USDT", "1h", forming - HOUR_MS, forming)) == [
        forming - HOUR_MS
    ]
    assert opens_of(client.klines("BTC-USDT", "1h", forming, forming + 10 * HOUR_MS)) == [forming]
    # a range before the history start is code 0 with an empty list
    assert client.klines("BTC-USDT", "1h", start_of(K - 1), start_of(K) - 1) == []
    inclusive = client_for(InclusiveEndBingX(SERVER, {"1h": CANDLES}))
    assert opens_of(inclusive.klines("BTC-USDT", "1h", start, last_open))[-1] == last_open


@pytest.mark.parametrize(
    "server_time_ms",
    [
        SERVER,  # mid-candle
        start_of(K + 3) + 400 * HOUR_MS,  # exactly at the forming candle's open
        start_of(K + 3) + 400 * HOUR_MS + SETTLE_MS,
        start_of(K + 4) + SETTLE_MS,  # the newest chunk just completed
    ],
)
def test_the_backfill_never_depends_on_an_exact_boundary_end_time(tmp_path, server_time_ms):
    archives = []
    for name, kind in (("observed", fakes.FakeBingX), ("inclusive", InclusiveEndBingX)):
        fake = kind(server_time_ms, {"1h": CANDLES})
        root = tmp_path / name
        summary = backfill(client_for(fake), "1h", root=root)
        assert all(call.end_ms % HOUR_MS != 0 for call in fake.klines_calls)
        files: dict[str, Any] = {}
        for path in sorted(archive_dir(root).iterdir()):
            if path.suffix == ".json":
                files[path.name] = json.loads(path.read_text())
                del files[path.name]["fetched_at"]  # the wall clock of the run
            else:
                files[path.name] = path.read_bytes()
        calls = [(c.start_ms, c.end_ms, c.limit) for c in fake.klines_calls]
        archives.append((summary, files, calls, read_history("1h", root=root)))
    assert archives[0] == archives[1]
