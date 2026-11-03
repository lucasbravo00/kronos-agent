"""Live checks of the backfill's request bounds against the PUBLIC real-market host (opt-in).

Marker ``network``; run once with ``KRONOS_AGENT_NETWORK_TESTS=1``. One module-scoped
``PublicMarketClient`` (default pacing, >= 2 s between requests) performs 7 unauthenticated GETs,
more only if the client retries: the server time; two pages of one long-closed 1h chunk inside
the history, requested with ``endTime`` at the next chunk start minus 1 ms (the backfill's form)
and at the exact boundary of its last open (whether that form includes the last candle is
recorded, never relied on); the two fetches of that chunk through ``fetch_chunk``; one page of
the chunk entirely before the 1h history start (``code 0`` and an empty list, the backfill's stop
signal); and one page of the chunk holding the history start. ``client.klines`` is called
directly: ``closed_klines`` and ``fetch_twice`` refuse internal gaps. The demo host is never
contacted: the ``conftest`` guard keeps it unresolvable even here.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from kronos_agent.data.bingx_history import (
    CHUNK_CANDLES,
    chunk_bounds,
    chunk_index,
    fetch_chunk,
    request_bounds,
)
from kronos_agent.data.timeutil import utc_text
from kronos_agent.exchange.public_market import (
    Candle,
    PublicMarketClient,
    canonical_bytes,
    interval_ms,
)

pytestmark = pytest.mark.network

SYMBOL = "BTC-USDT"
INTERVAL = "1h"
HOUR_MS = interval_ms(INTERVAL)
HISTORY_START_MS = 1_714_212_000_000  # 2024-04-27T10:00Z, the 1h history start observed in step 1
CLOSED_CHUNK = 480  # [2024-10-04T00:00Z, 2024-11-14T16:00Z): long closed, inside the history
HISTORY_CHUNK = chunk_index(HISTORY_START_MS, INTERVAL)  # 476, [2024-04-20T08:00Z, 2024-06-01)
PRE_HISTORY_CHUNK = HISTORY_CHUNK - 1  # 475: entirely before the history start


@pytest.fixture(scope="module")
def client():
    with PublicMarketClient(timeout=15.0) as live_client:
        yield live_client


@pytest.fixture(scope="module")
def server_time_ms(client) -> int:
    return client.server_time_ms()


@pytest.fixture(scope="module")
def seam_page(client, server_time_ms) -> list[Candle]:
    """The closed chunk as the backfill requests it: ``endTime`` = next chunk start - 1 ms."""
    start, end = request_bounds(CLOSED_CHUNK, INTERVAL, server_time_ms)
    assert (start, end + 1) == chunk_bounds(CLOSED_CHUNK, INTERVAL)
    return client.klines(SYMBOL, INTERVAL, start, end, CHUNK_CANDLES)


def test_end_time_forms_of_a_closed_chunk(client, seam_page, record_property):
    start, end = chunk_bounds(CLOSED_CHUNK, INTERVAL)
    last_open = end - HOUR_MS
    opens = [c.open_ms for c in seam_page]
    record_property("closed_chunk_start", utc_text(start))
    record_property("closed_chunk_candles", len(opens))
    # next start - 1 ms holds the whole range: startTime inclusive, the last open included, and
    # no truncation (a page cut at the limit would lose its OLDEST candles: limit anchors at end)
    assert opens, "no candles for a closed range inside the history"
    assert (opens[0], opens[-1]) == (start, last_open), (utc_text(opens[0]), utc_text(opens[-1]))
    assert all(start <= o < end and o % HOUR_MS == 0 for o in opens)
    assert all(later > earlier for earlier, later in pairwise(opens))
    exact = client.klines(SYMBOL, INTERVAL, start, last_open, CHUNK_CANDLES)
    includes_last = bool(exact) and exact[-1].open_ms == last_open
    record_property("end_time_exact_boundary_includes_last_candle", includes_last)
    expected = seam_page if includes_last else seam_page[:-1]
    assert canonical_bytes(exact) == canonical_bytes(expected)


def test_fetch_chunk_of_a_closed_chunk_returns_the_seam_page(
    client, server_time_ms, seam_page, record_property
):
    fetch = fetch_chunk(client, INTERVAL, CLOSED_CHUNK, server_time_ms=server_time_ms)
    record_property("closed_chunk_fetch_pairs", fetch.pairs)
    assert fetch.complete
    assert fetch.requests == 2 * fetch.pairs
    assert canonical_bytes(fetch.candles) == canonical_bytes(seam_page)


def test_a_range_before_the_history_start_is_code_0_with_an_empty_list(client, server_time_ms):
    start, end = request_bounds(PRE_HISTORY_CHUNK, INTERVAL, server_time_ms)
    assert end < HISTORY_START_MS
    # code != 0 would raise BingXApiError: an empty list here means "no data", the stop signal
    assert client.klines(SYMBOL, INTERVAL, start, end, CHUNK_CANDLES) == []


def test_the_1h_history_starts_where_step_1_observed_it(client, server_time_ms, record_property):
    start, end = request_bounds(HISTORY_CHUNK, INTERVAL, server_time_ms)
    page = client.klines(SYMBOL, INTERVAL, start, end, CHUNK_CANDLES)
    first = page[0].open_ms if page else None
    record_property("history_start_1h", None if first is None else utc_text(first))
    assert first == HISTORY_START_MS, "changed fact: the real-market 1h history start moved"
