"""Live checks against the PUBLIC BingX host only (marker ``network``, opt-in).

Run once with ``KRONOS_AGENT_NETWORK_TESTS=1``. One module-scoped client paces every request at
>= 2 s. With ``startTime`` inclusive at the exact candle boundary the module performs 8
unauthenticated GETs, 11 if it is exclusive (``closed_klines`` then asks for one extra, empty page
per call); retries add more. The demo host is never contacted: the ``conftest`` guard keeps it
unresolvable even here, and the last test proves it.
"""

from __future__ import annotations

import socket
import time
from itertools import pairwise
from typing import Any, NamedTuple

import pytest

from kronos_agent.exchange.public_market import (
    INTERVAL_MS,
    MAX_KLINES_PER_REQUEST,
    Candle,
    DataGapError,
    PublicMarketClient,
)

pytestmark = pytest.mark.network

SYMBOL = "BTC-USDT"
HOUR_MS = INTERVAL_MS["1h"]
EIGHT_HOURS_MS = 8 * HOUR_MS
FIVE_MINUTES_MS = 5 * 60_000
DEMO_HOST = "open-api-vst.bingx.com"


class Page(NamedTuple):
    """One klines page as requested and received (``None`` bounds for an empty page)."""

    start_ms: int
    end_ms: int
    count: int
    oldest_ms: int | None
    newest_ms: int | None


class PageRecordingClient(PublicMarketClient):
    """The production client, default transport and pin unchanged, plus a log of klines pages.

    Pages are logged after the client's own retries, so a rate-limit blip never adds one.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pages: list[Page] = []

    def klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = MAX_KLINES_PER_REQUEST,
    ) -> list[Candle]:
        page = super().klines(symbol, interval, start_ms, end_ms, limit)
        opens = [c.open_ms for c in page]
        oldest, newest = (opens[0], opens[-1]) if opens else (None, None)
        self.pages.append(Page(start_ms, end_ms, len(page), oldest, newest))
        return page


@pytest.fixture(scope="module")
def client():
    with PageRecordingClient(timeout=15.0) as live_client:
        yield live_client


@pytest.fixture(scope="module")
def server_time(client) -> tuple[int, int]:
    """``(server_time_ms, local_time_ms)`` sampled together, once for the module."""
    server_ms = client.server_time_ms()
    local_ms = int(time.time() * 1000)
    return server_ms, local_ms


def test_server_time_within_five_seconds_of_local_clock(server_time, record_property):
    server_ms, local_ms = server_time
    offset_ms = server_ms - local_ms
    record_property("bingx_server_time_offset_ms", offset_ms)
    assert abs(offset_ms) < 5_000, f"offset {offset_ms} ms"


def test_last_48_hours_of_closed_hourly_candles_are_contiguous_and_closed(
    client, server_time, record_property
):
    server_ms, _ = server_time
    start_ms = ((server_ms - 48 * HOUR_MS) // HOUR_MS) * HOUR_MS
    last_closed_open_ms = (server_ms // HOUR_MS) * HOUR_MS - HOUR_MS
    before = len(client.pages)
    candles = client.fetch_twice(SYMBOL, "1h", start_ms, server_ms, server_time_ms=server_ms)
    pages = len(client.pages) - before
    first = client.pages[before]
    record_property("closed_hourly_candles_in_48h", len(candles))
    record_property("klines_pages_in_48h_fetch_twice", pages)
    record_property("klines_first_page_count_in_48h", first.count)
    record_property("start_time_inclusive_at_exact_boundary", first.oldest_ms == start_ms)
    # startTime honoured: the first raw page starts at start_ms (inclusive) or one hour later
    # (exclusive). An ignored startTime would fill the page back from endTime (1000 candles),
    # which the candle count alone cannot reveal because closed_klines trims to the window.
    assert first.oldest_ms in (start_ms, start_ms + HOUR_MS), client.pages[before:]
    # 48 candles in one page per fetch when startTime is inclusive at the exact boundary; 47 if the
    # exchange treats it as exclusive, and then each fetch asks for one extra page that is empty.
    assert (len(candles), pages) in {(48, 2), (47, 4)}, client.pages[before:]
    assert candles[0].open_ms in (start_ms, start_ms + HOUR_MS)
    assert candles[-1].open_ms == last_closed_open_ms
    assert all(b.open_ms - a.open_ms == HOUR_MS for a, b in pairwise(candles))
    assert all(c.close_ms(HOUR_MS) <= server_ms for c in candles)
    assert all(c.open_ms % HOUR_MS == 0 for c in candles)
    for candle in candles:
        o, h, lo, c, v = candle.as_floats()
        assert lo <= min(o, c) <= max(o, c) <= h
        assert v > 0


def test_two_page_backfill_of_1100_closed_hours_is_contiguous_across_the_seam(
    client, server_time, record_property
):
    # 1100 closed hours need two pages (three if startTime is exclusive at the exact boundary: the
    # last one empty). The seam endTime = oldest_open - 1 must return the candle just before the
    # previous page's oldest as the next page's newest: no duplicate, no gap.
    server_ms, _ = server_time
    last_closed_open_ms = (server_ms // HOUR_MS) * HOUR_MS - HOUR_MS
    start_ms = last_closed_open_ms - 1099 * HOUR_MS
    before = len(client.pages)
    try:
        candles = client.closed_klines(SYMBOL, "1h", start_ms, server_ms, server_time_ms=server_ms)
    except DataGapError as exc:
        pytest.fail(
            f"{exc}; pages {client.pages[before:]}. A duplicate or gap at a page boundary means "
            "BingX's endTime semantic differs from the one the pager assumes; one inside a page is "
            "a gap in the exchange's own history"
        )
    pages = client.pages[before:]
    record_property("closed_hourly_candles_in_1100h", len(candles))
    record_property("klines_pages_in_1100h", len(pages))
    assert pages[0].count == MAX_KLINES_PER_REQUEST, f"changed fact: limit cap/anchor ({pages})"
    assert pages[0].start_ms == start_ms and pages[0].end_ms == server_ms, pages
    assert pages[1].end_ms == pages[0].oldest_ms - 1, pages
    # The seam fact this test exists for (first observed live on 2026-10-31).
    assert pages[1].newest_ms == pages[0].oldest_ms - HOUR_MS, pages
    assert pages[1].oldest_ms in (start_ms, start_ms + HOUR_MS), pages  # startTime honoured
    assert (len(candles), len(pages)) in {(1100, 2), (1099, 3)}, pages
    assert candles[0].open_ms in (start_ms, start_ms + HOUR_MS)
    assert candles[-1].open_ms == last_closed_open_ms
    assert all(b.open_ms - a.open_ms == HOUR_MS for a, b in pairwise(candles))
    assert all(c.close_ms(HOUR_MS) <= server_ms for c in candles)


def test_contract_precisions_and_fees(client):
    contract = client.contract(SYMBOL)
    assert int(contract["quantityPrecision"]) == 4
    assert int(contract["pricePrecision"]) == 1
    assert float(contract["tradeMinQuantity"]) == pytest.approx(0.0001)
    assert float(contract["takerFeeRate"]) == pytest.approx(0.0005)
    assert float(contract["makerFeeRate"]) == pytest.approx(0.0002)


def test_premium_index_funding_interval(client, server_time):
    server_ms, _ = server_time
    premium = client.premium_index(SYMBOL)
    assert int(premium["fundingIntervalHours"]) in {1, 2, 4, 8}
    assert int(premium["nextFundingTime"]) > server_ms
    assert float(premium["markPrice"]) > 0


def test_funding_history_has_eight_hour_spacing(client):
    items = client.funding_rate_history(SYMBOL, limit=10)
    assert len(items) >= 3
    assert all(item["symbol"] == SYMBOL for item in items)
    times = sorted(int(item["fundingTime"]) for item in items)
    for earlier, later in pairwise(times):
        assert abs((later - earlier) - EIGHT_HOURS_MS) <= FIVE_MINUTES_MS


def test_demo_host_is_unresolvable_even_with_network_enabled():
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo(DEMO_HOST, 443)
