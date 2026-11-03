"""Unit tests of the public market client: mock transports, injected clock and sleep, no network.

The fixtures reproduce the shapes verified live on 2026-10-29: klines objects with six string
fields, newest first, the unclosed candle included first, ``limit`` capped at 1000 anchored at
``endTime`` and ``code != 0`` as the error channel.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from kronos_agent.exchange import public_market as pm
from kronos_agent.exchange.public_market import (
    ALLOWED_PATHS,
    INTERVAL_MS,
    PUBLIC_HOST,
    BingXApiError,
    BingXBannedError,
    BingXRateLimited,
    BingXSchemaError,
    BingXUnavailableError,
    Candle,
    CandleMismatchError,
    DataGapError,
    HostNotAllowedError,
    PublicMarketClient,
    UnexpectedRedirect,
    canonical_bytes,
    interval_ms,
    window_sha256,
)
from kronos_agent.hashing import canonical_json, sha256_bytes

HOUR_MS = 3_600_000
SYMBOL = "BTC-USDT"
# Verbatim from the live probe: serverTime and the 16:00Z candle that was still open at that time.
SERVER_TIME_MS = 1_793_290_785_970
LIVE_UNCLOSED_CANDLE: dict[str, Any] = {
    "open": "81731.2",
    "close": "81471.3",
    "high": "81943.3",
    "low": "81466.6",
    "volume": "339.1763",
    "time": 1_793_289_600_000,
}
# Hosts that must be refused; the demo host is only ever used here to assert that it is refused.
DEMO_HOST_REFUSED = "open-api-vst.bingx.com"


class FakeClock:
    """Monotonic clock that only advances through ``sleep``; records every sleep."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def ok(data: Any) -> httpx.Response:
    return httpx.Response(200, json={"code": 0, "msg": "", "data": data})


def api_error(code: int, msg: str = "error") -> httpx.Response:
    return httpx.Response(200, json={"code": code, "msg": msg})


def hourly_candles(
    count: int, last_open_ms: int = LIVE_UNCLOSED_CANDLE["time"]
) -> list[dict[str, Any]]:
    """``count`` synthetic hourly candle objects ending at ``last_open_ms``, ascending, string fields."""
    out: list[dict[str, Any]] = []
    for i in range(count):
        open_ms = last_open_ms - (count - 1 - i) * HOUR_MS
        base = 80_000 + (i % 97) * 3
        out.append(
            {
                "open": f"{base}.0",
                "close": f"{base + 1}.5",
                "high": f"{base + 2}.0",
                "low": f"{base - 1}.0",
                "volume": f"{100 + i}.25",
                "time": open_ms,
            }
        )
    return out


def by_open_time(open_ms: int, start: int, end: int) -> bool:
    """The bounds observed live: ``startTime`` inclusive at an exact boundary (2026-10-31) and an
    ``endTime`` exactly at a candle's open excluding that candle (2026-11-01, D-041), no
    rounding: ``startTime <= open < endTime``."""
    return start <= open_ms < end


def start_time_exclusive(open_ms: int, start: int, end: int) -> bool:
    return start < open_ms < end


def end_time_inclusive(open_ms: int, start: int, end: int) -> bool:
    """What step 1 assumed before D-041: an ``endTime`` at a candle's open includes it."""
    return start <= open_ms <= end


def end_time_rounded_up(open_ms: int, start: int, end: int) -> bool:
    return start <= open_ms <= -(-end // HOUR_MS) * HOUR_MS


def end_time_against_close_time(open_ms: int, start: int, end: int) -> bool:
    return start <= open_ms and open_ms + HOUR_MS <= end


class FakeBingX:
    """Mock handler with the verified klines semantics: newest first, ``limit`` capped at 1000 at ``endTime``.

    ``selects(open_ms, startTime, endTime)`` picks the candles a klines request returns. The
    default (``startTime`` inclusive, an ``endTime`` exactly at a candle's open excluding it) is
    what the live modules observed on 2026-10-31 and 2026-11-01 (D-041); the alternatives above
    model what BingX could do instead.
    """

    def __init__(
        self,
        candles_asc: list[dict[str, Any]],
        server_time_ms: int = SERVER_TIME_MS,
        *,
        selects: Callable[[int, int, int], bool] = by_open_time,
    ):
        self.candles = candles_asc
        self.server_time_ms = server_time_ms
        self.selects = selects
        self.requests: list[httpx.Request] = []
        self.overrides: dict[str, Callable[[httpx.Request], httpx.Response]] = {}

    def klines_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == pm.KLINES_PATH]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.overrides:
            return self.overrides[path](request)
        if path == pm.SERVER_TIME_PATH:
            return ok({"serverTime": self.server_time_ms})
        if path == pm.KLINES_PATH:
            params = request.url.params
            start, end = int(params["startTime"]), int(params["endTime"])
            limit = min(int(params["limit"]), 1000)
            selected = [c for c in self.candles if self.selects(c["time"], start, end)]
            selected.sort(key=lambda c: c["time"], reverse=True)
            return ok(selected[:limit])
        return httpx.Response(404, text="not found")


def make_client(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[PublicMarketClient, FakeClock]:
    clock = FakeClock()
    client = PublicMarketClient(
        transport=httpx.MockTransport(handler), sleep=clock.sleep, clock=clock, **kwargs
    )
    return client, clock


def opens(candles: list[Candle]) -> list[int]:
    return [c.open_ms for c in candles]


# -- construction and allow-list --------------------------------------------------------------------


def test_constructor_has_no_host_url_or_headers_parameter():
    names = list(inspect.signature(PublicMarketClient).parameters)
    assert names == ["transport", "timeout", "min_interval_s", "max_retries", "sleep", "clock"]
    for name in names:
        assert not any(word in name.lower() for word in ("host", "url", "header", "base"))
    assert pm.PUBLIC_HOST == "open-api.bingx.com"


def test_allow_list_is_exactly_the_five_public_endpoints():
    assert ALLOWED_PATHS == {
        "/openApi/swap/v2/server/time",
        "/openApi/swap/v3/quote/klines",
        "/openApi/swap/v2/quote/contracts",
        "/openApi/swap/v2/quote/premiumIndex",
        "/openApi/swap/v2/quote/fundingRate",
    }


@pytest.mark.parametrize(
    "path",
    [
        "/openApi/swap/v3/user/balance",
        "/openApi/swap/v2/trade/order",
        "/openApi/swap/v2/quote/klines",
        "/openApi/swap/v3/quote/klines/",
        "openApi/swap/v2/server/time",
        "/",
        "",
    ],
)
def test_paths_outside_the_allow_list_raise_before_any_request(path):
    fake = FakeBingX([])
    client, clock = make_client(fake)
    with pytest.raises(ValueError, match="allow-list"):
        client._request_json(path)
    assert fake.requests == []
    assert clock.sleeps == []


def test_no_auth_header_is_ever_sent():
    fake = FakeBingX([])
    client, _ = make_client(fake)
    assert client.server_time_ms() == SERVER_TIME_MS
    headers = fake.requests[0].headers
    assert "x-bx-apikey" not in headers
    assert "authorization" not in headers
    assert "cookie" not in headers


def test_every_request_targets_https_public_host_default_port():
    fake = FakeBingX(hourly_candles(5))
    client, _ = make_client(fake)
    client.server_time_ms()
    client.klines(SYMBOL, "1h", fake.candles[0]["time"], fake.candles[-1]["time"])
    for request in fake.requests:
        assert request.url.scheme == "https"
        assert request.url.host == PUBLIC_HOST
        assert request.url.port is None
        assert request.url.userinfo == b""


# -- klines and closed windows ------------------------------------------------------------------------


def test_closed_klines_turns_descending_payload_with_unclosed_candle_into_ascending_closed():
    closed = hourly_candles(5, last_open_ms=LIVE_UNCLOSED_CANDLE["time"] - HOUR_MS)
    raw_newest_first = [LIVE_UNCLOSED_CANDLE, *reversed(closed)]
    fake = FakeBingX([])
    fake.overrides[pm.KLINES_PATH] = lambda request: ok(raw_newest_first)
    client, _ = make_client(fake)

    result = client.closed_klines(SYMBOL, "1h", closed[0]["time"], SERVER_TIME_MS)

    assert opens(result) == [c["time"] for c in closed]
    assert LIVE_UNCLOSED_CANDLE["time"] not in opens(result)
    assert result[-1].open_ms == 1_793_286_000_000  # 15:00Z, the last closed hour
    assert all(c.close_ms(HOUR_MS) <= SERVER_TIME_MS for c in result)
    assert result[0].open == closed[0]["open"] and result[0].volume == closed[0]["volume"]
    assert [r.url.path for r in fake.requests] == [pm.SERVER_TIME_PATH, pm.KLINES_PATH]
    params = fake.requests[1].url.params
    assert params["symbol"] == SYMBOL and params["interval"] == "1h"
    assert params["startTime"] == str(closed[0]["time"])
    assert params["endTime"] == str(SERVER_TIME_MS)
    assert params["limit"] == "1000"


def test_closed_klines_uses_given_server_time_without_requesting_it():
    candles = hourly_candles(4)
    fake = FakeBingX(candles)
    client, _ = make_client(fake)
    result = client.closed_klines(
        SYMBOL,
        "1h",
        candles[0]["time"],
        candles[-1]["time"] + HOUR_MS,  # half-open: up to the last candle's close
        server_time_ms=candles[-1]["time"] + HOUR_MS,
    )
    assert opens(result) == [c["time"] for c in candles]
    assert [r.url.path for r in fake.requests] == [pm.KLINES_PATH]


def test_klines_returns_ascending_and_keeps_raw_strings():
    candles = hourly_candles(6)
    fake = FakeBingX(candles)
    client, _ = make_client(fake)
    # the backfill's form: one millisecond before the next boundary (an exact one excludes)
    result = client.klines(SYMBOL, "1h", candles[0]["time"], candles[-1]["time"] + HOUR_MS - 1)
    assert opens(result) == sorted(opens(result)) == [c["time"] for c in candles]
    assert isinstance(result[0].open, str) and result[0].open == candles[0]["open"]
    assert result[0].as_floats() == (
        float(candles[0]["open"]),
        float(candles[0]["high"]),
        float(candles[0]["low"]),
        float(candles[0]["close"]),
        float(candles[0]["volume"]),
    )


@pytest.mark.parametrize(
    "start_ms, end_ms", [(None, 5), (5, None), (None, None), (10, 5), (-1, 5), (1.5, 5)]
)
def test_klines_requires_both_integer_bounds(start_ms, end_ms):
    fake = FakeBingX([])
    client, _ = make_client(fake)
    with pytest.raises(ValueError):
        client.klines(SYMBOL, "1h", start_ms, end_ms)
    assert fake.requests == []


@pytest.mark.parametrize("limit", [0, 1001, 1500, -5, True])
def test_klines_limit_is_capped_at_1000(limit):
    fake = FakeBingX([])
    client, _ = make_client(fake)
    with pytest.raises(ValueError, match="limit"):
        client.klines(SYMBOL, "1h", 0, 10, limit)
    assert fake.requests == []


def test_code_zero_with_empty_data_is_a_legitimate_empty_page():
    fake = FakeBingX([])
    client, _ = make_client(fake)
    assert client.klines(SYMBOL, "1h", 0, 10) == []


# -- errors, retries and backoff ----------------------------------------------------------------------------


@pytest.mark.parametrize("flavour", ["code_100410", "http_429"])
def test_rate_limit_retries_with_backoff_then_succeeds(flavour):
    candles = hourly_candles(3)
    fake = FakeBingX(candles)
    limited = (
        httpx.Response(429, text="too many")
        if flavour == "http_429"
        else api_error(100410, "rate limit")
    )
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return limited
        return ok(list(reversed(candles)))

    fake.overrides[pm.KLINES_PATH] = flaky
    client, clock = make_client(fake)
    result = client.klines(SYMBOL, "1h", candles[0]["time"], candles[-1]["time"])
    assert opens(result) == [c["time"] for c in candles]
    assert clock.sleeps == [10.0, 30.0]
    assert len(fake.requests) == 3


@pytest.mark.parametrize("flavour", ["code_100410", "http_429"])
def test_rate_limit_exhausts_backoff_then_raises(flavour):
    fake = FakeBingX([])
    limited = (
        httpx.Response(429, text="too many")
        if flavour == "http_429"
        else api_error(100410, "rate limit")
    )
    fake.overrides[pm.KLINES_PATH] = lambda request: limited
    client, clock = make_client(fake)
    with pytest.raises(BingXRateLimited):
        client.klines(SYMBOL, "1h", 0, 10)
    assert clock.sleeps == [10.0, 30.0, 90.0]
    assert len(fake.requests) == 4


@pytest.mark.parametrize("code", [100400, 80014, 109400, -1])
def test_other_api_codes_raise_and_never_look_like_no_data(code):
    fake = FakeBingX([])
    fake.overrides[pm.KLINES_PATH] = lambda request: httpx.Response(
        200, json={"code": code, "msg": "something", "data": []}
    )
    client, clock = make_client(fake)
    with pytest.raises(BingXApiError) as info:
        client.klines(SYMBOL, "1h", 0, 10)
    assert info.value.code == code
    assert info.value.path == pm.KLINES_PATH
    assert info.value.msg == "something"
    assert clock.sleeps == []
    assert len(fake.requests) == 1


def test_http_403_is_a_ban_without_retry():
    fake = FakeBingX([])
    fake.overrides[pm.SERVER_TIME_PATH] = lambda request: httpx.Response(403, text="forbidden")
    client, clock = make_client(fake)
    with pytest.raises(BingXBannedError):
        client.server_time_ms()
    assert len(fake.requests) == 1
    assert clock.sleeps == []


def test_other_4xx_statuses_raise_api_error_with_status():
    fake = FakeBingX([])
    fake.overrides[pm.SERVER_TIME_PATH] = lambda request: httpx.Response(404, text="nope")
    client, _ = make_client(fake)
    with pytest.raises(BingXApiError) as info:
        client.server_time_ms()
    assert info.value.status == 404 and info.value.code is None


def test_timeouts_retry_with_one_second_backoff_then_succeed():
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ReadTimeout("slow", request=request)
        return ok({"serverTime": SERVER_TIME_MS})

    client, clock = make_client(flaky, min_interval_s=0.0)
    assert client.server_time_ms() == SERVER_TIME_MS
    assert clock.sleeps == [1.0, 1.0]
    assert calls["n"] == 3


def test_timeouts_exhaust_max_retries():
    calls = {"n": 0}

    def always_timeout(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectTimeout("slow", request=request)

    client, clock = make_client(always_timeout, min_interval_s=0.0, max_retries=2)
    with pytest.raises(BingXUnavailableError) as info:
        client.server_time_ms()
    assert info.value.status is None
    assert calls["n"] == 3
    assert clock.sleeps == [1.0, 1.0]


def test_5xx_retries_then_raises_with_status():
    client, clock = make_client(
        lambda request: httpx.Response(503, text="down"), min_interval_s=0.0, max_retries=1
    )
    with pytest.raises(BingXUnavailableError) as info:
        client.server_time_ms()
    assert info.value.status == 503
    assert clock.sleeps == [1.0]


def test_redirect_is_refused_and_never_followed():
    redirect = httpx.Response(
        302, headers={"location": "https://open-api.bingx.pro/openApi/swap/v2/server/time"}
    )
    fake = FakeBingX([])
    fake.overrides[pm.SERVER_TIME_PATH] = lambda request: redirect
    client, clock = make_client(fake)
    with pytest.raises(UnexpectedRedirect):
        client.server_time_ms()
    assert len(fake.requests) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "location",
    ["https://xn--zzzzzzzzz.bingx.com/openApi/swap/v2/server/time", "https:x"],
    ids=["invalid-a-label", "no-host"],
)
def test_redirect_with_an_unparseable_location_is_refused(location):
    # httpx builds the never-followed redirect request while sending and fails outside HTTPError
    fake = FakeBingX([])
    fake.overrides[pm.SERVER_TIME_PATH] = lambda request: httpx.Response(
        302, headers={"location": location}
    )
    client, clock = make_client(fake)
    with pytest.raises(UnexpectedRedirect, match="unparseable Location"):
        client.server_time_ms()
    assert len(fake.requests) == 1
    assert clock.sleeps == []


# -- pagination and contiguity -----------------------------------------------------------------------------


def test_backwards_pagination_capped_at_1000_is_contiguous_and_ascending():
    candles = hourly_candles(2500)
    fake = FakeBingX(candles)
    client, _ = make_client(fake)
    start, end = candles[0]["time"], candles[-1]["time"] + HOUR_MS

    result = client.closed_klines(SYMBOL, "1h", start, end, server_time_ms=end + HOUR_MS)

    assert len(result) == 2500
    assert opens(result) == [c["time"] for c in candles]
    requests = fake.klines_requests()
    assert len(requests) == 3
    assert [r.url.params["limit"] for r in requests] == ["1000", "1000", "1000"]
    assert [int(r.url.params["startTime"]) for r in requests] == [start, start, start]
    expected_ends = [end, candles[1500]["time"] - 1, candles[500]["time"] - 1]
    assert [int(r.url.params["endTime"]) for r in requests] == expected_ends


def test_pagination_stops_on_an_empty_page_when_history_starts_late():
    candles = hourly_candles(50)
    fake = FakeBingX(candles)
    client, _ = make_client(fake)
    start, end = candles[0]["time"] - 10 * HOUR_MS, candles[-1]["time"] + HOUR_MS
    result = client.closed_klines(SYMBOL, "1h", start, end, server_time_ms=end)
    assert opens(result) == [c["time"] for c in candles]
    assert len(fake.klines_requests()) == 2


@pytest.mark.parametrize(
    "selects, closed_count, pages_per_fetch",
    [(by_open_time, 48, 1), (start_time_exclusive, 47, 2)],
    ids=["start_inclusive", "start_exclusive"],
)
def test_start_time_inclusivity_decides_the_live_48_hour_counts(
    selects, closed_count, pages_per_fetch
):
    # The live 48-hour test's arithmetic, at the verbatim live server time (inside 16:00Z).
    fake = FakeBingX(hourly_candles(60), selects=selects)
    client, _ = make_client(fake)
    start = ((SERVER_TIME_MS - 48 * HOUR_MS) // HOUR_MS) * HOUR_MS
    last_closed = (SERVER_TIME_MS // HOUR_MS) * HOUR_MS - HOUR_MS

    result = client.closed_klines(
        SYMBOL, "1h", start, SERVER_TIME_MS, server_time_ms=SERVER_TIME_MS
    )

    assert len(result) == closed_count
    assert result[0].open_ms == start + (48 - closed_count) * HOUR_MS
    assert result[-1].open_ms == last_closed
    requests = fake.klines_requests()
    assert [int(r.url.params["startTime"]) for r in requests] == [start] * pages_per_fetch
    # An exclusive start leaves the page's oldest at start + 1 h, so one more (empty) page is asked.
    expected_ends = [SERVER_TIME_MS, start + HOUR_MS - 1][:pages_per_fetch]
    assert [int(r.url.params["endTime"]) for r in requests] == expected_ends
    fake.requests.clear()
    client.fetch_twice(SYMBOL, "1h", start, SERVER_TIME_MS, server_time_ms=SERVER_TIME_MS)
    assert len(fake.klines_requests()) == 2 * pages_per_fetch


def test_start_candle_alone_on_the_next_page_is_still_fetched():
    # Why an exclusive startTime costs an extra page: a full page whose oldest is start + 1 h
    # cannot tell "startTime is exclusive" from "the start candle is on the next page".
    candles = hourly_candles(1001)
    fake = FakeBingX(candles)
    client, _ = make_client(fake)
    start, end = candles[0]["time"], candles[-1]["time"] + HOUR_MS
    result = client.closed_klines(SYMBOL, "1h", start, end, server_time_ms=end)
    assert opens(result) == [c["time"] for c in candles]
    page_ends = [int(r.url.params["endTime"]) for r in fake.klines_requests()]
    assert page_ends == [end, start + HOUR_MS - 1]


@pytest.mark.parametrize("selects", [by_open_time, end_time_inclusive], ids=["D-041", "inclusive"])
def test_closed_klines_is_half_open_whatever_the_exchange_does_at_end_time(selects):
    # tests-docs-2: an end exactly at a candle's open excludes that candle, as the exchange does
    # (D-041) and whatever it does; an end off the grid (the live window's D - 1 ms) keeps it
    candles = hourly_candles(10)
    fake = FakeBingX(candles, selects=selects)
    client, _ = make_client(fake)
    start, last = candles[0]["time"], candles[-1]["time"]
    server_time = last + HOUR_MS
    every = [c["time"] for c in candles]
    for end, expected in (
        (last, every[:-1]),
        (last + 1, every),
        (last + HOUR_MS - 1, every),
        (last + HOUR_MS, every),
    ):
        result = client.closed_klines(SYMBOL, "1h", start, end, server_time_ms=server_time)
        assert opens(result) == expected, end
        assert opens(client.fetch_twice(SYMBOL, "1h", start, end, server_time_ms=server_time)) == (
            expected
        )
    if selects is by_open_time:  # the raw page of the observed semantic
        assert opens(client.klines(SYMBOL, "1h", start, last)) == every[:-1]


def test_pagination_refuses_to_loop_when_the_server_ignores_end_time():
    candles = hourly_candles(2500)
    fake = FakeBingX(candles)
    newest_page = ok(list(reversed(candles))[:1000])
    fake.overrides[pm.KLINES_PATH] = lambda request: newest_page
    client, _ = make_client(fake)
    with pytest.raises(BingXSchemaError, match="endTime"):
        client.closed_klines(
            SYMBOL,
            "1h",
            candles[0]["time"],
            candles[-1]["time"],
            server_time_ms=candles[-1]["time"] + HOUR_MS,
        )
    assert len(fake.klines_requests()) == 2


@pytest.mark.parametrize(
    "selects, kind",
    [(end_time_rounded_up, "duplicate"), (end_time_against_close_time, "gap")],
    ids=["end_time_rounded_up_to_the_boundary", "end_time_compared_with_close_time"],
)
def test_pager_raises_when_the_end_time_seam_repeats_or_skips_a_candle(selects, kind):
    # endTime = oldest - 1 at each seam: a round-up repeats the oldest candle, a close-time filter
    # drops the one before it. Both must surface, never be deduplicated or returned short.
    candles = hourly_candles(2500)
    fake = FakeBingX(candles, selects=selects)
    client, _ = make_client(fake)
    with pytest.raises(DataGapError, match=kind):
        client.closed_klines(
            SYMBOL,
            "1h",
            candles[0]["time"],
            candles[-1]["time"],
            server_time_ms=candles[-1]["time"] + 2 * HOUR_MS,
        )
    assert len(fake.klines_requests()) == 3


def test_gap_raises_data_gap_error():
    candles = hourly_candles(100)
    del candles[40]
    fake = FakeBingX(candles)
    client, _ = make_client(fake)
    with pytest.raises(DataGapError, match="gap"):
        client.closed_klines(
            SYMBOL,
            "1h",
            candles[0]["time"],
            candles[-1]["time"],
            server_time_ms=candles[-1]["time"] + HOUR_MS,
        )


def test_duplicate_raises_data_gap_error():
    candles = hourly_candles(20)
    candles.insert(7, dict(candles[7]))
    fake = FakeBingX(candles)
    client, _ = make_client(fake)
    with pytest.raises(DataGapError, match="duplicate"):
        client.closed_klines(
            SYMBOL,
            "1h",
            candles[0]["time"],
            candles[-1]["time"],
            server_time_ms=candles[-1]["time"] + HOUR_MS,
        )


# -- schema guard -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        [
            [
                1_793_289_600_000,
                "81731.2",
                "81943.3",
                "81466.6",
                "81471.3",
                "339.1763",
                1_793_293_199_999,
                "0",
                0,
                "0",
                "0",
            ]
        ],
        [{k: v for k, v in LIVE_UNCLOSED_CANDLE.items() if k != "volume"}],
        [{**LIVE_UNCLOSED_CANDLE, "quoteVolume": "1"}],
        [{**LIVE_UNCLOSED_CANDLE, "open": "not-a-number"}],
        [{**LIVE_UNCLOSED_CANDLE, "time": "16:00"}],
        [{**LIVE_UNCLOSED_CANDLE, "close": None}],
        {"time": 1},
        "81731.2",
    ],
)
def test_kline_schema_other_than_six_key_objects_is_rejected(data):
    fake = FakeBingX([])
    fake.overrides[pm.KLINES_PATH] = lambda request: ok(data)
    client, _ = make_client(fake)
    with pytest.raises(BingXSchemaError):
        client.klines(SYMBOL, "1h", 0, 10)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>maintenance</html>"),
        httpx.Response(200, json=[{"code": 0}]),
        httpx.Response(200, json={"msg": "no code", "data": {}}),
        httpx.Response(200, json={"code": "0", "data": {}}),
        httpx.Response(200, json={"code": 0, "msg": ""}),
        httpx.Response(200, json={"code": 0, "data": {"serverTime": "soon"}}),
        httpx.Response(200, json={"code": 0, "data": [SERVER_TIME_MS]}),
    ],
)
def test_server_time_schema_guard(response):
    client, _ = make_client(lambda request: response)
    with pytest.raises(BingXSchemaError):
        client.server_time_ms()


def test_server_time_accepts_digit_strings_and_ints():
    client, _ = make_client(lambda request: ok({"serverTime": str(SERVER_TIME_MS)}))
    assert client.server_time_ms() == SERVER_TIME_MS


# -- transport pinning ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        f"https://{DEMO_HOST_REFUSED}/openApi/swap/v2/server/time",
        "https://open-api.bingx.pro/openApi/swap/v2/server/time",
        "http://open-api.bingx.com/openApi/swap/v2/server/time",
        "https://open-api.bingx.com:8443/openApi/swap/v2/server/time",
        f"https://{PUBLIC_HOST}.evil.invalid/openApi/swap/v2/server/time",
        f"https://{PUBLIC_HOST}@evil.invalid/openApi/swap/v2/server/time",  # userinfo trick
        f"https://user:pw@{PUBLIC_HOST}/openApi/swap/v2/server/time",
        f"https://evil.invalid/?u={PUBLIC_HOST}",
        f"{PUBLIC_HOST}/openApi/swap/v2/server/time",
        "",
    ],
)
def test_pinned_transport_refuses_everything_but_https_public_host(url):
    inner_calls: list[httpx.Request] = []

    def inner(request: httpx.Request) -> httpx.Response:
        inner_calls.append(request)
        return ok({})

    transport = pm._PublicPinnedTransport(httpx.MockTransport(inner))
    with pytest.raises(HostNotAllowedError):
        transport.handle_request(httpx.Request("GET", url))
    assert inner_calls == []


@pytest.mark.parametrize("host", ["xn--zzzzzzzzz.bingx.com", "xn--a.bingx.com"])
def test_invalid_punycode_host_is_refused_as_host_not_allowed(host):
    url = f"https://{host}/openApi/swap/v2/server/time"
    with pytest.raises(HostNotAllowedError):
        pm._check_public_url(httpx.URL(url))
    inner_calls: list[httpx.Request] = []

    def inner(request: httpx.Request) -> httpx.Response:
        inner_calls.append(request)
        return ok({})

    # httpx reads url.host while building a Request unless a Host header is given
    transport = pm._PublicPinnedTransport(httpx.MockTransport(inner))
    with pytest.raises(HostNotAllowedError):
        transport.handle_request(httpx.Request("GET", url, headers={"Host": "x"}))
    assert inner_calls == []


def test_pinned_transport_passes_the_public_host_through():
    inner_calls: list[httpx.Request] = []

    def inner(request: httpx.Request) -> httpx.Response:
        inner_calls.append(request)
        return ok({})

    transport = pm._PublicPinnedTransport(httpx.MockTransport(inner))
    response = transport.handle_request(
        httpx.Request("GET", f"https://{PUBLIC_HOST}{pm.SERVER_TIME_PATH}")
    )
    assert response.status_code == 200 and len(inner_calls) == 1


def test_client_uses_the_pinned_transport_without_redirects_or_env_trust():
    client, _ = make_client(FakeBingX([]))
    assert isinstance(client._transport, pm._PublicPinnedTransport)
    assert client._http.follow_redirects is False
    assert client._http.trust_env is False
    assert str(client._http.base_url) == ""


def test_proxy_environment_is_ignored(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.invalid:1080")
    fake = FakeBingX([])
    client, _ = make_client(fake)
    assert client.server_time_ms() == SERVER_TIME_MS
    assert fake.requests[0].url.host == PUBLIC_HOST
    # httpx 0.28 reads the proxy variables only when ``trust_env`` is true AND no ``transport`` is
    # given (``allow_env_proxies = trust_env and transport is None``); the client always passes its
    # pinned transport, so they are unreachable by construction. The client-level
    # ``trust_env=False`` is the second layer (it would keep them out if that transport were ever
    # dropped) and ``_mounts`` is the table ``_transport_for_url`` consults: ``{}`` means nothing
    # can route a request around the pinned transport. The inner transport's own
    # ``trust_env=False`` is load-bearing (SSL_CERT_FILE/SSL_CERT_DIR); the next test pins it.
    assert client._http._mounts == {}
    with PublicMarketClient() as default_client:
        assert default_client._http._mounts == {}
        assert default_client._http.trust_env is False


def test_default_inner_transport_is_built_without_env_trust(monkeypatch):
    built: list[tuple[httpx.HTTPTransport, tuple[Any, ...], dict[str, Any]]] = []

    class RecordingHTTPTransport(httpx.HTTPTransport):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append((self, args, kwargs))

    monkeypatch.setattr(httpx, "HTTPTransport", RecordingHTTPTransport)
    with PublicMarketClient() as client:
        assert [(args, kwargs) for _, args, kwargs in built] == [
            ((), {"trust_env": False, "retries": 0})
        ]
        assert client._transport._inner is built[0][0]


# -- intervals, pacing, hashing ----------------------------------------------------------------------------


def test_interval_whitelist():
    assert list(INTERVAL_MS) == [
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    ]
    assert (
        INTERVAL_MS["1h"] == HOUR_MS
        and INTERVAL_MS["1m"] == 60_000
        and INTERVAL_MS["1d"] == 86_400_000
    )
    for name, ms in INTERVAL_MS.items():
        assert interval_ms(name) == ms
    for bad in ("1w", "1M", "1s", "60m", "", None):
        with pytest.raises(ValueError):
            interval_ms(bad)


def test_unknown_interval_is_rejected_before_any_request():
    fake = FakeBingX([])
    client, _ = make_client(fake)
    with pytest.raises(ValueError, match="interval"):
        client.klines(SYMBOL, "1w", 0, 10)
    with pytest.raises(ValueError, match="interval"):
        client.closed_klines(SYMBOL, "1M", 0, 10, server_time_ms=10)
    assert fake.requests == []


def test_pacing_enforces_min_interval_with_injected_clock():
    fake = FakeBingX([])
    client, clock = make_client(fake)
    client.server_time_ms()
    assert clock.sleeps == []
    client.server_time_ms()
    assert clock.sleeps == [2.0]
    clock.now += 5.0
    client.server_time_ms()
    assert clock.sleeps == [2.0]
    clock.now += 0.5
    client.server_time_ms()
    assert clock.sleeps == [2.0, 1.5]
    assert len(fake.requests) == 4


def test_pacing_disabled_with_zero_interval():
    fake = FakeBingX([])
    client, clock = make_client(fake, min_interval_s=0.0)
    client.server_time_ms()
    client.server_time_ms()
    assert clock.sleeps == []


def test_fetch_twice_raises_on_mismatch_and_returns_on_match():
    candles = hourly_candles(10)
    fake = FakeBingX(candles)
    client, _ = make_client(fake)
    start, end = candles[0]["time"], candles[-1]["time"] + HOUR_MS
    server_time = end

    result = client.fetch_twice(SYMBOL, "1h", start, end, server_time_ms=server_time)
    assert opens(result) == [c["time"] for c in candles]
    assert len(fake.klines_requests()) == 2

    altered = [dict(c) for c in candles]
    altered[3]["volume"] = "999.9"
    pages = iter([ok(list(reversed(candles))), ok(list(reversed(altered)))])
    fake.overrides[pm.KLINES_PATH] = lambda request: next(pages)
    with pytest.raises(CandleMismatchError):
        client.fetch_twice(SYMBOL, "1h", start, end, server_time_ms=server_time)


def test_fetch_twice_reads_server_time_once():
    candles = hourly_candles(3)
    fake = FakeBingX(candles)
    client, _ = make_client(fake)
    client.fetch_twice(SYMBOL, "1h", candles[0]["time"], candles[-1]["time"])
    assert [r.url.path for r in fake.requests] == [
        pm.SERVER_TIME_PATH,
        pm.KLINES_PATH,
        pm.KLINES_PATH,
    ]


def test_window_sha256_is_stable_and_canonical():
    candles = [
        Candle(1_793_282_400_000, "81500.0", "81600.0", "81400.0", "81550.5", "120.5"),
        Candle(1_793_286_000_000, "81550.5", "81700.0", "81500.0", "81650.0", "98.25"),
    ]
    expected_bytes = (
        b'[[1793282400000,"81500.0","81600.0","81400.0","81550.5","120.5"],'
        b'[1793286000000,"81550.5","81700.0","81500.0","81650.0","98.25"]]'
    )
    assert canonical_bytes(candles) == expected_bytes
    assert canonical_bytes(candles) == canonical_json(
        [[c.open_ms, c.open, c.high, c.low, c.close, c.volume] for c in candles]
    )
    assert (
        window_sha256(candles) == "b751ace37c7437ca196f872a7f7efe1a4317ad9fff2f17de0a89562d4e0b5c59"
    )
    assert window_sha256(candles) == sha256_bytes(expected_bytes)
    assert window_sha256(candles) == PublicMarketClient.window_sha256(candles)
    changed = [
        candles[0],
        Candle(1_793_286_000_000, "81550.5", "81700.0", "81500.0", "81650.0", "98.26"),
    ]
    assert window_sha256(changed) != window_sha256(candles)
    assert window_sha256([]) == sha256_bytes(b"[]")


def test_candle_helpers():
    candle = Candle(
        **{k: v for k, v in LIVE_UNCLOSED_CANDLE.items() if k != "time"},
        open_ms=LIVE_UNCLOSED_CANDLE["time"],
    )
    assert candle.close_ms(HOUR_MS) == 1_793_293_200_000
    assert candle.close_ms(HOUR_MS) > SERVER_TIME_MS  # the verbatim live candle was unclosed
    assert candle.as_floats() == (81731.2, 81943.3, 81466.6, 81471.3, 339.1763)
    with pytest.raises(AttributeError):
        candle.open = "1"  # frozen


# -- the other three endpoints ---------------------------------------------------------------------------


LIVE_CONTRACT = {
    "symbol": SYMBOL,
    "quantityPrecision": 4,
    "pricePrecision": 1,
    "tradeMinQuantity": 0.0001,
    "takerFeeRate": 0.0005,
    "makerFeeRate": 0.0002,
}


def test_contract_returns_the_matching_entry():
    fake = FakeBingX([])
    fake.overrides[pm.CONTRACTS_PATH] = lambda request: ok(
        [{"symbol": "ETH-USDT", "pricePrecision": 2}, LIVE_CONTRACT]
    )
    client, _ = make_client(fake)
    contract = client.contract(SYMBOL)
    assert contract["quantityPrecision"] == 4 and contract["pricePrecision"] == 1
    assert contract["takerFeeRate"] == 0.0005 and contract["makerFeeRate"] == 0.0002
    assert fake.requests[0].url.params["symbol"] == SYMBOL


def test_contract_missing_symbol_raises_schema_error():
    fake = FakeBingX([])
    fake.overrides[pm.CONTRACTS_PATH] = lambda request: ok([{"symbol": "ETH-USDT"}])
    client, _ = make_client(fake)
    with pytest.raises(BingXSchemaError):
        client.contract(SYMBOL)


def test_premium_index_accepts_object_or_list():
    payload = {
        "symbol": SYMBOL,
        "markPrice": "81500.1",
        "fundingIntervalHours": 8,
        "nextFundingTime": 1_793_318_400_000,
    }
    fake = FakeBingX([])
    fake.overrides[pm.PREMIUM_INDEX_PATH] = lambda request: ok(payload)
    client, _ = make_client(fake)
    assert client.premium_index(SYMBOL) == payload
    fake.overrides[pm.PREMIUM_INDEX_PATH] = lambda request: ok([{"symbol": "ETH-USDT"}, payload])
    assert client.premium_index(SYMBOL) == payload
    fake.overrides[pm.PREMIUM_INDEX_PATH] = lambda request: ok([{"symbol": "ETH-USDT"}])
    with pytest.raises(BingXSchemaError):
        client.premium_index(SYMBOL)


def test_funding_rate_history_items_and_limit():
    items = [
        {
            "symbol": SYMBOL,
            "fundingRate": "0.0001",
            "fundingTime": 1_793_289_600_000,
            "markPrice": "81500",
        },
        {
            "symbol": SYMBOL,
            "fundingRate": "0.0001",
            "fundingTime": 1_793_260_800_000,
            "markPrice": "81000",
        },
    ]
    fake = FakeBingX([])
    fake.overrides[pm.FUNDING_RATE_PATH] = lambda request: ok(items)
    client, _ = make_client(fake)
    assert client.funding_rate_history(SYMBOL, limit=2) == items
    assert fake.requests[0].url.params["limit"] == "2"
    fake.overrides[pm.FUNDING_RATE_PATH] = lambda request: ok(
        [{"symbol": SYMBOL, "fundingRate": "0.0001"}]
    )
    with pytest.raises(BingXSchemaError):
        client.funding_rate_history(SYMBOL)
    with pytest.raises(ValueError):
        client.funding_rate_history(SYMBOL, limit=0)
    with pytest.raises(ValueError):
        client.funding_rate_history(SYMBOL, limit=1001)


@pytest.mark.parametrize("symbol", ["", None, 42])
def test_symbol_must_be_a_non_empty_string(symbol):
    fake = FakeBingX([])
    client, _ = make_client(fake)
    with pytest.raises(ValueError):
        client.contract(symbol)
    assert fake.requests == []
