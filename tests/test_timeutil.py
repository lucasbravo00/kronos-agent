"""UTC helpers: the three decision-hour forms, exact millisecond conversions, half-open ranges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from kronos_agent.data.timeutil import (
    HOUR_MS,
    MINUTE_MS,
    from_ms,
    hour_key,
    hour_range,
    parse_utc_hour,
    to_ms,
    utc_text,
)

VALIDATION_START = datetime(2024, 7, 1, tzinfo=UTC)
VALIDATION_START_MS = 1_719_792_000_000  # computed independently
LIVE_SERVER_TIME_MS = 1_793_290_785_970  # 2026-10-29T16:19:45.970Z, the step-1 live probe


@pytest.mark.parametrize("text", ["2024-07-01T00:00Z", "2024-07-01T00:00:00Z", "2024-07-01T00Z"])
def test_parse_utc_hour_accepts_the_three_forms(text):
    value = parse_utc_hour(text)
    assert value == VALIDATION_START
    assert value.tzinfo is UTC
    assert to_ms(value) == VALIDATION_START_MS


@pytest.mark.parametrize(
    "text",
    [
        "2024-07-01T00:30Z",
        "2024-07-01T00:00:01Z",
        "2024-07-01T00:00:00.000Z",
        "2024-07-01 00:00Z",
        "2024-07-01T00:00+00:00",
        "2024-07-01T00:00",
        "2024-07-01T00",
        "2024-07-01",
        "2024-13-01T00Z",
        "2024-02-30T00Z",
        "2024-07-01T24Z",
        " 2024-07-01T00Z",
        "2024-07-01T00Z\n",
        "2024-7-01T00Z",
        "2024-07-01t00z",
        "\uff12\uff10\uff12\uff14-07-01T00Z",  # full-width digits are not ASCII digits
        "",
    ],
)
def test_parse_utc_hour_refuses_every_other_string(text):
    with pytest.raises(ValueError):
        parse_utc_hour(text)


def test_parse_utc_hour_normalises_any_zero_offset_tzinfo():
    other_utc = timezone(timedelta(0), "another UTC")
    value = parse_utc_hour(datetime(2024, 7, 1, 5, tzinfo=other_utc))
    assert value == datetime(2024, 7, 1, 5, tzinfo=UTC)
    assert value.tzinfo is UTC
    assert type(value) is datetime


@pytest.mark.parametrize(
    "value",
    [
        datetime(2024, 7, 1, 5),  # naive
        datetime(2024, 7, 1, 5, tzinfo=timezone(timedelta(hours=1))),
        datetime(2024, 7, 1, 5, 1, tzinfo=UTC),
        datetime(2024, 7, 1, 5, 0, 1, tzinfo=UTC),
        datetime(2024, 7, 1, 5, 0, 0, 1, tzinfo=UTC),
        VALIDATION_START_MS,
        None,
    ],
)
def test_parse_utc_hour_refuses_other_datetimes_and_types(value):
    with pytest.raises(ValueError):
        parse_utc_hour(value)


def test_to_ms_and_from_ms_are_exact_inverses():
    with_millis = datetime(2026, 10, 29, 16, 19, 45, 970_000, tzinfo=UTC)
    assert to_ms(with_millis) == LIVE_SERVER_TIME_MS
    assert from_ms(LIVE_SERVER_TIME_MS) == with_millis
    assert from_ms(np.int64(VALIDATION_START_MS)) == VALIDATION_START
    assert to_ms(datetime(1970, 1, 1, tzinfo=UTC)) == 0
    assert to_ms(datetime(1969, 12, 31, 23, 59, 59, 999_000, tzinfo=UTC)) == -1
    # a sub-millisecond remainder is floored like an epoch-millisecond clock
    assert to_ms(datetime(2026, 1, 1, 0, 0, 0, 999, tzinfo=UTC)) == to_ms(
        datetime(2026, 1, 1, tzinfo=UTC)
    )
    hour = VALIDATION_START
    for _ in range(5000):
        assert from_ms(to_ms(hour)) == hour
        assert to_ms(hour) % HOUR_MS == 0
        hour += timedelta(hours=7)


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_from_ms_refuses_non_integers(bad):
    with pytest.raises(ValueError):
        from_ms(bad)


def test_to_ms_refuses_naive_and_non_utc_datetimes():
    with pytest.raises(ValueError):
        to_ms(datetime(2024, 7, 1))
    with pytest.raises(ValueError):
        to_ms(datetime(2024, 7, 1, tzinfo=timezone(timedelta(hours=-3))))


def test_hour_key_and_utc_text_forms():
    assert hour_key(datetime(2025, 1, 1, tzinfo=UTC)) == "2025-01-01T00Z"
    assert hour_key("2025-01-01T07:00Z") == "2025-01-01T07Z"
    assert parse_utc_hour(hour_key(VALIDATION_START)) == VALIDATION_START
    with pytest.raises(ValueError):
        hour_key(datetime(2025, 1, 1, 0, 30, tzinfo=UTC))
    assert utc_text(to_ms(datetime(2025, 1, 1, tzinfo=UTC))) == "2025-01-01T00:00Z"
    assert utc_text(5 * MINUTE_MS) == "1970-01-01T00:05Z"
    # a time off the minute grid is never truncated to the minute
    assert utc_text(LIVE_SERVER_TIME_MS) == "2026-10-29T16:19:45.970Z"
    assert utc_text(1000) == "1970-01-01T00:00:01.000Z"
    assert utc_text(np.int64(VALIDATION_START_MS)) == "2024-07-01T00:00Z"


def test_hour_range_is_half_open():
    start = VALIDATION_START
    assert hour_range(start, start + timedelta(hours=5)) == [
        start + timedelta(hours=k) for k in range(5)
    ]
    assert hour_range(start, start + timedelta(hours=5), step_hours=2) == [
        start,
        start + timedelta(hours=2),
        start + timedelta(hours=4),
    ]
    assert hour_range(start, start + timedelta(hours=4), step_hours=2) == [
        start,
        start + timedelta(hours=2),
    ]
    assert hour_range(start, start) == []
    assert hour_range(start + timedelta(hours=1), start) == []
    assert hour_range("2024-07-01T00Z", "2024-07-01T02:00Z") == [start, start + timedelta(hours=1)]
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            hour_range(start, start + timedelta(hours=2), step_hours=bad)
    with pytest.raises(ValueError):
        hour_range(start + timedelta(minutes=30), start + timedelta(hours=2))
