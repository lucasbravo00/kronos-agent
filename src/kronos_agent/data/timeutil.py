"""UTC time helpers: decision hours, epoch milliseconds and their text forms (stdlib only).

Everything in the data layer is UTC and every candle is keyed by the epoch milliseconds of its
open (``docs/DATA.md``, "Conventions"). A decision hour is an aware UTC ``datetime`` on an exact
hour; strings accepted for it are the three forms the command line and the reports use. The
conversions are exact integer arithmetic (no float ``timestamp()``), and every invalid input is a
``ValueError``: a wrong hour must never be rounded into a right-looking one.

This module imports only the standard library: the command line imports it at ``--help`` time.
"""

from __future__ import annotations

import numbers
import re
from datetime import UTC, datetime, timedelta

HOUR_MS = 3_600_000
MINUTE_MS = 60_000
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_HOUR_TEXT_RE = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2})(?::00(?::00)?)?Z")
_ONE_MS = timedelta(milliseconds=1)


def _require_utc(dt: object, name: str) -> datetime:
    if not isinstance(dt, datetime):
        raise ValueError(f"{name} must be a datetime, got {type(dt).__name__}")  # noqa: TRY004
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC, got {dt!r}")
    return dt


def parse_utc_hour(value: str | datetime) -> datetime:
    """An aware UTC ``datetime`` on an exact hour from a decision-hour string or datetime.

    Accepted strings: ``YYYY-MM-DDTHH:00Z``, ``YYYY-MM-DDTHH:00:00Z`` and ``YYYY-MM-DDTHHZ``
    (the file-name form). A datetime must be timezone-aware with a zero UTC offset and have no
    minutes, seconds or microseconds; it comes back as a plain ``datetime`` with ``tzinfo=UTC``.
    Anything else raises ``ValueError``.
    """
    if isinstance(value, str):
        match = _HOUR_TEXT_RE.fullmatch(value)
        if match is None:
            raise ValueError(
                f"decision hour {value!r} is not YYYY-MM-DDTHH:00Z, YYYY-MM-DDTHH:00:00Z or "
                "YYYY-MM-DDTHHZ"
            )
        year, month, day, hour = (int(group) for group in match.groups())
        try:
            return datetime(year, month, day, hour, tzinfo=UTC)
        except ValueError as exc:
            raise ValueError(f"decision hour {value!r} is not a valid date: {exc}") from None
    dt = _require_utc(value, "decision hour")
    if dt.minute or dt.second or dt.microsecond:
        raise ValueError(f"decision hour must be on an exact hour boundary, got {dt.isoformat()}")
    return datetime(dt.year, dt.month, dt.day, dt.hour, tzinfo=UTC)


def to_ms(dt: datetime) -> int:
    """Epoch milliseconds of an aware UTC datetime (exact integer arithmetic).

    A sub-millisecond remainder is floored, as an epoch-millisecond clock would.
    """
    dt = _require_utc(dt, "datetime")
    return (dt - EPOCH) // _ONE_MS


def _require_ms(ms: object) -> int:
    if isinstance(ms, bool) or not isinstance(ms, numbers.Integral):
        raise ValueError(  # noqa: TRY004 - ValueError is this module's contract
            f"epoch milliseconds must be an integer, got {ms!r}"
        )
    return int(ms)


def from_ms(ms: int) -> datetime:
    """Aware UTC datetime of epoch milliseconds (numpy integers accepted, booleans refused)."""
    return EPOCH + timedelta(milliseconds=_require_ms(ms))


def hour_key(dt: datetime) -> str:
    """``YYYY-MM-DDTHHZ`` (e.g. ``2025-01-01T00Z``): the file-name form of a decision hour."""
    return parse_utc_hour(dt).strftime("%Y-%m-%dT%HZ")


def utc_text(ms: int) -> str:
    """``YYYY-MM-DDTHH:MMZ`` for epoch milliseconds, e.g. ``2025-01-01T00:00Z``.

    Every candle open is on a whole minute and gets exactly that form. A time with seconds or
    milliseconds (a server time) is never truncated: it gets ``:SS.mmmZ`` appended to the minute.
    """
    ms = _require_ms(ms)
    dt = from_ms(ms)
    text = dt.strftime("%Y-%m-%dT%H:%M")
    remainder = ms % MINUTE_MS
    if remainder:
        text += f":{remainder // 1000:02d}.{remainder % 1000:03d}"
    return text + "Z"


def hour_range(start: datetime, end: datetime, step_hours: int = 1) -> list[datetime]:
    """Decision hours ``start, start + step, ...`` strictly before ``end`` (half-open).

    Both bounds must be exact UTC hours (strings in a :func:`parse_utc_hour` form are accepted);
    ``step_hours`` is a positive integer. ``end <= start`` gives an empty list.
    """
    if isinstance(step_hours, bool) or not isinstance(step_hours, int) or step_hours < 1:
        raise ValueError(f"step_hours must be a positive integer, got {step_hours!r}")
    first = parse_utc_hour(start)
    stop = parse_utc_hour(end)
    step = timedelta(hours=step_hours)
    hours: list[datetime] = []
    current = first
    while current < stop:
        hours.append(current)
        current += step
    return hours


__all__ = [
    "EPOCH",
    "HOUR_MS",
    "MINUTE_MS",
    "from_ms",
    "hour_key",
    "hour_range",
    "parse_utc_hour",
    "to_ms",
    "utc_text",
]
