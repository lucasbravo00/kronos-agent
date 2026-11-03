"""BingX real-market history: an archive of 1000-candle chunks, each fetched twice, then merged.

Why an archive: the real market serves 1h candles from 2024-04-27T10:00Z, but keeps 1m and 5m
candles on a rolling retention (about 450 and 48 days observed), so a candle fetched once may never
be fetchable again. Each interval is cut on an epoch-aligned grid of :data:`CHUNK_CANDLES` candles:
chunk ``k`` holds the opens ``[k * span, (k + 1) * span)`` with ``span = 1000 * step``. A chunk is
archived as ``raw/bingx/<interval>/<chunk_start_ms>.csv`` plus a JSON sidecar written after it,
which is its commit marker. Archived chunks are never deleted and archived candles never change;
complete chunks are never fetched again, and the store ``candles/bingx/<interval>`` is rebuilt
from all of them by :func:`merge_chunks` (``docs/DATA.md``, "BingX real market").

The rules, and why:

- **One page per fetch.** ``startTime = k * span`` (inclusive at an exact boundary: observed) and
  ``endTime = min((k + 1) * span - 1, server time)``, moved back one millisecond when it lands on a
  candle boundary, ``limit = 1000``. ``endTime`` is never an exact boundary, because an exact
  boundary excludes the candle opening there while the ``boundary - 1`` form returns the whole
  chunk (both observed live, D-041), and never in the future. :meth:`PublicMarketClient.klines`
  is called directly: ``closed_klines`` and ``fetch_twice`` refuse internal gaps, which a chunk
  may legitimately hold.
- **Settled candles only** (SPEC section 5: closed candles, at least 90 s after the close): a
  candle is kept only if ``open + step + 90 s <= server time``, with the server time read once per
  run; a chunk is complete when its last slot passes the same test, whatever it holds.
- **Twice until they match.** Each chunk is fetched twice and the settled candles of the two
  fetches must be byte-identical (``canonical_bytes``); a differing pair is repeated, at most
  :data:`MAX_FETCH_PAIRS` pairs, then the backfill fails naming the chunk. Every returned candle
  must lie inside the request, on the interval grid, once, with plain decimal strings (the store
  format): nothing is filled, deduplicated, re-formatted or repaired, and ``code != 0`` from the
  exchange is an error, never an empty chunk.
- **The archive is the record.** A CSV without a valid sidecar, a sidecar without its CSV, or a
  CSV whose size or sha256 differs from its sidecar stops the run before any request, naming the
  file. Incomplete chunks (normally the newest one) are fetched again every run, and archived
  candles are never overwritten or deleted. A candle the refetch returns with other strings than
  the archive is a revision (the owner's decision, D-045): the archive keeps the first-seen row,
  which is what a live decision would have used, the revision is recorded once in
  ``raw/bingx/revision-<interval>.json`` with its first and last observation
  (:func:`read_revisions`) before the chunk is written, and the backfill goes on; ``data bingx``
  reports it as a warning as soon as it is recorded (so also when a later chunk fails) and
  ``data status`` lists it. A chunk whose files were removed (the only repair of an
  inconsistent one) is fetched again like a new chunk, but the candles it serves with a recorded
  revision take their recorded first-seen rows. An archived candle the refetch no longer
  returns (rolled off the exchange's retention after a missed run, or no longer served) is kept
  from the archive and listed in the sidecar (``archive_only``, sidecar schema 2; schema 1
  sidecars are read as having none); new settled candles are added. A refreshed chunk completes only when the rolling retention explains
  every such candle (:func:`_not_retention`: 1m and 5m only, a candle older than the observed
  retention and than the first candle the refetch returned); otherwise it stays incomplete and
  is fetched again next run, so a transient answer never becomes a permanent gap. Empty chunks
  are never archived.
- **Walk and history start.** From the chunk holding the newest settled candle backwards;
  archived complete chunks are skipped without a request. The walk stops after
  :data:`STOP_AFTER_EMPTY_CHUNKS` consecutive empty chunks older than the oldest archived chunk.
  The first archived candle is then the history start if nothing older was served when it was
  archived: this run fetched it, or it opens after its chunk's start (the fetch that archived it
  found nothing before it in its chunk), or the recorded start already names it. Otherwise the
  chunks below may have rolled off the exchange's retention since a bounded run archived it, and
  the start stays open. An established start is recorded in ``raw/bingx/history-<interval>.json``
  (:func:`read_history`); a run whose walk its ``max_chunks`` bound stopped never writes it, and
  the merged store's provenance says whether its first candle is that start. An empty chunk between
  non-empty ones is a gap, reported and left to the merged store's gap list; three consecutive
  empty chunks inside the history would end the walk early, an accepted limit (3000 candles: 125
  days of 1h, 50 hours of 1m).
- **One run at a time.** :func:`backfill` and :func:`merge_chunks` hold the job lock
  ``raw/bingx/.lock`` exclusively (:func:`~kronos_agent.data.store.job_lock`); a second
  concurrent run is refused with ``LockBusyError``. Taking it removes the temporary files that a
  killed run left under ``raw/bingx`` and ``candles/bingx`` (no live writer can own one then).
"""

from __future__ import annotations

import json
import logging
import numbers
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exchange.public_market import Candle, PublicMarketClient, canonical_bytes, interval_ms
from ..hashing import sha256_bytes
from ..paths import data_dir
from ..provenance import code_version
from .fsutil import LockBusyError, atomic_write_bytes, shared_lock
from .store import (
    BASE_COLUMNS,
    DECIMAL_RE,
    SOURCE_SYMBOLS,
    StoreId,
    StoreWriter,
    find_gaps,
    job_lock,
)
from .timeutil import utc_text

CHUNK_CANDLES = 1000
SETTLE_MS = 90_000
MAX_FETCH_PAIRS = 3
STOP_AFTER_EMPTY_CHUNKS = 3
INTERVALS: tuple[str, ...] = ("1h", "15m", "5m", "1m")
# Observed rolling retention of the real market (step 1): the status warning and the completion
# of a refreshed chunk (_not_retention) use it, and the recorded coverage definition holds it, so
# a change makes coverage a new study (the first download's figures wait for D-049, D-050).
RETENTION_DAYS: dict[str, int] = {"1m": 450, "5m": 48}

SOURCE = "bingx"
SYMBOL = SOURCE_SYMBOLS[SOURCE]
# Version 2 adds ``archive_only``; version 1 sidecars (none listed there) are still read.
SIDECAR_SCHEMA_VERSION = 2
SIDECAR_SCHEMA_VERSIONS: tuple[int, ...] = (1, 2)
HISTORY_SCHEMA_VERSION = 1
HISTORY_STOP_REASONS: tuple[str, ...] = ("empty_chunks", "epoch")
# Version 2 lists every revision once; a version 1 record (the one revision of a backfill that
# stopped at it) is still read.
REVISION_SCHEMA_VERSION = 2
REVISION_SCHEMA_VERSIONS: tuple[int, ...] = (1, 2)
# Revisions named one by one in a status warning; the rest are counted.
LISTED_REVISIONS = 10
ARCHIVE_RELDIR = "raw/bingx"
LOCK_RELPATH = "raw/bingx/.lock"
DAY_MS = 86_400_000
ROW_COLUMNS = ",".join(BASE_COLUMNS)
CHUNK_HEADER = (ROW_COLUMNS + "\n").encode("ascii")
VALUE_FIELDS: tuple[str, ...] = BASE_COLUMNS[1:]

_PLAIN_DECIMAL = re.compile(DECIMAL_RE)
_OPEN_MS = re.compile(r"[0-9]+")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_CHUNK_FILE = re.compile(r"(0|[1-9][0-9]*)\.(csv|json)")

_LOG = logging.getLogger(__name__)


class BackfillError(Exception):
    """The chunk archive is inconsistent, or the exchange answered what a chunk cannot hold."""


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else data_dir()


def _check_interval(interval: object) -> str:
    if not isinstance(interval, str) or interval not in INTERVALS:
        raise ValueError(f"unsupported interval {interval!r}; expected one of {list(INTERVALS)}")
    return interval


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


# -- the chunk grid -------------------------------------------------------------------------------


def chunk_span_ms(interval: str) -> int:
    """Milliseconds covered by one chunk: ``CHUNK_CANDLES`` candles of ``interval``."""
    return CHUNK_CANDLES * interval_ms(_check_interval(interval))


def chunk_index(open_ms: int, interval: str) -> int:
    """Index of the chunk holding the open time ``open_ms`` (chunks are half-open)."""
    return _non_negative_int(open_ms, "open_ms") // chunk_span_ms(interval)


def chunk_bounds(index: int, interval: str) -> tuple[int, int]:
    """``(start_ms, end_ms)`` of chunk ``index``: the opens ``[start_ms, end_ms)``."""
    index = _non_negative_int(index, "chunk index")
    span = chunk_span_ms(interval)
    return index * span, (index + 1) * span


def request_bounds(index: int, interval: str, server_time_ms: int) -> tuple[int, int]:
    """``(startTime, endTime)`` sent for chunk ``index`` at ``server_time_ms``.

    ``startTime`` is the chunk start; ``endTime = min(chunk end - 1, server time)``, one
    millisecond earlier when that is a candle boundary, so it is never an exact boundary and never
    in the future. A chunk that starts at or after the server time has nothing to request
    (``ValueError``).
    """
    start, end = chunk_bounds(index, interval)
    server = _non_negative_int(server_time_ms, "server_time_ms")
    if server <= start:
        raise ValueError(
            f"{interval} chunk {utc_text(start)} starts at or after the server time "
            f"{utc_text(server)}: nothing to request"
        )
    step = interval_ms(interval)
    request_end = min(end - 1, server)
    if request_end % step == 0:
        request_end -= 1
    return start, request_end


def settled(candle_open_ms: int, step_ms: int, server_time_ms: int) -> bool:
    """True when the candle closed at least :data:`SETTLE_MS` before ``server_time_ms``."""
    return candle_open_ms + step_ms + SETTLE_MS <= server_time_ms


def _settled_slots(start_ms: int, step_ms: int, server_time_ms: int) -> int:
    """How many of a chunk's slots pass :func:`settled` at ``server_time_ms``."""
    count = (server_time_ms - SETTLE_MS - step_ms - start_ms) // step_ms + 1
    return max(0, min(CHUNK_CANDLES, count))


def _chunk_label(interval: str, start_ms: int) -> str:
    return f"{interval} chunk {utc_text(start_ms)}"


# -- fetching -------------------------------------------------------------------------------------


@dataclass
class ChunkFetch:
    """The settled candles of one chunk, from a pair of fetches that matched.

    ``pairs`` is the number of fetch pairs needed (1 to :data:`MAX_FETCH_PAIRS`), ``requests``
    the number of ``klines`` requests made (two per pair), ``complete`` whether the chunk's last
    slot is settled at the server time used.
    """

    candles: list[Candle]
    pairs: int
    requests: int
    complete: bool


def _check_page(
    page: Sequence[Candle], step_ms: int, request_start: int, request_end: int, label: str
) -> None:
    """Every candle inside ``[request_start, request_end]``, on the grid, once, plain decimals."""
    previous: int | None = None
    for candle in page:
        if not isinstance(candle, Candle):
            raise BackfillError(f"{label}: expected Candle objects, got {type(candle).__name__}")
        open_ms = candle.open_ms
        if not request_start <= open_ms <= request_end:
            raise BackfillError(
                f"{label}: the exchange returned a candle opening at {utc_text(open_ms)}, outside "
                f"the request [{utc_text(request_start)}, {utc_text(request_end)}]"
            )
        if open_ms % step_ms:
            raise BackfillError(f"{label}: candle at {utc_text(open_ms)} is off the interval grid")
        if previous is not None and open_ms <= previous:
            raise BackfillError(f"{label}: two candles open at {utc_text(open_ms)}")
        previous = open_ms
        for name in VALUE_FIELDS:
            value = getattr(candle, name)
            if not isinstance(value, str) or not _PLAIN_DECIMAL.fullmatch(value):
                raise BackfillError(
                    f"{label}: {name} of the candle at {utc_text(open_ms)} is not a plain decimal "
                    f"string ({value!r}); nothing is re-formatted"
                )


def _difference(first: Sequence[Candle], second: Sequence[Candle]) -> str:
    one = {c.open_ms: c for c in first}
    two = {c.open_ms: c for c in second}
    differing = sorted(key for key in one.keys() | two.keys() if one.get(key) != two.get(key))
    where = f", first at {utc_text(differing[0])}" if differing else ""
    return (
        f"{len(first)} vs {len(second)} settled candles, {len(differing)} differing{where} "
        f"(sha256 {sha256_bytes(canonical_bytes(first))[:12]} vs "
        f"{sha256_bytes(canonical_bytes(second))[:12]})"
    )


def fetch_chunk(
    client: PublicMarketClient,
    interval: str,
    index: int,
    *,
    server_time_ms: int,
    symbol: str = "BTC-USDT",
) -> ChunkFetch:
    """Fetch chunk ``index`` twice until the settled candles of a pair are byte-identical.

    Each fetch is one ``klines`` page over :func:`request_bounds`; its candles are checked
    (inside the request, on the grid, unique, plain decimal strings; :class:`BackfillError`
    otherwise) and the unsettled ones are dropped BEFORE the two fetches are compared, so a
    forming candle never makes a pair differ. After :data:`MAX_FETCH_PAIRS` differing pairs the
    chunk is refused (:class:`BackfillError` naming it). Errors of the client (``code != 0``,
    rate limits, transport) propagate: they are never an empty chunk.
    """
    step = interval_ms(_check_interval(interval))
    chunk_start, chunk_end = chunk_bounds(index, interval)
    request_start, request_end = request_bounds(index, interval, server_time_ms)
    complete = settled(chunk_end - step, step, server_time_ms)
    label = _chunk_label(interval, chunk_start)
    requests = 0
    differences: list[str] = []
    for pair in range(1, MAX_FETCH_PAIRS + 1):
        fetches: list[list[Candle]] = []
        for _ in range(2):
            page = client.klines(symbol, interval, request_start, request_end, CHUNK_CANDLES)
            requests += 1
            _check_page(page, step, request_start, request_end, label)
            fetches.append([c for c in page if settled(c.open_ms, step, server_time_ms)])
        first, second = fetches
        if canonical_bytes(first) == canonical_bytes(second):
            return ChunkFetch(candles=first, pairs=pair, requests=requests, complete=complete)
        differences.append(_difference(first, second))
    raise BackfillError(
        f"{label}: all {MAX_FETCH_PAIRS} fetch pairs differed after the settle filter; the "
        f"chunk is not archived (last pair: {differences[-1]})"
    )


# -- archive files --------------------------------------------------------------------------------


def _chunk_relpath(interval: str, start_ms: int, suffix: str) -> str:
    return f"{ARCHIVE_RELDIR}/{interval}/{start_ms}.{suffix}"


def chunk_paths(interval: str, index: int, root: Path | None = None) -> tuple[Path, Path]:
    """``(csv, json sidecar)`` paths of chunk ``index``: ``raw/bingx/<interval>/<start_ms>.*``."""
    start, _ = chunk_bounds(index, interval)
    directory = _root(root) / ARCHIVE_RELDIR / interval
    return directory / f"{start}.csv", directory / f"{start}.json"


def _row(c: Candle) -> str:
    """The candle's ``open_ms,open,high,low,close,volume`` row as a chunk CSV holds it (no line
    end)."""
    return f"{c.open_ms},{c.open},{c.high},{c.low},{c.close},{c.volume}"


def _row_fields(text: str) -> list[str] | None:
    """The six fields of a store-format row (``open_ms`` digits, then plain decimal strings), or
    ``None`` when it is not one."""
    fields = text.split(",")
    if (
        len(fields) != len(BASE_COLUMNS)
        or not _OPEN_MS.fullmatch(fields[0])
        or not all(_PLAIN_DECIMAL.fullmatch(value) for value in fields[1:])
    ):
        return None
    return fields


def _chunk_csv(candles: Sequence[Candle]) -> bytes:
    """The store format: header line, one ``open_ms,open,high,low,close,volume`` row per candle."""
    rows = "".join(f"{_row(c)}\n" for c in candles)
    return CHUNK_HEADER + rows.encode("ascii")


def _parse_chunk_csv(data: bytes, interval: str, index: int, relpath: str) -> list[Candle]:
    """Strict parse of an archived chunk CSV (the bytes already match the sidecar's sha256)."""
    step = interval_ms(interval)
    start, end = chunk_bounds(index, interval)
    if not data.startswith(CHUNK_HEADER):
        raise BackfillError(f"{relpath}: unexpected header line")
    body = data[len(CHUNK_HEADER) :]
    if body and not body.endswith(b"\n"):
        raise BackfillError(f"{relpath}: the last row is not terminated (truncated file)")
    candles: list[Candle] = []
    previous: int | None = None
    for number, raw in enumerate(body.split(b"\n")[:-1], start=2):
        try:
            fields = _row_fields(raw.decode("ascii"))
        except UnicodeDecodeError:
            raise BackfillError(f"{relpath}:{number}: not ASCII") from None
        if fields is None:
            raise BackfillError(f"{relpath}:{number}: malformed row")
        open_ms = int(fields[0])
        if not start <= open_ms < end or open_ms % step:
            raise BackfillError(f"{relpath}:{number}: open time outside the chunk or off the grid")
        if previous is not None and open_ms <= previous:
            raise BackfillError(f"{relpath}:{number}: open times are not strictly increasing")
        candles.append(Candle(open_ms, *fields[1:]))
        previous = open_ms
    return candles


def _same(value: object, expected: object) -> bool:
    """Equal and of the same type (``True == 1`` must not pass for an integer field)."""
    return type(value) is type(expected) and value == expected


def _is_int(value: object) -> bool:
    return type(value) is int


def _load_sidecar(interval: str, index: int, base: Path) -> dict[str, Any]:
    """The validated sidecar of chunk ``index`` (its CSV is not read here)."""
    start, end = chunk_bounds(index, interval)
    csv_rel = _chunk_relpath(interval, start, "csv")
    json_rel = _chunk_relpath(interval, start, "json")
    csv_path, json_path = chunk_paths(interval, index, base)
    try:
        sidecar = json.loads(json_path.read_bytes().decode("ascii"))
    except FileNotFoundError:
        if not csv_path.exists():
            raise BackfillError(f"{csv_rel} is not archived (no CSV, no sidecar)") from None
        raise BackfillError(
            f"{csv_rel} has no sidecar ({json_rel}): a chunk without its commit marker is never "
            "used; inspect it (an interrupted write leaves it), removing it lets the next backfill "
            "fetch the chunk again"
        ) from None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BackfillError(f"{json_rel} is not a valid sidecar ({exc})") from exc
    expected: dict[str, object] = {
        "source": SOURCE,
        "symbol": SYMBOL,
        "interval": interval,
        "step_ms": interval_ms(interval),
        "chunk_candles": CHUNK_CANDLES,
        "chunk_index": index,
        "chunk_start_ms": start,
        "chunk_end_ms": end,
        "csv_relpath": csv_rel,
    }
    if not isinstance(sidecar, dict):
        raise BackfillError(f"{json_rel} is not a sidecar object")
    wrong = sorted(key for key, value in expected.items() if not _same(sidecar.get(key), value))
    if not _is_int(sidecar.get("schema_version")) or (
        sidecar["schema_version"] not in SIDECAR_SCHEMA_VERSIONS
    ):
        wrong = sorted([*wrong, "schema_version"])
    if wrong:
        raise BackfillError(f"{json_rel} does not describe this chunk (fields {wrong})")
    rows, size = sidecar.get("rows"), sidecar.get("csv_size")
    first, last = sidecar.get("first_open_ms"), sidecar.get("last_open_ms")
    if (
        not _is_int(rows)
        or rows < 1
        or not _is_int(size)
        or size < len(CHUNK_HEADER)
        or not isinstance(sidecar.get("csv_sha256"), str)
        or not _SHA256_HEX.fullmatch(sidecar["csv_sha256"])
        or not isinstance(sidecar.get("complete"), bool)
        or not _is_int(sidecar.get("server_time_ms"))
        or not _is_int(first)
        or not _is_int(last)
        or not start <= first <= last < end
    ):
        raise BackfillError(
            f"{json_rel}: rows, size, sha256, completeness, server time or range is missing or "
            "malformed (the archive never holds an empty chunk)"
        )
    _archive_only_opens(sidecar, interval, index)
    return sidecar


def _open_ranges(opens: Sequence[int], step_ms: int) -> list[list[int]]:
    """Increasing open times merged into ``[first_open, end)`` ranges of consecutive slots."""
    ranges: list[list[int]] = []
    for open_ms in opens:
        if ranges and ranges[-1][1] == open_ms:
            ranges[-1][1] = open_ms + step_ms
        else:
            ranges.append([open_ms, open_ms + step_ms])
    return ranges


def _archive_only_opens(sidecar: dict[str, Any], interval: str, index: int) -> list[int]:
    """Opens of the sidecar's ``archive_only`` candles, validated (none in a version 1 sidecar).

    ``archive_only`` is ``{"rows": n, "ranges": [[first_open, end), ...]}``: the archived
    candles that the chunk's latest fetch no longer returned, as maximal ranges of consecutive
    slots inside the chunk, on the grid, increasing, ``n`` opens in all.
    """
    if sidecar["schema_version"] == 1:
        return []
    step = interval_ms(interval)
    start, end = chunk_bounds(index, interval)
    value = sidecar.get("archive_only")
    opens: list[int] = []
    valid = (
        isinstance(value, dict)
        and set(value) == {"rows", "ranges"}
        and _is_int(value["rows"])
        and isinstance(value["ranges"], list)
    )
    previous_end = start - step
    for item in value["ranges"] if valid else []:
        if not (
            isinstance(item, list)
            and len(item) == 2
            and all(_is_int(bound) for bound in item)
            and previous_end < item[0] < item[1] <= end
            and item[0] % step == 0
            and item[1] % step == 0
        ):
            valid = False
            break
        opens.extend(range(item[0], item[1], step))
        previous_end = item[1]
    if not valid or len(opens) != value["rows"]:
        raise BackfillError(
            f"{_chunk_relpath(interval, start, 'json')}: archive_only is missing or malformed"
        )
    return opens


def _verified_csv(interval: str, index: int, sidecar: dict[str, Any], base: Path) -> bytes:
    """The chunk CSV's bytes, which must match the sidecar's size and sha256."""
    start, _ = chunk_bounds(index, interval)
    csv_path, _ = chunk_paths(interval, index, base)
    csv_rel = _chunk_relpath(interval, start, "csv")
    try:
        data = csv_path.read_bytes()
    except FileNotFoundError:
        raise BackfillError(
            f"{_chunk_relpath(interval, start, 'json')} has no CSV ({csv_rel})"
        ) from None
    if len(data) != sidecar["csv_size"] or sha256_bytes(data) != sidecar["csv_sha256"]:
        raise BackfillError(
            f"{csv_rel}: size or sha256 differs from its sidecar (modified, or an interrupted "
            "replacement); nothing is repaired automatically. Removing both files of the chunk "
            "lets the next backfill fetch it again: the candles the exchange no longer serves "
            "(past its retention) are lost, and those it serves with a revision recorded in "
            f"{revision_relpath(interval)} are archived with their recorded first-seen rows"
        )
    return data


def read_chunk(
    interval: str, index: int, root: Path | None = None
) -> tuple[list[Candle], dict[str, Any]]:
    """Candles and sidecar of an archived chunk, verified (:class:`BackfillError` otherwise).

    The sidecar must describe this chunk; the CSV must match its size and sha256 and parse
    strictly (store format, opens inside the chunk, on the grid, strictly increasing); the rows
    must match the sidecar's count and range, and hold every ``archive_only`` candle.
    """
    _check_interval(interval)
    base = _root(root)
    sidecar = _load_sidecar(interval, index, base)
    data = _verified_csv(interval, index, sidecar, base)
    relpath = sidecar["csv_relpath"]
    candles = _parse_chunk_csv(data, interval, index, relpath)
    if (
        len(candles) != sidecar["rows"]
        or candles[0].open_ms != sidecar["first_open_ms"]
        or candles[-1].open_ms != sidecar["last_open_ms"]
        or not set(_archive_only_opens(sidecar, interval, index)) <= {c.open_ms for c in candles}
    ):
        raise BackfillError(f"{relpath}: rows do not match the count and range of its sidecar")
    return candles, sidecar


def _scan_archive(interval: str, base: Path, *, verify: bool) -> dict[int, dict[str, Any]]:
    """Validated sidecars of every archived chunk of ``interval`` by index (CSV hashed if
    ``verify``). Hidden files (interrupted temporary writes) are ignored; anything else that is
    not a ``<chunk_start_ms>.csv``/``.json`` pair on the chunk grid is refused."""
    directory = base / ARCHIVE_RELDIR / interval
    if not directory.is_dir():
        if directory.exists():
            raise BackfillError(f"{ARCHIVE_RELDIR}/{interval} is not a directory")
        return {}
    span = chunk_span_ms(interval)
    found: dict[int, set[str]] = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        relpath = f"{ARCHIVE_RELDIR}/{interval}/{path.name}"
        match = _CHUNK_FILE.fullmatch(path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise BackfillError(
                f"{relpath}: unexpected entry in the chunk archive (only regular "
                "<chunk_start_ms>.csv and .json files belong there)"
            )
        start = int(match.group(1))
        if start % span:
            raise BackfillError(f"{relpath}: {start} is not on the {interval} chunk grid")
        found.setdefault(start // span, set()).add(match.group(2))
    archive: dict[int, dict[str, Any]] = {}
    for index in sorted(found):
        start = index * span
        if "csv" not in found[index]:
            raise BackfillError(
                f"{_chunk_relpath(interval, start, 'json')} has no CSV "
                f"({_chunk_relpath(interval, start, 'csv')})"
            )
        sidecar = _load_sidecar(interval, index, base)
        if verify:
            _verified_csv(interval, index, sidecar, base)
        archive[index] = sidecar
    return archive


def _sidecar(
    interval: str,
    index: int,
    candles: Sequence[Candle],
    fetch: ChunkFetch,
    archive_only: Sequence[int],
    *,
    complete: bool,
    server_time_ms: int,
    symbol: str,
    data: bytes,
    code: dict[str, Any],
) -> dict[str, Any]:
    step = interval_ms(interval)
    start, end = chunk_bounds(index, interval)
    request_start, request_end = request_bounds(index, interval, server_time_ms)
    opens = np.array([c.open_ms for c in candles], dtype=np.int64)
    first, last = int(opens[0]), int(opens[-1])
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "source": SOURCE,
        "symbol": symbol,
        "interval": interval,
        "step_ms": step,
        "chunk_candles": CHUNK_CANDLES,
        "chunk_index": index,
        "chunk_start_ms": start,
        "chunk_end_ms": end,
        "chunk_start_utc": utc_text(start),
        "chunk_end_utc": utc_text(end),
        "request": {"startTime": request_start, "endTime": request_end, "limit": CHUNK_CANDLES},
        "server_time_ms": server_time_ms,
        "server_time_utc": utc_text(server_time_ms),
        "settle_ms": SETTLE_MS,
        "settled_slots": _settled_slots(start, step, server_time_ms),
        "complete": complete,
        "rows": len(candles),
        "first_open_ms": first,
        "last_open_ms": last,
        "first_open_utc": utc_text(first),
        "last_open_utc": utc_text(last),
        "gaps": find_gaps(opens, step),
        "archive_only": {"rows": len(archive_only), "ranges": _open_ranges(archive_only, step)},
        "csv_relpath": _chunk_relpath(interval, start, "csv"),
        "csv_sha256": sha256_bytes(data),
        "csv_size": len(data),
        "requests": fetch.requests,
        "pairs": fetch.pairs,
        "mismatch_retries": fetch.pairs - 1,
        "fetched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": code,
    }


def _write_chunk(
    interval: str,
    index: int,
    candles: Sequence[Candle],
    fetch: ChunkFetch,
    archive_only: Sequence[int],
    *,
    complete: bool,
    server_time_ms: int,
    symbol: str,
    code: dict[str, Any],
    base: Path,
    previous: bytes | None,
) -> dict[str, Any]:
    """Write the CSV (unless its bytes are unchanged), then the sidecar: the commit marker."""
    data = _chunk_csv(candles)
    sidecar = _sidecar(
        interval,
        index,
        candles,
        fetch,
        archive_only,
        complete=complete,
        server_time_ms=server_time_ms,
        symbol=symbol,
        data=data,
        code=code,
    )
    csv_path, json_path = chunk_paths(interval, index, base)
    if data != previous:
        atomic_write_bytes(csv_path, data)
    text = json.dumps(sidecar, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=True)
    atomic_write_bytes(json_path, (text + "\n").encode("ascii"))
    return sidecar


def _revisions(stored: Sequence[Candle], fetched: Sequence[Candle]) -> list[tuple[Candle, Candle]]:
    """``(archived, served)`` for every archived candle the fetch returns with other strings."""
    by_open = {c.open_ms: c for c in fetched}
    return [(c, by_open[c.open_ms]) for c in stored if by_open.get(c.open_ms, c) != c]


def _refreshed(
    stored: Sequence[Candle], fetched: Sequence[Candle]
) -> tuple[list[Candle], list[int]]:
    """An archived incomplete chunk after its refetch: its candles and its archive-only opens.

    Archived candles are immutable: a revised one (:func:`_revisions`, recorded by the caller)
    keeps its archived, first-seen row, and one that the refetch no longer returns (rolled off
    the exchange's retention, or no longer served) is kept from the archive. Refetched candles
    the archive does not hold are added. Returns the union in open order and the opens kept from
    the archive alone.
    """
    served = {c.open_ms for c in fetched}
    archived = {c.open_ms for c in stored}
    added = [c for c in fetched if c.open_ms not in archived]
    candles = sorted([*stored, *added], key=lambda candle: candle.open_ms)
    return candles, [c.open_ms for c in stored if c.open_ms not in served]


def _not_retention(
    interval: str, archive_only: Sequence[int], first_served: int | None, server_time_ms: int
) -> list[int]:
    """The archive-only opens of a refreshed chunk that the exchange's retention does not explain.

    Only 1m and 5m roll off (:data:`RETENTION_DAYS`): an archived candle of theirs rolled off
    when it opens more than the observed retention before ``server_time_ms`` and before the
    first candle the refetch returned (``first_served``, ``None`` when it returned nothing). The
    1h history is fixed and 15m has no known retention, so none of their candles is explained. A
    chunk holding an unexplained one stays incomplete: the next run fetches it again.
    """
    days = RETENTION_DAYS.get(interval)
    if days is None:
        return list(archive_only)
    cut = server_time_ms - days * DAY_MS
    if first_served is not None:
        cut = min(cut, first_served)
    return [open_ms for open_ms in archive_only if open_ms >= cut]


# -- revised candles ------------------------------------------------------------------------------


def revision_relpath(interval: str) -> str:
    """``raw/bingx/revision-<interval>.json``: every revision the backfills of ``interval``
    observed (:func:`read_revisions`)."""
    return f"{ARCHIVE_RELDIR}/revision-{_check_interval(interval)}.json"


# One revision: its open, the archived row and the other row the exchange served.
RevisionKey = tuple[int, str, str]
_REVISION_KEYS = frozenset(
    {
        "chunk_start_ms",
        "chunk_start_utc",
        "csv_relpath",
        "open_ms",
        "open_utc",
        "archived",
        "refetched",
        "first_server_time_ms",
        "first_server_time_utc",
        "last_server_time_ms",
        "last_server_time_utc",
    }
)


def _revision(
    interval: str, archived: Candle, refetched: Candle, server_time_ms: int
) -> dict[str, Any]:
    """The record entry of a revision observed (first and last) at ``server_time_ms``: the
    archived row and the other row the exchange served for its open."""
    start, _ = chunk_bounds(chunk_index(archived.open_ms, interval), interval)
    return {
        "chunk_start_ms": start,
        "chunk_start_utc": utc_text(start),
        "csv_relpath": _chunk_relpath(interval, start, "csv"),
        "open_ms": archived.open_ms,
        "open_utc": utc_text(archived.open_ms),
        "archived": _row(archived),
        "refetched": _row(refetched),
        "first_server_time_ms": server_time_ms,
        "first_server_time_utc": utc_text(server_time_ms),
        "last_server_time_ms": server_time_ms,
        "last_server_time_utc": utc_text(server_time_ms),
    }


def _revision_key(entry: dict[str, Any]) -> RevisionKey:
    """What makes a revision the same one on a later run: its open and both rows."""
    return entry["open_ms"], entry["archived"], entry["refetched"]


def _revision_order(entry: dict[str, Any]) -> tuple[int, int, str, str]:
    return entry["open_ms"], entry["first_server_time_ms"], entry["archived"], entry["refetched"]


def _row_candle(text: object) -> Candle | None:
    """The candle of a store-format row (``None`` when ``text`` is not one)."""
    fields = _row_fields(text) if isinstance(text, str) else None
    return None if fields is None else Candle(int(fields[0]), *fields[1:])


def _valid_revision(entry: object, interval: str) -> bool:
    """Whether ``entry`` is the record entry of a revision of an ``interval`` candle: two
    different rows of one open on the grid, settled at the first observation, which is not after
    the last, and every derived field (chunk, path, UTC texts) consistent with them."""
    if not isinstance(entry, dict) or set(entry) != _REVISION_KEYS:
        return False
    times = ("chunk_start_ms", "open_ms", "first_server_time_ms", "last_server_time_ms")
    if not all(_is_int(entry[key]) and entry[key] >= 0 for key in times):
        return False
    archived, refetched = _row_candle(entry["archived"]), _row_candle(entry["refetched"])
    if (
        archived is None
        or refetched is None
        or archived == refetched
        or archived.open_ms != refetched.open_ms
    ):
        return False
    step = interval_ms(interval)
    first, last = entry["first_server_time_ms"], entry["last_server_time_ms"]
    try:  # the open, the chunk, its path and every UTC text follow from the rows and the times
        rebuilt = _revision(interval, archived, refetched, first)
        rebuilt.update(last_server_time_ms=last, last_server_time_utc=utc_text(last))
    except (OverflowError, ValueError):
        return False
    return (
        rebuilt == entry
        and archived.open_ms % step == 0
        and settled(archived.open_ms, step, first)
        and first <= last
    )


def _schema_1_revisions(record: dict[str, Any], interval: str) -> list[dict[str, Any]] | None:
    """The revision a schema-1 record holds (the first revised candle of its chunk, observed once
    by a backfill that then stopped) as a current entry; ``None`` when the record is malformed."""
    values = [record.get("archived"), record.get("served")]
    if not (
        all(
            isinstance(value, dict)
            and set(value) == set(VALUE_FIELDS)
            and all(isinstance(text, str) for text in value.values())
            for value in values
        )
        and _is_int(record.get("first_open_ms"))
        and _is_int(record.get("server_time_ms"))
        and _is_int(record.get("revised_candles"))
        and record["revised_candles"] >= 1
    ):
        return None
    archived, refetched = (
        Candle(record["first_open_ms"], *(value[name] for name in VALUE_FIELDS)) for value in values
    )
    try:
        entry = _revision(interval, archived, refetched, record["server_time_ms"])
    except (OverflowError, ValueError):
        return None
    if not (
        _same(record.get("chunk_start_ms"), entry["chunk_start_ms"])
        and record.get("csv_relpath") == entry["csv_relpath"]
    ):
        return None
    return [entry]


def read_revisions(interval: str, root: Path | None = None) -> list[dict[str, Any]]:
    """The validated revisions recorded for ``interval``, in open order; ``[]`` when none is.

    A revision is an archived candle that a refetch of its chunk returned with other strings
    (D-045): the archive keeps the first-seen row and the backfill goes on. Each entry holds the
    chunk, the open time, the ``archived`` and the ``refetched`` rows (store format, ``open_ms``
    first, no line end) and the server times of the first and the last run that observed that
    pair of rows; a later run that observes it again changes only the last. A schema-1 record
    (one revision, from a backfill that stopped at it) is read as that revision. A record that
    does not describe revisions of ``interval`` raises :class:`BackfillError`.
    """
    relpath = revision_relpath(interval)
    try:
        record = json.loads((_root(root) / relpath).read_bytes().decode("ascii"))
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BackfillError(f"{relpath} is not a valid revision record ({exc})") from exc
    expected: dict[str, object] = {"source": SOURCE, "symbol": SYMBOL, "interval": interval}
    entries: object = None
    if isinstance(record, dict) and all(_same(record.get(k), v) for k, v in expected.items()):
        if _same(record.get("schema_version"), 1):
            entries = _schema_1_revisions(record, interval)
        elif _same(record.get("schema_version"), REVISION_SCHEMA_VERSION) and (
            record.get("columns") == ROW_COLUMNS
        ):
            entries = record.get("revisions")
    if not (
        isinstance(entries, list)
        and entries
        and all(_valid_revision(entry, interval) for entry in entries)
        and len({_revision_key(entry) for entry in entries}) == len(entries)
    ):
        raise BackfillError(
            f"{relpath} does not describe revised {interval} candles; nothing is repaired "
            "automatically: inspect it (it is the only record of what the exchange revised)"
        )
    return sorted(entries, key=_revision_order)


def _observe_revisions(
    known: dict[RevisionKey, dict[str, Any]],
    interval: str,
    revised: Sequence[tuple[Candle, Candle]],
    server_time_ms: int,
) -> int:
    """Add the ``(archived, refetched)`` pairs a refetch at ``server_time_ms`` returned to
    ``known``, each once: a pair recorded before gets this run as its last observation. Returns
    how many pairs are new."""
    new = 0
    for archived, refetched in revised:
        entry = _revision(interval, archived, refetched, server_time_ms)
        key = _revision_key(entry)
        if key not in known:
            known[key] = entry
            new += 1
        elif server_time_ms > known[key]["last_server_time_ms"]:
            known[key].update(
                last_server_time_ms=server_time_ms, last_server_time_utc=utc_text(server_time_ms)
            )
    return new


def _write_revisions(
    interval: str, known: dict[RevisionKey, dict[str, Any]], *, code: dict[str, Any], base: Path
) -> None:
    """Record every revision of ``known`` (:func:`read_revisions` reads them back)."""
    record = {
        "schema_version": REVISION_SCHEMA_VERSION,
        "source": SOURCE,
        "symbol": SYMBOL,
        "interval": interval,
        "columns": ROW_COLUMNS,
        "revisions": sorted(known.values(), key=_revision_order),
        "code": code,
    }
    text = json.dumps(record, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=True)
    atomic_write_bytes(base / revision_relpath(interval), (text + "\n").encode("ascii"))


def _recorded_first_seen(known: dict[RevisionKey, dict[str, Any]], start_ms: int) -> list[Candle]:
    """The first-seen rows that the recorded revisions hold for the chunk starting at
    ``start_ms``: for each open, the archived row of its earliest revision, in open order."""
    rows: dict[int, Candle] = {}
    for entry in sorted(known.values(), key=_revision_order):
        candle = _row_candle(entry["archived"])
        if entry["chunk_start_ms"] == start_ms and candle is not None:
            rows.setdefault(candle.open_ms, candle)
    return [rows[open_ms] for open_ms in sorted(rows)]


def _observed_warning(
    interval: str, start_ms: int, revised: Sequence[tuple[Candle, Candle]], new: int
) -> str:
    """The warning about the revised candles that a fetch of the chunk at ``start_ms`` returned,
    once they are recorded: open times only (the rows stay in the record, so no candle value is
    printed)."""
    opens = [utc_text(archived.open_ms) for archived, _ in revised]
    listed = ", ".join(opens[:LISTED_REVISIONS])
    if len(opens) > LISTED_REVISIONS:
        listed += f" and {len(opens) - LISTED_REVISIONS} more"
    return (
        f"bingx {_chunk_label(interval, start_ms)}: the exchange served {len(opens)} archived "
        f"candle(s) with other values ({new} of them not recorded before): {listed}; the archive "
        f"and the store keep the first-seen rows, and {revision_relpath(interval)} records every "
        "revision; report them to the owner"
    )


def _revisions_warning(interval: str, revisions: Sequence[dict[str, Any]]) -> str:
    """The status warning listing the recorded revisions of ``interval`` (open times and when
    they were seen; the rows stay in the record, so no candle value is printed)."""
    listed = []
    for entry in revisions[:LISTED_REVISIONS]:
        seen = entry["first_server_time_utc"]
        if entry["last_server_time_ms"] != entry["first_server_time_ms"]:
            seen += f" to {entry['last_server_time_utc']}"
        listed.append(f"{entry['open_utc']} (seen {seen})")
    more = len(revisions) - len(listed)
    return (
        f"bingx {interval}: {len(revisions)} revision(s) recorded in "
        f"{revision_relpath(interval)} (archived candles the exchange later served with other "
        f"values; the archive and the store keep the first-seen rows): {', '.join(listed)}"
        + (f" and {more} more" if more else "")
        + "; report them to the owner"
    )


# -- the history start ----------------------------------------------------------------------------


def history_relpath(interval: str) -> str:
    """``raw/bingx/history-<interval>.json``: the history start a walk established."""
    return f"{ARCHIVE_RELDIR}/history-{_check_interval(interval)}.json"


def read_history(interval: str, root: Path | None = None) -> dict[str, Any] | None:
    """The validated history-start record of ``interval``; ``None`` when no walk has written one.

    A backfill writes it when its walk established the history start (:func:`_established_start`)
    and the record does not already name it: the first candle of the oldest archived chunk, the
    stop reason and the server time of that run. A record that does not describe ``interval``
    raises :class:`BackfillError`.
    """
    relpath = history_relpath(interval)
    try:
        record = json.loads((_root(root) / relpath).read_bytes().decode("ascii"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BackfillError(f"{relpath} is not a valid history record ({exc})") from exc
    expected: dict[str, object] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "source": SOURCE,
        "symbol": SYMBOL,
        "interval": interval,
    }
    valid = (
        isinstance(record, dict)
        and all(_same(record.get(key), value) for key, value in expected.items())
        and record.get("stop_reason") in HISTORY_STOP_REASONS
        and all(
            _is_int(record.get(key)) and record[key] >= 0
            for key in ("history_start_ms", "oldest_chunk_start_ms", "server_time_ms")
        )
    )
    if valid:
        chunk_start, _ = chunk_bounds(chunk_index(record["history_start_ms"], interval), interval)
        valid = chunk_start == record["oldest_chunk_start_ms"]
    if not valid:
        raise BackfillError(
            f"{relpath} does not describe the {interval} history start; nothing is repaired "
            "automatically: inspect it (removing it loses only the record, and a later walk may "
            "establish the start again)"
        )
    return record


def _established_start(
    interval: str,
    archive: dict[int, dict[str, Any]],
    *,
    served_first: dict[int, int | None],
    record: dict[str, Any] | None,
) -> int | None:
    """The history start a walk that ended below the oldest archived chunk establishes, if any.

    That chunk's first archived candle ``F`` is the start when nothing older was served when
    ``F`` was: this run's fetch of the chunk returned ``F`` (``served_first`` maps every chunk
    fetched in this run to its first served open); or ``F`` opens after the chunk's start, so the
    fetch that archived it found nothing before it in its chunk; or ``record`` already names
    ``F``. Otherwise (``F`` opens at its chunk's start and an earlier run archived it) the empty
    chunks below may have rolled off the exchange's retention since a run bounded with
    ``max_chunks`` archived ``F``: ``None``, the start stays open.
    """
    oldest = min(archive)
    first = archive[oldest]["first_open_ms"]
    if (
        served_first.get(oldest) == first
        or first > chunk_bounds(oldest, interval)[0]
        or (record is not None and record["history_start_ms"] == first)
    ):
        return first
    return None


def _write_history(
    interval: str,
    history_start_ms: int,
    *,
    stop_reason: str,
    server_time_ms: int,
    code: dict[str, Any],
    base: Path,
) -> None:
    """Record the history start a walk established (:func:`read_history` reads it back)."""
    record = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "source": SOURCE,
        "symbol": SYMBOL,
        "interval": interval,
        "stop_reason": stop_reason,
        "history_start_ms": history_start_ms,
        "history_start_utc": utc_text(history_start_ms),
        "oldest_chunk_start_ms": chunk_bounds(chunk_index(history_start_ms, interval), interval)[0],
        "server_time_ms": server_time_ms,
        "server_time_utc": utc_text(server_time_ms),
        "code": code,
    }
    text = json.dumps(record, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=True)
    atomic_write_bytes(base / history_relpath(interval), (text + "\n").encode("ascii"))


def _history_provenance(
    interval: str, archive: dict[int, dict[str, Any]], base: Path
) -> dict[str, Any]:
    """Whether the archive's first candle is the history start a walk established, and why not."""
    first = archive[min(archive)]["first_open_ms"]
    record = read_history(interval, base)
    out: dict[str, Any] = {
        "established": False,
        "history_start_ms": None,
        "history_start_utc": None,
        "stop_reason": None,
        "server_time_ms": None,
        "relpath": history_relpath(interval),
        "reason": None,
    }
    if record is None:
        out["reason"] = (
            "no backfill has established the start of the exchange's history (a run that its "
            "--max-chunks bound stops does not reach it)"
        )
    elif record["history_start_ms"] != first:
        out["reason"] = (
            f"the recorded history start {utc_text(record['history_start_ms'])} is not the "
            f"archive's first candle {utc_text(first)}"
        )
    else:
        out.update(
            established=True,
            history_start_ms=first,
            history_start_utc=utc_text(first),
            stop_reason=record["stop_reason"],
            server_time_ms=record["server_time_ms"],
        )
    return out


# -- backfill and merge ---------------------------------------------------------------------------


def backfill(
    client: PublicMarketClient,
    interval: str,
    *,
    symbol: str = "BTC-USDT",
    root: Path | None = None,
    max_chunks: int | None = None,
    warn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Extend the chunk archive of ``interval`` from the real market; return a run summary.

    Under the job lock (which first removes the temporary files of killed runs), the whole
    archive, its history-start record and its revision record are verified (no request is made
    if they are inconsistent), then the server time is read once. The walk starts at the chunk
    holding the newest settled candle and goes backwards: archived complete chunks are skipped
    without a request, archived incomplete chunks are fetched again and refreshed (a revised
    candle keeps its archived, first-seen row and the revision is recorded for
    :func:`read_revisions` before the chunk is written; candles the exchange no longer returns
    are kept, new ones added, and the chunk completes only when the retention explains every
    kept one, :func:`_not_retention`), other chunks are fetched and archived when non-empty. A
    chunk archived again after its files were removed takes the first-seen rows that its
    recorded revisions hold for the opens the exchange serves (another value served is a
    revision like any other). Each chunk's revisions, once recorded, go to ``warn`` (default:
    this module's logger) with their open times, so a caller reports them even when a later
    chunk of the walk fails. It stops after :data:`STOP_AFTER_EMPTY_CHUNKS` consecutive empty
    chunks older than the oldest archived chunk (``stop_reason`` ``"empty_chunks"``), after
    ``max_chunks`` fetched chunks (``"max_chunks"``) or at the epoch. ``history_start_ms`` is
    the start this walk established (:func:`_established_start`, recorded for
    :func:`read_history`), ``None`` otherwise (always after ``"max_chunks"``, which leaves the
    record as it is). ``kept_from_archive`` counts the archived candles that refetches no longer
    returned, ``kept_inside_served_range`` those of them inside the range the exchange still
    serves (the whole history of 1h and 15m; for 1m and 5m, not older than the observed
    retention or than the first candle their chunk's refetch returned): candles the exchange
    stopped serving, not retention, whose chunk stays incomplete. ``revised_candles`` counts the
    first-seen candles this run's fetches returned with other strings (their opens in
    ``revised_opens``), ``new_revisions`` the revisions among them that no earlier run had
    recorded.
    """
    _check_interval(interval)
    if symbol != SYMBOL:
        raise ValueError(f"the BingX archive holds {SYMBOL} only, not {symbol!r}")
    if max_chunks is not None and (
        isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1
    ):
        raise ValueError(f"max_chunks must be a positive integer or None, got {max_chunks!r}")
    base = _root(root)
    step = interval_ms(interval)
    span = chunk_span_ms(interval)
    say = warn if warn is not None else _LOG.warning
    with job_lock(LOCK_RELPATH, base) as removed:
        archive = _scan_archive(interval, base, verify=True)
        record = read_history(interval, base)
        known = {_revision_key(entry): entry for entry in read_revisions(interval, base)}
        server_time_ms = _non_negative_int(client.server_time_ms(), "server time")
        latest = max((sidecar["server_time_ms"] for sidecar in archive.values()), default=None)
        if latest is not None and server_time_ms < latest:
            raise BackfillError(
                f"server time {utc_text(server_time_ms)} is earlier than the {utc_text(latest)} of "
                f"an archived {interval} chunk; refusing to run on a clock that went backwards"
            )
        newest_open = (server_time_ms - SETTLE_MS - step) // step * step
        if newest_open < 0:
            raise BackfillError(f"server time {server_time_ms} precedes every settled candle")
        code = code_version()
        new_chunks: list[int] = []
        refreshed: list[int] = []
        empty: list[int] = []
        served_first: dict[int, int | None] = {}
        revised_opens: list[int] = []
        new_candles = kept = kept_inside = new_revisions = 0
        fetched = skipped = requests = retries = 0
        oldest = min(archive, default=None)
        empty_run = 0
        stop_reason = "epoch"
        index = newest_open // span
        while index >= 0:
            stored = archive.get(index)
            if stored is not None and stored["complete"]:
                skipped += 1
                index -= 1
                continue
            if max_chunks is not None and fetched >= max_chunks:
                stop_reason = "max_chunks"
                break
            start = index * span
            fetch = fetch_chunk(
                client, interval, index, server_time_ms=server_time_ms, symbol=symbol
            )
            fetched += 1
            requests += fetch.requests
            retries += fetch.pairs - 1
            served_first[index] = fetch.candles[0].open_ms if fetch.candles else None
            _LOG.info(
                "bingx %s: %d settled candles, %s, %d pair(s)",
                _chunk_label(interval, start),
                len(fetch.candles),
                "complete" if fetch.complete else "incomplete",
                fetch.pairs,
            )
            previous: bytes | None = None
            candles: list[Candle] = fetch.candles
            archive_only: list[int] = []
            unexplained: list[int] = []
            if stored is not None:
                first_seen, _ = read_chunk(interval, index, root=base)
            else:  # only a chunk whose files were removed has recorded revisions
                first_seen = _recorded_first_seen(known, start)
            revised = _revisions(first_seen, fetch.candles)
            if revised:  # recorded before the chunk is written; the archive keeps them
                new = _observe_revisions(known, interval, revised, server_time_ms)
                _write_revisions(interval, known, code=code, base=base)
                new_revisions += new
                revised_opens += [archived.open_ms for archived, _ in revised]
                say(_observed_warning(interval, start, revised, new))
            if stored is not None:
                candles, archive_only = _refreshed(first_seen, fetch.candles)
                unexplained = _not_retention(
                    interval, archive_only, served_first[index], server_time_ms
                )
                previous = _chunk_csv(first_seen)
            elif revised:
                recorded = {archived.open_ms: archived for archived, _ in revised}
                candles = [recorded.get(c.open_ms, c) for c in fetch.candles]
            if candles:
                archive[index] = _write_chunk(
                    interval,
                    index,
                    candles,
                    fetch,
                    archive_only,
                    complete=fetch.complete and not unexplained,
                    server_time_ms=server_time_ms,
                    symbol=symbol,
                    code=code,
                    base=base,
                    previous=previous,
                )
            if stored is not None:
                refreshed.append(start)
                new_candles += len(candles) - stored["rows"]
                kept += len(archive_only)
                kept_inside += len(unexplained)
                empty_run = 0
            elif candles:
                new_chunks.append(start)
                new_candles += len(candles)
                oldest = index if oldest is None else min(oldest, index)
                empty_run = 0
            else:
                empty.append(start)
                if oldest is None or index < oldest:
                    empty_run += 1
                    if empty_run >= STOP_AFTER_EMPTY_CHUNKS:
                        stop_reason = "empty_chunks"
                        break
            index -= 1
        history_start: int | None = None
        if archive and stop_reason in HISTORY_STOP_REASONS:
            history_start = _established_start(
                interval, archive, served_first=served_first, record=record
            )
            if history_start is not None and (
                record is None or record["history_start_ms"] != history_start
            ):
                _write_history(
                    interval,
                    history_start,
                    stop_reason=stop_reason,
                    server_time_ms=server_time_ms,
                    code=code,
                    base=base,
                )
    return _summary(
        interval,
        archive,
        server_time_ms=server_time_ms,
        newest_open=newest_open,
        stop_reason=stop_reason,
        history_start=history_start,
        counts={
            "chunks_fetched": fetched,
            "chunks_skipped": skipped,
            "klines_requests": requests,
            "mismatch_retries": retries,
            "new_candles": new_candles,
            "kept_from_archive": kept,
            "kept_inside_served_range": kept_inside,
            "revised_candles": len(revised_opens),
            "new_revisions": new_revisions,
        },
        new_chunks=new_chunks,
        refreshed=refreshed,
        empty=empty,
        revised_opens=revised_opens,
        removed_temporaries=removed,
    )


def _summary(
    interval: str,
    archive: dict[int, dict[str, Any]],
    *,
    server_time_ms: int,
    newest_open: int,
    stop_reason: str,
    history_start: int | None,
    counts: dict[str, int],
    new_chunks: list[int],
    refreshed: list[int],
    empty: list[int],
    revised_opens: list[int],
    removed_temporaries: list[dict[str, Any]],
) -> dict[str, Any]:
    span = chunk_span_ms(interval)
    indexes = sorted(archive)
    gap_chunks = [start for start in empty if indexes and indexes[0] < start // span < indexes[-1]]
    return {
        "source": SOURCE,
        "symbol": SYMBOL,
        "interval": interval,
        "server_time_ms": server_time_ms,
        "server_time_utc": utc_text(server_time_ms),
        "settle_ms": SETTLE_MS,
        "newest_settled_open_ms": newest_open,
        **counts,
        "new_chunks": sorted(new_chunks),
        "refreshed_chunks": sorted(refreshed),
        "empty_chunks": sorted(empty),
        "gap_chunks": sorted(gap_chunks),
        "revised_opens": sorted(revised_opens),
        "revision_relpath": revision_relpath(interval),
        "stop_reason": stop_reason,
        "history_start_ms": history_start,
        "history_start_utc": None if history_start is None else utc_text(history_start),
        "removed_temporaries": list(removed_temporaries),
        "archive": {
            "chunks": len(indexes),
            "candles": sum(archive[i]["rows"] for i in indexes),
            "first_open_ms": archive[indexes[0]]["first_open_ms"] if indexes else None,
            "last_open_ms": archive[indexes[-1]]["last_open_ms"] if indexes else None,
            "incomplete_chunks": [i * span for i in indexes if not archive[i]["complete"]],
        },
    }


def _frame(candles: Sequence[Candle]) -> pd.DataFrame:
    """A ``BASE_COLUMNS`` string frame (``open_ms`` int64, values ``str``) of ``candles``."""
    data: dict[str, Any] = {"open_ms": np.array([c.open_ms for c in candles], dtype=np.int64)}
    for name in VALUE_FIELDS:
        data[name] = pd.Series([getattr(c, name) for c in candles], dtype=object)
    return pd.DataFrame(data)


def merge_chunks(interval: str, *, root: Path | None = None) -> dict[str, Any]:
    """Rebuild ``candles/bingx/<interval>`` from every archived chunk; return the store meta.

    Complete and incomplete chunks alike are verified (:func:`read_chunk`) and appended in
    order; the store's gaps include whole empty chunks between archived ones. The provenance
    lists every chunk (relative path, rows, sha256, completeness, server time, candles kept from
    the archive alone), names the incomplete ones and says whether the store's first candle is
    the history start a backfill established (``history_start``, from :func:`read_history`;
    otherwise the reason). Takes the job lock, so it never reads a chunk a backfill is
    replacing; taking it removes the temporary files of killed runs (logged).
    """
    _check_interval(interval)
    base = _root(root)
    with job_lock(LOCK_RELPATH, base) as removed:
        if removed:
            _LOG.warning(
                "removed %d temporary files of killed runs (%d bytes): %s",
                len(removed),
                sum(item["bytes"] for item in removed),
                ", ".join(item["relpath"] for item in removed),
            )
        archive = _scan_archive(interval, base, verify=False)
        if not archive:
            raise BackfillError(
                f"no archived {interval} chunks under {ARCHIVE_RELDIR}/{interval}: run the "
                "backfill first"
            )
        history_start = _history_provenance(interval, archive, base)
        chunks: list[dict[str, Any]] = []
        with StoreWriter(StoreId(SOURCE, interval), root=base) as writer:
            for index in sorted(archive):
                candles, sidecar = read_chunk(interval, index, root=base)
                writer.append(_frame(candles))
                chunks.append(
                    {
                        "relpath": sidecar["csv_relpath"],
                        "chunk_start_ms": sidecar["chunk_start_ms"],
                        "rows": sidecar["rows"],
                        "first_open_ms": sidecar["first_open_ms"],
                        "last_open_ms": sidecar["last_open_ms"],
                        "csv_sha256": sidecar["csv_sha256"],
                        "complete": sidecar["complete"],
                        "server_time_ms": sidecar["server_time_ms"],
                        "archive_only_rows": len(_archive_only_opens(sidecar, interval, index)),
                    }
                )
            provenance = {
                "kind": "bingx_chunks",
                "description": (
                    "BingX real-market public klines, archived in chunks of 1000 candles, each "
                    "fetched twice until the settled candles matched"
                ),
                "symbol": SYMBOL,
                "interval": interval,
                "chunk_candles": CHUNK_CANDLES,
                "settle_ms": SETTLE_MS,
                "chunks": chunks,
                "incomplete_chunks": [c["relpath"] for c in chunks if not c["complete"]],
                "history_start": history_start,
            }
            return writer.finish(provenance=provenance)


def retention_warnings(root: Path | None = None, *, now_ms: int) -> list[str]:
    """Warnings about the chunk archive: rolling retention, revisions and refreshes left behind.

    A rolling-retention interval (:data:`RETENTION_DAYS`) warns when it has no archived chunk,
    when its archive cannot be read, or when its newest archived candle opened more than a third
    of the observed retention before ``now_ms`` (from then on, candles not yet archived roll off
    the exchange within two thirds of it). Every interval lists its recorded revisions
    (:func:`read_revisions`: open times and when they were seen, never candle values, so no
    value of the sealed segment is printed; or says the record cannot be read); warns about an
    incomplete chunk holding archived candles its last refetch no longer returned (inside the
    range the exchange still serves: the chunk is fetched again every run); and about any other
    incomplete chunk older than its newest archived chunk (a backfill stopped before refreshing
    it: a failed run or a ``--max-chunks`` bound). The archive is read under
    ``raw/bingx/.lock`` shared; while a run holds it, a single warning says it was not checked.
    """
    now = _non_negative_int(now_ms, "now_ms")
    base = _root(root)
    try:
        with shared_lock(base / LOCK_RELPATH):
            return _archive_warnings(base, now)
    except LockBusyError:
        return [f"bingx: a run holds {LOCK_RELPATH}; the chunk archive was not checked"]


def _archive_warnings(base: Path, now: int) -> list[str]:
    archives: dict[str, dict[int, dict[str, Any]]] = {}
    warnings: list[str] = []
    for interval in INTERVALS:
        try:
            archives[interval] = _scan_archive(interval, base, verify=False)
        except BackfillError as exc:
            warnings.append(f"bingx {interval}: the chunk archive cannot be read ({exc})")
    for interval, days in RETENTION_DAYS.items():
        archive = archives.get(interval)
        command = f"`kronos-agent data bingx --interval {interval}`"
        if archive is None:
            continue
        if not archive:
            warnings.append(
                f"bingx {interval}: no archived chunks; the real market keeps about {days} days "
                f"of {interval} candles, so run {command} before they roll off"
            )
            continue
        newest = archive[max(archive)]["last_open_ms"]
        age = now - newest
        if age > days * DAY_MS // 3:
            warnings.append(
                f"bingx {interval}: the newest archived candle ({utc_text(newest)}) opened "
                f"{age / DAY_MS:.1f} days ago, more than a third of the observed {days}-day "
                f"retention; run {command}"
            )
    for interval in INTERVALS:
        try:
            revisions = read_revisions(interval, base)
        except BackfillError as exc:
            warnings.append(f"bingx {interval}: {exc}")
        else:
            if revisions:
                warnings.append(_revisions_warning(interval, revisions))
        archive = archives.get(interval)
        if archive is None:
            continue
        command = f"`kronos-agent data bingx --interval {interval}`"
        newest = max(archive, default=None)
        for index in sorted(archive):
            sidecar = archive[index]
            if sidecar["complete"]:
                continue
            kept = _archive_only_opens(sidecar, interval, index)
            if kept:
                warnings.append(
                    f"bingx {interval}: {sidecar['csv_relpath']} is incomplete: its last refetch "
                    f"({utc_text(sidecar['server_time_ms'])}) no longer returned {len(kept)} "
                    "archived candle(s) (kept, sidecar archive_only), not all explained by the "
                    "exchange's retention, so every backfill fetches the chunk again; report "
                    "this to the owner"
                )
            elif index != newest:
                warnings.append(
                    f"bingx {interval}: {sidecar['csv_relpath']} is incomplete but older than the "
                    "newest archived chunk, so a backfill stopped before refreshing it (a failed "
                    f"run, or a --max-chunks bound); run {command}"
                )
    return warnings


__all__ = [
    "ARCHIVE_RELDIR",
    "CHUNK_CANDLES",
    "HISTORY_SCHEMA_VERSION",
    "INTERVALS",
    "LOCK_RELPATH",
    "MAX_FETCH_PAIRS",
    "RETENTION_DAYS",
    "REVISION_SCHEMA_VERSION",
    "REVISION_SCHEMA_VERSIONS",
    "SETTLE_MS",
    "SIDECAR_SCHEMA_VERSION",
    "STOP_AFTER_EMPTY_CHUNKS",
    "BackfillError",
    "ChunkFetch",
    "backfill",
    "chunk_bounds",
    "chunk_index",
    "chunk_paths",
    "chunk_span_ms",
    "fetch_chunk",
    "history_relpath",
    "merge_chunks",
    "read_chunk",
    "read_history",
    "read_revisions",
    "request_bounds",
    "retention_warnings",
    "revision_relpath",
    "settled",
]
