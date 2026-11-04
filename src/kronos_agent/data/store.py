"""Candle stores: one CSV of the exchange's decimal strings per source and interval.

``candles/<source>/<interval>.csv`` holds a header line and one row per candle, ascending and
unique by ``open_ms``, ``\\n`` endings, no quoting; the values are the exchange's strings verbatim
(plain decimals, never re-formatted), so the same rows always give the same bytes and a window's
hash can be recomputed from the store. ``<interval>.meta.json`` is the commit marker written after
the CSV: row count, range, gaps, OHLC-relation violations (counted and sampled, never fixed), the
CSV sha256 and size, the data provenance and the code (``docs/DATA.md``, "Store format"). Nothing
here fills, interpolates or deduplicates candles: a missing candle is a gap.

Two read paths, on purpose:

- :func:`read_store` is for integrity work only (checksums, gaps, merges, the resample check,
  status), whose values never feed a study;
- :func:`read_study_frame` is the only path for studies and window sources: it refuses any range
  reaching a candle that opens at or after the sealed start without a ``SealedAccess``
  (:func:`kronos_agent.data.partition.require_study_range`).

A ranged read locates ``[start_ms, end_ms)`` by binary search over the byte offsets of the sorted
rows and parses only those rows, so values outside the range (sealed ones included) are never
loaded. Numbers are parsed with ``float_precision="round_trip"``, the correctly rounded conversion
that equals Python's ``float()`` (pandas' default parser and ``pandas.to_numeric`` are not, e.g.
for ``"465655238.20240235"``).

Stores are rebuilt only by the runs that hold their source's job lock (:data:`LOCK_RELPATHS`)
exclusively (:func:`job_lock`). Every read takes that lock shared (:func:`store_read_lock`), so it
never overlaps a rebuild (a read during one is refused with :class:`StoreBusyError`), and opens
the CSV once: the bytes whose sha256 is checked are the bytes parsed. A run killed by a signal
can leave its temporary files behind; :func:`leftover_temporaries` lists them and the next run
that takes the job lock removes them.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO, Self

import numpy as np
import pandas as pd

from ..exchange.public_market import interval_ms
from ..hashing import CHUNK_SIZE
from ..paths import data_dir
from ..provenance import code_version
from .fsutil import (
    FILE_MODE,
    LockBusyError,
    atomic_write_bytes,
    ensure_directory,
    exclusive_lock,
    fsync_directory,
    lock_is_held,
    remove_temporary_files,
    shared_lock,
    temporary_files,
)
from .partition import SealedAccess, require_study_range
from .timeutil import utc_text

SCHEMA_VERSION = 1
INTERVALS: tuple[str, ...] = ("1m", "5m", "15m", "1h")
SOURCE_SYMBOLS: dict[str, str] = {
    "binance": "BTCUSDT",
    "binance_official": "BTCUSDT",
    "bingx": "BTC-USDT",
}
# The job lock of each source: the runs that write its stores hold it exclusively (the Binance
# sync and builds, the BingX backfill and merge), every store read holds it shared.
LOCK_RELPATHS: dict[str, str] = {
    "binance": "raw/binance/.lock",
    "binance_official": "raw/binance/.lock",
    "bingx": "raw/bingx/.lock",
}
# The directories (subdirectories included) that only runs holding each job lock write: a
# temporary file found there while nobody holds the lock is the leftover of a killed write.
LOCKED_RELDIRS: dict[str, tuple[str, ...]] = {
    "raw/binance/.lock": ("candles/binance", "candles/binance_official", "raw/binance"),
    "raw/bingx/.lock": ("candles/bingx", "raw/bingx"),
}
BASE_COLUMNS: tuple[str, ...] = ("open_ms", "open", "high", "low", "close", "volume")
BINANCE_COLUMNS: tuple[str, ...] = BASE_COLUMNS + (
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
)
DECIMAL_RE = r"^-?[0-9]+(\.[0-9]+)?$"
COUNT_RE = r"^[0-9]+$"
MAX_VIOLATION_SAMPLES = 10

_DECIMAL = re.compile(DECIMAL_RE)
_COUNT = re.compile(COUNT_RE)
_INTEGER_COLUMNS = frozenset({"open_ms", "count"})


class StoreError(Exception):
    """A frame or a store that does not have the store format."""


class StoreMissingError(StoreError):
    """The store has not been built (no meta file)."""


class StoreIntegrityError(StoreError):
    """The stored CSV does not match its meta, or the files are malformed."""


class StoreBusyError(LockBusyError, StoreError):
    """A run that writes the store's source holds its job lock: the store is not read now."""


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else data_dir()


@dataclass(frozen=True)
class StoreId:
    """One store: a source (``binance``, ``binance_official``, ``bingx``) and an interval."""

    source: str
    interval: str

    def __post_init__(self) -> None:
        if self.source not in SOURCE_SYMBOLS:
            raise ValueError(
                f"unknown source {self.source!r}; expected one of {list(SOURCE_SYMBOLS)}"
            )
        if self.interval not in INTERVALS:
            raise ValueError(
                f"unknown interval {self.interval!r}; expected one of {list(INTERVALS)}"
            )

    @property
    def symbol(self) -> str:
        return SOURCE_SYMBOLS[self.source]

    @property
    def columns(self) -> tuple[str, ...]:
        return BASE_COLUMNS if self.source == "bingx" else BINANCE_COLUMNS

    @property
    def step_ms(self) -> int:
        return interval_ms(self.interval)

    @property
    def relpath(self) -> str:
        """``candles/<source>/<interval>.csv``, relative to the data root."""
        return f"candles/{self.source}/{self.interval}.csv"

    @property
    def meta_relpath(self) -> str:
        return f"candles/{self.source}/{self.interval}.meta.json"

    @property
    def lock_relpath(self) -> str:
        """The job lock of the store's source (:data:`LOCK_RELPATHS`)."""
        return LOCK_RELPATHS[self.source]

    def csv_path(self, root: Path | None = None) -> Path:
        return _root(root) / self.relpath

    def meta_path(self, root: Path | None = None) -> Path:
        return _root(root) / self.meta_relpath

    def lock_path(self, root: Path | None = None) -> Path:
        return _root(root) / self.lock_relpath


# -- validation -----------------------------------------------------------------------------------


def _ohlc_violation_open_ms(frame: pd.DataFrame) -> list[int]:
    """``open_ms`` of rows with ``high < max(o, c, l)``, ``low > min(o, c, h)`` or ``volume < 0``.

    Compared as decimals: ``float`` would call ``"100.09999999999999999"`` and ``"100.1"`` equal.
    """
    if frame.empty:
        return []
    columns = [frame[name].to_numpy(dtype=object) for name in ("open", "high", "low", "close")]
    opens, highs, lows, closes = ([Decimal(value) for value in column] for column in columns)
    volumes = [Decimal(value) for value in frame["volume"].to_numpy(dtype=object)]
    bad: list[int] = []
    for open_ms, op, hi, lo, cl, vol in zip(
        frame["open_ms"].to_numpy(dtype=np.int64), opens, highs, lows, closes, volumes, strict=True
    ):
        if hi < op or hi < cl or hi < lo or lo > op or lo > cl or vol < 0:
            bad.append(int(open_ms))
    return bad


def validate_frame(frame: pd.DataFrame, store_id: StoreId) -> dict[str, Any]:
    """Check a string frame against the store format; return its OHLC-relation violations.

    Structure errors raise :class:`StoreError`: the exact columns in order; ``open_ms`` int64,
    non-negative, strictly increasing and a multiple of the interval; every value a Python
    ``str`` matching :data:`DECIMAL_RE` (``count``: :data:`COUNT_RE`), which also excludes
    ``,``, ``"`` and newlines. Violations of the OHLC relations (``high < max(o, c, l)``,
    ``low > min(o, c, h)``, ``volume < 0``) are counted with ``Decimal`` comparisons, never
    fixed: ``{"ohlc_violations": n, "ohlc_violation_samples": [first 10 open_ms]}``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise StoreError(f"expected a pandas DataFrame, got {type(frame).__name__}")
    columns = store_id.columns
    if tuple(frame.columns) != columns:
        raise StoreError(
            f"{store_id.relpath}: columns {list(frame.columns)} != expected {list(columns)}"
        )
    open_ms = frame["open_ms"]
    if open_ms.dtype != np.int64:
        raise StoreError(f"{store_id.relpath}: open_ms must be int64, got {open_ms.dtype}")
    values = open_ms.to_numpy()
    step = store_id.step_ms
    if len(values):
        negative = np.flatnonzero(values < 0)
        if negative.size:
            raise StoreError(f"{store_id.relpath}: negative open_ms {int(values[negative[0]])}")
        misaligned = np.flatnonzero(values % step)
        if misaligned.size:
            raise StoreError(
                f"{store_id.relpath}: open_ms {int(values[misaligned[0]])} is not a multiple of "
                f"the {store_id.interval} step ({step} ms)"
            )
        not_increasing = np.flatnonzero(np.diff(values) <= 0)
        if not_increasing.size:
            index = int(not_increasing[0])
            kind = "duplicate" if values[index + 1] == values[index] else "decreasing"
            raise StoreError(
                f"{store_id.relpath}: open_ms must be strictly increasing ({kind} at "
                f"{int(values[index + 1])})"
            )
    for column in columns[1:]:
        series = frame[column]
        if series.empty:
            continue
        if pd.api.types.infer_dtype(series, skipna=False) != "string":
            raise StoreError(f"{store_id.relpath}: column {column} must hold Python str values")
        pattern = _COUNT if column == "count" else _DECIMAL
        matches = series.str.fullmatch(pattern).to_numpy(dtype=bool)
        if not matches.all():
            first = int(np.flatnonzero(~matches)[0])
            kind = "an unsigned integer" if column == "count" else "a plain decimal"
            raise StoreError(
                f"{store_id.relpath}: {column} at open_ms {int(values[first])} is not {kind} "
                f"(pattern {pattern.pattern})"
            )
    bad = _ohlc_violation_open_ms(frame)
    return {"ohlc_violations": len(bad), "ohlc_violation_samples": bad[:MAX_VIOLATION_SAMPLES]}


def find_gaps(open_ms: np.ndarray, step_ms: int) -> list[list[int]]:
    """Missing candles between the first and the last open: ``[first_missing, end)`` ranges.

    ``open_ms`` must be strictly increasing and spaced by multiples of ``step_ms``
    (``ValueError`` otherwise).
    """
    if isinstance(step_ms, bool) or not isinstance(step_ms, int) or step_ms < 1:
        raise ValueError(f"step_ms must be a positive integer, got {step_ms!r}")
    values = np.asarray(open_ms, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("open_ms must be one-dimensional")
    if values.size < 2:
        return []
    diffs = np.diff(values)
    if (diffs <= 0).any():
        raise ValueError("open_ms must be strictly increasing")
    if (diffs % step_ms).any():
        raise ValueError(f"open_ms must be spaced by multiples of {step_ms} ms")
    return [
        [int(values[index]) + step_ms, int(values[index + 1])]
        for index in np.flatnonzero(diffs > step_ms)
    ]


def _encode_rows(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    if frame.empty:
        return b""
    parts = [frame["open_ms"].to_numpy(dtype=np.int64).astype(str)]
    parts += [frame[column].to_numpy(dtype=object) for column in columns[1:]]
    lines = [",".join(row) for row in zip(*parts, strict=True)]
    return ("\n".join(lines) + "\n").encode("ascii")


def _header_bytes(store_id: StoreId) -> bytes:
    return (",".join(store_id.columns) + "\n").encode("ascii")


# -- writing --------------------------------------------------------------------------------------


class StoreWriter:
    """Streaming writer: append frames in ascending order, then :meth:`finish` publishes.

    Rows go to a temporary file next to the store, hashed as they are written; nothing at the
    store's path changes before :meth:`finish`, which publishes the CSV (``os.replace``) and then
    its meta. :meth:`abort` (or leaving a ``with`` block on an exception) removes the temporary
    file. The whole store is never held in memory as strings.
    """

    def __init__(self, store_id: StoreId, *, root: Path | None = None) -> None:
        if not isinstance(store_id, StoreId):
            raise TypeError(f"store_id must be a StoreId, got {type(store_id).__name__}")
        self.store_id = store_id
        self._root = _root(root)
        self._csv_path = store_id.csv_path(self._root)
        self._meta_path = store_id.meta_path(self._root)
        ensure_directory(self._csv_path.parent)
        fd, temp_name = tempfile.mkstemp(
            dir=self._csv_path.parent, prefix=f".{self._csv_path.name}.", suffix=".tmp"
        )
        self._temp = Path(temp_name)
        self._handle: BinaryIO | None = os.fdopen(fd, "wb")
        self._digest = hashlib.sha256()
        self._size = 0
        self._rows = 0
        self._first_open_ms: int | None = None
        self._last_open_ms: int | None = None
        self._gaps: list[list[int]] = []
        self._violations = 0
        self._violation_samples: list[int] = []
        self._state = "open"
        self._write(_header_bytes(store_id))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._state == "open":
            self.abort()

    @property
    def rows(self) -> int:
        return self._rows

    def _write(self, data: bytes) -> None:
        assert self._handle is not None
        self._handle.write(data)
        self._digest.update(data)
        self._size += len(data)

    def _require_open(self) -> None:
        if self._state != "open":
            raise StoreError(f"{self.store_id.relpath}: writer already {self._state}")

    def append(self, frame: pd.DataFrame) -> None:
        """Validate ``frame`` and append its rows; it must start after the last appended row.

        A gap at the seam with the previous frame is recorded like any other gap.
        """
        self._require_open()
        info = validate_frame(frame, self.store_id)
        if frame.empty:
            return
        open_ms = frame["open_ms"].to_numpy(dtype=np.int64)
        first = int(open_ms[0])
        step = self.store_id.step_ms
        if self._last_open_ms is None:
            self._first_open_ms = first
        else:
            if first <= self._last_open_ms:
                raise StoreError(
                    f"{self.store_id.relpath}: frames must be appended in ascending order "
                    f"(frame starts at {utc_text(first)}, last row {utc_text(self._last_open_ms)})"
                )
            if first - self._last_open_ms > step:
                self._gaps.append([self._last_open_ms + step, first])
        self._gaps.extend(find_gaps(open_ms, step))
        self._write(_encode_rows(frame, self.store_id.columns))
        self._rows += len(frame)
        self._last_open_ms = int(open_ms[-1])
        self._violations += info["ohlc_violations"]
        room = MAX_VIOLATION_SAMPLES - len(self._violation_samples)
        if room > 0:
            self._violation_samples.extend(info["ohlc_violation_samples"][:room])

    def _meta(self, provenance: dict[str, Any]) -> dict[str, Any]:
        store_id = self.store_id
        step = store_id.step_ms
        first, last = self._first_open_ms, self._last_open_ms
        return {
            "schema_version": SCHEMA_VERSION,
            "source": store_id.source,
            "symbol": store_id.symbol,
            "interval": store_id.interval,
            "step_ms": step,
            "columns": list(store_id.columns),
            "rows": self._rows,
            "first_open_ms": first,
            "last_open_ms": last,
            "first_open_utc": None if first is None else utc_text(first),
            "last_open_utc": None if last is None else utc_text(last),
            "gaps": [list(gap) for gap in self._gaps],
            "missing_candles": sum((end - start) // step for start, end in self._gaps),
            "ohlc_violations": self._violations,
            "ohlc_violation_samples": list(self._violation_samples),
            "csv_relpath": store_id.relpath,
            "csv_sha256": self._digest.hexdigest(),
            "csv_size": self._size,
            "provenance": provenance,
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "code": code_version(),
        }

    def finish(self, *, provenance: Mapping[str, Any]) -> dict[str, Any]:
        """Publish the CSV, then its meta; return the meta.

        ``provenance`` (archives with their checksums, or chunks with their sha256 and
        completeness) must be JSON-serialisable.
        """
        self._require_open()
        if not isinstance(provenance, Mapping):
            raise TypeError(f"provenance must be a mapping, got {type(provenance).__name__}")
        try:
            handle = self._handle
            assert handle is not None
            handle.flush()
            os.fchmod(handle.fileno(), FILE_MODE)
            os.fsync(handle.fileno())
            handle.close()
            self._handle = None
            meta = self._meta(dict(provenance))
            try:
                text = json.dumps(
                    meta, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=True
                )
            except (TypeError, ValueError) as exc:
                raise StoreError(
                    f"{self.store_id.relpath}: provenance is not JSON-serialisable ({exc})"
                ) from exc
            os.replace(self._temp, self._csv_path)
        except BaseException:
            self.abort()
            raise
        fsync_directory(self._csv_path.parent)
        atomic_write_bytes(self._meta_path, (text + "\n").encode("ascii"))
        self._state = "finished"
        return meta

    def abort(self) -> None:
        """Discard the temporary file; the store at its path is left as it was."""
        if self._state != "open":
            return
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._temp.unlink(missing_ok=True)
        self._state = "aborted"


def write_store(
    store_id: StoreId,
    frame: pd.DataFrame,
    *,
    provenance: Mapping[str, Any],
    root: Path | None = None,
) -> dict[str, Any]:
    """Write a whole store from one string frame; return its meta."""
    with StoreWriter(store_id, root=root) as writer:
        writer.append(frame)
        return writer.finish(provenance=provenance)


# -- reading --------------------------------------------------------------------------------------


def read_meta(store_id: StoreId, root: Path | None = None) -> dict[str, Any]:
    """The store's meta; :class:`StoreMissingError` when the store has not been built."""
    path = store_id.meta_path(root)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise StoreMissingError(f"store {store_id.relpath} has not been built (no meta)") from None
    try:
        meta = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StoreIntegrityError(f"{store_id.meta_relpath} is not valid JSON") from exc
    expected = {
        "schema_version": SCHEMA_VERSION,
        "source": store_id.source,
        "interval": store_id.interval,
        "columns": list(store_id.columns),
    }
    if not isinstance(meta, dict) or any(meta.get(key) != value for key, value in expected.items()):
        raise StoreIntegrityError(f"{store_id.meta_relpath} does not describe this store")
    for key in ("rows", "csv_size"):
        if isinstance(meta.get(key), bool) or not isinstance(meta.get(key), int):
            raise StoreIntegrityError(f"{store_id.meta_relpath}: {key} is not an integer")
    if not isinstance(meta.get("csv_sha256"), str) or len(meta["csv_sha256"]) != 64:
        raise StoreIntegrityError(f"{store_id.meta_relpath}: csv_sha256 is not a sha256 hex digest")
    return meta


def _selected_columns(store_id: StoreId, usecols: Iterable[str] | None) -> tuple[str, ...]:
    if usecols is None:
        return store_id.columns
    if isinstance(usecols, str):
        raise TypeError("usecols must be a sequence of column names, not a string")
    requested = set(usecols)
    unknown = sorted(requested - set(store_id.columns))
    if unknown:
        raise ValueError(f"unknown columns {unknown} for {store_id.relpath}")
    return tuple(c for c in store_id.columns if c == "open_ms" or c in requested)


def _optional_ms(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, float)) or not isinstance(value, (int, np.integer)):
        raise ValueError(  # noqa: TRY004 - a bad bound is a ValueError, as in partition
            f"{name} must be an integer of epoch milliseconds or None, got {value!r}"
        )
    return int(value)


def _line_open_ms(line: bytes, relpath: str) -> int:
    head, separator, _ = line.partition(b",")
    if not separator or not head.isdigit():
        raise StoreIntegrityError(f"{relpath}: malformed row while seeking a range")
    return int(head)


def _first_row_at_or_after(handle: BinaryIO, lo: int, hi: int, target: int, relpath: str) -> int:
    """Byte offset of the first row whose ``open_ms >= target`` in ``[lo, hi)`` (``hi`` if none).

    Binary search over byte offsets of a file whose rows are sorted by ``open_ms``. Invariants:
    ``lo`` is the start of a row and every row before it opens before ``target``; ``hi`` is a
    row start (or the end of the file) and the row there, if any, opens at or after ``target``.
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if mid <= lo:
            position = lo
        else:
            handle.seek(mid - 1)
            handle.readline()
            position = handle.tell()  # first row start at or after mid
        if position >= hi:
            handle.seek(lo)
            line = handle.readline()
            if _line_open_ms(line, relpath) >= target:
                return lo
            lo += len(line)
            continue
        handle.seek(position)
        line = handle.readline()
        if _line_open_ms(line, relpath) < target:
            lo = position + len(line)
        else:
            hi = position
    return lo


def _parse_csv(
    source: BinaryIO, store_id: StoreId, columns: tuple[str, ...], numeric: bool
) -> pd.DataFrame:
    try:
        if numeric:
            dtype = {c: ("int64" if c in _INTEGER_COLUMNS else "float64") for c in columns}
            frame = pd.read_csv(
                source,
                usecols=list(columns),
                dtype=dtype,
                float_precision="round_trip",
                keep_default_na=False,
                na_filter=False,
                engine="c",
            )
        else:
            frame = pd.read_csv(
                source,
                usecols=list(columns),
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                engine="c",
            )
            frame["open_ms"] = frame["open_ms"].astype(np.int64)
    except (ValueError, pd.errors.ParserError) as exc:
        raise StoreIntegrityError(f"{store_id.relpath}: cannot parse rows ({exc})") from exc
    return frame[list(columns)].reset_index(drop=True)


@contextlib.contextmanager
def store_read_lock(store_id: StoreId, root: Path | None = None) -> Iterator[None]:
    """Hold the job lock of the store's source shared (:func:`~kronos_agent.data.fsutil.shared_lock`).

    Every :func:`read_store` holds it for one read; a caller holds it around several reads and the
    meta it records, so that all of them describe one version of the store. The runs that rebuild
    stores hold it exclusively: while one runs, this raises :class:`StoreBusyError`, and while
    this is held, they are refused.
    """
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(shared_lock(store_id.lock_path(root)))
        except LockBusyError:
            raise StoreBusyError(
                f"{store_id.relpath}: a run that writes the {store_id.source} data holds "
                f"{store_id.lock_relpath}; read the store when that run is over"
            ) from None
        yield


def _sha256_of(handle: BinaryIO) -> str:
    """sha256 of the whole open file, streamed (the 1m store is never held in memory)."""
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
        digest.update(chunk)
    return digest.hexdigest()


def read_store(
    store_id: StoreId,
    *,
    numeric: bool = False,
    start_ms: int | None = None,
    end_ms: int | None = None,
    usecols: Iterable[str] | None = None,
    root: Path | None = None,
    verify: bool = True,
) -> pd.DataFrame:
    """INTEGRITY READS ONLY (checksums, gaps, merges, the resample check, status).

    Studies and window sources must use :func:`read_study_frame`. Returns the rows with
    ``open_ms`` in ``[start_ms, end_ms)`` (``None``: unbounded). ``verify`` checks the CSV's size
    and sha256 against the meta first. ``numeric=False`` keeps the stored strings (Python
    ``str`` values, ``open_ms`` int64); ``numeric=True`` gives float64 values parsed with
    ``float_precision="round_trip"`` and int64 ``open_ms``/``count``. ``usecols`` selects
    columns (``open_ms`` is always included, store order kept).

    The read holds :func:`store_read_lock` (a run that rebuilds the store's source makes it raise
    :class:`StoreBusyError`) and opens the CSV once: its size and sha256 are checked on the open
    file, which is then searched and parsed, so the rows returned come from the bytes verified
    even if the file were replaced meanwhile.
    """
    columns = _selected_columns(store_id, usecols)
    start = _optional_ms(start_ms, "start_ms")
    end = _optional_ms(end_ms, "end_ms")
    if start is not None and end is not None and start > end:
        raise ValueError(f"start_ms {start} must be <= end_ms {end} (half-open range)")
    with store_read_lock(store_id, root):
        meta = read_meta(store_id, root)
        path = store_id.csv_path(root)
        if not path.is_file():
            raise StoreIntegrityError(f"{store_id.relpath} is missing although its meta exists")
        with open(path, "rb") as handle:
            if verify and (
                os.fstat(handle.fileno()).st_size != meta["csv_size"]
                or _sha256_of(handle) != meta["csv_sha256"]
            ):
                raise StoreIntegrityError(
                    f"{store_id.relpath}: size or sha256 differs from its meta (modified or "
                    "partially written); rebuild the store"
                )
            return _read_rows(handle, store_id, columns, numeric=numeric, start=start, end=end)


def _read_rows(
    handle: BinaryIO,
    store_id: StoreId,
    columns: tuple[str, ...],
    *,
    numeric: bool,
    start: int | None,
    end: int | None,
) -> pd.DataFrame:
    """The rows of ``[start, end)`` of the open store CSV: a binary search, then one parse."""
    size = os.fstat(handle.fileno()).st_size
    header = _header_bytes(store_id)
    handle.seek(0)
    if handle.readline() != header:
        raise StoreIntegrityError(f"{store_id.relpath}: unexpected header line")
    data_start = handle.tell()
    lo = data_start
    if start is not None:
        lo = _first_row_at_or_after(handle, data_start, size, start, store_id.relpath)
    hi = size
    if end is not None:
        hi = _first_row_at_or_after(handle, lo, size, end, store_id.relpath)
    if lo == data_start and hi == size:
        handle.seek(0)
        source: BinaryIO = handle
    else:
        handle.seek(lo)
        source = io.BytesIO(header + handle.read(hi - lo))
    frame = _parse_csv(source, store_id, columns, numeric)
    opens = frame["open_ms"].to_numpy(dtype=np.int64)
    if opens.size and (
        (np.diff(opens) <= 0).any()
        or (start is not None and opens[0] < start)
        or (end is not None and opens[-1] >= end)
    ):
        raise StoreIntegrityError(f"{store_id.relpath}: rows are not sorted and unique by open_ms")
    return frame


def read_study_frame(
    store_id: StoreId,
    *,
    start_ms: int,
    end_ms: int,
    numeric: bool = True,
    usecols: Iterable[str] | None = None,
    sealed_access: SealedAccess | None = None,
    root: Path | None = None,
) -> pd.DataFrame:
    """The study read path: :func:`read_store` of ``[start_ms, end_ms)`` behind the sealed guard.

    ``end_ms > SEALED_START_MS`` without a ``SealedAccess`` raises ``SealedPartitionError``
    before anything is read; ``end_ms == SEALED_START_MS`` returns nothing at or after it. A
    ``SealedAccess`` issued for another data root than ``root`` is refused too
    (``SealedAccess.require_root``): an order verified for one data root never opens another's.
    """
    require_study_range(start_ms, end_ms, sealed_access)
    if isinstance(sealed_access, SealedAccess):
        sealed_access.require_root(root)
    return read_store(
        store_id,
        numeric=numeric,
        start_ms=start_ms,
        end_ms=end_ms,
        usecols=usecols,
        root=root,
        verify=True,
    )


def float_values(strings: Iterable[str]) -> np.ndarray:
    """``float()`` of each decimal string as a float64 array (the model-input conversion)."""
    return np.array([float(value) for value in strings], dtype=np.float64)


def _status_row(
    store_id: StoreId, root: Path | None, *, verify_sha256: bool, busy: bool
) -> dict[str, Any] | None:
    """The :func:`store_status` row of one store (``None``: not built)."""
    base: dict[str, Any] = {
        "source": store_id.source,
        "interval": store_id.interval,
        "relpath": store_id.relpath,
    }
    try:
        meta = read_meta(store_id, root)
    except StoreMissingError:
        return None
    except StoreIntegrityError as exc:
        return {**base, "error": str(exc)}
    row: dict[str, Any] = {
        **base,
        "rows": meta["rows"],
        "first_open_ms": meta.get("first_open_ms"),
        "last_open_ms": meta.get("last_open_ms"),
        "first": meta.get("first_open_utc"),
        "last": meta.get("last_open_utc"),
        "gaps": meta.get("gaps", []),
        "gap_count": len(meta.get("gaps", [])),
        "missing_candles": meta.get("missing_candles"),
        "ohlc_violations": meta.get("ohlc_violations"),
        "csv_size": meta["csv_size"],
        "busy": busy,
        "csv_present": None,
        "size_matches": None,
        "sha256_matches": None,
        "created_at": meta.get("created_at"),
    }
    if busy:
        return row
    csv_path = store_id.csv_path(root)
    if not csv_path.is_file():
        return {**row, "csv_present": False, "size_matches": False}
    with open(csv_path, "rb") as handle:
        row["csv_present"] = True
        row["size_matches"] = os.fstat(handle.fileno()).st_size == meta["csv_size"]
        if verify_sha256 and row["size_matches"]:
            row["sha256_matches"] = _sha256_of(handle) == meta["csv_sha256"]
    return row


def store_status(root: Path | None = None, *, verify_sha256: bool = True) -> list[dict[str, Any]]:
    """One row per built store: range, gaps, missing candles, violations and its CSV checked.

    Each store is checked under :func:`store_read_lock`: ``csv_present``, ``size_matches`` (the
    CSV's size against the meta) and, with ``verify_sha256``, ``sha256_matches`` (its sha256,
    streamed; ``None`` when not checked because the CSV is missing or of another size). Hashing
    reads every byte of every store, about 0.5 GB with the Binance 1m store (well under a second
    on the Mac with the pages cached). A store whose source's job lock a run holds is listed from
    its meta with ``busy`` true and its CSV not checked (``None``): the run may be replacing it.
    A store whose meta cannot be read is listed with an ``error``.
    """
    rows: list[dict[str, Any]] = []
    for source in SOURCE_SYMBOLS:
        for interval in INTERVALS:
            store_id = StoreId(source, interval)
            try:
                with store_read_lock(store_id, root):
                    row = _status_row(store_id, root, verify_sha256=verify_sha256, busy=False)
            except LockBusyError:
                row = _status_row(store_id, root, verify_sha256=False, busy=True)
            if row is not None:
                rows.append(row)
    return rows


def leftover_temporaries(root: Path | None = None) -> list[dict[str, Any]]:
    """Temporary files of killed writes in the directories of every job lock.

    One row per file under :data:`LOCKED_RELDIRS` (stores, raw archives, plans and records):
    ``relpath``, ``bytes``, ``lock`` (the job lock its writers hold) and ``running``, true when a
    run holds that lock now, so that the file may be the live temporary of that run rather than
    a leftover. Nothing is removed here: the next run that takes the lock does (:func:`job_lock`).
    """
    base = _root(root)
    rows: list[dict[str, Any]] = []
    for lock, reldirs in LOCKED_RELDIRS.items():
        running = lock_is_held(base / lock)
        for reldir in reldirs:
            for path in temporary_files(base / reldir):
                try:
                    size = path.stat().st_size
                except FileNotFoundError:  # a live writer just published or discarded it
                    continue
                relpath = path.relative_to(base).as_posix()
                rows.append({"relpath": relpath, "bytes": size, "lock": lock, "running": running})
    return rows


@contextlib.contextmanager
def job_lock(lock_relpath: str, root: Path | None = None) -> Iterator[list[dict[str, Any]]]:
    """Hold a job lock exclusively and first remove the temporary files that killed runs left.

    ``lock_relpath`` is one of :data:`LOCKED_RELDIRS` (``raw/binance/.lock``,
    ``raw/bingx/.lock``); a run that holds it already raises
    :class:`~kronos_agent.data.fsutil.LockBusyError` and nothing is removed. Every writer of the
    lock's directories holds that lock while it writes, and this run has written nothing yet, so
    no temporary file found there belongs to a live write: they are removed before the block
    runs (only names ``tempfile.mkstemp`` gives them; lock files, ``.DS_Store``, archived files
    and symlinks are never touched). Yields ``{"relpath", "bytes"}`` of each file removed.
    """
    if lock_relpath not in LOCKED_RELDIRS:
        raise ValueError(
            f"unknown job lock {lock_relpath!r}; expected one of {list(LOCKED_RELDIRS)}"
        )
    base = _root(root)
    with exclusive_lock(base / lock_relpath) as held:
        removed = [
            {"relpath": path.relative_to(base).as_posix(), "bytes": size}
            for reldir in LOCKED_RELDIRS[lock_relpath]
            for path, size in remove_temporary_files(base / reldir, held)
        ]
        yield removed


__all__ = [
    "BASE_COLUMNS",
    "BINANCE_COLUMNS",
    "COUNT_RE",
    "DECIMAL_RE",
    "INTERVALS",
    "LOCKED_RELDIRS",
    "LOCK_RELPATHS",
    "SCHEMA_VERSION",
    "SOURCE_SYMBOLS",
    "StoreBusyError",
    "StoreError",
    "StoreId",
    "StoreIntegrityError",
    "StoreMissingError",
    "StoreWriter",
    "find_gaps",
    "float_values",
    "job_lock",
    "leftover_temporaries",
    "read_meta",
    "read_store",
    "read_study_frame",
    "store_read_lock",
    "store_status",
    "validate_frame",
    "write_store",
]
