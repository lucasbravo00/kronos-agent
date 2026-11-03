"""Binance USD(S)-M perpetual BTCUSDT kline archives (data.binance.vision): training data.

SPEC section 5: Binance's public archives are the training and discovery source; download 1m,
resample locally to 5m, 15m and 1h, verify the checksums and check the resampling against
Binance's own 1h archives. Everything here follows ``docs/DATA.md`` ("Sources", "Resampling and
its check"); the facts below were observed on 2026-11-01.

- **Client.** :class:`BinanceVisionClient` sends GET requests only, to ``https://`` +
  :data:`HOST` on the default port (compared on ``url.raw_host``), for the paths
  ``/data/futures/um/{monthly,daily}/klines/BTCUSDT/<interval>/<archive name>[.CHECKSUM]`` of the
  four intervals and nothing else. A private transport wrapper checks every request before the
  wire, redirects are refused, the environment (proxies, certificates, netrc) is ignored and no
  credential is ever sent. Requests are paced (>= 0.25 s apart) and retried on timeouts and 5xx
  after 1, 5 and 15 s; 404 means "not published"; any other status is an error.
- **Plan.** For an interval, every month from :data:`FIRST_MONTH` (the perpetual's launch) to the
  month before the current one uses its monthly archive when published, otherwise each published
  daily archive of that month; the current month uses its daily archives up to yesterday (UTC).
  Publication is probed with the ``.CHECKSUM`` files (one small GET each), whose sha256 the
  download is then verified against. The monthly archives of 2019 return 404, so the plan probes
  the 2019 dailies; the monthly archive of the month just ended is published some days after its
  dailies. The plan is saved to ``raw/binance/plan-<interval>.json``.
- **Download.** Streamed to a temporary file, capped at :data:`ARCHIVE_MAX_BYTES`, verified
  against ``.CHECKSUM`` and only then published (a mismatch deletes the download). Gate 4: every
  byte already under ``raw/binance`` counts against :data:`BUDGET_BYTES`; a run plans every
  interval it syncs first (:func:`sync_intervals`) and is refused before anything is downloaded
  when the bytes on disk plus the estimate of all its intervals would pass the budget; every byte
  received is charged to one budget for the whole run. A remote ``.CHECKSUM`` that differs from a
  locally verified archive (a republished archive) is reported and the local copy kept, unless
  its period is refreshed explicitly.
- **Parse.** Archives are untrusted data: the single expected member is read in memory (capped at
  :data:`CSV_MAX_BYTES`, never extracted to disk) and parsed with the ``csv`` module: exactly 12
  fields per line, no blank lines, a header line only when the first field of the first line is
  not all digits and then exactly :data:`KLINE_COLUMNS` (2020 files have none, recent ones do),
  microsecond times (``open_time >= 1e15``) converted, ``close_time == open_time + interval - 1``,
  rows aligned, strictly increasing and inside the archive's period. Price strings change format
  across periods (``7225`` and ``7171.10`` in one 2020 file) and are kept verbatim. A malformed
  archive fails naming the archive and the line.
- **Stores.** ``binance/1m`` is built archive by archive with a streaming
  :class:`~kronos_agent.data.store.StoreWriter` (about 3.6 million rows, never held whole) and
  resampled to ``binance/{5m,15m,1h}`` on the way (:mod:`kronos_agent.data.resample`);
  ``binance_official/1h`` comes from Binance's own 1h archives and serves only the check. When a
  monthly archive replaces daily archives already on disk, their rows are compared as decimals and
  every difference is recorded in the store's provenance.
- **Check.** :func:`check_resample` compares every hour present in both ``binance/1h`` and
  ``binance_official/1h`` and returns the payload of ``reports/data/binance_resample_check``;
  values of hours at or after the sealed start are never written into it.

The sync and the builds hold the non-blocking lock ``raw/binance/.lock``, so a second concurrent
run is refused (:class:`~kronos_agent.data.fsutil.LockBusyError`); none of them may be called while
the caller itself holds that lock. The sync takes it through
:func:`~kronos_agent.data.store.job_lock`, which first removes the temporary files of killed runs.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import time
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Self

import httpx
import numpy as np
import pandas as pd

from ..exchange.public_market import interval_ms
from ..hashing import sha256_bytes, sha256_file
from ..paths import data_dir
from ..report.writer import record_config, report_header
from .fsutil import FILE_MODE, atomic_write_text, ensure_directory, exclusive_lock, fsync_directory
from .partition import PARTITION, SEALED_START_MS
from .resample import (
    EXACT_COLUMNS,
    ROUNDING_COLUMNS,
    TARGET_INTERVALS,
    TOLERANCE_COLUMNS,
    compare_with_official,
    quote_rounding,
    resample_many,
)
from .store import (
    BINANCE_COLUMNS,
    COUNT_RE,
    DECIMAL_RE,
    StoreId,
    StoreIntegrityError,
    StoreWriter,
    job_lock,
    read_meta,
    read_store,
)
from .timeutil import HOUR_MS, utc_text

HOST = "data.binance.vision"
SYMBOL = "BTCUSDT"
INTERVALS: tuple[str, ...] = ("1m", "5m", "15m", "1h")
KINDS: tuple[str, ...] = ("monthly", "daily")
FIRST_MONTH = "2019-09"
ARCHIVE_MAX_BYTES = 64 * 2**20
CSV_MAX_BYTES = 256 * 2**20
CHECKSUM_MAX_BYTES = 1024
BUDGET_BYTES = 4_500_000_000
MIN_INTERVAL_S = 0.25
RETRY_BACKOFF_S: tuple[float, ...] = (1.0, 5.0, 15.0)
# Deliberately high (docs/DATA.md expects about 0.15 GB of zips in all, some 40 bytes per row):
# the estimate only guards gate 4.
ESTIMATED_BYTES_PER_ROW = 80
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
MICROSECOND_THRESHOLD = 10**15
RAW_RELDIR = "raw/binance"
LOCK_RELPATH = f"{RAW_RELDIR}/.lock"
PLAN_SCHEMA_VERSION = 1
USER_AGENT = "kronos-agent/0.1 (public archive download)"
RESAMPLE_CHECK_STEM = "binance_resample_check"
LOST_HOURS_WINDOWS: tuple[int, ...] = (512,)  # the Kronos context
MAX_SAMPLES = 20
MAX_MARKDOWN_ROWS = 100

# store column -> field index in the kline CSV (close_time and ignore are dropped)
_STORE_FIELDS: tuple[tuple[str, int], ...] = (
    ("open", 1),
    ("high", 2),
    ("low", 3),
    ("close", 4),
    ("volume", 5),
    ("quote_volume", 7),
    ("count", 8),
    ("taker_buy_volume", 9),
    ("taker_buy_quote_volume", 10),
)
_ALL_DIGITS = re.compile(r"[0-9]+")
_TIME = re.compile(r"[0-9]{1,18}")
_DECIMAL = re.compile(DECIMAL_RE)
_COUNT = re.compile(COUNT_RE)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  (\S+)\n?")
_MONTH = re.compile(r"([0-9]{4})-([0-9]{2})")
_DAY = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})")
_PERIOD = re.compile(r"[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?")
_ALLOWED_PATH = re.compile(
    r"/data/futures/um/(monthly|daily)/klines/BTCUSDT/(1m|5m|15m|1h)/"
    r"BTCUSDT-(1m|5m|15m|1h)-([0-9]{4}-[0-9]{2}(?:-[0-9]{2})?)\.zip(?:\.CHECKSUM)?"
)
_TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


class BinanceVisionError(Exception):
    """Base class of every error raised by this module."""


class HostNotAllowedError(BinanceVisionError):
    """A request outside the pin: another origin, a path outside the allow-list, not a GET."""


class ChecksumMismatchError(BinanceVisionError):
    """An archive's sha256 differs from its ``.CHECKSUM``."""


class ArchiveFormatError(BinanceVisionError):
    """A malformed archive or ``.CHECKSUM`` file (named with the line where it fails)."""


class DownloadBudgetExceeded(BinanceVisionError):
    """Gate 4: the bytes under ``raw/binance`` plus this run's would pass the budget."""


class BinanceVisionUnavailable(BinanceVisionError):
    """Timeouts or 5xx persisted after every retry."""


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else data_dir()


def _require_interval(interval: str) -> str:
    if interval not in INTERVALS:
        raise ValueError(f"unknown interval {interval!r}; expected one of {list(INTERVALS)}")
    return interval


def _period_bounds(kind: str, period: str) -> tuple[datetime, datetime]:
    """``[start, end)`` of a monthly ``YYYY-MM`` or daily ``YYYY-MM-DD`` period, in UTC."""
    if kind not in KINDS:
        raise ValueError(f"unknown archive kind {kind!r}; expected one of {list(KINDS)}")
    pattern = _MONTH if kind == "monthly" else _DAY
    match = pattern.fullmatch(period) if isinstance(period, str) else None
    if match is None:
        form = "YYYY-MM" if kind == "monthly" else "YYYY-MM-DD"
        raise ValueError(f"a {kind} period is {form}, got {period!r}")
    parts = [int(group) for group in match.groups()]
    try:
        if kind == "monthly":
            year, month = parts
            start = datetime(year, month, 1, tzinfo=UTC)
            end = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=UTC)
        else:
            start = datetime(*parts, tzinfo=UTC)
            end = start + timedelta(days=1)
    except ValueError as exc:
        raise ValueError(f"invalid {kind} period {period!r}: {exc}") from None
    return start, end


def _to_ms(moment: datetime) -> int:
    return (moment - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)


@dataclass(frozen=True, order=True)
class Archive:
    """One archive: ``kind`` (``monthly`` | ``daily``), ``interval`` and ``period``.

    The period is ``YYYY-MM`` for a monthly archive and ``YYYY-MM-DD`` for a daily one; it covers
    the UTC open times ``[start_ms, end_ms)``. The derived names, URL paths and local paths are
    properties, so an ``Archive`` is fully described by its three fields.
    """

    kind: str
    interval: str
    period: str

    def __post_init__(self) -> None:
        _require_interval(self.interval)
        _period_bounds(self.kind, self.period)

    @property
    def name(self) -> str:
        """``BTCUSDT-<interval>-<period>.zip``."""
        return f"{SYMBOL}-{self.interval}-{self.period}.zip"

    @property
    def checksum_name(self) -> str:
        return f"{self.name}.CHECKSUM"

    @property
    def member_name(self) -> str:
        """The single CSV member expected inside the zip."""
        return f"{SYMBOL}-{self.interval}-{self.period}.csv"

    @property
    def url_path(self) -> str:
        return f"/data/futures/um/{self.kind}/klines/{SYMBOL}/{self.interval}/{self.name}"

    @property
    def checksum_url_path(self) -> str:
        return f"{self.url_path}.CHECKSUM"

    @property
    def relpath(self) -> str:
        """``raw/binance/<kind>/<interval>/<name>``, relative to the data root."""
        return f"{RAW_RELDIR}/{self.kind}/{self.interval}/{self.name}"

    @property
    def checksum_relpath(self) -> str:
        return f"{self.relpath}.CHECKSUM"

    @property
    def start_ms(self) -> int:
        return _to_ms(_period_bounds(self.kind, self.period)[0])

    @property
    def end_ms(self) -> int:
        return _to_ms(_period_bounds(self.kind, self.period)[1])

    @property
    def step_ms(self) -> int:
        return interval_ms(self.interval)

    @property
    def expected_rows(self) -> int:
        """Candles in a complete period (informational: archives may hold fewer)."""
        return (self.end_ms - self.start_ms) // self.step_ms

    def local_path(self, root: Path | None = None) -> Path:
        return _root(root) / self.relpath

    def checksum_path(self, root: Path | None = None) -> Path:
        return _root(root) / self.checksum_relpath


def _require_archive(archive: object) -> Archive:
    if not isinstance(archive, Archive):
        raise TypeError(f"expected an Archive, got {type(archive).__name__}")
    return archive


def _estimated_bytes(archive: Archive) -> int:
    return archive.expected_rows * ESTIMATED_BYTES_PER_ROW


# -- the pinned client -----------------------------------------------------------------------------


def _require_allowed_path(path: str) -> None:
    """Refuse a path outside the allow-list (the interval of the folder and of the name agree,
    the period has the form of its kind and is a real date)."""
    match = _ALLOWED_PATH.fullmatch(path) if isinstance(path, str) else None
    allowed = match is not None and match[2] == match[3]
    if allowed and match is not None:
        try:
            Archive(match[1], match[2], match[4])
        except ValueError:
            allowed = False
    if not allowed:
        raise HostNotAllowedError(
            f"path {str(path)[:120]!r} is not in the allow-list "
            "/data/futures/um/{monthly,daily}/klines/BTCUSDT/<interval>/"
            "BTCUSDT-<interval>-<period>.zip[.CHECKSUM]"
        )


def _check_url(url: httpx.URL) -> None:
    """Refuse anything but HTTPS to :data:`HOST` (default port, no userinfo) on an allowed path.

    The host is compared in its canonical ASCII form (``raw_host``) and the path as sent on the
    wire (``raw_path``, which also carries any query string), so neither IDNA decoding nor
    percent-encoding can smuggle another target past the check.
    """
    host = url.raw_host.decode("ascii")
    if url.scheme != "https" or host != HOST or url.port is not None:
        raise HostNotAllowedError(
            f"refusing request to {url.scheme!r}://{host!r} (port {url.port}); the Binance "
            f"archive client only talks to https://{HOST}"
        )
    if url.userinfo != b"":
        raise HostNotAllowedError("refusing a request URL with userinfo")
    _require_allowed_path(url.raw_path.decode("ascii", errors="replace"))


class _PinnedTransport(httpx.BaseTransport):
    """Last gate before the wire: GET only, pinned origin, allow-listed path."""

    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET":
            raise HostNotAllowedError(f"refusing {request.method}: the archive client only GETs")
        _check_url(request.url)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


def _content_length(response: httpx.Response) -> int | None:
    value = response.headers.get("content-length")
    return int(value) if value is not None and value.isdigit() else None


def _read_capped(response: httpx.Response, cap: int, what: str) -> bytes:
    declared = _content_length(response)
    if declared is not None and declared > cap:
        raise BinanceVisionError(f"{what}: {declared} bytes announced, cap {cap}")
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > cap:
            raise BinanceVisionError(f"{what}: body exceeds the {cap}-byte cap")
        chunks.append(chunk)
    return b"".join(chunks)


def checksum_line(sha256: str, archive: Archive) -> str:
    """``<sha256 hex>  <zip name>\\n``: the ``.CHECKSUM`` format, also written next to downloads."""
    return f"{_require_sha256(sha256)}  {_require_archive(archive).name}\n"


def _require_sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"expected a lowercase sha256 hex digest, got {value!r}")
    return value


class BinanceVisionClient:
    """Paced, retrying GETs of kline archives and their checksums from :data:`HOST`.

    Mirrors :class:`~kronos_agent.exchange.public_market.PublicMarketClient`: ``transport``
    (any ``httpx.BaseTransport``), ``sleep`` and ``clock`` (monotonic seconds) are injectable so
    unit tests run without network and without waiting, and there is deliberately no host, URL or
    headers parameter.
    """

    def __init__(
        self,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        min_interval_s: float = MIN_INTERVAL_S,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive seconds")
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be >= 0")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be an integer >= 0")
        inner = (
            transport if transport is not None else httpx.HTTPTransport(trust_env=False, retries=0)
        )
        self._transport = _PinnedTransport(inner)
        self._http = httpx.Client(
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        )
        self._min_interval_s = float(min_interval_s)
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __repr__(self) -> str:
        return (
            f"BinanceVisionClient(host={HOST!r}, min_interval_s={self._min_interval_s}, "
            f"max_retries={self._max_retries})"
        )

    def _pace(self) -> None:
        """Keep at least ``min_interval_s`` between request starts."""
        now = self._clock()
        if self._last_request_at is not None:
            wait = self._min_interval_s - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_request_at = now

    def _get[T](self, path: str, consume: Callable[[httpx.Response], T]) -> T | None:
        """One paced GET of ``path`` with retries; ``None`` on 404, ``consume(response)`` on 200.

        Timeouts, network errors, truncated bodies and 5xx are retried after
        :data:`RETRY_BACKOFF_S`; ``consume`` runs again from scratch on a retry (it must clean
        up after itself when it raises). Redirects and every other status raise.
        """
        _require_allowed_path(path)
        url = httpx.URL(scheme="https", host=HOST, path=path)
        _check_url(url)
        failures = 0
        while True:
            self._pace()
            self._http.cookies.clear()
            try:
                with self._http.stream("GET", url) as response:
                    status = response.status_code
                    if status == 200:
                        return consume(response)
                    if status == 404:
                        return None
                    if 300 <= status < 400:
                        raise BinanceVisionError(
                            f"{path}: HTTP {status} redirect refused (never followed)"
                        )
                    if status < 500:
                        raise BinanceVisionError(f"{path}: HTTP {status}")
                    problem = f"HTTP {status}"
            except _TRANSIENT_ERRORS as exc:
                problem = type(exc).__name__
            except (UnicodeError, httpx.InvalidURL):
                # httpx builds the never-followed redirect request itself: a Location it cannot
                # parse fails outside the HTTPError tree
                raise BinanceVisionError(
                    f"{path}: redirect with an unparseable Location refused"
                ) from None
            if failures >= self._max_retries:
                raise BinanceVisionUnavailable(f"{path}: {problem} after {failures + 1} attempts")
            self._sleep(RETRY_BACKOFF_S[min(failures, len(RETRY_BACKOFF_S) - 1)])
            failures += 1

    def fetch_checksum(self, archive: Archive) -> str | None:
        """The sha256 in the archive's remote ``.CHECKSUM``; ``None`` when it is not published."""
        _require_archive(archive)
        body = self._get(
            archive.checksum_url_path,
            lambda response: _read_capped(response, CHECKSUM_MAX_BYTES, archive.checksum_name),
        )
        if body is None:
            return None
        try:
            text = body.decode("ascii")
        except UnicodeDecodeError:
            raise ArchiveFormatError(f"{archive.checksum_name}: not ASCII text") from None
        return parse_checksum(text, archive)

    def download(
        self,
        archive: Archive,
        *,
        expected_sha256: str,
        root: Path | None = None,
        budget: DownloadBudget,
    ) -> Path:
        """Download ``archive`` to its local path, verified against ``expected_sha256``.

        The body streams into a temporary file next to the target, capped at
        :data:`ARCHIVE_MAX_BYTES` and charged to ``budget`` as it arrives; only a body whose
        sha256 matches is published (``os.replace``), followed by its local ``.CHECKSUM``. Any
        failure deletes the temporary file: nothing partial is ever left at the archive's path.
        """
        _require_archive(archive)
        expected = _require_sha256(expected_sha256)
        if not isinstance(budget, DownloadBudget):
            raise TypeError(f"budget must be a DownloadBudget, got {type(budget).__name__}")
        base = _root(root)
        target = archive.local_path(base)
        ensure_directory(target.parent)

        def consume(response: httpx.Response) -> Path:
            declared = _content_length(response)
            if declared is not None and declared > ARCHIVE_MAX_BYTES:
                raise BinanceVisionError(
                    f"{archive.name}: {declared} bytes announced, cap {ARCHIVE_MAX_BYTES}"
                )
            fd, temp_name = tempfile.mkstemp(
                dir=target.parent, prefix=f".{archive.name}.", suffix=".part"
            )
            temp = Path(temp_name)
            try:
                digest = hashlib.sha256()
                size = 0
                with os.fdopen(fd, "wb") as handle:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > ARCHIVE_MAX_BYTES:
                            raise BinanceVisionError(
                                f"{archive.name}: body exceeds the {ARCHIVE_MAX_BYTES}-byte cap"
                            )
                        budget.charge(len(chunk))
                        handle.write(chunk)
                        digest.update(chunk)
                    handle.flush()
                    os.fchmod(handle.fileno(), FILE_MODE)
                    os.fsync(handle.fileno())
                actual = digest.hexdigest()
                if actual != expected:
                    raise ChecksumMismatchError(
                        f"{archive.name}: sha256 {actual} differs from its .CHECKSUM {expected}; "
                        "the download was deleted"
                    )
                os.replace(temp, target)
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
            fsync_directory(target.parent)
            atomic_write_text(archive.checksum_path(base), checksum_line(expected, archive))
            return target

        path = self._get(archive.url_path, consume)
        if path is None:
            raise BinanceVisionError(f"{archive.name}: HTTP 404 (not published)")
        return path


# -- gate-4 budget ---------------------------------------------------------------------------------


def _bytes_under(directory: Path) -> int:
    """Total size of the regular files under ``directory`` (0 when it does not exist)."""
    total = 0
    if not directory.is_dir():
        return 0
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            try:
                info = os.lstat(os.path.join(dirpath, name))
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


@dataclass
class DownloadBudget:
    """Gate 4 for the Binance archives: bytes on disk under ``raw/binance`` plus this run's.

    :meth:`charge` refuses (:class:`DownloadBudgetExceeded`) a charge that would pass
    ``limit_bytes``; the downloads charge every byte they receive, failed attempts included.
    """

    limit_bytes: int = BUDGET_BYTES
    used_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("limit_bytes", "used_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

    @classmethod
    def from_disk(cls, root: Path | None = None) -> DownloadBudget:
        """A budget of :data:`BUDGET_BYTES` already charged with every byte under raw/binance."""
        return cls(limit_bytes=BUDGET_BYTES, used_bytes=_bytes_under(_root(root) / RAW_RELDIR))

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.limit_bytes - self.used_bytes)

    def charge(self, n: int) -> None:
        """Count ``n`` more bytes, or refuse when the total would pass the limit."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n!r}")
        if self.used_bytes + n > self.limit_bytes:
            raise DownloadBudgetExceeded(
                f"gate 4: {self.used_bytes} bytes under raw/binance and this run plus {n} more "
                f"would pass the {self.limit_bytes}-byte budget"
            )
        self.used_bytes += n


# -- checksums and local archives ------------------------------------------------------------------


def parse_checksum(text: str, archive: Archive) -> str:
    """The sha256 of a ``.CHECKSUM`` text ``<sha256 hex>  <zip name>`` (one optional newline).

    Strict: lowercase hex, two spaces, the archive's own name, nothing else
    (:class:`ArchiveFormatError` otherwise).
    """
    _require_archive(archive)
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    match = _CHECKSUM_LINE.fullmatch(text)
    if match is None:
        raise ArchiveFormatError(
            f"{archive.checksum_name}: expected '<sha256 hex>  {archive.name}', got {text[:120]!r}"
        )
    if match[2] != archive.name:
        raise ArchiveFormatError(
            f"{archive.checksum_name} names {match[2][:80]!r}, expected {archive.name!r}"
        )
    return match[1]


def _local_checksum(archive: Archive, root: Path) -> str | None:
    """The sha256 recorded in the local ``.CHECKSUM``; ``None`` when missing or malformed."""
    path = archive.checksum_path(root)
    try:
        if path.stat().st_size > CHECKSUM_MAX_BYTES:
            return None
        text = path.read_bytes().decode("ascii")
        return parse_checksum(text, archive)
    except (FileNotFoundError, UnicodeDecodeError, ArchiveFormatError):
        return None


def _local_sha256(archive: Archive, root: Path) -> str | None:
    """The verified sha256 of the local archive, or ``None`` (missing, oversized or mismatched)."""
    expected = _local_checksum(archive, root)
    if expected is None:
        return None
    path = archive.local_path(root)
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > ARCHIVE_MAX_BYTES:
        return None
    return expected if sha256_file(path) == expected else None


def verify_local(archive: Archive, root: Path | None = None) -> bool:
    """True iff the local archive exists and its sha256 equals its local ``.CHECKSUM``."""
    return _local_sha256(_require_archive(archive), _root(root)) is not None


# -- plans -----------------------------------------------------------------------------------------


def plan_relpath(interval: str) -> str:
    """``raw/binance/plan-<interval>.json``, relative to the data root."""
    return f"{RAW_RELDIR}/plan-{_require_interval(interval)}.json"


def _require_date(today: object) -> date:
    if isinstance(today, datetime) or not isinstance(today, date):
        raise TypeError(f"today must be a datetime.date (the UTC date), got {today!r}")
    return today


def _month_start(period: str) -> date:
    start, _ = _period_bounds("monthly", period)
    return start.date()


def _next_month(day: date) -> date:
    return date(day.year + day.month // 12, day.month % 12 + 1, 1)


def _days(start: date, stop: date) -> list[date]:
    """Every day in ``[start, stop)``."""
    return [start + timedelta(days=offset) for offset in range((stop - start).days)]


def _plan_entry(archive: Archive, sha256: str) -> dict[str, Any]:
    return {
        "kind": archive.kind,
        "period": archive.period,
        "name": archive.name,
        "sha256": sha256,
        "estimated_bytes": _estimated_bytes(archive),
    }


def plan_archives(client: BinanceVisionClient, interval: str, *, today: date) -> dict[str, Any]:
    """Which archives make up the history of ``interval`` up to yesterday (UTC).

    Months from :data:`FIRST_MONTH` to the month before ``today``'s use their monthly archive
    when its ``.CHECKSUM`` is published, otherwise every published daily archive of the month;
    the current month uses its daily archives before ``today``. Returns ``{"interval",
    "archives": [{kind, period, name, sha256, estimated_bytes}, ...] (chronological),
    "monthly_missing", "daily_missing", "estimated_bytes", ...}``; ``estimated_bytes`` is a
    deliberately high estimate of every planned archive (:data:`ESTIMATED_BYTES_PER_ROW`).
    """
    _require_interval(interval)
    today = _require_date(today)
    first = _month_start(FIRST_MONTH)
    current = today.replace(day=1)
    if current < first:
        raise ValueError(f"today {today.isoformat()} is before the first month {FIRST_MONTH}")
    archives: list[dict[str, Any]] = []
    monthly_missing: list[str] = []
    daily_missing: list[str] = []
    requests = 0

    def probe(archive: Archive) -> bool:
        nonlocal requests
        requests += 1
        sha256 = client.fetch_checksum(archive)
        if sha256 is None:
            return False
        archives.append(_plan_entry(archive, sha256))
        return True

    def probe_days(start: date, stop: date) -> None:
        for day in _days(start, stop):
            daily = Archive("daily", interval, day.isoformat())
            if not probe(daily):
                daily_missing.append(daily.period)

    month = first
    while month < current:
        following = _next_month(month)
        monthly = Archive("monthly", interval, f"{month:%Y-%m}")
        if not probe(monthly):
            monthly_missing.append(monthly.period)
            probe_days(month, following)
        month = following
    probe_days(current, today)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source": "binance",
        "host": HOST,
        "symbol": SYMBOL,
        "interval": interval,
        "first_month": FIRST_MONTH,
        "today": today.isoformat(),
        "archives": archives,
        "monthly_missing": monthly_missing,
        "daily_missing": daily_missing,
        "estimated_bytes": sum(entry["estimated_bytes"] for entry in archives),
        "checksum_requests": requests,
    }


def _planned_archives(
    plan: Mapping[str, Any], interval: str | None = None
) -> list[tuple[Archive, dict]]:
    """Validate a plan; its archives with their entries, in chronological order."""
    if not isinstance(plan, Mapping):
        raise BinanceVisionError(f"a plan is a mapping, got {type(plan).__name__}")
    plan_interval = plan.get("interval")
    if plan_interval not in INTERVALS or (interval is not None and plan_interval != interval):
        expected = list(INTERVALS) if interval is None else interval
        raise BinanceVisionError(f"plan interval {plan_interval!r}, expected {expected!r}")
    entries = plan.get("archives")
    if not isinstance(entries, list):
        raise BinanceVisionError("plan has no archive list")
    for key in ("monthly_missing", "daily_missing"):
        if not isinstance(plan.get(key), list):
            raise BinanceVisionError(f"plan has no {key} list")
    archives: list[tuple[Archive, dict]] = []
    previous_end: int | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            raise BinanceVisionError("plan archive entries are objects")
        try:
            archive = Archive(entry.get("kind"), plan_interval, entry.get("period"))
            _require_sha256(entry.get("sha256"))
        except (TypeError, ValueError) as exc:
            raise BinanceVisionError(f"malformed plan entry {entry!r}: {exc}") from None
        if previous_end is not None and archive.start_ms < previous_end:
            raise BinanceVisionError(f"plan archives overlap or are out of order at {archive.name}")
        previous_end = archive.end_ms
        archives.append((archive, entry))
    return archives


def save_plan(plan: Mapping[str, Any], root: Path | None = None) -> Path:
    """Write the plan to ``raw/binance/plan-<interval>.json`` (validated, atomic)."""
    _planned_archives(plan)
    path = _root(root) / plan_relpath(plan["interval"])
    atomic_write_text(path, json.dumps(plan, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
    return path


def load_plan(interval: str, root: Path | None = None) -> dict[str, Any]:
    """The saved plan of ``interval`` (:class:`BinanceVisionError` when absent or malformed)."""
    relpath = plan_relpath(interval)
    path = _root(root) / relpath
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise BinanceVisionError(
            f"no saved plan for {interval} ({relpath}): run the archive sync first"
        ) from None
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BinanceVisionError(f"{relpath} is not valid JSON: {exc}") from None
    _planned_archives(plan, interval)
    return plan


def _refresh_periods(refresh: Sequence[str]) -> set[str]:
    if isinstance(refresh, str):
        raise TypeError("refresh must be a sequence of periods, not a single string")
    periods = set()
    for period in refresh:
        if not isinstance(period, str) or _PERIOD.fullmatch(period) is None:
            raise ValueError(f"refresh periods are YYYY-MM or YYYY-MM-DD, got {period!r}")
        periods.add(period)
    return periods


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


@dataclass
class _IntervalSync:
    """One interval of a sync run: its saved plan and what is still to download."""

    interval: str
    plan: dict[str, Any]
    plan_path: Path
    present: int
    to_download: list[tuple[Archive, str, str]]
    republished: list[dict[str, Any]]

    @property
    def estimated(self) -> int:
        return sum(_estimated_bytes(archive) for archive, _, _ in self.to_download)


def _plan_interval(
    client: BinanceVisionClient,
    interval: str,
    *,
    today: date,
    base: Path,
    refresh_periods: set[str],
    say: Callable[[str], None],
) -> _IntervalSync:
    """Plan ``interval``, classify its archives against the local copies and save the plan."""
    plan = plan_archives(client, interval, today=today)
    to_download: list[tuple[Archive, str, str]] = []
    republished: list[dict[str, Any]] = []
    present = 0
    for archive, entry in _planned_archives(plan, interval):
        remote = entry["sha256"]
        local = _local_sha256(archive, base)
        if local is None:
            exists = archive.local_path(base).exists()
            to_download.append((archive, remote, "invalid" if exists else "missing"))
        elif local == remote:
            present += 1
        else:
            refreshed = archive.period in refresh_periods
            republished.append(
                {
                    "kind": archive.kind,
                    "period": archive.period,
                    "local_sha256": local,
                    "remote_sha256": remote,
                    "action": "refreshed" if refreshed else "kept_local",
                }
            )
            if refreshed:
                to_download.append((archive, remote, "republished"))
    plan["republished"] = republished
    planned = _IntervalSync(
        interval=interval,
        plan=plan,
        plan_path=save_plan(plan, base),
        present=present,
        to_download=to_download,
        republished=republished,
    )
    say(
        f"plan {interval}: {len(plan['archives'])} archives ({present} present, "
        f"{len(to_download)} to download, {len(republished)} republished); "
        f"{len(plan['monthly_missing'])} months and {len(plan['daily_missing'])} days not "
        f"published; estimated download {_mb(planned.estimated)}"
    )
    for item in republished:
        say(
            f"republished: {item['kind']} {item['period']} remote sha256 "
            f"{item['remote_sha256'][:12]} != local {item['local_sha256'][:12]} "
            f"({item['action'].replace('_', ' ')})"
        )
    return planned


def sync_intervals(
    client: BinanceVisionClient,
    intervals: Sequence[str],
    *,
    today: date,
    root: Path | None = None,
    refresh: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Plan every interval of ``intervals``, check gate 4 once for the run, then download.

    Holds the ``raw/binance`` job lock for the whole run (taking it removes the temporary files
    of killed runs, :func:`kronos_agent.data.store.job_lock`). Every interval is planned and its
    plan saved before anything is downloaded: a planned archive whose local copy verifies and
    matches the remote ``.CHECKSUM`` is kept; one whose verified local copy differs from the
    remote checksum (republished) is reported and kept unless its period is in ``refresh``; a
    missing or invalid one is to download. The estimate of every interval and the run's total
    go to ``progress``, and the run is refused (:class:`DownloadBudgetExceeded`) if the bytes
    already under ``raw/binance`` plus that total would pass :data:`BUDGET_BYTES` (gate 4), with
    nothing downloaded. The downloads of every interval then charge one budget, failed attempts
    included. Returns one summary per interval, in the order given; the first one carries the
    temporary files removed (``removed_temporaries``).
    """
    if isinstance(intervals, str):
        raise TypeError("intervals must be a sequence of intervals, not a single string")
    names = [_require_interval(interval) for interval in intervals]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"intervals must be distinct and at least one, got {names!r}")
    base = _root(root)
    refresh_periods = _refresh_periods(refresh)
    say = progress if progress is not None else (lambda message: None)
    with job_lock(LOCK_RELPATH, base) as removed:
        planned = [
            _plan_interval(
                client,
                interval,
                today=today,
                base=base,
                refresh_periods=refresh_periods,
                say=say,
            )
            for interval in names
        ]
        budget = DownloadBudget.from_disk(base)
        disk_before = budget.used_bytes
        estimated = sum(item.estimated for item in planned)
        say(
            f"gate 4: estimated download {_mb(estimated)} for {', '.join(names)}, "
            f"{_mb(disk_before)} already under {RAW_RELDIR}, budget {_mb(budget.limit_bytes)}"
        )
        if disk_before + estimated > budget.limit_bytes:
            raise DownloadBudgetExceeded(
                f"gate 4: {disk_before} bytes under {RAW_RELDIR} + an estimated {estimated} bytes "
                f"for this run ({', '.join(names)}) would pass the {budget.limit_bytes}-byte "
                "budget; nothing downloaded"
            )
        results: list[dict[str, Any]] = []
        for item in planned:
            downloaded: list[dict[str, Any]] = []
            for archive, sha256, reason in item.to_download:
                path = client.download(archive, expected_sha256=sha256, root=base, budget=budget)
                size = path.stat().st_size
                downloaded.append({"name": archive.name, "bytes": size, "reason": reason})
                say(f"downloaded {archive.name} ({_mb(size)}, sha256 verified, {reason})")
            refreshed = {e["period"] for e in item.republished if e["action"] == "refreshed"}
            results.append(
                {
                    "interval": item.interval,
                    "plan_relpath": item.plan_path.relative_to(base).as_posix(),
                    "archives": len(item.plan["archives"]),
                    "present": item.present,
                    "downloaded": downloaded,
                    "downloaded_bytes": sum(entry["bytes"] for entry in downloaded),
                    "republished": item.republished,
                    "refresh_unused": sorted(refresh_periods - refreshed),
                    "monthly_missing": item.plan["monthly_missing"],
                    "daily_missing": item.plan["daily_missing"],
                    "estimated_download_bytes": item.estimated,
                    "run_estimated_download_bytes": estimated,
                    "disk_bytes_before": disk_before,
                    "budget_limit_bytes": budget.limit_bytes,
                    "budget_used_bytes": budget.used_bytes,
                    "removed_temporaries": [] if results else list(removed),
                }
            )
    return results


def sync_archives(
    client: BinanceVisionClient,
    interval: str,
    *,
    today: date,
    root: Path | None = None,
    refresh: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """:func:`sync_intervals` of the single ``interval``: plan, gate 4, download; its summary."""
    (result,) = sync_intervals(
        client, [interval], today=today, root=root, refresh=refresh, progress=progress
    )
    return result


# -- parsing ---------------------------------------------------------------------------------------


def _zip_member(data: bytes, archive: Archive) -> bytes:
    """The bytes of the single expected CSV member, read in memory and capped."""
    where = archive.name
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            members = bundle.infolist()
            if len(members) != 1:
                raise ArchiveFormatError(
                    f"{where}: {len(members)} members, expected exactly {archive.member_name}"
                )
            info = members[0]
            if info.filename != archive.member_name or info.is_dir():
                raise ArchiveFormatError(
                    f"{where}: member {info.filename[:80]!r}, expected {archive.member_name!r}"
                )
            if info.flag_bits & 0x1:
                raise ArchiveFormatError(f"{where}: encrypted member")
            if info.file_size > CSV_MAX_BYTES:
                raise ArchiveFormatError(
                    f"{where}: member is {info.file_size} bytes uncompressed, cap {CSV_MAX_BYTES}"
                )
            with bundle.open(info) as handle:
                payload = handle.read(CSV_MAX_BYTES + 1)
    except (
        zipfile.BadZipFile,
        zlib.error,
        EOFError,
        NotImplementedError,
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        raise ArchiveFormatError(f"{where}: not a readable zip archive ({exc})") from None
    if len(payload) > CSV_MAX_BYTES:
        raise ArchiveFormatError(f"{where}: member exceeds the {CSV_MAX_BYTES}-byte cap")
    return payload


def _first_mismatch(values: Sequence[str], pattern: re.Pattern[str]) -> int | None:
    matches = list(map(pattern.fullmatch, values))
    return matches.index(None) if None in matches else None


def _parse_kline_csv(payload: bytes, archive: Archive) -> pd.DataFrame:
    """Strict parse of a kline CSV (see the module docstring) into a string frame."""
    where = archive.name
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        line = payload.count(b"\n", 0, exc.start) + 1
        raise ArchiveFormatError(f"{where} line {line}: non-ASCII byte") from None
    reader = csv.reader(io.StringIO(text, newline=""), quoting=csv.QUOTE_NONE, strict=True)
    rows: list[list[str]] = []
    try:
        for row in reader:
            if not row:
                raise ArchiveFormatError(f"{where} line {reader.line_num}: blank line")
            if len(row) != len(KLINE_COLUMNS):
                raise ArchiveFormatError(
                    f"{where} line {reader.line_num}: {len(row)} fields, expected "
                    f"{len(KLINE_COLUMNS)}"
                )
            rows.append(row)
    except csv.Error as exc:
        raise ArchiveFormatError(f"{where} line {reader.line_num}: {exc}") from None
    if not rows:
        raise ArchiveFormatError(f"{where}: no rows")
    first_line = 1
    if _ALL_DIGITS.fullmatch(rows[0][0]) is None:
        if tuple(rows[0]) != KLINE_COLUMNS:
            raise ArchiveFormatError(
                f"{where} line 1: header {[field[:40] for field in rows[0]]} != expected "
                f"{list(KLINE_COLUMNS)}"
            )
        rows = rows[1:]
        first_line = 2
        if not rows:
            raise ArchiveFormatError(f"{where}: a header and no rows")
    fields = list(zip(*rows, strict=True))
    checks: list[tuple[int, str, re.Pattern[str], str]] = [
        (0, "open_time", _TIME, "an integer time"),
        (6, "close_time", _TIME, "an integer time"),
    ]
    for name, index in _STORE_FIELDS:
        if name == "count":
            checks.append((index, name, _COUNT, "an unsigned integer"))
        else:
            checks.append((index, name, _DECIMAL, "a plain decimal"))
    worst: tuple[int, str] | None = None
    for index, name, pattern, description in checks:
        bad = _first_mismatch(fields[index], pattern)
        if bad is not None and (worst is None or bad < worst[0]):
            worst = (bad, f"{name} is not {description}")
    if worst is not None:
        raise ArchiveFormatError(f"{where} line {first_line + worst[0]}: {worst[1]}")
    opens = np.array([int(value) for value in fields[0]], dtype=np.int64)
    closes = np.array([int(value) for value in fields[6]], dtype=np.int64)
    step = archive.step_ms
    micro = bool(opens[0] >= MICROSECOND_THRESHOLD)
    if micro:
        open_ms = opens // 1000
        expected_close = opens + step * 1000 - 1
        unit_bad = (opens < MICROSECOND_THRESHOLD) | (opens % 1000 != 0)
        unit_reason = "open_time is not a whole millisecond in microseconds like the first row"
    else:
        open_ms = opens
        expected_close = opens + step - 1
        unit_bad = opens >= MICROSECOND_THRESHOLD
        unit_reason = "open_time is in microseconds while the first row is in milliseconds"
    order_bad = np.zeros(open_ms.size, dtype=bool)
    order_bad[1:] = np.diff(open_ms) <= 0
    problems = (
        (unit_bad, unit_reason),
        (open_ms % step != 0, f"open_time is not aligned to the {archive.interval} interval"),
        (closes != expected_close, "close_time != open_time + interval - 1"),
        (order_bad, "open_time is not strictly increasing (duplicate or out of order)"),
        (
            (open_ms < archive.start_ms) | (open_ms >= archive.end_ms),
            f"open_time is outside the archive's period {archive.period}",
        ),
    )
    first_bad: tuple[int, str] | None = None
    for mask, reason in problems:
        hits = np.flatnonzero(mask)
        if hits.size and (first_bad is None or int(hits[0]) < first_bad[0]):
            first_bad = (int(hits[0]), reason)
    if first_bad is not None:
        raise ArchiveFormatError(f"{where} line {first_line + first_bad[0]}: {first_bad[1]}")
    data: dict[str, Any] = {"open_ms": open_ms.astype(np.int64)}
    for name, index in _STORE_FIELDS:
        data[name] = pd.Series(fields[index], dtype=object)
    return pd.DataFrame(data, columns=list(BINANCE_COLUMNS))


def _read_verified(archive: Archive, root: Path) -> tuple[pd.DataFrame, str]:
    """Parse the local archive after checking its bytes against its local ``.CHECKSUM``."""
    expected = _local_checksum(archive, root)
    if expected is None:
        raise BinanceVisionError(
            f"{archive.relpath}: no valid local .CHECKSUM; run the archive sync first"
        )
    path = archive.local_path(root)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise BinanceVisionError(
            f"{archive.relpath} is missing; run the archive sync first"
        ) from None
    if size > ARCHIVE_MAX_BYTES:
        raise ArchiveFormatError(f"{archive.name}: {size} bytes, cap {ARCHIVE_MAX_BYTES}")
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != expected:
        raise ChecksumMismatchError(
            f"{archive.relpath}: sha256 {actual[:12]} differs from its .CHECKSUM "
            f"{expected[:12]}; delete it and run the archive sync again"
        )
    return _parse_kline_csv(_zip_member(data, archive), archive), actual


def read_archive(archive: Archive, root: Path | None = None) -> pd.DataFrame:
    """The ``BINANCE_COLUMNS`` string frame of a local archive (strict parse, checksum first).

    The zip's bytes must match the local ``.CHECKSUM`` (:class:`ChecksumMismatchError`); a
    malformed archive raises :class:`ArchiveFormatError` naming the archive and the line.
    """
    return _read_verified(_require_archive(archive), _root(root))[0]


# -- stores ----------------------------------------------------------------------------------------


def _require_verified(archives: Sequence[tuple[Archive, dict]], root: Path) -> None:
    bad = [archive.name for archive, _ in archives if _local_sha256(archive, root) is None]
    if bad:
        shown = ", ".join(bad[:10]) + (f" and {len(bad) - 10} more" if len(bad) > 10 else "")
        raise BinanceVisionError(
            f"{len(bad)} planned archive(s) missing or failing their checksum ({shown}); "
            "run the archive sync first"
        )


def _require_planned_sha(
    archive: Archive, entry: Mapping[str, Any], sha256: str, plan: Mapping[str, Any]
) -> None:
    """The local archive is the planned one, or a republished one the sync chose to keep."""
    if sha256 == entry["sha256"]:
        return
    for item in plan.get("republished", []):
        if (
            item.get("kind") == archive.kind
            and item.get("period") == archive.period
            and item.get("local_sha256") == sha256
            and item.get("action") == "kept_local"
        ):
            return
    raise BinanceVisionError(
        f"{archive.name}: the local archive (sha256 {sha256[:12]}) is not the one planned "
        f"({entry['sha256'][:12]}); run the archive sync again"
    )


def _archive_record(
    archive: Archive, sha256: str, frame: pd.DataFrame, entry: Mapping[str, Any]
) -> dict[str, Any]:
    opens = frame["open_ms"].to_numpy(dtype=np.int64)
    record: dict[str, Any] = {
        "kind": archive.kind,
        "period": archive.period,
        "name": archive.name,
        "sha256": sha256,
        "rows": int(opens.size),
        "first_open_ms": int(opens[0]),
        "last_open_ms": int(opens[-1]),
    }
    if entry["sha256"] != sha256:
        record["remote_sha256"] = entry["sha256"]
    return record


def _compare_rows(daily: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, Any]:
    """Rows of a daily archive vs the same day in the monthly archive, compared as decimals.

    Values are recorded only for candles opening before the sealed start; sealed differences
    are listed by open time and column.
    """
    d_open = daily["open_ms"].to_numpy(dtype=np.int64)
    m_open = monthly["open_ms"].to_numpy(dtype=np.int64)
    common, d_rows, m_rows = np.intersect1d(d_open, m_open, assume_unique=True, return_indices=True)
    only_daily = np.setdiff1d(d_open, m_open, assume_unique=True)
    only_monthly = np.setdiff1d(m_open, d_open, assume_unique=True)
    differing = 0
    samples: list[dict[str, Any]] = []
    for column in BINANCE_COLUMNS[1:]:
        ours = daily[column].to_numpy(dtype=object)[d_rows]
        theirs = monthly[column].to_numpy(dtype=object)[m_rows]
        for row in range(common.size):
            if ours[row] == theirs[row] or Decimal(ours[row]) == Decimal(theirs[row]):
                continue
            differing += 1
            if len(samples) < MAX_SAMPLES:
                open_ms = int(common[row])
                sample: dict[str, Any] = {"open_ms": open_ms, "column": column}
                if open_ms < SEALED_START_MS:
                    sample.update(daily=ours[row], monthly=theirs[row])
                samples.append(sample)
    return {
        "status": "compared",
        "rows_daily": int(d_open.size),
        "rows_monthly": int(m_open.size),
        "rows_compared": int(common.size),
        "only_in_daily": int(only_daily.size),
        "only_in_daily_open_ms": [int(v) for v in only_daily[:MAX_SAMPLES]],
        "only_in_monthly": int(only_monthly.size),
        "only_in_monthly_open_ms": [int(v) for v in only_monthly[:MAX_SAMPLES]],
        "differing_values": differing,
        "difference_samples": samples,
        "equal": differing == 0 and only_daily.size == 0 and only_monthly.size == 0,
    }


def _compare_replaced_dailies(
    monthly: Archive, frame: pd.DataFrame, root: Path
) -> list[dict[str, Any]]:
    """Compare every daily archive on disk for ``monthly``'s month with the monthly rows."""
    start = _month_start(monthly.period)
    results: list[dict[str, Any]] = []
    opens = frame["open_ms"].to_numpy(dtype=np.int64)
    for day in _days(start, _next_month(start)):
        daily = Archive("daily", monthly.interval, day.isoformat())
        if not (daily.local_path(root).exists() or daily.checksum_path(root).exists()):
            continue
        record: dict[str, Any] = {"period": daily.period, "name": daily.name}
        try:
            daily_frame, sha256 = _read_verified(daily, root)
        except BinanceVisionError as exc:
            record.update(status="unverified", error=str(exc))
            results.append(record)
            continue
        record["sha256"] = sha256
        inside = (opens >= daily.start_ms) & (opens < daily.end_ms)
        record.update(_compare_rows(daily_frame, frame[inside].reset_index(drop=True)))
        results.append(record)
    return results


def _plan_provenance(plan: Mapping[str, Any], root: Path) -> dict[str, Any]:
    relpath = plan_relpath(plan["interval"])
    return {
        "relpath": relpath,
        "sha256": sha256_file(root / relpath),
        "today": plan.get("today"),
        "first_month": plan.get("first_month"),
    }


def _archive_provenance(
    plan: Mapping[str, Any],
    root: Path,
    used: list[dict[str, Any]],
    replaced: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": "binance_vision_archives",
        "host": HOST,
        "symbol": SYMBOL,
        "interval": plan["interval"],
        "plan": _plan_provenance(plan, root),
        "archives": used,
        "monthly_missing": list(plan["monthly_missing"]),
        "daily_missing": list(plan["daily_missing"]),
        "republished": list(plan.get("republished", [])),
        "replaced_dailies": replaced,
    }


def build_binance_stores(*, root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Build ``binance/1m`` from the saved 1m plan and resample it to 5m, 15m and 1h.

    Every planned archive must be present and verified. The archives are read one at a time
    (checksum, strict parse), appended to the 1m store and resampled exactly to each target
    (complete buckets only; :mod:`kronos_agent.data.resample`), so neither the 1m store nor any
    target is ever held whole. Returns the four store metas keyed by interval; the resampled
    stores record the 1m CSV sha256 they come from and every incomplete bucket.
    """
    base = _root(root)
    with exclusive_lock(base / LOCK_RELPATH):
        plan = load_plan("1m", base)
        archives = _planned_archives(plan, "1m")
        if not archives:
            raise BinanceVisionError("the 1m plan holds no archive")
        _require_verified(archives, base)
        used: list[dict[str, Any]] = []
        replaced: list[dict[str, Any]] = []
        incomplete: dict[str, list[list[int]]] = {target: [] for target in TARGET_INTERVALS}
        with contextlib.ExitStack() as stack:
            writers = {
                interval: stack.enter_context(StoreWriter(StoreId("binance", interval), root=base))
                for interval in ("1m", *TARGET_INTERVALS)
            }
            for archive, entry in archives:
                frame, sha256 = _read_verified(archive, base)
                _require_planned_sha(archive, entry, sha256, plan)
                writers["1m"].append(frame)  # validates the frame in the store format
                resampled = resample_many(frame, TARGET_INTERVALS, validate=False)
                for target, (target_frame, missing) in resampled.items():
                    writers[target].append(target_frame)
                    incomplete[target].extend([open_ms, minutes] for open_ms, minutes in missing)
                used.append(_archive_record(archive, sha256, frame, entry))
                if archive.kind == "monthly":
                    replaced.extend(_compare_replaced_dailies(archive, frame, base))
            metas = {
                "1m": writers["1m"].finish(
                    provenance=_archive_provenance(plan, base, used, replaced)
                )
            }
            for target in TARGET_INTERVALS:
                metas[target] = writers[target].finish(
                    provenance={
                        "kind": "resampled",
                        "from": StoreId("binance", "1m").relpath,
                        "from_csv_sha256": metas["1m"]["csv_sha256"],
                        "from_rows": metas["1m"]["rows"],
                        "target": target,
                        "rule": (
                            "complete buckets only; open = first minute, close = last minute, "
                            "high/low = first largest/smallest decimal, exact decimal sums of "
                            "the four volumes, integer sum of count"
                        ),
                        "incomplete_bucket_count": len(incomplete[target]),
                        "incomplete_buckets": incomplete[target],
                    }
                )
    return metas


def build_official_1h(*, root: Path | None = None) -> dict[str, Any]:
    """Build ``binance_official/1h`` from the saved 1h plan (Binance's own 1h archives).

    Used only by the resample check. Returns the store meta.
    """
    base = _root(root)
    with exclusive_lock(base / LOCK_RELPATH):
        plan = load_plan("1h", base)
        archives = _planned_archives(plan, "1h")
        if not archives:
            raise BinanceVisionError("the 1h plan holds no archive")
        _require_verified(archives, base)
        used: list[dict[str, Any]] = []
        replaced: list[dict[str, Any]] = []
        with StoreWriter(StoreId("binance_official", "1h"), root=base) as writer:
            for archive, entry in archives:
                frame, sha256 = _read_verified(archive, base)
                _require_planned_sha(archive, entry, sha256, plan)
                writer.append(frame)
                used.append(_archive_record(archive, sha256, frame, entry))
                if archive.kind == "monthly":
                    replaced.extend(_compare_replaced_dailies(archive, frame, base))
            return writer.finish(provenance=_archive_provenance(plan, base, used, replaced))


# -- the resample check ----------------------------------------------------------------------------


def resample_check_config() -> dict[str, Any]:
    """The recorded definition of ``reports/data/binance_resample_check`` (SPEC rule 7)."""
    return {
        "study": RESAMPLE_CHECK_STEM,
        "version": 1,
        "stores": {
            "resampled": StoreId("binance", "1h").relpath,
            "official": StoreId("binance_official", "1h").relpath,
            "minutes": StoreId("binance", "1m").relpath,
        },
        "hours": "every hour present in both stores, all months (not a sample)",
        "comparison": "as decimals",
        "equal_columns": list(EXACT_COLUMNS),
        "tolerance_columns": list(TOLERANCE_COLUMNS),
        "tolerance": (
            "sum over the 60 minute values of 0.5 * 10^-d_i plus 0.5 * 10^-d_official, d = "
            "digits after the decimal point of each string"
        ),
        "bucket_rule": "a 1h bucket is kept only when all 60 minutes are present",
        "sealed_start": utc_text(SEALED_START_MS),
        "sealed_policy": "hours at or after the sealed start: counts and open times only",
        "lost_decision_hours": {
            "windows_hours": list(LOST_HOURS_WINDOWS),
            "definition": (
                "a decision hour D is lost when [D - window, D) holds an hour whose 1m bucket "
                "is incomplete (1 to 59 minutes present)"
            ),
            "segments": [
                [
                    segment.name,
                    utc_text(segment.start_ms),
                    None if segment.end_ms is None else utc_text(segment.end_ms),
                ]
                for segment in PARTITION
            ],
        },
        "max_samples_per_column": MAX_SAMPLES,
    }


def _verified_meta(store_id: StoreId, root: Path) -> dict[str, Any]:
    """The store's meta after checking its CSV's size and sha256 once."""
    meta = read_meta(store_id, root)
    path = store_id.csv_path(root)
    if (
        not path.is_file()
        or path.stat().st_size != meta["csv_size"]
        or sha256_file(path) != meta["csv_sha256"]
    ):
        raise StoreIntegrityError(
            f"{store_id.relpath}: size or sha256 differs from its meta; rebuild the store"
        )
    return meta


def _minute_rounding(store_id: StoreId, meta: Mapping[str, Any], root: Path) -> pd.DataFrame:
    """:func:`quote_rounding` per hour over the whole 1m store, read in 31-day slices.

    Slices start on whole hours, so no hour is split between two reads.
    """
    parts: list[pd.DataFrame] = []
    first, last = meta.get("first_open_ms"), meta.get("last_open_ms")
    if first is not None and last is not None:
        span = 31 * 24 * HOUR_MS
        start = int(first) - int(first) % HOUR_MS
        while start <= last:
            frame = read_store(
                store_id,
                start_ms=start,
                end_ms=start + span,
                usecols=list(TOLERANCE_COLUMNS),
                root=root,
                verify=False,  # verified once by _verified_meta; ranged reads never rehash
            )
            if not frame.empty:
                parts.append(quote_rounding(frame, "1h"))
            start += span
    if not parts:
        empty = pd.DataFrame(
            {
                "open_ms": np.empty(0, dtype=np.int64),
                **{name: pd.Series([], dtype=object) for name in TOLERANCE_COLUMNS},
            }
        )
        return quote_rounding(empty, "1h")
    return pd.concat(parts, ignore_index=True)


def _segment_order() -> list[str]:
    """``before_training`` and the partition's segments, in time order."""
    return ["before_training", *(segment.name for segment in PARTITION)]


def _segment_names(decision_ms: np.ndarray) -> np.ndarray:
    starts = np.array([segment.start_ms for segment in PARTITION], dtype=np.int64)
    names = np.array(_segment_order(), dtype=object)
    return names[np.searchsorted(starts, decision_ms, side="right")]


def _lost_decision_hours(
    incomplete_open_ms: np.ndarray, first_hour_ms: int, last_hour_ms: int
) -> dict[str, Any]:
    """Per window and segment: decision hours whose window holds an incomplete hour."""
    hours = (last_hour_ms - first_hour_ms) // HOUR_MS + 1
    flags = np.zeros(hours, dtype=np.int64)
    flags[(incomplete_open_ms - first_hour_ms) // HOUR_MS] = 1
    prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(flags)))
    segment_order = _segment_order()
    result: dict[str, Any] = {}
    for window in LOST_HOURS_WINDOWS:
        index = np.arange(window, hours + 1)
        decision_ms = first_hour_ms + index * HOUR_MS
        lost = (prefix[index] - prefix[index - window]) > 0
        names = _segment_names(decision_ms)
        result[f"{window}h"] = {
            name: {"eligible": int((names == name).sum()), "lost": int(lost[names == name].sum())}
            for name in segment_order
        }
    return result


def check_resample(*, root: Path | None = None, directory: Path | None = None) -> dict[str, Any]:
    """The payload of ``reports/data/binance_resample_check`` (header keys included).

    Records (or verifies) the study config first, under ``directory`` (default
    ``reports/data``). Then compares every hour present in both ``binance/1h`` and
    ``binance_official/1h`` (:func:`kronos_agent.data.resample.compare_with_official`), with the
    quote-volume tolerance computed from the 1m strings of each hour, lists the hours present on
    one side only with their 1m minute counts and counts the decision hours lost to incomplete
    buckets per segment. Integrity reads only (``read_store``); every value at or after the
    sealed start stays out of the payload.
    """
    base = _root(root)
    config = resample_check_config()
    record_config(RESAMPLE_CHECK_STEM, config, directory=directory)
    minute_id = StoreId("binance", "1m")
    hour_id = StoreId("binance", "1h")
    official_id = StoreId("binance_official", "1h")
    minute_meta = _verified_meta(minute_id, base)
    hour_meta = read_meta(hour_id, base)
    official_meta = read_meta(official_id, base)
    if hour_meta.get("provenance", {}).get("from_csv_sha256") != minute_meta["csv_sha256"]:
        raise StoreIntegrityError(
            f"{hour_id.relpath} was not resampled from the current {minute_id.relpath}; rebuild"
        )
    hourly = read_store(hour_id, root=base)
    official = read_store(official_id, root=base)
    rounding = _minute_rounding(minute_id, minute_meta, base)
    minutes = dict(
        zip(
            rounding["open_ms"].to_numpy(dtype=np.int64).tolist(),
            rounding["minutes"].to_numpy(dtype=np.int64).tolist(),
            strict=True,
        )
    )
    complete = rounding[rounding["minutes"] == 60].reset_index(drop=True)
    merged = hourly.merge(
        complete[["open_ms", *ROUNDING_COLUMNS]], on="open_ms", how="left", validate="one_to_one"
    )
    if merged[list(ROUNDING_COLUMNS)].isna().any().any():
        missing = merged.loc[merged[ROUNDING_COLUMNS[0]].isna(), "open_ms"].iloc[0]
        raise StoreIntegrityError(
            f"{hour_id.relpath}: hour {utc_text(int(missing))} has no complete 1m bucket; rebuild"
        )
    comparison = compare_with_official(merged, official, sealed_start_ms=SEALED_START_MS)
    only_official = [
        {
            "open_ms": open_ms,
            "hour": utc_text(open_ms),
            "minutes": int(minutes.get(open_ms, 0)),
            "sealed": open_ms >= SEALED_START_MS,
        }
        for open_ms in comparison.pop("only_official")
    ]
    only_resampled = [
        {"open_ms": open_ms, "hour": utc_text(open_ms), "sealed": open_ms >= SEALED_START_MS}
        for open_ms in comparison.pop("only_resampled")
    ]
    partial = rounding[(rounding["minutes"] > 0) & (rounding["minutes"] < 60)]
    incomplete_open_ms = partial["open_ms"].to_numpy(dtype=np.int64)
    first_hour = int(minute_meta["first_open_ms"]) // HOUR_MS * HOUR_MS
    last_hour = int(minute_meta["last_open_ms"]) // HOUR_MS * HOUR_MS
    incomplete_names = _segment_names(incomplete_open_ms)
    segment_order = _segment_order()
    if _verified_meta(minute_id, base)["csv_sha256"] != minute_meta["csv_sha256"]:
        raise StoreIntegrityError(f"{minute_id.relpath} changed during the check")
    mismatches = comparison["exact_mismatches"] + comparison["tolerance_mismatches"]
    body = {
        "inputs": {
            meta["csv_relpath"]: {
                "csv_sha256": meta["csv_sha256"],
                "rows": meta["rows"],
                "first_open": meta.get("first_open_utc"),
                "last_open": meta.get("last_open_utc"),
            }
            for meta in (minute_meta, hour_meta, official_meta)
        },
        "comparison": comparison,
        "only_official": only_official,
        "only_resampled": only_resampled,
        "incomplete_hours": {
            "count": int(incomplete_open_ms.size),
            "by_segment": {name: int((incomplete_names == name).sum()) for name in segment_order},
            "hours": [
                [int(open_ms), int(count)]
                for open_ms, count in zip(
                    incomplete_open_ms, partial["minutes"].to_numpy(dtype=np.int64), strict=True
                )
            ],
        },
        "lost_decision_hours": _lost_decision_hours(incomplete_open_ms, first_hour, last_hour),
        "owner_attention": mismatches > 0,
    }
    header = report_header(
        RESAMPLE_CHECK_STEM,
        config,
        statistics_with_intervals=0,
        seeds={"bootstrap": "none: a deterministic comparison of every hour, no random draws"},
        samples={
            "hours_compared": comparison["hours_compared"],
            "hours_compared_before_sealed": comparison["hours_compared_before_sealed"],
            "hours_compared_sealed": comparison["hours_compared_sealed"],
            "hours_only_resampled": len(only_resampled),
            "hours_only_official": len(only_official),
            "incomplete_hours": int(incomplete_open_ms.size),
        },
        comparators={
            "binance_official_1h": (
                "Binance's own 1h archives: each resampled hour is compared with the exchange's "
                "aggregate of the same trades"
            ),
            "quote_volume_tolerance": config["tolerance"],
        },
        directory=directory,
    )
    return {**header, **body}


def _md_cell(value: Any) -> str:
    return "-" if value is None else str(value).replace("|", "/")


def render_resample_markdown(report: Mapping[str, Any]) -> str:
    """The Markdown of the resample check (ASCII; sealed hours by open time only).

    Every table follows a fixed order, so the payload and the JSON written from it (sorted
    keys) render the same text.
    """
    comparison = report["comparison"]
    code = report.get("code", {})
    lines = [
        "# Binance resample check",
        "",
        (
            "`binance/1h` (resampled here from `binance/1m`) against `binance_official/1h` "
            "(Binance's own 1h archives): every hour present in both, compared as decimals "
            '(docs/DATA.md, "Resampling and its check").'
        ),
        "",
        (
            f"- Study config sha256 `{report.get('config_sha256')}`; DATA.md sha256 "
            f"`{report.get('data_md_sha256')}`."
        ),
        (
            f"- Code: tree sha256 `{code.get('source_tree_sha256')}`, commit "
            f"`{code.get('commit')}`, dirty {code.get('dirty')}. Generated at "
            f"{report.get('generated_at')}."
        ),
        (
            f"- Tests consumed: {report.get('tests_consumed')}; statistics with intervals: "
            f"{report.get('statistics_with_intervals')} (a deterministic comparison, no sampling)."
        ),
        "",
        "## Inputs",
        "",
        "| store | rows | first open | last open | csv sha256 |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for relpath in sorted(report["inputs"]):
        item = report["inputs"][relpath]
        lines.append(
            f"| `{relpath}` | {item['rows']} | {_md_cell(item['first_open'])} | "
            f"{_md_cell(item['last_open'])} | `{item['csv_sha256'][:16]}` |"
        )
    sealed_from = utc_text(int(comparison["sealed_start_ms"]))
    lines += [
        "",
        "## Hours",
        "",
        (
            f"- Compared: {comparison['hours_compared']} "
            f"({comparison['hours_compared_before_sealed']} before the sealed start, "
            f"{comparison['hours_compared_sealed']} at or after {sealed_from}: counts and open "
            "times only)."
        ),
        (
            f"- Only in `binance/1h`: {len(report['only_resampled'])}. Only in "
            f"`binance_official/1h`: {len(report['only_official'])}."
        ),
        "",
        "## Columns",
        "",
        (
            "| column | rule | compared | mismatches | before sealed | sealed | largest "
            "difference (before sealed) | at |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for column in (*EXACT_COLUMNS, *TOLERANCE_COLUMNS):
        stats = comparison["columns"][column]
        at = stats["max_abs_difference_open_ms"]
        lines.append(
            f"| {column} | {stats['rule']} | {stats['compared']} | {stats['mismatches']} | "
            f"{stats['mismatches_before_sealed']} | {stats['mismatches_sealed']} | "
            f"{stats['max_abs_difference']} | {'-' if at is None else utc_text(int(at))} |"
        )
    lines += [
        "",
        (
            "Quote volumes may differ by the rounding of the 60 minute values and of the "
            "official value: sum(0.5 * 10^-d_i) + 0.5 * 10^-d_official. Largest |difference| / "
            "tolerance before the sealed start:"
        ),
        "",
    ]
    for column in TOLERANCE_COLUMNS:
        stats = comparison["columns"][column]
        at = stats["max_ratio_open_ms"]
        lines.append(
            f"- {column}: {stats['max_ratio_to_tolerance']:.6g}"
            + ("" if at is None else f" at {utc_text(int(at))}")
        )
    lines += ["", "## Mismatches before the sealed start", ""]
    samples = comparison["mismatch_samples"]
    if samples:
        lines += [
            (
                f"At most {comparison['max_mismatch_samples_per_column']} per column; the "
                "counts above are complete."
            ),
            "",
            "| hour | column | resampled | official | difference | tolerance |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {s['hour']} | {s['column']} | {s['resampled']} | {s['official']} | "
            f"{s['difference']} | {_md_cell(s['tolerance'])} |"
            for s in samples
        ]
    else:
        lines.append("None.")
    lines += ["", f"## Mismatches at or after {sealed_from} (open times only)", ""]
    sealed = comparison["sealed_mismatches"]
    if sealed:
        lines += [f"- {s['hour']} ({s['open_ms']}): {s['column']}" for s in sealed]
    else:
        lines.append("None.")
    lines += ["", "## Hours on one side only", ""]
    one_side = [
        (item["hour"], "binance_official/1h only", item["minutes"])
        for item in report["only_official"]
    ] + [(item["hour"], "binance/1h only", None) for item in report["only_resampled"]]
    one_side.sort()
    if one_side:
        lines += ["| hour | present in | minutes in binance/1m |", "| --- | --- | ---: |"]
        lines += [
            f"| {hour} | {side} | {_md_cell(count)} |"
            for hour, side, count in one_side[:MAX_MARKDOWN_ROWS]
        ]
        if len(one_side) > MAX_MARKDOWN_ROWS:
            lines.append(f"\n{len(one_side) - MAX_MARKDOWN_ROWS} more in the JSON report.")
    else:
        lines.append("None.")
    lost = report["lost_decision_hours"]
    windows = [f"{window}h" for window in LOST_HOURS_WINDOWS if f"{window}h" in lost]
    lines += [
        "",
        "## Decision hours lost to incomplete buckets",
        "",
        (
            "A decision hour D is lost when its 512 h Kronos context [D - 512 h, D) holds an "
            "hour whose 1m bucket is incomplete (1 to 59 minutes): binance/1h keeps complete "
            "buckets only."
        ),
        "",
        "| segment | incomplete hours | "
        + " | ".join(f"eligible ({w}) | lost ({w})" for w in windows)
        + " |",
        "| --- | ---: | " + " | ".join("---: | ---:" for _ in windows) + " |",
    ]
    by_segment = report["incomplete_hours"]["by_segment"]
    for name in _segment_order():
        count = by_segment[name]
        cells = " | ".join(
            f"{lost[w][name]['eligible']} | {lost[w][name]['lost']}" for w in windows
        )
        lines.append(f"| {name} | {count} | {cells} |")
    exact = comparison["exact_mismatches"]
    tolerant = comparison["tolerance_mismatches"]
    lines += ["", "## Reading", ""]
    if exact or tolerant:
        lines.append(
            f"- {exact} OHLC, volume or count mismatches and {tolerant} quote-volume mismatches "
            "beyond the rounding tolerance: reported to the owner."
        )
    else:
        lines.append(
            f"- All {comparison['hours_compared']} compared hours agree: OHLC, volume, count and "
            "taker-buy volume equal as decimals, quote volumes within the rounding tolerance."
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "ARCHIVE_MAX_BYTES",
    "BUDGET_BYTES",
    "CHECKSUM_MAX_BYTES",
    "CSV_MAX_BYTES",
    "ESTIMATED_BYTES_PER_ROW",
    "FIRST_MONTH",
    "HOST",
    "INTERVALS",
    "KINDS",
    "KLINE_COLUMNS",
    "LOCK_RELPATH",
    "MIN_INTERVAL_S",
    "RAW_RELDIR",
    "RESAMPLE_CHECK_STEM",
    "RETRY_BACKOFF_S",
    "SYMBOL",
    "Archive",
    "ArchiveFormatError",
    "BinanceVisionClient",
    "BinanceVisionError",
    "BinanceVisionUnavailable",
    "ChecksumMismatchError",
    "DownloadBudget",
    "DownloadBudgetExceeded",
    "HostNotAllowedError",
    "build_binance_stores",
    "build_official_1h",
    "check_resample",
    "checksum_line",
    "load_plan",
    "parse_checksum",
    "plan_archives",
    "plan_relpath",
    "read_archive",
    "render_resample_markdown",
    "resample_check_config",
    "save_plan",
    "sync_archives",
    "sync_intervals",
    "verify_local",
]
