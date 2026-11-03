"""``kronos-agent data binance|bingx|status|coverage``: the candle stores (``docs/DATA.md``).

- ``binance [--interval 1m] [--interval 1h] [--refresh PERIOD] [--record-config]``: plan every
  requested interval (both by default) of the data.binance.vision kline archives, then download
  and verify them, then build ``candles/binance/1m`` with its exact 5m/15m/1h resamples (from the
  1m archives) and ``candles/binance_official/1h`` (from the 1h archives), and run the resample
  check (``reports/data/binance_resample_check``) once both 1h stores exist. The check's recorded
  definition is verified before anything is downloaded. Gate 4 is checked once for the whole
  run: a run whose estimate for all its intervals plus the bytes already under ``raw/binance``
  would pass the 4.5 GB budget is refused (exit 2) before any download; data.binance.vision
  being unreachable is a failure the owner must hear about (SPEC section 5).
- ``bingx --interval 1h|15m|5m|1m [--max-chunks N]``: extend the chunk archive of the BingX real
  market (public GET, settled candles only, every chunk fetched twice until the fetches match)
  and rebuild ``candles/bingx/<interval>`` from every archived chunk. Archived candles the
  exchange now serves with other values are a WARNING, printed for each chunk as soon as they
  are recorded (so also when a later chunk fails): the archive and the store keep the first-seen
  rows, ``raw/bingx/revision-<interval>.json`` records each revision once, and the run goes on.
  An interval whose backfill fails is reported (``ERROR:``) and its store is still rebuilt from
  the chunks archived so far; the other intervals run, and the command exits 1. A busy lock is a
  refusal (exit 2) at once.
- ``status``: every built store (range, gaps, missing candles, OHLC violations, its CSV's size
  and sha256 against its meta; a store whose source a run is writing is listed from its meta,
  its CSV not checked), the saved Binance archive plans, the temporary files of killed runs and
  the warnings of the BingX real-market archive (1m and 5m retention, the recorded revisions,
  chunks kept incomplete or left behind). Exit 1 when a store does not match its meta.
- ``coverage [--record-config]``: study 1, ``reports/data/coverage``.

The network clients come from :func:`_binance_client` and :func:`_bingx_client` and the clock
from :func:`_utc_today` and :func:`_now_ms`, so that tests replace them; nothing here imports
numpy, pandas or httpx before :func:`run`.
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..data.partition import SealedPartitionError
from ..data.timeutil import utc_text
from . import (
    EXIT_FAILURE,
    EXIT_OK,
    data_root,
    display_path,
    positive_int,
    report_directory,
    run_guarded,
)

if TYPE_CHECKING:
    from ..data.binance_vision import BinanceVisionClient
    from ..exchange.public_market import PublicMarketClient

BINANCE_INTERVALS: tuple[str, ...] = ("1m", "1h")
BINGX_INTERVALS: tuple[str, ...] = ("1h", "15m", "5m", "1m")
STORE_INTERVALS: tuple[str, ...] = ("1m", "5m", "15m", "1h")
MAX_LISTED = 10
_PERIOD_RE = re.compile(r"([0-9]{4})-([0-9]{2})(?:-([0-9]{2}))?")


def refresh_period(text: str) -> str:
    """argparse type of a ``--refresh`` period: an existing month ``YYYY-MM`` or day
    ``YYYY-MM-DD``."""
    match = _PERIOD_RE.fullmatch(text)
    valid = match is not None
    if match is not None:
        year, month, day = match.groups()
        try:
            date(int(year), int(month), int(day or 1))
        except ValueError:
            valid = False
    if not valid:
        raise argparse.ArgumentTypeError(f"a period is YYYY-MM or YYYY-MM-DD, got {text!r}")
    return text


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "data",
        help="candle stores: Binance archives, BingX real-market backfill, status, coverage",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    binance = actions.add_parser(
        "binance",
        help="download and verify the Binance archives, build the stores, run the resample check",
    )
    binance.add_argument(
        "--interval",
        action="append",
        choices=BINANCE_INTERVALS,
        help="archive interval to sync and build; repeatable (default: 1m and 1h)",
    )
    binance.add_argument(
        "--refresh",
        action="append",
        default=[],
        type=refresh_period,
        metavar="PERIOD",
        help="download this republished period again (YYYY-MM or YYYY-MM-DD); repeatable",
    )
    binance.add_argument(
        "--record-config",
        action="store_true",
        help="only record the resample check's definition (no download)",
    )
    bingx = actions.add_parser(
        "bingx",
        help="backfill the BingX real-market chunk archive and rebuild candles/bingx/<interval>",
    )
    bingx.add_argument(
        "--interval",
        action="append",
        required=True,
        choices=BINGX_INTERVALS,
        help="candle interval; repeatable",
    )
    bingx.add_argument(
        "--max-chunks",
        type=positive_int,
        default=None,
        metavar="N",
        help="fetch at most N chunks per interval in this run (if the bound stops the walk, the "
        "history start stays open)",
    )
    actions.add_parser(
        "status",
        help="stores (size and sha256 against their meta), gaps, archive plans, temporary files "
        "of killed runs and BingX archive warnings",
    )
    coverage = actions.add_parser("coverage", help="study 1: write reports/data/coverage")
    coverage.add_argument(
        "--record-config",
        action="store_true",
        help="only record the study's definition (reports/data/coverage.config.json)",
    )


def run(args: argparse.Namespace) -> int:
    import httpx

    from ..data.binance_vision import BinanceVisionError, DownloadBudgetExceeded
    from ..data.bingx_history import BackfillError
    from ..data.coverage import CoverageError
    from ..data.fsutil import LockBusyError
    from ..data.resample import ResampleError
    from ..data.store import StoreError
    from ..exchange.public_market import PublicMarketError
    from ..report.writer import ConfigChangedError, ReportError

    handlers = {"binance": _binance, "bingx": _bingx, "status": _status, "coverage": _coverage}
    return run_guarded(
        lambda: handlers[args.action](args),
        refusals=(LockBusyError, DownloadBudgetExceeded, ConfigChangedError, SealedPartitionError),
        failures=(
            BinanceVisionError,
            BackfillError,
            PublicMarketError,
            httpx.HTTPError,
            StoreError,
            ResampleError,
            CoverageError,
            ReportError,
        ),
    )


# -- injectable clients and clock ------------------------------------------------------------------


def _binance_client() -> BinanceVisionClient:
    from ..data.binance_vision import BinanceVisionClient

    return BinanceVisionClient()


def _bingx_client() -> PublicMarketClient:
    from ..exchange.public_market import PublicMarketClient

    return PublicMarketClient()


def _utc_today() -> date:
    return datetime.now(UTC).date()


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


# -- printing --------------------------------------------------------------------------------------


def _say(message: str) -> None:
    print(message, flush=True)


def _warn(message: str) -> None:
    _say(f"WARNING: {message}")


def _megabytes(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _listed(items: Sequence[Any]) -> str:
    shown = ", ".join(str(item) for item in items[:MAX_LISTED])
    more = f" and {len(items) - MAX_LISTED} more" if len(items) > MAX_LISTED else ""
    return shown + more


def _print_meta(meta: Mapping[str, Any]) -> None:
    """One line per store meta: rows, range, gaps, OHLC violations (and incomplete buckets or
    chunks and the BingX history start when the provenance names them)."""
    provenance = meta.get("provenance") or {}
    extra = ""
    if "incomplete_bucket_count" in provenance:
        extra = f", {provenance['incomplete_bucket_count']} incomplete buckets (gaps)"
    if provenance.get("incomplete_chunks"):
        extra = f", {len(provenance['incomplete_chunks'])} incomplete chunks (refetched next run)"
    history = provenance.get("history_start")
    if isinstance(history, Mapping):
        if history.get("established"):
            extra += f"; history start {history.get('history_start_utc')} (established)"
        else:
            extra += f"; history start not established ({history.get('reason')})"
    _say(
        f"{meta.get('csv_relpath')}: {meta.get('rows')} rows, {meta.get('first_open_utc')} .. "
        f"{meta.get('last_open_utc')}, {len(meta.get('gaps') or [])} gaps "
        f"({meta.get('missing_candles')} missing candles), {meta.get('ohlc_violations')} OHLC "
        f"violations{extra}"
    )


def _print_removed(removed: Sequence[Mapping[str, Any]]) -> None:
    if removed:
        total = sum(int(item.get("bytes") or 0) for item in removed)
        _say(f"removed {len(removed)} temporary files of killed runs ({_megabytes(total)})")


def _print_sync(result: Mapping[str, Any]) -> None:
    _print_removed(result.get("removed_temporaries") or [])
    downloaded = result.get("downloaded") or []
    _say(
        f"binance {result['interval']}: {result['archives']} archives planned "
        f"({result['present']} present, {len(downloaded)} downloaded, "
        f"{_megabytes(result.get('downloaded_bytes', 0))}); "
        f"{len(result.get('monthly_missing') or [])} months and "
        f"{len(result.get('daily_missing') or [])} days not published; "
        f"{len(result.get('republished') or [])} republished; plan {result.get('plan_relpath')}"
    )
    if result.get("refresh_unused"):
        _say(
            f"NOTE: --refresh periods not used for {result['interval']}: {result['refresh_unused']}"
        )


def _print_backfill(summary: Mapping[str, Any]) -> None:
    _print_removed(summary.get("removed_temporaries") or [])
    history = summary.get("history_start_utc") or "not established in this run"
    _say(
        f"bingx {summary['interval']}: server time {summary['server_time_utc']}; "
        f"{summary['chunks_fetched']} chunks fetched ({summary['klines_requests']} requests, "
        f"{summary['mismatch_retries']} mismatch retries), {summary['chunks_skipped']} complete "
        f"chunks skipped; {summary['new_candles']} new candles; stop: {summary['stop_reason']}; "
        f"history start {history}"
    )
    kept = summary.get("kept_from_archive") or 0
    if kept:
        _say(
            f"{kept} archived candles the exchange no longer returns were kept from the archive "
            "(sidecar archive_only)"
        )
    inside = summary.get("kept_inside_served_range") or 0
    if inside > 0:
        _say(
            f"WARNING: {inside} of them lie inside the range the exchange still serves (not "
            "retention): their chunk stays incomplete and the next run fetches it again; report "
            "them to the owner"
        )
    gaps = summary.get("gap_chunks") or []
    if gaps:
        _say(
            f"WARNING: {len(gaps)} empty chunks between archived ones (gaps in the store): "
            f"{_listed([utc_text(start) for start in gaps])}"
        )


# -- actions ---------------------------------------------------------------------------------------


def _binance(args: argparse.Namespace) -> int:
    from ..data import binance_vision as bv
    from ..data.store import StoreId, StoreMissingError, read_meta
    from ..report.writer import record_config, write_report

    root, directory = data_root(), report_directory()
    stem = bv.RESAMPLE_CHECK_STEM
    digest = record_config(stem, bv.resample_check_config(), directory=directory)
    config_path = display_path(directory / f"{stem}.config.json")
    if args.record_config:
        _say(f"recorded {config_path} (sha256 {digest})")
        return EXIT_OK
    _say(f"resample check config {config_path} (sha256 {digest[:12]}) verified")
    requested = args.interval or BINANCE_INTERVALS
    intervals = [interval for interval in BINANCE_INTERVALS if interval in requested]
    today = _utc_today()
    _say(f"data root {display_path(root)}; plans up to yesterday (UTC today {today.isoformat()})")
    with _binance_client() as client:
        results = bv.sync_intervals(
            client, intervals, today=today, root=root, refresh=args.refresh, progress=_say
        )
    for result in results:
        _print_sync(result)
    if "1m" in intervals:
        metas = bv.build_binance_stores(root=root)
        for interval in STORE_INTERVALS:
            _print_meta(metas[interval])
    if "1h" in intervals:
        _print_meta(bv.build_official_1h(root=root))
    try:
        for source in ("binance", "binance_official"):
            read_meta(StoreId(source, "1h"), root)
    except StoreMissingError as exc:
        _say(f"resample check not run: {exc}")
        return EXIT_OK
    report = bv.check_resample(root=root, directory=directory)
    json_path, markdown_path = write_report(
        stem, report, bv.render_resample_markdown(report), directory=directory
    )
    comparison = report["comparison"]
    _say(
        f"resample check: {comparison['hours_compared']} hours compared; "
        f"{comparison['exact_mismatches']} OHLC, volume or count mismatches; "
        f"{comparison['tolerance_mismatches']} quote-volume mismatches beyond the rounding "
        "tolerance"
    )
    if report.get("owner_attention"):
        _say("ATTENTION: the resample check found mismatches; report them to the owner")
    _say(f"wrote {display_path(json_path)} and {display_path(markdown_path)}")
    return EXIT_OK


def _has_archived_chunks(root: Path, interval: str) -> bool:
    """Whether ``raw/bingx/<interval>`` holds a chunk sidecar (a merge has something to build)."""
    from ..data.bingx_history import ARCHIVE_RELDIR

    directory = root / ARCHIVE_RELDIR / interval
    return directory.is_dir() and any(directory.glob("*.json"))


def _bingx(args: argparse.Namespace) -> int:
    """Each interval in turn; a failed backfill (an exchange error, fetch pairs that never
    matched, an inconsistent archive) still rebuilds the store from the chunks archived so far,
    and the remaining intervals still run. Revised candles are a warning, not a failure: the
    archive keeps their first-seen rows, and each chunk's revisions are printed as soon as the
    backfill has recorded them, so also when a later chunk of the walk fails. A busy lock
    refuses the whole command."""
    import httpx

    from ..data import bingx_history as bh
    from ..data.fsutil import LockBusyError
    from ..data.store import StoreError
    from ..exchange.public_market import PublicMarketError

    failures = (bh.BackfillError, PublicMarketError, httpx.HTTPError, StoreError)
    root = data_root()
    _say(f"data root {display_path(root)}")
    failed: list[str] = []
    with _bingx_client() as client:
        for interval in dict.fromkeys(args.interval):
            try:
                summary = bh.backfill(
                    client, interval, root=root, max_chunks=args.max_chunks, warn=_warn
                )
            except LockBusyError:
                raise
            except failures as exc:
                _say(f"ERROR: {type(exc).__name__}: {exc}")
                failed.append(interval)
                if not _has_archived_chunks(root, interval):
                    continue
                _say(f"bingx {interval}: rebuilding the store from the chunks archived so far")
            else:
                _print_backfill(summary)
            try:
                _print_meta(bh.merge_chunks(interval, root=root))
            except LockBusyError:
                raise
            except failures as exc:
                _say(f"ERROR: {type(exc).__name__}: {exc}")
                if interval not in failed:
                    failed.append(interval)
    if failed:
        _say(
            f"bingx: {len(failed)} of {len(dict.fromkeys(args.interval))} intervals failed: "
            f"{', '.join(failed)}"
        )
        return EXIT_FAILURE
    return EXIT_OK


def _print_temporaries(leftovers: Sequence[Mapping[str, Any]]) -> None:
    """Temporary files of killed runs (a warning) and of runs writing now (a note)."""
    killed = [item for item in leftovers if not item["running"]]
    if killed:
        total = sum(int(item["bytes"]) for item in killed)
        _say(
            f"WARNING: {len(killed)} temporary files of killed runs ({_megabytes(total)}) under "
            f"candles/ and raw/: {_listed([item['relpath'] for item in killed])}; the next "
            "`kronos-agent data binance` or `data bingx` run removes them"
        )
    running: dict[str, int] = {}
    for item in leftovers:
        if item["running"]:
            running[item["lock"]] = running.get(item["lock"], 0) + 1
    for lock, count in running.items():
        _say(
            f"NOTE: {count} temporary files under directories a run is writing now ({lock}): "
            "they may be its own"
        )


def _status(args: argparse.Namespace) -> int:
    from ..data import binance_vision as bv
    from ..data.bingx_history import retention_warnings
    from ..data.store import StoreId, leftover_temporaries, read_meta, store_status

    root = data_root()
    now_ms = _now_ms()
    _say(f"data root {display_path(root)}; now {utc_text(now_ms)}")
    problems: list[str] = []
    rows = store_status(root)
    if not rows:
        _say("no store is built yet")
    for row in rows:
        relpath = row["relpath"]
        if "error" in row:
            problems.append(f"{relpath}: {row['error']}")
            continue
        _say(
            f"{relpath}: {row['rows']} rows, {row['first']} .. {row['last']}, "
            f"{row['gap_count']} gaps ({row['missing_candles']} missing candles), "
            f"{row['ohlc_violations']} OHLC violations, built {row['created_at']}"
        )
        store_id = StoreId(row["source"], row["interval"])
        if row["source"] == "bingx":
            provenance = read_meta(store_id, root).get("provenance")
            incomplete = (provenance or {}).get("incomplete_chunks") or []
            if incomplete:
                _say(
                    f"  {len(incomplete)} incomplete chunks (refreshed by the next backfill; "
                    "archived candles are kept)"
                )
        if row.get("busy"):
            _say(f"  a run holds {store_id.lock_relpath}: its CSV was not checked")
            continue
        if not row["csv_present"]:
            problems.append(f"{relpath} is missing although its meta exists")
        elif not row["size_matches"]:
            problems.append(f"{relpath}: its size differs from its meta; rebuild the store")
        elif row["sha256_matches"] is False:
            problems.append(f"{relpath}: its sha256 differs from its meta; rebuild the store")
    for interval in BINANCE_INTERVALS:
        relpath = bv.plan_relpath(interval)
        if not (root / relpath).exists():
            continue
        try:
            plan = bv.load_plan(interval, root)
        except bv.BinanceVisionError as exc:
            problems.append(str(exc))
            continue
        _say(
            f"{relpath}: {len(plan['archives'])} archives as of {plan.get('today')}; "
            f"{len(plan['monthly_missing'])} months and {len(plan['daily_missing'])} days not "
            "published"
        )
    for warning in retention_warnings(root, now_ms=now_ms):
        _say(f"WARNING: {warning}")
    _print_temporaries(leftover_temporaries(root))
    for problem in problems:
        _say(f"PROBLEM: {problem}")
    return EXIT_FAILURE if problems else EXIT_OK


def _coverage(args: argparse.Namespace) -> int:
    from ..data import coverage
    from ..report.writer import record_config

    directory = report_directory()
    digest = record_config(coverage.STEM, coverage.study_config(), directory=directory)
    if args.record_config:
        _say(
            f"recorded {display_path(directory / f'{coverage.STEM}.config.json')} (sha256 {digest})"
        )
        return EXIT_OK
    payload = coverage.coverage_report(root=data_root(), now_ms=_now_ms(), directory=directory)
    for relpath, table in (payload.get("usable_decision_hours") or {}).items():
        if table is None:
            _say(f"{relpath}: not built")
            continue
        cells = []
        for window, entry in table.items():
            validation = (entry.get("segments") or {}).get("validation") or {}
            cells.append(f"{window} {validation.get('usable')}/{validation.get('eligible')}")
        _say(f"{relpath}: usable validation decision hours: {', '.join(cells)}")
    for warning in payload.get("retention_warnings") or []:
        _say(f"WARNING: {warning}")
    json_path = directory / f"{coverage.STEM}.json"
    _say(f"wrote {display_path(json_path)} and {display_path(json_path.with_suffix('.md'))}")
    return EXIT_OK
