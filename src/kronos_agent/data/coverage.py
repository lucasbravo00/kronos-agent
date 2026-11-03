"""Coverage of the candle stores (``docs/DATA.md``, study 1; SPEC section 5).

What the data can support before any study runs: for every built store, its first and last
candle with the date they were observed, its gaps; and, for the stores a decision can read (BingX
real market 1h, 15m, 5m and 1m; Binance 1m and 1h), how many decision hours of each segment of
the partition are eligible and how many have a complete window for the 512 h Kronos context. It
answers SPEC section 5's "verify how far back 1m and 5m go" (the BingX real market keeps them on
a rolling retention, so the first candle depends on the date it was observed).

Rules, and why:

- **Usable.** A decision hour ``D`` is usable for a window of ``H`` hours at an interval when
  every candle of that interval opening in ``[D - H h, D)`` is present; the candle opening at
  ``D`` is never needed, and a missing candle is never filled. The same rule as the decision path
  (:mod:`kronos_agent.data.windows`).
- **The sealed segment stays closed.** Open times are read through
  :func:`~kronos_agent.data.store.read_study_frame` only up to the sealed start (in calendar-year
  ranges, so that a 1m store is never held whole), and usable hours are counted for the decision
  hours before it. Ranges, counts and gaps at or after the sealed start come from the store meta:
  open times and counts only, never a value (``docs/DATA.md``, "Reports").
- **Observed.** A store's meta records when it was built; a BingX store's provenance records the
  server time at which each chunk was fetched, which dates its first and last candle; the
  history depth observed is the first candle's age at the fetch of its chunk, given only when
  the provenance says that a backfill established the start of the exchange's history at that
  candle (:func:`~kronos_agent.data.bingx_history.read_history`). A run that its
  ``--max-chunks`` bound stops leaves the start open: the report says so instead of presenting
  the archive's depth as the market's.
- **One version per store.** Each store's meta and reads are taken under its source's job lock
  held shared (:func:`~kronos_agent.data.store.store_read_lock`), so a rebuild cannot slip
  between them; a run that holds the lock makes the study refuse.

The definition is recorded in ``reports/data/coverage.config.json`` before the first run (SPEC
rule 7). These are counts of available data: no interval, no test budget, no outcome.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..report.writer import record_config, report_header, write_report
from .bingx_history import RETENTION_DAYS, retention_warnings
from .partition import BEFORE_PARTITION, PARTITION, SEALED_START_MS, TRAINING_START
from .store import (
    INTERVALS,
    SOURCE_SYMBOLS,
    StoreId,
    StoreMissingError,
    find_gaps,
    read_meta,
    read_study_frame,
    store_read_lock,
)
from .timeutil import HOUR_MS, from_ms, to_ms, utc_text

STEM = "coverage"
DAY_MS = 86_400_000
DECISION_STORES: tuple[tuple[str, str], ...] = (
    ("bingx", "1h"),
    ("bingx", "15m"),
    ("bingx", "5m"),
    ("bingx", "1m"),
    ("binance", "1m"),
    ("binance", "1h"),
)
WINDOWS: tuple[tuple[str, int], ...] = (("context_512h", 512),)
SEALED_SEGMENT = "sealed"
MARKDOWN_RANGES = 5


class CoverageError(ValueError):
    """A store whose meta does not describe its candles, or an invalid argument."""


def _non_negative_ms(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise CoverageError(f"{name} must be a non-negative integer of epoch milliseconds")
    return int(value)


def usable_hours(
    open_ms: np.ndarray, *, step_ms: int, window_hours: int, decision_ms: np.ndarray
) -> np.ndarray:
    """For each decision hour, whether every candle opening in ``[D - H h, D)`` is present.

    ``open_ms`` is the strictly increasing, step-aligned list of present opens; the window needs
    ``window_hours x 1 h / step`` candles.
    """
    opens = np.asarray(open_ms, dtype=np.int64)
    decisions = np.asarray(decision_ms, dtype=np.int64)
    if opens.size and ((np.diff(opens) <= 0).any() or (opens % step_ms).any()):
        raise CoverageError("open_ms must be strictly increasing multiples of the step")
    lo = np.searchsorted(opens, decisions - window_hours * HOUR_MS, side="left")
    hi = np.searchsorted(opens, decisions, side="left")
    return (hi - lo) == window_hours * HOUR_MS // step_ms


def _ranges(values_ms: Iterable[int], step_ms: int) -> list[list[str]]:
    """Merged ``[start, end)`` UTC text ranges of sorted times on a ``step_ms`` grid."""
    merged: list[list[int]] = []
    for value in values_ms:
        if merged and merged[-1][1] == value:
            merged[-1][1] = value + step_ms
        else:
            merged.append([value, value + step_ms])
    return [[utc_text(a), utc_text(b)] for a, b in merged]


def _next_year_ms(ms: int) -> int:
    return to_ms(datetime(from_ms(ms).year + 1, 1, 1, tzinfo=UTC))


def _open_times(store_id: StoreId, meta: Mapping[str, Any], root: Path | None) -> np.ndarray:
    """Open times of the store's candles before the sealed start, read year by year through
    ``read_study_frame`` (only ``open_ms``; the CSV is verified against the meta on every read)."""
    first, last = meta.get("first_open_ms"), meta.get("last_open_ms")
    if first is None or last is None:
        return np.zeros(0, dtype=np.int64)
    end = min(SEALED_START_MS, int(last) + store_id.step_ms)
    parts: list[np.ndarray] = []
    start = int(first)
    while start < end:
        stop = min(end, _next_year_ms(start))
        frame = read_study_frame(
            store_id, start_ms=start, end_ms=stop, numeric=True, usecols=(), root=root
        )
        parts.append(frame["open_ms"].to_numpy(dtype=np.int64))
        start = stop
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def _segments(first_open_ms: int | None) -> list[tuple[str, int, int]]:
    """``(name, start, end)`` of the decision hours counted: the partition before the sealed
    start, preceded by ``before_training`` from the hour of the store's first candle."""
    segments: list[tuple[str, int, int]] = []
    training_ms = to_ms(TRAINING_START)
    if first_open_ms is not None and first_open_ms < training_ms:
        segments.append((BEFORE_PARTITION, first_open_ms // HOUR_MS * HOUR_MS, training_ms))
    else:
        segments.append((BEFORE_PARTITION, training_ms, training_ms))
    for segment in PARTITION:
        if segment.name == SEALED_SEGMENT:
            continue
        end = SEALED_START_MS if segment.end_ms is None else min(segment.end_ms, SEALED_START_MS)
        segments.append((segment.name, segment.start_ms, end))
    return segments


def _usable_table(
    opens: np.ndarray, step_ms: int, first_open_ms: int | None
) -> dict[str, dict[str, dict[str, Any]]]:
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for window_name, window_hours in WINDOWS:
        per_segment: dict[str, dict[str, Any]] = {}
        for name, start, end in _segments(first_open_ms):
            decisions = np.arange(start, end, HOUR_MS, dtype=np.int64)
            flags = usable_hours(
                opens, step_ms=step_ms, window_hours=window_hours, decision_ms=decisions
            )
            usable = decisions[flags]
            per_segment[name] = {
                "eligible": int(decisions.size),
                "usable": int(usable.size),
                "share": float(usable.size / decisions.size) if decisions.size else None,
                "first_usable": utc_text(int(usable[0])) if usable.size else None,
                "last_usable": utc_text(int(usable[-1])) if usable.size else None,
                "unusable_ranges": _ranges(decisions[~flags].tolist(), HOUR_MS),
            }
        per_segment[SEALED_SEGMENT] = {
            "eligible": None,
            "usable": None,
            "reason": "not counted: decision hours at or after the sealed start stay closed "
            "(gate 6)",
        }
        table[window_name] = {"window_hours": window_hours, "segments": per_segment}
    return table


def _history_reason(provenance: Mapping[str, Any], first_open_ms: int) -> str | None:
    """Why the store's first candle is not an established history start (``None`` if it is)."""
    history = provenance.get("history_start")
    if not isinstance(history, Mapping):
        return "the store's provenance does not say whether a backfill established it"
    if history.get("established") is True and history.get("history_start_ms") == first_open_ms:
        return None
    reason = history.get("reason")
    return reason if isinstance(reason, str) else "the start was not established"


def _observed(meta: Mapping[str, Any]) -> dict[str, Any]:
    """When the store's candles were observed: the build time and, for BingX chunks, the server
    time of the fetch of the chunks holding the first and the last candle, and the history depth
    observed when a backfill established that the first candle is the start of the exchange's
    history (``None`` otherwise, with the reason)."""
    provenance = meta.get("provenance") or {}
    observed: dict[str, Any] = {
        "store_built_at": meta.get("created_at"),
        "first_candle_observed_at": None,
        "last_candle_observed_at": None,
        "history_start_established": None,
        "history_days_observed": None,
        "history_days_observed_reason": None,
        "plan_date": None,
    }
    chunks = provenance.get("chunks") if isinstance(provenance, Mapping) else None
    if isinstance(chunks, list) and chunks and meta.get("first_open_ms") is not None:
        valid = [
            chunk
            for chunk in chunks
            if isinstance(chunk, Mapping)
            and isinstance(chunk.get("server_time_ms"), int)
            and isinstance(chunk.get("first_open_ms"), int)
            and isinstance(chunk.get("last_open_ms"), int)
        ]
        if valid:
            oldest = min(valid, key=lambda chunk: chunk["first_open_ms"])
            newest = max(valid, key=lambda chunk: chunk["last_open_ms"])
            observed["first_candle_observed_at"] = utc_text(oldest["server_time_ms"])
            observed["last_candle_observed_at"] = utc_text(newest["server_time_ms"])
            first = int(meta["first_open_ms"])
            reason = _history_reason(provenance, first)
            observed["history_start_established"] = reason is None
            if reason is not None:
                observed["history_days_observed_reason"] = (
                    f"history start not established: {reason}; the first open is the oldest "
                    "archived candle, so the depth and the usable hours are lower bounds"
                )
            elif oldest.get("archive_only_rows"):
                observed["history_days_observed_reason"] = (
                    "the oldest chunk holds candles that its last fetch no longer returned, so "
                    "that fetch does not date the first candle"
                )
            else:
                observed["history_days_observed"] = (oldest["server_time_ms"] - first) / DAY_MS
    plan = provenance.get("plan") if isinstance(provenance, Mapping) else None
    if isinstance(plan, Mapping):
        observed["plan_date"] = plan.get("today")
    return observed


def _store_row(
    store_id: StoreId, meta: Mapping[str, Any], *, now_ms: int, opens: np.ndarray | None
) -> dict[str, Any]:
    step = store_id.step_ms
    gaps = [[int(a), int(b)] for a, b in meta.get("gaps") or []]
    last = meta.get("last_open_ms")
    row: dict[str, Any] = {
        "source": store_id.source,
        "interval": store_id.interval,
        "relpath": store_id.relpath,
        "built": True,
        "rows": meta.get("rows"),
        "first_open": meta.get("first_open_utc"),
        "last_open": meta.get("last_open_utc"),
        "first_open_ms": meta.get("first_open_ms"),
        "last_open_ms": last,
        "csv_sha256": meta.get("csv_sha256"),
        "gap_count": len(gaps),
        "missing_candles": sum((b - a) // step for a, b in gaps),
        "gaps": [[utc_text(a), utc_text(b), (b - a) // step] for a, b in gaps],
        "ohlc_violations": meta.get("ohlc_violations"),
        "observed": _observed(meta),
        "last_candle_age_days": None if last is None else (now_ms - int(last)) / DAY_MS,
        "incomplete_chunks": list((meta.get("provenance") or {}).get("incomplete_chunks") or []),
    }
    if opens is not None:
        row["rows_before_sealed_start"] = int(opens.size)
        row["rows_at_or_after_sealed_start"] = int(meta.get("rows") or 0) - int(opens.size)
    return row


def _check_against_meta(store_id: StoreId, meta: Mapping[str, Any], opens: np.ndarray) -> None:
    """The open times read must be the ones the meta describes before the sealed start."""
    if not opens.size:
        return
    first = meta.get("first_open_ms")
    if first is not None and int(first) < SEALED_START_MS and int(opens[0]) != int(first):
        raise CoverageError(f"{store_id.relpath}: the first candle read is not the meta's")
    read_gaps = find_gaps(opens, store_id.step_ms)
    last_read = int(opens[-1])
    meta_gaps = [[int(a), int(b)] for a, b in meta.get("gaps") or [] if int(b) <= last_read]
    if read_gaps != meta_gaps:
        raise CoverageError(
            f"{store_id.relpath}: the gaps of the candles read differ from the meta's before the "
            "sealed start; rebuild the store"
        )


def study_config() -> dict[str, Any]:
    """The recorded definition of ``reports/data/coverage`` (SPEC rule 7)."""
    return {
        "study": STEM,
        "title": "Coverage of the candle stores",
        "design": "docs/DATA.md, Studies and reports, study 1",
        "stores_listed": "every built store (source x interval) from its meta: rows, first and "
        "last open (UTC), the date observed, gaps as [first_missing, end) ranges",
        "usability_stores": [StoreId(s, i).relpath for s, i in DECISION_STORES],
        "windows": {name: hours for name, hours in WINDOWS},
        "usable_rule": "a decision hour D is usable for a window of H hours at an interval when "
        "every candle of that interval opening in [D - H h, D) is present; the candle opening at "
        "D is not needed; missing candles are never filled",
        "segments": {
            BEFORE_PARTITION: "from the hour of the store's first candle to the training start",
            **{
                segment.name: [
                    utc_text(segment.start_ms),
                    None if segment.end_ms is None else utc_text(segment.end_ms),
                ]
                for segment in PARTITION
            },
        },
        "eligible": "every decision hour of the segment (before the sealed start)",
        "sealed": "decision hours at or after the sealed start are not counted (gate 6); open "
        "times and counts at or after it come from the store meta, never a value",
        "reads": "open times only, through read_study_frame up to the sealed start, in "
        "calendar-year ranges, with the meta under the store's source lock held shared; each "
        "read verifies the CSV sha256 against the meta, and the gaps read must equal the meta's",
        "observed": "the store's build time (meta created_at); for BingX real-market stores the "
        "server time of the fetch of the chunks holding the first and the last candle; history "
        "depth observed = the first candle's age at its chunk's fetch, only when the store's "
        "provenance records that a backfill established the start of the exchange's history at "
        "that candle and the last fetch of its chunk kept no candle from the archive alone; "
        "otherwise null with the reason (a run that its --max-chunks bound stops leaves the "
        "history start open)",
        "retention": {
            "observed_days": dict(RETENTION_DAYS),
            "warning": "the newest archived 1m or 5m chunk is older than a third of its "
            "retention (kronos_agent.data.bingx_history.retention_warnings)",
        },
        "statistics": [],
        "comparators": {
            "usable": "eligible: every decision hour of the segment",
            "first_open": "the date observed",
            "history_days_observed": "the observed retention of the real market",
        },
        "tests_consumed": 0,
    }


def coverage_report(
    *, root: Path | None = None, now_ms: int, directory: Path | None = None
) -> dict[str, Any]:
    """Run study 1 and write ``reports/data/coverage.json`` and ``.md``; return the payload.

    The definition (:func:`study_config`) is recorded, or verified, under ``directory`` (default
    ``reports/data``) before any candle is read. Every built store of ``candles/`` under ``root``
    is listed from its meta, and the :data:`DECISION_STORES` always (absent ones as not built);
    these also get their open times read up to the sealed start and the usable decision hours
    per segment for each of :data:`WINDOWS`. ``now_ms`` (the time of the report) dates the age
    of the last candle and the retention warnings of the BingX real-market archive.
    """
    now = _non_negative_ms(now_ms, "now_ms")
    config = study_config()
    record_config(STEM, config, directory=directory)
    stores: list[dict[str, Any]] = []
    usable: dict[str, Any] = {StoreId(s, i).relpath: None for s, i in DECISION_STORES}
    for source in SOURCE_SYMBOLS:
        for interval in INTERVALS:
            store_id = StoreId(source, interval)
            decision_store = (source, interval) in DECISION_STORES
            with store_read_lock(store_id, root):  # one version: no rebuild between the reads
                try:
                    meta = read_meta(store_id, root)
                except StoreMissingError:
                    if decision_store:  # the stores a decision reads are listed even when absent
                        stores.append(
                            {
                                "source": source,
                                "interval": interval,
                                "relpath": store_id.relpath,
                                "built": False,
                            }
                        )
                    continue
                opens: np.ndarray | None = None
                if decision_store:
                    opens = _open_times(store_id, meta, root)
                    _check_against_meta(store_id, meta, opens)
                    usable[store_id.relpath] = _usable_table(
                        opens, store_id.step_ms, meta.get("first_open_ms")
                    )
                stores.append(_store_row(store_id, meta, now_ms=now, opens=opens))
    warnings = retention_warnings(root, now_ms=now)
    body = {
        "now": utc_text(now),
        "sealed_start": utc_text(SEALED_START_MS),
        "stores": stores,
        "usable_decision_hours": usable,
        "retention_warnings": warnings,
    }
    samples = {
        relpath: (
            None
            if table is None
            else {
                window: {
                    name: {
                        key: item.get(key)
                        for key in ("eligible", "usable", "unusable_ranges", "reason")
                        if key in item
                    }
                    for name, item in entry["segments"].items()
                }
                for window, entry in table.items()
            }
        )
        for relpath, table in usable.items()
    }
    header = report_header(
        STEM,
        config,
        statistics_with_intervals=0,
        seeds={},
        samples=samples,
        comparators=config["comparators"],
        directory=directory,
    )
    payload = {**body, **header}
    write_report(STEM, payload, render_coverage_markdown(payload), directory=directory)
    return payload


# -- Markdown -------------------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _segment_names() -> list[str]:
    return [BEFORE_PARTITION, *(s.name for s in PARTITION if s.name != SEALED_SEGMENT)]


def _ranges_text(ranges: Sequence[Sequence[Any]], limit: int = MARKDOWN_RANGES) -> str:
    shown = ", ".join(f"[{item[0]}, {item[1]})" for item in ranges[:limit])
    if len(ranges) > limit:
        shown += f" and {len(ranges) - limit} more (JSON)"
    return shown or "none"


def render_coverage_markdown(report: Mapping[str, Any]) -> str:
    """Markdown of the coverage report: stores, usable decision hours, gaps, retention."""
    code = report.get("code") or {}
    lines = [
        "# Coverage of the candle stores",
        "",
        (
            "Study 1 of docs/DATA.md: the first and last candle of every store with the date it "
            "was observed, the gaps, and the decision hours usable per segment for the 512 h "
            "Kronos context. A decision hour D is usable when every candle of the interval "
            "opening in [D - 512 h, D) is present; hours at or after the sealed start "
            f"({report.get('sealed_start')}) are not counted."
        ),
        "",
        (
            f"- Study: `{report.get('study')}`; config sha256 `{report.get('config_sha256')}`; "
            f"docs/DATA.md sha256 `{report.get('data_md_sha256')}`"
        ),
        (
            f"- Code: commit `{code.get('commit')}`, dirty {_fmt(code.get('dirty'))}, source "
            f"tree sha256 `{code.get('source_tree_sha256')}`"
        ),
        f"- Generated at {report.get('generated_at')} (report time {report.get('now')})",
        (
            f"- Tests consumed: {_fmt(report.get('tests_consumed'))}; statistics with intervals: "
            f"{_fmt(report.get('statistics_with_intervals'))} (counts of available data)"
        ),
        "",
        "## Stores",
        "",
        (
            "| store | rows | first open | last open | gaps | missing candles | built at | first "
            "candle observed at | history days observed |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    notes: list[str] = []
    open_start: set[str] = set()
    for store in report.get("stores") or []:
        if not store.get("built"):
            lines.append(f"| `{store.get('relpath')}` | not built | | | | | | | |")
            continue
        observed = store.get("observed") or {}
        depth = _fmt(observed.get("history_days_observed"))
        if observed.get("history_start_established") is False:
            depth = "not established"
            open_start.add(str(store.get("relpath")))
        if observed.get("history_days_observed_reason"):
            notes.append(f"- `{store.get('relpath')}`: {observed['history_days_observed_reason']}")
        lines.append(
            f"| `{store.get('relpath')}` | {_fmt(store.get('rows'))} | "
            f"{_fmt(store.get('first_open'))} | {_fmt(store.get('last_open'))} | "
            f"{_fmt(store.get('gap_count'))} | {_fmt(store.get('missing_candles'))} | "
            f"{_fmt(observed.get('store_built_at'))} | "
            f"{_fmt(observed.get('first_candle_observed_at'))} | {depth} |"
        )
    if notes:
        lines += ["", *notes]
    segments = _segment_names()
    usable = report.get("usable_decision_hours") or {}
    lines += ["", "## Decision hours usable before the sealed start (usable / eligible)", ""]
    header = "| store | window | " + " | ".join(segments) + " |"
    lines += [header, "| --- | --- | " + " | ".join("---" for _ in segments) + " |"]
    for relpath, table in usable.items():
        if table is None:
            lines.append(f"| `{relpath}` | not built | " + " | ".join("" for _ in segments) + " |")
            continue
        # an archive whose start is not established as the market's gives lower bounds
        marker = " (lower bound: history start not established)" if relpath in open_start else ""
        for window, entry in table.items():
            cells = []
            for name in segments:
                item = entry["segments"].get(name) or {}
                cells.append(f"{_fmt(item.get('usable'))} / {_fmt(item.get('eligible'))}")
            lines.append(f"| `{relpath}`{marker} | {window} | " + " | ".join(cells) + " |")
    lines += ["", "## Gaps", ""]
    for store in report.get("stores") or []:
        if store.get("built"):
            gaps = store.get("gaps") or []
            lines.append(f"- `{store.get('relpath')}`: {_ranges_text(gaps)}")
    lines += ["", "## Unusable decision hours before the sealed start", ""]
    for relpath, table in usable.items():
        if table is None:
            continue
        for window, entry in table.items():
            for name in segments:
                item = (entry.get("segments") or {}).get(name) or {}
                if item.get("eligible"):
                    ranges = _ranges_text(item.get("unusable_ranges") or [])
                    lines.append(f"- `{relpath}` {window} {name}: {ranges}")
    lines += ["", "## Retention of the BingX real market", ""]
    warnings = report.get("retention_warnings") or []
    lines += [f"- {text}" for text in warnings] or ["- no warning"]
    return "\n".join(lines) + "\n"


__all__ = [
    "DECISION_STORES",
    "STEM",
    "WINDOWS",
    "CoverageError",
    "coverage_report",
    "render_coverage_markdown",
    "study_config",
    "usable_hours",
]
