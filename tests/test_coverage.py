"""Coverage of the candle stores (study 1): usable decision hours at the window bounds, every
segment against an independent count, the sealed start, the observed dates and the report
(temporary data and report roots, no network)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import step2_fakes as fakes

from kronos_agent.data import bingx_history as bh
from kronos_agent.data import coverage as cv
from kronos_agent.data.fsutil import exclusive_lock
from kronos_agent.data.partition import PARTITION, SEALED_START_MS, TRAINING_START
from kronos_agent.data.store import StoreBusyError, StoreId, read_meta
from kronos_agent.data.timeutil import HOUR_MS, MINUTE_MS, to_ms, utc_text
from kronos_agent.exchange.public_market import PUBLIC_HOST
from kronos_agent.hashing import canonical_json, sha256_file
from kronos_agent.paths import REPO_ROOT
from kronos_agent.report.writer import REQUIRED_KEYS, ConfigChangedError

DAY_MS = 86_400_000
EXCHANGE_DOMAIN = PUBLIC_HOST.split(".", 1)[1]  # no host literal outside exchange/
NOW_MS = fakes.utc_ms(2026, 11, 1, 12)
STEPS = {"1m": MINUTE_MS, "5m": 5 * MINUTE_MS, "15m": 15 * MINUTE_MS, "1h": HOUR_MS}


def grid(start_ms: int, end_ms: int, step_ms: int, missing=()) -> np.ndarray:
    opens = np.arange(start_ms, end_ms, step_ms, dtype=np.int64)
    return opens[~np.isin(opens, np.asarray(missing, dtype=np.int64))]


# -- the usable rule ------------------------------------------------------------------------------


def test_usable_hours_at_the_window_bounds():
    start = fakes.utc_ms(2025, 1, 6)
    gap = start + 500 * HOUR_MS
    opens = grid(start, start + 1000 * HOUR_MS, HOUR_MS, [gap])
    cases = {
        99: False,  # the window would start one hour before the first candle
        100: True,
        500: True,  # [D - 100h, D) ends right before the missing candle
        501: False,
        600: False,  # the missing candle opens exactly at D - 100h
        601: True,
        1000: True,  # the last candle opens at D - 1h
        1001: False,
    }
    decisions = np.array([start + k * HOUR_MS for k in cases], dtype=np.int64)
    flags = cv.usable_hours(opens, step_ms=HOUR_MS, window_hours=100, decision_ms=decisions)
    assert flags.tolist() == list(cases.values())
    context = cv.usable_hours(
        opens, step_ms=HOUR_MS, window_hours=512, decision_ms=decisions[[0, 1, 5]]
    )
    assert context.tolist() == [False, False, False]
    with pytest.raises(cv.CoverageError):
        cv.usable_hours(opens[::-1], step_ms=HOUR_MS, window_hours=100, decision_ms=decisions)


def test_usable_hours_of_a_finer_interval():
    start = fakes.utc_ms(2025, 3, 3)
    step = 5 * MINUTE_MS
    at_hour = start + 150 * HOUR_MS  # a 5m candle opening exactly on an hour
    late = start + 300 * HOUR_MS + 55 * MINUTE_MS  # the last 5m candle of an hour
    opens = grid(start, start + 600 * HOUR_MS, step, [at_hour, late])
    cases = {
        150: True,  # the 5m candle opening at D is not needed
        151: False,
        250: False,  # D - 100h is the missing candle
        251: True,
        300: True,
        301: False,  # the candle opening at D - 65 min is missing
        400: False,
        401: True,
    }
    decisions = np.array([start + k * HOUR_MS for k in cases], dtype=np.int64)
    flags = cv.usable_hours(opens, step_ms=step, window_hours=100, decision_ms=decisions)
    assert flags.tolist() == list(cases.values())


# -- the report -----------------------------------------------------------------------------------


def independent_usable(opens: np.ndarray, step: int, hours: int, decisions: np.ndarray) -> int:
    """Usable decision hours by prefix sums over the candle grid (not searchsorted)."""
    if not opens.size:
        return 0
    first = int(opens[0])
    slots = (int(opens[-1]) - first) // step + 1
    present = np.zeros(slots, dtype=np.int64)
    present[(opens - first) // step] = 1
    prefix = np.concatenate(([0], np.cumsum(present)))
    need = hours * HOUR_MS // step
    count = 0
    for decision in decisions.tolist():
        lo = (decision - hours * HOUR_MS - first) // step
        hi = (decision - first) // step
        if lo < 0 or hi > slots:
            continue
        count += int(prefix[hi] - prefix[lo] == need)
    return count


def segment_bounds(first_open: int) -> dict[str, tuple[int, int]]:
    training = to_ms(TRAINING_START)
    start = first_open // HOUR_MS * HOUR_MS if first_open < training else training
    bounds = {"before_training": (start, training)}
    for segment in PARTITION:
        if segment.name != "sealed":
            bounds[segment.name] = (segment.start_ms, min(segment.end_ms, SEALED_START_MS))
    return bounds


@pytest.fixture
def stores(tmp_path):
    """Stores under a temporary root: BingX 1h across the sealed start with a gap on each side,
    BingX 5m (also across the sealed start, longer than one 512 h context) and 1m with a missing
    candle, Binance 1h and 1m around the training start, an extra Binance 5m store; BingX 15m and
    every other store are not built."""
    root = tmp_path / "data"
    opens: dict[tuple[str, str], np.ndarray] = {}
    hour_start = fakes.utc_ms(2024, 4, 27, 10)
    count = (SEALED_START_MS + 60 * HOUR_MS - hour_start) // HOUR_MS
    before_gap = fakes.utc_ms(2025, 2, 3, 7)
    after_gap = SEALED_START_MS + 10 * HOUR_MS
    candles = [
        c
        for c in fakes.synthetic_candles(hour_start, count, seed=1)
        if c.open_ms not in (before_gap, after_gap, after_gap + HOUR_MS)
    ]
    fakes.write_bingx_store(root, candles)
    opens[("bingx", "1h")] = np.array([c.open_ms for c in candles], dtype=np.int64)
    for interval, start, days, missing in (
        ("5m", fakes.utc_ms(2026, 7, 1), 35, fakes.utc_ms(2026, 7, 25, 3, 20)),
        ("1m", fakes.utc_ms(2026, 7, 26), 5, fakes.utc_ms(2026, 7, 28, 14)),
    ):
        step = STEPS[interval]
        series = [
            c
            for c in fakes.synthetic_candles(start, days * DAY_MS // step, interval=interval)
            if c.open_ms != missing
        ]
        fakes.write_bingx_store(root, series, interval=interval)
        opens[("bingx", interval)] = np.array([c.open_ms for c in series], dtype=np.int64)
    for interval, start, hours, missing in (
        ("1h", fakes.utc_ms(2019, 12, 2, 5), 24 * 60, fakes.utc_ms(2020, 1, 10, 3)),
        ("1m", fakes.utc_ms(2019, 12, 30, 17, 3), 60 * 24 * 4, fakes.utc_ms(2020, 1, 1, 0, 30)),
        ("5m", fakes.utc_ms(2019, 12, 30), 12 * 24, None),
    ):
        step = STEPS[interval]
        frame = fakes.binance_frame(start, hours, interval=interval, seed=4)
        frame = frame[frame["open_ms"] != missing].reset_index(drop=True)
        fakes.write_binance_store(root, frame, interval=interval)
        opens[("binance", interval)] = frame["open_ms"].to_numpy(dtype=np.int64)
    return root, opens


def test_coverage_counts_every_segment_and_window(tmp_path, stores):
    root, opens = stores
    reports = tmp_path / "reports"
    payload = cv.coverage_report(root=root, now_ms=NOW_MS, directory=reports)
    usable = payload["usable_decision_hours"]
    assert list(usable) == [StoreId(s, i).relpath for s, i in cv.DECISION_STORES]
    assert usable["candles/bingx/15m.csv"] is None
    checked = 0
    for (source, interval), present in opens.items():
        relpath = StoreId(source, interval).relpath
        if (source, interval) not in cv.DECISION_STORES:
            assert relpath not in usable
            continue
        before = present[present < SEALED_START_MS]
        for window, hours in cv.WINDOWS:
            segments = usable[relpath][window]["segments"]
            assert usable[relpath][window]["window_hours"] == hours
            for name, (start, end) in segment_bounds(int(present[0])).items():
                decisions = np.arange(start, end, HOUR_MS, dtype=np.int64)
                expected = independent_usable(before, STEPS[interval], hours, decisions)
                assert segments[name]["eligible"] == decisions.size, (relpath, window, name)
                assert segments[name]["usable"] == expected, (relpath, window, name)
                checked += 1
            assert segments["sealed"]["usable"] is None
            assert "gate 6" in segments["sealed"]["reason"]
    # only the 512 h Kronos context: the profile windows and question 5 left with D-042
    assert cv.WINDOWS == (("context_512h", 512),)
    assert checked == 5 * 1 * 5
    assert all(table is None or list(table) == ["context_512h"] for table in usable.values())
    assert "question_5" not in payload
    hour = usable["candles/bingx/1h.csv"]["context_512h"]["segments"]["validation"]
    gap = fakes.utc_ms(2025, 2, 3, 7)
    assert hour["unusable_ranges"] == [[utc_text(gap + HOUR_MS), utc_text(gap + 513 * HOUR_MS)]]
    assert hour["usable"] == hour["eligible"] - 512
    assert hour["first_usable"] == "2024-07-01T00:00Z"
    early = usable["candles/binance/1h.csv"]["context_512h"]["segments"]["before_training"]
    assert early["eligible"] == (to_ms(TRAINING_START) - fakes.utc_ms(2019, 12, 2, 5)) // HOUR_MS
    # a finer interval: 5m contexts are complete from 2026-07-22T08:00Z until the missing candle
    five = usable["candles/bingx/5m.csv"]["context_512h"]["segments"]["embargo_2"]
    assert (five["usable"], five["first_usable"], five["last_usable"]) == (
        68,
        "2026-07-22T08:00Z",
        "2026-07-25T03:00Z",
    )


def test_coverage_store_rows_and_the_sealed_start(tmp_path, stores, monkeypatch):
    root, opens = stores
    calls = []
    real_read = cv.read_study_frame

    def recording_read(store_id, **kwargs):
        calls.append((store_id.relpath, kwargs["start_ms"], kwargs["end_ms"], kwargs["usecols"]))
        return real_read(store_id, **kwargs)

    monkeypatch.setattr(cv, "read_study_frame", recording_read)
    payload = cv.coverage_report(root=root, now_ms=NOW_MS, directory=tmp_path / "reports")
    assert all(end <= SEALED_START_MS for _, _, end, _ in calls)
    assert all(tuple(usecols) == () for _, _, _, usecols in calls)  # open times only
    hourly = [(start, end) for relpath, start, end, _ in calls if relpath == "candles/bingx/1h.csv"]
    assert hourly == [
        (fakes.utc_ms(2024, 4, 27, 10), fakes.utc_ms(2025, 1, 1)),
        (fakes.utc_ms(2025, 1, 1), fakes.utc_ms(2026, 1, 1)),
        (fakes.utc_ms(2026, 1, 1), SEALED_START_MS),
    ]
    rows = {row["relpath"]: row for row in payload["stores"]}
    assert set(rows) == {
        "candles/bingx/1h.csv",
        "candles/bingx/5m.csv",
        "candles/bingx/1m.csv",
        "candles/bingx/15m.csv",
        "candles/binance/1h.csv",
        "candles/binance/1m.csv",
        "candles/binance/5m.csv",
    }
    assert rows["candles/bingx/15m.csv"]["built"] is False
    assert rows["candles/binance/1m.csv"]["built"] is True
    hourly_row = rows["candles/bingx/1h.csv"]
    present = opens[("bingx", "1h")]
    assert hourly_row["rows"] == present.size
    assert hourly_row["rows_before_sealed_start"] == int((present < SEALED_START_MS).sum())
    assert hourly_row["rows_at_or_after_sealed_start"] == int((present >= SEALED_START_MS).sum())
    assert hourly_row["last_open"] == utc_text(int(present[-1]))  # an open time, never a value
    after = SEALED_START_MS + 10 * HOUR_MS
    assert hourly_row["gaps"] == [
        [utc_text(fakes.utc_ms(2025, 2, 3, 7)), utc_text(fakes.utc_ms(2025, 2, 3, 8)), 1],
        [utc_text(after), utc_text(after + 2 * HOUR_MS), 2],
    ]
    assert hourly_row["missing_candles"] == 3
    assert hourly_row["last_candle_age_days"] == pytest.approx((NOW_MS - int(present[-1])) / DAY_MS)
    assert "rows_before_sealed_start" not in rows["candles/binance/5m.csv"]
    five = rows["candles/bingx/5m.csv"]
    assert five["rows_before_sealed_start"] + five["rows_at_or_after_sealed_start"] == five["rows"]
    document = (tmp_path / "reports" / "coverage.json").read_text()
    markdown = (tmp_path / "reports" / "coverage.md").read_text()
    hour_start = fakes.utc_ms(2024, 4, 27, 10)
    count = (SEALED_START_MS + 60 * HOUR_MS - hour_start) // HOUR_MS  # as the fixture
    sealed_closes = {
        c.close
        for c in fakes.synthetic_candles(hour_start, count, seed=1)
        if c.open_ms >= SEALED_START_MS
    }
    assert len(sealed_closes) > 50
    for text in (document, markdown):
        assert not any(close in text for close in sealed_closes)
        assert str(tmp_path) not in text and EXCHANGE_DOMAIN not in text
    assert markdown.isascii()


def test_coverage_report_header_config_and_markdown(tmp_path, stores, monkeypatch):
    root, _ = stores
    reports = tmp_path / "reports"
    payload = cv.coverage_report(root=root, now_ms=NOW_MS, directory=reports)
    document = json.loads((reports / "coverage.json").read_text())
    assert set(REQUIRED_KEYS) <= set(document)
    assert document["study"] == "coverage" and document["tests_consumed"] == 0
    assert document["statistics_with_intervals"] == 0 and document["seeds"] == {}
    config_path = reports / "coverage.config.json"
    assert config_path.read_bytes() == canonical_json(cv.study_config())
    assert document["config_sha256"] == sha256_file(config_path)
    assert document["now"] == utc_text(NOW_MS)
    samples = document["samples"]["candles/bingx/1h.csv"]["context_512h"]["validation"]
    assert (
        samples["usable"]
        == payload["usable_decision_hours"]["candles/bingx/1h.csv"]["context_512h"]["segments"][
            "validation"
        ]["usable"]
    )
    assert document["samples"]["candles/bingx/15m.csv"] is None
    warnings = document["retention_warnings"]
    assert len(warnings) == 2 and all("no archived chunks" in text for text in warnings)
    markdown = (reports / "coverage.md").read_text()
    assert "| `candles/bingx/15m.csv` | not built |" in markdown
    assert "| `candles/bingx/1h.csv` | context_512h | " in markdown
    gap = fakes.utc_ms(2025, 2, 3, 7)
    assert "## Unusable decision hours before the sealed start" in markdown
    assert (
        f"- `candles/bingx/1h.csv` context_512h validation: "
        f"[{utc_text(gap + HOUR_MS)}, {utc_text(gap + 513 * HOUR_MS)})\n" in markdown
    )
    assert "no archived chunks" in markdown
    text = canonical_json(cv.study_config()).decode("ascii")
    assert str(REPO_ROOT) not in text and str(Path.home()) not in text
    # no volume-profile window and no question-5 table (D-042)
    for written in (markdown, (reports / "coverage.json").read_text(), text):
        assert "profile" not in written and "question" not in written.lower()
    config_path.write_bytes(canonical_json({"study": "coverage", "changed": True}))

    def no_read(*args, **kwargs):
        raise AssertionError("candles read before the config check")

    monkeypatch.setattr(cv, "read_study_frame", no_read)
    with pytest.raises(ConfigChangedError):
        cv.coverage_report(root=root, now_ms=NOW_MS, directory=reports)


OBSERVED_START = fakes.utc_ms(2025, 7, 31, 6)
OLDEST_FETCH = fakes.utc_ms(2026, 10, 30, 9)
NEWEST_FETCH = fakes.utc_ms(2026, 10, 31, 9)
ESTABLISHED = {
    "established": True,
    "history_start_ms": OBSERVED_START,
    "history_start_utc": utc_text(OBSERVED_START),
    "stop_reason": "empty_chunks",
    "server_time_ms": OLDEST_FETCH,
    "relpath": "raw/bingx/history-1m.json",
    "reason": None,
}


def write_observed_store(root: Path, history, archive_only_rows: int = 0) -> None:
    """A BingX 1m store of two chunks fetched a day apart, with the ``history`` provenance."""
    candles = fakes.synthetic_candles(OBSERVED_START, 3000, interval="1m", seed=7)
    chunks = []
    for index, (fetched, complete) in enumerate(((OLDEST_FETCH, True), (NEWEST_FETCH, False))):
        first = OBSERVED_START + 1500 * index * MINUTE_MS
        chunks.append(
            {
                "relpath": f"raw/bingx/1m/{'xy'[index]}.csv",
                "chunk_start_ms": first,
                "rows": 1500,
                "first_open_ms": first,
                "last_open_ms": first + 1499 * MINUTE_MS,
                "csv_sha256": str(index) * 64,
                "complete": complete,
                "server_time_ms": fetched,
                "archive_only_rows": archive_only_rows if index == 0 else 0,
            }
        )
    provenance = {
        "kind": "bingx_chunks",
        "chunks": chunks,
        "incomplete_chunks": ["raw/bingx/1m/y.csv"],
        **({} if history is None else {"history_start": history}),
    }
    fakes.write_bingx_store(root, candles, interval="1m", provenance=provenance)


def observed_row(tmp_path, root: Path) -> tuple[dict, str]:
    reports = tmp_path / "reports"
    payload = cv.coverage_report(root=root, now_ms=NOW_MS, directory=reports)
    row = next(r for r in payload["stores"] if r["relpath"] == "candles/bingx/1m.csv")
    return row, (reports / "coverage.md").read_text()


def test_observed_dates_come_from_the_chunk_provenance(tmp_path):
    root = tmp_path / "data"
    write_observed_store(root, ESTABLISHED)
    row, markdown = observed_row(tmp_path, root)
    assert row["observed"]["first_candle_observed_at"] == utc_text(OLDEST_FETCH)
    assert row["observed"]["last_candle_observed_at"] == utc_text(NEWEST_FETCH)
    assert row["observed"]["history_start_established"] is True
    assert row["observed"]["history_days_observed"] == pytest.approx(
        (OLDEST_FETCH - OBSERVED_START) / DAY_MS
    )
    assert row["observed"]["history_days_observed_reason"] is None
    assert (
        row["observed"]["store_built_at"] == read_meta(StoreId("bingx", "1m"), root)["created_at"]
    )
    assert row["incomplete_chunks"] == ["raw/bingx/1m/y.csv"]
    assert f"| {utc_text(OLDEST_FETCH)} | 456.1 |" in markdown
    assert "not established" not in markdown and "lower bound" not in markdown
    assert "| `candles/bingx/1m.csv` | context_512h | " in markdown


@pytest.mark.parametrize(
    ("history", "archive_only_rows", "reason"),
    [
        (
            None,
            0,
            (
                "history start not established: the store's provenance does not say whether "
                "a backfill established it"
            ),
        ),
        (
            {**ESTABLISHED, "established": False, "reason": "no backfill has established it"},
            0,
            "history start not established: no backfill has established it",
        ),
        (
            {**ESTABLISHED, "history_start_ms": OBSERVED_START - MINUTE_MS},
            0,
            "history start not established: the start was not established",
        ),
        (
            ESTABLISHED,
            3,
            (
                "the oldest chunk holds candles that its last fetch no longer returned, so "
                "that fetch does not date the first candle"
            ),
        ),
    ],
    ids=["no record", "bounded run", "another candle", "archive-only candles"],
)
def test_the_archive_depth_is_never_reported_as_the_observed_history(
    tmp_path, history, archive_only_rows, reason
):
    root = tmp_path / "data"
    write_observed_store(root, history, archive_only_rows)
    row, markdown = observed_row(tmp_path, root)
    observed = row["observed"]
    assert observed["history_days_observed"] is None
    assert observed["history_days_observed_reason"].startswith(reason)
    assert observed["first_candle_observed_at"] == utc_text(OLDEST_FETCH)  # still dated
    assert f"- `candles/bingx/1m.csv`: {observed['history_days_observed_reason']}" in markdown
    if archive_only_rows:
        assert observed["history_start_established"] is True
        assert f"| {utc_text(OLDEST_FETCH)} | n/a |" in markdown
        assert "| `candles/bingx/1m.csv` | context_512h | " in markdown
        assert "lower bound: history start not established" not in markdown
    else:
        assert observed["history_start_established"] is False
        assert f"| {utc_text(OLDEST_FETCH)} | not established |" in markdown
        # the usable decision hours of the archive are lower bounds on the market's
        assert (
            "| `candles/bingx/1m.csv` (lower bound: history start not established) | "
            "context_512h | " in markdown
        )


def test_the_history_depth_is_reported_once_a_backfill_established_the_start(tmp_path):
    root = tmp_path / "data"
    server = fakes.utc_ms(2025, 6, 1, 7, 2)
    start = fakes.utc_ms(2025, 2, 13, 3)  # inside its 1000-hour chunk, about 2600 h back
    candles = fakes.synthetic_candles(start, (server - start) // HOUR_MS + 1, seed=5)
    fake = fakes.FakeBingX(server, {"1h": candles}, history_start_ms={"1h": start})
    client, _ = fakes.bingx_client(fake)

    def depth() -> tuple[dict, str]:
        bh.merge_chunks("1h", root=root)
        reports = tmp_path / "reports"
        payload = cv.coverage_report(root=root, now_ms=fake.server_time_ms, directory=reports)
        row = next(r for r in payload["stores"] if r["relpath"] == "candles/bingx/1h.csv")
        return row["observed"], (reports / "coverage.md").read_text()

    bh.backfill(client, "1h", root=root, max_chunks=1)  # the newest chunk only
    observed, markdown = depth()
    assert observed["history_start_established"] is False
    assert observed["history_days_observed"] is None
    assert "not established" in observed["history_days_observed_reason"]
    assert "| not established |" in markdown
    assert (
        "| `candles/bingx/1h.csv` (lower bound: history start not established) | "
        "context_512h | " in markdown
    )
    bh.backfill(client, "1h", root=root)  # the whole walk, to three empty chunks
    observed, markdown = depth()
    assert observed["history_start_established"] is True
    assert observed["history_days_observed"] == pytest.approx((server - start) / DAY_MS)
    assert "not established" not in markdown
    assert "| `candles/bingx/1h.csv` | context_512h | " in markdown
    fake.advance(48 * HOUR_MS)
    bh.backfill(client, "1h", root=root, max_chunks=1)  # a later bounded run keeps it
    observed, _ = depth()
    assert observed["history_start_established"] is True
    assert observed["history_days_observed"] == pytest.approx((server - start) / DAY_MS)


def test_coverage_refuses_while_a_run_rebuilds_a_store(tmp_path, stores):
    root, _ = stores
    with exclusive_lock(root / "raw" / "bingx" / ".lock"), pytest.raises(StoreBusyError):
        cv.coverage_report(root=root, now_ms=NOW_MS, directory=tmp_path / "reports")
    assert not (tmp_path / "reports" / "coverage.json").exists()


def test_a_meta_that_does_not_describe_its_candles_is_refused(tmp_path):
    root = tmp_path / "data"
    start = fakes.utc_ms(2025, 5, 5)
    fakes.write_bingx_store(root, fakes.synthetic_candles(start, 400, seed=3))
    meta_path = StoreId("bingx", "1h").meta_path(root)
    meta = json.loads(meta_path.read_text())
    meta["gaps"] = [[start + 10 * HOUR_MS, start + 11 * HOUR_MS]]  # a gap the CSV does not have
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    with pytest.raises(cv.CoverageError, match="gaps"):
        cv.coverage_report(root=root, now_ms=NOW_MS, directory=tmp_path / "reports")


@pytest.mark.parametrize("now_ms", [-1, 1.5, True, "2026-11-01"])
def test_now_must_be_epoch_milliseconds(tmp_path, now_ms):
    with pytest.raises(cv.CoverageError):
        cv.coverage_report(root=tmp_path, now_ms=now_ms, directory=tmp_path / "reports")


def test_an_empty_root_reports_every_decision_store_as_not_built(tmp_path):
    payload = cv.coverage_report(root=tmp_path / "data", now_ms=NOW_MS, directory=tmp_path / "r")
    assert [row["built"] for row in payload["stores"]] == [False] * len(cv.DECISION_STORES)
    assert set(payload["usable_decision_hours"].values()) == {None}
    assert "question_5" not in payload


def test_coverage_never_calls_read_store():
    tree = ast.parse((REPO_ROOT / "src/kronos_agent/data/coverage.py").read_text())
    names = {
        getattr(node, "id", None) or getattr(node, "attr", None) or getattr(node, "name", None)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute, ast.alias))
    }
    assert "read_store" not in names
    assert "read_study_frame" in names


def test_segments_and_windows_are_the_recorded_ones():
    config = cv.study_config()
    assert config["windows"] == {"context_512h": 512}  # the profile windows left with D-042
    assert config["usability_stores"] == [
        "candles/bingx/1h.csv",
        "candles/bingx/15m.csv",
        "candles/bingx/5m.csv",
        "candles/bingx/1m.csv",
        "candles/binance/1m.csv",
        "candles/binance/1h.csv",
    ]
    assert config["segments"]["validation"] == ["2024-07-01T00:00Z", "2026-07-10T08:00Z"]
    assert config["statistics"] == [] and config["tests_consumed"] == 0
