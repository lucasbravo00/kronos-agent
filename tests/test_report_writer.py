"""Report writer: the required header, the refusals, recorded configs, JSON layout, study series
and the append-only record files."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kronos_agent.eval.bootstrap import block_bootstrap, week_blocks
from kronos_agent.exchange.public_market import PUBLIC_HOST
from kronos_agent.hashing import canonical_json, sha256_bytes, sha256_file
from kronos_agent.paths import REPO_ROOT
from kronos_agent.report.writer import (
    RECORDS_SUFFIX,
    REQUIRED_KEYS,
    ConfigChangedError,
    ReportError,
    append_record,
    data_md_sha256,
    read_records,
    record_config,
    report_header,
    to_jsonable,
    write_report,
    write_study_series,
)

STEM = "unit_study"
CONFIG = {
    "samples": "every hour with a complete window",
    "statistics": ["close_abs_median_bp"],
    "blocks": "calendar weeks",
    "resamples": 1000,
}
MARKDOWN = "# Unit study\n\n| statistic | value |\n| --- | --- |\n| close | 0.5 bp |\n"


def header(directory: Path, config=CONFIG) -> dict:
    return report_header(
        STEM,
        config,
        statistics_with_intervals=3,
        seeds={"close_abs_median_bp": f"{STEM}/close_abs_median_bp"},
        samples={"eligible_hours": 3, "used_hours": 3},
        comparators={"close_abs_median_bp": "one price tick in bp"},
        directory=directory,
    )


def test_report_header_fills_every_required_key(tmp_path):
    payload = header(tmp_path)
    assert set(REQUIRED_KEYS) <= set(payload)
    assert payload["study"] == STEM
    assert payload["tests_consumed"] == 0
    assert payload["statistics_with_intervals"] == 3
    assert payload["expected_chance_misses"] == 0.15  # exactly, not 0.15000000000000002
    assert payload["config_sha256"] == sha256_bytes(canonical_json(CONFIG))
    assert payload["data_md_sha256"] == data_md_sha256() == sha256_file(REPO_ROOT / "docs/DATA.md")
    assert set(payload["code"]) == {"commit", "dirty", "source_tree_sha256"}
    assert payload["generated_at"].endswith("Z")
    recorded = tmp_path / f"{STEM}.config.json"
    assert recorded.read_bytes() == canonical_json(CONFIG)
    assert sha256_file(recorded) == payload["config_sha256"]
    with pytest.raises(ReportError):
        report_header(
            STEM,
            CONFIG,
            statistics_with_intervals=-1,
            seeds={},
            samples={},
            comparators={},
            directory=tmp_path,
        )


def test_write_report_layout(tmp_path):
    blocks = week_blocks(1_719_792_000_000 + 3_600_000 * np.arange(21 * 168))
    interval = block_bootstrap(
        [np.random.default_rng(0).normal(size=blocks.size)], blocks, np.mean, key=f"{STEM}/mean"
    )
    payload = {
        **header(tmp_path),
        "statistics": {
            "mean": np.float64(1.5),
            "count": np.int64(3),
            "flags": np.array([True, False]),
            "series": np.array([0.25, 0.5]),
            "lags": {1: 0.1, 2: -0.05},
            "interval": interval,
            "undefined": None,
            "undefined_reason": "no hour had a complete window",
        },
    }
    json_path, markdown_path = write_report(
        STEM, payload, MARKDOWN.rstrip("\n"), directory=tmp_path
    )
    assert (json_path, markdown_path) == (tmp_path / f"{STEM}.json", tmp_path / f"{STEM}.md")
    text = json_path.read_text(encoding="ascii")
    document = json.loads(text)
    assert text == json.dumps(document, indent=1, sort_keys=True) + "\n"
    assert document["statistics"]["count"] == 3 and document["statistics"]["flags"] == [True, False]
    assert document["statistics"]["lags"] == {"1": 0.1, "2": -0.05}
    assert document["statistics"]["interval"]["key"] == f"{STEM}/mean"
    assert document["statistics"]["interval"]["low"] == interval.low
    assert markdown_path.read_text() == MARKDOWN  # a final newline is added
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"{STEM}.config.json",
        f"{STEM}.json",
        f"{STEM}.md",
    ]


@pytest.mark.parametrize("missing", REQUIRED_KEYS)
def test_write_report_refuses_a_missing_header_key(tmp_path, missing):
    payload = header(tmp_path)
    del payload[missing]
    with pytest.raises(ReportError, match=missing):
        write_report(STEM, payload, MARKDOWN, directory=tmp_path)
    assert not (tmp_path / f"{STEM}.json").exists()


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        np.float64("nan"),
        np.float32("-inf"),
        np.array([1.0, np.nan]),
        [1.0, {"deep": -math.inf}],
        Decimal("NaN"),
    ],
)
def test_write_report_refuses_non_finite_numbers(tmp_path, value):
    payload = {**header(tmp_path), "statistics": {"value": value}}
    with pytest.raises(ReportError, match="non-finite value at statistics.value"):
        write_report(STEM, payload, MARKDOWN, directory=tmp_path)
    assert not (tmp_path / f"{STEM}.json").exists()


def test_write_report_refuses_personal_data(tmp_path):
    payload = header(tmp_path)
    home = str(Path.home())  # read at run time: never written into the repository
    with pytest.raises(ReportError, match="personal data"):
        write_report(STEM, payload, f"data under {home}/somewhere\n", directory=tmp_path)
    with pytest.raises(ReportError, match="personal data"):
        write_report(STEM, {**payload, "note": home}, MARKDOWN, directory=tmp_path)
    address = ".".join(str(part) for part in (10, 20, 30, 40))  # no IPv4 literal in this file
    with pytest.raises(ReportError, match="IPv4"):
        write_report(STEM, payload, f"server {address}\n", directory=tmp_path)
    assert not (tmp_path / f"{STEM}.md").exists()


def test_write_report_refuses_bingx_hosts_absolute_paths_and_non_ascii(tmp_path):
    payload = header(tmp_path)
    with pytest.raises(ReportError, match="BingX host"):
        write_report(STEM, payload, f"candles from {PUBLIC_HOST}\n", directory=tmp_path)
    with pytest.raises(ReportError, match="BingX host"):
        write_report(
            STEM, {**payload, "source": f"https://{PUBLIC_HOST}"}, MARKDOWN, directory=tmp_path
        )
    with pytest.raises(ReportError):
        write_report(STEM, payload, f"see {REPO_ROOT / 'data'}\n", directory=tmp_path)
    with pytest.raises(ReportError, match="ASCII"):
        write_report(STEM, payload, "difference \u00b1 2 bp\n", directory=tmp_path)
    assert not (tmp_path / f"{STEM}.json").exists()


def test_write_report_refuses_a_wrong_study_or_stem(tmp_path):
    payload = header(tmp_path)
    with pytest.raises(ReportError, match="carries study"):
        write_report(STEM, {**payload, "study": "other"}, MARKDOWN, directory=tmp_path)
    for stem in ("../escape", "a b", "", "x/y"):
        with pytest.raises(ReportError):
            write_report(stem, {**payload, "study": stem}, MARKDOWN, directory=tmp_path)
    with pytest.raises(ReportError):
        write_report(STEM, {**payload, "where": Path("x")}, MARKDOWN, directory=tmp_path)


def test_record_config_is_written_once_and_refuses_a_changed_definition(tmp_path):
    digest = record_config(STEM, CONFIG, directory=tmp_path)
    reordered = dict(reversed(list(CONFIG.items())))
    assert record_config(STEM, reordered, directory=tmp_path) == digest  # key order is irrelevant
    changed = {**CONFIG, "resamples": 999}
    with pytest.raises(ConfigChangedError, match="new stem"):
        record_config(STEM, changed, directory=tmp_path)
    with pytest.raises(ConfigChangedError):
        header(tmp_path, changed)
    assert (tmp_path / f"{STEM}.config.json").read_bytes() == canonical_json(CONFIG)
    with pytest.raises(ReportError, match="non-finite"):
        record_config("other_study", {"threshold": float("nan")}, directory=tmp_path)
    with pytest.raises(ReportError):
        record_config("other_study", ["not", "a", "mapping"], directory=tmp_path)


def test_to_jsonable_conversions():
    assert to_jsonable(np.float32(0.5)) == 0.5 and type(to_jsonable(np.float32(0.5))) is float
    assert to_jsonable(np.int32(3)) == 3 and type(to_jsonable(np.int32(3))) is int
    assert to_jsonable(np.bool_(True)) is True
    assert to_jsonable(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    assert to_jsonable([float("nan"), np.float64("inf"), 1.0]) == [None, None, 1.0]
    assert to_jsonable(Decimal("0.10")) == "0.10"
    assert to_jsonable(datetime(2024, 7, 1, tzinfo=UTC)) == "2024-07-01T00:00:00Z"
    assert to_jsonable(pd.Timestamp("2024-07-01 05:00")) == "2024-07-01T05:00:00Z"  # naive = UTC
    assert to_jsonable(date(2024, 7, 1)) == "2024-07-01"
    assert to_jsonable(pd.NaT) is None
    assert to_jsonable({1: "a", np.int64(2): (1, 2)}) == {"1": "a", "2": [1, 2]}
    for bad in ({1, 2}, Path("x"), {(1, 2): 3}, {True: 1}, object()):
        with pytest.raises(ReportError):
            to_jsonable(bad)


def test_append_record_is_append_only_and_refused_like_a_report(tmp_path):
    first = {"given_at": "2026-11-02T10:00:00Z", "figure": 1.5, "nested": {"n": np.int64(3)}}
    path = append_record("unit.records", first, directory=tmp_path)
    assert path == tmp_path / f"unit.records{RECORDS_SUFFIX}" == tmp_path / "unit.records.jsonl"
    data = path.read_bytes()
    assert data == canonical_json({**first, "nested": {"n": 3}}) + b"\n"
    second = {"given_at": "2026-11-02T11:00:00Z", "figure": None}
    append_record("unit.records", second, directory=tmp_path)
    assert path.read_bytes() == data + canonical_json(second) + b"\n"  # the first line is kept
    assert read_records("unit.records", directory=tmp_path) == [
        {**first, "nested": {"n": 3}},
        second,
    ]
    assert read_records("absent", directory=tmp_path) == []
    home = str(Path.home())  # read at run time: never written into the repository
    refused = [
        ({"figure": float("nan")}, "non-finite"),
        ({"note": f"saved under {home}/x"}, "personal data"),
        ({"source": f"https://{PUBLIC_HOST}"}, "BingX host"),
        ({"where": str(REPO_ROOT / "data")}, None),  # an absolute path (personal on a laptop)
        (["not", "a", "mapping"], "mapping"),
    ]
    for record, match in refused:
        with pytest.raises(ReportError, match=match):
            append_record("unit.records", record, directory=tmp_path)
    with pytest.raises(ReportError):
        append_record("../escape", second, directory=tmp_path)
    assert len(read_records("unit.records", directory=tmp_path)) == 2


def test_a_torn_or_malformed_record_file_is_refused_never_repaired(tmp_path):
    path = append_record("unit.records", {"n": 1}, directory=tmp_path)
    good = path.read_bytes()
    for content, match in (
        (good + b'{"n": ', "torn"),
        (good + b"[1, 2]\n", "JSON object"),
        (good + b"not json\n", "not a JSON record"),
    ):
        path.write_bytes(content)
        with pytest.raises(ReportError, match=match):
            read_records("unit.records", directory=tmp_path)
        with pytest.raises(ReportError, match=match):
            append_record("unit.records", {"n": 2}, directory=tmp_path)
        assert path.read_bytes() == content


def test_write_study_series(tmp_path):
    frame = pd.DataFrame(
        {"decision_ms": np.array([1, 2], dtype=np.int64), "dpoc_bp": [0.1, float("nan")]}
    )
    info = write_study_series("poc_unit", frame, root=tmp_path)
    path = tmp_path / "studies" / "poc_unit.csv"
    assert info == {"relpath": "studies/poc_unit.csv", "sha256": sha256_file(path), "rows": 2}
    assert path.read_text() == "decision_ms,dpoc_bp\n1,0.1\n2,\n"
    assert write_study_series("poc_unit", frame, root=tmp_path) == info  # same bytes again
    with pytest.raises(ReportError):
        write_study_series("poc_unit", frame.to_numpy(), root=tmp_path)
    with pytest.raises(ReportError):
        write_study_series("../poc", frame, root=tmp_path)
