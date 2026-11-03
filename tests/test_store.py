"""Candle stores: format, byte determinism, exact decimals, half-open ranges, integrity, sealing."""

from __future__ import annotations

import hashlib
import io
import json
import signal
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import replace

import httpx
import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes

from kronos_agent.data import partition as pt
from kronos_agent.data import store
from kronos_agent.data.fsutil import LockBusyError, exclusive_lock
from kronos_agent.data.partition import SEALED_START_MS, SealedPartitionError, sealed_access
from kronos_agent.data.store import (
    BASE_COLUMNS,
    BINANCE_COLUMNS,
    StoreBusyError,
    StoreError,
    StoreId,
    StoreIntegrityError,
    StoreMissingError,
    StoreWriter,
    find_gaps,
    float_values,
    job_lock,
    leftover_temporaries,
    read_meta,
    read_store,
    read_study_frame,
    store_read_lock,
    store_status,
    validate_frame,
    write_store,
)
from kronos_agent.hashing import sha256_file

HOUR_MS = 3_600_000
START_MS = fakes.utc_ms(2024, 1, 1)
# 17 significant digits: pandas' default parser and pandas.to_numeric read it one ulp off
VECTOR = "465655238.20240235"
BINGX_1H = StoreId("bingx", "1h")
PROVENANCE = {"chunks": [{"start_ms": START_MS, "sha256": "0" * 64, "complete": True}]}


def bingx_frame(count: int = 48, *, start_ms: int = START_MS, seed: int = 0) -> pd.DataFrame:
    return fakes.candles_to_frame(fakes.synthetic_candles(start_ms, count, seed=seed))


def csv_lines(root, store_id: StoreId = BINGX_1H) -> list[str]:
    return store_id.csv_path(root).read_bytes().decode("ascii").split("\n")


# -- identifiers and validation -------------------------------------------------------------------


def test_store_id_properties_and_validation(tmp_path):
    assert BINGX_1H.symbol == "BTC-USDT"
    assert BINGX_1H.columns == BASE_COLUMNS
    assert BINGX_1H.step_ms == HOUR_MS
    assert BINGX_1H.relpath == "candles/bingx/1h.csv"
    assert BINGX_1H.meta_relpath == "candles/bingx/1h.meta.json"
    assert BINGX_1H.csv_path(tmp_path) == tmp_path / "candles" / "bingx" / "1h.csv"
    assert BINGX_1H.meta_path(tmp_path) == tmp_path / "candles" / "bingx" / "1h.meta.json"
    minute = StoreId("binance", "1m")
    assert minute.columns == BINANCE_COLUMNS and minute.step_ms == 60_000
    assert minute.symbol == StoreId("binance_official", "1h").symbol == "BTCUSDT"
    assert StoreId("bingx", "15m").step_ms == 900_000
    for source, interval in (("bingx", "2h"), ("demo", "1h"), ("binance", "1d")):
        with pytest.raises(ValueError):
            StoreId(source, interval)


def mutate(frame: pd.DataFrame, column: str, row: int, value) -> pd.DataFrame:
    out = frame.copy()
    values = out[column].to_numpy(dtype=object).copy()
    values[row] = value
    out[column] = pd.Series(values, dtype=object)
    return out


@pytest.mark.parametrize(
    "value",
    [
        "1e5",
        ".5",
        "5.",
        "-",
        "+5",
        " 5",
        "5 ",
        "5\n",
        "1,5",
        '"5"',
        "nan",
        "inf",
        "",
        "0x10",
        "\u0661",  # an Arabic-Indic digit is not [0-9]
        5.0,
        None,
    ],
)
def test_validate_frame_refuses_values_that_are_not_plain_decimal_strings(value):
    with pytest.raises(StoreError):
        validate_frame(mutate(bingx_frame(5), "close", 2, value), BINGX_1H)


def test_validate_frame_refuses_bad_structure():
    frame = bingx_frame(5)
    with pytest.raises(StoreError, match="columns"):
        validate_frame(frame[["open_ms", "high", "open", "low", "close", "volume"]], BINGX_1H)
    with pytest.raises(StoreError, match="columns"):
        validate_frame(frame.assign(extra="1"), BINGX_1H)
    with pytest.raises(StoreError, match="int64"):
        validate_frame(frame.assign(open_ms=frame["open_ms"].astype(float)), BINGX_1H)
    with pytest.raises(StoreError, match="multiple"):
        validate_frame(frame.assign(open_ms=frame["open_ms"] + 1), BINGX_1H)
    duplicate = frame.assign(open_ms=frame["open_ms"].where(frame.index != 3, frame["open_ms"][2]))
    with pytest.raises(StoreError, match="duplicate"):
        validate_frame(duplicate, BINGX_1H)
    with pytest.raises(StoreError, match="increasing"):
        validate_frame(frame.iloc[::-1].reset_index(drop=True), BINGX_1H)
    with pytest.raises(StoreError, match="negative"):
        validate_frame(frame.assign(open_ms=frame["open_ms"] - START_MS - HOUR_MS), BINGX_1H)
    with pytest.raises(StoreError):
        validate_frame(frame.to_numpy(), BINGX_1H)


def test_validate_frame_count_must_be_an_unsigned_integer():
    frame = fakes.binance_frame(START_MS, 5, interval="1h")
    validate_frame(frame, StoreId("binance", "1h"))
    for bad in ("1.0", "-1", "1e3"):
        with pytest.raises(StoreError, match="count"):
            validate_frame(mutate(frame, "count", 1, bad), StoreId("binance", "1h"))


def test_ohlc_violations_are_counted_with_decimals_and_never_fixed(tmp_path):
    candles = fakes.synthetic_candles(START_MS, 12)
    # float("100.09999999999999999") == float("100.1"): only a decimal comparison sees high < close
    candles[3] = replace(
        candles[3], open="100.0", high="100.09999999999999999", low="99.9", close="100.1"
    )
    candles[5] = replace(candles[5], low=str(float(candles[5].open) + 1.0))  # low above open
    candles[7] = replace(candles[7], volume="-0.5")
    # equal as decimals, different strings: no violation
    candles[9] = replace(candles[9], open="100.0", high="100.10", low="99.9", close="100.1")
    frame = fakes.candles_to_frame(candles)
    info = validate_frame(frame, BINGX_1H)
    expected = [candles[3].open_ms, candles[5].open_ms, candles[7].open_ms]
    assert info == {"ohlc_violations": 3, "ohlc_violation_samples": expected}
    meta = write_store(BINGX_1H, frame, provenance=PROVENANCE, root=tmp_path)
    assert meta["ohlc_violations"] == 3 and meta["ohlc_violation_samples"] == expected
    stored = read_store(BINGX_1H, root=tmp_path)
    assert stored.loc[3, "high"] == "100.09999999999999999"  # kept verbatim, not fixed
    assert stored.loc[9, "high"] == "100.10"


def test_violation_samples_keep_the_first_ten_across_frames(tmp_path):
    candles = fakes.synthetic_candles(START_MS, 30)
    bad = [replace(c, volume="-1.0") if i % 2 == 0 else c for i, c in enumerate(candles)]
    with StoreWriter(BINGX_1H, root=tmp_path) as writer:
        writer.append(fakes.candles_to_frame(bad[:15]))
        writer.append(fakes.candles_to_frame(bad[15:]))
        meta = writer.finish(provenance=PROVENANCE)
    assert meta["ohlc_violations"] == 15
    assert meta["ohlc_violation_samples"] == [bad[i].open_ms for i in range(0, 20, 2)]


def test_find_gaps():
    assert find_gaps(np.array([0, 1, 2]), 1) == []
    assert find_gaps(np.array([0, 3, 4, 7]), 1) == [[1, 3], [5, 7]]
    hours = np.array([START_MS, START_MS + HOUR_MS, START_MS + 4 * HOUR_MS])
    assert find_gaps(hours, HOUR_MS) == [[START_MS + 2 * HOUR_MS, START_MS + 4 * HOUR_MS]]
    assert find_gaps(np.array([5]), 1) == [] and find_gaps(np.array([], dtype=np.int64), 1) == []
    with pytest.raises(ValueError):
        find_gaps(np.array([0, 2, 1]), 1)
    with pytest.raises(ValueError):
        find_gaps(np.array([0, 0]), 1)
    with pytest.raises(ValueError):
        find_gaps(np.array([0, 3]), 2)


# -- writing ---------------------------------------------------------------------------------------


def test_round_trip_and_byte_determinism(tmp_path):
    frame = bingx_frame(48)
    meta = write_store(BINGX_1H, frame, provenance=PROVENANCE, root=tmp_path / "a")
    lines = csv_lines(tmp_path / "a")
    assert lines[0] == "open_ms,open,high,low,close,volume"
    assert lines[-1] == "" and len(lines) == 50  # header + 48 rows, a final newline
    first = frame.iloc[0]
    assert lines[1] == ",".join([str(START_MS), *(first[c] for c in BASE_COLUMNS[1:])])
    raw = BINGX_1H.csv_path(tmp_path / "a").read_bytes()
    assert b'"' not in raw and b"\r" not in raw
    back = read_store(BINGX_1H, root=tmp_path / "a")
    pd.testing.assert_frame_equal(back, frame)
    assert all(isinstance(value, str) for value in back["close"])
    assert isinstance(back.index, pd.RangeIndex) and back["open_ms"].dtype == np.int64
    again = write_store(BINGX_1H, frame, provenance=PROVENANCE, root=tmp_path / "b")
    assert BINGX_1H.csv_path(tmp_path / "b").read_bytes() == raw
    assert (
        again["csv_sha256"] == meta["csv_sha256"] == sha256_file(BINGX_1H.csv_path(tmp_path / "a"))
    )
    rewritten = write_store(BINGX_1H, frame, provenance=PROVENANCE, root=tmp_path / "a")
    assert BINGX_1H.csv_path(tmp_path / "a").read_bytes() == raw
    assert rewritten["csv_sha256"] == meta["csv_sha256"]


def test_meta_records_range_gaps_provenance_and_code(tmp_path):
    frame = bingx_frame(48)
    frame = frame.drop(index=[10, 11, 30]).reset_index(drop=True)
    meta = write_store(BINGX_1H, frame, provenance=PROVENANCE, root=tmp_path)
    assert meta == read_meta(BINGX_1H, root=tmp_path)
    assert meta["schema_version"] == 1
    assert (meta["source"], meta["symbol"], meta["interval"]) == ("bingx", "BTC-USDT", "1h")
    assert meta["columns"] == list(BASE_COLUMNS) and meta["step_ms"] == HOUR_MS
    assert meta["rows"] == 45
    assert meta["first_open_ms"] == START_MS and meta["first_open_utc"] == "2024-01-01T00:00Z"
    assert meta["last_open_ms"] == START_MS + 47 * HOUR_MS
    assert meta["last_open_utc"] == "2024-01-02T23:00Z"
    assert meta["gaps"] == [
        [START_MS + 10 * HOUR_MS, START_MS + 12 * HOUR_MS],
        [START_MS + 30 * HOUR_MS, START_MS + 31 * HOUR_MS],
    ]
    assert meta["missing_candles"] == 3
    path = BINGX_1H.csv_path(tmp_path)
    assert meta["csv_sha256"] == sha256_file(path) and meta["csv_size"] == path.stat().st_size
    assert meta["csv_relpath"] == "candles/bingx/1h.csv"
    assert meta["provenance"] == PROVENANCE
    assert set(meta["code"]) == {"commit", "dirty", "source_tree_sha256"}
    assert meta["created_at"].endswith("Z")
    text = BINGX_1H.meta_path(tmp_path).read_text()
    assert text.endswith("\n") and json.loads(text) == meta


def test_streaming_writer_matches_a_one_shot_write_and_records_seam_gaps(tmp_path):
    frame = bingx_frame(100)
    kept = frame.drop(index=[60, 80]).reset_index(drop=True)
    pieces = [kept.iloc[:30], kept.iloc[30:60], kept.iloc[60:]]  # the second seam skips row 60
    writer = StoreWriter(BINGX_1H, root=tmp_path / "stream")
    for piece in pieces:
        writer.append(piece.reset_index(drop=True))
        assert not BINGX_1H.csv_path(tmp_path / "stream").exists()  # nothing visible before finish
    assert writer.rows == 98
    meta = writer.finish(provenance=PROVENANCE)
    one_shot = write_store(BINGX_1H, kept, provenance=PROVENANCE, root=tmp_path / "once")
    assert meta["csv_sha256"] == one_shot["csv_sha256"]
    assert (
        meta["gaps"]
        == one_shot["gaps"]
        == [
            [START_MS + 60 * HOUR_MS, START_MS + 61 * HOUR_MS],
            [START_MS + 80 * HOUR_MS, START_MS + 81 * HOUR_MS],
        ]
    )
    with pytest.raises(StoreError, match="already finished"):
        writer.append(kept.iloc[:1])
    with pytest.raises(StoreError):
        writer.finish(provenance=PROVENANCE)


def test_frames_out_of_order_are_refused(tmp_path):
    frame = bingx_frame(20)
    with StoreWriter(BINGX_1H, root=tmp_path) as writer:
        writer.append(frame.iloc[:10].reset_index(drop=True))
        with pytest.raises(StoreError, match="ascending"):
            writer.append(frame.iloc[9:].reset_index(drop=True))  # overlaps the last row
        writer.append(frame.iloc[10:].reset_index(drop=True))  # still usable
        meta = writer.finish(provenance=PROVENANCE)
    assert meta["rows"] == 20 and meta["gaps"] == []


def test_abort_leaves_nothing_and_keeps_an_existing_store(tmp_path):
    directory = BINGX_1H.csv_path(tmp_path).parent
    with pytest.raises(RuntimeError), StoreWriter(BINGX_1H, root=tmp_path) as writer:
        writer.append(bingx_frame(10))
        raise RuntimeError("build failed")
    assert sorted(p.name for p in directory.iterdir()) == []
    write_store(BINGX_1H, bingx_frame(10), provenance=PROVENANCE, root=tmp_path)
    before = BINGX_1H.csv_path(tmp_path).read_bytes()
    writer = StoreWriter(BINGX_1H, root=tmp_path)
    writer.append(bingx_frame(5, seed=3))
    writer.abort()
    writer.abort()  # idempotent
    assert BINGX_1H.csv_path(tmp_path).read_bytes() == before
    assert sorted(p.name for p in directory.iterdir()) == ["1h.csv", "1h.meta.json"]


def test_unserialisable_provenance_publishes_nothing(tmp_path):
    with pytest.raises(StoreError, match="JSON"):
        write_store(BINGX_1H, bingx_frame(5), provenance={"bad": {1, 2}}, root=tmp_path)
    assert list(BINGX_1H.csv_path(tmp_path).parent.iterdir()) == []


def test_empty_store(tmp_path):
    meta = write_store(BINGX_1H, bingx_frame(0), provenance=PROVENANCE, root=tmp_path)
    assert meta["rows"] == 0 and meta["first_open_ms"] is None and meta["gaps"] == []
    for numeric in (False, True):
        frame = read_store(BINGX_1H, numeric=numeric, root=tmp_path)
        assert list(frame.columns) == list(BASE_COLUMNS) and frame.empty
        assert frame["open_ms"].dtype == np.int64


def test_binance_store_keeps_every_string_format(tmp_path):
    frame = fakes.binance_frame(START_MS, 60, interval="1h")
    assert any("." not in value for value in frame["open"])  # whole-dollar rows like "7225"
    assert any(value.endswith("0") and "." in value for value in frame["close"])  # "7171.10"
    fakes.write_binance_store(tmp_path, frame)
    back = read_store(StoreId("binance", "1h"), root=tmp_path)
    pd.testing.assert_frame_equal(back, frame)
    numeric = read_store(StoreId("binance", "1h"), numeric=True, root=tmp_path)
    assert numeric["count"].dtype == np.int64
    assert numeric["quote_volume"].dtype == np.float64
    assert numeric["quote_volume"].tolist() == [float(v) for v in frame["quote_volume"]]


# -- reading ---------------------------------------------------------------------------------------


def test_numeric_reads_are_correctly_rounded(tmp_path):
    candles = fakes.synthetic_candles(START_MS, 10)
    candles[4] = replace(candles[4], volume=VECTOR)
    fakes.write_bingx_store(tmp_path, candles)
    numeric = read_store(BINGX_1H, numeric=True, root=tmp_path)
    assert numeric["volume"].dtype == np.float64 and numeric["open_ms"].dtype == np.int64
    expected = np.array([float(c.volume) for c in candles])
    assert numeric["volume"].to_numpy().view(np.int64).tolist() == expected.view(np.int64).tolist()
    assert numeric["volume"][4] == float(VECTOR)
    assert float_values([VECTOR])[0] == float(VECTOR)
    assert float_values(c.close for c in candles).tolist() == [float(c.close) for c in candles]
    assert read_store(BINGX_1H, root=tmp_path)["volume"][4] == VECTOR
    # why the vector: the default parser (and pandas.to_numeric) is one ulp off on it
    default = pd.read_csv(io.BytesIO(BINGX_1H.csv_path(tmp_path).read_bytes()))
    assert default["volume"][4] != float(VECTOR)


def range_rows(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    keep = np.ones(len(frame), dtype=bool)
    if start is not None:
        keep &= frame["open_ms"].to_numpy() >= start
    if end is not None:
        keep &= frame["open_ms"].to_numpy() < end
    return frame[keep].reset_index(drop=True)


def test_ranged_reads_are_half_open(tmp_path):
    frame = bingx_frame(200).drop(index=[0, 50, 51, 52, 120, 199]).reset_index(drop=True)
    write_store(BINGX_1H, frame, provenance=PROVENANCE, root=tmp_path)
    opens = [START_MS + k * HOUR_MS for k in range(200)]

    def read(start, end, numeric=False):
        return read_store(BINGX_1H, numeric=numeric, start_ms=start, end_ms=end, root=tmp_path)

    window = read(opens[10], opens[20])
    assert window["open_ms"].tolist() == opens[10:20]  # start included, end excluded
    assert read(opens[10], opens[10]).empty
    assert read(opens[10] + 1, opens[20] + 1)["open_ms"].tolist() == opens[11:21]
    assert read(opens[49], opens[54])["open_ms"].tolist() == [opens[49], opens[53]]  # a gap inside
    assert read(None, opens[3])["open_ms"].tolist() == opens[1:3]
    assert read(opens[197], None)["open_ms"].tolist() == opens[197:199]
    assert read(opens[199], None).empty
    assert read(0, opens[0]).empty
    rng = np.random.default_rng(5)
    bounds = [
        None,
        *opens,
        opens[0] - 1,
        opens[-1] + HOUR_MS,
        *(rng.integers(opens[0], opens[-1], 40)),
    ]
    for _ in range(150):
        start, end = (bounds[i] for i in rng.integers(0, len(bounds), size=2))
        if start is not None and end is not None and start > end:
            start, end = end, start
        numeric = bool(rng.integers(0, 2))
        got = read(
            None if start is None else int(start), None if end is None else int(end), numeric
        )
        reference = range_rows(read(None, None, numeric), start, end)
        pd.testing.assert_frame_equal(got, reference)
    with pytest.raises(ValueError):
        read(opens[20], opens[10])
    with pytest.raises(ValueError):
        read(float(opens[1]), None)


def test_usecols_keeps_open_ms_and_store_order(tmp_path):
    fakes.write_binance_store(tmp_path, fakes.binance_frame(START_MS, 20, interval="1h"))
    binance = StoreId("binance", "1h")
    frame = read_store(binance, usecols=["quote_volume", "close"], root=tmp_path)
    assert list(frame.columns) == ["open_ms", "close", "quote_volume"]
    numeric = read_store(binance, numeric=True, usecols=["count"], start_ms=START_MS, root=tmp_path)
    assert list(numeric.columns) == ["open_ms", "count"] and numeric["count"].dtype == np.int64
    with pytest.raises(ValueError, match="unknown"):
        read_store(binance, usecols=["amount"], root=tmp_path)
    with pytest.raises(TypeError):
        read_store(binance, usecols="close", root=tmp_path)


def test_reads_verify_the_csv_against_its_meta(tmp_path):
    frame = bingx_frame(24)
    write_store(BINGX_1H, frame, provenance=PROVENANCE, root=tmp_path)
    path = BINGX_1H.csv_path(tmp_path)
    lines = path.read_bytes().split(b"\n")
    fields = lines[6].split(b",")  # line 0 is the header, so row 5 is line 6
    fields[-1] = fields[-1][:-1] + (b"1" if fields[-1][-1:] != b"1" else b"2")  # same size
    lines[6] = b",".join(fields)
    data = b"\n".join(lines)
    path.write_bytes(data)
    with pytest.raises(StoreIntegrityError, match="sha256"):
        read_store(BINGX_1H, root=tmp_path)
    with pytest.raises(StoreIntegrityError):
        read_study_frame(BINGX_1H, start_ms=0, end_ms=SEALED_START_MS, root=tmp_path)
    tampered = read_store(BINGX_1H, root=tmp_path, verify=False)  # integrity work may look
    assert tampered["volume"][5] != frame["volume"][5]
    path.write_bytes(data[:-10])
    with pytest.raises(StoreIntegrityError):
        read_store(BINGX_1H, root=tmp_path)
    path.unlink()
    with pytest.raises(StoreIntegrityError, match="missing"):
        read_store(BINGX_1H, root=tmp_path)


def test_missing_and_foreign_metas(tmp_path):
    with pytest.raises(StoreMissingError):
        read_meta(BINGX_1H, root=tmp_path)
    with pytest.raises(StoreMissingError):
        read_store(BINGX_1H, root=tmp_path)
    write_store(BINGX_1H, bingx_frame(5), provenance=PROVENANCE, root=tmp_path)
    other = StoreId("bingx", "5m")
    other.meta_path(tmp_path).write_bytes(BINGX_1H.meta_path(tmp_path).read_bytes())
    with pytest.raises(StoreIntegrityError, match="does not describe"):
        read_meta(other, root=tmp_path)
    BINGX_1H.meta_path(tmp_path).write_text("{not json")
    with pytest.raises(StoreIntegrityError):
        read_meta(BINGX_1H, root=tmp_path)


def peek(source) -> bytes:
    """What a parse of ``source`` (a buffer or the open store file) reads, left unconsumed."""
    position = source.tell()
    data = source.read()
    source.seek(position)
    return data


def test_read_study_frame_stops_at_the_sealed_start(tmp_path, monkeypatch):
    first = SEALED_START_MS - 48 * HOUR_MS
    fakes.write_bingx_store(tmp_path, fakes.synthetic_candles(first, 72))
    parsed: list[bytes] = []
    real_parse = store._parse_csv

    def spy(source, *args, **kwargs):
        parsed.append(peek(source))
        return real_parse(source, *args, **kwargs)

    monkeypatch.setattr(store, "_parse_csv", spy)
    frame = read_study_frame(BINGX_1H, start_ms=0, end_ms=SEALED_START_MS, root=tmp_path)
    assert len(frame) == 48
    assert frame["open_ms"].iloc[-1] == SEALED_START_MS - HOUR_MS
    assert (frame["open_ms"] < SEALED_START_MS).all()
    assert str(SEALED_START_MS).encode() not in parsed[-1]  # sealed rows were never even parsed
    with pytest.raises(SealedPartitionError):
        read_study_frame(BINGX_1H, start_ms=0, end_ms=SEALED_START_MS + 1, root=tmp_path)
    with pytest.raises(SealedPartitionError):  # refused before anything is read
        read_study_frame(BINGX_1H, start_ms=0, end_ms=SEALED_START_MS + 1, root=tmp_path / "none")
    monkeypatch.setattr(pt, "_verify_order", lambda order_id: None)
    opened = read_study_frame(
        BINGX_1H,
        start_ms=0,
        end_ms=SEALED_START_MS + 24 * HOUR_MS,
        sealed_access=sealed_access("order-unit"),
        root=tmp_path,
    )
    assert len(opened) == 72


def replace_mid_read(monkeypatch, tmp_path, frame: pd.DataFrame) -> list[bool]:
    """Publish ``frame`` as a new version of ``bingx/1h`` once a read has checked the CSV.

    ``_header_bytes`` is what a read calls right after its size and sha256 check and before it
    searches and parses the rows (the next ``StoreWriter`` calls it too).
    """
    real_header = store._header_bytes
    replaced: list[bool] = []

    def header_then_replace(store_id):
        if not replaced:
            replaced.append(True)
            write_store(BINGX_1H, frame, provenance=PROVENANCE, root=tmp_path)
        return real_header(store_id)

    monkeypatch.setattr(store, "_header_bytes", header_then_replace)
    return replaced


@pytest.mark.parametrize(
    "bounds", [(None, None), (START_MS + 5 * HOUR_MS, START_MS + 30 * HOUR_MS)]
)
def test_a_store_replaced_during_a_read_hands_back_only_the_rows_verified(
    tmp_path, monkeypatch, bounds
):
    verified = bingx_frame(48)
    write_store(BINGX_1H, verified, provenance=PROVENANCE, root=tmp_path)
    other = bingx_frame(48, seed=1)  # another version of the same hours
    replaced = replace_mid_read(monkeypatch, tmp_path, other)
    start, end = bounds
    frame = read_store(BINGX_1H, start_ms=start, end_ms=end, root=tmp_path)
    assert replaced  # the new version was published between the check and the parse
    pd.testing.assert_frame_equal(frame, range_rows(verified, start, end))
    assert not frame.equals(range_rows(other, start, end))
    monkeypatch.undo()
    pd.testing.assert_frame_equal(read_store(BINGX_1H, root=tmp_path), other)  # the next read


def test_no_sealed_row_is_parsed_when_the_store_is_replaced_mid_read(tmp_path, monkeypatch):
    # the whole verified file is parsed from the handle that was checked, never reopened by path
    first = SEALED_START_MS - 48 * HOUR_MS
    candles = fakes.synthetic_candles(first, 72)
    fakes.write_bingx_store(tmp_path, candles[:48])
    parsed: list[bytes] = []
    real_parse = store._parse_csv

    def spy(source, *args, **kwargs):
        parsed.append(peek(source))
        return real_parse(source, *args, **kwargs)

    monkeypatch.setattr(store, "_parse_csv", spy)
    replaced = replace_mid_read(monkeypatch, tmp_path, fakes.candles_to_frame(candles))
    frame = read_study_frame(BINGX_1H, start_ms=0, end_ms=SEALED_START_MS, root=tmp_path)
    assert replaced and len(parsed) == 1
    assert frame["open_ms"].tolist() == [c.open_ms for c in candles[:48]]
    assert str(SEALED_START_MS).encode() not in parsed[0]


def test_reads_hold_the_job_lock_of_their_source_shared(tmp_path):
    frame = bingx_frame(24)
    write_store(BINGX_1H, frame, provenance=PROVENANCE, root=tmp_path)
    assert BINGX_1H.lock_relpath == "raw/bingx/.lock"
    assert StoreId("binance_official", "1h").lock_relpath == "raw/binance/.lock"
    lock = BINGX_1H.lock_path(tmp_path)
    assert len(read_store(BINGX_1H, root=tmp_path)) == 24  # no job has run: nothing to share
    assert not lock.exists()  # a read never creates the lock file
    with exclusive_lock(lock):  # a run rebuilds the BingX stores
        with pytest.raises(StoreBusyError, match="raw/bingx/.lock") as caught:
            read_study_frame(BINGX_1H, start_ms=0, end_ms=SEALED_START_MS, root=tmp_path)
        assert isinstance(caught.value, LockBusyError) and isinstance(caught.value, StoreError)
        with pytest.raises(StoreBusyError):
            read_store(BINGX_1H, root=tmp_path, verify=False)
        fakes.write_binance_store(tmp_path, fakes.binance_frame(START_MS, 10, interval="1h"))
        assert len(read_store(StoreId("binance", "1h"), root=tmp_path)) == 10  # another lock
    with store_read_lock(BINGX_1H, tmp_path):  # a study holding it around several reads
        pd.testing.assert_frame_equal(read_store(BINGX_1H, root=tmp_path), frame)  # shared
        with pytest.raises(LockBusyError), exclusive_lock(lock):
            pass  # a rebuild is refused meanwhile
    with exclusive_lock(lock):
        pass  # released


def test_store_status_lists_built_stores(tmp_path):
    assert store_status(tmp_path) == []
    candles = fakes.synthetic_candles(START_MS, 30)
    fakes.write_bingx_store(tmp_path, candles[:10] + candles[12:])
    fakes.write_binance_store(tmp_path, fakes.binance_frame(START_MS, 120), interval="1m")
    rows = store_status(tmp_path)
    assert [(row["source"], row["interval"]) for row in rows] == [
        ("binance", "1m"),
        ("bingx", "1h"),
    ]
    bingx = rows[1]
    assert bingx["rows"] == 28 and bingx["gap_count"] == 1 and bingx["missing_candles"] == 2
    assert bingx["first"] == "2024-01-01T00:00Z" and bingx["csv_present"] and bingx["size_matches"]
    assert (bingx["sha256_matches"], bingx["busy"]) == (True, False)
    path = BINGX_1H.csv_path(tmp_path)
    data = path.read_bytes()
    edited = data[:-3] + (b"1" if data[-3:-2] != b"1" else b"2") + data[-2:]  # one digit
    path.write_bytes(edited)
    assert len(edited) == len(data)
    bingx = store_status(tmp_path)[1]
    assert (bingx["size_matches"], bingx["sha256_matches"]) == (True, False)
    assert store_status(tmp_path, verify_sha256=False)[1]["sha256_matches"] is None
    with exclusive_lock(BINGX_1H.lock_path(tmp_path)):  # a run may be replacing it: not checked
        busy, other = store_status(tmp_path)[1], store_status(tmp_path)[0]
    assert (busy["busy"], busy["csv_present"], busy["size_matches"]) == (True, None, None)
    assert busy["sha256_matches"] is None and busy["rows"] == 28
    assert (other["busy"], other["sha256_matches"]) == (False, True)
    path.write_bytes(data[:-5])
    bingx = store_status(tmp_path)[1]
    assert (bingx["size_matches"], bingx["sha256_matches"]) == (False, None)
    path.unlink()
    bingx = store_status(tmp_path)[1]
    assert (bingx["csv_present"], bingx["size_matches"]) == (False, False)
    BINGX_1H.meta_path(tmp_path).write_text("[]")
    assert "error" in store_status(tmp_path)[1]


# -- temporary files of killed runs ----------------------------------------------------------------

LEFTOVERS = {  # what tempfile.mkstemp names the writes under each job lock
    "candles/binance/.1m.csv.abcdefgh.tmp": b"rows of a killed build",
    "candles/binance_official/.1h.meta.json.a1b2c3d4.tmp": b"{",
    "raw/binance/.plan-1m.json.zzzzzzzz.tmp": b"{}",
    "raw/binance/daily/1m/.BTCUSDT-1m-2019-09-01.zip.k2_9x0q1.part": b"PK",
    "candles/bingx/.1h.csv.qwertyui.tmp": b"partial",
    "raw/bingx/5m/.1790700000000.csv.mnbvcxz1.tmp": b"open_ms",
}
NOT_TEMPORARY = {
    "candles/binance/.DS_Store": b"\0",
    "candles/binance/1m.csv.abcdefgh.tmp": b"not hidden",
    "candles/binance/.1m.csv.abc.tmp": b"not a mkstemp name",
    "raw/binance/daily/1m/BTCUSDT-1m-2019-09-01.zip": b"an archive",
    "raw/bingx/5m/1790700000000.csv": b"an archived chunk",
    "forecasts/bingx/x/forecast/.2025-01-06T00Z.npz.abcdefgh.tmp": b"no job lock covers it",
}


def seed(root, files: dict[str, bytes]) -> None:
    for relpath, data in files.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def rows_of(relpaths, lock: str, running: bool = False) -> list[dict]:
    return [
        {"relpath": r, "bytes": len(LEFTOVERS[r]), "lock": lock, "running": running}
        for r in relpaths
    ]


BINANCE_LEFTOVERS = sorted(r for r in LEFTOVERS if "binance" in r)
BINGX_LEFTOVERS = sorted(r for r in LEFTOVERS if "bingx" in r)


def test_leftover_temporaries_are_listed_and_removed_by_the_next_job(tmp_path):
    assert store.LOCKED_RELDIRS == {
        "raw/binance/.lock": ("candles/binance", "candles/binance_official", "raw/binance"),
        "raw/bingx/.lock": ("candles/bingx", "raw/bingx"),
    }
    seed(tmp_path, LEFTOVERS | NOT_TEMPORARY)
    listed = leftover_temporaries(tmp_path)
    assert sorted(listed, key=lambda row: row["relpath"]) == sorted(
        rows_of(BINANCE_LEFTOVERS, "raw/binance/.lock")
        + rows_of(BINGX_LEFTOVERS, "raw/bingx/.lock"),
        key=lambda row: row["relpath"],
    )
    with exclusive_lock(tmp_path / "raw" / "bingx" / ".lock"):  # a BingX run is writing
        running = {row["relpath"]: row["running"] for row in leftover_temporaries(tmp_path)}
        assert running == {r: "bingx" in r for r in LEFTOVERS}
        with pytest.raises(LockBusyError), job_lock("raw/bingx/.lock", tmp_path):
            pass
        with job_lock("raw/binance/.lock", tmp_path) as removed:
            assert sorted(item["relpath"] for item in removed) == BINANCE_LEFTOVERS
            assert removed == [
                {"relpath": r, "bytes": len(LEFTOVERS[r])} for r in BINANCE_LEFTOVERS
            ]
    assert [row["relpath"] for row in leftover_temporaries(tmp_path)] == BINGX_LEFTOVERS
    with job_lock("raw/bingx/.lock", tmp_path) as removed:
        assert sorted(item["relpath"] for item in removed) == BINGX_LEFTOVERS
    assert leftover_temporaries(tmp_path) == []
    assert all((tmp_path / r).read_bytes() == data for r, data in NOT_TEMPORARY.items())
    assert (tmp_path / "raw" / "binance" / ".lock").exists()  # lock files are never removed
    with pytest.raises(ValueError, match="unknown job lock"), job_lock("raw/other/.lock", tmp_path):
        pass


KILLED_WRITER = textwrap.dedent(
    """
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from kronos_agent.data.store import StoreId, StoreWriter, job_lock

    frame = pd.DataFrame({"open_ms": np.array([0, 3_600_000], dtype=np.int64)})
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.Series(["1.0", "1.0"], dtype=object)
    with job_lock("raw/bingx/.lock", Path(sys.argv[1])):
        writer = StoreWriter(StoreId("bingx", "1h"), root=Path(sys.argv[1]))
        writer.append(frame)
        print("writing", flush=True)
        time.sleep(120)
    """
)


def test_a_killed_build_leaves_its_temporary_file_to_the_next_job(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", KILLED_WRITER, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline() == "writing\n", process.stderr.read()
        live = leftover_temporaries(tmp_path)
        assert [row["running"] for row in live] == [True]  # its own live file, not a leftover
        with pytest.raises(LockBusyError), job_lock("raw/bingx/.lock", tmp_path):
            pass
        assert (tmp_path / live[0]["relpath"]).exists()
    finally:
        process.kill()  # SIGKILL: no clean-up runs
        process.communicate(timeout=60)
    assert process.returncode == -signal.SIGKILL
    left = leftover_temporaries(tmp_path)
    assert [(row["relpath"], row["running"]) for row in left] == [(live[0]["relpath"], False)]
    assert left[0]["relpath"].startswith("candles/bingx/.1h.csv.")
    with job_lock("raw/bingx/.lock", tmp_path) as removed:
        assert removed == [{"relpath": left[0]["relpath"], "bytes": left[0]["bytes"]}]
    assert leftover_temporaries(tmp_path) == []
    assert not BINGX_1H.csv_path(tmp_path).exists()  # the killed build never published


# -- the Binance-shaped fixtures (later stages parse these archives) -------------------------------


def test_binance_fixtures_are_store_valid_and_archive_shaped():
    frame = fakes.period_frame("daily", "2020-02-03")
    assert len(frame) == 1440 and frame["open_ms"].iloc[0] == fakes.utc_ms(2020, 2, 3)
    assert validate_frame(frame, StoreId("binance", "1m"))["ohlc_violations"] == 0
    assert "0.000" in set(frame["volume"]) and "0" in set(frame["count"])  # zero-volume minutes
    lines = fakes.kline_csv(frame.iloc[:3], interval="1m").decode("ascii").split("\n")
    assert lines[0] == ",".join(fakes.KLINE_COLUMNS) and lines[-1] == ""
    fields = lines[1].split(",")
    assert len(fields) == 12 and fields[-1] == "0"
    assert int(fields[6]) == int(fields[0]) + 60_000 - 1  # close_time = open + interval - 1 ms
    micro = fakes.kline_csv(frame.iloc[:1], interval="1m", header=False, microseconds=True)
    open_us, close_us = (int(v) for v in micro.decode("ascii").split(",")[0:7:6])
    assert open_us == int(frame["open_ms"].iloc[0]) * 1000 and close_us == open_us + 60_000_000 - 1
    vision = fakes.FakeBinanceVision()
    zip_bytes = vision.add_archive("daily", "1m", "2020-02-03", frame, header=False)
    path = fakes.archive_url_path("daily", "1m", "2020-02-03")
    assert path == "/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2020-02-03.zip"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        assert archive.namelist() == ["BTCUSDT-1m-2020-02-03.csv"]
        member = archive.read("BTCUSDT-1m-2020-02-03.csv").decode("ascii")
    assert member.split("\n")[0].split(",")[1:6] == list(
        frame.iloc[0][["open", "high", "low", "close", "volume"]]
    )
    url = f"https://{fakes.BINANCE_HOST}{path}"
    assert vision(httpx.Request("GET", url)).content == zip_bytes
    checksum = vision(httpx.Request("GET", url + ".CHECKSUM")).text
    assert checksum == f"{hashlib.sha256(zip_bytes).hexdigest()}  BTCUSDT-1m-2020-02-03.zip\n"
    assert vision(httpx.Request("GET", url.replace("02-03", "02-04"))).status_code == 404
    assert vision(httpx.Request("HEAD", url)).status_code == 405
    assert fakes.period_bounds_ms("monthly", "2020-02") == (
        fakes.utc_ms(2020, 2, 1),
        fakes.utc_ms(2020, 3, 1),
    )
