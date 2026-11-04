"""The fold exporter: CSV boundaries (T2), invented future candles (T3), the 9973 rule (T5), the
sealed guard (T7), gaps (T8), the float32 equality (T9), no ``read_store`` (T10), file formats.

Everything runs on synthetic candles under temporary roots: a Binance-like 1h store from
2019-12-31 to 2026-08-20 (``step2_fakes.binance_frame``) for the planned folds, and
``synthetic_frame`` for the rules that need a crafted frame. Nothing touches ``data/``.
"""

from __future__ import annotations

import ast
import copy
import io
import json
import pickle
import zipfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes

from kronos_agent.data import partition
from kronos_agent.data.partition import SEALED_START_MS, SealedPartitionError
from kronos_agent.data.store import (
    StoreBusyError,
    StoreId,
    StoreMissingError,
    job_lock,
    read_study_frame,
)
from kronos_agent.data.timeutil import HOUR_MS, from_ms
from kronos_agent.data.windows import StoreWindowSource, context_frame
from kronos_agent.finetune import export_csv
from kronos_agent.finetune.export_csv import (
    CSV_COLUMNS,
    NLL_INPUT_MEMBERS,
    STORE_COLUMNS,
    TIMESTAMP_FORMAT,
    ZIP_DATE_TIME,
    ExportError,
    ExportRecord,
    ExportRefused,
    export_fold,
    export_frame,
    fold_export_dir,
    nll_inputs_bytes,
    read_export,
    synthetic_frame,
    vendored_float32,
    window_starts,
)
from kronos_agent.finetune.folds import (
    final_fold,
    fold_for,
    monthly_folds,
    quarterly_folds,
    sealed_fold,
    synthetic_fold,
)
from kronos_agent.finetune.recipe import R1, recipe8, recipe_id, train_ratio
from kronos_agent.hashing import canonical_json, sha256_bytes
from kronos_agent.kronos.features import FEATURE_COLUMNS
from kronos_agent.paths import REPO_ROOT, data_dir

STORE_START_MS = fakes.utc_ms(2019, 12, 31)
STORE_END_MS = fakes.utc_ms(2026, 8, 20)
EXPORTED = ("wf-2024-07", "q-2025-01", "wf-2026-06", "s-2026-08", "final-2026-08-01")
SYNTH_START_MS = fakes.utc_ms(2020, 1, 1)
FINETUNE_DIR = REPO_ROOT / "src" / "kronos_agent" / "finetune"
# 17 significant digits that pandas' default parser reads one float64 ulp off (D-023)
VECTOR = "465655238.20240235"


@pytest.fixture(scope="module")
def store_frame() -> pd.DataFrame:
    count = (STORE_END_MS - STORE_START_MS) // HOUR_MS
    frame = fakes.binance_frame(STORE_START_MS, count, interval="1h", seed=11)
    frame.loc[1000, "volume"] = VECTOR
    return frame


@pytest.fixture(scope="module")
def store_root(tmp_path_factory, store_frame) -> Path:
    root = tmp_path_factory.mktemp("binance-store")
    fakes.write_binance_store(root, store_frame)
    return root


@pytest.fixture(scope="module")
def exports(store_root) -> dict[str, ExportRecord]:
    """The representative subset of T2, exported once (no sealed access needed)."""
    return {label: export_fold(fold_for(label, R1), R1, root=store_root) for label in EXPORTED}


def synthetic(hours: int, *, seed: int = 0, recipe=R1, label: str = "synthetic"):
    fold = synthetic_fold(SYNTH_START_MS, SYNTH_START_MS + hours * HOUR_MS, recipe, label=label)
    return fold, synthetic_frame(SYNTH_START_MS, hours, seed=seed)


def without_hours(frame: pd.DataFrame, hours: list[int]) -> pd.DataFrame:
    gone = [SYNTH_START_MS + hour * HOUR_MS for hour in hours]
    return frame[~frame["open_ms"].isin(gone)].reset_index(drop=True)


def csv_opens(data: bytes) -> np.ndarray:
    stamps = pd.to_datetime(pd.read_csv(io.BytesIO(data))["timestamps"])
    return stamps.to_numpy(dtype="datetime64[ms]").astype(np.int64)


def brute_force_windows(opens: np.ndarray, rows: int) -> int:
    return sum(
        1
        for start in range(len(opens) - rows + 1)
        if opens[start + rows - 1] - opens[start] == (rows - 1) * HOUR_MS
    )


# -- T2: the CSV boundary ------------------------------------------------------------------------


def test_t2_boundary_arithmetic_of_every_planned_fold():
    folds = [*monthly_folds(R1), *quarterly_folds(R1)]
    folds += [sealed_fold(f"{m}-01T00:00Z", R1) for m in ("2026-08", "2026-09", "2026-10")]
    folds += [final_fold("2026-08-01T00:00Z", R1), final_fold("2026-11-02T00:00Z", R1)]
    for fold in folds:
        n_rows = (fold.csv_end_ms - fold.csv_start_ms) // HOUR_MS
        n_fit = (fold.inner_validation_start_ms - fold.csv_start_ms) // HOUR_MS
        ratio = train_ratio(n_rows, n_fit)
        train_end, val_end = int(n_rows * ratio), int(n_rows * (ratio + 1.0))
        opens = fold.csv_start_ms + np.arange(n_rows, dtype=np.int64) * HOUR_MS
        assert opens[-1] == fold.csv_end_ms - HOUR_MS
        assert opens[train_end - 1] < fold.inner_validation_start_ms == opens[train_end]
        assert val_end >= n_rows and n_rows - train_end == R1.inner_validation_hours


@pytest.mark.parametrize("label", EXPORTED)
def test_t2_the_csv_holds_the_fold_rows_and_splits_on_the_inner_validation(
    exports, store_frame, label
):
    fold = fold_for(label, R1)
    record = exports[label]
    data = record.csv_path.read_bytes()
    text = data.decode("ascii")
    assert text.endswith("\n") and "\r" not in text and '"' not in text
    lines = text.split("\n")[:-1]
    assert lines[0] == ",".join(CSV_COLUMNS)
    opens = csv_opens(data)
    # no row at or after the CSV end, none before its start (the synthetic store has no gap)
    assert np.array_equal(opens, np.arange(fold.csv_start_ms, fold.csv_end_ms, HOUR_MS))
    rows = len(opens)
    train_end = int(rows * record.train_ratio)
    val_end = int(rows * (record.train_ratio + 1.0))
    assert (opens[:train_end] < fold.inner_validation_start_ms).all()
    assert val_end >= rows
    assert np.array_equal(
        opens[train_end:val_end],
        np.arange(fold.inner_validation_start_ms, fold.csv_end_ms, HOUR_MS),
    )
    assert (record.rows, record.n_fit, record.n_val) == (rows, train_end, rows - train_end)
    selected = store_frame[
        (store_frame["open_ms"] >= fold.csv_start_ms) & (store_frame["open_ms"] < fold.csv_end_ms)
    ]
    cells = [line.split(",") for line in lines[1:]]
    for index, name in enumerate(STORE_COLUMNS, start=1):
        assert [cell[index] for cell in cells] == selected[name].tolist()  # verbatim
    proxy = context_frame(fakes.frame_to_candles(selected), "volume_x_mean_ohlc")["amount"]
    assert [cell[6] for cell in cells] == [repr(value) for value in proxy.tolist()]
    assert cells[0][0] == from_ms(fold.csv_start_ms).strftime(TIMESTAMP_FORMAT)


def test_t2_the_pilot_export_has_the_checked_numbers(exports, store_root):
    record = exports["wf-2024-07"]
    content = record.content
    assert (record.rows, record.n_fit, record.n_val) == (38_920, 37_576, 1_344)
    assert record.train_ratio == train_ratio(38_920, 37_576)
    assert (content["fit_windows"], content["fit_windows_kept"]) == (37_064, 37_064)
    assert (content["val_windows"], content["val_windows_kept"]) == (832, 832)
    assert (record.fit_windows_kept, record.val_windows_kept) == (37_064, 832)
    assert content["nll"]["planned"] == content["nll"]["kept"] == 336
    assert content["nll"]["dropped"] == ()
    assert not record.gap_aware and content["gaps"] == () and content["missing_candles"] == 0
    assert content["float32_check"] == {"rows": 38_920, "mismatches": 0}
    assert (content["csv_start"], content["csv_end"]) == ("2020-01-01T00:00Z", "2024-06-09T16:00Z")
    assert content["inner_validation_start"] == "2024-04-14T16:00Z"
    assert (content["fold"], content["layout"]) == ("wf-2024-07", "monthly")
    assert content["recipe_id"] == recipe_id(R1)
    assert (content["val_ratio"], content["test_ratio"], content["window_rows"]) == (1.0, 0.0, 513)
    assert record.directory == fold_export_dir(recipe8(R1), "wf-2024-07", root=store_root)
    assert record.nll_decisions_ms == fold_for("wf-2024-07", R1).nll_decisions_ms()


def test_nll_inputs_are_the_inference_floats_of_each_decision(exports, store_root):
    record = exports["wf-2024-07"]
    with zipfile.ZipFile(record.nll_inputs_path) as archive:
        infos = archive.infolist()
    assert [info.filename for info in infos] == list(NLL_INPUT_MEMBERS)
    assert all(info.compress_type == zipfile.ZIP_STORED for info in infos)
    assert all(info.date_time == ZIP_DATE_TIME for info in infos)
    with np.load(record.nll_inputs_path, allow_pickle=False) as npz:
        assert npz.files == ["decision_ms", "values", "meta"]
        decisions, values = npz["decision_ms"], npz["values"]
        meta = json.loads(str(npz["meta"][()]))
    assert decisions.dtype == np.int64 and tuple(decisions.tolist()) == record.nll_decisions_ms
    assert values.dtype == np.float64 and values.shape == (336, 516, 6)
    assert meta == {
        "schema_version": 1,
        "context": 512,
        "steps": 4,
        "columns": list(FEATURE_COLUMNS),
        "amount_policy": "volume_x_mean_ohlc",
        "decisions": 336,
        "grid_hours": 4,
    }
    source = StoreWindowSource("binance", root=store_root)
    for index in (0, 167, 335):
        decision = int(decisions[index])
        window = source.window(from_ms(decision), horizon=4)
        assert values[index, :512].tobytes() == window.context.to_numpy(np.float64).tobytes()
        targets = source.candles_between(decision, decision + 4 * HOUR_MS)
        expected = context_frame(targets, "volume_x_mean_ohlc").to_numpy(np.float64)
        assert values[index, 512:].tobytes() == expected.tobytes()


def test_export_fold_records_the_store_meta_it_read(exports, store_root):
    record = exports["wf-2024-07"]
    meta_bytes = StoreId("binance", "1h").meta_path(store_root).read_bytes()
    meta = json.loads(meta_bytes)
    assert record.provenance["store"] == {
        "relpath": "candles/binance/1h.csv",
        "meta_sha256": sha256_bytes(meta_bytes),
        "csv_sha256": meta["csv_sha256"],
        "rows": meta["rows"],
    }
    assert record.store_meta_sha256 == sha256_bytes(meta_bytes)
    assert set(record.provenance["code"]) == {"commit", "dirty", "source_tree_sha256"}
    text = record.export_json_path.read_text(encoding="ascii")
    assert str(store_root) not in text and str(Path.home()) not in text
    document = json.loads(text)
    assert set(document) == {"schema_version", "content", "content_sha256", "provenance"}
    assert text == json.dumps(document, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


# -- T3: invented future candles -----------------------------------------------------------------


def altered_after(frame: pd.DataFrame, cut_ms: int) -> pd.DataFrame:
    """Every price after ``cut_ms`` times 10 and every volume times 0.1 (plain decimals)."""
    out = frame.copy()
    later = out["open_ms"] >= cut_ms
    for name in ("open", "high", "low", "close"):
        out.loc[later, name] = [format(Decimal(v) * 10, "f") for v in out.loc[later, name]]
    out.loc[later, "volume"] = [format(Decimal(v) / 10, "f") for v in out.loc[later, "volume"]]
    return out


def test_t3_invented_future_candles_change_only_the_store_provenance(
    tmp_path, exports, store_frame, store_root
):
    fold = fold_for("wf-2024-07", R1)
    root = tmp_path / "altered"
    fakes.write_binance_store(root, altered_after(store_frame, fold.csv_end_ms))
    # positive control: the candles after the CSV end did change
    window = {"start_ms": fold.csv_end_ms, "end_ms": fold.csv_end_ms + 48 * HOUR_MS}
    store_id = StoreId("binance", "1h")
    before = read_study_frame(store_id, numeric=False, root=store_root, **window)
    after = read_study_frame(store_id, numeric=False, root=root, **window)
    assert (before["open_ms"] == after["open_ms"]).all()
    assert (before["close"] != after["close"]).all()
    altered = export_fold(fold, R1, root=root)
    base = exports["wf-2024-07"]
    assert altered.csv_path.read_bytes() == base.csv_path.read_bytes()
    assert altered.nll_inputs_path.read_bytes() == base.nll_inputs_path.read_bytes()
    assert altered.content == base.content and altered.content_sha256 == base.content_sha256
    assert altered.store_meta_sha256 != base.store_meta_sha256
    assert altered.provenance["store"]["csv_sha256"] != base.provenance["store"]["csv_sha256"]


# -- T7: the sealed guard ------------------------------------------------------------------------


def test_t7_a_csv_reading_sealed_candles_needs_the_owners_order(
    tmp_path, store_root, exports, monkeypatch
):
    september = fold_for("s-2026-09", R1)
    assert september.needs_sealed_access
    out = tmp_path / "s-2026-09"
    with pytest.raises(SealedPartitionError):
        export_fold(september, R1, root=store_root, out_dir=out)
    with pytest.raises(SealedPartitionError):
        export_fold(final_fold("2026-08-02T00:00Z", R1), R1, root=store_root, out_dir=out)
    assert not out.exists()
    monkeypatch.setattr(partition, "_verify_order", lambda order_id: None)
    access = partition.sealed_access("recorded-order-for-the-test", data_root=store_root)
    record = export_fold(september, R1, sealed_access=access, root=store_root, out_dir=out)
    assert record.content["last_open_ms"] == september.csv_end_ms - HOUR_MS > SEALED_START_MS
    # the first sealed fold and the model to date at 2026-08-01 read no sealed candle
    for label in ("s-2026-08", "final-2026-08-01"):
        assert exports[label].content["last_open_ms"] < SEALED_START_MS


# -- T5: the 9973 rule ---------------------------------------------------------------------------

GRID1 = replace(R1, name="grid1", nll_grid_hours=1, inner_validation_hours=520, min_nll_decisions=8)


def test_t5_a_fit_window_count_that_is_a_multiple_of_9973_is_refused(tmp_path):
    hours = 9973 + 512 + 520  # fit rows 10,485: 9,973 fit windows
    fold, frame = synthetic(hours, recipe=GRID1)
    out = tmp_path / "multiple"
    with pytest.raises(ExportRefused, match="9973"):
        export_frame(frame, fold, GRID1, out)
    assert not out.exists()
    fold, frame = synthetic(hours + 1, recipe=GRID1)
    record = export_frame(frame, fold, GRID1, tmp_path / "next")
    assert record.fit_windows_kept == 9974


def test_t5_the_kept_count_of_a_gap_aware_fold_is_checked(tmp_path):
    fold, frame = synthetic(11_832)  # 10,488 fit hours; without hour 2: 9,975 windows, 9,973 kept
    with pytest.raises(ExportRefused, match="9973 fit windows"):
        export_frame(without_hours(frame, [2]), fold, R1, tmp_path / "gap")


def test_t5_the_validation_count_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(export_csv, "VENDORED_INDEX_PRIME", 13)  # 832 = 13 x 64; 2,240 is not
    fold, frame = synthetic(4096)
    with pytest.raises(ExportRefused, match="832 inner-validation windows"):
        export_frame(frame, fold, R1, tmp_path / "val")


# -- T8: gaps ------------------------------------------------------------------------------------


def test_t8_windows_spanning_a_gap_are_dropped_and_nothing_is_fabricated(tmp_path):
    hours = 21_344  # 20,000 fit hours, then the inner validation
    missing = [10_000, 20_700]  # one fit hour, one inner-validation hour
    fold, full = synthetic(hours, seed=3)
    frame = without_hours(full, missing)
    record = export_frame(frame, fold, R1, tmp_path / "gaps")
    content = record.content
    data = record.csv_path.read_bytes()
    assert np.array_equal(csv_opens(data), frame["open_ms"].to_numpy())  # no filled row
    assert content["gaps"] == tuple(
        (SYNTH_START_MS + h * HOUR_MS, SYNTH_START_MS + (h + 1) * HOUR_MS) for h in missing
    )
    assert content["missing_candles"] == 2 and record.rows == hours - 2
    opens = frame["open_ms"].to_numpy()
    n_fit = record.n_fit
    assert n_fit == 19_999 and record.n_val == 1_343
    assert content["fit_windows"] == n_fit - 512
    assert content["fit_windows_kept"] == brute_force_windows(opens[:n_fit], 513) == 18_975
    assert content["fit_windows_dropped"] == content["fit_windows"] - 18_975 == 512
    assert content["val_windows_kept"] == brute_force_windows(opens[n_fit:], 513)
    assert content["val_windows_dropped"] == content["val_windows"] - content["val_windows_kept"]
    assert record.gap_aware and content["fit_dropped_share"] == 512 / content["fit_windows"]
    gone = SYNTH_START_MS + missing[1] * HOUR_MS
    expected = [d for d in fold.nll_decisions_ms() if d - 512 * HOUR_MS <= gone < d + 4 * HOUR_MS]
    assert [item["decision_ms"] for item in content["nll"]["dropped"]] == expected
    assert all(item["missing"] == ((gone, gone + HOUR_MS),) for item in content["nll"]["dropped"])
    assert content["nll"]["kept"] == 336 - len(expected) == 207
    with np.load(record.nll_inputs_path, allow_pickle=False) as npz:
        kept = npz["decision_ms"].tolist()
        values = npz["values"]
    assert kept == [d for d in fold.nll_decisions_ms() if d not in expected]
    candles = fakes.frame_to_candles(frame)
    by_open = {c.open_ms: c for c in candles}
    sample = kept[-1]
    rows = [by_open[sample + k * HOUR_MS] for k in range(-512, 4)]
    reference = context_frame(rows, "volume_x_mean_ohlc").to_numpy(np.float64)
    assert values[-1].tobytes() == reference.tobytes()


def test_t8_a_fold_losing_more_than_5_percent_of_its_fit_windows_is_refused(tmp_path):
    fold, full = synthetic(21_344)
    with pytest.raises(ExportRefused, match=r"1024 of 19486 fit windows \(5\.26%\)"):
        export_frame(without_hours(full, [5_000, 10_000]), fold, R1, tmp_path / "a")
    record = export_frame(without_hours(full, [10_000]), fold, R1, tmp_path / "b")
    assert record.content["fit_windows_dropped"] == 512


def test_t8_too_few_complete_nll_decisions_are_refused(tmp_path):
    fold, full = synthetic(21_344)
    out = tmp_path / "nll"
    with pytest.raises(ExportRefused, match="96 of 336 NLL4-TF decisions"):
        export_frame(without_hours(full, [20_300, 20_900]), fold, R1, out)
    assert not out.exists()


def test_a_fold_without_enough_rows_on_either_side_is_refused(tmp_path):
    fold, full = synthetic(4096)  # fit hours [0, 2752), inner validation [2752, 4096)
    with pytest.raises(ExportRefused, match="452 fit rows .* more than one window"):
        export_frame(full[full["open_ms"] >= SYNTH_START_MS + 2300 * HOUR_MS], fold, R1, tmp_path)
    with pytest.raises(ExportRefused, match="513 inner-validation rows .* more than one window"):
        export_frame(without_hours(full, list(range(3265, 4096))), fold, R1, tmp_path)
    # no stretch of the inner validation holds 513 consecutive hours
    with pytest.raises(ExportRefused, match="no inner-validation window"):
        export_frame(without_hours(full, [3200, 3700]), fold, R1, tmp_path)
    assert list(tmp_path.iterdir()) == []


# -- T9: the amount and the float32 equality -----------------------------------------------------


def test_t9_the_float32_check_holds_while_the_default_parser_misreads_float64(exports):
    record = exports["wf-2024-07"]
    data = record.csv_path.read_bytes()
    texts = pd.read_csv(io.BytesIO(data), dtype=str)
    default = pd.read_csv(io.BytesIO(data))
    for column in ("amount", "volume"):
        exact = np.array([float(text) for text in texts[column]])
        parsed = default[column].to_numpy(dtype=np.float64)
        assert (parsed != exact).any(), column  # pandas' default parser is not correctly rounded
        f32 = exact.astype(np.float32).view(np.uint32)
        assert (parsed.astype(np.float32).view(np.uint32) == f32).all(), column
    assert VECTOR in texts["volume"].tolist()
    assert record.content["float32_check"] == {"rows": record.rows, "mismatches": 0}
    assert vendored_float32(record.csv_path).shape == (record.rows, 6)


def tie_frame() -> tuple:
    """A synthetic fold whose row 1000 has an amount exactly halfway between two float32."""
    fold, frame = synthetic(4096)
    for name in ("open", "high", "low", "close"):
        frame.loc[1000, name] = "16777217"
    frame.loc[1000, "volume"] = "1"
    return fold, frame


def test_t9_a_one_ulp_change_of_one_amount_at_a_float32_tie_is_refused(tmp_path, monkeypatch):
    fold, frame = tie_frame()
    tie = 16777217.0  # 2**24 + 1: halfway between two float32, rounds to 2**24
    up = float(np.nextafter(tie, np.inf))  # one float64 ulp above: rounds up instead
    assert np.float32(tie) == np.float32(2.0**24) != np.float32(up)
    record = export_frame(frame, fold, R1, tmp_path / "honest")
    assert record.content["float32_check"]["mismatches"] == 0
    monkeypatch.setattr(
        export_csv, "_amount_text", lambda value: repr(up) if value == tie else repr(value)
    )
    out = tmp_path / "perturbed"
    with pytest.raises(ExportRefused, match=r"1 CSV rows .* \(amount\)"):
        export_frame(frame, fold, R1, out)
    assert not out.exists()


def test_t9_a_one_ulp_change_away_from_a_tie_keeps_the_float32(tmp_path, monkeypatch):
    fold, frame = synthetic(4096)
    honest = export_frame(frame, fold, R1, tmp_path / "honest")
    target = float(pd.read_csv(honest.csv_path, dtype=str)["amount"][2000])
    up = float(np.nextafter(target, np.inf))
    assert np.float32(up) == np.float32(target)
    monkeypatch.setattr(
        export_csv, "_amount_text", lambda value: repr(up) if value == target else repr(value)
    )
    record = export_frame(frame, fold, R1, tmp_path / "perturbed")
    assert record.content["float32_check"]["mismatches"] == 0
    assert record.csv_sha256 != honest.csv_sha256


# -- T10: no read_store --------------------------------------------------------------------------


def read_store_mentions(tree: ast.AST) -> list[int]:
    """Lines where the syntax tree names ``read_store`` (call, attribute, import, string)."""
    found = []
    for node in ast.walk(tree):
        names: list[object] = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.alias):
            names = [node.name.rsplit(".", 1)[-1], node.asname]
        elif isinstance(node, (ast.arg, ast.keyword)):
            names = [node.arg]
        elif isinstance(node, ast.Constant):
            names = [node.value]
        if "read_store" in names:
            found.append(getattr(node, "lineno", 0))
    return found


def test_t10_no_finetune_module_names_read_store():
    files = sorted(FINETUNE_DIR.glob("*.py"))
    assert {"__init__.py", "recipe.py", "folds.py", "export_csv.py"} <= {f.name for f in files}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        assert read_store_mentions(tree) == [], path.name


@pytest.mark.parametrize(
    "snippet",
    [
        "from ..data.store import read_store",
        "from ..data.store import read_store as reader",
        "from ..data import store\nstore.read_store(x)",
        "import kronos_agent.data.store as read_store",
        "getattr(store, 'read_store')(x)",
        "f(read_store=1)",
    ],
)
def test_t10_the_scan_sees_every_form(snippet):
    assert read_store_mentions(ast.parse(snippet))


# -- publication: never overwritten, verified on read --------------------------------------------


def test_an_identical_export_is_kept_and_another_one_refused(tmp_path):
    fold, frame = synthetic(4096)
    out = tmp_path / "export"
    first = export_frame(frame, fold, R1, out)
    names = ("train.csv", "nll_inputs.npz", "export.json")
    before = {name: ((out / name).read_bytes(), (out / name).stat().st_mtime_ns) for name in names}
    again = export_frame(frame, fold, R1, out)
    assert again == first  # the first provenance is kept
    assert {n: ((out / n).read_bytes(), (out / n).stat().st_mtime_ns) for n in names} == before
    other = frame.copy()
    other.loc[100, "volume"] = "1.0000"
    with pytest.raises(ExportRefused, match="never overwritten"):
        export_frame(other, fold, R1, out)
    assert {n: ((out / n).read_bytes(), (out / n).stat().st_mtime_ns) for n in names} == before


def test_an_interrupted_export_completes_only_over_identical_bytes(tmp_path):
    fold, frame = synthetic(4096)
    reference = export_frame(frame, fold, R1, tmp_path / "reference")
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    (resumed / "train.csv").write_bytes(reference.csv_path.read_bytes())
    record = export_frame(frame, fold, R1, resumed)
    assert record.content_sha256 == reference.content_sha256
    clashing = tmp_path / "clashing"
    clashing.mkdir()
    (clashing / "train.csv").write_bytes(b"timestamps\n")
    with pytest.raises(ExportRefused, match="exists with other bytes"):
        export_frame(frame, fold, R1, clashing)
    assert not (clashing / "export.json").exists()


def test_read_export_reverifies_every_file(tmp_path):
    fold, frame = synthetic(4096)
    record = export_frame(frame, fold, R1, tmp_path / "export")
    assert read_export(record.directory) == record
    assert sha256_bytes(canonical_json(record.content)) == record.content_sha256
    assert json.loads(json.dumps(record.content)) == json.loads(canonical_json(record.content))
    with pytest.raises(TypeError):
        record.content["rows"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        record.content["nll"]["kept"] = 1  # type: ignore[index]
    assert copy.deepcopy(record) == record and pickle.loads(pickle.dumps(record)) == record
    csv = record.csv_path
    data = csv.read_bytes()
    csv.write_bytes(data[:-2] + (b"0\n" if data[-2:] != b"0\n" else b"1\n"))
    with pytest.raises(ExportError, match="sha256 differs"):
        read_export(record.directory)
    csv.write_bytes(data + b"\n")
    with pytest.raises(ExportError, match="size differs"):
        read_export(record.directory)
    csv.write_bytes(data)
    document = json.loads(record.export_json_path.read_text("ascii"))
    document["content"]["rows"] += 1
    record.export_json_path.write_text(json.dumps(document), "ascii")
    with pytest.raises(ExportError, match="content_sha256"):
        read_export(record.directory)
    record.nll_inputs_path.unlink()
    with pytest.raises(ExportError):
        read_export(record.directory)
    with pytest.raises(ExportError, match="cannot read"):
        read_export(tmp_path / "absent")


def test_the_record_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fold, frame = synthetic(4096)
    record = export_frame(frame, fold, R1, Path("relative"))
    assert record.directory == tmp_path.absolute() / "relative"
    assert record.csv_path.is_absolute() and record.nll_inputs_path.is_absolute()


# -- export_fold and its inputs ------------------------------------------------------------------


def test_export_fold_refuses_while_a_data_run_holds_the_job_lock(store_root, tmp_path):
    out = tmp_path / "busy"
    with job_lock("raw/binance/.lock", store_root), pytest.raises(StoreBusyError):
        export_fold(fold_for("wf-2024-07", R1), R1, root=store_root, out_dir=out)
    assert not out.exists()


def test_export_fold_needs_a_built_store(tmp_path):
    with pytest.raises(StoreMissingError):
        export_fold(fold_for("wf-2024-07", R1), R1, root=tmp_path)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda f: f.iloc[::-1].reset_index(drop=True), id="decreasing"),
        pytest.param(lambda f: pd.concat([f, f.iloc[[5]]]).sort_values("open_ms"), id="duplicate"),
        pytest.param(lambda f: f.assign(open_ms=f["open_ms"] + 1), id="off-the-hour"),
        pytest.param(lambda f: f.assign(open_ms=f["open_ms"] + HOUR_MS), id="row-at-csv-end"),
        pytest.param(lambda f: f.assign(open_ms=f["open_ms"] - HOUR_MS), id="row-before-start"),
        pytest.param(lambda f: f.assign(open_ms=f["open_ms"].astype(np.int32)), id="int32"),
        pytest.param(lambda f: f.drop(columns="volume"), id="no-volume"),
        pytest.param(lambda f: f.assign(close=f["close"].astype(float)), id="floats"),
        pytest.param(lambda f: f.assign(open=f["open"].str.replace(".", ",")), id="comma"),
        pytest.param(lambda f: f.assign(high="1e5"), id="exponent"),
        pytest.param(lambda f: f.assign(volume="1" + "0" * 400), id="infinite-float"),
    ],
)
def test_malformed_frames_are_errors_not_refusals(tmp_path, mutate):
    fold, frame = synthetic(4096)
    with pytest.raises(ExportError) as info:
        export_frame(mutate(frame), fold, R1, tmp_path / "out")
    assert not isinstance(info.value, ExportRefused)
    assert not (tmp_path / "out").exists()


def test_export_frame_checks_its_other_inputs(tmp_path):
    fold, frame = synthetic(4096)
    with pytest.raises(ExportError):
        export_frame(frame, fold, R1, tmp_path / "a", store={"relpath": "x"})
    with pytest.raises(TypeError):
        export_frame(frame, fold, R1.name, tmp_path / "b")
    other = replace(R1, nll_grid_hours=8)
    with pytest.raises(ValueError):
        export_frame(frame, fold, other, tmp_path / "c")  # the fold was planned with R1


def test_fold_export_dir(tmp_path):
    expected = tmp_path / "finetune" / "0ee0d123" / "wf-2024-07"
    assert fold_export_dir("0ee0d123", "wf-2024-07", root=tmp_path) == expected
    assert fold_export_dir("0ee0d123", "synthetic") == data_dir() / "finetune/0ee0d123/synthetic"
    for recipe8_text, label in [
        ("0EE0D123", "wf-2024-07"),
        ("0ee0d12", "wf-2024-07"),
        ("0ee0d123", "wf-2024-07-r1"),
        ("0ee0d123", "../wf-2024-07"),
    ]:
        with pytest.raises(ExportError):
            fold_export_dir(recipe8_text, label, root=tmp_path)


# -- helpers ---------------------------------------------------------------------------------------


def test_synthetic_frame_is_deterministic_and_store_like():
    frame = synthetic_frame(SYNTH_START_MS, 600, seed=4)
    assert list(frame.columns) == ["open_ms", *STORE_COLUMNS]
    assert frame["open_ms"].dtype == np.int64
    expected_opens = SYNTH_START_MS + np.arange(600) * HOUR_MS
    assert np.array_equal(frame["open_ms"].to_numpy(), expected_opens)
    for name in STORE_COLUMNS:
        decimals = 4 if name == "volume" else 2
        assert frame[name].str.fullmatch(rf"[0-9]+\.[0-9]{{{decimals}}}").all(), name
    assert synthetic_frame(SYNTH_START_MS, 600, seed=4).equals(frame)
    assert not synthetic_frame(SYNTH_START_MS, 600, seed=5).equals(frame)
    for bad in [(SYNTH_START_MS + 1, 10), (SYNTH_START_MS, 0), (True, 10), (SYNTH_START_MS, 1.5)]:
        with pytest.raises(ExportError):
            synthetic_frame(*bad)


def test_window_starts_match_a_brute_force_count():
    rng = np.random.default_rng(0)
    opens = np.sort(rng.choice(2000, size=1800, replace=False)).astype(np.int64) * HOUR_MS
    for rows in (1, 2, 5, 50, 513):
        expected = [
            start
            for start in range(len(opens) - rows + 1)
            if opens[start + rows - 1] - opens[start] == (rows - 1) * HOUR_MS
        ]
        starts = window_starts(opens, rows)
        assert starts.dtype == np.int64 and starts.tolist() == expected
    assert window_starts(opens[:3], 5).size == 0
    for bad_rows in (0, True):
        with pytest.raises(ExportError):
            window_starts(opens, bad_rows)


def test_vendored_float32_parses_sorts_and_casts_as_the_vendored_dataset(tmp_path):
    csv = (
        b"timestamps,open,high,low,close,volume,amount\n"
        b"2024-01-01 01:00:00,7225,7230.5,7200,7210.10," + VECTOR.encode() + b",0.0\n"
        b"2024-01-01 00:00:00,1,2,0.5,1.5,0.001,1.5e-05\n"
    )
    values = vendored_float32(csv)
    assert values.dtype == np.float32 and values.shape == (2, 6)
    assert values[0].tolist() == np.array([1, 2, 0.5, 1.5, 0.001, 1.5e-05], np.float32).tolist()
    assert values[1, 4] == np.float32(float(VECTOR))
    path = tmp_path / "train.csv"
    path.write_bytes(csv)
    assert vendored_float32(path).tobytes() == values.tobytes()


def test_nll_inputs_bytes_are_deterministic_and_load_without_pickle():
    base = SYNTH_START_MS + 4000 * HOUR_MS
    decisions = np.array([base, base + 4 * HOUR_MS, base + 12 * HOUR_MS], dtype=np.int64)
    values = np.random.default_rng(1).normal(size=(3, 516, 6))
    data = nll_inputs_bytes(decisions, values)
    assert data == nll_inputs_bytes(decisions.copy(), values.copy())
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        assert np.array_equal(npz["decision_ms"], decisions)
        assert npz["values"].tobytes() == values.tobytes()
        assert json.loads(str(npz["meta"][()]))["decisions"] == 3
    hourly = nll_inputs_bytes(decisions + HOUR_MS, values, grid_hours=1)
    with np.load(io.BytesIO(hourly), allow_pickle=False) as npz:
        assert json.loads(str(npz["meta"][()]))["grid_hours"] == 1
    empty = nll_inputs_bytes(np.empty(0, np.int64), np.empty((0, 516, 6)))
    with np.load(io.BytesIO(empty), allow_pickle=False) as npz:
        assert npz["values"].shape == (0, 516, 6)
    bad_inputs = [
        (decisions[::-1], values),
        (decisions + HOUR_MS, values),  # off the 4-hour grid
        (np.array([base, base, base]), values),
        (decisions.astype(float), values),
        (decisions, values[:, :512]),
        (decisions, values.astype(np.float32)),
        (decisions, np.where(values > 2, np.nan, values)),
    ]
    for bad_decisions, bad_values in bad_inputs:
        with pytest.raises(ExportError):
            nll_inputs_bytes(bad_decisions, bad_values)
    with pytest.raises(ExportError):
        nll_inputs_bytes(decisions, values, grid_hours=True)
