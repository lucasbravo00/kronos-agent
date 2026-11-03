"""Fold CSVs for the vendored ``finetune_csv`` and the NLL4-TF inputs (step 6; design 1.5-1.7).

One core, :func:`export_frame`, turns a fold's hourly candles (the store's decimal strings) into
the three files of ``<data>/finetune/<recipe8>/<fold label>/``:

- ``train.csv``: ``timestamps,open,high,low,close,volume,amount``, one row per candle of
  ``[csv_start, csv_end)``, gaps left as missing rows; ``timestamps`` is the candle's OPEN time
  (tz-naive ``YYYY-MM-DD HH:MM:SS``, so the vendored time features equal the inference ones,
  FB:55-64, K:472-479); OHLCV are the store's strings verbatim; ``amount`` is ``repr`` of the
  float64 proxy ``volume * mean(open, high, low, close)`` that the inference window computes
  (``data.windows.context_frame`` under ``volume_x_mean_ohlc``);
- ``nll_inputs.npz``: the float64 channels of the 516 candles (512 context rows, 4 targets) of
  every NLL4-TF decision of the inner validation, in deterministic bytes, so the training child
  never builds windows or reads a store;
- ``export.json``: what was exported and checked (``content``, whose sha256 identifies the
  export and depends only on the candles before the CSV end, the fold and the recipe) and where
  it came from (``provenance``: the store meta read, the code, the time). It is the commit
  marker, written last.

Before anything is published the exporter checks, and refuses with :class:`ExportRefused` (a
pre-registered data rule; the owner is told) or :class:`ExportError` (malformed input):

1. the frame: opens strictly increasing, on the hour, inside ``[csv_start, csv_end)``, values
   plain decimals (no row at or after the CSV end can enter);
2. the split: ``n_fit`` rows open before the inner validation, and both parts hold more rows
   than a vendored window (FB:108-110 raises otherwise); ``train_ratio = (n_fit + 0.5) / N``
   must make the vendored split arithmetic land on that row (FB:75-97);
3. the float32 equality (design 1.5): the vendored reader parses the CSV with pandas' default
   float parser, which is not correctly rounded, then casts to float32 (FB:53, FB:122); every
   value must equal, bit for bit, the float32 of the inference floats;
4. gaps (design 1.6): windows of ``window_rows`` rows that span a missing hour are dropped and
   counted, separately in the fit rows and in the inner validation; a fold dropping more than
   ``max_dropped_fit_share`` of its fit windows is refused;
5. the vendored index mapping ``(idx * 9973 + ...) mod n`` (FB:112-114) is a bijection only when
   9973 does not divide ``n``: a fit or validation window count that is a multiple is refused;
6. NLL4-TF decisions whose 516 candles miss one are dropped with their missing ranges; fewer than
   ``min_nll_decisions`` kept is refused.

:func:`export_fold` reads the fold's candles through ``read_study_frame`` (behind the sealed
guard: a CSV end after 2026-08-01T00:00Z needs a ``SealedAccess``) under one hold of the
store's read lock, and records the store meta it read. Exports are never overwritten: an
existing export with the same content is kept (identical files), anything else is refused.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.fsutil import publish_new
from ..data.partition import MAX_CONTEXT, SealedAccess, require_study_range
from ..data.store import (
    DECIMAL_RE,
    StoreId,
    find_gaps,
    read_meta,
    read_study_frame,
    store_read_lock,
)
from ..data.timeutil import HOUR_MS, from_ms, utc_text
from ..data.windows import context_frame
from ..exchange.public_market import Candle
from ..hashing import canonical_json, sha256_bytes, sha256_file
from ..kronos.features import FEATURE_COLUMNS, synthetic_window
from ..paths import data_dir
from ..provenance import code_version
from .folds import FOLD_LABEL_RE, Fold, check_fold
from .recipe import NLL_STEPS, R1, Recipe, RecipeError, recipe_id, train_ratio
from .recipe import recipe8 as recipe8_of

CSV_NAME = "train.csv"
NLL_INPUTS_NAME = "nll_inputs.npz"
EXPORT_JSON_NAME = "export.json"
EXPORTS_DIRNAME = "finetune"  # <data>/finetune/<recipe8>/<fold label>/
CSV_COLUMNS: tuple[str, ...] = ("timestamps", "open", "high", "low", "close", "volume", "amount")
STORE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPORT_SCHEMA_VERSION = 1
NLL_INPUTS_SCHEMA_VERSION = 1
VENDORED_INDEX_PRIME = 9973
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
NLL_INPUT_MEMBERS: tuple[str, ...] = ("decision_ms.npy", "values.npy", "meta.npy")
NLL_AMOUNT_POLICY = "volume_x_mean_ohlc"

_DECIMAL = re.compile(DECIMAL_RE)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_RECIPE8 = re.compile(r"[0-9a-f]{8}")
_DOCUMENT_KEYS = frozenset({"schema_version", "content", "content_sha256", "provenance"})
_PROVENANCE_KEYS = frozenset({"store", "code", "created_at"})
_STORE_KEYS = frozenset({"relpath", "meta_sha256", "csv_sha256", "rows"})


class ExportError(ValueError):
    """Malformed input, an unreadable or inconsistent export, an I/O problem (exit 1)."""


class ExportRefused(ExportError):
    """A pre-registered data rule refuses the fold (exit 2); the owner is told."""


class _ReadOnlyDict(dict):
    """A dict that refuses changes; JSON-serialisable, unlike a ``mappingproxy``."""

    def _refuse(self, *args: object, **kwargs: object) -> Any:
        raise TypeError("an export record is read-only")

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self) -> tuple[Any, ...]:
        return (_ReadOnlyDict, (dict(self),))


def _frozen(value: Any) -> Any:
    """``value`` with every dict read-only and every list a tuple (still JSON-serialisable)."""
    if isinstance(value, Mapping):
        return _ReadOnlyDict((key, _frozen(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


@dataclass(frozen=True)
class ExportRecord:
    """A verified export (:func:`read_export`); the scalar fields are views of ``content``."""

    directory: Path
    csv_path: Path
    nll_inputs_path: Path
    export_json_path: Path
    content: Mapping[str, Any]
    content_sha256: str
    provenance: Mapping[str, Any]
    csv_sha256: str
    nll_inputs_sha256: str
    rows: int
    n_fit: int
    n_val: int
    train_ratio: float
    gap_aware: bool
    fit_windows_kept: int
    val_windows_kept: int
    nll_decisions_ms: tuple[int, ...]
    store_meta_sha256: str | None


def fold_export_dir(recipe8: str, fold_label: str, *, root: Path | None = None) -> Path:
    """``<data root>/finetune/<recipe8>/<fold label>`` (replicates share their fold's export)."""
    if not isinstance(recipe8, str) or not _RECIPE8.fullmatch(recipe8):
        raise ExportError(f"recipe8 must be 8 lowercase hex characters, got {recipe8!r}")
    if not isinstance(fold_label, str) or FOLD_LABEL_RE.fullmatch(fold_label) is None:
        raise ExportError(f"not a fold label (no replicate suffix): {fold_label!r}")
    base = Path(root) if root is not None else data_dir()
    return base / EXPORTS_DIRNAME / recipe8 / fold_label


def export_fold(
    fold: Fold,
    recipe: Recipe,
    *,
    sealed_access: SealedAccess | None = None,
    root: Path | None = None,
    out_dir: Path | None = None,
) -> ExportRecord:
    """Export ``fold`` from the recipe's store (``candles/binance/1h``) under the data ``root``.

    The sealed guard applies before anything is read. Under one hold of the store's read lock
    the meta file's bytes are read (their sha256 is recorded as ``store.meta_sha256``), then the
    fold's rows through ``read_study_frame``, which verifies the CSV against that meta. The
    export goes to ``out_dir`` or :func:`fold_export_dir`.
    """
    if not isinstance(recipe, Recipe):
        raise TypeError(f"expected a Recipe, got {type(recipe).__name__}")
    check_fold(fold, recipe=recipe)
    require_study_range(fold.csv_start_ms, fold.csv_end_ms, sealed_access)
    store_id = StoreId(recipe.source, recipe.interval)
    with store_read_lock(store_id, root):
        meta = read_meta(store_id, root)
        meta_bytes = store_id.meta_path(root).read_bytes()
        if json.loads(meta_bytes) != meta:
            raise ExportError(f"{store_id.meta_relpath} changed while it was read; run again")
        frame = read_study_frame(
            store_id,
            start_ms=fold.csv_start_ms,
            end_ms=fold.csv_end_ms,
            numeric=False,
            usecols=STORE_COLUMNS,
            sealed_access=sealed_access,
            root=root,
        )
        if store_id.meta_path(root).read_bytes() != meta_bytes:
            raise ExportError(f"{store_id.relpath} was rebuilt while it was read; run again")
    store = {
        "relpath": store_id.relpath,
        "meta_sha256": sha256_bytes(meta_bytes),
        "csv_sha256": meta["csv_sha256"],
        "rows": meta["rows"],
    }
    if out_dir is None:
        out_dir = fold_export_dir(recipe8_of(recipe), fold.label, root=root)
    return export_frame(frame, fold, recipe, Path(out_dir), store=store)


# -- the core -------------------------------------------------------------------------------------


def _check_store(store: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if store is None:
        return None
    if not isinstance(store, Mapping) or set(store) != _STORE_KEYS:
        raise ExportError(f"store must be None or a mapping of exactly {sorted(_STORE_KEYS)}")
    if not isinstance(store["relpath"], str) or not all(
        isinstance(store[key], str) and _SHA256_HEX.fullmatch(store[key])
        for key in ("meta_sha256", "csv_sha256")
    ):
        raise ExportError("store: relpath is a string and the sha256 values are hex digests")
    if isinstance(store["rows"], bool) or not isinstance(store["rows"], int):
        raise ExportError("store: rows is an integer")
    return dict(store)


def _check_frame(frame: pd.DataFrame, fold: Fold) -> np.ndarray:
    """The ``open_ms`` of a valid fold frame (``ExportError`` otherwise)."""
    if not isinstance(frame, pd.DataFrame):
        raise ExportError(f"expected a pandas DataFrame, got {type(frame).__name__}")
    missing = [name for name in ("open_ms", *STORE_COLUMNS) if name not in frame.columns]
    if missing:
        raise ExportError(f"fold {fold.label}: the frame misses columns {missing}")
    if frame["open_ms"].dtype != np.int64:
        raise ExportError(f"fold {fold.label}: open_ms must be int64, got {frame['open_ms'].dtype}")
    open_ms = frame["open_ms"].to_numpy(dtype=np.int64)
    if open_ms.size:
        if (open_ms % HOUR_MS).any():
            first = int(open_ms[np.flatnonzero(open_ms % HOUR_MS)[0]])
            raise ExportError(f"fold {fold.label}: open_ms {first} is not on an exact hour")
        if (np.diff(open_ms) <= 0).any():
            raise ExportError(f"fold {fold.label}: opens must be strictly increasing")
        if open_ms[0] < fold.csv_start_ms or open_ms[-1] >= fold.csv_end_ms:
            raise ExportError(
                f"fold {fold.label}: every open lies in [{utc_text(fold.csv_start_ms)}, "
                f"{utc_text(fold.csv_end_ms)}), got {utc_text(int(open_ms[0]))} to "
                f"{utc_text(int(open_ms[-1]))}"
            )
        for column in STORE_COLUMNS:
            series = frame[column]
            if pd.api.types.infer_dtype(series, skipna=False) != "string":
                raise ExportError(f"fold {fold.label}: {column} must hold Python str values")
            matches = series.str.fullmatch(_DECIMAL).to_numpy(dtype=bool)
            if not matches.all():
                first = int(open_ms[np.flatnonzero(~matches)[0]])
                raise ExportError(
                    f"fold {fold.label}: {column} at {utc_text(first)} is not a plain decimal"
                )
    return open_ms


def _candles(frame: pd.DataFrame, open_ms: np.ndarray) -> list[Candle]:
    columns = [frame[name].to_numpy(dtype=object) for name in STORE_COLUMNS]
    return [
        Candle(int(ms), o, h, lo, c, v)
        for ms, o, h, lo, c, v in zip(open_ms.tolist(), *columns, strict=True)
    ]


def _gaps(open_ms: np.ndarray, start_ms: int, end_ms: int) -> list[list[int]]:
    """The missing hourly opens of ``[start_ms, end_ms)`` as ``[first_missing, end)`` ranges."""
    if open_ms.size == 0:
        return [[start_ms, end_ms]] if end_ms > start_ms else []
    gaps: list[list[int]] = []
    if int(open_ms[0]) > start_ms:
        gaps.append([start_ms, int(open_ms[0])])
    gaps.extend(find_gaps(open_ms, HOUR_MS))
    tail = int(open_ms[-1]) + HOUR_MS
    if tail < end_ms:
        gaps.append([tail, end_ms])
    return gaps


def _amount_text(value: float) -> str:
    """The CSV text of one ``amount``: the shortest text that round-trips the float64."""
    return repr(value)


def _csv_bytes(frame: pd.DataFrame, open_ms: np.ndarray, amounts: np.ndarray) -> bytes:
    stamps = [from_ms(ms).strftime(TIMESTAMP_FORMAT) for ms in open_ms.tolist()]
    columns = [frame[name].to_numpy(dtype=object) for name in STORE_COLUMNS]
    texts = [_amount_text(value) for value in amounts.tolist()]
    lines = [",".join(CSV_COLUMNS)]
    lines.extend(",".join(row) for row in zip(stamps, *columns, texts, strict=True))
    return ("\n".join(lines) + "\n").encode("ascii")


def _check_finite(floats: np.ndarray, open_ms: np.ndarray, label: str) -> None:
    bad = ~(np.isfinite(floats) & np.isfinite(floats.astype(np.float32)))
    if bad.any():
        row, column = (int(index) for index in np.argwhere(bad)[0])
        raise ExportError(
            f"fold {label}: {FEATURE_COLUMNS[column]} at {utc_text(int(open_ms[row]))} is not a "
            "finite float32"
        )


def _float32_mismatch_rows(csv_bytes: bytes, floats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose vendored float32 differs from the inference float32, and the bad cells."""
    vendored = np.ascontiguousarray(vendored_float32(csv_bytes))
    expected = np.ascontiguousarray(floats.astype(np.float32))
    if vendored.shape != expected.shape:
        raise ExportRefused(
            f"the vendored reader sees {vendored.shape} values, the inference path {expected.shape}"
        )
    cells = vendored.view(np.uint32) != expected.view(np.uint32)
    return np.flatnonzero(cells.any(axis=1)), cells


def _nll_inputs(
    fold: Fold, recipe: Recipe, open_ms: np.ndarray, candles: Sequence[Candle]
) -> tuple[list[int], list[np.ndarray], list[dict[str, Any]]]:
    """Kept decisions, their 516-row float64 channels, and the dropped ones with their gaps."""
    context = recipe.lookback_window
    rows = context + recipe.nll_steps
    kept: list[int] = []
    values: list[np.ndarray] = []
    dropped: list[dict[str, Any]] = []
    for decision in fold.nll_decisions_ms():
        start = decision - context * HOUR_MS
        end = decision + recipe.nll_steps * HOUR_MS
        lo = int(np.searchsorted(open_ms, start, side="left"))
        hi = int(np.searchsorted(open_ms, end, side="left"))
        if hi - lo != rows:
            dropped.append({"decision_ms": decision, "missing": _gaps(open_ms[lo:hi], start, end)})
            continue
        frame = context_frame(candles[lo:hi], recipe.amount_policy)
        kept.append(decision)
        values.append(frame.to_numpy(dtype=np.float64))
    return kept, values, dropped


def export_frame(
    frame: pd.DataFrame,
    fold: Fold,
    recipe: Recipe,
    out_dir: Path,
    *,
    store: Mapping[str, Any] | None = None,
) -> ExportRecord:
    """Check and publish the export of ``fold`` from ``frame`` into ``out_dir`` (module docstring).

    ``frame``: ``open_ms`` int64 and :data:`STORE_COLUMNS` as decimal strings (other columns are
    ignored). ``store`` is the store meta read (``None`` for synthetic data). Returns the record
    of the export as published (or as found, when an identical export exists).
    """
    if not isinstance(recipe, Recipe):
        raise TypeError(f"expected a Recipe, got {type(recipe).__name__}")
    check_fold(fold, recipe=recipe)
    store_doc = _check_store(store)
    label = fold.label
    open_ms = _check_frame(frame, fold)
    rows = int(open_ms.size)
    window = recipe.window_rows
    gaps = _gaps(open_ms, fold.csv_start_ms, fold.csv_end_ms)
    missing = sum((end - start) // HOUR_MS for start, end in gaps)
    n_fit = int(np.searchsorted(open_ms, fold.inner_validation_start_ms, side="left"))
    n_val = rows - n_fit
    if n_fit < window + 1 or n_val < window + 1:
        raise ExportRefused(
            f"fold {label}: {n_fit} fit rows and {n_val} inner-validation rows ({missing} "
            f"candles missing); each part needs more than one window of {window} rows"
        )
    try:
        ratio = train_ratio(rows, n_fit)
    except RecipeError as exc:
        raise ExportRefused(f"fold {label}: {exc}") from exc

    candles = _candles(frame, open_ms)
    floats = context_frame(candles, recipe.amount_policy).to_numpy(dtype=np.float64)
    _check_finite(floats, open_ms, label)
    csv_bytes = _csv_bytes(frame, open_ms, floats[:, -1])
    bad_rows, cells = _float32_mismatch_rows(csv_bytes, floats)
    if bad_rows.size:
        first = int(bad_rows[0])
        names = [FEATURE_COLUMNS[i] for i in np.flatnonzero(cells[first])]
        raise ExportRefused(
            f"fold {label}: {bad_rows.size} CSV rows read by the vendored parser differ in "
            f"float32 from the inference values; first at {utc_text(int(open_ms[first]))} "
            f"({', '.join(names)})"
        )

    fit_windows = n_fit - window + 1
    val_windows = n_val - window + 1
    fit_kept = int(window_starts(open_ms[:n_fit], window).size)
    val_kept = int(window_starts(open_ms[n_fit:], window).size)
    fit_dropped = fit_windows - fit_kept
    fit_share = fit_dropped / fit_windows
    if fit_share > recipe.max_dropped_fit_share:
        raise ExportRefused(
            f"fold {label}: gaps ({missing} candles missing) drop {fit_dropped} of "
            f"{fit_windows} fit windows ({fit_share:.2%}), more than the "
            f"{recipe.max_dropped_fit_share:.0%} recipe {recipe.name} allows; the owner decides"
        )
    if val_kept == 0:
        raise ExportRefused(f"fold {label}: no inner-validation window is free of gaps")
    for name, count in (("fit", fit_kept), ("inner-validation", val_kept)):
        if count % VENDORED_INDEX_PRIME == 0:
            raise ExportRefused(
                f"fold {label}: {count} {name} windows is a multiple of "
                f"{VENDORED_INDEX_PRIME}, so the vendored index mapping would reuse windows "
                "(FB:112-114)"
            )

    planned = len(fold.nll_decisions_ms())
    kept, values, dropped = _nll_inputs(fold, recipe, open_ms, candles)
    if len(kept) < recipe.min_nll_decisions:
        raise ExportRefused(
            f"fold {label}: {len(kept)} of {planned} NLL4-TF decisions have every candle, fewer "
            f"than the {recipe.min_nll_decisions} recipe {recipe.name} needs"
        )
    shape = (0, recipe.lookback_window + recipe.nll_steps, len(FEATURE_COLUMNS))
    npz_bytes = nll_inputs_bytes(
        np.asarray(kept, dtype=np.int64),
        np.stack(values) if values else np.empty(shape, dtype=np.float64),
        grid_hours=recipe.nll_grid_hours,
    )

    content: dict[str, Any] = {
        "fold": label,
        "layout": fold.layout,
        "recipe_id": recipe_id(recipe),
        "source": recipe.source,
        "interval": recipe.interval,
        "amount_policy": recipe.amount_policy,
        "csv_start_ms": fold.csv_start_ms,
        "csv_end_ms": fold.csv_end_ms,
        "inner_validation_start_ms": fold.inner_validation_start_ms,
        "csv_start": utc_text(fold.csv_start_ms),
        "csv_end": utc_text(fold.csv_end_ms),
        "inner_validation_start": utc_text(fold.inner_validation_start_ms),
        "columns": list(CSV_COLUMNS),
        "timestamp_format": TIMESTAMP_FORMAT,
        "rows": rows,
        "first_open_ms": int(open_ms[0]),
        "last_open_ms": int(open_ms[-1]),
        "first_open": utc_text(int(open_ms[0])),
        "last_open": utc_text(int(open_ms[-1])),
        "n_fit": n_fit,
        "n_val": n_val,
        "train_ratio": ratio,
        "val_ratio": 1.0,
        "test_ratio": 0.0,
        "window_rows": window,
        "gaps": gaps,
        "missing_candles": missing,
        "fit_windows": fit_windows,
        "fit_windows_kept": fit_kept,
        "fit_windows_dropped": fit_dropped,
        "fit_dropped_share": fit_share,
        "val_windows": val_windows,
        "val_windows_kept": val_kept,
        "val_windows_dropped": val_windows - val_kept,
        "gap_aware": fit_kept < fit_windows or val_kept < val_windows,
        "nll": {
            "planned": planned,
            "kept": len(kept),
            "decision_ms": kept,
            "dropped": dropped,
            "grid_hours": recipe.nll_grid_hours,
            "steps": recipe.nll_steps,
        },
        "csv": {"sha256": sha256_bytes(csv_bytes), "size": len(csv_bytes)},
        "nll_inputs": {"sha256": sha256_bytes(npz_bytes), "size": len(npz_bytes)},
        "float32_check": {"rows": rows, "mismatches": 0},
    }
    content_sha256 = sha256_bytes(canonical_json(content))
    document = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "content": content,
        "content_sha256": content_sha256,
        "provenance": {
            "store": store_doc,
            "code": code_version(),
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }
    text = json.dumps(document, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return _publish(Path(out_dir), csv_bytes, npz_bytes, (text + "\n").encode("ascii"), content)


# -- publishing and reading back ------------------------------------------------------------------


def _publish_or_match(path: Path, data: bytes) -> None:
    """Publish ``data`` at ``path``; an existing file must hold exactly ``data`` (then kept)."""
    try:
        publish_new(path, data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ExportRefused(
                f"{path.parent.name}/{path.name} exists with other bytes; exports are never "
                "overwritten (an interrupted or different export: inspect the directory and "
                "remove it)"
            ) from None


def _existing_content_sha256(path: Path) -> str | None:
    """``content_sha256`` of an existing, self-consistent ``export.json``; ``None`` when absent."""
    if not path.exists():
        return None
    document = _load_document(path)
    return document["content_sha256"]


def _publish(
    directory: Path, csv_bytes: bytes, npz_bytes: bytes, document_bytes: bytes, content: dict
) -> ExportRecord:
    """Publish the CSV, the NLL inputs and last ``export.json``; return the record read back."""
    content_sha256 = sha256_bytes(canonical_json(content))
    json_path = directory / EXPORT_JSON_NAME
    existing = _existing_content_sha256(json_path)
    if existing is not None and existing != content_sha256:
        raise ExportRefused(
            f"{directory.name}/{EXPORT_JSON_NAME} records another export of fold "
            f"{content['fold']} (content {existing[:16]}, now {content_sha256[:16]}): the "
            "candles or the code changed; exports are never overwritten"
        )
    _publish_or_match(directory / CSV_NAME, csv_bytes)
    _publish_or_match(directory / NLL_INPUTS_NAME, npz_bytes)
    if existing is None:
        try:
            publish_new(json_path, document_bytes)
        except FileExistsError:
            if _existing_content_sha256(json_path) != content_sha256:
                raise ExportRefused(
                    f"{directory.name}/{EXPORT_JSON_NAME} was published concurrently with "
                    "another content"
                ) from None
    return read_export(directory)


def _load_document(path: Path) -> dict[str, Any]:
    """A parsed ``export.json`` whose ``content_sha256`` matches its content."""
    label = f"{path.parent.name}/{path.name}"
    try:
        document = json.loads(path.read_bytes().decode("ascii"))
    except OSError as exc:
        raise ExportError(f"cannot read {label} ({type(exc).__name__})") from None
    except ValueError as exc:
        raise ExportError(f"{label} is not ASCII JSON ({exc})") from None
    if not isinstance(document, dict) or set(document) != _DOCUMENT_KEYS:
        raise ExportError(f"{label} is not an export record")
    if document["schema_version"] != EXPORT_SCHEMA_VERSION or isinstance(
        document["schema_version"], bool
    ):
        raise ExportError(f"{label}: schema_version must be {EXPORT_SCHEMA_VERSION}")
    content = document["content"]
    if not isinstance(content, dict):
        raise ExportError(f"{label}: content is not an object")
    if document["content_sha256"] != sha256_bytes(canonical_json(content)):
        raise ExportError(f"{label}: content_sha256 does not match its content (edited?)")
    provenance = document["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_KEYS:
        raise ExportError(f"{label}: provenance must hold exactly {sorted(_PROVENANCE_KEYS)}")
    store = provenance["store"]
    if store is not None and (not isinstance(store, dict) or set(store) != _STORE_KEYS):
        raise ExportError(f"{label}: provenance.store must be null or {sorted(_STORE_KEYS)}")
    return document


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExportError(f"export content: {name} must be an integer, got {value!r}")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise ExportError(f"export content: {name} must be a sha256 hex digest")
    return value


def read_export(directory: Path) -> ExportRecord:
    """Read and re-verify the export in ``directory`` (``ExportError`` on any mismatch).

    ``export.json`` must parse with schema 1 and its ``content_sha256`` must match its content;
    ``train.csv`` and ``nll_inputs.npz`` must have the sizes and sha256 the content records. The
    training parent calls it before every run.
    """
    directory = Path(directory).absolute()
    json_path = directory / EXPORT_JSON_NAME
    document = _load_document(json_path)
    content = document["content"]
    provenance = document["provenance"]
    try:
        files: dict[str, str] = {}
        for key, name in (("csv", CSV_NAME), ("nll_inputs", NLL_INPUTS_NAME)):
            entry = content[key]
            path = directory / name
            if not path.is_file():
                raise ExportError(f"{directory.name}/{name} is missing")
            if path.stat().st_size != _int(entry["size"], f"{key}.size"):
                raise ExportError(f"{directory.name}/{name}: size differs from {EXPORT_JSON_NAME}")
            files[key] = _sha(entry["sha256"], f"{key}.sha256")
            if sha256_file(path) != files[key]:
                raise ExportError(
                    f"{directory.name}/{name}: sha256 differs from {EXPORT_JSON_NAME} (modified)"
                )
        decisions = content["nll"]["decision_ms"]
        if not isinstance(decisions, list):
            raise ExportError("export content: nll.decision_ms must be a list")
        ratio = content["train_ratio"]
        if not isinstance(ratio, float):
            raise ExportError("export content: train_ratio must be a float")
        gap_aware = content["gap_aware"]
        if not isinstance(gap_aware, bool):
            raise ExportError("export content: gap_aware must be a boolean")
        store = provenance["store"]
        return ExportRecord(
            directory=directory,
            csv_path=directory / CSV_NAME,
            nll_inputs_path=directory / NLL_INPUTS_NAME,
            export_json_path=json_path,
            content=_frozen(content),
            content_sha256=document["content_sha256"],
            provenance=_frozen(provenance),
            csv_sha256=files["csv"],
            nll_inputs_sha256=files["nll_inputs"],
            rows=_int(content["rows"], "rows"),
            n_fit=_int(content["n_fit"], "n_fit"),
            n_val=_int(content["n_val"], "n_val"),
            train_ratio=ratio,
            gap_aware=gap_aware,
            fit_windows_kept=_int(content["fit_windows_kept"], "fit_windows_kept"),
            val_windows_kept=_int(content["val_windows_kept"], "val_windows_kept"),
            nll_decisions_ms=tuple(_int(value, "nll.decision_ms") for value in decisions),
            store_meta_sha256=None
            if store is None
            else _sha(store["meta_sha256"], "store.meta_sha256"),
        )
    except (KeyError, TypeError) as exc:
        raise ExportError(f"{directory.name}/{EXPORT_JSON_NAME} is malformed: {exc!r}") from None


# -- helpers shared with the training side and the tests -----------------------------------------


def synthetic_frame(csv_start_ms: int, rows: int, *, seed: int = 0) -> pd.DataFrame:
    """``rows`` hourly synthetic candles from ``csv_start_ms`` as a store-like string frame.

    The geometric-Brownian-motion candles of ``kronos.features.synthetic_window`` as decimal
    strings (prices with 2 decimals, volume with 4: the formats of the batch timing window),
    ``open_ms`` int64 from ``csv_start_ms``. No market data.
    """
    if isinstance(csv_start_ms, bool) or not isinstance(csv_start_ms, int):
        raise ExportError(f"csv_start_ms must be an integer, got {csv_start_ms!r}")
    if csv_start_ms % HOUR_MS:
        raise ExportError(f"csv_start_ms {csv_start_ms} is not on an exact hour")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ExportError(f"rows must be a positive integer, got {rows!r}")
    end = pd.Timestamp(csv_start_ms + (rows - 1) * HOUR_MS, unit="ms")
    candles, _, _ = synthetic_window(rows, step=pd.Timedelta(hours=1), end=end, seed=seed)
    data: dict[str, Any] = {
        "open_ms": csv_start_ms + np.arange(rows, dtype=np.int64) * HOUR_MS,
    }
    for name in STORE_COLUMNS:
        pattern = "{:.4f}" if name == "volume" else "{:.2f}"
        data[name] = pd.Series([pattern.format(x) for x in candles[name].tolist()], dtype=object)
    return pd.DataFrame(data)


def vendored_float32(csv: bytes | Path) -> np.ndarray:
    """The six channels as the vendored dataset reads them (FB:53-56, FB:122): float32 ``(n, 6)``.

    ``pd.read_csv`` with pandas' DEFAULT float parser, timestamps parsed and sorted, the columns
    ``open .. amount`` as ``.values.astype(np.float32)``. Missing values are not forward-filled
    (the vendored dataset fills them; an exported CSV has none).
    """
    source = io.BytesIO(csv) if isinstance(csv, (bytes, bytearray)) else Path(csv)
    frame = pd.read_csv(source)
    frame["timestamps"] = pd.to_datetime(frame["timestamps"])
    frame = frame.sort_values("timestamps").reset_index(drop=True)
    return frame[list(FEATURE_COLUMNS)].values.astype(np.float32)


def window_starts(open_ms: np.ndarray, rows: int) -> np.ndarray:
    """Indices ``s`` whose ``rows`` rows span exactly ``rows - 1`` hours (int64, ascending)."""
    values = np.asarray(open_ms, dtype=np.int64)
    if values.ndim != 1:
        raise ExportError("open_ms must be one-dimensional")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ExportError(f"rows must be a positive integer, got {rows!r}")
    if values.size < rows:
        return np.empty(0, dtype=np.int64)
    span = values[rows - 1 :] - values[: values.size - rows + 1]
    return np.flatnonzero(span == (rows - 1) * HOUR_MS).astype(np.int64)


def nll_inputs_bytes(
    decision_ms: np.ndarray, values: np.ndarray, *, grid_hours: int = R1.nll_grid_hours
) -> bytes:
    """The deterministic ``nll_inputs.npz`` bytes of ``decision_ms`` and their 516-row channels.

    A ZIP_STORED archive of ``decision_ms.npy`` (int64 ``(n,)``, ascending, unique, on the
    ``grid_hours`` grid), ``values.npy`` (float64 ``(n, 516, 6)``, finite) and ``meta.npy`` (a 0-d
    unicode array of canonical JSON), each member dated :data:`ZIP_DATE_TIME` and written by
    ``np.lib.format.write_array`` without pickle: the same inputs always give the same bytes.
    """
    if isinstance(grid_hours, bool) or not isinstance(grid_hours, int) or grid_hours < 1:
        raise ExportError(f"grid_hours must be a positive integer, got {grid_hours!r}")
    decisions = np.asarray(decision_ms)
    if decisions.ndim != 1 or not np.issubdtype(decisions.dtype, np.integer):
        raise ExportError("decision_ms must be a one-dimensional integer array")
    decisions = decisions.astype("<i8")
    if (np.diff(decisions) <= 0).any() or (decisions % (grid_hours * HOUR_MS)).any():
        raise ExportError(f"decision_ms must be ascending, unique and on the {grid_hours} h grid")
    channels = np.asarray(values)
    expected = (decisions.size, MAX_CONTEXT + NLL_STEPS, len(FEATURE_COLUMNS))
    if channels.dtype != np.float64 or channels.shape != expected:
        raise ExportError(f"values must be float64 of shape {expected}, got {channels.shape}")
    if not np.isfinite(channels).all():
        raise ExportError("values must be finite")
    meta = {
        "schema_version": NLL_INPUTS_SCHEMA_VERSION,
        "context": MAX_CONTEXT,
        "steps": NLL_STEPS,
        "columns": list(FEATURE_COLUMNS),
        "amount_policy": NLL_AMOUNT_POLICY,
        "decisions": int(decisions.size),
        "grid_hours": grid_hours,
    }
    arrays = (
        np.ascontiguousarray(decisions, dtype="<i8"),
        np.ascontiguousarray(channels, dtype="<f8"),
        np.array(canonical_json(meta).decode("ascii")),
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in zip(NLL_INPUT_MEMBERS, arrays, strict=True):
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


__all__ = [
    "CSV_COLUMNS",
    "CSV_NAME",
    "EXPORTS_DIRNAME",
    "EXPORT_JSON_NAME",
    "EXPORT_SCHEMA_VERSION",
    "NLL_AMOUNT_POLICY",
    "NLL_INPUTS_NAME",
    "NLL_INPUTS_SCHEMA_VERSION",
    "NLL_INPUT_MEMBERS",
    "STORE_COLUMNS",
    "TIMESTAMP_FORMAT",
    "VENDORED_INDEX_PRIME",
    "ZIP_DATE_TIME",
    "ExportError",
    "ExportRecord",
    "ExportRefused",
    "export_fold",
    "export_frame",
    "fold_export_dir",
    "nll_inputs_bytes",
    "read_export",
    "synthetic_frame",
    "vendored_float32",
    "window_starts",
]
