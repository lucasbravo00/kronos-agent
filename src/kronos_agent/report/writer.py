"""Study reports: ``reports/data/<stem>.json`` + ``<stem>.md``, recorded configs and study series.

Every study records its definition BEFORE its first run (SPEC rule 7): :func:`record_config`
writes ``<stem>.config.json`` once as canonical JSON and a later run whose definition differs is
refused (:class:`ConfigChangedError`): a changed study is a new, separately reported study.
:func:`report_header` fills the keys every report carries (``docs/DATA.md``, "Studies and
reports"): the study, the config and ``DATA.md`` hashes, the code, the test budget consumed (0:
these studies measure the data, not a strategy), the number of statistics with intervals and how
many of them are expected to miss by chance at 95 %, seeds, samples and comparators.

:func:`write_report` refuses a report without those keys, a non-finite number (an undefined
statistic is written as ``None`` WITH a reason, never as a silent ``NaN``), personal data, a
non-ASCII Markdown text, an absolute data path and any BingX host name (reports never name one
outside ``kronos_agent.exchange``). JSON is written with ``indent=1``, sorted keys and
``allow_nan=False``; every file goes through the durable atomic write. Per-hour series go to
``data/studies/<stem>.csv`` (gitignored) and the report records their sha256.

:func:`append_record` keeps an append-only record file next to the reports
(``<name>.jsonl``, one canonical-JSON object per line) under the same refusals; the lines already
there are never rewritten, and :func:`read_records` reads them back.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.fsutil import atomic_write_bytes, atomic_write_text, publish_new
from ..envinfo import assert_no_personal_data
from ..exchange.public_market import PUBLIC_HOST
from ..hashing import canonical_json, sha256_bytes, sha256_file
from ..paths import REPO_ROOT, data_dir, reports_dir, validate_label
from ..provenance import code_version

REQUIRED_KEYS: tuple[str, ...] = (
    "study",
    "config_sha256",
    "data_md_sha256",
    "code",
    "generated_at",
    "tests_consumed",
    "statistics_with_intervals",
    "expected_chance_misses",
    "seeds",
    "samples",
    "comparators",
)
DATA_MD_RELPATH = "docs/DATA.md"
REPORTS_SUBDIR = "data"
STUDIES_SUBDIR = "studies"
RECORDS_SUFFIX = ".jsonl"
# the registered domain of both BingX hosts, taken from the public client (no literal here)
_EXCHANGE_DOMAIN = PUBLIC_HOST.split(".", 1)[1]


class ReportError(ValueError):
    """A report, config or series that must not be written as given."""


class ConfigChangedError(ReportError):
    """The study's recorded definition differs from the one this run uses."""


def _report_dir(directory: Path | None) -> Path:
    return Path(directory) if directory is not None else reports_dir() / REPORTS_SUBDIR


def _stem(stem: str) -> str:
    try:
        return validate_label(stem)
    except ValueError as exc:
        raise ReportError(str(exc)) from None


def _utc_text(value: dt.datetime) -> str:
    if value.tzinfo is None:  # tz-naive timestamps mean UTC in this project
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def _key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
        return str(int(key))
    raise ReportError(f"JSON object keys must be strings or integers, got {type(key).__name__}")


def to_jsonable(obj: Any) -> Any:
    """Plain JSON types from nested reports: numpy scalars and arrays become Python values.

    Non-finite floats become ``None``; ``Decimal`` becomes its exact string; datetimes become
    UTC ISO text (tz-naive ones are UTC by convention); objects with ``as_dict()`` (bootstrap
    results) are converted through it; integer keys become strings. Anything else (sets, paths,
    arbitrary objects) raises :class:`ReportError`.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Decimal):
        return str(obj) if obj.is_finite() else None
    if obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, dt.datetime):
        return _utc_text(obj)
    if isinstance(obj, dt.date):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {_key(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        return to_jsonable(as_dict())
    raise ReportError(f"cannot write a {type(obj).__name__} into a report")


def _non_finite_path(obj: Any, path: str = "") -> str | None:
    """Key path of the first NaN or infinity in ``obj`` (``None`` when there is none)."""
    if isinstance(obj, (float, np.floating)):
        return None if math.isfinite(float(obj)) else (path or "<root>")
    if isinstance(obj, Decimal):
        return None if obj.is_finite() else (path or "<root>")
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.floating) or obj.dtype == object:
            for index, value in enumerate(obj.tolist()):
                found = _non_finite_path(value, f"{path}[{index}]")
                if found:
                    return found
        return None
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            found = _non_finite_path(value, f"{path}.{key}" if path else str(key))
            if found:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found = _non_finite_path(value, f"{path}[{index}]")
            if found:
                return found
        return None
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        return _non_finite_path(as_dict(), path)
    return None


def _absolute_roots() -> list[str]:
    """The repository and data roots as text (a one-level root such as ``/data`` is skipped: it
    would match relative paths like ``reports/data``)."""
    return [str(path) for path in (REPO_ROOT, data_dir()) if len(path.parts) >= 3]


def _check_publishable(document: Mapping[str, Any], texts: Mapping[str, str]) -> None:
    """Refuse personal data in ``document`` and ``texts``; absolute roots and BingX host names
    in ``texts`` (the serialised forms)."""
    try:
        assert_no_personal_data(document)
        assert_no_personal_data(dict(texts))
    except ValueError as exc:
        raise ReportError(str(exc)) from exc
    roots = _absolute_roots()
    for name, text in texts.items():
        if _EXCHANGE_DOMAIN in text.lower():
            raise ReportError(f"{name} names a BingX host; reports say 'BingX real market'")
        if any(root in text for root in roots):
            raise ReportError(f"{name} holds an absolute path; record paths relative to a root")


def data_md_sha256() -> str:
    """sha256 of ``docs/DATA.md``, the design every step-2 report embeds."""
    return sha256_file(REPO_ROOT / DATA_MD_RELPATH)


def record_config(stem: str, config: Mapping[str, Any], *, directory: Path | None = None) -> str:
    """Record a study's definition once; return the sha256 of the recorded bytes.

    ``<directory>/<stem>.config.json`` (default ``reports/data``) holds exactly
    ``canonical_json(config)``, so ``shasum -a 256`` of the file is the returned hash. An
    identical existing file is kept; a different one raises :class:`ConfigChangedError`.
    """
    stem = _stem(stem)
    if not isinstance(config, Mapping):
        raise ReportError(f"a study config is a mapping, got {type(config).__name__}")
    bad = _non_finite_path(config)
    if bad:
        raise ReportError(f"config {stem}: non-finite value at {bad}")
    jsonable = to_jsonable(config)
    data = canonical_json(jsonable)
    _check_publishable(jsonable, {"config": data.decode("ascii")})
    digest = sha256_bytes(data)
    path = _report_dir(directory) / f"{stem}.config.json"
    try:
        publish_new(path, data)
    except FileExistsError:
        existing = path.read_bytes()
        if existing != data:
            raise ConfigChangedError(
                f"study {stem}: its recorded config (sha256 {sha256_bytes(existing)[:12]}) differs "
                f"from this run's ({digest[:12]}); a changed study is a new study with a new stem"
            ) from None
    return digest


def report_header(
    stem: str,
    config: Mapping[str, Any],
    *,
    statistics_with_intervals: int,
    seeds: Mapping[str, Any],
    samples: Mapping[str, Any],
    comparators: Mapping[str, Any],
    directory: Path | None = None,
) -> dict[str, Any]:
    """Every :data:`REQUIRED_KEYS` entry of a report, recording or verifying its config first.

    ``tests_consumed`` is 0 and ``expected_chance_misses`` is ``0.05`` per interval (computed as
    ``n * 5 / 100``, so 3 intervals give exactly ``0.15``).
    """
    if (
        isinstance(statistics_with_intervals, bool)
        or not isinstance(statistics_with_intervals, int)
        or statistics_with_intervals < 0
    ):
        raise ReportError("statistics_with_intervals must be a non-negative integer")
    for name, value in (("seeds", seeds), ("samples", samples), ("comparators", comparators)):
        if not isinstance(value, Mapping):
            raise ReportError(f"{name} must be a mapping, got {type(value).__name__}")
    config_sha256 = record_config(stem, config, directory=directory)
    return {
        "study": stem,
        "config_sha256": config_sha256,
        "data_md_sha256": data_md_sha256(),
        "code": code_version(),
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tests_consumed": 0,
        "statistics_with_intervals": statistics_with_intervals,
        "expected_chance_misses": statistics_with_intervals * 5 / 100,
        "seeds": dict(seeds),
        "samples": dict(samples),
        "comparators": dict(comparators),
    }


def write_report(
    stem: str, payload: Mapping[str, Any], markdown: str, *, directory: Path | None = None
) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` (default directory ``reports/data``).

    Refused (:class:`ReportError`): a missing :data:`REQUIRED_KEYS` entry or a ``study`` that is
    not ``stem``; a NaN or infinity anywhere in ``payload``; a non-ASCII Markdown text; personal
    data (:func:`kronos_agent.envinfo.assert_no_personal_data`), an absolute data path or a BingX
    host name in either file. The JSON is ``to_jsonable`` then
    ``json.dumps(indent=1, sort_keys=True, allow_nan=False) + "\\n"``.
    """
    stem = _stem(stem)
    if not isinstance(payload, Mapping):
        raise ReportError(f"a report payload is a mapping, got {type(payload).__name__}")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ReportError(f"report {stem} lacks the header keys {missing}")
    if payload["study"] != stem:
        raise ReportError(f"report {stem} carries study {payload['study']!r}")
    bad = _non_finite_path(payload)
    if bad:
        raise ReportError(
            f"report {stem}: non-finite value at {bad}; write None with a reason instead"
        )
    if not isinstance(markdown, str) or not markdown.isascii():
        raise ReportError(f"report {stem}: the Markdown text must be an ASCII string")
    jsonable = to_jsonable(payload)
    try:
        text = json.dumps(jsonable, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ReportError(f"report {stem} is not JSON-serialisable: {exc}") from exc
    if not markdown.endswith("\n"):
        markdown += "\n"
    _check_publishable(jsonable, {"json": text, "markdown": markdown})
    directory = _report_dir(directory)
    json_path = directory / f"{stem}.json"
    markdown_path = directory / f"{stem}.md"
    atomic_write_text(json_path, text + "\n")
    atomic_write_text(markdown_path, markdown)
    return json_path, markdown_path


def write_study_series(
    stem: str, frame: pd.DataFrame, *, root: Path | None = None
) -> dict[str, Any]:
    """Write a per-hour series to ``data/studies/<stem>.csv``; return its relpath, sha256, rows.

    ``frame.to_csv(index=False, lineterminator="\\n")``: floats in their shortest round-trip
    form, so the same frame always gives the same bytes. The relpath is relative to the data
    root, the form a report records.
    """
    stem = _stem(stem)
    if not isinstance(frame, pd.DataFrame):
        raise ReportError(f"a study series is a DataFrame, got {type(frame).__name__}")
    text = frame.to_csv(index=False, lineterminator="\n")
    if not text.isascii():
        raise ReportError(f"study series {stem} is not ASCII")
    data = text.encode("ascii")
    relpath = f"{STUDIES_SUBDIR}/{stem}.csv"
    base = Path(root) if root is not None else data_dir()
    atomic_write_bytes(base / relpath, data)
    return {"relpath": relpath, "sha256": sha256_bytes(data), "rows": len(frame)}


def _parse_records(data: bytes, name: str) -> list[dict[str, Any]]:
    """The records of a record file's bytes; a torn or malformed file is refused, not repaired."""
    if data and not data.endswith(b"\n"):
        raise ReportError(f"{name}: the last record is not terminated (a torn write); inspect it")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(data.split(b"\n")[:-1], start=1):
        try:
            record = json.loads(line.decode("ascii"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ReportError(f"{name}:{number}: not a JSON record ({exc})") from None
        if not isinstance(record, dict):
            raise ReportError(f"{name}:{number}: a record is a JSON object")
        records.append(record)
    return records


def append_record(name: str, record: Mapping[str, Any], *, directory: Path | None = None) -> Path:
    """Append ``record`` to ``<directory>/<name>.jsonl`` (default ``reports/data``); its path.

    The record is refused like a report (:class:`ReportError`: a non-finite number, personal
    data, an absolute data path, a BingX host name) and becomes one line of canonical JSON. The
    lines already in the file must parse, and they are kept byte for byte: the file is replaced
    atomically by its old bytes plus the new line, so a crash leaves the old file or the new one
    and a record, once written, never changes.
    """
    name = _stem(name)
    if not isinstance(record, Mapping):
        raise ReportError(f"a record is a mapping, got {type(record).__name__}")
    bad = _non_finite_path(record)
    if bad:
        raise ReportError(f"record {name}: non-finite value at {bad}; write None with a reason")
    jsonable = to_jsonable(record)
    line = canonical_json(jsonable)
    _check_publishable(jsonable, {"record": line.decode("ascii")})
    path = _report_dir(directory) / f"{name}{RECORDS_SUFFIX}"
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        existing = b""
    _parse_records(existing, path.name)
    atomic_write_bytes(path, existing + line + b"\n")
    return path


def read_records(name: str, *, directory: Path | None = None) -> list[dict[str, Any]]:
    """The records of ``<directory>/<name>.jsonl`` in the order they were appended (an empty list
    when the file does not exist); a line that is not a JSON object raises :class:`ReportError`."""
    name = _stem(name)
    path = _report_dir(directory) / f"{name}{RECORDS_SUFFIX}"
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    return _parse_records(data, path.name)


__all__ = [
    "RECORDS_SUFFIX",
    "REQUIRED_KEYS",
    "ConfigChangedError",
    "ReportError",
    "append_record",
    "data_md_sha256",
    "read_records",
    "record_config",
    "report_header",
    "to_jsonable",
    "write_report",
    "write_study_series",
]
