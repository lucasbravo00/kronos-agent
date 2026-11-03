"""Registry of fine-tuned Kronos checkpoints (``docs/RETRAINING.md``, 3.6).

``<ft>/registry.jsonl``, where ``<ft>`` is a fine-tune root (``<weights>/finetuned`` by default,
:func:`finetuned_dir`; the identity control and the tests use scratch roots), lists every
registered checkpoint, one canonical-JSON object per line (sorted keys, no whitespace, ASCII),
each terminated by LF. It is append-only and hash-chained: ``prev_sha256`` of the first line is
:data:`GENESIS_SHA256` and of every later line the sha256 of the previous line's bytes without
its LF, so an edited, removed or reordered line breaks the chain at the line after it. The last
line has no line after it: an edit that keeps it a valid entry, or its removal, leaves a valid
chain. Its protection is external: whoever records a checkpoint's use keeps the ``line_sha256``
of its entry (``finetune.train.fit`` returns the entry; the experiment ledger and the model card,
built later, are to record it) and names the checkpoint with it, ``CheckpointRef(id, root,
line_sha256)``; every reader of such a reference (:func:`find_entry`, hence
``kronos.weights.load_models`` and the batch layer, which pins the line of the entry it prepared
a batch with) refuses an entry that no longer hashes to it. A line holds the 33 keys of the
caller's payload (:data:`PAYLOAD_KEYS`: what was trained, from what, how, on which data, by
which code) and the four the registry fills (``schema_version``, ``prev_sha256``, ``files``,
``created_at``). Selection values (losses, NLL4-TF, the selected pseudo-epoch) never enter it:
they stay in the blinded fold record that ``fit_record_sha256`` identifies.

The files of a checkpoint live in ``<ft>/<checkpoint_id>/`` (``config.json`` and
``model.safetensors``, mode ``0o644``). :func:`install_checkpoint` copies them from the run
directory into a staging directory, checks them, renames the staging directory into place and
only then appends the line, all under the root's exclusive lock: a kill between the rename and
the append leaves an unregistered directory, which nothing loads (no entry names it) and
:func:`registry_status` reports. Registered files are never overwritten or removed.

The registry stays outside ``manifests/``, which enters ``source_tree_sha256``: a tracked
registry would change the recorded code hash at every registration. This module imports only the
standard library and light project modules (never ``kronos.*``, numpy, pandas or torch):
``kronos.weights`` imports it on the fan path. ``kronos.weights.load_models`` verifies the bytes
it loads with its own code, which is hashed into the forecast ``config_id``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..data.fsutil import FILE_MODE, atomic_write_bytes, exclusive_lock, fsync_directory
from ..data.timeutil import HOUR_MS, utc_text
from ..hashing import canonical_json, sha256_bytes, sha256_file
from ..models import MODEL_SPECS
from ..paths import LABEL_RE, weights_dir

FINETUNED_DIRNAME = "finetuned"
REGISTRY_NAME = "registry.jsonl"
LOCK_NAME = ".lock"
RUNS_DIRNAME = ".runs"
SCRATCH_DIRNAME = ".scratch"
STAGING_PREFIX = ".staging-"
REGISTRY_SCHEMA_VERSION = 1
GENESIS_SHA256 = "0" * 64
CHECKPOINT_FILES: tuple[str, ...] = ("config.json", "model.safetensors")
CHECKPOINT_ID_RE = re.compile(r"ks-[a-z0-9]+(?:-[a-z0-9]+)*")
CHECKPOINT_ID_MAX_LENGTH = 64
CHECKPOINT_ID_PREFIX = "ks-btc-"
ENTRY_KINDS: tuple[str, ...] = ("fold", "identity")
PAYLOAD_KEYS: tuple[str, ...] = tuple(
    sorted(
        (
            "kind",
            "checkpoint_id",
            "model",
            "base",
            "tokenizer",
            "expected_config",
            "state_dict_numel",
            "data_end_ms",
            "data_end",
            "recipe_id",
            "fold",
            "replicate",
            "fit_range_ms",
            "inner_validation_range_ms",
            "test_start_ms",
            "test_end_ms",
            "train_csv_sha256",
            "nll_inputs_sha256",
            "export_content_sha256",
            "store_meta_sha256",
            "run_id",
            "fit_record_sha256",
            "vendored_best_model_sha256",
            "device",
            "threads",
            "deterministic",
            "activation_checkpointing",
            "torch_version",
            "platform",
            "seeds",
            "code",
            "training_code_sha256",
            "vendored_manifest_sha256",
        )
    )
)
# the keys the registry fills; with the payload they make every line
REGISTRY_KEYS: tuple[str, ...] = ("schema_version", "prev_sha256", "files", "created_at")
ENTRY_KEYS: tuple[str, ...] = tuple(sorted((*PAYLOAD_KEYS, *REGISTRY_KEYS)))
# non-null for kind "fold", null for kind "identity"
FOLD_ONLY_KEYS: tuple[str, ...] = (
    "recipe_id",
    "fold",
    "replicate",
    "fit_range_ms",
    "inner_validation_range_ms",
    "train_csv_sha256",
    "nll_inputs_sha256",
    "export_content_sha256",
    "run_id",
    "fit_record_sha256",
    "seeds",
    "training_code_sha256",
)
# may be null in a fold entry (final folds, synthetic data, no vendored keep-best), always null
# in an identity entry
FOLD_OPTIONAL_KEYS: tuple[str, ...] = (
    "test_start_ms",
    "test_end_ms",
    "store_meta_sha256",
    "vendored_best_model_sha256",
)
CODE_KEYS: tuple[str, ...] = ("commit", "dirty", "source_tree_sha256")
REPO_KEYS: tuple[str, ...] = ("repo_id", "sha256")
FILE_KEYS: tuple[str, ...] = ("name", "sha256", "size")
DIRECTORY_MODE = 0o755

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_RECIPE8_RE = re.compile(r"[0-9a-f]{8}")
_RUN_LABEL_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_CREATED_AT_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_COPY_CHUNK = 1 << 20
_MAX_SHOWN = 8


class RegistryError(RuntimeError):
    """A checkpoint that cannot be registered or found, or a registry that cannot be read."""


class RegistryChainError(RegistryError):
    """A line of the registry, its hash chain or its schema is broken (never repaired)."""


class CheckpointNotFound(RegistryError):
    """No entry of the registry has this checkpoint id."""


class CheckpointExists(RegistryError):
    """The checkpoint id is registered already, or its directory exists without an entry."""


def _check_checkpoint_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) > CHECKPOINT_ID_MAX_LENGTH
        or not CHECKPOINT_ID_RE.fullmatch(value)
    ):
        raise ValueError(
            f"a checkpoint id matches {CHECKPOINT_ID_RE.pattern} in at most "
            f"{CHECKPOINT_ID_MAX_LENGTH} characters, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class CheckpointRef:
    """A checkpoint by id in a fine-tune root (``root=None``: :func:`finetuned_dir`).

    ``line_sha256``, when given, is the sha256 of the checkpoint's registry line as it was
    recorded (at registration, in the ledger, by a prepared batch): :func:`find_entry` refuses an
    entry that no longer hashes to it. The hash chain alone cannot protect the registry's last
    line (module docstring).
    """

    checkpoint_id: str
    root: Path | None = None
    line_sha256: str | None = None

    def __post_init__(self) -> None:
        _check_checkpoint_id(self.checkpoint_id)
        if self.root is not None:
            if not isinstance(self.root, (str, os.PathLike)):
                raise TypeError(f"root must be a path, got {type(self.root).__name__}")
            object.__setattr__(self, "root", Path(self.root))
        if self.line_sha256 is not None and not _is_hex(self.line_sha256):
            raise ValueError(
                f"line_sha256 must be a lowercase sha256 hex or None, got {self.line_sha256!r}"
            )

    def resolved_root(self) -> Path:
        """The fine-tune root that holds this checkpoint's ``registry.jsonl``."""
        return self.root if self.root is not None else finetuned_dir()


@dataclass(frozen=True)
class CheckpointFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True, eq=False)
class RegistryEntry:
    """One verified line of the registry; ``document`` is a read-only view of the parsed line.

    ``eq=False``: entries are compared by their ``line`` (or ``line_sha256``), never field by
    field.
    """

    line: bytes
    line_sha256: str
    document: Mapping[str, Any]
    checkpoint_id: str
    kind: str
    model: str
    files: tuple[CheckpointFile, ...]
    data_end_ms: int
    expected_config: Mapping[str, Any]
    state_dict_numel: int
    tokenizer_repo_id: str
    tokenizer_sha256: Mapping[str, str]

    def weights_sha256(self) -> dict[str, str]:
        """The ``weights_sha256`` of the models loaded from this checkpoint (the only definition).

        ``finetuned/<checkpoint_id>/<file>`` for the checkpoint's own files and
        ``<tokenizer repo_id>/<file>`` for the tokenizer it was trained with; the shipped weights
        record ``<repo_id>/<file>`` keys instead, so the two never share a forecast config.
        """
        weights = {
            f"{FINETUNED_DIRNAME}/{self.checkpoint_id}/{item.name}": item.sha256
            for item in self.files
        }
        weights.update(
            {
                f"{self.tokenizer_repo_id}/{name}": value
                for name, value in self.tokenizer_sha256.items()
            }
        )
        return weights


# -- layout ----------------------------------------------------------------------------------------


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else finetuned_dir()


def finetuned_dir(base: Path | None = None) -> Path:
    """The default fine-tune root: ``(base or weights_dir()) / "finetuned"``."""
    return (Path(base) if base is not None else weights_dir()) / FINETUNED_DIRNAME


def runs_dir(root: Path | None = None) -> Path:
    """``<root>/.runs``: one directory per training child run (the fold records)."""
    return _root(root) / RUNS_DIRNAME


def scratch_dir(root: Path | None = None) -> Path:
    """``<root>/.scratch``: scratch space of the timing command."""
    return _root(root) / SCRATCH_DIRNAME


def registry_path(root: Path | None = None) -> Path:
    """``<root>/registry.jsonl``."""
    return _root(root) / REGISTRY_NAME


def checkpoint_dir(ref: CheckpointRef) -> Path:
    """``<ref root>/<checkpoint_id>``: the registered files of one checkpoint."""
    if not isinstance(ref, CheckpointRef):
        raise TypeError(f"expected a CheckpointRef, got {type(ref).__name__}")
    return ref.resolved_root() / ref.checkpoint_id


def checkpoint_id_for(recipe8: str, run_label: str) -> str:
    """``ks-btc-<recipe8>-<run_label>``: the id of a fold's checkpoint.

    ``recipe8`` is the first 8 hex characters of the recipe id; ``run_label`` the fold's label
    with its replicate suffix (``wf-2024-07``, ``wf-2025-06-r1``), lowercase letters, digits and
    single hyphens. ``ValueError`` otherwise, or when the id would exceed 64 characters.
    """
    if not isinstance(recipe8, str) or not _RECIPE8_RE.fullmatch(recipe8):
        raise ValueError(f"recipe8 must be 8 lowercase hex characters, got {recipe8!r}")
    if not isinstance(run_label, str) or not _RUN_LABEL_RE.fullmatch(run_label):
        raise ValueError(
            f"a run label in a checkpoint id matches {_RUN_LABEL_RE.pattern}, got {run_label!r}"
        )
    return _check_checkpoint_id(f"{CHECKPOINT_ID_PREFIX}{recipe8}-{run_label}")


def _label(root: Path) -> str:
    """A path-free name of the registry of ``root`` for messages."""
    return f"{root.name}/{REGISTRY_NAME}"


# -- validation ------------------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value: object) -> bool:
    return isinstance(value, str) and _HEX64_RE.fullmatch(value) is not None


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.isascii() and value.isprintable()


def _is_label(value: object) -> bool:
    return isinstance(value, str) and LABEL_RE.fullmatch(value) is not None


def _is_range(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_int(item) for item in value)
        and value[0] <= value[1]
    )


def _repo_problems(name: str, value: object) -> list[str]:
    if not isinstance(value, dict) or sorted(value) != sorted(REPO_KEYS):
        return [f"{name} must be an object with exactly {list(REPO_KEYS)}"]
    problems: list[str] = []
    if not _is_text(value["repo_id"]):
        problems.append(f"{name}.repo_id must be a non-empty string")
    hashes = value["sha256"]
    if (
        not isinstance(hashes, dict)
        or not hashes
        or not all(_is_text(key) and _is_hex(item) for key, item in hashes.items())
    ):
        problems.append(f"{name}.sha256 must map each file name to its sha256 hex")
    return problems


def _code_problems(value: object) -> list[str]:
    if not isinstance(value, dict) or sorted(value) != sorted(CODE_KEYS):
        return [f"code must be an object with exactly {list(CODE_KEYS)}"]
    problems: list[str] = []
    if value["commit"] is not None and not _is_text(value["commit"]):
        problems.append("code.commit must be a string or null")
    if value["dirty"] is not None and not isinstance(value["dirty"], bool):
        problems.append("code.dirty must be a boolean or null")
    if not _is_hex(value["source_tree_sha256"]):
        problems.append("code.source_tree_sha256 must be a sha256 hex")
    return problems


_HEX_KEYS: tuple[str, ...] = (
    "recipe_id",
    "train_csv_sha256",
    "nll_inputs_sha256",
    "export_content_sha256",
    "store_meta_sha256",
    "fit_record_sha256",
    "vendored_best_model_sha256",
    "training_code_sha256",
)


def _payload_problems(document: Mapping[str, Any]) -> list[str]:
    """Why ``document`` (a JSON object holding every key of :data:`PAYLOAD_KEYS`) is not a valid
    payload: types (section 5.8 of the interfaces), the rules of its kind and the fold ranges."""
    problems: list[str] = []
    kind = document["kind"]
    if kind not in ENTRY_KINDS:
        problems.append(f"kind must be one of {list(ENTRY_KINDS)}, got {kind!r}")
    try:
        _check_checkpoint_id(document["checkpoint_id"])
    except ValueError as exc:
        problems.append(str(exc))
    if not isinstance(document["model"], str) or document["model"] not in MODEL_SPECS:
        problems.append(f"model must be one of {sorted(MODEL_SPECS)}, got {document['model']!r}")
    problems += _repo_problems("base", document["base"])
    problems += _repo_problems("tokenizer", document["tokenizer"])
    if not isinstance(document["expected_config"], dict) or not document["expected_config"]:
        problems.append("expected_config must be a non-empty object")
    if not _is_int(document["state_dict_numel"]) or document["state_dict_numel"] < 1:
        problems.append("state_dict_numel must be a positive integer")
    data_end_ms = document["data_end_ms"]
    if not _is_int(data_end_ms) or data_end_ms % HOUR_MS:
        problems.append("data_end_ms must be an integer of epoch milliseconds on an exact hour")
    elif document["data_end"] != utc_text(data_end_ms):
        problems.append(f"data_end must be {utc_text(data_end_ms)!r}, the text of data_end_ms")
    for key in _HEX_KEYS:
        if document[key] is not None and not _is_hex(document[key]):
            problems.append(f"{key} must be a sha256 hex or null")
    if not _is_hex(document["vendored_manifest_sha256"]):
        problems.append("vendored_manifest_sha256 must be a sha256 hex")
    for key in ("fold", "run_id"):
        if document[key] is not None and not _is_label(document[key]):
            problems.append(f"{key} must match {LABEL_RE.pattern} or be null")
    replicate = document["replicate"]
    if replicate is not None and (not _is_int(replicate) or replicate < 0):
        problems.append("replicate must be an integer >= 0 or null")
    for key in ("fit_range_ms", "inner_validation_range_ms"):
        if document[key] is not None and not _is_range(document[key]):
            problems.append(f"{key} must be [start_ms, end_ms] with start_ms <= end_ms, or null")
    for key in ("test_start_ms", "test_end_ms"):
        if document[key] is not None and not _is_int(document[key]):
            problems.append(f"{key} must be an integer of epoch milliseconds or null")
    if not _is_text(document["device"]):
        problems.append("device must be a non-empty string")
    if not _is_int(document["threads"]) or document["threads"] < 1:
        problems.append("threads must be a positive integer")
    for key in ("deterministic", "activation_checkpointing"):
        if not isinstance(document[key], bool):
            problems.append(f"{key} must be a boolean")
    for key in ("torch_version", "platform"):
        if not _is_text(document[key]):
            problems.append(f"{key} must be a non-empty string")
    if document["seeds"] is not None and (
        not isinstance(document["seeds"], dict) or not document["seeds"]
    ):
        problems.append("seeds must be a non-empty object or null")
    problems += _code_problems(document["code"])
    if problems:
        return problems
    return _kind_problems(document)


def _kind_problems(document: Mapping[str, Any]) -> list[str]:
    """The rules of each kind, on a document whose types are valid."""
    problems: list[str] = []
    if document["kind"] == "identity":
        set_keys = [
            key for key in (*FOLD_ONLY_KEYS, *FOLD_OPTIONAL_KEYS) if document[key] is not None
        ]
        if set_keys:
            problems.append(f"an identity entry has null {set_keys}")
        return problems
    unset = [key for key in FOLD_ONLY_KEYS if document[key] is None]
    if unset:
        problems.append(f"a fold entry needs {unset}")
        return problems
    fit_range, inner_range = document["fit_range_ms"], document["inner_validation_range_ms"]
    if fit_range[1] != inner_range[0]:
        problems.append("fit_range_ms must end where inner_validation_range_ms starts")
    if inner_range[1] != document["data_end_ms"]:
        problems.append("inner_validation_range_ms must end at data_end_ms, the CSV end")
    start, end = document["test_start_ms"], document["test_end_ms"]
    if (start is None) != (end is None):
        problems.append("test_start_ms and test_end_ms are both set or both null")
    elif start is not None and not document["data_end_ms"] <= start < end:
        problems.append("a fold's test months start at or after data_end_ms and are not empty")
    return problems


def _registry_key_problems(document: Mapping[str, Any]) -> list[str]:
    """The four keys the registry fills, on a line read back."""
    problems: list[str] = []
    version = document["schema_version"]
    if not _is_int(version) or version != REGISTRY_SCHEMA_VERSION:
        problems.append(f"unknown schema_version {version!r}")
    if not _is_hex(document["prev_sha256"]):
        problems.append("prev_sha256 must be a sha256 hex")
    files = document["files"]
    if (
        not isinstance(files, list)
        or [item.get("name") if isinstance(item, dict) else None for item in files]
        != list(CHECKPOINT_FILES)
        or not all(
            sorted(item) == sorted(FILE_KEYS)
            and _is_int(item["size"])
            and item["size"] >= 0
            and _is_hex(item["sha256"])
            for item in files
        )
    ):
        problems.append(
            f"files must list {list(CHECKPOINT_FILES)} in that order with their size and sha256"
        )
    created_at = document["created_at"]
    if not isinstance(created_at, str) or not _CREATED_AT_RE.fullmatch(created_at):
        problems.append("created_at must be YYYY-MM-DDTHH:MM:SSZ")
    return problems


def _shown(problems: Sequence[str]) -> str:
    text = "; ".join(problems[:_MAX_SHOWN])
    if len(problems) > _MAX_SHOWN:
        text += f"; ... ({len(problems)} problems)"
    return text


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a finite JSON number")


def _json_copy(value: Mapping[str, Any], what: str) -> dict[str, Any]:
    """A pure-JSON copy of ``value`` (tuples become lists); non-finite floats and values JSON
    cannot hold are a :class:`RegistryError`."""
    try:
        text = json.dumps(dict(value), sort_keys=True, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RegistryError(f"{what} is not JSON-serialisable with finite numbers: {exc}") from None
    return json.loads(text)


def _validated_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RegistryError(f"a payload is a mapping, got {type(payload).__name__}")
    missing = sorted(set(PAYLOAD_KEYS) - set(payload))
    unknown = sorted(str(key) for key in set(payload) - set(PAYLOAD_KEYS))
    if missing or unknown:
        raise RegistryError(f"the payload misses {missing} and has unknown keys {unknown}")
    document = _json_copy(payload, "the payload")
    problems = _payload_problems(document)
    if problems:
        raise RegistryError(
            f"invalid payload for {payload.get('checkpoint_id')!r}: {_shown(problems)}"
        )
    return document


def _entry_from_line(line: bytes, number: int, previous_sha256: str) -> RegistryEntry:
    """The entry of line ``number`` (1-based), whose predecessor's line hashes to
    ``previous_sha256``; :class:`RegistryChainError` naming the line for any problem."""

    def broken(reason: str) -> RegistryChainError:
        return RegistryChainError(f"registry line {number}: {reason}")

    try:
        document = json.loads(line.decode("ascii"), parse_constant=_refuse_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise broken(f"not JSON ({exc})") from None
    if not isinstance(document, dict):
        raise broken("not a JSON object")
    if canonical_json(document) != line:
        raise broken("not canonical JSON")
    if sorted(document) != list(ENTRY_KEYS):
        missing = sorted(set(ENTRY_KEYS) - set(document))
        unknown = sorted(set(document) - set(ENTRY_KEYS))
        raise broken(f"misses {missing} and has unknown keys {unknown}")
    problems = _registry_key_problems(document) or _payload_problems(document)
    if problems:
        raise broken(_shown(problems))
    if document["prev_sha256"] != previous_sha256:
        raise broken("prev_sha256 is not the sha256 of the previous line: the chain is broken")
    return RegistryEntry(
        line=line,
        line_sha256=sha256_bytes(line),
        document=types.MappingProxyType(document),
        checkpoint_id=document["checkpoint_id"],
        kind=document["kind"],
        model=document["model"],
        files=tuple(
            CheckpointFile(item["name"], item["size"], item["sha256"]) for item in document["files"]
        ),
        data_end_ms=document["data_end_ms"],
        expected_config=types.MappingProxyType(dict(document["expected_config"])),
        state_dict_numel=document["state_dict_numel"],
        tokenizer_repo_id=document["tokenizer"]["repo_id"],
        tokenizer_sha256=types.MappingProxyType(dict(document["tokenizer"]["sha256"])),
    )


def _scan(data: bytes) -> tuple[list[RegistryEntry], RegistryChainError | None]:
    """The entries of ``data`` that verify, in order, and the first problem (``None``: all)."""
    entries: list[RegistryEntry] = []
    if not data:
        return entries, None
    *complete, tail = data.split(b"\n")
    previous = GENESIS_SHA256
    seen: set[str] = set()
    for number, line in enumerate(complete, 1):
        try:
            entry = _entry_from_line(line, number, previous)
        except RegistryChainError as exc:
            return entries, exc
        if entry.checkpoint_id in seen:
            return entries, RegistryChainError(
                f"registry line {number}: checkpoint {entry.checkpoint_id} is registered twice"
            )
        seen.add(entry.checkpoint_id)
        entries.append(entry)
        previous = entry.line_sha256
    if tail:
        return entries, RegistryChainError(
            f"registry line {len(complete) + 1}: no LF at the end of the file (a torn write); "
            "the registry is never repaired"
        )
    return entries, None


def _read(root: Path) -> tuple[bytes, tuple[RegistryEntry, ...]]:
    path = registry_path(root)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return b"", ()
    except OSError as exc:
        raise RegistryError(f"cannot read {_label(root)}: {exc.strerror}") from None
    entries, problem = _scan(data)
    if problem is not None:
        raise RegistryChainError(f"{_label(root)}: {problem}")
    return data, tuple(entries)


# -- reading ---------------------------------------------------------------------------------------


def read_registry(root: Path | None = None) -> tuple[RegistryEntry, ...]:
    """Every entry of ``<root>/registry.jsonl``, verified; a missing file is an empty registry.

    Every line must end with LF (a torn last line is refused, never repaired), be the canonical
    JSON of an object with exactly :data:`ENTRY_KEYS`, schema 1, a valid id, kind and types, and
    carry the sha256 of the previous line (:data:`GENESIS_SHA256` for the first); ids are unique.
    Any problem is a :class:`RegistryChainError` naming the line.
    """
    return _read(_root(root))[1]


def registry_head(root: Path | None = None) -> str:
    """The ``line_sha256`` of the last entry, or :data:`GENESIS_SHA256` for an empty registry."""
    entries = read_registry(root)
    return entries[-1].line_sha256 if entries else GENESIS_SHA256


def _check_recorded_line(ref: CheckpointRef, entry: RegistryEntry) -> None:
    """With ``ref.line_sha256``, ``entry`` must still be the line that was recorded."""
    if ref.line_sha256 is not None and entry.line_sha256 != ref.line_sha256:
        raise RegistryChainError(
            f"{_label(ref.resolved_root())}: the entry of checkpoint {ref.checkpoint_id} hashes "
            f"to {entry.line_sha256[:12]}..., not to the line sha256 recorded for it "
            f"({ref.line_sha256[:12]}...): it changed after it was recorded; the registry is "
            "never repaired"
        )


def find_entry(ref: CheckpointRef) -> RegistryEntry:
    """The verified entry of ``ref`` (:class:`CheckpointNotFound` when it has none).

    With ``ref.line_sha256`` the entry's line must still hash to it (:class:`RegistryChainError`
    otherwise): an edit that keeps the registry's last line a valid entry leaves the chain
    valid, so only the hash recorded where the checkpoint is used detects it.
    """
    if not isinstance(ref, CheckpointRef):
        raise TypeError(f"expected a CheckpointRef, got {type(ref).__name__}")
    root = ref.resolved_root()
    for entry in read_registry(root):
        if entry.checkpoint_id == ref.checkpoint_id:
            _check_recorded_line(ref, entry)
            return entry
    raise CheckpointNotFound(f"checkpoint {ref.checkpoint_id} is not in {_label(root)}")


def checkpoint_problems(ref: CheckpointRef, entry: RegistryEntry | None = None) -> list[str]:
    """Why the files of ``ref`` do not match its entry (an empty list: they do).

    The size and sha256 of every listed file of :func:`checkpoint_dir` and the ``config.json``
    content against the entry's ``expected_config``; problems are labelled
    ``finetuned/<id>/<file>``. For status reports and tests: ``kronos.weights.load_models``
    verifies the bytes it loads with its own code. A given ``entry`` must be the one of ``ref``
    (its id, and its recorded line when ``ref`` carries one).
    """
    if entry is None:
        entry = find_entry(ref)
    elif entry.checkpoint_id != ref.checkpoint_id:
        raise ValueError(
            f"the entry of {entry.checkpoint_id} is not the one of {ref.checkpoint_id}"
        )
    else:
        _check_recorded_line(ref, entry)
    directory = checkpoint_dir(ref)
    label = f"{FINETUNED_DIRNAME}/{ref.checkpoint_id}"
    problems: list[str] = []
    for item in entry.files:
        path = directory / item.name
        if not path.is_file():
            problems.append(f"missing: {label}/{item.name}")
            continue
        size = path.stat().st_size
        if size != item.size:
            problems.append(f"size mismatch: {label}/{item.name} ({size} != {item.size})")
            continue
        if sha256_file(path) != item.sha256:
            problems.append(f"sha256 mismatch: {label}/{item.name}")
    config_path = directory / "config.json"
    if config_path.is_file():
        try:
            content = json.loads(config_path.read_text(encoding="utf-8"))
        except ValueError:
            content = None
        if content != dict(entry.expected_config):
            problems.append(f"config content mismatch: {label}/config.json")
    return problems


def registry_status(root: Path | None = None) -> dict[str, Any]:
    """What the fine-tune root holds, without changing anything.

    ``entries`` and ``head`` describe the lines that verify (all of them when ``chain_ok``; the
    prefix before the first broken line otherwise, ``problem`` naming it);
    ``unregistered_dirs`` are checkpoint directories no verified entry names (a kill between an
    install's rename and its append leaves one; nothing loads it); ``staging_leftovers`` are
    staging directories (an install in progress or killed; never loaded).
    """
    base = _root(root)
    status: dict[str, Any] = {
        "entries": 0,
        "head": GENESIS_SHA256,
        "chain_ok": True,
        "problem": None,
        "unregistered_dirs": [],
        "staging_leftovers": [],
    }
    try:
        data = registry_path(base).read_bytes()
    except FileNotFoundError:
        data = b""
    except OSError as exc:
        data = b""
        status.update(chain_ok=False, problem=f"cannot read {_label(base)}: {exc.strerror}")
    entries, problem = _scan(data)
    if problem is not None:
        status.update(chain_ok=False, problem=str(problem))
    status["entries"] = len(entries)
    if entries:
        status["head"] = entries[-1].line_sha256
    registered = {entry.checkpoint_id for entry in entries}
    if base.is_dir():
        for child in sorted(base.iterdir(), key=lambda path: path.name):
            name = child.name
            if name.startswith(STAGING_PREFIX):
                status["staging_leftovers"].append(name)
            elif child.is_dir() and CHECKPOINT_ID_RE.fullmatch(name) and name not in registered:
                status["unregistered_dirs"].append(name)
    return status


# -- installing ------------------------------------------------------------------------------------


def _copy_file(source: Path, destination: Path) -> None:
    """Copy ``source`` to the new file ``destination`` (synced, mode ``0o644``)."""
    with open(source, "rb") as reader, open(destination, "xb") as writer:
        shutil.copyfileobj(reader, writer, _COPY_CHUNK)
        writer.flush()
        os.fchmod(writer.fileno(), FILE_MODE)
        os.fsync(writer.fileno())


def _staged_files(staging: Path) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "sha256": sha256_file(staging / name),
            "size": (staging / name).stat().st_size,
        }
        for name in CHECKPOINT_FILES
    ]


def _stage(
    source: Path, staging: Path, document: Mapping[str, Any], check: Callable[[Path], Sequence[str]]
) -> list[dict[str, Any]]:
    """Copy and check the checkpoint files in ``staging``; return their names, sizes and sha256."""
    checkpoint_id = document["checkpoint_id"]
    for name in CHECKPOINT_FILES:
        if not (source / name).is_file():
            raise RegistryError(f"the checkpoint {checkpoint_id} to install has no {name}")
        _copy_file(source / name, staging / name)
    try:
        staged_config = json.loads((staging / "config.json").read_text(encoding="utf-8"))
    except ValueError:
        staged_config = None
    if staged_config != document["expected_config"]:
        raise RegistryError(
            f"the config.json of {checkpoint_id} differs from the payload's expected_config"
        )
    problems = list(check(staging))
    if problems:
        raise RegistryError(f"checkpoint {checkpoint_id} is not loadable: {_shown(problems)}")
    files = _staged_files(staging)
    os.chmod(staging, DIRECTORY_MODE)
    fsync_directory(staging)
    return files


def install_checkpoint(
    source_dir: Path,
    *,
    payload: Mapping[str, Any],
    check: Callable[[Path], Sequence[str]],
    root: Path | None = None,
) -> RegistryEntry:
    """Install the checkpoint in ``source_dir`` and append its entry; return the entry read back.

    In this order: the payload must hold exactly :data:`PAYLOAD_KEYS` with valid types, every
    :data:`FOLD_ONLY_KEYS` value set for kind ``"fold"`` and null for kind ``"identity"``
    (:class:`RegistryError`); under the root's exclusive lock (a busy lock is a
    ``LockBusyError``, a refusal): the registry must verify and neither name the id nor hold its
    directory (:class:`CheckpointExists`); exactly :data:`CHECKPOINT_FILES` are copied into a
    staging directory (anything else in ``source_dir``, such as the generated ``README.md``, is
    ignored), its ``config.json`` must equal the payload's ``expected_config`` and
    ``check(staging)`` (``kronos.weights.checkpoint_dir_problems``) must find no problem
    (:class:`RegistryError`); the staged files are hashed, the staging directory renamed to
    ``<root>/<checkpoint_id>`` and the line appended (old bytes plus the line, replaced
    atomically). A failure before the rename removes the staging directory.
    """
    if not callable(check):
        raise TypeError("check must be a callable returning the problems of a directory")
    base = _root(root)
    document = _validated_payload(payload)
    checkpoint_id = document["checkpoint_id"]
    source = Path(source_dir)
    with exclusive_lock(base / LOCK_NAME):
        old, entries = _read(base)
        if any(entry.checkpoint_id == checkpoint_id for entry in entries):
            raise CheckpointExists(f"checkpoint {checkpoint_id} is registered already")
        target = base / checkpoint_id
        if target.exists() or target.is_symlink():
            raise CheckpointExists(
                f"{FINETUNED_DIRNAME}/{checkpoint_id} exists without a registry entry (an "
                "interrupted install left it): inspect it and remove it before installing again"
            )
        staging = Path(tempfile.mkdtemp(prefix=f"{STAGING_PREFIX}{checkpoint_id}-", dir=base))
        try:
            files = _stage(source, staging, document, check)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        os.rename(staging, target)
        fsync_directory(base)
        head = entries[-1].line_sha256 if entries else GENESIS_SHA256
        line = canonical_json(
            {
                **document,
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "prev_sha256": head,
                "files": files,
                "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        atomic_write_bytes(registry_path(base), old + line + b"\n")
        written = _read(base)[1]
    if not written or written[-1].line != line:
        raise RegistryChainError(f"{_label(base)} does not end with the line just appended")
    return written[-1]


__all__ = [
    "CHECKPOINT_FILES",
    "CHECKPOINT_ID_MAX_LENGTH",
    "CHECKPOINT_ID_PREFIX",
    "CHECKPOINT_ID_RE",
    "CODE_KEYS",
    "DIRECTORY_MODE",
    "ENTRY_KEYS",
    "ENTRY_KINDS",
    "FINETUNED_DIRNAME",
    "FOLD_ONLY_KEYS",
    "FOLD_OPTIONAL_KEYS",
    "GENESIS_SHA256",
    "LOCK_NAME",
    "PAYLOAD_KEYS",
    "REGISTRY_KEYS",
    "REGISTRY_NAME",
    "REGISTRY_SCHEMA_VERSION",
    "RUNS_DIRNAME",
    "SCRATCH_DIRNAME",
    "STAGING_PREFIX",
    "CheckpointExists",
    "CheckpointFile",
    "CheckpointNotFound",
    "CheckpointRef",
    "RegistryChainError",
    "RegistryEntry",
    "RegistryError",
    "checkpoint_dir",
    "checkpoint_id_for",
    "checkpoint_problems",
    "find_entry",
    "finetuned_dir",
    "install_checkpoint",
    "read_registry",
    "registry_head",
    "registry_path",
    "registry_status",
    "runs_dir",
    "scratch_dir",
]
