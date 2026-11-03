"""The parent of one training child (``docs/RETRAINING.md`` 3.1, 3.5, 3.6; interfaces 4.11).

A :class:`RunRequest` names everything one child run needs: the mode (``train``, ``timing`` or
``preflight``), the recipe, the fold, its verified export, the compute choices (device, threads,
determinism, activation checkpointing, the MPS memory watermark, the memory guard), the model
files (the shipped, verified Kronos-small and Kronos-Tokenizer-base by default), which wrappers
of ``finetune.runner`` are on, and the spend limits. :func:`start_run`:

1. verifies the vendored tree (a ``__pycache__`` fails it, D-014), the shipped weights when the
   request uses them, and the export (``export_csv.read_export``);
2. creates ``<ft>/.runs/<run_label>-<UTC stamp>`` exclusively and writes ``run.yaml`` (the
   recipe's run YAML, every key the vendored loader reads) and ``run.json`` (interfaces 5.5);
3. starts ``<python> -B -P -m kronos_agent.finetune.runner --run-dir <dir>`` in a new session,
   in the run directory, stdin ``/dev/null``, stdout and stderr appended to ``child.log``, with
   an environment built from an allow-list (:func:`child_environment`): no credential or other
   variable of the parent reaches the child.

:meth:`RunningChild.wait` terminates the child when the parent's timeout fires or the parent is
interrupted (SIGTERM, a grace period, then SIGKILL), verifies the vendored tree again and reads
``runner_result.json``. :func:`fit` runs a ``train`` request and, by default, registers the
selected checkpoint (``finetune.registry.install_checkpoint`` with
``kronos.weights.checkpoint_dir_problems`` as its check), then removes the run's ``selected/`` and
vendored keep-best directories: their sha256 are in the registry entry and the fold record. Only a
recipe run is registered under the recipe's checkpoint id: every wrapper on, from the registered
shipped weights, an id not registered yet (all three checked before the child starts, so no fold
is trained only to be refused), and a record showing the state NLL4-TF selected. A diagnostic
configuration (a wrapper off, test T1) runs with ``register=False`` or :func:`run`.

The run directory keeps ``run.json``, ``run.yaml``, ``child.log`` and ``runner_result.json`` (the
fold record). ``child.log`` and the vendored logs hold absolute paths and the blinded loss values:
they are never printed or read into a report. :meth:`FitOutcome.printable` is all a command may
show of a blinded fold. ``torch`` and the vendored code are never imported here.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..data.fsutil import atomic_write_text, ensure_directory, fsync_directory
from ..data.timeutil import utc_text
from ..hashing import sha256_bytes, sha256_file
from ..paths import validate_label
from ..provenance import code_version, files_sha256
from ..runtime import OFFLINE_ENV, THREAD_ENV_VARS
from . import runner
from .nll import NLL_BATCH_SIZE

if TYPE_CHECKING:
    from .export_csv import ExportRecord
    from .folds import Fold
    from .recipe import Recipe
    from .registry import RegistryEntry

CHILD_MODULE = "kronos_agent.finetune.runner"
CHILD_PYTHON_FLAGS: tuple[str, ...] = ("-B", "-P")
CHILD_ENV_PASSTHROUGH: tuple[str, ...] = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "KRONOS_AGENT_WEIGHTS_DIR",
    "KRONOS_AGENT_DATA_DIR",
)
CHILD_ENV_FIXED: Mapping[str, str] = MappingProxyType(
    {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", **OFFLINE_ENV}
)
MPS_WATERMARK_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
DEFAULT_MPS_WATERMARK_RATIO = 1.0
MAX_MPS_WATERMARK_RATIO = 1.7
# torch refuses a low watermark above the high one ("invalid low watermark ratio 1.4" at the first
# MPS allocation, torch 2.14.1 on unified memory, whose low default is 1.4): the low ratio is set
# with the high one, to min(1.4, high), so a high ratio below 1.4 does not break every MPS run
MPS_LOW_WATERMARK_ENV = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
TORCH_MPS_LOW_WATERMARK_DEFAULT = 1.4
KILL_GRACE_SECONDS = 30.0
TRAINING_CODE_FILES: tuple[str, ...] = (
    "src/kronos_agent/finetune/recipe.py",
    "src/kronos_agent/finetune/export_csv.py",
    "src/kronos_agent/finetune/nll.py",
    "src/kronos_agent/finetune/runner.py",
    "src/kronos_agent/finetune/train.py",
    "src/kronos_agent/kronos/features.py",
    "src/kronos_agent/data/windows.py",
    "manifests/kronos_source.sha256",
)
RUN_DIR_STAMP = "%Y%m%dT%H%M%SZ"
MAX_RUN_DIR_SUFFIX = 99
JSON_INDENT = 1
NO_RESULT_STATUSES: tuple[str, ...] = ("killed", "no_result")  # RunnerResult.status, no file
# runner_result.json "selected.source" of a recipe run: the state NLL4-TF selected (design 3.3);
# with early stopping off the child keeps the vendored keep-best ("vendored_best") instead
RECIPE_SELECTION = "nll4_tf"


class TrainError(RuntimeError):
    """A training request that cannot run, or a run whose outcome cannot be used (exit 1)."""


class ChildFailed(TrainError):
    """The child failed, was killed or wrote no result."""


class ChildMemoryAbort(TrainError):
    """The child stopped on memory pressure: not a fold failure; the owner is told (design 3.5)."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _now_text() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_text(document: Mapping[str, Any]) -> str:
    """The JSON form of this phase's files (interfaces section 3)."""
    return (
        json.dumps(document, indent=JSON_INDENT, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    )


@dataclass(frozen=True)
class Compute:
    """Compute choices of one run; recorded with it, outside the recipe id (interfaces 9.10)."""

    device: str = "cpu"  # "cpu" | "mps"
    threads: int = 2
    deterministic: bool = True
    activation_checkpointing: bool = False
    mps_watermark_ratio: float | None = None  # MPS only: (0, 1.7]; None means 1.0 on MPS
    memory_guard: bool = True

    def __post_init__(self) -> None:
        if self.device not in runner.DEVICES:
            raise TrainError(f"device must be one of {list(runner.DEVICES)}, got {self.device!r}")
        if not _is_int(self.threads) or self.threads < 1:
            raise TrainError(f"threads must be a positive integer, got {self.threads!r}")
        for name in ("deterministic", "activation_checkpointing", "memory_guard"):
            if not isinstance(getattr(self, name), bool):
                raise TrainError(f"{name} must be a boolean")
        ratio = self.mps_watermark_ratio
        if ratio is not None:
            if self.device != "mps":
                raise TrainError("mps_watermark_ratio applies to MPS runs only")
            if not _is_number(ratio) or not 0 < ratio <= MAX_MPS_WATERMARK_RATIO:
                raise TrainError(
                    f"mps_watermark_ratio must be in (0, {MAX_MPS_WATERMARK_RATIO}], got {ratio!r}"
                )
            object.__setattr__(self, "mps_watermark_ratio", float(ratio))


@dataclass(frozen=True)
class Wrappers:
    """Which wrappers of ``finetune.runner`` are on (all of them in a recipe run)."""

    device: bool = True
    router: bool = True
    clean_validation: bool = True
    early_stopping: bool = True
    pseudo_epochs: bool = True
    gap_aware: bool = True
    weight_check: bool = True

    def __post_init__(self) -> None:
        if tuple(item.name for item in fields(self)) != runner.WRAPPERS:
            raise TrainError("the wrapper switches must be exactly runner.WRAPPERS")
        for item in fields(self):
            if not isinstance(getattr(self, item.name), bool):
                raise TrainError(f"wrapper {item.name} must be a boolean")

    @classmethod
    def none(cls) -> Wrappers:
        """Every wrapper off: the child then reproduces the vendored script (test T1)."""
        return cls(**{name: False for name in runner.WRAPPERS})


@dataclass(frozen=True)
class TimingSpec:
    """Warm-up steps, then exactly one of a number of timed steps or a timed duration."""

    warmup_steps: int
    timed_steps: int | None = None
    timed_seconds: float | None = None

    def __post_init__(self) -> None:
        if not _is_int(self.warmup_steps) or self.warmup_steps < 0:
            raise TrainError(f"warmup_steps must be an integer >= 0, got {self.warmup_steps!r}")
        if (self.timed_steps is None) == (self.timed_seconds is None):
            raise TrainError("a timing needs exactly one of timed_steps and timed_seconds")
        if self.timed_steps is not None and (not _is_int(self.timed_steps) or self.timed_steps < 1):
            raise TrainError(f"timed_steps must be a positive integer, got {self.timed_steps!r}")
        if self.timed_seconds is not None:
            if not _is_number(self.timed_seconds) or not self.timed_seconds > 0:
                raise TrainError(f"timed_seconds must be positive, got {self.timed_seconds!r}")
            object.__setattr__(self, "timed_seconds", float(self.timed_seconds))


@dataclass(frozen=True)
class ModelFiles:
    """The predictor and tokenizer directories a run loads, with the sha256 of their files."""

    predictor_dir: Path
    tokenizer_dir: Path
    predictor_sha256: Mapping[str, str]
    tokenizer_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in ("predictor_dir", "tokenizer_dir"):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                raise TrainError(f"{name} must be an absolute path")
            object.__setattr__(self, name, path)
        for name in ("predictor_sha256", "tokenizer_sha256"):
            hashes = dict(getattr(self, name))
            if sorted(hashes) != sorted(runner.CHECKPOINT_FILE_NAMES) or not all(
                isinstance(value, str) and len(value) == 64 for value in hashes.values()
            ):
                raise TrainError(
                    f"{name} must map exactly {list(runner.CHECKPOINT_FILE_NAMES)} to sha256 hex"
                )
            object.__setattr__(self, name, hashes)


@dataclass(frozen=True)
class RunRequest:
    """One child run: what to train or measure, on what, how (interfaces 4.11)."""

    mode: str
    recipe: Recipe
    fold: Fold
    export: ExportRecord
    compute: Compute
    model_files: ModelFiles | None = None  # None: shipped_model_files(recipe.model)
    wrappers: Wrappers = field(default_factory=Wrappers)
    timing: TimingSpec | None = None  # required for mode "timing"
    preflight_checks: tuple[str, ...] = ()  # required for mode "preflight"
    deadline_seconds: float | None = None
    blinded: bool = False

    def __post_init__(self) -> None:
        from .export_csv import ExportRecord
        from .folds import Fold
        from .recipe import Recipe

        if self.mode not in runner.MODES:
            raise TrainError(f"mode must be one of {list(runner.MODES)}, got {self.mode!r}")
        for name, expected in (("recipe", Recipe), ("fold", Fold), ("export", ExportRecord)):
            if not isinstance(getattr(self, name), expected):
                raise TrainError(f"{name} must be a {expected.__name__}")
        if not isinstance(self.compute, Compute) or not isinstance(self.wrappers, Wrappers):
            raise TrainError("compute and wrappers must be Compute and Wrappers")
        if self.export.gap_aware and not self.wrappers.gap_aware:
            raise TrainError(
                "the export has a gap: a run on it needs the gap_aware wrapper (design 1.6: no "
                "window across a missing hour is trained or validated on)"
            )
        if self.model_files is not None and not isinstance(self.model_files, ModelFiles):
            raise TrainError("model_files must be ModelFiles or None")
        if (self.mode == "timing") != (self.timing is not None):
            raise TrainError("a timing (and only a timing) request needs a TimingSpec")
        if self.timing is not None and not isinstance(self.timing, TimingSpec):
            raise TrainError("timing must be a TimingSpec")
        checks = tuple(self.preflight_checks)
        if self.mode == "preflight":
            if not checks or len(set(checks)) != len(checks):
                raise TrainError("a preflight request needs distinct checks")
            unknown = sorted(set(checks) - set(runner.PREFLIGHT_CHECKS))
            if unknown:
                raise TrainError(f"unknown preflight checks {unknown}")
        elif checks:
            raise TrainError("preflight checks belong to a preflight request")
        object.__setattr__(self, "preflight_checks", checks)
        if self.mode == "timing" and self.wrappers.early_stopping:
            raise TrainError("a timing request measures steps: switch early_stopping off")
        if self.compute.device != "cpu" and not self.wrappers.device:
            raise TrainError("a run on another device than the CPU needs the device wrapper")
        if self.deadline_seconds is not None:
            if not _is_number(self.deadline_seconds) or not self.deadline_seconds > 0:
                raise TrainError("deadline_seconds must be positive")
            object.__setattr__(self, "deadline_seconds", float(self.deadline_seconds))
        if not isinstance(self.blinded, bool):
            raise TrainError("blinded must be a boolean")


@dataclass(frozen=True)
class RunnerResult:
    """What a child run left: its status, exit code and ``runner_result.json``."""

    run_id: str
    run_dir: Path
    mode: str
    status: str  # the file's status, "killed" or "no_result" (no file written)
    exit_code: int  # the child's return code (negative: a signal)
    timed_out: bool  # the parent's timeout fired
    document: Mapping[str, Any]  # runner_result.json ({} when absent)


@dataclass(frozen=True)
class FitOutcome:
    """A finished fit; :meth:`printable` is all a command may show of a blinded fold."""

    run_id: str
    run_dir: Path
    status: str
    finite: bool
    wall_seconds: float
    cpu_seconds: float
    selected_sha256: str | None  # model.safetensors of the selected state
    checkpoint: RegistryEntry | None  # None with register=False
    blinded: bool

    def printable(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "wall_seconds": self.wall_seconds,
            "finite": self.finite,
            "selected_sha256": self.selected_sha256,
        }


def training_code_sha256() -> str:
    """``provenance.files_sha256`` of :data:`TRAINING_CODE_FILES` (whole files)."""
    return files_sha256(TRAINING_CODE_FILES)


def shipped_model_files(name: str = "small", *, base: Path | None = None) -> ModelFiles:
    """The registered model and tokenizer directories of ``name`` with their registered sha256
    (``manifests/weights.json``); nothing is read or verified here."""
    from ..kronos.weights import repo_dir, repos_for_model

    model_repo, tokenizer_repo = repos_for_model(name)

    def hashes(repo: Any) -> dict[str, str]:
        found = {item.name: item.sha256 for item in repo.files}
        if sorted(found) != sorted(runner.CHECKPOINT_FILE_NAMES) or None in found.values():
            raise TrainError(f"{repo.repo_id} has no registered sha256 for every file")
        return found

    return ModelFiles(
        predictor_dir=repo_dir(model_repo.repo_id, base).absolute(),
        tokenizer_dir=repo_dir(tokenizer_repo.repo_id, base).absolute(),
        predictor_sha256=hashes(model_repo),
        tokenizer_sha256=hashes(tokenizer_repo),
    )


def new_run_dir(run_label: str, *, root: Path | None = None) -> Path:
    """Create ``<root>/.runs/<run_label>-<YYYYMMDDTHHMMSSZ>`` exclusively (``-1`` .. ``-99``
    appended on a collision); the directory name is the run id."""
    from .registry import runs_dir

    validate_label(run_label)
    parent = runs_dir(root).absolute()
    ensure_directory(parent)
    stamp = datetime.now(UTC).strftime(RUN_DIR_STAMP)
    for index in range(MAX_RUN_DIR_SUFFIX + 1):
        name = f"{run_label}-{stamp}" if index == 0 else f"{run_label}-{stamp}-{index}"
        validate_label(name)
        path = parent / name
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
        fsync_directory(parent)
        return path
    raise TrainError(f"no free run directory for {run_label} at {stamp}")


def _model_files(request: RunRequest) -> ModelFiles:
    return request.model_files or shipped_model_files(request.recipe.model)


def _run_document(run_dir: Path, request: RunRequest, yaml_sha256: str) -> dict[str, Any]:
    from .recipe import recipe_id, recipe_mapping, training_seeds

    recipe, fold, export = request.recipe, request.fold, request.export
    files = _model_files(request)
    identifier = recipe_id(recipe)
    seeds = training_seeds(identifier, fold.label, fold.replicate)
    nll_inputs = None
    if request.mode == "train":
        nll_inputs = {
            "path": str(Path(export.nll_inputs_path).absolute()),
            "sha256": export.nll_inputs_sha256,
            "decisions": len(export.nll_decisions_ms),
        }
    timing = None
    if request.timing is not None:
        timing = {
            "warmup_steps": request.timing.warmup_steps,
            "timed_steps": request.timing.timed_steps,
            "timed_seconds": request.timing.timed_seconds,
        }
    preflight = None
    if request.mode == "preflight":
        preflight = {
            "checks": list(request.preflight_checks),
            "multinomial_rows": runner.PREFLIGHT_MULTINOMIAL_ROWS,
            "multinomial_repetitions": runner.PREFLIGHT_MULTINOMIAL_REPETITIONS,
            "steps": runner.PREFLIGHT_STEPS,
        }
    return {
        "schema_version": runner.RUN_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "mode": request.mode,
        "created_at": _now_text(),
        "recipe": recipe_mapping(recipe),
        "recipe_id": identifier,
        "fold": {
            "label": fold.label,
            "replicate": fold.replicate,
            "run_label": fold.run_label,
            "layout": fold.layout,
            "csv_start_ms": fold.csv_start_ms,
            "csv_end_ms": fold.csv_end_ms,
            "inner_validation_start_ms": fold.inner_validation_start_ms,
        },
        "seeds": asdict(seeds),
        "yaml": {"name": runner.RUN_YAML, "sha256": yaml_sha256},
        "train_csv": {
            "path": str(Path(export.csv_path).absolute()),
            "sha256": export.csv_sha256,
            "rows": export.rows,
            "n_fit": export.n_fit,
            "n_val": export.n_val,
        },
        "nll_inputs": nll_inputs,
        "export_content_sha256": export.content_sha256,
        "windows": {
            "gap_aware": export.gap_aware,
            "fit_windows_kept": export.fit_windows_kept,
            "val_windows_kept": export.val_windows_kept,
        },
        "model_files": {
            "predictor": {"dir": str(files.predictor_dir), "sha256": dict(files.predictor_sha256)},
            "tokenizer": {"dir": str(files.tokenizer_dir), "sha256": dict(files.tokenizer_sha256)},
        },
        "compute": asdict(request.compute),
        "wrappers": asdict(request.wrappers),
        "nll_batch_size": NLL_BATCH_SIZE,
        "memory": {
            "every_steps": runner.MEMORY_CHECK_EVERY_STEPS,
            "max_swap_growth_bytes": runner.MAX_SWAP_GROWTH_BYTES,
            "critical_pressure_level": runner.CRITICAL_PRESSURE_LEVEL,
        },
        "timing": timing,
        "preflight": preflight,
        "deadline_seconds": request.deadline_seconds,
        "blinded": request.blinded,
    }


def write_run_inputs(run_dir: Path, request: RunRequest) -> dict[str, str]:
    """Write ``run.yaml`` (the recipe's run YAML) and ``run.json`` (interfaces 5.5) into
    ``run_dir``; return their sha256 as ``{"run.yaml": ..., "run.json": ...}``."""
    from .recipe import YamlPaths, run_yaml_document, training_seeds, write_run_yaml
    from .recipe import recipe_id as recipe_id_of

    run_dir = Path(run_dir).absolute()
    validate_label(run_dir.name)
    recipe, fold, export = request.recipe, request.fold, request.export
    files = _model_files(request)
    seeds = training_seeds(recipe_id_of(recipe), fold.label, fold.replicate)
    document = run_yaml_document(
        recipe,
        paths=YamlPaths(
            data_path=Path(export.csv_path).absolute(),
            predictor_dir=files.predictor_dir,
            tokenizer_dir=files.tokenizer_dir,
            save_dir=run_dir / runner.VENDORED_SAVE_DIRNAME,
            base_path=run_dir,
        ),
        n_rows=export.rows,
        n_fit=export.n_fit,
        seed=seeds.yaml_seed,
        exp_name=run_dir.name,
        description=f"{recipe.name} {fold.run_label} {request.mode}",
    )
    yaml_sha256 = write_run_yaml(run_dir / runner.RUN_YAML, document)
    text = _json_text(_run_document(run_dir, request, yaml_sha256))
    atomic_write_text(run_dir / runner.RUN_JSON, text)
    return {runner.RUN_YAML: yaml_sha256, runner.RUN_JSON: sha256_bytes(text.encode("ascii"))}


def child_command(run_dir: Path) -> list[str]:
    """``[<python>, -B, -P, -m, kronos_agent.finetune.runner, --run-dir, <dir>]`` (interfaces
    5.6); the interpreter is the parent's (uv's)."""
    return [
        sys.executable,
        *CHILD_PYTHON_FLAGS,
        "-m",
        CHILD_MODULE,
        "--run-dir",
        str(Path(run_dir).absolute()),
    ]


def child_environment(
    compute: Compute, *, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The child's environment, built from scratch: the :data:`CHILD_ENV_PASSTHROUGH` variables
    present in ``environ`` (``os.environ`` by default), :data:`CHILD_ENV_FIXED`, every thread
    variable of ``runtime.THREAD_ENV_VARS`` at ``compute.threads`` and, on MPS only,
    :data:`MPS_WATERMARK_ENV` (``compute.mps_watermark_ratio``, 1.0 by default) with
    :data:`MPS_LOW_WATERMARK_ENV` at ``min(1.4, high)`` (torch refuses a low ratio above the high
    one). Nothing else of the parent reaches the child."""
    if not isinstance(compute, Compute):
        raise TrainError(f"expected Compute, got {type(compute).__name__}")
    source = os.environ if environ is None else environ
    environment = {name: source[name] for name in CHILD_ENV_PASSTHROUGH if name in source}
    environment.update(CHILD_ENV_FIXED)
    for name in THREAD_ENV_VARS:
        environment[name] = str(compute.threads)
    if compute.device == "mps":
        ratio = compute.mps_watermark_ratio
        high = DEFAULT_MPS_WATERMARK_RATIO if ratio is None else float(ratio)
        environment[MPS_WATERMARK_ENV] = repr(high)
        environment[MPS_LOW_WATERMARK_ENV] = repr(min(TORCH_MPS_LOW_WATERMARK_DEFAULT, high))
    return environment


def _read_result(run_dir: Path) -> dict[str, Any]:
    path = run_dir / runner.RESULT_JSON
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def _verify_vendored_tree() -> None:
    from ..kronos.vendored import VendoredTreeError, verify_vendored_tree
    from ..manifests import ManifestMismatch

    report = verify_vendored_tree(strict=True)
    if not report.ok:
        raise VendoredTreeError(str(ManifestMismatch(report.problems)))


class RunningChild:
    """A started child run; :meth:`wait` reaps it, verifies the tree and reads its result."""

    def __init__(
        self, run_id: str, run_dir: Path, process: subprocess.Popen[bytes], mode: str
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.process = process
        self.mode = mode
        self.started = time.monotonic()
        self.terminated = False

    def poll(self) -> int | None:
        return self.process.poll()

    def _signal(self, number: int) -> None:
        try:
            os.killpg(self.process.pid, number)  # the child leads its own session and group
        except ProcessLookupError:
            pass
        except PermissionError:
            self.process.send_signal(number)

    def terminate(self) -> None:
        """SIGTERM to the child's process group, :data:`KILL_GRACE_SECONDS`, then SIGKILL."""
        if self.process.poll() is not None:
            return
        self.terminated = True
        self._signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            self._signal(signal.SIGKILL)
            self.process.wait()

    def wait(self, *, timeout_seconds: float | None = None) -> RunnerResult:
        """Wait for the child (terminating it on timeout or on any exception in the parent),
        verify the vendored tree again and read ``runner_result.json``."""
        timed_out = False
        try:
            try:
                self.process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                self.terminate()
        except BaseException:
            self.terminate()
            raise
        _verify_vendored_tree()
        document = _read_result(self.run_dir)
        status = document.get("status")
        if status not in runner.STATUSES:
            killed = self.process.returncode < 0 or self.terminated
            status = "killed" if killed else "no_result"
        return RunnerResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            mode=self.mode,
            status=status,
            exit_code=int(self.process.returncode),
            timed_out=timed_out,
            document=document,
        )


def _verified_export(request: RunRequest) -> ExportRecord:
    """The request's export read back and verified (``export_csv.read_export``), unchanged since
    the request was made and made for the request's fold and window length (its recipe may differ
    in what the export does not depend on, the batch size of a timing configuration say)."""
    from .export_csv import read_export

    record = read_export(Path(request.export.directory))
    if record.content_sha256 != request.export.content_sha256:
        raise TrainError("the export changed since the request was made")
    fold = request.fold
    expected = {
        "fold": fold.label,
        "csv_start_ms": fold.csv_start_ms,
        "csv_end_ms": fold.csv_end_ms,
        "inner_validation_start_ms": fold.inner_validation_start_ms,
        "window_rows": request.recipe.window_rows,
    }
    differing = sorted(key for key, value in expected.items() if record.content.get(key) != value)
    if differing:
        raise TrainError(f"the export does not belong to fold {fold.label}: {differing} differ")
    return record


def start_run(request: RunRequest, *, root: Path | None = None) -> RunningChild:
    """Verify, write the run inputs and start the child (see the module docstring)."""
    if not isinstance(request, RunRequest):
        raise TrainError(f"expected a RunRequest, got {type(request).__name__}")
    _verify_vendored_tree()
    if request.model_files is None:
        from ..kronos.weights import WeightsError, verify_model
        from ..manifests import ManifestMismatch

        report = verify_model(request.recipe.model)
        if not report.ok:
            raise WeightsError(
                f"weights for {request.recipe.model!r} not verified: "
                f"{ManifestMismatch(report.problems)}"
            )
    _verified_export(request)
    run_dir = new_run_dir(request.fold.run_label, root=root)
    write_run_inputs(run_dir, request)
    with open(run_dir / runner.CHILD_LOG, "ab") as log:
        process = subprocess.Popen(
            child_command(run_dir),
            cwd=run_dir,
            env=child_environment(request.compute),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return RunningChild(run_dir.name, run_dir, process, request.mode)


def run(
    request: RunRequest, *, root: Path | None = None, timeout_seconds: float | None = None
) -> RunnerResult:
    """``start_run(...).wait(...)``: any mode, nothing registered (timing and preflight)."""
    return start_run(request, root=root).wait(timeout_seconds=timeout_seconds)


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _selected_files(result: RunnerResult) -> tuple[Path, dict[str, dict[str, Any]]]:
    """The selected checkpoint's directory, its recorded files verified on disk."""
    selected = result.document.get("selected")
    if not isinstance(selected, dict) or not isinstance(selected.get("dir"), str):
        raise ChildFailed(f"run {result.run_id} recorded no selected checkpoint")
    directory = (result.run_dir / selected["dir"]).resolve()
    if not directory.is_relative_to(result.run_dir.resolve()):
        raise ChildFailed(f"run {result.run_id}: the selected directory is outside its run")
    files = {item["name"]: item for item in selected.get("files", []) if isinstance(item, dict)}
    if sorted(files) != sorted(runner.CHECKPOINT_FILE_NAMES):
        raise ChildFailed(f"run {result.run_id}: the selected checkpoint lists other files")
    for name, item in files.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ChildFailed(f"run {result.run_id}: the selected {name} does not verify")
    return directory, files


def _shipped_start(request: RunRequest) -> tuple[Any, Any]:
    """The registered model and tokenizer repositories of the request's model; model files other
    than exactly those are refused (:class:`TrainError`): a fold checkpoint starts from the
    registered shipped weights (cold start, design 6.1)."""
    from ..kronos.weights import repos_for_model

    model_repo, tokenizer_repo = repos_for_model(request.recipe.model)
    files = _model_files(request)
    registered = (
        {item.name: item.sha256 for item in model_repo.files},
        {item.name: item.sha256 for item in tokenizer_repo.files},
    )
    if (dict(files.predictor_sha256), dict(files.tokenizer_sha256)) != registered:
        raise TrainError("a fold checkpoint must start from the registered shipped weights")
    return model_repo, tokenizer_repo


def _registration_id(request: RunRequest, *, root: Path | None) -> str:
    """The checkpoint id ``fit(register=True)`` installs, checked before the child starts.

    Refused (:class:`TrainError`): a wrapper of ``runner.WRAPPERS`` off, since only a recipe run
    takes the recipe's checkpoint id (a diagnostic configuration selects by the vendored
    validation loss, whose eval-mode cross-attention reads the target, or skips a guard; design
    3.1, 3.3), and model files other than the registered shipped ones. Refused
    (``registry.CheckpointExists``): an id the registry holds, or whose directory exists in the
    fine-tune root: a registered fold is never trained again (design 3.5).
    ``registry.install_checkpoint`` checks the id again under its lock.
    """
    from .recipe import recipe8
    from .registry import (
        FINETUNED_DIRNAME,
        CheckpointExists,
        CheckpointRef,
        checkpoint_dir,
        checkpoint_id_for,
        read_registry,
    )

    off = [name for name, on in asdict(request.wrappers).items() if not on]
    if off:
        raise TrainError(
            f"only a recipe run, every wrapper on, is registered; {off} are off (run it with "
            "register=False: design 3.1)"
        )
    _shipped_start(request)
    checkpoint_id = checkpoint_id_for(recipe8(request.recipe), request.fold.run_label)
    if any(entry.checkpoint_id == checkpoint_id for entry in read_registry(root)):
        raise CheckpointExists(
            f"checkpoint {checkpoint_id} is registered already: a registered fold is never "
            "trained again (nothing was started)"
        )
    target = checkpoint_dir(CheckpointRef(checkpoint_id, root=root))
    if target.exists() or target.is_symlink():
        raise CheckpointExists(
            f"{FINETUNED_DIRNAME}/{checkpoint_id} exists without a registry entry (an interrupted "
            "install left it): inspect it and remove it before training the fold again (nothing "
            "was started)"
        )
    return checkpoint_id


def _require_recipe_selection(result: RunnerResult) -> None:
    """The child's record must show a recipe run: every wrapper on and the state NLL4-TF selected
    (``selected.source``); any other run is never registered (:class:`TrainError`)."""
    document = result.document
    selected = document.get("selected")
    source = selected.get("source") if isinstance(selected, dict) else None
    wrappers = document.get("wrappers")
    if wrappers != dict.fromkeys(runner.WRAPPERS, True) or source != RECIPE_SELECTION:
        raise TrainError(
            f"run {result.run_id} is not a recipe run (wrappers {wrappers}, checkpoint selected "
            f"by {source!r}, not {RECIPE_SELECTION!r}): it is not registered"
        )


def fit(
    request: RunRequest,
    *,
    root: Path | None = None,
    timeout_seconds: float | None = None,
    register: bool = True,
) -> FitOutcome:
    """Run a ``train`` request; with ``register`` install the selected checkpoint in the
    registry, then remove the run's ``selected/`` and vendored keep-best directories.

    With ``register``, before the child starts: a request with a wrapper off or other model
    files than the registered shipped ones is refused (:class:`TrainError`), and so is a
    checkpoint id that is registered already or whose directory exists
    (``registry.CheckpointExists``); after it, a record that does not show every wrapper on and
    the state NLL4-TF selected (:class:`TrainError`). Diagnostic configurations run with
    ``register=False`` or :func:`run`.

    :class:`ChildMemoryAbort` for a memory abort, :class:`ChildFailed` for anything but
    ``completed`` (the child's error type, and its message unless the run is blinded).
    """
    if request.mode != "train":
        raise TrainError(f"fit runs a train request, got mode {request.mode!r}")
    checkpoint_id = _registration_id(request, root=root) if register else None
    started = time.monotonic()
    cpu_before = _children_cpu_seconds()
    result = run(request, root=root, timeout_seconds=timeout_seconds)
    wall_seconds = time.monotonic() - started
    cpu_seconds = _children_cpu_seconds() - cpu_before
    if result.status != "completed":
        error = result.document.get("error")
        error = error if isinstance(error, dict) else {}
        detail = str(error.get("type") or "")
        if not request.blinded and error.get("message"):
            detail += f": {error['message']}"
        timed_out = ", timed out" if result.timed_out else ""
        message = f"run {result.run_id} ended {result.status} (exit {result.exit_code}{timed_out})"
        if detail:
            message += f" ({detail})"
        if result.status == "memory_abort":
            raise ChildMemoryAbort(message)
        raise ChildFailed(message)
    directory, files = _selected_files(result)
    entry = None
    if checkpoint_id is not None:
        from ..kronos.weights import checkpoint_dir_problems
        from .registry import install_checkpoint

        _require_recipe_selection(result)
        entry = install_checkpoint(
            directory,
            payload=registry_payload(request, result, checkpoint_id=checkpoint_id),
            check=checkpoint_dir_problems,
            root=root,
        )
        for relative in (runner.SELECTED_DIRNAME, runner.VENDORED_BEST_RELPATH):
            target = (result.run_dir / relative).resolve()
            if target.is_relative_to(result.run_dir.resolve()) and target.is_dir():
                shutil.rmtree(target)
    return FitOutcome(
        run_id=result.run_id,
        run_dir=result.run_dir,
        status=result.status,
        finite=bool(result.document.get("finite")),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        selected_sha256=files["model.safetensors"]["sha256"],
        checkpoint=entry,
        blinded=request.blinded,
    )


def registry_payload(
    request: RunRequest, result: RunnerResult, *, checkpoint_id: str
) -> dict[str, Any]:
    """The 33 payload keys of a fold entry (interfaces 5.8, with the compute's activation
    checkpointing next to its device, threads and determinism) for the run ``result`` of
    ``request``. A fold checkpoint starts from the registered shipped weights: other model files
    are refused (:class:`TrainError`)."""
    from ..kronos.vendored import source_manifest_path
    from ..models import spec_for
    from .recipe import recipe_id, training_seeds

    recipe, fold, export, compute = request.recipe, request.fold, request.export, request.compute
    spec = spec_for(recipe.model)
    model_repo, tokenizer_repo = _shipped_start(request)
    base_sha256 = {item.name: item.sha256 for item in model_repo.files}
    tokenizer_sha256 = {item.name: item.sha256 for item in tokenizer_repo.files}
    document = result.document
    best = document.get("vendored_best_model") or {}
    best_sha256 = next(
        (
            item.get("sha256")
            for item in best.get("files", [])
            if isinstance(item, dict) and item.get("name") == "model.safetensors"
        ),
        None,
    )
    identifier = recipe_id(recipe)
    return {
        "kind": "fold",
        "checkpoint_id": checkpoint_id,
        "model": recipe.model,
        "base": {"repo_id": model_repo.repo_id, "sha256": base_sha256},
        "tokenizer": {"repo_id": tokenizer_repo.repo_id, "sha256": tokenizer_sha256},
        "expected_config": dict(model_repo.expected_config or {}),
        "state_dict_numel": spec.state_dict_numel,
        "data_end_ms": fold.data_end_ms,
        "data_end": utc_text(fold.data_end_ms),
        "recipe_id": identifier,
        "fold": fold.label,
        "replicate": fold.replicate,
        "fit_range_ms": list(fold.fit_range_ms),
        "inner_validation_range_ms": list(fold.inner_validation_range_ms),
        "test_start_ms": fold.test_start_ms,
        "test_end_ms": fold.test_end_ms,
        "train_csv_sha256": export.csv_sha256,
        "nll_inputs_sha256": export.nll_inputs_sha256,
        "export_content_sha256": export.content_sha256,
        "store_meta_sha256": export.store_meta_sha256,
        "run_id": result.run_id,
        "fit_record_sha256": sha256_file(result.run_dir / runner.RESULT_JSON),
        "vendored_best_model_sha256": best_sha256,
        "device": compute.device,
        "threads": compute.threads,
        "deterministic": compute.deterministic,
        "activation_checkpointing": compute.activation_checkpointing,
        "torch_version": document.get("torch_version"),
        "platform": document.get("platform"),
        "seeds": asdict(training_seeds(identifier, fold.label, fold.replicate)),
        "code": code_version(),
        "training_code_sha256": training_code_sha256(),
        "vendored_manifest_sha256": sha256_file(source_manifest_path()),
    }


__all__ = [
    "CHILD_ENV_FIXED",
    "CHILD_ENV_PASSTHROUGH",
    "CHILD_MODULE",
    "CHILD_PYTHON_FLAGS",
    "DEFAULT_MPS_WATERMARK_RATIO",
    "KILL_GRACE_SECONDS",
    "MAX_MPS_WATERMARK_RATIO",
    "MPS_LOW_WATERMARK_ENV",
    "MPS_WATERMARK_ENV",
    "RECIPE_SELECTION",
    "TORCH_MPS_LOW_WATERMARK_DEFAULT",
    "TRAINING_CODE_FILES",
    "ChildFailed",
    "ChildMemoryAbort",
    "Compute",
    "FitOutcome",
    "ModelFiles",
    "RunRequest",
    "RunnerResult",
    "RunningChild",
    "TimingSpec",
    "TrainError",
    "Wrappers",
    "child_command",
    "child_environment",
    "fit",
    "new_run_dir",
    "registry_payload",
    "run",
    "shipped_model_files",
    "start_run",
    "training_code_sha256",
    "write_run_inputs",
]
