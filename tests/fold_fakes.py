"""Stand-ins for the fold command's tests (``test_finetune_walkforward.py``,
``test_finetune_fold_cli.py``): a support module, not a test module.

- A timing record with the measurements of ``reports/finetune/timing-2.json`` (D-056: MPS,
  2 threads, batch 32 with activation checkpointing, 5.449 s per step, theta 1.004; M8 and M4 as
  measured), tied to this process's torch, platform and training code so that it is fresh, and
  written to the test's own reports root, where the real ``timing.load_timing`` reads it.
- A stand-in export: the tiny synthetic fold of ``finetune_fakes`` exported by the real exporter
  into the directory of the fold asked for (``fold_export_dir``), so that ``read_export`` and the
  pilot's resolution work; the fold's own candles are never read.
- A stand-in training child: it creates the run directory as ``train.fit`` does, writes a fold
  record (``runner_result.json``) with distinctive loss values (searched for by the blinding
  tests), and either registers a stand-in checkpoint (``finetune_fakes.install_stand_in``) and
  returns a ``train.FitOutcome``, or raises what ``train.fit`` raises for a failed child.
- A stand-in clock and CPU counter that the stand-in child advances.

Nothing here trains, times, exports real candles or reads a real root (D-055 (6)): every root is
the test's own (``conftest.isolated_roots``).
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import finetune_fakes as ff

from kronos_agent.finetune import registry, timing, train
from kronos_agent.finetune.export_csv import export_frame, fold_export_dir, read_export
from kronos_agent.finetune.recipe import recipe8, recipe_id, training_seeds
from kronos_agent.finetune.walkforward import FoldSteps
from kronos_agent.hashing import sha256_file
from kronos_agent.paths import default_label, reports_dir
from kronos_agent.runtime import require_configured

# the measurements of reports/finetune/timing-2.json (sha256 73832b1a...), as recorded
SECONDS_PER_STEP = 5.448582562501542
THETA = 1.004073166669094
M8_SECONDS_PER_DECISION = 0.15522049479174407
M4_CPU_SECONDS = 73.749217
M4_SECONDS = 1200.765995499998
PILOT_HOURS = 4.945  # E_max 6: 17,801.8 s (INTERFACES-3a section 1)
REFIT_HOURS = 9.779  # E_max 12: 35,204.3 s
PILOT_SECONDS = 17_801.848
REFIT_SECONDS = 35_204.322

# distinctive values of the stand-in fold records: a blinded fold's output never shows them
SHIPPED_NLL = 3.2198765
NLL_STEP = 0.0123457
TRAIN_LOSS = 4.5678912
VAL_LOSS = 5.6789123
STEP_MEDIAN = 7.7654321
DISTINCTIVE = ("3.2198", "3.20753", "4.5678", "5.6789", "7.7654")


def timing_choice(**changes: Any) -> dict[str, Any]:
    """The choice of ``timing-2.json`` (``changes`` replace keys)."""
    choice: dict[str, Any] = {
        "device": "mps",
        "threads": 2,
        "batch_size": 32,
        "deterministic": True,
        "activation_checkpointing": True,
        "mps_watermark_ratio": 1.0,
        "batch_32_fits": False,
        "seconds_per_step": SECONDS_PER_STEP,
        "theta": THETA,
        "reasons": ["device MPS: as measured (stand-in)"],
    }
    choice.update(changes)
    return choice


def timing_record(**choice: Any) -> dict[str, Any]:
    """A fresh timing record (this process's torch, platform and training code)."""
    return {
        "study": timing.TIMING_STEM,
        "schema_version": timing.TIMING_SCHEMA_VERSION,
        "status": "completed",
        "environment": {
            "torch_version": require_configured().torch_version,
            "platform": default_label(),
        },
        "code": {"training_code_sha256": train.training_code_sha256()},
        "estimate": {},
        "spend": {},
        "approval": None,
        "measurements": {
            "M4": {
                "status": "completed",
                "cpu_seconds": M4_CPU_SECONDS,
                "seconds": M4_SECONDS,
            },
            "M8": {"status": "completed", "seconds_per_decision": M8_SECONDS_PER_DECISION},
        },
        "choice": timing_choice(**choice),
    }


def timing_directory() -> Path:
    return reports_dir() / timing.FINETUNE_REPORTS_SUBDIR


def write_timing_record(record: dict[str, Any] | None = None, **choice: Any) -> dict[str, Any]:
    """Write ``record`` (default: :func:`timing_record`) as the test's ``timing-2.json``, the
    way the report writer does (indented, sorted keys); returns it."""
    record = record if record is not None else timing_record(**choice)
    directory = timing_directory()
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
    (directory / f"{timing.TIMING_STEM}.json").write_text(text + "\n", encoding="ascii")
    return record


@dataclass
class Script:
    """What one call of the stand-in child does.

    ``status``: ``completed`` (registered), ``failed``, ``memory_abort``, ``time_abort``,
    ``no_result`` (no record written) or ``killed`` (no run directory at all); ``interrupt``
    raises ``KeyboardInterrupt`` after writing a ``failed`` record. ``seconds`` and
    ``cpu_seconds`` advance the stand-in clock and CPU counter. ``lock_record`` makes the fold
    record unreadable (mode 0): opening it raises. ``error_type`` names the error of a record
    that is not completed (``SystemExit``: the child stopped by a signal, as ``finetune.runner``
    records it). ``after_registration`` raises that exception (an instance) once the stand-in
    checkpoint is registered, as an interrupt or an error after ``train.fit``'s registration
    would. ``timeout`` makes the call last as long as ``train.fit`` waits when its timeout fires
    (the timeout, then the kill grace), instead of ``seconds``."""

    status: str = "completed"
    selected: int = 3
    seconds: float = 600.0
    cpu_seconds: float = 60.0
    stop_reason: str = "early_stop"
    lock_record: bool = False
    interrupt: bool = False
    error_type: str = "RunnerError"
    after_registration: BaseException | None = None
    timeout: bool = False


@dataclass
class World:
    """The stand-ins of one test and what they saw."""

    now: float = 0.0  # the stand-in clock, seconds
    cpu: float = 0.0  # the stand-in children's CPU-seconds
    mps: bool = True
    script: list[Script] = field(default_factory=list)
    fits: list[dict[str, Any]] = field(default_factory=list)
    exports: list[dict[str, Any]] = field(default_factory=list)
    # the live training children the stand-in process list shows: (pid, run directory)
    running: list[tuple[int, str]] = field(default_factory=list)

    # -- the steps ------------------------------------------------------------------------------

    def steps(self) -> FoldSteps:
        return FoldSteps(
            export_fold=self.export_fold,
            fit=self.fit,
            load_timing=timing.load_timing,
            mps_available=lambda: self.mps,
            clock=lambda: self.now,
            children_cpu_seconds=lambda: self.cpu,
            running_children=lambda: tuple(self.running),
        )

    def export_fold(self, fold: Any, recipe: Any, *, root: Any = None, sealed_access: Any = None):
        """The tiny synthetic export, written into the fold's export directory once."""
        directory = fold_export_dir(recipe8(recipe), fold.label, root=root)
        self.exports.append(
            {"fold": fold.run_label, "recipe_id": recipe_id(recipe), "access": sealed_access}
        )
        if not (directory / "export.json").is_file():
            tiny = ff.tiny_recipe()
            tiny_fold = ff.tiny_fold(tiny)
            export_frame(ff.tiny_frame(tiny_fold), tiny_fold, tiny, directory)
        return read_export(directory)

    def fit(self, request: Any, *, root: Any, timeout_seconds: float, register: bool):
        from kronos_agent import ledger

        script = self.script.pop(0) if self.script else Script()
        self.fits.append(
            {
                "request": request,
                "root": root,
                "timeout_seconds": timeout_seconds,
                "register": register,
                "ledger_kinds": [entry.kind for entry in ledger.read_ledger()],
                "status": script.status,
            }
        )
        self.now += timeout_seconds + train.KILL_GRACE_SECONDS if script.timeout else script.seconds
        self.cpu += script.cpu_seconds
        if script.status == "killed":
            raise train.ChildFailed("the stand-in child was killed before its run directory")
        run_dir = train.new_run_dir(request.fold.run_label, root=root)
        if script.interrupt:
            write_fold_record(run_dir, request, script, status="failed")
            raise KeyboardInterrupt
        if script.status != "no_result":
            write_fold_record(run_dir, request, script, status=script.status)
        if script.status == "memory_abort":
            raise train.ChildMemoryAbort(f"run {run_dir.name} ended memory_abort")
        if script.status != "completed":
            raise train.ChildFailed(f"run {run_dir.name} ended {script.status}")
        entry = register_stand_in(request, run_dir, root=root)
        if script.after_registration is not None:
            raise script.after_registration
        record = run_dir / "runner_result.json"
        if script.lock_record:
            os.chmod(record, 0)
        weights = next(item.sha256 for item in entry.files if item.name == "model.safetensors")
        return train.FitOutcome(
            run_id=run_dir.name,
            run_dir=run_dir,
            status="completed",
            finite=True,
            wall_seconds=script.seconds,
            cpu_seconds=script.cpu_seconds,
            selected_sha256=weights,
            checkpoint=entry,
            blinded=request.blinded,
        )


def fold_record(
    run_id: str,
    *,
    status: str,
    selected: int,
    max_epochs: int,
    stop: str,
    seconds: float,
    error_type: str = "RunnerError",
) -> dict[str, Any]:
    """A ``runner_result.json`` with the keys the fold command reads and distinctive values."""
    last = min(max_epochs, selected + 2) if stop == "early_stop" else max_epochs
    epochs: list[dict[str, Any]] = [
        {
            "pseudo_epoch": 0,
            "nll4_tf": SHIPPED_NLL,
            "train_loss": None,
            "vendored_val_loss": None,
            "lr_end": None,
            "seconds": 52.0,
        }
    ]
    for epoch in range(1, last + 1):
        epochs.append(
            {
                "pseudo_epoch": epoch,
                "nll4_tf": SHIPPED_NLL - NLL_STEP * (epoch if epoch <= selected else selected - 1),
                "train_loss": TRAIN_LOSS,
                "vendored_val_loss": VAL_LOSS,
                "lr_end": 1e-7,
                "seconds": 2800.0,
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "mode": "train",
        "status": status,
        "stop_reason": stop if status == "completed" else None,
        "error": None if status == "completed" else {"type": error_type, "message": "x"},
        "finite": status == "completed",
        "wall_seconds": seconds,
        "cpu_seconds": seconds * 0.06,
        "epochs": epochs,
        "selected_pseudo_epoch": selected if status == "completed" else None,
        "step_seconds": {"count": 10, "median": STEP_MEDIAN, "p10": 5.4, "p90": 5.5},
    }


def write_fold_record(run_dir: Path, request: Any, script: Script, *, status: str) -> Path:
    document = fold_record(
        run_dir.name,
        status=status,
        selected=script.selected,
        max_epochs=request.recipe.max_epochs,
        stop=script.stop_reason,
        seconds=script.seconds,
        error_type=script.error_type,
    )
    path = run_dir / "runner_result.json"
    text = json.dumps(document, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
    path.write_text(text + "\n", encoding="ascii")
    return path


def register_stand_in(request: Any, run_dir: Path, *, root: Any) -> Any:
    """Register the run's stand-in checkpoint as ``train.fit`` would (the recipe's id for the
    fold's run label; the fold record's sha256; the export's hashes)."""
    from dataclasses import asdict

    recipe, fold, export, compute = request.recipe, request.fold, request.export, request.compute
    identifier = recipe_id(recipe)
    checkpoint_id = registry.checkpoint_id_for(recipe8(recipe), fold.run_label)
    return ff.install_stand_in(
        root if root is not None else registry.finetuned_dir(),
        checkpoint_id,
        data_end_ms=fold.data_end_ms,
        recipe_id=identifier,
        fold=fold.label,
        replicate=fold.replicate,
        run_id=run_dir.name,
        fit_record_sha256=sha256_file(run_dir / "runner_result.json"),
        export_content_sha256=export.content_sha256,
        train_csv_sha256=export.csv_sha256,
        nll_inputs_sha256=export.nll_inputs_sha256,
        device=compute.device,
        threads=compute.threads,
        deterministic=compute.deterministic,
        activation_checkpointing=compute.activation_checkpointing,
        seeds=asdict(training_seeds(identifier, fold.label, fold.replicate)),
    )


def unlock(path: Path) -> None:
    """Make a record locked by ``Script(lock_record=True)`` readable again (teardown)."""
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def write_bundle(recipe: Any, *, tests: tuple[str, ...] = ("T-A", "T-B"), now: Any = None) -> Any:
    """A stand-in pre-registration bundle of ``recipe`` under the test's reports root and its
    ``preregistration`` entry in the test's ledger (at ``now``, default the current time; the
    ledger's times never decrease); returns the entry."""
    from kronos_agent import ledger
    from kronos_agent.provenance import forecasting_code_sha256

    identifier = recipe_id(recipe)
    relpath = f"finetune/prereg-{identifier[:8]}.config.json"
    path = reports_dir() / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps({"stand_in_bundle": identifier}, sort_keys=True).encode("ascii"))
    return ledger.record_preregistration(
        bundle_relpath=relpath,
        recipe_id=identifier,
        tests=tests,
        forecasting_code_sha256=forecasting_code_sha256(),
        training_code_sha256=train.training_code_sha256(),
        now=now,
    )


def bundle_path(recipe: Any) -> Path:
    return reports_dir() / f"finetune/prereg-{recipe_id(recipe)[:8]}.config.json"


def ledger_kinds() -> list[str]:
    from kronos_agent import ledger

    return [entry.kind for entry in ledger.read_ledger()]


def ledger_text() -> str:
    from kronos_agent import ledger

    path = ledger.ledger_path()
    return path.read_text(encoding="ascii") if path.is_file() else ""
