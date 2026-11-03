"""The training child: one fit, timing measurement or preflight of Kronos-small in its own process.

``python -B -P -m kronos_agent.finetune.runner --run-dir DIR`` (``docs/RETRAINING.md`` 3.1). A
child process keeps the patched vendored module globals, the device state and the training RNG
out of the parent. It reads only ``run.json`` (the run specification ``finetune.train`` wrote),
``run.yaml`` (what the vendored loader reads), the fold's export files and the verified weights,
and always writes ``runner_result.json`` (the fold record; blinded for folds 2 onwards).

The pinned predictor phase of ``finetune_csv`` runs unmodified: ``SequentialTrainer.
train_basemodel_phase()`` (TS:148-262, never ``run_training``, which would turn the early stop
into a failure) calls ``train_model`` (FB:239-364). The project reaches the vendored code only
through module globals, a subclass and instance attributes, each switched by ``run.json``
(:data:`WRAPPERS`; design 3.1, rows):

- ``device`` (1): a subclass of ``SequentialTrainer`` overriding only ``_setup_device``;
- ``router`` (2): ``torch.multinomial`` is a module attribute looked up at call time (K:271);
  :class:`MultinomialRouter` draws on the CPU from the fold's own generator, on every device;
- ``clean_validation`` (3, 4): :class:`CleanValidation` keeps the vendored validation as a
  diagnostic, causal (the dependency layer's cross-attention in train mode, its dropouts are 0)
  and reseeded, with every RNG state restored;
- ``early_stopping`` (5): NLL4-TF (``finetune.nll``) on a CPU copy of the weights at every
  pseudo-epoch boundary, patience, the best state kept as a CPU copy and saved by the child
  (with it off, a train run still records NLL4-TF and keeps the vendored keep-best);
- ``pseudo_epochs`` (6): the train loader rebuilt as ``pseudo_epoch_steps`` batches drawn
  without replacement by a ``RandomSampler`` on the fold's own generator;
- ``gap_aware`` (8): the dataset subclass of :func:`make_dataset_class` when the export has a
  gap or the normalisation is not the vendored one; a run on an export with a gap is refused
  with it off (the vendored dataset would build windows across the missing hours);
- ``weight_check`` (9): every tensor of the loaded models equals the verified safetensors file.

Always on, changing no value (the equivalence test runs with them): :class:`TrainLoaderGate`
around the train loader (pseudo-epoch boundaries, step timing, the memory guard, the deadline,
the end of a timing measurement), loss recording through an instance attribute on
``model.head`` and a global optimizer step hook. With every wrapper off and deterministic
algorithms off, the child reproduces ``python -B train_sequential.py --skip-tokenizer`` bit for
bit (test T1). ``num_workers`` is 0 (D-014): loader workers would import the vendored tree
outside :func:`import_vendored_finetune`.

Exit codes: 0 completed, 1 failed, 3 memory abort, 4 time abort, ``128 + n`` after SIGTERM or
SIGHUP (the result is still written, status ``failed``). Nothing absolute reaches the result
except through ``paths.redact_home``; ``run.json``, ``run.yaml`` and ``child.log`` (vendored
stdout and logs) hold absolute paths and stay in the gitignored run directory.

Only the standard library and light project modules are imported here; ``torch``, numpy and the
vendored code are imported after ``runtime.configure_runtime()``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from ..data.fsutil import atomic_write_text
from ..hashing import sha256_bytes, sha256_file
from ..paths import default_label, redact_home, vendored_kronos_dir
from ..runtime import configure_runtime

RUN_JSON = "run.json"
RUN_YAML = "run.yaml"
RESULT_JSON = "runner_result.json"
CHILD_LOG = "child.log"
VENDORED_SAVE_DIRNAME = "vendored"  # base_save_path
SELECTED_DIRNAME = "selected"
VENDORED_BEST_RELPATH = f"{VENDORED_SAVE_DIRNAME}/basemodel/best_model"
CHECKPOINT_FILE_NAMES: tuple[str, ...] = ("config.json", "model.safetensors")
RUN_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
MODES: tuple[str, ...] = ("train", "timing", "preflight")
STATUSES: tuple[str, ...] = ("completed", "failed", "memory_abort", "time_abort")
STOP_REASONS: tuple[str, ...] = ("max_epochs", "early_stop", "timing_done", "preflight_done")
EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_MEMORY_ABORT = 3
EXIT_TIME_ABORT = 4
WRAPPERS: tuple[str, ...] = (
    "device",
    "router",
    "clean_validation",
    "early_stopping",
    "pseudo_epochs",
    "gap_aware",
    "weight_check",
)
PREFLIGHT_CHECKS: tuple[str, ...] = (
    "multinomial",
    "parity",
    "steps",
    "deterministic",
    "checkpoint_gradients",
)
DEVICES: tuple[str, ...] = ("cpu", "mps")
MEMORY_CHECK_EVERY_STEPS = 50
MAX_SWAP_GROWTH_BYTES = 1 << 30
CRITICAL_PRESSURE_LEVEL = 4  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
VENDORED_FINETUNE_MODULES: tuple[str, ...] = (
    "train_sequential",
    "finetune_base_model",
    "finetune_tokenizer",
    "config_loader",
)
FINETUNE_CSV_DIRNAME = "finetune_csv"
SYSCTL = "/usr/sbin/sysctl"
SYSCTL_TIMEOUT_SECONDS = 10.0
# preflight (design (a), M2-M3): the defaults the parent writes into run.json
PREFLIGHT_MULTINOMIAL_ROWS = 16_384  # one batch of 32 windows x 512 positions
PREFLIGHT_MULTINOMIAL_REPETITIONS = 64
PREFLIGHT_STEPS = 20
PREFLIGHT_ALPHA = 1e-3  # the chi-square p-value below which a sampler is refused
PREFLIGHT_MIN_EXPECTED = 5.0  # chi-square cells with fewer expected draws are pooled
PREFLIGHT_PARITY_MAX_RELATIVE = 1e-3
PREFLIGHT_PARITY_MIN_COSINE = 0.9999
PREFLIGHT_STEPS_MAX_RELATIVE = 0.01
VENDORED_GRAD_CLIP = 3.0  # FB:296, hard-coded
NORMALISATION_EPSILON = 1e-5  # FB:126
VENDORED_INDEX_PRIME = 9973  # FB:114
VENDORED_INDEX_OFFSET = 104729  # FB:114
ERROR_MESSAGE_MAX = 500

_RUN_KEYS: tuple[str, ...] = (
    "blinded",
    "compute",
    "created_at",
    "deadline_seconds",
    "export_content_sha256",
    "fold",
    "memory",
    "mode",
    "model_files",
    "nll_batch_size",
    "nll_inputs",
    "preflight",
    "recipe",
    "recipe_id",
    "run_id",
    "schema_version",
    "seeds",
    "timing",
    "train_csv",
    "windows",
    "wrappers",
    "yaml",
)
_FOLD_KEYS = (
    "csv_end_ms",
    "csv_start_ms",
    "inner_validation_start_ms",
    "label",
    "layout",
    "replicate",
    "run_label",
)
_SEED_KEYS = ("canonical", "digest", "router_seed", "sampler_seed", "val_seed", "yaml_seed")
_TRAIN_CSV_KEYS = ("n_fit", "n_val", "path", "rows", "sha256")
_NLL_INPUT_KEYS = ("decisions", "path", "sha256")
_WINDOW_KEYS = ("fit_windows_kept", "gap_aware", "val_windows_kept")
_MODEL_FILE_KEYS = ("dir", "sha256")
_COMPUTE_KEYS = (
    "activation_checkpointing",
    "deterministic",
    "device",
    "memory_guard",
    "mps_watermark_ratio",
    "threads",
)
_MEMORY_KEYS = ("critical_pressure_level", "every_steps", "max_swap_growth_bytes")
_TIMING_KEYS = ("timed_seconds", "timed_steps", "warmup_steps")
_PREFLIGHT_KEYS = ("checks", "multinomial_repetitions", "multinomial_rows", "steps")
_YAML_KEYS = ("name", "sha256")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SWAP_USED = re.compile(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)([BKMGT])", re.IGNORECASE)
_SWAP_UNITS = {"B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
_DETERMINISTIC_OP = re.compile(r"^\s*(\S+) does not have a deterministic implementation")


class RunnerError(RuntimeError):
    """The run cannot proceed as specified (child exit 1)."""


class EarlyStop(Exception):
    """Control flow: NLL4-TF did not improve for ``patience`` pseudo-epochs."""


class TimingDone(Exception):
    """Control flow: the timed steps of a measurement are done."""


class MemoryAbort(RuntimeError):
    """Memory pressure or swap growth passed its limit (child exit 3; not a fold failure)."""


class TimeAbort(RuntimeError):
    """The run passed its ``deadline_seconds`` (child exit 4)."""


@dataclass(frozen=True)
class VendoredFinetune:
    """The vendored ``finetune_csv`` modules, imported once per process."""

    train_sequential: ModuleType
    finetune_base_model: ModuleType
    config_loader: ModuleType
    CustomKlineDataset: type
    SequentialTrainer: type


@dataclass(frozen=True)
class MemoryReading:
    pressure_level: int
    swap_used_bytes: int


def _now_text() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else None


# -- the vendored finetune_csv modules ----------------------------------------------------------

_vendored: VendoredFinetune | None = None


def _finetune_csv_dir() -> Path:
    return vendored_kronos_dir() / FINETUNE_CSV_DIRNAME


def import_vendored_finetune() -> VendoredFinetune:
    """Verify the vendored tree, then import ``train_sequential`` and the modules it imports.

    ``kronos.vendored.import_kronos()`` runs first, so their ``from model import ...`` binds the
    verified upstream package. ``third_party/kronos/finetune_csv`` is put at ``sys.path[0]``
    with bytecode writing and the pycache prefix off for the import only; that entry and every
    ``'../'`` the modules append (TS:10, FB:20) are removed afterwards, also on failure. A module
    of one of their names imported from anywhere else is refused. Cached per process.
    """
    global _vendored
    if _vendored is not None:
        return _vendored
    from ..kronos.vendored import VendoredTreeError, import_kronos, verify_vendored_tree
    from ..manifests import ManifestMismatch

    import_kronos()
    report = verify_vendored_tree(strict=True)
    if not report.ok:
        raise VendoredTreeError(str(ManifestMismatch(report.problems)))
    directory = _finetune_csv_dir().resolve()
    for name in VENDORED_FINETUNE_MODULES:
        existing = sys.modules.get(name)
        if existing is None:
            continue
        location = getattr(existing, "__file__", None)
        if not location or not Path(location).resolve().is_relative_to(directory):
            raise VendoredTreeError(f"a foreign module named {name!r} is already imported")
    entry = str(_finetune_csv_dir())
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_pycache_prefix = sys.pycache_prefix
    sys.path.insert(0, entry)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = None
    try:
        import train_sequential
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        sys.pycache_prefix = previous_pycache_prefix
        while entry in sys.path:
            sys.path.remove(entry)
        while "../" in sys.path:
            sys.path.remove("../")
    modules = {name: sys.modules.get(name) for name in VENDORED_FINETUNE_MODULES}
    for name, module in modules.items():
        location = getattr(module, "__file__", None)
        if module is None or not location or not Path(location).resolve().is_relative_to(directory):
            raise VendoredTreeError(f"{name!r} did not resolve inside the vendored finetune_csv")
    finetune_base_model = modules["finetune_base_model"]
    _vendored = VendoredFinetune(
        train_sequential=train_sequential,
        finetune_base_model=finetune_base_model,
        config_loader=modules["config_loader"],
        CustomKlineDataset=finetune_base_model.CustomKlineDataset,
        SequentialTrainer=train_sequential.SequentialTrainer,
    )
    return _vendored


# -- datasets and loaders ------------------------------------------------------------------------


def make_dataset_class(base: type, *, normalisation: str) -> type:
    """A subclass of the vendored ``CustomKlineDataset`` that skips gaps and can normalise over
    the context rows only (design 1.6, 2.2).

    ``__init__`` calls the vendored one, then keeps the window starts whose ``W`` rows span
    exactly ``W - 1`` hours (``n_samples`` is their count). ``__getitem__`` repeats FB:118-132
    with the vendored index mappings restricted to the kept starts (train ``k = (idx * 9973 +
    (epoch + 1) * 104729) % n_kept``, validation ``k = idx % n_kept``). ``"vendored"``
    normalises over all ``W`` rows (FB:125-127); ``"context_only"`` uses the statistics of rows
    ``0 .. lookback - 1``, reduced on a Fortran-ordered copy as the predictor reduces its context
    (K:629), and applies them to every row. Without a gap and with ``"vendored"`` it returns
    exactly what the vendored class returns.
    """
    import numpy as np
    import pandas as pd
    import torch

    if normalisation not in ("vendored", "context_only"):
        raise RunnerError(f"unknown normalisation {normalisation!r}")
    hour_ns = pd.Timedelta(hours=1).value

    class GapAwareKlineDataset(base):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.normalisation = normalisation
            stamps = self.timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            vendored_count = max(len(stamps) - self.window + 1, 0)
            if vendored_count:
                span = stamps[self.window - 1 :] - stamps[:vendored_count]
                starts = np.flatnonzero(span == (self.window - 1) * hour_ns)
            else:
                starts = np.empty(0, dtype=np.int64)
            self.kept_starts = starts.astype(np.int64)
            self.vendored_samples = vendored_count
            self.n_samples = int(self.kept_starts.size)

        def __getitem__(self, idx: int) -> tuple[Any, Any]:
            max_start = len(self.data) - self.window
            if max_start <= 0:
                raise ValueError("Data length insufficient to create samples")
            if self.n_samples == 0:
                raise ValueError("no window of this split spans consecutive hours")
            if self.data_type == "train":
                epoch = getattr(self, "current_epoch", 0)
                k = (idx * VENDORED_INDEX_PRIME + (epoch + 1) * VENDORED_INDEX_OFFSET) % (
                    self.n_samples
                )
            else:
                k = idx % self.n_samples
            start_idx = int(self.kept_starts[k])
            end_idx = start_idx + self.window
            window_data = self.data.iloc[start_idx:end_idx]
            x = window_data[self.feature_list].values.astype(np.float32)
            x_stamp = window_data[self.time_feature_list].values.astype(np.float32)
            if self.normalisation == "vendored":
                x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            else:
                context = np.asfortranarray(x[: self.lookback_window])
                x_mean, x_std = np.mean(context, axis=0), np.std(context, axis=0)
            x = (x - x_mean) / (x_std + NORMALISATION_EPSILON)
            x = np.clip(x, -self.clip, self.clip)
            return torch.from_numpy(x), torch.from_numpy(x_stamp)

    return GapAwareKlineDataset


def pseudo_epoch_loader(dataset: Any, *, batch_size: int, steps: int, seed: int) -> Any:
    """The train loader of the ``pseudo_epochs`` wrapper (design 3.1 row 6, 3.4).

    ``steps`` batches of ``batch_size`` windows drawn without replacement by a ``RandomSampler``
    whose generator is seeded with ``seed``; the loader shares that generator, so its per-epoch
    base-seed draw does not touch the global RNG either. No workers, no pinned memory.
    """
    import torch
    from torch.utils.data import DataLoader, RandomSampler

    if not _is_int(steps) or steps < 1 or not _is_int(batch_size) or batch_size < 1:
        raise RunnerError(f"steps and batch_size must be positive, got {steps}, {batch_size}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(
        dataset, replacement=False, num_samples=steps * batch_size, generator=generator
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        generator=generator,
    )


# -- memory --------------------------------------------------------------------------------------


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            [SYSCTL, "-n", name],
            capture_output=True,
            text=True,
            timeout=SYSCTL_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def parse_swap_used(text: str) -> int | None:
    """Bytes of ``used = <n><unit>`` in the output of ``sysctl -n vm.swapusage``."""
    match = _SWAP_USED.search(text or "")
    if match is None:
        return None
    return round(float(match.group(1)) * _SWAP_UNITS[match.group(2).upper()])


def read_memory() -> MemoryReading | None:
    """The memory-pressure level and the swap in use (macOS ``sysctl``); ``None`` elsewhere or
    when a reading fails."""
    if sys.platform != "darwin" or not Path(SYSCTL).is_file():
        return None
    level_text = _sysctl("kern.memorystatus_vm_pressure_level")
    swap_text = _sysctl("vm.swapusage")
    if level_text is None or swap_text is None:
        return None
    try:
        level = int(level_text)
    except ValueError:
        return None
    swap = parse_swap_used(swap_text)
    if swap is None:
        return None
    return MemoryReading(pressure_level=level, swap_used_bytes=swap)


class MemoryGuard:
    """Abort a run under memory pressure or swap growth (design 3.5).

    :meth:`check` reads ``read()`` every ``every_steps`` steps: :class:`MemoryAbort` when the
    pressure level reaches ``critical_level`` or the swap in use has grown by more than
    ``max_swap_growth_bytes`` since the first reading (taken by :meth:`start`, or by the first
    check). A reading that is unavailable is a :class:`RunnerError`.
    """

    def __init__(
        self,
        read: Callable[[], MemoryReading | None] = read_memory,
        *,
        every_steps: int = MEMORY_CHECK_EVERY_STEPS,
        max_swap_growth_bytes: int = MAX_SWAP_GROWTH_BYTES,
        critical_level: int = CRITICAL_PRESSURE_LEVEL,
    ) -> None:
        if not _is_int(every_steps) or every_steps < 1:
            raise RunnerError(f"every_steps must be a positive integer, got {every_steps!r}")
        if not _is_int(max_swap_growth_bytes) or max_swap_growth_bytes < 0:
            raise RunnerError("max_swap_growth_bytes must be an integer >= 0")
        if not _is_int(critical_level) or critical_level < 1:
            raise RunnerError("critical_level must be a positive integer")
        self._read = read
        self.every_steps = every_steps
        self.max_swap_growth_bytes = max_swap_growth_bytes
        self.critical_level = critical_level
        self.swap_start_bytes: int | None = None
        self.swap_growth_bytes: int | None = None
        self.max_pressure_level: int | None = None
        self.checks = 0

    def _reading(self) -> MemoryReading:
        reading = self._read()
        if reading is None:
            raise RunnerError(
                "memory readings (sysctl) are unavailable on this machine; the memory guard "
                "needs them"
            )
        self.checks += 1
        if self.swap_start_bytes is None:
            self.swap_start_bytes = reading.swap_used_bytes
        growth = reading.swap_used_bytes - self.swap_start_bytes
        self.swap_growth_bytes = max(growth, self.swap_growth_bytes or 0)
        self.max_pressure_level = max(reading.pressure_level, self.max_pressure_level or 0)
        return reading

    def start(self) -> None:
        """Take the first reading (the swap baseline); fails when readings are unavailable."""
        self._reading()

    def check(self, step: int) -> None:
        if step % self.every_steps:
            return
        reading = self._reading()
        if reading.pressure_level >= self.critical_level:
            raise MemoryAbort(
                f"memory pressure level {reading.pressure_level} at step {step} (critical is "
                f"{self.critical_level})"
            )
        growth = reading.swap_used_bytes - (self.swap_start_bytes or 0)
        if growth > self.max_swap_growth_bytes:
            raise MemoryAbort(
                f"swap use grew by {growth} bytes since the run started (limit "
                f"{self.max_swap_growth_bytes}) at step {step}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "swap_start_bytes": self.swap_start_bytes,
            "swap_growth_bytes": self.swap_growth_bytes,
            "max_pressure_level": self.max_pressure_level,
            "checks": self.checks,
        }


def peak_rss_bytes() -> int:
    """Peak resident set size of this process (``ru_maxrss``: bytes on macOS, KiB on Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


# -- the multinomial router ----------------------------------------------------------------------


class MultinomialRouter:
    """``torch.multinomial`` routed to the CPU with the fold's own generator (design 3.1 row 2).

    While :meth:`installed`, ``torch.multinomial`` is this object (K:271 looks it up at call
    time): every draw is the original ``torch.multinomial`` on a CPU copy of the probabilities
    with the active generator, returned on the input's device. The training generator is seeded
    with ``seed``; :meth:`using` switches to another one for a block (the validation diagnostic).
    ``generator=`` and ``out=`` arguments are refused. Leaving :meth:`installed` restores the
    original, so nothing outside the training phase (inference fans run in other processes
    anyway) ever sees the router.
    """

    def __init__(self, seed: int) -> None:
        import torch

        if not _is_int(seed) or seed < 0:
            raise RunnerError(f"the router seed must be an integer >= 0, got {seed!r}")
        self.seed = seed
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self._active = self.generator
        original = torch.multinomial
        while isinstance(original, MultinomialRouter):  # built while another one is installed
            original = original._multinomial
        self._multinomial = original
        self._installed = False
        self.calls = 0

    @property
    def is_installed(self) -> bool:
        return self._installed

    @contextlib.contextmanager
    def installed(self) -> Iterator[MultinomialRouter]:
        import torch

        if (
            self._installed
            or isinstance(torch.multinomial, MultinomialRouter)
            or torch.multinomial is not self._multinomial
        ):
            raise RunnerError("torch.multinomial is already replaced; refusing to stack routers")
        torch.multinomial = self  # type: ignore[assignment]
        self._installed = True
        try:
            yield self
        finally:
            torch.multinomial = self._multinomial
            self._installed = False

    @contextlib.contextmanager
    def using(self, generator: Any) -> Iterator[None]:
        previous = self._active
        self._active = generator
        try:
            yield
        finally:
            self._active = previous

    def __call__(
        self,
        input: Any,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: Any = None,
        out: Any = None,
    ) -> Any:
        if generator is not None or out is not None:
            raise RunnerError(
                "the multinomial router draws from its own generator: generator= and out= are "
                "refused"
            )
        self.calls += 1
        draws = self._multinomial(
            input.detach().to("cpu"), num_samples, replacement, generator=self._active
        )
        return draws.to(input.device)


# -- RNG states ----------------------------------------------------------------------------------


def _rng_states() -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    states: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        states["mps"] = torch.mps.get_rng_state()
    return states


def _restore_rng_states(states: Mapping[str, Any]) -> None:
    import random

    import numpy as np
    import torch

    torch.set_rng_state(states["torch"])
    np.random.set_state(states["numpy"])
    random.setstate(states["python"])
    if "mps" in states:
        torch.mps.set_rng_state(states["mps"])


# -- the validation diagnostic -------------------------------------------------------------------


def _cross_attention(model: Any) -> Any:
    cross_attn = model.dep_layer.cross_attn
    dropout = getattr(cross_attn, "resid_dropout", None)
    if getattr(cross_attn, "attn_dropout_p", 0.0) != 0.0 or getattr(dropout, "p", 0.0) != 0.0:
        raise RunnerError(
            "the dependency layer's cross-attention has dropout: in train mode the validation "
            "diagnostic would apply it"
        )
    return cross_attn


class CleanValidation:
    """The vendored validation loader as a causal, reseeded diagnostic (design 3.1 rows 3-4).

    Iterating it (the vendored loop does, right after ``model.eval()``, FB:313-318) first saves
    every RNG state (torch CPU and MPS, numpy, python), BEFORE the inner iterator is created so
    the loader's base-seed draw is undone too; then puts ``model.dep_layer.cross_attn`` in train
    mode (causal, M:387; its dropouts are 0, M:447, K:221, checked) and draws the s1 samples
    of the pass from a generator seeded with ``val_seed`` (through the router when it is
    installed, from the reseeded global generators otherwise). Afterwards, also when the pass is
    abandoned, the cross-attention's mode and every RNG state are restored: the same weights give
    the same diagnostic loss, and the pass leaves training untouched.
    """

    def __init__(
        self, loader: Any, *, model: Any, val_seed: int, router: MultinomialRouter | None = None
    ) -> None:
        if not _is_int(val_seed) or val_seed < 0:
            raise RunnerError(f"val_seed must be an integer >= 0, got {val_seed!r}")
        self.loader = loader
        self.model = model
        self.val_seed = val_seed
        self.router = router
        self.passes = 0
        _cross_attention(model)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        import torch

        states = _rng_states()
        cross_attn = _cross_attention(self.model)
        was_training = cross_attn.training
        try:
            cross_attn.train(True)
            if self.router is not None and self.router.is_installed:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.val_seed)
                draws: contextlib.AbstractContextManager[Any] = self.router.using(generator)
            else:
                torch.manual_seed(self.val_seed)
                draws = contextlib.nullcontext()
            with draws:
                yield from self.loader
            self.passes += 1
        finally:
            cross_attn.train(was_training)
            _restore_rng_states(states)


# -- the train-loader gate and the run monitor ---------------------------------------------------


class TrainLoaderGate:
    """The train loader as the vendored loop sees it (always installed; design 3.3, 3.5).

    ``__len__`` forwards, so OneCycle is sized as before (FB:256). Each ``__iter__`` starts a
    pseudo-epoch (FB:281): first ``monitor.epoch_boundary()`` (the previous pseudo-epoch's record,
    NLL4-TF, selection, :class:`EarlyStop`), then the inner loader's batches, each one preceded by
    ``monitor.before_batch()`` (step timing, the memory guard, the deadline, :class:`TimingDone`)
    and followed by ``monitor.batch_given()``; ``monitor.after_last_batch()`` closes the last
    step. Nothing here draws a random number or changes a batch.
    """

    def __init__(self, loader: Any, monitor: Any) -> None:
        self.loader = loader
        self.monitor = monitor

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Any]:
        monitor = self.monitor
        monitor.epoch_boundary()
        iterator = iter(self.loader)
        while True:
            monitor.before_batch()
            try:
                batch = next(iterator)
            except StopIteration:
                monitor.after_last_batch()
                return
            monitor.batch_given()
            yield batch


def _snapshot(model: Any) -> dict[str, Any]:
    """A CPU copy of every tensor of ``model.state_dict()`` (cloned also when already on CPU)."""
    return {
        name: tensor.detach().to("cpu", copy=True) for name, tensor in model.state_dict().items()
    }


class _Monitor:
    """What one run measures and decides at steps and pseudo-epoch boundaries.

    ``evaluate(epoch)`` returns NLL4-TF of the current weights (``None``: no NLL4-TF, as in a
    timing run). Pseudo-epoch 0 is the shipped weights (a diagnostic, never selected); with
    ``early_stopping`` the best pseudo-epoch is the argmin over ``e >= 1`` with strict
    improvement (ties to the earlier one), its state is kept as a CPU copy, and
    :class:`EarlyStop` is raised once ``patience`` consecutive evaluations failed to beat it.
    A non-finite training loss or NLL4-TF value fails the run (:class:`RunnerError`).
    """

    def __init__(
        self,
        *,
        evaluate: Callable[[int], float] | None = None,
        early_stopping: bool = False,
        patience: int = 1,
        memory_guard: MemoryGuard | None = None,
        deadline: float | None = None,
        timing: Mapping[str, Any] | None = None,
        device_memory: Callable[[], int | None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if early_stopping and evaluate is None:
            raise RunnerError("early stopping needs an NLL4-TF evaluation")
        if not _is_int(patience) or patience < 1:
            raise RunnerError(f"patience must be a positive integer, got {patience!r}")
        self._evaluate = evaluate
        self.early_stopping = early_stopping
        self.patience = patience
        self.memory_guard = memory_guard
        self.deadline = deadline
        self.timing = dict(timing) if timing is not None else None
        self._device_memory = device_memory
        self._clock = clock
        self.model: Any = None
        self.optimizer: Any = None
        self.optimizer_steps = 0
        self.completed_steps = 0
        self.step_durations: list[float] = []
        self.timed_durations: list[float] = []
        self.timed_offsets: list[float] = []
        self._first_timed_start: float | None = None
        self._open_step: float | None = None
        self._request_at: float | None = None
        self._pending_train_losses: list[Any] = []
        self._epoch_train_losses: list[float] = []
        self._epoch_val_losses: list[Any] = []
        self._boundaries = 0
        self._epoch_started: float | None = None
        self.epochs: list[dict[str, Any]] = []
        self.nll_seconds: list[float] = []
        self.best_epoch: int | None = None
        self.best_value: float | None = None
        self.best_state: dict[str, Any] | None = None
        self.stopped_at: int | None = None
        self.finite = True
        self.peak_device_bytes: int | None = None
        self.onecycle_total_steps: int | None = None

    # instrumentation hooks ---------------------------------------------------------------------

    def record_loss(self, loss: Any) -> None:
        """Called with every loss ``model.head.compute_loss`` returns (no value changes)."""
        import torch

        if torch.is_grad_enabled():
            self._pending_train_losses.append(loss.detach())
        else:
            self._epoch_val_losses.append(loss.detach())

    def optimizer_stepped(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        """Global optimizer step post-hook (``torch.optim.optimizer``)."""
        self.optimizer = optimizer
        self.optimizer_steps += 1

    # deadline and timing -----------------------------------------------------------------------

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeAbort("the run passed its deadline_seconds")

    def _timing_complete(self, now: float) -> bool:
        if self.timing is None:
            return False
        timed = len(self.timed_durations)
        if self.timing.get("timed_steps") is not None:
            return timed >= self.timing["timed_steps"]
        if timed == 0 or self._first_timed_start is None:
            return False
        return now - self._first_timed_start >= self.timing["timed_seconds"]

    # steps -------------------------------------------------------------------------------------

    def _settle_train_losses(self) -> None:
        for loss in self._pending_train_losses:
            value = float(loss)
            self._epoch_train_losses.append(value)
            if not math.isfinite(value):
                self.finite = False
                raise RunnerError(f"non-finite training loss at step {self.completed_steps}")
        self._pending_train_losses.clear()

    def _close_step(self, end: float) -> None:
        start = self._open_step
        self._open_step = None
        if start is None:
            return
        index = self.completed_steps
        self.completed_steps += 1
        duration = end - start
        self.step_durations.append(duration)
        if self.timing is not None and index >= self.timing["warmup_steps"]:
            if self._first_timed_start is None:
                self._first_timed_start = start
            self.timed_durations.append(duration)
            self.timed_offsets.append(start - self._first_timed_start)
        self._settle_train_losses()

    def sample_device_memory(self) -> None:
        if self._device_memory is None:
            return
        value = self._device_memory()
        if value is not None:
            self.peak_device_bytes = max(value, self.peak_device_bytes or 0)

    def before_batch(self) -> None:
        now = self._clock()
        self._close_step(now)
        self._request_at = now
        self.sample_device_memory()
        if self.memory_guard is not None:
            self.memory_guard.check(self.completed_steps)
        self.check_deadline()
        if self._timing_complete(now):
            raise TimingDone(f"{len(self.timed_durations)} timed steps done")

    def batch_given(self) -> None:
        self._open_step = self._request_at

    def after_last_batch(self) -> None:
        self._close_step(self._clock())

    # pseudo-epochs -----------------------------------------------------------------------------

    def _current_lr(self) -> float | None:
        if self.optimizer is None:
            return None
        return float(self.optimizer.param_groups[0]["lr"])

    def _evaluate_nll(self, epoch: int) -> float | None:
        if self._evaluate is None:
            return None
        started = self._clock()
        value = float(self._evaluate(epoch))
        self.nll_seconds.append(self._clock() - started)
        self.check_deadline()
        return value

    def _select(self, epoch: int, value: float, *, may_stop: bool) -> None:
        if not self.early_stopping:
            return
        if self.best_value is None or value < self.best_value:
            self.best_epoch, self.best_value = epoch, value
            self.best_state = _snapshot(self.model)
            return
        assert self.best_epoch is not None
        if may_stop and epoch - self.best_epoch >= self.patience:
            self.stopped_at = epoch
            raise EarlyStop(
                f"NLL4-TF did not improve for {self.patience} pseudo-epochs (stopped after "
                f"pseudo-epoch {epoch})"
            )

    def _close_epoch(self, epoch: int, *, may_stop: bool) -> None:
        self._settle_train_losses()
        now = self._clock()
        train = list(self._epoch_train_losses)
        validation = [float(loss) for loss in self._epoch_val_losses]
        self._epoch_train_losses.clear()
        self._epoch_val_losses.clear()
        train_loss = sum(train) / len(train) if train else None
        val_loss = sum(validation) / len(validation) if validation else None
        entry: dict[str, Any] = {
            "pseudo_epoch": epoch,
            "nll4_tf": None,
            "train_loss": train_loss,
            "vendored_val_loss": val_loss,
            "lr_end": self._current_lr(),
            "seconds": now - (self._epoch_started if self._epoch_started is not None else now),
        }
        self.epochs.append(entry)
        if train_loss is not None and not math.isfinite(train_loss):
            self.finite = False
            raise RunnerError(f"non-finite training loss in pseudo-epoch {epoch}")
        if val_loss is not None and not math.isfinite(val_loss):
            self.finite = False
        value = self._evaluate_nll(epoch)
        entry["nll4_tf"] = value
        if value is None:
            return
        if not math.isfinite(value):
            self.finite = False
            raise RunnerError(f"non-finite NLL4-TF after pseudo-epoch {epoch}")
        self._select(epoch, value, may_stop=may_stop)

    def epoch_boundary(self) -> None:
        """Start of a pseudo-epoch (the gate's ``__iter__``): close the previous one."""
        self.check_deadline()
        completed = self._boundaries
        self._boundaries += 1
        if completed == 0:
            if self._evaluate is not None:
                started = self._clock()
                value = self._evaluate_nll(0)
                self.epochs.append(
                    {
                        "pseudo_epoch": 0,
                        "nll4_tf": value,
                        "train_loss": None,
                        "vendored_val_loss": None,
                        "lr_end": None,
                        "seconds": self._clock() - started,
                    }
                )
                if value is not None and not math.isfinite(value):
                    self.finite = False
                    raise RunnerError("non-finite NLL4-TF of the shipped weights")
        else:
            self._close_epoch(completed, may_stop=True)
        self._epoch_started = self._clock()

    def finish(self) -> None:
        """After a normal return of the vendored loop: close the last pseudo-epoch."""
        if self._boundaries == 0:
            raise RunnerError("the training loop never started a pseudo-epoch")
        self._close_epoch(self._boundaries, may_stop=False)

    @property
    def pseudo_epochs_started(self) -> int:
        return self._boundaries

    def vendored_best_epoch(self) -> int | None:
        """The pseudo-epoch whose vendored keep-best was saved last (FB:354-362: strict, from
        infinity; a NaN or missing loss never saves)."""
        best = math.inf
        chosen: int | None = None
        for entry in self.epochs:
            value = entry["vendored_val_loss"]
            if entry["pseudo_epoch"] >= 1 and value is not None and value < best:
                best, chosen = value, entry["pseudo_epoch"]
        return chosen

    def step_summary(self) -> dict[str, Any]:
        import numpy as np

        values = self.timed_durations if self.timing is not None else self.step_durations
        if not values:
            return {"count": 0, "median": None, "p10": None, "p90": None}
        p10, median, p90 = np.percentile(np.asarray(values, dtype=np.float64), [10, 50, 90])
        return {"count": len(values), "median": float(median), "p10": float(p10), "p90": float(p90)}


# -- instance-level wrappers ---------------------------------------------------------------------


class _ReplayMpsGenerator:
    """The recompute context of one checkpointed call on MPS: the recomputation draws from the
    MPS generator state its forward started from, and the generator is put back afterwards."""

    def __init__(self, state: Any) -> None:
        self.state = state
        self.saved: list[Any] = []

    def __enter__(self) -> None:
        import torch

        self.saved.append(torch.mps.get_rng_state())
        torch.mps.set_rng_state(self.state)

    def __exit__(self, *exc_info: object) -> bool:
        import torch

        torch.mps.set_rng_state(self.saved.pop())
        return False


def _mps_generator_contexts() -> tuple[Any, Any]:
    """``context_fn`` of a checkpointed call on MPS, called before its forward: no forward
    context, and a recompute context that replays the MPS generator from this point."""
    import torch

    return contextlib.nullcontext(), _ReplayMpsGenerator(torch.mps.get_rng_state())


def enable_activation_checkpointing(model: Any) -> None:
    """Wrap each ``TransformerBlock.forward`` of ``model.transformer`` with
    ``torch.utils.checkpoint`` at instance level (non-reentrant, RNG state preserved): CPU
    gradients stay bit-identical (test T17).

    ``preserve_rng_state`` restores only the CPU generator for the recomputation on MPS (torch
    2.14.1: ``torch.mps`` has no ``_initialized``), so the recomputed dropout masks would differ
    from the forward's and so would the gradients. On MPS the call therefore also passes
    :func:`_mps_generator_contexts`: the recomputation replays the forward's MPS generator state
    and leaves the generator where an unwrapped step leaves it. A CPU call is unchanged.
    """
    from torch.utils.checkpoint import checkpoint

    for block in model.transformer:
        if "forward" in vars(block):
            raise RunnerError("activation checkpointing is already enabled on this model")
        original = block.forward

        def forward(x: Any, key_padding_mask: Any = None, _original: Any = original) -> Any:
            if x.device.type == "mps":
                return checkpoint(
                    _original,
                    x,
                    key_padding_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    context_fn=_mps_generator_contexts,
                )
            return checkpoint(
                _original, x, key_padding_mask, use_reentrant=False, preserve_rng_state=True
            )

        block.forward = forward


def _install_loss_recorder(model: Any, monitor: _Monitor) -> Callable[[], None]:
    """Record every loss through an instance attribute on ``model.head`` (FB:292 and FB:327 call
    ``model.head.compute_loss``); returns what removes it."""
    head = model.head
    original = head.compute_loss

    def compute_loss(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        monitor.record_loss(result[0])
        return result

    head.compute_loss = compute_loss

    def remove() -> None:
        if vars(head).get("compute_loss") is compute_loss:
            del head.compute_loss

    return remove


def _same_bits(first: Any, second: Any) -> bool:
    import torch

    left = first.detach().to("cpu").contiguous().reshape(-1)
    right = second.detach().to("cpu").contiguous().reshape(-1)
    if left.dtype != right.dtype or left.numel() != right.numel():
        return False
    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def state_mismatches(module: Any, safetensors_path: Path) -> list[str]:
    """Why ``module.state_dict()`` is not, bit for bit, the tensors of ``safetensors_path``.

    Names missing on either side, shapes or dtypes that differ and tensors whose bytes differ
    (the vendored loader uses ``strict=False`` and checks nothing, HM:785). Empty: identical.
    """
    from safetensors.torch import load_file

    expected = load_file(str(safetensors_path), device="cpu")
    actual = module.state_dict()
    problems: list[str] = []
    for name in sorted(set(expected) - set(actual)):
        problems.append(f"{name}: in the file, not in the module")
    for name in sorted(set(actual) - set(expected)):
        problems.append(f"{name}: in the module, not in the file")
    for name in sorted(set(expected) & set(actual)):
        mine, theirs = actual[name], expected[name]
        if mine.dtype != theirs.dtype or tuple(mine.shape) != tuple(theirs.shape):
            problems.append(
                f"{name}: {mine.dtype} {tuple(mine.shape)} != {theirs.dtype} {tuple(theirs.shape)}"
            )
        elif not _same_bits(mine, theirs):
            problems.append(f"{name}: values differ")
    return problems


# -- run.json --------------------------------------------------------------------------------------


def _object(value: object, keys: Sequence[str], what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunnerError(f"{what} must be an object")
    unknown = sorted(str(key) for key in set(value) - set(keys))
    missing = sorted(set(keys) - set(value))
    if unknown or missing:
        raise RunnerError(f"{what}: unknown keys {unknown}, missing keys {missing}")
    return value


def _int_field(mapping: Mapping[str, Any], key: str, what: str, *, minimum: int = 0) -> int:
    value = mapping[key]
    if not _is_int(value) or value < minimum:
        raise RunnerError(f"{what}.{key} must be an integer >= {minimum}, got {value!r}")
    return value


def _hex_field(mapping: Mapping[str, Any], key: str, what: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise RunnerError(f"{what}.{key} must be a sha256 hex")
    return value


def _path_field(mapping: Mapping[str, Any], key: str, what: str) -> Path:
    value = mapping[key]
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise RunnerError(f"{what}.{key} must be an absolute path")
    return Path(value)


@dataclass(frozen=True)
class _ModelFileSpec:
    directory: Path
    sha256: Mapping[str, str]


@dataclass(frozen=True)
class _RunSpec:
    """``run.json``, validated (interfaces 5.5)."""

    run_dir: Path
    run_json_sha256: str
    run_id: str
    mode: str
    recipe: Any
    recipe_id: str
    fold: Mapping[str, Any]
    seeds: Mapping[str, Any]
    yaml_sha256: str
    train_csv: Mapping[str, Any]
    train_csv_path: Path
    nll_inputs: Mapping[str, Any] | None
    nll_inputs_path: Path | None
    export_content_sha256: str
    windows: Mapping[str, Any]
    predictor: _ModelFileSpec
    tokenizer: _ModelFileSpec
    compute: Mapping[str, Any]
    wrappers: Mapping[str, bool]
    nll_batch_size: int
    memory: Mapping[str, Any]
    timing: Mapping[str, Any] | None
    preflight: Mapping[str, Any] | None
    deadline_seconds: float | None
    blinded: bool


def _model_file_spec(value: object, what: str) -> _ModelFileSpec:
    mapping = _object(value, _MODEL_FILE_KEYS, what)
    hashes = _object(mapping["sha256"], CHECKPOINT_FILE_NAMES, f"{what}.sha256")
    for name in CHECKPOINT_FILE_NAMES:
        _hex_field(hashes, name, f"{what}.sha256")
    return _ModelFileSpec(directory=_path_field(mapping, "dir", what), sha256=dict(hashes))


def _compute_spec(value: object) -> dict[str, Any]:
    compute = _object(value, _COMPUTE_KEYS, "compute")
    if compute["device"] not in DEVICES:
        raise RunnerError(f"compute.device must be one of {list(DEVICES)}")
    _int_field(compute, "threads", "compute", minimum=1)
    for key in ("deterministic", "activation_checkpointing", "memory_guard"):
        if not isinstance(compute[key], bool):
            raise RunnerError(f"compute.{key} must be a boolean")
    ratio = compute["mps_watermark_ratio"]
    if ratio is not None and (not _is_number(ratio) or not 0 < ratio <= 1.7):
        raise RunnerError("compute.mps_watermark_ratio must be null or in (0, 1.7]")
    return compute


def _timing_spec(value: object) -> dict[str, Any]:
    timing = _object(value, _TIMING_KEYS, "timing")
    _int_field(timing, "warmup_steps", "timing")
    steps, seconds = timing["timed_steps"], timing["timed_seconds"]
    if (steps is None) == (seconds is None):
        raise RunnerError("timing needs exactly one of timed_steps and timed_seconds")
    if steps is not None:
        _int_field(timing, "timed_steps", "timing", minimum=1)
    elif not _is_number(seconds) or not seconds > 0:
        raise RunnerError("timing.timed_seconds must be a positive number")
    return timing


def _preflight_spec(value: object) -> dict[str, Any]:
    preflight = _object(value, _PREFLIGHT_KEYS, "preflight")
    checks = preflight["checks"]
    if (
        not isinstance(checks, list)
        or not checks
        or len(set(checks)) != len(checks)
        or not set(checks) <= set(PREFLIGHT_CHECKS)
    ):
        raise RunnerError(f"preflight.checks must be distinct names of {list(PREFLIGHT_CHECKS)}")
    for key in ("multinomial_rows", "multinomial_repetitions", "steps"):
        _int_field(preflight, key, "preflight", minimum=1)
    return preflight


def _read_run_spec(run_dir: Path) -> _RunSpec:
    """Parse and validate ``run.json`` (unknown or missing keys, a wrong schema: RunnerError)."""
    from .recipe import RecipeError, recipe_from_mapping, recipe_id, training_seeds

    path = run_dir / RUN_JSON
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RunnerError(f"cannot read {RUN_JSON}: {exc.strerror}") from None
    try:
        document = json.loads(data)
    except ValueError:
        raise RunnerError(f"{RUN_JSON} is not JSON") from None
    document = _object(document, _RUN_KEYS, RUN_JSON)
    if document["schema_version"] != RUN_SCHEMA_VERSION or not _is_int(document["schema_version"]):
        raise RunnerError(f"{RUN_JSON} schema_version must be {RUN_SCHEMA_VERSION}")
    run_id = document["run_id"]
    if run_id != run_dir.name:
        raise RunnerError(f"run_id {run_id!r} is not the run directory's name")
    mode = document["mode"]
    if mode not in MODES:
        raise RunnerError(f"mode must be one of {list(MODES)}, got {mode!r}")
    try:
        recipe = recipe_from_mapping(document["recipe"])
        if recipe_id(recipe) != document["recipe_id"]:
            raise RunnerError("recipe_id is not the id of the recipe")
    except RecipeError as exc:
        raise RunnerError(f"invalid recipe: {exc}") from None
    fold = _object(document["fold"], _FOLD_KEYS, "fold")
    for key in ("csv_start_ms", "csv_end_ms", "inner_validation_start_ms", "replicate"):
        _int_field(fold, key, "fold")
    suffix = f"-r{fold['replicate']}" if fold["replicate"] else ""
    if not isinstance(fold["label"], str) or fold["run_label"] != f"{fold['label']}{suffix}":
        raise RunnerError("fold.run_label must be the fold label plus its replicate suffix")
    if not (fold["csv_start_ms"] < fold["inner_validation_start_ms"] < fold["csv_end_ms"]):
        raise RunnerError("fold ranges must satisfy csv_start < inner_validation_start < csv_end")
    seeds = _object(document["seeds"], _SEED_KEYS, "seeds")
    try:
        expected_seeds = training_seeds(document["recipe_id"], fold["label"], fold["replicate"])
    except RecipeError as exc:
        raise RunnerError(f"invalid seeds request: {exc}") from None
    if seeds != vars(expected_seeds):
        raise RunnerError("seeds are not the training seeds of this recipe, fold and replicate")
    yaml_spec = _object(document["yaml"], _YAML_KEYS, "yaml")
    if yaml_spec["name"] != RUN_YAML:
        raise RunnerError(f"yaml.name must be {RUN_YAML!r}")
    train_csv = _object(document["train_csv"], _TRAIN_CSV_KEYS, "train_csv")
    for key in ("rows", "n_fit", "n_val"):
        _int_field(train_csv, key, "train_csv", minimum=1)
    if train_csv["n_fit"] + train_csv["n_val"] != train_csv["rows"]:
        raise RunnerError("train_csv.n_fit + n_val must equal rows")
    nll_inputs = document["nll_inputs"]
    nll_path: Path | None = None
    if mode == "train":
        nll_inputs = _object(nll_inputs, _NLL_INPUT_KEYS, "nll_inputs")
        _int_field(nll_inputs, "decisions", "nll_inputs", minimum=1)
        _hex_field(nll_inputs, "sha256", "nll_inputs")
        nll_path = _path_field(nll_inputs, "path", "nll_inputs")
    elif nll_inputs is not None:
        raise RunnerError("nll_inputs must be null outside mode train")
    windows = _object(document["windows"], _WINDOW_KEYS, "windows")
    if not isinstance(windows["gap_aware"], bool):
        raise RunnerError("windows.gap_aware must be a boolean")
    for key in ("fit_windows_kept", "val_windows_kept"):
        _int_field(windows, key, "windows", minimum=1)
    model_files = _object(document["model_files"], ("predictor", "tokenizer"), "model_files")
    wrappers = _object(document["wrappers"], WRAPPERS, "wrappers")
    if not all(isinstance(value, bool) for value in wrappers.values()):
        raise RunnerError("every wrapper switch must be a boolean")
    compute = _compute_spec(document["compute"])
    if compute["device"] != "cpu" and not wrappers["device"]:
        raise RunnerError("a run on another device than the CPU needs the device wrapper")
    if compute["mps_watermark_ratio"] is not None and compute["device"] != "mps":
        raise RunnerError("compute.mps_watermark_ratio applies to MPS runs only")
    nll_batch_size = document["nll_batch_size"]
    if not _is_int(nll_batch_size) or nll_batch_size < 1:
        raise RunnerError("nll_batch_size must be a positive integer")
    memory = _object(document["memory"], _MEMORY_KEYS, "memory")
    _int_field(memory, "every_steps", "memory", minimum=1)
    _int_field(memory, "max_swap_growth_bytes", "memory")
    _int_field(memory, "critical_pressure_level", "memory", minimum=1)
    timing = None
    if mode == "timing":
        timing = _timing_spec(document["timing"])
        if wrappers["early_stopping"]:
            raise RunnerError("a timing run measures steps only: early_stopping must be off")
    elif document["timing"] is not None:
        raise RunnerError("timing must be null outside mode timing")
    preflight = None
    if mode == "preflight":
        preflight = _preflight_spec(document["preflight"])
    elif document["preflight"] is not None:
        raise RunnerError("preflight must be null outside mode preflight")
    if mode == "train" and wrappers["early_stopping"] and recipe.early_stopping != "nll4_tf":
        raise RunnerError(f"unknown early-stopping metric {recipe.early_stopping!r}")
    if recipe.normalisation != "vendored" and not wrappers["gap_aware"]:
        raise RunnerError("a normalisation other than the vendored one needs the gap_aware wrapper")
    if windows["gap_aware"] and not wrappers["gap_aware"]:
        raise RunnerError(
            "the export has a gap: a run on it needs the gap_aware wrapper (design 1.6: no "
            "window across a missing hour)"
        )
    deadline = document["deadline_seconds"]
    if deadline is not None and (not _is_number(deadline) or not deadline > 0):
        raise RunnerError("deadline_seconds must be null or a positive number")
    if not isinstance(document["blinded"], bool):
        raise RunnerError("blinded must be a boolean")
    return _RunSpec(
        run_dir=run_dir,
        run_json_sha256=sha256_bytes(data),
        run_id=run_id,
        mode=mode,
        recipe=recipe,
        recipe_id=document["recipe_id"],
        fold=fold,
        seeds=seeds,
        yaml_sha256=_hex_field(yaml_spec, "sha256", "yaml"),
        train_csv=train_csv,
        train_csv_path=_path_field(train_csv, "path", "train_csv"),
        nll_inputs=nll_inputs,
        nll_inputs_path=nll_path,
        export_content_sha256=_hex_field(document, "export_content_sha256", RUN_JSON),
        windows=windows,
        predictor=_model_file_spec(model_files["predictor"], "model_files.predictor"),
        tokenizer=_model_file_spec(model_files["tokenizer"], "model_files.tokenizer"),
        compute=compute,
        wrappers={name: bool(wrappers[name]) for name in WRAPPERS},
        nll_batch_size=nll_batch_size,
        memory=memory,
        timing=timing,
        preflight=preflight,
        deadline_seconds=None if deadline is None else float(deadline),
        blinded=document["blinded"],
    )


def _verified_sha256(path: Path, expected: str, label: str) -> str:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise RunnerError(f"cannot read {label}: {exc.strerror}") from None
    if actual != expected:
        raise RunnerError(f"sha256 mismatch: {label}")
    return actual


def _verify_inputs(spec: _RunSpec) -> dict[str, Any]:
    """Hash every input against ``run.json`` and check that ``run.yaml`` is the document of this
    run (``recipe.run_yaml_document``); returns the ``inputs`` record."""
    import yaml

    from .recipe import YamlPaths, run_yaml_document

    yaml_path = spec.run_dir / RUN_YAML
    _verified_sha256(yaml_path, spec.yaml_sha256, RUN_YAML)
    expected = run_yaml_document(
        spec.recipe,
        paths=YamlPaths(
            data_path=spec.train_csv_path,
            predictor_dir=spec.predictor.directory,
            tokenizer_dir=spec.tokenizer.directory,
            save_dir=spec.run_dir / VENDORED_SAVE_DIRNAME,
            base_path=spec.run_dir,
        ),
        n_rows=spec.train_csv["rows"],
        n_fit=spec.train_csv["n_fit"],
        seed=spec.seeds["yaml_seed"],
        exp_name=spec.run_id,
        description=f"{spec.recipe.name} {spec.fold['run_label']} {spec.mode}",
    )
    if yaml.safe_load(yaml_path.read_bytes()) != expected:
        raise RunnerError(f"{RUN_YAML} is not the run YAML of {RUN_JSON}")
    _verified_sha256(spec.train_csv_path, spec.train_csv["sha256"], "the fold CSV")
    nll_sha256 = None
    if spec.nll_inputs_path is not None and spec.nll_inputs is not None:
        nll_sha256 = _verified_sha256(
            spec.nll_inputs_path, spec.nll_inputs["sha256"], "the NLL4-TF inputs"
        )
    model_files: dict[str, dict[str, str]] = {}
    for role, files in (("predictor", spec.predictor), ("tokenizer", spec.tokenizer)):
        model_files[role] = {
            name: _verified_sha256(files.directory / name, files.sha256[name], f"{role} {name}")
            for name in CHECKPOINT_FILE_NAMES
        }
    return {
        "run_json_sha256": spec.run_json_sha256,
        "run_yaml_sha256": spec.yaml_sha256,
        "train_csv_sha256": spec.train_csv["sha256"],
        "nll_inputs_sha256": nll_sha256,
        "model_files": model_files,
    }


# -- the session: wrappers around the vendored predictor phase -----------------------------------


def _trainer_class(vendored: VendoredFinetune, spec: _RunSpec) -> type:
    """The vendored ``SequentialTrainer``, or (device wrapper) a subclass overriding only
    ``_setup_device`` (design 3.1 row 1)."""
    base = vendored.SequentialTrainer
    if not spec.wrappers["device"]:
        return base
    device_name = spec.compute["device"]

    class DeviceTrainer(base):  # type: ignore[valid-type, misc]
        def _setup_device(self) -> Any:
            import torch

            device = torch.device(device_name)
            if self.rank == 0:
                print(f"Using device: {device} (device wrapper; rank={self.rank})")
            return device

    return DeviceTrainer


class _Session:
    """The wrappers of one run around ``train_model`` and ``create_dataloaders``."""

    def __init__(
        self,
        spec: _RunSpec,
        vendored: VendoredFinetune,
        monitor: _Monitor,
        *,
        router: MultinomialRouter | None = None,
    ) -> None:
        self.spec = spec
        self.vendored = vendored
        self.monitor = monitor
        self.router = router
        self.model: Any = None
        self.dataset: dict[str, Any] | None = None
        self._train_model = vendored.finetune_base_model.train_model
        self._create_dataloaders = vendored.finetune_base_model.create_dataloaders

    @contextlib.contextmanager
    def installed(self) -> Iterator[_Session]:
        """Patch ``train_sequential.train_model`` and ``finetune_base_model.create_dataloaders``
        (module globals the vendored code looks up at call time); restored on exit."""
        sequential = self.vendored.train_sequential
        base_model = self.vendored.finetune_base_model
        if (
            sequential.train_model is not self._train_model
            or base_model.create_dataloaders is not self._create_dataloaders
        ):
            raise RunnerError("the vendored training functions are already replaced")
        sequential.train_model = self.train_model
        base_model.create_dataloaders = self.create_dataloaders
        try:
            yield self
        finally:
            sequential.train_model = self._train_model
            base_model.create_dataloaders = self._create_dataloaders

    def train_model(
        self, model: Any, tokenizer: Any, device: Any, config: Any, save_dir: str, logger: Any
    ) -> Any:
        """In place of ``train_sequential.train_model`` (TS:247): the weight check, activation
        checkpointing, the router and the instrumentation, then the vendored ``train_model``."""
        from torch.optim.optimizer import register_optimizer_step_post_hook

        spec = self.spec
        self.model = model
        self.monitor.model = model
        if spec.wrappers["weight_check"]:
            problems = state_mismatches(
                model, spec.predictor.directory / "model.safetensors"
            ) + state_mismatches(tokenizer, spec.tokenizer.directory / "model.safetensors")
            if problems:
                shown = "; ".join(problems[:8])
                raise RunnerError(f"the loaded weights differ from the verified files: {shown}")
        if spec.compute["activation_checkpointing"]:
            enable_activation_checkpointing(model)
        remove_recorder = _install_loss_recorder(model, self.monitor)
        hook = register_optimizer_step_post_hook(self.monitor.optimizer_stepped)
        routing = self.router.installed() if self.router is not None else contextlib.nullcontext()
        try:
            with routing:
                return self._train_model(model, tokenizer, device, config, save_dir, logger)
        finally:
            hook.remove()
            remove_recorder()

    def _datasets(self, config: Any) -> tuple[Any, Any, Any, Any]:
        spec = self.spec
        use_project = spec.wrappers["gap_aware"] and (
            spec.windows["gap_aware"] or spec.recipe.normalisation != "vendored"
        )
        if not use_project:
            loaders = self._create_dataloaders(config)
            train_loader, val_loader, train_ds, val_ds, train_sampler, val_sampler = loaders
            if train_sampler is not None or val_sampler is not None:
                raise RunnerError("distributed samplers are not supported")
            return train_loader, val_loader, train_ds, val_ds
        from torch.utils.data import DataLoader

        dataset_class = make_dataset_class(
            self.vendored.CustomKlineDataset, normalisation=spec.recipe.normalisation
        )
        arguments = {
            "data_path": config.data_path,
            "lookback_window": config.lookback_window,
            "predict_window": config.predict_window,
            "clip": config.clip,
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
        }
        # FB:185-231 with the project dataset in place of the vendored one
        train_ds = dataset_class(data_type="train", seed=config.seed, **arguments)
        val_ds = dataset_class(data_type="val", seed=config.seed + 1, **arguments)
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, val_loader, train_ds, val_ds

    def create_dataloaders(self, config: Any) -> tuple[Any, ...]:
        """In place of ``finetune_base_model.create_dataloaders`` (FB:245). The datasets' window
        counts must be the export's, whichever dataset runs (without a gap the vendored one
        counts every window, as the exporter does)."""
        spec = self.spec
        recipe = spec.recipe
        train_loader, val_loader, train_ds, val_ds = self._datasets(config)
        if (
            len(train_ds.data) != spec.train_csv["n_fit"]
            or len(val_ds.data) != (spec.train_csv["n_val"])
        ):
            raise RunnerError(
                f"the vendored split gave {len(train_ds.data)} + {len(val_ds.data)} rows, the "
                f"export {spec.train_csv['n_fit']} + {spec.train_csv['n_val']}"
            )
        if (
            len(train_ds) != spec.windows["fit_windows_kept"]
            or len(val_ds) != spec.windows["val_windows_kept"]
        ):
            raise RunnerError(
                f"the datasets hold {len(train_ds)} fit and {len(val_ds)} validation windows, "
                f"the export {spec.windows['fit_windows_kept']} and "
                f"{spec.windows['val_windows_kept']}"
            )
        self.dataset = {
            "class": type(train_ds).__name__,
            "fit_windows": len(train_ds),
            "val_windows": len(val_ds),
        }
        if spec.wrappers["pseudo_epochs"] and recipe.budget == "pseudo_epochs":
            draws = recipe.pseudo_epoch_steps * config.batch_size
            if spec.mode == "train" and draws > len(train_ds):
                raise RunnerError(
                    f"a pseudo-epoch draws {draws} windows without replacement, the fold has "
                    f"{len(train_ds)}"
                )
            train_loader = pseudo_epoch_loader(
                train_ds,
                batch_size=config.batch_size,
                steps=recipe.pseudo_epoch_steps,
                seed=spec.seeds["sampler_seed"],
            )
        gate = TrainLoaderGate(train_loader, self.monitor)
        total = len(gate) * config.basemodel_epochs
        self.monitor.onecycle_total_steps = total  # what OneCycleLR is given (FB:253-260)
        if (
            spec.wrappers["pseudo_epochs"]
            and recipe.onecycle_total_steps is not None
            and total != recipe.onecycle_total_steps
        ):
            raise RunnerError(
                f"OneCycle would span {total} steps, the recipe {recipe.onecycle_total_steps}"
            )
        if spec.wrappers["clean_validation"]:
            val_loader = CleanValidation(
                val_loader,
                model=self.model,
                val_seed=spec.seeds["val_seed"],
                router=self.router,
            )
        return gate, val_loader, train_ds, val_ds, None, None


# -- models ----------------------------------------------------------------------------------------


def _eval_models(api: Any, spec: _RunSpec) -> tuple[Any, Any]:
    """CPU copies of the run's tokenizer and predictor in eval mode, without gradients."""
    tokenizer = api.KronosTokenizer.from_pretrained(str(spec.tokenizer.directory), strict=True)
    model = api.Kronos.from_pretrained(str(spec.predictor.directory), strict=True)
    for module in (tokenizer, model):
        module.eval()
        module.requires_grad_(False)
        module.to("cpu")
    return tokenizer, model


def _predictor_config(spec: _RunSpec) -> dict[str, Any]:
    return json.loads((spec.predictor.directory / "config.json").read_text(encoding="utf-8"))


def _file_records(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "size": (directory / name).stat().st_size,
            "sha256": sha256_file(directory / name),
        }
        for name in CHECKPOINT_FILE_NAMES
        if (directory / name).is_file()
    ]


def _save_selected(api: Any, spec: _RunSpec, state: Mapping[str, Any]) -> dict[str, Any]:
    """``save_pretrained`` of the selected state into ``selected/`` from a fresh CPU ``Kronos``
    built from the predictor's ``config.json``."""
    model = api.Kronos(**_predictor_config(spec))
    model.load_state_dict(dict(state), strict=True)
    model.eval()
    directory = spec.run_dir / SELECTED_DIRNAME
    if directory.exists():
        raise RunnerError(f"{SELECTED_DIRNAME}/ exists already in the run directory")
    model.save_pretrained(str(directory))
    return {"dir": SELECTED_DIRNAME, "source": "nll4_tf", "files": _file_records(directory)}


# -- preflight (design (a), M2-M3) --------------------------------------------------------------


def _zero_dropout(model: Any) -> None:
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attn_dropout_p"):
            module.attn_dropout_p = 0.0


def _fresh_model(api: Any, spec: _RunSpec, state: Mapping[str, Any], device: Any) -> Any:
    model = api.Kronos(**_predictor_config(spec))
    model.load_state_dict(dict(state), strict=True)
    return model.to(device)


def _training_model(api: Any, spec: _RunSpec, state: Mapping[str, Any], device: Any) -> Any:
    """A fresh model for a check that trains (parity, steps, deterministic), with activation
    checkpointing when the run asks for it, on the device and on the CPU alike: on the CPU the
    wrap changes no value (test T17, M2), and it keeps the CPU side of an MPS preflight from
    holding the unwrapped batch-32 activations next to the MPS allocations."""
    model = _fresh_model(api, spec, state, device)
    if spec.compute["activation_checkpointing"]:
        enable_activation_checkpointing(model)
    return model


def _cpu_float64(tensor: Any) -> Any:
    """``tensor`` as float64 on the CPU, moved first and converted after: one ``.to("cpu",
    torch.float64)`` of an MPS tensor returns NaN (torch 2.14.1; MPS has no float64)."""
    import torch

    return tensor.detach().to("cpu").to(torch.float64)


def _chi_square(counts: Any, expected: Any) -> dict[str, Any]:
    """Pearson's statistic over categories, cells with fewer than 5 expected draws pooled."""
    import numpy as np
    import torch

    observed = np.asarray(counts, dtype=np.float64)
    wanted = np.asarray(expected, dtype=np.float64)
    small = wanted < PREFLIGHT_MIN_EXPECTED
    cells_observed = list(observed[~small])
    cells_expected = list(wanted[~small])
    if small.any():
        cells_observed.append(observed[small].sum())
        cells_expected.append(wanted[small].sum())
    observed_cells = np.asarray(cells_observed)
    expected_cells = np.asarray(cells_expected)
    keep = expected_cells > 0
    statistic = float((((observed_cells - expected_cells) ** 2)[keep] / expected_cells[keep]).sum())
    dof = int(keep.sum()) - 1
    p_value = None
    if dof >= 1:
        p_value = float(
            torch.special.gammaincc(
                torch.tensor(dof / 2.0, dtype=torch.float64),
                torch.tensor(statistic / 2.0, dtype=torch.float64),
            )
        )
    return {"statistic": statistic, "dof": dof, "p_value": p_value}


def _preflight_multinomial(
    probabilities: Any, *, device: Any, repetitions: int, seed: int
) -> dict[str, Any]:
    """Device draws against the probabilities they come from (pooled chi-square), with a CPU
    control on a seeded generator; draws of zero-probability categories are counted."""
    import torch

    probs_cpu = _cpu_float64(probabilities)
    expected = (probs_cpu.sum(dim=0) * repetitions).numpy()
    rows, categories = probs_cpu.shape
    outcome: dict[str, Any] = {"rows": rows, "categories": categories, "repetitions": repetitions}
    for label, target in (("device", device), ("cpu", torch.device("cpu"))):
        generator = None
        if target.type == "cpu":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
        source = probabilities.detach().to(target)
        counts = torch.zeros(categories, dtype=torch.float64)
        zero_draws = 0
        for _ in range(repetitions):
            if generator is None:
                draws = torch.multinomial(source, 1)
            else:
                draws = torch.multinomial(source, 1, generator=generator)
            draws = draws.to("cpu").reshape(-1)
            counts += torch.bincount(draws, minlength=categories).to(torch.float64)
            zero_draws += int((probs_cpu[torch.arange(rows), draws] == 0).sum())
        test = _chi_square(counts.numpy(), expected)
        outcome[label] = {**test, "zero_probability_draws": zero_draws}
    device_test = outcome["device"]
    outcome["passed"] = bool(
        device_test["p_value"] is not None
        and device_test["p_value"] >= PREFLIGHT_ALPHA
        and device_test["zero_probability_draws"] == 0
    )
    return outcome


def _tokens(tokenizer: Any, batch_x: Any) -> tuple[Any, Any]:
    import torch

    with torch.no_grad():
        return tokenizer.encode(batch_x, half=True)


def _s1_loss(model: Any, tokens: tuple[Any, Any], stamps: Any) -> tuple[Any, Any]:
    from torch.nn import functional

    s1, s2 = tokens
    logits, _ = model.decode_s1(s1[:, :-1], s2[:, :-1], stamps[:, :-1, :])
    loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), s1[:, 1:].reshape(-1))
    return logits, loss


def _preflight_parity(
    api: Any, spec: _RunSpec, state: Mapping[str, Any], tokenizer: Any, batch: Any, device: Any
) -> dict[str, Any]:
    """s1 logits and s1-loss gradients on the device against the CPU, dropout set to 0."""
    import torch

    batch_x, batch_stamp = batch
    tokens = _tokens(tokenizer, batch_x)
    results = {}
    for label, target in (("device", device), ("cpu", torch.device("cpu"))):
        model = _training_model(api, spec, state, target)
        _zero_dropout(model)
        model.train()
        logits, loss = _s1_loss(
            model, (tokens[0].to(target), tokens[1].to(target)), batch_stamp.to(target)
        )
        loss.backward()
        gradient = torch.cat(
            [_cpu_float64(p.grad).reshape(-1) for p in model.parameters() if p.grad is not None]
        )
        results[label] = (_cpu_float64(logits), gradient)
    logits_device, gradient_device = results["device"]
    logits_cpu, gradient_cpu = results["cpu"]
    scale = float(logits_cpu.abs().max()) or 1.0
    relative = float((logits_device - logits_cpu).abs().max()) / scale
    cosine = float(
        torch.nn.functional.cosine_similarity(gradient_device, gradient_cpu, dim=0, eps=1e-30)
    )
    return {
        "passed": bool(
            relative < PREFLIGHT_PARITY_MAX_RELATIVE and cosine > PREFLIGHT_PARITY_MIN_COSINE
        ),
        "max_relative_logit_difference": relative,
        "gradient_cosine": cosine,
    }


def _training_step(model: Any, tokenizer: Any, optimizer: Any, batch: Any, device: Any) -> float:
    """One step of the vendored loop body (FB:282-297) on ``device``."""
    import torch

    batch_x, batch_stamp = (tensor.to(device) for tensor in batch)
    with torch.no_grad():
        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
    logits = model(token_seq_0[:, :-1], token_seq_1[:, :-1], batch_stamp[:, :-1, :])
    loss, _, _ = model.head.compute_loss(
        logits[0], logits[1], token_seq_0[:, 1:], token_seq_1[:, 1:]
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=VENDORED_GRAD_CLIP)
    optimizer.step()
    return float(loss.detach())


def _optimizer(model: Any, recipe: Any) -> Any:
    import torch

    return torch.optim.AdamW(
        model.parameters(),
        lr=recipe.predictor_learning_rate,
        betas=(recipe.adam_beta1, recipe.adam_beta2),
        weight_decay=recipe.adam_weight_decay,
    )


def _preflight_steps(
    api: Any,
    spec: _RunSpec,
    state: Mapping[str, Any],
    tokenizers: Mapping[str, Any],
    batches: Sequence[Any],
    device: Any,
    steps: int,
) -> dict[str, Any]:
    """``steps`` training steps on the device and on the CPU from the same weights and batches,
    s1 draws through the router: finite losses, means within 1 %."""
    import torch

    losses: dict[str, list[float]] = {}
    for label, target in (("device", device), ("cpu", torch.device("cpu"))):
        model = _training_model(api, spec, state, target)
        model.train()
        optimizer = _optimizer(model, spec.recipe)
        torch.manual_seed(spec.seeds["yaml_seed"])
        router = MultinomialRouter(spec.seeds["router_seed"])
        values: list[float] = []
        with router.installed():
            for step in range(steps):
                batch = batches[step % len(batches)]
                values.append(_training_step(model, tokenizers[label], optimizer, batch, target))
        losses[label] = values
    finite = all(math.isfinite(value) for series in losses.values() for value in series)
    mean_device = sum(losses["device"]) / steps
    mean_cpu = sum(losses["cpu"]) / steps
    relative = abs(mean_device - mean_cpu) / abs(mean_cpu) if mean_cpu else math.inf
    return {
        "passed": bool(finite and relative <= PREFLIGHT_STEPS_MAX_RELATIVE),
        "steps": steps,
        "finite": finite,
        "mean_loss_device": mean_device if finite else None,
        "mean_loss_cpu": mean_cpu if finite else None,
        "relative_difference": relative if finite and math.isfinite(relative) else None,
    }


def _preflight_deterministic(
    api: Any, spec: _RunSpec, state: Mapping[str, Any], tokenizer: Any, batch: Any, device: Any
) -> dict[str, Any]:
    """Whether one training step on the device runs under deterministic algorithms."""
    import torch

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    refused, op = False, None
    try:
        torch.use_deterministic_algorithms(True)
        model = _training_model(api, spec, state, device)
        model.train()
        router = MultinomialRouter(spec.seeds["router_seed"])
        with router.installed():
            _training_step(model, tokenizer, _optimizer(model, spec.recipe), batch, device)
    except RuntimeError as exc:
        if "deterministic" not in str(exc):
            raise
        refused = True
        match = _DETERMINISTIC_OP.search(str(exc))
        op = match.group(1) if match else None
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    return {"passed": not refused, "refused": refused, "op": op}


def _preflight_checkpoint_gradients(
    api: Any, spec: _RunSpec, state: Mapping[str, Any], tokenizer: Any, batch: Any
) -> dict[str, Any]:
    """CPU gradients of one training forward and backward, with and without activation
    checkpointing, compared bit for bit (dropout active, the same RNG streams)."""
    import torch

    gradients: list[dict[str, Any]] = []
    cpu = torch.device("cpu")
    batch_x, batch_stamp = batch
    tokens = _tokens(tokenizer, batch_x)
    for wrapped in (False, True):
        model = _fresh_model(api, spec, state, cpu)
        model.train()
        if wrapped:
            enable_activation_checkpointing(model)
        torch.manual_seed(spec.seeds["yaml_seed"])
        router = MultinomialRouter(spec.seeds["router_seed"])
        with router.installed():
            logits = model(tokens[0][:, :-1], tokens[1][:, :-1], batch_stamp[:, :-1, :])
            loss, _, _ = model.head.compute_loss(
                logits[0], logits[1], tokens[0][:, 1:], tokens[1][:, 1:]
            )
            loss.backward()
        gradients.append(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )
    plain, wrapped_gradients = gradients
    mismatches = sorted(
        name
        for name in set(plain) | set(wrapped_gradients)
        if name not in plain
        or name not in wrapped_gradients
        or not _same_bits(plain[name], wrapped_gradients[name])
    )
    return {"passed": not mismatches, "tensors": len(plain), "mismatches": mismatches[:8]}


def _train_dataset(vendored: VendoredFinetune, spec: _RunSpec, config: Any) -> Any:
    use_project = spec.wrappers["gap_aware"] and (
        spec.windows["gap_aware"] or spec.recipe.normalisation != "vendored"
    )
    dataset_class = (
        make_dataset_class(vendored.CustomKlineDataset, normalisation=spec.recipe.normalisation)
        if use_project
        else vendored.CustomKlineDataset
    )
    return dataset_class(
        data_path=config.data_path,
        data_type="train",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )


def _is_out_of_memory(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _run_preflight(
    spec: _RunSpec, vendored: VendoredFinetune, api: Any, monitor: _Monitor
) -> dict[str, Any]:
    """The requested :data:`PREFLIGHT_CHECKS` on ``compute.device`` (a check that raises is
    recorded as not passed; memory, time and signals still end the run). With
    ``compute.activation_checkpointing`` the checks that train (parity, steps, deterministic)
    run wrapped models (:func:`_training_model`); ``multinomial`` runs without gradients and
    ``checkpoint_gradients`` compares both forms by design, so neither changes."""
    import torch

    preflight = spec.preflight
    assert preflight is not None
    device = torch.device(spec.compute["device"])
    config = vendored.config_loader.CustomFinetuneConfig(str(spec.run_dir / RUN_YAML))
    torch.manual_seed(spec.seeds["yaml_seed"])
    tokenizer_cpu, model_cpu = _eval_models(api, spec)
    if spec.wrappers["weight_check"]:
        problems = state_mismatches(
            model_cpu, spec.predictor.directory / "model.safetensors"
        ) + state_mismatches(tokenizer_cpu, spec.tokenizer.directory / "model.safetensors")
        if problems:
            raise RunnerError(f"the loaded weights differ from the verified files: {problems[0]}")
    state = _snapshot(model_cpu)
    tokenizer_device = api.KronosTokenizer.from_pretrained(
        str(spec.tokenizer.directory), strict=True
    )
    tokenizer_device.eval()
    tokenizer_device.to(device)
    tokenizers = {"device": tokenizer_device, "cpu": tokenizer_cpu}
    dataset = _train_dataset(vendored, spec, config)
    loader = pseudo_epoch_loader(
        dataset,
        batch_size=spec.recipe.batch_size,
        steps=max(1, min(preflight["steps"], max(1, len(dataset) // spec.recipe.batch_size))),
        seed=spec.seeds["sampler_seed"],
    )
    batches = list(loader)
    if not batches:
        raise RunnerError("the fold has too few windows for one preflight batch")
    batch = batches[0]
    results: dict[str, Any] = {}
    for check in preflight["checks"]:
        monitor.check_deadline()
        try:
            if check == "multinomial":
                with torch.no_grad():
                    model_device = _fresh_model(api, spec, state, device)
                    model_device.eval()
                    batch_x, batch_stamp = (tensor.to(device) for tensor in batch)
                    s1, s2 = tokenizer_device.encode(batch_x, half=True)
                    logits, _ = model_device.decode_s1(
                        s1[:, :-1], s2[:, :-1], batch_stamp[:, :-1, :]
                    )
                    probabilities = torch.softmax(logits.float(), dim=-1)
                    probabilities = probabilities.reshape(-1, probabilities.shape[-1])
                    probabilities = probabilities[: preflight["multinomial_rows"]].contiguous()
                results[check] = _preflight_multinomial(
                    probabilities,
                    device=device,
                    repetitions=preflight["multinomial_repetitions"],
                    seed=spec.seeds["router_seed"],
                )
            elif check == "parity":
                results[check] = _preflight_parity(api, spec, state, tokenizer_cpu, batch, device)
            elif check == "steps":
                results[check] = _preflight_steps(
                    api, spec, state, tokenizers, batches, device, preflight["steps"]
                )
            elif check == "deterministic":
                results[check] = _preflight_deterministic(
                    api, spec, state, tokenizer_device, batch, device
                )
            else:
                results[check] = _preflight_checkpoint_gradients(
                    api, spec, state, tokenizer_cpu, batch
                )
        except (MemoryAbort, TimeAbort):
            raise
        except Exception as exc:  # a failing check is a measurement, not a crash
            if _is_out_of_memory(exc):
                raise
            results[check] = {"passed": False, "error": _error_record(exc)}
        monitor.sample_device_memory()
    return results


# -- the result ------------------------------------------------------------------------------


def _error_record(exc: BaseException) -> dict[str, str]:
    lines = str(exc).strip().splitlines()
    message = redact_home(lines[0] if lines else "")[:ERROR_MESSAGE_MAX]
    return {"type": type(exc).__name__, "message": message}


def _blank_result(run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "mode": None,
        "status": "failed",
        "stop_reason": None,
        "error": None,
        "finite": True,
        "started_at": _now_text(),
        "finished_at": None,
        "wall_seconds": None,
        "cpu_seconds": None,
        "device": None,
        "threads": None,
        "deterministic": None,
        "torch_version": None,
        "numpy_version": None,
        "pandas_version": None,
        "platform": default_label(),
        "inputs": None,
        "wrappers": None,
        "dataset": None,
        "steps": 0,
        "onecycle_total_steps": None,
        "lr_last": None,
        "step_seconds": {"count": 0, "median": None, "p10": None, "p90": None},
        "step_times": None,
        "step_offsets": None,
        "memory": {
            "peak_rss_bytes": None,
            "peak_mps_driver_bytes": None,
            "swap_start_bytes": None,
            "swap_growth_bytes": None,
            "max_pressure_level": None,
            "checks": 0,
        },
        "epochs": [],
        "selected_pseudo_epoch": None,
        "selected": None,
        "vendored_best_model": None,
        "nll": None,
        "preflight": None,
    }


def _json_safe(value: Any) -> tuple[Any, bool]:
    """``value`` with every non-finite float replaced by ``None``; whether one was replaced."""
    if isinstance(value, float):
        return (value, False) if math.isfinite(value) else (None, True)
    if isinstance(value, Mapping):
        replaced = False
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[key], hit = _json_safe(item)
            replaced = replaced or hit
        return out, replaced
    if isinstance(value, (list, tuple)):
        items = [_json_safe(item) for item in value]
        return [item for item, _ in items], any(hit for _, hit in items)
    return value, False


def _write_result(run_dir: Path, result: Mapping[str, Any]) -> None:
    document, replaced = _json_safe(dict(result))
    if replaced:
        document["finite"] = False
    text = json.dumps(document, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
    atomic_write_text(run_dir / RESULT_JSON, text + "\n")


def _fill_monitor(result: dict[str, Any], monitor: _Monitor | None, session: Any) -> None:
    if monitor is None:
        return
    result["steps"] = monitor.optimizer_steps
    result["onecycle_total_steps"] = monitor.onecycle_total_steps
    result["lr_last"] = monitor._current_lr()
    result["step_seconds"] = monitor.step_summary()
    if monitor.timing is not None:
        result["step_times"] = list(monitor.timed_durations)
        result["step_offsets"] = list(monitor.timed_offsets)
    result["epochs"] = [dict(entry) for entry in monitor.epochs]
    result["finite"] = result["finite"] and monitor.finite
    memory = result["memory"]
    memory["peak_mps_driver_bytes"] = monitor.peak_device_bytes
    if monitor.memory_guard is not None:
        memory.update(monitor.memory_guard.summary())
    if session is not None and session.dataset is not None:
        result["dataset"] = dict(session.dataset)


# -- the run ---------------------------------------------------------------------------------------


def _install_signal_handlers() -> None:
    def handler(signum: int, frame: Any) -> None:
        for name in ("SIGTERM", "SIGHUP"):
            signal.signal(getattr(signal, name), signal.SIG_IGN)
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGHUP"):
        signal.signal(getattr(signal, name), handler)


def _device_memory_sampler(device: str) -> Callable[[], int | None] | None:
    if device != "mps":
        return None

    def sample() -> int | None:
        import torch

        try:
            return int(torch.mps.driver_allocated_memory())
        except (AttributeError, RuntimeError):
            return None

    return sample


def _train_or_time(
    spec: _RunSpec,
    vendored: VendoredFinetune,
    api: Any,
    monitor: _Monitor,
    session_box: list[Any],
    result: dict[str, Any],
) -> str:
    """Mode ``train`` or ``timing``: the vendored predictor phase with the wrappers; returns the
    stop reason."""
    router = MultinomialRouter(spec.seeds["router_seed"]) if spec.wrappers["router"] else None
    session = _Session(spec, vendored, monitor, router=router)
    session_box.append(session)
    trainer_class = _trainer_class(vendored, spec)
    stop_reason: str
    with session.installed():
        trainer = trainer_class(str(spec.run_dir / RUN_YAML))
        try:
            trainer.train_basemodel_phase()
        except EarlyStop:
            stop_reason = "early_stop"
        except TimingDone:
            stop_reason = "timing_done"
        else:
            if spec.mode == "timing":
                raise RunnerError("the training loop ended before the timed steps were done")
            monitor.finish()
            stop_reason = "max_epochs"
    if spec.mode != "train":
        return stop_reason
    best_files = _file_records(spec.run_dir / VENDORED_BEST_RELPATH)
    result["vendored_best_model"] = {"dir": VENDORED_BEST_RELPATH, "files": best_files}
    if spec.wrappers["early_stopping"]:
        if monitor.best_state is None:
            raise RunnerError("no pseudo-epoch was evaluated, nothing to select")
        result["selected_pseudo_epoch"] = monitor.best_epoch
        result["selected"] = _save_selected(api, spec, monitor.best_state)
    else:
        if len(best_files) != len(CHECKPOINT_FILE_NAMES):
            raise RunnerError("the vendored keep-best checkpoint was never saved")
        result["selected_pseudo_epoch"] = monitor.vendored_best_epoch()
        result["selected"] = {
            "dir": VENDORED_BEST_RELPATH,
            "source": "vendored_best",
            "files": best_files,
        }
    return stop_reason


def run(run_dir: Path) -> int:
    """One child run in ``run_dir``; writes ``runner_result.json`` in every case and returns the
    exit code of its status (``128 + n`` after SIGTERM or SIGHUP)."""
    started = time.monotonic()
    run_dir = Path(run_dir).absolute()
    result = _blank_result(run_dir)
    exit_code = EXIT_FAILED
    monitor: _Monitor | None = None
    session_box: list[Any] = []
    try:
        _install_signal_handlers()
        spec = _read_run_spec(run_dir)
        result.update(
            mode=spec.mode,
            device=spec.compute["device"],
            threads=spec.compute["threads"],
            deterministic=spec.compute["deterministic"],
            wrappers=dict(spec.wrappers),
        )
        configure_runtime(spec.compute["threads"], deterministic=spec.compute["deterministic"])
        import numpy
        import pandas
        import torch

        result.update(
            torch_version=torch.__version__,
            numpy_version=numpy.__version__,
            pandas_version=pandas.__version__,
        )
        from ..kronos.vendored import import_kronos

        api = import_kronos()
        vendored = import_vendored_finetune()
        result["inputs"] = _verify_inputs(spec)
        if spec.compute["device"] == "mps" and not torch.backends.mps.is_available():
            raise RunnerError("MPS is not available on this machine")
        deadline = None if spec.deadline_seconds is None else started + spec.deadline_seconds
        guard = None
        if spec.compute["memory_guard"]:
            guard = MemoryGuard(
                read_memory,
                every_steps=spec.memory["every_steps"],
                max_swap_growth_bytes=spec.memory["max_swap_growth_bytes"],
                critical_level=spec.memory["critical_pressure_level"],
            )
            guard.start()
        evaluate = None
        if spec.mode == "train":
            from .nll import NLL_STEPS, load_nll_inputs, nll4_tf

            assert spec.nll_inputs_path is not None and spec.nll_inputs is not None
            inputs = load_nll_inputs(
                spec.nll_inputs_path, expected_sha256=spec.nll_inputs["sha256"]
            )
            if inputs.decisions != spec.nll_inputs["decisions"]:
                raise RunnerError("the NLL4-TF inputs hold another number of decisions")
            nll_tokenizer, nll_model = _eval_models(api, spec)
            result["nll"] = {
                "decisions": inputs.decisions,
                "steps": NLL_STEPS,
                "batch_size": spec.nll_batch_size,
                "evaluation_seconds": [],
            }

            def evaluate(epoch: int) -> float:
                assert monitor is not None
                nll_model.load_state_dict(monitor.model.state_dict())
                return nll4_tf(
                    nll_tokenizer,
                    nll_model,
                    inputs,
                    clip=spec.recipe.clip,
                    batch_size=spec.nll_batch_size,
                ).value

        monitor = _Monitor(
            evaluate=evaluate,
            early_stopping=spec.mode == "train" and spec.wrappers["early_stopping"],
            patience=spec.recipe.patience,
            memory_guard=guard,
            deadline=deadline,
            timing=spec.timing,
            device_memory=_device_memory_sampler(spec.compute["device"]),
        )
        if deadline is not None:
            monitor.check_deadline()
        if spec.mode == "preflight":
            result["preflight"] = _run_preflight(spec, vendored, api, monitor)
            stop_reason = "preflight_done"
        else:
            stop_reason = _train_or_time(spec, vendored, api, monitor, session_box, result)
        result.update(status="completed", stop_reason=stop_reason)
        exit_code = EXIT_COMPLETED
    except MemoryAbort as exc:
        result.update(status="memory_abort", error=_error_record(exc))
        exit_code = EXIT_MEMORY_ABORT
    except TimeAbort as exc:
        result.update(status="time_abort", error=_error_record(exc))
        exit_code = EXIT_TIME_ABORT
    except SystemExit as exc:
        result.update(status="failed", error=_error_record(exc))
        code = exc.code if _is_int(exc.code) else EXIT_FAILED
        exit_code = code if code != EXIT_COMPLETED else EXIT_FAILED
        result["error"]["message"] = f"stopped by a signal (exit {exit_code})"
    except KeyboardInterrupt as exc:
        result.update(status="failed", error=_error_record(exc))
        exit_code = 128 + signal.SIGINT
    except BaseException as exc:  # noqa: BLE001 - recorded; the exit code says what happened
        traceback.print_exc()
        if _is_out_of_memory(exc):
            result.update(status="memory_abort", error=_error_record(exc))
            exit_code = EXIT_MEMORY_ABORT
        else:
            result.update(status="failed", error=_error_record(exc))
            exit_code = EXIT_FAILED
    session = session_box[0] if session_box else None
    try:
        _fill_monitor(result, monitor, session)
        if monitor is not None and result["nll"] is not None:
            result["nll"]["evaluation_seconds"] = list(monitor.nll_seconds)
        result["memory"]["peak_rss_bytes"] = peak_rss_bytes()
        result["finished_at"] = _now_text()
        result["wall_seconds"] = time.monotonic() - started
        result["cpu_seconds"] = time.process_time()
        _write_result(run_dir, result)
    except BaseException:  # noqa: BLE001 - nothing more can be recorded
        traceback.print_exc()
        return exit_code if exit_code != EXIT_COMPLETED else EXIT_FAILED
    sys.stdout.flush()
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """``python -B -P -m kronos_agent.finetune.runner --run-dir DIR``."""
    parser = argparse.ArgumentParser(
        prog="kronos_agent.finetune.runner",
        description="Run one fine-tune, timing measurement or preflight described by run.json.",
    )
    parser.add_argument("--run-dir", required=True, help="the run directory holding run.json")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir.name}", file=sys.stderr)
        return EXIT_FAILED
    return run(run_dir)


__all__ = [
    "CHILD_LOG",
    "CRITICAL_PRESSURE_LEVEL",
    "DEVICES",
    "EXIT_COMPLETED",
    "EXIT_FAILED",
    "EXIT_MEMORY_ABORT",
    "EXIT_TIME_ABORT",
    "MAX_SWAP_GROWTH_BYTES",
    "MEMORY_CHECK_EVERY_STEPS",
    "MODES",
    "PREFLIGHT_CHECKS",
    "PREFLIGHT_MULTINOMIAL_REPETITIONS",
    "PREFLIGHT_MULTINOMIAL_ROWS",
    "PREFLIGHT_STEPS",
    "RESULT_JSON",
    "RESULT_SCHEMA_VERSION",
    "RUN_JSON",
    "RUN_SCHEMA_VERSION",
    "RUN_YAML",
    "SELECTED_DIRNAME",
    "STATUSES",
    "STOP_REASONS",
    "VENDORED_BEST_RELPATH",
    "VENDORED_FINETUNE_MODULES",
    "VENDORED_SAVE_DIRNAME",
    "WRAPPERS",
    "CleanValidation",
    "EarlyStop",
    "MemoryAbort",
    "MemoryGuard",
    "MemoryReading",
    "MultinomialRouter",
    "RunnerError",
    "TimeAbort",
    "TimingDone",
    "TrainLoaderGate",
    "VendoredFinetune",
    "enable_activation_checkpointing",
    "import_vendored_finetune",
    "main",
    "make_dataset_class",
    "parse_swap_used",
    "peak_rss_bytes",
    "pseudo_epoch_loader",
    "read_memory",
    "run",
    "state_mismatches",
]


if __name__ == "__main__":
    sys.exit(main())
