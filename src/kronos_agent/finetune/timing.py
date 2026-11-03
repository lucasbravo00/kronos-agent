"""The measurement step of the retraining: ``kronos-agent finetune timing`` (``docs/RETRAINING.md``,
section (a), phase P0).

It measures on this machine, without market data, what the gate-3 estimates of the pilot, the
walk-forward and the evaluation fans need, and records the compute choice of design 3.4 (device,
threads, batch, deterministic mode, activation checkpointing) in
``reports/finetune/timing-2.json`` and ``.md`` (interfaces 5.9):

- M1 training step time and peak memory: CPU at 2 and 8 threads and MPS (2 CPU threads) at batch
  8, 16 and 32, smallest batch first (a configuration that fails or aborts skips its larger
  batches); a fresh training child each, 3 untimed and 10 timed optimizer steps;
- M2 activation checkpointing at batch 32 where batch 32 does not fit as is (a peak above 9 GiB,
  swap growth, or on the CPU a memory abort): 3 + 10 steps with the wrap, and once, before the
  first configuration that relies on the wrap, the CPU gradient bit-identity of one step with
  and without it;
- M3 the MPS preflight (``multinomial``, ``parity``, ``steps``, ``deterministic``) before any
  MPS timing: MPS is timed only once its multinomial, parity and steps checks passed, and keeps
  deterministic mode only if its deterministic check passed;
- M4 sustained MPS throughput at batch 32 for 20 minutes; the thermal factor ``theta`` is the
  median step time of the last 10 minutes over the median of the first 2. M7 (contention: 12
  fans with 1, then 2 workers) runs during it, between those two windows;
- M5 fan speed ``t(4)``, ``t(8)``, ``t(24)``: one warm-up and 3 timed fans each, 2 threads;
- M6 worker scaling: 12 fans with 1, 2, 3 and 4 spawn-started workers;
- M8 NLL4-TF time on 48 synthetic decisions (CPU, 2 threads);
- M9 the identity control: the shipped weights through ``save_pretrained``, a scratch registry,
  ``load_models(checkpoint=...)`` and a spawn-started worker; its fan must be bit-identical.

The training measurements run the real child (``finetune.train`` and ``finetune.runner``, modes
``timing`` and ``preflight``) with every R1 wrapper on and early stopping off, on a synthetic fold
exported by the fold exporter (:data:`TIMING_SYNTHETIC_ROWS` hours of
``export_csv.synthetic_frame`` candles from 2020-01-01) and the verified shipped weights; only the
batch size changes between configurations, and a pseudo-epoch of
:data:`TIMING_PSEUDO_EPOCH_STEPS` steps keeps the vendored validation out of every measurement.
The fan, NLL4-TF and identity measurements run the batch and NLL code in this process (the
governed 2 threads) and in spawn-started workers. No market data, no outcome, no test.

Order: the synthetic export and the shipped models; M9, M8 and M5 (cheap, and every later
estimate needs them); the models released; M1 on the CPU and M2 there when its batch 32 does not
fit; M3, M1 on MPS, M2 there when needed, M4 with M7; M6.

MPS gets the remedy M2 gives the CPU (design 3.5: activation checkpointing before any batch
change). When the chosen CPU configuration's batch 32 does not fit as is and M2 passed there, M3,
the MPS batch 32 of M1 and M4 run with activation checkpointing from their first attempt. An MPS
unit (M3, an M1 batch, M4) that ends ``memory_abort`` without it is retried once with it, once
M2's gradient bit-identity has passed (run first when it has not) and the retry's prior fits the
budget; the MPS units after it, from that batch up, run with it. Both attempts are recorded: the
unit's entry holds the retry, ``first_attempt`` the aborted attempt (a retried M4 also keeps the
M7 that ran during it); a unit that is not retried says why under ``retry``. A retry is charged
to ``retries``, whose prior the estimate includes (one retry of the preflight's size).

Budget (gate 3, SPEC section 17): a run's limits (:class:`TimingBudget`) are the step's own,
:data:`TIMING_MAX_CPU_SECONDS` and :data:`TIMING_MAX_GPU_MINUTES`, below gate 3, unless the owner
approved more CPU in chat for this measurement: ``finetune timing --approved-by-owner CPU_HOURS``
makes ``CPU_HOURS x 3600`` CPU-seconds its CPU limit, a ceiling that may pass gate 3 under that
approval, while the GPU limit stays (D-055). The budget is how far a run may go, not what it
measures: it is not part of the recorded plan, and the record states it (``spend``, ``approval``:
the approved CPU-hours, when the flag was given, the estimate and whether the limit passes gate
3). The CPU limit a run works against, its stop limit, is the step's own limit, which stays
:data:`APPROVAL_RESERVE_CPU_SECONDS` below gate 3, or an approved ceiling minus that reserve, so
that the unit running when the spend reaches the stop limit cannot take it past the ceiling: an
approved ceiling has no tolerance. The estimate from the priors of design (a) is printed first
with the budget, M3 and the retries priced by their content and M4 by its whole lifetime
(D-054), and a run whose estimate passes the stop limit or the GPU limit is refused
(:class:`TimingBudgetExceeded`) before anything runs. While it runs, each measurement is charged
the larger of the CPU time of this process and its reaped children and what it reports (a
child's ``cpu_seconds``; fans: wall x threads, the gate-3 accounting of D-033), and the wall
time of each MPS child as GPU time. A measurement or retry whose prior would pass the stop limit
or the GPU limit is not started, and the run stops (status ``stopped``) as soon as the spend
passes one; everything after is recorded as not run. Each training child gets the wall time the
remaining budget allows it as its ``deadline_seconds`` (a CPU child: the CPU-seconds left below
the stop limit / its threads; a short MPS child: the smaller of the remaining GPU-minutes x 60
and those CPU-seconds / 2) and a parent timeout of that plus 60 s. The 20-minute M4 child, whose
CPU use that bound would cut short, starts (or is retried) only when its whole lifetime fits the
remaining GPU-minutes, gets the remaining GPU time and is watched while it runs: every few
seconds its CPU time (``ps``) plus the spend so far is checked, and it is stopped once a limit is
passed. M7, which runs during M4 and holds the watch while it runs, admits each of its
configurations on the M4 child's CPU time read again at its start. Without the owner's approval
the limits stay below gate 3 (3,600 CPU-seconds, one GPU-hour).

The record is tied to the chosen device, the torch version, the platform label and the training
code (:func:`require_fresh_timing`): later estimates refuse a stale one. Its stem is this plan's
(:data:`TIMING_STEM`): the first run recorded another plan under ``timing`` (D-053), and a changed
study is a new study with a new stem (D-034, D-054), so those three files stay as they are. No
path, user or host name enters it: runs are named by their id, the scratch by its name under
``<fine-tune root>/.scratch/``, and every error text is cleaned of paths.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import io
import json
import math
import os
import re
import resource
import statistics
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..data.fsutil import ensure_directory, exclusive_lock, fsync_directory
from ..data.partition import TRAINING_START
from ..data.timeutil import HOUR_MS, to_ms, utc_text
from ..hashing import sha256_file
from ..params import Params
from ..paths import REPO_ROOT, default_label, redact_home, reports_dir
from ..runtime import require_configured
from .recipe import R1, Recipe
from .runner import PREFLIGHT_STEPS

if TYPE_CHECKING:
    from ..kronos.weights import LoadedModels
    from .export_csv import ExportRecord
    from .folds import Fold
    from .nll import NllResult
    from .train import RunnerResult, RunningChild

# the plan of D-054; reports/finetune/timing.* hold the first run's plan and record (D-053), which
# a run never rewrites: a changed plan takes a new stem (D-034)
TIMING_STEM = "timing-2"
FINETUNE_REPORTS_SUBDIR = "finetune"  # reports/finetune/
TIMING_SCHEMA_VERSION = 2  # 2: CPU before MPS, activation checkpointing and retries on MPS
# the step's own limits (a run's budget without the owner's approval; never part of the plan)
TIMING_MAX_CPU_SECONDS = 3000
TIMING_MAX_GPU_MINUTES = 45.0
# gate 3 (SPEC section 17; kronos.batch.GATE3_CPU_SECONDS): the limits above stay below it; only
# the CPU-hours the owner approved (--approved-by-owner, D-055) take the CPU limit past it
GATE3_CPU_SECONDS = 3600
GATE3_GPU_MINUTES = 60.0
# under the owner's approval the run works against its ceiling minus this reserve (its stop
# limit), so that the unit running when the spend reaches the stop limit cannot take it past the
# ceiling (D-055): it covers the most one admitted unit can add past the stop limit, M6's
# four-worker configuration (admitted on its 75 CPU-seconds of prior, charged 289 in the first
# run, D-053 (8)) or M7 with the M4 child's CPU, which is not watched while M7 runs; the step's
# own limit stays as far below gate 3
APPROVAL_RESERVE_CPU_SECONDS = 600.0
SECONDS_PER_HOUR = 3600.0
TIMING_SYNTHETIC_ROWS = 4096
TIMING_CSV_START = TRAINING_START  # synthetic candles 2020-01-01 onwards
TIMING_SYNTHETIC_SEED = 0
TIMING_FOLD_LABEL = "synthetic-timing"
TIMING_PSEUDO_EPOCH_STEPS = 4096  # no vendored validation inside a measurement
RECIPE_BATCH = R1.batch_size  # 32: the batch the device rule compares
MPS_THREADS = 2  # MPS runs use 2 CPU threads (design 3.4)
M1_CONFIGS: tuple[tuple[str, int], ...] = (("cpu", 2), ("cpu", 8), ("mps", MPS_THREADS))
M1_BATCHES: tuple[int, ...] = (8, 16, 32)  # smallest first
M1_WARMUP_STEPS = 3
M1_TIMED_STEPS = 10
M2_PEAK_LIMIT_BYTES = 9 * 2**30
M2_GRADIENT_CHECKS: tuple[str, ...] = ("checkpoint_gradients",)
M2_GRADIENT_PASSES = 2  # the gradient check: one forward and backward with the wrap, one without
# swap growth up to this much across a child's run is other processes' noise, not "swapped"
SWAP_GROWTH_TOLERANCE_BYTES = 256 * 2**20
M3_CHECKS: tuple[str, ...] = ("multinomial", "parity", "steps", "deterministic")
M3_REQUIRED_CHECKS: tuple[str, ...] = ("multinomial", "parity", "steps")
M4_SECONDS = 1200.0
# the M4 child's life besides its 20 timed minutes: start-up, warm-up steps and the step that
# passes the 20 minutes (the tests' stand-in: 30 s); M4 needs all of it of the GPU budget (D-054)
M4_OVERHEAD_SECONDS = 60.0
M4_FIRST_WINDOW_SECONDS = 120.0
M4_LAST_WINDOW_SECONDS = 600.0
# M7 starts this long after the M4 child: past its start-up and first window, before its last one
M7_START_DELAY_SECONDS = 180.0
M5_LENGTHS: tuple[int, ...] = (4, 8, 24)
M5_REPEATS = 3
M6_WORKERS: tuple[int, ...] = (1, 2, 3, 4)
M7_WORKERS: tuple[int, ...] = (1, 2)
FANS_PER_WORKER_TEST = 12
M8_DECISIONS = 48
MEASUREMENTS: tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9")
RETRIES = "retries"  # what the retries with activation checkpointing are charged to
SPEND_ITEMS: tuple[str, ...] = (*MEASUREMENTS, "loads", RETRIES)
MPS_ONLY: tuple[str, ...] = ("M3", "M4", "M7", RETRIES)
PRIOR_SECONDS_PER_STEP_B32: Mapping[tuple[str, int], float] = MappingProxyType(
    {("cpu", 2): 6.1, ("cpu", 8): 5.0, ("mps", MPS_THREADS): 3.2}
)
# M3 priced by its content (D-054; the design's 150 CPU-seconds and 2 GPU-minutes left out the CPU
# side): the child runs check (b)'s forward and backward and check (c)'s PREFLIGHT_STEPS training
# steps at batch 32 on the CPU at 2 threads, at the step time the first run measured there with
# activation checkpointing (D-053 (2)), and those plus check (a)'s forward and check (d)'s step on
# MPS at the design's prior; CPU-seconds at wall x 2 threads (D-033), the child's whole wall as
# GPU time (an MPS child is charged so)
PREFLIGHT_CPU_STEPS = PREFLIGHT_STEPS + 1
PREFLIGHT_MPS_STEPS = PREFLIGHT_STEPS + 3
PREFLIGHT_CPU_SECONDS_PER_STEP = 9.68
_PREFLIGHT_WALL_SECONDS = (
    PREFLIGHT_CPU_STEPS * PREFLIGHT_CPU_SECONDS_PER_STEP
    + PREFLIGHT_MPS_STEPS * PRIOR_SECONDS_PER_STEP_B32[("mps", MPS_THREADS)]
)
PREFLIGHT_PRIOR: tuple[float, float] = (
    _PREFLIGHT_WALL_SECONDS * MPS_THREADS,
    _PREFLIGHT_WALL_SECONDS / 60.0,
)
# (CPU-seconds, GPU-minutes) per measurement: the table of design section (a), as D-054 amends it
PRIORS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "M1": (1350.0, 2.0),
        "M2": (200.0, 1.0),
        "M3": PREFLIGHT_PRIOR,
        "M4": (120.0, (M4_SECONDS + M4_OVERHEAD_SECONDS) / 60.0),  # its whole lifetime
        "M5": (210.0, 0.0),
        "M6": (300.0, 0.0),
        "M7": (150.0, 0.0),
        "M8": (20.0, 0.0),
        "M9": (20.0, 0.0),
        "loads": (100.0, 0.0),
        RETRIES: PREFLIGHT_PRIOR,  # one MPS retry with activation checkpointing, the size of M3
    }
)
# what M1 and M2 spend on MPS; left out, with MPS_ONLY, on a machine without MPS
PRIORS_MPS_SIDE: Mapping[str, tuple[float, float]] = MappingProxyType(
    {"M1": (150.0, 2.0), "M2": (0.0, 1.0)}
)
PRIOR_FAN_SECONDS: tuple[float, float] = (0.44, 0.678)  # t(L) = 0.44 + 0.678 L at 2 threads (5.2)
MPS_REQUIRED_SPEEDUP = 1.5
EIGHT_THREADS_REQUIRED_SPEEDUP = 1.3
DEFAULT_MPS_WATERMARK_RATIO = 1.0  # design 3.5; finetune.train applies the same default
CHILD_TIMEOUT_MARGIN_SECONDS = 60.0
CHILD_POLL_SECONDS = 5.0
STATUSES: tuple[str, ...] = ("completed", "stopped", "failed")
UNIT_ORDER: tuple[str, ...] = (
    "setup: the synthetic export and the shipped models",
    "M9",
    "M8",
    "M5",
    "M1 on the CPU (cpu x2, cpu x8; batch 8, 16, 32)",
    "M2 on the CPU (only if the chosen CPU configuration's batch 32 does not fit)",
    "M3 (MPS only; with activation checkpointing if the CPU's batch 32 needs it)",
    (
        "M1 on MPS (mps x2 once M3 passed; batch 8, 16, 32; batch 32 with activation "
        "checkpointing if the CPU's needs it)"
    ),
    "M2 on MPS (only if its batch 32 completed without activation checkpointing and does not fit)",
    "M4 with M7 during it (MPS only)",
    "M6",
    "a memory abort of an MPS unit without activation checkpointing: one retry with it",
)
SCRATCH_PREFIX = "timing-"
LOCK_NAME = "timing.lock"  # <fine-tune root>/.scratch/timing.lock: one timing run at a time
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
MAX_SCRATCH_SUFFIX = 99
PS = "/bin/ps"
PS_TIMEOUT_SECONDS = 10.0
ERROR_TEXT_MAX = 300
GIB = float(2**30)
RETRAINING_MD_RELPATH = "docs/RETRAINING.md"
RECORD_KEYS: tuple[str, ...] = (
    "schema_version",
    "status",
    "environment",
    "code",
    "estimate",
    "spend",
    "approval",
    "measurements",
    "choice",
)

# an absolute path in an error text (also ``~/...`` once the home directory is redacted)
_PATH_TOKEN = re.compile(r"(?<![\w.~/-])(?:~/|/)[^\s'\"`<>,;:()\[\]{}]+")
_PS_TIME = re.compile(r"(?:(\d+)-)?(\d+(?::\d+){0,2})(?:\.(\d+))?")

# clocks, readers and probes are module attributes so that tests can drive them
_clock: Callable[[], float] = time.monotonic
_sleep: Callable[[float], None] = time.sleep


def _cpu_clock() -> float:
    """CPU-seconds (user + system) of this process and of its children reaped so far."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _read_memory() -> Any:
    """``finetune.runner.read_memory()``: the memory-pressure level and the swap in use."""
    from .runner import read_memory

    return read_memory()


def _mps_available() -> bool:
    """Whether torch can use MPS here (read from the configured runtime)."""
    return bool(require_configured().mps_available)


def _parse_ps_time(text: str) -> float | None:
    """Seconds of a ``ps -o time=`` value: ``M:SS.cc`` (macOS) or ``[D-]HH:MM:SS`` (Linux)."""
    match = _PS_TIME.fullmatch(text.strip())
    if match is None:
        return None
    days, clock, fraction = match.groups()
    seconds = 0.0
    for part in clock.split(":"):
        seconds = seconds * 60.0 + int(part)
    if fraction:
        seconds += int(fraction) / 10 ** len(fraction)
    return seconds + 86400.0 * int(days or 0)


def _child_cpu_seconds(child: Any) -> float | None:
    """CPU-seconds the running child has used so far, read with ``ps`` (``None`` when it cannot
    be read: no ``ps``, or the process already ended)."""
    pid = getattr(getattr(child, "process", None), "pid", None)
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    try:
        completed = subprocess.run(
            [PS, "-o", "time=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return _parse_ps_time(completed.stdout)


class TimingError(RuntimeError):
    """The measurement step cannot run, or its record cannot be read (exit 1)."""


class TimingBudgetExceeded(PermissionError):
    """The estimate passes the measurement step's own budget (exit 2); nothing ran."""


class StaleTimingError(PermissionError):
    """The timing record was measured for another device, torch, platform or training code
    (exit 2): measure again before estimating from it."""


@dataclass(frozen=True)
class TimingSteps:
    """The heavy measurements, injectable (:func:`real_steps` gives the real ones; tests pass
    stand-ins). Each is called as its real counterpart: ``run_child(request, root=,
    timeout_seconds=)``, ``start_child(request, root=)``, ``seconds_per_fan(params=, models=,
    evaluation_length=, repeats=)``, ``time_fan_workers(params, workers=, fans=)``,
    ``load_models(name)``, ``nll4_tf(tokenizer, model, inputs, clip=, batch_size=)`` and
    ``identity_control(params, scratch_root=, models=)``."""

    run_child: Callable[..., RunnerResult]  # finetune.train.run
    start_child: Callable[..., RunningChild]  # finetune.train.start_run
    seconds_per_fan: Callable[..., float]  # kronos.batch.measure_seconds_per_fan
    time_fan_workers: Callable[..., dict[str, Any]]  # kronos.batch.time_fan_workers
    load_models: Callable[..., LoadedModels]  # kronos.weights.load_models
    nll4_tf: Callable[..., NllResult]  # finetune.nll.nll4_tf
    identity_control: Callable[..., dict[str, Any]]  # kronos.batch.identity_control


def real_steps() -> TimingSteps:
    """The real measurements (imported here, after the runtime is configured)."""
    from ..kronos import batch
    from ..kronos.weights import load_models
    from . import train
    from .nll import nll4_tf

    return TimingSteps(
        run_child=train.run,
        start_child=train.start_run,
        seconds_per_fan=batch.measure_seconds_per_fan,
        time_fan_workers=batch.time_fan_workers,
        load_models=load_models,
        nll4_tf=nll4_tf,
        identity_control=batch.identity_control,
    )


# -- the plan and its estimate ----------------------------------------------------------------


def _prior_mapping(priors: Mapping[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {"cpu_seconds": float(cpu), "gpu_minutes": float(gpu)}
        for name, (cpu, gpu) in priors.items()
    }


def timing_plan() -> dict[str, Any]:
    """The recorded definition of the measurement step: constants only, identical on every
    machine (what a machine cannot run is recorded in the report as not run) and for every
    budget: the plan is what is measured, the limits a run may spend are not in it
    (:class:`TimingBudget`, recorded with the run's spend; D-055)."""
    from ..kronos.batch import IDENTITY_CHECKPOINT_ID, TIMING_DECISION_HOUR
    from .nll import NLL_BATCH_SIZE

    intercept, slope = PRIOR_FAN_SECONDS
    return {
        "schema_version": TIMING_SCHEMA_VERSION,
        "design": f"{RETRAINING_MD_RELPATH} section (a)",
        "synthetic": {
            "rows": TIMING_SYNTHETIC_ROWS,
            "csv_start": utc_text(to_ms(TIMING_CSV_START)),
            "seed": TIMING_SYNTHETIC_SEED,
            "fold": TIMING_FOLD_LABEL,
            "timing_window": utc_text(to_ms(TIMING_DECISION_HOUR)),
        },
        "recipe": {
            "base": R1.name,
            "model": R1.model,
            "window_rows": R1.window_rows,
            "batch_size": RECIPE_BATCH,
            "pseudo_epoch_steps": TIMING_PSEUDO_EPOCH_STEPS,
            "wrappers": "every R1 wrapper on, early stopping off",
        },
        "measurements": {
            "M1": {
                "configs": [[device, threads] for device, threads in M1_CONFIGS],
                "batches": list(M1_BATCHES),
                "warmup_steps": M1_WARMUP_STEPS,
                "timed_steps": M1_TIMED_STEPS,
                "rule": "smallest batch first; a failed or aborted configuration skips its "
                "larger batches; the CPU first, MPS only after M3 passed; MPS activation "
                "checkpointing as rules.mps_checkpointing",
            },
            "M2": {
                "batch_size": RECIPE_BATCH,
                "warmup_steps": M1_WARMUP_STEPS,
                "timed_steps": M1_TIMED_STEPS,
                "when": "on the chosen CPU configuration, before any MPS unit, when its batch 32 "
                "has a peak above peak_limit_bytes, swap growth above "
                "swap_growth_tolerance_bytes, or a memory abort; on MPS when its batch 32 "
                "completed without activation checkpointing and does not fit",
                "gradient_check": list(M2_GRADIENT_CHECKS),
                "gradient_check_when": "once, before the first configuration that relies on "
                "activation checkpointing",
            },
            "M3": {
                "checks": list(M3_CHECKS),
                "required": list(M3_REQUIRED_CHECKS),
                "batch_size": RECIPE_BATCH,
                "activation_checkpointing": "as rules.mps_checkpointing; the checks that train "
                "wrap their models, on MPS and on the CPU",
                "prior": {
                    "cpu_steps": PREFLIGHT_CPU_STEPS,
                    "cpu_seconds_per_step": PREFLIGHT_CPU_SECONDS_PER_STEP,
                    "mps_steps": PREFLIGHT_MPS_STEPS,
                    "mps_seconds_per_step": PRIOR_SECONDS_PER_STEP_B32[("mps", MPS_THREADS)],
                    "rule": "priced by its content: check (b)'s forward and backward and check "
                    "(c)'s steps on the CPU at 2 threads, at the first run's step time there "
                    "with activation checkpointing, and those plus checks (a) and (d) on MPS; "
                    "CPU-seconds at wall x 2 threads, the child's whole wall as GPU time; "
                    f"{RETRIES} priced as one M3",
                },
            },
            "M4": {
                "seconds": M4_SECONDS,
                "overhead_seconds": M4_OVERHEAD_SECONDS,
                "first_window_seconds": M4_FIRST_WINDOW_SECONDS,
                "last_window_seconds": M4_LAST_WINDOW_SECONDS,
                "batch_size": RECIPE_BATCH,
                "warmup_steps": M1_WARMUP_STEPS,
                "activation_checkpointing": "as the MPS batch 32 of M1 and M2; a retried M4 runs "
                "M7 again",
            },
            "M5": {"lengths": list(M5_LENGTHS), "warmup_fans": 1, "repeats": M5_REPEATS},
            "M6": {"workers": list(M6_WORKERS), "fans": FANS_PER_WORKER_TEST},
            "M7": {
                "workers": list(M7_WORKERS),
                "fans": FANS_PER_WORKER_TEST,
                "start_after_m4_seconds": M7_START_DELAY_SECONDS,
            },
            "M8": {"decisions": M8_DECISIONS, "batch_size": NLL_BATCH_SIZE},
            "M9": {"checkpoint_id": IDENTITY_CHECKPOINT_ID, "device": "cpu"},
        },
        "order": list(UNIT_ORDER),
        "priors": _prior_mapping(PRIORS),
        "priors_mps_side": _prior_mapping(PRIORS_MPS_SIDE),
        "mps_only": list(MPS_ONLY),
        "prior_seconds_per_step_b32": {
            f"{device}-{threads}": seconds
            for (device, threads), seconds in PRIOR_SECONDS_PER_STEP_B32.items()
        },
        "prior_fan_seconds": {"intercept": intercept, "slope": slope},
        "rules": {
            "mps_required_speedup": MPS_REQUIRED_SPEEDUP,
            "eight_threads_required_speedup": EIGHT_THREADS_REQUIRED_SPEEDUP,
            "peak_limit_bytes": M2_PEAK_LIMIT_BYTES,
            "swap_growth_tolerance_bytes": SWAP_GROWTH_TOLERANCE_BYTES,
            "mps_watermark_ratio": DEFAULT_MPS_WATERMARK_RATIO,
            "mps_checkpointing": "M3, the MPS batch 32 of M1 and M4 run with activation "
            "checkpointing from their first attempt when the chosen CPU configuration's batch "
            "32 does not fit as is and M2 passed there; an MPS unit that ends memory_abort "
            "without it is retried once with it, once M2's gradient bit-identity passed (run "
            "first when needed) and the retry's prior fits the budget; the MPS units after it, "
            "from that batch up, run with it; both attempts are recorded; a retry is charged to "
            f"{RETRIES}",
            "device_comparison": "the sustained MPS step time (M4, with activation checkpointing "
            "when it ran with it) against the fastest CPU configuration that runs batch 32: as "
            "is where it fits, with activation checkpointing where M2 passed (the times as is "
            "when none does)",
            "budget": "the run's limits are not part of this plan: the record states them "
            "(spend, approval), the step's own below gate 3 or the CPU-hours the owner approved; "
            "refused before the start above the limits; a measurement or retry whose prior "
            "would pass a limit is not started (M4 and its retry: unless their whole lifetime, "
            "seconds plus overhead_seconds, fits the remaining GPU-minutes); the run stops once "
            "the measured spend passes one",
            "child_deadline": "CPU child: remaining CPU-seconds / threads; MPS child: the "
            "smaller of remaining GPU-minutes x 60 and remaining CPU-seconds / 2; M4: remaining "
            "GPU-minutes x 60, its CPU time watched; parent timeout: deadline + "
            f"{CHILD_TIMEOUT_MARGIN_SECONDS:g} s",
        },
    }


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return float(value)


def estimate_plan(plan: Mapping[str, Any], *, mps_available: bool) -> dict[str, Any]:
    """The prior estimate of ``plan``: CPU-seconds and GPU-minutes per measurement and in all.

    Without MPS the MPS-only measurements (``plan["mps_only"]``: M3, M4, M7) and the MPS side
    of M1 and M2 (``plan["priors_mps_side"]``) are left out.
    """
    priors = plan["priors"]
    side = plan.get("priors_mps_side") or {}
    mps_only = set(plan.get("mps_only") or ())
    per: dict[str, dict[str, float]] = {}
    order = [name for name in (*MEASUREMENTS, "loads") if name in priors]
    order += sorted(name for name in priors if name not in order)
    for name in order:
        cpu = _number(priors[name]["cpu_seconds"], f"prior {name} cpu_seconds")
        gpu = _number(priors[name]["gpu_minutes"], f"prior {name} gpu_minutes")
        if not mps_available:
            if name in mps_only:
                cpu = gpu = 0.0
            elif name in side:
                cpu = max(0.0, cpu - float(side[name]["cpu_seconds"]))
                gpu = max(0.0, gpu - float(side[name]["gpu_minutes"]))
        per[name] = {"cpu_seconds": cpu, "gpu_minutes": gpu}
    return {
        "mps_available": bool(mps_available),
        "cpu_seconds": sum(item["cpu_seconds"] for item in per.values()),
        "gpu_minutes": sum(item["gpu_minutes"] for item in per.values()),
        "per_measurement": per,
    }


def timing_estimate() -> dict[str, Any]:
    """:func:`estimate_plan` of :func:`timing_plan` on this machine (configured runtime)."""
    return estimate_plan(timing_plan(), mps_available=_mps_available())


@dataclass(frozen=True)
class TimingBudget:
    """How far one run of the measurement step may spend: its limits, which are not part of the
    plan (:func:`timing_plan`), so a budget change is not a changed study.

    Without the owner's approval they are the step's own, :data:`TIMING_MAX_CPU_SECONDS` and
    :data:`TIMING_MAX_GPU_MINUTES`, below gate 3. With ``finetune timing --approved-by-owner
    CPU_HOURS`` the CPU limit is ``CPU_HOURS x 3600`` CPU-seconds, the ceiling the owner approved
    in chat for this measurement (gate 3; D-055), and ``given_at`` is when the flag was given
    (the command's start, ``YYYY-MM-DDTHH:MM:SSZ``); the GPU limit stays. The run works against
    :attr:`stop_cpu_seconds`, so that its spend never passes an approved ceiling. A CPU limit
    above gate 3 without an approval, or a GPU limit above it, is a ``ValueError``. Built by
    :func:`timing_budget`.
    """

    cpu_seconds: float
    gpu_minutes: float
    approved_cpu_hours: float | None = None
    given_at: str | None = None

    def __post_init__(self) -> None:
        cpu = _number(self.cpu_seconds, "the budget's cpu_seconds")
        if _number(self.gpu_minutes, "the budget's gpu_minutes") > GATE3_GPU_MINUTES:
            raise ValueError(f"a GPU limit above gate 3 ({GATE3_GPU_MINUTES:g} GPU-minutes)")
        hours = self.approved_cpu_hours
        if hours is None:
            if self.given_at is not None or cpu > GATE3_CPU_SECONDS:
                raise ValueError(
                    f"a CPU limit above gate 3 ({GATE3_CPU_SECONDS} CPU-seconds) needs the "
                    "CPU-hours the owner approved and the time the flag was given"
                )
            return
        if (
            _number(hours, "the approved CPU-hours") <= 0
            or cpu != hours * SECONDS_PER_HOUR
            or not isinstance(self.given_at, str)
        ):
            raise ValueError(
                "an approved CPU limit is the approved CPU-hours x 3600 CPU-seconds, with the "
                "time the flag was given"
            )

    @property
    def above_gate3(self) -> bool:
        """Whether the CPU limit passes gate 3 (only under the owner's approval)."""
        return self.cpu_seconds > GATE3_CPU_SECONDS

    @property
    def reserve_cpu_seconds(self) -> float:
        """What the run keeps below its CPU limit: :data:`APPROVAL_RESERVE_CPU_SECONDS` below an
        approved ceiling; 0 below the step's own limit, which gate 3 is that far above."""
        return 0.0 if self.approved_cpu_hours is None else APPROVAL_RESERVE_CPU_SECONDS

    @property
    def stop_cpu_seconds(self) -> float:
        """The CPU limit the run works against (its stop limit): the estimate must fit it, a
        unit or retry whose prior would pass it does not start, the child deadlines and the
        watch of M4 follow it, and the run stops once its measured spend passes it. The CPU limit
        minus :attr:`reserve_cpu_seconds`, never below 0."""
        return max(0.0, self.cpu_seconds - self.reserve_cpu_seconds)


def timing_budget(
    approved_cpu_hours: float | None = None, *, given_at: datetime | None = None
) -> TimingBudget:
    """The budget of one run: the step's own limits, or with ``approved_cpu_hours`` (the figure
    given with ``--approved-by-owner``, a finite number > 0) a CPU limit of that many CPU-hours,
    given at ``given_at`` (an aware datetime: the command's start, ignored without a figure). A
    malformed figure or time is a ``ValueError``."""
    if approved_cpu_hours is None:
        return TimingBudget(TIMING_MAX_CPU_SECONDS, TIMING_MAX_GPU_MINUTES)
    if (
        isinstance(approved_cpu_hours, bool)
        or not isinstance(approved_cpu_hours, (int, float))
        or not math.isfinite(approved_cpu_hours)
        or approved_cpu_hours <= 0
    ):
        raise ValueError(
            f"the approved CPU-hours must be a finite number > 0, got {approved_cpu_hours!r}"
        )
    if not isinstance(given_at, datetime) or given_at.tzinfo is None:
        raise ValueError("an approval needs given_at, the command's start (an aware datetime)")
    hours = float(approved_cpu_hours)
    return TimingBudget(
        cpu_seconds=hours * SECONDS_PER_HOUR,
        gpu_minutes=TIMING_MAX_GPU_MINUTES,
        approved_cpu_hours=hours,
        given_at=given_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _gate3_text() -> str:
    return f"gate 3 ({GATE3_CPU_SECONDS} CPU-seconds, {GATE3_GPU_MINUTES:g} GPU-minutes)"


def _ceiling_text(stop_cpu_seconds: float, reserve_cpu_seconds: float) -> str:
    """How the run keeps within an approved ceiling, in words."""
    return (
        "a ceiling the spend never passes: the run is planned and stopped against its stop limit "
        f"of {stop_cpu_seconds:.0f} CPU-seconds, a reserve of {reserve_cpu_seconds:.0f} below it "
        "for the unit running then"
    )


def _budget_text(budget: TimingBudget) -> str:
    """The limits of ``budget`` in words: where the CPU limit comes from, gate 3 and, under an
    approval, the stop limit."""
    limits = f"{budget.cpu_seconds:.0f} CPU-seconds and {budget.gpu_minutes:g} GPU-minutes"
    if budget.approved_cpu_hours is None:
        return f"{limits}, below {_gate3_text()}"
    if budget.above_gate3:
        where = f"above {_gate3_text()} under that approval"
    else:
        where = f"within {_gate3_text()}"
    return (
        f"{limits}: the {budget.approved_cpu_hours:g} CPU-hours the owner approved "
        f"(--approved-by-owner given at {budget.given_at}), {where}; "
        + _ceiling_text(budget.stop_cpu_seconds, budget.reserve_cpu_seconds)
    )


def _stop_limit_text(budget: TimingBudget) -> str:
    """The CPU limit a run stops at, in words (the step's own limit, or the stop limit below an
    approved ceiling)."""
    if budget.approved_cpu_hours is None:
        return f"the {budget.cpu_seconds:.0f} of the budget"
    return (
        f"the stop limit of {budget.stop_cpu_seconds:.0f} (the {budget.cpu_seconds:.0f} the owner "
        f"approved minus a reserve of {budget.reserve_cpu_seconds:.0f})"
    )


def _approval_record(budget: TimingBudget, estimate: Mapping[str, Any]) -> dict[str, Any] | None:
    """What a run records about ``--approved-by-owner`` (``None`` without the flag): the
    CPU-hours the owner approved, when the flag was given (the command's start), the CPU limit
    they give against the step's own and gate 3, the reserve and stop limit that keep the spend
    within it, and the estimate shown first."""
    if budget.approved_cpu_hours is None:
        return None
    return {
        "approved_cpu_hours": budget.approved_cpu_hours,
        "given_at": budget.given_at,
        "limit_cpu_seconds": budget.cpu_seconds,
        "reserve_cpu_seconds": budget.reserve_cpu_seconds,
        "stop_cpu_seconds": budget.stop_cpu_seconds,
        "own_limit_cpu_seconds": TIMING_MAX_CPU_SECONDS,
        "gate3_cpu_seconds": GATE3_CPU_SECONDS,
        "above_gate3": budget.above_gate3,
        "estimate_cpu_seconds": float(estimate["cpu_seconds"]),
        "estimate_gpu_minutes": float(estimate["gpu_minutes"]),
    }


def _approval_text(approval: Mapping[str, Any]) -> str:
    """An approval record in words (printed at the start and in the summary, and in the
    Markdown)."""
    gate3 = approval["gate3_cpu_seconds"]
    where = (
        f"above the gate-3 limit ({gate3} CPU-seconds) under the owner's approval"
        if approval["above_gate3"]
        else f"within the gate-3 limit ({gate3} CPU-seconds)"
    )
    return (
        f"the owner approved {approval['approved_cpu_hours']:g} CPU-hours (--approved-by-owner "
        f"given at {approval['given_at']}, the command's start): a CPU limit of "
        f"{approval['limit_cpu_seconds']:.0f} CPU-seconds instead of the step's own "
        f"{approval['own_limit_cpu_seconds']:.0f}, {where}, "
        + _ceiling_text(approval["stop_cpu_seconds"], approval["reserve_cpu_seconds"])
        + f"; estimate about {approval['estimate_cpu_seconds']:.0f} CPU-seconds and "
        f"{approval['estimate_gpu_minutes']:.1f} GPU-minutes"
    )


def require_within_budget(estimate: Mapping[str, Any], budget: TimingBudget | None = None) -> None:
    """Refuse (:class:`TimingBudgetExceeded`) an estimate above the stop limit or the GPU limit
    of ``budget`` (default: the step's own, :data:`TIMING_MAX_CPU_SECONDS` and
    :data:`TIMING_MAX_GPU_MINUTES`); exactly a limit passes. With the owner's approval the stop
    limit is the approved ceiling minus :data:`APPROVAL_RESERVE_CPU_SECONDS`, so the ceiling has
    no tolerance. A malformed estimate is a ``ValueError``."""
    budget = timing_budget() if budget is None else budget
    cpu = _number(estimate.get("cpu_seconds"), "estimate cpu_seconds")
    gpu = _number(estimate.get("gpu_minutes"), "estimate gpu_minutes")
    if cpu <= budget.stop_cpu_seconds and gpu <= budget.gpu_minutes:
        return
    shown = (
        f"the measurement step's estimate is about {cpu:.0f} CPU-seconds and {gpu:.1f} GPU-minutes"
    )
    if budget.approved_cpu_hours is None:
        raise TimingBudgetExceeded(
            f"{shown}, above its budget of {budget.cpu_seconds:.0f} CPU-seconds and "
            f"{budget.gpu_minutes:g} GPU-minutes, which keeps it below {_gate3_text()}; "
            "nothing ran"
        )
    raise TimingBudgetExceeded(
        f"{shown}, which does not fit its budget of {_budget_text(budget)}: the owner's approval "
        "does not cover it; nothing ran"
    )


def estimate_lines(estimate: Mapping[str, Any], budget: TimingBudget | None = None) -> list[str]:
    """The printed form of an estimate: the totals against the budget (default: the step's
    own; an approved ceiling is printed as such), one line per item."""
    budget = timing_budget() if budget is None else budget
    mps = (
        "MPS available"
        if estimate.get("mps_available")
        else f"MPS not available: M3, M4, M7, the {RETRIES} and the MPS side of M1 and M2 left out"
    )
    lines = [
        (
            f"timing estimate (the priors of {RETRAINING_MD_RELPATH} section (a), M3 and the "
            f"{RETRIES} priced by their content, D-054; {mps}): about "
            f"{estimate['cpu_seconds']:.0f} CPU-seconds and {estimate['gpu_minutes']:.1f} "
            f"GPU-minutes; budget {_budget_text(budget)}"
        )
    ]
    cpu, gpu = estimate["cpu_seconds"], estimate["gpu_minutes"]
    if cpu > budget.stop_cpu_seconds or gpu > budget.gpu_minutes:
        lines.append("  above the budget: `finetune timing` refuses to start")
    for name, item in estimate["per_measurement"].items():
        note = " (one MPS retry with activation checkpointing, as M3)" if name == RETRIES else ""
        lines.append(
            f"  {name}: about {item['cpu_seconds']:.0f} CPU-seconds, "
            f"{item['gpu_minutes']:.1f} GPU-minutes{note}"
        )
    return lines


# -- small helpers ----------------------------------------------------------------------------


def _now_text() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number_or_none(value: object) -> float | None:
    return float(value) if _finite(value) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _max_known(*values: object) -> int | None:
    known = [number for number in (_int_or_none(value) for value in values) if number is not None]
    return max(known) if known else None


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _seconds_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"median": None, "min": None, "max": None, "values": []}
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "values": [float(value) for value in values],
    }


def _linear_fit(points: Sequence[tuple[float, float]]) -> dict[str, float] | None:
    """Least squares ``y = intercept + slope x`` (``None`` below two distinct ``x``)."""
    if len({x for x, _ in points}) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
    return {"intercept": mean_y - slope * mean_x, "slope": slope}


def _public_text(value: object) -> str:
    """One line of ``value`` without any path, user or home directory (at most 300 chars)."""
    text = _PATH_TOKEN.sub("<path>", redact_home(str(value)))
    lines = text.strip().splitlines()
    return (lines[0] if lines else "")[:ERROR_TEXT_MAX]


def _public(value: Any) -> Any:
    """``value`` (nested mappings and lists) with every string passed through
    :func:`_public_text`."""
    if isinstance(value, str):
        return _public_text(value)
    if isinstance(value, Mapping):
        return {str(key): _public(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    return value


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": _public_text(exc)}


def _child_error(result: RunnerResult) -> dict[str, str] | None:
    error = (result.document or {}).get("error")
    if isinstance(error, Mapping):
        return {
            "type": _public_text(error.get("type") or "error"),
            "message": _public_text(error.get("message") or ""),
        }
    if result.status in ("killed", "no_result"):
        return {
            "type": result.status,
            "message": f"the child left no result (exit {result.exit_code})",
        }
    return None


def _swap_growth(before: Any, after: Any) -> int | None:
    if before is None or after is None:
        return None
    return max(0, int(after.swap_used_bytes) - int(before.swap_used_bytes))


def _gib(value: object) -> str:
    return f"{value / GIB:.2f}" if _finite(value) else "-"


def _timing_recipe(batch_size: int) -> Recipe:
    """R1 at ``batch_size``, with pseudo-epochs of :data:`TIMING_PSEUDO_EPOCH_STEPS` steps."""
    return replace(
        R1,
        name=f"{R1.name}-timing-b{batch_size}",
        batch_size=batch_size,
        pseudo_epoch_steps=TIMING_PSEUDO_EPOCH_STEPS,
    )


def _child_prior(device: str, threads: int, batch: int, steps: int) -> tuple[float, float]:
    """The prior of one training child: ``steps`` optimizer steps at the design's seconds per
    step at batch 32 scaled by the batch; CPU-seconds at its threads, an MPS child's wall as GPU
    time."""
    wall = steps * PRIOR_SECONDS_PER_STEP_B32[(device, threads)] * batch / RECIPE_BATCH
    return wall * threads, (wall / 60.0 if device == "mps" else 0.0)


def _fan_prior(length: int) -> float:
    intercept, slope = PRIOR_FAN_SECONDS
    return intercept + slope * length


def _fits(entry: Mapping[str, Any] | None) -> bool | None:
    """Whether a batch-32 configuration fits: completed with a peak of at most 9 GiB and no swap
    growth (False after a memory abort; ``None`` when unknown)."""
    if not entry:
        return None
    status = entry.get("status")
    if status == "memory_abort":
        return False
    if status != "completed" or not _finite(entry.get("peak_bytes")):
        return None
    growth = entry.get("swap_growth_bytes")
    swapped = _finite(growth) and growth > SWAP_GROWTH_TOLERANCE_BYTES
    return entry["peak_bytes"] <= M2_PEAK_LIMIT_BYTES and not swapped


def _checkpointed(entry: Mapping[str, Any] | None) -> bool:
    """Whether a training child ran with activation checkpointing (its last attempt)."""
    return bool(entry) and entry.get("activation_checkpointing") is True


def _m2_runs(measurements: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    m2 = measurements.get("M2")
    return list(m2.get("runs") or []) if isinstance(m2, Mapping) else []


def _checkpointing_valid(measurements: Mapping[str, Any]) -> bool | None:
    """Whether activation checkpointing passed M2's CPU gradient bit-identity (``None``: the
    check did not run or did not finish)."""
    m2 = measurements.get("M2")
    check = (m2.get("gradient_check") if isinstance(m2, Mapping) else None) or {}
    passed = check.get("passed")
    return passed if isinstance(passed, bool) else None


def _gradient_check_prior() -> tuple[float, float]:
    """The prior of M2's gradient check: two passes at batch 32 on the CPU at 2 threads."""
    return _child_prior("cpu", MPS_THREADS, RECIPE_BATCH, M2_GRADIENT_PASSES)


def _thermal(document: Mapping[str, Any], *, completed: bool) -> dict[str, Any]:
    """``theta`` of an M4 run: the median step time of the steps starting in the last
    :data:`M4_LAST_WINDOW_SECONDS` of its timed span over that of the first
    :data:`M4_FIRST_WINDOW_SECONDS`; ``None`` unless the run completed a span holding both
    windows."""
    times = document.get("step_times") or []
    offsets = document.get("step_offsets") or []
    pairs = [
        (float(offset), float(duration))
        for offset, duration in zip(offsets, times, strict=False)
        if _finite(offset) and _finite(duration)
    ]
    out: dict[str, Any] = {
        "theta": None,
        "median_first_seconds": None,
        "median_last_seconds": None,
        "median_seconds": _median([duration for _, duration in pairs]),
        "steps": len(pairs),
        "seconds": None,
        "first_window_steps": 0,
        "last_window_steps": 0,
        "theta_reason": None,
    }
    if not pairs:
        out["theta_reason"] = "no timed steps"
        return out
    end = max(offset + duration for offset, duration in pairs)
    first = [duration for offset, duration in pairs if offset < M4_FIRST_WINDOW_SECONDS]
    last = [duration for offset, duration in pairs if offset >= end - M4_LAST_WINDOW_SECONDS]
    out.update(
        seconds=end,
        median_first_seconds=_median(first),
        median_last_seconds=_median(last),
        first_window_steps=len(first),
        last_window_steps=len(last),
    )
    if not completed:
        out["theta_reason"] = "the run did not complete"
    elif end < M4_FIRST_WINDOW_SECONDS + M4_LAST_WINDOW_SECONDS or not first or not last:
        out["theta_reason"] = f"the timed span ({end:.0f} s) does not hold both windows"
    else:
        out["theta"] = out["median_last_seconds"] / out["median_first_seconds"]
    return out


def _fan_workers_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {"status": "completed"}
    for key in ("workers", "fans", "evaluation_length"):
        entry[key] = _int_or_none(result.get(key))
    for key in ("startup_seconds", "wall_seconds", "seconds_per_fan", "fans_per_second"):
        entry[key] = _number_or_none(result.get(key))
    entry["cpu_seconds"] = _number_or_none(result.get("cpu_seconds"))
    return entry


def _ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _environment(mps_available: bool) -> dict[str, Any]:
    import numpy
    import pandas

    info = require_configured()
    return {
        "torch_version": info.torch_version,
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "platform": default_label(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "mps_available": bool(mps_available),
        "threads": info.threads,
    }


def _code() -> dict[str, Any]:
    from ..kronos.vendored import source_manifest_path
    from ..provenance import forecasting_code_sha256
    from .train import training_code_sha256

    return {
        "training_code_sha256": training_code_sha256(),
        "forecasting_code_sha256": forecasting_code_sha256(),
        "vendored_manifest_sha256": sha256_file(source_manifest_path()),
    }


def _retraining_md_sha256() -> str | None:
    path = REPO_ROOT / RETRAINING_MD_RELPATH
    return sha256_file(path) if path.is_file() else None


def _error_classes() -> tuple[tuple[type[BaseException], ...], tuple[type[BaseException], ...]]:
    """``(environment, measurement)`` errors: the first stop the run (nothing after can work),
    the second fail one measurement; both are recorded and make the command exit 1."""
    from ..data.forecast_store import ForecastStoreError
    from ..data.partition import PartitionError
    from ..data.windows import WindowError
    from ..kronos.batch import BatchError
    from ..kronos.decision import DecisionError
    from ..kronos.vendored import VendoredTreeError
    from ..kronos.weights import WeightsError
    from ..manifests import ManifestMismatch
    from ..runtime import RuntimeConfigError
    from .export_csv import ExportError
    from .nll import NllError
    from .registry import RegistryError
    from .train import TrainError

    environment = (VendoredTreeError, ManifestMismatch, WeightsError, RuntimeConfigError, OSError)
    measurement = (
        TrainError,
        BatchError,
        NllError,
        RegistryError,
        ExportError,
        DecisionError,
        WindowError,
        ForecastStoreError,
        PartitionError,
        TimingError,
    )
    return environment, measurement


# -- one run ----------------------------------------------------------------------------------


@dataclass
class _Charge:
    """What one metered block costs: the larger of the measured and the declared CPU-seconds
    (minus what nested blocks charged themselves), and its GPU-minutes."""

    declared_cpu: float = 0.0
    wall_threads: int = 0  # > 0: declare at least wall x threads (D-033)
    excluded_cpu: float = 0.0
    gpu_minutes: float = 0.0
    measured_cpu: float = 0.0
    wall: float = 0.0
    charged_cpu: float = 0.0


class _Run:
    """One run of the measurement step: its budget, its record and the measurements in order."""

    def __init__(
        self,
        *,
        params: Params,
        steps: TimingSteps,
        root: Path,
        scratch: Path,
        estimate: Mapping[str, Any],
        budget: TimingBudget,
        mps_available: bool,
        say: Callable[[str], None],
    ) -> None:
        self.params = params
        self.steps = steps
        self.root = root
        self.scratch = scratch
        self.estimate = dict(estimate)
        self.budget = budget
        self.mps_available = mps_available
        self.say = say
        self.threads = params.kronos.threads
        self.cpu_seconds = 0.0
        self.gpu_minutes = 0.0
        self.spent: dict[str, dict[str, float]] = {
            name: {"cpu_seconds": 0.0, "gpu_minutes": 0.0} for name in SPEND_ITEMS
        }
        self.stop_reason: str | None = None
        self.failures: list[str] = []
        self.measurements: dict[str, Any] = {
            "M1": [],
            "M2": {"gradient_check": None, "runs": []},
            "M3": None,
            "M4": None,
            "M5": None,
            "M6": [],
            "M7": [],
            "M8": None,
            "M9": None,
        }
        # the smallest batch from which MPS units run with activation checkpointing, and why
        self.mps_checkpointing_from: int | None = None
        self.mps_checkpointing_reason: str | None = None
        self.fold: Fold | None = None
        self.export: ExportRecord | None = None
        self.models: Any = None
        self.samples: dict[str, Any] = {}
        self.environment_errors, self.measurement_errors = _error_classes()

    # budget -----------------------------------------------------------------------------------

    def _limit_reason(self, cpu: float, gpu: float, what: str) -> str | None:
        """Why ``cpu`` CPU-seconds or ``gpu`` GPU-minutes pass a limit the run works against
        (the stop limit, :attr:`TimingBudget.stop_cpu_seconds`, and the GPU limit); else
        ``None``."""
        budget = self.budget
        if cpu > budget.stop_cpu_seconds:
            return f"{what} {cpu:.0f} CPU-seconds, above {_stop_limit_text(budget)}"
        if gpu > budget.gpu_minutes:
            return f"{what} {gpu:.1f} GPU-minutes, above the {budget.gpu_minutes:g} of the budget"
        return None

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            self.say(f"STOP: {reason}; what follows is recorded as not run")

    def fail(self, message: str, *, stop: bool = False) -> None:
        self.failures.append(message)
        self.say(f"FAILED: {message}")
        if stop:
            self.stop(f"stopped after a failure ({message})")

    def may_start(
        self, label: str, prior: tuple[float, float], *, pending_cpu: float = 0.0
    ) -> str | None:
        """``None`` when ``label`` may start; else why not (a prior that would pass a limit
        stops the run)."""
        if self.stop_reason is not None:
            return "not run: the run stopped before it (spend.stop_reason)"
        cpu, gpu = prior
        reason = self._limit_reason(
            self.cpu_seconds + pending_cpu + cpu,
            self.gpu_minutes + gpu,
            f"{label} (prior about {cpu:.0f} CPU-seconds, {gpu:.1f} GPU-minutes) would bring the "
            "spend to about",
        )
        if reason is None:
            return None
        self.stop(reason)
        return "not run: its prior would pass the budget (spend.stop_reason)"

    def charge(self, measurement: str, cpu: float, gpu: float) -> None:
        self.cpu_seconds += cpu
        self.gpu_minutes += gpu
        self.spent[measurement]["cpu_seconds"] += cpu
        self.spent[measurement]["gpu_minutes"] += gpu
        reason = self._limit_reason(self.cpu_seconds, self.gpu_minutes, "the measured spend is")
        if reason is not None:
            self.stop(reason)

    def spend_text(self) -> str:
        return (
            f"spent {self.cpu_seconds:.0f} of {self.budget.cpu_seconds:.0f} CPU-seconds, "
            f"{self.gpu_minutes:.1f} of {self.budget.gpu_minutes:g} GPU-minutes"
        )

    @contextlib.contextmanager
    def measure(self, measurement: str) -> Iterator[_Charge]:
        """Meter a block and charge it to ``measurement`` when it ends (also on an error)."""
        charge = _Charge()
        wall_start = _clock()
        cpu_start = _cpu_clock()
        try:
            yield charge
        finally:
            charge.wall = max(0.0, _clock() - wall_start)
            charge.measured_cpu = max(0.0, _cpu_clock() - cpu_start - charge.excluded_cpu)
            declared = charge.declared_cpu
            if charge.wall_threads:
                declared = max(declared, charge.wall * charge.wall_threads)
            charge.charged_cpu = max(charge.measured_cpu, declared)
            self.charge(measurement, charge.charged_cpu, charge.gpu_minutes)

    @contextlib.contextmanager
    def guard(self, label: str, entry: dict[str, Any]) -> Iterator[None]:
        """Record a known error of a measurement in ``entry`` instead of raising it; an error
        of the environment also stops the run. Anything else propagates (a bug)."""
        try:
            yield
        except (TimingBudgetExceeded, StaleTimingError):
            raise
        except self.environment_errors as exc:
            entry.update(status="failed", error=_error(exc))
            self.fail(f"{label}: {type(exc).__name__}: {_public_text(exc)}", stop=True)
        except self.measurement_errors as exc:
            entry.update(status="failed", error=_error(exc))
            self.fail(f"{label}: {type(exc).__name__}: {_public_text(exc)}")

    # set-up -----------------------------------------------------------------------------------

    def setup(self) -> None:
        """The synthetic export (into the scratch) and the shipped models (in this process)."""
        from .export_csv import export_frame, synthetic_frame
        from .folds import synthetic_fold

        recipe = _timing_recipe(RECIPE_BATCH)
        start = to_ms(TIMING_CSV_START)
        end = start + TIMING_SYNTHETIC_ROWS * HOUR_MS
        with self.measure("loads"):
            self.fold = synthetic_fold(start, end, recipe, label=TIMING_FOLD_LABEL)
            frame = synthetic_frame(start, TIMING_SYNTHETIC_ROWS, seed=TIMING_SYNTHETIC_SEED)
            self.export = export_frame(frame, self.fold, recipe, self.scratch / "export")
            with contextlib.redirect_stdout(io.StringIO()):  # the Hub mixin prints while loading
                self.models = self.steps.load_models(self.params.kronos.model)
        export = self.export
        self.samples = {
            "market_data": "none: synthetic candles only",
            "synthetic_rows": export.rows,
            "csv_start": utc_text(self.fold.csv_start_ms),
            "csv_end": utc_text(self.fold.csv_end_ms),
            "fit_windows": export.fit_windows_kept,
            "val_windows": export.val_windows_kept,
            "nll_decisions_exported": len(export.nll_decisions_ms),
            "nll_decisions_timed": min(M8_DECISIONS, len(export.nll_decisions_ms)),
            "export_content_sha256": export.content_sha256,
        }
        self.say(
            f"synthetic export: {export.rows} hourly candles, {export.fit_windows_kept} fit and "
            f"{export.val_windows_kept} validation windows of {R1.window_rows} rows; shipped "
            f"{self.params.kronos.model} loaded; {self.spend_text()}"
        )

    def release_models(self) -> None:
        """Free this process's models before the training children (memory)."""
        self.models = None
        gc.collect()

    # M9, M8, M5 (this process) -----------------------------------------------------------------

    def m9(self) -> None:
        entry: dict[str, Any] = {"status": "not_run"}
        self.measurements["M9"] = entry
        reason = self.may_start("M9", PRIORS["M9"])
        if reason is not None:
            entry["reason"] = reason
            return
        self.say("M9 identity control: the shipped weights through the checkpoint path (CPU)")
        with self.guard("M9", entry), self.measure("M9") as charge:
            charge.wall_threads = self.threads
            result = self.steps.identity_control(
                self.params, scratch_root=self.scratch / "identity", models=self.models
            )
            identical = result.get("identical") is True
            entry.update(
                status="completed" if identical else "mismatch",
                identical=identical,
                checkpoint_id=_public(result.get("checkpoint_id")),
                shipped_config_id=_public(result.get("shipped_config_id")),
                checkpoint_config_id=_public(result.get("checkpoint_config_id")),
                paths_sha256=_public(dict(result.get("paths_sha256") or {})),
                files=_public(list(result.get("files") or [])),
                seconds=_number_or_none(result.get("seconds")),
            )
            if not identical:
                self.fail("M9: the checkpoint's fan differs from the shipped weights' fan")
        self.say(f"M9: {entry['status']}; {self.spend_text()}")

    def m8(self) -> None:
        from .nll import NLL_BATCH_SIZE, load_nll_inputs

        entry: dict[str, Any] = {"status": "not_run", "decisions": M8_DECISIONS}
        self.measurements["M8"] = entry
        reason = self.may_start("M8", PRIORS["M8"])
        if reason is not None:
            entry["reason"] = reason
            return
        assert self.export is not None
        with self.guard("M8", entry), self.measure("M8") as charge:
            charge.wall_threads = self.threads
            inputs = load_nll_inputs(
                self.export.nll_inputs_path, expected_sha256=self.export.nll_inputs_sha256
            )
            inputs = inputs.head(min(M8_DECISIONS, inputs.decisions))
            started = _clock()
            result = self.steps.nll4_tf(
                self.models.tokenizer,
                self.models.model,
                inputs,
                clip=R1.clip,
                batch_size=NLL_BATCH_SIZE,
            )
            seconds = _clock() - started
            per_decision = seconds / inputs.decisions
            planned = len(self.export.nll_decisions_ms)
            entry.update(
                status="completed",
                decisions=inputs.decisions,
                seconds=seconds,
                seconds_per_decision=per_decision,
                batch_size=NLL_BATCH_SIZE,
                threads=_int_or_none(getattr(result, "threads", None)) or self.threads,
                planned_decisions=planned,
                seconds_per_evaluation=per_decision * planned,
            )
        self.say(
            f"M8 NLL4-TF: {entry['status']}"
            + (
                f", {entry['seconds_per_decision']:.3f} s per decision"
                if entry["status"] == "completed"
                else ""
            )
            + f"; {self.spend_text()}"
        )

    def m5(self) -> None:
        entry: dict[str, Any] = {"status": "not_run", "repeats": M5_REPEATS, "per_length": []}
        self.measurements["M5"] = entry
        for length in M5_LENGTHS:
            item: dict[str, Any] = {
                "length": length,
                "status": "not_run",
                "prior_seconds_per_fan": _fan_prior(length),
            }
            entry["per_length"].append(item)
            prior = (M5_REPEATS + 1) * _fan_prior(length) * self.threads
            reason = self.may_start(f"M5 t({length})", (prior, 0.0))
            if reason is not None:
                item["reason"] = reason
                continue
            with self.guard(f"M5 t({length})", item), self.measure("M5") as charge:
                charge.wall_threads = self.threads
                seconds = self.steps.seconds_per_fan(
                    params=self.params,
                    models=self.models,
                    evaluation_length=length,
                    repeats=M5_REPEATS,
                )
                if not _finite(seconds) or seconds <= 0:
                    raise TimingError(f"t({length}) measured {seconds!r} s")
                item.update(status="completed", seconds_per_fan=float(seconds))
                self.say(f"M5 t({length}) = {seconds:.2f} s per fan; {self.spend_text()}")
        done = {
            item["length"]: item["seconds_per_fan"]
            for item in entry["per_length"]
            if item["status"] == "completed"
        }
        statuses = [item["status"] for item in entry["per_length"]]
        entry.update(
            status=next((status for status in statuses if status != "completed"), "completed"),
            t4=done.get(4),
            t8=done.get(8),
            t24=done.get(24),
            fit=_linear_fit([(float(length), seconds) for length, seconds in done.items()]),
        )

    # training children ------------------------------------------------------------------------

    def deterministic(self, device: str) -> bool:
        """CPU children run deterministic; MPS ones only if M3's deterministic check passed."""
        if device == "cpu":
            return True
        m3 = self.measurements["M3"] or {}
        return m3.get("deterministic_passed") is True

    def mps_reason(self) -> str | None:
        """``None`` when MPS may be timed (available, M3 passed); else why not."""
        if not self.mps_available:
            return "not run: MPS is not available on this machine"
        m3 = self.measurements["M3"] or {}
        if m3.get("status") in (None, "not_run"):
            return "not run: the MPS preflight (M3) did not run"
        if m3.get("passed") is not True:
            return "not run: the MPS preflight (M3) did not pass"
        return None

    def child_deadline(self, device: str, threads: int) -> float:
        """The wall seconds a training child may run: the CPU-seconds left below the stop limit
        at its threads, on MPS also at most the GPU-minutes left."""
        cpu = (self.budget.stop_cpu_seconds - self.cpu_seconds) / threads
        if device == "mps":
            return min((self.budget.gpu_minutes - self.gpu_minutes) * 60.0, cpu)
        return cpu

    def record_child(
        self,
        label: str,
        entry: dict[str, Any],
        result: RunnerResult,
        *,
        device: str,
        batch: int,
        wall: float,
        before: Any,
        after: Any,
        budget_stop: bool = False,
    ) -> None:
        """Fill ``entry`` from a child's result; a deadline or timeout stops the run (the
        deadline is what the budget allowed), a crash is a failure."""
        document = dict(result.document or {})
        memory = document.get("memory") or {}
        status = result.status
        if budget_stop or result.timed_out or status == "time_abort":
            recorded = "stopped"
        elif status in ("completed", "memory_abort"):
            recorded = status
        else:
            recorded = "failed"
        rss = _int_or_none(memory.get("peak_rss_bytes"))
        mps = _int_or_none(memory.get("peak_mps_driver_bytes"))
        driver = (mps or 0) if device == "mps" else 0  # unified memory: the GPU's share too
        peak = None if rss is None else rss + driver
        times = [float(value) for value in document.get("step_times") or [] if _finite(value)]
        median = _median(times)
        entry.update(
            status=recorded,
            run_id=_public_text(result.run_id),
            child_status=status,
            stop_reason=_public(document.get("stop_reason")),
            wall_seconds=wall,
            cpu_seconds=_number_or_none(document.get("cpu_seconds")),
            peak_rss_bytes=rss,
            peak_mps_driver_bytes=mps,
            peak_bytes=peak,
            swap_growth_bytes=_max_known(
                memory.get("swap_growth_bytes"), _swap_growth(before, after)
            ),
            max_pressure_level=_max_known(
                memory.get("max_pressure_level"),
                None if after is None else after.pressure_level,
            ),
            error=_child_error(result),
        )
        if result.mode == "timing":
            entry.update(
                timed_steps=len(times),
                seconds_per_step=_seconds_summary(times),
                windows_per_second=(batch / median) if median else None,
            )
        if recorded == "stopped" and not budget_stop:
            self.stop(f"{label} reached the time the remaining budget allowed it ({status})")
        elif recorded == "failed":
            self.fail(f"{label}: the training child ended {status} (exit {result.exit_code})")

    def run_child(
        self,
        measurement: str,
        label: str,
        entry: dict[str, Any],
        *,
        mode: str,
        device: str,
        threads: int,
        batch: int,
        checkpointing: bool = False,
        timed_steps: int | None = None,
        checks: tuple[str, ...] = (),
        deterministic: bool | None = None,
    ) -> RunnerResult | None:
        """One training child through ``steps.run_child``, metered and recorded in ``entry``."""
        from .train import Compute, RunRequest, TimingSpec, Wrappers

        deterministic = self.deterministic(device) if deterministic is None else deterministic
        deadline = self.child_deadline(device, threads)
        entry.update(
            deterministic=deterministic,
            activation_checkpointing=checkpointing,
            deadline_seconds=deadline,
        )
        if checkpointing and device == "mps" and self.mps_checkpointing_reason is not None:
            entry["checkpointing_reason"] = self.mps_checkpointing_reason
        result: RunnerResult | None = None
        with self.guard(label, entry), self.measure(measurement) as charge:
            request = RunRequest(
                mode=mode,
                recipe=_timing_recipe(batch),
                fold=self.fold,
                export=self.export,
                compute=Compute(
                    device=device,
                    threads=threads,
                    deterministic=deterministic,
                    activation_checkpointing=checkpointing,
                    memory_guard=True,
                ),
                wrappers=Wrappers(early_stopping=False),
                timing=(
                    None
                    if timed_steps is None
                    else TimingSpec(warmup_steps=M1_WARMUP_STEPS, timed_steps=timed_steps)
                ),
                preflight_checks=checks,
                deadline_seconds=deadline,
            )
            before = _read_memory()
            started = _clock()
            result = self.steps.run_child(
                request, root=self.root, timeout_seconds=deadline + CHILD_TIMEOUT_MARGIN_SECONDS
            )
            wall = _clock() - started
            charge.declared_cpu = _number_or_none(result.document.get("cpu_seconds")) or 0.0
            if device == "mps":
                charge.gpu_minutes = wall / 60.0
            self.record_child(
                label,
                entry,
                result,
                device=device,
                batch=batch,
                wall=wall,
                before=before,
                after=_read_memory(),
            )
        return result

    # activation checkpointing on MPS and the retry ----------------------------------------------

    def note_mps_checkpointing(self, batch: int, reason: str) -> None:
        """MPS units from ``batch`` up run with activation checkpointing from now on."""
        if self.mps_checkpointing_from is None or batch < self.mps_checkpointing_from:
            self.mps_checkpointing_from = batch
            self.mps_checkpointing_reason = reason

    def mps_starts_with_checkpointing(self, batch: int) -> bool:
        return self.mps_checkpointing_from is not None and batch >= self.mps_checkpointing_from

    def checkpointing_from_cpu(self) -> None:
        """Before any MPS unit: when the chosen CPU configuration's batch 32 does not fit as is
        (M1) and activation checkpointing passed there (M2), the MPS units at batch 32 (M3, M1,
        M4) run with it from their first attempt."""
        choice = choose_compute(self.measurements)
        threads = choice["threads"]
        if self.stop_reason is not None or choice["device"] != "cpu" or threads is None:
            return
        if _recipe_batch(self.measurements, "cpu", threads)["checkpointing"] is not True:
            return
        self.note_mps_checkpointing(
            RECIPE_BATCH,
            f"the batch 32 of CPU x{threads} does not fit as is (M1) and activation "
            "checkpointing passed there (M2)",
        )
        self.say(f"MPS at batch 32 with activation checkpointing: {self.mps_checkpointing_reason}")

    def gradient_check(self) -> None:
        """M2's CPU gradient bit-identity of one step with and without activation checkpointing,
        once (charged to M2); nothing when it ran already or the run stopped."""
        m2 = self.measurements["M2"]
        if m2["gradient_check"] is not None or self.stop_reason is not None:
            return
        entry: dict[str, Any] = {"device": "cpu", "threads": MPS_THREADS, "status": "not_run"}
        m2["gradient_check"] = entry
        result = self.run_child(
            "M2",
            "M2 CPU gradient bit-identity",
            entry,
            mode="preflight",
            device="cpu",
            threads=MPS_THREADS,
            batch=RECIPE_BATCH,
            checks=M2_GRADIENT_CHECKS,
        )
        checks = _public(dict((result.document if result else {}).get("preflight") or {}))
        entry.update(
            checks=checks,
            passed=entry["status"] == "completed"
            and (checks.get("checkpoint_gradients") or {}).get("passed") is True,
        )
        self.say(
            f"M2 CPU gradient bit-identity: {'passed' if entry['passed'] else 'not passed'}; "
            f"{self.spend_text()}"
        )

    def retry_with_checkpointing(
        self,
        label: str,
        entry: dict[str, Any],
        base: Mapping[str, Any],
        *,
        batch: int,
        prior: tuple[float, float],
    ) -> bool:
        """After the MPS unit ``label`` ended memory_abort without activation checkpointing:
        whether it is retried once with it. A retry needs M2's gradient bit-identity (run first
        when it has not run) and its prior within the budget; ``entry`` then holds ``base``
        again with the aborted attempt under ``first_attempt``, and the MPS units after it,
        from ``batch`` up, run with activation checkpointing. Otherwise ``entry["retry"]`` says
        why it was not retried."""
        aborted = entry.get("status") == "memory_abort"
        if not aborted or entry.get("activation_checkpointing") is not False:
            return False
        invalid = "not retried: activation checkpointing failed M2's CPU gradient bit-identity"
        valid = _checkpointing_valid(self.measurements)
        why = invalid if valid is False else None
        if why is None:
            cpu, gpu = prior
            if valid is None:
                cpu += _gradient_check_prior()[0]
            refused = self.may_start(f"{label} retried with activation checkpointing", (cpu, gpu))
            if refused is not None:
                why = refused.replace("not run:", "not retried:", 1)
            elif valid is None:
                self.gradient_check()
                if self.stop_reason is not None:
                    why = "not retried: the run stopped before it (spend.stop_reason)"
                elif _checkpointing_valid(self.measurements) is not True:
                    why = invalid
        if why is not None:
            entry["retry"] = {"status": "not_run", "reason": why}
            self.say(f"{label}: memory abort without activation checkpointing; {why}")
            return False
        first = dict(entry)
        entry.clear()
        entry.update(copy.deepcopy(dict(base)))
        entry["first_attempt"] = first
        self.note_mps_checkpointing(batch, f"{label} ended memory_abort without it")
        self.say(f"{label}: memory abort without activation checkpointing; retried once with it")
        return True

    # M3, M1, M2 ------------------------------------------------------------------------------

    def m3(self) -> None:
        base: dict[str, Any] = {
            "status": "not_run",
            "device": "mps",
            "threads": MPS_THREADS,
            "batch_size": RECIPE_BATCH,
            "requested": list(M3_CHECKS),
            "checks": {},
            "passed": False,
            "deterministic_passed": False,
        }
        entry = copy.deepcopy(base)
        self.measurements["M3"] = entry
        reason = (
            "not run: MPS is not available on this machine"
            if not self.mps_available
            else self.may_start("M3", PRIORS["M3"])
        )
        if reason is not None:
            entry["reason"] = reason
            return
        checkpointing = self.mps_starts_with_checkpointing(RECIPE_BATCH)
        self.say(
            "M3 MPS preflight at batch 32 (before any MPS timing)"
            + (", with activation checkpointing" if checkpointing else "")
        )
        self.m3_attempt(entry, checkpointing, "M3")
        if self.retry_with_checkpointing(
            "M3 MPS preflight", entry, base, batch=RECIPE_BATCH, prior=PRIORS["M3"]
        ):
            self.m3_attempt(entry, True, RETRIES)
        self.say(
            f"M3: {entry['status']}, preflight {'passed' if entry['passed'] else 'not passed'}; "
            f"{self.spend_text()}"
        )

    def m3_attempt(self, entry: dict[str, Any], checkpointing: bool, measurement: str) -> None:
        """One M3 child (charged to ``measurement``) and what its checks decide."""
        result = self.run_child(
            measurement,
            "M3 MPS preflight",
            entry,
            mode="preflight",
            device="mps",
            threads=MPS_THREADS,
            batch=RECIPE_BATCH,
            checkpointing=checkpointing,
            checks=M3_CHECKS,
            deterministic=False,  # check (d) asks deterministic mode itself
        )
        checks = _public(dict((result.document if result else {}).get("preflight") or {}))
        deterministic = checks.get("deterministic") or {}
        passed = entry["status"] == "completed" and all(
            (checks.get(name) or {}).get("passed") is True for name in M3_REQUIRED_CHECKS
        )
        entry.update(
            checks=checks,
            passed=passed,
            deterministic_passed=deterministic.get("passed") is True,
            deterministic_refused=deterministic.get("refused") is True,
            deterministic_op=deterministic.get("op"),
        )

    def m1(self, device: str) -> None:
        """M1 on ``device``'s configurations; on MPS a memory abort without activation
        checkpointing is retried once with it."""
        for config_device, threads in M1_CONFIGS:
            if config_device != device:
                continue
            skip = self.mps_reason() if device == "mps" else None
            for batch in M1_BATCHES:
                label = f"M1 {device} x{threads} batch {batch}"
                base: dict[str, Any] = {
                    "device": device,
                    "threads": threads,
                    "batch_size": batch,
                    "status": "not_run",
                    "warmup_steps": M1_WARMUP_STEPS,
                    "timed_steps": 0,
                }
                entry = dict(base)
                self.measurements["M1"].append(entry)
                prior = _child_prior(device, threads, batch, M1_WARMUP_STEPS + M1_TIMED_STEPS)
                reason = skip or self.may_start(label, prior)
                if reason is not None:
                    entry["reason"] = reason
                    continue
                checkpointing = device == "mps" and self.mps_starts_with_checkpointing(batch)
                self.m1_attempt(label, entry, checkpointing, "M1")
                if device == "mps" and self.retry_with_checkpointing(
                    label, entry, base, batch=batch, prior=prior
                ):
                    self.m1_attempt(label, entry, True, RETRIES)
                if entry["status"] != "completed":
                    skip = f"not run: batch {batch} of {device} x{threads} ended {entry['status']}"

    def m1_attempt(
        self, label: str, entry: dict[str, Any], checkpointing: bool, measurement: str
    ) -> None:
        """One M1 child (charged to ``measurement``)."""
        self.run_child(
            measurement,
            label,
            entry,
            mode="timing",
            device=entry["device"],
            threads=entry["threads"],
            batch=entry["batch_size"],
            checkpointing=checkpointing,
            timed_steps=M1_TIMED_STEPS,
        )
        median = (entry.get("seconds_per_step") or {}).get("median")
        self.say(
            f"{label}{' with activation checkpointing' if checkpointing else ''}: "
            f"{entry['status']}"
            + (f", {median:.3f} s per step" if median else "")
            + f", peak {_gib(entry.get('peak_bytes'))} GiB; {self.spend_text()}"
        )

    def m1_entry(self, device: str, threads: int, batch: int) -> dict[str, Any] | None:
        for entry in self.measurements["M1"]:
            if (entry["device"], entry["threads"], entry["batch_size"]) == (device, threads, batch):
                return entry
        return None

    def m2_if_needed(self, device: str) -> None:
        """M2 on ``device`` when its batch 32 ran without activation checkpointing and does not
        fit: on the CPU configuration chosen so far, before any MPS unit; on MPS (if MPS may be
        timed) when its batch 32 completed but does not fit, a memory abort there being retried
        in M1 instead."""
        if device == "mps":
            if self.mps_reason() is not None:
                return
            threads = MPS_THREADS
        else:
            choice = choose_compute(self.measurements)
            if choice["device"] != "cpu" or choice["threads"] is None:
                return
            threads = choice["threads"]
        entry = self.m1_entry(device, threads, RECIPE_BATCH)
        if entry is None or entry.get("activation_checkpointing") is not False:
            return
        if _fits(entry) is not False or (device == "mps" and entry["status"] == "memory_abort"):
            return
        if any(item["device"] == device for item in self.measurements["M2"]["runs"]):
            return
        self.m2(device, threads, entry)

    def m2(self, device: str, threads: int, b32: Mapping[str, Any]) -> None:
        runs = self.measurements["M2"]["runs"]
        why = (
            "batch 32 ended memory_abort"
            if b32.get("status") == "memory_abort"
            else f"batch 32 peak {_gib(b32.get('peak_bytes'))} GiB, swap growth "
            f"{_gib(b32.get('swap_growth_bytes'))} GiB"
        )
        entry: dict[str, Any] = {
            "device": device,
            "threads": threads,
            "batch_size": RECIPE_BATCH,
            "status": "not_run",
            "why": why,
            "timing": None,
            "passed": False,
        }
        runs.append(entry)
        cpu, gpu = PRIORS["M2"]
        reason = self.may_start(f"M2 {device}", (cpu, gpu if device == "mps" else 0.0))
        if reason is not None:
            entry["reason"] = reason
            return
        self.say(f"M2 activation checkpointing on {device} x{threads} at batch 32 ({why})")
        timing_entry: dict[str, Any] = {
            "device": device,
            "threads": threads,
            "batch_size": RECIPE_BATCH,
            "status": "not_run",
            "warmup_steps": M1_WARMUP_STEPS,
            "timed_steps": 0,
        }
        entry["timing"] = timing_entry
        self.run_child(
            "M2",
            f"M2 {device} x{threads} batch 32 with checkpointing",
            timing_entry,
            mode="timing",
            device=device,
            threads=threads,
            batch=RECIPE_BATCH,
            checkpointing=True,
            timed_steps=M1_TIMED_STEPS,
        )
        self.gradient_check()
        gradient = self.measurements["M2"]["gradient_check"] or {}
        statuses = [timing_entry["status"], gradient.get("status", "not_run")]
        entry["status"] = next((item for item in statuses if item != "completed"), "completed")
        entry["passed"] = (
            entry["status"] == "completed"
            and _fits(timing_entry) is True
            and gradient.get("passed") is True
        )
        if device == "mps" and entry["passed"]:
            self.note_mps_checkpointing(
                RECIPE_BATCH, "the MPS batch 32 does not fit as is (M1) and M2 passed there"
            )
        self.say(
            f"M2 {device}: {'passed' if entry['passed'] else 'not passed'}; {self.spend_text()}"
        )

    def mps_checkpointing(self) -> bool | None:
        """How M4 runs batch 32 on MPS (M1, M2): False as is, True with activation
        checkpointing, ``None`` when neither fits (M4 cannot run at the recipe batch)."""
        return _recipe_batch(self.measurements, "mps", MPS_THREADS)["checkpointing"]

    # M4 with M7 --------------------------------------------------------------------------------

    def m7_not_run(self, reason: str) -> None:
        listed = {item["requested_workers"] for item in self.measurements["M7"]}
        for workers in M7_WORKERS:
            if workers not in listed:
                self.measurements["M7"].append(
                    {
                        "requested_workers": workers,
                        "fans": FANS_PER_WORKER_TEST,
                        "status": "not_run",
                        "reason": reason,
                    }
                )

    def m4(self) -> None:
        base: dict[str, Any] = {
            "status": "not_run",
            "device": "mps",
            "threads": MPS_THREADS,
            "batch_size": RECIPE_BATCH,
            "warmup_steps": M1_WARMUP_STEPS,
            "timed_seconds": M4_SECONDS,
            "theta": None,
        }
        entry = dict(base)
        self.measurements["M4"] = entry
        reason = self.mps_reason()
        checkpointing = None if reason is not None else self.mps_checkpointing()
        if reason is None and checkpointing is None:
            reason = (
                "not run: batch 32 did not complete on MPS in M1 or does not fit there, and "
                "activation checkpointing (M2) did not pass; batch 16 needs the owner"
            )
        prior = (PRIORS["M4"][0] + PRIORS["M7"][0], PRIORS["M4"][1] + PRIORS["M7"][1])
        if reason is None:
            reason = self.may_start("M4 with M7", prior)
        if reason is not None:
            entry["reason"] = reason
            self.m7_not_run("not run: M7 runs during M4, which did not run")
            return
        self.m4_attempt(entry, bool(checkpointing), "M4")
        if self.retry_with_checkpointing("M4", entry, base, batch=RECIPE_BATCH, prior=prior):
            entry["first_attempt"]["m7"] = self.measurements["M7"]  # M7 runs again in the retry
            self.measurements["M7"] = []
            self.m4_attempt(entry, True, RETRIES)

    def m4_attempt(self, entry: dict[str, Any], checkpointing: bool, measurement: str) -> None:
        """One M4 child with M7 during it, charged to ``measurement`` (M7 to M7)."""
        from .train import Compute, RunRequest, TimingSpec, Wrappers

        deterministic = self.deterministic("mps")
        deadline = (self.budget.gpu_minutes - self.gpu_minutes) * 60.0
        entry.update(
            deterministic=deterministic,
            activation_checkpointing=checkpointing,
            deadline_seconds=deadline,
        )
        if checkpointing and self.mps_checkpointing_reason is not None:
            entry["checkpointing_reason"] = self.mps_checkpointing_reason
        self.say(
            f"M4 sustained MPS throughput: {M4_SECONDS / 60:.0f} minutes at batch 32"
            + (" with activation checkpointing" if checkpointing else "")
            + f" (M7 starts after {M7_START_DELAY_SECONDS:.0f} s); {self.spend_text()}"
        )
        with self.guard("M4", entry), self.measure(measurement) as charge:
            request = RunRequest(
                mode="timing",
                recipe=_timing_recipe(RECIPE_BATCH),
                fold=self.fold,
                export=self.export,
                compute=Compute(
                    device="mps",
                    threads=MPS_THREADS,
                    deterministic=deterministic,
                    activation_checkpointing=checkpointing,
                    memory_guard=True,
                ),
                wrappers=Wrappers(early_stopping=False),
                timing=TimingSpec(warmup_steps=M1_WARMUP_STEPS, timed_seconds=M4_SECONDS),
                deadline_seconds=deadline,
            )
            before = _read_memory()
            started = _clock()
            child = self.steps.start_child(request, root=self.root)
            try:
                stopped, watched, m7_cpu = self.watch(child, started)
                result = child.wait(timeout_seconds=CHILD_TIMEOUT_MARGIN_SECONDS)
            except BaseException:
                child.terminate()  # an interrupt or a failing M7 never leaves it running
                raise
            wall = _clock() - started
            charge.excluded_cpu = m7_cpu
            charge.declared_cpu = max(
                _number_or_none(result.document.get("cpu_seconds")) or 0.0, watched
            )
            charge.gpu_minutes = wall / 60.0
            self.record_child(
                "M4",
                entry,
                result,
                device="mps",
                batch=RECIPE_BATCH,
                wall=wall,
                before=before,
                after=_read_memory(),
                budget_stop=stopped,
            )
            entry.update(_thermal(result.document, completed=entry["status"] == "completed"))
        self.m7_not_run("not run: M4 ended before M7 could start")
        theta = entry.get("theta")
        self.say(
            f"M4: {entry['status']}"
            + (f", theta {theta:.3f}" if _finite(theta) else "")
            + f"; {self.spend_text()}"
        )

    def m4_cpu(self, child: Any, started: float) -> float:
        """CPU-seconds the running M4 child has used so far: read with ``ps``, or its wall time
        x its :data:`MPS_THREADS` threads when that cannot be read."""
        used = _child_cpu_seconds(child)
        return used if used is not None else (_clock() - started) * MPS_THREADS

    def watch(self, child: Any, started: float) -> tuple[bool, float, float]:
        """Poll the running M4 child until it ends, stopping it when the spend with its CPU
        time passes a limit; M7 runs once it has run :data:`M7_START_DELAY_SECONDS` (the child
        is not polled while M7 runs, which the stop limit's reserve covers). Returns whether the
        budget stopped it, its last watched CPU-seconds and M7's measured CPU-seconds (charged
        by M7 itself)."""
        watched = 0.0
        m7_cpu = 0.0
        m7_done = False
        while child.poll() is None:
            elapsed = _clock() - started
            watched = self.m4_cpu(child, started)
            reason = self._limit_reason(
                self.cpu_seconds + watched,
                self.gpu_minutes + elapsed / 60.0,
                "the spend with M4 running is",
            )
            if reason is not None:
                self.stop(reason)
                child.terminate()
                return True, watched, m7_cpu
            if not m7_done and elapsed >= M7_START_DELAY_SECONDS:
                m7_done = True
                m7_cpu = self.m7(child, started)
                continue
            _sleep(CHILD_POLL_SECONDS)
        return False, watched, m7_cpu

    def m7(self, child: Any, started: float) -> float:
        """M7: 12 fans with 1, then 2 workers while the M4 child trains, each admitted on the
        child's CPU time read again at its start (:meth:`m4_cpu`); returns the measured
        CPU-seconds of its blocks."""
        measured = 0.0
        for workers in M7_WORKERS:
            entry: dict[str, Any] = {
                "requested_workers": workers,
                "fans": FANS_PER_WORKER_TEST,
                "status": "not_run",
            }
            self.measurements["M7"].append(entry)
            if child.poll() is not None:
                entry["reason"] = "not run: M4 ended before it"
                continue
            prior = (PRIORS["M7"][0] / len(M7_WORKERS), 0.0)
            reason = self.may_start(
                f"M7 {workers} worker(s)", prior, pending_cpu=self.m4_cpu(child, started)
            )
            if reason is not None:
                entry["reason"] = reason
                continue
            entry["m4_elapsed_start_seconds"] = _clock() - started
            with self.guard(f"M7 {workers} worker(s)", entry), self.measure("M7") as charge:
                result = self.steps.time_fan_workers(
                    self.params, workers=workers, fans=FANS_PER_WORKER_TEST
                )
                charge.declared_cpu = _number_or_none(result.get("cpu_seconds")) or 0.0
                entry.update(_fan_workers_entry(result))
            entry["m4_elapsed_end_seconds"] = _clock() - started
            measured += charge.measured_cpu
            self.say(f"M7 {workers} worker(s) during M4: {entry['status']}; {self.spend_text()}")
        return measured

    # M6 ---------------------------------------------------------------------------------------

    def m6(self) -> None:
        for workers in M6_WORKERS:
            entry: dict[str, Any] = {
                "requested_workers": workers,
                "fans": FANS_PER_WORKER_TEST,
                "status": "not_run",
            }
            self.measurements["M6"].append(entry)
            prior = (PRIORS["M6"][0] / len(M6_WORKERS), 0.0)
            reason = self.may_start(f"M6 {workers} worker(s)", prior)
            if reason is not None:
                entry["reason"] = reason
                continue
            with self.guard(f"M6 {workers} worker(s)", entry), self.measure("M6") as charge:
                result = self.steps.time_fan_workers(
                    self.params, workers=workers, fans=FANS_PER_WORKER_TEST
                )
                charge.declared_cpu = _number_or_none(result.get("cpu_seconds")) or 0.0
                entry.update(_fan_workers_entry(result))
            self.say(f"M6 {workers} worker(s): {entry['status']}; {self.spend_text()}")

    # the whole run ------------------------------------------------------------------------------

    def run(self) -> None:
        self.setup()
        self.m9()
        self.m8()
        self.m5()
        self.release_models()
        self.m1("cpu")
        self.m2_if_needed("cpu")
        if self.mps_available:
            self.checkpointing_from_cpu()
        self.m3()
        self.m1("mps")
        self.m2_if_needed("mps")
        self.m4()
        self.m6()

    def record(
        self,
        *,
        started_at: str,
        finished_at: str,
        scratch: str,
        environment: Mapping[str, Any],
        code: Mapping[str, Any],
    ) -> dict[str, Any]:
        status = "failed" if self.failures else ("stopped" if self.stop_reason else "completed")
        measurements = self.measurements
        single = {
            item.get("workers"): item
            for item in measurements["M6"]
            if item["status"] == "completed"
        }
        for item in _m7_entries(measurements):
            alone = single.get(item.get("workers"))
            item["slowdown_vs_m6"] = (
                item["seconds_per_fan"] / alone["seconds_per_fan"]
                if item["status"] == "completed"
                and alone
                and _finite(item.get("seconds_per_fan"))
                and _finite(alone.get("seconds_per_fan"))
                and alone["seconds_per_fan"] > 0
                else None
            )
        return {
            "schema_version": TIMING_SCHEMA_VERSION,
            "status": status,
            "retraining_md_sha256": _retraining_md_sha256(),
            "environment": dict(environment),
            "code": dict(code),
            "estimate": self.estimate,
            "spend": {
                "cpu_seconds": self.cpu_seconds,
                "gpu_minutes": self.gpu_minutes,
                "limit_cpu_seconds": self.budget.cpu_seconds,
                "limit_gpu_minutes": self.budget.gpu_minutes,
                "gate3_cpu_seconds": GATE3_CPU_SECONDS,
                "gate3_gpu_minutes": GATE3_GPU_MINUTES,
                "stopped": self.stop_reason is not None,
                "stop_reason": None if self.stop_reason is None else _public_text(self.stop_reason),
                "per_measurement": self.spent,
            },
            "approval": _approval_record(self.budget, self.estimate),
            "measurements": measurements,
            "mps_checkpointing": {
                "from_batch": self.mps_checkpointing_from,
                "reason": self.mps_checkpointing_reason,
            },
            "memory_slopes": _memory_slopes(measurements["M1"]),
            "choice": choose_compute(measurements),
            "failures": [_public_text(item) for item in self.failures],
            "samples": self.samples,
            "scratch": scratch,
            "started_at": started_at,
            "finished_at": finished_at,
        }


def _memory_slopes(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per configuration, the least-squares slope of the peak memory over the batch size (bytes
    per window; about 0.2 GiB means the attention probabilities are materialised), over the
    batches that ran without activation checkpointing."""
    slopes: list[dict[str, Any]] = []
    for device, threads in M1_CONFIGS:
        points = [
            (float(entry["batch_size"]), float(entry["peak_bytes"]))
            for entry in entries
            if (entry["device"], entry["threads"]) == (device, threads)
            and entry["status"] == "completed"
            and not _checkpointed(entry)
            and _finite(entry.get("peak_bytes"))
        ]
        fit = _linear_fit(points)
        slopes.append(
            {
                "device": device,
                "threads": threads,
                "batches": [int(batch) for batch, _ in points],
                "bytes_per_window": None if fit is None else fit["slope"],
                "gib_per_window": None if fit is None else fit["slope"] / GIB,
            }
        )
    return slopes


# -- the choice -------------------------------------------------------------------------------


def _b32(measurements: Mapping[str, Any], device: str, threads: int) -> Mapping[str, Any] | None:
    for entry in measurements.get("M1") or []:
        if (entry.get("device"), entry.get("threads"), entry.get("batch_size")) == (
            device,
            threads,
            RECIPE_BATCH,
        ):
            return entry
    return None


def _step_median(entry: Mapping[str, Any] | None) -> float | None:
    if not entry or entry.get("status") != "completed":
        return None
    value = (entry.get("seconds_per_step") or {}).get("median")
    return float(value) if _finite(value) and value > 0 else None


def _m7_entries(measurements: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every M7 entry: those of M4's last attempt, then those of a retried M4's first one."""
    first = (measurements.get("M4") or {}).get("first_attempt") or {}
    return [*(measurements.get("M7") or []), *(first.get("m7") or [])]


def _retried(measurements: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """``(unit, entry)`` of every MPS unit that ended memory_abort without activation
    checkpointing: retried (``first_attempt``) or not (``retry`` says why)."""
    units: list[tuple[str, Any]] = [("M3", measurements.get("M3"))]
    units += [
        (f"M1 {item.get('device')} x{item.get('threads')} batch {item.get('batch_size')}", item)
        for item in measurements.get("M1") or []
    ]
    units.append(("M4", measurements.get("M4")))
    return [
        (unit, entry)
        for unit, entry in units
        if isinstance(entry, Mapping) and ("first_attempt" in entry or "retry" in entry)
    ]


def _recipe_batch(measurements: Mapping[str, Any], device: str, threads: int) -> dict[str, Any]:
    """How ``device`` at ``threads`` runs the recipe batch by M1 and M2: as is (``checkpointing``
    False) when its batch 32 completed without activation checkpointing and fits; with it (True)
    when a batch-32 timing with it (M1's own or an M2 run's) completed and fits and M2's CPU
    gradient bit-identity passed; ``None`` when neither. ``timing`` is the entry timed."""
    b32 = _b32(measurements, device, threads)
    if b32 is not None and not _checkpointed(b32) and _fits(b32) is True:
        return {"checkpointing": False, "timing": b32}
    timings: list[Any] = [b32] if _checkpointed(b32) else []
    timings += [
        run.get("timing")
        for run in _m2_runs(measurements)
        if (run.get("device"), run.get("threads")) == (device, threads)
    ]
    timing = next((item for item in timings if item and _fits(item) is True), None)
    if timing is not None and _checkpointing_valid(measurements) is True:
        return {"checkpointing": True, "timing": timing}
    return {"checkpointing": None, "timing": b32}


def _best_cpu(measurements: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """The batch-32 step time of the fastest CPU configuration that runs batch 32 (as is where
    it fits, with activation checkpointing where M2 passed) and its description; the fastest
    time as is when no CPU configuration runs batch 32; ``None`` without any."""
    runnable: list[tuple[float, int, bool]] = []
    as_is: list[tuple[float, int]] = []
    for threads in (threads for device, threads in M1_CONFIGS if device == "cpu"):
        config = _recipe_batch(measurements, "cpu", threads)
        seconds = _step_median(config["timing"])
        if config["checkpointing"] is not None and seconds is not None:
            runnable.append((seconds, threads, config["checkpointing"]))
        plain = _step_median(_b32(measurements, "cpu", threads))
        if plain is not None:
            as_is.append((plain, threads))
    if runnable:
        seconds, threads, checkpointing = min(runnable)
        how = "with activation checkpointing" if checkpointing else "as is"
        return seconds, f"CPU x{threads} {how}, {seconds:.3f} s per step"
    if as_is:
        seconds, threads = min(as_is)
        return seconds, (
            f"CPU x{threads} as is, {seconds:.3f} s per step; no CPU configuration runs batch 32"
        )
    return None, None


def choose_compute(measurements: Mapping[str, Any]) -> dict[str, Any]:
    """The compute choice of design 3.4 and (a) from the measurements (interfaces 4.12).

    CPU threads: 8 only if 8 threads are at least 1.3 times faster than 2 at batch 32 as is.
    How a device runs batch 32 (:func:`_recipe_batch`): as is when it fits at a peak of at most
    9 GiB with no swap growth, otherwise with activation checkpointing where a timing with it
    fits and M2's CPU gradient bit-identity passed. Device: MPS only if the M3 multinomial,
    parity and steps checks passed and the M4 sustained median step time (its last 10 minutes;
    with activation checkpointing when M4 ran with it, which then needs that gradient check) is
    at least 1.5 times faster than the batch-32 median of the fastest CPU configuration that runs
    batch 32 (the times as is when none does); otherwise the CPU. Deterministic mode: always on
    the CPU; on MPS only if M3's deterministic check passed. A device that runs batch 32 neither
    way keeps batch 32 and the reasons say that batch 16 needs the owner (a recipe change, never
    made here). ``device`` is ``None`` when the measurements decide nothing.
    """
    reasons: list[str] = []
    t2 = _step_median(_b32(measurements, "cpu", 2))
    t8 = _step_median(_b32(measurements, "cpu", 8))
    cpu_threads: int | None
    eight_vs_two = t2 / t8 if t2 is not None and t8 is not None else None
    if t2 is None and t8 is None:
        cpu_threads = None
        reasons.append("no CPU configuration completed batch 32 in M1")
    elif t8 is not None and (
        eight_vs_two is None or eight_vs_two >= EIGHT_THREADS_REQUIRED_SPEEDUP
    ):
        cpu_threads = 8
        reasons.append(
            "CPU threads 8: "
            + (
                "2 threads did not complete batch 32"
                if eight_vs_two is None
                else f"{eight_vs_two:.2f} times faster than 2 at batch 32 (at least "
                f"{EIGHT_THREADS_REQUIRED_SPEEDUP:g} needed)"
            )
        )
    else:
        cpu_threads = 2
        reasons.append(
            "CPU threads 2: "
            + (
                "8 threads did not complete batch 32"
                if eight_vs_two is None
                else f"8 threads only {eight_vs_two:.2f} times faster at batch 32 (at least "
                f"{EIGHT_THREADS_REQUIRED_SPEEDUP:g} needed)"
            )
        )
    best_cpu, best_cpu_configuration = _best_cpu(measurements)
    m3 = measurements.get("M3") or {}
    m4 = measurements.get("M4") or {}
    sustained = m4.get("median_last_seconds") if m4.get("status") == "completed" else None
    sustained = float(sustained) if _finite(sustained) and sustained > 0 else None
    mps_batch = _recipe_batch(measurements, "mps", MPS_THREADS)
    mps_checkpointing = _checkpointed(m4) or mps_batch["checkpointing"] is True
    wrapped = " (with activation checkpointing)" if mps_checkpointing else ""
    mps_vs_cpu = best_cpu / sustained if best_cpu is not None and sustained is not None else None
    device: str | None = None
    if m3.get("passed") is not True:
        reasons.append(f"MPS not usable: {m3.get('reason') or 'its preflight (M3) did not pass'}")
    elif sustained is None:
        why = m4.get("reason") or m4.get("status")
        reasons.append(f"MPS not chosen: M4 gave no sustained step time ({why})")
    elif mps_checkpointing and _checkpointing_valid(measurements) is not True:
        reasons.append(
            "MPS not chosen: it runs batch 32 with activation checkpointing, which did not pass "
            "M2's CPU gradient bit-identity"
        )
    elif mps_vs_cpu is None:
        reasons.append("MPS not chosen: no CPU batch-32 time to compare its sustained time with")
    elif mps_vs_cpu >= MPS_REQUIRED_SPEEDUP:
        device = "mps"
        reasons.append(
            f"device MPS: its sustained step time{wrapped} is {mps_vs_cpu:.2f} times faster than "
            f"the best CPU configuration's ({best_cpu_configuration}; at least "
            f"{MPS_REQUIRED_SPEEDUP:g} needed)"
        )
    else:
        reasons.append(
            f"MPS not chosen: its sustained step time{wrapped} is only {mps_vs_cpu:.2f} times "
            f"faster than the best CPU configuration's ({best_cpu_configuration}; at least "
            f"{MPS_REQUIRED_SPEEDUP:g} needed)"
        )
    if device is None and cpu_threads is not None:
        device = "cpu"  # a CPU time exists, so MPS was refused, not left uncompared
    threads: int | None
    deterministic: bool | None
    if device is None:
        reasons.append("no device decided: the owner decides with this record")
        threads, deterministic = None, None
    elif device == "mps":
        threads, deterministic = MPS_THREADS, m3.get("deterministic_passed") is True
    else:
        threads, deterministic = cpu_threads, True
    batch = (
        None if device is None or threads is None else _recipe_batch(measurements, device, threads)
    )
    checkpointing = False
    fits = False
    seconds_per_step: float | None = None
    if batch is None:
        pass
    elif device == "mps" and mps_checkpointing:
        checkpointing = True
        why = m4.get("checkpointing_reason") or (mps_batch["timing"] or {}).get(
            "checkpointing_reason"
        )
        if batch["checkpointing"] is True:
            seconds_per_step = _step_median(batch["timing"])
        else:  # only M4 needed it: the step time with it is M4's first 2 minutes
            seconds_per_step = _number_or_none(m4.get("median_first_seconds"))
            why = f"{why or 'M4 needed it'}; its step time is M4's first 2 minutes"
        reasons.append(
            "batch 32 runs with activation checkpointing on MPS" + (f": {why}" if why else "")
        )
    elif batch["checkpointing"] is False:
        fits = True
        seconds_per_step = _step_median(batch["timing"])
        reasons.append("batch 32 fits: peak at most 9 GiB and no swap growth")
    elif batch["checkpointing"] is True:
        checkpointing = True
        seconds_per_step = _step_median(batch["timing"])
        reasons.append("batch 32 does not fit as is: activation checkpointing (M2 passed)")
    else:
        seconds_per_step = _step_median(batch["timing"])
        reasons.append(
            "batch 32 does not fit (or was not measured) and activation checkpointing did not "
            "pass: batch 16 needs the owner (a recipe change)"
        )
    theta = None
    if device == "mps":
        measured = m4.get("theta")
        if _finite(measured):
            theta = max(1.0, float(measured))
            reasons.append(f"theta {theta:.3f} from M4 (at least 1)")
    elif device == "cpu":
        reasons.append("theta for the CPU comes from the pilot's first pseudo-epoch (no M4)")
    return {
        "device": device,
        "threads": threads,
        "batch_size": RECIPE_BATCH,
        "deterministic": deterministic,
        "activation_checkpointing": checkpointing,
        "mps_watermark_ratio": DEFAULT_MPS_WATERMARK_RATIO if device == "mps" else None,
        "batch_32_fits": fits,
        "seconds_per_step": seconds_per_step,
        "theta": theta,
        "speedups": {"mps_vs_best_cpu": mps_vs_cpu, "eight_vs_two_threads": eight_vs_two},
        "best_cpu": {"configuration": best_cpu_configuration, "seconds_per_step": best_cpu},
        "mps_sustained_seconds_per_step": sustained,
        "reasons": reasons,
    }


# -- the report -------------------------------------------------------------------------------


def _report_dir(directory: Path | None) -> Path:
    return Path(directory) if directory is not None else reports_dir() / FINETUNE_REPORTS_SUBDIR


def _comparators() -> dict[str, Any]:
    intercept, slope = PRIOR_FAN_SECONDS
    return {
        "priors": f"the per-measurement priors of {RETRAINING_MD_RELPATH} section (a)",
        "prior_seconds_per_step_b32": {
            f"{device}-{threads}": seconds
            for (device, threads), seconds in PRIOR_SECONDS_PER_STEP_B32.items()
        },
        "prior_fan_seconds": f"t(L) = {intercept:g} + {slope:g} L s at 2 threads (design 5.2)",
        "prior_nll4_tf": "about 50 s per 336 decisions at 2 threads (design 3.3)",
        "device_rule": f"MPS if its preflight passed and its sustained step time (with "
        f"activation checkpointing when it ran with it) is at least {MPS_REQUIRED_SPEEDUP:g} "
        "times faster than that of the fastest CPU configuration that runs batch 32 (as is "
        "where it fits, with activation checkpointing where M2 passed)",
        "threads_rule": f"8 CPU threads only if at least {EIGHT_THREADS_REQUIRED_SPEEDUP:g} "
        "times faster than 2 at batch 32",
        "memory_rule": "batch 32 fits at a peak of at most 9 GiB with no swap growth; "
        "otherwise activation checkpointing (M2), on MPS also after a memory abort (one retry)",
    }


def _seeds() -> dict[str, Any]:
    return {
        "training": "finetune.recipe.training_seeds of each timing recipe for fold "
        f"{TIMING_FOLD_LABEL}, replicate 0",
        "fans": "kronos_agent.seeds of the timing window's decision hour for the streams "
        "timing-warmup, timing and timing-w<i>",
        "synthetic_candles": TIMING_SYNTHETIC_SEED,
    }


def _payload(record: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    from ..report.writer import report_header

    missing = [key for key in RECORD_KEYS if key not in record]
    if missing:
        raise TimingError(f"a timing record lacks {missing}")
    header = report_header(
        TIMING_STEM,
        timing_plan(),
        statistics_with_intervals=0,
        seeds=_seeds(),
        samples=dict(record.get("samples") or {}),
        comparators=_comparators(),
        directory=directory,
    )
    payload = {**dict(record), **{key: value for key, value in header.items() if key != "code"}}
    payload["code"] = {**header["code"], **dict(record["code"])}
    return payload


def write_timing_report(
    record: Mapping[str, Any], *, directory: Path | None = None
) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` (:data:`TIMING_STEM`; default
    ``reports/finetune``) through the report layer: the header of
    :func:`kronos_agent.report.writer.report_header` (the plan recorded once as
    ``<stem>.config.json``, 0 tests consumed, 0 statistics with intervals) and the body of
    interfaces 5.9; ``code`` holds both the header's provenance and the record's hashes."""
    from ..report.writer import write_report

    directory = _report_dir(directory)
    payload = _payload(record, directory)
    return write_report(TIMING_STEM, payload, _markdown(payload), directory=directory)


def load_timing(*, directory: Path | None = None) -> dict[str, Any]:
    """The timing record (``<stem>.json``, :data:`TIMING_STEM`) as written; :class:`TimingError`
    when it is missing or is not a timing record of this stem and schema (the first run's
    ``timing.json``, schema 1, is neither)."""
    path = _report_dir(directory) / f"{TIMING_STEM}.json"
    shown = f"{FINETUNE_REPORTS_SUBDIR}/{path.name}"
    try:
        record = json.loads(path.read_text(encoding="ascii"))
    except FileNotFoundError:
        raise TimingError(
            f"no timing record ({shown}): run `kronos-agent finetune timing` first"
        ) from None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TimingError(f"{shown} cannot be read: {_public_text(exc)}") from None
    if not isinstance(record, dict) or record.get("study") != TIMING_STEM:
        raise TimingError(f"{shown} is not a timing record")
    if record.get("schema_version") != TIMING_SCHEMA_VERSION:
        raise TimingError(
            f"{shown} has schema {record.get('schema_version')!r}, expected {TIMING_SCHEMA_VERSION}"
        )
    missing = [key for key in RECORD_KEYS if key not in record]
    if missing:
        raise TimingError(f"{shown} lacks {missing}")
    return record


def require_fresh_timing(
    record: Mapping[str, Any],
    *,
    device: str,
    torch_version: str,
    platform: str,
    training_code_sha256: str,
) -> None:
    """Refuse (:class:`StaleTimingError`) a timing record measured for another chosen device,
    torch version, platform label (``paths.default_label()``) or training code
    (``finetune.train.training_code_sha256()``)."""
    if not isinstance(record, Mapping):
        raise StaleTimingError(f"a timing record is a mapping, got {type(record).__name__}")
    choice = record.get("choice") or {}
    environment = record.get("environment") or {}
    code = record.get("code") or {}
    problems = []
    if record.get("schema_version") != TIMING_SCHEMA_VERSION:
        problems.append(f"schema {record.get('schema_version')!r}")
    for label, recorded, wanted in (
        ("device", choice.get("device"), device),
        ("torch", environment.get("torch_version"), torch_version),
        ("platform", environment.get("platform"), platform),
        ("training code", code.get("training_code_sha256"), training_code_sha256),
    ):
        if recorded != wanted:
            if label == "training code":
                recorded = str(recorded)[:12]
                wanted = str(wanted)[:12]
            problems.append(f"{label} {recorded!r} (now {wanted!r})")
    if problems:
        raise StaleTimingError(
            f"the timing record ({FINETUNE_REPORTS_SUBDIR}/{TIMING_STEM}.json) is stale: "
            + "; ".join(problems)
            + "; run `kronos-agent finetune timing` again before estimating from it"
        )


# -- the command ------------------------------------------------------------------------------


def _new_scratch(parent: Path) -> Path:
    """Create ``<parent>/timing-<UTC stamp>`` exclusively (``-1`` .. ``-99`` on a collision)."""
    ensure_directory(parent)
    stamp = datetime.now(UTC).strftime(STAMP_FORMAT)
    for index in range(MAX_SCRATCH_SUFFIX + 1):
        name = f"{SCRATCH_PREFIX}{stamp}" if index == 0 else f"{SCRATCH_PREFIX}{stamp}-{index}"
        path = parent / name
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
        fsync_directory(parent)
        return path
    raise TimingError(f"no free timing scratch directory at {stamp}")


def run_timing(
    *,
    params: Params,
    steps: TimingSteps | None = None,
    root: Path | None = None,
    directory: Path | None = None,
    progress: Callable[[str], None] = print,
    approved_cpu_hours: float | None = None,
    given_at: datetime | None = None,
) -> dict[str, Any]:
    """Run the measurement step and write its report; return the report as written.

    ``root`` is the fine-tune root (``registry.finetuned_dir()`` by default: runs under
    ``<root>/.runs``, the scratch under ``<root>/.scratch/timing-<stamp>``); ``directory`` the
    report directory (``reports/finetune``). The runtime must run ``params``' thread count.
    ``approved_cpu_hours`` is the figure given with ``--approved-by-owner`` (``None`` without
    it) and ``given_at`` the command's start, when the flag was given: the run's CPU limit is
    then that many CPU-hours instead of :data:`TIMING_MAX_CPU_SECONDS` (:func:`timing_budget`),
    a ceiling the run keeps within by working against its stop limit. Printed first: the prior
    estimate with the budget, refused above the stop limit or the GPU limit
    (:class:`TimingBudgetExceeded`) before anything runs; then the approval, if any, and the
    plan is recorded or verified (a changed plan is refused, ``ConfigChangedError``; the budget
    is not part of it) and the scratch lock taken (a concurrent run is refused,
    ``LockBusyError``). The returned report's ``status`` is ``completed``, ``stopped`` (the
    budget stopped it) or ``failed`` (an M9 mismatch, a crashed child or a measurement error;
    ``failures`` lists them).
    """
    if not isinstance(params, Params):
        raise TypeError(f"expected Params, got {type(params).__name__}")
    info = require_configured()
    if info.threads != params.kronos.threads:
        raise TimingError(
            f"the runtime runs {info.threads} threads, params.yaml {params.kronos.threads}: the "
            "measurement step runs the governed thread count"
        )
    budget = timing_budget(approved_cpu_hours, given_at=given_at)
    from ..report.writer import record_config
    from . import registry

    steps = steps if steps is not None else real_steps()
    directory = _report_dir(directory)
    root = Path(root) if root is not None else registry.finetuned_dir()
    mps_available = _mps_available()
    plan = timing_plan()
    estimate = estimate_plan(plan, mps_available=mps_available)
    for line in estimate_lines(estimate, budget):
        progress(line)
    require_within_budget(estimate, budget)
    approval = _approval_record(budget, estimate)
    if approval is not None:
        progress(f"gate 3: {_approval_text(approval)}")
    progress(
        "run it when the machine is otherwise idle, on power, lid open, under caffeinate -i -s"
    )
    digest = record_config(TIMING_STEM, plan, directory=directory)
    progress(f"timing plan {TIMING_STEM}.config.json (sha256 {digest[:12]}) verified")
    environment = _environment(mps_available)
    code = _code()
    scratch_parent = registry.scratch_dir(root)
    with exclusive_lock(scratch_parent / LOCK_NAME):
        started_at = _now_text()
        scratch = _new_scratch(scratch_parent)
        progress(f"scratch {registry.SCRATCH_DIRNAME}/{scratch.name} under the fine-tune root")
        run = _Run(
            params=params,
            steps=steps,
            root=root,
            scratch=scratch,
            estimate=estimate,
            budget=budget,
            mps_available=mps_available,
            say=progress,
        )
        run.run()
        record = run.record(
            started_at=started_at,
            finished_at=_now_text(),
            scratch=scratch.name,
            environment=environment,
            code=code,
        )
        write_timing_report(record, directory=directory)
    return load_timing(directory=directory)


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    """What the command prints after a run: status, spend against the run's limits, the owner's
    approval (if any), the choice and its reasons."""
    spend = report.get("spend") or {}
    choice = report.get("choice") or {}
    lines = [
        (
            f"timing {report.get('status')}: spent {spend.get('cpu_seconds', 0.0):.0f} of "
            f"{spend.get('limit_cpu_seconds', TIMING_MAX_CPU_SECONDS):.0f} CPU-seconds and "
            f"{spend.get('gpu_minutes', 0.0):.1f} of "
            f"{spend.get('limit_gpu_minutes', TIMING_MAX_GPU_MINUTES):g} GPU-minutes"
        )
    ]
    if report.get("approval"):
        lines.append(f"gate 3: {_approval_text(report['approval'])}")
    if spend.get("stop_reason"):
        lines.append(f"stopped: {spend['stop_reason']}")
    measurements = report.get("measurements") or {}
    m9 = measurements.get("M9") or {}
    lines.append(f"M9 identity control: {m9.get('status')}")
    for unit, entry in _retried(measurements):
        lines.append(f"{unit}: memory_abort without activation checkpointing; {_retry_text(entry)}")
    lines.append(
        f"choice: device {choice.get('device')}, threads {choice.get('threads')}, batch "
        f"{choice.get('batch_size')}, deterministic {choice.get('deterministic')}, activation "
        f"checkpointing {choice.get('activation_checkpointing')}, MPS watermark "
        f"{choice.get('mps_watermark_ratio')}"
    )
    lines.extend(f"  - {reason}" for reason in choice.get("reasons") or [])
    lines.extend(f"failure: {item}" for item in report.get("failures") or [])
    return lines


# -- Markdown ---------------------------------------------------------------------------------


def _cell(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "/")


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(_cell(item) for item in row) + " |" for row in rows)
    return lines


def _attempts(entry: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """``(attempt, record)`` of a training unit: a retried one gives its aborted attempt first."""
    first = entry.get("first_attempt")
    if isinstance(first, Mapping):
        return [("1", first), ("2 (retry)", entry)]
    return [("1" if entry.get("status") != "not_run" else "-", entry)]


def _error_text(record: Mapping[str, Any]) -> str | None:
    error = record.get("error")
    return f"{error.get('type')}: {error.get('message')}" if isinstance(error, Mapping) else None


def _attempt_text(record: Mapping[str, Any]) -> str:
    """One attempt of a training unit in words: status, error, peak and swap growth."""
    error = _error_text(record)
    return (
        f"{record.get('status')}"
        + (f" ({error})" if error else "")
        + f", peak {_gib(record.get('peak_bytes'))} GiB, swap growth "
        f"{_gib(record.get('swap_growth_bytes'))} GiB"
    )


def _retry_text(entry: Mapping[str, Any]) -> str:
    """What followed a memory abort without activation checkpointing: the retry or why none."""
    retry = entry.get("retry")
    if isinstance(retry, Mapping):
        return str(retry.get("reason"))
    return f"retried once with it: {entry.get('status')}"


def _section_header(payload: Mapping[str, Any]) -> list[str]:
    code = payload["code"]
    environment = payload["environment"]
    return [
        "# Measurement step of the retraining (finetune timing)",
        "",
        (
            f"Status: **{payload['status']}**. Design: {RETRAINING_MD_RELPATH} section (a), "
            "phase P0. Synthetic data only: no market data and no outcome read; tests consumed: "
            f"{payload['tests_consumed']}. Generated {payload['generated_at']}; run "
            f"{payload.get('started_at')} to {payload.get('finished_at')}; scratch "
            f"`.scratch/{payload.get('scratch')}` under the fine-tune root."
        ),
        "",
        "## Environment and code",
        "",
        *_table(
            ["Item", "Value"],
            [
                ["platform", environment.get("platform")],
                ["torch", environment.get("torch_version")],
                ["numpy", environment.get("numpy_version")],
                ["pandas", environment.get("pandas_version")],
                ["CPU count", environment.get("cpu_count")],
                ["RAM GiB", _gib(environment.get("ram_bytes"))],
                ["MPS available", environment.get("mps_available")],
                ["training code sha256", code.get("training_code_sha256")],
                ["forecasting code sha256", code.get("forecasting_code_sha256")],
                ["vendored manifest sha256", code.get("vendored_manifest_sha256")],
                ["source tree sha256", code.get("source_tree_sha256")],
                ["timing plan sha256", payload.get("config_sha256")],
                ["RETRAINING.md sha256", payload.get("retraining_md_sha256")],
            ],
        ),
        "",
        (
            "A later estimate refuses this record when the chosen device, the torch version, "
            "the platform or the training code differ."
        ),
        "",
    ]


def _approval_paragraph(approval: Mapping[str, Any], spend: Mapping[str, Any]) -> str:
    """The Markdown's statement of the owner's approval and of the spend against the ceiling the
    owner approved and against gate 3."""
    cpu = spend["cpu_seconds"]
    ceiling = approval["limit_cpu_seconds"]
    gate3 = approval["gate3_cpu_seconds"]
    within = "stayed within" if cpu <= ceiling else "passed"
    passed = "passed" if cpu > gate3 else "stayed below"
    return (
        f"Gate 3: {_approval_text(approval)}. The CPU limit is the ceiling the owner approved in "
        "chat for this measurement: an estimate above the stop limit is refused before anything "
        "runs, a unit or retry whose prior would pass it is not started, each child's deadline "
        "and the watch of M4 follow it, and the run stops once its measured spend passes it. The "
        f"run's spend, {cpu:.0f} CPU-seconds, {within} the {ceiling:.0f} CPU-seconds the owner "
        f"approved and {passed} the gate-3 limit ({gate3} CPU-seconds)."
    )


def _section_budget(payload: Mapping[str, Any]) -> list[str]:
    spend = payload["spend"]
    estimate = payload["estimate"]
    approval = payload.get("approval")
    stopped = f" ({spend['stop_reason']})." if spend["stopped"] else "."
    rows = [
        [
            name,
            estimate["per_measurement"].get(name, {}).get("cpu_seconds"),
            estimate["per_measurement"].get(name, {}).get("gpu_minutes"),
            item["cpu_seconds"],
            item["gpu_minutes"],
        ]
        for name, item in spend["per_measurement"].items()
    ]
    cpu_limit = f"{spend['limit_cpu_seconds']:.0f} CPU-seconds"
    gpu_limit = f"{spend['limit_gpu_minutes']:g} GPU-minutes"
    limits = (
        f"{cpu_limit}, the ceiling the owner approved (gate 3 below), and {gpu_limit}"
        if approval
        else f"{cpu_limit} and {gpu_limit}, below gate 3 ({spend['gate3_cpu_seconds']} "
        f"CPU-seconds, {spend['gate3_gpu_minutes']:g} GPU-minutes)"
    )
    return [
        "## Budget",
        "",
        (
            f"Estimate (priors) about {estimate['cpu_seconds']:.0f} CPU-seconds and "
            f"{estimate['gpu_minutes']:.1f} GPU-minutes; spent {spend['cpu_seconds']:.0f} "
            f"CPU-seconds and {spend['gpu_minutes']:.1f} GPU-minutes; limits {limits}. Stopped: "
            f"{_cell(spend['stopped'])}" + stopped
        ),
        "",
        *([_approval_paragraph(approval, spend), ""] if approval else []),
        *_table(["Item", "Prior CPU-s", "Prior GPU-min", "Spent CPU-s", "Spent GPU-min"], rows),
        "",
    ]


def _section_choice(payload: Mapping[str, Any]) -> list[str]:
    choice = payload["choice"]
    speedups = choice.get("speedups") or {}
    lines = [
        "## Choice (design 3.4)",
        "",
        *_table(
            ["Item", "Value"],
            [
                ["device", choice.get("device")],
                ["threads", choice.get("threads")],
                ["batch", choice.get("batch_size")],
                ["deterministic", choice.get("deterministic")],
                ["activation checkpointing", choice.get("activation_checkpointing")],
                ["MPS watermark ratio", choice.get("mps_watermark_ratio")],
                ["batch 32 fits", choice.get("batch_32_fits")],
                ["seconds per step (batch 32)", choice.get("seconds_per_step")],
                ["theta", choice.get("theta")],
                ["MPS sustained s/step", choice.get("mps_sustained_seconds_per_step")],
                ["best CPU configuration", (choice.get("best_cpu") or {}).get("configuration")],
                ["MPS vs best CPU", speedups.get("mps_vs_best_cpu")],
                ["8 vs 2 threads", speedups.get("eight_vs_two_threads")],
            ],
        ),
        "",
        *[f"- {reason}" for reason in choice.get("reasons") or []],
        "",
    ]
    if payload.get("failures"):
        lines += ["## Failures", "", *[f"- {item}" for item in payload["failures"]], ""]
    return lines


def _section_retries(payload: Mapping[str, Any]) -> list[str]:
    note = payload.get("mps_checkpointing") or {}
    lines = ["## Activation checkpointing on MPS and retries", ""]
    if note.get("from_batch") is not None:
        lines.append(
            f"MPS units from batch {note['from_batch']} up ran with activation checkpointing: "
            f"{note.get('reason')}."
        )
    else:
        lines.append("No MPS unit was set to run with activation checkpointing.")
    lines.append("")
    retried = _retried(payload["measurements"])
    if not retried:
        return lines + [
            "No MPS unit ended memory_abort without activation checkpointing: none was retried.",
            "",
        ]
    rows: list[list[object]] = []
    for unit, entry in retried:
        for attempt, record in _attempts(entry):
            rows.append(
                [
                    unit,
                    attempt,
                    record.get("activation_checkpointing"),
                    record.get("status"),
                    _gib(record.get("peak_bytes")),
                    _gib(record.get("swap_growth_bytes")),
                    _error_text(record),
                ]
            )
        if isinstance(entry.get("retry"), Mapping):
            rows.append([unit, "2", True, "not_run", "-", "-", entry["retry"].get("reason")])
    header = ["Unit", "Attempt", "Checkpointing", "Status", "Peak GiB", "Swap growth GiB"]
    return lines + [
        (
            "Each MPS unit that ended memory_abort without activation checkpointing, with what "
            "followed: one retry with it, or why there was none."
        ),
        "",
        *_table([*header, "Error or reason"], rows),
        "",
    ]


def _section_training(payload: Mapping[str, Any]) -> list[str]:
    measurements = payload["measurements"]
    rows = []
    for item in measurements["M1"]:
        for attempt, record in _attempts(item):
            steps = record.get("seconds_per_step") or {}
            rows.append(
                [
                    item["device"],
                    item["threads"],
                    item["batch_size"],
                    record.get("activation_checkpointing"),
                    attempt,
                    record.get("status"),
                    steps.get("median"),
                    steps.get("min"),
                    steps.get("max"),
                    record.get("windows_per_second"),
                    _gib(record.get("peak_bytes")),
                    _gib(record.get("swap_growth_bytes")),
                    record.get("max_pressure_level"),
                ]
            )
    header = ["Device", "Threads", "Batch", "Checkpointing", "Attempt", "Status", "Median s/step"]
    header += ["Min", "Max", "Windows/s", "Peak GiB", "Swap growth GiB", "Pressure"]
    slopes = [
        [item["device"], item["threads"], item["batches"], item["gib_per_window"]]
        for item in payload.get("memory_slopes") or []
    ]
    lines = [
        "## M1 training step time and peak memory",
        "",
        *_table(header, rows),
        "",
        "Peak: the child's peak RSS, plus its largest MPS driver allocation on MPS.",
        "",
        *_table(["Device", "Threads", "Batches", "GiB per window"], slopes),
        "",
        "## M2 activation checkpointing",
        "",
    ]
    m2 = measurements["M2"] or {}
    gradient = m2.get("gradient_check")
    runs = m2.get("runs") or []
    if gradient is None and not runs:
        return lines + ["Not run: no usable device needed it at batch 32.", ""]
    if gradient is not None:
        result = (gradient.get("checks") or {}).get("checkpoint_gradients") or {}
        mismatches = ", ".join(str(name) for name in result.get("mismatches") or []) or "none"
        lines += [
            (
                "CPU gradient bit-identity of one step with and without the wrap (dropout "
                f"active): status {gradient.get('status')}, passed "
                f"{_cell(gradient.get('passed'))}; {_cell(result.get('tensors'))} tensors, "
                f"mismatches: {mismatches}."
            ),
            "",
        ]
    if not runs:
        return lines
    rows = []
    for item in runs:
        timing = item.get("timing") or {}
        rows.append(
            [
                item["device"],
                item["threads"],
                item["why"],
                item["status"],
                (timing.get("seconds_per_step") or {}).get("median"),
                _gib(timing.get("peak_bytes")),
                item["passed"],
            ]
        )
    header = ["Device", "Threads", "Why", "Status", "Median s/step", "Peak GiB", "Passed"]
    return lines + [*_table(header, rows), ""]


def _section_mps(payload: Mapping[str, Any]) -> list[str]:
    m3 = payload["measurements"]["M3"] or {}
    m4 = payload["measurements"]["M4"] or {}
    checks = m3.get("checks") or {}
    rows = []
    for name in M3_CHECKS:
        result = checks.get(name) or {}
        detail = ", ".join(  # 6 significant digits: a cosine of 0.99999 is not 1.000
            f"{key} {value:.6g}" if isinstance(value, float) else f"{key} {_cell(value)}"
            for key, value in sorted(result.items())
            if key != "passed" and not isinstance(value, (dict, list))
        )
        rows.append([name, result.get("passed"), detail])
    lines = [
        "## M3 MPS preflight",
        "",
        (
            f"Status {m3.get('status')}; passed {_cell(m3.get('passed'))}; activation "
            f"checkpointing {_cell(m3.get('activation_checkpointing'))}."
        ),
    ]
    if m3.get("reason"):
        lines.append(f"Reason: {m3['reason']}.")
    first = m3.get("first_attempt")
    if isinstance(first, Mapping):
        lines.append(
            f"Attempt 1, without activation checkpointing: {_attempt_text(first)}. Attempt 2 "
            "retried it with activation checkpointing; the checks below are attempt 2's."
        )
    elif isinstance(m3.get("retry"), Mapping):
        lines.append(f"Without activation checkpointing; {m3['retry'].get('reason')}.")
    attempts = []
    if isinstance(m4.get("first_attempt"), Mapping):
        attempts.append(
            ["attempt 1, without activation checkpointing", _attempt_text(m4["first_attempt"])]
        )
    if isinstance(m4.get("retry"), Mapping):
        attempts.append(["not retried because", m4["retry"].get("reason")])
    return lines + [
        "",
        *_table(["Check", "Passed", "Detail"], rows),
        "",
        "## M4 sustained MPS throughput",
        "",
        *_table(
            ["Item", "Value"],
            [
                ["status", m4.get("status")],
                ["reason", m4.get("reason")],
                ["timed seconds", m4.get("seconds")],
                ["timed steps", m4.get("steps")],
                ["median s/step, first 2 minutes", m4.get("median_first_seconds")],
                ["median s/step, last 10 minutes", m4.get("median_last_seconds")],
                ["theta", m4.get("theta")],
                ["theta not computed because", m4.get("theta_reason")],
                ["activation checkpointing", m4.get("activation_checkpointing")],
                ["peak MPS driver GiB", _gib(m4.get("peak_mps_driver_bytes"))],
                *attempts,
            ],
        ),
        "",
    ]


def _section_fans(payload: Mapping[str, Any]) -> list[str]:
    measurements = payload["measurements"]
    m5 = measurements["M5"] or {}
    m8 = measurements["M8"] or {}
    m9 = measurements["M9"] or {}
    lengths = [
        [item["length"], item["status"], item.get("seconds_per_fan"), item["prior_seconds_per_fan"]]
        for item in m5.get("per_length") or []
    ]
    first_m4 = (measurements["M4"] or {}).get("first_attempt") or {}
    tests = (
        ("M6", measurements["M6"]),
        ("M7", measurements["M7"]),
        ("M7 (M4 attempt 1)", first_m4.get("m7") or []),
    )
    workers = [
        [
            name,
            item.get("workers") or item["requested_workers"],
            item["status"],
            item.get("wall_seconds"),
            item.get("seconds_per_fan"),
            item.get("fans_per_second"),
            item.get("startup_seconds"),
            item.get("slowdown_vs_m6"),
        ]
        for name, items in tests
        for item in items
    ]
    fit = m5.get("fit")
    fitted = (
        [f"Fit: t(L) = {fit['intercept']:.3f} + {fit['slope']:.3f} L seconds.", ""] if fit else []
    )
    header = ["Test", "Workers", "Status", "Wall s", "s per fan", "Fans/s", "Start-up s"]
    return [
        "## M5 fan speed (one warm-up and 3 timed fans, 2 threads)",
        "",
        *_table(["Length", "Status", "Seconds per fan", "Prior"], lengths),
        "",
        *fitted,
        "## M6 worker scaling and M7 contention (during M4)",
        "",
        *_table([*header, "Slowdown"], workers),
        "",
        "## M8 NLL4-TF time and M9 identity control",
        "",
        *_table(
            ["Item", "Value"],
            [
                ["M8 status", m8.get("status")],
                ["M8 decisions", m8.get("decisions")],
                ["M8 seconds per decision", m8.get("seconds_per_decision")],
                ["M8 seconds per fold evaluation", m8.get("seconds_per_evaluation")],
                ["M9 status", m9.get("status")],
                ["M9 identical fan", m9.get("identical")],
                ["M9 shipped config id", m9.get("shipped_config_id")],
                ["M9 checkpoint config id", m9.get("checkpoint_config_id")],
            ],
        ),
        "",
    ]


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        *_section_header(payload),
        *_section_budget(payload),
        *_section_choice(payload),
        *_section_retries(payload),
        *_section_training(payload),
        *_section_mps(payload),
        *_section_fans(payload),
    ]
    return "\n".join(str(line) for line in lines)


__all__ = [
    "APPROVAL_RESERVE_CPU_SECONDS",
    "CHILD_POLL_SECONDS",
    "CHILD_TIMEOUT_MARGIN_SECONDS",
    "EIGHT_THREADS_REQUIRED_SPEEDUP",
    "FANS_PER_WORKER_TEST",
    "FINETUNE_REPORTS_SUBDIR",
    "GATE3_CPU_SECONDS",
    "GATE3_GPU_MINUTES",
    "M1_BATCHES",
    "M1_CONFIGS",
    "M1_TIMED_STEPS",
    "M1_WARMUP_STEPS",
    "M2_GRADIENT_PASSES",
    "M2_PEAK_LIMIT_BYTES",
    "M3_CHECKS",
    "M3_REQUIRED_CHECKS",
    "M4_FIRST_WINDOW_SECONDS",
    "M4_LAST_WINDOW_SECONDS",
    "M4_OVERHEAD_SECONDS",
    "M4_SECONDS",
    "M5_LENGTHS",
    "M5_REPEATS",
    "M6_WORKERS",
    "M7_START_DELAY_SECONDS",
    "M7_WORKERS",
    "M8_DECISIONS",
    "MEASUREMENTS",
    "MPS_ONLY",
    "MPS_REQUIRED_SPEEDUP",
    "PREFLIGHT_CPU_SECONDS_PER_STEP",
    "PREFLIGHT_CPU_STEPS",
    "PREFLIGHT_MPS_STEPS",
    "PREFLIGHT_PRIOR",
    "PRIORS",
    "PRIORS_MPS_SIDE",
    "PRIOR_FAN_SECONDS",
    "PRIOR_SECONDS_PER_STEP_B32",
    "RETRIES",
    "SPEND_ITEMS",
    "STATUSES",
    "SWAP_GROWTH_TOLERANCE_BYTES",
    "TIMING_CSV_START",
    "TIMING_FOLD_LABEL",
    "TIMING_MAX_CPU_SECONDS",
    "TIMING_MAX_GPU_MINUTES",
    "TIMING_PSEUDO_EPOCH_STEPS",
    "TIMING_SCHEMA_VERSION",
    "TIMING_STEM",
    "TIMING_SYNTHETIC_ROWS",
    "UNIT_ORDER",
    "StaleTimingError",
    "TimingBudget",
    "TimingBudgetExceeded",
    "TimingError",
    "TimingSteps",
    "choose_compute",
    "estimate_lines",
    "estimate_plan",
    "load_timing",
    "real_steps",
    "require_fresh_timing",
    "require_within_budget",
    "run_timing",
    "summary_lines",
    "timing_budget",
    "timing_estimate",
    "timing_plan",
    "write_timing_report",
]
