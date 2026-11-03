"""The measurement step ``kronos-agent finetune timing`` (``docs/RETRAINING.md`` section (a)):
the plan and its prior estimate (M3 and the retries priced by their content, M4 by its whole
lifetime: D-054), the budget refusal and the stop rule, the MPS preflight before any MPS timing,
activation checkpointing on MPS (from the CPU's M1 and M2, or one retry after a memory abort
without it, both attempts reported), the compute choice, the staleness of a record, the report's
stem beside the first run's record, its hygiene and the command's exit codes.

Every heavy measurement is a stand-in (:class:`FakeWorld`): training children return scripted
``runner_result.json`` documents, the fan, NLL4-TF and identity measurements scripted values, a
fake clock advances with them, and the CPU clock of this process reads 0, so the spend is exactly
what the stand-ins declare (a child's ``cpu_seconds``; wall x threads for the fans). The synthetic
export is the real exporter's (a fraction of a second). One opt-in test (``weights`` marker) runs
the real identity control M9 through the command's code path on CPU (about 10 s measured on
2026-11-02, under the minute that test may take).

The estimate with MPS passes the default limit of 3,000 CPU-seconds, so a run without the owner's
approval is refused (D-054 (8)); the owner approved up to 1.5 CPU-hours for this measurement
(D-055), given with ``--approved-by-owner`` and tested with the default limit (the approval's
own section). A run under it works against a stop limit 600 CPU-seconds below that ceiling, so
that its spend never passes it, also when a unit costs more than its prior (its own section).
The other stand-in runs use :data:`RAISED_CPU_LIMIT`, the highest limit below gate 3, and the
refusal under the default limit has its own tests.

The child side of activation checkpointing in the MPS preflight runs for real on tiny random
models (``finetune_fakes``): its CPU values are bit-identical with and without the wrap, and on MPS
(skipped without it) the wrap reproduces the unwrapped gradients and generator state.

Nothing here writes under the repository's ``data/``, ``reports/`` or ``weights/``: the report
directory and the fine-tune root are pytest temporary directories.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kronos_agent import cli
from kronos_agent.cli import finetune as finetune_cli
from kronos_agent.data.fsutil import LockBusyError, exclusive_lock
from kronos_agent.finetune import timing
from kronos_agent.finetune.runner import MemoryReading
from kronos_agent.finetune.train import RunnerResult
from kronos_agent.hashing import canonical_json
from kronos_agent.kronos.batch import GATE3_CPU_SECONDS, BatchError
from kronos_agent.kronos.vendored import VendoredTreeError
from kronos_agent.params import load_params
from kronos_agent.paths import REPO_ROOT, default_label
from kronos_agent.report.writer import ConfigChangedError, record_config
from kronos_agent.runtime import require_configured

PARAMS = load_params()
GIB = 2**30
DEFAULT_CPU_LIMIT = timing.TIMING_MAX_CPU_SECONDS  # 3,000: the estimate with MPS passes it
RAISED_CPU_LIMIT = timing.GATE3_CPU_SECONDS - 1  # the stand-in runs' limit, below gate 3
APPROVED_HOURS = 1.5  # the CPU-hours the owner approved on 2026-11-02 (D-055)
APPROVED_CPU_LIMIT = APPROVED_HOURS * 3600  # 5,400 CPU-seconds: above gate 3, never passed
APPROVED_STOP_LIMIT = APPROVED_CPU_LIMIT - 600  # 4,800: the run works against it (the reserve)
GIVEN_AT = datetime(2026, 11, 2, 22, 0, 5, tzinfo=UTC)  # the command's start, in the tests
GIVEN_AT_TEXT = "2026-11-02T22:00:05Z"
STEP_B32 = {("cpu", 2): 6.1, ("cpu", 8): 5.0, ("mps", 2): 2.0}
M4_LIFETIME = 1230.0  # seconds the stand-in M4 child runs: start-up, warm-up, 20 timed minutes
# what the first measurement step met in M3 (reports/finetune/timing.json, 2026-11-02)
MPS_OUT_OF_MEMORY = {
    "type": "RuntimeError",
    "message": "MPS backend out of memory (MPS allocated: 12.17 GiB, other allocations: 1.72 "
    "MiB, max allowed: 11.84 GiB). Tried to allocate 32.00 MiB on shared pool.",
}
MEMORY_ABORT = {
    "status": "memory_abort",
    "error": MPS_OUT_OF_MEMORY,
    "peak": 11 * GIB,
    "swap": 4 * GIB,
    "wall": 100.0,
}
CPU_NEEDS_CHECKPOINTING = {  # the chosen CPU configuration's batch 32 swaps, as on 2026-11-02
    ("timing", "cpu", 2, 32, False): {"peak": int(9.23 * GIB), "swap": int(1.52 * GIB)},
    ("timing", "cpu", 8, 32, False): {"peak": int(9.16 * GIB), "swap": int(1.34 * GIB)},
}
# what the first run charged M6 per worker count (reports/finetune/timing.json, D-053 (8): 779
# CPU-seconds against its prior of 300, 75 per configuration)
M6_FIRST_RUN = {1: 103.35, 2: 165.24, 3: 221.23, 4: 289.18}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeChild:
    """The M4 child of ``start_child``: alive for :data:`M4_LIFETIME` fake seconds, or as
    ``world.m4_overrides`` (keyed by activation checkpointing) says: ``lifetime``, ``status``,
    ``swap`` (its swap growth)."""

    def __init__(self, world: FakeWorld, request: Any, root: Path) -> None:
        self.world = world
        self.request = request
        self.root = root
        self.started = world.clock()
        self.terminated = False
        self.override = world.m4_overrides.get(request.compute.activation_checkpointing, {})
        self.lifetime = self.override.get("lifetime", world.m4_lifetime)

    def poll(self) -> int | None:
        if self.terminated:
            return -15
        return None if self.world.clock() - self.started < self.lifetime else 0

    def terminate(self) -> None:
        if not self.terminated:
            self.terminated = True
            self.world.events.append("terminate M4")

    def wait(self, *, timeout_seconds: float | None = None) -> RunnerResult:
        self.world.active = None
        status = "failed" if self.terminated else self.override.get("status", "completed")
        offsets: list[float] = []
        times: list[float] = []
        offset = 0.0
        while offset < min(1200.0, self.lifetime - 30.0):
            duration = self.world.m4_step * (1.0 if offset < 600.0 else self.world.m4_heat)
            offsets.append(offset)
            times.append(duration)
            offset += duration
        error = None
        if self.terminated:
            error = {"type": "SystemExit", "message": "stopped by a signal (exit 143)"}
        elif status == "memory_abort":
            error = {"type": "MemoryAbort", "message": "swap use grew by 1100000000 bytes"}
        document = {
            "status": status,
            "stop_reason": "timing_done" if status == "completed" else None,
            "error": error,
            "cpu_seconds": self.world.m4_cpu_rate * (self.world.clock() - self.started),
            "wall_seconds": self.world.clock() - self.started,
            "step_times": times,
            "step_offsets": offsets,
            "memory": {
                "peak_rss_bytes": 2 * GIB,
                "peak_mps_driver_bytes": 3 * GIB,
                "swap_growth_bytes": self.override.get("swap", 0),
                "max_pressure_level": 1,
            },
        }
        return RunnerResult(
            run_id="synthetic-timing-m4",
            run_dir=self.root / ".runs" / "synthetic-timing-m4",
            mode="timing",
            status=status,
            exit_code=143 if self.terminated else {"completed": 0, "memory_abort": 3}[status],
            timed_out=False,
            document=document,
        )


class FakeWorld:
    """Scripted stand-ins for every heavy measurement, and the order they were called in."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.mps = True
        self.events: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.overrides: dict[tuple, dict[str, Any]] = {}
        self.preflight: dict[str, dict[str, Any]] = {
            "multinomial": {"passed": True, "rows": 16384},
            "parity": {"passed": True, "gradient_cosine": 0.99999},
            "steps": {"passed": True, "relative_difference": 0.001},
            "deterministic": {"passed": True, "refused": False, "op": None},
            "checkpoint_gradients": {"passed": True, "tensors": 36, "mismatches": []},
        }
        self.identical = True
        self.fan_errors: dict[int, BaseException] = {}
        self.child_error: BaseException | None = None
        self.m4_lifetime = M4_LIFETIME
        self.m4_overrides: dict[bool, dict[str, Any]] = {}
        self.m4_step = 2.0
        self.m4_heat = 1.1
        self.m4_cpu_rate = 0.1
        self.active: FakeChild | None = None
        self.runs = 0

    def steps(self) -> timing.TimingSteps:
        return timing.TimingSteps(
            run_child=self.run_child,
            start_child=self.start_child,
            seconds_per_fan=self.seconds_per_fan,
            time_fan_workers=self.time_fan_workers,
            load_models=self.load_models,
            nll4_tf=self.nll4_tf,
            identity_control=self.identity_control,
        )

    @staticmethod
    def key(request: Any) -> tuple:
        compute = request.compute
        return (
            request.mode,
            compute.device,
            compute.threads,
            request.recipe.batch_size,
            compute.activation_checkpointing,
        )

    def _note(self, verb: str, request: Any, **extra: Any) -> tuple:
        key = self.key(request)
        mode, device, threads, batch, checkpointing = key
        self.events.append(
            f"{verb} {mode} {device} x{threads} b{batch}" + (" ckpt" if checkpointing else "")
        )
        self.requests.append(
            {
                "key": key,
                "deadline": request.deadline_seconds,
                "deterministic": request.compute.deterministic,
                "memory_guard": request.compute.memory_guard,
                "early_stopping": request.wrappers.early_stopping,
                "wrappers": request.wrappers,
                "checks": request.preflight_checks,
                "timing": request.timing,
                "pseudo_epoch_steps": request.recipe.pseudo_epoch_steps,
                "window_rows": request.recipe.window_rows,
                "fold": request.fold.label,
                **extra,
            }
        )
        return key

    def run_child(self, request: Any, *, root: Path, timeout_seconds: float) -> RunnerResult:
        key = self._note("child", request, timeout=timeout_seconds, root=root)
        if self.child_error is not None:
            raise self.child_error
        mode, device, threads, batch, checkpointing = key
        override = self.overrides.get(key, {})
        past = override.get("past_deadline")
        if past is not None:  # a child that runs on until its deadline stops it, ``past`` s late
            wall = request.deadline_seconds + past
            override = {**override, "status": "time_abort", "cpu": wall * threads, "wall": wall}
        status = override.get("status", "completed")
        self.runs += 1
        run_id = f"synthetic-timing-{self.runs}"
        document: dict[str, Any] = {
            "status": status,
            "stop_reason": {"timing": "timing_done", "preflight": "preflight_done"}[mode]
            if status == "completed"
            else None,
            "error": override.get("error"),
            "cpu_seconds": override.get("cpu", 10.0),
            "memory": {
                "peak_rss_bytes": override.get("peak", 2 * GIB),
                "peak_mps_driver_bytes": GIB if device == "mps" else None,
                "swap_growth_bytes": override.get("swap", 0),
                "max_pressure_level": 1,
            },
        }
        if mode == "timing" and status == "completed":
            step = STEP_B32[(device, threads)] * batch / 32 * (1.3 if checkpointing else 1.0)
            step = override.get("step", step)
            document["step_times"] = [step] * timing.M1_TIMED_STEPS
            document["step_offsets"] = [index * step for index in range(timing.M1_TIMED_STEPS)]
        if mode == "preflight" and status == "completed":
            document["preflight"] = {
                check: dict(self.preflight[check]) for check in request.preflight_checks
            }
        self.clock.advance(override.get("wall", 30.0))
        return RunnerResult(
            run_id=run_id,
            run_dir=root / ".runs" / run_id,
            mode=mode,
            status=status,
            exit_code={"completed": 0, "memory_abort": 3, "time_abort": 4}.get(status, 1),
            timed_out=override.get("timed_out", False),
            document=document,
        )

    def start_child(self, request: Any, *, root: Path) -> FakeChild:
        self._note("start", request, root=root)
        self.active = FakeChild(self, request, root)
        return self.active

    def seconds_per_fan(
        self, *, params: Any, models: Any, evaluation_length: int, repeats: int
    ) -> float:
        self.events.append(f"fan t({evaluation_length})")
        if evaluation_length in self.fan_errors:
            raise self.fan_errors[evaluation_length]
        seconds = 0.44 + 0.678 * evaluation_length
        self.clock.advance((repeats + 1) * seconds)
        return seconds

    def time_fan_workers(self, params: Any, *, workers: int, fans: int) -> dict[str, Any]:
        during = self.active is not None and self.active.poll() is None
        self.events.append(f"workers {workers}" + (" during M4" if during else ""))
        wall = 3.15 * fans / workers * (1.25 if during else 1.0)
        self.clock.advance(5.0 + wall)
        return {
            "workers": workers,
            "fans": fans,
            "evaluation_length": 4,
            "startup_seconds": 5.0,
            "wall_seconds": wall,
            "seconds_per_fan": wall * workers / fans,
            "fans_per_second": fans / wall,
            "cpu_seconds": wall * workers * 2,
        }

    def load_models(self, name: str) -> Any:
        self.events.append(f"load {name}")
        return SimpleNamespace(tokenizer="tokenizer", model="model")

    def nll4_tf(
        self, tokenizer: Any, model: Any, inputs: Any, *, clip: float, batch_size: int
    ) -> Any:
        self.events.append(f"nll {inputs.decisions}")
        self.clock.advance(7.0)
        return SimpleNamespace(threads=2, seconds=7.0)

    def identity_control(self, params: Any, *, scratch_root: Path, models: Any) -> dict[str, Any]:
        self.events.append("identity")
        assert not scratch_root.exists()
        self.clock.advance(9.0)
        return {
            "identical": self.identical,
            "checkpoint_id": "ks-identity-shipped",
            "shipped_config_id": "a" * 16,
            "checkpoint_config_id": "b" * 16,
            "paths_sha256": {
                "shipped": "c" * 64,
                "checkpoint": ("c" if self.identical else "d") * 64,
            },
            "files": [{"name": "config.json", "size": 228, "sha256": "e" * 64}],
            "seconds": 9.0,
        }


@pytest.fixture
def world(monkeypatch) -> FakeWorld:
    clock = FakeClock()
    fake = FakeWorld(clock)
    monkeypatch.setattr(timing, "TIMING_MAX_CPU_SECONDS", RAISED_CPU_LIMIT)
    monkeypatch.setattr(timing, "_clock", clock)
    monkeypatch.setattr(timing, "_sleep", clock.advance)
    monkeypatch.setattr(timing, "_cpu_clock", lambda: 0.0)
    monkeypatch.setattr(timing, "_read_memory", lambda: MemoryReading(1, 4 * GIB))
    monkeypatch.setattr(timing, "_mps_available", lambda: fake.mps)
    monkeypatch.setattr(
        timing,
        "_child_cpu_seconds",
        lambda child: fake.m4_cpu_rate * (clock() - child.started),
    )
    return fake


def run_timing(world: FakeWorld, tmp_path: Path, **kwargs: Any) -> tuple[dict[str, Any], list]:
    printed: list[str] = []
    report = timing.run_timing(
        params=kwargs.pop("params", PARAMS),
        steps=world.steps(),
        root=tmp_path / "ft",
        directory=tmp_path / "reports",
        progress=printed.append,
        **kwargs,
    )
    return report, printed


def children(world: FakeWorld) -> list[str]:
    return [event for event in world.events if event.startswith(("child", "start"))]


def m1(report: dict[str, Any], device: str, threads: int, batch: int) -> dict[str, Any]:
    (entry,) = [
        item
        for item in report["measurements"]["M1"]
        if (item["device"], item["threads"], item["batch_size"]) == (device, threads, batch)
    ]
    return entry


def report_names() -> list[str]:
    """The files a run writes: the plan recorded once, the JSON record and the Markdown."""
    return [f"{timing.TIMING_STEM}{suffix}" for suffix in (".config.json", ".json", ".md")]


def read_markdown(directory: Path) -> str:
    return (directory / f"{timing.TIMING_STEM}.md").read_text(encoding="ascii")


# == the plan and its estimate =====================================================================


def test_the_plan_is_constant_and_machine_free(monkeypatch):
    first = timing.timing_plan()
    monkeypatch.setattr(timing, "_mps_available", lambda: False)
    assert timing.timing_plan() == first
    text = canonical_json(first).decode("ascii")
    for root in (REPO_ROOT, Path.home(), Path(tempfile.gettempdir())):
        assert str(root) not in text
    # the plan is what is measured; how far a run may spend is its budget, not in the plan (D-055)
    assert "limits" not in first
    monkeypatch.setattr(timing, "TIMING_MAX_CPU_SECONDS", 1000)  # another budget, the same plan
    assert timing.timing_plan() == first
    assert timing.GATE3_CPU_SECONDS == GATE3_CPU_SECONDS  # the step's own budget is below gate 3
    assert DEFAULT_CPU_LIMIT == 3000 < GATE3_CPU_SECONDS
    assert timing.TIMING_MAX_GPU_MINUTES == 45.0 < timing.GATE3_GPU_MINUTES
    assert first["recipe"]["window_rows"] == 513
    assert first["measurements"]["M3"]["checks"] == list(timing.M3_CHECKS)
    assert first["measurements"]["M3"]["prior"]["cpu_steps"] == 21
    assert first["measurements"]["M4"]["overhead_seconds"] == 60.0
    assert finetune_cli.FINETUNE_REPORTS_SUBDIR == timing.FINETUNE_REPORTS_SUBDIR


def test_the_preflight_and_its_retry_are_priced_by_their_content():
    """M3 runs check (b)'s forward and backward and check (c)'s 20 training steps at batch 32 on
    the CPU at 2 threads besides its MPS side: about 387 CPU-seconds for check (c)'s CPU side alone
    at the first run's 9.68 s per step, which the design's 150 left out (D-054). The retries' prior
    is one retry of that size; the child's whole wall is GPU time."""
    from kronos_agent.finetune.runner import PREFLIGHT_STEPS

    wall = (PREFLIGHT_STEPS + 1) * 9.68 + (PREFLIGHT_STEPS + 3) * 3.2  # MPS: (a), (b), (c), (d)
    assert timing.PRIORS["M3"] == pytest.approx((wall * 2, wall / 60))
    assert timing.PRIORS["M3"] == pytest.approx((553.76, 4.6147), abs=1e-4)
    assert timing.PRIORS["retries"] == timing.PRIORS["M3"]
    assert timing.PRIORS["M3"][0] > PREFLIGHT_STEPS * 9.68 * 2  # check (c)'s CPU side is in it
    # M4 needs its whole lifetime of the GPU budget: 20 timed minutes and a minute more
    assert timing.PRIORS["M4"] == (120.0, 21.0)


def test_the_estimate_comes_from_the_priors_with_and_without_mps():
    plan = timing.timing_plan()
    with_mps = timing.estimate_plan(plan, mps_available=True)
    # the design's 2,620 CPU-seconds and 25 GPU-minutes, M3 priced by its content (instead of 150
    # and 2), M4 by its lifetime (21 GPU-minutes instead of 20) and one retry of M3's size
    m3_cpu, m3_gpu = timing.PREFLIGHT_PRIOR
    assert with_mps["cpu_seconds"] == pytest.approx(2620.0 - 150.0 + 2 * m3_cpu)  # 3,577.5
    assert with_mps["gpu_minutes"] == pytest.approx(25.0 - 2.0 + 1.0 + 2 * m3_gpu)  # 33.2
    assert with_mps["per_measurement"]["M1"] == {"cpu_seconds": 1350.0, "gpu_minutes": 2.0}
    assert with_mps["per_measurement"]["retries"] == {"cpu_seconds": m3_cpu, "gpu_minutes": m3_gpu}
    assert list(with_mps["per_measurement"])[-1] == "retries"
    without = timing.estimate_plan(plan, mps_available=False)
    assert without["cpu_seconds"] == pytest.approx(2050.0)
    assert without["gpu_minutes"] == 0.0
    for name in ("M3", "M4", "M7", "retries"):
        assert without["per_measurement"][name] == {"cpu_seconds": 0.0, "gpu_minutes": 0.0}
    assert without["per_measurement"]["M1"] == {"cpu_seconds": 1200.0, "gpu_minutes": 0.0}
    assert without["per_measurement"]["M2"] == {"cpu_seconds": 200.0, "gpu_minutes": 0.0}
    # with MPS the estimate passes the step's own 3,000 CPU-seconds: refused without the owner's
    # approval (D-054 (8); the owner approved up to 1.5 CPU-hours, D-055); without MPS it fits
    with pytest.raises(timing.TimingBudgetExceeded, match="about 3578 CPU-seconds"):
        timing.require_within_budget(with_mps)
    timing.require_within_budget(without)
    lines = timing.estimate_lines(with_mps)
    assert "3578 CPU-seconds" in lines[0] and "33.2 GPU-minutes" in lines[0]
    assert lines[1] == "  above the budget: `finetune timing` refuses to start"
    assert len(lines) == 2 + len(with_mps["per_measurement"])
    assert lines[-1].startswith("  retries: about 554 CPU-seconds, 4.6 GPU-minutes (one MPS retry")
    assert len(timing.estimate_lines(without)) == 1 + len(without["per_measurement"])
    assert "the retries" in timing.estimate_lines(without)[0]


@pytest.mark.parametrize(
    "cpu, gpu, refused",
    [
        (3000, 45.0, False),  # exactly the limit passes
        (2999.9, 44.9, False),
        (3000.001, 0.0, True),
        (0.0, 45.001, True),
        (3600, 60.0, True),
    ],
)
def test_the_budget_refuses_an_estimate_above_either_limit(cpu, gpu, refused):
    estimate = {"cpu_seconds": cpu, "gpu_minutes": gpu}
    if refused:
        with pytest.raises(timing.TimingBudgetExceeded, match="nothing ran"):
            timing.require_within_budget(estimate)
    else:
        timing.require_within_budget(estimate)


@pytest.mark.parametrize(
    "estimate",
    [
        {"cpu_seconds": float("nan"), "gpu_minutes": 0.0},
        {"cpu_seconds": -1.0, "gpu_minutes": 0.0},
        {"cpu_seconds": True, "gpu_minutes": 0.0},
        {"gpu_minutes": 0.0},
    ],
)
def test_a_malformed_estimate_is_a_value_error(estimate):
    with pytest.raises(ValueError):
        timing.require_within_budget(estimate)


@pytest.mark.parametrize(
    "priors, limit, mps, refused",
    [
        ({}, DEFAULT_CPU_LIMIT, True, True),  # 3,578 CPU-seconds: the plan as D-054 leaves it
        ({}, DEFAULT_CPU_LIMIT, False, False),  # 2,050 CPU-seconds without MPS
        ({"M5": (1400.0, 0.0)}, RAISED_CPU_LIMIT, True, True),  # 4,768 CPU-seconds
        ({"M4": (120.0, 41.0)}, RAISED_CPU_LIMIT, True, True),  # 53.2 GPU-minutes
        ({"M4": (120.0, 41.0)}, RAISED_CPU_LIMIT, False, False),  # M4 is left out without MPS
    ],
)
def test_a_run_above_the_budget_is_refused_before_anything_runs(
    world, tmp_path, monkeypatch, priors, limit, mps, refused
):
    monkeypatch.setattr(timing, "PRIORS", {**timing.PRIORS, **priors})
    monkeypatch.setattr(timing, "TIMING_MAX_CPU_SECONDS", limit)
    world.mps = mps
    if not refused:
        report, _ = run_timing(world, tmp_path)
        assert report["status"] == "completed"
        return
    printed: list[str] = []
    with pytest.raises(timing.TimingBudgetExceeded):
        timing.run_timing(
            params=PARAMS,
            steps=world.steps(),
            root=tmp_path / "ft",
            directory=tmp_path / "reports",
            progress=printed.append,
        )
    assert printed[0].startswith("timing estimate")  # the estimate is printed first
    assert world.events == []
    assert not (tmp_path / "ft").exists() and not (tmp_path / "reports").exists()


# == the owner's approval: gate 3 as a ceiling (D-055) =============================================

APPROVAL_TEXT = (
    f"the owner approved 1.5 CPU-hours (--approved-by-owner given at {GIVEN_AT_TEXT}, the "
    "command's start): a CPU limit of 5400 CPU-seconds instead of the step's own 3000, above the "
    "gate-3 limit (3600 CPU-seconds) under the owner's approval, a ceiling the spend never "
    "passes: the run is planned and stopped against its stop limit of 4800 CPU-seconds, a reserve "
    "of 600 below it for the unit running then; estimate about 3578 CPU-seconds and 33.2 "
    "GPU-minutes"
)
APPROVED_STOP_TEXT = (  # how a stop under the approval names its limit
    "above the stop limit of 4800 (the 5400 the owner approved minus a reserve of 600)"
)


def approved_run(
    world: FakeWorld, tmp_path: Path, monkeypatch, hours: float | None = APPROVED_HOURS
) -> tuple[dict[str, Any], list]:
    """A run with the step's own default limit (3,000 CPU-seconds) and ``hours`` approved."""
    monkeypatch.setattr(timing, "TIMING_MAX_CPU_SECONDS", DEFAULT_CPU_LIMIT)
    return run_timing(world, tmp_path, approved_cpu_hours=hours, given_at=GIVEN_AT)


def test_a_budget_is_the_steps_own_or_the_cpu_hours_the_owner_approved():
    own = timing.timing_budget()
    assert own == timing.TimingBudget(3000, 45.0)
    assert (own.approved_cpu_hours, own.given_at, own.above_gate3) == (None, None, False)
    # the step's own limit is where its run stops; gate 3 is 600 CPU-seconds above it
    assert (own.stop_cpu_seconds, own.reserve_cpu_seconds) == (3000, 0.0)
    assert timing.timing_budget(given_at=GIVEN_AT) == own  # a start without a figure approves none
    approved = timing.timing_budget(APPROVED_HOURS, given_at=GIVEN_AT)
    assert approved == timing.TimingBudget(5400.0, 45.0, 1.5, GIVEN_AT_TEXT)
    assert approved.above_gate3 is True  # 5,400 CPU-seconds against the 3,600 of gate 3
    # the approved ceiling is never passed: the run works 600 CPU-seconds below it, more than the
    # most one unit added past its prior in the first run (M6 with 4 workers: 289 against 75)
    assert timing.APPROVAL_RESERVE_CPU_SECONDS == GATE3_CPU_SECONDS - DEFAULT_CPU_LIMIT == 600
    assert (approved.stop_cpu_seconds, approved.reserve_cpu_seconds) == (APPROVED_STOP_LIMIT, 600)
    assert max(M6_FIRST_RUN.values()) - timing.PRIORS["M6"][0] / 4 < 600
    # the time the flag was given is the command's start, in UTC whatever its zone
    elsewhere = GIVEN_AT.astimezone(timezone(timedelta(hours=-3)))
    assert timing.timing_budget(APPROVED_HOURS, given_at=elsewhere).given_at == GIVEN_AT_TEXT
    exactly = timing.timing_budget(1, given_at=GIVEN_AT)  # exactly gate 3 is not above it
    assert (exactly.cpu_seconds, exactly.above_gate3) == (3600.0, False)
    assert exactly.stop_cpu_seconds == 3000.0
    # an approval smaller than the reserve leaves nothing to plan for
    assert timing.timing_budget(0.1, given_at=GIVEN_AT).stop_cpu_seconds == 0.0


@pytest.mark.parametrize(
    "hours, given_at",
    [
        (0, GIVEN_AT),
        (-1.5, GIVEN_AT),
        (float("nan"), GIVEN_AT),
        (float("inf"), GIVEN_AT),
        (True, GIVEN_AT),
        ("1.5", GIVEN_AT),
        (1.5, None),  # an approval needs the time it was given
        (1.5, datetime(2026, 11, 2, 22, 0, 5)),  # a naive time
    ],
)
def test_a_malformed_approval_is_a_value_error(hours, given_at):
    with pytest.raises(ValueError):
        timing.timing_budget(hours, given_at=given_at)


@pytest.mark.parametrize(
    "fields",
    [
        (3601, 45.0),  # a CPU limit above gate 3 without the owner's approval
        (3000, 61.0),  # a GPU limit above gate 3: never approved here
        (3000, 45.0, None, GIVEN_AT_TEXT),  # a time without a figure
        (5400.0, 45.0, 1.4, GIVEN_AT_TEXT),  # not the approved figure x 3600
        (5400.0, 45.0, 1.5, None),  # no time
        (float("nan"), 45.0),
    ],
)
def test_a_budget_above_gate_3_exists_only_under_the_owners_approval(fields):
    with pytest.raises(ValueError):
        timing.TimingBudget(*fields)


def test_the_estimate_is_checked_against_the_approved_ceiling():
    approved = timing.timing_budget(APPROVED_HOURS, given_at=GIVEN_AT)
    with_mps = timing.estimate_plan(timing.timing_plan(), mps_available=True)  # 3,578 CPU-seconds
    with pytest.raises(timing.TimingBudgetExceeded, match="which keeps it below gate 3"):
        timing.require_within_budget(with_mps)  # the step's own 3,000
    timing.require_within_budget(with_mps, approved)
    lines = timing.estimate_lines(with_mps, approved)
    assert lines[0].endswith(
        "about 3578 CPU-seconds and 33.2 GPU-minutes; budget 5400 CPU-seconds and 45 GPU-minutes: "
        f"the 1.5 CPU-hours the owner approved (--approved-by-owner given at {GIVEN_AT_TEXT}), "
        "above gate 3 (3600 CPU-seconds, 60 GPU-minutes) under that approval; a ceiling the spend "
        "never passes: the run is planned and stopped against its stop limit of 4800 CPU-seconds, "
        "a reserve of 600 below it for the unit running then"
    )
    assert len(lines) == 1 + len(with_mps["per_measurement"])  # no "above the budget" line
    # a ceiling, without tolerance: the estimate must fit the stop limit 600 CPU-seconds below it
    # (exactly it passes); between the stop limit and the ceiling, or above 45 GPU-minutes, it is
    # refused
    timing.require_within_budget(
        {"cpu_seconds": APPROVED_STOP_LIMIT, "gpu_minutes": 45.0}, approved
    )
    for cpu, gpu in ((4800.001, 0.0), (5400.0, 0.0), (5400.001, 0.0), (0.0, 45.001)):
        estimate = {"cpu_seconds": cpu, "gpu_minutes": gpu}
        with pytest.raises(timing.TimingBudgetExceeded, match="does not cover it; nothing ran"):
            timing.require_within_budget(estimate, approved)
        assert timing.estimate_lines({**estimate, "per_measurement": {}}, approved)[1] == (
            "  above the budget: `finetune timing` refuses to start"
        )
    below = timing.timing_budget(0.9, given_at=GIVEN_AT)
    with pytest.raises(
        timing.TimingBudgetExceeded,
        match=r"about 3578 CPU-seconds and 33\.2 GPU-minutes, which does not fit its budget of "
        r"3240 CPU-seconds and 45 GPU-minutes: the 0\.9 CPU-hours the owner approved",
    ):
        timing.require_within_budget(with_mps, below)
    lines = timing.estimate_lines(with_mps, below)
    assert "within gate 3 (3600 CPU-seconds, 60 GPU-minutes)" in lines[0]
    assert lines[1] == "  above the budget: `finetune timing` refuses to start"


@pytest.mark.parametrize(
    "hours, priors, runs",
    [
        (None, {}, False),  # 3,578 CPU-seconds, above the step's own 3,000
        (APPROVED_HOURS, {}, True),  # within the 5,400 the owner approved
        (0.9, {}, False),  # 3,240 approved: below the estimate
        (1.3, {"M5": (1400.0, 0.0)}, False),  # 4,768 CPU-seconds against 4,680 approved
        (APPROVED_HOURS, {"M5": (1400.0, 0.0)}, True),  # 4,768: within the stop limit of 4,800
        (APPROVED_HOURS, {"M5": (1450.0, 0.0)}, False),  # 4,818: within 5,400, above 4,800
        (APPROVED_HOURS, {"M4": (120.0, 41.0)}, False),  # 53.2 GPU-minutes: the GPU limit stays
    ],
)
def test_an_estimate_above_the_steps_own_limit_runs_only_within_an_approved_ceiling(
    world, tmp_path, monkeypatch, hours, priors, runs
):
    monkeypatch.setattr(timing, "PRIORS", {**timing.PRIORS, **priors})
    if runs:
        report, printed = approved_run(world, tmp_path, monkeypatch, hours)
        assert report["status"] == "completed"
        assert report["approval"]["approved_cpu_hours"] == hours
        assert "the 1.5 CPU-hours the owner approved" in printed[0]
        return
    monkeypatch.setattr(timing, "TIMING_MAX_CPU_SECONDS", DEFAULT_CPU_LIMIT)
    printed: list[str] = []
    with pytest.raises(timing.TimingBudgetExceeded, match="nothing ran"):
        timing.run_timing(
            params=PARAMS,
            steps=world.steps(),
            root=tmp_path / "ft",
            directory=tmp_path / "reports",
            progress=printed.append,
            approved_cpu_hours=hours,
            given_at=GIVEN_AT,
        )
    assert printed[0].startswith("timing estimate")  # the estimate with the budget, first
    assert printed[1] == "  above the budget: `finetune timing` refuses to start"
    assert world.events == []
    assert not (tmp_path / "ft").exists() and not (tmp_path / "reports").exists()


def test_with_the_approval_the_spend_goes_past_the_steps_own_limit_up_to_the_ceiling(
    world, tmp_path, monkeypatch
):
    """One child's 3,300 CPU-seconds take the spend past the step's own 3,000 and, with the rest,
    past gate 3: under the owner's 1.5 CPU-hours the run goes on and completes (a run without
    the approval stops there: test_the_run_stops_once_the_measured_spend_passes_the_limit)."""
    world.overrides[("timing", "cpu", 8, 8, False)] = {"cpu": 3300.0}
    report, printed = approved_run(world, tmp_path, monkeypatch)
    assert report["status"] == "completed" and report["failures"] == []
    spend = report["spend"]
    assert spend["stopped"] is False and spend["limit_cpu_seconds"] == APPROVED_CPU_LIMIT
    assert GATE3_CPU_SECONDS < spend["cpu_seconds"] <= APPROVED_STOP_LIMIT
    assert [item["status"] for item in report["measurements"]["M6"]] == ["completed"] * 4
    assert any(line.endswith("of 5400 CPU-seconds, 0.0 of 45 GPU-minutes") for line in printed)
    markdown = read_markdown(tmp_path / "reports")
    assert (
        f"The run's spend, {spend['cpu_seconds']:.0f} CPU-seconds, stayed within the 5400 "
        "CPU-seconds the owner approved and passed the gate-3 limit (3600 CPU-seconds)." in markdown
    )


def test_the_record_states_the_owners_approval(world, tmp_path, monkeypatch):
    report, printed = approved_run(world, tmp_path, monkeypatch)
    assert report["status"] == "completed"
    estimate = report["estimate"]
    assert estimate["cpu_seconds"] == pytest.approx(3577.52)
    assert report["approval"] == {
        "approved_cpu_hours": 1.5,
        "given_at": GIVEN_AT_TEXT,  # the command's start, when the flag was given
        "limit_cpu_seconds": 5400.0,
        "reserve_cpu_seconds": 600.0,
        "stop_cpu_seconds": 4800.0,
        "own_limit_cpu_seconds": 3000,
        "gate3_cpu_seconds": 3600,
        "above_gate3": True,
        "estimate_cpu_seconds": estimate["cpu_seconds"],
        "estimate_gpu_minutes": estimate["gpu_minutes"],
    }
    spend = report["spend"]
    assert (spend["limit_cpu_seconds"], spend["limit_gpu_minutes"]) == (5400.0, 45.0)
    assert (spend["gate3_cpu_seconds"], spend["gate3_gpu_minutes"]) == (3600, 60.0)
    assert timing.load_timing(directory=tmp_path / "reports") == report
    # printed once the estimate passed its ceiling, before anything ran
    assert printed.index(f"gate 3: {APPROVAL_TEXT}") < min(
        index for index, line in enumerate(printed) if line.startswith("synthetic export")
    )
    summary = timing.summary_lines(report)
    assert summary[0] == (
        f"timing completed: spent {spend['cpu_seconds']:.0f} of 5400 CPU-seconds and "
        f"{spend['gpu_minutes']:.1f} of 45 GPU-minutes"
    )
    assert summary[1] == f"gate 3: {APPROVAL_TEXT}"
    markdown = read_markdown(tmp_path / "reports")
    assert (
        "limits 5400 CPU-seconds, the ceiling the owner approved (gate 3 below), and 45 "
        "GPU-minutes. Stopped: no." in markdown
    )
    assert f"Gate 3: {APPROVAL_TEXT}. The CPU limit is the ceiling the owner approved" in markdown
    assert (
        f"The run's spend, {spend['cpu_seconds']:.0f} CPU-seconds, stayed within the 5400 "
        "CPU-seconds the owner approved and stayed below the gate-3 limit (3600 CPU-seconds)."
        in markdown
    )
    # the Markdown compares the spend with the ceiling itself, and would say so if it passed it
    above = timing._approval_paragraph(report["approval"], {"cpu_seconds": 5401.0})
    assert above.endswith(
        "The run's spend, 5401 CPU-seconds, passed the 5400 CPU-seconds the owner approved and "
        "passed the gate-3 limit (3600 CPU-seconds)."
    )


def test_without_the_flag_nothing_changes(world, tmp_path, monkeypatch):
    """The step's own limits below gate 3, as before D-055: the same estimate line, limits and
    Markdown, and a record whose approval is null."""
    monkeypatch.setattr(timing, "TIMING_MAX_CPU_SECONDS", DEFAULT_CPU_LIMIT)
    world.mps = False  # 2,050 CPU-seconds: within the step's own 3,000
    report, printed = run_timing(world, tmp_path, given_at=GIVEN_AT)  # a start approves nothing
    assert report["status"] == "completed" and report["approval"] is None
    spend = report["spend"]
    assert (spend["limit_cpu_seconds"], spend["limit_gpu_minutes"]) == (3000, 45.0)
    assert printed[0].endswith(
        "about 2050 CPU-seconds and 0.0 GPU-minutes; budget 3000 CPU-seconds and 45 GPU-minutes, "
        "below gate 3 (3600 CPU-seconds, 60 GPU-minutes)"
    )
    summary = timing.summary_lines(report)
    assert summary[0].startswith("timing completed: spent ") and summary[0].endswith(
        " of 3000 CPU-seconds and 0.0 of 45 GPU-minutes"
    )
    assert not any(line.startswith("gate 3:") for line in [*printed, *summary])
    markdown = read_markdown(tmp_path / "reports")
    assert (
        "limits 3000 CPU-seconds and 45 GPU-minutes, below gate 3 (3600 CPU-seconds, 60 "
        "GPU-minutes). Stopped: no." in markdown
    )
    assert "Gate 3:" not in markdown and "--approved-by-owner" not in markdown
    # the budget the functions take by default is the step's own
    assert timing.estimate_lines(report["estimate"]) == timing.estimate_lines(
        report["estimate"], timing.timing_budget()
    )


def test_a_budget_change_is_not_a_changed_study(world, tmp_path, monkeypatch):
    """The plan a run without the flag records is the plan of a run with it: the second run
    verifies the same timing-2.config.json instead of being refused as a changed study."""
    monkeypatch.setattr(timing, "TIMING_MAX_CPU_SECONDS", DEFAULT_CPU_LIMIT)
    world.mps = False
    first, _ = run_timing(world, tmp_path)
    config = tmp_path / "reports" / f"{timing.TIMING_STEM}.config.json"
    recorded = config.read_bytes()
    world.mps = True
    second, printed = run_timing(
        world, tmp_path, approved_cpu_hours=APPROVED_HOURS, given_at=GIVEN_AT
    )
    assert (first["approval"], second["approval"]["approved_cpu_hours"]) == (None, 1.5)
    assert second["status"] == "completed"
    assert config.read_bytes() == recorded
    assert first["config_sha256"] == second["config_sha256"]
    digest = first["config_sha256"][:12]
    assert f"timing plan timing-2.config.json (sha256 {digest}) verified" in printed


# == a whole run ===================================================================================


def test_a_whole_run_measures_everything_in_order_and_writes_the_report(world, tmp_path):
    report, printed = run_timing(world, tmp_path)
    assert report["status"] == "completed"
    assert report["failures"] == [] and report["spend"]["stopped"] is False
    assert world.events == [
        "load small",
        "identity",
        "nll 48",
        "fan t(4)",
        "fan t(8)",
        "fan t(24)",
        *[
            f"child timing cpu x{threads} b{batch}"
            for threads in (2, 8)
            for batch in timing.M1_BATCHES
        ],
        "child preflight mps x2 b32",
        *[f"child timing mps x2 b{batch}" for batch in timing.M1_BATCHES],
        "start timing mps x2 b32",
        "workers 1 during M4",
        "workers 2 during M4",
        "workers 1",
        "workers 2",
        "workers 3",
        "workers 4",
    ]
    measurements = report["measurements"]
    assert [item["status"] for item in measurements["M1"]] == ["completed"] * 9
    entry = m1(report, "cpu", 8, 16)
    assert entry["seconds_per_step"]["median"] == pytest.approx(2.5)
    assert entry["windows_per_second"] == pytest.approx(16 / 2.5)
    assert entry["timed_steps"] == timing.M1_TIMED_STEPS
    assert m1(report, "mps", 2, 32)["peak_bytes"] == 3 * GIB  # RSS plus the MPS driver
    assert m1(report, "cpu", 2, 32)["peak_bytes"] == 2 * GIB
    # batch 32 fits everywhere: no activation checkpointing, no retry
    assert measurements["M2"] == {"gradient_check": None, "runs": []}
    assert report["mps_checkpointing"] == {"from_batch": None, "reason": None}
    for item in [*measurements["M1"], measurements["M3"], measurements["M4"]]:
        assert item["activation_checkpointing"] is False
        assert "first_attempt" not in item and "retry" not in item
    assert report["spend"]["per_measurement"]["retries"] == {"cpu_seconds": 0.0, "gpu_minutes": 0.0}
    assert measurements["M3"]["passed"] is True and measurements["M3"]["status"] == "completed"
    m4 = measurements["M4"]
    assert m4["status"] == "completed" and m4["theta"] == pytest.approx(1.1)
    assert m4["median_last_seconds"] == pytest.approx(2.2)
    assert [item["status"] for item in measurements["M7"]] == ["completed"] * 2
    for item in measurements["M7"]:
        # between the end of M4's first window and the start of its last one
        assert 180.0 <= item["m4_elapsed_start_seconds"] < item["m4_elapsed_end_seconds"] < 630
        assert item["slowdown_vs_m6"] == pytest.approx(1.25)
    assert [item["status"] for item in measurements["M6"]] == ["completed"] * 4
    m5 = measurements["M5"]
    assert (m5["t4"], m5["t8"], m5["t24"]) == pytest.approx((3.152, 5.864, 16.712))
    assert m5["fit"] == pytest.approx({"intercept": 0.44, "slope": 0.678})
    assert measurements["M8"]["decisions"] == 48
    assert measurements["M8"]["seconds_per_decision"] == pytest.approx(7.0 / 48)
    assert measurements["M8"]["planned_decisions"] == 336
    assert measurements["M9"]["status"] == "completed" and measurements["M9"]["identical"]
    choice = report["choice"]
    assert (choice["device"], choice["threads"], choice["batch_size"]) == ("mps", 2, 32)
    assert choice["deterministic"] is True and choice["activation_checkpointing"] is False
    assert choice["mps_watermark_ratio"] == 1.0 and choice["batch_32_fits"] is True
    assert choice["theta"] == pytest.approx(1.1)
    assert choice["speedups"]["mps_vs_best_cpu"] == pytest.approx(5.0 / 2.2)
    assert choice["speedups"]["eight_vs_two_threads"] == pytest.approx(6.1 / 5.0)
    assert choice["best_cpu"] == {
        "configuration": "CPU x8 as is, 5.000 s per step",
        "seconds_per_step": 5.0,
    }
    assert choice["mps_sustained_seconds_per_step"] == pytest.approx(2.2)
    # every training child: the real child's request, every R1 wrapper on, early stopping off
    for request in world.requests:
        assert request["early_stopping"] is False and request["memory_guard"] is True
        wrappers = request["wrappers"]
        assert all(getattr(wrappers, name) for name in ("device", "router", "pseudo_epochs"))
        assert request["pseudo_epoch_steps"] == timing.TIMING_PSEUDO_EPOCH_STEPS
        assert request["window_rows"] == 513 and request["fold"] == timing.TIMING_FOLD_LABEL
    m1_requests = [item for item in world.requests if item["key"][0] == "timing"][:9]
    assert all(item["timing"].warmup_steps == 3 for item in m1_requests)
    assert all(item["timing"].timed_steps == 10 for item in m1_requests)
    m4_request = world.requests[-1]
    assert m4_request["timing"].timed_seconds == timing.M4_SECONDS
    # the report: written through the report layer, the plan recorded once beside it
    directory = tmp_path / "reports"
    assert sorted(path.name for path in directory.iterdir()) == sorted(report_names())
    assert report_names() == ["timing-2.config.json", "timing-2.json", "timing-2.md"]
    assert timing.load_timing(directory=directory) == report
    assert report["study"] == "timing-2" and report["tests_consumed"] == 0
    assert report["statistics_with_intervals"] == 0
    assert report["environment"]["platform"] == default_label()
    assert set(report["code"]) >= {
        "training_code_sha256",
        "forecasting_code_sha256",
        "vendored_manifest_sha256",
        "source_tree_sha256",
    }
    assert report["scratch"].startswith("timing-")
    assert (tmp_path / "ft" / ".scratch" / report["scratch"] / "export" / "train.csv").is_file()
    assert printed[0].startswith("timing estimate")
    markdown = read_markdown(directory)
    assert markdown.startswith("# Measurement step of the retraining (finetune timing)\n")
    for heading in ("## Budget", "## Choice", "## M1", "## M3", "## M4", "## M5", "## M8"):
        assert heading in markdown
    assert "## Activation checkpointing on MPS and retries" in markdown
    assert "none was retried" in markdown


def test_the_spend_is_what_each_measurement_reports(world, tmp_path):
    report, _ = run_timing(world, tmp_path)
    spent = report["spend"]["per_measurement"]
    # in this process: wall x the 2 governed threads (D-033)
    assert spent["M9"]["cpu_seconds"] == pytest.approx(9.0 * 2)
    assert spent["M8"]["cpu_seconds"] == pytest.approx(7.0 * 2)
    fans = sum(4 * (0.44 + 0.678 * length) for length in (4, 8, 24))
    assert spent["M5"]["cpu_seconds"] == pytest.approx(fans * 2)
    # a training child: its cpu_seconds; an MPS child's wall as GPU time
    assert spent["M1"]["cpu_seconds"] == pytest.approx(9 * 10.0)
    assert spent["M1"]["gpu_minutes"] == pytest.approx(3 * 30.0 / 60)
    assert spent["M3"]["gpu_minutes"] == pytest.approx(0.5)
    # the watch polls every 5 s, so the child is seen ending within a poll of its lifetime
    assert spent["M4"]["gpu_minutes"] == pytest.approx(M4_LIFETIME / 60, abs=5 / 60)
    assert spent["M4"]["cpu_seconds"] == pytest.approx(0.1 * M4_LIFETIME, abs=0.5)
    total = sum(item["cpu_seconds"] for item in spent.values())
    assert report["spend"]["cpu_seconds"] == pytest.approx(total)
    assert report["spend"]["cpu_seconds"] <= timing.TIMING_MAX_CPU_SECONDS


@pytest.mark.parametrize("hours", [None, APPROVED_HOURS])
def test_child_deadlines_and_timeouts_follow_the_remaining_budget(world, tmp_path, hours):
    """The CPU limit the deadlines leave room for is the step's own, or with the owner's approval
    the stop limit 600 CPU-seconds below the ceiling it approved (D-055); the GPU limit is 45
    minutes either way."""
    report, _ = run_timing(world, tmp_path, approved_cpu_hours=hours, given_at=GIVEN_AT)
    spent = report["spend"]["per_measurement"]
    cpu = sum(spent[name]["cpu_seconds"] for name in ("loads", "M9", "M8", "M5"))
    gpu = 0.0
    ceiling = timing.TIMING_MAX_CPU_SECONDS if hours is None else APPROVED_CPU_LIMIT
    limit = timing.TIMING_MAX_CPU_SECONDS if hours is None else APPROVED_STOP_LIMIT
    assert report["spend"]["limit_cpu_seconds"] == ceiling
    assert report["spend"]["limit_gpu_minutes"] == 45.0
    for request in world.requests:
        _, device, threads, _, _ = request["key"]
        if "timeout" not in request:  # the M4 child: the remaining GPU time, its CPU watched
            assert request["deadline"] == pytest.approx((45.0 - gpu) * 60.0)
            continue
        expected = (limit - cpu) / threads
        if device == "mps":
            expected = min((45.0 - gpu) * 60.0, (limit - cpu) / 2)
        assert request["deadline"] == pytest.approx(expected)
        assert request["timeout"] == pytest.approx(expected + 60.0)
        cpu += 10.0
        gpu += 0.5 if device == "mps" else 0.0


# == the MPS preflight =============================================================================


def test_the_mps_preflight_runs_before_any_mps_timing(world, tmp_path):
    run_timing(world, tmp_path)
    mps_events = [event for event in children(world) if " mps " in event]
    assert mps_events[0] == "child preflight mps x2 b32"
    (preflight,) = [item for item in world.requests if item["key"][:2] == ("preflight", "mps")]
    assert preflight["checks"] == timing.M3_CHECKS
    assert preflight["deterministic"] is False  # check (d) asks deterministic mode itself
    assert all(item["deterministic"] for item in world.requests if item is not preflight)
    # the CPU's M1 runs first: whether batch 32 needs activation checkpointing comes from it
    events = children(world)
    assert events.index("child timing cpu x8 b32") < events.index("child preflight mps x2 b32")


@pytest.mark.parametrize("failing", ["multinomial", "parity", "steps"])
def test_no_mps_timing_runs_after_a_failed_preflight(world, tmp_path, failing):
    world.preflight[failing] = {"passed": False}
    report, _ = run_timing(world, tmp_path)
    assert [event for event in children(world) if " mps " in event] == [
        "child preflight mps x2 b32"
    ]
    measurements = report["measurements"]
    assert measurements["M3"]["passed"] is False
    for batch in timing.M1_BATCHES:
        entry = m1(report, "mps", 2, batch)
        assert entry["status"] == "not_run" and "preflight (M3) did not pass" in entry["reason"]
    assert measurements["M4"]["status"] == "not_run"
    assert [item["status"] for item in measurements["M7"]] == ["not_run"] * 2
    assert report["choice"]["device"] == "cpu"
    assert report["status"] == "completed"  # an unusable MPS is a finding, not a failure


def test_a_refused_deterministic_mode_turns_it_off_on_mps(world, tmp_path):
    world.preflight["deterministic"] = {"passed": False, "refused": True, "op": "some_op"}
    report, _ = run_timing(world, tmp_path)
    for request in world.requests[1:]:
        assert request["deterministic"] is (request["key"][1] == "cpu")
    assert report["measurements"]["M3"]["deterministic_refused"] is True
    assert report["choice"]["device"] == "mps" and report["choice"]["deterministic"] is False


def test_without_mps_the_mps_measurements_are_recorded_as_not_run(world, tmp_path):
    world.mps = False
    report, _ = run_timing(world, tmp_path)
    assert not [event for event in world.events if "mps" in event or "during" in event]
    measurements = report["measurements"]
    assert measurements["M3"]["status"] == "not_run"
    assert "not available" in measurements["M3"]["reason"]
    assert measurements["M4"]["status"] == "not_run"
    assert report["estimate"]["cpu_seconds"] == pytest.approx(2050.0)
    assert report["choice"]["device"] == "cpu" and report["choice"]["mps_watermark_ratio"] is None
    assert report["status"] == "completed"


# == activation checkpointing on MPS: from the CPU's M1 and M2, or one retry ======================

CPU_M1 = [f"child timing cpu x{threads} b{batch}" for threads in (2, 8) for batch in (8, 16, 32)]


@pytest.mark.parametrize("m4_step, device", [(2.0, "mps"), (5.0, "cpu")])
def test_when_the_cpu_needs_checkpointing_mps_runs_batch_32_with_it_from_the_start(
    world, tmp_path, m4_step, device
):
    world.overrides.update(CPU_NEEDS_CHECKPOINTING)
    world.m4_step = m4_step
    report, printed = run_timing(world, tmp_path)
    assert report["status"] == "completed" and report["failures"] == []
    assert children(world) == [
        *CPU_M1,
        "child timing cpu x2 b32 ckpt",  # M2 on the chosen CPU configuration
        "child preflight cpu x2 b32",  # its gradient bit-identity
        "child preflight mps x2 b32 ckpt",  # M3 at batch 32 with activation checkpointing
        "child timing mps x2 b8",
        "child timing mps x2 b16",
        "child timing mps x2 b32 ckpt",
        "start timing mps x2 b32 ckpt",  # M4 as M1's batch 32
    ]
    reason = (
        "the batch 32 of CPU x2 does not fit as is (M1) and activation checkpointing passed "
        "there (M2)"
    )
    assert report["mps_checkpointing"] == {"from_batch": 32, "reason": reason}
    assert f"MPS at batch 32 with activation checkpointing: {reason}" in printed
    measurements = report["measurements"]
    for entry in (measurements["M3"], m1(report, "mps", 2, 32), measurements["M4"]):
        assert entry["activation_checkpointing"] is True and entry["checkpointing_reason"] == reason
        assert "first_attempt" not in entry  # nothing aborted: no retry
    assert measurements["M3"]["passed"] is True
    for batch in (8, 16):
        assert m1(report, "mps", 2, batch)["activation_checkpointing"] is False
    (mps_slope,) = [item for item in report["memory_slopes"] if item["device"] == "mps"]
    assert mps_slope["batches"] == [8, 16]  # the checkpointed batch 32 is not on the slope
    (m2,) = measurements["M2"]["runs"]  # the CPU's only: MPS's batch 32 was never run as is
    assert (m2["device"], m2["threads"], m2["passed"]) == ("cpu", 2, True)
    choice = report["choice"]
    best = "CPU x2 with activation checkpointing, 7.930 s per step"  # x8 does not fit as is
    assert choice["best_cpu"] == {"configuration": best, "seconds_per_step": pytest.approx(7.93)}
    assert choice["speedups"]["mps_vs_best_cpu"] == pytest.approx(7.93 / (m4_step * 1.1))
    assert choice["device"] == device and choice["activation_checkpointing"] is True
    assert choice["batch_32_fits"] is False and choice["threads"] == 2
    if device == "mps":  # MPS with activation checkpointing, at least 1.5 times faster
        assert choice["seconds_per_step"] == pytest.approx(2.0 * 1.3)
        assert choice["theta"] == pytest.approx(1.1)
        assert f"batch 32 runs with activation checkpointing on MPS: {reason}" in choice["reasons"]
    else:  # 7.93 / 5.5 = 1.44: the CPU with activation checkpointing (M2)
        assert choice["seconds_per_step"] == pytest.approx(6.1 * 1.3)
        assert any("only 1.44 times faster" in item and best in item for item in choice["reasons"])


def test_a_memory_abort_of_the_mps_preflight_is_retried_once_with_checkpointing(world, tmp_path):
    world.overrides[("preflight", "mps", 2, 32, False)] = dict(MEMORY_ABORT)
    report, printed = run_timing(world, tmp_path)
    assert report["status"] == "completed" and report["failures"] == []
    assert children(world) == [
        *CPU_M1,  # batch 32 fits on the CPU: M3 starts as designed, without the wrap
        "child preflight mps x2 b32",  # attempt 1: memory abort
        "child preflight cpu x2 b32",  # M2's gradient bit-identity, before the retry relies on it
        "child preflight mps x2 b32 ckpt",  # attempt 2
        "child timing mps x2 b8",
        "child timing mps x2 b16",
        "child timing mps x2 b32 ckpt",  # from batch 32 up with activation checkpointing
        "start timing mps x2 b32 ckpt",
    ]
    reason = "M3 MPS preflight ended memory_abort without it"
    measurements = report["measurements"]
    m3 = measurements["M3"]
    assert (m3["status"], m3["passed"], m3["activation_checkpointing"]) == ("completed", True, True)
    assert m3["checkpointing_reason"] == reason and m3["checks"]["parity"]["passed"] is True
    first = m3["first_attempt"]
    assert (first["status"], first["activation_checkpointing"]) == ("memory_abort", False)
    assert first["error"] == MPS_OUT_OF_MEMORY and first["checks"] == {}
    assert first["passed"] is False and first["child_status"] == "memory_abort"
    assert first["peak_bytes"] == 12 * GIB and first["swap_growth_bytes"] == 4 * GIB
    assert first["run_id"] != m3["run_id"]
    assert report["mps_checkpointing"] == {"from_batch": 32, "reason": reason}
    assert measurements["M2"]["gradient_check"]["passed"] is True
    assert measurements["M2"]["runs"] == []
    assert m1(report, "mps", 2, 32)["checkpointing_reason"] == reason
    # attempt 1 is M3's, the retry is charged to the retries
    spent = report["spend"]["per_measurement"]
    assert spent["M3"] == {"cpu_seconds": 10.0, "gpu_minutes": pytest.approx(100.0 / 60)}
    assert spent["retries"] == {"cpu_seconds": 10.0, "gpu_minutes": pytest.approx(0.5)}
    assert spent["M2"]["cpu_seconds"] == 10.0  # the gradient check
    choice = report["choice"]
    assert (choice["device"], choice["activation_checkpointing"]) == ("mps", True)
    assert choice["seconds_per_step"] == pytest.approx(2.6)
    assert f"batch 32 runs with activation checkpointing on MPS: {reason}" in choice["reasons"]
    assert (
        "M3 MPS preflight: memory abort without activation checkpointing; retried once with it"
        in printed
    )
    assert "M3: memory_abort without activation checkpointing; retried once with it: completed" in (
        timing.summary_lines(report)
    )
    # both attempts in the Markdown: the retries table and the M3 section
    markdown = read_markdown(tmp_path / "reports")
    assert f"MPS units from batch 32 up ran with activation checkpointing: {reason}." in markdown
    assert "| M3 | 1 | no | memory_abort | 12.00 | 4.00 | RuntimeError: MPS backend out of" in (
        markdown
    )
    assert "| M3 | 2 (retry) | yes | completed | 3.00 | 0.00 | - |" in markdown
    assert (
        "Attempt 1, without activation checkpointing: memory_abort (RuntimeError: MPS backend "
        "out of memory" in markdown
    )
    assert "| mps | 2 | 32 | yes | 1 | completed |" in markdown


def test_a_memory_abort_of_an_mps_batch_is_retried_and_the_larger_ones_use_checkpointing(
    world, tmp_path
):
    world.overrides[("timing", "mps", 2, 16, False)] = dict(MEMORY_ABORT)
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "completed"
    assert children(world)[len(CPU_M1) :] == [
        "child preflight mps x2 b32",
        "child timing mps x2 b8",
        "child timing mps x2 b16",  # attempt 1: memory abort
        "child preflight cpu x2 b32",  # the gradient check
        "child timing mps x2 b16 ckpt",  # attempt 2
        "child timing mps x2 b32 ckpt",  # from batch 16 up with activation checkpointing
        "start timing mps x2 b32 ckpt",
    ]
    reason = "M1 mps x2 batch 16 ended memory_abort without it"
    assert report["mps_checkpointing"] == {"from_batch": 16, "reason": reason}
    b16 = m1(report, "mps", 2, 16)
    assert (b16["status"], b16["activation_checkpointing"]) == ("completed", True)
    assert b16["seconds_per_step"]["median"] == pytest.approx(2.0 / 2 * 1.3)
    assert (b16["first_attempt"]["status"], b16["first_attempt"]["timed_steps"]) == (
        "memory_abort",
        0,
    )
    b32 = m1(report, "mps", 2, 32)
    assert b32["activation_checkpointing"] is True and "first_attempt" not in b32
    assert b32["checkpointing_reason"] == reason
    (mps_slope,) = [item for item in report["memory_slopes"] if item["device"] == "mps"]
    assert mps_slope["batches"] == [8]
    markdown = read_markdown(tmp_path / "reports")
    assert "| mps | 2 | 16 | no | 1 | memory_abort |" in markdown
    assert "| mps | 2 | 16 | yes | 2 (retry) | completed |" in markdown
    assert "| M1 mps x2 batch 16 | 1 | no | memory_abort |" in markdown


def test_a_memory_abort_of_m4_is_retried_with_checkpointing_and_m7_runs_again(world, tmp_path):
    world.m4_overrides[False] = {"lifetime": 400.0, "status": "memory_abort", "swap": 2 * GIB}
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "completed" and report["failures"] == []
    assert children(world)[-3:] == [
        "start timing mps x2 b32",  # attempt 1: batch 32 fit as is in M1
        "child preflight cpu x2 b32",  # the gradient check
        "start timing mps x2 b32 ckpt",  # attempt 2
    ]
    assert world.events.count("workers 1 during M4") == world.events.count("workers 2 during M4")
    assert world.events.count("workers 2 during M4") == 2  # M7 during each attempt
    measurements = report["measurements"]
    m4 = measurements["M4"]
    assert (m4["status"], m4["activation_checkpointing"]) == ("completed", True)
    assert m4["theta"] == pytest.approx(1.1)
    first = m4["first_attempt"]
    assert (first["status"], first["activation_checkpointing"]) == ("memory_abort", False)
    assert first["theta"] is None and first["theta_reason"] == "the run did not complete"
    assert [item["status"] for item in first["m7"]] == ["completed"] * 2
    assert all(item["slowdown_vs_m6"] == pytest.approx(1.25) for item in first["m7"])
    assert [item["status"] for item in measurements["M7"]] == ["completed"] * 2
    spent = report["spend"]["per_measurement"]
    assert spent["M4"]["gpu_minutes"] == pytest.approx(400.0 / 60, abs=5 / 60)
    assert spent["retries"]["gpu_minutes"] == pytest.approx(M4_LIFETIME / 60, abs=5 / 60)
    choice = report["choice"]
    assert (choice["device"], choice["activation_checkpointing"]) == ("mps", True)
    assert choice["seconds_per_step"] == pytest.approx(2.0)  # M4's first 2 minutes, with it
    assert any("M4 ended memory_abort without it" in item for item in choice["reasons"])
    markdown = read_markdown(tmp_path / "reports")
    assert "| M7 (M4 attempt 1) | 1 | completed |" in markdown
    assert "| M4 | 1 | no | memory_abort |" in markdown and "| M4 | 2 (retry) | yes |" in markdown


def test_no_retry_when_checkpointing_fails_its_gradient_check(world, tmp_path):
    world.overrides[("preflight", "mps", 2, 32, False)] = dict(MEMORY_ABORT)
    world.preflight["checkpoint_gradients"] = {"passed": False, "tensors": 36, "mismatches": ["x"]}
    report, _ = run_timing(world, tmp_path)
    assert children(world)[len(CPU_M1) :] == [
        "child preflight mps x2 b32",
        "child preflight cpu x2 b32",  # the gradient check fails: nothing relies on the wrap
    ]
    m3 = report["measurements"]["M3"]
    assert (m3["status"], m3["passed"], m3["activation_checkpointing"]) == (
        "memory_abort",
        False,
        False,
    )
    assert m3["retry"] == {
        "status": "not_run",
        "reason": "not retried: activation checkpointing failed M2's CPU gradient bit-identity",
    }
    assert report["measurements"]["M4"]["status"] == "not_run"
    assert report["choice"]["device"] == "cpu" and report["status"] == "completed"
    markdown = read_markdown(tmp_path / "reports")
    assert "| M3 | 2 | yes | not_run | - | - | not retried: activation checkpointing failed" in (
        markdown
    )


def test_a_retried_preflight_that_fails_a_check_leaves_mps_unused(world, tmp_path):
    world.overrides[("preflight", "mps", 2, 32, False)] = dict(MEMORY_ABORT)
    world.preflight["parity"] = {"passed": False, "gradient_cosine": 0.5}
    report, _ = run_timing(world, tmp_path)
    m3 = report["measurements"]["M3"]
    assert (m3["status"], m3["passed"], m3["activation_checkpointing"]) == (
        "completed",
        False,
        True,
    )
    assert m3["first_attempt"]["status"] == "memory_abort"
    assert [event for event in children(world) if " mps " in event] == [
        "child preflight mps x2 b32",
        "child preflight mps x2 b32 ckpt",
    ]
    choice = report["choice"]
    assert choice["device"] == "cpu" and choice["activation_checkpointing"] is False
    assert any("MPS not usable" in item for item in choice["reasons"])


# == the stop rule =================================================================================


@pytest.mark.parametrize("hours", [None, APPROVED_HOURS])
def test_the_run_stops_once_the_measured_spend_passes_the_limit(world, tmp_path, hours):
    """At the step's own limit, or at the stop limit 600 CPU-seconds below the ceiling the owner
    approved, so that the child that passed it leaves the spend within that ceiling (D-055)."""
    limit = timing.TIMING_MAX_CPU_SECONDS if hours is None else APPROVED_STOP_LIMIT
    world.overrides[("timing", "cpu", 8, 8, False)] = {"cpu": limit - 100.0}
    report, printed = run_timing(world, tmp_path, approved_cpu_hours=hours, given_at=GIVEN_AT)
    assert report["status"] == "stopped"
    spend = report["spend"]
    assert spend["stopped"] is True and spend["cpu_seconds"] > limit
    assert "measured spend" in spend["stop_reason"]
    if hours is None:
        assert spend["stop_reason"].endswith(f"CPU-seconds, above the {limit:.0f} of the budget")
    else:
        assert spend["stop_reason"].endswith(f"CPU-seconds, {APPROVED_STOP_TEXT}")
        assert spend["cpu_seconds"] <= APPROVED_CPU_LIMIT
    assert children(world)[-1] == "child timing cpu x8 b8"  # nothing started after it
    assert m1(report, "cpu", 8, 16)["status"] == "not_run"
    assert report["measurements"]["M4"]["status"] == "not_run"
    assert [item["status"] for item in report["measurements"]["M6"]] == ["not_run"] * 4
    assert any(line.startswith("STOP:") for line in printed)
    assert (tmp_path / "reports" / f"{timing.TIMING_STEM}.json").is_file()


def test_a_measurement_whose_prior_would_pass_the_limit_is_not_started(world, tmp_path):
    limit = timing.TIMING_MAX_CPU_SECONDS
    world.overrides[("timing", "cpu", 2, 32, False)] = {"cpu": limit - 350.0}
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "stopped"
    assert children(world)[-1] == "child timing cpu x2 b32"
    entry = m1(report, "cpu", 8, 8)
    assert entry["status"] == "not_run" and "prior" in entry["reason"]
    prior, _ = timing._child_prior("cpu", 8, 8, 13)
    assert report["spend"]["cpu_seconds"] <= limit < report["spend"]["cpu_seconds"] + prior
    assert "would bring the spend" in report["spend"]["stop_reason"]


def test_the_preflight_starts_only_when_its_content_priced_prior_fits(world, tmp_path):
    """With room for the design's 150 CPU-seconds but not for M3's content (check (c)'s CPU side
    included, about 554), M3 does not start and the run stops before any MPS unit (D-054)."""
    limit = timing.TIMING_MAX_CPU_SECONDS
    world.overrides[("timing", "cpu", 8, 32, False)] = {"cpu": limit - 600.0}
    report, _ = run_timing(world, tmp_path)
    spend = report["spend"]
    assert spend["cpu_seconds"] + 150.0 <= limit < spend["cpu_seconds"] + timing.PRIORS["M3"][0]
    assert children(world) == CPU_M1  # no MPS child
    m3 = report["measurements"]["M3"]
    assert (m3["status"], m3["reason"]) == (
        "not_run",
        "not run: its prior would pass the budget (spend.stop_reason)",
    )
    assert spend["stop_reason"].startswith("M3 (prior about 554 CPU-seconds, 4.6 GPU-minutes)")
    assert report["status"] == "stopped"


def test_m4_starts_only_when_its_whole_lifetime_fits_the_gpu_budget(world, tmp_path):
    """24.75 GPU-minutes spent before M4 leave room for its 20 timed minutes but not for its
    start-up, warm-up and last step (the stand-in child lives 20.5 minutes): M4 does not start,
    instead of running into the 45-minute stop (D-054)."""
    world.overrides[("preflight", "mps", 2, 32, False)] = {"wall": 1395.0}  # 23.25 GPU-minutes
    report, _ = run_timing(world, tmp_path)
    assert not [event for event in world.events if event.startswith("start")]
    spend = report["spend"]
    assert spend["gpu_minutes"] == pytest.approx(24.75)
    assert 45.0 - 21.0 < spend["gpu_minutes"] <= 45.0 - 20.0  # a 20-minute prior would start it
    m4 = report["measurements"]["M4"]
    assert (m4["status"], m4["reason"]) == (
        "not_run",
        "not run: its prior would pass the budget (spend.stop_reason)",
    )
    assert spend["stop_reason"].startswith("M4 with M7 (prior about 270 CPU-seconds, 21.0 GPU")
    assert [item["status"] for item in report["measurements"]["M7"]] == ["not_run"] * 2
    assert report["status"] == "stopped"


def test_an_m4_retry_starts_only_when_its_whole_lifetime_fits_the_gpu_budget(world, tmp_path):
    """A first M4 attempt that ends memory_abort after 22.7 minutes leaves 20 to 21 GPU-minutes:
    room for the retry's 20 timed minutes but not for its whole lifetime, so it is not retried
    (no gradient check either), instead of a retry that could only end at the 45-minute stop."""
    world.m4_overrides[False] = {"lifetime": 1360.0, "status": "memory_abort"}
    report, _ = run_timing(world, tmp_path)
    assert children(world)[-1] == "start timing mps x2 b32"  # attempt 1 only
    m4 = report["measurements"]["M4"]
    assert (m4["status"], m4["activation_checkpointing"]) == ("memory_abort", False)
    assert "first_attempt" not in m4
    assert m4["retry"] == {
        "status": "not_run",
        "reason": "not retried: its prior would pass the budget (spend.stop_reason)",
    }
    spend = report["spend"]
    assert 45.0 - 21.0 < spend["gpu_minutes"] <= 45.0 - 20.0
    assert spend["stop_reason"].startswith("M4 retried with activation checkpointing (prior about")
    assert "21.0 GPU-minutes" in spend["stop_reason"]
    assert report["measurements"]["M2"]["gradient_check"] is None
    assert [item["status"] for item in report["measurements"]["M6"]] == ["not_run"] * 4
    assert report["status"] == "stopped" and report["failures"] == []


@pytest.mark.parametrize(
    "ending", [{"status": "time_abort"}, {"timed_out": True, "status": "killed"}]
)
def test_a_child_that_reaches_its_budget_deadline_stops_the_run(world, tmp_path, ending):
    world.overrides[("timing", "cpu", 2, 16, False)] = ending
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "stopped" and report["failures"] == []
    assert m1(report, "cpu", 2, 16)["status"] == "stopped"
    assert children(world)[-1] == "child timing cpu x2 b16"
    assert "remaining budget" in report["spend"]["stop_reason"]


@pytest.mark.parametrize("hours, rate", [(None, 3.0), (APPROVED_HOURS, 5.0)])
def test_the_m4_child_is_stopped_when_its_cpu_passes_the_limit(world, tmp_path, hours, rate):
    """A runaway MPS child (``rate`` CPU-seconds per second) is stopped at the step's own limit,
    or at the stop limit below the ceiling the owner approved (where 3 per second would fit) the
    same way, which keeps the spend within that ceiling."""
    world.m4_cpu_rate = rate
    report, _ = run_timing(world, tmp_path, approved_cpu_hours=hours, given_at=GIVEN_AT)
    assert "terminate M4" in world.events
    m4 = report["measurements"]["M4"]
    assert m4["status"] == "stopped" and m4["theta"] is None
    assert report["status"] == "stopped" and "M4 running" in report["spend"]["stop_reason"]
    assert [item["status"] for item in report["measurements"]["M6"]] == ["not_run"] * 4
    # stopped within a poll of the limit: the overshoot is at most one poll of its CPU rate
    limit = timing.TIMING_MAX_CPU_SECONDS if hours is None else APPROVED_STOP_LIMIT
    spend = report["spend"]["cpu_seconds"]
    assert limit < spend <= limit + rate * timing.CHILD_POLL_SECONDS + 1e-6
    if hours is not None:
        assert report["spend"]["stop_reason"].endswith(APPROVED_STOP_TEXT)
        assert spend <= APPROVED_CPU_LIMIT


@pytest.mark.parametrize("hours", [None, APPROVED_HOURS])
def test_the_m4_child_is_stopped_when_the_gpu_time_passes_the_limit(world, tmp_path, hours):
    """The GPU limit stays 45 minutes with the owner's approval of CPU-hours (D-055)."""
    world.m4_lifetime = 3600.0  # it never ends by itself
    report, _ = run_timing(world, tmp_path, approved_cpu_hours=hours, given_at=GIVEN_AT)
    assert "terminate M4" in world.events
    assert report["measurements"]["M4"]["status"] == "stopped"
    assert report["spend"]["gpu_minutes"] <= 45.0 + timing.CHILD_POLL_SECONDS / 60 + 1e-6
    assert "GPU-minutes" in report["spend"]["stop_reason"]


# == the approved ceiling is never passed (D-055) ==================================================


def first_run_fans(world: FakeWorld) -> None:
    """Charge the stand-in fans what the first run's M6 cost (1.25 times that during M4)."""
    plain = world.time_fan_workers

    def fans(params: Any, *, workers: int, fans: int) -> dict[str, Any]:
        result = plain(params, workers=workers, fans=fans)
        during = world.events[-1].endswith("during M4")
        return {**result, "cpu_seconds": M6_FIRST_RUN[workers] * (1.25 if during else 1.0)}

    world.time_fan_workers = fans


def test_m6_never_takes_the_spend_past_the_approved_ceiling(world, tmp_path):
    """M6 starts each worker count on its prior of 75 CPU-seconds, but the first run charged them
    103, 165, 221 and 289 (D-053 (8)). Stopped only once its spend passed the ceiling itself, a
    run under the owner's 1.5 CPU-hours ended at 5,479 CPU-seconds when M6 started at 4,700 and
    at 5,614 from 4,835. Against the stop limit 600 below the ceiling the spend stays within
    5,400 from any start up to the stop limit; the worst start lets all four run and ends 214
    past the stop limit."""
    first_run_fans(world)
    budget = timing.timing_budget(APPROVED_HOURS, given_at=GIVEN_AT)
    estimate = timing.timing_estimate()
    worst = 0.0
    for start in range(3800, int(APPROVED_STOP_LIMIT) + 1, 5):
        run = timing._Run(
            params=PARAMS,
            steps=world.steps(),
            root=tmp_path / "ft",
            scratch=tmp_path / "scratch",
            estimate=estimate,
            budget=budget,
            mps_available=True,
            say=lambda line: None,
        )
        run.cpu_seconds = float(start)
        run.m6()
        assert run.cpu_seconds <= APPROVED_CPU_LIMIT, start
        fits = start + sum(M6_FIRST_RUN.values()) <= APPROVED_STOP_LIMIT
        assert (run.stop_reason is None) is fits, start  # all four ran, or the run stopped
        worst = max(worst, run.cpu_seconds)
    # the last worker count started at the stop limit minus its prior
    assert worst == pytest.approx(APPROVED_STOP_LIMIT - 75 + M6_FIRST_RUN[4], abs=5)


def test_m7_admits_each_configuration_on_the_m4_cpu_read_at_its_start(world, tmp_path):
    """M4 is not polled while M7 runs: at 2 CPU-seconds per second the M4 child adds about 105
    while M7's first configuration runs (52 s), which here take the second past the stop limit.
    It is not started, because it is admitted on the M4 CPU read at its own start, not on the one
    read when M7 began (which would have started it); the watch then stops M4 at the stop limit,
    within the approved ceiling."""
    world.m4_cpu_rate = 2.0
    base, _ = run_timing(
        world, tmp_path / "base", approved_cpu_hours=APPROVED_HOURS, given_at=GIVEN_AT
    )
    assert base["status"] == "completed"
    spent = base["spend"]["per_measurement"]
    later = ("M4", "M7", "M6")
    before_m4 = sum(item["cpu_seconds"] for name, item in spent.items() if name not in later)
    one = base["measurements"]["M7"][0]  # 1 worker, from M7's start to its second configuration
    stale = world.m4_cpu_rate * one["m4_elapsed_start_seconds"]  # 360 CPU-seconds
    fresh = world.m4_cpu_rate * one["m4_elapsed_end_seconds"]  # 464.5
    prior = timing.PRIORS["M7"][0] / len(timing.M7_WORKERS)
    # the spend when M7's second configuration starts: the stale M4 CPU would fit the stop limit
    target = APPROVED_STOP_LIMIT - prior - (stale + fresh) / 2
    extra = target - one["cpu_seconds"] - before_m4  # on the last child before M4
    world.overrides[("timing", "mps", 2, 32, False)] = {"cpu": 10.0 + extra}
    report, _ = run_timing(
        world, tmp_path / "run", approved_cpu_hours=APPROVED_HOURS, given_at=GIVEN_AT
    )
    m7 = report["measurements"]["M7"]
    assert [item["status"] for item in m7] == ["completed", "not_run"]
    assert m7[1]["reason"] == "not run: its prior would pass the budget (spend.stop_reason)"
    spend = report["spend"]
    assert spend["stop_reason"] == (
        "M7 2 worker(s) (prior about 75 CPU-seconds, 0.0 GPU-minutes) would bring the spend to "
        f"about {target + fresh + prior:.0f} CPU-seconds, {APPROVED_STOP_TEXT}"
    )
    assert "terminate M4" in world.events and report["measurements"]["M4"]["status"] == "stopped"
    assert APPROVED_STOP_LIMIT < spend["cpu_seconds"] <= APPROVED_CPU_LIMIT
    assert [item["status"] for item in report["measurements"]["M6"]] == ["not_run"] * 4


def test_a_child_stopped_at_its_deadline_stays_within_the_approved_ceiling(world, tmp_path):
    """A training child that would run on (CPU x8 at batch 32: 8 CPU-seconds per second) is
    stopped by the deadline the stop limit leaves it, here one 10.6 s step late (D-053 (1)): the
    spend passes the stop limit by those 85 CPU-seconds, within the reserve below the ceiling the
    owner approved (a deadline from the ceiling itself ended the spend 85 past it)."""
    world.overrides[("timing", "cpu", 8, 32, False)] = {"past_deadline": 10.6}
    report, _ = run_timing(world, tmp_path, approved_cpu_hours=APPROVED_HOURS, given_at=GIVEN_AT)
    assert report["status"] == "stopped" and report["failures"] == []
    assert m1(report, "cpu", 8, 32)["status"] == "stopped"
    assert children(world)[-1] == "child timing cpu x8 b32"
    assert "remaining budget" in report["spend"]["stop_reason"]
    spend = report["spend"]["cpu_seconds"]
    assert spend == pytest.approx(APPROVED_STOP_LIMIT + 10.6 * 8)
    assert spend <= APPROVED_CPU_LIMIT


def test_a_retry_whose_prior_would_pass_the_budget_is_not_started(world, tmp_path):
    limit = timing.TIMING_MAX_CPU_SECONDS
    world.overrides[("preflight", "mps", 2, 32, False)] = {**MEMORY_ABORT, "cpu": limit - 400.0}
    report, printed = run_timing(world, tmp_path)
    assert report["status"] == "stopped" and report["failures"] == []
    assert children(world)[-1] == "child preflight mps x2 b32"  # no gradient check, no retry
    spend = report["spend"]
    # one retry of M3's size (priced by its content) and the gradient check it needs first
    retry_prior = timing.PRIORS["M3"][0] + timing._gradient_check_prior()[0]
    assert retry_prior == pytest.approx(553.76 + 24.4)
    assert spend["cpu_seconds"] <= limit < spend["cpu_seconds"] + retry_prior
    assert spend["stop_reason"].startswith(
        "M3 MPS preflight retried with activation checkpointing (prior about 578 CPU-seconds"
    )
    m3 = report["measurements"]["M3"]
    assert (m3["status"], m3["activation_checkpointing"]) == ("memory_abort", False)
    assert m3["retry"] == {
        "status": "not_run",
        "reason": "not retried: its prior would pass the budget (spend.stop_reason)",
    }
    assert "first_attempt" not in m3 and report["measurements"]["M2"]["gradient_check"] is None
    assert all(m1(report, "mps", 2, batch)["status"] == "not_run" for batch in (8, 16, 32))
    assert report["measurements"]["M4"]["status"] == "not_run"
    assert [item["status"] for item in report["measurements"]["M6"]] == ["not_run"] * 4
    assert any(line.startswith("STOP:") for line in printed)
    markdown = read_markdown(tmp_path / "reports")
    assert "| M3 | 1 | no | memory_abort |" in markdown
    assert "| M3 | 2 | yes | not_run | - | - | not retried: its prior would pass the budget" in (
        markdown
    )


def test_a_retry_that_reaches_its_budget_deadline_stops_the_run(world, tmp_path):
    world.overrides[("preflight", "mps", 2, 32, False)] = dict(MEMORY_ABORT)
    world.overrides[("preflight", "mps", 2, 32, True)] = {"status": "time_abort"}
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "stopped" and report["failures"] == []
    assert children(world)[-1] == "child preflight mps x2 b32 ckpt"
    m3 = report["measurements"]["M3"]
    assert (m3["status"], m3["activation_checkpointing"]) == ("stopped", True)
    assert m3["first_attempt"]["status"] == "memory_abort" and m3["passed"] is False
    assert report["spend"]["stop_reason"] == (
        "M3 MPS preflight reached the time the remaining budget allowed it (time_abort)"
    )
    # the retry got the wall time the remaining budget allowed: CPU-seconds / 2 here
    spent = report["spend"]["per_measurement"]
    before = sum(item["cpu_seconds"] for name, item in spent.items() if name != "retries")
    (retry,) = [item for item in world.requests if item["key"] == ("preflight", "mps", 2, 32, True)]
    assert retry["deadline"] == pytest.approx((timing.TIMING_MAX_CPU_SECONDS - before) / 2)
    assert retry["timeout"] == pytest.approx(retry["deadline"] + 60.0)
    assert all(m1(report, "mps", 2, batch)["status"] == "not_run" for batch in (8, 16, 32))
    assert [item["status"] for item in report["measurements"]["M6"]] == ["not_run"] * 4


def test_a_retried_m4_is_stopped_at_the_gpu_limit(world, tmp_path):
    world.m4_overrides[False] = {"lifetime": 400.0, "status": "memory_abort"}
    world.m4_overrides[True] = {"lifetime": 3600.0}  # it never ends by itself
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "stopped" and "terminate M4" in world.events
    m4 = report["measurements"]["M4"]
    assert (m4["status"], m4["activation_checkpointing"]) == ("stopped", True)
    assert m4["first_attempt"]["status"] == "memory_abort" and m4["theta"] is None
    spend = report["spend"]
    assert spend["gpu_minutes"] <= 45.0 + timing.CHILD_POLL_SECONDS / 60 + 1e-6
    assert "GPU-minutes" in spend["stop_reason"] and "M4 running" in spend["stop_reason"]
    assert spend["per_measurement"]["retries"]["gpu_minutes"] > 30.0
    assert [item["status"] for item in report["measurements"]["M6"]] == ["not_run"] * 4


# == measurement outcomes ==========================================================================


def test_an_aborted_configuration_skips_its_larger_batches(world, tmp_path):
    world.overrides[("timing", "cpu", 8, 16, False)] = {"status": "memory_abort"}
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "completed"
    assert m1(report, "cpu", 8, 16)["status"] == "memory_abort"
    skipped = m1(report, "cpu", 8, 32)
    assert skipped["status"] == "not_run" and "ended memory_abort" in skipped["reason"]
    assert "child timing cpu x8 b32" not in world.events
    assert "child timing mps x2 b8" in world.events  # the other configurations still run
    assert report["choice"]["speedups"]["eight_vs_two_threads"] is None


def test_m2_runs_when_batch_32_does_not_fit_and_m4_uses_its_checkpointing(world, tmp_path):
    world.overrides[("timing", "mps", 2, 32, False)] = {"peak": 9 * GIB}  # plus 1 GiB of MPS
    report, _ = run_timing(world, tmp_path)
    events = children(world)
    m2_at = events.index("child timing mps x2 b32 ckpt")
    assert events[m2_at + 1] == "child preflight cpu x2 b32"  # the CPU gradient bit-identity
    assert events[m2_at + 2] == "start timing mps x2 b32 ckpt"
    assert world.requests[m2_at + 1]["checks"] == ("checkpoint_gradients",)
    m2s = report["measurements"]["M2"]
    (m2,) = m2s["runs"]
    assert (m2["device"], m2["status"], m2["passed"]) == ("mps", "completed", True)
    assert m2s["gradient_check"]["passed"] is True
    assert m1(report, "mps", 2, 32)["activation_checkpointing"] is False  # it completed as is
    assert report["mps_checkpointing"]["reason"].startswith("the MPS batch 32 does not fit")
    assert report["measurements"]["M4"]["activation_checkpointing"] is True
    choice = report["choice"]
    assert choice["device"] == "mps" and choice["activation_checkpointing"] is True
    assert choice["batch_32_fits"] is False
    assert choice["seconds_per_step"] == pytest.approx(2.0 * 1.3)


def test_batch_32_that_fits_nowhere_goes_to_the_owner(world, tmp_path):
    world.mps = False
    world.overrides[("timing", "cpu", 2, 32, False)] = {"peak": 10 * GIB}
    world.overrides[("timing", "cpu", 2, 32, True)] = {"peak": 10 * GIB}
    report, _ = run_timing(world, tmp_path)
    (m2,) = report["measurements"]["M2"]["runs"]
    assert m2["device"] == "cpu" and m2["passed"] is False
    choice = report["choice"]
    assert choice["device"] == "cpu" and choice["batch_size"] == 32  # never changed here
    assert choice["batch_32_fits"] is False and choice["activation_checkpointing"] is False
    assert any("batch 16 needs the owner" in reason for reason in choice["reasons"])


def test_an_m9_mismatch_fails_the_run_after_writing_the_report(world, tmp_path):
    world.identical = False
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "failed"
    assert report["measurements"]["M9"]["status"] == "mismatch"
    assert any("M9" in item for item in report["failures"])
    assert len(children(world)) == 11  # the other measurements still ran
    assert timing.load_timing(directory=tmp_path / "reports")["status"] == "failed"


def test_a_crashed_child_fails_the_run_and_its_configuration_only(world, tmp_path):
    world.overrides[("timing", "cpu", 2, 8, False)] = {
        "status": "failed",
        "error": {"type": "RunnerError", "message": "the loss is not finite"},
    }
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "failed"
    entry = m1(report, "cpu", 2, 8)
    assert entry["status"] == "failed" and entry["error"]["type"] == "RunnerError"
    assert m1(report, "cpu", 2, 16)["status"] == "not_run"
    assert m1(report, "cpu", 8, 32)["status"] == "completed"
    assert any("cpu x2 batch 8" in item for item in report["failures"])


def test_a_measurement_error_is_recorded_and_the_run_goes_on(world, tmp_path):
    world.fan_errors[8] = BatchError("the timed fan took 0.0 s")
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "failed"
    per_length = report["measurements"]["M5"]["per_length"]
    assert [item["status"] for item in per_length] == ["completed", "failed", "completed"]
    assert per_length[1]["error"]["type"] == "BatchError"
    assert report["measurements"]["M5"]["t8"] is None
    assert "fan t(24)" in world.events and len(children(world)) == 11


def test_an_environment_error_stops_the_run_and_is_reported(world, tmp_path):
    world.child_error = VendoredTreeError("third_party/kronos holds a __pycache__ directory")
    report, _ = run_timing(world, tmp_path)
    assert report["status"] == "failed" and report["spend"]["stopped"] is True
    first, *others = report["measurements"]["M1"]  # the CPU's batch 8 is the first child now
    assert (first["device"], first["threads"], first["batch_size"]) == ("cpu", 2, 8)
    assert first["status"] == "failed"
    assert len(children(world)) == 1  # nothing after the first child
    assert all(item["status"] == "not_run" for item in others)
    assert report["measurements"]["M3"]["status"] == "not_run"


# == the record: staleness, hygiene, refusals ======================================================


def test_a_stale_record_is_refused(world, tmp_path):
    from kronos_agent.finetune.train import training_code_sha256
    from kronos_agent.runtime import require_configured

    run_timing(world, tmp_path)
    record = timing.load_timing(directory=tmp_path / "reports")
    fresh = {
        "device": "mps",
        "torch_version": require_configured().torch_version,
        "platform": default_label(),
        "training_code_sha256": training_code_sha256(),
    }
    timing.require_fresh_timing(record, **fresh)
    for key, other, word in (
        ("device", "cpu", "device"),
        ("torch_version", "0.0.0", "torch"),
        ("platform", "linux-x86_64", "platform"),
        ("training_code_sha256", "0" * 64, "training code"),
    ):
        with pytest.raises(timing.StaleTimingError, match=word):
            timing.require_fresh_timing(record, **{**fresh, key: other})
    with pytest.raises(timing.StaleTimingError, match="schema"):
        timing.require_fresh_timing({**record, "schema_version": 1}, **fresh)  # the first run's


def test_load_timing_refuses_a_missing_or_foreign_record(tmp_path):
    with pytest.raises(timing.TimingError, match="run `kronos-agent finetune timing` first"):
        timing.load_timing(directory=tmp_path)
    # the first run's record keeps its stem: it is never read as this plan's record
    first = {"study": "timing", "schema_version": 1}
    (tmp_path / "timing.json").write_text(json.dumps(first), encoding="ascii")
    with pytest.raises(timing.TimingError, match=r"no timing record \(finetune/timing-2\.json\)"):
        timing.load_timing(directory=tmp_path)
    path = tmp_path / f"{timing.TIMING_STEM}.json"
    for document, problem in (
        ({"study": "coverage"}, "not a timing record"),
        (first, "not a timing record"),
        ({**first, "study": timing.TIMING_STEM}, "has schema 1, expected 2"),
    ):
        path.write_text(json.dumps(document), encoding="ascii")
        with pytest.raises(timing.TimingError, match=problem):
            timing.load_timing(directory=tmp_path)
    path.write_text("{", encoding="ascii")
    with pytest.raises(timing.TimingError, match="cannot be read"):
        timing.load_timing(directory=tmp_path)


def test_a_run_writes_its_own_stem_beside_the_first_runs_record(world, tmp_path):
    """The first run recorded another plan under ``timing`` (D-053): this plan's run writes
    ``timing-2`` and leaves those three files byte for byte, where under ``timing`` it would be
    refused before anything ran (D-034: a changed study is a new study with a new stem; D-054)."""
    directory = tmp_path / "reports"
    directory.mkdir()
    first = {
        "timing.config.json": canonical_json({**timing.timing_plan(), "schema_version": 1}),
        "timing.json": json.dumps({"study": "timing", "schema_version": 1}).encode("ascii"),
        "timing.md": b"# Measurement step of the retraining (finetune timing)\n",
    }
    for name, data in first.items():
        (directory / name).write_bytes(data)
    report, printed = run_timing(world, tmp_path)
    assert report["status"] == "completed" and report["study"] == "timing-2"
    for name, data in first.items():
        assert (directory / name).read_bytes() == data
    assert sorted(path.name for path in directory.iterdir()) == sorted([*first, *report_names()])
    assert timing.load_timing(directory=directory) == report
    assert any(line.startswith("timing plan timing-2.config.json (sha256") for line in printed)
    with pytest.raises(ConfigChangedError, match="study timing: its recorded config"):
        record_config("timing", timing.timing_plan(), directory=directory)


def test_the_report_holds_no_path_user_or_host_name(world, tmp_path):
    home = Path.home()
    message = (
        f"cannot read {tmp_path / 'ft' / '.runs' / 'x' / 'run.json'}: missing; also "
        f"{home / 'secret' / 'file'} and ~/other/thing and {REPO_ROOT / 'weights'}"
    )
    world.overrides[("timing", "cpu", 2, 8, False)] = {
        "status": "failed",
        "error": {"type": "RunnerError", "message": message},
    }
    world.fan_errors[4] = BatchError(f"no window under {tmp_path / 'data'}")
    report, printed = run_timing(world, tmp_path)
    assert m1(report, "cpu", 2, 8)["error"]["message"].startswith("cannot read <path>: missing")
    texts = [(tmp_path / "reports" / name).read_text(encoding="ascii") for name in report_names()]
    texts.append("\n".join(printed))
    for text in texts:
        for root in (tmp_path, home, REPO_ROOT, Path(tempfile.gettempdir()).resolve()):
            assert str(root) not in text
        assert "~/" not in text
        assert re.search(r"(?<![\w.])/(?:private|var|tmp|" + "Users" + r")/", text) is None


def test_a_concurrent_run_is_refused(world, tmp_path):
    lock = tmp_path / "ft" / ".scratch" / timing.LOCK_NAME
    with exclusive_lock(lock), pytest.raises(LockBusyError):
        run_timing(world, tmp_path)
    assert world.events == []


def test_a_changed_plan_is_refused_before_anything_runs(world, tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    recorded = directory / f"{timing.TIMING_STEM}.config.json"
    recorded.write_bytes(canonical_json({"another": "plan"}))
    with pytest.raises(ConfigChangedError):
        run_timing(world, tmp_path)
    assert world.events == []
    assert recorded.read_bytes() == canonical_json({"another": "plan"})  # never overwritten


def test_the_runtime_must_run_the_governed_thread_count(world, tmp_path):
    params = replace(PARAMS, kronos=replace(PARAMS.kronos, threads=4))
    with pytest.raises(timing.TimingError, match="threads"):
        run_timing(world, tmp_path, params=params)
    assert world.events == []


# == the compute choice ============================================================================


def _m1_entry(device, threads, median, *, status="completed", peak=2 * GIB, growth=0):
    return {
        "device": device,
        "threads": threads,
        "batch_size": 32,
        "status": status,
        "seconds_per_step": {"median": median},
        "peak_bytes": peak,
        "swap_growth_bytes": growth,
    }


def _m2(runs, *, gradient=True):
    """M2 as recorded: its CPU gradient bit-identity (passed or not) and its runs."""
    return {"gradient_check": {"status": "completed", "passed": gradient}, "runs": runs}


def _measurements(cpu2=7.0, cpu8=6.0, mps=2.0, sustained=2.0, *, preflight=True, theta=1.1, **kw):
    entries = []
    for device, threads, median in (("cpu", 2, cpu2), ("cpu", 8, cpu8), ("mps", 2, mps)):
        if median is not None:
            entries.append(_m1_entry(device, threads, median, **kw.get(device + str(threads), {})))
    m4 = {"status": "completed", "median_last_seconds": sustained, "theta": theta}
    return {
        "M1": entries,
        "M2": kw.get("m2") or {"gradient_check": None, "runs": []},
        "M3": {"passed": preflight, "deterministic_passed": kw.get("deterministic", True)},
        "M4": {**m4, **kw.get("m4", {})}
        if sustained is not None
        else {"status": "not_run", "reason": "not run: MPS is not available on this machine"},
    }


@pytest.mark.parametrize(
    "sustained, device",
    [(4.0, "mps"), (6.0 / 1.49, "cpu"), (1.0, "mps"), (None, "cpu")],  # best CPU: 6.0 s
)
def test_mps_is_chosen_only_when_at_least_1_5_times_faster_than_the_best_cpu(sustained, device):
    choice = timing.choose_compute(_measurements(sustained=sustained))
    assert choice["device"] == device
    assert choice["threads"] == 2  # MPS runs use 2 threads; 8 CPU threads are not 1.3 times faster


@pytest.mark.parametrize("cpu8, threads", [(5.0, 8), (6.5 / 1.29, 2), (None, 2)])
def test_eight_cpu_threads_only_when_at_least_1_3_times_faster(cpu8, threads):
    choice = timing.choose_compute(_measurements(cpu2=6.5, cpu8=cpu8, sustained=None))
    assert (choice["device"], choice["threads"]) == ("cpu", threads)
    assert choice["deterministic"] is True and choice["theta"] is None


def test_a_failed_preflight_never_chooses_mps():
    choice = timing.choose_compute(_measurements(sustained=0.5, preflight=False))
    assert choice["device"] == "cpu"
    assert any("MPS not usable" in reason for reason in choice["reasons"])


def test_nothing_measured_decides_nothing():
    choice = timing.choose_compute(_measurements(cpu2=None, cpu8=None, sustained=2.0))
    assert choice["device"] is None and choice["threads"] is None
    assert any("owner" in reason for reason in choice["reasons"])


def test_the_mps_choice_carries_its_determinism_and_a_theta_of_at_least_1():
    choice = timing.choose_compute(_measurements(theta=0.97, deterministic=False))
    assert choice["device"] == "mps" and choice["deterministic"] is False
    assert choice["theta"] == 1.0


@pytest.mark.parametrize(
    "mps_fit, m2, checkpointing, fits",
    [
        ({}, None, False, True),
        ({"peak": 9 * GIB + 1}, None, False, False),
        ({"growth": 300 * 2**20}, None, False, False),  # swapped
        ({"growth": 100 * 2**20}, None, False, True),  # noise below the tolerance
        (
            {"peak": 10 * GIB},
            _m2(
                [
                    {
                        "device": "mps",
                        "threads": 2,
                        "passed": True,
                        "timing": _m1_entry("mps", 2, 2.6),
                    }
                ]
            ),
            True,
            False,
        ),
        ({"status": "memory_abort"}, None, False, False),
    ],
)
def test_batch_32_fits_or_needs_checkpointing_or_the_owner(mps_fit, m2, checkpointing, fits):
    choice = timing.choose_compute(_measurements(mps2=mps_fit, m2=m2))
    assert choice["device"] == "mps"
    assert choice["activation_checkpointing"] is checkpointing
    assert choice["batch_32_fits"] is fits and choice["batch_size"] == 32
    if checkpointing:
        assert choice["seconds_per_step"] == 2.6
    if not fits and not checkpointing:
        assert any("batch 16 needs the owner" in reason for reason in choice["reasons"])


def _measured_on_this_mac(sustained, *, gradient=True):
    """M1 and M2 on the CPU as measured on 2026-11-02 (batch 32 swaps as is at 2 and 8 threads;
    2 threads with activation checkpointing runs at 9.68 s), M3 passed, and M1 and M4 on MPS
    with activation checkpointing."""
    swapping = {"peak": int(9.2 * GIB), "growth": int(1.5 * GIB)}
    mps_b32 = {**_m1_entry("mps", 2, 3.0), "activation_checkpointing": True}
    return {
        "M1": [
            _m1_entry("cpu", 2, 10.36, **swapping),
            _m1_entry("cpu", 8, 10.61, **swapping),
            mps_b32,
        ],
        "M2": _m2(
            [{"device": "cpu", "threads": 2, "passed": True, "timing": _m1_entry("cpu", 2, 9.68)}],
            gradient=gradient,
        ),
        "M3": {"passed": True, "deterministic_passed": True},
        "M4": {
            "status": "completed",
            "median_last_seconds": sustained,
            "theta": 1.05,
            "activation_checkpointing": True,
        },
    }


@pytest.mark.parametrize("sustained, device", [(6.4, "mps"), (6.6, "cpu")])
def test_mps_is_compared_with_the_fastest_cpu_configuration_that_runs_batch_32(sustained, device):
    """The CPU runs batch 32 only with activation checkpointing, at 9.68 s: MPS must be 1.5 times
    faster than that (6.45 s), not than the 10.36 s of a batch 32 that does not fit."""
    choice = timing.choose_compute(_measured_on_this_mac(sustained))
    assert choice["best_cpu"] == {
        "configuration": "CPU x2 with activation checkpointing, 9.680 s per step",
        "seconds_per_step": 9.68,
    }
    assert choice["speedups"]["mps_vs_best_cpu"] == pytest.approx(9.68 / sustained)
    assert choice["device"] == device and choice["activation_checkpointing"] is True
    assert choice["batch_32_fits"] is False and choice["threads"] == 2
    if device == "mps":
        assert choice["seconds_per_step"] == 3.0 and choice["theta"] == 1.05
        assert any(
            item.startswith("device MPS: its sustained step time (with activation")
            for item in choice["reasons"]
        )
    else:
        assert choice["seconds_per_step"] == 9.68 and choice["theta"] is None
        assert (
            "batch 32 does not fit as is: activation checkpointing (M2 passed)"
            in (choice["reasons"])
        )


def test_activation_checkpointing_that_fails_its_gradient_check_is_used_nowhere():
    choice = timing.choose_compute(_measured_on_this_mac(2.0, gradient=False))
    assert choice["device"] == "cpu" and choice["activation_checkpointing"] is False
    assert any("did not pass M2's CPU gradient bit-identity" in item for item in choice["reasons"])
    assert any("batch 16 needs the owner" in item for item in choice["reasons"])
    # no CPU configuration runs batch 32: MPS would be compared with the times as is
    assert choice["best_cpu"]["seconds_per_step"] == 10.36
    assert choice["best_cpu"]["configuration"].endswith("no CPU configuration runs batch 32")


def test_mps_checkpointed_only_by_its_m4_retry_takes_its_step_time_from_m4():
    m4 = {
        "activation_checkpointing": True,
        "median_first_seconds": 1.9,
        "checkpointing_reason": "M4 ended memory_abort without it",
        "first_attempt": {"status": "memory_abort", "activation_checkpointing": False},
    }
    choice = timing.choose_compute(_measurements(m4=m4, m2=_m2([])))
    assert choice["device"] == "mps" and choice["activation_checkpointing"] is True
    assert choice["seconds_per_step"] == 1.9 and choice["batch_32_fits"] is False
    assert (
        "batch 32 runs with activation checkpointing on MPS: M4 ended memory_abort without it; "
        "its step time is M4's first 2 minutes" in choice["reasons"]
    )
    # without the gradient check MPS is not chosen at all
    unchecked = timing.choose_compute(_measurements(m4=m4))
    assert unchecked["device"] == "cpu"


# == helpers =======================================================================================


def test_theta_compares_the_last_ten_minutes_with_the_first_two():
    offsets = [float(second) for second in range(0, 1200, 2)]
    times = [2.0 if offset < 600 else 2.4 for offset in offsets]
    document = {"step_times": times, "step_offsets": offsets}
    result = timing._thermal(document, completed=True)
    assert result["theta"] == pytest.approx(1.2)
    assert result["first_window_steps"] == 60 and result["steps"] == 600
    assert timing._thermal(document, completed=False)["theta"] is None
    short = {"step_times": times[:200], "step_offsets": offsets[:200]}
    assert timing._thermal(short, completed=True)["theta"] is None
    assert timing._thermal({}, completed=True)["theta_reason"] == "no timed steps"


@pytest.mark.parametrize(
    "text, seconds",
    [
        ("  0:00.04", 0.04),
        ("26:10.62", 1570.62),
        ("01:02:03", 3723.0),
        ("2-01:00:00", 2 * 86400 + 3600.0),
        ("garbage", None),
        ("", None),
    ],
)
def test_ps_cpu_times_of_macos_and_linux_are_parsed(text, seconds):
    parsed = timing._parse_ps_time(text)
    assert parsed == (None if seconds is None else pytest.approx(seconds))


def test_the_cpu_time_of_a_running_process_is_read_with_ps():
    own = timing._child_cpu_seconds(SimpleNamespace(process=SimpleNamespace(pid=os.getpid())))
    assert own is not None and own >= 0.0
    assert timing._child_cpu_seconds(SimpleNamespace(process=SimpleNamespace(pid=-1))) is None
    assert timing._child_cpu_seconds(object()) is None


def test_the_real_steps_are_the_real_measurements():
    from kronos_agent.finetune import nll, train
    from kronos_agent.kronos import batch, weights

    steps = timing.real_steps()
    assert steps.run_child is train.run and steps.start_child is train.start_run
    assert steps.seconds_per_fan is batch.measure_seconds_per_fan
    assert steps.time_fan_workers is batch.time_fan_workers
    assert steps.identity_control is batch.identity_control
    assert steps.load_models is weights.load_models and steps.nll4_tf is nll.nll4_tf


# == the command ===================================================================================


@pytest.fixture
def command(world, tmp_path, monkeypatch) -> FakeWorld:
    """``kronos-agent finetune timing`` on the stand-ins, its roots in ``tmp_path``."""
    monkeypatch.setattr(finetune_cli, "reports_dir", lambda: tmp_path / "reports")
    monkeypatch.setenv("KRONOS_AGENT_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("KRONOS_AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(timing, "real_steps", world.steps)
    return world


def test_the_command_estimate_runs_nothing(command, tmp_path, capsys):
    assert cli.main(["finetune", "timing", "--estimate"]) == 0
    out = capsys.readouterr().out
    assert "timing estimate" in out and "3578 CPU-seconds" in out and "nothing ran" in out
    assert "order: setup" in out and "M4 with M7 during it" in out
    assert "one retry with it" in out
    assert command.events == [] and not (tmp_path / "reports").exists()
    assert not (tmp_path / "weights").exists()


def test_under_the_default_limit_the_command_refuses_to_start_until_the_owner_decides(
    command, tmp_path, monkeypatch, capsys
):
    """With MPS the estimate (about 3,578 CPU-seconds) passes the default 3,000: `--estimate` says
    so and `finetune timing` without the owner's approval refuses (exit 2) before anything runs or
    is written; without MPS (2,050) it runs (D-054 (8); the approval is D-055)."""
    monkeypatch.setattr(timing, "TIMING_MAX_CPU_SECONDS", DEFAULT_CPU_LIMIT)
    assert cli.main(["finetune", "timing", "--estimate"]) == 0
    out = capsys.readouterr().out
    assert "about 3578 CPU-seconds" in out and "budget 3000 CPU-seconds" in out
    assert "above the budget: `finetune timing` refuses to start" in out
    assert cli.main(["finetune", "timing"]) == 2
    out = capsys.readouterr().out
    assert "REFUSED:" in out and "nothing ran" in out
    assert command.events == [] and not (tmp_path / "reports").exists()
    assert not (tmp_path / "weights").exists()
    command.mps = False
    assert cli.main(["finetune", "timing"]) == 0
    assert timing.load_timing(directory=tmp_path / "reports" / "finetune")["status"] == "completed"


def test_the_command_runs_above_the_default_limit_only_with_the_owners_approval(
    command, tmp_path, monkeypatch, capsys
):
    """D-055: the owner approved up to 1.5 CPU-hours for this measurement. `--estimate` prints
    that ceiling; without the flag, or with a figure below the estimate, the command refuses
    (exit 2) before anything runs or is written; `--approved-by-owner 1.5` runs and records the
    approval with the command's start as the time the flag was given."""
    monkeypatch.setattr(timing, "TIMING_MAX_CPU_SECONDS", DEFAULT_CPU_LIMIT)
    monkeypatch.setattr(finetune_cli, "_now", lambda: GIVEN_AT)
    assert cli.main(["finetune", "timing", "--estimate", "--approved-by-owner", "1.5"]) == 0
    out = capsys.readouterr().out
    assert (
        "about 3578 CPU-seconds and 33.2 GPU-minutes; budget 5400 CPU-seconds and 45 "
        "GPU-minutes: the 1.5 CPU-hours the owner approved (--approved-by-owner given at "
        f"{GIVEN_AT_TEXT}), above gate 3 (3600 CPU-seconds, 60 GPU-minutes) under that approval"
        in out
    )
    assert "above the budget" not in out and "--estimate: nothing ran" in out
    for argv, refusal in (
        ([], "which keeps it below gate 3"),
        (["--approved-by-owner", "0.9"], "the owner's approval does not cover it"),
    ):
        assert cli.main(["finetune", "timing", *argv]) == 2
        out = capsys.readouterr().out
        assert "REFUSED:" in out and refusal in out and "nothing ran" in out
    assert command.events == [] and not (tmp_path / "reports").exists()
    assert not (tmp_path / "weights").exists()
    assert cli.main(["finetune", "timing", "--approved-by-owner", "1.5"]) == 0
    out = capsys.readouterr().out
    assert f"gate 3: {APPROVAL_TEXT}\n" in out
    assert re.search(r"timing completed: spent \d+ of 5400 CPU-seconds and ", out)
    report = timing.load_timing(directory=tmp_path / "reports" / "finetune")
    assert report["status"] == "completed"
    assert report["approval"]["approved_cpu_hours"] == 1.5
    assert report["approval"]["given_at"] == GIVEN_AT_TEXT


def test_the_command_runs_writes_and_exits_0(command, tmp_path, capsys):
    assert cli.main(["finetune", "timing"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("timing estimate")  # its estimate first
    assert "timing completed" in out and "choice: device mps" in out
    assert re.search(r"wrote \S*reports/finetune/timing-2\.json and \S*/timing-2\.md\n", out)
    report = timing.load_timing(directory=tmp_path / "reports" / "finetune")
    assert report["status"] == "completed"
    assert (tmp_path / "weights" / "finetuned" / ".scratch" / report["scratch"]).is_dir()


@pytest.mark.parametrize(
    "scenario, code, prefix",
    [("mismatch", 1, "ERROR:"), ("stopped", 2, "REFUSED:"), ("over_budget", 2, "REFUSED:")],
)
def test_the_command_exit_codes(command, tmp_path, monkeypatch, capsys, scenario, code, prefix):
    if scenario == "mismatch":
        command.identical = False
    elif scenario == "stopped":
        limit = timing.TIMING_MAX_CPU_SECONDS
        command.overrides[("timing", "cpu", 8, 8, False)] = {"cpu": limit - 100.0}
    else:
        monkeypatch.setattr(timing, "PRIORS", {**timing.PRIORS, "M5": (1400.0, 0.0)})
    assert cli.main(["finetune", "timing"]) == code
    out = capsys.readouterr().out
    assert prefix in out
    written = (tmp_path / "reports" / "finetune" / f"{timing.TIMING_STEM}.json").is_file()
    assert written is (scenario != "over_budget")  # a refused estimate writes nothing


def test_the_command_refuses_a_concurrent_run(command, tmp_path, capsys):
    lock = tmp_path / "weights" / "finetuned" / ".scratch" / timing.LOCK_NAME
    with exclusive_lock(lock):
        assert cli.main(["finetune", "timing"]) == 2
    assert "REFUSED:" in capsys.readouterr().out and command.events == []


def test_the_command_refuses_a_changed_plan(command, tmp_path, capsys):
    directory = tmp_path / "reports" / "finetune"
    directory.mkdir(parents=True)
    (directory / f"{timing.TIMING_STEM}.config.json").write_bytes(canonical_json({"another": 1}))
    assert cli.main(["finetune", "timing"]) == 2
    assert "REFUSED:" in capsys.readouterr().out and command.events == []


@pytest.mark.parametrize(
    "argv",
    [
        ["finetune"],
        ["finetune", "fold"],
        ["finetune", "timing", "extra"],
        # the approval names the CPU-hours the owner approved, a finite number > 0 (D-046, D-055)
        ["finetune", "timing", "--approved-by-owner"],
        ["finetune", "timing", "--estimate", "--approved-by-owner"],
        ["finetune", "timing", "--approved-by-owner", "0"],
        ["finetune", "timing", "--approved-by-owner", "-1.5"],
        ["finetune", "timing", "--approved-by-owner", "nan"],
        ["finetune", "timing", "--approved-by-owner", "inf"],
        ["finetune", "timing", "--approved-by-owner", "yes"],
        ["finetune", "timing", "--approved-by-owner", "1.5h"],
    ],
)
def test_finetune_usage_errors_exit_2(command, tmp_path, argv, capsys):
    """A usage error exits 2 before anything runs or is written."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err and captured.out == ""
    assert command.events == [] and not (tmp_path / "reports").exists()
    assert not (tmp_path / "weights").exists()


# == the child: activation checkpointing in the preflight and on MPS ===============================

MPS = require_configured().mps_available


def test_the_preflight_checks_that_train_wrap_their_models_when_the_run_asks(tmp_path):
    import finetune_fakes as ff

    from kronos_agent.finetune import runner
    from kronos_agent.kronos.vendored import import_kronos

    files = ff.save_tiny_models(tmp_path)
    _, model = ff.load_saved(files)
    state = runner._snapshot(model)
    for wrapped in (False, True):
        spec = SimpleNamespace(
            compute={"activation_checkpointing": wrapped},
            predictor=SimpleNamespace(directory=files.predictor_dir),
        )
        built = runner._training_model(import_kronos(), spec, state, "cpu")
        assert [("forward" in vars(block)) for block in built.transformer] == [wrapped] * len(
            built.transformer
        )


def test_a_cpu_preflight_with_checkpointing_gives_the_values_without_it(tmp_path):
    """The wrap the MPS preflight asks for changes no CPU value: the CPU side of parity, steps
    and the deterministic check is bit-identical with and without it (test T17 for one step)."""
    import finetune_fakes as ff

    from kronos_agent.finetune import train
    from kronos_agent.finetune.train import Compute

    tiny = ff.tiny_run(tmp_path)
    checks: dict[bool, Any] = {}
    for wrapped in (False, True):
        request = tiny.request(
            mode="preflight",
            compute=Compute(threads=2, memory_guard=False, activation_checkpointing=wrapped),
            preflight_checks=("parity", "steps", "deterministic"),
        )
        result = train.run(request, root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
        assert result.status == "completed", ff.child_log(result.run_dir)
        checks[wrapped] = result.document["preflight"]
    assert checks[True] == checks[False]
    assert checks[True]["steps"]["finite"] is True and checks[True]["steps"]["steps"] == 20


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not MPS, reason="MPS is not available"))],
)
def test_checkpointing_reproduces_the_unwrapped_training_steps(device):
    """Three training steps with dropout and the router, with and without the wrap: bit-identical
    on the CPU; on MPS the same losses, gradients within float rounding and the generator where an
    unwrapped run leaves it (torch 2.14.1 restores only the CPU generator for the recomputation;
    without the replay the gradients differ by about 5e-2 and the generator moves on)."""
    import copy

    import finetune_fakes as ff
    import torch

    from kronos_agent.finetune import runner

    _, base = ff.build_tiny_models(seed=2)
    generator = torch.Generator().manual_seed(1)
    s1 = torch.randint(0, 16, (3, 41), generator=generator).to(device)
    s2 = torch.randint(0, 16, (3, 41), generator=generator).to(device)
    stamp = torch.zeros(3, 41, 5)
    stamp[..., 1] = torch.arange(41) % 24
    stamp = stamp.to(device)
    runs = {}
    for wrapped in (False, True):
        model = copy.deepcopy(base).to(device)
        if wrapped:
            runner.enable_activation_checkpointing(model)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.manual_seed(21)  # dropout on the device, the same stream in both runs
        losses = []
        with runner.MultinomialRouter(7).installed():
            for _ in range(3):
                logits = model(s1[:, :-1], s2[:, :-1], stamp[:, :-1])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])
                optimizer.zero_grad()
                loss.backward()
                gradients = {
                    name: parameter.grad.detach().to("cpu").clone()
                    for name, parameter in model.named_parameters()
                }
                optimizer.step()
                losses.append(float(loss.detach()))
        state = torch.mps.get_rng_state() if device == "mps" else torch.get_rng_state()
        runs[wrapped] = (losses, gradients, state)
    (plain_losses, plain, plain_state), (losses, wrapped_gradients, state) = runs[False], runs[True]
    assert torch.equal(state, plain_state)
    if device == "cpu":
        assert losses == plain_losses
        assert all(torch.equal(wrapped_gradients[name], plain[name]) for name in plain)
        return
    assert losses == pytest.approx(plain_losses, rel=1e-6)
    for name in plain:
        assert torch.allclose(wrapped_gradients[name], plain[name], rtol=1e-4, atol=1e-6), name


@pytest.mark.skipif(not MPS, reason="MPS is not available")
def test_an_mps_preflight_with_checkpointing_runs_its_checks(tmp_path):
    """The child accepts activation checkpointing in an MPS preflight and records every check
    (whether MPS passes is the measurement step's question, not this test's)."""
    import finetune_fakes as ff

    from kronos_agent.finetune import train
    from kronos_agent.finetune.train import Compute

    tiny = ff.tiny_run(tmp_path)
    compute = Compute(
        device="mps",
        threads=2,
        deterministic=False,
        memory_guard=False,
        activation_checkpointing=True,
    )
    request = tiny.request(
        mode="preflight", compute=compute, preflight_checks=("parity", "steps", "deterministic")
    )
    result = train.run(request, root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "completed", ff.child_log(result.run_dir)
    checks = result.document["preflight"]
    for outcome in checks.values():
        assert isinstance(outcome["passed"], bool) and "error" not in outcome, outcome
    assert checks["steps"]["finite"] is True
    assert checks["parity"]["gradient_cosine"] is not None


# == the real identity control (weights) ===========================================================


@pytest.mark.weights
def test_the_identity_control_runs_through_the_timing_path_with_the_real_weights(
    world, tmp_path, monkeypatch
):
    """M9 with the verified shipped weights (read-only, through symlinks in a temporary weights
    root), a scratch registry and a spawn-started worker, on CPU; the other measurements are
    stand-ins and MPS is left out."""
    from kronos_agent.kronos import batch, weights
    from kronos_agent.kronos.vendored import verify_vendored_tree
    from kronos_agent.paths import weights_dir

    real = weights_dir()
    staged = tmp_path / "weights"
    (staged / "NeoQuasar").mkdir(parents=True)
    for name in ("Kronos-small", "Kronos-Tokenizer-base"):
        (staged / "NeoQuasar" / name).symlink_to(real / "NeoQuasar" / name)
    monkeypatch.setenv("KRONOS_AGENT_WEIGHTS_DIR", str(staged))
    monkeypatch.setenv("KRONOS_AGENT_DATA_DIR", str(tmp_path / "data"))
    world.mps = False
    steps = replace(
        world.steps(),
        load_models=weights.load_models,
        identity_control=batch.identity_control,
    )
    report = timing.run_timing(
        params=PARAMS,
        steps=steps,
        root=tmp_path / "ft",
        directory=tmp_path / "reports",
        progress=lambda line: None,
    )
    m9 = report["measurements"]["M9"]
    assert m9["status"] == "completed" and m9["identical"] is True
    assert m9["checkpoint_id"] == batch.IDENTITY_CHECKPOINT_ID
    assert m9["shipped_config_id"] != m9["checkpoint_config_id"]
    assert m9["paths_sha256"]["shipped"] == m9["paths_sha256"]["checkpoint"]
    assert sorted(item["name"] for item in m9["files"]) == ["config.json", "model.safetensors"]
    assert report["status"] == "completed"
    assert verify_vendored_tree(strict=True).ok  # no bytecode under third_party/kronos
