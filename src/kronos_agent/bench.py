"""Benchmark of the Kronos fan: timing, memory, determinism and the upstream fixture replica.

In-process measurements run on the reference device (CPU, with the thread count fixed by
``kronos_agent.runtime.configure_runtime``). Thread sweeps and the MPS comparison run in
subprocesses (``python -m kronos_agent.bench --child ...``, one JSON line each) because torch
cannot change its thread pool once started and MPS is not a reference device. Results are
JSON-serialisable dictionaries; ``kronos-agent bench`` writes them to
``reports/environment/<label>.bench.json`` and ``.bench.md``. They never contain hostnames, user
names or absolute home paths.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .envinfo import assert_no_personal_data
from .kronos.features import (
    FEATURE_COLUMNS,
    UPSTREAM_FIXTURE_RELPATH,
    future_timestamps,
    synthetic_window,
    upstream_fixture_frame,
    upstream_fixture_window,
)
from .kronos.forecast import Fan, ForecastRequest, forecast_fan
from .kronos.weights import LoadedModels, load_models
from .models import MODEL_SPECS
from .paths import (
    REPO_ROOT,
    default_label,
    redact_home,
    reports_dir,
    validate_label,
    vendored_kronos_dir,
)
from .runtime import RuntimeConfigError, configure_runtime, require_configured
from .seeds import Seeds, seed_everything, seed_for_decision

SCHEMA_VERSION = 1
KIND = "kronos_forecast_benchmark"
SOURCES: tuple[str, ...] = ("upstream_fixture", "synthetic")
REFERENCE_DEVICE = "cpu"

BENCH_DECISION_HOUR = datetime(2026, 10, 29, 16, tzinfo=UTC)
BENCH_STREAM = "bench"
WARMUP_SALT = "warmup"
REFERENCE_SALT = "r0"
ALTERNATIVE_SALT = "alt"
TEMPERATURE = 1.0
TOP_P = 0.9
TOP_K = 0

FIXTURE_CONTEXTS: tuple[int, ...] = (512, 256)
FIXTURE_RTOL = 1e-5
UPSTREAM_SEEDS = Seeds(
    canonical="third_party/kronos/tests/test_kronos_regression.py set_seed(123)",
    digest="",
    torch_seed=123,
    numpy_seed=123,
)

CHILD_TIMEOUT_S = 1800.0
_PERSONAL_PATH_RE = re.compile(r"/(?:Users|home)/[^/\s'\"]+")

FAN_METHOD = "predict_batch([context] * K, [x_ts] * K, [y_ts] * K, pred_len=N, sample_count=1)"
DETERMINISM_NOTE = (
    "same machine, same build, same thread count and same seeds reproduce a fan bit for bit; "
    "other thread counts, devices or architectures are not guaranteed to (recorded, not required)"
)
MEAN_PATH_NOTE = (
    "predict(sample_count=K) averages K sampled paths and returns ONE path; it is timed for "
    "comparison only, the fan is built with predict_batch and sample_count=1"
)
FIXTURE_NOTE = (
    "offline replica of the upstream greedy regression test (seed 123, top_k=1, top_p=1.0, "
    "T=1.0, raw fixture timestamps, predictor.predict); the platform that generated the fixtures "
    "is unknown, so the comparison is report-only (decision D-007)"
)


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB.

    ``ru_maxrss`` is reported in bytes on macOS and in KiB on Linux.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return usage / divisor


def redact(text: str) -> str:
    """Replace the home directory, and any macOS- or Linux-style user-home prefix, with ``~``."""
    return _PERSONAL_PATH_RE.sub("~", redact_home(text))


def fan_fingerprint(fan: Fan) -> str:
    """sha256 of the fan's shape and float64 bytes; equal fingerprints mean bit-identical fans."""
    digest = hashlib.sha256()
    digest.update(repr(fan.paths.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(fan.paths, dtype=np.float64).tobytes())
    return digest.hexdigest()


def bench_seeds(salt: str) -> Seeds:
    """Seeds of the benchmark stream for the fixed decision hour 2026-10-29T16:00Z."""
    return seed_for_decision(BENCH_DECISION_HOUR, stream=BENCH_STREAM, salt=salt)


def parse_threads_sweep(text: str | None) -> list[int]:
    """``"1,2,10"`` -> ``[1, 2, 10]`` (deduplicated, order kept); empty or ``None`` -> ``[]``."""
    if text is None or not text.strip():
        return []
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise ValueError(
                f"--threads-sweep expects integers separated by commas, got {text!r}"
            ) from None
        if value < 1:
            raise ValueError(f"thread counts must be >= 1, got {value}")
        if value not in values:
            values.append(value)
    return values


@dataclass(frozen=True)
class BenchInput:
    """One forecast context with its continuation timestamps and a JSON-ready description."""

    context: pd.DataFrame
    x_timestamps: pd.Series
    y_timestamps: pd.Series
    step: pd.Timedelta
    description: dict[str, Any]


def build_input(source: str, max_context: int, horizon: int) -> BenchInput:
    """Context of ``max_context`` rows from ``source`` plus ``horizon`` future timestamps."""
    if source == "upstream_fixture":
        context, x_timestamps, step = upstream_fixture_window(max_context)
        description: dict[str, Any] = {
            "source": source,
            "file": "third_party/kronos/" + UPSTREAM_FIXTURE_RELPATH,
            "timestamps": (
                "regular 5-minute grid from the fixture's first timestamp "
                "(the fixture itself has session gaps)"
            ),
            "amount": "provided by the fixture",
        }
    elif source == "synthetic":
        context, x_timestamps, step = synthetic_window(max_context)
        description = {
            "source": source,
            "generator": (
                "seeded geometric Brownian motion (kronos_agent.kronos.features.synthetic_window)"
            ),
            "seed": 0,
            "amount": "volume * mean(open, high, low, close) proxy",
        }
    else:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    y_timestamps = future_timestamps(x_timestamps, step, horizon)
    description.update(
        {
            "rows": len(context),
            "columns": list(context.columns),
            "step_seconds": float(step.total_seconds()),
            "first_timestamp": x_timestamps.iloc[0].isoformat(),
            "last_timestamp": x_timestamps.iloc[-1].isoformat(),
            "horizon": int(horizon),
        }
    )
    return BenchInput(context, x_timestamps, y_timestamps, step, description)


def load_models_quiet(
    model: str, device: str = REFERENCE_DEVICE
) -> tuple[LoadedModels, float, list[str]]:
    """Load the models, timing the call and capturing what the Hub mixin prints to stdout.

    ``PyTorchModelHubMixin`` prints ``Loading weights from local directory`` for every local
    load; the captured lines are returned (redacted) as evidence that no Hub lookup happened.
    """
    buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        models = load_models(
            model, device=device, allow_non_reference_device=device != REFERENCE_DEVICE
        )
    wall = time.perf_counter() - started
    messages = [redact(line.strip()) for line in buffer.getvalue().splitlines() if line.strip()]
    return models, wall, messages


def run_fan(
    models: LoadedModels, inp: BenchInput, trajectories: int, horizon: int, salt: str
) -> tuple[Fan, float]:
    """One fan with the benchmark seeds of ``salt``; returns it with its wall-clock seconds."""
    request = ForecastRequest(
        context=inp.context,
        x_timestamps=inp.x_timestamps,
        y_timestamps=inp.y_timestamps,
        horizon=horizon,
        trajectories=trajectories,
        seeds=bench_seeds(salt),
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )
    started = time.perf_counter()
    fan = forecast_fan(models, request)
    return fan, time.perf_counter() - started


def measure_fans(
    models: LoadedModels,
    inp: BenchInput,
    *,
    trajectories: int,
    horizon: int,
    repeats: int,
    determinism: bool = True,
) -> dict[str, Any]:
    """Warm-up fan (untimed), ``repeats`` timed fans with salts ``r0, r1, ...`` and, optionally,
    the same-seed / different-seed checks against the ``r0`` fan."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    _, warmup_seconds = run_fan(models, inp, trajectories, horizon, WARMUP_SALT)
    fans: list[Fan] = []
    seconds: list[float] = []
    salts: list[str] = []
    for index in range(repeats):
        salt = f"r{index}"
        fan, wall = run_fan(models, inp, trajectories, horizon, salt)
        fans.append(fan)
        seconds.append(wall)
        salts.append(salt)
    reference = fans[0]
    median = statistics.median(seconds)
    result: dict[str, Any] = {
        "warmup_seconds": warmup_seconds,
        "seconds": seconds,
        "median_seconds": median,
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "seconds_per_trajectory_candle": median / (trajectories * horizon),
        "model_elapsed_seconds": [fan.elapsed_s for fan in fans],
        "seed_salts": salts,
        "reference_seed": {
            "salt": REFERENCE_SALT,
            "canonical": reference.seeds.canonical,
            "torch_seed": reference.seeds.torch_seed,
            "numpy_seed": reference.seeds.numpy_seed,
        },
        "fan_fingerprint": fan_fingerprint(reference),
        "fan_shape": list(reference.paths.shape),
        "finite": bool(np.isfinite(reference.paths).all()),
    }
    if determinism:
        again, again_seconds = run_fan(models, inp, trajectories, horizon, REFERENCE_SALT)
        if repeats >= 2:
            other, other_salt = fans[1], salts[1]
        else:
            other, _ = run_fan(models, inp, trajectories, horizon, ALTERNATIVE_SALT)
            other_salt = ALTERNATIVE_SALT
        result["determinism"] = {
            "same_seed_identical": bool(np.array_equal(reference.paths, again.paths)),
            "same_seed_max_abs_diff": float(np.max(np.abs(reference.paths - again.paths))),
            "same_seed_repeat_seconds": again_seconds,
            "different_seed_differs": bool(not np.array_equal(reference.paths, other.paths)),
            "different_seed_salt": other_salt,
        }
    return result


def measure_mean_path(
    models: LoadedModels, inp: BenchInput, *, sample_count: int, horizon: int
) -> dict[str, Any]:
    """Time ``predictor.predict(sample_count=K)``: one averaged path, documented as NOT a fan."""
    import torch

    seed_everything(bench_seeds(REFERENCE_SALT))
    started = time.perf_counter()
    with torch.no_grad():
        frame = models.predictor.predict(
            inp.context,
            inp.x_timestamps,
            inp.y_timestamps,
            pred_len=horizon,
            T=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            sample_count=sample_count,
            verbose=False,
        )
    seconds = time.perf_counter() - started
    return {
        "method": f"predictor.predict(sample_count={sample_count})",
        "sample_count": sample_count,
        "seconds": seconds,
        "shape": list(frame.shape),
        "is_fan": False,
        "note": MEAN_PATH_NOTE,
    }


@dataclass
class FixtureReplica:
    """Greedy prediction versus an upstream ``regression_output_<n>.csv`` fixture (float32)."""

    context_len: int
    pred_len: int
    obtained: np.ndarray
    expected: np.ndarray
    seconds: float

    @property
    def abs_diff(self) -> np.ndarray:
        return np.abs(self.obtained.astype(np.float64) - self.expected.astype(np.float64))

    @property
    def max_abs_diff(self) -> float:
        return float(self.abs_diff.max())

    @property
    def max_rel_diff(self) -> float:
        return float((self.abs_diff / (np.abs(self.expected.astype(np.float64)) + 1e-9)).max())

    @property
    def passes_rtol(self) -> bool:
        """The criterion of ``np.testing.assert_allclose(obtained, expected, rtol=1e-5)``."""
        return bool(
            np.all(self.abs_diff <= FIXTURE_RTOL * np.abs(self.expected.astype(np.float64)))
        )

    def summary(self) -> dict[str, Any]:
        return {
            "context_len": self.context_len,
            "pred_len": self.pred_len,
            "shape": list(self.obtained.shape),
            "finite": bool(np.isfinite(self.obtained).all()),
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "passes_rtol_1e-5": self.passes_rtol,
            "seconds": self.seconds,
        }


def greedy_fixture_replica(models: LoadedModels, context_len: int) -> FixtureReplica:
    """Offline replica of upstream ``test_kronos_predictor_regression`` for one context length.

    Seeds 123 for ``random``/``numpy``/``torch`` (the upstream ``set_seed``), greedy decoding
    (``top_k=1, top_p=1.0, T=1.0``), the first ``context_len`` fixture rows with their RAW
    timestamps, ``pred_len`` taken from the fixture (8), through ``predictor.predict`` as
    upstream does; compared with ``regression_output_<context_len>.csv``.
    """
    if context_len not in FIXTURE_CONTEXTS:
        raise ValueError(f"context_len must be one of {FIXTURE_CONTEXTS}, got {context_len}")
    frame = upstream_fixture_frame()
    expected_frame = pd.read_csv(
        vendored_kronos_dir() / "tests" / "data" / f"regression_output_{context_len}.csv",
        parse_dates=["timestamps"],
    )
    pred_len = len(expected_frame)
    context = frame.iloc[:context_len]
    features = context[list(FEATURE_COLUMNS)].reset_index(drop=True)
    x_timestamps = context["timestamps"].reset_index(drop=True)
    y_timestamps = (
        frame["timestamps"].iloc[context_len : context_len + pred_len].reset_index(drop=True)
    )
    if not y_timestamps.equals(expected_frame["timestamps"]):
        raise ValueError(f"fixture timestamps do not line up for context {context_len}")
    expected = expected_frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    import torch

    started = time.perf_counter()
    seed_everything(UPSTREAM_SEEDS)
    with torch.no_grad():
        prediction = models.predictor.predict(
            df=features,
            x_timestamp=x_timestamps,
            y_timestamp=y_timestamps,
            pred_len=pred_len,
            T=1.0,
            top_k=1,
            top_p=1.0,
            verbose=False,
            sample_count=1,
        )
    seconds = time.perf_counter() - started
    obtained = prediction[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    return FixtureReplica(context_len, pred_len, obtained, expected, seconds)


def child_measurement(
    *, model: str, device: str, trajectories: int, horizon: int, repeats: int, source: str
) -> dict[str, Any]:
    """The worker's measurement: load, warm-up, timed fans and the ``r0`` fingerprint."""
    info = require_configured()
    models, load_wall, hub_messages = load_models_quiet(model, device)
    inp = build_input(source, models.spec.max_context, horizon)
    stats = measure_fans(
        models, inp, trajectories=trajectories, horizon=horizon, repeats=repeats, determinism=False
    )
    return {
        "threads": info.threads,
        "device": models.device,
        "deterministic": info.deterministic,
        "torch_version": info.torch_version,
        "load_seconds": models.load_seconds,
        "load_wall_seconds": load_wall,
        "hub_messages": hub_messages,
        **stats,
        "peak_rss_mb": peak_rss_mb(),
    }


def run_child(
    *,
    threads: int,
    device: str,
    model: str,
    trajectories: int,
    horizon: int,
    repeats: int,
    source: str,
    timeout: float = CHILD_TIMEOUT_S,
) -> dict[str, Any]:
    """Run ``python -m kronos_agent.bench --child ...`` and parse its JSON line.

    A failing worker never raises here: the record carries ``error`` (redacted) instead.
    """
    command = [
        sys.executable,
        "-m",
        "kronos_agent.bench",
        "--child",
        "--threads",
        str(threads),
        "--device",
        device,
        "--model",
        model,
        "--trajectories",
        str(trajectories),
        "--horizon",
        str(horizon),
        "--repeats",
        str(repeats),
        "--source",
        source,
    ]
    record: dict[str, Any] = {"threads": threads, "device": device, "error": None}
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        record["error"] = f"worker timed out after {timeout:.0f} s"
        return record
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload: Any = None
    if stdout_lines:
        try:
            payload = json.loads(stdout_lines[-1])
        except json.JSONDecodeError:
            payload = None
    if not isinstance(payload, dict):
        stderr_lines = [line for line in completed.stderr.splitlines() if line.strip()]
        detail = f": {stderr_lines[-1]}" if stderr_lines else ""
        record["error"] = redact(
            f"worker exited with code {completed.returncode} without a JSON result{detail}"
        )
        return record
    record.update(payload)
    return record


def _identical_to_reference(record: dict[str, Any], fingerprint: str) -> bool | None:
    if record.get("error") or "fan_fingerprint" not in record:
        return None
    return bool(record["fan_fingerprint"] == fingerprint)


def _iter_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _iter_strings(key)
            yield from _iter_strings(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_strings(item)


def assert_no_personal_data_in(result: dict[str, Any]) -> None:
    """Raise ``ValueError`` if the report carries personal data or the home directory path.

    ``kronos_agent.envinfo.assert_no_personal_data`` checks the user name, host name, home
    directory and IPv4 literals; the loop below repeats the home-directory check with a
    benchmark-specific message.
    """
    assert_no_personal_data(result)
    home = str(Path.home())
    if home and home != "/":
        for text in _iter_strings(result):
            if home in text:
                raise ValueError("benchmark report contains the home directory path")


def run_benchmark(
    *,
    model: str = "small",
    trajectories: int = 20,
    horizon: int = 8,
    threads: int = 2,
    repeats: int = 3,
    source: str = "upstream_fixture",
    include_fixture_replica: bool = True,
    threads_sweep: Sequence[int] | None = None,
    include_mps: bool = False,
    label: str | None = None,
) -> dict[str, Any]:
    """Measure the fan of ``trajectories`` x ``horizon`` on CPU and return a JSON-ready report.

    The runtime must already be configured with ``threads`` (the CLI does it before importing
    torch). In-process: timed model load, context from ``source``, one warm-up fan, ``repeats``
    timed fans (seeds of the benchmark stream, salts ``r0, r1, ...``), same-seed and
    different-seed checks, ``predict(sample_count=trajectories)`` for comparison, peak RSS and,
    with ``include_fixture_replica``, the greedy replica of the upstream fixtures (report-only).
    ``threads_sweep`` and ``include_mps`` spawn one worker subprocess per entry and record
    whether its ``r0`` fan is bit-identical to the in-process one.
    """
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODEL_SPECS)}")
    for name, value in (
        ("trajectories", trajectories),
        ("horizon", horizon),
        ("threads", threads),
        ("repeats", repeats),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    label = validate_label(label or default_label())
    sweep = list(dict.fromkeys(threads_sweep or []))
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in sweep):
        raise ValueError(f"threads_sweep entries must be positive integers, got {sweep}")
    info = require_configured()
    if info.threads != threads:
        raise RuntimeConfigError(
            f"runtime configured with {info.threads} threads but the benchmark asked for "
            f"{threads}; call configure_runtime({threads}) before importing torch"
        )

    models, load_wall, hub_messages = load_models_quiet(model, REFERENCE_DEVICE)
    inp = build_input(source, models.spec.max_context, horizon)
    fan_stats = measure_fans(
        models, inp, trajectories=trajectories, horizon=horizon, repeats=repeats, determinism=True
    )
    determinism = fan_stats.pop("determinism")
    mean_path = measure_mean_path(models, inp, sample_count=trajectories, horizon=horizon)

    fixture: dict[str, Any] | None = None
    if include_fixture_replica:
        fixture = {
            "note": FIXTURE_NOTE,
            "rtol": FIXTURE_RTOL,
            "report_only": True,
            "contexts": {
                str(context_len): greedy_fixture_replica(models, context_len).summary()
                for context_len in FIXTURE_CONTEXTS
            },
        }

    reference_fingerprint = fan_stats["fan_fingerprint"]
    worker_args = {
        "model": model,
        "trajectories": trajectories,
        "horizon": horizon,
        "repeats": repeats,
        "source": source,
    }
    sweep_records: list[dict[str, Any]] | None = None
    if sweep:
        sweep_records = []
        for count in sweep:
            record = run_child(threads=count, device=REFERENCE_DEVICE, **worker_args)
            record["identical_to_reference_fan"] = _identical_to_reference(
                record, reference_fingerprint
            )
            sweep_records.append(record)
    mps: dict[str, Any] | None = None
    if include_mps:
        mps = run_child(threads=threads, device="mps", **worker_args)
        mps["identical_to_reference_fan"] = _identical_to_reference(mps, reference_fingerprint)
        mps["note"] = "MPS is not a reference device (SPEC section 16); informational only"

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "label": label,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": {
            "system": sys.platform,
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "runtime": {
            "device": models.device,
            "threads": info.threads,
            "interop_threads": info.interop_threads,
            "deterministic": info.deterministic,
            "torch_version": info.torch_version,
            "hf_offline": info.hf_offline,
            "mps_available": info.mps_available,
            "cuda_available": info.cuda_available,
        },
        "threads": info.threads,
        "model": {
            "name": models.spec.name,
            "repo_id": models.spec.repo_id,
            "tokenizer_repo_id": models.spec.tokenizer_repo_id,
            "max_context": models.spec.max_context,
            "weights_sha256": dict(models.weights_sha256),
            "load_seconds": models.load_seconds,
            "load_wall_seconds": load_wall,
            "hub_messages": hub_messages,
        },
        "input": inp.description,
        "fan": {
            "method": FAN_METHOD,
            "trajectories": trajectories,
            "horizon": horizon,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repeats": repeats,
            **fan_stats,
        },
        "determinism": {**determinism, "note": DETERMINISM_NOTE},
        "mean_path": mean_path,
        "peak_rss_mb": peak_rss_mb(),
        "fixture_replica": fixture,
        "threads_sweep": sweep_records,
        "mps": mps,
    }
    assert_no_personal_data_in(result)
    return result


def _seconds(value: Any) -> str:
    return f"{value:.2f} s" if isinstance(value, (int, float)) else "n/a"


def _flag(value: Any) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def headline_lines(result: dict[str, Any]) -> list[str]:
    """The numbers worth printing after a run."""
    fan = result["fan"]
    model = result["model"]
    runtime = result["runtime"]
    determinism = result["determinism"]
    mean_path = result["mean_path"]
    lines = [
        (
            f"Kronos-{model['name']} fan {fan['trajectories']} x {fan['horizon']} on "
            f"{runtime['device']} with {result['threads']} threads: median "
            f"{fan['median_seconds']:.2f} s (min {fan['min_seconds']:.2f}, max "
            f"{fan['max_seconds']:.2f}; {fan['repeats']} run(s) after 1 warm-up), "
            f"{fan['seconds_per_trajectory_candle'] * 1000:.1f} ms per trajectory-candle"
        ),
        (
            f"model load {model['load_seconds']:.2f} s; peak RSS {result['peak_rss_mb']:.0f} MiB; "
            f"torch {runtime['torch_version']}"
        ),
        (
            f"same seed identical: {determinism['same_seed_identical']}; "
            f"different seed differs: {determinism['different_seed_differs']}"
        ),
        (
            f"predict(sample_count={mean_path['sample_count']}) mean path: "
            f"{mean_path['seconds']:.2f} s (one path, not a fan)"
        ),
    ]
    fixture = result.get("fixture_replica")
    if fixture:
        for context_len, summary in fixture["contexts"].items():
            verdict = "pass" if summary["passes_rtol_1e-5"] else "fail"
            lines.append(
                f"upstream greedy fixture, context {context_len}: max_rel_diff "
                f"{summary['max_rel_diff']:.2e} (rtol 1e-5: {verdict}, report-only)"
            )
    for record in result.get("threads_sweep") or []:
        if record.get("error"):
            lines.append(f"threads {record['threads']}: error: {record['error']}")
        else:
            lines.append(
                f"threads {record['threads']}: median {record['median_seconds']:.2f} s; "
                "fan identical "
                f"to the reference fan: {_flag(record.get('identical_to_reference_fan'))}"
            )
    mps = result.get("mps")
    if mps:
        if mps.get("error"):
            lines.append(f"mps: error: {mps['error']}")
        else:
            lines.append(
                f"mps: median {mps['median_seconds']:.2f} s; fan identical to the CPU fan: "
                f"{_flag(mps.get('identical_to_reference_fan'))} (not a reference device)"
            )
    return lines


def render_markdown(result: dict[str, Any]) -> str:
    """A short Markdown report with the same numbers as the JSON."""
    fan = result["fan"]
    model = result["model"]
    runtime = result["runtime"]
    determinism = result["determinism"]
    mean_path = result["mean_path"]
    inp = result["input"]
    lines = [
        f"# Kronos forecast benchmark: {result['label']}",
        "",
        (
            f"Generated {result['generated_at']}. Kronos-{model['name']} with its paired "
            "tokenizer, "
            f"device {runtime['device']}, {runtime['threads']} threads, torch "
            f"{runtime['torch_version']}, deterministic algorithms "
            f"{'on' if runtime['deterministic'] else 'off'}. Input: {inp['source']} "
            f"({inp['rows']} rows, step {inp['step_seconds']:.0f} s, horizon {inp['horizon']})."
        ),
        "",
        "| Metric | Value |",
        "| --- | --- |",
        (
            f"| Fan {fan['trajectories']} x {fan['horizon']} (`predict_batch`, `sample_count=1`), "
            f"median of {fan['repeats']} run(s) after 1 warm-up | {fan['median_seconds']:.2f} s "
            f"(min {fan['min_seconds']:.2f}, max {fan['max_seconds']:.2f}) |"
        ),
        f"| Seconds per trajectory-candle | {fan['seconds_per_trajectory_candle']:.4f} |",
        f"| Model load (`load_models`) | {model['load_seconds']:.2f} s |",
        f"| Peak RSS of the benchmark process | {result['peak_rss_mb']:.0f} MiB |",
        f"| Same seed twice gives an identical fan | {_flag(determinism['same_seed_identical'])} |",
        (
            "| Different seed gives a different fan | "
            f"{_flag(determinism['different_seed_differs'])} |"
        ),
        (
            f"| `predict(sample_count={mean_path['sample_count']})`, one mean path (not a fan) | "
            f"{mean_path['seconds']:.2f} s |"
        ),
    ]
    fixture = result.get("fixture_replica")
    if fixture:
        for context_len, summary in fixture["contexts"].items():
            verdict = "pass" if summary["passes_rtol_1e-5"] else "fail"
            lines.append(
                f"| Upstream greedy fixture, context {context_len} (report-only) | max_rel_diff "
                f"{summary['max_rel_diff']:.2e}, rtol 1e-5 {verdict} |"
            )
    sweep = result.get("threads_sweep")
    if sweep:
        lines += [
            "",
            "## Thread sweep (subprocesses, informational)",
            "",
            "| Threads | Median | Min | Max | Identical to the reference fan | Peak RSS | Error |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for record in sweep:
            rss = record.get("peak_rss_mb")
            lines.append(
                f"| {record['threads']} | {_seconds(record.get('median_seconds'))} | "
                f"{_seconds(record.get('min_seconds'))} | {_seconds(record.get('max_seconds'))} | "
                f"{_flag(record.get('identical_to_reference_fan'))} | "
                f"{f'{rss:.0f} MiB' if isinstance(rss, (int, float)) else 'n/a'} | "
                f"{record.get('error') or '-'} |"
            )
    mps = result.get("mps")
    if mps:
        lines += [
            "",
            "## MPS comparison (informational; MPS is not a reference device)",
            "",
            "| Device | Median | Identical to the CPU fan | Error |",
            "| --- | --- | --- | --- |",
            (
                f"| mps | {_seconds(mps.get('median_seconds'))} | "
                f"{_flag(mps.get('identical_to_reference_fan'))} | {mps.get('error') or '-'} |"
            ),
        ]
    lines += [
        "",
        "Notes:",
        "",
        f"- {DETERMINISM_NOTE}.",
        f"- {MEAN_PATH_NOTE}.",
        f"- {FIXTURE_NOTE}.",
        "",
    ]
    return "\n".join(lines)


def write_reports(
    result: dict[str, Any], *, label: str | None = None, directory: Path | None = None
) -> tuple[Path, Path]:
    """Write ``<label>.bench.json`` and ``<label>.bench.md`` (default: ``reports/environment/``).

    An explicit ``label`` renames the report: file names, JSON and Markdown all carry it.
    """
    label = validate_label(label or result["label"])
    report = {**result, "label": label}
    directory = directory or reports_dir() / "environment"
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{label}.bench.json"
    md_path = directory / f"{label}.bench.md"
    json_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _child_main(argv: Sequence[str]) -> int:
    """Entry point of ``python -m kronos_agent.bench --child ...``: prints exactly one JSON line."""
    parser = argparse.ArgumentParser(
        prog="python -m kronos_agent.bench",
        description="benchmark worker spawned by kronos_agent.bench.run_benchmark",
    )
    parser.add_argument(
        "--child", action="store_true", required=True, help="worker invocation marker"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default=REFERENCE_DEVICE)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="small")
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source", choices=SOURCES, default="upstream_fixture")
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {"threads": args.threads, "device": args.device, "error": None}
    try:
        configure_runtime(args.threads)
        payload.update(
            child_measurement(
                model=args.model,
                device=args.device,
                trajectories=args.trajectories,
                horizon=args.horizon,
                repeats=args.repeats,
                source=args.source,
            )
        )
    except Exception as exc:  # noqa: BLE001 - the parent records the failure instead of dying
        payload["error"] = redact(f"{type(exc).__name__}: {exc}")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    sys.exit(_child_main(sys.argv[1:]))
