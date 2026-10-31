"""Redacted environment report (SPEC section 16): what the machine is, never whose it is.

The report covers the operating system, CPU, memory, free disk space, the interpreter, the pinned
packages, the torch build and runtime settings, the state of the manifests and, on request, the
clock offset against the public BingX server time (one GET through the public market client).
It never records ``platform.node()``, the login name, the raw home directory or an IP address:
the few paths it needs pass through :func:`kronos_agent.paths.redact_home`, and
:func:`assert_no_personal_data` scans every string before :func:`collect` returns.
"""

from __future__ import annotations

import getpass
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from . import runtime
from .exchange.public_market import BingXRateLimited, PublicMarketClient, PublicMarketError
from .kronos.vendored import verify_vendored_tree
from .kronos.weights import load_registry, verify_repo
from .params import params_version
from .paths import REPO_ROOT, data_dir, redact_home, weights_dir

PACKAGES: tuple[str, ...] = (
    "torch",
    "numpy",
    "pandas",
    "huggingface_hub",
    "safetensors",
    "einops",
    "httpx",
    "pyyaml",
)
BUILD_FLAGS: tuple[str, ...] = (
    "BLAS_INFO",
    "LAPACK_INFO",
    "USE_OPENMP",
    "USE_MKL",
    "USE_MKLDNN",
    "USE_CUDA",
)
PROJECT_NAMES: tuple[str, ...] = ("kronos-agent", "kronos_agent")
PROJECT_WORDS = frozenset({"kronos", "agent", *PROJECT_NAMES})
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
GIB = float(1 << 30)
COMMAND_TIMEOUT_S = 15.0


def _run(*command: str, env: Mapping[str, str] | None = None) -> str | None:
    """Stripped stdout of a short external command; ``None`` when it is absent, fails or hangs."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            check=False,
            env=dict(env) if env is not None else None,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _sysctl(name: str) -> str | None:
    return _run("sysctl", "-n", name) if sys.platform == "darwin" else None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _to_int(text: str | None) -> int | None:
    try:
        return int(text) if text else None
    except ValueError:
        return None


def _os_pretty() -> str | None:
    """``macOS 26.3 (25D125)`` from ``sw_vers``; ``PRETTY_NAME`` of ``/etc/os-release`` on Linux."""
    if sys.platform == "darwin":
        name, version, build = (
            _run("sw_vers", flag) for flag in ("-productName", "-productVersion", "-buildVersion")
        )
        if name and version:
            return f"{name} {version}" + (f" ({build})" if build else "")
        return None
    if sys.platform.startswith("linux"):
        for line in (_read_text(Path("/etc/os-release")) or "").splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.partition("=")[2].strip().strip('"')
    return None


def collect_os() -> dict[str, Any]:
    """Kernel name, release and version plus the distribution name; never the node name."""
    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "pretty": _os_pretty(),
    }


def _linux_cpu() -> tuple[str | None, int | None]:
    """(model name, physical core count) from ``/proc/cpuinfo``."""
    text = _read_text(Path("/proc/cpuinfo"))
    if not text:
        return None, None
    brand: str | None = None
    cores: set[tuple[str, str]] = set()
    for block in text.strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        brand = brand or fields.get("model name")
        if "core id" in fields:
            cores.add((fields.get("physical id", "0"), fields["core id"]))
    return brand, (len(cores) or None)


def collect_cpu() -> dict[str, Any]:
    """Brand string and core counts; the performance/efficiency split only where macOS reports it."""
    brand = physical = perf = eff = None
    if sys.platform == "darwin":
        brand = _sysctl("machdep.cpu.brand_string") or None
        physical = _to_int(_sysctl("hw.physicalcpu"))
        perf = _to_int(_sysctl("hw.perflevel0.physicalcpu"))
        eff = _to_int(_sysctl("hw.perflevel1.physicalcpu"))
    elif sys.platform.startswith("linux"):
        brand, physical = _linux_cpu()
    return {
        "brand": brand,
        "logical": os.cpu_count(),
        "physical": physical,
        "perf_cores": perf,
        "eff_cores": eff,
    }


def _ram_gb() -> float | None:
    """Installed memory in GiB (``sysconf`` first, ``hw.memsize`` as the macOS fallback)."""
    try:
        total: int | None = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total = _to_int(_sysctl("hw.memsize"))
    return round(total / GIB, 2) if total else None


def _swap_gb() -> float | None:
    """Configured swap in GiB; ``None`` when the platform does not expose it."""
    if sys.platform == "darwin":
        match = re.search(r"total = ([\d.]+)([KMGT])", _sysctl("vm.swapusage") or "")
        if not match:
            return None
        scale = {"K": 1.0 / (1 << 20), "M": 1.0 / 1024, "G": 1.0, "T": 1024.0}[match.group(2)]
        return round(float(match.group(1)) * scale, 2)
    meminfo = _read_text(Path("/proc/meminfo")) or ""
    match = re.search(r"^SwapTotal:\s+(\d+) kB", meminfo, re.MULTILINE)
    return round(int(match.group(1)) * 1024 / GIB, 2) if match else None


def _free_gb(path: Path) -> float | None:
    """Free space in GiB on the volume holding ``path`` (or its nearest existing parent)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    try:
        return round(shutil.disk_usage(path).free / GIB, 2)
    except OSError:
        return None


def collect_disk() -> dict[str, float | None]:
    """Free space of the volumes holding the repository, ``data/`` and ``weights/``."""
    return {
        "repo": _free_gb(REPO_ROOT),
        "data_dir": _free_gb(data_dir()),
        "weights_dir": _free_gb(weights_dir()),
    }


def _uv_managed() -> bool:
    """True when the interpreter comes from uv's Python store or the venv was created by uv."""
    candidates = (sys.executable, sys.base_prefix, getattr(sys, "_base_executable", None) or "")
    if any("/uv/python/" in str(location).replace("\\", "/") for location in candidates):
        return True
    for cfg in (Path(sys.prefix) / "pyvenv.cfg", REPO_ROOT / ".venv" / "pyvenv.cfg"):
        lines = (_read_text(cfg) or "").splitlines()
        if any(line.partition("=")[0].strip() == "uv" for line in lines):
            return True
    return False


def collect_python() -> dict[str, Any]:
    """Interpreter version, implementation, redacted executable path and uv provenance."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable_redacted": redact_home(sys.executable),
        "uv_managed": _uv_managed(),
    }


def collect_packages() -> dict[str, str | None]:
    """Installed versions of the pinned packages from distribution metadata (nothing imported)."""
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_torch_runtime() -> dict[str, Any]:
    """Thread, determinism and build facts of torch, read after the runtime is configured."""
    info = runtime.current() or runtime.configure_runtime(runtime.DEFAULT_THREADS, strict=False)
    import torch  # only after configure_runtime(): see kronos_agent.runtime

    parallel_info = torch.__config__.parallel_info()
    build_info = torch.__config__.show()
    backend = re.search(r"^ATen parallel backend: (.+)$", parallel_info, re.MULTILINE)
    concurrency = re.search(r"hardware_concurrency\(\) : (\d+)", parallel_info)
    flags: dict[str, str] = {}
    for key in BUILD_FLAGS:
        match = re.search(rf"\b{key}=([^,\n]*)", build_info)
        if match:
            flags[key] = match.group(1).strip()
    return {
        "threads": info.threads,
        "interop_threads": info.interop_threads,
        "deterministic": info.deterministic,
        "torch_version": info.torch_version,
        "mps_available": info.mps_available,
        "cuda_available": info.cuda_available,
        "hf_offline": info.hf_offline,
        "get_num_threads": torch.get_num_threads(),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "parallel": {
            "backend": backend.group(1).strip() if backend else None,
            "hardware_concurrency": int(concurrency.group(1)) if concurrency else None,
            "openmp": bool(torch.backends.openmp.is_available()),
            "mkl": bool(torch.backends.mkl.is_available()),
            "mkldnn": bool(torch.backends.mkldnn.is_available()),
            "accelerate": flags.get("BLAS_INFO") == "accelerate",
        },
        "build": flags,
    }


def collect_manifests() -> dict[str, Any]:
    """Verification state of the vendored Kronos tree and of every registered weights repository."""
    source = verify_vendored_tree(strict=True, check_blob_ids=True)
    weights: dict[str, Any] = {}
    for repo_id, repo in sorted(load_registry().items()):
        report = verify_repo(repo)
        if report.ok:
            status = "ok"
        elif report.checked == 0:  # none of its files is present; a partial repo is a problem
            status = "absent"
        else:
            status = "problems"
        weights[repo_id] = {
            "status": status,
            "policy": repo.policy,
            "checked": report.checked,
            "problems": len(report.problems),
        }
    return {
        "kronos_source": {
            "ok": source.ok,
            "checked": source.checked,
            "problems": len(source.problems),
        },
        "weights": weights,
    }


def _refuse_to_wait(seconds: float) -> None:
    """``sleep`` of the clock probe's client: with pacing off and no transient retries, the only
    wait left is the rate-limit backoff, and the probe never sends a second GET."""
    raise BingXRateLimited(f"rate limited; the clock probe does not wait {seconds:g} s to retry")


def measure_clock(transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    """One unauthenticated GET to the public BingX server time; offset = server - local midpoint.

    The GET goes through :class:`~kronos_agent.exchange.public_market.PublicMarketClient`: host
    pin, path allow-list, status checked before the body and schema guard are the public client's.
    Pacing is off and nothing is retried (a transport error, a 5xx or a rate-limit answer raises at
    once), so the probe is exactly one request; the round trip excludes building the client.
    ``transport`` lets tests substitute ``httpx.MockTransport``.
    """
    with PublicMarketClient(
        transport=transport, min_interval_s=0.0, max_retries=0, sleep=_refuse_to_wait
    ) as client:
        sent = datetime.now(UTC)
        started = time.perf_counter()
        server_ms = client.server_time_ms()
        round_trip_ms = (time.perf_counter() - started) * 1000.0
    local_ms = sent.timestamp() * 1000.0 + round_trip_ms / 2.0
    return {
        "local_utc_iso": sent.isoformat(timespec="milliseconds"),
        "bingx_server_time_ms": server_ms,
        "offset_ms": round(server_ms - local_ms, 1),
        "round_trip_ms": round(round_trip_ms, 1),
    }


def _clock_or_error(transport: httpx.BaseTransport | None) -> dict[str, Any]:
    """``measure_clock`` reduced to the exception class name on failure (messages may carry hosts).

    ``httpx.HTTPError`` stays in the list for what the public client does not translate (protocol
    errors such as ``RemoteProtocolError``).
    """
    try:
        return measure_clock(transport)
    except (
        PublicMarketError,
        httpx.HTTPError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        RuntimeError,
    ) as exc:
        return {"error": type(exc).__name__}


def collect_git() -> dict[str, Any]:
    """Commit, branch, dirty flag and remote count of the repository at ``REPO_ROOT`` (no URLs)."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}

    def git(*args: str) -> str | None:
        return _run("git", "-C", str(REPO_ROOT), *args, env=env)

    toplevel = git("rev-parse", "--show-toplevel")
    if not toplevel or Path(toplevel).resolve() != REPO_ROOT.resolve():
        return {
            "is_repository": False,
            "commit": None,
            "branch": None,
            "dirty": False,
            "remotes_count": 0,
        }
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or git("symbolic-ref", "--short", "HEAD")
    status = git("status", "--porcelain")
    remotes = git("remote")
    return {
        "is_repository": True,
        "commit": git("rev-parse", "HEAD") or None,
        "branch": branch or None,
        "dirty": bool(status) if status is not None else None,
        "remotes_count": len(remotes.splitlines()) if remotes is not None else None,
    }


def is_project_name_fragment(value: str) -> bool:
    """True when ``value`` (any case) is the project's name or one of its words, e.g. ``kronos``.

    Exact words only: a shorter piece such as ``ron`` or ``gent`` can be a person's name.
    """
    return value.lower() in PROJECT_WORDS


def personal_markers() -> list[tuple[str, str]]:
    """(kind, value) pairs that must never appear in a report: login name, hostname, home directory.

    A login or host name that is the project's name or one of its words (a service account
    ``kronos``, a server ``kronos-agent``) identifies nobody and sits in every report and nearly
    every file of the repository, so it is skipped; the home directory never is.
    """
    markers: list[tuple[str, str]] = []
    try:
        markers.append(("username", getpass.getuser()))
    except (OSError, KeyError, ImportError):
        pass  # no account database (containers, CI)
    try:
        markers.append(("hostname", socket.gethostname()))
    except OSError:
        pass
    markers.append(("home directory", str(Path.home())))
    return [
        (kind, value)
        for kind, value in markers
        if len(value) > 1 and (kind == "home directory" or not is_project_name_fragment(value))
    ]


def _strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Every string in a nested report (keys included) with its dotted key path."""
    if isinstance(value, str):
        yield path or "<root>", value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield child, str(key)
            yield from _strings(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def assert_no_personal_data(report: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` naming the key path if a string in ``report`` identifies the owner.

    Checked as substrings: the login name, the hostname and the home directory of the current
    process (:func:`personal_markers`, which skips a login or host name that is the project's
    name or one of its words); additionally any IPv4 literal.
    """
    markers = personal_markers()
    for path, text in _strings(report):
        for kind, marker in markers:
            if marker in text:
                raise ValueError(f"personal data ({kind}) in report at {path}")
        if IPV4_RE.search(text):
            raise ValueError(f"IPv4 literal in report at {path}")


def collect(
    *, label: str, network: bool = False, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """Build the redacted environment report (JSON-serialisable, ASCII).

    ``network=True`` adds exactly one unauthenticated GET to the public BingX server time under
    ``clock``; otherwise ``clock`` is ``None``. ``transport`` is for tests. The runtime is
    configured (2 threads, non-strict) if nobody did it yet, so the torch facts describe the
    configured state. The result passes :func:`assert_no_personal_data` before it is returned.
    """
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "network": network,
        "os": collect_os(),
        "arch": platform.machine(),
        "cpu": collect_cpu(),
        "ram_gb": _ram_gb(),
        "swap_gb": _swap_gb(),
        "disk_free_gb": collect_disk(),
        "paths": {
            "repo": redact_home(str(REPO_ROOT)),
            "data_dir": redact_home(str(data_dir())),
            "weights_dir": redact_home(str(weights_dir())),
        },
        "python": collect_python(),
        "uv_version": _run("uv", "--version") or None,
        "packages": collect_packages(),
        "torch_runtime": collect_torch_runtime(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "manifests": collect_manifests(),
        "clock": _clock_or_error(transport) if network else None,
        "git": collect_git(),
        "params_version": params_version(),
    }
    assert_no_personal_data(report)
    return report


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            rows.extend(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return rows
    return [(prefix, value)]


def _cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[tuple[str, Any]]) -> list[str]:
    return [
        "| key | value |",
        "| --- | --- |",
        *(f"| {key} | {_cell(value)} |" for key, value in rows),
    ]


def to_markdown(report: Mapping[str, Any]) -> str:
    """Compact Markdown: ``## summary`` for the top-level scalars, then one table per section."""
    lines = [f"# Environment report: {report.get('label', '')}", "", "## summary", ""]
    lines += _table(
        [(key, value) for key, value in report.items() if not isinstance(value, Mapping)]
    )
    for name, section in report.items():
        if isinstance(section, Mapping):
            lines += ["", f"## {name}", "", *_table(_flatten(section))]
    return "\n".join(lines) + "\n"


def headline(report: Mapping[str, Any]) -> str:
    """One line for the terminal: OS, CPU, RAM, Python, torch and manifests status."""
    os_info, cpu, python, torch_info, manifests = (
        report[key] for key in ("os", "cpu", "python", "torch_runtime", "manifests")
    )
    cores = f"{cpu['logical']} logical"
    if cpu["perf_cores"] is not None and cpu["eff_cores"] is not None:
        cores += f", {cpu['perf_cores']}P+{cpu['eff_cores']}E"
    weights_ok = sum(1 for item in manifests["weights"].values() if item["status"] == "ok")
    return " | ".join(
        [
            os_info["pretty"] or f"{os_info['system']} {os_info['release']}",
            f"{cpu['brand'] or report['arch']} ({cores})",
            f"RAM {report['ram_gb']} GiB",
            f"Python {python['version']}" + (" (uv)" if python["uv_managed"] else ""),
            f"torch {torch_info['torch_version']}, {torch_info['threads']} threads"
            + (", deterministic" if torch_info["deterministic"] else ""),
            (
                f"manifests: kronos source "
                f"{'OK' if manifests['kronos_source']['ok'] else 'PROBLEMS'}, "
                f"weights {weights_ok}/{len(manifests['weights'])} verified"
            ),
        ]
    )
