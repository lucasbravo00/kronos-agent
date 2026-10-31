"""Process-level runtime configuration (offline Hub, fixed threads, deterministic algorithms).

Import this module, or call :func:`configure_runtime`, BEFORE ``torch`` and ``huggingface_hub``:
``HF_HUB_OFFLINE`` is read by huggingface_hub at import time and the thread-count environment
variables are read by the BLAS/OpenMP runtimes when torch loads. Importing this module already
exports the environment with the default thread count; :func:`configure_runtime` then fixes the
torch thread pool. A second call with a different thread count raises: the thread count cannot
change inside a process without breaking comparability.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

DEFAULT_THREADS = 2
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
HEAVY_MODULES = ("torch", "huggingface_hub")


class RuntimeConfigError(RuntimeError):
    """The runtime was configured too late, twice with different settings, or not at all."""


@dataclass(frozen=True)
class RuntimeInfo:
    threads: int
    interop_threads: int
    deterministic: bool
    torch_version: str
    mps_available: bool
    cuda_available: bool
    hf_offline: bool


_configured: RuntimeInfo | None = None


def set_process_env(threads: int = DEFAULT_THREADS) -> None:
    """Export the offline and thread-count variables (explicit set, never ``setdefault``)."""
    if threads < 1:
        raise ValueError("threads must be >= 1")
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value
    for key in THREAD_ENV_VARS:
        os.environ[key] = str(threads)


def heavy_modules_imported() -> list[str]:
    return [name for name in HEAVY_MODULES if name in sys.modules]


def configure_runtime(
    threads: int = DEFAULT_THREADS, *, deterministic: bool = True, strict: bool = True
) -> RuntimeInfo:
    """Fix threads and determinism for this process; idempotent for the same settings."""
    global _configured
    if _configured is not None:
        if threads != _configured.threads:
            raise RuntimeConfigError(
                f"runtime already configured with {_configured.threads} threads; "
                f"cannot switch to {threads} inside the same process"
            )
        return _configured
    already = heavy_modules_imported()
    if already and strict:
        raise RuntimeConfigError(
            f"configure_runtime() must run before importing {already}: "
            "environment variables set now would be ignored"
        )
    set_process_env(threads)
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # the inter-op pool was already started by an earlier parallel region
    if deterministic:
        torch.use_deterministic_algorithms(True)
    import huggingface_hub.constants as hf_constants

    mps = getattr(torch.backends, "mps", None)
    _configured = RuntimeInfo(
        threads=threads,
        interop_threads=torch.get_num_interop_threads(),
        deterministic=deterministic,
        torch_version=torch.__version__,
        mps_available=bool(mps is not None and mps.is_available()),
        cuda_available=bool(torch.cuda.is_available()),
        hf_offline=bool(hf_constants.HF_HUB_OFFLINE),
    )
    return _configured


def current() -> RuntimeInfo | None:
    return _configured


def require_configured() -> RuntimeInfo:
    if _configured is None:
        raise RuntimeConfigError("call kronos_agent.runtime.configure_runtime() first")
    return _configured


def assert_hf_offline() -> None:
    """Fail loudly if the Hub could be contacted."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeConfigError("HF_HUB_OFFLINE=1 must be set before importing huggingface_hub")
    if "huggingface_hub" in sys.modules:
        import huggingface_hub.constants as hf_constants

        if not hf_constants.HF_HUB_OFFLINE:
            raise RuntimeConfigError("huggingface_hub was imported before HF_HUB_OFFLINE=1 was set")


set_process_env(DEFAULT_THREADS)
