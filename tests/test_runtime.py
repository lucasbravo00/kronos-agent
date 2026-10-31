"""kronos_agent.runtime: offline Hub, fixed threads and determinism, enforced in fresh processes.

Process-level facts (environment exported at import time, torch imported too early) can only be
proven in a subprocess with a clean interpreter; the in-process checks rely on ``conftest.py``
having configured this process with 2 threads.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from kronos_agent import runtime
from kronos_agent.paths import REPO_ROOT
from kronos_agent.runtime import RuntimeConfigError

IMPORT_EXPORTS_ENV = """
import os
before = (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("OMP_NUM_THREADS"))
import kronos_agent.runtime
after = (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("OMP_NUM_THREADS"))
print(before, after)
"""

CONFIGURE_THEN_INSPECT = """
from kronos_agent.runtime import configure_runtime, current
assert current() is None
info = configure_runtime(2)
import torch
import huggingface_hub.constants as hf_constants
print(
    bool(hf_constants.HF_HUB_OFFLINE),
    torch.get_num_threads(),
    torch.are_deterministic_algorithms_enabled(),
    info.threads,
    info.hf_offline,
    current() is info,
)
"""

TORCH_IMPORTED_FIRST = """
import torch
from kronos_agent.runtime import RuntimeConfigError, configure_runtime
try:
    configure_runtime(2)
except RuntimeConfigError as exc:
    print("RAISED", "torch" in str(exc))
else:
    print("NOT RAISED")
"""

TORCH_IMPORTED_FIRST_NON_STRICT = """
import torch
from kronos_agent.runtime import configure_runtime
info = configure_runtime(2, strict=False)
print(info.threads, torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
"""

REQUIRE_UNCONFIGURED = """
from kronos_agent.runtime import RuntimeConfigError, current, require_configured
assert current() is None
try:
    require_configured()
except RuntimeConfigError:
    print("RAISED")
"""


def run_python(code: str, *, drop: tuple[str, ...] = ()) -> str:
    """Run ``code`` in a fresh interpreter (the project venv) and return its stripped stdout."""
    env = {key: value for key, value in os.environ.items() if key not in drop}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_importing_runtime_exports_offline_and_thread_variables():
    output = run_python(IMPORT_EXPORTS_ENV, drop=("HF_HUB_OFFLINE", "OMP_NUM_THREADS"))
    assert output == "(None, None) ('1', '2')"


def test_configure_runtime_fixes_offline_threads_and_determinism():
    output = run_python(CONFIGURE_THEN_INSPECT, drop=("HF_HUB_OFFLINE",))
    assert output == "True 2 True 2 True True"


def test_torch_imported_before_strict_configuration_raises():
    assert run_python(TORCH_IMPORTED_FIRST) == "RAISED True"


def test_torch_imported_before_non_strict_configuration_is_tolerated():
    assert run_python(TORCH_IMPORTED_FIRST_NON_STRICT) == "2 2 True"


def test_require_configured_raises_in_an_unconfigured_process():
    assert run_python(REQUIRE_UNCONFIGURED) == "RAISED"


def test_reconfiguring_with_another_thread_count_raises():
    info = runtime.current()
    assert info is not None and info.threads == 2  # configured by conftest.py
    with pytest.raises(RuntimeConfigError, match="already configured with 2 threads"):
        runtime.configure_runtime(3)
    assert runtime.configure_runtime(2) is info  # idempotent for the same settings
    assert runtime.require_configured() is info


def test_in_process_state_matches_the_configuration():
    import huggingface_hub.constants as hf_constants
    import torch

    info = runtime.require_configured()
    assert hf_constants.HF_HUB_OFFLINE
    assert info.hf_offline is True
    assert torch.get_num_threads() == 2 == info.threads
    assert torch.are_deterministic_algorithms_enabled()
    assert info.deterministic is True
    assert info.torch_version == torch.__version__
    for key in runtime.THREAD_ENV_VARS:
        assert os.environ[key] == "2"
    for key, value in runtime.OFFLINE_ENV.items():
        assert os.environ[key] == value


def test_assert_hf_offline_raises_without_the_variable(monkeypatch):
    runtime.assert_hf_offline()
    monkeypatch.delenv("HF_HUB_OFFLINE")
    with pytest.raises(RuntimeConfigError, match="HF_HUB_OFFLINE=1"):
        runtime.assert_hf_offline()
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with pytest.raises(RuntimeConfigError):
        runtime.assert_hf_offline()


def test_set_process_env_rejects_non_positive_threads():
    with pytest.raises(ValueError):
        runtime.set_process_env(0)
