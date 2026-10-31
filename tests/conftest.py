"""Shared test configuration.

Order matters: the runtime is configured before any test module can import ``torch`` or
``huggingface_hub``. A socket guard wraps every test from the start of its setup to the end of
its teardown, so it also covers the session- and module-scoped fixtures set up for the test
(``loaded_small`` among them), which pytest creates before any function-scoped fixture. The guard
refuses every socket connection unless the test carries the ``network`` marker, and keeps the
demo host unresolvable even when it does: no test, and no fixture set up for one, can reach
``open-api-vst.bingx.com`` (SPEC section 17, gate 1).
"""

from __future__ import annotations

import os
import socket

from kronos_agent import runtime

runtime.configure_runtime(2, strict=False)

import pandas as pd
import pytest

from kronos_agent.kronos.weights import weights_available
from kronos_agent.paths import vendored_kronos_dir

NETWORK_OPT_IN = os.environ.get("KRONOS_AGENT_NETWORK_TESTS") == "1"
DEMO_HOST = "open-api-vst.bingx.com"
WEIGHTS_SKIP_REASON = (
    "verified weights not present under weights/ "
    "(run: uv run kronos-agent fetch-weights --model small)"
)
NETWORK_SKIP_REASON = "network tests are opt-in: set KRONOS_AGENT_NETWORK_TESTS=1"


def has_marker(item: pytest.Item, name: str) -> bool:
    """Markers only: ``item.keywords`` also holds test names and parametrize ids."""
    return item.get_closest_marker(name) is not None


def pytest_collection_modifyitems(config, items):
    weights_ok = weights_available("small")
    for item in items:
        if has_marker(item, "weights") and not weights_ok:
            item.add_marker(pytest.mark.skip(reason=WEIGHTS_SKIP_REASON))
        if has_marker(item, "network") and not NETWORK_OPT_IN:
            item.add_marker(pytest.mark.skip(reason=NETWORK_SKIP_REASON))


def _install_network_guard(patcher: pytest.MonkeyPatch, *, network_marked: bool) -> None:
    """Socket-level guard: no network without the marker; the demo host never resolves."""
    real_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else str(host or "")
        if name.lower().rstrip(".") == DEMO_HOST:
            raise socket.gaierror(
                f"test network guard: {DEMO_HOST} is unreachable from the test suite"
            )
        return real_getaddrinfo(host, *args, **kwargs)

    patcher.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    if network_marked:
        return

    def refuse(*args, **kwargs):
        raise AssertionError("test network guard: unexpected network access")

    patcher.setattr(socket.socket, "connect", refuse)
    patcher.setattr(socket.socket, "connect_ex", refuse)
    patcher.setattr(socket, "create_connection", refuse)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Run the whole protocol of each test (setup, call, teardown) inside the socket guard.

    Not an autouse fixture: pytest sets every session- and module-scoped fixture up before the
    function-scoped ones, so such a fixture would leave ``loaded_small`` unguarded. The ``network``
    marker lifts the refusal for everything run on the test's behalf, wider fixtures included
    (the live module samples the server time in a module-scoped fixture).
    """
    with pytest.MonkeyPatch.context() as patcher:
        _install_network_guard(patcher, network_marked=has_marker(item, "network"))
        return (yield)


@pytest.fixture(scope="session")
def upstream_fixture_frame() -> pd.DataFrame:
    """The upstream regression input (5-minute bars with ``amount``), as shipped at the pinned commit."""
    return pd.read_csv(vendored_kronos_dir() / "tests" / "data" / "regression_input.csv")


@pytest.fixture(scope="session")
def loaded_small():
    """Kronos-small + Tokenizer-base loaded once per session (tests must carry the ``weights`` marker)."""
    from kronos_agent.kronos.weights import load_models

    return load_models("small")
