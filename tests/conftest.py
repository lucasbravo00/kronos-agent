"""Shared test configuration.

Order matters: the runtime is configured before any test module can import ``torch`` or
``huggingface_hub``. A socket guard wraps every test from the start of its setup to the end of
its teardown, so it also covers the session- and module-scoped fixtures set up for the test
(``loaded_small`` among them), which pytest creates before any function-scoped fixture. The guard
refuses every socket connection unless the test carries the ``network`` marker, and keeps the
demo host unresolvable even when it does: no test, and no fixture set up for one, can reach
``open-api-vst.bingx.com`` (SPEC section 17, gate 1).

The suite never writes under the real data, weights or reports roots (D-055 (6); the full rules
are in ``root_guard.py``). Every test runs on its own temporary roots: the autouse fixture
``isolated_roots`` points ``KRONOS_AGENT_DATA_DIR``, ``KRONOS_AGENT_WEIGHTS_DIR`` and
``KRONOS_AGENT_REPORTS_DIR`` at them, and between tests (collection, session- and module-scoped
fixtures) the variables point at the session's own temporary roots. An audit hook refuses every
write under a real root that the test process attempts through Python, the native writers the
suite uses in this process (safetensors' ``save_file`` and ``save_model``, ``torch.save``) are
wrapped to refuse the same, and a refused write fails the test that attempted it. Other native
writes and child processes are seen only by the listing, after the fact: the session fails when
the real roots, listed at its start, differ at its end. Opt-ins read a real root and never write
it: the ``weights`` marker makes the test's weights root a view of the shipped weights (its
fine-tune root stays the test's own), and ``real_roots("data", "weights", "reports")`` makes each
named root a view of the whole real root. With ``KRONOS_AGENT_TEST_ROOTS_LISTING`` set to a file
outside the real roots, the listing taken at session start is written there.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator

from kronos_agent import runtime

runtime.configure_runtime(2, strict=False)

import pandas as pd
import pytest
import root_guard

from kronos_agent.kronos.weights import weights_available
from kronos_agent.paths import vendored_kronos_dir

NETWORK_OPT_IN = os.environ.get("KRONOS_AGENT_NETWORK_TESTS") == "1"
DEMO_HOST = "open-api-vst.bingx.com"
WEIGHTS_SKIP_REASON = (
    "verified weights not present under weights/ "
    "(run: uv run kronos-agent fetch-weights --model small)"
)
NETWORK_SKIP_REASON = "network tests are opt-in: set KRONOS_AGENT_NETWORK_TESTS=1"
# read when this module is imported, before the session or any test changes the variables
REAL_ROOTS = root_guard.RealRoots.from_environment()


def has_marker(item: pytest.Item, name: str) -> bool:
    """Markers only: ``item.keywords`` also holds test names and parametrize ids."""
    return item.get_closest_marker(name) is not None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{root_guard.VIEWS_MARKER}(*names): read the real 'data', 'weights' or 'reports' root "
        "through a view of it, never write it (tests/root_guard.py)",
    )
    listing = root_guard.listing_path(os.environ.get(root_guard.LISTING_ENV), REAL_ROOTS.protected)
    config.pluginmanager.register(
        root_guard.RootGuardPlugin(
            REAL_ROOTS, guard=root_guard.GUARD, listing=listing, isolate=True
        ),
        root_guard.PLUGIN_NAME,
    )


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


@pytest.fixture(autouse=True)
def isolated_roots(request) -> Iterator[root_guard.IsolatedRoots]:
    """The test's own data, weights and report roots (new, empty temporary directories, or views
    of the real roots for the ``weights`` marker and ``real_roots``), set in the environment for
    the test and the processes it starts. Its own monkeypatch, so that a test's
    ``monkeypatch.undo()`` keeps them. At teardown, fails the test that (itself or through a
    fixture) attempted a write under a real root, which the guard refused."""
    plugin = request.config.pluginmanager.get_plugin(root_guard.PLUGIN_NAME)
    if plugin is None:  # blocked with -p no:...: no test runs unguarded
        pytest.fail(f"the {root_guard.PLUGIN_NAME} plugin is not registered", pytrace=False)
    try:
        views = root_guard.view_names(request.node.iter_markers(root_guard.VIEWS_MARKER))
    except ValueError as exc:
        pytest.fail(str(exc), pytrace=False)
    roots = plugin.factory.make(views=views, shipped_weights=has_marker(request.node, "weights"))
    with pytest.MonkeyPatch.context() as patcher:
        for variable, value in roots.environment().items():
            patcher.setenv(variable, value)
        with root_guard.GUARD.collect() as refused:
            yield roots
    root_guard.fail_on_refusals(refused)


@pytest.fixture(scope="session")
def upstream_fixture_frame() -> pd.DataFrame:
    """The upstream regression input (5-minute bars with ``amount``), as shipped at the pinned commit."""
    return pd.read_csv(vendored_kronos_dir() / "tests" / "data" / "regression_input.csv")


@pytest.fixture(scope="session")
def loaded_small():
    """Kronos-small + Tokenizer-base loaded once per session (tests must carry the ``weights`` marker)."""
    from kronos_agent.kronos.weights import load_models

    return load_models("small")
