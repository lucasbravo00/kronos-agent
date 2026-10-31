"""The ``conftest`` socket guard covers fixtures of every scope, not only the test body.

pytest sets session- and module-scoped fixtures up before any function-scoped one, so a guard
installed by a function-scoped autouse fixture would leave ``loaded_small`` and every other wider
fixture unguarded. The probes below try each guarded way out from such fixtures. The connection
probes target the loopback discard port and the resolution probe runs with the real resolver
stubbed out, so even a broken guard sends nothing off this machine.
"""

from __future__ import annotations

import _socket
import contextlib
import socket

import pytest

DEMO_HOST = "open-api-vst.bingx.com"
LOOPBACK_DISCARD = ("localhost", 9)
GUARD_MESSAGE = "test network guard"


def resolve_demo_host_fail_closed() -> None:
    """The guard answers for the demo host; the real resolver is stubbed so that a broken guard
    fails here instead of sending a DNS query."""

    def real_resolver(*args, **kwargs):
        raise AssertionError("the demo host reached the real resolver: the guard is broken")

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(_socket, "getaddrinfo", real_resolver)
        with pytest.raises(socket.gaierror, match=GUARD_MESSAGE):
            socket.getaddrinfo(DEMO_HOST, 443)


def assert_refused_by_the_guard() -> None:
    """Connections are refused before reaching the socket layer; the demo host never resolves."""
    with pytest.raises(AssertionError, match=GUARD_MESSAGE):
        socket.create_connection(LOOPBACK_DISCARD, timeout=1)
    with socket.socket() as sock:
        for connect in (sock.connect, sock.connect_ex):
            with pytest.raises(AssertionError, match=GUARD_MESSAGE):
                connect(LOOPBACK_DISCARD)
    resolve_demo_host_fail_closed()


@pytest.fixture(scope="session")
def session_fixture_probe() -> str:
    """Session scope, like ``loaded_small``: set up before every function-scoped fixture."""
    assert_refused_by_the_guard()
    return "session"


@pytest.fixture(scope="module")
def module_fixture_probe() -> str:
    """Module scope, like ``greedy`` or ``reference_fan``: set up before the function scope."""
    assert_refused_by_the_guard()
    return "module"


def test_session_scoped_fixtures_are_set_up_under_the_guard(session_fixture_probe):
    assert session_fixture_probe == "session"


def test_module_scoped_fixtures_are_set_up_under_the_guard(module_fixture_probe):
    assert module_fixture_probe == "module"


def test_the_test_body_runs_under_the_guard():
    assert_refused_by_the_guard()


@pytest.mark.parametrize("label", ["network", "weights"])
def test_a_parametrize_id_named_like_a_marker_changes_nothing(label, request):
    # parametrize ids become pytest keywords; only a real marker may skip a test or lift the guard
    assert label in request.node.keywords
    assert request.node.get_closest_marker(label) is None
    assert_refused_by_the_guard()


def test_marker_named_ids_were_not_skipped_at_collection(request):
    prefix = "test_a_parametrize_id_named_like_a_marker_changes_nothing["
    collected = [item for item in request.session.items if item.name.startswith(prefix)]
    for item in collected:  # empty when ``-k`` deselected them: nothing to check then
        assert item.get_closest_marker("skip") is None, item.nodeid


@pytest.fixture(scope="module")
def module_fixture_of_a_network_test() -> str:
    """Set up for a ``network`` test, like the live module's ``server_time``: the refusal is lifted
    for it as well (only the guard's AssertionError would escape), the demo host stays hidden."""
    with contextlib.suppress(OSError):  # nothing listens there: the attempt reached the socket
        socket.create_connection(LOOPBACK_DISCARD, timeout=1).close()
    resolve_demo_host_fail_closed()
    return "module"


@pytest.mark.network
def test_the_network_marker_also_lifts_the_refusal_for_wider_fixtures(
    module_fixture_of_a_network_test,
):
    """Opt-in like every ``network`` test, although it only touches the loopback interface."""
    assert module_fixture_of_a_network_test == "module"
