"""kronos_agent.envinfo: required keys, redaction, JSON, Markdown, weights and the clock probe."""

from __future__ import annotations

import argparse
import ast
import copy
import ipaddress
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from kronos_agent import envinfo
from kronos_agent.cli import verify as verify_cli
from kronos_agent.exchange import public_market
from kronos_agent.hashing import sha256_bytes
from kronos_agent.kronos.weights import load_registry
from kronos_agent.paths import REPO_ROOT, WEIGHTS_DIR_ENV

REQUIRED_TOP_LEVEL = {
    "timestamp_utc",
    "label",
    "network",
    "os",
    "arch",
    "cpu",
    "ram_gb",
    "swap_gb",
    "disk_free_gb",
    "paths",
    "python",
    "uv_version",
    "packages",
    "torch_runtime",
    "hf_hub_offline",
    "manifests",
    "clock",
    "git",
    "params_version",
}
REQUIRED_NESTED = {
    "os": {"system", "release", "version", "pretty"},
    "cpu": {"brand", "logical", "physical", "perf_cores", "eff_cores"},
    "disk_free_gb": {"repo", "data_dir", "weights_dir"},
    "paths": {"repo", "data_dir", "weights_dir"},
    "python": {"version", "implementation", "executable_redacted", "uv_managed"},
    "packages": set(envinfo.PACKAGES),
    "torch_runtime": {
        "threads",
        "interop_threads",
        "deterministic",
        "torch_version",
        "mps_available",
        "cuda_available",
        "hf_offline",
        "parallel",
        "build",
    },
    "manifests": {"kronos_source", "weights"},
    "git": {"is_repository", "commit", "branch", "dirty", "remotes_count"},
}
SECTION_HEADINGS = (
    "## summary",
    "## os",
    "## cpu",
    "## disk_free_gb",
    "## python",
    "## packages",
    "## torch_runtime",
    "## manifests",
    "## git",
)
SERVICE_ACCOUNT = "kronos"  # a project-named login: SPEC section 15 asks for a service user
SERVICE_HOST = "kronos-agent"
FAKE_REPO = "synthetic/Kronos-fake"
FAKE_CONFIG = b'{"d_model": 8}'
FAKE_WEIGHTS = b"synthetic weights, not a tensor file"


@pytest.fixture
def report() -> dict:
    """A fresh report per test, collected under the socket guard (no network allowed)."""
    return envinfo.collect(label="unit-test", network=False)


def test_collect_returns_required_keys(report):
    assert REQUIRED_TOP_LEVEL <= set(report)
    for section, keys in REQUIRED_NESTED.items():
        assert keys <= set(report[section]), section
    assert report["label"] == "unit-test"
    assert report["network"] is False
    assert report["clock"] is None
    assert report["arch"] == platform.machine()
    assert report["cpu"]["logical"] == os.cpu_count()
    assert report["ram_gb"] > 0
    assert report["disk_free_gb"]["repo"] > 0
    assert report["python"]["version"] == platform.python_version()
    assert report["packages"]["torch"] == report["torch_runtime"]["torch_version"]
    assert report["torch_runtime"]["threads"] == 2
    assert report["torch_runtime"]["deterministic"] is True
    assert report["torch_runtime"]["get_num_threads"] == 2
    assert report["torch_runtime"]["parallel"]["backend"]
    assert report["hf_hub_offline"] == "1"
    assert report["manifests"]["kronos_source"]["ok"] is True
    assert set(report["manifests"]["weights"]) == set(load_registry())
    statuses = {item["status"] for item in report["manifests"]["weights"].values()}
    assert statuses <= {"ok", "absent", "problems"}
    assert re.fullmatch(r"[0-9a-f]{12}", report["params_version"])
    assert isinstance(report["git"]["is_repository"], bool)


def test_collected_report_carries_no_identity(report):
    envinfo.assert_no_personal_data(report)
    dumped = json.dumps(report)
    node = platform.node()
    if len(node) > 1 and not envinfo.is_project_name_fragment(node):
        assert node not in dumped
    assert str(Path.home()) not in dumped
    for prefix in ("/Users", "/home"):
        assert prefix + "/" not in dumped
    assert report["python"]["executable_redacted"].startswith(("~", ".", "/"))
    assert not envinfo.IPV4_RE.search(dumped)


@pytest.mark.parametrize("kind", ["username", "hostname", "home", "ipv4"])
def test_assert_no_personal_data_rejects_injected_values(report, kind):
    if kind == "ipv4":
        marker = str(ipaddress.IPv4Address(0xC0A80114))  # no IPv4 literal may appear in a file
    else:
        marker = dict(envinfo.personal_markers()).get("home directory" if kind == "home" else kind)
    if marker is None:
        pytest.skip(f"{kind} is not a marker here (unavailable or part of the project's name)")
    tainted = copy.deepcopy(report)
    tainted["os"]["pretty"] = f"some system owned by {marker}"
    with pytest.raises(ValueError, match=r"os\.pretty"):
        envinfo.assert_no_personal_data(tainted)


def test_assert_no_personal_data_names_nested_paths_and_keys():
    ip = str(ipaddress.IPv4Address(0x0A000001))
    with pytest.raises(ValueError, match=r"IPv4 literal in report at a\.b\[1\]"):
        envinfo.assert_no_personal_data({"a": {"b": ["clean", f"host {ip}"]}})
    with pytest.raises(ValueError, match="IPv4"):
        envinfo.assert_no_personal_data({"section": {ip: "value stored under an address key"}})
    envinfo.assert_no_personal_data(
        {"version": "2.14.1", "count": 3, "ok": True, "none": None, "items": [1.5, "x", "a.b.c.d"]}
    )


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        ("kronos", True),
        ("Kronos", True),
        ("agent", True),
        ("kronos-agent", True),
        ("kronos_agent", True),
        ("kronos-agent-ops", False),
        ("kronosagent", False),
        ("zq-synthetic-user", False),
        # pieces of the project's words can be people's names: still identity
        ("ron", False),
        ("gent", False),
        ("ag", False),
    ],
)
def test_is_project_name_fragment(value, fragment):
    assert envinfo.is_project_name_fragment(value) is fragment


def test_project_named_account_and_host_are_not_identity(monkeypatch):
    """env-report must work for a ``kronos`` service account: "kronos" is in every report."""
    monkeypatch.setattr(envinfo.getpass, "getuser", lambda: SERVICE_ACCOUNT)
    monkeypatch.setattr(envinfo.socket, "gethostname", lambda: SERVICE_HOST)
    kinds = {kind for kind, _ in envinfo.personal_markers()}
    assert "username" not in kinds and "hostname" not in kinds
    report = envinfo.collect(label="svc", network=False)
    assert "kronos_source" in report["manifests"]  # a key the scan reads, in every report
    envinfo.assert_no_personal_data(report)


def test_home_directory_of_a_project_named_account_is_still_caught(monkeypatch):
    home = Path("/") / "home" / SERVICE_ACCOUNT  # assembled: no personal path literal in a file
    monkeypatch.setattr(envinfo.getpass, "getuser", lambda: SERVICE_ACCOUNT)
    monkeypatch.setenv("HOME", str(home))
    assert ("home directory", str(home)) in envinfo.personal_markers()
    envinfo.assert_no_personal_data({"paths": {"repo": "~/kronos-agent"}})
    with pytest.raises(ValueError, match=r"home directory\) in report at paths\.repo"):
        envinfo.assert_no_personal_data({"paths": {"repo": str(home / "kronos-agent")}})


@pytest.mark.parametrize("login", ["zq-synthetic-user", "kronos-agent-ops", "ron"])
def test_other_login_and_host_names_are_still_caught(monkeypatch, login):
    monkeypatch.setattr(envinfo.getpass, "getuser", lambda: login)
    monkeypatch.setattr(envinfo.socket, "gethostname", lambda: "zq-synthetic-host")
    with pytest.raises(ValueError, match=r"\(username\) in report at os\.pretty"):
        envinfo.assert_no_personal_data({"os": {"pretty": f"owned by {login}"}})
    with pytest.raises(ValueError, match=r"\(hostname\) in report at os\.pretty"):
        envinfo.assert_no_personal_data({"os": {"pretty": "built on zq-synthetic-host"}})


def test_json_round_trip_is_ascii(report):
    text = json.dumps(report, indent=2)
    assert text.isascii()
    assert json.loads(text) == report


def test_to_markdown_has_section_headings(report):
    text = envinfo.to_markdown(report)
    assert text.startswith("# Environment report: unit-test\n")
    for heading in SECTION_HEADINGS:
        assert f"\n{heading}\n" in text, heading
    assert "| key | value |" in text
    assert "| ram_gb |" in text
    assert "| parallel.backend |" in text
    assert "| version | " in text
    assert text.endswith("\n")


def test_headline_mentions_key_facts(report):
    line = envinfo.headline(report)
    assert "\n" not in line
    assert report["torch_runtime"]["torch_version"] in line
    assert report["python"]["version"] in line
    assert "kronos source OK" in line
    assert f"/{len(report['manifests']['weights'])} verified" in line


# --- weights status -------------------------------------------------------------------------------


@pytest.fixture
def fake_weights(tmp_path, monkeypatch) -> Path:
    """A one-repository registry (config + weights) wired into the environment report and
    ``kronos-agent verify``, with an empty weights root; returns the repository's directory."""
    files = [
        {"name": name, "size": len(content), "sha256": sha256_bytes(content)}
        for name, content in (("config.json", FAKE_CONFIG), ("model.safetensors", FAKE_WEIGHTS))
    ]
    repo = {
        "repo_id": FAKE_REPO,
        "role": "model",
        "revision": "0" * 40,
        "policy": "download_allowed",
        "files": files,
    }
    registry_path = tmp_path / "weights.json"
    registry_path.write_text(json.dumps({"schema_version": 1, "repos": [repo]}), encoding="utf-8")
    registry = load_registry(registry_path)
    monkeypatch.setattr(envinfo, "load_registry", lambda: registry)
    monkeypatch.setattr(verify_cli, "load_registry", lambda: registry)
    monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path / "weights"))
    return tmp_path / "weights" / FAKE_REPO


def _verify_weights(capsys) -> tuple[int, str]:
    parser = argparse.ArgumentParser()
    verify_cli.register(parser.add_subparsers(dest="command"))
    code = verify_cli.run(parser.parse_args(["verify", "--weights"]))
    return code, capsys.readouterr().out


def test_partial_weights_are_a_problem_not_absent(fake_weights, capsys):
    """Absent means none of the registered files is present; an interrupted download is a
    problem, both in the environment report and for ``kronos-agent verify``."""
    assert envinfo.collect_manifests()["weights"] == {
        FAKE_REPO: {"status": "absent", "policy": "download_allowed", "checked": 0, "problems": 2}
    }
    code, out = _verify_weights(capsys)
    assert (code, out) == (0, f"weights {FAKE_REPO}: absent (policy download_allowed)\n")

    fake_weights.mkdir(parents=True)
    (fake_weights / "config.json").write_bytes(FAKE_CONFIG)
    assert envinfo.collect_manifests()["weights"] == {
        FAKE_REPO: {"status": "problems", "policy": "download_allowed", "checked": 1, "problems": 1}
    }
    code, out = _verify_weights(capsys)
    assert code == 1
    assert f"weights {FAKE_REPO} PROBLEM: missing: {FAKE_REPO}/model.safetensors" in out
    assert "absent" not in out

    (fake_weights / "model.safetensors").write_bytes(FAKE_WEIGHTS)
    assert envinfo.collect_manifests()["weights"][FAKE_REPO]["status"] == "ok"
    assert _verify_weights(capsys) == (0, f"weights {FAKE_REPO}: OK (2 files)\n")


# --- clock probe ----------------------------------------------------------------------------------


def _one_shot(respond, calls: list[httpx.Request]) -> httpx.MockTransport:
    """Records the request; a second one fails the test (the probe never retries or backs off)."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert len(calls) == 1, "the clock probe sent a second request"
        return respond(request)

    return httpx.MockTransport(handler)


def _server_time(offset_ms: int):
    def respond(request: httpx.Request) -> httpx.Response:
        server_time = int(time.time() * 1000) + offset_ms
        return httpx.Response(200, json={"code": 0, "msg": "", "data": {"serverTime": server_time}})

    return respond


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _protocol_error(request: httpx.Request) -> httpx.Response:
    raise httpx.RemoteProtocolError("server closed the connection", request=request)


# case -> (responder, exception measure_clock raises, text its message carries)
CLOCK_FAILURES = {
    "connect-error": (_connect_error, public_market.BingXUnavailableError, "transport error"),
    "http-503-html": (
        lambda request: httpx.Response(503, text="<html>maintenance</html>"),
        public_market.BingXUnavailableError,
        "HTTP 503",
    ),
    "http-429": (lambda request: httpx.Response(429), public_market.BingXRateLimited, "rate"),
    "code-100410": (
        lambda request: httpx.Response(200, json={"code": 100410, "msg": "too frequent"}),
        public_market.BingXRateLimited,
        "rate",
    ),
    "code-100400": (
        lambda request: httpx.Response(200, json={"code": 100400, "msg": "bad request"}),
        public_market.BingXApiError,
        "code 100400",
    ),
    "redirect": (
        lambda request: httpx.Response(302, headers={"location": "https://example.invalid/"}),
        public_market.UnexpectedRedirect,
        "HTTP 302",
    ),
    "redirect-malformed-location": (
        lambda request: httpx.Response(302, headers={"location": "https:x"}),
        public_market.UnexpectedRedirect,
        "unparseable Location",
    ),
    "not-an-object": (
        lambda request: httpx.Response(200, json=[1, 2, 3]),
        public_market.BingXSchemaError,
        "JSON object",
    ),
    "protocol-error": (_protocol_error, httpx.RemoteProtocolError, "closed"),
}


def test_clock_offset_measured_through_one_public_get():
    calls: list[httpx.Request] = []
    report = envinfo.collect(
        label="unit-test", network=True, transport=_one_shot(_server_time(250), calls)
    )
    assert len(calls) == 1
    request = calls[0]
    assert request.method == "GET"
    assert request.url.scheme == "https"
    assert request.url.host == "open-api.bingx.com"
    assert request.url.path == "/openApi/swap/v2/server/time"
    assert "x-bx-apikey" not in {name.lower() for name in request.headers}
    assert request.headers["user-agent"] == public_market.USER_AGENT  # sent by the public client
    clock = report["clock"]
    assert {"local_utc_iso", "bingx_server_time_ms", "offset_ms", "round_trip_ms"} <= set(clock)
    assert abs(clock["offset_ms"] - 250) < 200
    assert clock["round_trip_ms"] >= 0
    assert report["network"] is True
    envinfo.assert_no_personal_data(report)


@pytest.mark.parametrize("case", sorted(CLOCK_FAILURES))
def test_clock_failure_is_one_get_reduced_to_the_exception_name(case):
    """Status before body (a 503 HTML page is not a JSON error), no retry and no backoff wait."""
    respond, raised, message = CLOCK_FAILURES[case]
    calls: list[httpx.Request] = []
    with pytest.raises(raised, match=message):
        envinfo.measure_clock(_one_shot(respond, calls))
    assert len(calls) == 1
    calls.clear()
    assert envinfo._clock_or_error(_one_shot(respond, calls)) == {"error": raised.__name__}
    assert len(calls) == 1


def test_collect_records_a_failed_clock_probe_by_its_exception_name():
    calls: list[httpx.Request] = []
    report = envinfo.collect(
        label="unit-test", network=True, transport=_one_shot(_connect_error, calls)
    )
    assert report["clock"] == {"error": "BingXUnavailableError"}
    assert len(calls) == 1


def test_envinfo_loads_the_public_client_and_never_the_demo_client():
    """The clock probe reuses the pinned public client; nothing that can sign is imported."""
    code = (
        "import sys, kronos_agent.envinfo\n"
        "assert 'kronos_agent.exchange.public_market' in sys.modules, 'public client missing'\n"
        "assert 'kronos_agent.exchange.demo_account' not in sys.modules, 'demo client imported'\n"
        "assert 'hmac' not in sys.modules, 'hmac imported'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-1000:]


def test_envinfo_builds_no_http_client_and_names_no_exchange_host():
    tree = ast.parse(Path(envinfo.__file__).read_text(encoding="utf-8"))
    hosts = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "bingx.com" in node.value
    ]
    assert hosts == []
    called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert {name for name in called if name.startswith("httpx.")} == set()
