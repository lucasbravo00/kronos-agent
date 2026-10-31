"""The ``kronos-agent`` console script: registry, help, params, verify, gate-4 refusal, download
errors, vendor check, env-report.

``--help`` is exercised through ``uv run kronos-agent`` in a subprocess (the real entry point);
the other commands run in-process through ``kronos_agent.cli.main`` so that the socket guard of
``conftest.py`` proves they never touch the network. No wall-clock budget is asserted:
a cheap ``--help`` is checked by what it imports, and timing is the benchmark's job.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys

import httpx
import pytest

from kronos_agent import cli
from kronos_agent.kronos.weights import WeightsError
from kronos_agent.params import params_version
from kronos_agent.paths import REPO_ROOT

COMMANDS = ("env-report", "verify", "fetch-weights", "bench", "vendor-kronos", "params")


def console_script(*args: str) -> list[str]:
    """``uv run --offline kronos-agent ...`` when uv is available, else ``cli.main`` in a fresh
    interpreter (``kronos_agent.cli`` has no ``__main__`` module, so ``-m`` cannot run it)."""
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "--offline", "kronos-agent", *args]
    code = "import sys; from kronos_agent.cli import main; sys.exit(main(sys.argv[1:]))"
    return [sys.executable, "-c", code, *args]


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        console_script(*args),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
        check=False,
    )


def test_command_registry_names_every_step_one_command():
    assert set(cli.COMMAND_MODULES) == set(COMMANDS)
    for name, module_name in cli.COMMAND_MODULES.items():
        module = importlib.import_module(module_name)
        assert callable(module.register) and callable(module.run), name
    _parser, runners = cli.build_parser()
    assert set(runners) == set(COMMANDS)


def test_build_parser_fails_loudly_on_a_missing_command_module(monkeypatch):
    missing = "kronos_agent.cli.bench_missing"
    monkeypatch.setitem(cli.COMMAND_MODULES, "bench", missing)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        cli.build_parser()
    assert excinfo.value.name == missing


def test_help_lists_commands():
    completed = run_script("--help")
    assert completed.returncode == 0, completed.stderr
    # argparse prints the sub-command choices as one "{a,b,...}" group in the usage line.
    choices = re.search(r"\{([^{}]+)\}", completed.stdout)
    assert choices is not None, completed.stdout
    assert set(choices.group(1).split(",")) == set(COMMANDS)


def test_console_script_fallback_runs_without_uv(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert console_script("params", "version")[0] == sys.executable
    completed = run_script("params", "version")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == params_version()


def test_help_imports_no_heavy_modules():
    code = (
        "import sys, kronos_agent.cli as cli; cli.build_parser(); "
        "heavy = ('torch', 'huggingface_hub', 'pandas', 'numpy'); "
        "print(sorted(m for m in heavy if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_params_validate(capsys):
    assert cli.main(["params", "validate"]) == 0
    out = capsys.readouterr().out
    assert "params.yaml valid" in out
    assert "model=small" in out


def test_params_version_prints_12_hex(capsys):
    assert cli.main(["params", "version"]) == 0
    printed = capsys.readouterr().out.strip()
    assert re.fullmatch(r"[0-9a-f]{12}", printed)
    assert printed == params_version()


def test_params_version_via_console_script():
    completed = run_script("params", "version")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == params_version()


def test_verify_kronos(capsys):
    assert cli.main(["verify", "--kronos"]) == 0
    assert "third_party/kronos: OK (27 files verified)" in capsys.readouterr().out


@pytest.mark.parametrize("model", ["base", "mini"])
def test_fetch_weights_gated_model_is_refused_without_network(capsys, model):
    # Any connection attempt would raise AssertionError from the socket guard of conftest.py.
    assert cli.main(["fetch-weights", "--model", model]) == 2
    out = capsys.readouterr().out
    assert out.startswith("REFUSED")
    assert "gate 4" in out
    assert "--approved-by-owner" in out


@pytest.mark.parametrize(
    "error",
    [
        WeightsError("HTTP 404 for config.json"),
        httpx.ConnectError("name resolution failed"),
        httpx.ReadError("connection reset while streaming"),
    ],
    ids=["weights-error", "connect-error", "read-error"],
)
def test_fetch_weights_reports_download_failures_without_a_traceback(capsys, monkeypatch, error):
    def failing_download(*args, **kwargs):
        raise error

    # download_model takes no transport: the failure is injected where run() calls it.
    monkeypatch.setattr("kronos_agent.cli.fetch_weights.download_model", failing_download)
    assert cli.main(["fetch-weights", "--model", "small"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("ERROR:")
    assert type(error).__name__ in out and str(error) in out


def test_vendor_kronos_verify_only(capsys):
    assert cli.main(["vendor-kronos", "--verify-only"]) == 0
    assert "third_party/kronos: OK (27 files verified)" in capsys.readouterr().out


def test_env_report_json_to_stdout(capsys):
    assert cli.main(["env-report", "--json", "--label", "cli-test"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["label"] == "cli-test"
    assert report["network"] is False
    assert report["clock"] is None
    assert {"os", "cpu", "python", "torch_runtime", "manifests", "git"} <= set(report)


def test_env_report_writes_json_and_markdown(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("kronos_agent.cli.env_report.reports_dir", lambda: tmp_path)
    assert cli.main(["env-report", "--label", "unit-test"]) == 0
    out = capsys.readouterr().out
    json_path = tmp_path / "environment" / "unit-test.json"
    md_path = tmp_path / "environment" / "unit-test.md"
    assert json_path.is_file() and md_path.is_file()
    report = json.loads(json_path.read_text(encoding="ascii"))
    assert report["label"] == "unit-test"
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Environment report: unit-test\n")
    assert "## torch_runtime" in markdown
    assert "wrote" in out and "unit-test.json" in out and "unit-test.md" in out
    assert report["torch_runtime"]["torch_version"] in out


def test_env_report_arguments():
    parser, runners = cli.build_parser()
    assert "env-report" in runners
    args = parser.parse_args(["env-report"])
    assert (args.label, args.network, args.json) == (None, False, False)
    args = parser.parse_args(["env-report", "--label", "linux-x86_64", "--network", "--json"])
    assert (args.label, args.network, args.json) == ("linux-x86_64", True, True)
    with pytest.raises(SystemExit):
        parser.parse_args(["env-report", "--label", "../escape"])
    with pytest.raises(SystemExit):
        parser.parse_args(["env-report", "--label", "sub/dir"])
