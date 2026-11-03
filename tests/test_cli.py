"""The ``kronos-agent`` console script: registry, help, params, verify, gate-4 refusal, download
errors, vendor check, env-report, and the step-2 commands ``data``, ``forecast`` and
``mismatch``.

``--help`` is exercised through ``uv run kronos-agent`` in a subprocess (the real entry point);
the other commands run in-process through ``kronos_agent.cli.main`` so that the socket guard of
``conftest.py`` proves they never touch the network. No wall-clock budget is asserted:
a cheap ``--help`` is checked by what it imports, and timing is the benchmark's job.

The step-2 commands run on temporary data and report roots (``KRONOS_AGENT_DATA_DIR`` and a
patched ``reports_dir``), with the shared fakes of ``step2_fakes``: the BingX real-market and
Binance Vision handlers behind real clients, synthetic stores, and a stand-in Kronos predictor
(a seeded random walk) wherever a fan is needed, except one test with the real weights. The
expensive study runners are replaced by recording stubs; their own modules test them.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes

from kronos_agent import cli
from kronos_agent.cli import data as data_cli
from kronos_agent.cli import forecast as forecast_cli
from kronos_agent.cli import mismatch as mismatch_cli
from kronos_agent.data import binance_vision as bv
from kronos_agent.data import coverage
from kronos_agent.data import mismatch as mm
from kronos_agent.data.forecast_store import config_id, fan_relpath, load_fan
from kronos_agent.data.fsutil import exclusive_lock
from kronos_agent.data.resample import resample_exact
from kronos_agent.data.store import StoreId, read_meta, read_study_frame
from kronos_agent.data.timeutil import HOUR_MS, MINUTE_MS, from_ms, hour_key, to_ms, utc_text
from kronos_agent.data.windows import (
    StoreWindowSource,
    build_window,
    inputs_sha256,
    load_window,
    save_window,
)
from kronos_agent.exchange.public_market import KLINES_PATH, Candle
from kronos_agent.hashing import canonical_json
from kronos_agent.kronos import batch
from kronos_agent.kronos.batch import registered_weights_sha256, registry_config
from kronos_agent.kronos.decision import fan_p_up, forecast_window
from kronos_agent.kronos.features import FEATURE_COLUMNS
from kronos_agent.kronos.weights import LoadedModels, WeightsError
from kronos_agent.models import spec_for
from kronos_agent.params import load_params, params_version
from kronos_agent.paths import REPO_ROOT

COMMANDS = (
    "env-report",
    "verify",
    "fetch-weights",
    "bench",
    "vendor-kronos",
    "params",
    "data",
    "forecast",
    "mismatch",
    "finetune",
    "ledger",
)
STEP2_MODULES = ("data", "forecast", "mismatch")
# the command modules held to the import and sealed-guard rules below (step 2 and step 6)
RULED_MODULES = (*STEP2_MODULES, "finetune", "ledger")
# cli/__init__.py rule: command modules import only these (and the standard library) at load
LIGHT_MODULES = {
    "kronos_agent.cli",
    "kronos_agent.paths",
    "kronos_agent.params",
    "kronos_agent.models",
    "kronos_agent.data.timeutil",
    "kronos_agent.data.partition",
}
HEAVY_MODULES = ("torch", "huggingface_hub", "pandas", "numpy", "matplotlib")

PARAMS = load_params()
KRONOS = PARAMS.kronos
APPROVAL_PREFIX = "gate 3: --approved-by-owner "
D1 = datetime(2025, 3, 3, 12, tzinfo=UTC)
D1_MS = to_ms(D1)
FIRST_MS = D1_MS - 600 * HOUR_MS
BINGX = fakes.synthetic_candles(FIRST_MS, 700, seed=3)
BINANCE_FRAME = fakes.binance_frame(FIRST_MS, 700, interval="1h", seed=4)
CONFIG_ID = config_id(registry_config(KRONOS))


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


def test_command_registry_names_every_command():
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
    assert "profile" not in completed.stdout  # no volume-profile command or wording (D-042)


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


# == step 2: registry rules ========================================================================


def module_level_imports(path: Path, package: str) -> set[str]:
    """Absolute names of the modules imported by the top-level statements of ``path`` (an ``if
    TYPE_CHECKING:`` block is not executed at runtime and is skipped)."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                names.add(node.module or "")
                continue
            base = package.split(".")[: len(package.split(".")) - node.level + 1]
            names.add(".".join(base + ([node.module] if node.module else [])))
    return names


@pytest.mark.parametrize("name", RULED_MODULES)
def test_step_two_command_modules_import_only_light_modules_at_load(name):
    path = REPO_ROOT / "src" / "kronos_agent" / "cli" / f"{name}.py"
    imports = module_level_imports(path, "kronos_agent.cli")
    stdlib = {item for item in imports if item.split(".")[0] in sys.stdlib_module_names}
    assert imports - stdlib <= LIGHT_MODULES, sorted(imports - stdlib - LIGHT_MODULES)
    assert "__future__" in imports


def test_the_import_rule_check_sees_a_heavy_import(tmp_path):
    module = tmp_path / "bad.py"
    module.write_text(
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from ..data import store\n"
        "from . import utc_hour\n"
        "if True:\n    import pandas\n",
        encoding="utf-8",
    )
    assert module_level_imports(module, "kronos_agent.cli") == {
        "__future__",
        "numpy",
        "kronos_agent.data",
        "kronos_agent.cli",
    }


def referenced_names(path: Path) -> set[str]:
    """Every name, attribute and imported alias the syntax tree of ``path`` mentions."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name)
    return names


@pytest.mark.parametrize("name", RULED_MODULES)
def test_step_two_commands_read_candles_only_through_the_sealed_guard(name):
    """docs/DATA.md: studies and window sources read candles only through ``read_study_frame``;
    ``read_store`` (integrity work) never feeds a study or a fan."""
    names = referenced_names(REPO_ROOT / "src" / "kronos_agent" / "cli" / f"{name}.py")
    assert "read_store" not in names


STEP2_HELP = (
    ["data", "--help"],
    ["data", "binance", "--help"],
    ["data", "bingx", "--help"],
    ["data", "status", "--help"],
    ["data", "coverage", "--help"],
    ["forecast", "--help"],
    ["mismatch", "--help"],
    ["finetune", "--help"],
    ["finetune", "timing", "--help"],
    ["finetune", "fold", "--help"],
    ["finetune", "pilot-check", "--help"],
    ["finetune", "tokenizer-diag", "--help"],
)


def test_step_two_help_runs_in_a_fresh_interpreter_without_heavy_imports():
    code = (
        "import json, sys\n"
        "from kronos_agent import cli\n"
        f"codes = []\nfor argv in {[list(argv) for argv in STEP2_HELP]!r}:\n"
        "    try:\n        cli.main(argv)\n    except SystemExit as exc:\n"
        "        codes.append(exc.code)\n"
        f"heavy = sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules)\n"
        "print('RESULT ' + json.dumps([codes, heavy]))\n"
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
    result = completed.stdout.splitlines()[-1]
    assert result.startswith("RESULT "), completed.stdout[-500:]
    codes, heavy = json.loads(result.removeprefix("RESULT "))
    assert codes == [0] * len(STEP2_HELP)
    assert heavy == []
    assert "--approved-by-owner" in completed.stdout and "--record-config" in completed.stdout
    assert "--estimate" in completed.stdout and "reports/finetune/timing" in completed.stdout


# == step 2: argument validation ===================================================================


def parse(*argv: str):
    parser, _ = cli.build_parser()
    return parser.parse_args(list(argv))


@pytest.mark.parametrize(
    "argv",
    [
        ["forecast", "--source", "bingx", "--hour", "2025-03-03T12:30Z"],  # not an exact hour
        ["forecast", "--source", "bingx", "--hour", "2025-03-03T12:00"],  # no UTC marker
        ["forecast", "--source", "bingx", "--hour", "2025-02-30T12Z"],  # no such day
        ["forecast", "--source", "kraken", "--hour", "2025-03-03T12Z"],
        ["forecast", "--source", "bingx"],  # neither --hour nor --from
        ["forecast", "--hour", "2025-03-03T12Z"],  # no --source
        ["forecast", "--source", "bingx", "--hour", "2025-03-03T12Z", "--from", "2025-03-03T13Z"],
        ["forecast", "--source", "bingx", "--from", "2025-03-03T12Z", "--workers", "0"],
        ["forecast", "--source", "bingx", "--from", "2025-03-03T12Z", "--step-hours", "-1"],
        ["forecast", "--source", "bingx", "--from", "2025-03-03T12Z", "--step-hours", "1.5"],
        ["data"],
        ["data", "bingx"],  # --interval is required
        ["data", "bingx", "--interval", "4h"],
        ["data", "bingx", "--interval", "1h", "--max-chunks", "0"],
        ["data", "binance", "--interval", "5m"],  # 5m and 15m are resampled, never downloaded
        ["data", "binance", "--refresh", "2024-13"],
        ["data", "binance", "--refresh", "2024-02-30"],
        ["data", "binance", "--refresh", "202402"],
        ["mismatch", "--workers", "zero"],
        ["profile", "poc", "--source", "bingx"],  # the volume-profile command left (D-042)
        # gate 3: the approval names the CPU-hours the owner approved, a finite number > 0
        ["forecast", "--source", "bingx", "--from", "2025-03-03T12Z", "--approved-by-owner"],
        ["mismatch", "--kronos", "--approved-by-owner"],
        ["mismatch", "--kronos", "--approved-by-owner", "0"],
        ["mismatch", "--kronos", "--approved-by-owner", "-1.5"],
        ["mismatch", "--kronos", "--approved-by-owner", "nan"],
        ["mismatch", "--kronos", "--approved-by-owner", "inf"],
        ["mismatch", "--kronos", "--approved-by-owner", "yes"],
        ["finetune"],  # a subcommand is required
        # the fold command (step 6): --fold is required; a replicate counts from 1; the approval
        # names the hours the owner approved, a finite number > 0
        ["finetune", "fold"],
        ["finetune", "fold", "--fold", "q-2024-07", "--replicate", "0"],
        ["finetune", "fold", "--fold", "q-2024-07", "--approved-by-owner"],
        ["finetune", "fold", "--fold", "q-2024-07", "--approved-by-owner", "0"],
        ["finetune", "fold", "--fold", "q-2024-07", "--approved-by-owner", "nan"],
        # --record-config runs nothing: it goes alone, in either order
        ["finetune", "pilot-check", "--record-config", "--estimate"],
        ["finetune", "pilot-check", "--approved-by-owner", "1", "--record-config"],
        ["finetune", "pilot-check", "--approved-by-owner", "-1"],
        ["finetune", "tokenizer-diag"],  # --part is required
        ["finetune", "tokenizer-diag", "--part", "test"],
        # the measurement step's approval is the CPU-hours the owner approved, as above (D-055)
        ["finetune", "timing", "--approved-by-owner"],
        ["finetune", "timing", "--approved-by-owner", "0"],
        ["finetune", "timing", "--approved-by-owner", "-1.5"],
        ["finetune", "timing", "--approved-by-owner", "nan"],
        ["finetune", "timing", "--approved-by-owner", "inf"],
        ["finetune", "timing", "--approved-by-owner", "yes"],
    ],
)
def test_invalid_arguments_are_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse(*argv)
    assert excinfo.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_valid_arguments_parse_to_typed_values():
    args = parse("forecast", "--source", "bingx", "--hour", "2025-03-03T12Z")
    assert args.hour == D1 and args.hour.tzinfo is UTC
    assert (args.start, args.end, args.step_hours, args.workers) == (None, None, None, None)
    assert (args.estimate, args.approved_by_owner, args.save) == (False, None, False)
    batch_argv = "forecast --source binance --from 2025-03-03T12:00Z --to 2025-03-04T00:00:00Z"
    args = parse(*batch_argv.split(), "--step-hours", "3", "--workers", "2", "--estimate")
    assert (args.start, args.end) == (D1, D1 + timedelta(hours=12))
    assert (args.step_hours, args.workers, args.estimate) == (3, 2, True)
    assert parse(*batch_argv.split(), "--approved-by-owner", "1.53").approved_by_owner == 1.53
    args = parse("data", "binance", "--refresh", "2024-02", "--refresh", "2024-02-29")
    assert args.refresh == ["2024-02", "2024-02-29"]
    assert (args.interval, args.record_config) == (None, False)
    args = parse("data", "bingx", "--interval", "1m", "--interval", "5m", "--max-chunks", "4")
    assert (args.interval, args.max_chunks) == (["1m", "5m"], 4)
    args = parse(
        "mismatch", "--kronos", "--fallback", "--workers", "4", "--approved-by-owner", "0.96"
    )
    assert (args.kronos, args.fallback, args.workers) == (True, True, 4)
    assert args.approved_by_owner == 0.96
    assert parse("mismatch").approved_by_owner is None
    args = parse("finetune", "timing")
    assert (args.command, args.finetune_command, args.estimate) == ("finetune", "timing", False)
    assert args.approved_by_owner is None
    assert parse("finetune", "timing", "--estimate").estimate is True
    args = parse("finetune", "timing", "--approved-by-owner", "1.5")
    assert (args.approved_by_owner, args.estimate) == (1.5, False)
    args = parse("finetune", "timing", "--estimate", "--approved-by-owner", "1.5")
    assert (args.approved_by_owner, args.estimate) == (1.5, True)
    args = parse("finetune", "fold", "--fold", "q-2024-07")
    assert (args.finetune_command, args.fold, args.replicate, args.order) == (
        "fold",
        "q-2024-07",
        None,
        None,
    )
    assert (args.estimate, args.approved_by_owner) == (False, None)
    argv = ("--fold", "s-2026-09", "--replicate", "2", "--order", "order-0001")
    args = parse("finetune", "fold", *argv, "--estimate", "--approved-by-owner", "10")
    assert (args.replicate, args.order, args.estimate, args.approved_by_owner) == (
        2,
        "order-0001",
        True,
        10.0,
    )
    args = parse("finetune", "pilot-check")
    assert (args.record_config, args.estimate, args.approved_by_owner) == (False, False, None)
    assert parse("finetune", "pilot-check", "--record-config").record_config is True
    args = parse("finetune", "pilot-check", "--estimate", "--approved-by-owner", "0.9")
    assert (args.record_config, args.estimate, args.approved_by_owner) == (False, True, 0.9)
    args = parse("finetune", "tokenizer-diag", "--part", "validation", "--record-config")
    assert (args.finetune_command, args.part, args.record_config) == (
        "tokenizer-diag",
        "validation",
        True,
    )
    assert parse("finetune", "tokenizer-diag", "--part", "training").record_config is False


def test_the_literal_choices_agree_with_the_modules():
    """argparse choices are literals so that ``--help`` imports nothing heavy; they must name
    exactly what the modules behind them accept."""
    from kronos_agent.data import bingx_history
    from kronos_agent.data import store as store_module

    assert set(data_cli.BINGX_INTERVALS) == set(bingx_history.INTERVALS)
    assert data_cli.BINANCE_INTERVALS == ("1m", "1h")  # 5m and 15m are resampled from 1m
    assert set(data_cli.BINANCE_INTERVALS) <= set(bv.INTERVALS)
    assert data_cli.STORE_INTERVALS == store_module.INTERVALS
    assert mm.SOURCES == forecast_cli.SOURCES


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--hour", "2025-03-03T12Z", "--to", "2025-03-03T15Z"], "--to belong to a batch"),
        (["--hour", "2025-03-03T12Z", "--workers", "2"], "--workers belong to a batch"),
        (["--hour", "2025-03-03T12Z", "--step-hours", "2"], "--step-hours belong"),
        (["--hour", "2025-03-03T12Z", "--estimate"], "--estimate belong"),
        (["--hour", "2025-03-03T12Z", "--approved-by-owner", "1.5"], "--approved-by-owner belong"),
        (["--from", "2025-03-03T12Z"], "needs --to"),
        (["--from", "2025-03-03T12Z", "--to", "2025-03-03T12Z"], "later than --from"),
        (["--from", "2025-03-03T12Z", "--to", "2025-03-03T11Z"], "later than --from"),
        (["--from", "2025-03-03T12Z", "--to", "2025-03-03T15Z", "--save"], "--save belongs"),
    ],
)
def test_forecast_mode_conflicts_are_usage_errors_before_anything_runs(
    argv, message, capsys, monkeypatch
):
    monkeypatch.setattr(forecast_cli, "load_params", fail_if_called("load_params"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["forecast", "--source", "bingx", *argv])
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--estimate"], "--estimate need --kronos"),
        (["--workers", "2"], "--workers need --kronos"),
        (["--approved-by-owner", "1.5"], "--approved-by-owner need --kronos"),
        (["--fallback"], "--fallback need --kronos"),
        (["--record-config", "--kronos"], "give it alone"),
        (["--record-config", "--estimate"], "give it alone"),
    ],
)
def test_mismatch_option_conflicts_are_usage_errors(argv, message, capsys, monkeypatch):
    monkeypatch.setattr(mm, "study_config", fail_if_called("study_config"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["mismatch", *argv])
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


# == step 2: shared fixtures =======================================================================


def fail_if_called(name: str):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be called")

    return fail


class Roots:
    """The temporary data root and report directory of one test."""

    def __init__(self, tmp_path: Path) -> None:
        self.data = tmp_path / "data"
        self.reports = tmp_path / "reports" / "data"


@pytest.fixture
def roots(tmp_path, monkeypatch) -> Roots:
    """Data under ``tmp/data`` (``KRONOS_AGENT_DATA_DIR``), reports under ``tmp/reports/data``."""
    paths = Roots(tmp_path)
    monkeypatch.setenv("KRONOS_AGENT_DATA_DIR", str(paths.data))
    monkeypatch.setattr(cli, "reports_dir", lambda: tmp_path / "reports")
    return paths


def test_the_roots_fixture_redirects_both_roots(roots):
    assert cli.data_root() == roots.data.resolve()
    assert cli.report_directory() == roots.reports


def write_stores(root: Path) -> None:
    fakes.write_bingx_store(root, BINGX)
    fakes.write_binance_store(root, BINANCE_FRAME)


class WalkPredictor:
    """``predict_batch`` stand-in: a seeded random walk from the last close of each context."""

    def __init__(self) -> None:
        self.calls = 0

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, **_):
        self.calls += 1
        frames = []
        for df, y in zip(df_list, y_timestamp_list, strict=True):
            last = float(df["close"].iloc[-1])
            close = last * np.exp(np.cumsum(np.random.normal(0.0, 0.01, size=pred_len)))
            values = np.column_stack(
                [close, close * 1.001, close * 0.999, close, np.ones(pred_len), close]
            )
            frames.append(pd.DataFrame(values, columns=list(FEATURE_COLUMNS), index=y))
        return frames


def fake_models() -> LoadedModels:
    """Stand-in models with the registered weight hashes: the batch configuration is the real
    one, so stored fans land in the real config directory."""
    return LoadedModels(
        spec=spec_for("small"),
        tokenizer=None,
        model=None,
        predictor=WalkPredictor(),
        weights_sha256=registered_weights_sha256("small"),
        load_seconds=0.0,
    )


@pytest.fixture
def models(monkeypatch) -> list[LoadedModels]:
    """``_load_models`` returns stand-in models; the list holds every one handed out."""
    loaded: list[LoadedModels] = []

    def load(params):
        assert params.kronos.model == "small"
        loaded.append(fake_models())
        return loaded[-1]

    monkeypatch.setattr(forecast_cli, "_load_models", load)
    return loaded


def fans(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in (root / "forecasts").rglob("*.npz"))


# == step 2: data ==================================================================================


def binance_client(fake: fakes.FakeBinanceVision):
    clock = fakes.FakeClock()
    return bv.BinanceVisionClient(
        transport=httpx.MockTransport(fake), min_interval_s=0.0, sleep=clock.sleep, clock=clock
    )


def binance_fake() -> tuple[fakes.FakeBinanceVision, pd.DataFrame]:
    """1m dailies 2019-09-29..2019-10-01 and, as Binance's own 1h archives, their exact
    resample: the check must find every hour equal."""
    fake = fakes.FakeBinanceVision()
    frames = []
    for offset, day in enumerate(("2019-09-29", "2019-09-30", "2019-10-01")):
        minutes = fakes.period_frame("daily", day, seed=offset)
        fake.add_archive("daily", "1m", day, minutes, header=offset != 1)
        fake.add_archive("daily", "1h", day, resample_exact(minutes, "1h"))
        frames.append(minutes)
    return fake, pd.concat(frames, ignore_index=True)


@pytest.fixture
def binance_world(roots, monkeypatch):
    fake, minutes = binance_fake()
    monkeypatch.setattr(data_cli, "_binance_client", lambda: binance_client(fake))
    monkeypatch.setattr(data_cli, "_utc_today", lambda: date(2019, 10, 2))
    return fake, minutes


def test_data_binance_downloads_builds_and_checks(roots, binance_world, capsys):
    fake, minutes = binance_world
    assert cli.main(["data", "binance"]) == 0
    out = capsys.readouterr().out
    assert "binance 1m: 3 archives planned (0 present, 3 downloaded" in out
    assert "binance 1h: 3 archives planned (0 present, 3 downloaded" in out
    assert "resample check: 72 hours compared; 0 OHLC, volume or count mismatches; 0 " in out
    assert "ATTENTION" not in out
    stored = read_study_frame(
        StoreId("binance", "1m"), start_ms=0, end_ms=to_ms(D1), numeric=False, root=roots.data
    )
    assert stored.equals(minutes)
    assert read_meta(StoreId("binance", "1h"), roots.data)["rows"] == 72
    assert read_meta(StoreId("binance_official", "1h"), roots.data)["rows"] == 72
    report = json.loads((roots.reports / "binance_resample_check.json").read_text())
    assert report["comparison"]["hours_compared"] == 72 and report["owner_attention"] is False
    config = roots.reports / "binance_resample_check.config.json"
    assert config.read_bytes() == canonical_json(bv.resample_check_config())
    assert (roots.reports / "binance_resample_check.md").is_file()
    zips = [path for path in fake.requested_paths() if path.endswith(".zip")]
    assert len(zips) == 6

    # a second run downloads nothing and reproduces the stores
    sha_1m = read_meta(StoreId("binance", "1m"), roots.data)["csv_sha256"]
    assert cli.main(["data", "binance", "--interval", "1m"]) == 0
    out = capsys.readouterr().out
    assert "binance 1m: 3 archives planned (3 present, 0 downloaded" in out
    assert "binance 1h:" not in out
    assert len([path for path in fake.requested_paths() if path.endswith(".zip")]) == 6
    assert read_meta(StoreId("binance", "1m"), roots.data)["csv_sha256"] == sha_1m
    assert "resample check: 72 hours compared" in out  # both 1h stores exist


def test_data_binance_only_1h_skips_the_check_until_the_1m_stores_exist(
    roots, binance_world, capsys
):
    assert cli.main(["data", "binance", "--interval", "1h"]) == 0
    out = capsys.readouterr().out
    assert "resample check not run: store candles/binance/1h.csv has not been built" in out
    assert not (roots.reports / "binance_resample_check.json").exists()
    assert read_meta(StoreId("binance_official", "1h"), roots.data)["rows"] == 72


def test_data_status_lists_the_binance_stores_and_the_saved_plans(
    roots, binance_world, monkeypatch, capsys
):
    assert cli.main(["data", "binance"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(data_cli, "_now_ms", lambda: fakes.utc_ms(2019, 10, 2))
    assert cli.main(["data", "status"]) == 0
    out = capsys.readouterr().out
    for interval in ("1m", "1h"):
        plan = bv.load_plan(interval, roots.data)
        assert (
            f"raw/binance/plan-{interval}.json: {len(plan['archives'])} archives as of "
            f"2019-10-02; {len(plan['monthly_missing'])} months and "
            f"{len(plan['daily_missing'])} days not published"
        ) in out
        assert len(plan["archives"]) == 3
    for source, interval, rows in (
        ("binance", "1m", 3 * 1440),
        ("binance", "5m", 3 * 288),
        ("binance", "15m", 3 * 96),
        ("binance", "1h", 72),
        ("binance_official", "1h", 72),
    ):
        assert f"candles/{source}/{interval}.csv: {rows} rows, 2019-09-29T00:00Z .. " in out
    assert "PROBLEM" not in out


def test_data_binance_gate_4_refuses_before_any_download(roots, binance_world, monkeypatch, capsys):
    fake, _ = binance_world
    monkeypatch.setattr(bv, "BUDGET_BYTES", 1000)
    assert cli.main(["data", "binance"]) == 2
    out = capsys.readouterr().out
    assert "REFUSED: gate 4" in out
    assert not [path for path in fake.requested_paths() if path.endswith(".zip")]
    assert not list((roots.data / "raw" / "binance").rglob("*.zip"))
    assert not (roots.data / "candles").exists()


def add_day(fake: fakes.FakeBinanceVision, day: str) -> None:
    minutes = fakes.period_frame("daily", day, seed=7)
    fake.add_archive("daily", "1m", day, minutes)
    fake.add_archive("daily", "1h", day, resample_exact(minutes, "1h"))


def test_data_binance_gate_4_is_checked_once_for_the_whole_run(
    roots, binance_world, monkeypatch, capsys
):
    """1m is synced but for one new day and 1h never was: the 1m estimate fits the budget alone,
    1m plus 1h does not, so the run is refused before any download (not after the 1m ones)."""
    fake, _ = binance_world
    assert cli.main(["data", "binance", "--interval", "1m"]) == 0
    add_day(fake, "2019-10-02")
    monkeypatch.setattr(data_cli, "_utc_today", lambda: date(2019, 10, 3))
    probe = roots.data.parent / "probe"
    shutil.copytree(roots.data, probe)
    planned = bv.sync_intervals(
        binance_client(fake), ["1m", "1h"], today=date(2019, 10, 3), root=probe
    )
    disk, run_estimate = planned[0]["disk_bytes_before"], planned[0]["run_estimated_download_bytes"]
    estimate_1m, estimate_1h = (item["estimated_download_bytes"] for item in planned)
    assert (estimate_1m, estimate_1h) == (1440 * 80, 4 * 24 * 80)
    assert run_estimate == estimate_1m + estimate_1h
    capsys.readouterr()
    fake.requests.clear()
    zips_before = sorted(roots.data.rglob("*.zip"))
    monkeypatch.setattr(bv, "BUDGET_BYTES", disk + estimate_1m)  # 1m alone fits, so does 1h
    assert cli.main(["data", "binance"]) == 2
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("REFUSED: gate 4: ")
    assert f"+ an estimated {run_estimate} bytes for this run (1m, 1h)" in out
    assert "nothing downloaded" in out
    assert not [path for path in fake.requested_paths() if path.endswith(".zip")]
    assert sorted(roots.data.rglob("*.zip")) == zips_before
    lines = out.splitlines()
    plans = [i for i, line in enumerate(lines) if line.startswith("plan ")]
    assert [lines[i].split(":")[0] for i in plans] == ["plan 1m", "plan 1h"]
    assert "gate 4: estimated download 0.1 MB for 1m, 1h" in lines[plans[-1] + 1]

    monkeypatch.setattr(bv, "BUDGET_BYTES", disk + run_estimate)
    assert cli.main(["data", "binance"]) == 0
    lines = capsys.readouterr().out.splitlines()
    first_download = next(i for i, line in enumerate(lines) if line.startswith("downloaded "))
    gate = next(i for i, line in enumerate(lines) if line.startswith("gate 4: estimated"))
    assert max(i for i, line in enumerate(lines) if line.startswith("plan ")) < gate
    assert gate < first_download
    assert len([path for path in fake.requested_paths() if path.endswith(".zip")]) == 5


def test_data_binance_record_config_only_and_changed_config(roots, monkeypatch, capsys):
    monkeypatch.setattr(data_cli, "_binance_client", fail_if_called("the Binance client"))
    assert cli.main(["data", "binance", "--record-config"]) == 0
    out = capsys.readouterr().out
    config = roots.reports / "binance_resample_check.config.json"
    assert config.read_bytes() == canonical_json(bv.resample_check_config())
    assert out.startswith("recorded ") and "binance_resample_check.config.json" in out
    config.write_bytes(canonical_json({"study": "binance_resample_check", "changed": True}))
    assert cli.main(["data", "binance"]) == 2  # refused before the client exists
    assert capsys.readouterr().out.startswith("REFUSED: study binance_resample_check")


def test_data_binance_unreachable_host_is_a_failure(roots, monkeypatch, capsys):
    def unavailable(request):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(
        data_cli,
        "_binance_client",
        lambda: bv.BinanceVisionClient(
            transport=httpx.MockTransport(unavailable),
            min_interval_s=0.0,
            max_retries=0,
            sleep=lambda seconds: None,
        ),
    )
    monkeypatch.setattr(data_cli, "_utc_today", lambda: date(2019, 10, 2))
    assert cli.main(["data", "binance"]) == 1
    assert capsys.readouterr().out.splitlines()[-1].startswith("ERROR: BinanceVisionUnavailable: ")


def bingx_world(server_ms: int, count: int = 1500) -> tuple[fakes.FakeBingX, list]:
    start = server_ms - count * HOUR_MS - 2 * MINUTE_MS
    start -= start % HOUR_MS
    candles = fakes.synthetic_candles(start, count + 1, seed=9)  # the last one is unclosed
    fake = fakes.FakeBingX(server_ms, {"1h": candles}, history_start_ms={"1h": start})
    return fake, candles


def test_data_bingx_backfills_merges_and_resumes(roots, monkeypatch, capsys):
    server_ms = fakes.utc_ms(2025, 6, 1, 7) + 2 * MINUTE_MS  # 7:02, the 6:00 candle settled
    fake, candles = bingx_world(server_ms)
    clients = []

    def client():
        clients.append(fakes.bingx_client(fake)[0])
        return clients[-1]

    monkeypatch.setattr(data_cli, "_bingx_client", client)
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 0
    out = capsys.readouterr().out
    settled = [c for c in candles if c.open_ms + HOUR_MS + 90_000 <= server_ms]
    assert f"history start {utc_text(candles[0].open_ms)}" in out and "stop: empty_chunks" in out
    meta = read_meta(StoreId("bingx", "1h"), roots.data)
    assert meta["rows"] == len(settled) == 1500
    assert meta["last_open_ms"] == to_ms(datetime(2025, 6, 1, 6, tzinfo=UTC))
    assert f"candles/bingx/1h.csv: 1500 rows, {utc_text(settled[0].open_ms)}" in out
    first_requests = len(fake.klines_requests())

    fake.advance(HOUR_MS)  # one more settled candle; complete chunks are not fetched again
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 0
    out = capsys.readouterr().out
    assert "1 new candles" in out and "stop: empty_chunks" in out
    skipped = int(re.search(r"(\d+) complete chunks skipped", out).group(1))
    assert skipped >= 1
    # the newest (incomplete) chunk and the three empty chunks that end the walk, twice each
    assert len(fake.klines_requests()) - first_requests == 2 * (1 + 3)
    assert read_meta(StoreId("bingx", "1h"), roots.data)["rows"] == 1501


def test_data_bingx_refuses_a_concurrent_run(roots, monkeypatch, capsys):
    fake, _ = bingx_world(fakes.utc_ms(2025, 6, 1, 7) + 2 * MINUTE_MS)
    monkeypatch.setattr(data_cli, "_bingx_client", lambda: fakes.bingx_client(fake)[0])
    with exclusive_lock(roots.data / "raw" / "bingx" / ".lock"):
        assert cli.main(["data", "bingx", "--interval", "1h"]) == 2
    assert "REFUSED: bingx/.lock is locked by another run" in capsys.readouterr().out
    assert fake.requests == []


def test_data_bingx_exchange_errors_are_failures(roots, monkeypatch, capsys):
    fake, _ = bingx_world(fakes.utc_ms(2025, 6, 1, 7) + 2 * MINUTE_MS)
    fake.queue(KLINES_PATH, fakes.api_error(109400, "invalid"))
    monkeypatch.setattr(data_cli, "_bingx_client", lambda: fakes.bingx_client(fake)[0])
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 1
    assert "ERROR: BingXApiError: " in capsys.readouterr().out
    assert not (roots.data / "candles").exists()


def test_data_bingx_reports_removed_temporaries_and_candles_kept_from_the_archive(
    roots, monkeypatch, capsys
):
    server_ms = fakes.utc_ms(2025, 6, 1, 7) + 2 * MINUTE_MS
    fake, candles = bingx_world(server_ms)
    monkeypatch.setattr(data_cli, "_bingx_client", lambda: fakes.bingx_client(fake)[0])
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 0
    out = capsys.readouterr().out
    assert "removed" not in out and "kept from the archive" not in out
    # the merged store's meta line says whether its first candle is the established start
    assert f"; history start {utc_text(candles[0].open_ms)} (established)" in out
    leftover = roots.data / "raw" / "bingx" / "1h" / ".0.csv.k1ll3d00.tmp"
    leftover.write_bytes(b"x" * 10)
    newest = to_ms(datetime(2025, 6, 1, 6, tzinfo=UTC))
    fake.missing["1h"] = {newest - HOUR_MS}  # archived, no longer served inside the range
    fake.advance(HOUR_MS)
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 0
    out = capsys.readouterr().out
    assert "removed 1 temporary files of killed runs (0.0 MB)" in out
    assert not leftover.exists()
    assert (
        "1 archived candles the exchange no longer returns were kept from the archive (sidecar "
        "archive_only)"
    ) in out
    assert (
        "WARNING: 1 of them lie inside the range the exchange still serves (not retention): "
        "their chunk stays incomplete and the next run fetches it again; report them to the owner"
    ) in out
    assert read_meta(StoreId("bingx", "1h"), roots.data)["rows"] == 1501


def test_data_bingx_a_failed_interval_is_still_merged_and_the_others_run(
    roots, monkeypatch, capsys
):
    server_ms = fakes.utc_ms(2025, 6, 1, 7) + 2 * MINUTE_MS
    fake, _candles = bingx_world(server_ms)
    start_15m = to_ms(datetime(2025, 5, 28, tzinfo=UTC))
    fake.set_candles("15m", fakes.synthetic_candles(start_15m, 401, interval="15m", seed=11))
    fake.history_start_ms["15m"] = start_15m
    monkeypatch.setattr(data_cli, "_bingx_client", lambda: fakes.bingx_client(fake)[0])
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 0
    capsys.readouterr()
    unstable = to_ms(datetime(2025, 6, 1, 5, tzinfo=UTC))  # archived in the incomplete chunk
    fake.mutator = fakes.for_interval("1h", fakes.vary_each_call(unstable))  # no pair matches
    fake.advance(HOUR_MS)
    assert cli.main(["data", "bingx", "--interval", "1h", "--interval", "15m"]) == 1
    out = capsys.readouterr().out
    lines = out.splitlines()
    error = next(i for i, line in enumerate(lines) if line.startswith("ERROR: BackfillError: "))
    assert "all 3 fetch pairs differed" in lines[error]
    assert lines[error + 1] == "bingx 1h: rebuilding the store from the chunks archived so far"
    assert lines[error + 2].startswith("candles/bingx/1h.csv: 1500 rows")
    assert "bingx 15m: server time" in out  # the next interval still ran
    assert read_meta(StoreId("bingx", "15m"), roots.data)["rows"] > 0
    assert lines[-1] == "bingx: 1 of 2 intervals failed: 1h"


def test_data_bingx_reports_a_revision_as_a_warning_and_status_lists_it(roots, monkeypatch, capsys):
    """The owner's decision on revised candles (D-045): the archive and the store keep the
    first-seen row, the revision is recorded once, the run goes on and exits 0 unless something
    else failed, and the status lists every recorded revision."""
    server_ms = fakes.utc_ms(2025, 6, 1, 7) + 2 * MINUTE_MS
    fake, _candles = bingx_world(server_ms)
    monkeypatch.setattr(data_cli, "_bingx_client", lambda: fakes.bingx_client(fake)[0])
    monkeypatch.setattr(data_cli, "_now_ms", lambda: fake.server_time_ms)
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 0
    assert cli.main(["data", "status"]) == 0
    assert "revision" not in capsys.readouterr().out
    revised = to_ms(datetime(2025, 6, 1, 5, tzinfo=UTC))  # archived in the newest chunk
    original = fake.candle("1h", revised)
    fake.put("1h", replace(original, volume="1.0000"))
    fake.advance(HOUR_MS)
    first_seen = fake.server_time_ms
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 0
    out = capsys.readouterr().out
    assert "ERROR" not in out and "1 new candles; stop: empty_chunks" in out
    chunk = to_ms(datetime(2025, 4, 30, 8, tzinfo=UTC))  # the newest 1h chunk, 485
    warning = (
        f"WARNING: bingx 1h chunk {utc_text(chunk)}: the exchange served 1 archived candle(s) "
        f"with other values ({{new}} of them not recorded before): {utc_text(revised)}; the "
        "archive and the store keep the first-seen rows, and raw/bingx/revision-1h.json records "
        "every revision; report them to the owner"
    )
    assert warning.format(new=1) in out.splitlines()
    stored = read_study_frame(
        StoreId("bingx", "1h"),
        start_ms=revised,
        end_ms=revised + HOUR_MS,
        numeric=False,
        root=roots.data,
    )
    assert fakes.frame_to_candles(stored) == [original]  # the first-seen row
    listed = (
        "WARNING: bingx 1h: 1 revision(s) recorded in raw/bingx/revision-1h.json (archived "
        "candles the exchange later served with other values; the archive and the store keep "
        f"the first-seen rows): {utc_text(revised)} (seen {{seen}}); report them to the owner"
    )
    assert cli.main(["data", "status"]) == 0
    out = capsys.readouterr().out
    assert listed.format(seen=utc_text(first_seen)) in out.splitlines()
    assert "PROBLEM" not in out
    # the next run sees it again (recorded once) while another interval fails: exit 1 for that
    start_15m = to_ms(datetime(2025, 5, 28, tzinfo=UTC))
    fake.set_candles("15m", fakes.synthetic_candles(start_15m, 401, interval="15m", seed=11))
    fake.history_start_ms["15m"] = start_15m
    newest_15m = start_15m + 400 * 15 * MINUTE_MS
    fake.mutator = fakes.for_interval("15m", fakes.vary_each_call(newest_15m))
    fake.advance(HOUR_MS)
    assert cli.main(["data", "bingx", "--interval", "1h", "--interval", "15m"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert warning.format(new=0) in lines
    assert any(line.startswith("ERROR: BackfillError: 15m chunk ") for line in lines)
    assert lines[-1] == "bingx: 1 of 2 intervals failed: 15m"
    assert cli.main(["data", "status"]) == 0
    seen = f"{utc_text(first_seen)} to {utc_text(fake.server_time_ms)}"
    assert listed.format(seen=seen) in capsys.readouterr().out.splitlines()


def test_data_bingx_reports_a_revision_when_a_later_chunk_of_its_walk_fails(
    roots, monkeypatch, capsys
):
    """A revision is printed as soon as it is recorded: an interval whose walk then fails (a
    rate limit, transport error or fetch pairs that never match in an older chunk) still names
    it on stdout, not only in the record and in `data status` (D-045)."""
    older = to_ms(datetime(2025, 4, 30, 8, tzinfo=UTC))  # 1h chunk 485
    newer = to_ms(datetime(2025, 6, 11, tzinfo=UTC))  # 1h chunk 486
    fake, _candles = bingx_world(newer + 4 * HOUR_MS + 2 * MINUTE_MS, count=1800)
    monkeypatch.setattr(data_cli, "_bingx_client", lambda: fakes.bingx_client(fake)[0])
    fake.server_time_ms = fakes.utc_ms(2025, 6, 1, 7) + 2 * MINUTE_MS
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 0
    # a bounded run archives the first candles of chunk 486 and leaves 485 incomplete behind it
    fake.server_time_ms = newer + 3 * HOUR_MS + 2 * MINUTE_MS
    assert cli.main(["data", "bingx", "--interval", "1h", "--max-chunks", "1"]) == 0
    capsys.readouterr()
    revised = newer + HOUR_MS  # archived by the bounded run
    original = fake.candle("1h", revised)
    fake.put("1h", replace(original, volume="1.0000"))
    unstable = to_ms(datetime(2025, 6, 1, 5, tzinfo=UTC))  # archived in chunk 485
    fake.mutator = fakes.for_interval("1h", fakes.vary_each_call(unstable))  # no pair matches
    fake.advance(HOUR_MS)
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 1
    lines = capsys.readouterr().out.splitlines()
    warning = (
        f"WARNING: bingx 1h chunk {utc_text(newer)}: the exchange served 1 archived candle(s) "
        f"with other values (1 of them not recorded before): {utc_text(revised)}; the archive "
        "and the store keep the first-seen rows, and raw/bingx/revision-1h.json records every "
        "revision; report them to the owner"
    )
    error = next(i for i, line in enumerate(lines) if line.startswith("ERROR: BackfillError: "))
    assert lines.index(warning) < error
    assert f"1h chunk {utc_text(older)}: all 3 fetch pairs differed" in lines[error]
    assert lines[error + 1] == "bingx 1h: rebuilding the store from the chunks archived so far"
    assert lines[-1] == "bingx: 1 of 1 intervals failed: 1h"
    stored = read_study_frame(
        StoreId("bingx", "1h"),
        start_ms=revised,
        end_ms=revised + HOUR_MS,
        numeric=False,
        root=roots.data,
    )
    assert fakes.frame_to_candles(stored) == [original]  # the first-seen row


def test_data_status_lists_stores_plans_and_warnings(roots, monkeypatch, capsys):
    monkeypatch.setattr(data_cli, "_now_ms", lambda: fakes.utc_ms(2025, 3, 5))
    assert cli.main(["data", "status"]) == 0
    out = capsys.readouterr().out
    assert "no store is built yet" in out and "now 2025-03-05T00:00Z" in out
    assert "WARNING: bingx 1m: no archived chunks" in out
    assert "WARNING: bingx 5m: no archived chunks" in out

    write_stores(roots.data)
    assert cli.main(["data", "status"]) == 0
    out = capsys.readouterr().out
    assert f"candles/bingx/1h.csv: 700 rows, {utc_text(FIRST_MS)} .. " in out
    assert "candles/binance/1h.csv: 700 rows" in out and "PROBLEM" not in out

    csv = roots.data / "candles" / "bingx" / "1h.csv"
    original = csv.read_bytes()
    csv.write_bytes(original + b"1,2,3,4,5,6\n")  # modified after its meta
    assert cli.main(["data", "status"]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM: candles/bingx/1h.csv: its size differs from its meta" in out

    # a same-size edit (one digit of a price) is caught by the sha256
    row = original.splitlines()[5]
    digit = row.rfind(b".") - 1
    edited = row[:digit] + (b"1" if row[digit : digit + 1] != b"1" else b"2") + row[digit + 1 :]
    csv.write_bytes(original.replace(row, edited, 1))
    assert len(csv.read_bytes()) == len(original)
    assert cli.main(["data", "status"]) == 1
    out = capsys.readouterr().out
    assert (
        "PROBLEM: candles/bingx/1h.csv: its sha256 differs from its meta; rebuild the store" in out
    )
    assert out.count("PROBLEM:") == 1


def test_data_status_skips_the_stores_a_run_is_writing_and_lists_temporaries(
    roots, monkeypatch, capsys
):
    write_stores(roots.data)
    monkeypatch.setattr(data_cli, "_now_ms", lambda: fakes.utc_ms(2025, 3, 5))
    csv = roots.data / "candles" / "bingx" / "1h.csv"
    csv.write_bytes(csv.read_bytes() + b"1,2,3,4,5,6\n")  # being replaced by the run, say
    killed = roots.data / "candles" / "binance" / ".1h.csv.k1ll3d00.tmp"
    killed.write_bytes(b"x" * 2_500_000)
    live = roots.data / "raw" / "bingx" / "1h" / ".0.csv.l1ve0000.tmp"
    live.parent.mkdir(parents=True)
    live.write_bytes(b"y" * 10)
    with exclusive_lock(roots.data / "raw" / "bingx" / ".lock"):
        assert cli.main(["data", "status"]) == 0  # no false PROBLEM for a busy store
    out = capsys.readouterr().out
    assert f"candles/bingx/1h.csv: 700 rows, {utc_text(FIRST_MS)} .. " in out
    assert "  a run holds raw/bingx/.lock: its CSV was not checked" in out
    assert "PROBLEM" not in out
    assert (
        "WARNING: 1 temporary files of killed runs (2.5 MB) under candles/ and raw/: "
        "candles/binance/.1h.csv.k1ll3d00.tmp; the next `kronos-agent data binance` or `data "
        "bingx` run removes them"
    ) in out
    assert (
        "NOTE: 1 temporary files under directories a run is writing now (raw/bingx/.lock): "
        "they may be its own"
    ) in out
    assert killed.exists() and live.exists()  # status removes nothing
    assert cli.main(["data", "status"]) == 1  # the run is over: the CSV is checked again
    out = capsys.readouterr().out
    assert "PROBLEM: candles/bingx/1h.csv: its size differs from its meta" in out
    assert "WARNING: 2 temporary files of killed runs" in out


def test_data_coverage_writes_the_report_and_refuses_a_changed_config(roots, monkeypatch, capsys):
    write_stores(roots.data)
    monkeypatch.setattr(data_cli, "_now_ms", lambda: fakes.utc_ms(2025, 3, 5))
    assert cli.main(["data", "coverage", "--record-config"]) == 0
    config = roots.reports / "coverage.config.json"
    assert config.read_bytes() == canonical_json(coverage.study_config())
    assert not (roots.reports / "coverage.json").exists()
    assert "recorded " in capsys.readouterr().out

    assert cli.main(["data", "coverage"]) == 0
    out = capsys.readouterr().out
    document = json.loads((roots.reports / "coverage.json").read_text())
    usable = document["usable_decision_hours"]["candles/bingx/1h.csv"]
    validation = usable["context_512h"]["segments"]["validation"]
    assert validation["usable"] == 700 - 512 + 1  # D from the 512th candle's close to the last
    assert list(usable) == ["context_512h"]  # the profile windows left with D-042
    assert (
        f"candles/bingx/1h.csv: usable validation decision hours: context_512h "
        f"{validation['usable']}/{validation['eligible']}\n" in out
    )
    assert "profile" not in out
    assert "candles/bingx/5m.csv: not built" in out
    assert "WARNING: bingx 1m: no archived chunks" in out
    assert (roots.reports / "coverage.md").is_file()

    config.write_bytes(canonical_json({"study": "coverage", "changed": True}))
    monkeypatch.setattr(coverage, "coverage_report", fail_if_called("coverage_report"))
    assert cli.main(["data", "coverage"]) == 2
    assert capsys.readouterr().out.startswith("REFUSED: study coverage")


# == step 2: forecast ==============================================================================


def expected_window(source: str, hour: datetime):
    candles = BINGX if source == "bingx" else fakes.frame_to_candles(BINANCE_FRAME)
    return build_window(candles, hour, source=source, horizon=KRONOS.horizon_n)


def test_forecast_one_hour_prints_the_signal_and_writes_nothing(roots, models, capsys):
    write_stores(roots.data)
    assert cli.main(["forecast", "--source", "bingx", "--hour", "2025-03-03T12Z"]) == 0
    out = capsys.readouterr().out
    window = expected_window("bingx", D1)
    fan = forecast_window(fake_models(), window, PARAMS)
    first = utc_text(D1_MS - 512 * HOUR_MS)
    assert out.startswith(f"bingx window of 2025-03-03T12:00Z: 512 candles from {first}; ")
    assert f"sha256 {window.sha256}; inputs sha256 {window.inputs_sha256}" in out
    assert f"seeds {fan.seeds.canonical} (torch {fan.seeds.torch_seed}" in out
    assert f"p_up {fan_p_up(fan.paths, window.last_close):.3f}" in out
    assert len(models) == 1 and models[0].predictor.calls == 1
    assert not (roots.data / "forecasts").exists() and not (roots.data / "windows").exists()


def window_file_of(root: Path, source: str, hour: datetime) -> Path:
    return root / "windows" / source / f"context-{KRONOS.max_context}" / f"{hour_key(hour)}.json"


def test_forecast_one_hour_save_stores_once_and_reuses(roots, models, monkeypatch, capsys):
    write_stores(roots.data)
    argv = ["forecast", "--source", "binance", "--hour", "2025-03-03T12Z", "--save"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    relpath = fan_relpath("binance", CONFIG_ID, "forecast", D1)
    assert fans(roots.data) == [relpath]
    window_file = window_file_of(roots.data, "binance", D1)
    assert window_file.is_file()
    stored = load_fan(roots.data / relpath)
    window = expected_window("binance", D1)
    assert stored.meta["inputs_sha256"] == window.inputs_sha256
    assert f"p_up {fan_p_up(stored.paths, window.last_close):.3f}" in out
    assert f"fan {relpath} (computed and stored)" in out
    assert f"seeds {stored.meta['seeds']['canonical']}" in out

    monkeypatch.setattr(forecast_cli, "_load_models", fail_if_called("_load_models"))
    data_before = (roots.data / relpath).read_bytes()
    assert cli.main(argv) == 0
    assert f"fan {relpath} (stored fan reused)" in capsys.readouterr().out
    assert (roots.data / relpath).read_bytes() == data_before


def test_forecast_save_survives_a_governed_amount_policy_change(roots, models, monkeypatch, capsys):
    """kronos-1: the saved window holds the inputs only, so a fan under another amount policy
    (another config directory) is saved with the same window file, which is kept as it was."""
    write_stores(roots.data)
    argv = ["forecast", "--source", "binance", "--hour", "2025-03-03T12Z", "--save"]
    assert cli.main(argv) == 0
    window_file = window_file_of(roots.data, "binance", D1)
    saved = window_file.read_bytes()
    quoted = replace(PARAMS, kronos=replace(KRONOS, amount_policy="binance_quote_volume"))
    monkeypatch.setattr(forecast_cli, "load_params", lambda: quoted)
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    quoted_id = config_id(registry_config(quoted.kronos))
    assert quoted_id != CONFIG_ID
    relpaths = [fan_relpath("binance", cid, "forecast", D1) for cid in (CONFIG_ID, quoted_id)]
    assert fans(roots.data) == sorted(relpaths)
    assert f"fan {relpaths[1]} (computed and stored)" in out
    assert window_file.read_bytes() == saved
    assert [path.name for path in (roots.data / "windows").rglob("*.json")] == [window_file.name]
    stored = load_fan(roots.data / relpaths[1])
    assert stored.meta["amount_policy"] == "binance_quote_volume"
    candles, metadata = load_window(window_file)  # the saved inputs give the fan's inputs back
    assert (
        inputs_sha256(candles, "binance_quote_volume", metadata["quote_volumes"])
        == (stored.meta["inputs_sha256"])
    )
    monkeypatch.setattr(forecast_cli, "_load_models", fail_if_called("_load_models"))
    assert cli.main(argv) == 0  # a rerun reuses the fan and keeps the window
    assert f"fan {relpaths[1]} (stored fan reused)" in capsys.readouterr().out
    assert window_file.read_bytes() == saved


def test_forecast_save_refuses_a_different_saved_window_before_any_fan(roots, monkeypatch, capsys):
    """The window is checked and saved before any model is loaded or fan stored: a different
    window already saved for the hour (another store version) leaves no fan behind."""
    write_stores(roots.data)
    other = roots.data.parent / "other"
    altered = BINANCE_FRAME.copy()
    altered.loc[200, "volume"] = "1.000"  # inside the window of D1 (FIRST_MS + 200 h)
    fakes.write_binance_store(other, altered)
    previous = StoreWindowSource("binance", root=other).window(D1, horizon=KRONOS.horizon_n)
    save_window(previous, root=roots.data)
    monkeypatch.setattr(forecast_cli, "_load_models", fail_if_called("_load_models"))
    argv = ["forecast", "--source", "binance", "--hour", "2025-03-03T12Z", "--save"]
    assert cli.main(argv) == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith(
        "ERROR: WindowError: a different window is already saved for binance 2025-03-03T12Z"
    )
    assert fans(roots.data) == []


@pytest.mark.parametrize(
    "hour, code, start",
    [
        ("2026-08-01T00Z", 2, "REFUSED: decision hour 2026-08-01T00:00Z is in the sealed segment"),
        ("2026-09-15T07Z", 2, "REFUSED: decision hour 2026-09-15T07:00Z is in the sealed segment"),
        ("2024-06-30T23Z", 2, "REFUSED: decision hour 2024-06-30T23:00Z is before 2024-07-01"),
        ("2025-03-02T12Z", 1, "ERROR: WindowGapError: bingx window of 2025-03-02T12:00Z misses"),
    ],
)
@pytest.mark.parametrize("save", [False, True])
def test_forecast_one_hour_refusals_and_gaps(roots, models, hour, code, start, save, capsys):
    gap = D1_MS - 30 * HOUR_MS
    fakes.write_bingx_store(roots.data, [c for c in BINGX if c.open_ms != gap])
    argv = ["forecast", "--source", "bingx", "--hour", hour] + (["--save"] if save else [])
    assert cli.main(argv) == code
    out = capsys.readouterr().out
    if save and code == 1:
        start = "ERROR: bingx window of 2025-03-02T12:00Z misses"  # the dropped task's reason
    assert out.splitlines()[-1].startswith(start), out
    assert models == [] and not (roots.data / "forecasts").exists()


def test_forecast_batch_estimate_runs_no_fan(roots, models, capsys):
    write_stores(roots.data)
    argv = ["forecast", "--source", "bingx", "--from", "2025-03-03T12Z", "--to", "2025-03-03T15Z"]
    assert cli.main([*argv, "--estimate"]) == 0
    out = capsys.readouterr().out
    assert "bingx batch: 3 decision hours from 2025-03-03T12:00Z every 1 h" in out
    assert "gate 3 estimate: 3 fans (0 stored, 3 pending) x " in out and "within the limit" in out
    assert fans(roots.data) == []
    assert models[0].predictor.calls == 2  # the warm-up and the timed fan, nothing else


@pytest.fixture
def slow_fans(monkeypatch) -> list:
    """2000 s per fan: 3 fans x 2 threads = 12000 CPU-seconds, above gate 3."""
    measured: list = []

    def measure(*, params, root=None, models=None):
        measured.append(models)
        return 2000.0

    monkeypatch.setattr(batch, "measure_seconds_per_fan", measure)
    return measured


def test_forecast_batch_gate_3_refuses_without_the_owner_approval(roots, models, slow_fans, capsys):
    write_stores(roots.data)
    argv = ["forecast", "--source", "bingx", "--from", "2025-03-03T12Z", "--to", "2025-03-03T15Z"]
    assert cli.main(argv) == 2
    out = capsys.readouterr().out
    assert "12000 CPU-seconds (3.33 CPU-hours)" in out
    assert "needs the CPU-hours the owner approves (--approved-by-owner CPU_HOURS)" in out
    assert out.splitlines()[-1].startswith("REFUSED: the batch needs about 12000 CPU-seconds")
    assert APPROVAL_PREFIX not in out
    assert fans(roots.data) == []
    assert slow_fans == [models[0]]  # timed with the loaded models

    assert cli.main([*argv, "--estimate"]) == 0  # the estimate alone is never refused
    assert fans(roots.data) == []


@pytest.mark.parametrize("figure, ratio", [("2.5", "1.33"), ("1", "3.33"), ("0.0333", "100.10")])
def test_forecast_batch_refuses_an_estimate_well_above_the_approved_figure(
    roots, models, slow_fans, figure, ratio, capsys
):
    """D-040: a measured estimate more than 25 % above the CPU-hours the owner approved goes back
    to the owner before any fan runs; the refusal names both figures and their ratio."""
    write_stores(roots.data)
    argv = ["forecast", "--source", "bingx", "--from", "2025-03-03T12Z", "--to", "2025-03-03T15Z"]
    assert cli.main([*argv, "--approved-by-owner", figure]) == 2
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == (
        f"REFUSED: gate 3: the measured 3.33 CPU-hours (12000 CPU-seconds) are {ratio} times the "
        f"{float(figure):g} CPU-hours the owner approved, more than 1.25 times (D-040): show the "
        "new estimate to the owner before any fan runs"
    )
    assert APPROVAL_PREFIX not in out
    assert fans(roots.data) == [] and models[0].predictor.calls == 0


def test_forecast_batch_runs_half_open_with_the_owner_approval(
    roots, models, slow_fans, monkeypatch, capsys
):
    write_stores(roots.data)
    monkeypatch.setattr(forecast_cli, "now", lambda: datetime(2026, 11, 2, 9, 0, 5, tzinfo=UTC))
    argv = "forecast --source bingx --from 2025-03-03T12Z --to 2025-03-03T18Z --step-hours 2"
    argv = [*argv.split(), "--approved-by-owner", "2.7"]  # 3.33 measured: 1.23 times the figure
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    hours = [D1, D1 + timedelta(hours=2), D1 + timedelta(hours=4)]  # 18:00 is the end, excluded
    assert fans(roots.data) == sorted(fan_relpath("bingx", CONFIG_ID, "forecast", h) for h in hours)
    # the estimate, then the approval with both figures, then the fans (docs/DATA.md)
    lines = out.splitlines()
    estimate_line = next(i for i, line in enumerate(lines) if line.startswith("gate 3 estimate:"))
    assert lines[estimate_line + 1] == (
        "gate 3: --approved-by-owner 2.7 CPU-hours given at 2026-11-02T09:00:05Z; measured 3.33 "
        "CPU-hours, 1.23 times the approved figure (at most 1.25): the owner's go-ahead covers "
        "this estimate"
    )
    progress = [line for line in out.splitlines() if line.startswith("[")]
    assert [line.split()[2] for line in progress] == [utc_text(to_ms(h)) for h in hours]
    assert all("computed in" in line for line in progress)
    assert "3 fans: 3 computed, 0 stored fans reused" in out

    assert cli.main(argv) == 0  # everything stored: nothing pending, no approval needed
    out = capsys.readouterr().out
    assert "(3 stored, 0 pending)" in out and "3 fans: 0 computed, 3 stored fans reused" in out
    assert "gate 3: within the limit, no approval needed (--approved-by-owner 2.7" in out


def test_forecast_batch_lists_dropped_hours_and_runs_the_rest(roots, models, capsys):
    gap = FIRST_MS + 15 * HOUR_MS  # in the window of every D in (gap, gap + 512h]
    fakes.write_bingx_store(roots.data, [c for c in BINGX if c.open_ms != gap])
    first, end = D1_MS - 88 * HOUR_MS, D1_MS - 70 * HOUR_MS  # 18 hours; D1 - 88h reads FIRST_MS
    argv = ["forecast", "--source", "bingx", "--from", utc_text(first), "--to", utc_text(end)]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "dropped 16 of 18 tasks: their window misses candles" in out
    assert APPROVAL_PREFIX not in out  # within the limit, and the flag was not given
    assert f"bingx {utc_text(first)} forecast: 1 missing candles [{utc_text(gap)}, " in out
    assert out.count(f"1 missing candles [{utc_text(gap)}, {utc_text(gap + HOUR_MS)})") == 10
    assert "  and 6 more" in out
    kept = [D1_MS - 72 * HOUR_MS, D1_MS - 71 * HOUR_MS]  # gap + 513h and after: complete
    assert fans(roots.data) == [
        fan_relpath("bingx", CONFIG_ID, "forecast", from_ms(ms)) for ms in kept
    ]
    assert "2 fans: 2 computed, 0 stored fans reused" in out


def test_forecast_parameter_errors_are_failures(roots, monkeypatch, capsys):
    def broken():
        raise forecast_cli.ParamsError("kronos.threads: expected an integer")

    monkeypatch.setattr(forecast_cli, "load_params", broken)
    assert cli.main(["forecast", "--source", "bingx", "--hour", "2025-03-03T12Z"]) == 1
    assert capsys.readouterr().out == "ERROR: ParamsError: kronos.threads: expected an integer\n"


@pytest.mark.weights
def test_forecast_one_hour_with_the_real_weights(roots, capsys):
    """One real Kronos-small fan through the command: the loader prints nothing, the stored fan
    is the one printed."""
    write_stores(roots.data)
    argv = ["forecast", "--source", "bingx", "--hour", "2025-03-03T12Z", "--save"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("bingx window of 2025-03-03T12:00Z: 512 candles")
    relpath = fan_relpath("bingx", CONFIG_ID, "forecast", D1)
    stored = load_fan(roots.data / relpath)
    window = expected_window("bingx", D1)
    assert stored.meta["model"] == "small" and stored.paths.shape == (20, KRONOS.horizon_n, 6)
    assert f"p_up {fan_p_up(stored.paths, window.last_close):.3f}" in out
    assert f"fan {relpath} (computed and stored)" in out


# == step 2: mismatch ==============================================================================


SAMPLE = [datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 7, 3, 13, tzinfo=UTC)]  # D_0, D_1
SAMPLE_FANS = (("binance", "forecast"), ("bingx", "forecast"), ("bingx", "mismatch-floor"))
TICK = Decimal("0.1")


def bingx_like(frame: pd.DataFrame, *, seed: int) -> list:
    """BingX real-market candles following a Binance 1h frame: its prices on the 0.1 tick with
    a few ticks of noise and 4.5 times its volume, so the two stores are aligned."""
    rng = np.random.default_rng(seed)
    candles = []
    for row in frame.itertuples(index=False):
        o, h, lo, c = (Decimal(text) for text in (row.open, row.high, row.low, row.close))
        noise = [Decimal(int(v)) * TICK for v in rng.integers(-3, 4, size=4)]
        o2, c2 = (o + noise[0]).quantize(TICK), (c + noise[1]).quantize(TICK)
        h2 = max((h + abs(noise[2])).quantize(TICK), o2, c2)
        l2 = min((lo - abs(noise[3])).quantize(TICK), o2, c2)
        volume = (Decimal(row.volume) * Decimal("4.5")).quantize(Decimal("0.0001"))
        candles.append(
            Candle(int(row.open_ms), *(format(value, "f") for value in (o2, h2, l2, c2, volume)))
        )
    return candles


WORLD_START = to_ms(SAMPLE[0]) - 520 * HOUR_MS
WORLD_BINANCE = fakes.binance_frame(WORLD_START, 700, interval="1h", seed=22)
WORLD_BINGX = bingx_like(WORLD_BINANCE, seed=21)


@pytest.fixture
def kronos_world(roots, monkeypatch):
    """Both 1h stores around the first two hours of the 61-hour sample (D_0, D_1), BingX
    following Binance (the study's alignment check passes); the sample and the fallback sample
    are cut to them so that six fans stand for the 873."""
    fakes.write_binance_store(roots.data, WORLD_BINANCE)
    fakes.write_bingx_store(roots.data, WORLD_BINGX)
    monkeypatch.setattr(mm, "kronos_sample_hours", lambda: list(SAMPLE))
    monkeypatch.setattr(mm, "kronos_fallback_hours", lambda: list(SAMPLE[:1]))
    return roots


@pytest.fixture
def misaligned_world(kronos_world):
    """The same stores with the BingX candles labelled one hour late (close-time labels)."""
    late = [replace(candle, open_ms=candle.open_ms + HOUR_MS) for candle in WORLD_BINGX]
    fakes.write_bingx_store(kronos_world.data, late)
    return kronos_world


@pytest.fixture
def report_calls(monkeypatch) -> list:
    """``write_study`` replaced by a recorder (``prepare_study`` runs for real before it)."""
    calls: list = []

    def write_study(study, **kwargs):
        assert isinstance(study, mm.PreparedStudy)
        assert study.sections["alignment"]["lag0_is_largest"] is True
        calls.append(kwargs)
        summaries = kwargs["kronos_summaries"]
        kronos = (
            {"status": "not_run", "reason": kwargs["kronos_reason"] or "not run: stub"}
            if summaries is None
            else {
                "status": "run",
                "readings": {
                    "a_x_minus_a_f": {"reading": "not distinguishable", "interval_position": "x"},
                    "a_xd_minus_a_f": {"reading": "not distinguishable", "interval_position": "y"},
                },
            }
        )
        return {
            "samples": {"hours": {"used": 10, "eligible": 12}},
            "statistics_with_intervals": 40,
            "expected_chance_misses": 2.0,
            "sections": {"kronos": kronos},
        }

    monkeypatch.setattr(mm, "write_study", write_study)
    return calls


def sample_fans(config: str = CONFIG_ID) -> list[str]:
    return sorted(
        fan_relpath(source, config, stream, hour)
        for hour in SAMPLE
        for source, stream in SAMPLE_FANS
    )


def gate3_lines(reports: Path) -> list[dict]:
    path = reports / "mismatch.gate3.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_mismatch_records_its_config_then_runs_without_kronos(kronos_world, report_calls, capsys):
    roots = kronos_world
    assert cli.main(["mismatch", "--record-config"]) == 0
    config = roots.reports / "mismatch.config.json"
    assert config.read_bytes() == canonical_json(mm.study_config())
    assert report_calls == []
    capsys.readouterr()

    assert cli.main(["mismatch"]) == 0
    reason = (
        "not run: not every fan of a pre-registered sample is stored under forecast config "
        f"{CONFIG_ID} (primary sample 0 of 6; fallback sample 0 of 3 fans stored); "
        "`kronos-agent mismatch --kronos` computes them behind gate 3"
    )
    assert report_calls == [
        {
            "directory": roots.reports,
            "kronos_summaries": None,
            "gate3": None,
            "kronos_reason": reason,
        }
    ]
    out = capsys.readouterr().out
    assert "sections without Kronos computed (lag 0 is the largest return correlation)" in out
    assert "hours present in both sources: 10 of 12; 40 statistics with intervals" in out
    assert f"Kronos signal agreement: {reason}" in out
    assert "POC" not in out  # the POC part left the study (D-042)

    config.write_bytes(canonical_json({"study": "mismatch", "changed": True}))
    assert cli.main(["mismatch"]) == 2
    assert capsys.readouterr().out.startswith("REFUSED: study mismatch")
    assert len(report_calls) == 1


def test_mismatch_kronos_estimate_runs_no_fan(kronos_world, models, report_calls, capsys):
    assert cli.main(["mismatch", "--kronos", "--estimate"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    sections = next(i for i, line in enumerate(lines) if line.startswith("sections without"))
    assert sections < next(i for i, line in enumerate(lines) if line.startswith("gate 3 estimate"))
    assert "Kronos sample (primary): 2 decision hours x 3 fans" in out
    assert "gate 3 estimate: 6 fans (0 stored, 6 pending)" in out
    assert report_calls == [] and fans(kronos_world.data) == []
    assert gate3_lines(kronos_world.reports) == []  # an estimate approves nothing
    assert (kronos_world.reports / "mismatch.config.json").is_file()  # verified before the fans

    assert cli.main(["mismatch", "--kronos", "--fallback", "--estimate"]) == 0
    out = capsys.readouterr().out
    assert "Kronos sample (fallback): 1 decision hours x 3 fans" in out
    assert "gate 3 estimate: 3 fans" in out


def test_mismatch_kronos_gate_3_refuses_without_or_well_above_the_approved_figure(
    kronos_world, models, slow_fans, report_calls, capsys
):
    # 6 fans x 2000 s x 2 threads = 24000 CPU-seconds, 6.67 CPU-hours
    assert cli.main(["mismatch", "--kronos"]) == 2
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("REFUSED: the batch needs about 24000 CPU-seconds")
    assert cli.main(["mismatch", "--kronos", "--approved-by-owner", "5"]) == 2
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith(
        "REFUSED: gate 3: the measured 6.67 CPU-hours (24000 CPU-seconds) are 1.33 times the 5 "
        "CPU-hours the owner approved, more than 1.25 times (D-040)"
    )
    assert APPROVAL_PREFIX not in out
    assert report_calls == [] and fans(kronos_world.data) == []
    assert gate3_lines(kronos_world.reports) == [] and models[0].predictor.calls == 0


def test_mismatch_kronos_records_the_approval_before_the_first_fan_and_keeps_it(
    kronos_world, models, slow_fans, monkeypatch, capsys
):
    """The approved run appends its gate-3 record (the command's start, the approved figure, the
    measured estimate) before its first fan; reruns with nothing pending add none and every later
    report, plain `mismatch` included, lists it with the Kronos part rebuilt from the fans."""
    reports, data = kronos_world.reports, kronos_world.data
    records = reports / "mismatch.gate3.jsonl"
    clock = {"now": datetime(2026, 11, 2, 9, 30, tzinfo=UTC)}
    monkeypatch.setattr(mismatch_cli, "_now", lambda: clock["now"])
    seen: list[bool] = []
    real_predict = WalkPredictor.predict_batch

    def predict(self, *args, **kwargs):
        seen.append(records.is_file())
        clock["now"] += timedelta(hours=1)  # the batch takes time: the approval is its start
        return real_predict(self, *args, **kwargs)

    monkeypatch.setattr(WalkPredictor, "predict_batch", predict)
    assert cli.main(["mismatch", "--kronos", "--approved-by-owner", "6"]) == 0
    out = capsys.readouterr().out
    assert (
        "gate 3: --approved-by-owner 6 CPU-hours given at 2026-11-02T09:30:00Z; measured 6.67 "
        "CPU-hours, 1.11 times the approved figure (at most 1.25)"
    ) in out
    assert fans(data) == sample_fans()
    assert seen == [True] * 6  # the record was on disk before the first fan
    (record,) = gate3_lines(reports)
    assert record["given_at"] == "2026-11-02T09:30:00Z"  # not the end of the batch (15:30)
    assert (record["approved_cpu_hours"], record["design"], record["config_id"]) == (
        6.0,
        "primary",
        CONFIG_ID,
    )
    assert record["estimate"]["pending"] == 6 and record["estimate"]["cpu_seconds"] == 24000.0
    assert record["ratio"] == pytest.approx(24000 / 3600 / 6) and record["dropped"] == []
    report = json.loads((reports / "mismatch.json").read_text())
    kronos = report["sections"]["kronos"]
    assert kronos["status"] == "run" and kronos["gate3"]["records"] == [record]
    assert kronos["gate3"]["invocation"]["approved_cpu_hours"] == 6.0
    markdown = (reports / "mismatch.md").read_text()
    assert "the owner approved 6 CPU-hours (--approved-by-owner given at 2026-11-02T09:30:00Z)" in (
        markdown
    )
    first = records.read_bytes()

    # a rerun with everything stored: no approval needed, no record added, the original listed
    assert cli.main(["mismatch", "--kronos"]) == 0
    out = capsys.readouterr().out
    assert "(6 stored, 0 pending)" in out and "gate-3 record appended" not in out
    assert records.read_bytes() == first
    report = json.loads((reports / "mismatch.json").read_text())
    gate3 = report["sections"]["kronos"]["gate3"]
    assert gate3["records"] == [record] and gate3["invocation"]["estimate"]["pending"] == 0

    # plain `mismatch`: the part is rebuilt from the stored fans, nothing is computed or measured
    monkeypatch.setattr(forecast_cli, "_load_models", fail_if_called("_load_models"))
    monkeypatch.setattr(batch, "measure_seconds_per_fan", fail_if_called("measure_seconds"))
    assert cli.main(["mismatch"]) == 0
    out = capsys.readouterr().out
    assert (
        f"Kronos part rebuilt from the 6 stored fans of the primary sample (forecast config "
        f"{CONFIG_ID}); no fan computed"
    ) in out
    report = json.loads((reports / "mismatch.json").read_text())
    kronos = report["sections"]["kronos"]
    assert kronos["status"] == "run" and kronos["gate3"]["records"] == [record]
    assert kronos["gate3"]["invocation"]["rebuilt_from_stored_fans"] is True
    assert records.read_bytes() == first
    agreement = reports / "figures" / mm.AGREEMENT_FIGURE
    series = data / "studies" / "mismatch_kronos.csv"
    assert agreement.is_file() and series.is_file()

    # a fan missing: the part has not run, and the files beside the report are the ones it lists
    (data / sample_fans()[0]).unlink()
    assert cli.main(["mismatch"]) == 0
    out = capsys.readouterr().out
    assert "primary sample 5 of 6; fallback sample 2 of 3 fans stored" in out
    report = json.loads((reports / "mismatch.json").read_text())
    assert report["sections"]["kronos"]["status"] == "not_run"
    assert not agreement.exists() and not series.exists()
    assert records.read_bytes() == first


def test_plain_mismatch_with_a_stale_fan_writes_the_report_without_the_kronos_part(
    kronos_world, models, capsys
):
    """mismatch-core-3: a store corrected after the fans were computed leaves stale fans. A plain
    `mismatch` still writes the report of the current stores, the reason naming the first stale
    fan, and removes the earlier report's Kronos part; `--kronos` refuses the stale fans (never
    overwritten) until they are moved aside, then computes them again."""
    reports, data = kronos_world.reports, kronos_world.data
    assert cli.main(["mismatch", "--kronos"]) == 0  # 6 fast fans, within gate 3
    capsys.readouterr()
    agreement = reports / "figures" / mm.AGREEMENT_FIGURE
    series = data / "studies" / "mismatch_kronos.csv"
    assert agreement.is_file() and series.is_file()
    target = to_ms(SAMPLE[1]) - 3 * HOUR_MS  # inside the windows of D_1 only
    corrected = [
        replace(c, volume=format(Decimal(c.volume) + 1, "f")) if c.open_ms == target else c
        for c in WORLD_BINGX
    ]
    fakes.write_bingx_store(data, corrected)  # still aligned with Binance
    stale = [
        fan_relpath(source, CONFIG_ID, stream, SAMPLE[1])
        for source, stream in SAMPLE_FANS
        if source == "bingx"
    ]

    assert cli.main(["mismatch"]) == 0
    out = capsys.readouterr().out
    report = json.loads((reports / "mismatch.json").read_text())
    kronos = report["sections"]["kronos"]
    assert kronos["status"] == "not_run"
    assert kronos["reason"].startswith(
        f"not run: a stored fan of the primary sample cannot be reused (stored fan {stale[0]} "
        f"cannot be reused for the bingx window of {utc_text(to_ms(SAMPLE[1]))} (stream "
        "forecast): window sha256, inputs_sha256, context_sha256 differ"
    )
    assert kronos["reason"].endswith(
        "move that fan aside (other stored fans of the sample may be stale too), then "
        "`kronos-agent mismatch --kronos` computes the missing fans behind gate 3"
    )
    assert f"Kronos signal agreement: {kronos['reason']}" in out
    assert not agreement.exists() and not series.exists()  # the old stores' part is gone
    bingx_meta = read_meta(StoreId("bingx", "1h"), data)
    assert bingx_meta["csv_sha256"] in json.dumps(report["data"])  # the store as read now

    assert cli.main(["mismatch", "--kronos"]) == 1
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith(f"ERROR: ForecastStoreError: stored fan {stale[0]} cannot be reused")
    aside = data / "aside"
    aside.mkdir()
    for relpath in stale:
        (data / relpath).rename(aside / relpath.replace("/", "_"))
    assert cli.main(["mismatch", "--kronos"]) == 0
    out = capsys.readouterr().out
    assert "(4 stored, 2 pending)" in out
    report = json.loads((reports / "mismatch.json").read_text())
    assert report["sections"]["kronos"]["status"] == "run"
    assert agreement.is_file() and series.is_file()


def test_mismatch_kronos_dropped_hours_reach_the_report(kronos_world, models, report_calls, capsys):
    store = StoreId("bingx", "1h")
    gap = to_ms(SAMPLE[1]) - 2 * HOUR_MS  # inside D_1's window only
    fakes.write_bingx_store(kronos_world.data, [c for c in WORLD_BINGX if c.open_ms != gap])
    assert read_meta(store, kronos_world.data)["missing_candles"] == 1
    assert cli.main(["mismatch", "--kronos"]) == 0  # 4 fast fans, within gate 3
    out = capsys.readouterr().out
    assert "dropped 2 of 6 tasks" in out
    ((call),) = report_calls
    assert len(call["kronos_summaries"]) == 4
    assert [(d["source"], d["stream"]) for d in call["gate3"]["dropped"]] == [
        ("bingx", "forecast"),
        ("bingx", "mismatch-floor"),
    ]
    assert call["gate3"]["approved_cpu_hours"] is None
    assert call["gate3"]["estimate"]["pending"] == 4
    # a batch within the limit records its run too, with no approval
    (record,) = gate3_lines(kronos_world.reports)
    assert record["approved_cpu_hours"] is None and record["ratio"] is None
    assert [(d["source"], d["stream"]) for d in record["dropped"]] == [
        ("bingx", "forecast"),
        ("bingx", "mismatch-floor"),
    ]


@pytest.mark.parametrize(
    "argv",
    [[], ["--kronos", "--estimate"], ["--kronos", "--approved-by-owner", "10"]],
    ids=["plain", "estimate", "approved"],
)
def test_mismatch_checks_the_alignment_before_any_kronos_work(
    misaligned_world, models, slow_fans, argv, capsys
):
    """mismatch-core-1: a misaligned pair of stores fails before any model load, timing run,
    estimate, approval record or fan: approved compute is never spent on a failing report."""
    assert cli.main(["mismatch", *argv]) == 1
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith(
        "ERROR: AlignmentError: the 1h return correlation is not largest at lag 0"
    )
    assert "gate 3 estimate" not in out and APPROVAL_PREFIX not in out
    assert models == [] and slow_fans == []
    assert fans(misaligned_world.data) == []
    assert gate3_lines(misaligned_world.reports) == []
    assert not (misaligned_world.reports / "mismatch.json").exists()


def test_forecast_and_mismatch_refuse_to_read_stores_a_data_run_is_writing(
    kronos_world, models, capsys
):
    """D-039: a busy lock is a refusal (exit 2), not a failure."""
    hour = utc_text(to_ms(SAMPLE[1]))
    with exclusive_lock(kronos_world.data / "raw" / "bingx" / ".lock"):
        for argv in (
            ["forecast", "--source", "bingx", "--hour", hour],
            ["forecast", "--source", "bingx", "--hour", hour, "--save"],
            ["mismatch"],
            ["mismatch", "--kronos", "--estimate"],
        ):
            assert cli.main(argv) == 2, argv
            last = capsys.readouterr().out.splitlines()[-1]
            assert last.startswith(
                "REFUSED: candles/bingx/1h.csv: a run that writes the bingx data holds "
                "raw/bingx/.lock"
            ), argv
    assert models == [] and fans(kronos_world.data) == []


def test_a_signal_ends_a_command_through_its_clean_up(tmp_path):
    """SIGTERM (or SIGHUP) raises SystemExit inside the command, so ``finally`` blocks run and
    the previous handlers come back; a signal the parent set to be ignored stays ignored."""
    code = (
        "import os, signal\n"
        "from kronos_agent.cli import exit_on_signals\n"
        "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
        "cleaned = []\n"
        "try:\n"
        "    with exit_on_signals():\n"
        "        ignored = signal.getsignal(signal.SIGHUP) is signal.SIG_IGN\n"
        "        try:\n"
        "            os.kill(os.getpid(), signal.SIGTERM)\n"
        "            for _ in range(10**8):\n"
        "                pass\n"
        "        finally:\n"
        "            cleaned.append(True)\n"
        "except SystemExit as exc:\n"
        "    restored = signal.getsignal(signal.SIGTERM) is signal.SIG_DFL\n"
        "    print('RESULT', exc.code, cleaned, ignored, restored)\n"
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
    assert completed.stdout.strip() == f"RESULT {128 + signal.SIGTERM} [True] True True"


def test_main_restores_the_signal_handlers(capsys):
    before = signal.getsignal(signal.SIGTERM)
    assert cli.main(["params", "version"]) == 0
    assert signal.getsignal(signal.SIGTERM) is before
    capsys.readouterr()
