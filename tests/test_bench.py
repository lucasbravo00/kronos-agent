"""Benchmark: a small in-process run, report rendering, the worker protocol and the CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kronos_agent.bench import (
    build_input,
    headline_lines,
    parse_threads_sweep,
    peak_rss_mb,
    redact,
    render_markdown,
    run_benchmark,
    run_child,
    write_reports,
)
from kronos_agent.cli import bench as bench_cli
from kronos_agent.envinfo import personal_markers
from kronos_agent.paths import REPO_ROOT, default_label
from kronos_agent.runtime import RuntimeConfigError

TINY = {"trajectories": 2, "horizon": 2, "repeats": 1, "include_fixture_replica": False}


@pytest.fixture(scope="module")
def tiny_result():
    return run_benchmark(**TINY)


def _cli_args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    bench_cli.register(parser.add_subparsers(dest="command"))
    return parser.parse_args(["bench", *argv])


# --- no weights needed ----------------------------------------------------------------------------


def test_parse_threads_sweep():
    assert parse_threads_sweep(None) == []
    assert parse_threads_sweep("") == []
    assert parse_threads_sweep("1,2,10") == [1, 2, 10]
    assert parse_threads_sweep(" 2, 2 ,1") == [2, 1]
    with pytest.raises(ValueError):
        parse_threads_sweep("1,x")
    with pytest.raises(ValueError):
        parse_threads_sweep("0")


def test_peak_rss_is_positive_mib():
    assert 10 < peak_rss_mb() < 1_000_000


def test_redact_removes_the_home_directory():
    home = str(Path.home())
    redacted = redact(f"failed at {home}/weights/x.safetensors")
    assert home not in redacted
    assert redacted.endswith("/weights/x.safetensors")
    assert redact("plain text") == "plain text"


def test_build_input_describes_both_sources():
    for source in ("upstream_fixture", "synthetic"):
        inp = build_input(source, 512, 3)
        assert len(inp.context) == 512
        assert len(inp.y_timestamps) == 3
        assert inp.y_timestamps.iloc[0] == inp.x_timestamps.iloc[-1] + inp.step
        assert inp.description["source"] == source
        assert inp.description["rows"] == 512
        assert inp.description["horizon"] == 3
        json.dumps(inp.description)
    with pytest.raises(ValueError):
        build_input("other", 512, 3)


def test_run_benchmark_rejects_bad_arguments():
    with pytest.raises(ValueError, match="model"):
        run_benchmark(model="huge", **TINY)
    with pytest.raises(ValueError, match="source"):
        run_benchmark(source="csv", **TINY)
    with pytest.raises(ValueError, match="trajectories"):
        run_benchmark(trajectories=0, horizon=2, repeats=1)
    with pytest.raises(ValueError, match="label"):
        run_benchmark(label="../escape", **TINY)
    with pytest.raises(RuntimeConfigError):
        run_benchmark(threads=3, **TINY)


def test_worker_failure_is_recorded_not_raised():
    record = run_child(
        threads=2,
        device="cpu",
        model="small",
        trajectories=1,
        horizon=1,
        repeats=1,
        source="not-a-source",
        timeout=120,
    )
    assert record["threads"] == 2 and record["device"] == "cpu"
    assert record["error"] is not None
    assert "without a JSON result" in record["error"]
    assert str(Path.home()) not in record["error"]


def test_cli_refuses_a_thread_count_that_differs_from_the_configured_runtime(capsys):
    assert bench_cli.run(_cli_args("--threads", "3", "--json")) == 2
    assert "ERROR" in capsys.readouterr().out


def test_cli_module_imports_no_torch():
    code = "import sys, kronos_agent.cli.bench; print('torch' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-1000:]
    assert completed.stdout.strip() == "False"


def test_cli_registers_bench_with_the_contract_options():
    from kronos_agent.cli import build_parser

    parser, runners = build_parser()
    assert "bench" in runners
    assert "bench" in parser.format_help()
    args = _cli_args()
    assert args.model == "small" and args.trajectories == 20 and args.horizon == 8
    assert args.threads == 2 and args.repeats == 3 and args.source == "upstream_fixture"
    assert args.threads_sweep is None and args.include_mps is False and args.json is False
    assert args.label == default_label()


# --- weights needed -------------------------------------------------------------------------------


@pytest.mark.weights
def test_small_benchmark_reports_positive_timings_and_determinism(tiny_result):
    result = tiny_result
    fan = result["fan"]
    assert fan["trajectories"] == 2 and fan["horizon"] == 2 and fan["repeats"] == 1
    assert fan["warmup_seconds"] > 0
    assert len(fan["seconds"]) == 1 and fan["seconds"][0] > 0
    assert fan["median_seconds"] == fan["min_seconds"] == fan["max_seconds"] == fan["seconds"][0]
    assert fan["seconds_per_trajectory_candle"] == pytest.approx(fan["median_seconds"] / 4)
    assert fan["fan_shape"] == [2, 2, 6] and fan["finite"] is True
    assert len(fan["fan_fingerprint"]) == 64
    assert fan["seed_salts"] == ["r0"]
    assert (
        fan["reference_seed"]["canonical"]
        == "kronos-agent/seed/v1/bench/BTC-USDT/2026-10-29T16:00:00Z/r0"
    )
    assert result["determinism"]["same_seed_identical"] is True
    assert result["determinism"]["same_seed_max_abs_diff"] == 0.0
    assert result["determinism"]["different_seed_differs"] is True
    assert result["determinism"]["different_seed_salt"] == "alt"
    assert result["threads"] == 2 == result["runtime"]["threads"]
    assert result["peak_rss_mb"] > 0
    assert result["model"]["name"] == "small"
    assert result["model"]["load_seconds"] > 0
    assert result["model"]["load_wall_seconds"] >= result["model"]["load_seconds"]
    assert result["model"]["hub_messages"] == ["Loading weights from local directory"] * 2
    assert result["mean_path"]["is_fan"] is False
    assert result["mean_path"]["shape"] == [2, 6] and result["mean_path"]["seconds"] > 0
    assert result["fixture_replica"] is None
    assert result["threads_sweep"] is None and result["mps"] is None
    assert result["runtime"]["device"] == "cpu"
    assert result["runtime"]["deterministic"] is True and result["runtime"]["hf_offline"] is True
    assert result["input"]["source"] == "upstream_fixture"
    assert result["input"]["rows"] == 512 and result["input"]["horizon"] == 2


@pytest.mark.weights
def test_result_is_json_serialisable_and_free_of_personal_data(tiny_result):
    text = json.dumps(tiny_result)
    assert json.loads(text) == tiny_result
    assert str(Path.home()) not in text
    for kind, marker in personal_markers():  # a project-named login or host is no identity
        assert marker not in text, kind


@pytest.mark.weights
def test_reports_are_written_under_the_label(tiny_result, tmp_path):
    json_path, md_path = write_reports(tiny_result, label="test-label", directory=tmp_path)
    assert json_path == tmp_path / "test-label.bench.json"
    assert md_path == tmp_path / "test-label.bench.md"
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        **tiny_result,
        "label": "test-label",
    }
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Kronos forecast benchmark: test-label")
    assert "| Same seed twice gives an identical fan | yes |" in markdown
    assert "| Different seed gives a different fan | yes |" in markdown
    assert str(Path.home()) not in markdown
    with pytest.raises(ValueError):
        write_reports(tiny_result, label="../escape", directory=tmp_path)


@pytest.mark.weights
def test_headline_lines_and_markdown_cover_sweep_and_mps_records(tiny_result):
    lines = headline_lines(tiny_result)
    assert lines[0].startswith("Kronos-small fan 2 x 2 on cpu with 2 threads: median")
    assert "same seed identical: True; different seed differs: True" in lines
    result = json.loads(json.dumps(tiny_result))
    result["threads_sweep"] = [
        {
            "threads": 1,
            "device": "cpu",
            "error": None,
            "median_seconds": 2.0,
            "min_seconds": 1.9,
            "max_seconds": 2.1,
            "peak_rss_mb": 500.0,
            "identical_to_reference_fan": False,
        },
        {
            "threads": 10,
            "device": "cpu",
            "error": "worker timed out after 5 s",
            "identical_to_reference_fan": None,
        },
    ]
    result["mps"] = {
        "threads": 2,
        "device": "mps",
        "error": "RuntimeError: no MPS",
        "identical_to_reference_fan": None,
    }
    markdown = render_markdown(result)
    assert "| 1 | 2.00 s | 1.90 s | 2.10 s | no | 500 MiB | - |" in markdown
    assert "| 10 | n/a | n/a | n/a | n/a | n/a | worker timed out after 5 s |" in markdown
    assert "| mps | n/a | n/a | RuntimeError: no MPS |" in markdown
    lines = headline_lines(result)
    assert any(
        "threads 1: median 2.00 s; fan identical to the reference fan: no" in line for line in lines
    )
    assert any(line.startswith("threads 10: error") for line in lines)
    assert any(line.startswith("mps: error") for line in lines)


@pytest.mark.weights
def test_worker_reproduces_the_parent_fan_bit_for_bit(tiny_result):
    record = run_child(
        threads=2,
        device="cpu",
        model="small",
        trajectories=2,
        horizon=2,
        repeats=1,
        source="upstream_fixture",
        timeout=600,
    )
    assert record["error"] is None, record["error"]
    assert record["threads"] == 2 and record["device"] == "cpu" and record["deterministic"] is True
    assert record["median_seconds"] > 0 and record["peak_rss_mb"] > 0
    assert record["fan_shape"] == [2, 2, 6]
    assert "determinism" not in record
    assert record["fan_fingerprint"] == tiny_result["fan"]["fan_fingerprint"]


@pytest.mark.weights
def test_cli_json_mode_prints_only_the_report(capsys):
    args = _cli_args(
        "--trajectories", "1", "--horizon", "1", "--repeats", "1", "--json", "--label", "cli-test"
    )
    assert bench_cli.run(args) == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["label"] == "cli-test"
    assert result["fan"]["trajectories"] == 1 and result["fan"]["horizon"] == 1
    assert result["fixture_replica"]["contexts"]["512"]["shape"] == [8, 6]
    assert result["fixture_replica"]["contexts"]["256"]["shape"] == [8, 6]
    assert isinstance(result["fixture_replica"]["contexts"]["512"]["passes_rtol_1e-5"], bool)
