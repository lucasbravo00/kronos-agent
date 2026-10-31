"""kronos_agent.paths: report labels, defined once and shared by env-report and bench."""

from __future__ import annotations

import argparse
import platform
import re
import sys

import pytest

from kronos_agent import paths

VALID_LABELS = ("darwin-arm64", "linux-x86_64", "a.b_c-1", "x" * 64)
INVALID_LABELS = ("../escape", "sub/dir", "", ".hidden", "-dash", "sp ace", "x" * 65, "end\n", None)


def test_default_label():
    label = paths.default_label()
    assert label == f"{sys.platform}-{platform.machine()}"
    assert re.fullmatch(r"[a-z0-9]+-[A-Za-z0-9_]+", label)
    assert paths.validate_label(label) == label


@pytest.mark.parametrize("label", VALID_LABELS)
def test_validate_label_accepts_file_stems(label):
    assert paths.validate_label(label) == label


@pytest.mark.parametrize("label", INVALID_LABELS)
def test_validate_label_rejects_anything_but_a_plain_file_stem(label):
    with pytest.raises(ValueError, match="label"):
        paths.validate_label(label)


def test_env_report_label_option_keeps_its_message():
    from kronos_agent.cli import env_report

    assert env_report._label("darwin-arm64") == "darwin-arm64"
    for label in ("../escape", "end\n"):
        with pytest.raises(argparse.ArgumentTypeError, match="plain file stem"):
            env_report._label(label)


def test_report_labels_have_one_definition():
    from kronos_agent import bench, envinfo
    from kronos_agent.cli import bench as bench_cli
    from kronos_agent.cli import env_report

    assert bench.default_label is paths.default_label
    assert bench.validate_label is paths.validate_label
    assert bench_cli.default_label is paths.default_label
    assert env_report.default_label is paths.default_label
    assert env_report.validate_label is paths.validate_label
    assert not hasattr(envinfo, "default_label")
    for module in (bench, bench_cli, env_report, envinfo):
        assert not hasattr(module, "LABEL_RE"), module.__name__
