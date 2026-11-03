"""Figures: fixed size and style, no Software metadata, identical bytes across renders."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pytest

from kronos_agent.hashing import sha256_bytes
from kronos_agent.paths import reports_dir
from kronos_agent.report.figures import (
    FIGURE_DPI,
    FIGURE_SIZE,
    RC_PARAMS,
    apply_style,
    figures_dir,
    new_figure,
    render_png,
    save_png,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def draw():
    figure, axes = new_figure(1, 2)
    x = np.linspace(0.0, 10.0, 200)
    axes[0].plot(x, np.sin(x), label="sin")
    axes[0].plot(x, np.cos(x), label="cos")
    axes[0].legend()
    axes[0].set_title("Unit figure")
    axes[1].hist(np.random.default_rng(0).normal(size=500), bins=30)
    axes[1].set_xlabel("|dPOC| (bp)")
    figure.suptitle("Determinism check")
    return figure


def draw_dates():
    """72 hourly points from 2024-07-01T00:00Z on a datetime64 axis labelled UTC."""
    figure, axis = new_figure()
    hours = np.datetime64("2024-07-01T00:00", "ms") + np.arange(72) * np.timedelta64(1, "h")
    axis.plot(hours, np.sin(np.arange(72) / 6.0))
    axis.set_xlabel("decision hour D (UTC)")
    return figure, axis


def ticks(axis) -> tuple[list[datetime], list[str]]:
    """Major tick instants (UTC) and labels of the x axis of a rendered figure."""
    instants = [mdates.num2date(value, tz=UTC) for value in axis.xaxis.get_majorticklocs()]
    return instants, [label.get_text() for label in axis.get_xticklabels()]


OTHER_ZONE = "Asia/Tokyo"  # UTC+9, no DST


def png_size(data: bytes) -> tuple[int, int]:
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def test_two_renders_give_identical_bytes(tmp_path):
    path = figures_dir(tmp_path) / "unit.png"
    first = save_png(draw(), path)
    data = path.read_bytes()
    second = save_png(draw(), path)
    assert path.read_bytes() == data
    assert first == second == {"relpath": "figures/unit.png", "sha256": sha256_bytes(data)}
    assert data.startswith(PNG_SIGNATURE)
    assert png_size(data) == (int(FIGURE_SIZE[0] * FIGURE_DPI), int(FIGURE_SIZE[1] * FIGURE_DPI))
    assert png_size(data) == (800, 450)
    assert [p.name for p in path.parent.iterdir()] == ["unit.png"]  # no temporary file left


def test_png_carries_no_software_metadata():
    data = render_png(draw())
    assert b"Software" not in data
    assert b"matplotlib" not in data.lower()


def test_process_rcparams_do_not_reach_the_figure():
    reference = render_png(draw())
    changed = {
        "lines.linewidth": 4.0,
        "font.size": 20.0,
        "axes.facecolor": "black",
        "savefig.dpi": 300,
        "figure.dpi": 50,
        "axes.grid": False,
    }
    with matplotlib.rc_context(changed):
        assert render_png(draw()) == reference


def test_new_figure_shapes_and_checks():
    figure, axes = new_figure()
    assert hasattr(axes, "plot")
    assert tuple(figure.get_size_inches()) == FIGURE_SIZE and figure.dpi == FIGURE_DPI
    _, grid = new_figure(2, 3)
    assert grid.shape == (2, 3)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            new_figure(bad, 1)


def test_save_png_refuses_other_formats(tmp_path):
    with pytest.raises(ValueError):
        save_png(draw(), tmp_path / "unit.svg")
    assert list(tmp_path.iterdir()) == []


def test_figures_dir():
    assert figures_dir() == reports_dir() / "data" / "figures"


def test_figures_never_import_pyplot():
    code = (
        "import sys\n"
        "from kronos_agent.report.figures import new_figure, render_png\n"
        "figure, axes = new_figure()\n"
        "axes.plot([1, 2], [3, 4])\n"
        "render_png(figure)\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_style_pins_the_keys_rcdefaults_leaves_alone():
    assert (RC_PARAMS["timezone"], RC_PARAMS["date.epoch"]) == ("UTC", "1970-01-01T00:00:00")
    with matplotlib.rc_context({"timezone": OTHER_ZONE}):
        matplotlib.rcdefaults()
        assert matplotlib.rcParams["timezone"] == OTHER_ZONE  # what rcdefaults() keeps
        apply_style()
        assert matplotlib.rcParams["timezone"] == "UTC"


def test_date_axes_are_placed_and_labelled_in_utc_whatever_the_process_timezone():
    figure, axis = draw_dates()
    reference = render_png(figure)
    instants, labels = ticks(axis)
    assert instants[:2] == [
        datetime(2024, 7, 1, 0, tzinfo=UTC),
        datetime(2024, 7, 1, 12, tzinfo=UTC),
    ]
    assert labels[:2] == ["07-01 00", "07-01 12"]
    with matplotlib.rc_context({"timezone": OTHER_ZONE}):
        figure, axis = draw_dates()
        assert render_png(figure) == reference
        assert ticks(axis) == (instants, labels)  # read here: labels are formatted on access


DATE_FIGURE_IN_A_FRESH_PROCESS = textwrap.dedent(
    """
    import sys

    import matplotlib

    sys.path.insert(0, sys.argv[1])
    from test_figures import draw_dates

    from kronos_agent.hashing import sha256_bytes
    from kronos_agent.report.figures import render_png

    print(matplotlib.rcParams["timezone"], matplotlib.rcParams["lines.linewidth"])
    figure, _ = draw_dates()
    print(sha256_bytes(render_png(figure)))
    """
)


def test_a_user_matplotlibrc_never_reaches_the_figure(tmp_path):
    figure, _ = draw_dates()
    reference = sha256_bytes(render_png(figure))
    rc_file = tmp_path / "config" / "matplotlibrc"
    rc_file.parent.mkdir()
    rc_file.write_text(f"timezone: {OTHER_ZONE}\nlines.linewidth: 5\n")
    completed = subprocess.run(
        [sys.executable, "-B", "-c", DATE_FIGURE_IN_A_FRESH_PROCESS, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "MATPLOTLIBRC": str(rc_file)},
    )
    assert completed.returncode == 0, completed.stderr
    loaded, digest = completed.stdout.splitlines()
    assert loaded == f"{OTHER_ZONE} 5.0"  # the user's file was read by matplotlib
    assert digest == reference
