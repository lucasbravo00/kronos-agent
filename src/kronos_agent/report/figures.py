"""Deterministic report figures (SPEC section 14): two renders give identical PNG bytes.

Figures are built with the object-oriented API on an Agg canvas (no ``pyplot``, no GUI backend,
no global figure registry) at a fixed size and DPI. Style comes from matplotlib's built-in
defaults plus :data:`RC_PARAMS`, never from a user's ``matplotlibrc``: :func:`new_figure` resets
the global rcParams to that style before the caller draws (artists read most rcParams when they
are created) and :func:`save_png` renders under the same style. ``matplotlib.rcdefaults()`` leaves
the keys of ``matplotlib.style.core.STYLE_BLACKLIST`` alone, among them ``timezone`` and
``date.epoch``, so :data:`RC_PARAMS` pins both: date axes are placed and labelled in UTC
whatever a ``matplotlibrc`` says. Text uses DejaVu Sans, which ships with matplotlib, so no
system font is involved. The PNG is written without the ``Software`` metadata (the only
metadata Agg writes by default) through the durable atomic write.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..data.fsutil import atomic_write_bytes
from ..hashing import sha256_bytes
from ..paths import reports_dir

FIGURE_SIZE: tuple[float, float] = (8.0, 4.5)
FIGURE_DPI = 100
FIGURES_SUBDIR = "figures"
RC_PARAMS: dict[str, Any] = {
    "font.family": ["sans-serif"],
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "figure.figsize": list(FIGURE_SIZE),
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "standard",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "axes.formatter.use_locale": False,
    "svg.hashsalt": "kronos-agent",
    # not reset by rcdefaults(): a user's matplotlibrc would otherwise place and label date axes
    "timezone": "UTC",
    "date.epoch": "1970-01-01T00:00:00",
}


def apply_style() -> None:
    """Reset the global rcParams to matplotlib's built-in defaults plus :data:`RC_PARAMS`."""
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(RC_PARAMS)


def new_figure(nrows: int = 1, ncols: int = 1) -> tuple[Figure, Any]:
    """A fixed-size figure on an Agg canvas and its axes (one ``Axes`` or an array of them).

    The global style is reset first (:func:`apply_style`), so whatever the process changed in
    rcParams before does not reach the figure.
    """
    for name, value in (("nrows", nrows), ("ncols", ncols)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    apply_style()
    figure = Figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(nrows, ncols)
    return figure, axes


def render_png(figure: Figure) -> bytes:
    """PNG bytes of ``figure`` under the fixed style, without the ``Software`` metadata."""
    buffer = io.BytesIO()
    with matplotlib.rc_context():
        apply_style()
        figure.savefig(buffer, format="png", dpi=FIGURE_DPI, metadata={"Software": None})
    return buffer.getvalue()


def save_png(fig: Figure, path: Path) -> dict[str, str]:
    """Render ``fig`` and write it atomically to ``path`` (a ``.png``).

    Returns ``{"relpath", "sha256"}``; ``relpath`` is ``<figures dir name>/<file name>``, the
    path relative to the report directory that a Markdown report beside it links.
    """
    path = Path(path)
    if path.suffix != ".png":
        raise ValueError(f"figures are PNG files, got {path.name!r}")
    data = render_png(fig)
    atomic_write_bytes(path, data)
    return {"relpath": f"{path.parent.name}/{path.name}", "sha256": sha256_bytes(data)}


def figures_dir(directory: Path | None = None) -> Path:
    """``<report directory>/figures``; the report directory defaults to ``reports/data``."""
    base = Path(directory) if directory is not None else reports_dir() / "data"
    return base / FIGURES_SUBDIR


__all__ = [
    "FIGURE_DPI",
    "FIGURE_SIZE",
    "RC_PARAMS",
    "apply_style",
    "figures_dir",
    "new_figure",
    "render_png",
    "save_png",
]
