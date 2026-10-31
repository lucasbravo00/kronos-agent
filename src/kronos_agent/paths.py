"""Repository paths with environment overrides, home-directory redaction and report labels.

Nothing here reads secrets. Personal paths never reach the repository: pass every path that
ends up in a report through :func:`redact_home`. Report files are named after a label, a plain
file stem checked by :func:`validate_label` (``<platform>-<machine>`` by default).
"""

from __future__ import annotations

import os
import platform
import re
import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

WEIGHTS_DIR_ENV = "KRONOS_AGENT_WEIGHTS_DIR"
DATA_DIR_ENV = "KRONOS_AGENT_DATA_DIR"
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def repo_root() -> Path:
    return REPO_ROOT


def _dir_from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def weights_dir() -> Path:
    """Local weights root (``weights/`` by default; gitignored; relocatable)."""
    return _dir_from_env(WEIGHTS_DIR_ENV, REPO_ROOT / "weights")


def data_dir() -> Path:
    """Local data root (``data/`` by default; gitignored; relocatable)."""
    return _dir_from_env(DATA_DIR_ENV, REPO_ROOT / "data")


def manifests_dir() -> Path:
    return REPO_ROOT / "manifests"


def vendored_kronos_dir() -> Path:
    return REPO_ROOT / "third_party" / "kronos"


def reports_dir() -> Path:
    return REPO_ROOT / "reports"


def redact_home(text: str) -> str:
    """Replace the current user's home directory with ``~`` wherever it appears in ``text``."""
    home = str(Path.home())
    if home and home != "/":
        text = text.replace(home, "~")
    return text


def default_label() -> str:
    """``<sys.platform>-<machine>``, e.g. ``darwin-arm64`` or ``linux-x86_64``."""
    return f"{sys.platform}-{platform.machine()}"


def validate_label(label: str) -> str:
    """A report label is a short file-name-safe token (letters, digits, ``.``, ``_``, ``-``)."""
    if not isinstance(label, str) or not LABEL_RE.fullmatch(label):
        raise ValueError(f"label must match {LABEL_RE.pattern}, got {label!r}")
    return label
