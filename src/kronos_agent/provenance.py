"""Code provenance for store metas, reports and stored fans (``docs/DATA.md``, "Provenance").

The repository stays at one commit for the whole project (CLAUDE.md, "Git"), so the commit alone
does not identify the code that produced a result: every record carries
``{commit, dirty, source_tree_sha256}``, and the tree hash is what identifies the code. It is the
sha256 over the sorted relative paths and contents of the package sources, ``params.yaml``,
``pyproject.toml``, ``uv.lock`` and ``manifests/*`` (:data:`SOURCE_GLOBS`). The narrower
:func:`forecasting_code_sha256` hashes the whole files of the fan path's own modules, the
construction of the models included (``kronos/weights.py`` loads the weights and builds the
predictor, ``kronos/vendored.py`` imports the classes it builds them from), and the
vendored-source manifest (:data:`FORECASTING_FILES`); it goes into the forecast ``config_id``.
Whole files err on the safe side: an edit anywhere in them, even one that cannot change a fan
(the weights download, saving a window, the signal of a fan), gives a new ``config_id`` and costs
recomputation, never a wrong reuse. What changes a fan outside these files is covered by
:mod:`kronos_agent.data.forecast_store` instead: the conversion of candle strings into floats
(``data/store.py``) by the ``context_sha256`` every stored fan records and every reuse compares;
the weight files by their hashes; torch, numpy and pandas by their versions. Models changed in
memory after ``load_models`` returned them are refused only for what
:func:`~kronos_agent.kronos.weights.inference_problems` checks (clip, context length, device,
training mode, gradients); only ``load_models`` builds the models of stored fans.

The digest is reproducible without this package: for every file, in sorted order of its POSIX
path relative to the repository root, the line ``<path> NUL <sha256 hex of the bytes> LF`` is fed
to one sha256. Git is only asked read-only questions, with optional locks off so that nothing
under ``.git`` is written.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from .hashing import sha256_file
from .manifests import IGNORED_FILE_NAMES
from .paths import REPO_ROOT

SOURCE_GLOBS: tuple[str, ...] = (
    "src/kronos_agent/**/*.py",
    "params.yaml",
    "pyproject.toml",
    "uv.lock",
    "manifests/*",
)
# whole files, on purpose (see the module docstring); docs/DATA.md "Forecasts" lists them
FORECASTING_FILES: tuple[str, ...] = (
    "src/kronos_agent/kronos/forecast.py",
    "src/kronos_agent/kronos/features.py",
    "src/kronos_agent/kronos/decision.py",
    "src/kronos_agent/kronos/weights.py",
    "src/kronos_agent/kronos/vendored.py",
    "src/kronos_agent/data/windows.py",
    "src/kronos_agent/seeds.py",
    "src/kronos_agent/runtime.py",
    "manifests/kronos_source.sha256",
)
GIT_TIMEOUT_S = 30.0


class ProvenanceError(RuntimeError):
    """A file the code hash needs is missing, or a path is not a plain relative POSIX path."""


def _relative_posix(relpath: str) -> str:
    if not isinstance(relpath, str) or not relpath:
        raise ProvenanceError(f"code-hash paths are non-empty strings, got {relpath!r}")
    pure = PurePosixPath(relpath)
    if pure.is_absolute() or "\\" in relpath or ".." in pure.parts or pure.as_posix() != relpath:
        raise ProvenanceError(f"code-hash paths are normalised relative POSIX paths: {relpath!r}")
    return relpath


def files_sha256(relpaths: Iterable[str], root: Path = REPO_ROOT) -> str:
    """sha256 over ``path + NUL + sha256_hex(bytes) + LF`` for each path, in sorted order.

    Paths are relative POSIX paths under ``root``; duplicates count once. A missing file raises
    :class:`ProvenanceError` naming it.
    """
    digest = hashlib.sha256()
    for relpath in sorted({_relative_posix(item) for item in relpaths}):
        path = Path(root) / relpath
        if not path.is_file():
            raise ProvenanceError(f"file needed for the code hash is missing: {relpath}")
        digest.update(relpath.encode("utf-8") + b"\0" + sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def source_tree_files(root: Path = REPO_ROOT) -> list[str]:
    """Sorted relative paths matched by :data:`SOURCE_GLOBS` (``__pycache__`` and ``.DS_Store``
    excluded)."""
    root = Path(root)
    found: set[str] = set()
    for pattern in SOURCE_GLOBS:
        for path in root.glob(pattern):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.name in IGNORED_FILE_NAMES:
                continue
            if path.is_file():
                found.add(relative.as_posix())
    return sorted(found)


def source_tree_sha256(root: Path = REPO_ROOT) -> str:
    """The tree hash of :data:`SOURCE_GLOBS` under ``root`` (see the module docstring)."""
    return files_sha256(source_tree_files(root), root)


def forecasting_code_sha256(root: Path = REPO_ROOT) -> str:
    """Hash of the whole :data:`FORECASTING_FILES`; every one must exist (``ProvenanceError``
    otherwise)."""
    return files_sha256(FORECASTING_FILES, root)


def _git(root: Path, *args: str) -> str | None:
    """Stripped stdout of a read-only git command at ``root``; ``None`` when git fails."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def code_version() -> dict[str, Any]:
    """``{"commit": str | None, "dirty": bool | None, "source_tree_sha256": str}``.

    ``commit`` and ``dirty`` are ``None`` when the repository root is not a git work tree (or git
    is unavailable); ``dirty`` counts untracked files, so it is true while new step files are not
    committed. The tree hash is always computed.
    """
    root = REPO_ROOT
    commit: str | None = None
    dirty: bool | None = None
    toplevel = _git(root, "rev-parse", "--show-toplevel")
    if toplevel and Path(toplevel).resolve() == root.resolve():
        commit = _git(root, "rev-parse", "--verify", "HEAD") or None
        status = _git(root, "status", "--porcelain", "--untracked-files=normal")
        dirty = None if status is None else bool(status)
    return {"commit": commit, "dirty": dirty, "source_tree_sha256": source_tree_sha256(root)}


__all__ = [
    "FORECASTING_FILES",
    "SOURCE_GLOBS",
    "ProvenanceError",
    "code_version",
    "files_sha256",
    "forecasting_code_sha256",
    "source_tree_files",
    "source_tree_sha256",
]
