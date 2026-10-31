"""GNU ``sha256sum``-format manifests.

One line per file: ``<64 lowercase hex><two spaces><path>\\n``. Paths are POSIX, relative to the
repository root and sorted, so ``shasum -a 256 -c manifests/<name>.sha256`` (macOS) or
``sha256sum -c`` (Linux), run from the repository root, verifies a manifest without this package.

Verification is strict by default: a listed file that is missing or altered, an unlisted file and
a symlink are all reported. Bytecode caches (``__pycache__``, ``*.pyc``) are unlisted files like
any other, because the interpreter runs them in place of the hashed sources (``import_kronos``
never writes them); only ``.DS_Store`` is ignored.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import sha256_file

LINE_RE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")
IGNORED_FILE_NAMES = frozenset({".DS_Store"})


class ManifestMismatch(Exception):
    """Raised when a manifest verification finds problems."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        shown = "; ".join(self.problems[:5])
        if len(self.problems) > 5:
            shown += f"; ... ({len(self.problems)} problems)"
        super().__init__(shown)


@dataclass(frozen=True)
class ManifestEntry:
    sha256: str
    path: str


@dataclass
class VerifyReport:
    checked: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def raise_if_problems(self) -> None:
        if self.problems:
            raise ManifestMismatch(self.problems)


def is_ignored(rel_path: str) -> bool:
    return rel_path.rsplit("/", 1)[-1] in IGNORED_FILE_NAMES


def list_files(root: Path) -> list[str]:
    """Regular files under ``root`` as sorted POSIX paths relative to ``root`` (``.DS_Store``
    ignored)."""
    found: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if not is_ignored(rel):
            found.append(rel)
    return sorted(found)


def list_symlinks(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_symlink())


def parse_manifest(text: str) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for number, line in enumerate(text.splitlines(), 1):
        match = LINE_RE.match(line)
        if not match:
            raise ValueError(f"line {number}: not a GNU sha256sum line")
        entries.append(ManifestEntry(match.group(1), match.group(2)))
    paths = [entry.path for entry in entries]
    if paths != sorted(paths):
        raise ValueError("manifest paths are not sorted")
    if len(set(paths)) != len(paths):
        raise ValueError("manifest contains duplicate paths")
    return entries


def format_manifest(entries: Iterable[ManifestEntry]) -> str:
    ordered = sorted(entries, key=lambda entry: entry.path)
    return "".join(f"{entry.sha256}  {entry.path}\n" for entry in ordered)


def read_manifest(path: Path | str) -> list[ManifestEntry]:
    return parse_manifest(Path(path).read_text(encoding="ascii"))


def write_manifest(path: Path | str, entries: Iterable[ManifestEntry]) -> None:
    Path(path).write_text(format_manifest(entries), encoding="ascii")


def build_entries(root: Path, prefix: str, rel_files: Iterable[str]) -> list[ManifestEntry]:
    """Entries for ``rel_files`` under ``root``; manifest paths are ``prefix + rel``."""
    entries = [ManifestEntry(sha256_file(root / rel), prefix + rel) for rel in rel_files]
    return sorted(entries, key=lambda entry: entry.path)


def verify_tree(
    entries: Iterable[ManifestEntry], *, root: Path, prefix: str, strict: bool = True
) -> VerifyReport:
    """Verify the files under ``root`` against the entries whose manifest path starts with ``prefix``.

    Every selected entry must exist with the recorded sha256. With ``strict`` (default) every
    regular file under ``root`` must be listed and no symlink may exist.
    """
    report = VerifyReport()
    expected = {e.path[len(prefix) :]: e.sha256 for e in entries if e.path.startswith(prefix)}
    if not expected:
        report.problems.append(f"no manifest entries with prefix {prefix!r}")
        return report
    for rel, digest in sorted(expected.items()):
        path = root / rel
        if path.is_symlink():
            report.problems.append(f"symlink instead of file: {prefix}{rel}")
            continue
        if not path.is_file():
            report.problems.append(f"missing: {prefix}{rel}")
            continue
        report.checked += 1
        if sha256_file(path) != digest:
            report.problems.append(f"sha256 mismatch: {prefix}{rel}")
    if strict and root.is_dir():
        for rel in list_files(root):
            if rel not in expected:
                report.problems.append(f"unlisted file: {prefix}{rel}")
        for rel in list_symlinks(root):
            if rel not in expected:
                report.problems.append(f"symlink: {prefix}{rel}")
    return report
