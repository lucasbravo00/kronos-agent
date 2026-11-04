"""Repository hygiene (CLAUDE.md, "Secrets"): nothing personal, nothing secret, English only.

The scan walks every text file of the repository except ``.git``, ``.venv``, ``weights/``,
``data/``, ``third_party/`` (upstream code, verified by manifest), ``research/`` (another
session's gitignored working folder) and caches; ``reports/`` is included on purpose.
``.claude/settings.local.json`` is excluded because it is gitignored and personal by design. The
runtime identity (login name, hostname, home directory) is read at test time and never written
down; it is the environment report's (``kronos_agent.envinfo.personal_markers``), so a login or
host name that is the project's name or one of its words (a ``kronos`` service account)
identifies nobody and is skipped, while the home directory never is. Patterns this file has to mention are
assembled at runtime so the scanner stays clean under its own rules; IPv4 literals are tolerated
in this file only.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import socket
import subprocess
from pathlib import Path

import pytest

from kronos_agent.envinfo import IPV4_RE, personal_markers
from kronos_agent.manifests import read_manifest
from kronos_agent.paths import REPO_ROOT

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "weights",
    "data",
    "third_party",
    "research",  # another session's working folder (gitignored); never read here
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
}
EXCLUDED_FILES = {Path(".claude/settings.local.json")}
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".safetensors",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".zip",
    ".gz",
    ".tar",
    ".whl",
    ".so",
    ".dylib",
    ".pyc",
    ".pyo",
    ".woff",
    ".woff2",
    ".ttf",
}
BINARY_NAMES = {".DS_Store"}
SPEC = Path("docs/SPEC.md")
THIS_FILE = Path(__file__).resolve().relative_to(REPO_ROOT)
MANIFEST = Path("manifests/kronos_source.sha256")

# a e i o u (acute), n (tilde), their capitals, inverted question and exclamation marks; built from
# code points so that this file stays ASCII under its own rule
SPANISH_LETTERS = "".join(
    chr(code)
    for code in (
        0xE1,
        0xE9,
        0xED,
        0xF3,
        0xFA,
        0xF1,
        0xC1,
        0xC9,
        0xCD,
        0xD3,
        0xDA,
        0xD1,
        0xBF,
        0xA1,
    )
)
SPANISH_RE = re.compile(f"[{SPANISH_LETTERS}]")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CREDENTIAL_RE = re.compile("BINGX_API_" + r"(?:KEY|SECRET)\s*=\s*\S+")
PERSONAL_PATH_PREFIXES = ("/Users" + "/", "/home" + "/")
CLOUD_HOST_FRAGMENT = "amazon" + "aws"

REQUIRED_GITIGNORE = {
    ".env",
    ".env.*",
    "/weights/",
    "/data/",
    ".venv/",
    "*.safetensors",
    ".claude/settings.local.json",
    ".DS_Store",
    "/research/",
    "/docs/SPEC.md",
    "/docs/reference/",
}
REQUIRED_DENY = {
    "Read(.env)",
    "Read(.env.*)",
    "Edit(.env)",
    "Edit(.env.*)",
    "Edit(/third_party/**)",
    "Edit(/docs/SPEC.md)",
    "Bash(env)",
    "Bash(env *)",
    "Bash(printenv)",
    "Bash(printenv *)",
    "Bash(uv run --env-file*)",
    "Bash(uv run * --env-file*)",
}
REQUIRED_RUFF_EXCLUDES = {"third_party", "research"}
REQUIRED_TRACKED = {"CLAUDE.md", ".claude/settings.json", str(MANIFEST)}
# the owner's Spanish specification and its screenshot stay on the development machine (D-063)
NEVER_TRACKED = (str(SPEC), "docs/reference/")


def iter_text_files() -> list[tuple[Path, str]]:
    """(relative path, text) of every scannable text file, binaries and excluded trees skipped."""
    found: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = sorted(name for name in dirnames if name not in EXCLUDED_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = path.relative_to(REPO_ROOT)
            if rel in EXCLUDED_FILES or path.is_symlink() or not path.is_file():
                continue
            if name in BINARY_NAMES or path.suffix.lower() in BINARY_SUFFIXES:
                continue
            data = path.read_bytes()
            if b"\0" in data[:8192]:
                continue
            found.append((rel, data.decode("utf-8", errors="replace")))
    return found


@pytest.fixture(scope="module")
def corpus() -> dict[Path, str]:
    files = dict(iter_text_files())
    for required in (
        Path("CLAUDE.md"),
        Path("pyproject.toml"),
        Path(".claude/settings.json"),
        MANIFEST,
    ):
        assert required in files, f"{required} missing from the scan"
    assert THIS_FILE in files
    assert not any(rel.parts[0] == "third_party" for rel in files)
    return files


def locations(corpus: dict[Path, str], predicate, *, exempt: set[Path] = frozenset()) -> list[str]:
    """``file:line`` for every line satisfying ``predicate`` (the line itself is never printed)."""
    hits: list[str] = []
    for rel, text in corpus.items():
        if rel in exempt:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if predicate(line):
                hits.append(f"{rel}:{number}")
    return hits


def runtime_identity() -> dict[str, str]:
    """Values that must never be written into the repository, read from the running process."""
    return dict(personal_markers())


def identity_leaks(corpus: dict[Path, str]) -> dict[str, list[str]]:
    """``file:line`` hits per runtime identity kind (login name, hostname, home directory)."""
    leaks: dict[str, list[str]] = {}
    for kind, value in runtime_identity().items():
        exempt = {SPEC} if kind == "username" else set()  # the SPEC carries the owner's byline
        leaks[kind] = locations(corpus, lambda line, v=value: v in line, exempt=exempt)
    return leaks


def has_personal_path(line: str) -> bool:
    return any(prefix in line for prefix in PERSONAL_PATH_PREFIXES)


# --- configuration files -----------------------------------------------------------------------


def test_gitignore_has_the_required_entries():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert REQUIRED_GITIGNORE <= lines, sorted(REQUIRED_GITIGNORE - lines)
    assert "/research/strategy/" in lines  # another session's folder never enters the index


def test_ruff_never_touches_the_vendored_tree():
    """Hard rule 4: a repo-root ``ruff check --fix`` or ``ruff format`` must skip third_party/."""
    import tomllib

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excludes = set(config["tool"]["ruff"].get("extend-exclude", []))
    assert REQUIRED_RUFF_EXCLUDES <= excludes, sorted(REQUIRED_RUFF_EXCLUDES - excludes)


def test_claude_settings_deny_rules():
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    deny = set(settings["permissions"]["deny"])
    assert REQUIRED_DENY <= deny, sorted(REQUIRED_DENY - deny)


def test_no_project_module_is_named_model():
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "src").rglob("model*")
        if path.name == "model.py" or (path.is_dir() and path.name == "model")
    ]
    assert offenders == []


# --- content scan ------------------------------------------------------------------------------


def test_no_ipv4_literals(corpus):
    assert locations(corpus, IPV4_RE.search, exempt={THIS_FILE}) == []


def test_no_personal_paths(corpus):
    assert locations(corpus, has_personal_path) == []


def test_no_cloud_hostnames(corpus):
    assert locations(corpus, lambda line: CLOUD_HOST_FRAGMENT in line.lower()) == []


def test_no_email_addresses(corpus):
    assert locations(corpus, EMAIL_RE.search) == []


def test_no_runtime_identity(corpus):
    leaks = identity_leaks(corpus)
    assert "home directory" in leaks
    assert {kind: hits for kind, hits in leaks.items() if hits} == {}


def test_identity_scan_skips_project_words_but_still_catches_home_paths(monkeypatch, corpus):
    """A ``kronos`` service account on a ``kronos-agent`` host (SPEC section 15) must not turn the
    scan red; its home directory, other login names and personal path prefixes still do."""
    monkeypatch.setattr(getpass, "getuser", lambda: "kronos")
    monkeypatch.setattr(socket, "gethostname", lambda: "kronos-agent")
    assert locations(corpus, lambda line: "kronos" in line) != []  # the exemption is load-bearing
    assert set(runtime_identity()) == {"home directory"}
    assert {kind: hits for kind, hits in identity_leaks(corpus).items() if hits} == {}

    home = Path("/") / "home" / "kronos"  # assembled at runtime: no personal path literal here
    monkeypatch.setenv("HOME", str(home))
    synthetic = {
        Path("README.md"): "kronos-agent runs as kronos on the kronos-agent server",
        Path("notes.md"): f"cache in {home}/.cache",
    }
    assert identity_leaks(synthetic) == {"home directory": ["notes.md:1"]}
    assert locations(synthetic, has_personal_path) == ["notes.md:1"]

    monkeypatch.setattr(getpass, "getuser", lambda: "zq-synthetic-user")
    assert identity_leaks({Path("a.md"): "owner: zq-synthetic-user"}) == {
        "username": ["a.md:1"],
        "home directory": [],
    }


def test_no_spanish_characters_outside_the_spec(corpus):
    assert locations(corpus, SPANISH_RE.search, exempt={SPEC}) == []


def test_no_credential_assignments(corpus):
    assert locations(corpus, CREDENTIAL_RE.search) == []


# --- git index ---------------------------------------------------------------------------------


def git_ls_files(*args: str) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


@pytest.fixture(scope="module")
def tracked() -> list[str]:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("git repository not initialised yet (it is created at the end of step 1)")
    return git_ls_files()


def test_git_index_has_no_secrets_weights_or_data(tracked):
    offenders = [
        path
        for path in tracked
        if Path(path).name.startswith(".env")
        or path.startswith(("weights/", "data/"))
        or path.endswith(".safetensors")
        or path.startswith(NEVER_TRACKED)
    ]
    assert offenders == []


def test_git_index_has_the_governing_files(tracked):
    assert REQUIRED_TRACKED <= set(tracked), sorted(REQUIRED_TRACKED - set(tracked))


def test_git_tracks_exactly_the_vendored_files(tracked):
    vendored = git_ls_files("third_party/kronos")
    assert len(vendored) == len(read_manifest(REPO_ROOT / MANIFEST))
