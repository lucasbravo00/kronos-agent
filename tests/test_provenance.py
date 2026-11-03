"""Code provenance: the tree hash identifies the code while the repository stays at one commit."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from kronos_agent import provenance
from kronos_agent.paths import REPO_ROOT
from kronos_agent.provenance import (
    FORECASTING_FILES,
    SOURCE_GLOBS,
    ProvenanceError,
    code_version,
    files_sha256,
    forecasting_code_sha256,
    source_tree_files,
    source_tree_sha256,
)

# sha256 over "a.txt\0<sha256(alpha)>\n" + "b/c.py\0<sha256(print(1)\n)>\n", computed by hand once
FILES_VECTOR = "2d925cca4481bef39187488e809578cb691c726df44fc78770e756bdbfdf7d77"

# Every kronos_agent module the fan path imports (forecast_window, load_models and context_frame,
# transitively, lazy imports included) and why a change in it cannot reuse a stored fan silently:
# hashed whole into config_id (FORECASTING_FILES) or covered otherwise (docs/DATA.md,
# "Forecasts"). A module new to the path fails the test below until it is classified here.
FAN_PATH_ENTRIES: tuple[str, ...] = (
    "kronos_agent.kronos.decision",
    "kronos_agent.kronos.weights",
    "kronos_agent.data.windows",
)
HASHED = "hashed whole into config_id"
FAN_PATH_COVERAGE: dict[str, str] = {
    "kronos_agent.kronos.decision": HASHED,
    "kronos_agent.kronos.features": HASHED,
    "kronos_agent.kronos.forecast": HASHED,
    "kronos_agent.data.windows": HASHED,
    "kronos_agent.runtime": HASHED,
    "kronos_agent.seeds": HASHED,
    "kronos_agent.kronos.weights": HASHED,
    "kronos_agent.kronos.vendored": HASHED,
    "kronos_agent.data.store": "the context floats: context_sha256, compared on every reuse",
    "kronos_agent.data.timeutil": "the timestamps: context_sha256; D and target opens on reuse",
    "kronos_agent.exchange.public_market": "the candle strings: window hashes, on every reuse",
    "kronos_agent.params": "config (every governed value)",
    "kronos_agent.models": "config (model, max_context)",
    "kronos_agent.paths": "config (platform label); file locations",
    "kronos_agent.data.partition": "refusals only",
    "kronos_agent.data.fsutil": "file writes only",
    "kronos_agent.hashing": "digests only",
    "kronos_agent.manifests": "verification only",
    "kronos_agent.provenance": "the code hash itself",
    "kronos_agent.finetune.registry": "a checkpoint's entry: its file hashes in config, "
    "verified by load_models itself",
    "kronos_agent.ledger": "refusals only (the sealed guard's order lookup)",
    "kronos_agent.envinfo": "refusals only (personal data in ledger text)",
}


def module_path(name: str, root: Path = REPO_ROOT) -> Path:
    """The source file of module or package ``name`` under ``root / "src"``."""
    path = root / "src" / Path(*name.split("."))
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def imported_modules(name: str, root: Path = REPO_ROOT) -> set[str]:
    """kronos_agent modules imported anywhere in the source of ``name`` (lazy imports and
    ``from . import module`` included)."""
    path = module_path(name, root)
    package = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            parts = package[: len(package) - node.level + 1] if node.level else []
            base = ".".join([*parts, *([node.module] if node.module else [])])
            for alias in node.names:
                submodule = f"{base}.{alias.name}"
                found.add(submodule if module_path(submodule, root).is_file() else base)
    return {item for item in found if item.split(".")[0] == "kronos_agent"}


def import_closure(entries: tuple[str, ...], root: Path = REPO_ROOT) -> set[str]:
    """The kronos_agent modules ``entries`` import, transitively; packages (whose ``__init__``
    runs on import) are followed but not listed."""
    seen: set[str] = set()
    todo = list(entries)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        parents = [name.rsplit(".", depth)[0] for depth in range(1, name.count(".") + 1)]
        todo.extend([*imported_modules(name, root), *parents])
    return {name for name in seen if module_path(name, root).name != "__init__.py"}


def write(root, relpath: str, data: bytes) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def expected_digest(entries: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relpath in sorted(entries):
        file_hash = hashlib.sha256(entries[relpath]).hexdigest()
        digest.update(relpath.encode() + b"\0" + file_hash.encode() + b"\n")
    return digest.hexdigest()


def test_files_sha256_vector_order_and_duplicates(tmp_path):
    entries = {"b/c.py": b"print(1)\n", "a.txt": b"alpha"}
    for relpath, data in entries.items():
        write(tmp_path, relpath, data)
    assert files_sha256(["b/c.py", "a.txt"], tmp_path) == FILES_VECTOR == expected_digest(entries)
    assert files_sha256(["a.txt", "b/c.py", "a.txt"], tmp_path) == FILES_VECTOR
    assert files_sha256([], tmp_path) == hashlib.sha256(b"").hexdigest()


def test_files_sha256_changes_with_content_and_with_names(tmp_path):
    write(tmp_path, "a.txt", b"alpha")
    write(tmp_path, "b.txt", b"alpha")
    reference = files_sha256(["a.txt"], tmp_path)
    assert files_sha256(["b.txt"], tmp_path) != reference  # same bytes, other name
    write(tmp_path, "a.txt", b"alphb")
    assert files_sha256(["a.txt"], tmp_path) != reference


@pytest.mark.parametrize("bad", ["/etc/hosts", "../x.py", "a/../b.py", "./a.py", "a\\b.py", ""])
def test_files_sha256_refuses_non_relative_paths(tmp_path, bad):
    with pytest.raises(ProvenanceError):
        files_sha256([bad], tmp_path)


def test_files_sha256_refuses_a_missing_file(tmp_path):
    with pytest.raises(ProvenanceError, match="missing: nope.py"):
        files_sha256(["nope.py"], tmp_path)


def test_source_tree_covers_exactly_the_documented_globs(tmp_path):
    assert SOURCE_GLOBS == (
        "src/kronos_agent/**/*.py",
        "params.yaml",
        "pyproject.toml",
        "uv.lock",
        "manifests/*",
    )
    included = {
        "src/kronos_agent/__init__.py": b"",
        "src/kronos_agent/data/store.py": b"x = 1\n",
        "src/kronos_agent/kronos/deep/inner.py": b"y = 2\n",
        "params.yaml": b"a: 1\n",
        "pyproject.toml": b"[project]\n",
        "uv.lock": b"lock\n",
        "manifests/weights.json": b"{}\n",
        "manifests/kronos_source.sha256": b"0  third_party/kronos/x\n",
    }
    excluded = {
        "src/kronos_agent/__pycache__/store.cpython-312.pyc": b"\0",
        "src/kronos_agent/data/__pycache__/helper.py": b"cached\n",
        "src/kronos_agent/data/notes.txt": b"notes\n",
        "manifests/.DS_Store": b"\0",
        "README.md": b"readme\n",
        "src/other/mod.py": b"z = 3\n",
        "tests/test_x.py": b"t = 4\n",
        "docs/DATA.md": b"design\n",
    }
    for relpath, data in {**included, **excluded}.items():
        write(tmp_path, relpath, data)
    assert source_tree_files(tmp_path) == sorted(included)
    assert source_tree_sha256(tmp_path) == expected_digest(included)
    for relpath in excluded:
        write(tmp_path, relpath, b"changed\n")
    assert source_tree_sha256(tmp_path) == expected_digest(included)
    write(tmp_path, "src/kronos_agent/data/store.py", b"x = 2\n")
    assert source_tree_sha256(tmp_path) != expected_digest(included)


def test_forecasting_code_sha256_needs_every_file(tmp_path):
    # whole files of the fan path's own modules, as documented, the construction of the models
    # included (kronos/weights.py, kronos/vendored.py: foundation-1); the string-to-float
    # conversion (data/store.py) is covered by the stored fans' context_sha256 instead
    assert FORECASTING_FILES == (
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
    entries = {relpath: relpath.encode() for relpath in FORECASTING_FILES}
    for relpath, data in entries.items():
        write(tmp_path, relpath, data)
    assert forecasting_code_sha256(tmp_path) == expected_digest(entries)
    (tmp_path / "src/kronos_agent/kronos/decision.py").unlink()
    with pytest.raises(ProvenanceError, match="decision.py"):
        forecasting_code_sha256(tmp_path)


def test_every_module_on_the_fan_path_is_hashed_or_covered_otherwise():
    """A change in a module a fan depends on gives a new config_id (hashed whole) or is caught
    otherwise (FAN_PATH_COVERAGE); a module that joins the path fails here until classified."""
    hashed = {name for name, how in FAN_PATH_COVERAGE.items() if how == HASHED}
    assert {module_path(name).relative_to(REPO_ROOT).as_posix() for name in hashed} == {
        relpath for relpath in FORECASTING_FILES if relpath.endswith(".py")
    }
    on_path = import_closure(FAN_PATH_ENTRIES)
    assert hashed | {"kronos_agent.data.store"} <= on_path
    unclassified = sorted(on_path - set(FAN_PATH_COVERAGE))
    assert not unclassified, f"classify in FAN_PATH_COVERAGE: {unclassified}"


def test_the_import_closure_sees_every_form_of_import(tmp_path):
    for relpath in ("__init__.py", "a/b.py", "a/c.py", "d.py", "e.py", "g.py"):
        write(tmp_path, f"src/kronos_agent/{relpath}", b"")
    write(tmp_path, "src/kronos_agent/a/__init__.py", b"from ..f import name\n")  # runs first
    write(tmp_path, "src/kronos_agent/f.py", b"import kronos_agent.g\n")
    write(
        tmp_path,
        "src/kronos_agent/a/m.py",
        b"import numpy\nfrom . import b\nfrom .c import name\nfrom kronos_agent import d\n"
        b"def lazy():\n    from ..e import other\n",
    )
    direct = {"kronos_agent.a.b", "kronos_agent.a.c", "kronos_agent.d", "kronos_agent.e"}
    assert imported_modules("kronos_agent.a.m", tmp_path) == direct
    assert import_closure(("kronos_agent.a.m",), tmp_path) == {
        "kronos_agent.a.m",
        *direct,
        "kronos_agent.f",
        "kronos_agent.g",
    }


def test_code_version_of_the_repository():
    version = code_version()
    assert set(version) == {"commit", "dirty", "source_tree_sha256"}
    assert version["source_tree_sha256"] == source_tree_sha256(REPO_ROOT)
    commit = version["commit"]
    assert commit is None or (len(commit) == 40 and set(commit) <= set("0123456789abcdef"))
    assert version["dirty"] in (None, True, False)
    assert "src/kronos_agent/provenance.py" in source_tree_files(REPO_ROOT)
    assert not any("__pycache__" in relpath for relpath in source_tree_files(REPO_ROOT))


def test_code_version_without_git(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda root, *args: None)
    version = code_version()
    assert version["commit"] is None and version["dirty"] is None
    assert version["source_tree_sha256"] == source_tree_sha256(REPO_ROOT)


def test_code_version_outside_a_work_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    version = code_version()
    assert version == {
        "commit": None,
        "dirty": None,
        "source_tree_sha256": hashlib.sha256(b"").hexdigest(),
    }
