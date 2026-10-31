"""Vendored Kronos: the manifest gate, where the import resolves, what it leaves on ``sys.path``,
that it never runs or writes bytecode caches, and the upstream API facts.

None of this needs weights; the import itself needs the verified ``third_party/kronos`` tree.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kronos_agent.kronos import vendored
from kronos_agent.paths import REPO_ROOT, vendored_kronos_dir

PREDICT_PARAMETERS = (
    "self",
    "df",
    "x_timestamp",
    "y_timestamp",
    "pred_len",
    "T",
    "top_k",
    "top_p",
    "sample_count",
    "verbose",
)
PREDICT_BATCH_PARAMETERS = (
    "self",
    "df_list",
    "x_timestamp_list",
    "y_timestamp_list",
    "pred_len",
    "T",
    "top_k",
    "top_p",
    "sample_count",
    "verbose",
)

FAILING_VERIFIER_SCRIPT = """
import json, sys
from kronos_agent import runtime  # exports HF_HUB_OFFLINE before anything heavy
from kronos_agent.manifests import VerifyReport
from kronos_agent.kronos import vendored

path_before = list(sys.path)
vendored.verify_vendored_tree = lambda **kwargs: VerifyReport(
    problems=["sha256 mismatch: third_party/kronos/model/kronos.py"]
)
outcome = "imported"
try:
    vendored.import_kronos()
except vendored.VendoredTreeError as exc:
    outcome = f"VendoredTreeError: {exc}"
print(json.dumps({
    "outcome": outcome,
    "model_imported": "model" in sys.modules,
    "torch_imported": "torch" in sys.modules,
    "sys_path_unchanged": sys.path == path_before,
    "root_in_sys_path": str(vendored.vendored_kronos_dir()) in sys.path,
}))
"""

SUCCESSFUL_IMPORT_SCRIPT = """
import json, sys
from importlib.machinery import PathFinder
from pathlib import Path
from kronos_agent import runtime  # exports HF_HUB_OFFLINE before anything heavy
from kronos_agent.kronos import vendored

root = vendored.vendored_kronos_dir().resolve()


def found_in_root(name):
    # what a spawn-started worker (same sys.path, empty sys.modules) would import by this name
    spec = PathFinder.find_spec(name)
    locations = [] if spec is None else [spec.origin, *(spec.submodule_search_locations or [])]
    return any(loc and Path(loc).resolve().is_relative_to(root) for loc in locations)


path_before = list(sys.path)
api = vendored.import_kronos()
top_level = sorted(p.name for p in root.iterdir() if p.is_dir())
print(json.dumps({
    "model_file": api.module.__file__,
    "model_bound": sys.modules.get("model") is api.module,
    "sys_path_unchanged": sys.path == path_before,
    "root_in_sys_path": str(vendored.vendored_kronos_dir()) in sys.path,
    "top_level": top_level,
    "importable_by_name": [name for name in top_level if found_in_root(name)],
}))
"""

FAILING_IMPORT_SCRIPT = """
import json, sys
from kronos_agent import runtime  # exports HF_HUB_OFFLINE before anything heavy
from kronos_agent.kronos import vendored

sys.modules["einops"] = None  # model/module.py then fails, after model/kronos.py appended "../"
path_before = list(sys.path)
outcome = "imported"
try:
    vendored.import_kronos()
except ImportError as exc:
    outcome = f"{type(exc).__name__}: {exc}"
print(json.dumps({
    "outcome": outcome,
    "model_imported": "model" in sys.modules,
    "sys_path_unchanged": sys.path == path_before,
}))
"""

# Both scripts import from a temporary copy of the verified tree, never from third_party/ itself.
NO_BYTECODE_SCRIPT = """
import json, shutil, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = False  # behave like a default interpreter, whatever the environment
sys.pycache_prefix = None
from kronos_agent import runtime  # exports HF_HUB_OFFLINE before anything heavy
from kronos_agent.kronos import vendored

base = Path(tempfile.mkdtemp())
try:
    copy = base / "kronos"
    shutil.copytree(vendored.vendored_kronos_dir(), copy, ignore=shutil.ignore_patterns("__pycache__"))
    vendored.vendored_kronos_dir = lambda: copy
    api = vendored.import_kronos()
    print(json.dumps({
        "model_in_copy": Path(api.module.__file__).resolve().is_relative_to(copy.resolve()),
        "caches": sorted(
            p.relative_to(copy).as_posix()
            for p in copy.rglob("*")
            if p.name == "__pycache__" or p.suffix in (".pyc", ".pyo")
        ),
        "dont_write_bytecode": sys.dont_write_bytecode,
    }))
finally:
    shutil.rmtree(base)
"""

PLANTED_BYTECODE_SCRIPT = """
import json, shutil, sys, tempfile
from pathlib import Path
from kronos_agent import runtime  # exports HF_HUB_OFFLINE before anything heavy
from kronos_agent.kronos import vendored

base = Path(tempfile.mkdtemp())
try:
    copy = base / "kronos"
    shutil.copytree(vendored.vendored_kronos_dir(), copy, ignore=shutil.ignore_patterns("__pycache__"))
    cache = copy / "model" / "__pycache__"
    cache.mkdir()
    (cache / f"kronos.{sys.implementation.cache_tag}.pyc").write_bytes(b"planted")
    vendored.vendored_kronos_dir = lambda: copy
    outcome = "imported"
    try:
        vendored.import_kronos()
    except vendored.VendoredTreeError as exc:
        outcome = f"VendoredTreeError: {exc}"
    print(json.dumps({
        "outcome": outcome,
        "cache_tag": sys.implementation.cache_tag,
        "model_imported": "model" in sys.modules,
        "torch_imported": "torch" in sys.modules,
    }))
finally:
    shutil.rmtree(base)
"""

PREFIXED_BYTECODE_SCRIPT = """
import importlib.util, json, marshal, shutil, sys, tempfile
from pathlib import Path
from kronos_agent import runtime  # exports HF_HUB_OFFLINE before anything heavy
from kronos_agent.kronos import vendored

base = Path(tempfile.mkdtemp())
try:
    copy = base / "kronos"
    shutil.copytree(vendored.vendored_kronos_dir(), copy, ignore=shutil.ignore_patterns("__pycache__"))
    prefix = str(base / "prefix")
    sys.pycache_prefix = prefix  # as PYTHONPYCACHEPREFIX would: caches live outside the tree
    source = copy / "model" / "kronos.py"
    stat = source.stat()
    # a VALID cache of model/kronos.py (matching mtime and size) whose code is not the source's
    planted = compile("raise RuntimeError('planted bytecode ran')", str(source), "exec")
    header = importlib.util.MAGIC_NUMBER + (0).to_bytes(4, "little")
    header += (int(stat.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
    header += (stat.st_size & 0xFFFFFFFF).to_bytes(4, "little")
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(parents=True)
    cache.write_bytes(header + marshal.dumps(planted))
    vendored.vendored_kronos_dir = lambda: copy
    outcome = "imported"
    try:
        vendored.import_kronos()
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    print(json.dumps({
        "outcome": outcome,
        "cache_under_prefix": cache.is_relative_to(base / "prefix"),
        "pycache_prefix_restored": sys.pycache_prefix == prefix,
    }))
finally:
    shutil.rmtree(base)
"""


def _run_python(code: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_manifest_failure_imports_nothing_in_a_fresh_process():
    outcome = _run_python(FAILING_VERIFIER_SCRIPT)
    assert outcome["outcome"].startswith("VendoredTreeError")
    assert "sha256 mismatch" in outcome["outcome"]
    assert outcome["model_imported"] is False
    assert outcome["torch_imported"] is False
    assert outcome["sys_path_unchanged"] is True
    assert outcome["root_in_sys_path"] is False


def test_successful_import_leaves_sys_path_unchanged_in_a_fresh_process():
    outcome = _run_python(SUCCESSFUL_IMPORT_SCRIPT)
    root = vendored_kronos_dir().resolve()
    assert Path(outcome["model_file"]).resolve().is_relative_to(root)
    assert outcome["model_bound"] is True
    assert outcome["sys_path_unchanged"] is True
    assert outcome["root_in_sys_path"] is False
    # neither ``model`` nor upstream's finetune/finetune_csv/tests stay importable by name
    assert {"model", "finetune", "finetune_csv", "tests"} <= set(outcome["top_level"])
    assert outcome["importable_by_name"] == []


def test_failed_import_still_restores_sys_path_in_a_fresh_process():
    outcome = _run_python(FAILING_IMPORT_SCRIPT)
    assert outcome["outcome"].startswith("ModuleNotFoundError")
    assert "einops" in outcome["outcome"]
    assert outcome["model_imported"] is False
    assert outcome["sys_path_unchanged"] is True


def test_import_writes_no_bytecode_into_the_vendored_tree():
    outcome = _run_python(NO_BYTECODE_SCRIPT)
    assert outcome["model_in_copy"] is True
    assert outcome["caches"] == []
    assert outcome["dont_write_bytecode"] is False  # restored after the import


def test_planted_bytecode_cache_is_refused_before_import():
    outcome = _run_python(PLANTED_BYTECODE_SCRIPT)
    planted = f"third_party/kronos/model/__pycache__/kronos.{outcome['cache_tag']}.pyc"
    assert outcome["outcome"] == f"VendoredTreeError: unlisted file: {planted}"
    assert outcome["model_imported"] is False
    assert outcome["torch_imported"] is False


def test_bytecode_under_a_pycache_prefix_is_never_read():
    outcome = _run_python(PREFIXED_BYTECODE_SCRIPT)
    assert outcome["cache_under_prefix"] is True  # the plant sits where the loader would look
    assert outcome["outcome"] == "imported"  # the source ran, not the planted code
    assert outcome["pycache_prefix_restored"] is True


@pytest.fixture(scope="module")
def api():
    return vendored.import_kronos()


def test_import_resolves_inside_the_vendored_tree(api):
    root = vendored_kronos_dir().resolve()
    assert Path(api.module.__file__).resolve().is_relative_to(root)
    for cls in (api.KronosTokenizer, api.Kronos, api.KronosPredictor):
        location = Path(sys.modules[cls.__module__].__file__).resolve()
        assert location.is_relative_to(root), cls
    assert sys.modules["model"] is api.module
    assert vendored.import_kronos() is api


def test_predict_signature_matches_the_pinned_commit(api):
    parameters = inspect.signature(api.KronosPredictor.predict).parameters
    assert tuple(parameters) == PREDICT_PARAMETERS
    assert parameters["T"].default == 1.0
    assert parameters["top_k"].default == 0
    assert parameters["top_p"].default == 0.9
    assert parameters["sample_count"].default == 1
    assert parameters["verbose"].default is True


def test_predict_batch_exists_with_the_batched_signature(api):
    assert callable(api.KronosPredictor.predict_batch)
    parameters = inspect.signature(api.KronosPredictor.predict_batch).parameters
    assert tuple(parameters) == PREDICT_BATCH_PARAMETERS
    assert parameters["sample_count"].default == 1


def test_predictor_defaults_max_context_512_and_auto_device(api):
    parameters = inspect.signature(api.KronosPredictor.__init__).parameters
    assert tuple(parameters) == ("self", "model", "tokenizer", "device", "max_context", "clip")
    assert parameters["max_context"].default == 512
    # ``device=None`` auto-selects CUDA/MPS when available: the loader always passes "cpu".
    assert parameters["device"].default is None
    assert parameters["clip"].default == 5


def test_sys_path_entries_removed_after_import(api):
    assert "../" not in sys.path
    assert str(vendored_kronos_dir()) not in sys.path
    assert sys.modules["model"] is api.module


def test_no_project_package_named_model(api):
    source_root = REPO_ROOT / "src" / "kronos_agent"
    offenders = [
        path
        for path in source_root.rglob("*")
        if path.name in ("model", "model.py") and "__pycache__" not in path.parts
    ]
    assert offenders == []
    location = Path(sys.modules["model"].__file__).resolve()
    assert location.is_relative_to(vendored_kronos_dir().resolve())
