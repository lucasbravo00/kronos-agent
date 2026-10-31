"""Import the vendored Kronos code only after verifying it against its manifest.

Upstream (``third_party/kronos/model/kronos.py``) appends the cwd-relative ``"../"`` to
``sys.path`` and imports itself as the top-level package ``model``; this module inserts the
vendored root at ``sys.path[0]`` instead, imports ``model`` and removes both entries again, also
when the import fails. Afterwards ``model`` resolves only through ``sys.modules``, never through
``sys.path``: a process that unpickles upstream objects without having imported them (a
spawn-started worker, for instance) must call :func:`import_kronos` first, which also verifies
the tree in that process.

The import compiles the verified sources and reads or writes no bytecode: a cached ``.pyc`` runs
in place of the source it was compiled from, so the strict manifest reports any bytecode cache in
the tree as an unlisted file and the import is refused, and ``sys.pycache_prefix`` (which would
make the loader read caches kept outside the tree) is switched off for the import.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ..hashing import git_blob_sha1_file
from ..manifests import ManifestMismatch, VerifyReport, read_manifest, verify_tree
from ..paths import manifests_dir, vendored_kronos_dir
from ..runtime import assert_hf_offline

UPSTREAM_URL = "https://github.com/shiyu-coder/Kronos"
UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
UPSTREAM_TREE = "2a7aada0cadce7cab6071b9bf4fdd55216dcb16e"
MANIFEST_PREFIX = "third_party/kronos/"
SOURCE_MANIFEST = "kronos_source.sha256"
SOURCE_PROVENANCE = "kronos_source.json"


class VendoredTreeError(RuntimeError):
    """The vendored tree is missing, altered or shadowed."""


@dataclass(frozen=True)
class KronosApi:
    KronosTokenizer: Any
    Kronos: Any
    KronosPredictor: Any
    module: ModuleType


_api: KronosApi | None = None


def source_manifest_path() -> Path:
    return manifests_dir() / SOURCE_MANIFEST


def provenance_path() -> Path:
    return manifests_dir() / SOURCE_PROVENANCE


def verify_vendored_tree(*, strict: bool = True, check_blob_ids: bool = False) -> VerifyReport:
    """sha256 manifest check of ``third_party/kronos`` (optionally git blob ids vs provenance)."""
    root = vendored_kronos_dir()
    manifest = source_manifest_path()
    if not manifest.is_file():
        return VerifyReport(problems=[f"manifest not found: {manifest.name}"])
    report = verify_tree(read_manifest(manifest), root=root, prefix=MANIFEST_PREFIX, strict=strict)
    if check_blob_ids and provenance_path().is_file():
        provenance = json.loads(provenance_path().read_text(encoding="utf-8"))
        for entry in provenance["files"]:
            if not entry.get("vendored"):
                continue
            path = root / entry["path"]
            if path.is_file() and git_blob_sha1_file(path) != entry["git_blob_sha1"]:
                report.problems.append(f"git blob id mismatch: {MANIFEST_PREFIX}{entry['path']}")
    return report


def import_kronos() -> KronosApi:
    """Verify the vendored tree, then import and return the upstream classes (cached)."""
    global _api
    if _api is not None:
        return _api
    assert_hf_offline()
    root = vendored_kronos_dir()
    report = verify_vendored_tree(strict=True)
    if not report.ok:
        raise VendoredTreeError(str(ManifestMismatch(report.problems)))
    existing = sys.modules.get("model")
    if existing is not None:
        location = getattr(existing, "__file__", None)
        if not location or not Path(location).resolve().is_relative_to(root.resolve()):
            raise VendoredTreeError("a foreign package named 'model' is already imported")
    root_str = str(root)
    inserted = root_str not in sys.path
    if inserted:
        sys.path.insert(0, root_str)
    # a .pyc would run instead of the verified source: the tree holds none (strict manifest), a
    # pycache prefix would point the loader at caches outside it, and nothing is written
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_pycache_prefix = sys.pycache_prefix
    sys.dont_write_bytecode = True
    sys.pycache_prefix = None
    try:
        import model
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        sys.pycache_prefix = previous_pycache_prefix
        # the bound modules stay in sys.modules; a root already on sys.path is left alone
        if inserted:
            while root_str in sys.path:
                sys.path.remove(root_str)
        while "../" in sys.path:
            sys.path.remove("../")
    location = Path(model.__file__).resolve()
    if not location.is_relative_to(root.resolve()):
        raise VendoredTreeError(f"'model' resolved outside the vendored tree: {location}")
    _api = KronosApi(
        KronosTokenizer=model.KronosTokenizer,
        Kronos=model.Kronos,
        KronosPredictor=model.KronosPredictor,
        module=model,
    )
    return _api
