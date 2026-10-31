"""``kronos-agent vendor-kronos``: copy the pinned Kronos commit into ``third_party/kronos``.

No git clone. The commit tarball is downloaded into an isolated temporary directory and every
extracted file is tied to the commit by its git blob sha1, compared with the tree that GitHub's
API reports for the pinned sha. The vendored subset (code, configs, tests, licence) is copied
into ``third_party/kronos``; ``manifests/kronos_source.json`` records all upstream blobs (with
``vendored: true/false``) and ``manifests/kronos_source.sha256`` lists the vendored files in GNU
``sha256sum`` format.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ..hashing import git_blob_sha1, sha256_bytes
from ..kronos.vendored import (
    MANIFEST_PREFIX,
    UPSTREAM_COMMIT,
    UPSTREAM_TREE,
    UPSTREAM_URL,
    provenance_path,
    source_manifest_path,
    verify_vendored_tree,
)
from ..manifests import ManifestEntry, write_manifest
from ..paths import vendored_kronos_dir

UPSTREAM_REPO = "shiyu-coder/Kronos"
TREE_URL = f"https://api.github.com/repos/{UPSTREAM_REPO}/git/trees/{UPSTREAM_TREE}?recursive=1"
TARBALL_URL = f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/{UPSTREAM_COMMIT}"
COMMIT_URL = f"https://api.github.com/repos/{UPSTREAM_REPO}/commits/{UPSTREAM_COMMIT}"

VENDORED_FILES = {"LICENSE", "README.md", "requirements.txt", ".gitignore"}
VENDORED_PREFIXES = ("model/", "finetune/", "finetune_csv/", "tests/")
EXCLUDED_PREFIXES = ("finetune_csv/data/", "finetune_csv/examples/")
VENDORING_RULE = (
    "LICENSE, README.md, requirements.txt, .gitignore and everything under model/, finetune/, "
    "finetune_csv/ (except finetune_csv/data/ and finetune_csv/examples/) and tests/; "
    "examples/, figures/ and webui/ are not vendored"
)


def is_vendored(path: str) -> bool:
    if path in VENDORED_FILES:
        return True
    if path.startswith(EXCLUDED_PREFIXES):
        return False
    return path.startswith(VENDORED_PREFIXES)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "vendor-kronos", help="vendor the pinned Kronos commit (or re-verify it)"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="only verify the existing vendored tree"
    )


def _get(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} for {url}")
    return response


def _safe_extract(archive: bytes, destination: Path) -> Path:
    """Extract the tarball, refusing links, devices, unknown types, absolute names and path
    traversal (each member before it is written); return the top directory."""
    top: str | None = None
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            parts = name.parts
            if (
                not parts
                or name.is_absolute()
                or parts[0] in ("", ".", "..")
                or any(p == ".." for p in parts)
            ):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsupported member type in archive: {member.name}")
            target = destination / name
            if not target.resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            if top is None:
                top = parts[0]
            elif parts[0] != top:
                raise RuntimeError("archive has more than one top-level directory")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)
    if top is None:
        raise RuntimeError("empty archive")
    return destination / top


def run(args: argparse.Namespace) -> int:
    root = vendored_kronos_dir()
    if args.verify_only:
        return _report(verify_vendored_tree(strict=True, check_blob_ids=True))
    if root.exists() and any(root.iterdir()):
        print(f"{root} already exists; verifying instead (delete it to re-vendor)")
        return _report(verify_vendored_tree(strict=True, check_blob_ids=True))

    with httpx.Client(
        follow_redirects=False,
        trust_env=False,
        timeout=httpx.Timeout(120.0, connect=15.0),
        headers={
            "User-Agent": "kronos-agent/0.1 (vendoring)",
            "Accept": "application/vnd.github+json",
        },
    ) as client:
        commit = _get(client, COMMIT_URL).json()
        if commit.get("sha") != UPSTREAM_COMMIT:
            raise RuntimeError(f"commit sha {commit.get('sha')} != expected {UPSTREAM_COMMIT}")
        if commit["commit"]["tree"]["sha"] != UPSTREAM_TREE:
            raise RuntimeError(
                f"commit tree {commit['commit']['tree']['sha']} != expected {UPSTREAM_TREE}"
            )
        commit_date = commit["commit"]["committer"]["date"]
        tree = _get(client, TREE_URL).json()
        if tree.get("truncated"):
            raise RuntimeError("GitHub tree listing is truncated")
        if tree.get("sha") != UPSTREAM_TREE:
            raise RuntimeError(f"tree sha {tree.get('sha')} != expected {UPSTREAM_TREE}")
        blobs = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
        archive = _get(client, TARBALL_URL).content

    with tempfile.TemporaryDirectory(prefix="kronos-vendor-") as tmp:
        top = _safe_extract(archive, Path(tmp))
        extracted = sorted(p.relative_to(top).as_posix() for p in top.rglob("*") if p.is_file())
        if set(extracted) != set(blobs):
            missing = sorted(set(blobs) - set(extracted))[:5]
            extra = sorted(set(extracted) - set(blobs))[:5]
            raise RuntimeError(f"archive/tree mismatch: missing {missing} extra {extra}")
        records = []
        for path in sorted(blobs):
            data = (top / path).read_bytes()
            blob = blobs[path]
            if git_blob_sha1(data) != blob["sha"] or len(data) != blob["size"]:
                raise RuntimeError(f"blob mismatch for {path}")
            records.append(
                {
                    "path": path,
                    "size": blob["size"],
                    "git_blob_sha1": blob["sha"],
                    "sha256": sha256_bytes(data),
                    "vendored": is_vendored(path),
                }
            )
        root.mkdir(parents=True, exist_ok=True)
        entries = []
        for record in records:
            if not record["vendored"]:
                continue
            source = top / record["path"]
            target = root / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            entries.append(ManifestEntry(record["sha256"], MANIFEST_PREFIX + record["path"]))

    provenance = {
        "upstream": UPSTREAM_URL,
        "commit": UPSTREAM_COMMIT,
        "tree": UPSTREAM_TREE,
        "commit_date": commit_date,
        "license": "MIT",
        "fetched_on": datetime.now(UTC).strftime("%Y-%m-%d"),
        "vendoring_rule": VENDORING_RULE,
        "blob_count": len(records),
        "total_bytes": sum(r["size"] for r in records),
        "vendored_count": sum(1 for r in records if r["vendored"]),
        "vendored_bytes": sum(r["size"] for r in records if r["vendored"]),
        "files": records,
    }
    provenance_path().parent.mkdir(parents=True, exist_ok=True)
    provenance_path().write_text(
        json.dumps(provenance, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    write_manifest(source_manifest_path(), entries)
    print(
        f"vendored {provenance['vendored_count']} of {provenance['blob_count']} files "
        f"({provenance['vendored_bytes']:,} of {provenance['total_bytes']:,} bytes) from {UPSTREAM_COMMIT}"
    )
    return _report(verify_vendored_tree(strict=True, check_blob_ids=True))


def _report(report) -> int:
    if report.ok:
        print(f"third_party/kronos: OK ({report.checked} files verified)")
        return 0
    for problem in report.problems:
        print(f"PROBLEM: {problem}")
    return 1
