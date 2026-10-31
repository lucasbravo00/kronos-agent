"""GNU sha256sum manifests, the vendored Kronos manifest and the pre-registered weights."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kronos_agent.hashing import git_blob_sha1, git_blob_sha1_file, sha256_bytes, sha256_file
from kronos_agent.kronos.vendored import (
    MANIFEST_PREFIX,
    UPSTREAM_COMMIT,
    provenance_path,
    source_manifest_path,
    verify_vendored_tree,
)
from kronos_agent.kronos.weights import (
    load_registry,
    registry_lines,
    registry_path,
    repos_for_model,
)
from kronos_agent.manifests import (
    ManifestEntry,
    ManifestMismatch,
    build_entries,
    format_manifest,
    is_ignored,
    list_files,
    parse_manifest,
    read_manifest,
    verify_tree,
    write_manifest,
)
from kronos_agent.paths import REPO_ROOT, manifests_dir, vendored_kronos_dir, weights_dir

VENDORED_FILE_COUNT = 27
UPSTREAM_BLOB_COUNT = 92
SHA_A, SHA_B, SHA_C = "a" * 64, "b" * 64, "c" * 64

# Hugging Face API identities (plan appendix A, read on 2026-10-29): policy, revision, files.
EXPECTED_REGISTRY = {
    "NeoQuasar/Kronos-Tokenizer-base": (
        "download_allowed",
        "0e0117387f39004a9016484a186a908917e22426",
        {
            "config.json": (
                301,
                "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
            ),
            "model.safetensors": (
                15842368,
                "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
            ),
        },
    ),
    "NeoQuasar/Kronos-small": (
        "download_allowed",
        "901c26c1332695a2a8f243eb2f37243a37bea320",
        {
            "config.json": (
                228,
                "5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396",
            ),
            "model.safetensors": (
                98980656,
                "b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020",
            ),
        },
    ),
    "NeoQuasar/Kronos-base": (
        "needs_owner_approval",
        "2b554741eca47781b64468546e77fef3e85130e6",
        {
            "model.safetensors": (
                409264008,
                "abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83",
            ),
        },
    ),
    "NeoQuasar/Kronos-mini": (
        "needs_owner_approval",
        "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        {
            "model.safetensors": (
                16440776,
                "a7d5f37e2e9fbd9891f7d7d4f72574512dd1f704fee14223e0a8cd0fbf54197c",
            ),
        },
    ),
    "NeoQuasar/Kronos-Tokenizer-2k": (
        "needs_owner_approval",
        "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        {
            "model.safetensors": (
                15842376,
                "b97ec46b3b72160509e289183eaf7bdf5f0dac5bb9b49522f6d46638a99a8717",
            ),
        },
    ),
}
VENDORED_BLOB_IDS = {
    "model/kronos.py": "ce4494ee0b3ec8751b09d5488c93bde995e008e0",
    "model/module.py": "f2a05158b48a56e9235426f6583e384360d761f9",
    "model/__init__.py": "718d07a21b53b7eff4a6564e6dfcae8ee7e8c6b1",
}


# --- format and parse -------------------------------------------------------------------------


def test_format_and_parse_round_trip_exactly():
    entries = [
        ManifestEntry(SHA_B, "tree/b.txt"),
        ManifestEntry(SHA_A, "tree/a.txt"),
        ManifestEntry(SHA_C, "tree/sub/c.bin"),
    ]
    text = format_manifest(entries)
    assert text == f"{SHA_A}  tree/a.txt\n{SHA_B}  tree/b.txt\n{SHA_C}  tree/sub/c.bin\n"
    assert parse_manifest(text) == sorted(entries, key=lambda entry: entry.path)
    assert format_manifest(parse_manifest(text)) == text
    assert parse_manifest("") == []


@pytest.mark.parametrize(
    "text, message",
    [
        (f"{SHA_B}  z.txt\n{SHA_A}  a.txt\n", "not sorted"),
        (f"{SHA_A}  a.txt\n{SHA_B}  a.txt\n", "duplicate"),
        (f"{SHA_A} a.txt\n", "not a GNU sha256sum line"),
        (f"{SHA_A.upper()}  a.txt\n", "not a GNU sha256sum line"),
        (f"{SHA_A[:-1]}  a.txt\n", "not a GNU sha256sum line"),
        (f"{SHA_A}  \n", "not a GNU sha256sum line"),
        ("not a manifest\n", "line 1: not a GNU sha256sum line"),
    ],
)
def test_parse_rejects_malformed_manifests(text, message):
    with pytest.raises(ValueError, match=message):
        parse_manifest(text)


def test_write_and_read_manifest(tmp_path):
    entries = [ManifestEntry(SHA_A, "x/a"), ManifestEntry(SHA_B, "x/b")]
    path = tmp_path / "x.sha256"
    write_manifest(path, reversed(entries))
    assert path.read_text(encoding="ascii") == format_manifest(entries)
    assert read_manifest(path) == entries


def test_manifest_mismatch_message_is_truncated():
    problems = [f"missing: file{index}" for index in range(7)]
    exc = ManifestMismatch(problems)
    assert exc.problems == problems
    assert "missing: file4" in str(exc)
    assert "missing: file5" not in str(exc)
    assert str(exc).endswith("; ... (7 problems)")
    assert str(ManifestMismatch(["only one"])) == "only one"


# --- verify_tree on a temporary tree -----------------------------------------------------------


@pytest.fixture
def tree(tmp_path) -> tuple[Path, list[ManifestEntry]]:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"alpha\n")
    (root / "sub" / "b.bin").write_bytes(bytes(range(256)))
    return root, build_entries(root, "tree/", list_files(root))


def test_verify_tree_ok(tree):
    root, entries = tree
    assert [entry.path for entry in entries] == ["tree/a.txt", "tree/sub/b.bin"]
    assert entries[0].sha256 == sha256_bytes(b"alpha\n")
    report = verify_tree(entries, root=root, prefix="tree/")
    assert report.ok and report.checked == 2 and report.problems == []
    report.raise_if_problems()


def test_verify_tree_reports_byte_flip(tree):
    root, entries = tree
    data = bytearray((root / "a.txt").read_bytes())
    data[0] ^= 0x01
    (root / "a.txt").write_bytes(bytes(data))
    report = verify_tree(entries, root=root, prefix="tree/")
    assert report.problems == ["sha256 mismatch: tree/a.txt"]
    assert report.checked == 2
    with pytest.raises(ManifestMismatch, match="sha256 mismatch: tree/a.txt"):
        report.raise_if_problems()


def test_verify_tree_reports_missing_file(tree):
    root, entries = tree
    (root / "sub" / "b.bin").unlink()
    report = verify_tree(entries, root=root, prefix="tree/")
    assert report.problems == ["missing: tree/sub/b.bin"]
    assert report.checked == 1


def test_verify_tree_reports_extra_file_only_when_strict(tree):
    root, entries = tree
    (root / "sub" / "extra.txt").write_bytes(b"unexpected")
    assert verify_tree(entries, root=root, prefix="tree/").problems == [
        "unlisted file: tree/sub/extra.txt"
    ]
    assert verify_tree(entries, root=root, prefix="tree/", strict=False).ok


def test_verify_tree_reports_symlinks(tree):
    root, entries = tree
    (root / "link.txt").symlink_to(root / "a.txt")
    assert verify_tree(entries, root=root, prefix="tree/").problems == ["symlink: tree/link.txt"]
    (root / "a.txt").unlink()
    (root / "a.txt").symlink_to(root / "sub" / "b.bin")
    assert verify_tree(entries, root=root, prefix="tree/").problems == [
        "symlink instead of file: tree/a.txt",
        "symlink: tree/link.txt",
    ]


def test_verify_tree_needs_entries_for_the_prefix(tree):
    root, entries = tree
    report = verify_tree(entries, root=root, prefix="other/")
    assert report.problems == ["no manifest entries with prefix 'other/'"]


def test_only_ds_store_is_ignored_bytecode_caches_are_reported(tree):
    # a cached .pyc runs in place of the hashed source, so it is an unlisted file like any other
    root, entries = tree
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    (root / "sub" / "b.pyc").write_bytes(b"\0")
    (root / "sub" / "c.pyo").write_bytes(b"\0")
    (root / ".DS_Store").write_bytes(b"\0")
    (root / "sub" / ".DS_Store").write_bytes(b"\0")
    assert list_files(root) == [
        "__pycache__/a.cpython-312.pyc",
        "a.txt",
        "sub/b.bin",
        "sub/b.pyc",
        "sub/c.pyo",
    ]
    assert verify_tree(entries, root=root, prefix="tree/").problems == [
        "unlisted file: tree/__pycache__/a.cpython-312.pyc",
        "unlisted file: tree/sub/b.pyc",
        "unlisted file: tree/sub/c.pyo",
    ]
    assert verify_tree(entries, root=root, prefix="tree/", strict=False).ok
    assert is_ignored(".DS_Store")
    assert is_ignored("a/.DS_Store")
    assert not is_ignored("pkg/__pycache__/x.cpython-312.pyc")
    assert not is_ignored("x.pyc")
    assert not is_ignored("x.pyo")
    assert not is_ignored("a/.DS_Store.txt")


# --- the vendored Kronos tree ------------------------------------------------------------------


def test_vendored_manifest_verifies_strictly_with_blob_ids():
    report = verify_vendored_tree(strict=True, check_blob_ids=True)
    assert report.ok, report.problems
    assert report.checked == VENDORED_FILE_COUNT
    entries = read_manifest(source_manifest_path())
    assert len(entries) == VENDORED_FILE_COUNT
    assert all(entry.path.startswith(MANIFEST_PREFIX) for entry in entries)
    provenance = json.loads(provenance_path().read_text(encoding="utf-8"))
    assert provenance["commit"] == UPSTREAM_COMMIT
    assert provenance["blob_count"] == UPSTREAM_BLOB_COUNT == len(provenance["files"])
    assert provenance["vendored_count"] == VENDORED_FILE_COUNT
    vendored = sorted(MANIFEST_PREFIX + f["path"] for f in provenance["files"] if f["vendored"])
    assert vendored == [entry.path for entry in entries]
    by_path = {MANIFEST_PREFIX + f["path"]: f["sha256"] for f in provenance["files"]}
    assert all(by_path[entry.path] == entry.sha256 for entry in entries)


@pytest.mark.parametrize("relative, blob_id", sorted(VENDORED_BLOB_IDS.items()))
def test_vendored_git_blob_ids_match_the_pinned_tree(relative, blob_id):
    assert git_blob_sha1_file(vendored_kronos_dir() / relative) == blob_id


def test_hash_vectors():
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_file(
        vendored_kronos_dir() / "finetune" / "utils" / "__init__.py"
    ) == sha256_bytes(b"")


@pytest.mark.parametrize(
    "tool, options",
    [("shasum", ["-a", "256", "-c"]), ("sha256sum", ["-c"])],
)
def test_coreutils_verify_the_vendored_manifest(tool, options):
    executable = shutil.which(tool)
    if executable is None:
        pytest.skip(f"{tool} is not installed")
    completed = subprocess.run(
        [executable, *options, "manifests/kronos_source.sha256"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count(": OK") == VENDORED_FILE_COUNT


# --- weights manifest and registry -------------------------------------------------------------


def test_weights_manifest_parses_and_present_files_match():
    entries = read_manifest(manifests_dir() / "weights.sha256")
    assert len(entries) == 7
    assert all(entry.path.startswith("weights/") for entry in entries)
    by_path = sorted(registry_lines(), key=lambda line: line.split("  ", 1)[1])
    assert by_path == [f"{entry.sha256}  {entry.path}" for entry in entries]
    for entry in entries:
        path = weights_dir() / entry.path.removeprefix("weights/")
        if path.is_file():  # absent weights are allowed (another machine, gated repositories)
            assert sha256_file(path) == entry.sha256, entry.path


def test_weights_registry_identities_and_gate_policy():
    registry = load_registry()
    assert set(registry) == set(EXPECTED_REGISTRY)
    for repo_id, (policy, revision, files) in EXPECTED_REGISTRY.items():
        repo = registry[repo_id]
        assert repo.policy == policy, repo_id
        assert repo.revision == revision, repo_id
        assert re.fullmatch(r"[0-9a-f]{40}", repo.revision)
        for name, (size, digest) in files.items():
            item = repo.file(name)
            assert (item.size, item.sha256) == (size, digest), f"{repo_id}/{name}"
    raw = json.loads(registry_path().read_text(encoding="utf-8"))
    assert len(raw["repos"]) == 5
    gated = {repo_id for repo_id, repo in registry.items() if repo.policy == "needs_owner_approval"}
    assert gated == {
        "NeoQuasar/Kronos-base",
        "NeoQuasar/Kronos-mini",
        "NeoQuasar/Kronos-Tokenizer-2k",
    }


def test_model_tokenizer_pairing_comes_from_the_registry():
    small, small_tokenizer = repos_for_model("small")
    assert (small.repo_id, small_tokenizer.repo_id) == (
        "NeoQuasar/Kronos-small",
        "NeoQuasar/Kronos-Tokenizer-base",
    )
    base, base_tokenizer = repos_for_model("base")
    assert (base.repo_id, base_tokenizer.repo_id) == (
        "NeoQuasar/Kronos-base",
        "NeoQuasar/Kronos-Tokenizer-base",
    )
    mini, mini_tokenizer = repos_for_model("mini")
    assert (mini.repo_id, mini_tokenizer.repo_id) == (
        "NeoQuasar/Kronos-mini",
        "NeoQuasar/Kronos-Tokenizer-2k",
    )
    with pytest.raises(KeyError, match="unknown Kronos model"):
        repos_for_model("large")
