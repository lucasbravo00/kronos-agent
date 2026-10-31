"""Hashing helpers shared by manifests, ledgers and candle-window fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """Git object id of a blob: ``sha1(b"blob <size>\\0" + data)``."""
    digest = hashlib.sha1(b"blob %d\0" % len(data))
    digest.update(data)
    return digest.hexdigest()


def git_blob_sha1_file(path: Path | str) -> str:
    return git_blob_sha1(Path(path).read_bytes())


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding (sorted keys, no whitespace, ASCII only)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
