"""Durable file writes and the network-job lock (``docs/DATA.md``, "Durable writes").

Every file is written to a unique temporary file in its own directory (``tempfile.mkstemp``),
flushed and ``fsync``ed, moved into place with ``os.replace`` and the directory is ``fsync``ed, so a
crash leaves either the old file or the new one, never a torn one. Files that must never be
overwritten (stored fans, saved windows, recorded study configs) are published with ``os.link``,
which fails when the name exists. Network jobs serialise on a non-blocking ``fcntl.flock``: a
second concurrent run is refused instead of waiting; readers of what a job writes take the same
lock shared (:func:`shared_lock`), so a read never overlaps a job that writes.

A process killed by a signal (SIGTERM, SIGKILL, the OOM killer) skips the clean-up of its
temporary file, so a killed write can leave one behind: :func:`temporary_files` lists them and
:func:`remove_temporary_files` removes them. Removing is safe only in a directory whose every
writer holds the same exclusive lock, and only while the caller holds it: no live writer can then
own one of the files. The caller proves it with the :class:`HeldLock` that :func:`exclusive_lock`
yields.

Published files get mode ``0o644`` (``mkstemp`` creates ``0o600``); nothing written through here is
secret. Missing parent directories are created, and each new directory entry is synced too.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

FILE_MODE = 0o644
# What ``tempfile.mkstemp(prefix=".<name>.", suffix=...)`` creates: the target's name behind a dot,
# 8 random characters, and ``.tmp`` (these writes, the store writer) or ``.part`` (downloads).
_TEMPORARY_NAME = re.compile(r"\.(?P<target>.+)\.[a-z0-9_]{8}\.(?:tmp|part)")


class LockBusyError(RuntimeError):
    """Another process (or another open handle) holds the exclusive lock."""


@dataclass
class HeldLock:
    """What :func:`exclusive_lock` yields: ``held`` is true until its ``with`` block exits.

    Operations that are safe only under a job lock (:func:`remove_temporary_files`) take it as
    proof that the caller holds the lock now, instead of asking whether anybody does.
    """

    path: Path
    held: bool = True


def fsync_directory(directory: Path) -> None:
    """``fsync`` a directory so that renames and new entries in it are durable."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(directory: Path) -> None:
    """Create ``directory`` and missing parents, syncing each parent that gains an entry."""
    directory = Path(directory)
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for path in reversed(missing):
        try:
            path.mkdir()
        except FileExistsError:
            if not path.is_dir():
                raise
            continue
        fsync_directory(path.parent)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")


def _write_temp(directory: Path, name: str, data: bytes) -> Path:
    """Write ``data`` to a new temporary file in ``directory`` (flushed, synced, mode 0o644)."""
    fd, temp_name = tempfile.mkstemp(dir=directory, prefix=f".{name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), FILE_MODE)
            os.fsync(handle.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` atomically and durably (temp, fsync, rename, dir fsync)."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes, got {type(data).__name__}")
    path = Path(path)
    ensure_directory(path.parent)
    temp = _write_temp(path.parent, path.name, bytes(data))
    try:
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    """:func:`atomic_write_bytes` of ``text`` encoded as UTF-8 (no newline translation)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    atomic_write_bytes(path, text.encode("utf-8"))


def publish_new(path: Path, data: bytes) -> None:
    """Publish ``data`` at ``path`` only if nothing is there yet (``FileExistsError`` otherwise).

    The bytes are written and synced under a temporary name, then hard-linked to ``path``
    (``os.link`` refuses an existing name atomically) and the temporary name is removed.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes, got {type(data).__name__}")
    path = Path(path)
    ensure_directory(path.parent)
    temp = _write_temp(path.parent, path.name, bytes(data))
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    fsync_directory(path.parent)


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[HeldLock]:
    """Hold ``fcntl.flock(LOCK_EX | LOCK_NB)`` on ``path`` for the duration of the block.

    The lock file is created if needed and never deleted (deleting it would let two runs lock
    two different files). A held lock raises :class:`LockBusyError` immediately; the lock is
    released when the block exits, also on an exception. Yields a :class:`HeldLock`.
    """
    path = Path(path)
    ensure_directory(path.parent)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, FILE_MODE)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockBusyError(
                f"{path.parent.name}/{path.name} is locked by another run; refusing to run "
                "concurrently"
            ) from None
        held = HeldLock(path)
        try:
            yield held
        finally:
            held.held = False
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextlib.contextmanager
def shared_lock(path: Path) -> Iterator[bool]:
    """Hold ``fcntl.flock(LOCK_SH | LOCK_NB)`` on ``path`` for the duration of the block.

    Readers take it on the lock that the writers of the same data hold with
    :func:`exclusive_lock`: any number of readers share it, none overlaps a writer, and a writer
    is refused while a reader holds it. A held exclusive lock raises :class:`LockBusyError`
    immediately. The lock file is never created here: every writer creates it before its first
    write and it is never deleted, so a missing one means that no job has ever written under
    it, and the block runs unlocked. Yields whether the lock is held.
    """
    path = Path(path)
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        fd = None
    if fd is None:  # yielded outside the except block, so the body's errors carry no context
        yield False
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LockBusyError(
                f"{path.parent.name}/{path.name} is locked by a run that writes this data; "
                "refusing to read it concurrently"
            ) from None
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def lock_is_held(path: Path) -> bool:
    """True when a run holds the exclusive lock ``path`` now (shared holders do not count)."""
    try:
        with shared_lock(path):
            return False
    except LockBusyError:
        return True


def temporary_files(directory: Path) -> list[Path]:
    """Temporary files of interrupted writes under ``directory`` (its subdirectories included).

    Only regular files named as :func:`tempfile.mkstemp` names them here, ``.<target>.<8 random
    characters>.tmp`` or ``.part``, are listed: never a lock file, a ``.DS_Store``, a symlink
    or a file in a symlinked directory. Sorted by path; a missing directory has none.
    """
    found: list[Path] = []
    for current, _, names in os.walk(directory):
        for name in names:
            path = Path(current) / name
            if _TEMPORARY_NAME.fullmatch(name) and path.is_file() and not path.is_symlink():
                found.append(path)
    return sorted(found)


def remove_temporary_files(directory: Path, lock: HeldLock) -> list[tuple[Path, int]]:
    """Remove :func:`temporary_files` under ``directory``; return ``(path, size)`` of each.

    ``lock`` is the :class:`HeldLock` of the exclusive lock that every writer of ``directory``
    holds while it writes, and it must be held now: no live writer can then own one of these
    files. A released (or any other) lock is refused with ``RuntimeError`` before anything is
    removed.
    """
    if not isinstance(lock, HeldLock) or not lock.held:
        raise RuntimeError(
            f"refusing to remove temporary files under {Path(directory).name}: the caller does "
            "not hold the lock of its writers"
        )
    removed: list[tuple[Path, int]] = []
    for path in temporary_files(directory):
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append((path, size))
    for parent in sorted({path.parent for path, _ in removed}):
        fsync_directory(parent)
    return removed


__all__ = [
    "FILE_MODE",
    "HeldLock",
    "LockBusyError",
    "atomic_write_bytes",
    "atomic_write_text",
    "ensure_directory",
    "exclusive_lock",
    "fsync_directory",
    "lock_is_held",
    "publish_new",
    "remove_temporary_files",
    "shared_lock",
    "temporary_files",
]
