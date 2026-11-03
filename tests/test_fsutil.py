"""Durable writes (temp file, fsync, rename, directory fsync), publish-once and the job lock."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import textwrap

import pytest

from kronos_agent.data.fsutil import (
    HeldLock,
    LockBusyError,
    atomic_write_bytes,
    atomic_write_text,
    exclusive_lock,
    lock_is_held,
    publish_new,
    remove_temporary_files,
    shared_lock,
    temporary_files,
)

LOCK_PROBE = textwrap.dedent(
    """
    import sys
    from kronos_agent.data.fsutil import LockBusyError, exclusive_lock, shared_lock
    lock = exclusive_lock if sys.argv[2] == "exclusive" else shared_lock
    try:
        with lock(sys.argv[1]):
            pass
    except LockBusyError:
        sys.exit(3)
    sys.exit(0)
    """
)


def leftovers(directory) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.endswith(".tmp"))


def test_atomic_write_creates_replaces_and_leaves_no_temp(tmp_path):
    target = tmp_path / "a" / "b" / "file.json"
    atomic_write_bytes(target, b"one")
    assert target.read_bytes() == b"one"
    atomic_write_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert leftovers(target.parent) == []
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    atomic_write_text(target, "line\r\nnext\n")
    assert target.read_bytes() == b"line\r\nnext\n"  # no newline translation
    with pytest.raises(TypeError):
        atomic_write_bytes(target, "text")
    with pytest.raises(TypeError):
        atomic_write_text(target, b"bytes")


def test_atomic_write_syncs_the_file_before_the_rename_and_the_directory_after(
    tmp_path, monkeypatch
):
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def recording_fsync(fd):
        events.append("fsync-dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync-file")
        real_fsync(fd)

    def recording_replace(src, dst):
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    atomic_write_bytes(tmp_path / "x.bin", b"data")
    assert events == ["fsync-file", "replace", "fsync-dir"]


def test_a_failed_rename_keeps_the_old_file_and_removes_the_temp(tmp_path, monkeypatch):
    target = tmp_path / "keep.txt"
    atomic_write_bytes(target, b"old")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert leftovers(tmp_path) == []


def test_new_parent_directories_are_synced(tmp_path, monkeypatch):
    synced_dirs: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced_dirs.append("dir")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    atomic_write_bytes(tmp_path / "x" / "y" / "z.bin", b"data")
    # tmp_path gained x, x gained y, y gained z.bin (after the rename)
    assert synced_dirs == ["dir", "dir", "dir"]


def test_publish_new_never_overwrites(tmp_path):
    target = tmp_path / "forecasts" / "fan.npz"
    publish_new(target, b"first")
    assert target.read_bytes() == b"first"
    with pytest.raises(FileExistsError):
        publish_new(target, b"second")
    assert target.read_bytes() == b"first"
    assert leftovers(target.parent) == []
    assert target.stat().st_nlink == 1  # the temporary name is gone
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_exclusive_lock_refuses_a_second_holder_and_releases(tmp_path):
    lock = tmp_path / "raw" / "bingx" / ".lock"
    with exclusive_lock(lock):
        assert lock.exists()
        with pytest.raises(LockBusyError, match="bingx/.lock"), exclusive_lock(lock):
            pass
    with exclusive_lock(lock):
        pass
    assert lock.exists()  # never deleted


def test_exclusive_lock_is_released_when_the_block_raises(tmp_path):
    lock = tmp_path / ".lock"
    with pytest.raises(RuntimeError), exclusive_lock(lock):
        raise RuntimeError("job failed")
    with exclusive_lock(lock):
        pass


def probe(lock, mode: str) -> int:
    """Exit code of another process taking ``lock`` in ``mode``: 0 taken, 3 refused."""
    command = [sys.executable, "-B", "-c", LOCK_PROBE, str(lock), mode]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode in (0, 3), completed.stderr
    return completed.returncode


def test_exclusive_lock_refuses_another_process(tmp_path):
    lock = tmp_path / ".lock"
    with exclusive_lock(lock):
        assert probe(lock, "exclusive") == 3
    assert probe(lock, "exclusive") == 0


def test_exclusive_lock_yields_the_proof_that_it_is_held(tmp_path):
    lock = tmp_path / "raw" / "bingx" / ".lock"
    with exclusive_lock(lock) as held:
        assert isinstance(held, HeldLock) and held.path == lock and held.held
    assert not held.held


def test_readers_share_the_lock_and_exclude_writers_across_processes(tmp_path):
    lock = tmp_path / ".lock"
    with shared_lock(lock) as locked:  # the lock file does not exist: no job ever wrote
        assert locked is False and probe(lock, "exclusive") == 0
    assert lock.exists()  # created by the writer probe, never by a reader
    with shared_lock(lock) as locked:
        assert locked is True
        assert probe(lock, "shared") == 0  # readers share it
        assert probe(lock, "exclusive") == 3  # a writer is refused while one reads
        assert not lock_is_held(lock)  # shared holders do not count as a run
        with pytest.raises(LockBusyError), exclusive_lock(lock):
            pass
    with exclusive_lock(lock):
        assert probe(lock, "shared") == 3  # no reader while a writer holds it
        assert lock_is_held(lock)
        with (
            pytest.raises(LockBusyError, match="refusing to read it concurrently"),
            shared_lock(lock),
        ):
            pass
    assert not lock_is_held(lock) and not lock_is_held(tmp_path / "missing" / ".lock")


def test_temporary_files_are_the_names_mkstemp_gives_them(tmp_path):
    directory = tmp_path / "candles"
    (directory / "sub").mkdir(parents=True)
    expected = []
    for folder, prefix, suffix in (
        (directory, ".1m.csv.", ".tmp"),
        (directory / "sub", ".BTCUSDT-1m-2020-01.zip.", ".part"),
    ):
        fd, name = tempfile.mkstemp(dir=folder, prefix=prefix, suffix=suffix)
        os.close(fd)
        expected.append(name)
    for name in (".lock", ".DS_Store", "1m.csv", "1m.csv.abcdefgh.tmp", ".1m.csv.abcdefg.tmp"):
        (directory / name).write_bytes(b"x")
    (directory / ".1m.csv.abcdefgh.part.bak").write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".x.abcdefgh.tmp").write_bytes(b"x")
    (directory / ".linked.abcdefgh.tmp").symlink_to(outside / ".x.abcdefgh.tmp")
    (directory / "linked_directory").symlink_to(outside, target_is_directory=True)
    assert [str(path) for path in temporary_files(directory)] == sorted(expected)
    assert temporary_files(tmp_path / "missing") == []


def test_temporary_files_are_removed_only_under_the_held_lock(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    fd, name = tempfile.mkstemp(dir=directory, prefix=".plan-1m.json.", suffix=".tmp")
    os.write(fd, b"partial")
    os.close(fd)
    (directory / "plan-1m.json").write_bytes(b"{}")
    lock = tmp_path / ".lock"
    for proof in (lock, None, HeldLock(lock, held=False)):
        with pytest.raises(RuntimeError, match="does not hold the lock"):
            remove_temporary_files(directory, proof)
    with exclusive_lock(lock) as held:
        assert remove_temporary_files(directory, held) == [(directory / os.path.basename(name), 7)]
        assert remove_temporary_files(directory, held) == []
    with pytest.raises(RuntimeError):
        remove_temporary_files(directory, held)  # released
    assert sorted(path.name for path in directory.iterdir()) == ["plan-1m.json"]
