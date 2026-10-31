"""``kronos-agent vendor-kronos``: the tarball guard that runs before any tree or blob check.

Every archive is synthetic, built in memory and extracted under pytest's ``tmp_path``: no network
and nothing under ``third_party/``. Escape targets are absolute paths inside ``tmp_path`` too, so
even a broken guard cannot write outside the test sandbox.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from kronos_agent.cli.vendor_kronos import _safe_extract

TOP = "Kronos-0123abc"
COMMIT_COMMENT = "0" * 40  # GitHub's commit tarballs carry the commit id in a pax global header
UNKNOWN_TYPE = b"Z"


def make_archive(*members: tuple[str, bytes, bytes], comment: str | None = None) -> bytes:
    """A ``.tar.gz`` of ``(name, type, data)`` members; names are written verbatim."""
    buffer = io.BytesIO()
    pax_headers = {"comment": comment} if comment else None
    with tarfile.open(
        fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers=pax_headers
    ) as tar:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(data)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "../../outside"
            tar.addfile(info, io.BytesIO(data) if data else None)
    return buffer.getvalue()


def regular(name: str, data: bytes = b"x\n") -> tuple[str, bytes, bytes]:
    return (name, tarfile.REGTYPE, data)


@pytest.fixture
def destination(tmp_path: Path) -> Path:
    path = tmp_path / "extract"
    path.mkdir()
    return path


def tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_extracts_a_github_style_archive(destination):
    archive = make_archive(
        (f"{TOP}/", tarfile.DIRTYPE, b""),
        regular(f"{TOP}/model/kronos.py", b"import torch\n"),
        regular(f"./{TOP}/LICENSE", b"MIT\n"),
        comment=COMMIT_COMMENT,
    )
    top = _safe_extract(archive, destination)
    assert top == destination / TOP
    assert (top / "model" / "kronos.py").read_bytes() == b"import torch\n"
    assert (top / "LICENSE").read_bytes() == b"MIT\n"
    assert tree(destination) == [TOP, f"{TOP}/LICENSE", f"{TOP}/model", f"{TOP}/model/kronos.py"]


@pytest.mark.parametrize("followed_by_a_normal_member", [False, True])
@pytest.mark.parametrize("prefix", ["", "/"], ids=["absolute", "double-slash"])
def test_absolute_member_is_refused_before_it_is_written(
    tmp_path, destination, prefix, followed_by_a_normal_member
):
    escaped = tmp_path / "escaped.txt"  # absolute and outside the destination, inside the sandbox
    members = [regular(prefix + str(escaped))]
    if followed_by_a_normal_member:
        members.append(regular(f"{TOP}/README.md"))
    with pytest.raises(RuntimeError, match="unsafe path"):
        _safe_extract(make_archive(*members), destination)
    assert not escaped.exists()
    assert tree(destination) == []


@pytest.mark.parametrize(
    "name", [f"{TOP}/../escaped.txt", f"{TOP}/../../escaped.txt", "../escaped.txt", ".", ""]
)
def test_traversal_and_empty_names_are_refused(tmp_path, destination, name):
    with pytest.raises(RuntimeError, match="unsafe path"):
        _safe_extract(make_archive(regular(name)), destination)
    assert not (tmp_path / "escaped.txt").exists()
    assert tree(destination) == []


def test_member_resolving_outside_the_destination_is_refused(tmp_path, destination):
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / TOP).symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unsafe path"):
        _safe_extract(make_archive(regular(f"{TOP}/README.md")), destination)
    assert tree(outside) == []


@pytest.mark.parametrize(
    "kind",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.BLKTYPE,
        UNKNOWN_TYPE,
    ],
    ids=["symlink", "hardlink", "fifo", "char-device", "block-device", "unknown"],
)
def test_links_devices_and_unknown_types_are_refused(destination, kind):
    data = b"payload" if kind == UNKNOWN_TYPE else b""  # tarfile skips unknown types as data
    archive = make_archive((f"{TOP}/member", kind, data), regular(f"{TOP}/README.md"))
    with pytest.raises(RuntimeError, match="unsupported member type"):
        _safe_extract(archive, destination)
    assert tree(destination) == []


def test_more_than_one_top_level_directory_is_refused(destination):
    archive = make_archive(regular(f"{TOP}/README.md"), regular("Other-top/README.md"))
    with pytest.raises(RuntimeError, match="more than one top-level directory"):
        _safe_extract(archive, destination)
    assert not (destination / "Other-top").exists()


def test_archive_without_members_is_refused(destination):
    with pytest.raises(RuntimeError, match="empty archive"):
        _safe_extract(make_archive(), destination)
    assert tree(destination) == []
