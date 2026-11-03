"""The test suite's guard of the real data, weights and report roots (D-055 (6)).

A test once called ``kronos-agent finetune timing`` with a figure that had become valid and with
the repository's roots: the real measurement step started and wrote under ``reports/`` and
``weights/``. ``conftest.py`` sets up three lines of defence with the helpers below (a support
module, not a test module, like ``step2_fakes``):

1. Redirection. The autouse fixture ``isolated_roots`` points ``KRONOS_AGENT_DATA_DIR``,
   ``KRONOS_AGENT_WEIGHTS_DIR`` and ``KRONOS_AGENT_REPORTS_DIR`` at a fresh temporary directory
   for every test, so that every default root of the code (``paths.data_dir()``,
   ``weights_dir()``, ``reports_dir()`` and what derives from them: the stores, fans and fold
   exports, the fine-tune root, the report directories) is the test's own, also in the child
   processes the test starts, which inherit the variables. From session start to session end the
   variables point at the session's own roots too (data and reports empty, weights a view of the
   shipped weights), so module- and session-scoped fixtures and collection get temporary roots as
   well. A test that forgets its stand-ins finds no weights to load. The temporary roots live
   under one directory per session in the system's temporary directory, outside pytest's
   ``pytest-of-<login>`` base, because some reports (the environment report) refuse a path
   naming the login; it is removed at the end of a session that passed.
2. Refusal. An audit hook (:func:`sys.addaudithook`), armed from session start to session end,
   refuses every write under a real root that this process attempts through Python, before it
   happens and whatever path reaches it (symlinks are resolved): opening a file for writing,
   creating, renaming, linking, removing or truncating an entry, changing its metadata, copying,
   moving, archiving or extracting into it. Native code that writes a file without Python's
   ``open`` raises no audit event, so the plugin also wraps, for the session, the native writers
   the code and the tests use in this process (:data:`NATIVE_WRITERS`: safetensors' ``save_file``
   and ``save_model``, behind ``save_pretrained``, and ``torch.save`` to a path): each refuses a
   target under a real root before it writes. A refusal raises :class:`RealRootWriteError` (a
   ``PermissionError``, as on a read-only volume) and is recorded: one its test did not take
   (:meth:`WriteGuard.collect`) fails that test at teardown, one outside any test fails the
   session. What the refusal cannot see: other native code that writes without an audit event,
   a native writer bound before the session started (a ``from safetensors.torch import
   save_file`` run earlier), and child processes; the listing (3) catches those, after the fact.
3. Listing. Path, kind, size and mtime of every entry under the real roots (no hashing) are
   listed at session start and at session end; any entry added, removed or changed fails the
   session (exit status 1). This also covers child processes and native writes, which the
   refusal cannot see, but only after they happened. With :data:`LISTING_ENV` set to a file
   outside the real roots, the start listing is written there (:func:`render_listing`).

The real roots are the repository's ``data/``, ``weights/`` and ``reports/``, plus the roots the
environment relocates them to when the session starts with ``KRONOS_AGENT_*_DIR`` set.

Opt-ins to read a real root, never to write it. The ``weights`` marker gives the test a weights
root that is a view of the shipped weights: its top-level entries are symlinks to the real
root's, except the fine-tune root ``finetuned/``, which stays the test's own.
``real_roots("data", "weights", "reports")`` gives each named root as a view of the whole real
root (``finetuned/`` included). What a test creates at the top level of a view is its own; a
write through a link reaches the real root and is refused like any other.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import importlib
import itertools
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import pytest

from kronos_agent.finetune.registry import FINETUNED_DIRNAME
from kronos_agent.paths import (
    DATA_DIR_ENV,
    REPO_ROOT,
    REPORTS_DIR_ENV,
    WEIGHTS_DIR_ENV,
    redact_home,
)

ROOT_ENV: Mapping[str, str] = MappingProxyType(
    {"data": DATA_DIR_ENV, "weights": WEIGHTS_DIR_ENV, "reports": REPORTS_DIR_ENV}
)
LISTING_ENV = "KRONOS_AGENT_TEST_ROOTS_LISTING"
PLUGIN_NAME = "real-roots-guard"
VIEWS_MARKER = "real_roots"
ROOTS_PREFIX = "kronos-agent-test-roots-"
SESSION_ROOTS = "session"
FINDER_FILE = ".DS_Store"
IGNORED_NAMES = frozenset({FINDER_FILE})  # listed, never compared, never linked
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
MAXPATHLEN = 1024  # the buffer of fcntl F_GETPATH (macOS)
# audit event -> its path operands: (argument index, index of its dir_fd argument or None, whether
# the operation follows a symlink in the last component). Python 3.12 argument orders.
WRITE_EVENTS: Mapping[str, tuple[tuple[int, int | None, bool], ...]] = MappingProxyType(
    {
        "open": ((0, None, True),),  # only with a write flag (WRITE_FLAGS)
        "os.mkdir": ((0, 2, False),),
        "os.mkfifo": ((0, 2, False),),
        "os.mknod": ((0, 3, False),),
        "os.remove": ((0, 1, False),),
        "os.rmdir": ((0, 1, False),),
        "os.rename": ((0, 2, False), (1, 3, False)),
        "os.link": ((0, 2, True), (1, 3, False)),  # a hard link to a real file writes it later
        "os.symlink": ((1, 2, False),),  # the link only: a view links into a real root
        "os.truncate": ((0, None, True),),
        "os.chmod": ((0, 2, True),),
        "os.chown": ((0, 3, True),),
        "os.chflags": ((0, None, True),),
        "os.lchflags": ((0, None, False),),
        "os.utime": ((0, 3, True),),
        "os.setxattr": ((0, None, True),),
        "os.removexattr": ((0, None, True),),
        "shutil.chown": ((0, None, True),),
        "shutil.copyfile": ((1, None, True),),
        "shutil.copymode": ((1, None, True),),
        "shutil.copystat": ((1, None, True),),
        "shutil.copytree": ((1, None, True),),
        "shutil.move": ((0, None, False), (1, None, True)),
        "shutil.rmtree": ((0, 1, False),),
        "shutil.make_archive": ((0, None, True),),
        "shutil.unpack_archive": ((1, None, True),),
        "tempfile.mkstemp": ((0, None, True),),
        "tempfile.mkdtemp": ((0, None, True),),
    }
)
# writers whose native code writes a file without Python's open (no audit event), wrapped for the
# session: (module, function, position and keyword of the target path). safetensors.torch's
# save_model, and huggingface_hub's save_pretrained through it, write with save_file.
NATIVE_WRITERS: tuple[tuple[str, str, int, str], ...] = (
    ("safetensors.torch", "save_file", 1, "filename"),
    ("safetensors.torch", "save_model", 1, "filename"),
    ("safetensors.numpy", "save_file", 1, "filename"),
    ("torch", "save", 1, "f"),
    ("torch.serialization", "save", 1, "f"),
)
WRAPPED_ATTRIBUTE = "__root_guard_original__"  # on a wrapped writer: the function it wraps


class RealRootWriteError(PermissionError):
    """A write under a real root, refused by the test suite's guard before it happened."""


# -- the roots -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RealRoots:
    """The real roots of a session.

    ``protected`` maps a label to a directory the suite never writes: ``data``, ``weights`` and
    ``reports`` (the repository's) and ``$KRONOS_AGENT_*_DIR`` (a relocated root). ``code`` maps
    each root's name to the directory the code resolved when the session started: what an opt-in
    views.
    """

    protected: Mapping[str, Path]
    code: Mapping[str, Path]

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> RealRoots:
        environ = os.environ if environ is None else environ
        protected: dict[str, Path] = {}
        code: dict[str, Path] = {}
        for name, variable in ROOT_ENV.items():
            default = REPO_ROOT / name
            protected[name] = default
            raw = environ.get(variable)
            relocated = Path(raw).expanduser().resolve() if raw else None  # as paths.py reads it
            code[name] = relocated or default
            if relocated is not None and relocated != default.resolve():
                protected[f"${variable}"] = relocated
        return cls(MappingProxyType(protected), MappingProxyType(code))


@dataclass(frozen=True)
class IsolatedRoots:
    """The roots of one test (or of the session's wider fixtures): ``data``, ``weights`` and
    ``reports``; ``views`` names those that are views of a real root (``weights`` also for a
    view of the shipped weights)."""

    data: Path
    weights: Path
    reports: Path
    views: frozenset[str] = frozenset()

    def environment(self) -> dict[str, str]:
        """The ``KRONOS_AGENT_*_DIR`` variables that point the code at these roots."""
        return {variable: str(getattr(self, name)) for name, variable in ROOT_ENV.items()}


def stage_view(source: Path, target: Path, *, exclude: Collection[str] = ()) -> Path:
    """Make ``target`` (new) a view of the real root ``source``: a directory whose top-level
    entries are symlinks to ``source``'s, ``exclude`` and :data:`IGNORED_NAMES` left out (a missing
    ``source`` gives an empty view). Returns ``target``."""
    target.mkdir(parents=True)
    source = Path(source)
    if source.is_dir():
        for entry in sorted(source.iterdir()):
            if entry.name in exclude or entry.name in IGNORED_NAMES:
                continue
            (target / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    return target


def view_names(marks: Iterable[pytest.Mark]) -> frozenset[str]:
    """The roots that :data:`VIEWS_MARKER` marks name; ``ValueError`` for an empty mark, a
    keyword or an unknown name."""
    names: set[str] = set()
    for mark in marks:
        unknown = [name for name in mark.args if name not in ROOT_ENV]
        if not mark.args or mark.kwargs or unknown:
            raise ValueError(
                f"{VIEWS_MARKER} takes one or more of {sorted(ROOT_ENV)}, got "
                f"args {mark.args!r} and keywords {sorted(mark.kwargs)}"
            )
        names.update(mark.args)
    return frozenset(names)


class RootFactory:
    """Fresh roots under one temporary directory per session (module docstring, 1)."""

    def __init__(self, sources: Mapping[str, Path]) -> None:
        self.sources = dict(sources)
        self.base: Path | None = None
        self._numbers = itertools.count(1)

    def open(self) -> Path:
        if self.base is None:
            self.base = Path(tempfile.mkdtemp(prefix=ROOTS_PREFIX)).resolve()
        return self.base

    def make(
        self,
        *,
        views: Collection[str] = frozenset(),
        shipped_weights: bool = False,
        name: str | None = None,
    ) -> IsolatedRoots:
        """New roots, nothing created but the views: ``views`` name the roots that view the
        whole real root; ``shipped_weights`` makes the weights root a view of the shipped
        weights (every entry but the fine-tune root) unless ``views`` names it already."""
        directory = self.open() / (name or f"{next(self._numbers):05d}")
        roots = IsolatedRoots(
            data=directory / "data",
            weights=directory / "weights",
            reports=directory / "reports",
            views=frozenset(views) | ({"weights"} if shipped_weights else set()),
        )
        for view in sorted(views):
            stage_view(self.sources[view], getattr(roots, view))
        if shipped_weights and "weights" not in views:
            stage_view(self.sources["weights"], roots.weights, exclude=(FINETUNED_DIRNAME,))
        return roots

    def close(self) -> None:
        """Remove the session's directory (links are removed, never followed)."""
        if self.base is not None:
            shutil.rmtree(self.base, ignore_errors=True)
            self.base = None


# -- the listing -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One listed path: ``kind`` (``dir``, ``file``, ``link``, ``other`` or ``absent``), ``size``
    in bytes (``None`` for a directory), ``mtime_ns`` and a link's ``target``."""

    kind: str
    size: int | None = None
    mtime_ns: int | None = None
    target: str | None = None

    def describe(self) -> str:
        if self.kind == "absent":
            return "absent"
        mtime = datetime.fromtimestamp(self.mtime_ns / 1e9, UTC).isoformat(timespec="microseconds")
        parts = [self.kind]
        if self.kind == "link":
            parts.append(f"to {redact_home(str(self.target))}")
        elif self.size is not None:
            parts.append(f"{self.size} bytes")
        return ", ".join([*parts, f"mtime {mtime.replace('+00:00', 'Z')}"])


def _entry(path: Path, *, follow: bool = False) -> Entry:
    try:
        info = os.stat(path) if follow else os.lstat(path)
    except FileNotFoundError:
        return Entry("absent")
    if stat.S_ISLNK(info.st_mode):
        try:
            target = os.readlink(path)
        except OSError:
            target = None
        return Entry("link", info.st_size, info.st_mtime_ns, target)
    if stat.S_ISDIR(info.st_mode):
        return Entry("dir", None, info.st_mtime_ns)
    kind = "file" if stat.S_ISREG(info.st_mode) else "other"
    return Entry(kind, info.st_size, info.st_mtime_ns)


def list_roots(roots: Mapping[str, Path]) -> dict[str, Entry]:
    """``"<label>/<relative path>" -> Entry`` of every entry under each root, and the root itself
    as ``"<label>"`` (followed when it is a symlink; ``absent`` when it does not exist). Symlinks
    inside are listed, never followed."""
    found: dict[str, Entry] = {}
    for label, root in roots.items():
        root = Path(root)
        found[label] = _entry(root, follow=True)
        if found[label].kind != "dir":
            continue
        for current, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(dirnames + filenames):
                path = Path(current) / name
                entry = _entry(path)
                if entry.kind != "absent":  # removed while listed
                    found[f"{label}/{path.relative_to(root).as_posix()}"] = entry
    return found


def listing_changes(before: Mapping[str, Entry], after: Mapping[str, Entry]) -> list[str]:
    """One line per entry added, removed or changed between two listings, sorted by path.

    Finder's ``.DS_Store`` files are no change, nor is the new mtime of a directory whose
    ``.DS_Store`` appeared or changed (Finder may write one while a folder is open)."""
    lines: list[str] = []
    absent = Entry("absent")
    for key in sorted(set(before) | set(after)):
        if PurePosixPath(key).name in IGNORED_NAMES:
            continue
        old, new = before.get(key, absent), after.get(key, absent)
        if old == new:
            continue
        if old.kind == "absent":
            lines.append(f"added    {key} ({new.describe()})")
        elif new.kind == "absent":
            lines.append(f"removed  {key} ({old.describe()})")
        else:
            finder = f"{key}/{FINDER_FILE}"
            finder_wrote = before.get(finder, absent) != after.get(finder, absent)
            if old.kind == new.kind == "dir" and finder_wrote:
                continue
            lines.append(f"changed  {key} ({old.describe()} -> {new.describe()})")
    return lines


def render_listing(
    entries: Mapping[str, Entry], roots: Mapping[str, Path], *, listed_at: datetime
) -> str:
    """The listing as text: a ``#`` header naming the roots (home-redacted), then one line per
    entry, tab-separated: path, kind, size (``-`` for none), mtime in ns, and a link's target."""
    stamp = listed_at.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    lines = [
        f"# real roots of the kronos-agent test suite, listed at session start {stamp}",
        *(f"# {label} = {redact_home(str(root))}" for label, root in roots.items()),
        "# path<TAB>kind<TAB>size<TAB>mtime_ns[<TAB>link target]; .DS_Store is never compared",
    ]
    for key in sorted(entries):
        entry = entries[key]
        fields = [
            key,
            entry.kind,
            "-" if entry.size is None else str(entry.size),
            "-" if entry.mtime_ns is None else str(entry.mtime_ns),
        ]
        if entry.target is not None:
            fields.append(entry.target)
        lines.append("\t".join(fields))
    return "\n".join(lines) + "\n"


def listing_path(raw: str | None, roots: Mapping[str, Path]) -> Path | None:
    """The file :data:`LISTING_ENV` names (``None`` when unset); ``pytest.UsageError`` when it
    lies under a real root."""
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    for label, root in roots.items():
        if path.is_relative_to(Path(root).resolve()):
            raise pytest.UsageError(
                f"{LISTING_ENV} must name a file outside the real roots, not one under {label}"
            )
    return path


# -- the refusal -----------------------------------------------------------------------------------


def _fold(text: str) -> str:
    """macOS volumes ignore case by default, so ``Data/x`` is ``data/x`` there."""
    return text.casefold() if sys.platform == "darwin" else text


def _fd_directory(fd: int) -> str | None:
    """The path of an open directory descriptor (the ``dir_fd`` of ``shutil.rmtree``'s walk)."""
    try:
        if sys.platform == "darwin":
            raw = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(MAXPATHLEN))
            return os.fsdecode(raw.split(b"\0", 1)[0])
        return os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, ValueError, AttributeError):
        return None


def _resolve(value: object, dir_fd: object, follow: bool) -> str | None:
    """The absolute, symlink-resolved path an operation reaches (``None``: a descriptor, or a
    path that cannot be told). Without ``follow`` the last component is the entry itself."""
    if value is None or isinstance(value, int):
        return None
    try:
        text = os.fsdecode(os.fspath(value))
        if not text:
            return None
        if not os.path.isabs(text):
            if isinstance(dir_fd, int) and dir_fd >= 0:
                base = _fd_directory(dir_fd)
            else:
                base = os.getcwd()
            if base is None:
                return None
            text = os.path.join(base, text)
        parent, name = os.path.split(text.rstrip(os.sep) or os.sep)
        if follow or name in ("", ".", ".."):
            return os.path.realpath(text)
        return os.path.join(os.path.realpath(parent), name)
    except (TypeError, ValueError, OSError):
        return None


class WriteGuard:
    """The refusal of the module docstring (2). One per process, because an audit hook cannot be
    removed: :meth:`disarm` only stops it from refusing."""

    def __init__(self) -> None:
        self.refusals: list[str] = []
        self._roots: tuple[tuple[str, str, str], ...] = ()  # (folded path, path, label)
        self._installed = False

    @property
    def armed(self) -> bool:
        return bool(self._roots)

    def arm(self, roots: Mapping[str, Path]) -> None:
        """Refuse writes under ``roots`` (label -> directory) from now on."""
        resolved = []
        for label, root in roots.items():
            text = os.path.realpath(root)
            resolved.append((_fold(text), text, label))
        self._roots = tuple(resolved)
        if not self._installed:
            sys.addaudithook(self._audit)
            self._installed = True

    def disarm(self) -> None:
        self._roots = ()

    @contextlib.contextmanager
    def disarmed(self) -> Iterator[None]:
        """No refusal inside the block: only to remove a probe that a failed guard let through."""
        roots, self._roots = self._roots, ()
        try:
            yield
        finally:
            self._roots = roots

    @contextlib.contextmanager
    def collect(self) -> Iterator[list[str]]:
        """Take the refusals recorded inside the block into the yielded list, so that nobody
        else accounts for them: a test's (``isolated_roots``) or a probe's (the guard's tests)."""
        start = len(self.refusals)
        taken: list[str] = []
        try:
            yield taken
        finally:
            taken.extend(self.refusals[start:])
            del self.refusals[start:]

    def take_all(self) -> list[str]:
        taken, self.refusals[:] = list(self.refusals), []
        return taken

    def locate(self, path: object, *, follow: bool = True) -> str | None:
        """``<label>/<relative path>`` when ``path`` reaches under an armed root, else ``None``."""
        resolved = _resolve(path, None, follow)
        return None if resolved is None else self._locate(resolved)

    def _locate(self, resolved: str) -> str | None:
        folded = _fold(resolved)
        for root_folded, root, label in self._roots:
            if folded == root_folded:
                return label
            if folded.startswith(root_folded + os.sep):
                return f"{label}/{resolved[len(root) + 1 :]}"
        return None

    def _refuse(self, event: str, where: str | None) -> None:
        if where is not None:
            self.refusals.append(f"{event} {where}")
            raise RealRootWriteError(f"test root guard: refused {event} under a real root: {where}")

    def _audit(self, event: str, args: tuple) -> None:
        if not self._roots:
            return
        operands = WRITE_EVENTS.get(event)
        if operands is None:
            return
        if event == "open":
            flags = args[2] if len(args) > 2 else None
            if not isinstance(flags, int) or not flags & WRITE_FLAGS:
                return
        for index, dir_fd_index, follow in operands:
            if index >= len(args):
                continue
            dir_fd = None
            if dir_fd_index is not None and dir_fd_index < len(args):
                dir_fd = args[dir_fd_index]
            resolved = _resolve(args[index], dir_fd, follow)
            self._refuse(event, None if resolved is None else self._locate(resolved))

    # -- the native writers ------------------------------------------------------------------

    def _native_writer(
        self, event: str, function: Callable, position: int, keyword: str
    ) -> Callable:
        """``function`` refusing, while the guard is armed, a target path under a real root
        before it writes (a file object is Python's own: the audit hook saw its open)."""
        guard = self

        @functools.wraps(function)
        def guarded(*args, **kwargs):
            target = kwargs.get(keyword, args[position] if len(args) > position else None)
            if guard._roots and isinstance(target, (str, bytes, os.PathLike)):
                guard._refuse(event, guard.locate(target))
            return function(*args, **kwargs)

        setattr(guarded, WRAPPED_ATTRIBUTE, function)
        return guarded

    def wrap_native_writers(self) -> list[str]:
        """Wrap every writer of :data:`NATIVE_WRITERS` in its module (importing it), once: a
        wrapped writer refuses a target under an armed root as the audit hook would. Returns the
        writers wrapped by this call; a module that cannot be imported is left out."""
        wrapped: list[str] = []
        for module_name, name, position, keyword in NATIVE_WRITERS:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            function = getattr(module, name)
            if getattr(function, WRAPPED_ATTRIBUTE, None) is not None:
                continue
            event = f"{module_name}.{name}"
            setattr(module, name, self._native_writer(event, function, position, keyword))
            wrapped.append(event)
        return wrapped


GUARD = WriteGuard()


def fail_on_refusals(refused: Collection[str]) -> None:
    """Fail the test (at its teardown) that attempted the writes ``refused`` lists."""
    if refused:
        pytest.fail(
            "test root guard: this test attempted writes under a real root, each refused before "
            "it happened:\n" + "\n".join(f"  {line}" for line in refused),
            pytrace=False,
        )


# -- the plugin ------------------------------------------------------------------------------------


class RootGuardPlugin:
    """The session side of the guard, registered by ``conftest.py`` as :data:`PLUGIN_NAME`: lists
    ``real.protected`` at session start and end (3), arms ``guard`` for the session and wraps the
    native writers (2) and, with ``isolate``, points the roots' variables at the session's own
    temporary roots and makes the tests' (``factory``, 1).
    """

    def __init__(
        self,
        real: RealRoots,
        *,
        guard: WriteGuard | None = None,
        listing: Path | None = None,
        isolate: bool = False,
    ) -> None:
        self.real = real
        self.guard = guard
        self.listing = listing
        self.factory = RootFactory(real.code) if isolate else None
        self.before: dict[str, Entry] | None = None
        self.problems: list[str] = []
        self._saved: dict[str, str | None] = {}

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.before = list_roots(self.real.protected)
        if self.listing is not None:
            self.listing.parent.mkdir(parents=True, exist_ok=True)
            self.listing.write_text(
                render_listing(self.before, self.real.protected, listed_at=datetime.now(UTC)),
                encoding="utf-8",
            )
        if self.factory is not None:
            defaults = self.factory.make(name=SESSION_ROOTS, shipped_weights=True)
            for variable, value in defaults.environment().items():
                self._saved[variable] = os.environ.get(variable)
                os.environ[variable] = value
        if self.guard is not None:
            self.guard.arm(self.real.protected)
            self.guard.wrap_native_writers()

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int):
        """Outermost, so that the check runs and prints after every other hook (the terminal
        summary included) and its exit status is the session's."""
        try:
            return (yield)
        finally:
            self._finish(session)

    def _finish(self, session: pytest.Session) -> None:
        for variable, value in self._saved.items():
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = value
        self._saved.clear()
        refused = self.guard.take_all() if self.guard is not None else []
        if self.before is not None:
            self.problems = listing_changes(self.before, list_roots(self.real.protected))
        self.problems += [f"refused  {line} (not taken by a test)" for line in refused]
        if self.factory is not None and not self.problems and session.exitstatus == 0:
            self.factory.close()  # kept after a failure, for a look at what the tests wrote
        if self.guard is not None:
            self.guard.disarm()
        if self.problems:
            self._report(session.config)
            if session.exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
                session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def _report(self, config: pytest.Config) -> None:
        title = "test root guard: the session wrote, or tried to write, under a real root"
        lines = [
            (
                "the test suite must never write under the real data, weights or reports roots "
                "(D-055 (6)); this session fails:"
            ),
            *(f"  {line}" for line in self.problems),
            "a run outside the suite that wrote there during the session shows here too",
        ]
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is None:
            sys.stderr.write("\n".join([title, *lines]) + "\n")
            return
        reporter.write_sep("=", title, red=True, bold=True)
        for line in lines:
            reporter.write_line(line, red=True)


__all__ = [
    "GUARD",
    "LISTING_ENV",
    "NATIVE_WRITERS",
    "PLUGIN_NAME",
    "ROOT_ENV",
    "VIEWS_MARKER",
    "WRAPPED_ATTRIBUTE",
    "Entry",
    "IsolatedRoots",
    "RealRootWriteError",
    "RealRoots",
    "RootFactory",
    "RootGuardPlugin",
    "WriteGuard",
    "fail_on_refusals",
    "list_roots",
    "listing_changes",
    "listing_path",
    "render_listing",
    "stage_view",
    "view_names",
]
