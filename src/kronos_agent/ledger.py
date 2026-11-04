"""The experiment ledger (``docs/RETRAINING.md``, 7.3; SPEC rules 6 to 8 and gate 6).

``<reports>/ledger/experiments.jsonl`` (:func:`ledger_path`; :func:`kronos_agent.paths.ledger_dir`,
which follows ``KRONOS_AGENT_REPORTS_DIR``) records what the retraining looked at, ran, was
approved for and read: one canonical-JSON object per line (sorted keys, no whitespace, ASCII),
each terminated by LF and at most :data:`MAX_LINE_BYTES` bytes long, with exactly
:data:`LINE_KEYS`: ``schema`` and ``schema_version``, ``seq`` (the 1-based line number), ``ts``
(the UTC append time, ``YYYY-MM-DDTHH:MM:SSZ``), ``prev_sha256`` (the sha256 of the previous
line's bytes without its LF; :data:`GENESIS_SHA256` for line 1), ``kind`` (:data:`KINDS`),
``params_version``, ``code`` (:func:`kronos_agent.provenance.code_version`) and the kind's
``payload`` (exactly its :data:`PAYLOAD_KEYS`). The file is tracked and published with the
reports (SPEC section 19), so no line holds personal data, an IPv4 literal, an absolute path, a
BingX host name or anything else the repository's hygiene scan of reports/ refuses (an email
address, a personal path prefix, the cloud host name fragment, a credential assignment); every
string is printable ASCII and every number finite.

Append-only and hash-chained, as the fine-tune registry: an edited, removed or reordered line
breaks the chain at the line after it, every reader verifies every line (:func:`read_ledger`) and
refuses a broken ledger (:class:`LedgerChainError` naming the line), and nothing is ever repaired
or rewritten. The last line has no line after it: whoever records the use of an entry records its
``line_sha256``, which the commands print after every append. An append takes the exclusive lock
``<ledger dir>/.lock`` (a held lock is a ``LockBusyError``, a refusal), reads and verifies the
chain under it, builds the payload from the entries read there (:func:`append_entry` takes a
callable, so a check and its append are one atomic step), validates the new line exactly as a
reader would and replaces the file with the old bytes plus the line (a durable atomic write).
Readers take no lock: they see the old file or the new one.

Kinds (each payload's keys and rules are those of the interface specification of this build):

- ``exploration``: something looked at on training-segment data or on inputs only, 0 tests (the
  pilot fold, the pilot check, the tokenizer diagnostic). Reading validation or sealed outcomes
  is a test, never exploration, and is refused, as is any sealed input.
- ``fit``: one training child of a recipe run: its checkpoint and registry line when registered,
  its status, attempt, device and spend, never a loss, an NLL value or a selected pseudo-epoch
  (the blinding of design 6.3).
- ``gate3_approval``: the hours the owner approved in chat, the measured estimate shown and their
  ratio, at most :data:`APPROVAL_TOLERANCE` (D-040, D-046), appended by the command that runs
  under it before its first unit of work. The owner's words stay in ``docs/DECISIONS.md``.
- ``preregistration``: the pre-registration bundle of design 7.1, a file under the reports root
  whose sha256 is recorded once per recipe and verified by every ``finetune`` command
  (:func:`verify_bundles`).
- ``owner_order``: the owner's order to open the sealed segment for one T-B candidate (gate 6),
  appended by ``kronos-agent ledger record-order`` only after the owner's explicit order in chat.
  ``data.partition.sealed_access`` verifies it here (:func:`verify_open_order`).
- ``test``: one reading of reserved hours, appended by :func:`kronos_agent.budget.spend_test`
  before any statistic is computed (SPEC rule 8).

A payload that breaks its kind's schema is a :class:`LedgerError` (a programming error, exit 1);
one that breaks a governance rule (validation or sealed outcomes in an exploration, an approval
above the tolerance, a dangling or repeated reference, a second bundle or order, a test past the
budget or read out of order, a time earlier than the previous line's, personal data) is a
:class:`LedgerRefused` (exit 2). Every check that needs only the ledger runs on read too, where
any problem is a :class:`LedgerChainError` naming its line: the references, the uniqueness
rules, the times (``ts`` never decreases, so a reading cannot be dated into an earlier week),
and SPEC rule 8 as the ``test`` lines record it (at most :data:`TESTS_PER_ISO_WEEK` in an ISO
week and :data:`TESTS_PER_QUARTER` in a calendar quarter, T-B after a T-A reading of the same
recipe in an earlier ISO week when its bundle registers both). Only the rules that need more than
the ledger run at append alone: the bundle files (``budget.spend_test``,
:func:`record_preregistration`) and the personal data of the machine that writes the line (its
login, host name and home directory, which only the writer knows: a line written on one machine
must verify on every other, whatever its login). The checks that do not depend on the machine
(printable ASCII, finite numbers, IPv4 literals, absolute paths, the BingX host name, the
hygiene scan's email, personal-path, cloud-host and credential rules) run on read too.

Which ledger may authorize a read of which data (:func:`binding_problem`): the reports root, and
the ledger with it, is relocatable (``KRONOS_AGENT_REPORTS_DIR``), and the data root on its own
(``KRONOS_AGENT_DATA_DIR``). A ledger other than the repository's ``reports/ledger`` never
authorizes a read of the repository's ``data/`` or of a data root that links into it (a view):
its owner's order does not open the sealed candles (``data.partition.sealed_access``), its
readings spend no test there (``budget.spend_test``), and its pre-registration bundle does not
admit the reads that wait for one (Part B of the tokenizer diagnostic, the folds after the
pilot: ``finetune.tokenizer_diag.prepare_diag``, ``finetune.walkforward.plan_fold``). An order,
a reading or a bundle recorded in a scratch ledger stays in its scratch world (gate 6, hard rule
7). Reads that need no authorization (the training segment: the pilot, Part A of the tokenizer
diagnostic) are not bound.

At load this module imports only the standard library and light project modules, because
``data.partition`` imports it (lazily) for the sealed guard: :mod:`kronos_agent.params` (the
params version) and :mod:`kronos_agent.envinfo` (personal data) are imported where an append
needs them, reading imports neither, and :mod:`kronos_agent.exchange.public_market` (the BingX
domain; host literals live only in the exchange package) is imported where a line is checked.
It keeps its own copies of the fold-label and checkpoint-id patterns (``finetune.folds`` and
``finetune.registry`` import heavier modules), of the IPv4 pattern (``envinfo``) and of the
hygiene scan's patterns (``tests/test_repo_hygiene.py``); the tests assert that they are equal.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
import types
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import provenance
from .data.fsutil import atomic_write_bytes, exclusive_lock
from .hashing import canonical_json, sha256_bytes, sha256_file
from .paths import LABEL_RE, LEDGER_SUBDIR, REPO_ROOT, data_dir, ledger_dir, reports_dir

LEDGER_NAME = "experiments.jsonl"
LOCK_NAME = ".lock"
LEDGER_SCHEMA = "experiment"
LEDGER_SCHEMA_VERSION = 1
GENESIS_SHA256 = "0" * 64
KINDS: tuple[str, ...] = (
    "exploration",
    "fit",
    "gate3_approval",
    "owner_order",
    "preregistration",
    "test",
)
LINE_KEYS: tuple[str, ...] = (
    "schema",
    "schema_version",
    "seq",
    "ts",
    "prev_sha256",
    "kind",
    "params_version",
    "code",
    "payload",
)
PAYLOAD_KEYS: Mapping[str, tuple[str, ...]] = types.MappingProxyType(
    {
        "exploration": (
            "activity",
            "subject",
            "stage",
            "recipe_id",
            "inputs",
            "outcomes",
            "details",
        ),
        "fit": (
            "checkpoint_id",
            "registry_line_sha256",
            "recipe_id",
            "fold",
            "replicate",
            "run_id",
            "status",
            "attempt",
            "blinded",
            "fit_record_sha256",
            "device",
            "threads",
            "wall_seconds",
            "cpu_seconds",
            "approval_seq",
            "order_id",
        ),
        "gate3_approval": (
            "command",
            "subject",
            "unit",
            "approved",
            "estimate",
            "ratio",
            "tolerance",
            "gate3_hours",
            "above_gate3",
            "given_at",
            "estimate_detail",
        ),
        "owner_order": ("order_id", "scope", "candidate", "text"),
        "preregistration": (
            "bundle_relpath",
            "bundle_sha256",
            "recipe_id",
            "tests",
            "forecasting_code_sha256",
            "training_code_sha256",
        ),
        "test": (
            "test_id",
            "test",
            "recipe_id",
            "bundle_relpath",
            "bundle_sha256",
            "preregistration_seq",
            "iso_week",
            "quarter",
            "week_count",
            "quarter_count",
            "order_id",
        ),
    }
)
CODE_KEYS: tuple[str, ...] = ("commit", "dirty", "source_tree_sha256")
EXPLORATION_ACTIVITIES: tuple[str, ...] = ("pilot-fold", "pilot-check", "tokenizer-diag")
# "stopped": a run its spend control stopped (design 3.5), with what it spent, so that a lost
# run's compute counts against the next run of the same study
EXPLORATION_STAGES: tuple[str, ...] = ("planned", "done", "stopped")
SEGMENTS: tuple[str, ...] = ("training", "validation", "sealed")
# what an exploration may read: these segments' candles as inputs, and these segments' outcomes
# (realised candles after a decision hour); reading validation or sealed outcomes is a test
EXPLORATION_INPUTS: tuple[str, ...] = ("training", "validation")
EXPLORATION_OUTCOMES: tuple[str, ...] = ("training",)
FIT_STATUSES: tuple[str, ...] = (
    "registered",
    "failed",
    "memory_abort",
    "time_abort",
    "killed",
    "no_result",
)
FIT_ATTEMPTS: tuple[int, ...] = (1, 2)  # design 6.3: one retry with the same seed
FIT_DEVICES: tuple[str, ...] = ("cpu", "mps")
APPROVAL_COMMANDS: tuple[str, ...] = ("finetune fold", "finetune pilot-check")
APPROVAL_UNITS: tuple[str, ...] = ("mps_hours", "cpu_hours")
APPROVAL_TOLERANCE = 1.25  # D-040, D-046: a measured estimate above 1.25 x the figure goes back
GATE3_HOURS = 1.0
ORDER_SCOPE_OPEN_SEALED = "open_sealed"
ORDER_SCOPES: tuple[str, ...] = (ORDER_SCOPE_OPEN_SEALED,)
TEST_NAMES: tuple[str, ...] = ("T-A", "T-B", "T-C")
TESTS_PER_ISO_WEEK = 3  # SPEC rule 8 (kronos_agent.budget enforces it before an append)
TESTS_PER_QUARTER = 15
TEST_ID_RE = re.compile(r"T-[ABC]-[0-9a-f]{8}")
ORDER_ID_RE = re.compile(r"order-[0-9]{4}")
BUNDLE_RELPATH_RE = re.compile(r"finetune/prereg-(?P<recipe8>[0-9a-f]{8})\.config\.json")
# copies of finetune.folds.FOLD_LABEL_RE and finetune.registry.CHECKPOINT_ID_RE (asserted equal)
FOLD_LABEL_RE = re.compile(
    r"(?P<kind>wf|q|s)-(?P<year>[0-9]{4})-(?P<month>[0-9]{2})"
    r"|final-(?P<fyear>[0-9]{4})-(?P<fmonth>[0-9]{2})-(?P<fday>[0-9]{2})"
    r"|(?P<synthetic>synthetic(?:-(?!r[0-9]+(?:-|$))[a-z0-9]+)*)"
)
CHECKPOINT_ID_RE = re.compile(r"ks-[a-z0-9]+(?:-[a-z0-9]+)*")
CHECKPOINT_ID_MAX_LENGTH = 64
MAX_LINE_BYTES = 65536
MAX_DETAILS_BYTES = 16384
MAX_ORDER_TEXT = 1000
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_HEX12_RE = re.compile(r"[0-9a-f]{12}")
_TS_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
# an absolute path: a token starting with "/" or "~/" that no word character, dot, tilde, slash or
# hyphen precedes (the pattern of finetune.timing, which redacts such tokens from error texts)
_PATH_TOKEN = re.compile(r"(?<![\w.~/-])(?:~/|/)[^\s'\"`<>,;:()\[\]{}]+")
# a copy of envinfo.IPV4_RE (asserted equal by the tests): reading the ledger never imports the
# personal-data check, which depends on the machine that runs it
_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
# copies of the machine-independent rules of the repository's hygiene scan of every file under
# reports/ (tests/test_repo_hygiene.py; asserted equal by the tests), assembled the same way so
# that this file passes that scan: a ledger line it flags would fail it on every later run, and
# the ledger is never repaired
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PERSONAL_PATH_PREFIXES: tuple[str, ...] = ("/Users" + "/", "/home" + "/")
_CLOUD_HOST_FRAGMENT = "amazon" + "aws"
_CREDENTIAL_RE = re.compile("BINGX_API_" + r"(?:KEY|SECRET)\s*=\s*\S+")
# how deep below a data root a link into the repository's data/ is looked for: the stores are
# candles/<source>/<interval>.csv, and a view (tests/root_guard.py) links its top-level entries
DATA_LINK_DEPTH = 3


class LedgerError(RuntimeError):
    """The ledger cannot be read, or a payload or argument is malformed (a programming error)."""


class LedgerChainError(LedgerError):
    """A line of the ledger, its hash chain or a rule it records is broken (never repaired)."""


class LedgerRefused(PermissionError):
    """A governance rule refuses the append or the lookup."""


class OrderNotFound(LedgerRefused):
    """No owner_order of that id and scope is recorded."""


class BundleChanged(LedgerRefused):
    """A recorded pre-registration bundle file is missing or changed."""


@dataclass(frozen=True, eq=False)
class LedgerEntry:
    """One verified line of the ledger.

    ``document``, ``payload`` and ``code`` are read-only views of the parsed line (shallow: the
    values inside are the parsed JSON); the line's bytes are the record. ``eq=False``: entries
    are compared by their ``line`` (or ``line_sha256``), never field by field.
    """

    line: bytes
    line_sha256: str
    seq: int
    kind: str
    ts: str
    params_version: str
    code: Mapping[str, Any]
    payload: Mapping[str, Any]
    document: Mapping[str, Any]

    @property
    def moment(self) -> datetime:
        """``ts`` as an aware UTC datetime."""
        return parse_ts(self.ts)


# -- times -----------------------------------------------------------------------------------------


def _aware(moment: object, name: str) -> datetime:
    if not isinstance(moment, datetime) or moment.utcoffset() is None:
        raise LedgerError(f"{name} must be a timezone-aware datetime, got {moment!r}")
    return moment.astimezone(UTC)


def format_ts(moment: datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ`` of an aware datetime, in UTC, whole seconds (``ts``)."""
    return _aware(moment, "a ledger time").strftime(TS_FORMAT)


def parse_ts(text: str) -> datetime:
    """The aware UTC datetime of a ledger time text (:class:`LedgerError` for any other text)."""
    if not isinstance(text, str) or not _TS_RE.fullmatch(text):
        raise LedgerError(f"a ledger time is YYYY-MM-DDTHH:MM:SSZ, got {text!r}")
    try:
        return datetime.strptime(text, TS_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        raise LedgerError(f"{text!r} is not a valid UTC time") from None


def iso_week_key(moment: datetime) -> str:
    """``YYYY-Www`` of the ISO week of the UTC date of ``moment`` (``2026-W45``)."""
    year, week, _ = _aware(moment, "moment").date().isocalendar()
    return f"{year}-W{week:02d}"


def quarter_key(moment: datetime) -> str:
    """``YYYY-Qn`` of the calendar quarter of the UTC date of ``moment`` (``2026-Q4``)."""
    day = _aware(moment, "moment").date()
    return f"{day.year}-Q{(day.month - 1) // 3 + 1}"


def week_number(moment: datetime) -> tuple[int, int]:
    """``(ISO year, ISO week)`` of the UTC date of ``moment``, for ordering readings by week."""
    year, week, _ = _aware(moment, "moment").date().isocalendar()
    return year, week


# -- layout ----------------------------------------------------------------------------------------


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else ledger_dir()


def _label(base: Path) -> str:
    """A path-free name of the ledger of ``base`` for messages."""
    return f"{base.name}/{LEDGER_NAME}"


def ledger_path(root: Path | None = None) -> Path:
    """``<root>/experiments.jsonl``; ``root=None`` is :func:`kronos_agent.paths.ledger_dir`."""
    return _root(root) / LEDGER_NAME


# -- which ledger may authorize which data ---------------------------------------------------------


def _inside(path: str, directory: os.stat_result) -> bool:
    """``path`` (resolved) is the directory ``directory`` describes, or lies under it; compared by
    device and inode, so that a case-insensitive volume or another spelling cannot hide it."""
    current = path
    while True:
        try:
            if os.path.samestat(os.stat(current), directory):
                return True
        except OSError:
            pass  # not created yet: its parents decide
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def _same_directory(first: Path, second: Path) -> bool:
    """``first`` and ``second`` name one directory (both resolved; compared by device and inode
    when both exist, by their resolved text otherwise)."""
    one, two = os.path.realpath(first), os.path.realpath(second)
    try:
        return os.path.samestat(os.stat(one), os.stat(two))
    except OSError:
        return os.path.normcase(one) == os.path.normcase(two)


def _reach(data: Path, repository_data: Path) -> str | None:
    """How the data root ``data`` reaches the repository's ``data/``, or ``None``: it is that
    directory or lies inside it (resolved), or an entry at most :data:`DATA_LINK_DEPTH` levels
    below it is a symlink that resolves into it (a view). Nothing is read but metadata."""
    try:
        target = os.stat(repository_data)
    except OSError:
        return None  # no data/ in the repository: nothing to reach
    base = os.path.realpath(data)
    if _inside(base, target):
        return "the data root is the repository's data/ (or lies inside it)"
    if not os.path.isdir(base):
        return None
    depth = base.rstrip(os.sep).count(os.sep)
    for current, dirnames, filenames in os.walk(base):  # links are listed, never followed
        for name in (*dirnames, *filenames):
            path = os.path.join(current, name)
            if os.path.islink(path) and _inside(os.path.realpath(path), target):
                return "the data root links into the repository's data/ (a view of it)"
        if current.count(os.sep) - depth >= DATA_LINK_DEPTH - 1:
            dirnames[:] = []
    return None


def binding_problem(*, root: Path | None = None, data_root: Path | None = None) -> str | None:
    """Why the ledger at ``root`` (default :func:`kronos_agent.paths.ledger_dir`) may not
    authorize reading the data root ``data_root`` (default :func:`kronos_agent.paths.data_dir`):
    the sealed segment it would open (gate 6), the outcomes a test it records would read (SPEC
    rule 8) or the reads its pre-registration bundle would admit (hard rule 7: Part B of the
    tokenizer diagnostic, the folds after the pilot); ``None`` when it may.

    The repository's own ledger, ``reports/ledger`` (also through a relocated reports root that
    links to it), authorizes any data root. Another ledger (``KRONOS_AGENT_REPORTS_DIR`` pointing
    at another reports root: a scratch run, a test) authorizes only a data root that neither is
    the repository's ``data/`` nor links into it: an order or a reading recorded there never
    reaches the repository's candles or fans. A copy of the data elsewhere is not recognised
    (mistake-proofing, not security)."""
    ledger = Path(root) if root is not None else ledger_dir()
    data = Path(data_root) if data_root is not None else data_dir()
    if _same_directory(ledger, REPO_ROOT / "reports" / LEDGER_SUBDIR):
        return None
    reach = _reach(data, REPO_ROOT / "data")
    if reach is None:
        return None
    return (
        "the experiment ledger in use is not the repository's reports/ledger (a relocated "
        f"reports root, KRONOS_AGENT_REPORTS_DIR, or another ledger root), and {reach}: an "
        "order, a test or a pre-registration bundle recorded in another ledger never opens or "
        "reads the repository's data (gate 6, hard rule 7); run with the repository's reports "
        "root, or relocate the data root too (KRONOS_AGENT_DATA_DIR)"
    )


def require_binding(*, root: Path | None = None, data_root: Path | None = None) -> None:
    """Refuse (:class:`LedgerRefused`) a ledger that may not authorize reading ``data_root``
    (:func:`binding_problem`)."""
    problem = binding_problem(root=root, data_root=data_root)
    if problem is not None:
        raise LedgerRefused(problem)


# -- values ----------------------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    """A finite JSON number (an int, or a finite float; never a bool)."""
    return _is_int(value) or (isinstance(value, float) and math.isfinite(value))


def _is_hex(value: object) -> bool:
    return isinstance(value, str) and _HEX64_RE.fullmatch(value) is not None


def _is_label(value: object) -> bool:
    return isinstance(value, str) and LABEL_RE.fullmatch(value) is not None


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.isascii() and value.isprintable()


def _join(where: str, key: object) -> str:
    return f"{where}.{key}" if where else str(key)


def _plain(value: Any, where: str) -> Any:
    """A plain-JSON copy of ``value``: mappings become dicts with string keys (integer keys become
    their text), tuples lists, numpy values their Python values (``tolist``). A non-finite number
    or a value JSON cannot hold is a :class:`LedgerError` naming where it is."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LedgerError(f"non-finite number at {where}")
        return float(value)
    if isinstance(value, Mapping):
        copy: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                text = key
            elif _is_int(key):
                text = str(key)
            else:
                raise LedgerError(f"a {type(key).__name__} key under {where}: keys are strings")
            copy[text] = _plain(item, _join(where, text))
        return copy
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{where}[{index}]") for index, item in enumerate(value)]
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and type(value).__module__.partition(".")[0] == "numpy":
        return _plain(tolist(), where)
    raise LedgerError(f"a {type(value).__name__} at {where} is not a JSON value")


def _check_values(value: Any, where: str) -> None:
    """Only JSON values; every string, keys included, printable ASCII; every number finite."""
    if isinstance(value, str):
        if not (value.isascii() and value.isprintable()):
            raise LedgerError(f"the string at {where} is not printable ASCII")
    elif value is None or isinstance(value, bool) or _is_int(value):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise LedgerError(f"non-finite number at {where}")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not (key.isascii() and key.isprintable()):
                raise LedgerError(f"a key under {where or 'the line'} is not printable ASCII")
            _check_values(item, _join(where, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_values(item, f"{where}[{index}]")
    else:
        raise LedgerError(f"a {type(value).__name__} at {where} is not a JSON value")


def _strings(value: Any, where: str) -> Iterator[tuple[str, str]]:
    """Every string in ``value``, keys included, with where it is."""
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield _join(where, key), key
            yield from _strings(item, _join(where, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{where}[{index}]")


def _hygiene_problem(text: str) -> str | None:
    """What the repository's hygiene scan of reports/ would flag in ``text`` (an email address,
    a personal path prefix, the cloud host name fragment, a credential assignment, an IPv4
    literal), or ``None``."""
    if _EMAIL_RE.search(text):
        return "an email address"
    if any(prefix in text for prefix in _PERSONAL_PATH_PREFIXES):
        return "a personal path"
    if _CLOUD_HOST_FRAGMENT in text.lower():
        return "a cloud host name"
    if _CREDENTIAL_RE.search(text):
        return "a credential assignment"
    if _IPV4_RE.search(text):
        return "an IPv4 literal"
    return None


def _check_publishable(document: Mapping[str, Any], *, personal: bool) -> None:
    """Refuse an IPv4 literal, a BingX host name, an absolute path, an email address, a personal
    path prefix (the home directories of macOS and Linux, anywhere in a string), the cloud host
    name fragment or a credential assignment anywhere in a line (keys included; the last four are
    the rules of the repository's hygiene scan of reports/, also checked on the whole canonical
    line, which is what that scan reads), and with ``personal`` (at append: the writer's
    machine) the login, host name and home directory of this process
    (``envinfo.assert_no_personal_data``): the ledger is published with the reports. The personal
    check depends on the machine that runs it, so it runs only where the line is written; the
    others are the same on every machine and run on read too."""
    from .exchange.public_market import PUBLIC_HOST  # the host literals live in exchange/

    if personal:
        from .envinfo import assert_no_personal_data

        try:
            assert_no_personal_data(document)
        except ValueError as exc:
            reason = str(exc).replace(" in report at ", " at ")
            raise LedgerRefused(f"{reason}: the ledger is published, so it holds none") from None
    domain = PUBLIC_HOST.split(".", 1)[1]
    for where, text in _strings(document, ""):
        if _IPV4_RE.search(text):
            raise LedgerRefused(
                f"IPv4 literal at {where}: the ledger is published, so it holds none"
            )
        if domain in text.lower():
            raise LedgerRefused(
                f"a BingX host name at {where}: the ledger names it 'BingX real market'"
            )
        if _PATH_TOKEN.search(text):
            raise LedgerRefused(
                f"an absolute path at {where}: the ledger is published, so it records paths "
                "relative to a root"
            )
        problem = _hygiene_problem(text)
        if problem is not None:
            raise LedgerRefused(
                f"{problem} at {where}: the ledger is published with the reports, whose "
                "hygiene scan refuses it (tests/test_repo_hygiene.py)"
            )
    problem = _hygiene_problem(canonical_json(dict(document)).decode("ascii"))
    if problem is not None:
        raise LedgerRefused(
            f"{problem} across the strings of the line: the ledger is published with the "
            "reports, whose hygiene scan reads whole lines (tests/test_repo_hygiene.py)"
        )


# -- the rules of each kind ------------------------------------------------------------------------


class _State:
    """What the entries before a line hold, for that line's references, counts and rules."""

    def __init__(self) -> None:
        self.approvals: set[int] = set()
        self.orders: dict[str, LedgerEntry] = {}
        self.order_of_candidate: dict[str, str] = {}
        self.preregistrations: dict[int, LedgerEntry] = {}
        self.bundle_of_recipe: dict[str, int] = {}
        self.bundle_relpaths: dict[str, int] = {}
        self.tests: dict[str, LedgerEntry] = {}
        self.week_counts: Counter[str] = Counter()
        self.quarter_counts: Counter[str] = Counter()
        self.used_orders: dict[str, int] = {}
        self.last: LedgerEntry | None = None  # the previous line: its ts bounds the next one's

    def add(self, entry: LedgerEntry) -> None:
        self.last = entry
        payload = entry.payload
        if entry.kind == "gate3_approval":
            self.approvals.add(entry.seq)
        elif entry.kind == "owner_order":
            self.orders[payload["order_id"]] = entry
            self.order_of_candidate[payload["candidate"]] = payload["order_id"]
        elif entry.kind == "preregistration":
            self.preregistrations[entry.seq] = entry
            self.bundle_of_recipe[payload["recipe_id"]] = entry.seq
            self.bundle_relpaths[payload["bundle_relpath"]] = entry.seq
        elif entry.kind == "test":
            self.tests[payload["test_id"]] = entry
            self.week_counts[payload["iso_week"]] += 1
            self.quarter_counts[payload["quarter"]] += 1
            if payload["order_id"] is not None:
                self.used_orders[payload["order_id"]] = entry.seq


def _malformed(kind: str, message: str) -> LedgerError:
    return LedgerError(f"{kind} payload: {message}")


def _check_details(kind: str, value: object, key: str) -> None:
    if not isinstance(value, dict):
        raise _malformed(kind, f"{key} must be a JSON object")
    size = len(canonical_json(value))
    if size > MAX_DETAILS_BYTES:
        raise _malformed(
            kind, f"{key} holds {size} bytes of canonical JSON, more than {MAX_DETAILS_BYTES}"
        )


def _segments(kind: str, payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload[key]
    if (
        not isinstance(value, list)
        or not all(item in SEGMENTS for item in value)
        or value != sorted(set(value))
    ):
        raise _malformed(
            kind, f"{key} must be a sorted list of distinct segments of {list(SEGMENTS)}"
        )
    return value


def _check_exploration(payload: Mapping[str, Any], state: _State, moment: datetime) -> None:
    kind = "exploration"
    if payload["activity"] not in EXPLORATION_ACTIVITIES:
        raise _malformed(kind, f"activity must be one of {list(EXPLORATION_ACTIVITIES)}")
    if not _is_label(payload["subject"]):
        raise _malformed(kind, f"subject must match {LABEL_RE.pattern}")
    if payload["stage"] not in EXPLORATION_STAGES:
        raise _malformed(kind, f"stage must be one of {list(EXPLORATION_STAGES)}")
    if payload["recipe_id"] is not None and not _is_hex(payload["recipe_id"]):
        raise _malformed(kind, "recipe_id must be a sha256 hex or null")
    inputs = _segments(kind, payload, "inputs")
    outcomes = _segments(kind, payload, "outcomes")
    if not inputs:
        raise _malformed(kind, "inputs must name the segments whose candles were read")
    _check_details(kind, payload["details"], "details")
    if any(segment not in EXPLORATION_INPUTS for segment in inputs):
        raise LedgerRefused(
            "an exploration never reads the sealed segment: opening it needs the owner's "
            "recorded order (gate 6)"
        )
    tested = [segment for segment in outcomes if segment not in EXPLORATION_OUTCOMES]
    if tested:
        raise LedgerRefused(
            f"reading {' and '.join(tested)} outcomes is a test, never exploration "
            "(docs/RETRAINING.md 7.2): it is read through the test budget, after the "
            "pre-registration bundle"
        )


def _check_fit(payload: Mapping[str, Any], state: _State, moment: datetime) -> None:
    kind = "fit"
    status = payload["status"]
    if status not in FIT_STATUSES:
        raise _malformed(kind, f"status must be one of {list(FIT_STATUSES)}")
    registered = status == "registered"
    for key in ("checkpoint_id", "registry_line_sha256"):
        if (payload[key] is not None) != registered:
            raise _malformed(
                kind, f"{key} is set exactly when the status is registered (status {status})"
            )
    checkpoint_id = payload["checkpoint_id"]
    if registered and (
        not isinstance(checkpoint_id, str)
        or len(checkpoint_id) > CHECKPOINT_ID_MAX_LENGTH
        or not CHECKPOINT_ID_RE.fullmatch(checkpoint_id)
    ):
        raise _malformed(
            kind,
            f"checkpoint_id must match {CHECKPOINT_ID_RE.pattern} in at most "
            f"{CHECKPOINT_ID_MAX_LENGTH} characters",
        )
    if registered and not _is_hex(payload["registry_line_sha256"]):
        raise _malformed(kind, "registry_line_sha256 must be a sha256 hex")
    if not _is_hex(payload["recipe_id"]):
        raise _malformed(kind, "recipe_id must be a sha256 hex")
    fold = payload["fold"]
    match = FOLD_LABEL_RE.fullmatch(fold) if isinstance(fold, str) else None
    if match is None or match.group("synthetic") is not None:
        raise _malformed(
            kind, "fold must be a planned fold label without its replicate suffix, not synthetic"
        )
    if not _is_int(payload["replicate"]) or payload["replicate"] < 0:
        raise _malformed(kind, "replicate must be an integer >= 0")
    if payload["run_id"] is not None and not _is_label(payload["run_id"]):
        raise _malformed(kind, f"run_id must match {LABEL_RE.pattern} or be null")
    if not _is_int(payload["attempt"]) or payload["attempt"] not in FIT_ATTEMPTS:
        raise _malformed(kind, f"attempt must be one of {list(FIT_ATTEMPTS)}")
    if not isinstance(payload["blinded"], bool):
        raise _malformed(kind, "blinded must be a boolean")
    if payload["fit_record_sha256"] is not None and not _is_hex(payload["fit_record_sha256"]):
        raise _malformed(kind, "fit_record_sha256 must be a sha256 hex or null")
    if payload["device"] not in FIT_DEVICES:
        raise _malformed(kind, f"device must be one of {list(FIT_DEVICES)}")
    if not _is_int(payload["threads"]) or payload["threads"] < 1:
        raise _malformed(kind, "threads must be a positive integer")
    for key in ("wall_seconds", "cpu_seconds"):
        if not _is_number(payload[key]) or payload[key] < 0:
            raise _malformed(kind, f"{key} must be a finite number >= 0")
    approval_seq = payload["approval_seq"]
    if approval_seq is not None and (not _is_int(approval_seq) or approval_seq < 1):
        raise _malformed(kind, "approval_seq must be a positive integer or null")
    order_id = payload["order_id"]
    if order_id is not None and (
        not isinstance(order_id, str) or not ORDER_ID_RE.fullmatch(order_id)
    ):
        raise _malformed(kind, f"order_id must match {ORDER_ID_RE.pattern} or be null")
    if approval_seq is not None and approval_seq not in state.approvals:
        raise LedgerRefused(f"fit: approval_seq {approval_seq} names no earlier gate3_approval")
    if order_id is not None and order_id not in state.orders:
        raise OrderNotFound(f"fit: no owner_order {order_id} is recorded before this entry")


def _check_gate3_approval(payload: Mapping[str, Any], state: _State, moment: datetime) -> None:
    kind = "gate3_approval"
    if payload["command"] not in APPROVAL_COMMANDS:
        raise _malformed(kind, f"command must be one of {list(APPROVAL_COMMANDS)}")
    if not _is_label(payload["subject"]):
        raise _malformed(kind, f"subject must match {LABEL_RE.pattern}")
    unit = payload["unit"]
    if unit not in APPROVAL_UNITS:
        raise _malformed(kind, f"unit must be one of {list(APPROVAL_UNITS)}")
    approved, estimate, ratio = payload["approved"], payload["estimate"], payload["ratio"]
    if not _is_number(approved) or approved <= 0:
        raise _malformed(kind, "approved must be a finite number > 0")
    if not _is_number(estimate) or estimate < 0:
        raise _malformed(kind, "estimate must be a finite number >= 0")
    if not _is_number(ratio) or ratio != estimate / approved:
        raise _malformed(kind, "ratio must be estimate / approved")
    if not _is_number(payload["tolerance"]) or payload["tolerance"] != APPROVAL_TOLERANCE:
        raise _malformed(kind, f"tolerance must be {APPROVAL_TOLERANCE}")
    if not _is_number(payload["gate3_hours"]) or payload["gate3_hours"] != GATE3_HOURS:
        raise _malformed(kind, f"gate3_hours must be {GATE3_HOURS}")
    if payload["above_gate3"] is not (estimate > GATE3_HOURS):
        raise _malformed(kind, "above_gate3 must be estimate > gate3_hours")
    try:
        parse_ts(payload["given_at"])
    except LedgerError:
        raise _malformed(kind, "given_at must be the UTC text of the command's start") from None
    _check_details(kind, payload["estimate_detail"], "estimate_detail")
    if ratio > APPROVAL_TOLERANCE:
        raise LedgerRefused(
            f"gate 3: the measured estimate of {estimate:g} {unit} is {ratio:.3f} times the "
            f"{approved:g} the owner approved, above the tolerance of {APPROVAL_TOLERANCE} "
            "(D-040, D-046): show the owner the estimate"
        )


def _check_preregistration(payload: Mapping[str, Any], state: _State, moment: datetime) -> None:
    kind = "preregistration"
    recipe_id = payload["recipe_id"]
    if not _is_hex(recipe_id):
        raise _malformed(kind, "recipe_id must be a sha256 hex")
    relpath = payload["bundle_relpath"]
    match = BUNDLE_RELPATH_RE.fullmatch(relpath) if isinstance(relpath, str) else None
    if match is None or match.group("recipe8") != recipe_id[:8]:
        raise _malformed(
            kind, f"bundle_relpath must be finetune/prereg-{recipe_id[:8]}.config.json"
        )
    if not _is_hex(payload["bundle_sha256"]):
        raise _malformed(kind, "bundle_sha256 must be a sha256 hex")
    tests = payload["tests"]
    if (
        not isinstance(tests, list)
        or not tests
        or not all(test in TEST_NAMES for test in tests)
        or tests != sorted(set(tests))
    ):
        raise _malformed(kind, f"tests must be a sorted non-empty subset of {list(TEST_NAMES)}")
    for key in ("forecasting_code_sha256", "training_code_sha256"):
        if not _is_hex(payload[key]):
            raise _malformed(kind, f"{key} must be a sha256 hex")
    earlier = state.bundle_of_recipe.get(recipe_id, state.bundle_relpaths.get(relpath))
    if earlier is not None:
        raise LedgerRefused(
            f"recipe {recipe_id[:8]} has its pre-registration bundle already (ledger entry "
            f"#{earlier}): a bundle is recorded once (docs/RETRAINING.md 7.1)"
        )


def _check_owner_order(payload: Mapping[str, Any], state: _State, moment: datetime) -> None:
    kind = "owner_order"
    expected = f"order-{len(state.orders) + 1:04d}"
    if payload["order_id"] != expected or not ORDER_ID_RE.fullmatch(expected):
        raise _malformed(kind, f"order_id must be {expected}, the next order id")
    if payload["scope"] not in ORDER_SCOPES:
        raise _malformed(kind, f"scope must be one of {list(ORDER_SCOPES)}")
    candidate = payload["candidate"]
    if not isinstance(candidate, str) or not TEST_ID_RE.fullmatch(candidate):
        raise _malformed(kind, f"candidate must be a test id ({TEST_ID_RE.pattern})")
    text = payload["text"]
    if not _is_text(text) or len(text) > MAX_ORDER_TEXT:
        raise _malformed(kind, f"text must be 1 to {MAX_ORDER_TEXT} printable ASCII characters")
    reading = state.tests.get(candidate)
    if reading is None or reading.payload["test"] != "T-B":
        raise LedgerRefused(
            f"no T-B test entry {candidate}: the owner orders the sealed test only for a T-B "
            "candidate whose reading is recorded (docs/RETRAINING.md 6.6)"
        )
    earlier = state.order_of_candidate.get(candidate)
    if earlier is not None:
        raise LedgerRefused(
            f"candidate {candidate} has an order already ({earlier}): the sealed segment opens "
            "once per candidate (D-048 (3))"
        )


def _check_test(payload: Mapping[str, Any], state: _State, moment: datetime) -> None:
    kind = "test"
    test, recipe_id = payload["test"], payload["recipe_id"]
    if test not in TEST_NAMES:
        raise _malformed(kind, f"test must be one of {list(TEST_NAMES)}")
    if not _is_hex(recipe_id):
        raise _malformed(kind, "recipe_id must be a sha256 hex")
    test_id = f"{test}-{recipe_id[:8]}"
    if payload["test_id"] != test_id:
        raise _malformed(kind, f"test_id must be {test_id}")
    relpath = payload["bundle_relpath"]
    if not isinstance(relpath, str) or not BUNDLE_RELPATH_RE.fullmatch(relpath):
        raise _malformed(kind, "bundle_relpath must name a pre-registration bundle")
    if not _is_hex(payload["bundle_sha256"]):
        raise _malformed(kind, "bundle_sha256 must be a sha256 hex")
    bundle_seq = payload["preregistration_seq"]
    if not _is_int(bundle_seq) or bundle_seq < 1:
        raise _malformed(kind, "preregistration_seq must be a positive integer")
    week, quarter = iso_week_key(moment), quarter_key(moment)
    if payload["iso_week"] != week or payload["quarter"] != quarter:
        raise _malformed(kind, f"iso_week and quarter must be {week} and {quarter}, those of ts")
    counts = (state.week_counts[week] + 1, state.quarter_counts[quarter] + 1)
    if (payload["week_count"], payload["quarter_count"]) != counts or not all(
        _is_int(payload[key]) for key in ("week_count", "quarter_count")
    ):
        raise _malformed(
            kind,
            f"week_count and quarter_count must be {counts[0]} and {counts[1]}: the test "
            f"entries of {week} and of {quarter}, this one included",
        )
    order_id = payload["order_id"]
    if order_id is not None and (
        not isinstance(order_id, str) or not ORDER_ID_RE.fullmatch(order_id)
    ):
        raise _malformed(kind, f"order_id must match {ORDER_ID_RE.pattern} or be null")
    bundle = state.preregistrations.get(bundle_seq)
    if bundle is None or bundle.payload["recipe_id"] != recipe_id:
        raise LedgerRefused(
            f"{test_id}: preregistration_seq {bundle_seq} is not the pre-registration entry of "
            f"recipe {recipe_id[:8]}"
        )
    if (relpath, payload["bundle_sha256"]) != (
        bundle.payload["bundle_relpath"],
        bundle.payload["bundle_sha256"],
    ):
        raise LedgerRefused(
            f"{test_id}: its bundle is not the one ledger entry #{bundle_seq} holds"
        )
    if test not in bundle.payload["tests"]:
        raise LedgerRefused(f"{test_id}: {test} is not registered in the bundle {relpath}")
    earlier = state.tests.get(test_id)
    if earlier is not None:
        raise LedgerRefused(
            f"{test_id} was read already (ledger entry #{earlier.seq}): a second reading of the "
            "same test id is refused"
        )
    if test != "T-C":
        if order_id is not None:
            raise LedgerRefused(f"{test_id}: only T-C reads on an order of the owner")
        if test == "T-B" and "T-A" in bundle.payload["tests"]:
            _check_t_a_first(test_id, recipe_id, state, moment)
    else:
        _check_t_c_order(test_id, recipe_id, order_id, state)
    week_count, quarter_count = counts
    if week_count > TESTS_PER_ISO_WEEK or quarter_count > TESTS_PER_QUARTER:
        raise LedgerRefused(
            f"{test_id} would be test {week_count} of ISO week {week} and {quarter_count} of "
            f"{quarter}: at most {TESTS_PER_ISO_WEEK} in an ISO week and {TESTS_PER_QUARTER} in "
            "a quarter (SPEC rule 8)"
        )


def _check_t_a_first(test_id: str, recipe_id: str, state: _State, moment: datetime) -> None:
    """A bundle that registers T-A and T-B reads T-A first, in an earlier ISO week (design 6.4,
    7.2)."""
    first = state.tests.get(f"T-A-{recipe_id[:8]}")
    if first is None or first.payload["recipe_id"] != recipe_id:
        raise LedgerRefused(
            f"{test_id}: the bundle registers T-A and T-B, and T-A is read first "
            "(docs/RETRAINING.md 6.4)"
        )
    if week_number(first.moment) >= week_number(moment):
        raise LedgerRefused(
            f"{test_id}: T-A was read in ISO week {iso_week_key(first.moment)} (ledger entry "
            f"#{first.seq}); T-B is read in a later ISO week (docs/RETRAINING.md 7.2)"
        )


def _check_t_c_order(test_id: str, recipe_id: str, order_id: str | None, state: _State) -> None:
    """T-C reads on the owner's recorded order for its recipe's T-B candidate, used once."""
    if order_id is None:
        raise LedgerRefused(f"{test_id}: T-C reads the sealed segment only on the owner's order")
    order = state.orders.get(order_id)
    if order is None:
        raise OrderNotFound(f"{test_id}: no owner_order {order_id} is recorded before it")
    candidate = f"T-B-{recipe_id[:8]}"
    if order.payload["candidate"] != candidate:
        raise LedgerRefused(
            f"{test_id}: order {order_id} is for candidate {order.payload['candidate']}, not for "
            f"{candidate}"
        )
    used = state.used_orders.get(order_id)
    if used is not None:
        raise LedgerRefused(
            f"{test_id}: order {order_id} was used by the T-C reading of ledger entry #{used}"
        )


_KIND_CHECKS: Mapping[str, Callable[[Mapping[str, Any], _State, datetime], None]] = {
    "exploration": _check_exploration,
    "fit": _check_fit,
    "gate3_approval": _check_gate3_approval,
    "owner_order": _check_owner_order,
    "preregistration": _check_preregistration,
    "test": _check_test,
}


def _check_code(value: object) -> None:
    if not isinstance(value, dict) or sorted(value) != sorted(CODE_KEYS):
        raise LedgerError(f"code must be an object with exactly {list(CODE_KEYS)}")
    if value["commit"] is not None and not _is_text(value["commit"]):
        raise LedgerError("code.commit must be a string or null")
    if value["dirty"] is not None and not isinstance(value["dirty"], bool):
        raise LedgerError("code.dirty must be a boolean or null")
    if not _is_hex(value["source_tree_sha256"]):
        raise LedgerError("code.source_tree_sha256 must be a sha256 hex")


def _key_problem(found: Iterable[str], expected: Sequence[str], what: str) -> str | None:
    missing = sorted(set(expected) - set(found))
    unknown = sorted(set(found) - set(expected))
    if not missing and not unknown:
        return None
    return f"{what} misses {missing} and has unknown keys {unknown}"


def _check_time(moment: datetime, state: _State) -> None:
    """A line's ``ts`` is never earlier than the previous line's (:class:`LedgerRefused`): the
    budget counts tests by the week of their ``ts``, so a reading dated back (a backdated
    ``now``, a clock set back) could escape a full week."""
    last = state.last
    if last is not None and moment < last.moment:
        raise LedgerRefused(
            f"ts {format_ts(moment)} is earlier than {last.ts}, the ts of the previous entry "
            f"#{last.seq}: the ledger's times never decrease (a backdated time, or a system "
            "clock set back: append once the clock has passed it)"
        )


def _check_document(
    document: dict[str, Any], *, seq: int, previous: str, state: _State, appending: bool
) -> None:
    """Raise :class:`LedgerError` (malformed) or :class:`LedgerRefused` (a rule) unless
    ``document`` is a valid line number ``seq`` after a line that hashes to ``previous``, given
    the entries before it (``state``). ``appending``: the line is being written on this machine,
    so its writer's personal data is checked too (never on read: it depends on the reader)."""
    problem = _key_problem(document, LINE_KEYS, "the line")
    if problem:
        raise LedgerError(problem)
    _check_values(document, "")
    if document["schema"] != LEDGER_SCHEMA:
        raise LedgerError(f"schema must be {LEDGER_SCHEMA!r}")
    version = document["schema_version"]
    if not _is_int(version) or version != LEDGER_SCHEMA_VERSION:
        raise LedgerError(f"unknown schema_version {version!r}")
    if not _is_int(document["seq"]) or document["seq"] != seq:
        raise LedgerError(f"seq {document['seq']!r} is not the line number {seq}")
    if document["prev_sha256"] != previous:
        raise LedgerError("prev_sha256 is not the sha256 of the previous line: the chain is broken")
    moment = parse_ts(document["ts"])
    _check_time(moment, state)
    kind = document["kind"]
    if kind not in KINDS:
        raise LedgerError(f"unknown kind {kind!r}")
    params_version = document["params_version"]
    if not isinstance(params_version, str) or not _HEX12_RE.fullmatch(params_version):
        raise LedgerError("params_version must be 12 lowercase hex characters")
    _check_code(document["code"])
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise LedgerError(f"the {kind} payload must be a JSON object")
    problem = _key_problem(payload, PAYLOAD_KEYS[kind], f"the {kind} payload")
    if problem:
        raise LedgerError(problem)
    _KIND_CHECKS[kind](payload, state, moment)
    _check_publishable(document, personal=appending)


# -- reading ---------------------------------------------------------------------------------------


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a finite JSON number")


def _entry(line: bytes, document: dict[str, Any]) -> LedgerEntry:
    return LedgerEntry(
        line=line,
        line_sha256=sha256_bytes(line),
        seq=document["seq"],
        kind=document["kind"],
        ts=document["ts"],
        params_version=document["params_version"],
        code=types.MappingProxyType(document["code"]),
        payload=types.MappingProxyType(document["payload"]),
        document=types.MappingProxyType(document),
    )


def _entry_from_line(line: bytes, number: int, previous: str, state: _State) -> LedgerEntry:
    """The entry of line ``number`` (1-based) after a line that hashes to ``previous``;
    :class:`LedgerChainError` naming the line for any problem."""

    def broken(reason: str) -> LedgerChainError:
        return LedgerChainError(f"ledger line {number}: {reason}")

    if len(line) > MAX_LINE_BYTES:
        raise broken(f"longer than {MAX_LINE_BYTES} bytes")
    try:
        document = json.loads(line.decode("ascii"), parse_constant=_refuse_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise broken(f"not JSON ({exc})") from None
    if not isinstance(document, dict):
        raise broken("not a JSON object")
    if canonical_json(document) != line:
        raise broken("not canonical JSON")
    try:
        _check_document(document, seq=number, previous=previous, state=state, appending=False)
    except (LedgerError, LedgerRefused) as exc:
        raise broken(str(exc)) from None
    return _entry(line, document)


def _scan(data: bytes) -> tuple[list[LedgerEntry], _State, LedgerChainError | None]:
    """The entries of ``data`` that verify, in order, what they hold, and the first problem
    (``None``: every line verifies)."""
    entries: list[LedgerEntry] = []
    state = _State()
    if not data:
        return entries, state, None
    *complete, tail = data.split(b"\n")
    previous = GENESIS_SHA256
    for number, line in enumerate(complete, 1):
        try:
            entry = _entry_from_line(line, number, previous, state)
        except LedgerChainError as exc:
            return entries, state, exc
        state.add(entry)
        entries.append(entry)
        previous = entry.line_sha256
    if tail:
        torn = LedgerChainError(
            f"ledger line {len(complete) + 1}: no LF at the end of the file (a torn write); "
            "the ledger is never repaired"
        )
        return entries, state, torn
    return entries, state, None


def _read(base: Path) -> tuple[bytes, tuple[LedgerEntry, ...], _State]:
    path = base / LEDGER_NAME
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return b"", (), _State()
    except OSError as exc:
        raise LedgerError(f"cannot read {_label(base)}: {exc.strerror}") from None
    entries, state, problem = _scan(data)
    if problem is not None:
        raise LedgerChainError(f"{_label(base)}: {problem}")
    return data, tuple(entries), state


def read_ledger(root: Path | None = None) -> tuple[LedgerEntry, ...]:
    """Every entry of the ledger, verified; a missing file is an empty ledger (nothing is created).

    Every line must end with LF (a torn last line is refused, never repaired), be the canonical
    JSON of an object with exactly :data:`LINE_KEYS`, carry its line number as ``seq``, the
    sha256 of the previous line (:data:`GENESIS_SHA256` for the first) and a ``ts`` no earlier
    than the previous line's, a known kind and a valid payload whose references point at earlier
    entries of the right kind (a ``test`` within the budget and after its T-A reading where its
    bundle asks for one), and hold nothing that may not be published. The writer's personal data
    was checked when the line was appended; the reader's is never compared, so a ledger verifies
    the same on every machine. Any problem is a :class:`LedgerChainError` naming the line.
    """
    return _read(_root(root))[1]


def ledger_status(root: Path | None = None) -> dict[str, Any]:
    """``{"entries", "head", "chain_ok", "problem"}`` of the ledger, without raising for a chain
    problem: ``entries`` and ``head`` describe the lines that verify (all of them when
    ``chain_ok``; the prefix before the first broken line otherwise, ``problem`` naming it)."""
    base = _root(root)
    status: dict[str, Any] = {
        "entries": 0,
        "head": GENESIS_SHA256,
        "chain_ok": True,
        "problem": None,
    }
    try:
        data = (base / LEDGER_NAME).read_bytes()
    except FileNotFoundError:
        data = b""
    except OSError as exc:
        data = b""
        status.update(chain_ok=False, problem=f"cannot read {_label(base)}: {exc.strerror}")
    entries, _, problem = _scan(data)
    if problem is not None:
        status.update(chain_ok=False, problem=str(problem))
    status["entries"] = len(entries)
    if entries:
        status["head"] = entries[-1].line_sha256
    return status


def entries_of(
    kind: str, *, root: Path | None = None, entries: Sequence[LedgerEntry] | None = None
) -> tuple[LedgerEntry, ...]:
    """The entries of ``kind`` in ``entries`` (default: :func:`read_ledger` of ``root``)."""
    if kind not in KINDS:
        raise LedgerError(f"kind must be one of {list(KINDS)}, got {kind!r}")
    source = read_ledger(root) if entries is None else entries
    return tuple(entry for entry in source if entry.kind == kind)


# -- appending -------------------------------------------------------------------------------------


def _params_version() -> str:
    from .params import params_version

    return params_version()


def append_entry(
    kind: str,
    payload: Mapping[str, Any] | Callable[[tuple[LedgerEntry, ...]], Mapping[str, Any]],
    *,
    root: Path | None = None,
    now: datetime | None = None,
) -> LedgerEntry:
    """Append one ``kind`` entry and return it, as read back.

    Under the ledger's exclusive lock (``LockBusyError`` when another append holds it): read and
    verify every line (:class:`LedgerChainError`), call ``payload(entries)`` when it is callable
    (it sees the entries read under the lock and may raise :class:`LedgerRefused`: nothing is
    appended then), copy the payload as plain JSON, build the line (``ts`` from ``now``, an aware
    datetime for tests, default the current time; ``params_version``; ``code``), validate it as
    a reader would, plus the writer's personal data (:class:`LedgerError` for a malformed
    payload, :class:`LedgerRefused` for a rule, among them a ``ts`` earlier than the last line's)
    and replace the file atomically with the old bytes plus the line.
    """
    if kind not in KINDS:
        raise LedgerError(f"kind must be one of {list(KINDS)}, got {kind!r}")
    if not callable(payload) and not isinstance(payload, Mapping):
        raise LedgerError(f"a payload is a mapping or a callable, got {type(payload).__name__}")
    if now is not None:
        _aware(now, "now")
    base = _root(root)
    with exclusive_lock(base / LOCK_NAME):
        old, entries, state = _read(base)
        content = payload(entries) if callable(payload) else payload
        if not isinstance(content, Mapping):
            raise LedgerError(f"the {kind} payload must be a mapping, got {type(content).__name__}")
        moment = _aware(now, "now") if now is not None else datetime.now(UTC)
        previous = entries[-1].line_sha256 if entries else GENESIS_SHA256
        seq = len(entries) + 1
        document = {
            "schema": LEDGER_SCHEMA,
            "schema_version": LEDGER_SCHEMA_VERSION,
            "seq": seq,
            "ts": format_ts(moment),
            "prev_sha256": previous,
            "kind": kind,
            "params_version": _params_version(),
            "code": _plain(provenance.code_version(), "code"),
            "payload": _plain(content, "payload"),
        }
        _check_document(document, seq=seq, previous=previous, state=state, appending=True)
        line = canonical_json(document)
        if len(line) > MAX_LINE_BYTES:
            raise LedgerError(
                f"the {kind} line would hold {len(line)} bytes, more than {MAX_LINE_BYTES}"
            )
        atomic_write_bytes(base / LEDGER_NAME, old + line + b"\n")
        written = _read(base)[1]
    if not written or written[-1].line != line:
        raise LedgerChainError(f"{_label(base)} does not end with the line just appended")
    return written[-1]


def _names(values: object, what: str) -> list[str]:
    """``values`` (an iterable of names, never a single string) as a sorted list of distinct
    names."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise LedgerError(f"{what} must be a list of names, got {values!r}")
    items = list(values)
    if not all(isinstance(item, str) for item in items):
        raise LedgerError(f"{what} must be a list of names, got {items!r}")
    return sorted(set(items))


def _hours(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise LedgerError(f"{what} must be a finite number, got {value!r}")
    hours = float(value)
    if not math.isfinite(hours):
        raise LedgerError(f"{what} must be a finite number, got {value!r}")
    return hours


def record_exploration(
    *,
    activity: str,
    subject: str,
    stage: str,
    recipe_id: str | None,
    inputs: Iterable[str],
    outcomes: Iterable[str],
    details: Mapping[str, Any],
    root: Path | None = None,
    now: datetime | None = None,
) -> LedgerEntry:
    """Append an ``exploration`` entry (0 tests): ``inputs`` are the segments whose candles were
    read as inputs (``training``, ``validation``), ``outcomes`` the segments whose realised
    candles after a decision hour were read (at most ``training``: :class:`LedgerRefused`
    otherwise); both are stored sorted and distinct. ``details`` is a publishable JSON object
    of at most :data:`MAX_DETAILS_BYTES` bytes."""
    payload = {
        "activity": activity,
        "subject": subject,
        "stage": stage,
        "recipe_id": recipe_id,
        "inputs": _names(inputs, "inputs"),
        "outcomes": _names(outcomes, "outcomes"),
        "details": details,
    }
    return append_entry("exploration", payload, root=root, now=now)


def record_fit(
    *,
    checkpoint_id: str | None,
    registry_line_sha256: str | None,
    recipe_id: str,
    fold: str,
    replicate: int,
    run_id: str | None,
    status: str,
    attempt: int,
    blinded: bool,
    fit_record_sha256: str | None,
    device: str,
    threads: int,
    wall_seconds: float,
    cpu_seconds: float,
    approval_seq: int | None,
    order_id: str | None,
    root: Path | None = None,
    now: datetime | None = None,
) -> LedgerEntry:
    """Append a ``fit`` entry: one training child of a recipe run. ``checkpoint_id`` and
    ``registry_line_sha256`` are set exactly when ``status`` is ``registered``; ``approval_seq``
    names an earlier ``gate3_approval`` and ``order_id`` an earlier ``owner_order``, or are
    null. It never holds a loss, an NLL value or a selected pseudo-epoch."""
    payload = {
        "checkpoint_id": checkpoint_id,
        "registry_line_sha256": registry_line_sha256,
        "recipe_id": recipe_id,
        "fold": fold,
        "replicate": replicate,
        "run_id": run_id,
        "status": status,
        "attempt": attempt,
        "blinded": blinded,
        "fit_record_sha256": fit_record_sha256,
        "device": device,
        "threads": threads,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "approval_seq": approval_seq,
        "order_id": order_id,
    }
    return append_entry("fit", payload, root=root, now=now)


def record_gate3_approval(
    *,
    command: str,
    subject: str,
    unit: str,
    approved: float,
    estimate: float,
    given_at: datetime,
    estimate_detail: Mapping[str, Any],
    root: Path | None = None,
    now: datetime | None = None,
) -> LedgerEntry:
    """Append a ``gate3_approval`` entry: ``approved`` is the figure of ``--approved-by-owner``
    (the hours the owner approved in chat), ``estimate`` the measured estimate shown, both in
    ``unit``; ``given_at`` is the command's start. ``ratio`` (``estimate / approved``),
    ``tolerance``, ``gate3_hours`` and ``above_gate3`` are derived here; a ratio above
    :data:`APPROVAL_TOLERANCE` is refused (:class:`LedgerRefused`)."""
    approved_hours = _hours(approved, "approved")
    estimate_hours = _hours(estimate, "estimate")
    if approved_hours <= 0:
        raise LedgerError(f"approved must be a finite number > 0, got {approved!r}")
    payload = {
        "command": command,
        "subject": subject,
        "unit": unit,
        "approved": approved_hours,
        "estimate": estimate_hours,
        "ratio": estimate_hours / approved_hours,
        "tolerance": APPROVAL_TOLERANCE,
        "gate3_hours": GATE3_HOURS,
        "above_gate3": estimate_hours > GATE3_HOURS,
        "given_at": format_ts(given_at),
        "estimate_detail": estimate_detail,
    }
    return append_entry("gate3_approval", payload, root=root, now=now)


def record_preregistration(
    *,
    bundle_relpath: str,
    recipe_id: str,
    tests: Iterable[str],
    forecasting_code_sha256: str,
    training_code_sha256: str,
    root: Path | None = None,
    reports_root: Path | None = None,
    now: datetime | None = None,
) -> LedgerEntry:
    """Append the ``preregistration`` entry of a bundle already written at ``bundle_relpath``
    under the reports root (``finetune/prereg-<recipe8>.config.json``), with its sha256 taken
    under the ledger lock. One bundle per recipe: a second is refused (:class:`LedgerRefused`);
    a missing file is a :class:`LedgerError`."""
    if not _is_hex(recipe_id):
        raise LedgerError(f"recipe_id must be a sha256 hex, got {recipe_id!r}")
    expected = f"finetune/prereg-{recipe_id[:8]}.config.json"
    if bundle_relpath != expected:
        raise LedgerError(
            f"the bundle of recipe {recipe_id[:8]} is {expected}, not {bundle_relpath!r}"
        )
    names = _names(tests, "tests")
    base = Path(reports_root) if reports_root is not None else reports_dir()

    def build(entries: tuple[LedgerEntry, ...]) -> dict[str, Any]:
        earlier = preregistration_for(recipe_id, entries=entries)
        if earlier is not None:
            raise LedgerRefused(
                f"recipe {recipe_id[:8]} has its pre-registration bundle already (ledger entry "
                f"#{earlier.seq}): a bundle is recorded once (docs/RETRAINING.md 7.1)"
            )
        path = base / bundle_relpath
        if not path.is_file():
            raise LedgerError(f"the bundle {bundle_relpath} does not exist under the reports root")
        return {
            "bundle_relpath": bundle_relpath,
            "bundle_sha256": sha256_file(path),
            "recipe_id": recipe_id,
            "tests": names,
            "forecasting_code_sha256": forecasting_code_sha256,
            "training_code_sha256": training_code_sha256,
        }

    return append_entry("preregistration", build, root=root, now=now)


def record_order(
    *,
    scope: str,
    candidate: str,
    text: str,
    root: Path | None = None,
    now: datetime | None = None,
) -> LedgerEntry:
    """Append the owner's order (``owner_order``; gate 6), only after the owner's explicit order
    in chat; ``text`` is their words translated into English. The id ``order-NNNN`` is the next
    one, assigned under the lock. Refused (:class:`LedgerRefused`): a candidate without a
    recorded T-B test entry, a second order for the same candidate."""
    if scope not in ORDER_SCOPES:
        raise LedgerError(f"scope must be one of {list(ORDER_SCOPES)}, got {scope!r}")
    if not isinstance(candidate, str) or not TEST_ID_RE.fullmatch(candidate):
        raise LedgerError(f"candidate must be a test id ({TEST_ID_RE.pattern}), got {candidate!r}")
    if not _is_text(text) or len(text) > MAX_ORDER_TEXT:
        raise LedgerError(
            f"the order text is 1 to {MAX_ORDER_TEXT} printable ASCII characters (the owner's "
            "words translated into English)"
        )

    def build(entries: tuple[LedgerEntry, ...]) -> dict[str, Any]:
        reading = next(
            (
                entry
                for entry in entries_of("test", entries=entries)
                if entry.payload["test_id"] == candidate
            ),
            None,
        )
        if reading is None or reading.payload["test"] != "T-B":
            raise LedgerRefused(
                f"no T-B test entry for the candidate {candidate}: the owner orders the sealed "
                "test only for a T-B candidate whose reading is recorded (docs/RETRAINING.md 6.6)"
            )
        orders = entries_of("owner_order", entries=entries)
        for order in orders:
            if order.payload["candidate"] == candidate:
                raise LedgerRefused(
                    f"candidate {candidate} has an order already ({order.payload['order_id']}, "
                    f"ledger entry #{order.seq}): the sealed segment opens once per candidate "
                    "(D-048 (3))"
                )
        return {
            "order_id": f"order-{len(orders) + 1:04d}",
            "scope": scope,
            "candidate": candidate,
            "text": text,
        }

    return append_entry("owner_order", build, root=root, now=now)


# -- lookups ---------------------------------------------------------------------------------------


def find_order(
    order_id: str, *, scope: str = ORDER_SCOPE_OPEN_SEALED, root: Path | None = None
) -> LedgerEntry:
    """The ``owner_order`` entry ``order_id`` of ``scope`` (:class:`OrderNotFound` otherwise)."""
    if not isinstance(order_id, str) or not ORDER_ID_RE.fullmatch(order_id):
        raise OrderNotFound(
            f"{order_id!r} is not an order id (order-NNNN, as 'ledger record-order' prints it)"
        )
    for entry in entries_of("owner_order", root=root):
        if entry.payload["order_id"] == order_id:
            if entry.payload["scope"] != scope:
                raise OrderNotFound(
                    f"order {order_id} has scope {entry.payload['scope']}, not {scope!r}"
                )
            return entry
    raise OrderNotFound(
        f"no owner_order {order_id} of scope {scope!r} is recorded in {_label(_root(root))}"
    )


def verify_open_order(
    order_id: str, *, scope: str = ORDER_SCOPE_OPEN_SEALED, root: Path | None = None
) -> LedgerEntry:
    """The owner's order ``order_id`` to open the sealed segment, for the sealed guard
    (``data.partition``): an ``owner_order`` entry of ``scope`` in a ledger whose every line
    verifies (:class:`LedgerChainError` otherwise, :class:`OrderNotFound` without the entry).

    An order is not refused after its T-C reading: the same order also covers the sealed folds,
    the sealed fans and the model to date (D-048 (3)). "Once" is enforced where it matters: one
    order per candidate (:func:`record_order`) and one T-C reading per order
    (``budget.spend_test``); the commands that use an order check that its candidate is their
    recipe's T-B test id.
    """
    if scope not in ORDER_SCOPES:
        raise OrderNotFound(f"unknown order scope {scope!r}; the scopes are {list(ORDER_SCOPES)}")
    return find_order(order_id, scope=scope, root=root)


def order_used(
    order_id: str, *, root: Path | None = None, entries: Sequence[LedgerEntry] | None = None
) -> bool:
    """True when a T-C ``test`` entry names ``order_id`` (the order's sealed reading is spent)."""
    return any(
        entry.payload["order_id"] == order_id
        for entry in entries_of("test", root=root, entries=entries)
    )


def preregistration_for(
    recipe_id: str, *, root: Path | None = None, entries: Sequence[LedgerEntry] | None = None
) -> LedgerEntry | None:
    """The ``preregistration`` entry of ``recipe_id``, or ``None`` when its bundle is not
    recorded."""
    for entry in entries_of("preregistration", root=root, entries=entries):
        if entry.payload["recipe_id"] == recipe_id:
            return entry
    return None


def verify_bundle(entry: LedgerEntry, *, reports_root: Path | None = None) -> None:
    """The bundle file of the ``preregistration`` entry ``entry`` must hash to its recorded
    ``bundle_sha256`` (:class:`BundleChanged` when it is missing, unreadable or changed)."""
    if not isinstance(entry, LedgerEntry) or entry.kind != "preregistration":
        raise LedgerError("verify_bundle takes a preregistration entry of the ledger")
    relpath = entry.payload["bundle_relpath"]
    recorded = entry.payload["bundle_sha256"]
    base = Path(reports_root) if reports_root is not None else reports_dir()
    try:
        digest = sha256_file(base / relpath)
    except FileNotFoundError:
        raise BundleChanged(
            f"the pre-registration bundle {relpath} of ledger entry #{entry.seq} is missing: a "
            "recorded bundle is never removed or changed (docs/RETRAINING.md 7.1)"
        ) from None
    except OSError as exc:
        raise BundleChanged(
            f"the pre-registration bundle {relpath} of ledger entry #{entry.seq} cannot be read "
            f"({exc.strerror})"
        ) from None
    if digest != recorded:
        raise BundleChanged(
            f"the pre-registration bundle {relpath} changed after ledger entry #{entry.seq} "
            f"recorded it (sha256 {recorded[:12]}..., now {digest[:12]}...): a recorded bundle "
            "is never changed (docs/RETRAINING.md 7.1)"
        )


def verify_bundles(
    *, root: Path | None = None, reports_root: Path | None = None
) -> tuple[LedgerEntry, ...]:
    """Every ``preregistration`` entry, after checking that its bundle file still hashes to the
    recorded sha256 (:class:`BundleChanged` otherwise). Every ``finetune`` subcommand calls it
    first (design 7.1)."""
    bundles = entries_of("preregistration", root=root)
    for entry in bundles:
        verify_bundle(entry, reports_root=reports_root)
    return bundles


def id_for_test(test: str, recipe_id: str) -> str:
    """``<test>-<recipe8>``, the id of a test of a recipe (``T-B-0ee0d123``)."""
    if test not in TEST_NAMES:
        raise LedgerError(f"test must be one of {list(TEST_NAMES)}, got {test!r}")
    if not _is_hex(recipe_id):
        raise LedgerError(f"recipe_id must be a sha256 hex, got {recipe_id!r}")
    return f"{test}-{recipe_id[:8]}"


__all__ = [
    "APPROVAL_COMMANDS",
    "APPROVAL_TOLERANCE",
    "APPROVAL_UNITS",
    "BUNDLE_RELPATH_RE",
    "CHECKPOINT_ID_MAX_LENGTH",
    "CHECKPOINT_ID_RE",
    "CODE_KEYS",
    "DATA_LINK_DEPTH",
    "EXPLORATION_ACTIVITIES",
    "EXPLORATION_INPUTS",
    "EXPLORATION_OUTCOMES",
    "EXPLORATION_STAGES",
    "FIT_ATTEMPTS",
    "FIT_DEVICES",
    "FIT_STATUSES",
    "FOLD_LABEL_RE",
    "GATE3_HOURS",
    "GENESIS_SHA256",
    "KINDS",
    "LEDGER_NAME",
    "LEDGER_SCHEMA",
    "LEDGER_SCHEMA_VERSION",
    "LINE_KEYS",
    "LOCK_NAME",
    "MAX_DETAILS_BYTES",
    "MAX_LINE_BYTES",
    "MAX_ORDER_TEXT",
    "ORDER_ID_RE",
    "ORDER_SCOPES",
    "ORDER_SCOPE_OPEN_SEALED",
    "PAYLOAD_KEYS",
    "SEGMENTS",
    "TESTS_PER_ISO_WEEK",
    "TESTS_PER_QUARTER",
    "TEST_ID_RE",
    "TEST_NAMES",
    "TS_FORMAT",
    "BundleChanged",
    "LedgerChainError",
    "LedgerEntry",
    "LedgerError",
    "LedgerRefused",
    "OrderNotFound",
    "append_entry",
    "binding_problem",
    "entries_of",
    "find_order",
    "format_ts",
    "id_for_test",
    "iso_week_key",
    "ledger_path",
    "ledger_status",
    "order_used",
    "parse_ts",
    "preregistration_for",
    "quarter_key",
    "read_ledger",
    "record_exploration",
    "record_fit",
    "record_gate3_approval",
    "record_order",
    "record_preregistration",
    "require_binding",
    "verify_bundle",
    "verify_bundles",
    "verify_open_order",
    "week_number",
]
