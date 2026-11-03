"""``kronos-agent ledger <subcommand>``: the experiment ledger of the retraining (step 6;
``docs/RETRAINING.md`` 7.3; :mod:`kronos_agent.ledger`).

- ``show [--kind KIND]``: one line per entry, ``#<seq> <ts> <kind> <summary>`` (an exploration's
  details are not printed), then the ledger's size and head (the sha256 of its last line) and the
  tests spent in the current ISO week and calendar quarter.
- ``verify``: the hash chain and every line, then every pre-registration bundle against the
  sha256 its entry recorded.
- ``record-order --scope open_sealed --candidate TEST_ID --text "<words>"``: the owner's order to
  open the sealed segment for the T-B candidate ``TEST_ID`` (gate 6). Run it only after the
  owner's explicit order in chat; the text is their words translated into English. It prints the
  order id that the sealed commands take, the entry's seq and its line sha256.

Exit codes (D-039): 0 done; 1 failure (``ERROR:``): a ledger that cannot be read or whose chain is
broken (never repaired); 2 refusal (``REFUSED:``): a changed or missing pre-registration bundle,
an order whose candidate has no recorded T-B reading or has an order already, personal data in
the order text, a clock earlier than the ledger's last entry, a busy ledger lock. argparse usage
errors exit 2 as well (a candidate that is not a test id, an order text that is not 1 to 1,000
printable ASCII characters).

At load this module imports only what ``kronos_agent.cli`` allows: the choices and the test-id
pattern are literals here, and the tests assert that they equal the ledger's. The ledger needs no
runtime configuration: nothing here imports torch.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from . import EXIT_OK, ExceptionTypes, display_path, run_guarded

if TYPE_CHECKING:
    from ..ledger import LedgerEntry

# literals of kronos_agent.ledger (KINDS, ORDER_SCOPES, TEST_ID_RE, MAX_ORDER_TEXT)
KINDS: tuple[str, ...] = (
    "exploration",
    "fit",
    "gate3_approval",
    "owner_order",
    "preregistration",
    "test",
)
ORDER_SCOPES: tuple[str, ...] = ("open_sealed",)
TEST_ID_RE = re.compile(r"T-[ABC]-[0-9a-f]{8}")
MAX_ORDER_TEXT = 1000


def _say(message: str) -> None:
    print(message, flush=True)


def _now() -> datetime:
    """The UTC clock of the command (a module function, so tests replace it)."""
    return datetime.now(UTC)


def _errors() -> tuple[ExceptionTypes, ExceptionTypes]:
    """``(refusals, failures)`` of every ``ledger`` subcommand (imported lazily)."""
    from ..data.fsutil import LockBusyError
    from ..ledger import LedgerError, LedgerRefused
    from ..params import ParamsError
    from ..provenance import ProvenanceError

    return (LedgerRefused, LockBusyError), (LedgerError, ParamsError, ProvenanceError)


def _guarded(action: Callable[[], int]) -> int:
    refusals, failures = _errors()
    return run_guarded(action, refusals=refusals, failures=failures)


# -- argparse types --------------------------------------------------------------------------------


def candidate_id(text: str) -> str:
    """argparse type of a test id, ``T-<A|B|C>-<recipe8>`` (never named ``test...``: pytest
    would collect it from a test module that imports it)."""
    if not TEST_ID_RE.fullmatch(text):
        raise argparse.ArgumentTypeError(
            f"a test id matches {TEST_ID_RE.pattern} (the candidate is T-B-<recipe8>), got {text!r}"
        )
    return text


def order_text(text: str) -> str:
    """argparse type of the owner's words: 1 to 1,000 printable ASCII characters."""
    if not text or len(text) > MAX_ORDER_TEXT or not text.isascii() or not text.isprintable():
        raise argparse.ArgumentTypeError(
            f"the order text is 1 to {MAX_ORDER_TEXT:,} printable ASCII characters: the owner's "
            "words translated into English"
        )
    return text


# -- show ------------------------------------------------------------------------------------------


def _summary(entry: LedgerEntry, used_orders: set[str]) -> str:
    from ..budget import TESTS_PER_ISO_WEEK, TESTS_PER_QUARTER

    payload: Any = entry.payload
    if entry.kind == "exploration":
        return f"{payload['activity']} {payload['subject']} {payload['stage']}"
    if entry.kind == "fit":
        run = payload["fold"]
        if payload["replicate"]:
            run = f"{run}-r{payload['replicate']}"
        return (
            f"{payload['checkpoint_id'] or '-'} {payload['status']} {run} attempt "
            f"{payload['attempt']} blinded {str(payload['blinded']).lower()}"
        )
    if entry.kind == "gate3_approval":
        return (
            f"{payload['command']} {payload['subject']}: approved {payload['approved']:g} "
            f"{payload['unit']}, estimate {payload['estimate']:.4g} (ratio "
            f"{payload['ratio']:.3f}), given at {payload['given_at']}"
        )
    if entry.kind == "preregistration":
        return (
            f"{payload['bundle_relpath']} sha256 {payload['bundle_sha256'][:12]} recipe "
            f"{payload['recipe_id'][:8]} tests {','.join(payload['tests'])}"
        )
    if entry.kind == "test":
        return (
            f"{payload['test_id']} week {payload['iso_week']} ({payload['week_count']}/"
            f"{TESTS_PER_ISO_WEEK}) quarter {payload['quarter']} ({payload['quarter_count']}/"
            f"{TESTS_PER_QUARTER})"
        )
    read = "yes" if payload["order_id"] in used_orders else "no"
    return (
        f"{payload['order_id']} {payload['scope']} candidate {payload['candidate']} (T-C read: "
        f'{read}): "{payload["text"]}"'
    )


def _show_run(kind: str | None) -> int:
    from .. import budget, ledger

    entries = ledger.read_ledger()
    location = display_path(ledger.ledger_path())
    if not entries:
        _say(f"ledger {location}: no entries")
        return EXIT_OK
    used_orders = {
        entry.payload["order_id"]
        for entry in ledger.entries_of("test", entries=entries)
        if entry.payload["order_id"] is not None
    }
    for entry in entries:
        if kind is None or entry.kind == kind:
            _say(f"#{entry.seq} {entry.ts} {entry.kind} {_summary(entry, used_orders)}")
    status = budget.budget_status(at=_now(), entries=entries)
    _say(
        f"ledger {location}: {len(entries)} entries, head {entries[-1].line_sha256}; tests in "
        f"{status.iso_week} {status.week_used}/{status.week_limit}, in {status.quarter} "
        f"{status.quarter_used}/{status.quarter_limit}"
    )
    return EXIT_OK


def _show(args: argparse.Namespace) -> int:
    return _guarded(lambda: _show_run(args.kind))


# -- verify ----------------------------------------------------------------------------------------


def _verify_run() -> int:
    from .. import ledger

    entries = ledger.read_ledger()  # every line and the chain (LedgerChainError: exit 1)
    location = display_path(ledger.ledger_path())
    if not entries:
        _say(f"ledger {location}: no entries, no bundle to verify")
        return EXIT_OK
    _say(
        f"ledger {location}: {len(entries)} entries verified, chain intact, head "
        f"{entries[-1].line_sha256}"
    )
    bundles = ledger.entries_of("preregistration", entries=entries)
    for entry in bundles:
        ledger.verify_bundle(entry)  # BundleChanged: exit 2
        _say(
            f"bundle {entry.payload['bundle_relpath']} sha256 "
            f"{entry.payload['bundle_sha256'][:12]} unchanged (ledger entry #{entry.seq})"
        )
    _say(f"bundles verified: {len(bundles)}")
    return EXIT_OK


def _verify(args: argparse.Namespace) -> int:
    return _guarded(_verify_run)


# -- record-order ----------------------------------------------------------------------------------


def _record_order_run(args: argparse.Namespace) -> int:
    from .. import ledger

    entry = ledger.record_order(
        scope=args.scope, candidate=args.candidate, text=args.text, now=_now()
    )
    _say(
        f"recorded {entry.payload['order_id']} ({entry.payload['scope']}, candidate "
        f"{entry.payload['candidate']}): ledger entry #{entry.seq}, line sha256 "
        f"{entry.line_sha256}"
    )
    _say(f"ledger {display_path(ledger.ledger_path())}: {entry.seq} entries")
    return EXIT_OK


def _record_order(args: argparse.Namespace) -> int:
    return _guarded(lambda: _record_order_run(args))


# -- the command -----------------------------------------------------------------------------------

SUBCOMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "show": _show,
    "verify": _verify,
    "record-order": _record_order,
}


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "ledger",
        help="the experiment ledger (docs/RETRAINING.md 7.3): show, verify, record the owner's "
        "order to open the sealed segment",
    )
    actions = parser.add_subparsers(dest="ledger_command", required=True)
    show_help = "one line per entry, then the head and the tests spent this week and quarter"
    show = actions.add_parser("show", help=show_help, description=show_help)
    show.add_argument("--kind", choices=KINDS, default=None, help="only the entries of this kind")
    verify_help = "verify every line, the hash chain and every pre-registration bundle"
    actions.add_parser("verify", help=verify_help, description=verify_help)
    order_help = (
        "record an order only after the owner's explicit order in chat (gate 6); the text is "
        "their words translated into English"
    )
    order = actions.add_parser("record-order", help=order_help, description=order_help)
    order.add_argument(
        "--scope",
        required=True,
        choices=ORDER_SCOPES,
        help="what the order opens: open_sealed, the sealed segment for one candidate",
    )
    order.add_argument(
        "--candidate",
        required=True,
        type=candidate_id,
        metavar="TEST_ID",
        help="the T-B test id the order is for (T-B-<recipe8>; its reading must be recorded)",
    )
    order.add_argument(
        "--text",
        required=True,
        type=order_text,
        help="the owner's words translated into English (1 to 1,000 printable ASCII characters)",
    )


def run(args: argparse.Namespace) -> int:
    return SUBCOMMANDS[args.ledger_command](args)
