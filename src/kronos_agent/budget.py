"""The test budget (SPEC rule 8; ``docs/RETRAINING.md``, 7.2 and 7.3): at most
:data:`TESTS_PER_ISO_WEEK` tests in an ISO week and :data:`TESTS_PER_QUARTER` in a calendar
quarter, both in UTC, over every recipe, enforced by the command that reads a test.

A test is any statistic that compares a configuration's forecasts with realised outcomes on hours
reserved for evaluation (design 7.2): T-A (Kronos as shipped on validation), T-B (the recipe's
walk-forward) and T-C (the sealed test, only on the owner's recorded order). :func:`spend_test`
appends the ``test`` entry of the experiment ledger BEFORE any statistic on reserved hours is
computed, so "compute, peek, abandon" still spends the test. Its checks run under the ledger's
lock against the entries read there, and the entry is appended only if all of them pass:

0. the ledger may authorize reading the data root whose outcomes the reading reads
   (``ledger.binding_problem``: a ledger other than the repository's ``reports/ledger`` never
   spends a test on the repository's ``data/``);
1. a known test and a recipe id;
2. the recipe's pre-registration bundle is recorded (a test without a matching bundle is refused),
   its file is unchanged and it registers the test;
3. the test id was never read (a second reading of the same id is refused);
4. the order of tests: when the bundle registers T-A and T-B, T-B follows a T-A reading of the
   same recipe in an earlier ISO week (T-A is read first, design 6.4); T-C follows the recipe's
   T-B reading and names the owner's ``open_sealed`` order for that T-B candidate, used by no
   other T-C reading; only T-C names an order;
5. the budget: refused when the ISO week of ``now`` already holds 3 tests or its quarter 15.

The ``test`` entry records its ISO week and quarter and how many tests each holds with it. The
ledger itself refuses a ``now`` earlier than its last line's ``ts`` (a reading cannot be dated
back into a week with room), and checks rules 4 and 5 again on every read, as the ``test`` lines
record them. Changing ``goals.yaml`` also counts as a test (SPEC rule 8); no ``goals.yaml``
exists in this build, so nothing spends a test for it yet. This module imports the ledger and the
standard library only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ledger import (
    ORDER_ID_RE,
    ORDER_SCOPE_OPEN_SEALED,
    TESTS_PER_ISO_WEEK,
    TESTS_PER_QUARTER,
    LedgerEntry,
    LedgerError,
    LedgerRefused,
    append_entry,
    entries_of,
    format_ts,
    id_for_test,
    iso_week_key,
    preregistration_for,
    quarter_key,
    require_binding,
    verify_bundle,
    week_number,
)


class BudgetExceeded(LedgerRefused):
    """A 4th test in an ISO week or a 16th in a calendar quarter (SPEC rule 8)."""


class ReadingRefused(LedgerRefused):
    """A reading the bundle, the registration, a second reading or the order rules refuse."""


@dataclass(frozen=True)
class BudgetStatus:
    """The tests spent in the ISO week and the quarter of ``at`` (UTC text), over all recipes."""

    at: str
    iso_week: str
    week_used: int
    week_limit: int
    quarter: str
    quarter_used: int
    quarter_limit: int

    @property
    def remaining(self) -> int:
        """The tests that may still be read now: the smaller of the week's and the quarter's."""
        return min(self.week_limit - self.week_used, self.quarter_limit - self.quarter_used)


def _moment(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise LedgerError(f"a budget time is a timezone-aware datetime, got {value!r}")
    return value.astimezone(UTC)


def budget_status(
    *,
    at: datetime | None = None,
    root: Path | None = None,
    entries: Sequence[LedgerEntry] | None = None,
) -> BudgetStatus:
    """The tests spent in the ISO week and the calendar quarter of ``at`` (default now), counted
    over every ``test`` entry of ``entries`` (default: the ledger of ``root``)."""
    moment = _moment(at)
    week, quarter = iso_week_key(moment), quarter_key(moment)
    tests = entries_of("test", root=root, entries=entries)
    return BudgetStatus(
        at=format_ts(moment),
        iso_week=week,
        week_used=sum(1 for entry in tests if iso_week_key(entry.moment) == week),
        week_limit=TESTS_PER_ISO_WEEK,
        quarter=quarter,
        quarter_used=sum(1 for entry in tests if quarter_key(entry.moment) == quarter),
        quarter_limit=TESTS_PER_QUARTER,
    )


def require_budget(status: BudgetStatus) -> None:
    """Refuse (:class:`BudgetExceeded`) when the week or the quarter of ``status`` is spent."""
    if status.week_used >= status.week_limit or status.quarter_used >= status.quarter_limit:
        raise BudgetExceeded(
            f"the test budget is spent: {status.week_used} of {status.week_limit} tests in ISO "
            f"week {status.iso_week} and {status.quarter_used} of {status.quarter_limit} in "
            f"{status.quarter} (SPEC rule 8); no test is read until the budget allows it"
        )


def _reading(entries: Sequence[LedgerEntry], test_id: str) -> LedgerEntry | None:
    for entry in entries_of("test", entries=entries):
        if entry.payload["test_id"] == test_id:
            return entry
    return None


def _check_order_of_tests(
    test: str,
    recipe_id: str,
    order_id: str | None,
    bundle: LedgerEntry,
    entries: Sequence[LedgerEntry],
    moment: datetime,
) -> None:
    """Rule 4 of the module docstring (:class:`ReadingRefused`)."""
    test_id = id_for_test(test, recipe_id)
    if test != "T-C":
        if order_id is not None:
            raise ReadingRefused(f"{test_id}: only T-C reads on an order of the owner")
        if test == "T-B" and "T-A" in bundle.payload["tests"]:
            first = _reading(entries, id_for_test("T-A", recipe_id))
            if first is None:
                raise ReadingRefused(
                    f"{test_id}: the bundle registers T-A and T-B, and T-A is read first "
                    "(docs/RETRAINING.md 6.4)"
                )
            if week_number(first.moment) >= week_number(moment):
                raise ReadingRefused(
                    f"{test_id}: T-A was read in ISO week {iso_week_key(first.moment)} (ledger "
                    f"entry #{first.seq}); T-B is read in a later ISO week (docs/RETRAINING.md "
                    "7.2)"
                )
        return
    candidate = id_for_test("T-B", recipe_id)
    if _reading(entries, candidate) is None:
        raise ReadingRefused(f"{test_id}: T-C follows the T-B reading {candidate}, not recorded")
    if order_id is None:
        raise ReadingRefused(
            f"{test_id}: T-C reads the sealed segment only on the owner's recorded order (gate 6)"
        )
    order = next(
        (
            entry
            for entry in entries_of("owner_order", entries=entries)
            if entry.payload["order_id"] == order_id
        ),
        None,
    )
    if order is None or order.payload["scope"] != ORDER_SCOPE_OPEN_SEALED:
        raise ReadingRefused(
            f"{test_id}: no {ORDER_SCOPE_OPEN_SEALED} order {order_id} is recorded in the ledger"
        )
    if order.payload["candidate"] != candidate:
        raise ReadingRefused(
            f"{test_id}: order {order_id} is the owner's order for {order.payload['candidate']}, "
            f"not for {candidate}"
        )
    for entry in entries_of("test", entries=entries):
        if entry.payload["order_id"] == order_id:
            raise ReadingRefused(
                f"{test_id}: order {order_id} was used by the T-C reading of ledger entry "
                f"#{entry.seq}: the sealed segment opens once per candidate"
            )


def spend_test(
    test: str,
    *,
    recipe_id: str,
    order_id: str | None = None,
    root: Path | None = None,
    reports_root: Path | None = None,
    data_root: Path | None = None,
    now: datetime | None = None,
) -> LedgerEntry:
    """Spend one test: append the ``test`` entry of ``test`` for ``recipe_id`` once rules 0 to 5
    of the module docstring pass under the ledger lock, and return it.

    Call it before computing any statistic on reserved hours. ``order_id`` is T-C's only (the
    owner's recorded order); ``reports_root`` holds the bundle files (default the reports root);
    ``data_root`` is the data root whose fans and candles the reading reads (default the data
    root); ``now`` (an aware datetime) is for tests, and the ledger refuses one earlier than its
    last line. Refusals: ``ledger.require_binding`` (a ledger that may not authorize
    ``data_root``), :class:`ReadingRefused`, ``BundleChanged``, :class:`BudgetExceeded` and a
    backdated ``now`` (all ``LedgerRefused``); a busy lock is a ``LockBusyError``.
    """
    test_id = id_for_test(test, recipe_id)  # LedgerError for an unknown test or recipe id
    if order_id is not None and (
        not isinstance(order_id, str) or not ORDER_ID_RE.fullmatch(order_id)
    ):
        raise LedgerError(f"order_id must match {ORDER_ID_RE.pattern}, got {order_id!r}")
    moment = _moment(now)
    require_binding(root=root, data_root=data_root)

    def build(entries: tuple[LedgerEntry, ...]) -> dict[str, Any]:
        bundle = preregistration_for(recipe_id, entries=entries)
        if bundle is None:
            raise ReadingRefused(
                f"{test_id}: a test without a matching bundle: no pre-registration entry for "
                f"recipe {recipe_id[:8]} (docs/RETRAINING.md 7.1)"
            )
        verify_bundle(bundle, reports_root=reports_root)
        relpath, registered = bundle.payload["bundle_relpath"], bundle.payload["tests"]
        if test not in registered:
            raise ReadingRefused(
                f"{test_id}: {test} is not registered in the bundle {relpath} (it registers "
                f"{', '.join(registered)})"
            )
        earlier = _reading(entries, test_id)
        if earlier is not None:
            raise ReadingRefused(
                f"{test_id} was read already (ledger entry #{earlier.seq} at {earlier.ts}): a "
                "second reading of the same test id is refused"
            )
        _check_order_of_tests(test, recipe_id, order_id, bundle, entries, moment)
        status = budget_status(at=moment, entries=entries)
        require_budget(status)
        return {
            "test_id": test_id,
            "test": test,
            "recipe_id": recipe_id,
            "bundle_relpath": bundle.payload["bundle_relpath"],
            "bundle_sha256": bundle.payload["bundle_sha256"],
            "preregistration_seq": bundle.seq,
            "iso_week": status.iso_week,
            "quarter": status.quarter,
            "week_count": status.week_used + 1,
            "quarter_count": status.quarter_used + 1,
            "order_id": order_id if test == "T-C" else None,
        }

    return append_entry("test", build, root=root, now=moment)


__all__ = [
    "TESTS_PER_ISO_WEEK",
    "TESTS_PER_QUARTER",
    "BudgetExceeded",
    "BudgetStatus",
    "ReadingRefused",
    "budget_status",
    "iso_week_key",
    "quarter_key",
    "require_budget",
    "spend_test",
]
