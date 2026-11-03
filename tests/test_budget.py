"""The test budget (``kronos_agent.budget``; SPEC rule 8; ``docs/RETRAINING.md`` 7.2 and 7.3; test
T27): at most 3 tests in an ISO week and 15 in a calendar quarter (UTC), over every recipe.

``spend_test`` checks, under the ledger lock and in this order, the bundle (recorded, unchanged,
registering the test), a second reading of the same id, the order of the readings (T-B in a later
ISO week than T-A; T-C on the owner's order for its T-B candidate, used once) and the budget, and
appends the ``test`` entry only when all pass: a refusal appends nothing. The week and quarter
boundaries are checked on the ISO-year boundary (2026-W53 spans 2026-Q4 and 2027-Q1) and on a
quarter boundary inside one week (2026-W40). Every line records a stand-in code provenance
(``ledger_fakes.fixed_code``): these tests append many lines.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, timezone

import ledger_fakes as lf
import pytest

from kronos_agent import budget, ledger, paths
from kronos_agent.budget import BudgetExceeded, BudgetStatus, ReadingRefused
from kronos_agent.data.fsutil import LockBusyError, exclusive_lock
from kronos_agent.ledger import BundleChanged, LedgerError, LedgerRefused

SECOND = timedelta(seconds=1)


@pytest.fixture(autouse=True)
def stand_in_code(monkeypatch):
    lf.fixed_code(monkeypatch)


def at(day: date | str, hour: int = 12, minute: int = 0, second: int = 0) -> datetime:
    day = date.fromisoformat(day) if isinstance(day, str) else day
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=UTC)


def ledger_bytes() -> bytes:
    return ledger.ledger_path().read_bytes()


def recipes(count: int, tests: tuple[str, ...] = ("T-A",), *, prefix: str = "r") -> list[str]:
    """``count`` stand-in recipes, each with a recorded bundle registering ``tests``, recorded
    at 2026-07-01 or, after it, at the time of the ledger's last entry (times never decrease)."""
    found = [lf.recipe_id(f"{prefix}{index}") for index in range(count)]
    for recipe in found:
        lf.register_bundle(recipe, tests, now=max(at("2026-07-01"), lf.last_moment()))
    return found


def counts(entry: ledger.LedgerEntry) -> tuple[str, int, str, int]:
    payload = entry.payload
    return payload["iso_week"], payload["week_count"], payload["quarter"], payload["quarter_count"]


# -- weeks and quarters ----------------------------------------------------------------------------


def test_iso_weeks_and_quarters_of_the_utc_date():
    for offset in range(7):  # Monday 2026-12-28 to Sunday 2027-01-03: one ISO week
        day = date(2026, 12, 28) + timedelta(days=offset)
        assert budget.iso_week_key(at(day)) == "2026-W53"
        assert budget.quarter_key(at(day)) == ("2026-Q4" if day.year == 2026 else "2027-Q1")
    assert budget.iso_week_key(at("2027-01-04")) == "2027-W01"
    for day, quarter in (("2026-09-28", "2026-Q3"), ("2026-10-01", "2026-Q4")):
        assert (budget.iso_week_key(at(day)), budget.quarter_key(at(day))) == ("2026-W40", quarter)
    assert (budget.iso_week_key(at("2026-11-03")), budget.quarter_key(at("2026-11-03"))) == (
        "2026-W45",
        "2026-Q4",
    )
    # the UTC date counts, not the local one: Sunday 22:00 at UTC-3 is Monday 01:00 UTC
    evening = datetime(2026, 11, 1, 22, tzinfo=timezone(timedelta(hours=-3)))
    assert budget.iso_week_key(evening) == "2026-W45" and evening.date().isoformat() == "2026-11-01"
    assert budget.iso_week_key is ledger.iso_week_key and budget.quarter_key is ledger.quarter_key
    with pytest.raises(LedgerError, match="timezone-aware"):
        budget.iso_week_key(datetime(2026, 11, 3))


def test_budget_status_counts_every_recipe_and_requires_room():
    empty = budget.budget_status(at=at("2026-11-03"))
    assert empty == BudgetStatus(
        at="2026-11-03T12:00:00Z",
        iso_week="2026-W45",
        week_used=0,
        week_limit=3,
        quarter="2026-Q4",
        quarter_used=0,
        quarter_limit=15,
    )
    assert empty.remaining == 3
    budget.require_budget(empty)
    full_week = BudgetStatus("x", "2026-W45", 3, 3, "2026-Q4", 3, 15)
    full_quarter = BudgetStatus("x", "2026-W46", 0, 3, "2026-Q4", 15, 15)
    assert (full_week.remaining, full_quarter.remaining) == (0, 0)
    for status in (full_week, full_quarter):
        with pytest.raises(BudgetExceeded, match="the test budget is spent"):
            budget.require_budget(status)
    assert (budget.TESTS_PER_ISO_WEEK, budget.TESTS_PER_QUARTER) == (3, 15)
    assert budget.budget_status().quarter_limit == 15  # now, by default
    with pytest.raises(LedgerError, match="timezone-aware"):
        budget.budget_status(at=datetime(2026, 11, 3))


# -- the budget ------------------------------------------------------------------------------------


def test_three_tests_in_an_iso_week_then_the_fourth_is_refused():
    r0, r1, r2, r3 = recipes(4)
    days = (at("2026-11-02", 0), at("2026-11-04"), at("2026-11-08", 23, 59, 59))
    spent = [budget.spend_test("T-A", recipe_id=r, now=d) for r, d in zip((r0, r1, r2), days)]
    assert [counts(entry) for entry in spent] == [
        ("2026-W45", 1, "2026-Q4", 1),
        ("2026-W45", 2, "2026-Q4", 2),
        ("2026-W45", 3, "2026-Q4", 3),
    ]
    status = budget.budget_status(at=days[-1])
    assert (status.week_used, status.quarter_used, status.remaining) == (3, 3, 0)
    before = ledger_bytes()
    message = "3 of 3 tests in ISO week 2026-W45 and 3 of 15 in 2026-Q4"
    with pytest.raises(BudgetExceeded, match=re.escape(message)):
        budget.spend_test("T-A", recipe_id=r3, now=at("2026-11-08", 23, 59, 59))
    assert ledger_bytes() == before  # a refusal appends nothing
    entry = budget.spend_test("T-A", recipe_id=r3, now=at("2026-11-09", 0))  # the next week
    assert counts(entry) == ("2026-W46", 1, "2026-Q4", 4)
    assert entry.payload["test_id"] == f"T-A-{r3[:8]}" and entry.payload["order_id"] is None
    assert budget.budget_status(at=at("2026-11-09")).remaining == 2


def test_fifteen_tests_in_a_quarter_then_the_sixteenth_is_refused():
    found = recipes(17)
    mondays = [at("2026-10-05") + timedelta(weeks=week) for week in range(5)]  # W41 to W45
    for index, recipe in enumerate(found[:15]):
        moment = mondays[index // 3] + timedelta(days=index % 3)
        entry = budget.spend_test("T-A", recipe_id=recipe, now=moment)
        assert entry.payload["quarter_count"] == index + 1
        assert entry.payload["week_count"] == index % 3 + 1
    before = ledger_bytes()
    message = "0 of 3 tests in ISO week 2026-W46 and 15 of 15 in 2026-Q4"
    with pytest.raises(BudgetExceeded, match=re.escape(message)):
        budget.spend_test("T-A", recipe_id=found[15], now=at("2026-11-12"))
    with pytest.raises(BudgetExceeded):
        budget.spend_test("T-A", recipe_id=found[15], now=at("2026-12-31", 23, 59, 59))
    assert ledger_bytes() == before
    entry = budget.spend_test("T-A", recipe_id=found[15], now=at("2027-01-04", 0))
    assert counts(entry) == ("2027-W01", 1, "2027-Q1", 1)


def test_the_iso_year_boundary_week_spans_two_quarters():
    r0, r1, r2, r3 = recipes(4)
    first = budget.spend_test("T-A", recipe_id=r0, now=at("2026-12-29"))
    second = budget.spend_test("T-A", recipe_id=r1, now=at("2027-01-02"))
    third = budget.spend_test("T-A", recipe_id=r2, now=at("2027-01-03", 23, 59, 59))
    assert [counts(entry) for entry in (first, second, third)] == [
        ("2026-W53", 1, "2026-Q4", 1),
        ("2026-W53", 2, "2027-Q1", 1),
        ("2026-W53", 3, "2027-Q1", 2),
    ]
    with pytest.raises(BudgetExceeded, match="3 of 3 tests in ISO week 2026-W53 and 2 of 15"):
        budget.spend_test("T-A", recipe_id=r3, now=at("2027-01-03", 23, 59, 59))
    fourth = budget.spend_test("T-A", recipe_id=r3, now=at("2027-01-04", 0))
    assert counts(fourth) == ("2027-W01", 1, "2027-Q1", 3)


def test_a_quarter_boundary_inside_one_iso_week():
    r0, r1, r2, r3 = recipes(4)
    spent = [
        budget.spend_test("T-A", recipe_id=r0, now=at("2026-09-28", 0)),
        budget.spend_test("T-A", recipe_id=r1, now=at("2026-09-30", 23, 59, 59)),
        budget.spend_test("T-A", recipe_id=r2, now=at("2026-10-01", 0)),
    ]
    assert [counts(entry) for entry in spent] == [
        ("2026-W40", 1, "2026-Q3", 1),
        ("2026-W40", 2, "2026-Q3", 2),
        ("2026-W40", 3, "2026-Q4", 1),
    ]
    with pytest.raises(BudgetExceeded, match="3 of 3 tests in ISO week 2026-W40 and 1 of 15 in"):
        budget.spend_test("T-A", recipe_id=r3, now=at("2026-10-04", 23, 59, 59))
    september = budget.budget_status(at=at("2026-09-29"))
    assert (september.week_used, september.quarter, september.quarter_used) == (3, "2026-Q3", 2)


def test_seeded_test_entries_count_through_the_ledger():
    """Entries appended directly must carry the counts the ledger implies; spend_test counts
    them with its own."""
    (recipe,) = recipes(1, ("T-A", "T-B"))
    bundle = ledger.preregistration_for(recipe)
    moment = at("2026-11-03")
    payload = {
        "test_id": f"T-A-{recipe[:8]}",
        "test": "T-A",
        "recipe_id": recipe,
        "bundle_relpath": bundle.payload["bundle_relpath"],
        "bundle_sha256": bundle.payload["bundle_sha256"],
        "preregistration_seq": bundle.seq,
        "iso_week": "2026-W45",
        "quarter": "2026-Q4",
        "week_count": 2,
        "quarter_count": 1,
        "order_id": None,
    }
    with pytest.raises(LedgerError, match="week_count and quarter_count must be 1 and 1"):
        ledger.append_entry("test", payload, now=moment)
    with pytest.raises(LedgerError, match="iso_week and quarter must be 2026-W46 and 2026-Q4"):
        ledger.append_entry("test", {**payload, "week_count": 1}, now=moment + timedelta(days=6))
    seeded = ledger.append_entry("test", {**payload, "week_count": 1}, now=moment)
    assert counts(seeded) == ("2026-W45", 1, "2026-Q4", 1)
    with pytest.raises(LedgerRefused, match="was read already"):
        ledger.append_entry("test", {**payload, "week_count": 2, "quarter_count": 2}, now=moment)
    later = budget.spend_test("T-B", recipe_id=recipe, now=at("2026-11-10"))
    assert counts(later) == ("2026-W46", 1, "2026-Q4", 2)


# -- what a reading needs --------------------------------------------------------------------------


def test_the_bundle_must_be_recorded_unchanged_and_register_the_test():
    recipe = lf.recipe_id("lonely")
    with pytest.raises(ReadingRefused, match="a test without a matching bundle"):
        budget.spend_test("T-A", recipe_id=recipe, now=at("2026-11-03"))
    lf.register_bundle(recipe, ("T-A",), now=at("2026-11-03"))
    before = ledger_bytes()
    with pytest.raises(
        ReadingRefused, match=r"T-B is not registered in the bundle .* registers T-A"
    ):
        budget.spend_test("T-B", recipe_id=recipe, now=at("2026-11-03"))
    bundle = paths.reports_dir() / lf.bundle_relpath(recipe)
    original = bundle.read_bytes()
    bundle.write_bytes(original + b"\n")
    with pytest.raises(BundleChanged, match="changed after ledger entry #1 recorded it"):
        budget.spend_test("T-A", recipe_id=recipe, now=at("2026-11-03"))
    assert ledger_bytes() == before
    bundle.write_bytes(original)
    first = budget.spend_test("T-A", recipe_id=recipe, now=at("2026-11-03"))
    before = ledger_bytes()
    with pytest.raises(ReadingRefused, match=f"T-A-{recipe[:8]} was read already .ledger entry #2"):
        budget.spend_test("T-A", recipe_id=recipe, now=at("2026-11-17"))
    assert ledger_bytes() == before and first.seq == 2
    with pytest.raises(LedgerError, match="test must be one of"):
        budget.spend_test("T-D", recipe_id=recipe)
    with pytest.raises(LedgerError, match="recipe_id must be a sha256 hex"):
        budget.spend_test("T-A", recipe_id=recipe[:8])
    with pytest.raises(LedgerError, match="order_id must match"):
        budget.spend_test("T-C", recipe_id=recipe, order_id="order-1")
    with pytest.raises(LedgerError, match="timezone-aware"):
        budget.spend_test("T-A", recipe_id=recipe, now=datetime(2026, 11, 3))


def test_t_b_is_read_in_a_later_iso_week_than_t_a():
    (recipe,) = recipes(1, ("T-A", "T-B"))
    with pytest.raises(ReadingRefused, match="the bundle registers T-A and T-B, and T-A is read"):
        budget.spend_test("T-B", recipe_id=recipe, now=at("2026-11-02"))
    budget.spend_test("T-A", recipe_id=recipe, now=at("2026-11-02", 0))
    before = ledger_bytes()
    with pytest.raises(ReadingRefused, match="T-A was read in ISO week 2026-W45"):
        budget.spend_test("T-B", recipe_id=recipe, now=at("2026-11-08", 23, 59, 59))
    assert ledger_bytes() == before
    entry = budget.spend_test("T-B", recipe_id=recipe, now=at("2026-11-09", 0))
    assert counts(entry) == ("2026-W46", 1, "2026-Q4", 2)
    (alone,) = recipes(1, ("T-B",), prefix="t-b-only")  # a bundle without T-A: no order
    assert budget.spend_test("T-B", recipe_id=alone, now=at("2026-11-09")).seq == 5


def test_t_c_reads_on_the_owner_order_for_its_t_b_candidate_once():
    x, y = lf.recipe_id("x"), lf.recipe_id("y")
    order_x = lf.recorded_order(x, start=at("2026-11-02"))  # T-A in W45, T-B in W46
    order_y = lf.recorded_order(y, start=at("2026-11-09"))  # T-A in W46, T-B in W47
    assert (order_x.payload["order_id"], order_y.payload["order_id"]) == (
        "order-0001",
        "order-0002",
    )
    (z,) = recipes(1, prefix="z")
    moment = at("2026-11-17")
    before = ledger_bytes()
    for order_id, message in [
        (None, "T-C reads the sealed segment only on the owner's recorded order"),
        ("order-0002", f"order order-0002 is the owner's order for T-B-{y[:8]}, not for"),
        ("order-0009", "no open_sealed order order-0009 is recorded"),
    ]:
        with pytest.raises(ReadingRefused, match=re.escape(message)):
            budget.spend_test("T-C", recipe_id=x, order_id=order_id, now=moment)
    with pytest.raises(ReadingRefused, match="only T-C reads on an order of the owner"):
        budget.spend_test("T-A", recipe_id=z, order_id="order-0001", now=moment)
    assert ledger_bytes() == before
    t_c = budget.spend_test("T-C", recipe_id=x, order_id="order-0001", now=moment)
    assert t_c.payload["order_id"] == "order-0001" and ledger.order_used("order-0001")
    with pytest.raises(ReadingRefused):  # the order is used and the id was read
        budget.spend_test("T-C", recipe_id=x, order_id="order-0001", now=moment)
    (lone,) = recipes(1, ("T-C",), prefix="lone")
    with pytest.raises(ReadingRefused, match="T-C follows the T-B reading"):
        budget.spend_test("T-C", recipe_id=lone, order_id="order-0001", now=moment)


def test_a_reading_while_another_append_holds_the_lock_is_refused():
    (recipe,) = recipes(1)
    before = ledger_bytes()
    with exclusive_lock(paths.ledger_dir() / ledger.LOCK_NAME), pytest.raises(LockBusyError):
        budget.spend_test("T-A", recipe_id=recipe, now=at("2026-11-03"))
    assert ledger_bytes() == before
    assert issubclass(BudgetExceeded, ledger.LedgerRefused)
    assert issubclass(ReadingRefused, ledger.LedgerRefused)
