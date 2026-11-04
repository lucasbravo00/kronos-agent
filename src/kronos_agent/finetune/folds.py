"""The fold plan of the walk-forward (step 6; ``docs/RETRAINING.md`` 1.2 to 1.4).

A fold is one fine-tune and the test month (or quarter) that starts at ``T``, 00:00Z on the 1st.
Its CSV holds the Binance candles opening in ``[history_start, T - 512 h)`` (the per-model test
embargo of D-048 (6)); the CSV's last 1,344 hours are its inner validation, where NLL4-TF early
stops on the 4-hour grid of decisions ``D`` whose 4 targets open before the CSV end. Layouts:

- ``monthly``: ``wf-YYYY-MM``, test months 2024-07 to 2026-06 (24 folds), the layout the
  walk-forward does not run (D-056: its measured worst case passes the 72 h of D-048 (1));
- ``quarterly``: ``q-YYYY-MM``, the compute fallback over the same test decisions (8 folds) and
  the walk-forward's layout (D-056); ``q-2024-07`` is the pilot, whose CSV ends at
  2024-06-09T16:00Z, the last candle a training-segment sample may read (``wf-2024-07``'s CSV);
- ``sealed``: ``s-YYYY-MM``, test months from 2026-08 (only after the owner's recorded order);
- ``final``: ``final-YYYY-MM-DD``, the model to date: its CSV ends at ``T_final`` (00:00Z of a
  day, from 2026-08-01) and no test follows;
- ``synthetic``: the measurement step and tests, on synthetic candles.

Grids (design "Conventions"): ``G4`` = ``D.hour % 4 == 0``; ``G8`` = ``D.hour % 8 == 0``; ``G25`` =
hours whose count since 2020-01-01T00:00Z is divisible by 25. :func:`check_fold` enforces the
boundaries of design 1.4: a partition rule broken raises ``PartitionError``, anything else
:class:`FoldError`. Every planner checks the folds it returns.

Standard library only at module level (with the project's light modules): no numpy or pandas.
"""

from __future__ import annotations

import numbers
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from ..data.partition import (
    EVAL_LENGTH_MAX,
    MAX_CONTEXT,
    SEALED_START,
    SEALED_START_MS,
    TRAINING_START,
    PartitionError,
    sample_fits,
)
from ..data.timeutil import HOUR_MS, from_ms, parse_utc_hour, to_ms, utc_text
from .recipe import R1, Recipe

LAYOUTS: tuple[str, ...] = ("monthly", "quarterly", "sealed", "final", "synthetic")
GRIDS: tuple[str, ...] = ("G4", "G8", "G25")
G25_ORIGIN_MS: int = to_ms(TRAINING_START)
_GRID_HOURS: dict[str, int] = {"G4": 4, "G8": 8, "G25": 25}


def _add_months(start: datetime, months: int) -> datetime:
    index = start.year * 12 + (start.month - 1) + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


_WALK_FORWARD_START = datetime(2024, 7, 1, tzinfo=UTC)
MONTHLY_TEST_STARTS: tuple[datetime, ...] = tuple(
    _add_months(_WALK_FORWARD_START, k) for k in range(24)
)
QUARTERLY_TEST_STARTS: tuple[datetime, ...] = tuple(
    _add_months(_WALK_FORWARD_START, 3 * k) for k in range(8)
)
# The excluded tail (2026-07-01 to 2026-07-10T08:00Z, 56 G4 decisions) is pre-registered (1.3).
WALK_FORWARD_END: datetime = datetime(2026, 7, 1, tzinfo=UTC)
SEALED_FIRST_TEST_START: datetime = SEALED_START
REPLICATE_FOLDS: tuple[str, ...] = ("wf-2024-07", "wf-2025-06", "wf-2026-06")
REPLICATES: tuple[int, ...] = (1, 2)
# A synthetic label may carry lowercase segments ("synthetic-tiny"), never a replicate-like one.
FOLD_LABEL_RE: re.Pattern[str] = re.compile(
    r"(?P<kind>wf|q|s)-(?P<year>[0-9]{4})-(?P<month>[0-9]{2})"
    r"|final-(?P<fyear>[0-9]{4})-(?P<fmonth>[0-9]{2})-(?P<fday>[0-9]{2})"
    r"|(?P<synthetic>synthetic(?:-(?!r[0-9]+(?:-|$))[a-z0-9]+)*)"
)
# ks-btc-<recipe8>-<run label> must stay within the 64 characters of a checkpoint id.
MAX_SYNTHETIC_LABEL = 48
_KIND_LAYOUT = {"wf": "monthly", "q": "quarterly", "s": "sealed"}
_LAYOUT_PREFIX = {"monthly": "wf", "quarterly": "q", "sealed": "s"}
_TEST_MONTHS = {"monthly": 1, "quarterly": 3, "sealed": 1}


class FoldError(ValueError):
    """An invalid fold label or an inconsistent fold plan."""


@dataclass(frozen=True)
class Fold:
    """One fold: its CSV range, inner validation, test range and NLL4-TF grid (epoch ms).

    ``label`` carries no replicate suffix (``run_label`` does); ``test_start_ms`` and
    ``test_end_ms`` are ``None`` for final and synthetic folds. Built by the planners below,
    which check it (:func:`check_fold`).
    """

    label: str
    layout: str
    csv_start_ms: int
    csv_end_ms: int
    inner_validation_start_ms: int
    test_start_ms: int | None
    test_end_ms: int | None
    nll_grid_hours: int
    nll_steps: int
    replicate: int = 0

    @property
    def run_label(self) -> str:
        """``label``, plus ``-r<k>`` for replicate ``k > 0``."""
        return self.label if self.replicate == 0 else f"{self.label}-r{self.replicate}"

    @property
    def data_end_ms(self) -> int:
        """The CSV end: a checkpoint of this fold forecasts only decisions from it on."""
        return self.csv_end_ms

    @property
    def fit_range_ms(self) -> tuple[int, int]:
        return (self.csv_start_ms, self.inner_validation_start_ms)

    @property
    def inner_validation_range_ms(self) -> tuple[int, int]:
        return (self.inner_validation_start_ms, self.csv_end_ms)

    @property
    def needs_sealed_access(self) -> bool:
        """True when the CSV reads a candle opening at or after the sealed start."""
        return self.csv_end_ms > SEALED_START_MS

    def nll_decisions_ms(self) -> tuple[int, ...]:
        """NLL4-TF decisions: grid hours ``D`` in ``[inner start, csv_end - nll_steps h]``.

        Every target ``D .. D + (nll_steps - 1) h`` opens before the CSV end (336 for R1).
        """
        step = self.nll_grid_hours * HOUR_MS
        first = -(-self.inner_validation_start_ms // step) * step
        last = self.csv_end_ms - self.nll_steps * HOUR_MS
        return tuple(range(first, last + 1, step))

    def test_decisions_ms(self, grid: str) -> tuple[int, ...]:
        """The ``grid`` hours in ``[test_start, test_end)`` (none for final and synthetic folds)."""
        if self.test_start_ms is None or self.test_end_ms is None:
            _grid_step(grid)
            return ()
        return grid_decisions_ms(self.test_start_ms, self.test_end_ms, grid)


def _grid_step(grid: str) -> int:
    if grid not in _GRID_HOURS:
        raise FoldError(f"unknown grid {grid!r}; expected one of {list(GRIDS)}")
    return _GRID_HOURS[grid] * HOUR_MS


def _ms(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise FoldError(f"{name} must be an integer of epoch milliseconds, got {value!r}")
    return int(value)


def on_grid(decision_ms: int, grid: str) -> bool:
    """True when ``decision_ms`` is an exact hour on ``grid``."""
    step = _grid_step(grid)
    value = _ms(decision_ms, "decision_ms")
    origin = G25_ORIGIN_MS if grid == "G25" else 0
    return value % HOUR_MS == 0 and (value - origin) % step == 0


def grid_decisions_ms(start_ms: int, end_ms: int, grid: str) -> tuple[int, ...]:
    """The ``grid`` hours in ``[start_ms, end_ms)``, ascending."""
    step = _grid_step(grid)
    start = _ms(start_ms, "start_ms")
    end = _ms(end_ms, "end_ms")
    origin = G25_ORIGIN_MS if grid == "G25" else 0
    first = origin + -(-(start - origin) // step) * step
    return tuple(range(first, end, step))


def _month_label(prefix: str, test_start: datetime) -> str:
    return f"{prefix}-{test_start:%Y-%m}"


def _history_start_ms(recipe: Recipe) -> int:
    return to_ms(parse_utc_hour(recipe.history_start))


def _build(
    label: str,
    layout: str,
    csv_end_ms: int,
    test_start_ms: int | None,
    test_end_ms: int | None,
    recipe: Recipe,
    *,
    csv_start_ms: int | None = None,
) -> Fold:
    if not isinstance(recipe, Recipe):
        raise TypeError(f"expected a Recipe, got {type(recipe).__name__}")
    start = _history_start_ms(recipe) if csv_start_ms is None else csv_start_ms
    fold = Fold(
        label=label,
        layout=layout,
        csv_start_ms=start,
        csv_end_ms=csv_end_ms,
        inner_validation_start_ms=csv_end_ms - recipe.inner_validation_hours * HOUR_MS,
        test_start_ms=test_start_ms,
        test_end_ms=test_end_ms,
        nll_grid_hours=recipe.nll_grid_hours,
        nll_steps=recipe.nll_steps,
    )
    check_fold(fold, recipe=recipe)
    return fold


def _test_fold(layout: str, test_start: datetime, recipe: Recipe) -> Fold:
    test_end = _add_months(test_start, _TEST_MONTHS[layout])
    test_start_ms = to_ms(test_start)
    return _build(
        _month_label(_LAYOUT_PREFIX[layout], test_start),
        layout,
        test_start_ms - recipe.test_embargo_hours * HOUR_MS,
        test_start_ms,
        to_ms(test_end),
        recipe,
    )


def monthly_folds(recipe: Recipe) -> tuple[Fold, ...]:
    """The 24 monthly folds ``wf-2024-07`` .. ``wf-2026-06`` (the first has the pilot's CSV;
    the walk-forward runs the quarterly layout, D-056)."""
    return tuple(_test_fold("monthly", start, recipe) for start in MONTHLY_TEST_STARTS)


def quarterly_folds(recipe: Recipe) -> tuple[Fold, ...]:
    """The 8 quarterly folds ``q-2024-07`` .. ``q-2026-04`` (the compute fallback, the
    walk-forward's layout since D-056; the first is the pilot)."""
    return tuple(_test_fold("quarterly", start, recipe) for start in QUARTERLY_TEST_STARTS)


def _month_start(value: datetime | str, name: str) -> datetime:
    try:
        hour = parse_utc_hour(value)
    except ValueError as exc:
        raise FoldError(f"{name}: {exc}") from None
    if hour.day != 1 or hour.hour != 0:
        raise FoldError(f"{name} must be 00:00Z on the 1st of a month, got {utc_text(to_ms(hour))}")
    return hour


def sealed_fold(test_start: datetime | str, recipe: Recipe) -> Fold:
    """The sealed fold ``s-YYYY-MM`` testing the month that starts at ``test_start`` (from 2026-08).

    Its CSV reads sealed candles from ``s-2026-09`` on (:attr:`Fold.needs_sealed_access`).
    """
    start = _month_start(test_start, "test_start")
    if start < SEALED_FIRST_TEST_START:
        raise FoldError(
            f"sealed test months start at {utc_text(to_ms(SEALED_FIRST_TEST_START))}, got "
            f"{utc_text(to_ms(start))}"
        )
    return _test_fold("sealed", start, recipe)


def final_fold(t_final: datetime | str, recipe: Recipe) -> Fold:
    """The model to date: CSV ``[history_start, t_final)``, ``t_final`` 00:00Z from 2026-08-01."""
    try:
        end = parse_utc_hour(t_final)
    except ValueError as exc:
        raise FoldError(f"t_final: {exc}") from None
    if end.hour != 0:
        raise FoldError(f"t_final must be 00:00Z of a day, got {utc_text(to_ms(end))}")
    return _build(f"final-{end:%Y-%m-%d}", "final", to_ms(end), None, None, recipe)


def synthetic_fold(
    csv_start_ms: int, csv_end_ms: int, recipe: Recipe, *, label: str = "synthetic"
) -> Fold:
    """A fold over synthetic candles in ``[csv_start_ms, csv_end_ms)`` (timing and tests).

    Both ends lie on the recipe's NLL grid, with at least ``inner_validation_hours +
    window_rows + 1`` hours between them; no test follows.
    """
    start = _ms(csv_start_ms, "csv_start_ms")
    end = _ms(csv_end_ms, "csv_end_ms")
    return _build(label, "synthetic", end, None, None, recipe, csv_start_ms=start)


def fold_for(label: str, recipe: Recipe, *, replicate: int = 0) -> Fold:
    """The fold of a label (``wf-``, ``q-``, ``s-YYYY-MM`` or ``final-YYYY-MM-DD``).

    A replicate ``k > 0`` exists only for :data:`REPLICATE_FOLDS` and ``k`` in
    :data:`REPLICATES`; synthetic folds are built with :func:`synthetic_fold`.
    """
    if not isinstance(label, str):
        raise FoldError(f"a fold label is a string, got {label!r}")
    match = FOLD_LABEL_RE.fullmatch(label)
    if match is None:
        raise FoldError(
            f"fold label {label!r} is not wf-YYYY-MM, q-YYYY-MM, s-YYYY-MM or final-YYYY-MM-DD"
        )
    if match["synthetic"] is not None:
        raise FoldError("a synthetic fold has no planned range: use synthetic_fold()")
    if match["kind"] is not None:
        try:
            start = datetime(int(match["year"]), int(match["month"]), 1, tzinfo=UTC)
        except ValueError as exc:
            raise FoldError(f"fold label {label!r}: {exc}") from None
        layout = _KIND_LAYOUT[match["kind"]]
        if layout == "sealed":
            fold = sealed_fold(start, recipe)
        else:
            allowed = MONTHLY_TEST_STARTS if layout == "monthly" else QUARTERLY_TEST_STARTS
            if start not in allowed:
                raise FoldError(f"{label!r} is not a planned {layout} fold")
            fold = _test_fold(layout, start, recipe)
    else:
        try:
            end = datetime(
                int(match["fyear"]), int(match["fmonth"]), int(match["fday"]), tzinfo=UTC
            )
        except ValueError as exc:
            raise FoldError(f"fold label {label!r}: {exc}") from None
        fold = final_fold(end, recipe)
    if isinstance(replicate, bool) or not isinstance(replicate, int):
        raise FoldError(f"replicate must be an integer, got {replicate!r}")
    if replicate:
        fold = replace(fold, replicate=replicate)
        check_fold(fold, recipe=recipe)
    return fold


def _check_label(fold: Fold) -> re.Match[str]:
    if not isinstance(fold.label, str):
        raise FoldError(f"a fold label is a string, got {fold.label!r}")
    match = FOLD_LABEL_RE.fullmatch(fold.label)
    if match is None:
        raise FoldError(f"fold label {fold.label!r} does not match {FOLD_LABEL_RE.pattern}")
    if match["synthetic"] is not None:
        kind = "synthetic"
        if len(fold.label) > MAX_SYNTHETIC_LABEL:
            raise FoldError(f"a synthetic label has at most {MAX_SYNTHETIC_LABEL} characters")
    elif match["kind"] is not None:
        kind = _KIND_LAYOUT[match["kind"]]
    else:
        kind = "final"
    if kind != fold.layout:
        raise FoldError(f"fold label {fold.label!r} does not name a {fold.layout} fold")
    return match


def _check_replicate(fold: Fold) -> None:
    replicate = fold.replicate
    if isinstance(replicate, bool) or not isinstance(replicate, int):
        raise FoldError(f"replicate must be an integer, got {replicate!r}")
    if replicate == 0:
        return
    if replicate not in REPLICATES or fold.label not in REPLICATE_FOLDS:
        raise FoldError(
            f"replicates {list(REPLICATES)} exist only for the folds {list(REPLICATE_FOLDS)} "
            f"(D-048 (4)), not {fold.label!r} r{replicate}"
        )


def _check_test_range(fold: Fold, match: re.Match[str], recipe: Recipe) -> None:
    """Monthly, quarterly and sealed folds: the test range, its label and the test embargo."""
    if fold.test_start_ms is None or fold.test_end_ms is None:
        raise FoldError(f"{fold.layout} fold {fold.label} needs a test range")
    test_start_ms = _ms(fold.test_start_ms, "test_start_ms")
    start = from_ms(test_start_ms)
    if start.day != 1 or start.hour or start.minute or start.second or start.microsecond:
        raise FoldError(f"fold {fold.label}: a test range starts at 00:00Z on the 1st of a month")
    if fold.layout == "monthly" and start not in MONTHLY_TEST_STARTS:
        raise FoldError(f"fold {fold.label}: {utc_text(test_start_ms)} is not a monthly test start")
    if fold.layout == "quarterly" and start not in QUARTERLY_TEST_STARTS:
        raise FoldError(
            f"fold {fold.label}: {utc_text(test_start_ms)} is not a quarterly test start"
        )
    if fold.layout == "sealed" and start < SEALED_FIRST_TEST_START:
        raise FoldError(f"fold {fold.label}: sealed test months start at 2026-08-01")
    if fold.label != _month_label(match["kind"], start):
        raise FoldError(f"fold label {fold.label!r} does not name its test start")
    expected_end = to_ms(_add_months(start, _TEST_MONTHS[fold.layout]))
    if fold.test_end_ms != expected_end:
        raise FoldError(
            f"fold {fold.label}: the test range ends at {utc_text(expected_end)}, got "
            f"{fold.test_end_ms!r}"
        )
    expected = test_start_ms - recipe.test_embargo_hours * HOUR_MS
    if fold.csv_end_ms > expected:
        raise PartitionError(
            f"fold {fold.label}: the CSV ends at {utc_text(fold.csv_end_ms)}, inside the "
            f"{recipe.test_embargo_hours} h test embargo before {utc_text(test_start_ms)} "
            "(D-048 (6))"
        )
    if fold.csv_end_ms != expected:
        raise FoldError(
            f"fold {fold.label}: the CSV ends {recipe.test_embargo_hours} h before the test "
            f"start ({utc_text(expected)}), got {utc_text(fold.csv_end_ms)}"
        )


def _check_partition_rules(fold: Fold, recipe: Recipe) -> None:
    """The partition rules of design 1.4 for a fold with a test range (``PartitionError``)."""
    assert fold.test_start_ms is not None and fold.test_end_ms is not None
    decisions = sorted({d for grid in GRIDS for d in fold.test_decisions_ms(grid)})
    first_allowed = fold.data_end_ms + recipe.test_embargo_hours * HOUR_MS
    early = [d for d in decisions if d < first_allowed]
    if early:
        raise PartitionError(
            f"fold {fold.label}: test decision {utc_text(early[0])} is less than "
            f"{recipe.test_embargo_hours} h after the CSV end {utc_text(fold.data_end_ms)}"
        )
    if fold.layout in ("monthly", "quarterly"):
        if fold.test_end_ms > to_ms(WALK_FORWARD_END):
            raise PartitionError(
                f"fold {fold.label}: walk-forward tests end by {utc_text(to_ms(WALK_FORWARD_END))}"
            )
        boundary = SEALED_START_MS - MAX_CONTEXT * HOUR_MS
        if fold.csv_end_ms > boundary:
            raise PartitionError(
                f"fold {fold.label}: a validation-phase CSV ends by {utc_text(boundary)}, the "
                "last candle a validation sample may read"
            )
    for decision in decisions:
        if not sample_fits(from_ms(decision), EVAL_LENGTH_MAX):
            raise PartitionError(
                f"fold {fold.label}: test decision {utc_text(decision)} does not fit "
                f"{EVAL_LENGTH_MAX} target candles before the next segment (horizon purge)"
            )


def check_fold(fold: Fold, *, recipe: Recipe = R1) -> None:
    """Check ``fold`` against ``recipe`` and the boundaries of design 1.4.

    ``FoldError`` for an inconsistent fold: the label, layout and replicate; integer hours; the
    recipe's NLL grid; ``csv_start`` = the recipe's history start (synthetic folds: both ends on
    the NLL grid); ``inner start = csv_end - inner_validation_hours h``; enough hours for a fit
    and an inner validation of more than one window each; ``csv_end = test_start -
    test_embargo_hours h`` with the layout's test range, or ``t_final`` (00:00Z, from 2026-08-01)
    for a final fold. ``PartitionError`` for a partition rule: a CSV ending inside the test
    embargo, and every ``G4``, ``G8`` and ``G25`` test decision at least ``test_embargo_hours``
    after the CSV end; monthly and quarterly tests ending by :data:`WALK_FORWARD_END`, so their
    CSVs end by ``SEALED_START - 512 h``; and every test decision fitting ``EVAL_LENGTH_MAX``
    targets before the next segment (``sample_fits``). ``recipe`` defaults to :data:`R1`; a fold
    alone does not carry the history start, inner validation and embargo it was planned with.
    """
    if not isinstance(fold, Fold):
        raise TypeError(f"expected a Fold, got {type(fold).__name__}")
    if not isinstance(recipe, Recipe):
        raise TypeError(f"expected a Recipe, got {type(recipe).__name__}")
    if fold.layout not in LAYOUTS:
        raise FoldError(f"layout {fold.layout!r} not in {list(LAYOUTS)}")
    match = _check_label(fold)
    _check_replicate(fold)
    for name in ("csv_start_ms", "csv_end_ms", "inner_validation_start_ms"):
        if _ms(getattr(fold, name), name) % HOUR_MS:
            raise FoldError(f"fold {fold.label}: {name} is not on an exact hour")
    for name in ("test_start_ms", "test_end_ms"):
        value = getattr(fold, name)
        if value is not None and _ms(value, name) % HOUR_MS:
            raise FoldError(f"fold {fold.label}: {name} is not on an exact hour")
    if (fold.nll_grid_hours, fold.nll_steps) != (recipe.nll_grid_hours, recipe.nll_steps):
        raise FoldError(
            f"fold {fold.label}: NLL grid {fold.nll_grid_hours} h x {fold.nll_steps} steps differs "
            f"from recipe {recipe.name}'s {recipe.nll_grid_hours} h x {recipe.nll_steps}"
        )
    if fold.inner_validation_start_ms != fold.csv_end_ms - recipe.inner_validation_hours * HOUR_MS:
        raise FoldError(
            f"fold {fold.label}: the inner validation is the CSV's last "
            f"{recipe.inner_validation_hours} hours"
        )
    min_hours = recipe.inner_validation_hours + recipe.window_rows + 1
    if fold.csv_end_ms - fold.csv_start_ms < min_hours * HOUR_MS:
        raise FoldError(
            f"fold {fold.label}: the CSV range needs at least {min_hours} hours (more than one "
            "window of fit rows after the inner validation)"
        )
    if fold.layout == "synthetic":
        step = fold.nll_grid_hours * HOUR_MS
        if fold.csv_start_ms % step or fold.csv_end_ms % step:
            raise FoldError(
                f"fold {fold.label}: a synthetic fold starts and ends on the "
                f"{fold.nll_grid_hours}-hour NLL grid"
            )
        if fold.test_start_ms is not None or fold.test_end_ms is not None:
            raise FoldError(f"fold {fold.label}: a synthetic fold has no test range")
        return
    if fold.csv_start_ms != _history_start_ms(recipe):
        raise FoldError(
            f"fold {fold.label}: the CSV starts at the recipe's history start "
            f"{recipe.history_start}, got {utc_text(fold.csv_start_ms)}"
        )
    if fold.layout == "final":
        _check_final(fold)
        return
    _check_test_range(fold, match, recipe)
    _check_partition_rules(fold, recipe)


def _check_final(fold: Fold) -> None:
    if fold.test_start_ms is not None or fold.test_end_ms is not None:
        raise FoldError(f"fold {fold.label}: the model to date has no test range")
    end = from_ms(fold.csv_end_ms)
    if end.hour:
        raise FoldError(f"fold {fold.label}: t_final is 00:00Z of a day")
    if fold.csv_end_ms < SEALED_START_MS:
        raise FoldError(
            f"fold {fold.label}: the model to date ends at {utc_text(SEALED_START_MS)} or later "
            "(D-048 (3))"
        )
    if fold.label != f"final-{end:%Y-%m-%d}":
        raise FoldError(f"fold label {fold.label!r} does not name its t_final")


__all__ = [
    "FOLD_LABEL_RE",
    "G25_ORIGIN_MS",
    "GRIDS",
    "LAYOUTS",
    "MAX_SYNTHETIC_LABEL",
    "MONTHLY_TEST_STARTS",
    "QUARTERLY_TEST_STARTS",
    "REPLICATES",
    "REPLICATE_FOLDS",
    "SEALED_FIRST_TEST_START",
    "WALK_FORWARD_END",
    "Fold",
    "FoldError",
    "check_fold",
    "final_fold",
    "fold_for",
    "grid_decisions_ms",
    "monthly_folds",
    "on_grid",
    "quarterly_folds",
    "sealed_fold",
    "synthetic_fold",
]
