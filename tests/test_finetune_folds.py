"""The fold plan: the design's table 1.2, the test embargo (T6, data side), grids and labels.

Every number here is the design's (``docs/RETRAINING.md`` 1.2 and 1.3) and assumes no gap; the
exporter recounts on the real candles (``tests/test_finetune_export.py``).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from kronos_agent.data.partition import (
    EVAL_LENGTH_MAX,
    SEALED_START_MS,
    TRAINING_START,
    VALIDATION_START,
    PartitionError,
    sample_fits,
)
from kronos_agent.data.timeutil import HOUR_MS, from_ms, to_ms, utc_text
from kronos_agent.finetune import folds as fd
from kronos_agent.finetune.folds import (
    FOLD_LABEL_RE,
    GRIDS,
    MONTHLY_TEST_STARTS,
    QUARTERLY_TEST_STARTS,
    REPLICATE_FOLDS,
    REPLICATES,
    WALK_FORWARD_END,
    Fold,
    FoldError,
    check_fold,
    final_fold,
    fold_for,
    grid_decisions_ms,
    monthly_folds,
    on_grid,
    quarterly_folds,
    sealed_fold,
    synthetic_fold,
)
from kronos_agent.finetune.recipe import R1

# docs/RETRAINING.md, table 1.2: fold, CSV end (excl.), inner validation from, fit windows, G4, G25
DESIGN_TABLE = [
    ("wf-2024-07", "2024-06-09T16:00Z", "2024-04-14T16:00Z", 37_064, 186, 30),
    ("wf-2024-08", "2024-07-10T16:00Z", "2024-05-15T16:00Z", 37_808, 186, 29),
    ("wf-2024-09", "2024-08-10T16:00Z", "2024-06-15T16:00Z", 38_552, 180, 29),
    ("wf-2024-10", "2024-09-09T16:00Z", "2024-07-15T16:00Z", 39_272, 186, 30),
    ("wf-2024-11", "2024-10-10T16:00Z", "2024-08-15T16:00Z", 40_016, 180, 29),
    ("wf-2024-12", "2024-11-09T16:00Z", "2024-09-14T16:00Z", 40_736, 186, 29),
    ("wf-2025-01", "2024-12-10T16:00Z", "2024-10-15T16:00Z", 41_480, 186, 30),
    ("wf-2025-02", "2025-01-10T16:00Z", "2024-11-15T16:00Z", 42_224, 168, 27),
    ("wf-2025-03", "2025-02-07T16:00Z", "2024-12-13T16:00Z", 42_896, 186, 30),
    ("wf-2025-04", "2025-03-10T16:00Z", "2025-01-13T16:00Z", 43_640, 180, 29),
    ("wf-2025-05", "2025-04-09T16:00Z", "2025-02-12T16:00Z", 44_360, 186, 29),
    ("wf-2025-06", "2025-05-10T16:00Z", "2025-03-15T16:00Z", 45_104, 180, 29),
    ("wf-2025-07", "2025-06-09T16:00Z", "2025-04-14T16:00Z", 45_824, 186, 30),
    ("wf-2025-08", "2025-07-10T16:00Z", "2025-05-15T16:00Z", 46_568, 186, 30),
    ("wf-2025-09", "2025-08-10T16:00Z", "2025-06-15T16:00Z", 47_312, 180, 28),
    ("wf-2025-10", "2025-09-09T16:00Z", "2025-07-15T16:00Z", 48_032, 186, 30),
    ("wf-2025-11", "2025-10-10T16:00Z", "2025-08-15T16:00Z", 48_776, 180, 29),
    ("wf-2025-12", "2025-11-09T16:00Z", "2025-09-14T16:00Z", 49_496, 186, 30),
    ("wf-2026-01", "2025-12-10T16:00Z", "2025-10-15T16:00Z", 50_240, 186, 30),
    ("wf-2026-02", "2026-01-10T16:00Z", "2025-11-15T16:00Z", 50_984, 168, 26),
    ("wf-2026-03", "2026-02-07T16:00Z", "2025-12-13T16:00Z", 51_656, 186, 30),
    ("wf-2026-04", "2026-03-10T16:00Z", "2026-01-13T16:00Z", 52_400, 180, 29),
    ("wf-2026-05", "2026-04-09T16:00Z", "2026-02-12T16:00Z", 53_120, 186, 30),
    ("wf-2026-06", "2026-05-10T16:00Z", "2026-03-15T16:00Z", 53_864, 180, 29),
]
SYNTHETIC_START_MS = to_ms(datetime(2020, 1, 1, tzinfo=UTC))


def fit_windows(fold: Fold) -> int:
    """Vendored fit windows without gaps: fit rows minus a window plus one."""
    return (fold.inner_validation_start_ms - fold.csv_start_ms) // HOUR_MS - R1.window_rows + 1


def test_the_monthly_plan_is_the_design_table():
    plan = monthly_folds(R1)
    assert [fold.label for fold in plan] == [row[0] for row in DESIGN_TABLE]
    for fold, (_, csv_end, inner, windows, g4, g25) in zip(plan, DESIGN_TABLE, strict=True):
        assert fold.layout == "monthly" and fold.replicate == 0
        assert utc_text(fold.csv_end_ms) == csv_end
        assert utc_text(fold.inner_validation_start_ms) == inner
        assert fold.csv_start_ms == to_ms(TRAINING_START)
        assert fit_windows(fold) == windows
        assert len(fold.test_decisions_ms("G4")) == g4
        assert len(fold.test_decisions_ms("G25")) == g25
    assert sum(len(fold.test_decisions_ms("G4")) for fold in plan) == 4_380
    assert sum(len(fold.test_decisions_ms("G25")) for fold in plan) == 701
    assert [from_ms(fold.test_start_ms) for fold in plan] == list(MONTHLY_TEST_STARTS)
    assert from_ms(plan[-1].test_end_ms) == WALK_FORWARD_END


def test_the_pilot_ends_at_the_partition_boundary_with_the_checked_numbers():
    pilot = fold_for("wf-2024-07", R1)
    assert pilot == monthly_folds(R1)[0]
    assert pilot.csv_end_ms == to_ms(VALIDATION_START) - 512 * HOUR_MS  # 2024-06-09T16:00Z
    rows = (pilot.csv_end_ms - pilot.csv_start_ms) // HOUR_MS
    fit_rows = (pilot.inner_validation_start_ms - pilot.csv_start_ms) // HOUR_MS
    assert (rows, fit_rows, rows - fit_rows) == (38_920, 37_576, 1_344)
    assert rows - fit_rows - R1.window_rows + 1 == 832  # vendored validation windows
    decisions = pilot.nll_decisions_ms()
    t_ms = pilot.test_start_ms
    assert len(decisions) == 336
    assert decisions[0] == t_ms - 1_856 * HOUR_MS and decisions[-1] == t_ms - 516 * HOUR_MS
    assert all(d % (4 * HOUR_MS) == 0 for d in decisions)
    assert decisions[-1] + 3 * HOUR_MS < pilot.csv_end_ms  # every target opens in the CSV
    assert pilot.fit_range_ms == (pilot.csv_start_ms, pilot.inner_validation_start_ms)
    assert pilot.inner_validation_range_ms == (pilot.inner_validation_start_ms, pilot.csv_end_ms)
    assert pilot.data_end_ms == pilot.csv_end_ms and not pilot.needs_sealed_access


def test_the_quarterly_fallback_covers_the_same_decisions():
    quarters = quarterly_folds(R1)
    months = monthly_folds(R1)
    assert [fold.label for fold in quarters] == [
        f"q-{start:%Y-%m}" for start in QUARTERLY_TEST_STARTS
    ]
    assert len(quarters) == 8 and quarters[0].csv_end_ms == months[0].csv_end_ms
    for grid in GRIDS:
        by_quarter = [d for fold in quarters for d in fold.test_decisions_ms(grid)]
        by_month = [d for fold in months for d in fold.test_decisions_ms(grid)]
        assert by_quarter == by_month
    for fold in quarters:
        assert fold.test_end_ms - fold.test_start_ms >= 89 * 24 * HOUR_MS
        assert fold.csv_end_ms == fold.test_start_ms - 512 * HOUR_MS


def test_the_quarterly_pilot_q_2024_07_is_fold_1_of_the_chosen_layout():
    """D-056 chose the quarterly layout, whose first fold is the pilot: the CSV of wf-2024-07,
    a test quarter, R1's seeds and checkpoint ids, and no quarterly replicate (INTERFACES-3a 1,
    16: a replicate mapping needs the owner's decision)."""
    from kronos_agent.finetune.export_csv import fold_export_dir
    from kronos_agent.finetune.recipe import recipe8, recipe_id, training_seeds
    from kronos_agent.finetune.registry import checkpoint_id_for

    pilot = fold_for("q-2024-07", R1)
    monthly = fold_for("wf-2024-07", R1)
    assert pilot == quarterly_folds(R1)[0] and pilot.layout == "quarterly"
    assert (pilot.csv_start_ms, pilot.csv_end_ms) == (monthly.csv_start_ms, monthly.csv_end_ms)
    assert pilot.inner_validation_start_ms == monthly.inner_validation_start_ms
    assert utc_text(pilot.csv_end_ms) == "2024-06-09T16:00Z" and not pilot.needs_sealed_access
    assert utc_text(pilot.inner_validation_start_ms) == "2024-04-14T16:00Z"
    assert (utc_text(pilot.test_start_ms), utc_text(pilot.test_end_ms)) == (
        "2024-07-01T00:00Z",
        "2024-10-01T00:00Z",
    )
    assert len(pilot.test_decisions_ms("G4")) == 552 and len(pilot.test_decisions_ms("G25")) == 88
    decisions = pilot.nll_decisions_ms()
    assert len(decisions) == 336 and decisions == monthly.nll_decisions_ms()
    assert (utc_text(decisions[0]), utc_text(decisions[-1])) == (
        "2024-04-14T16:00Z",
        "2024-06-09T12:00Z",
    )
    pilot_check = grid_decisions_ms(pilot.inner_validation_start_ms, pilot.csv_end_ms, "G8")
    assert len(pilot_check) == 168 and utc_text(pilot_check[-1]) == "2024-06-09T08:00Z"
    identifier = recipe_id(R1)
    seeds = training_seeds(identifier, "q-2024-07", 0)
    assert seeds.canonical == f"kronos-agent/train-seed/v1/{identifier}/q-2024-07/r0"
    assert seeds.yaml_seed == 81454873
    doubled = replace(R1, max_epochs=12)
    assert checkpoint_id_for(recipe8(R1), pilot.run_label) == "ks-btc-0ee0d123-q-2024-07"
    assert checkpoint_id_for(recipe8(doubled), pilot.run_label) == "ks-btc-56e3d2f9-q-2024-07"
    assert fold_export_dir(recipe8(R1), pilot.label).parts[-3:] == (
        "finetune",
        "0ee0d123",
        "q-2024-07",
    )
    quarters = quarterly_folds(R1)
    assert [len(fold.test_decisions_ms("G4")) for fold in quarters] == [
        552,
        552,
        540,
        546,
        552,
        552,
        540,
        546,
    ]
    assert [len(fold.test_decisions_ms("G25")) for fold in quarters] == [
        88,
        88,
        87,
        87,
        88,
        89,
        86,
        88,
    ]
    for replicate in REPLICATES:
        with pytest.raises(FoldError, match="replicates"):
            fold_for("q-2024-07", R1, replicate=replicate)


def every_test_fold() -> list[Fold]:
    sealed = [sealed_fold(f"{month}-01T00:00Z", R1) for month in ("2026-08", "2026-09", "2027-03")]
    return [*monthly_folds(R1), *quarterly_folds(R1), *sealed]


def test_t6_every_test_decision_is_512_hours_after_the_csv_end():
    for fold in every_test_fold():
        assert fold.csv_end_ms == fold.test_start_ms - 512 * HOUR_MS
        for grid in GRIDS:
            decisions = fold.test_decisions_ms(grid)
            assert decisions and min(decisions) >= fold.data_end_ms + 512 * HOUR_MS
            assert all(fold.test_start_ms <= d < fold.test_end_ms for d in decisions)
        check_fold(fold)


def test_t6_walk_forward_decisions_fit_the_longest_evaluation_before_the_sealed_start():
    boundary = SEALED_START_MS - 512 * HOUR_MS  # 2026-07-10T16:00Z
    for fold in [*monthly_folds(R1), *quarterly_folds(R1)]:
        assert fold.csv_end_ms <= boundary
        for d in fold.test_decisions_ms("G4") + fold.test_decisions_ms("G25"):
            assert sample_fits(from_ms(d), EVAL_LENGTH_MAX)
            assert d + (EVAL_LENGTH_MAX - 1) * HOUR_MS < boundary


def test_t6_a_csv_ending_inside_the_embargo_is_a_partition_error():
    pilot = fold_for("wf-2024-07", R1)
    late = replace(
        pilot,
        csv_end_ms=pilot.csv_end_ms + HOUR_MS,
        inner_validation_start_ms=pilot.inner_validation_start_ms + HOUR_MS,
    )
    with pytest.raises(PartitionError, match="embargo"):
        check_fold(late)
    early = replace(
        pilot,
        csv_end_ms=pilot.csv_end_ms - HOUR_MS,
        inner_validation_start_ms=pilot.inner_validation_start_ms - HOUR_MS,
    )
    with pytest.raises(FoldError):
        check_fold(early)
    no_embargo = replace(R1, test_embargo_hours=0)
    with pytest.raises(PartitionError):
        check_fold(pilot, recipe=replace(R1, test_embargo_hours=600))
    unembargoed = fold_for("wf-2024-07", no_embargo)
    assert unembargoed.csv_end_ms == unembargoed.test_start_ms
    check_fold(unembargoed, recipe=no_embargo)


def test_extending_the_walk_forward_into_the_excluded_tail_is_refused(monkeypatch):
    july = datetime(2026, 7, 1, tzinfo=UTC)
    monkeypatch.setattr(fd, "MONTHLY_TEST_STARTS", (*MONTHLY_TEST_STARTS, july))
    with pytest.raises(PartitionError, match="walk-forward tests end by"):
        fold_for("wf-2026-07", R1)
    # with the end moved as well, the decisions of July 2026 no longer fit 24 targets
    monkeypatch.setattr(fd, "WALK_FORWARD_END", datetime(2026, 8, 1, tzinfo=UTC))
    with pytest.raises(PartitionError, match="horizon purge"):
        fold_for("wf-2026-07", R1)
    # and a validation-phase CSV may not read past the last candle a validation sample reads
    august = datetime(2026, 8, 1, tzinfo=UTC)
    monkeypatch.setattr(fd, "MONTHLY_TEST_STARTS", (*MONTHLY_TEST_STARTS, july, august))
    monkeypatch.setattr(fd, "WALK_FORWARD_END", datetime(2026, 9, 1, tzinfo=UTC))
    with pytest.raises(PartitionError, match="validation-phase CSV"):
        fold_for("wf-2026-08", replace(R1, test_embargo_hours=0))


# -- sealed and final folds -----------------------------------------------------------------------


def test_sealed_folds_follow_the_design():
    expected = {
        "2026-08": (186, 30, False),
        "2026-09": (180, 29, True),
        "2026-10": (186, 30, True),
        "2026-11": (180, 28, True),
    }
    for month, (g4, g25, needs_access) in expected.items():
        fold = sealed_fold(f"{month}-01T00:00Z", R1)
        assert fold == fold_for(f"s-{month}", R1)
        assert (fold.label, fold.layout) == (f"s-{month}", "sealed")
        assert len(fold.test_decisions_ms("G4")) == g4
        assert len(fold.test_decisions_ms("G25")) == g25
        assert fold.needs_sealed_access is needs_access
    first = fold_for("s-2026-08", R1)
    assert first.csv_end_ms == SEALED_START_MS - 512 * HOUR_MS  # readable without access
    for bad in ("2026-07-01T00:00Z", "2026-09-02T00:00Z", "2026-09-01T01:00Z", "2026-09"):
        with pytest.raises(FoldError):
            sealed_fold(bad, R1)


def test_the_model_to_date():
    fold = final_fold("2026-08-01T00:00Z", R1)
    assert fold == fold_for("final-2026-08-01", R1)
    assert (fold.label, fold.layout, fold.test_start_ms, fold.test_end_ms) == (
        "final-2026-08-01",
        "final",
        None,
        None,
    )
    assert fold.csv_end_ms == SEALED_START_MS and not fold.needs_sealed_access
    assert fold.test_decisions_ms("G4") == ()
    later = final_fold(datetime(2026, 11, 2, tzinfo=UTC), R1)
    assert later.label == "final-2026-11-02" and later.needs_sealed_access
    assert len(later.nll_decisions_ms()) == 336
    for bad in ("2026-07-31T00:00Z", "2026-08-01T16:00Z"):
        with pytest.raises(FoldError):
            final_fold(bad, R1)


def test_no_planned_window_count_is_a_multiple_of_9973():
    """Design 1.2: the vendored index mapping is a bijection for every planned count."""
    folds = [*monthly_folds(R1), *quarterly_folds(R1)]
    folds += [sealed_fold(f"{m}-01T00:00Z", R1) for m in ("2026-08", "2026-09", "2026-10")]
    day = datetime(2026, 8, 1, tzinfo=UTC)
    while day <= datetime(2027, 3, 31, tzinfo=UTC):
        folds.append(final_fold(day, R1))
        day += timedelta(days=1)
    for fold in folds:
        assert fit_windows(fold) % 9973, fold.label
        assert (R1.inner_validation_hours - R1.window_rows + 1) % 9973
    assert len(folds) == 24 + 8 + 3 + 243


# -- labels, replicates and synthetic folds ------------------------------------------------------


def test_replicates_exist_only_in_the_three_folds():
    for label in REPLICATE_FOLDS:
        base = fold_for(label, R1)
        for replicate in REPLICATES:
            fold = fold_for(label, R1, replicate=replicate)
            assert fold == replace(base, replicate=replicate)
            assert fold.run_label == f"{label}-r{replicate}"
            assert fold.label == label
    assert fold_for("wf-2024-07", R1).run_label == "wf-2024-07"
    for label, replicate in [("wf-2024-08", 1), ("wf-2024-07", 3), ("q-2024-07", 1)]:
        with pytest.raises(FoldError):
            fold_for(label, R1, replicate=replicate)
    for replicate in (-1, True, 1.0):
        with pytest.raises(FoldError):
            fold_for("wf-2024-07", R1, replicate=replicate)


@pytest.mark.parametrize(
    "label",
    [
        "wf-2024-06",
        "wf-2026-07",
        "wf-2024-13",
        "q-2024-08",
        "q-2026-07",
        "s-2026-07",
        "final-2026-07-31",
        "final-2026-02-30",
        "final-2026-08-01T",
        "synthetic",
        "wf-2024-07-r1",
        "WF-2024-07",
        "",
    ],
)
def test_fold_for_refuses_unplanned_labels(label):
    with pytest.raises(FoldError):
        fold_for(label, R1)


def test_synthetic_folds():
    end = SYNTHETIC_START_MS + 4096 * HOUR_MS
    fold = synthetic_fold(SYNTHETIC_START_MS, end, R1)
    assert (fold.label, fold.layout, fold.test_start_ms) == ("synthetic", "synthetic", None)
    assert fold.inner_validation_start_ms == end - 1344 * HOUR_MS
    assert len(fold.nll_decisions_ms()) == 336
    assert synthetic_fold(SYNTHETIC_START_MS, end, R1, label="synthetic-tiny").label == (
        "synthetic-tiny"
    )
    shortest = (R1.inner_validation_hours + R1.window_rows + 1 + 3) // 4 * 4
    synthetic_fold(SYNTHETIC_START_MS, SYNTHETIC_START_MS + shortest * HOUR_MS, R1)
    bad_calls = [
        (SYNTHETIC_START_MS + HOUR_MS, end, "synthetic"),  # off the 4-hour NLL grid
        (SYNTHETIC_START_MS, end - HOUR_MS, "synthetic"),
        (SYNTHETIC_START_MS, SYNTHETIC_START_MS + (shortest - 4) * HOUR_MS, "synthetic"),
        (SYNTHETIC_START_MS, end, "synthetic-r1"),  # reads like a replicate
        (SYNTHETIC_START_MS, end, "Synthetic"),
        (SYNTHETIC_START_MS, end, "wf-2024-07"),
        (SYNTHETIC_START_MS, end, "synthetic-" + "x" * 40),
        (float(SYNTHETIC_START_MS), end, "synthetic"),
    ]
    for start, stop, label in bad_calls:
        with pytest.raises(FoldError):
            synthetic_fold(start, stop, R1, label=label)
    with pytest.raises(FoldError):
        check_fold(replace(fold, replicate=1))
    tiny_grid = replace(R1, nll_grid_hours=1, inner_validation_hours=520, min_nll_decisions=8)
    odd = synthetic_fold(SYNTHETIC_START_MS + HOUR_MS, end + HOUR_MS, tiny_grid)
    assert len(odd.nll_decisions_ms()) == 517


def test_check_fold_refuses_tampered_folds():
    pilot = fold_for("wf-2024-07", R1)
    tampered = [
        replace(pilot, label="q-2024-07"),
        replace(pilot, label="wf-2024-08"),
        replace(pilot, layout="quarterly"),
        replace(pilot, layout="weekly"),
        replace(pilot, inner_validation_start_ms=pilot.inner_validation_start_ms + HOUR_MS),
        replace(pilot, csv_start_ms=pilot.csv_start_ms + HOUR_MS),
        replace(pilot, csv_end_ms=pilot.csv_end_ms + 1),
        replace(pilot, test_end_ms=pilot.test_end_ms + HOUR_MS),
        replace(pilot, test_end_ms=None),
        replace(pilot, nll_grid_hours=8),
        replace(pilot, nll_steps=8),
        replace(pilot, csv_end_ms=float(pilot.csv_end_ms)),
        replace(pilot, replicate=True),
    ]
    for fold in tampered:
        with pytest.raises(FoldError):
            check_fold(fold)
    with pytest.raises(FoldError):
        check_fold(pilot, recipe=replace(R1, history_start="2021-01-01T00:00Z"))
    with pytest.raises(FoldError):
        check_fold(pilot, recipe=replace(R1, nll_grid_hours=8))
    with pytest.raises(TypeError):
        check_fold("wf-2024-07")
    final = fold_for("final-2026-08-01", R1)
    with pytest.raises(FoldError):
        check_fold(replace(final, test_start_ms=SEALED_START_MS, test_end_ms=SEALED_START_MS))
    with pytest.raises(FoldError):
        check_fold(replace(final, label="final-2026-08-02"))


def test_the_fold_label_grammar():
    for label in ("wf-2024-07", "q-2025-01", "s-2026-08", "final-2026-08-01", "synthetic"):
        assert FOLD_LABEL_RE.fullmatch(label)
    for label in ("synthetic-tiny", "synthetic-a1-b2", "synthetic-r1x"):
        assert FOLD_LABEL_RE.fullmatch(label)
    for label in ("wf-2024-7", "x-2024-07", "final-2026-8-1", "synthetic-r2", "synthetic-"):
        assert FOLD_LABEL_RE.fullmatch(label) is None


# -- grids -------------------------------------------------------------------------------------


def test_grids():
    day = to_ms(datetime(2025, 3, 1, tzinfo=UTC))
    hours = [day + k * HOUR_MS for k in range(24)]
    assert [from_ms(h).hour for h in hours if on_grid(h, "G4")] == [0, 4, 8, 12, 16, 20]
    assert [from_ms(h).hour for h in hours if on_grid(h, "G8")] == [0, 8, 16]
    assert on_grid(to_ms(TRAINING_START), "G25")
    assert on_grid(to_ms(TRAINING_START) + 25 * HOUR_MS, "G25")
    assert not on_grid(to_ms(TRAINING_START) + 24 * HOUR_MS, "G25")
    assert not on_grid(day + 1, "G4")
    g25 = grid_decisions_ms(day, day + 25 * 24 * HOUR_MS, "G25")
    assert len(g25) == 24 and sorted(from_ms(d).hour for d in g25) == list(range(24))
    assert all((d - to_ms(TRAINING_START)) % (25 * HOUR_MS) == 0 for d in g25)
    g4 = grid_decisions_ms(day, day + 30 * 24 * HOUR_MS, "G4")
    g8 = grid_decisions_ms(day, day + 30 * 24 * HOUR_MS, "G8")
    assert len(g4) == 180 and set(g8) <= set(g4) and len(g8) == 90
    assert grid_decisions_ms(day + 1, day + 4 * HOUR_MS + 1, "G4") == (day + 4 * HOUR_MS,)
    assert grid_decisions_ms(day, day, "G4") == ()
    for bad in ("G5", "g4"):
        with pytest.raises(FoldError):
            on_grid(day, bad)
        with pytest.raises(FoldError):
            fold_for("wf-2024-07", R1).test_decisions_ms(bad)
    with pytest.raises(FoldError):
        on_grid(True, "G4")
