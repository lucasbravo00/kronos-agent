"""Partition, embargo purge and the sealed guard (SPEC section 6; SPEC section 17, gate 6), the
evaluation length and the model periods of step 6 (``docs/RETRAINING.md``, 1.4 and 5.3).

The sealed guard verifies the owner's order in the experiment ledger (test T7): on the test's
isolated ledger, empty unless a test records an order through ``ledger_fakes``. That ledger is
not the repository's, so its orders open only a data root that is not the repository's data/
(nor links into it): two tests point the data root at the repository's data/, and at a view of
it, and read nothing but its metadata."""

from __future__ import annotations

import functools
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import ledger_fakes as lf
import numpy as np
import pytest

from kronos_agent import budget, ledger, paths
from kronos_agent.data import partition as pt
from kronos_agent.data.partition import (
    EMBARGO_HOURS,
    EVAL_LENGTH_MAX,
    KRONOS_FIRST_DECISION,
    KRONOS_FIRST_DECISION_MS,
    MAX_CONTEXT,
    MODEL_PERIOD_ROLES,
    N1_PARTITION,
    N1_TRAINING_END,
    N1_VALIDATION_START,
    PARTITION,
    SEALED_START,
    SEALED_START_MS,
    TRAINING_END,
    TRAINING_START,
    VALIDATION_END,
    VALIDATION_START,
    PartitionError,
    SealedAccess,
    SealedPartitionError,
    check_embargo,
    check_evaluation_length,
    first_model_decision_ms,
    is_sealed,
    require_accessible,
    require_kronos_period,
    require_model_period,
    require_study_range,
    sample_fits,
    sealed_access,
    segment_named,
    segment_of,
)
from kronos_agent.data.timeutil import HOUR_MS, to_ms, utc_text

H = timedelta(hours=1)
ORDER_ID = "order-2026-001"


def test_constants_are_the_question_8_defaults():
    assert TRAINING_START == datetime(2020, 1, 1, tzinfo=UTC)
    assert TRAINING_END == datetime(2024, 6, 9, 8, tzinfo=UTC)
    assert VALIDATION_START == KRONOS_FIRST_DECISION == datetime(2024, 7, 1, tzinfo=UTC)
    assert VALIDATION_END == datetime(2026, 7, 10, 8, tzinfo=UTC)
    assert SEALED_START == datetime(2026, 8, 1, tzinfo=UTC)
    assert SEALED_START_MS == 1_785_542_400_000 == to_ms(SEALED_START)
    assert EMBARGO_HOURS == 520 == MAX_CONTEXT + 8
    assert TRAINING_END + EMBARGO_HOURS * H == VALIDATION_START
    assert VALIDATION_END + EMBARGO_HOURS * H == SEALED_START
    assert N1_TRAINING_END == datetime(2026, 1, 1, tzinfo=UTC)
    assert N1_TRAINING_END + EMBARGO_HOURS * H == N1_VALIDATION_START
    assert N1_VALIDATION_START == datetime(2026, 1, 22, 16, tzinfo=UTC)


def test_segments_are_contiguous_and_named():
    assert [s.name for s in PARTITION] == [
        "training",
        "embargo_1",
        "validation",
        "embargo_2",
        "sealed",
    ]
    assert [s.name for s in N1_PARTITION] == ["n1_training", "n1_embargo", "n1_validation"]
    for partition in (PARTITION, N1_PARTITION):
        for previous, current in pairwise(partition):
            assert previous.end == current.start
    assert PARTITION[-1].end is None and PARTITION[-1].end_ms is None
    assert N1_PARTITION[0].start == VALIDATION_START
    assert N1_PARTITION[-1].end == VALIDATION_END
    assert [s.holds_samples for s in PARTITION] == [True, False, True, False, True]
    assert segment_named("validation").start_ms == to_ms(VALIDATION_START)
    with pytest.raises(KeyError):
        segment_named("validation", N1_PARTITION)


@pytest.mark.parametrize(
    "hour, expected",
    [
        (TRAINING_START - H, "before_training"),
        (TRAINING_START, "training"),
        (TRAINING_END - H, "training"),
        (TRAINING_END, "embargo_1"),
        (VALIDATION_START - H, "embargo_1"),
        (VALIDATION_START, "validation"),
        (VALIDATION_END - H, "validation"),
        (VALIDATION_END, "embargo_2"),
        (SEALED_START - H, "embargo_2"),
        (SEALED_START, "sealed"),
        (SEALED_START + 10_000 * H, "sealed"),
    ],
)
def test_segment_of_is_half_open(hour, expected):
    assert segment_of(hour) == expected


def test_segment_of_the_n1_partition():
    assert segment_of(VALIDATION_START - H, N1_PARTITION) == "before_training"
    assert segment_of(VALIDATION_START, N1_PARTITION) == "n1_training"
    assert segment_of(N1_TRAINING_END - H, N1_PARTITION) == "n1_training"
    assert segment_of(N1_TRAINING_END, N1_PARTITION) == "n1_embargo"
    assert segment_of(N1_VALIDATION_START, N1_PARTITION) == "n1_validation"
    assert segment_of(VALIDATION_END - H, N1_PARTITION) == "n1_validation"
    assert segment_of(VALIDATION_END, N1_PARTITION) == "after_partition"


def test_segment_of_refuses_a_non_hour():
    with pytest.raises(ValueError):
        segment_of(datetime(2024, 7, 1, 0, 30, tzinfo=UTC))
    with pytest.raises(ValueError):
        segment_of(datetime(2024, 7, 1))


def test_is_sealed_and_require_accessible():
    assert not is_sealed(SEALED_START - H)
    assert is_sealed(SEALED_START)
    require_accessible(SEALED_START - H)
    with pytest.raises(SealedPartitionError, match="gate 6"):
        require_accessible(SEALED_START)
    with pytest.raises(SealedPartitionError):
        require_accessible(SEALED_START, ORDER_ID)  # a string is not a SealedAccess


def test_require_study_range_end_is_exclusive_at_the_sealed_start():
    require_study_range(0, SEALED_START_MS)  # the last allowed end: candles open before it
    require_study_range(SEALED_START_MS, SEALED_START_MS)  # empty range at the boundary
    with pytest.raises(SealedPartitionError):
        require_study_range(0, SEALED_START_MS + 1)
    with pytest.raises(SealedPartitionError):
        require_study_range(SEALED_START_MS, SEALED_START_MS + HOUR_MS)
    with pytest.raises(ValueError):
        require_study_range(10, 5)
    with pytest.raises(ValueError):
        require_study_range(0.0, 5)
    with pytest.raises(ValueError):
        require_study_range(True, 5)


def test_sealed_access_raises_without_a_recorded_order():
    """The test's ledger is empty: no order verifies, whatever its id."""
    for order_id in (ORDER_ID, "order-0001"):
        with pytest.raises(SealedPartitionError, match="in the experiment ledger"):
            sealed_access(order_id)
    with pytest.raises(SealedPartitionError, match="issued only by sealed_access"):
        SealedAccess(ORDER_ID)
    with pytest.raises(ValueError):
        sealed_access("  ")
    assert not ledger.ledger_path().exists()  # the guard only reads the ledger


def test_a_recorded_order_opens_the_guard():
    """T7 with the ledger: the owner's order, recorded after its candidate's T-B reading, opens
    the guard, also after its T-C reading (it covers the sealed folds and the model to date,
    D-048 (3)); an unknown id and a broken ledger do not."""
    recipe = lf.recipe_id("sealed-guard")
    order = lf.recorded_order(recipe)
    assert order.payload["order_id"] == "order-0001"
    with pytest.raises(SealedPartitionError, match="no owner_order order-0002"):
        sealed_access("order-0002")
    access = sealed_access("order-0001")
    assert access.order_id == "order-0001"
    require_accessible(SEALED_START, access)
    require_study_range(0, SEALED_START_MS + HOUR_MS, access)
    with pytest.raises(SealedPartitionError):
        require_study_range(0, SEALED_START_MS + HOUR_MS)
    budget.spend_test("T-C", recipe_id=recipe, order_id="order-0001", now=lf.MONDAY + 2 * lf.WEEK)
    assert sealed_access("order-0001").order_id == "order-0001"
    path = ledger.ledger_path()
    path.write_bytes(path.read_bytes()[:-1])  # a torn last line: the ledger no longer verifies
    with pytest.raises(SealedPartitionError, match="experiment ledger: .*no LF at the end"):
        sealed_access("order-0001")


def test_a_verified_order_opens_the_guard(monkeypatch):
    verified: list[str] = []
    monkeypatch.setattr(pt, "_verify_order", verified.append)
    access = sealed_access(ORDER_ID)
    assert verified == [ORDER_ID]
    assert access.order_id == ORDER_ID
    require_accessible(SEALED_START, access)
    require_study_range(0, SEALED_START_MS + HOUR_MS, access)


REAL_DATA = paths.REPO_ROOT / "data"
needs_real_data = pytest.mark.skipif(
    not REAL_DATA.is_dir(), reason="the repository has no data/ to protect"
)


@needs_real_data
def test_an_order_in_a_relocated_ledger_never_opens_the_repositorys_data(monkeypatch):
    """The test's ledger lives under its own (relocated) reports root and holds the owner's
    order; with the data root left at the repository's data/ (KRONOS_AGENT_DATA_DIR unset), the
    order does not open the guard (the binding of the ledger to the data root). Only metadata of
    the repository's data/ is read: nothing in it is opened."""
    lf.recorded_order(lf.recipe_id("relocated"))
    assert sealed_access("order-0001").order_id == "order-0001"  # its own data root: opens
    monkeypatch.delenv(paths.DATA_DIR_ENV)
    assert paths.data_dir() == paths.REPO_ROOT / "data"
    assert paths.ledger_dir() != paths.REPO_ROOT / "reports" / "ledger"
    for order_id in ("order-0001", "order-0009"):  # refused before the order is even looked up
        with pytest.raises(
            SealedPartitionError, match="not the repository's reports/ledger"
        ) as info:
            sealed_access(order_id)
        assert "the data root is the repository's data/" in str(info.value)
        assert "gate 6" in str(info.value)
    verified: list[str] = []
    monkeypatch.setattr(pt, "_verify_order", verified.append)  # a stand-in cannot skip it
    with pytest.raises(SealedPartitionError, match="not the repository's reports/ledger"):
        sealed_access("order-0001")
    assert verified == []


@needs_real_data
@pytest.mark.real_roots("data")
def test_an_order_in_a_relocated_ledger_never_opens_a_view_of_the_repositorys_data(
    isolated_roots, tmp_path
):
    """The test's data root is a view of the repository's data/ (its entries link there): the
    test's ledger can neither spend a test on it nor open its sealed segment."""
    assert isolated_roots.views == {"data"} and any(paths.data_dir().iterdir())
    lf.register_bundle(lf.recipe_id("view"), now=lf.MONDAY)
    with pytest.raises(ledger.LedgerRefused, match="links into the repository's data/"):
        budget.spend_test("T-A", recipe_id=lf.recipe_id("view"), now=lf.MONDAY)
    lf.recorded_order(lf.recipe_id("elsewhere"), data_root=tmp_path / "own-data")
    with pytest.raises(SealedPartitionError, match="links into the repository's data/"):
        sealed_access("order-0001")
    access = sealed_access("order-0001", data_root=tmp_path / "own-data")
    assert access.order_id == "order-0001"


def test_the_sealed_guard_reads_the_order_from_the_ledger_it_is_given(tmp_path):
    """An explicit ledger and data root (the fold command's): the binding and the order are both
    checked there, not in the default ledger."""
    elsewhere = tmp_path / "ledger"
    lf.recorded_order(lf.recipe_id("explicit"), root=elsewhere, reports_root=tmp_path / "reports")
    with pytest.raises(SealedPartitionError, match="no owner_order order-0001"):
        sealed_access("order-0001")  # the default (isolated) ledger holds no order
    access = sealed_access("order-0001", ledger_root=elsewhere, data_root=tmp_path / "data")
    assert access.order_id == "order-0001"
    if REAL_DATA.is_dir():
        with pytest.raises(SealedPartitionError, match="the repository's data/"):
            sealed_access("order-0001", ledger_root=elsewhere, data_root=REAL_DATA)


def test_check_embargo():
    check_embargo(512, 8)
    check_embargo(512, 4)
    check_embargo(511, 9)
    with pytest.raises(PartitionError, match="521") as info:
        check_embargo(512, 9)
    # the embargo bounds the trading horizon N; longer evaluation horizons go through the purge
    message = str(info.value)
    for part in (
        "exceeds the 520 h embargo, which bounds the context plus the trading horizon N",
        "(SPEC section 6: embargo = context + N)",
        "evaluation horizons longer than N (the skill curve of SPEC section 8 reaches 24 candles)",
        "horizon purge with an explicit evaluation length designed with the retraining",
    ):
        assert part in message
    assert message.endswith("not by recomputing the partition")
    assert "must be recomputed" not in message
    with pytest.raises(PartitionError, match="2052 h exceeds the 520 h embargo") as info:
        check_embargo(2048, 4)  # Kronos-mini
    assert str(info.value).endswith("not by recomputing the partition")
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            check_embargo(512, bad)


BOUNDARIES = [
    pytest.param(TRAINING_END - H, PARTITION, id="training-to-validation"),
    pytest.param(VALIDATION_END - H, PARTITION, id="validation-to-sealed"),
    pytest.param(N1_TRAINING_END - H, N1_PARTITION, id="n1_training-to-n1_validation"),
    pytest.param(VALIDATION_END - H, N1_PARTITION, id="n1_validation-to-sealed"),
]


@pytest.mark.parametrize("last_hour, partition", BOUNDARIES)
def test_sample_fits_keeps_h9_and_purges_h10_at_every_boundary(last_hour, partition):
    assert sample_fits(last_hour, 9, partition=partition)
    assert not sample_fits(last_hour, 10, partition=partition)
    # one hour earlier everything shifts by one candle
    assert sample_fits(last_hour - H, 10, partition=partition)
    assert not sample_fits(last_hour - H, 11, partition=partition)
    # a shorter context moves the next segment's first context later
    assert sample_fits(last_hour, 10, max_context=511, partition=partition)
    assert not sample_fits(last_hour, 11, max_context=511, partition=partition)


@pytest.mark.parametrize("partition", [PARTITION, N1_PARTITION], ids=["main", "n1"])
def test_sample_fits_long_horizon_cutoff_at_the_end_of_validation(partition):
    # docs/DATA.md, "Horizon purge": the last target must open before 2026-07-10T16:00Z, the
    # sealed start minus 512 h; decisions up to 2026-07-10T16:00Z - h are kept for h > 9
    cutoff = datetime(2026, 7, 10, 16, tzinfo=UTC)
    assert SEALED_START - MAX_CONTEXT * H == cutoff
    assert VALIDATION_END - 16 * H == datetime(2026, 7, 9, 16, tzinfo=UTC)
    assert sample_fits(VALIDATION_END - 16 * H, 24, partition=partition)
    assert not sample_fits(VALIDATION_END - 15 * H, 24, partition=partition)
    for horizon in (10, 12, 24, 48):
        assert sample_fits(cutoff - horizon * H, horizon, partition=partition)
        assert not sample_fits(cutoff - horizon * H + H, horizon, partition=partition)


def test_sample_fits_outside_sample_segments():
    assert not sample_fits(TRAINING_END, 1)  # embargo
    assert not sample_fits(VALIDATION_START - H, 1)  # embargo
    assert not sample_fits(TRAINING_START - H, 1)  # before the partition
    assert sample_fits(VALIDATION_START, 24)  # the first validation hour, far from the end
    assert sample_fits(SEALED_START + 1000 * H, 48)  # the open-ended last segment
    assert not sample_fits(VALIDATION_END, 1, partition=N1_PARTITION)  # after the N1 partition
    assert not sample_fits(N1_TRAINING_END, 1, partition=N1_PARTITION)  # N1 embargo
    with pytest.raises(ValueError):
        sample_fits(VALIDATION_START, 0)


def test_require_kronos_period():
    require_kronos_period(KRONOS_FIRST_DECISION)
    with pytest.raises(PartitionError, match="June 2024"):
        require_kronos_period(KRONOS_FIRST_DECISION - H)
    require_kronos_period(KRONOS_FIRST_DECISION - H, allow_pretraining_period=True)


# -- evaluation length (docs/RETRAINING.md, 5.3) ---------------------------------------------------


def test_check_evaluation_length_allows_1_to_24():
    assert EVAL_LENGTH_MAX == 24
    assert [check_evaluation_length(length) for length in range(1, 25)] == list(range(1, 25))
    numpy_length = check_evaluation_length(np.int64(24))
    assert numpy_length == 24 and type(numpy_length) is int
    for too_long in (25, 48):
        with pytest.raises(PartitionError, match="EVAL_LENGTH_MAX = 24") as info:
            check_evaluation_length(too_long)
        assert "horizon purge" in str(info.value)
    for bad in (0, -1, True, False, 4.0, "4", None):
        with pytest.raises(ValueError, match="evaluation length"):
            check_evaluation_length(bad)


def test_the_purge_of_the_longest_evaluation_length():
    """T23: 24-step samples of the validation segment end at 2026-07-09T16:00Z, the sealed start
    minus the 512-candle context minus 24 targets; the embargo keeps bounding N only."""
    last = datetime(2026, 7, 9, 16, tzinfo=UTC)
    assert last == SEALED_START - (MAX_CONTEXT + EVAL_LENGTH_MAX) * H
    assert sample_fits(last, EVAL_LENGTH_MAX)
    assert not sample_fits(last + H, EVAL_LENGTH_MAX)
    assert sample_fits(last + H, EVAL_LENGTH_MAX - 1)
    check_embargo(MAX_CONTEXT, 4)  # the governed N passes ...
    with pytest.raises(PartitionError, match="embargo"):
        check_embargo(MAX_CONTEXT, EVAL_LENGTH_MAX)  # ... the evaluation length is not its job


# -- model periods (docs/RETRAINING.md, 1.4) -------------------------------------------------------

PILOT_END = datetime(2024, 6, 9, 16, tzinfo=UTC)  # wf-2024-07: the end of its training data
LATER_END = datetime(2025, 1, 10, 16, tzinfo=UTC)  # wf-2025-02


def test_first_model_decision_ms():
    assert first_model_decision_ms(None) == KRONOS_FIRST_DECISION_MS
    # a checkpoint starts from the shipped weights: never before their first decision hour
    assert first_model_decision_ms(to_ms(PILOT_END)) == KRONOS_FIRST_DECISION_MS
    assert first_model_decision_ms(to_ms(LATER_END)) == to_ms(LATER_END)
    assert first_model_decision_ms(np.int64(to_ms(LATER_END))) == to_ms(LATER_END)
    for bad in (to_ms(LATER_END) + 1, to_ms(LATER_END) + HOUR_MS // 2, 1.5e12, True, "1"):
        with pytest.raises(ValueError, match="data_end_ms"):
            first_model_decision_ms(bad)


SHIPPED = None
PILOT = to_ms(PILOT_END)
LATER = to_ms(LATER_END)


@pytest.mark.parametrize(
    "role, data_end_ms, hour, allowed",
    [
        ("evaluation", SHIPPED, KRONOS_FIRST_DECISION - H, False),
        ("evaluation", SHIPPED, KRONOS_FIRST_DECISION, True),
        ("evaluation", PILOT, PILOT_END, False),  # its data end is before July 2024
        ("evaluation", PILOT, KRONOS_FIRST_DECISION - H, False),
        ("evaluation", PILOT, KRONOS_FIRST_DECISION, True),
        ("evaluation", LATER, LATER_END - H, False),
        ("evaluation", LATER, LATER_END, True),
        ("timing", SHIPPED, TRAINING_START, True),
        ("timing", LATER, TRAINING_START, True),
        ("pilot-check", SHIPPED, datetime(2024, 5, 1, tzinfo=UTC), True),
        ("pilot-check", PILOT, datetime(2024, 5, 1, tzinfo=UTC), True),
        ("cutoff-check", SHIPPED, datetime(2024, 1, 15, tzinfo=UTC), True),
        ("cutoff-check", SHIPPED, datetime(2024, 12, 15, tzinfo=UTC), True),
        ("cutoff-check", PILOT, datetime(2024, 12, 15, tzinfo=UTC), False),
    ],
)
def test_require_model_period_table(role, data_end_ms, hour, allowed):
    assert role in MODEL_PERIOD_ROLES
    if allowed:
        require_model_period(hour, role=role, data_end_ms=data_end_ms)
        require_model_period(utc_text(to_ms(hour)), role=role, data_end_ms=data_end_ms)
        return
    with pytest.raises(PartitionError) as info:
        require_model_period(hour, role=role, data_end_ms=data_end_ms)
    message = str(info.value)
    if role == "cutoff-check":
        assert "reads the shipped weights only" in message
    elif data_end_ms is None:
        assert "June 2024" in message  # tests/test_decision.py matches these words
    else:
        first = utc_text(first_model_decision_ms(data_end_ms))
        assert f"is before {first}, the first hour this checkpoint may forecast" in message
        assert f"trained on candles before {utc_text(data_end_ms)}" in message


def test_allow_pretraining_period_lifts_only_the_evaluation_rule():
    for data_end_ms in (SHIPPED, PILOT, LATER):
        require_model_period(
            TRAINING_START,
            role="evaluation",
            data_end_ms=data_end_ms,
            allow_pretraining_period=True,
        )
    with pytest.raises(PartitionError, match="shipped weights only"):
        require_model_period(
            KRONOS_FIRST_DECISION,
            role="cutoff-check",
            data_end_ms=PILOT,
            allow_pretraining_period=True,
        )


def test_require_model_period_argument_checks():
    for role in ("Evaluation", "", None, "training"):
        with pytest.raises(ValueError, match="role must be one of"):
            require_model_period(KRONOS_FIRST_DECISION, role=role)
    with pytest.raises(ValueError):
        require_model_period(datetime(2024, 7, 1, 0, 30, tzinfo=UTC), role="evaluation")
    with pytest.raises(ValueError, match="data_end_ms"):
        require_model_period(KRONOS_FIRST_DECISION, role="timing", data_end_ms=LATER + 1)


def test_require_kronos_period_is_the_shipped_evaluation_rule():
    for hour in (KRONOS_FIRST_DECISION - H, KRONOS_FIRST_DECISION):
        for allow in (False, True):
            outcomes = []
            for check in (
                functools.partial(require_kronos_period, hour, allow_pretraining_period=allow),
                functools.partial(
                    require_model_period, hour, role="evaluation", allow_pretraining_period=allow
                ),
            ):
                try:
                    check()
                    outcomes.append(None)
                except PartitionError as exc:
                    outcomes.append(str(exc))
            assert outcomes[0] == outcomes[1]


def test_timeutil_and_partition_import_only_the_standard_library():
    code = (
        "import sys\n"
        "import kronos_agent.data.partition, kronos_agent.data.timeutil\n"
        "heavy = sorted(name for name in ('numpy', 'pandas', 'torch', 'httpx', 'yaml', "
        "'matplotlib', 'kronos_agent.data.store', 'kronos_agent.provenance') "
        "if name in sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
