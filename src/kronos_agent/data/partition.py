"""Chronological partition by decision hour and the sealed-segment guard (SPEC section 6).

``training | embargo | validation | embargo | sealed``, half-open UTC intervals of decision hours
that hold for every source (``docs/DATA.md``, "Partition"; question-8 defaults in
``docs/DECISIONS.md``). Each embargo is 520 h = the 512-candle context plus ``N = 8`` targets:
a sample at ``D`` in one segment may read targets up to ``D + (h - 1) h`` only if that time is
earlier than the open of the first candle that the next sample-holding segment's first context
reads (:func:`sample_fits`), so a configuration whose context plus horizon exceeds the embargo
is refused (:func:`check_embargo`).

The sealed segment (gate 6) is guarded at the data-access layer by candle open time: a study
range reaching a candle that opens at or after :data:`SEALED_START` needs a
:class:`SealedAccess`, and the only way to obtain one is :func:`sealed_access`, which verifies the
owner's order recorded in the experiment ledger (:mod:`kronos_agent.ledger`, ``docs/RETRAINING.md``
7.3): an ``owner_order`` entry of scope ``open_sealed``, which ``kronos-agent ledger
record-order`` appends only after the owner's explicit order in chat, in a ledger whose chain
verifies. The ledger lives under the reports root, which is relocatable on its own, so the order
must also come from a ledger that may open the data root the access is for
(``ledger.binding_problem``): a ledger other than the repository's ``reports/ledger`` (a scratch
or test reports root) never opens the repository's ``data/`` or a data root that links into it.
Without such an order every attempt raises :class:`SealedPartitionError`, so nothing can open the
sealed segment of the repository's data before the owner orders it in its own ledger.

Evaluation horizons longer than the trading ``N`` (``docs/RETRAINING.md``, 5.3): the embargo
keeps bounding the context plus ``N`` (:func:`check_embargo`), an evaluation length ``L`` of at
most :data:`EVAL_LENGTH_MAX` candles is checked by :func:`check_evaluation_length`, and a sample
whose ``L`` targets reach past the purge is refused (:func:`sample_fits`), never shortened. The
model period (:func:`require_model_period`) refuses decision hours a model may have seen: the
shipped weights forecast only from :data:`KRONOS_FIRST_DECISION`, a fine-tuned checkpoint only
from the later of its data end and that date; the named opt-in roles (:data:`MODEL_PERIOD_ROLES`)
are recorded by the callers that use them.

This module imports only the standard library (and :mod:`kronos_agent.data.timeutil`): the
command line imports it at ``--help`` time. The ledger, itself light, is imported only inside
:func:`_verify_order` and :func:`_verify_binding`.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .timeutil import HOUR_MS, parse_utc_hour, to_ms, utc_text

TRAINING_START = datetime(2020, 1, 1, tzinfo=UTC)
TRAINING_END = datetime(2024, 6, 9, 8, tzinfo=UTC)
VALIDATION_START = datetime(2024, 7, 1, tzinfo=UTC)
VALIDATION_END = datetime(2026, 7, 10, 8, tzinfo=UTC)
SEALED_START = datetime(2026, 8, 1, tzinfo=UTC)
EMBARGO_HOURS = 520
MAX_CONTEXT = 512
# Kronos as shipped saw Binance up to June 2024 (SPEC section 6).
KRONOS_FIRST_DECISION = datetime(2024, 7, 1, tzinfo=UTC)
# N1/N2 sub-partition inside validation (models fed with Kronos forecasts, SPEC sections 6 and 10).
N1_TRAINING_END = datetime(2026, 1, 1, tzinfo=UTC)
N1_VALIDATION_START = datetime(2026, 1, 22, 16, tzinfo=UTC)

SEALED_START_MS = to_ms(SEALED_START)
KRONOS_FIRST_DECISION_MS = to_ms(KRONOS_FIRST_DECISION)

BEFORE_PARTITION = "before_training"
AFTER_PARTITION = "after_partition"

# Evaluation fans may run longer than the trading N (the SPEC section 8 skill curve reaches 24
# candles); their samples go through the horizon purge (docs/RETRAINING.md, 5.3).
EVAL_LENGTH_MAX = 24
# Who forecasts a decision hour (docs/RETRAINING.md, 1.4, 5.4 and 6.2): "evaluation" is the
# default everywhere; the other roles are named opt-ins that their callers record.
MODEL_PERIOD_EVALUATION = "evaluation"
MODEL_PERIOD_TIMING = "timing"
MODEL_PERIOD_PILOT_CHECK = "pilot-check"
MODEL_PERIOD_CUTOFF_CHECK = "cutoff-check"
MODEL_PERIOD_ROLES: tuple[str, ...] = (
    MODEL_PERIOD_EVALUATION,
    MODEL_PERIOD_TIMING,
    MODEL_PERIOD_PILOT_CHECK,
    MODEL_PERIOD_CUTOFF_CHECK,
)


class PartitionError(ValueError):
    """A configuration or a decision hour the partition does not allow."""


class SealedPartitionError(PermissionError):
    """Access to the sealed segment without the owner's recorded order (SPEC section 17, gate 6)."""


@dataclass(frozen=True)
class Segment:
    """Decision hours in ``[start, end)``; ``end=None`` is open-ended (the sealed segment).

    Segments whose name contains ``embargo`` hold no samples: they only separate the context of
    one sample-holding segment from the targets of the previous one.
    """

    name: str
    start: datetime
    end: datetime | None

    @property
    def holds_samples(self) -> bool:
        return "embargo" not in self.name

    @property
    def start_ms(self) -> int:
        return to_ms(self.start)

    @property
    def end_ms(self) -> int | None:
        return None if self.end is None else to_ms(self.end)

    def contains(self, decision_hour: datetime) -> bool:
        hour = parse_utc_hour(decision_hour)
        return self.start <= hour and (self.end is None or hour < self.end)


PARTITION: tuple[Segment, ...] = (
    Segment("training", TRAINING_START, TRAINING_END),
    Segment("embargo_1", TRAINING_END, VALIDATION_START),
    Segment("validation", VALIDATION_START, VALIDATION_END),
    Segment("embargo_2", VALIDATION_END, SEALED_START),
    Segment("sealed", SEALED_START, None),
)
N1_PARTITION: tuple[Segment, ...] = (
    Segment("n1_training", VALIDATION_START, N1_TRAINING_END),
    Segment("n1_embargo", N1_TRAINING_END, N1_VALIDATION_START),
    Segment("n1_validation", N1_VALIDATION_START, VALIDATION_END),
)

_ISSUER = object()


@dataclass(frozen=True)
class SealedAccess:
    """Permission to read the sealed segment, tied to one recorded order of the owner (an
    ``owner_order`` entry of the experiment ledger, named by its id ``order-NNNN``) in a ledger
    that may open the data root it was issued for.

    Never constructed directly: :func:`sealed_access` verifies the binding and the order and
    issues it. A direct ``SealedAccess(...)`` raises :class:`SealedPartitionError`
    (mistake-proofing, not security).
    """

    order_id: str
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _ISSUER:
            raise SealedPartitionError(
                "a SealedAccess is issued only by sealed_access(order_id), which verifies the "
                "owner's order recorded in the experiment ledger"
            )


def _verify_order(order_id: str, root: Path | None = None) -> None:
    """Check that ``order_id`` is the owner's order to open the sealed segment, recorded in the
    experiment ledger at ``root`` (default ``paths.ledger_dir()``; :mod:`kronos_agent.ledger`,
    ``docs/RETRAINING.md`` 7.3): an ``owner_order`` entry of scope ``open_sealed`` in a ledger
    whose chain verifies. A missing order, a broken or unreadable ledger:
    :class:`SealedPartitionError` (tests that only need an access monkeypatch this function with a
    one-argument stand-in)."""
    from ..ledger import LedgerError, LedgerRefused, verify_open_order  # light; read at call time

    try:
        verify_open_order(order_id, root=root)
    except (LedgerRefused, LedgerError) as exc:
        raise SealedPartitionError(
            f"cannot verify order {order_id!r} in the experiment ledger: {exc} "
            "(SPEC section 17, gate 6)"
        ) from None


def _verify_binding(order_id: str, *, ledger_root: Path | None, data_root: Path | None) -> None:
    """The ledger the order is read from may open ``data_root`` (``ledger.binding_problem``;
    :class:`SealedPartitionError` otherwise). Never replaced by the tests."""
    from ..ledger import binding_problem  # light; read at call time

    problem = binding_problem(root=ledger_root, data_root=data_root)
    if problem is not None:
        raise SealedPartitionError(
            f"cannot open the sealed segment on order {order_id!r}: {problem} (SPEC section 17, "
            "gate 6)"
        )


def sealed_access(
    order_id: str, *, ledger_root: Path | None = None, data_root: Path | None = None
) -> SealedAccess:
    """Issue a :class:`SealedAccess` for the data root ``data_root`` (default
    ``paths.data_dir()``) after checking that the ledger at ``ledger_root`` (default
    ``paths.ledger_dir()``) may open it, then verifying the owner's order recorded there
    (gate 6). Pass the access only to reads of that data root."""
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("order_id must be a non-empty string naming the recorded order")
    _verify_binding(order_id, ledger_root=ledger_root, data_root=data_root)
    if ledger_root is None:
        _verify_order(order_id)  # the one-argument form the tests' stand-ins take
    else:
        _verify_order(order_id, Path(ledger_root))
    return SealedAccess(order_id, _issuer=_ISSUER)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _integer_ms(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(  # noqa: TRY004 - ValueError is this module's contract
            f"{name} must be an integer of epoch milliseconds, got {value!r}"
        )
    return int(value)


def _check_partition(partition: tuple[Segment, ...]) -> tuple[Segment, ...]:
    """A non-empty sequence of contiguous segments; only the last one may be open-ended."""
    segments = tuple(partition)
    if not segments or not all(isinstance(segment, Segment) for segment in segments):
        raise PartitionError("a partition is a non-empty tuple of Segment")
    for index, segment in enumerate(segments):
        if segment.end is None:
            if index != len(segments) - 1:
                raise PartitionError(f"only the last segment may be open-ended ({segment.name})")
        elif not segment.start < segment.end:
            raise PartitionError(f"segment {segment.name} is empty or reversed")
        if index and segments[index - 1].end != segment.start:
            raise PartitionError(f"segment {segment.name} does not start where the previous ends")
    return segments


_check_partition(PARTITION)
_check_partition(N1_PARTITION)


def segment_named(name: str, partition: tuple[Segment, ...] = PARTITION) -> Segment:
    """The segment called ``name`` (``KeyError`` when the partition has none)."""
    for segment in _check_partition(partition):
        if segment.name == name:
            return segment
    raise KeyError(f"no segment named {name!r} in {[s.name for s in partition]}")


def _segment_index(hour: datetime, segments: tuple[Segment, ...]) -> int | None:
    for index, segment in enumerate(segments):
        if segment.start <= hour and (segment.end is None or hour < segment.end):
            return index
    return None


def segment_of(decision_hour: datetime, partition: tuple[Segment, ...] = PARTITION) -> str:
    """Name of the segment holding ``decision_hour``.

    ``"before_training"`` before the first segment starts (2020-01-01T00:00Z for
    :data:`PARTITION`), ``"after_partition"`` at or after the end of a partition whose last
    segment is not open-ended (:data:`N1_PARTITION`).
    """
    hour = parse_utc_hour(decision_hour)
    segments = _check_partition(partition)
    if hour < segments[0].start:
        return BEFORE_PARTITION
    index = _segment_index(hour, segments)
    return AFTER_PARTITION if index is None else segments[index].name


def is_sealed(decision_hour: datetime) -> bool:
    """True for decision hours at or after :data:`SEALED_START`."""
    return parse_utc_hour(decision_hour) >= SEALED_START


def _has_access(sealed_access: SealedAccess | None) -> bool:
    if sealed_access is None:
        return False
    if not isinstance(sealed_access, SealedAccess):
        raise SealedPartitionError(
            f"sealed_access must be a SealedAccess issued by sealed_access(), got "
            f"{type(sealed_access).__name__}"
        )
    return True


def require_accessible(decision_hour: datetime, sealed_access: SealedAccess | None = None) -> None:
    """Refuse a sealed decision hour unless a :class:`SealedAccess` is given."""
    hour = parse_utc_hour(decision_hour)
    if hour >= SEALED_START and not _has_access(sealed_access):
        raise SealedPartitionError(
            f"decision hour {utc_text(to_ms(hour))} is in the sealed segment "
            f"(from {utc_text(SEALED_START_MS)}); opening it needs the owner's recorded order "
            "(SPEC section 17, gate 6)"
        )


def require_study_range(
    start_ms: int, end_ms: int, sealed_access: SealedAccess | None = None
) -> None:
    """Refuse a candle range ``[start_ms, end_ms)`` that reaches the sealed segment.

    The range reaches it when it can hold a candle opening at or after :data:`SEALED_START`,
    i.e. when ``end_ms > SEALED_START_MS``; ``end_ms == SEALED_START_MS`` is the last allowed
    end. A :class:`SealedAccess` lifts the refusal.
    """
    start = _integer_ms(start_ms, "start_ms")
    end = _integer_ms(end_ms, "end_ms")
    if start > end:
        raise ValueError(f"start_ms {start} must be <= end_ms {end} (half-open range)")
    if end > SEALED_START_MS and not _has_access(sealed_access):
        raise SealedPartitionError(
            f"candle range [{utc_text(start)}, {utc_text(end)}) reaches the sealed segment "
            f"(candles opening at or after {utc_text(SEALED_START_MS)}); opening it needs the "
            "owner's recorded order (SPEC section 17, gate 6)"
        )


def check_embargo(max_context: int, horizon: int) -> None:
    """Refuse a configuration whose context plus trading horizon ``N`` exceeds the 520 h embargo.

    The embargo bounds the context plus the trading horizon ``N`` (SPEC section 6: embargo =
    context + ``N``): 512 + 8 = 520 passes; 512 + 9 (``N >= 9``) and 2048 + 4 (Kronos-mini) are
    refused. Evaluation horizons longer than ``N`` (the skill curve of SPEC section 8 reaches 24
    candles) go through the horizon purge (:func:`sample_fits`) with an explicit evaluation
    length (:func:`check_evaluation_length`), not by recomputing the partition: callers pass the
    governed ``N`` here whatever length they evaluate.
    """
    context = _positive_int(max_context, "max_context")
    targets = _positive_int(horizon, "horizon")
    if context + targets > EMBARGO_HOURS:
        raise PartitionError(
            f"max_context {context} + horizon {targets} = {context + targets} h exceeds the "
            f"{EMBARGO_HOURS} h embargo, which bounds the context plus the trading horizon N "
            "(SPEC section 6: embargo = context + N); evaluation horizons longer than N (the "
            "skill curve of SPEC section 8 reaches 24 candles) go through the horizon purge "
            "with an explicit evaluation length designed with the retraining "
            "(check_evaluation_length), not by recomputing the partition"
        )


def _next_sample_start(segments: tuple[Segment, ...], index: int) -> datetime | None:
    """Start of the next sample-holding segment after ``segments[index]``.

    A partition that stops at a finite end is followed by an embargo of :data:`EMBARGO_HOURS`
    (as :data:`N1_PARTITION` is followed by the main partition's second embargo); ``None`` when
    nothing follows an open-ended last segment.
    """
    for later in segments[index + 1 :]:
        if later.holds_samples:
            return later.start
    final_end = segments[-1].end
    if final_end is None:
        return None
    return final_end + timedelta(hours=EMBARGO_HOURS)


def sample_fits(
    decision_hour: datetime,
    horizon_candles: int,
    *,
    max_context: int = MAX_CONTEXT,
    partition: tuple[Segment, ...] = PARTITION,
) -> bool:
    """True iff the last target candle of a sample at ``D`` ends before the first candle that the
    next sample-holding segment's first context reads.

    The sample reads ``h`` targets, the last one opening at ``D + (h - 1) h``. It must open before
    the start of the next sample-holding segment minus ``max_context`` hours, the open time of
    that first context candle, so on the hourly grid the last target closes no later than that
    candle opens. A longer sample is purged, never shortened. A partition that stops at a finite
    end is followed by an embargo of :data:`EMBARGO_HOURS` (:data:`N1_PARTITION` by the main
    partition's second embargo). The open-ended last segment always fits. Hours in an embargo or
    outside the partition hold no samples (False).
    """
    hour = parse_utc_hour(decision_hour)
    horizon = _positive_int(horizon_candles, "horizon_candles")
    context = _positive_int(max_context, "max_context")
    segments = _check_partition(partition)
    index = _segment_index(hour, segments)
    if index is None or not segments[index].holds_samples:
        return False
    boundary = _next_sample_start(segments, index)
    if boundary is None:
        return True
    last_target = hour + timedelta(hours=horizon - 1)
    return last_target < boundary - timedelta(hours=context)


def check_evaluation_length(length: int) -> int:
    """The evaluation length ``L`` of a fan: 1 to :data:`EVAL_LENGTH_MAX` candles.

    A bool, a non-integer or ``L < 1`` is a ``ValueError``; ``L > EVAL_LENGTH_MAX`` is a
    :class:`PartitionError`. The embargo keeps bounding the trading ``N`` only
    (:func:`check_embargo`); each sample of length ``L`` must also fit the horizon purge
    (:func:`sample_fits` with ``L``), which refuses it rather than shortening it.
    """
    if isinstance(length, bool) or not isinstance(length, numbers.Integral) or length < 1:
        raise ValueError(
            f"an evaluation length must be a positive integer of candles, got {length!r}"
        )
    if length > EVAL_LENGTH_MAX:
        raise PartitionError(
            f"evaluation length {length} exceeds EVAL_LENGTH_MAX = {EVAL_LENGTH_MAX} candles: "
            "longer samples are not covered by the horizon purge designed with the retraining "
            "(docs/RETRAINING.md, 5.3)"
        )
    return int(length)


def first_model_decision_ms(data_end_ms: int | None) -> int:
    """The first decision hour a model may forecast, in epoch milliseconds.

    ``None`` (the shipped weights): :data:`KRONOS_FIRST_DECISION_MS`. A checkpoint's data end
    ``e`` (the end of the candles it was trained on, an integer of epoch milliseconds on an exact
    hour, ``ValueError`` otherwise): ``max(e, KRONOS_FIRST_DECISION_MS)``, because every
    checkpoint starts from the shipped weights, which saw Binance up to June 2024.
    """
    if data_end_ms is None:
        return KRONOS_FIRST_DECISION_MS
    end = _integer_ms(data_end_ms, "data_end_ms")
    if end % HOUR_MS:
        raise ValueError(f"data_end_ms must be on an exact hour, got {utc_text(end)}")
    return max(end, KRONOS_FIRST_DECISION_MS)


def require_model_period(
    decision_hour: datetime | str,
    *,
    role: str,
    data_end_ms: int | None = None,
    allow_pretraining_period: bool = False,
) -> None:
    """Refuse a decision hour the model may have seen (``docs/RETRAINING.md``, 1.4).

    ``data_end_ms`` is ``None`` for the shipped weights and a checkpoint's data end otherwise.
    ``role`` (:data:`MODEL_PERIOD_ROLES`; another value is a ``ValueError``):

    - ``"evaluation"`` (the default of every caller): the shipped weights forecast only from
      :data:`KRONOS_FIRST_DECISION`, a checkpoint only from :func:`first_model_decision_ms` of
      its data end; earlier hours raise :class:`PartitionError`. ``allow_pretraining_period``
      (tests only) lifts this rule.
    - ``"timing"`` (the synthetic timing window) and ``"pilot-check"`` (the pilot's inner
      validation, 6.2): every hour is allowed, for both.
    - ``"cutoff-check"`` (5.4): every hour is allowed for the shipped weights; a checkpoint is
      refused, because the cutoff check reads the shipped weights only.

    The callers that use an opt-in role record it in their report.
    """
    if not isinstance(role, str) or role not in MODEL_PERIOD_ROLES:
        raise ValueError(f"role must be one of {list(MODEL_PERIOD_ROLES)}, got {role!r}")
    decision_ms = to_ms(parse_utc_hour(decision_hour))
    first_ms = first_model_decision_ms(data_end_ms)
    if role == MODEL_PERIOD_CUTOFF_CHECK and data_end_ms is not None:
        raise PartitionError(
            f"the {MODEL_PERIOD_CUTOFF_CHECK} role reads the shipped weights only "
            f"(docs/RETRAINING.md, 5.4): a checkpoint trained on candles before "
            f"{utc_text(data_end_ms)} is refused"
        )
    if role != MODEL_PERIOD_EVALUATION or allow_pretraining_period or decision_ms >= first_ms:
        return
    if data_end_ms is None:
        raise PartitionError(
            f"decision hour {utc_text(decision_ms)} is before "
            f"{utc_text(KRONOS_FIRST_DECISION_MS)}: "
            "the pretrained Kronos saw Binance up to June 2024 (pass "
            "allow_pretraining_period=True only in tests)"
        )
    raise PartitionError(
        f"decision hour {utc_text(decision_ms)} is before {utc_text(first_ms)}, the first hour "
        f"this checkpoint may forecast: it was trained on candles before {utc_text(data_end_ms)} "
        "and starts from the pretrained Kronos, which saw Binance up to June 2024 (pass "
        "allow_pretraining_period=True only in tests)"
    )


def require_kronos_period(
    decision_hour: datetime, *, allow_pretraining_period: bool = False
) -> None:
    """Refuse a forecast with the pretrained weights before :data:`KRONOS_FIRST_DECISION`.

    Kronos as shipped saw Binance up to June 2024 (SPEC section 6); tests opt in explicitly.
    The shipped-weights ``"evaluation"`` rule of :func:`require_model_period`, kept for the
    callers that forecast with the shipped weights only.
    """
    require_model_period(
        decision_hour,
        role=MODEL_PERIOD_EVALUATION,
        allow_pretraining_period=allow_pretraining_period,
    )


__all__ = [
    "AFTER_PARTITION",
    "BEFORE_PARTITION",
    "EMBARGO_HOURS",
    "EVAL_LENGTH_MAX",
    "KRONOS_FIRST_DECISION",
    "KRONOS_FIRST_DECISION_MS",
    "MAX_CONTEXT",
    "MODEL_PERIOD_CUTOFF_CHECK",
    "MODEL_PERIOD_EVALUATION",
    "MODEL_PERIOD_PILOT_CHECK",
    "MODEL_PERIOD_ROLES",
    "MODEL_PERIOD_TIMING",
    "N1_PARTITION",
    "N1_TRAINING_END",
    "N1_VALIDATION_START",
    "PARTITION",
    "SEALED_START",
    "SEALED_START_MS",
    "TRAINING_END",
    "TRAINING_START",
    "VALIDATION_END",
    "VALIDATION_START",
    "PartitionError",
    "SealedAccess",
    "SealedPartitionError",
    "Segment",
    "check_embargo",
    "check_evaluation_length",
    "first_model_decision_ms",
    "is_sealed",
    "require_accessible",
    "require_kronos_period",
    "require_model_period",
    "require_study_range",
    "sample_fits",
    "sealed_access",
    "segment_named",
    "segment_of",
]
