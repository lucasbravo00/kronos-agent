"""``kronos-agent mismatch``: study 3, Binance vs BingX real market (``reports/data/mismatch``).

- ``mismatch``: every section without Kronos (alignment, returns, prices, volume, the ``amount``
  proxy) on every hour present in both 1h stores from 2024-04-27T10:00Z to the sealed start.
  When every fan of a pre-registered sample (the primary one first, then the fallback) is stored
  under the current forecast config, the Kronos part is rebuilt from those fans (nothing is
  computed, so gate 3 is not involved) with the gate-3 records that cover them; otherwise the
  report says the part has not run and how many of its fans are stored. A stored fan that no
  longer matches its window (it is never overwritten) leaves the part not run too, the reason
  naming the fan: move it aside, then ``mismatch --kronos`` computes it again (``--kronos``
  itself fails on such a fan).
- ``mismatch --kronos [--fallback] --estimate``: the gate-3 estimate of the Kronos signal
  agreement (three fans per sample hour: ``binance/forecast``, ``bingx/forecast`` with the same
  seeds and ``bingx/mismatch-floor``); no fan of the sample runs.
- ``mismatch --kronos [--fallback] [--workers N] [--approved-by-owner CPU_HOURS]``: computes
  (or reuses) the fans behind gate 3 and writes the whole report. ``CPU_HOURS`` is the figure the
  owner approved in chat for the estimate shown; above 3600 CPU-seconds the batch runs only with
  it and only when the measured CPU-hours are at most 1.25 times it. Before the first fan the
  batch appends its gate-3 record (the approved figure, the measured estimate, the forecast
  config, the design, the dropped tasks and the command's start, when the flag was given) to
  ``reports/data/mismatch.gate3.jsonl``; a rerun with nothing pending adds none. ``--fallback``
  uses the pre-registered 97-hour sample (183 hours, about 0.96 CPU-hours at the measured speed)
  if the owner declines the compute of the 61-hour one.
- ``mismatch --record-config``: records the study's definition only.

The study's definition is verified (or recorded) first, so a changed definition is refused (exit
2) before any candle is read or any fan runs. Every section without Kronos, the alignment check
included, is computed before any Kronos work (also with ``--estimate``), so approved compute is
never spent on a report that then fails. The runtime is configured with the governed thread
count before anything imports torch; the batch flow is :func:`kronos_agent.cli.forecast.run_batch`.
A store read while a ``data`` run holds its job lock is refused (exit 2).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..data.partition import SealedPartitionError
from ..params import Params, ParamsError
from . import (
    EXIT_FAILURE,
    EXIT_OK,
    data_root,
    display_path,
    positive_int,
    positive_number,
    report_directory,
    run_guarded,
)

if TYPE_CHECKING:
    from ..kronos.batch import FanSummary
    from .forecast import BatchRun


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "mismatch", help="study 3: Binance vs BingX real-market mismatch (reports/data/mismatch)"
    )
    parser.add_argument(
        "--kronos",
        action="store_true",
        help="add the Kronos signal agreement: 3 fans per sample hour behind gate 3",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="with --kronos: the pre-registered 97-hour fallback sample instead of the 61-hour one",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="with --kronos: measure and print the gate-3 estimate, run nothing else",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        metavar="N",
        help="with --kronos: spawn-started worker processes (default 1; 4 GiB of RAM for more)",
    )
    parser.add_argument(
        "--approved-by-owner",
        type=positive_number,
        default=None,
        metavar="CPU_HOURS",
        help="with --kronos: the CPU-hours the owner approved in chat for the estimate shown "
        "(gate 3); a measured estimate well above it is refused",
    )
    parser.add_argument(
        "--record-config",
        action="store_true",
        help="only record the study's definition (reports/data/mismatch.config.json)",
    )
    parser.set_defaults(usage_error=parser.error)


def _check_usage(args: argparse.Namespace) -> None:
    kronos_only = [
        flag
        for flag, value in (
            ("--fallback", args.fallback),
            ("--estimate", args.estimate),
            ("--workers", args.workers),
            ("--approved-by-owner", args.approved_by_owner),
        )
        if value is not None and value is not False
    ]
    if args.record_config and (args.kronos or kronos_only):
        args.usage_error("--record-config runs nothing: give it alone")
    if kronos_only and not args.kronos:
        args.usage_error(f"{', '.join(kronos_only)} need --kronos")


def _now() -> datetime:
    """The UTC clock of the command (a module function, so tests replace it)."""
    return datetime.now(UTC)


def run(args: argparse.Namespace) -> int:
    started_at = _now()  # the command's start: when --approved-by-owner was given
    _check_usage(args)
    params: Params | None = None
    refusals: tuple[type[BaseException], ...] = ()
    failures: tuple[type[BaseException], ...] = ()
    if not args.record_config:
        from .forecast import kronos_errors, load_and_configure

        params = load_and_configure()  # the Kronos part, run or rebuilt, needs the runtime
        if params is None:
            return EXIT_FAILURE
        refusals, failures = kronos_errors()
    from ..data.fsutil import LockBusyError
    from ..data.mismatch import MismatchError
    from ..data.store import StoreError
    from ..report.writer import ConfigChangedError, ReportError

    return run_guarded(
        lambda: _study(args, params, started_at),
        refusals=(ConfigChangedError, SealedPartitionError, LockBusyError, *refusals),
        failures=(MismatchError, StoreError, ReportError, ParamsError, *failures),
    )


def _study(args: argparse.Namespace, params: Params | None, started_at: datetime) -> int:
    from ..data import mismatch
    from ..report.writer import record_config

    directory = report_directory()
    root = data_root()
    digest = record_config(mismatch.STEM, mismatch.study_config(), directory=directory)
    config_path = display_path(directory / f"{mismatch.STEM}.config.json")
    if args.record_config:
        print(f"recorded {config_path} (sha256 {digest})")
        return EXIT_OK
    print(f"study config {config_path} (sha256 {digest[:12]}) verified", flush=True)
    assert params is not None
    study = mismatch.prepare_study(root=root, directory=directory)
    print(
        "sections without Kronos computed (lag 0 is the largest return correlation); "
        "nothing written yet",
        flush=True,
    )
    reason: str | None = None
    if args.kronos:
        outcome = _kronos_batch(args, params, started_at, directory)
        if outcome.summaries is None:
            return EXIT_OK
        summaries: list[FanSummary] | None = list(outcome.summaries)
        gate3: dict[str, Any] | None = {
            "estimate": outcome.estimate,
            "approved_cpu_hours": outcome.approved_cpu_hours,
            "dropped": [dict(item) for item in outcome.prepared.dropped],
        }
    else:
        summaries, gate3, reason = _stored_part(params, root)
    payload = mismatch.write_study(
        study, kronos_summaries=summaries, gate3=gate3, kronos_reason=reason, directory=directory
    )
    _print_report(payload)
    json_path = directory / f"{mismatch.STEM}.json"
    print(f"wrote {display_path(json_path)} and {display_path(json_path.with_suffix('.md'))}")
    return EXIT_OK


def _kronos_batch(
    args: argparse.Namespace, params: Params, started_at: datetime, directory: Path
) -> BatchRun:
    """The gate-3 batch of the sample; its record is appended once gate 3 has passed and before
    the first fan (only when some fan is pending)."""
    from ..data import mismatch
    from .forecast import run_batch

    design = "fallback" if args.fallback else "primary"
    hours = mismatch.kronos_fallback_hours() if args.fallback else mismatch.kronos_sample_hours()
    print(f"Kronos sample ({design}): {len(hours)} decision hours x 3 fans", flush=True)

    def record(planned: BatchRun) -> None:
        path = mismatch.append_gate3_record(
            mismatch.gate3_record(
                design=design,
                estimate=planned.estimate,
                approved_cpu_hours=planned.approved_cpu_hours,
                dropped=planned.prepared.dropped,
                given_at=started_at,
            ),
            directory=directory,
        )
        print(f"gate-3 record appended to {display_path(path)} before the first fan", flush=True)

    return run_batch(
        mismatch.kronos_tasks(hours),
        params=params,
        workers=args.workers or 1,
        estimate_only=args.estimate,
        approved_cpu_hours=args.approved_by_owner,
        started_at=started_at,
        before_fans=record,
    )


def _stored_part(
    params: Params, root: Path
) -> tuple[list[FanSummary] | None, dict[str, Any] | None, str | None]:
    """``(summaries, gate3, None)`` when every fan of a pre-registered sample is stored under
    the current forecast config (no fan is computed: a fan found missing on the way is refused
    as a stale estimate), else ``(None, None, reason)`` with the stored counts.

    A stored fan of a sample that no longer matches its window (its store changed after the fan
    was computed, after a ``data binance --refresh`` say) is never overwritten: the part is not
    rebuilt, and the reason names the fan and the way out, so the report of the current stores
    is still written and an earlier report's Kronos part, made from the old stores, is removed.
    """
    from ..data import mismatch
    from ..data.forecast_store import ForecastStoreError
    from ..kronos.batch import prepare_tasks, run_fan_batch

    counts: list[str] = []
    config_id = ""
    for design, hours in (
        ("primary", mismatch.kronos_sample_hours()),
        ("fallback", mismatch.kronos_fallback_hours()),
    ):
        try:
            prepared = prepare_tasks(mismatch.kronos_tasks(hours), params=params, root=root)
            config_id = prepared.config_id
            summaries = None
            if prepared.tasks and not prepared.pending:
                summaries = run_fan_batch(prepared, workers=1, params=params, root=root)
        except ForecastStoreError as exc:
            reason = (
                f"not run: a stored fan of the {design} sample cannot be reused ({exc}); move "
                "that fan aside (other stored fans of the sample may be stale too), then "
                "`kronos-agent mismatch --kronos` computes the missing fans behind gate 3"
            )
            return None, None, reason
        if summaries is not None:
            print(
                f"Kronos part rebuilt from the {len(summaries)} stored fans of the {design} "
                f"sample (forecast config {config_id}); no fan computed",
                flush=True,
            )
            gate3 = {"estimate": None, "dropped": [dict(item) for item in prepared.dropped]}
            return summaries, gate3, None
        stored = len(prepared.tasks) - prepared.pending
        counts.append(f"{design} sample {stored} of {len(prepared.tasks)}")
    reason = (
        "not run: not every fan of a pre-registered sample is stored under forecast config "
        f"{config_id} ({'; '.join(counts)} fans stored); `kronos-agent mismatch --kronos` "
        "computes them behind gate 3"
    )
    return None, None, reason


def _print_report(payload: dict[str, Any]) -> None:
    hours = (payload.get("samples") or {}).get("hours") or {}
    print(
        f"hours present in both sources: {hours.get('used')} of {hours.get('eligible')}; "
        f"{payload.get('statistics_with_intervals')} statistics with intervals "
        f"({payload.get('expected_chance_misses')} expected to miss by chance at 95 %)"
    )
    kronos = (payload.get("sections") or {}).get("kronos") or {}
    if kronos.get("status") != "run":
        print(f"Kronos signal agreement: {kronos.get('reason')}")
        return
    readings = kronos.get("readings") or {}
    for name in ("a_x_minus_a_f", "a_xd_minus_a_f"):
        item = readings.get(name) or {}
        print(
            f"Kronos {name} (pre-registered): {item.get('reading')} (interval "
            f"{item.get('interval_position')})"
        )
