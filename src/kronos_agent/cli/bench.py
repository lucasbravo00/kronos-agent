"""``kronos-agent bench``: time the fan of trajectories on CPU and report determinism.

``run()`` configures the runtime (threads, offline Hub, deterministic algorithms) BEFORE the
benchmark and Kronos modules are imported; nothing heavy is imported at module level so that
``kronos-agent --help`` stays cheap. Reports go to ``reports/environment/<label>.bench.json``
and ``.bench.md`` unless ``--json`` prints the JSON to stdout.
"""

from __future__ import annotations

import argparse
import json

from ..models import MODEL_SPECS
from ..paths import REPO_ROOT, default_label

SOURCES = ("upstream_fixture", "synthetic")


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "bench", help="time a Kronos fan of trajectories on CPU and record determinism"
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="small")
    parser.add_argument("--trajectories", type=int, default=20, help="paths per fan (default 20)")
    parser.add_argument("--horizon", type=int, default=8, help="candles per path (default 8)")
    parser.add_argument("--threads", type=int, default=2, help="torch threads (default 2)")
    parser.add_argument("--repeats", type=int, default=3, help="timed fans after one warm-up")
    parser.add_argument("--source", choices=SOURCES, default="upstream_fixture")
    parser.add_argument(
        "--threads-sweep",
        default=None,
        help="comma-separated thread counts timed in subprocesses, e.g. 1,2,10",
    )
    parser.add_argument(
        "--include-mps",
        action="store_true",
        help="also time the fan on MPS in a subprocess (informational, never a reference)",
    )
    parser.add_argument(
        "--label", default=default_label(), help="report name (default: platform-machine)"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the JSON to stdout instead of writing"
    )


def run(args: argparse.Namespace) -> int:
    from ..runtime import RuntimeConfigError, configure_runtime

    try:
        configure_runtime(args.threads)
    except RuntimeConfigError as exc:
        print(f"ERROR: {exc}")
        return 2
    from ..bench import headline_lines, parse_threads_sweep, run_benchmark, write_reports

    try:
        sweep = parse_threads_sweep(args.threads_sweep)
        result = run_benchmark(
            model=args.model,
            trajectories=args.trajectories,
            horizon=args.horizon,
            threads=args.threads,
            repeats=args.repeats,
            source=args.source,
            include_fixture_replica=True,
            threads_sweep=sweep,
            include_mps=args.include_mps,
            label=args.label,
        )
    except (ValueError, RuntimeConfigError) as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    json_path, md_path = write_reports(result, label=args.label)
    for line in headline_lines(result):
        print(line)
    print(f"wrote {json_path.relative_to(REPO_ROOT)} and {md_path.relative_to(REPO_ROOT)}")
    return 0
