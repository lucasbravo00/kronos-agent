"""``kronos-agent env-report [--label LABEL] [--network] [--json]``: the redacted environment report.

Writes ``reports/environment/<label>.json`` and ``<label>.md`` (``--json`` prints the JSON to
stdout instead). ``--network`` adds one unauthenticated GET to the public BingX server time to
measure the clock offset; nothing else leaves the machine. ``kronos_agent.envinfo`` is imported
inside :func:`run` so that ``--help`` stays cheap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..paths import REPO_ROOT, default_label, redact_home, reports_dir, validate_label
from ..runtime import configure_runtime


def _label(text: str) -> str:
    """A plain file stem: letters, digits, ``.``, ``_`` and ``-`` only (no path separators)."""
    try:
        return validate_label(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "label must be a plain file stem (letters, digits, '.', '_' or '-')"
        ) from None


def _display(path: Path) -> str:
    """Repository-relative path when possible, otherwise the home-redacted absolute path."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return redact_home(str(path))


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "env-report", help="write the redacted environment report to reports/environment/"
    )
    parser.add_argument(
        "--label",
        type=_label,
        default=None,
        help="report name; default <platform>-<machine>, e.g. darwin-arm64",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="also measure the clock offset against the public BingX server time (one GET)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the JSON report to stdout instead of writing files",
    )


def run(args: argparse.Namespace) -> int:
    configure_runtime(2, strict=False)
    from ..envinfo import collect, headline, to_markdown

    label = args.label or default_label()
    report = collect(label=label, network=args.network)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    out_dir = reports_dir() / "environment"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{label}.json"
    md_path = out_dir / f"{label}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    print(f"wrote {_display(json_path)} and {_display(md_path)}")
    print(headline(report))
    return 0
