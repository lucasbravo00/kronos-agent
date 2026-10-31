"""``kronos-agent <command>``: an argparse registry with one module per command.

Command modules expose ``register(subparsers)`` and ``run(args) -> int``. They must not import
``torch``, ``huggingface_hub``, ``numpy`` or ``pandas`` at module level, directly or through a
``kronos_agent`` module (``kronos.vendored`` and ``kronos.weights`` defer theirs and are fine):
``--help`` has to stay cheap and ``kronos_agent.runtime.configure_runtime()`` must run inside
``run()`` before those imports. Every registered module is imported strictly, so an entry and its
module are added together: a missing or broken module fails ``--help`` instead of dropping the
command.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable

COMMAND_MODULES: dict[str, str] = {
    "env-report": "kronos_agent.cli.env_report",
    "verify": "kronos_agent.cli.verify",
    "fetch-weights": "kronos_agent.cli.fetch_weights",
    "bench": "kronos_agent.cli.bench",
    "vendor-kronos": "kronos_agent.cli.vendor_kronos",
    "params": "kronos_agent.cli.params",
}


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, Callable]]:
    parser = argparse.ArgumentParser(
        prog="kronos-agent",
        description="Kronos x volume-profile criterion research system (BingX demo only).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    runners: dict[str, Callable] = {}
    for name, module_name in COMMAND_MODULES.items():
        module = importlib.import_module(module_name)
        module.register(subparsers)
        runners[name] = module.run
    return parser, runners


def main(argv: list[str] | None = None) -> int:
    parser, runners = build_parser()
    args = parser.parse_args(argv)
    return int(runners[args.command](args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
