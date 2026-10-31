"""``kronos-agent params show|version``."""

from __future__ import annotations

import argparse

from ..params import DEFAULT_PARAMS_PATH, load_params, params_version


def register(subparsers) -> None:
    parser = subparsers.add_parser("params", help="show or version the governed parameters")
    parser.add_argument("action", choices=["show", "version", "validate"])


def run(args: argparse.Namespace) -> int:
    if args.action == "show":
        print(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"), end="")
        return 0
    if args.action == "version":
        print(params_version())
        return 0
    params = load_params()
    print(f"params.yaml valid (version {params.version}); model={params.kronos.model}")
    return 0
