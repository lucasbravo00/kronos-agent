"""``kronos-agent fetch-weights --model small [--approved-by-owner]``.

Downloads a model and its paired tokenizer at the registered revisions, verifying size and sha256
against ``manifests/weights.json``. Repositories other than Kronos-small and Kronos-Tokenizer-base
are gated (SPEC section 17, gate 4) and need ``--approved-by-owner``. A gate-4 refusal prints
``REFUSED:`` and exits 2; a failed download (HTTP error, refused redirect, size or sha256 mismatch,
dropped connection) prints ``ERROR:`` and exits 1, leaving no partial file behind.
"""

from __future__ import annotations

import argparse

import httpx

from ..kronos.weights import Gate4Error, WeightsError, download_model, verify_model
from ..models import MODEL_SPECS


def register(subparsers) -> None:
    parser = subparsers.add_parser("fetch-weights", help="download and verify Kronos weights")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="small")
    parser.add_argument(
        "--approved-by-owner",
        action="store_true",
        help="the owner explicitly approved downloading a gated repository (gate 4)",
    )


def run(args: argparse.Namespace) -> int:
    try:
        written = download_model(args.model, approved_by_owner=args.approved_by_owner)
    except Gate4Error as exc:
        print(f"REFUSED: {exc}")
        return 2
    except (WeightsError, httpx.HTTPError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    report = verify_model(args.model)
    if not report.ok:
        for problem in report.problems:
            print(f"PROBLEM: {problem}")
        return 1
    print(f"{args.model}: {len(written)} file(s) downloaded; {report.checked} files verified")
    return 0
