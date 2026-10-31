"""``kronos-agent verify [--kronos] [--weights]``: manifests of the vendored code and the weights.

A weights repository with none of its registered files present is reported as absent (exit 0, so
the check passes on machines without the gated repositories); a partial one, e.g. an interrupted
download, is a problem (exit 1).
"""

from __future__ import annotations

import argparse

from ..kronos.vendored import verify_vendored_tree
from ..kronos.weights import load_registry, verify_repo


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "verify", help="verify third_party/kronos and weights/ against the manifests"
    )
    parser.add_argument("--kronos", action="store_true", help="only the vendored tree")
    parser.add_argument("--weights", action="store_true", help="only the weights")


def run(args: argparse.Namespace) -> int:
    do_kronos = args.kronos or not args.weights
    do_weights = args.weights or not args.kronos
    failed = False
    if do_kronos:
        report = verify_vendored_tree(strict=True, check_blob_ids=True)
        if report.ok:
            print(f"third_party/kronos: OK ({report.checked} files verified)")
        else:
            failed = True
            for problem in report.problems:
                print(f"third_party/kronos PROBLEM: {problem}")
    if do_weights:
        for repo in load_registry().values():
            report = verify_repo(repo)
            if report.ok:
                print(f"weights {repo.repo_id}: OK ({report.checked} files)")
            elif report.checked == 0:
                print(f"weights {repo.repo_id}: absent (policy {repo.policy})")
            else:
                failed = True
                for problem in report.problems:
                    print(f"weights {repo.repo_id} PROBLEM: {problem}")
    return 1 if failed else 0
