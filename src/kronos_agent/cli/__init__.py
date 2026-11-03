"""``kronos-agent <command>``: an argparse registry with one module per command.

Command modules expose ``register(subparsers)`` and ``run(args) -> int``. At module level they
import only the standard library, the helpers of this package and
``kronos_agent.{paths, params, models, data.timeutil, data.partition}`` (``kronos.vendored`` and
``kronos.weights`` defer their heavy imports and are fine for the step-1 commands): ``--help`` has
to stay cheap (no ``torch``, ``huggingface_hub``, ``numpy`` or ``pandas``) and
``kronos_agent.runtime.configure_runtime()`` must run inside ``run()`` before anything imports
``torch``. Every registered module is imported strictly, so an entry and its module are added
together: a missing or broken module fails ``--help`` instead of dropping the command.

Exit codes of the step-2 commands (``data``, ``forecast``, ``mismatch``), of ``finetune`` and of
``ledger``: 0 done; 1 failure, printed as ``ERROR: ...``; 2 refusal by a governance rule, printed
as ``REFUSED: ...``: gate 3 without ``--approved-by-owner CPU_HOURS`` or well above that figure,
the gate-4 download budget, the sealed partition, the other partition rules (the Kronos cutoff,
the embargo), a changed study config or a busy lock, the measurement step's own budget
(``finetune timing``), and the experiment ledger's rules (a changed pre-registration bundle, an
order the ledger refuses). argparse usage errors exit 2 as well. An unexpected exception is a
bug and keeps its traceback.

A command stopped by SIGTERM (a service stop, ``kill``) or SIGHUP (its terminal closed) exits
through ``SystemExit`` (code 128 + the signal number), so its ``with`` and ``except`` blocks
remove their temporary files and release their locks; a signal the parent process set to be
ignored (``nohup``) stays ignored. SIGKILL and the OOM killer skip that clean-up: the next run
that takes the job lock removes what they left (:func:`kronos_agent.data.store.job_lock`).

The helpers below are shared by the command modules; they import nothing heavy.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import math
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

from ..data.timeutil import parse_utc_hour
from ..paths import REPO_ROOT, data_dir, redact_home, reports_dir

COMMAND_MODULES: dict[str, str] = {
    "env-report": "kronos_agent.cli.env_report",
    "verify": "kronos_agent.cli.verify",
    "fetch-weights": "kronos_agent.cli.fetch_weights",
    "bench": "kronos_agent.cli.bench",
    "vendor-kronos": "kronos_agent.cli.vendor_kronos",
    "params": "kronos_agent.cli.params",
    "data": "kronos_agent.cli.data",
    "forecast": "kronos_agent.cli.forecast",
    "mismatch": "kronos_agent.cli.mismatch",
    "finetune": "kronos_agent.cli.finetune",
    "ledger": "kronos_agent.cli.ledger",
}

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_REFUSED = 2
REPORTS_SUBDIR = "data"
EXIT_SIGNALS: tuple[str, ...] = ("SIGTERM", "SIGHUP")

ExceptionTypes = tuple[type[BaseException], ...]


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, Callable]]:
    parser = argparse.ArgumentParser(
        prog="kronos-agent",
        description="Kronos x the owner's criterion: research system (BingX demo only).",
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
    with exit_on_signals():
        return int(runners[args.command](args) or 0)


def _raise_system_exit(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def exit_on_signals() -> Iterator[None]:
    """Turn :data:`EXIT_SIGNALS` into ``SystemExit`` for the block, then restore the handlers.

    Only signals whose handler is the default one are changed (one ignored by the parent
    process, as under ``nohup``, stays ignored), and only in the main thread, where Python runs
    signal handlers.
    """
    installed: dict[int, Callable | int] = {}
    if threading.current_thread() is threading.main_thread():
        for name in EXIT_SIGNALS:
            number = getattr(signal, name, None)
            if number is not None and signal.getsignal(number) is signal.SIG_DFL:
                installed[number] = signal.signal(number, _raise_system_exit)
    try:
        yield
    finally:
        for number, previous in installed.items():
            signal.signal(number, previous)


# -- helpers shared by the command modules ---------------------------------------------------------


def utc_hour(text: str) -> datetime:
    """argparse type of a decision hour: ``YYYY-MM-DDTHH:00Z``, ``YYYY-MM-DDTHH:00:00Z`` or
    ``YYYY-MM-DDTHHZ`` (UTC, an exact hour)."""
    try:
        return parse_utc_hour(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def positive_int(text: str) -> int:
    """argparse type of a positive integer."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def positive_number(text: str) -> float:
    """argparse type of a finite number > 0 (the CPU-hours of a gate-3 approval)."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number > 0, got {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"expected a finite number > 0, got {text!r}")
    return value


def data_root() -> Path:
    """The data root the commands read and write (``KRONOS_AGENT_DATA_DIR``, else ``data/``)."""
    return data_dir()


def report_directory() -> Path:
    """``reports/data``: the step-2 study reports, their recorded configs and their figures."""
    return reports_dir() / REPORTS_SUBDIR


def display_path(path: Path) -> str:
    """A path for the terminal: repository-relative when possible, otherwise home-redacted."""
    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return redact_home(str(path))


def run_guarded(
    action: Callable[[], int], *, refusals: ExceptionTypes, failures: ExceptionTypes
) -> int:
    """Run ``action``; a refusal prints ``REFUSED:`` (exit 2), a failure ``ERROR:`` (exit 1).

    Refusals are matched first, so a refusal class that subclasses a failure class (the gate-4
    budget is a Binance Vision error, a changed config a report error) is still a refusal. Any
    other exception propagates with its traceback.
    """
    try:
        return int(action())
    except refusals as exc:
        print(f"REFUSED: {exc}")
        return EXIT_REFUSED
    except failures as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
