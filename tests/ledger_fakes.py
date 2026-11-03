"""Stand-ins for the tests of the experiment ledger and the test budget (``kronos_agent.ledger``,
``kronos_agent.budget``): recipe ids, pre-registration bundle files, valid payloads and a recorded
order of the owner (a support module, not a test module, like ``finetune_fakes``).

Everything lands on the test's isolated roots (``conftest.isolated_roots``): the ledger under
``paths.ledger_dir()``, the bundles under ``paths.reports_dir()``, unless a test passes its own
roots. The texts hold no personal data, path or host name: the ledger refuses them.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kronos_agent import budget, ledger, paths, provenance
from kronos_agent.hashing import canonical_json

# a constant code provenance for the tests that append many lines (code_version runs git and
# hashes the source tree at every append)
STAND_IN_CODE: dict[str, Any] = {"commit": None, "dirty": None, "source_tree_sha256": "c" * 64}
FORECASTING_CODE = "f" * 64
TRAINING_CODE = "e" * 64
MONDAY = datetime(2026, 11, 2, 12, tzinfo=UTC)  # ISO week 2026-W45, quarter 2026-Q4
WEEK = timedelta(days=7)
ORDER_TEXT = "open the sealed segment for this candidate, once"


def recipe_id(name: object) -> str:
    """A stand-in recipe id: the sha256 of ``stand-in recipe <name>``."""
    return hashlib.sha256(f"stand-in recipe {name}".encode()).hexdigest()


R1 = recipe_id("R1")


def bundle_relpath(recipe: str) -> str:
    return f"finetune/prereg-{recipe[:8]}.config.json"


def write_bundle(recipe: str, *, reports_root: Path | None = None, content: Any = None) -> Path:
    """A stand-in bundle file of ``recipe`` under the reports root; returns its path."""
    base = Path(reports_root) if reports_root is not None else paths.reports_dir()
    path = base / bundle_relpath(recipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = content if content is not None else {"recipe_id": recipe, "stand_in": True}
    path.write_bytes(canonical_json(document))
    return path


def register_bundle(
    recipe: str,
    tests: tuple[str, ...] = ledger.TEST_NAMES,
    *,
    root: Path | None = None,
    reports_root: Path | None = None,
    now: datetime | None = None,
) -> ledger.LedgerEntry:
    """Write the stand-in bundle of ``recipe`` and record its ``preregistration`` entry."""
    write_bundle(recipe, reports_root=reports_root)
    return ledger.record_preregistration(
        bundle_relpath=bundle_relpath(recipe),
        recipe_id=recipe,
        tests=tests,
        forecasting_code_sha256=FORECASTING_CODE,
        training_code_sha256=TRAINING_CODE,
        root=root,
        reports_root=reports_root,
        now=now,
    )


def last_moment(root: Path | None = None) -> datetime:
    """The time of the ledger's last entry (the epoch for an empty ledger): an append dated
    earlier is refused, because the ledger's times never decrease."""
    entries = ledger.read_ledger(root)
    return entries[-1].moment if entries else datetime(1970, 1, 1, tzinfo=UTC)


def recorded_order(
    recipe: str,
    *,
    start: datetime = MONDAY,
    root: Path | None = None,
    reports_root: Path | None = None,
    data_root: Path | None = None,
) -> ledger.LedgerEntry:
    """The bundle of ``recipe``, its T-A reading at ``start`` and its T-B reading one ISO week
    later (on ``data_root``, default the data root), then the owner's order for the T-B
    candidate; returns the ``owner_order`` entry. ``start`` must not be earlier than the
    ledger's last entry (:func:`last_moment`)."""
    roots = {"root": root, "reports_root": reports_root, "data_root": data_root}
    register_bundle(recipe, root=root, reports_root=reports_root, now=start)
    budget.spend_test("T-A", recipe_id=recipe, now=start, **roots)
    budget.spend_test("T-B", recipe_id=recipe, now=start + WEEK, **roots)
    return ledger.record_order(
        scope=ledger.ORDER_SCOPE_OPEN_SEALED,
        candidate=ledger.id_for_test("T-B", recipe),
        text=ORDER_TEXT,
        root=root,
        now=start + WEEK,
    )


def exploration(**changes: Any) -> dict[str, Any]:
    """Keyword arguments of a valid :func:`kronos_agent.ledger.record_exploration` call."""
    return {
        "activity": "pilot-fold",
        "subject": "q-2024-07",
        "stage": "planned",
        "recipe_id": R1,
        "inputs": ["training"],
        "outcomes": ["training"],
        "details": {"max_epochs": 6, "unit": "mps_hours", "timing_record": "timing-2"},
        **changes,
    }


def fit(**changes: Any) -> dict[str, Any]:
    """Keyword arguments of a valid :func:`kronos_agent.ledger.record_fit` call (registered)."""
    return {
        "checkpoint_id": f"ks-btc-{R1[:8]}-q-2024-07",
        "registry_line_sha256": "1" * 64,
        "recipe_id": R1,
        "fold": "q-2024-07",
        "replicate": 0,
        "run_id": "q-2024-07-20261103T120000Z",
        "status": "registered",
        "attempt": 1,
        "blinded": False,
        "fit_record_sha256": "2" * 64,
        "device": "mps",
        "threads": 2,
        "wall_seconds": 17012.5,
        "cpu_seconds": 2049.0,
        "approval_seq": None,
        "order_id": None,
        **changes,
    }


def approval(**changes: Any) -> dict[str, Any]:
    """Keyword arguments of a valid :func:`kronos_agent.ledger.record_gate3_approval` call."""
    return {
        "command": "finetune fold",
        "subject": "q-2024-07",
        "unit": "mps_hours",
        "approved": 10.0,
        "estimate": 4.945,
        "given_at": MONDAY,
        "estimate_detail": {"first": {"wall_seconds": 17801.8}, "refit": None},
        **changes,
    }


def fixed_code(monkeypatch: Any) -> None:
    """Every line appended from now on records :data:`STAND_IN_CODE` as its ``code``."""
    monkeypatch.setattr(provenance, "code_version", lambda: dict(STAND_IN_CODE))
