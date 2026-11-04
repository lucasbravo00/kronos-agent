"""The experiment ledger (``kronos_agent.ledger``; ``docs/RETRAINING.md`` 7.3; test T27) and its
command, ``kronos-agent ledger``.

Appends are hash-chained canonical-JSON lines written under the ledger's lock, from a payload
that may be built from the entries read under it; reading verifies every line and refuses a torn,
non-canonical, edited, removed or reordered one, and one that breaks a rule, naming the line and
never repairing it. The rules: no validation or sealed outcomes (or sealed inputs) in an
exploration, a fit's checkpoint only when registered, a gate-3 approval within 1.25 times the
figure, references to earlier entries of the right kind, one bundle per recipe and one order per
T-B candidate, times that never decrease, the test budget and the order of the readings as the
test lines record them, and nothing that may not be published (personal data at append, IPv4
literals, absolute paths, BingX host names, non-ASCII text, non-finite numbers): a ledger
verifies the same whatever the reader's login, host and home. Only the repository's ledger
authorizes reading the repository's data (a stand-in repository under the test's own
directory). The command runs through ``cli.main`` on the test's isolated roots; nothing here
trains, times or forecasts. The test budget has its own module (``tests/test_budget.py``); the
sealed guard's use of the ledger is in ``tests/test_partition.py``.
"""

from __future__ import annotations

import ast
import getpass
import json
import re
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ledger_fakes as lf
import numpy as np
import pytest

from kronos_agent import budget, cli, ledger, paths
from kronos_agent.cli import ledger as ledger_cli
from kronos_agent.data import partition
from kronos_agent.data.fsutil import LockBusyError, exclusive_lock, lock_is_held
from kronos_agent.exchange.public_market import PUBLIC_HOST
from kronos_agent.hashing import canonical_json, sha256_bytes, sha256_file
from kronos_agent.ledger import (
    BundleChanged,
    LedgerChainError,
    LedgerEntry,
    LedgerError,
    LedgerRefused,
    OrderNotFound,
)
from kronos_agent.params import params_version
from kronos_agent.paths import REPO_ROOT

HEAVY_MODULES = ("torch", "huggingface_hub", "pandas", "numpy", "matplotlib")
# an absolute path in printed text (the ledger's own pattern, from finetune.timing)
ABSOLUTE_PATH = re.compile(r"(?<![\w.~/-])(?:~/|/)[^\s'\"`<>,;:()\[\]{}]+")
T_B = ledger.id_for_test("T-B", lf.R1)
NON_ASCII = "caf" + chr(0xE9)  # built from its code point: the sources stay ASCII


def ledger_bytes() -> bytes:
    return ledger.ledger_path().read_bytes()


def three() -> tuple[LedgerEntry, LedgerEntry, LedgerEntry]:
    """An exploration, a gate-3 approval and the fit that names the approval, in order."""
    first = ledger.record_exploration(**lf.exploration())
    second = ledger.record_gate3_approval(**lf.approval())
    third = ledger.record_fit(**lf.fit(approval_seq=second.seq))
    return first, second, third


def relined(entry: LedgerEntry, previous: bytes | None, **changes: object) -> bytes:
    """``entry``'s line with ``changes``, chained after the line ``previous``."""
    document = {**json.loads(entry.line), **changes}
    document["prev_sha256"] = ledger.GENESIS_SHA256 if previous is None else sha256_bytes(previous)
    return canonical_json(document)


def with_payload(entry: LedgerEntry, previous: bytes | None, **changes: object) -> bytes:
    """``entry``'s line with ``changes`` in its payload, chained after the line ``previous``."""
    payload = {**json.loads(entry.line)["payload"], **changes}
    return relined(entry, previous, payload=payload)


def lines(*items: bytes) -> bytes:
    return b"".join(item + b"\n" for item in items)


# -- appending and reading -------------------------------------------------------------------------


def test_appends_are_hash_chained_canonical_lines():
    first, second, third = three()
    assert ledger_bytes() == lines(first.line, second.line, third.line)
    assert [entry.seq for entry in (first, second, third)] == [1, 2, 3]
    assert first.document["prev_sha256"] == ledger.GENESIS_SHA256
    assert second.document["prev_sha256"] == first.line_sha256 == sha256_bytes(first.line)
    assert third.document["prev_sha256"] == second.line_sha256
    read = ledger.read_ledger()
    assert [entry.line for entry in read] == [first.line, second.line, third.line]
    for entry in read:
        assert canonical_json(dict(entry.document)) == entry.line
        assert sorted(entry.document) == sorted(ledger.LINE_KEYS)
        assert sorted(entry.payload) == sorted(ledger.PAYLOAD_KEYS[entry.kind])
        assert entry.document["schema"] == "experiment" and entry.document["schema_version"] == 1
        assert entry.params_version == params_version()
        assert sorted(entry.code) == sorted(ledger.CODE_KEYS)
        assert re.fullmatch(r"[0-9a-f]{64}", entry.code["source_tree_sha256"])
        assert re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", entry.ts)
        assert entry.moment == ledger.parse_ts(entry.ts) and entry.moment.tzinfo is UTC
    assert [entry.kind for entry in read] == ["exploration", "gate3_approval", "fit"]
    assert [entry.line for entry in ledger.entries_of("fit")] == [third.line]
    assert [entry.line for entry in ledger.entries_of("fit", entries=read[:2])] == []
    assert ledger.ledger_status() == {
        "entries": 3,
        "head": third.line_sha256,
        "chain_ok": True,
        "problem": None,
    }
    approval = second.payload
    assert (approval["approved"], approval["estimate"]) == (10.0, 4.945)
    assert approval["ratio"] == 4.945 / 10.0 and approval["above_gate3"] is True
    assert (approval["tolerance"], approval["gate3_hours"]) == (1.25, 1.0)
    assert approval["given_at"] == "2026-11-02T12:00:00Z"
    assert third.payload["approval_seq"] == 2 and third.payload["checkpoint_id"].startswith("ks-")


def test_an_absent_ledger_is_empty_and_reading_creates_nothing():
    assert ledger.ledger_path() == paths.ledger_dir() / "experiments.jsonl"
    assert paths.ledger_dir() == paths.reports_dir() / "ledger"
    assert ledger.read_ledger() == ()
    assert ledger.ledger_status() == {
        "entries": 0,
        "head": ledger.GENESIS_SHA256,
        "chain_ok": True,
        "problem": None,
    }
    assert ledger.entries_of("test") == () and ledger.verify_bundles() == ()
    assert ledger.preregistration_for(lf.R1) is None and ledger.order_used("order-0001") is False
    with pytest.raises(OrderNotFound, match="no owner_order order-0001"):
        ledger.find_order("order-0001")
    assert not paths.ledger_dir().exists()  # reading creates nothing, not even the lock


def test_an_explicit_root_holds_its_own_ledger(tmp_path):
    root = tmp_path / "ledger"
    entry = ledger.record_exploration(**lf.exploration(), root=root)
    assert ledger.ledger_path(root).read_bytes() == entry.line + b"\n"
    assert [item.line for item in ledger.read_ledger(root)] == [entry.line]
    assert ledger.read_ledger() == ()  # the default ledger is another file


# -- a broken ledger is refused, never repaired ----------------------------------------------------


def torn(a, b, c):
    return lines(a.line, b.line) + c.line


def non_canonical(a, b, c):
    return lines(a.line.replace(b'"kind":"exploration"', b'"kind": "exploration"'), b.line)


def not_json(a, b, c):
    return lines(b"{not json", b.line, c.line)


def too_long(a, b, c):
    return lines(b"x" * (ledger.MAX_LINE_BYTES + 1), b.line)


def edited_first(a, b, c):
    return lines(with_payload(a, None, stage="done"), b.line, c.line)


def removed_first(a, b, c):
    return lines(b.line, c.line)


def reordered(a, b, c):
    return lines(b.line, a.line, c.line)


def unknown_kind(a, b, c):
    return lines(a.line, relined(b, a.line, kind="opinion"), c.line)


def missing_key(a, b, c):
    document = json.loads(b.line)
    del document["params_version"]
    return lines(a.line, canonical_json(document), c.line)


def extra_key(a, b, c):
    return lines(a.line, relined(b, a.line, note="x"), c.line)


def extra_payload_key(a, b, c):
    return lines(a.line, with_payload(b, a.line, note="x"), c.line)


def wrong_seq(a, b, c):
    return lines(a.line, relined(b, a.line, seq=5), c.line)


def non_finite(a, b, c):
    return lines(a.line, b.line.replace(b'"approved":10.0', b'"approved":NaN'), c.line)


def non_ascii(a, b, c):
    return lines(with_payload(a, None, details={"note": NON_ASCII}), b.line, c.line)


def validation_outcome(a, b, c):
    return lines(with_payload(a, None, outcomes=["validation"]), b.line, c.line)


def dangling_approval(a, b, c):
    return lines(a.line, b.line, with_payload(c, b.line, approval_seq=1))


BROKEN_CHAIN = "prev_sha256 is not the sha256 of the previous line: the chain is broken"


@pytest.mark.parametrize(
    "mutation, message, verified",
    [
        (torn, "ledger line 3: no LF at the end of the file (a torn write)", 2),
        (non_canonical, "ledger line 1: not canonical JSON", 0),
        (not_json, "ledger line 1: not JSON", 0),
        (too_long, "ledger line 1: longer than 65536 bytes", 0),
        (edited_first, f"ledger line 2: {BROKEN_CHAIN}", 1),
        (removed_first, "ledger line 1: seq 2 is not the line number 1", 0),
        (reordered, "ledger line 1: seq 2 is not the line number 1", 0),
        (unknown_kind, "ledger line 2: unknown kind 'opinion'", 1),
        (missing_key, "ledger line 2: the line misses ['params_version']", 1),
        (extra_key, "ledger line 2: the line misses [] and has unknown keys ['note']", 1),
        (extra_payload_key, "ledger line 2: the gate3_approval payload misses [] and has", 1),
        (wrong_seq, "ledger line 2: seq 5 is not the line number 2", 1),
        (non_finite, "ledger line 2: not JSON (NaN is not a finite JSON number)", 1),
        (non_ascii, "ledger line 1: the string at payload.details.note is not printable", 0),
        (validation_outcome, "ledger line 1: reading validation outcomes is a test", 0),
        (dangling_approval, "ledger line 3: fit: approval_seq 1 names no earlier gate3_", 2),
    ],
)
def test_a_broken_ledger_is_refused_and_never_repaired(mutation, message, verified):
    data = mutation(*three())
    ledger.ledger_path().write_bytes(data)
    with pytest.raises(LedgerChainError, match=re.escape(message)):
        ledger.read_ledger()
    status = ledger.ledger_status()
    assert status["chain_ok"] is False and message in status["problem"]
    assert status["entries"] == verified
    with pytest.raises(LedgerChainError, match=re.escape(message)):
        ledger.record_exploration(**lf.exploration())
    with pytest.raises(LedgerChainError, match=re.escape(message)):
        ledger.find_order("order-0001")
    with pytest.raises(partition.SealedPartitionError, match="experiment ledger"):
        partition.sealed_access("order-0001")
    assert ledger_bytes() == data  # never repaired


def test_the_last_line_is_protected_only_by_its_recorded_line_sha256():
    """The chain cannot see an edit that keeps the last line a valid entry, nor its removal: the
    line sha256 recorded where the entry is used (and printed by the commands) does."""
    first, second, third = three()
    edited = with_payload(third, second.line, wall_seconds=1.0)
    ledger.ledger_path().write_bytes(lines(first.line, second.line, edited))
    status = ledger.ledger_status()
    assert status["chain_ok"] is True and status["head"] == sha256_bytes(edited)
    assert ledger.read_ledger()[-1].line_sha256 != third.line_sha256  # only this tells
    ledger.ledger_path().write_bytes(lines(first.line, second.line))  # the last line removed
    assert ledger.ledger_status() == {
        "entries": 2,
        "head": second.line_sha256,
        "chain_ok": True,
        "problem": None,
    }


# -- appends: the lock and the callable payload ----------------------------------------------------


def test_an_append_while_another_holds_the_lock_is_refused():
    first = ledger.record_exploration(**lf.exploration())
    lock = paths.ledger_dir() / ledger.LOCK_NAME
    busy = "ledger/.lock is locked by another run"
    with exclusive_lock(lock), pytest.raises(LockBusyError, match=busy):
        ledger.record_exploration(**lf.exploration(stage="done"))
    assert ledger_bytes() == first.line + b"\n"


def test_a_callable_payload_sees_the_entries_read_under_the_lock():
    first = ledger.record_exploration(**lf.exploration())
    payload = dict(lf.exploration(stage="done"))
    seen: list[tuple[bytes, ...]] = []

    def build(entries):
        assert lock_is_held(paths.ledger_dir() / ledger.LOCK_NAME)
        seen.append(tuple(entry.line for entry in entries))
        return payload

    second = ledger.append_entry("exploration", build)
    assert seen == [(first.line,)]
    assert second.seq == 2 and second.payload["stage"] == "done"
    assert not lock_is_held(paths.ledger_dir() / ledger.LOCK_NAME)
    before = ledger_bytes()

    def refuse(entries):
        raise LedgerRefused("a rule refuses it")

    with pytest.raises(LedgerRefused, match="a rule refuses it"):
        ledger.append_entry("exploration", refuse)
    with pytest.raises(LedgerError, match="must be a mapping"):
        ledger.append_entry("exploration", lambda entries: ["not", "a", "mapping"])
    assert ledger_bytes() == before  # nothing appended


def test_append_entry_refuses_unknown_kinds_and_malformed_payloads():
    payload = {
        key: value
        for key, value in lf.exploration().items()
        if key in ledger.PAYLOAD_KEYS["exploration"]
    }
    ledger.append_entry("exploration", payload)
    before = ledger_bytes()
    cases = [
        ("opinion", payload, "kind must be one of"),
        ("exploration", ["not a mapping"], "a mapping or a callable"),
        ("exploration", {k: v for k, v in payload.items() if k != "details"}, "misses ['details']"),
        ("exploration", {**payload, "note": "x"}, "has unknown keys ['note']"),
        ("exploration", {**payload, "inputs": ["validation", "training"]}, "sorted list"),
        ("exploration", {**payload, "inputs": "training"}, "sorted list"),
        ("fit", payload, "the fit payload misses"),
    ]
    for kind, content, message in cases:
        with pytest.raises(LedgerError, match=re.escape(message)):
            ledger.append_entry(kind, content)
    naive = datetime(2026, 11, 3, 12)
    with pytest.raises(LedgerError, match="timezone-aware"):
        ledger.append_entry("exploration", payload, now=naive)
    assert ledger_bytes() == before


# -- payload rules ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcomes", [["validation"], ["sealed"], ["training", "validation"], ["sealed", "training"]]
)
def test_an_exploration_never_reads_validation_or_sealed_outcomes(outcomes):
    with pytest.raises(LedgerRefused, match="outcomes is a test, never exploration"):
        ledger.record_exploration(**lf.exploration(outcomes=outcomes))
    assert not ledger.ledger_path().exists()


def test_exploration_entries():
    with pytest.raises(LedgerRefused, match="never reads the sealed segment"):
        ledger.record_exploration(**lf.exploration(inputs=["training", "sealed"]))
    entry = ledger.record_exploration(
        **lf.exploration(
            activity="tokenizer-diag",
            subject="tokenizer-validation",
            stage="done",
            recipe_id=None,
            inputs=("validation", "training", "validation"),
            outcomes=[],
        )
    )
    assert entry.payload["inputs"] == ["training", "validation"]  # sorted and distinct
    assert entry.payload["outcomes"] == [] and entry.payload["recipe_id"] is None
    for changes, message in [
        ({"activity": "peek"}, "activity must be one of"),
        ({"subject": "q-2024-07/fold"}, "subject must match"),
        ({"stage": "started"}, "stage must be one of"),
        ({"recipe_id": "0ee0d123"}, "recipe_id must be a sha256 hex or null"),
        ({"inputs": []}, "inputs must name the segments"),
        ({"inputs": ["test"]}, "sorted list of distinct segments"),
        ({"inputs": "training"}, "inputs must be a list of names"),
        ({"details": ["a list"]}, "details must be a JSON object"),
        ({"details": {"big": "x" * ledger.MAX_DETAILS_BYTES}}, "more than 16384"),
    ]:
        with pytest.raises(LedgerError, match=re.escape(message)):
            ledger.record_exploration(**lf.exploration(**changes))
    assert len(ledger.read_ledger()) == 1


def test_fit_entries():
    approval = ledger.record_gate3_approval(**lf.approval())
    failed = ledger.record_fit(
        **lf.fit(
            checkpoint_id=None,
            registry_line_sha256=None,
            status="failed",
            run_id=None,
            fit_record_sha256=None,
            blinded=True,
            replicate=2,
            fold="wf-2025-06",
            approval_seq=approval.seq,
        )
    )
    assert failed.payload["checkpoint_id"] is None and failed.payload["status"] == "failed"
    retried = ledger.record_fit(**lf.fit(attempt=2, approval_seq=approval.seq))
    assert retried.payload["attempt"] == 2
    for changes, message in [
        ({"status": "failed"}, "checkpoint_id is set exactly when the status is registered"),
        ({"checkpoint_id": None}, "checkpoint_id is set exactly when the status is registered"),
        ({"registry_line_sha256": None}, "registry_line_sha256 is set exactly when"),
        ({"status": "completed"}, "status must be one of"),
        ({"checkpoint_id": "KS-btc-x"}, "checkpoint_id must match"),
        ({"checkpoint_id": "ks-" + "a" * 62}, "in at most 64 characters"),
        ({"fold": "synthetic-timing"}, "not synthetic"),
        ({"fold": "wf-2025-06-r1"}, "without its replicate suffix"),
        ({"replicate": -1}, "replicate must be an integer >= 0"),
        ({"run_id": "a/b"}, "run_id must match"),
        ({"attempt": 3}, "attempt must be one of [1, 2]"),
        ({"attempt": True}, "attempt must be one of [1, 2]"),
        ({"blinded": "no"}, "blinded must be a boolean"),
        ({"device": "cuda"}, "device must be one of ['cpu', 'mps']"),
        ({"threads": 0}, "threads must be a positive integer"),
        ({"wall_seconds": -1.0}, "wall_seconds must be a finite number >= 0"),
        ({"cpu_seconds": float("nan")}, "non-finite number at payload.cpu_seconds"),
        ({"recipe_id": "x" * 64}, "recipe_id must be a sha256 hex"),
        ({"order_id": "order-1"}, "order_id must match"),
    ]:
        with pytest.raises(LedgerError, match=re.escape(message)):
            ledger.record_fit(**lf.fit(**changes))
    with pytest.raises(LedgerRefused, match="approval_seq 2 names no earlier gate3_approval"):
        ledger.record_fit(**lf.fit(approval_seq=2))  # entry 2 is a fit
    with pytest.raises(OrderNotFound, match="no owner_order order-0001"):
        ledger.record_fit(**lf.fit(order_id="order-0001"))
    assert len(ledger.read_ledger()) == 3


def test_gate3_approvals_stay_within_the_tolerance():
    at_tolerance = ledger.record_gate3_approval(**lf.approval(estimate=12.5, approved=10))
    assert at_tolerance.payload["ratio"] == 1.25 and at_tolerance.payload["approved"] == 10.0
    below = ledger.record_gate3_approval(
        **lf.approval(
            command="finetune pilot-check",
            subject="pilot-check-0ee0d123",
            unit="cpu_hours",
            approved=1.5,
            estimate=0.89,
        )
    )
    assert below.payload["above_gate3"] is False
    before = ledger_bytes()
    with pytest.raises(
        LedgerRefused,
        match=r"is 1\.260 times the 10 the owner approved, above the tolerance of 1\.25",
    ) as info:
        ledger.record_gate3_approval(**lf.approval(estimate=12.6, approved=10))
    assert "D-040, D-046" in str(info.value)
    for changes, message in [
        ({"approved": 0}, "approved must be a finite number > 0"),
        ({"approved": True}, "approved must be a finite number"),
        ({"estimate": float("inf")}, "estimate must be a finite number"),
        ({"estimate": -1.0}, "estimate must be a finite number >= 0"),
        ({"command": "finetune walk-forward"}, "command must be one of"),
        ({"unit": "gpu_hours"}, "unit must be one of"),
        ({"subject": ""}, "subject must match"),
        ({"given_at": datetime(2026, 11, 3, 12)}, "timezone-aware"),
        ({"estimate_detail": {"x": "y" * ledger.MAX_DETAILS_BYTES}}, "more than 16384"),
    ]:
        with pytest.raises(LedgerError, match=re.escape(message)):
            ledger.record_gate3_approval(**lf.approval(**changes))
    payload = {**json.loads(at_tolerance.line)["payload"], "ratio": 1.2}
    with pytest.raises(LedgerError, match="ratio must be estimate / approved"):
        ledger.append_entry("gate3_approval", payload)
    payload = {**json.loads(at_tolerance.line)["payload"], "above_gate3": False}
    with pytest.raises(LedgerError, match="above_gate3 must be estimate > gate3_hours"):
        ledger.append_entry("gate3_approval", payload)
    assert ledger_bytes() == before


def test_what_may_not_be_published_is_refused(monkeypatch):
    lf.fixed_code(monkeypatch)  # git would read the synthetic home below
    monkeypatch.setattr(getpass, "getuser", lambda: "zq-synthetic-user")
    home = "/" + "srv/zq-synthetic-home"  # assembled: no path literal for the hygiene scan
    monkeypatch.setenv("HOME", home)
    peer = ".".join(str(octet) for octet in (192, 0, 2, 17))  # no IPv4 literal in the sources
    for details, message in [
        ({"owner": "zq-synthetic-user"}, "personal data (username) at payload.details.owner"),
        ({"cache": home + "-cache"}, "personal data (home directory) at payload.details.cache"),
        ({"peer": peer}, "IPv4 literal at payload.details.peer"),
        ({"log": "/opt/kronos/child.log"}, "an absolute path at payload.details.log"),
        ({"log": "see ~/child.log"}, "an absolute path at payload.details.log"),
        ({"/opt/kronos": 1}, "an absolute path at payload.details."),
        ({"host": PUBLIC_HOST}, "a BingX host name at payload.details.host"),
        ({"host": f"https://{PUBLIC_HOST.upper()}/x"}, "a BingX host name at payload.details.host"),
    ]:
        with pytest.raises(LedgerRefused, match=re.escape(message)):
            ledger.record_exploration(**lf.exploration(details=details))
    assert not ledger.ledger_path().exists()
    relative = {
        "report": "reports/finetune/pilot-check-0ee0d123.json",
        "share": "1/3",
        "spacing": "a / b",
        "source": "BingX real market",
    }
    entry = ledger.record_exploration(**lf.exploration(details=relative))
    assert entry.payload["details"] == relative


# what the repository's hygiene scan of reports/ refuses (assembled: no literal for that scan)
EMAIL = "owner" + "@" + "example.org"
CLOUD_HOST = "s3." + "amazon" + "aws.com"
HOME_AFTER_A_WORD = "x" + "/home" + "/" + "someone"
USERS_AFTER_A_WORD = "notes" + "/Users" + "/" + "someone"
CREDENTIAL = "BINGX_API_" + "KEY=" + "abc"


def test_what_the_hygiene_scan_of_reports_refuses_is_refused_at_append_and_on_read(monkeypatch):
    """The ledger is a file under reports/: a line the hygiene scan flags (an email address, a
    personal path prefix after a word character, the cloud host fragment, a credential
    assignment) would fail ``tests/test_repo_hygiene.py`` on every later run, because the ledger
    is never repaired. It is refused at append, and a ledger file holding one does not verify;
    the check is machine-independent, so it runs on read too."""
    lf.fixed_code(monkeypatch)
    for text, problem in [
        (f"approved by the owner, {EMAIL}", "an email address"),
        (f"mirror on {CLOUD_HOST}", "a cloud host name"),
        (f"mirror on {CLOUD_HOST.upper()}", "a cloud host name"),
        (f"notes in {HOME_AFTER_A_WORD}", "a personal path"),
        (f"notes in {USERS_AFTER_A_WORD}", "a personal path"),
        (f"set {CREDENTIAL}", "a credential assignment"),
    ]:
        with pytest.raises(LedgerRefused, match=f"^{problem} at payload.details.note: ") as info:
            ledger.record_exploration(**lf.exploration(details={"note": text}))
        assert "tests/test_repo_hygiene.py" in str(info.value)
    with pytest.raises(LedgerRefused, match="an email address at payload.details.owner" + "@"):
        ledger.record_exploration(**lf.exploration(details={"owner" + "@" + "example.org": 1}))
    across = {"a": "BINGX_API_" + "KEY =", "b": "x"}  # flagged only on the whole line
    with pytest.raises(LedgerRefused, match="a credential assignment across the strings"):
        ledger.record_exploration(**lf.exploration(details=across))
    assert not ledger.ledger_path().exists()
    first = ledger.record_exploration(**lf.exploration(), now=lf.MONDAY)
    seeded = json.loads(first.line)
    seeded["payload"]["details"] = {"note": f"approved by the owner, {EMAIL}"}
    ledger.ledger_path().write_bytes(chained(json.loads(first.line), seeded))
    with pytest.raises(LedgerChainError, match="ledger line 2: an email address at payload"):
        ledger.read_ledger()
    assert ledger.ledger_status()["entries"] == 1


def test_values_are_plain_json_with_printable_ascii_strings_and_finite_numbers():
    for details, message in [
        ({"note": NON_ASCII}, "the string at payload.details.note is not printable ASCII"),
        ({"note": "two\nlines"}, "the string at payload.details.note is not printable ASCII"),
        ({"x": float("nan")}, "non-finite number at payload.details.x"),
        ({"x": [1.0, float("-inf")]}, "non-finite number at payload.details.x[1]"),
        ({"when": datetime(2026, 1, 1, tzinfo=UTC)}, "a datetime at payload.details.when"),
        ({1.5: "x"}, "a float key under payload.details"),
    ]:
        with pytest.raises(LedgerError, match=re.escape(message)):
            ledger.record_exploration(**lf.exploration(details=details))
    entry = ledger.record_exploration(
        **lf.exploration(
            details={
                "n": np.int64(3),
                "x": np.float64(0.25),
                "v": np.array([1, 2]),
                "t": (1, 2),
                7: "seven",
            }
        )
    )
    assert dict(entry.payload["details"]) == {
        "7": "seven",
        "n": 3,
        "t": [1, 2],
        "v": [1, 2],
        "x": 0.25,
    }


# -- times, the test budget and the order of tests hold on read too --------------------------------


def chained(*documents: dict) -> bytes:
    """Lines of ``documents`` chained as the ledger chains them (no rule checked): a ledger file
    that a writer bypassing ``append_entry`` could leave."""
    out: list[bytes] = []
    for seq, document in enumerate(documents, 1):
        document = {**document, "seq": seq}
        document["prev_sha256"] = sha256_bytes(out[-1]) if out else ledger.GENESIS_SHA256
        out.append(canonical_json(document))
    return lines(*out)


def documents() -> list[dict]:
    return [json.loads(entry.line) for entry in ledger.read_ledger()]


def test_times_never_decrease(monkeypatch):
    lf.fixed_code(monkeypatch)
    first = ledger.record_exploration(**lf.exploration(), now=lf.MONDAY)
    same = ledger.record_exploration(**lf.exploration(stage="done"), now=lf.MONDAY)  # equal: kept
    before = ledger_bytes()
    with pytest.raises(LedgerRefused, match="the ledger's times never decrease") as info:
        ledger.record_gate3_approval(**lf.approval(), now=lf.MONDAY - timedelta(seconds=1))
    assert "earlier than 2026-11-02T12:00:00Z, the ts of the previous entry #2" in str(info.value)
    assert ledger_bytes() == before
    backdated = {**json.loads(same.line), "ts": "2026-11-02T11:59:59Z"}
    ledger.ledger_path().write_bytes(chained(json.loads(first.line), backdated))
    with pytest.raises(LedgerChainError, match="ledger line 2: ts 2026-11-02T11:59:59Z is earlier"):
        ledger.read_ledger()
    assert ledger.ledger_status()["entries"] == 1


def a_test_payload(recipe: str, test: str, moment: datetime, week: int, quarter: int) -> dict:
    """A ``test`` payload with the counts the caller gives (the ledger checks them)."""
    bundle = ledger.preregistration_for(recipe)
    return {
        "test_id": ledger.id_for_test(test, recipe),
        "test": test,
        "recipe_id": recipe,
        "bundle_relpath": bundle.payload["bundle_relpath"],
        "bundle_sha256": bundle.payload["bundle_sha256"],
        "preregistration_seq": bundle.seq,
        "iso_week": ledger.iso_week_key(moment),
        "quarter": ledger.quarter_key(moment),
        "week_count": week,
        "quarter_count": quarter,
        "order_id": None,
    }


def test_a_test_line_past_the_budget_is_refused_at_append_and_on_read(monkeypatch, capsys, shown):
    """SPEC rule 8 as the test lines record it: a 4th test of an ISO week, appended past
    ``budget.spend_test`` through the public ``append_entry``, is refused; a ledger file holding
    one (written by anything else) no longer verifies."""
    lf.fixed_code(monkeypatch)
    recipes = [lf.recipe_id(f"budget-{index}") for index in range(4)]
    for recipe in recipes:
        lf.register_bundle(recipe, ("T-A",), now=lf.MONDAY)
    for day, recipe in enumerate(recipes[:3]):
        budget.spend_test("T-A", recipe_id=recipe, now=lf.MONDAY + timedelta(days=day))
    moment = lf.MONDAY + timedelta(days=3)
    fourth = a_test_payload(recipes[3], "T-A", moment, 4, 4)
    before = ledger_bytes()
    with pytest.raises(LedgerRefused, match=re.escape("would be test 4 of ISO week 2026-W45")):
        ledger.append_entry("test", fourth, now=moment)
    assert ledger_bytes() == before
    seeded = {**documents()[-1], "ts": "2026-11-05T12:00:00Z", "payload": fourth}
    ledger.ledger_path().write_bytes(chained(*documents(), seeded))
    message = "ledger line 8: T-A-" + recipes[3][:8] + " would be test 4 of ISO week 2026-W45"
    with pytest.raises(LedgerChainError, match=re.escape(message)):
        ledger.read_ledger()
    assert ledger.ledger_status()["chain_ok"] is False
    code, output = run_ledger(capsys, "verify")
    assert code == 1 and "at most 3 in an ISO week and 15 in a quarter (SPEC rule 8)" in output


def test_a_sixteenth_test_of_a_quarter_is_refused_on_append(monkeypatch):
    lf.fixed_code(monkeypatch)
    recipes = [lf.recipe_id(f"quarter-{index}") for index in range(16)]
    start = datetime(2026, 10, 5, 12, tzinfo=UTC)  # Monday of 2026-W41, 2026-Q4
    for recipe in recipes:
        lf.register_bundle(recipe, ("T-A",), now=start)
    for index, recipe in enumerate(recipes[:15]):
        moment = start + timedelta(weeks=index // 3, days=index % 3)
        budget.spend_test("T-A", recipe_id=recipe, now=moment)
    moment = start + timedelta(weeks=5)  # 2026-W46: a week with room, the quarter without
    sixteenth = a_test_payload(recipes[15], "T-A", moment, 1, 16)
    with pytest.raises(LedgerRefused, match=re.escape("and 16 of 2026-Q4: at most 3 in an ISO")):
        ledger.append_entry("test", sixteenth, now=moment)


def test_t_b_without_an_earlier_week_t_a_is_refused_at_append_and_on_read(monkeypatch):
    lf.fixed_code(monkeypatch)
    lf.register_bundle(lf.R1, ("T-A", "T-B"), now=lf.MONDAY)
    early = a_test_payload(lf.R1, "T-B", lf.MONDAY, 1, 1)
    with pytest.raises(LedgerRefused, match="the bundle registers T-A and T-B, and T-A is read"):
        ledger.append_entry("test", early, now=lf.MONDAY)
    budget.spend_test("T-A", recipe_id=lf.R1, now=lf.MONDAY)
    sunday = lf.MONDAY + timedelta(days=6)  # the same ISO week, 2026-W45
    same_week = a_test_payload(lf.R1, "T-B", sunday, 2, 2)
    with pytest.raises(LedgerRefused, match="T-A was read in ISO week 2026-W45 .ledger entry #2"):
        ledger.append_entry("test", same_week, now=sunday)
    seeded = {**documents()[-1], "ts": "2026-11-08T12:00:00Z", "payload": same_week}
    ledger.ledger_path().write_bytes(chained(*documents(), seeded))
    with pytest.raises(LedgerChainError, match="ledger line 3: T-B-.* T-A was read in ISO week"):
        ledger.read_ledger()


def test_a_backdated_reading_cannot_escape_a_full_week(monkeypatch):
    """The reviewer's case: 2026-W46 holds 3 tests, and a 4th dated into 2026-W45 (before the
    ledger's last entry) is refused; the ledger is unchanged."""
    lf.fixed_code(monkeypatch)
    recipes = [lf.recipe_id(f"backdated-{index}") for index in range(4)]
    monday = lf.MONDAY + lf.WEEK  # 2026-11-09, 2026-W46
    for recipe in recipes:
        lf.register_bundle(recipe, ("T-A",), now=monday)
    for day, recipe in enumerate(recipes[:3]):
        budget.spend_test("T-A", recipe_id=recipe, now=monday + timedelta(days=day))
    with pytest.raises(budget.BudgetExceeded):
        budget.spend_test("T-A", recipe_id=recipes[3], now=monday + timedelta(days=3))
    before = ledger_bytes()
    with pytest.raises(LedgerRefused, match="the ledger's times never decrease"):
        budget.spend_test("T-A", recipe_id=recipes[3], now=monday - timedelta(days=4))
    assert ledger_bytes() == before
    assert budget.budget_status(at=monday, entries=ledger.read_ledger()).week_used == 3


# -- a ledger verifies the same on every machine ---------------------------------------------------


def test_a_ledger_verifies_whatever_the_readers_login_host_and_home(monkeypatch):
    """The writer's personal data is checked when a line is appended; a reader's login, host
    name or home directory found in a line (``pi`` in ``pilot-fold``, ``dev`` in ``device``,
    ``mps`` in a fit's device, ``/prereg`` in a bundle path) never breaks the chain."""
    lf.fixed_code(monkeypatch)  # git would read the synthetic homes below
    order = lf.recorded_order(lf.R1)
    ledger.record_exploration(**lf.exploration(), now=order.moment)
    ledger.record_fit(**lf.fit(), now=order.moment)
    written = ledger_bytes()
    readers = [("pi", "kronos"), ("dev", "mps"), ("prereg", "fit")]
    for login, host in readers:
        monkeypatch.setattr(getpass, "getuser", lambda login=login: login)
        monkeypatch.setattr(socket, "gethostname", lambda host=host: host)
        monkeypatch.setenv("HOME", "/" + "prereg")  # assembled: no path literal for hygiene
        assert len(ledger.read_ledger()) == 6 and ledger.ledger_status()["chain_ok"] is True
        assert ledger.find_order("order-0001").line == order.line
        assert partition.sealed_access("order-0001").order_id == "order-0001"
        assert budget.budget_status(at=order.moment).week_used == 1
    monkeypatch.setattr(getpass, "getuser", lambda: "pi")  # the writer's own login: refused
    monkeypatch.setattr(socket, "gethostname", lambda: "zq-synthetic-host")
    monkeypatch.setenv("HOME", "/" + "srv/zq-synthetic-home")
    with pytest.raises(LedgerRefused, match="personal data .username. at payload.activity"):
        ledger.record_exploration(**lf.exploration(), now=order.moment)
    assert ledger_bytes() == written


def test_reading_the_ledger_and_opening_the_guard_never_run_the_personal_data_check():
    """Reading a ledger with entries (and verifying an order on it) imports neither the
    personal-data check, which depends on the machine, nor anything heavy (only the exchange
    module that holds the BingX host literal, for the host check)."""
    lf.recorded_order(lf.recipe_id("light"))
    code = (
        "import json, sys\n"
        "from kronos_agent import ledger\n"
        "from kronos_agent.data import partition\n"
        "entries = ledger.read_ledger()\n"
        "access = partition.sealed_access('order-0001')\n"
        "heavy = ('torch', 'numpy', 'pandas', 'yaml', 'huggingface_hub', 'kronos_agent.envinfo',\n"
        "         'kronos_agent.params')\n"
        "loaded = sorted(m for m in heavy if m in sys.modules)\n"
        "print('RESULT ' + json.dumps([len(entries), access.order_id, loaded]))\n"
    )
    (count, order_id, loaded), _ = fresh_interpreter(code)
    assert (count, order_id, loaded) == (4, "order-0001", [])


# -- which ledger may authorize which data ---------------------------------------------------------


@pytest.fixture
def repository(tmp_path, monkeypatch) -> Path:
    """A stand-in repository: ``reports/ledger`` and ``data/candles/binance`` under ``tmp_path /
    "repo"``, which ``ledger`` takes for the repository's (never the real one's)."""
    repo = tmp_path / "repo"
    (repo / "reports" / "ledger").mkdir(parents=True)
    (repo / "data" / "candles" / "binance").mkdir(parents=True)
    (repo / "data" / "candles" / "binance" / "1h.csv").write_text("open_ms\n")
    monkeypatch.setattr(ledger, "REPO_ROOT", repo)
    return repo


def test_only_the_repositorys_ledger_authorizes_the_repositorys_data(tmp_path, repository):
    data = repository / "data"
    own = repository / "reports" / "ledger"
    scratch = tmp_path / "scratch" / "reports" / "ledger"
    assert ledger.binding_problem(root=own, data_root=data) is None
    assert ledger.binding_problem(root=own, data_root=tmp_path / "elsewhere") is None
    assert ledger.binding_problem(root=scratch, data_root=tmp_path / "elsewhere") is None
    for target in (data, data / "candles", tmp_path / "link-to-data"):
        if target.name == "link-to-data":
            target.symlink_to(data, target_is_directory=True)
        problem = ledger.binding_problem(root=scratch, data_root=target)
        assert problem is not None and "is the repository's data/" in problem, target
    with pytest.raises(LedgerRefused, match="KRONOS_AGENT_REPORTS_DIR"):
        ledger.require_binding(root=scratch, data_root=data)
    linked_reports = tmp_path / "linked" / "reports"  # a reports root linking to the ledger
    linked_reports.mkdir(parents=True)
    (linked_reports / "ledger").symlink_to(own, target_is_directory=True)
    assert ledger.binding_problem(root=linked_reports / "ledger", data_root=data) is None
    if (repository / "DATA").exists():  # a case-insensitive volume: another spelling, one place
        assert ledger.binding_problem(root=scratch, data_root=repository / "DATA") is not None


def test_a_data_root_linking_into_the_repositorys_data_is_refused(tmp_path, repository):
    data = repository / "data"
    scratch = tmp_path / "scratch" / "ledger"
    view = tmp_path / "view"  # tests/root_guard.py's views: top-level links
    view.mkdir()
    (view / "candles").symlink_to(data / "candles", target_is_directory=True)
    deep = tmp_path / "deep" / "candles" / "binance"  # a link at the stores' depth
    deep.mkdir(parents=True)
    (deep / "1h.csv").symlink_to(data / "candles" / "binance" / "1h.csv")
    deeper = tmp_path / "deeper" / "a" / "b" / "c"  # past DATA_LINK_DEPTH: not looked for
    deeper.mkdir(parents=True)
    (deeper / "1h.csv").symlink_to(data / "candles" / "binance" / "1h.csv")
    for root in (view, tmp_path / "deep"):
        problem = ledger.binding_problem(root=scratch, data_root=root)
        assert problem is not None and "links into the repository's data/ (a view" in problem
    assert ledger.DATA_LINK_DEPTH == 3
    assert ledger.binding_problem(root=scratch, data_root=tmp_path / "deeper") is None
    shutil.rmtree(data)  # a repository without data/: nothing to protect
    assert ledger.binding_problem(root=scratch, data_root=view) is None


def test_spend_test_refuses_a_ledger_that_may_not_read_the_data(tmp_path, monkeypatch):
    """The test's ledger is relocated: it spends tests on its own data root, never on the
    repository's (``finetune report`` reads the fans of the data root after the entry)."""
    lf.fixed_code(monkeypatch)
    lf.register_bundle(lf.R1, ("T-A",), now=lf.MONDAY)
    real_data = paths.REPO_ROOT / "data"
    if real_data.is_dir():
        with pytest.raises(LedgerRefused, match="not the repository's reports/ledger"):
            budget.spend_test("T-A", recipe_id=lf.R1, data_root=real_data, now=lf.MONDAY)
    stand_in = tmp_path / "repo"
    (stand_in / "data").mkdir(parents=True)
    monkeypatch.setattr(ledger, "REPO_ROOT", stand_in)
    before = ledger_bytes()
    with pytest.raises(LedgerRefused, match="the data root is the repository's data/"):
        budget.spend_test("T-A", recipe_id=lf.R1, data_root=stand_in / "data", now=lf.MONDAY)
    assert ledger_bytes() == before
    entry = budget.spend_test("T-A", recipe_id=lf.R1, now=lf.MONDAY)  # its own data root
    assert entry.payload["test_id"] == ledger.id_for_test("T-A", lf.R1)


# -- pre-registration bundles ----------------------------------------------------------------------


def test_a_preregistration_records_its_bundle_once():
    path = lf.write_bundle(lf.R1)
    entry = ledger.record_preregistration(
        bundle_relpath=lf.bundle_relpath(lf.R1),
        recipe_id=lf.R1,
        tests=["T-B", "T-A", "T-A"],
        forecasting_code_sha256=lf.FORECASTING_CODE,
        training_code_sha256=lf.TRAINING_CODE,
    )
    assert entry.payload["tests"] == ["T-A", "T-B"]
    assert entry.payload["bundle_sha256"] == sha256_file(path)
    assert ledger.preregistration_for(lf.R1).line == entry.line
    assert [item.line for item in ledger.verify_bundles()] == [entry.line]
    with pytest.raises(LedgerRefused, match="has its pre-registration bundle already"):
        lf.register_bundle(lf.R1)
    other = lf.recipe_id("other")
    with pytest.raises(LedgerError, match=f"the bundle of recipe {other[:8]} is"):
        ledger.record_preregistration(
            bundle_relpath=lf.bundle_relpath(lf.R1),
            recipe_id=other,
            tests=["T-A"],
            forecasting_code_sha256=lf.FORECASTING_CODE,
            training_code_sha256=lf.TRAINING_CODE,
        )
    with pytest.raises(LedgerError, match="does not exist under the reports root"):
        ledger.record_preregistration(
            bundle_relpath=lf.bundle_relpath(other),
            recipe_id=other,
            tests=["T-A"],
            forecasting_code_sha256=lf.FORECASTING_CODE,
            training_code_sha256=lf.TRAINING_CODE,
        )
    for tests, message in [([], "tests must be a sorted non-empty subset"), (["T-D"], "subset")]:
        with pytest.raises(LedgerError, match=message):
            lf.register_bundle(other, tests=tests)
    with pytest.raises(LedgerError, match="forecasting_code_sha256 must be a sha256 hex"):
        ledger.record_preregistration(
            bundle_relpath=lf.bundle_relpath(other),
            recipe_id=other,
            tests=["T-A"],
            forecasting_code_sha256="0ee0d123",
            training_code_sha256=lf.TRAINING_CODE,
        )
    assert len(ledger.read_ledger()) == 1


def test_a_changed_or_missing_bundle_is_refused(tmp_path):
    entry = lf.register_bundle(lf.R1)
    path = paths.reports_dir() / lf.bundle_relpath(lf.R1)
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(BundleChanged, match="changed after ledger entry #1 recorded it"):
        ledger.verify_bundles()
    with pytest.raises(BundleChanged, match="changed after"):
        ledger.verify_bundle(entry)
    path.write_bytes(original)
    assert [item.line for item in ledger.verify_bundles()] == [entry.line]
    path.unlink()
    with pytest.raises(BundleChanged, match="is missing"):
        ledger.verify_bundles()
    other_reports = tmp_path / "reports"  # the same bundle under another reports root
    lf.write_bundle(lf.R1, reports_root=other_reports)
    assert ledger.verify_bundles(reports_root=other_reports)[0].line == entry.line
    assert issubclass(BundleChanged, LedgerRefused)


# -- the owner's orders ----------------------------------------------------------------------------


def test_record_order_needs_a_t_b_reading_and_is_recorded_once_per_candidate():
    lf.register_bundle(lf.R1, now=lf.MONDAY)
    with pytest.raises(LedgerRefused, match=f"no T-B test entry for the candidate {T_B}"):
        ledger.record_order(scope="open_sealed", candidate=T_B, text=lf.ORDER_TEXT)
    budget.spend_test("T-A", recipe_id=lf.R1, now=lf.MONDAY)
    t_a = ledger.id_for_test("T-A", lf.R1)
    with pytest.raises(LedgerRefused, match=f"no T-B test entry for the candidate {t_a}"):
        ledger.record_order(scope="open_sealed", candidate=t_a, text=lf.ORDER_TEXT)
    budget.spend_test("T-B", recipe_id=lf.R1, now=lf.MONDAY + lf.WEEK)
    first = ledger.record_order(
        scope="open_sealed", candidate=T_B, text=lf.ORDER_TEXT, now=lf.MONDAY + lf.WEEK
    )
    assert first.payload == {
        "order_id": "order-0001",
        "scope": "open_sealed",
        "candidate": T_B,
        "text": lf.ORDER_TEXT,
    }
    with pytest.raises(LedgerRefused, match="has an order already .order-0001"):
        ledger.record_order(scope="open_sealed", candidate=T_B, text="again")
    second = lf.recorded_order(lf.recipe_id("second"), start=lf.MONDAY + 2 * lf.WEEK)
    assert second.payload["order_id"] == "order-0002"
    assert ledger.find_order("order-0001").line == first.line
    assert ledger.find_order("order-0002", scope="open_sealed").line == second.line
    assert ledger.verify_open_order("order-0001").line == first.line
    for order_id, scope, message in [
        ("order-0001", "close_sealed", "order order-0001 has scope open_sealed, not"),
        ("order-0003", "open_sealed", "no owner_order order-0003 of scope 'open_sealed'"),
        ("order-1", "open_sealed", "'order-1' is not an order id"),
        (None, "open_sealed", "None is not an order id"),
    ]:
        with pytest.raises(OrderNotFound, match=re.escape(message)):
            ledger.find_order(order_id, scope=scope)
    with pytest.raises(OrderNotFound, match="unknown order scope"):
        ledger.verify_open_order("order-0001", scope="close_sealed")
    assert ledger.order_used("order-0001") is False
    budget.spend_test("T-C", recipe_id=lf.R1, order_id="order-0001", now=lf.MONDAY + 3 * lf.WEEK)
    assert ledger.order_used("order-0001") is True and ledger.order_used("order-0002") is False
    # the order stays valid after its T-C reading: it also covers the sealed folds (D-048 (3))
    assert ledger.verify_open_order("order-0001").line == first.line


def test_record_order_arguments_are_checked_before_the_ledger_is_read():
    for changes, message in [
        ({"scope": "open"}, "scope must be one of"),
        ({"candidate": "T-B-0EE0D123"}, "candidate must be a test id"),
        ({"candidate": "T-D-0ee0d123"}, "candidate must be a test id"),
        ({"text": ""}, "the order text is 1 to 1000 printable ASCII characters"),
        ({"text": "x" * 1001}, "the order text is 1 to 1000"),
        ({"text": "two\nlines"}, "the order text is 1 to 1000"),
    ]:
        arguments = {"scope": "open_sealed", "candidate": T_B, "text": lf.ORDER_TEXT, **changes}
        with pytest.raises(LedgerError, match=re.escape(message)):
            ledger.record_order(**arguments)
    assert not paths.ledger_dir().exists()


def test_test_ids():
    assert ledger.id_for_test("T-B", lf.R1) == f"T-B-{lf.R1[:8]}"
    assert ledger.TEST_ID_RE.fullmatch(ledger.id_for_test("T-C", lf.R1))
    with pytest.raises(LedgerError, match="test must be one of"):
        ledger.id_for_test("T-D", lf.R1)
    with pytest.raises(LedgerError, match="recipe_id must be a sha256 hex"):
        ledger.id_for_test("T-A", lf.R1[:8])


# -- module rules ----------------------------------------------------------------------------------


def test_the_copied_patterns_equal_their_sources():
    import test_repo_hygiene as hygiene

    from kronos_agent import envinfo
    from kronos_agent.finetune import folds, registry, runner, timing, train

    assert ledger.FOLD_LABEL_RE.pattern == folds.FOLD_LABEL_RE.pattern
    assert ledger.CHECKPOINT_ID_RE.pattern == registry.CHECKPOINT_ID_RE.pattern
    assert ledger.CHECKPOINT_ID_MAX_LENGTH == registry.CHECKPOINT_ID_MAX_LENGTH
    assert ledger._PATH_TOKEN.pattern == timing._PATH_TOKEN.pattern == ABSOLUTE_PATH.pattern
    assert ledger._IPV4_RE.pattern == envinfo.IPV4_RE.pattern == hygiene.IPV4_RE.pattern
    # the machine-independent rules of the hygiene scan of reports/, where the ledger lives
    assert ledger._EMAIL_RE.pattern == hygiene.EMAIL_RE.pattern
    assert ledger._PERSONAL_PATH_PREFIXES == hygiene.PERSONAL_PATH_PREFIXES
    assert ledger._CLOUD_HOST_FRAGMENT == hygiene.CLOUD_HOST_FRAGMENT
    assert ledger._CREDENTIAL_RE.pattern == hygiene.CREDENTIAL_RE.pattern
    assert "reports" not in hygiene.EXCLUDED_DIRS  # the scan reads the ledger's directory
    assert (budget.TESTS_PER_ISO_WEEK, budget.TESTS_PER_QUARTER) == (3, 15)
    assert budget.TESTS_PER_ISO_WEEK is ledger.TESTS_PER_ISO_WEEK
    # every status a training child can end with, besides the completed run that registers
    assert set(runner.STATUSES) - {"completed"} <= set(ledger.FIT_STATUSES)
    assert set(train.NO_RESULT_STATUSES) <= set(ledger.FIT_STATUSES)
    assert ledger.LEDGER_NAME == "experiments.jsonl" and paths.LEDGER_SUBDIR == "ledger"


def test_the_ledger_lock_is_ignored_by_git_and_the_ledger_is_not():
    """The first append creates ``reports/ledger/.lock``: a lock file, never published; the
    ledger beside it is tracked and published with the reports (SPEC section 19)."""
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert f"/reports/{paths.LEDGER_SUBDIR}/{ledger.LOCK_NAME}" in lines
    assert not any("experiments" in line or line.rstrip("/") == "/reports" for line in lines)


LEDGER_IMPORTS = {
    "kronos_agent.hashing",
    "kronos_agent.paths",
    "kronos_agent.data.fsutil",
    "kronos_agent.provenance",
}


def module_level_imports(name: str) -> set[str]:
    """Absolute names of the modules the top-level statements of module ``name`` import
    (``from . import x`` names the submodule ``x``)."""
    path = REPO_ROOT / "src" / Path(*name.split(".")).with_suffix(".py")
    package = name.split(".")[:-1]
    found: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
                continue
            base = ".".join(package[: len(package) - node.level + 1] + [node.module or ""])
            if node.module:
                found.add(base)
            else:
                found.update(f"{base.rstrip('.')}.{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize(
    "name, allowed",
    [("kronos_agent.ledger", LEDGER_IMPORTS), ("kronos_agent.budget", {"kronos_agent.ledger"})],
)
def test_ledger_and_budget_import_only_light_modules_at_load(name, allowed):
    imports = module_level_imports(name)
    stdlib = {item for item in imports if item.split(".")[0] in sys.stdlib_module_names}
    assert imports - stdlib == allowed
    assert "__future__" in imports


def test_the_import_rule_check_sees_every_form(tmp_path, monkeypatch):
    source = tmp_path / "src" / "kronos_agent" / "probe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import json\nfrom . import provenance\nfrom .data.fsutil import x\n"
        "def lazy():\n    from .params import params_version\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    assert module_level_imports("kronos_agent.probe") == {
        "json",
        "kronos_agent.provenance",
        "kronos_agent.data.fsutil",
    }


def fresh_interpreter(code: str) -> list:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = completed.stdout.splitlines()[-1]
    assert result.startswith("RESULT "), completed.stdout[-500:]
    return [json.loads(result.removeprefix("RESULT ")), completed.stdout]


def test_the_ledger_and_the_sealed_guard_load_nothing_heavy():
    code = (
        "import json, sys\n"
        "from kronos_agent import budget, ledger\n"
        "from kronos_agent.data import partition\n"
        "try:\n    partition.sealed_access('order-0001')\n"
        "except partition.SealedPartitionError:\n    pass\n"
        "heavy = ('torch', 'numpy', 'pandas', 'yaml', 'httpx', 'huggingface_hub')\n"
        "print('RESULT ' + json.dumps(sorted(m for m in heavy if m in sys.modules)))\n"
    )
    loaded, _ = fresh_interpreter(code)
    assert loaded == []


# -- the command -----------------------------------------------------------------------------------

LEDGER_HELP = (
    ["ledger", "--help"],
    ["ledger", "show", "--help"],
    ["ledger", "verify", "--help"],
    ["ledger", "record-order", "--help"],
)


def test_ledger_help_runs_in_a_fresh_interpreter_without_heavy_imports():
    code = (
        "import json, sys\n"
        "from kronos_agent import cli\n"
        f"codes = []\nfor argv in {[list(argv) for argv in LEDGER_HELP]!r}:\n"
        "    try:\n        cli.main(argv)\n    except SystemExit as exc:\n"
        "        codes.append(exc.code)\n"
        f"heavy = sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules)\n"
        "print('RESULT ' + json.dumps([codes, heavy, 'kronos_agent.ledger' in sys.modules]))\n"
    )
    (codes, heavy, ledger_loaded), output = fresh_interpreter(code)
    assert codes == [0] * len(LEDGER_HELP)
    assert heavy == [] and ledger_loaded is False
    text = " ".join(output.split())
    assert (
        "record an order only after the owner's explicit order in chat (gate 6); the text is "
        "their words translated into English" in text
    )
    for option in ("--kind", "--scope", "--candidate", "--text"):
        assert option in text


def test_the_literal_choices_agree_with_the_ledger():
    assert ledger_cli.KINDS == ledger.KINDS
    assert ledger_cli.ORDER_SCOPES == ledger.ORDER_SCOPES
    assert ledger_cli.TEST_ID_RE.pattern == ledger.TEST_ID_RE.pattern
    assert ledger_cli.MAX_ORDER_TEXT == ledger.MAX_ORDER_TEXT


@pytest.fixture
def shown(monkeypatch):
    """``display_path`` shows the isolated reports root as it shows the repository's
    ``reports/``, so that the printed lines can be compared and searched for absolute paths."""
    monkeypatch.setattr(cli, "REPO_ROOT", paths.reports_dir().parent)
    return "reports/ledger/experiments.jsonl"


def run_ledger(capsys, *argv: str) -> tuple[int, str]:
    code = cli.main(["ledger", *argv])
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert not ABSOLUTE_PATH.search(output), output
    return code, output


def usage_error(capsys, *argv: str) -> str:
    with pytest.raises(SystemExit) as info:
        cli.main(["ledger", *argv])
    assert info.value.code == 2
    error = capsys.readouterr().err
    assert "error:" in error and not ABSOLUTE_PATH.search(error)
    return error


def test_show_on_an_empty_and_on_a_filled_ledger(capsys, monkeypatch, shown):
    assert run_ledger(capsys, "show") == (0, f"ledger {shown}: no entries\n")
    start = lf.MONDAY
    exploration = ledger.record_exploration(**lf.exploration(), now=start)
    approval = ledger.record_gate3_approval(**lf.approval(), now=start)
    fit = ledger.record_fit(**lf.fit(approval_seq=approval.seq), now=start)
    failed = ledger.record_fit(
        **lf.fit(
            checkpoint_id=None,
            registry_line_sha256=None,
            status="memory_abort",
            fold="wf-2025-06",
            replicate=1,
            blinded=True,
        ),
        now=start,
    )
    order = lf.recorded_order(lf.R1, start=start)
    t_c = budget.spend_test(
        "T-C", recipe_id=lf.R1, order_id="order-0001", now=start + lf.WEEK + timedelta(days=1)
    )
    monkeypatch.setattr(ledger_cli, "_now", lambda: start + lf.WEEK + timedelta(days=2))
    code, output = run_ledger(capsys, "show")
    recipe8 = lf.R1[:8]
    bundle8 = ledger.preregistration_for(lf.R1).payload["bundle_sha256"][:12]
    expected = [
        "#1 2026-11-02T12:00:00Z exploration pilot-fold q-2024-07 planned",
        (
            "#2 2026-11-02T12:00:00Z gate3_approval finetune fold q-2024-07: approved 10 "
            "mps_hours, estimate 4.945 (ratio 0.495), given at 2026-11-02T12:00:00Z"
        ),
        (
            f"#3 2026-11-02T12:00:00Z fit ks-btc-{recipe8}-q-2024-07 registered q-2024-07 "
            "attempt 1 blinded false"
        ),
        "#4 2026-11-02T12:00:00Z fit - memory_abort wf-2025-06-r1 attempt 1 blinded true",
        (
            f"#5 2026-11-02T12:00:00Z preregistration finetune/prereg-{recipe8}.config.json "
            f"sha256 {bundle8} recipe {recipe8} tests T-A,T-B,T-C"
        ),
        f"#6 2026-11-02T12:00:00Z test T-A-{recipe8} week 2026-W45 (1/3) quarter 2026-Q4 (1/15)",
        f"#7 2026-11-09T12:00:00Z test T-B-{recipe8} week 2026-W46 (1/3) quarter 2026-Q4 (2/15)",
        (
            f"#8 2026-11-09T12:00:00Z owner_order order-0001 open_sealed candidate T-B-{recipe8} "
            f'(T-C read: yes): "{lf.ORDER_TEXT}"'
        ),
        f"#9 2026-11-10T12:00:00Z test T-C-{recipe8} week 2026-W46 (2/3) quarter 2026-Q4 (3/15)",
        (
            f"ledger {shown}: 9 entries, head {t_c.line_sha256}; tests in 2026-W46 2/3, in "
            "2026-Q4 3/15"
        ),
    ]
    assert code == 0 and output.splitlines() == expected
    assert (exploration.seq, fit.seq, failed.seq, order.seq) == (1, 3, 4, 8)
    code, output = run_ledger(capsys, "show", "--kind", "test")
    assert code == 0 and output.splitlines() == [expected[i] for i in (5, 6, 8, 9)]
    code, output = run_ledger(capsys, "show", "--kind", "owner_order")
    assert output.splitlines()[0] == expected[7]
    assert "max_epochs" not in run_ledger(capsys, "show")[1]  # details are not printed
    assert "invalid choice" in usage_error(capsys, "show", "--kind", "opinion")


def test_verify_exit_codes(capsys, shown):
    assert run_ledger(capsys, "verify") == (0, f"ledger {shown}: no entries, no bundle to verify\n")
    entry = lf.register_bundle(lf.R1)
    code, output = run_ledger(capsys, "verify")
    assert code == 0
    bundle8 = entry.payload["bundle_sha256"][:12]
    assert output.splitlines() == [
        f"ledger {shown}: 1 entries verified, chain intact, head {entry.line_sha256}",
        f"bundle {lf.bundle_relpath(lf.R1)} sha256 {bundle8} unchanged (ledger entry #1)",
        "bundles verified: 1",
    ]
    bundle = paths.reports_dir() / lf.bundle_relpath(lf.R1)
    original = bundle.read_bytes()
    bundle.write_bytes(b"{}")
    code, output = run_ledger(capsys, "verify")
    assert code == 2
    assert output.splitlines()[-1].startswith(
        f"REFUSED: the pre-registration bundle {lf.bundle_relpath(lf.R1)} changed after ledger "
        "entry #1 recorded it"
    )
    bundle.write_bytes(original)
    ledger.ledger_path().write_bytes(ledger_bytes()[:-1])  # a torn last line
    code, output = run_ledger(capsys, "verify")
    assert code == 1
    assert output.strip() == (
        "ERROR: LedgerChainError: ledger/experiments.jsonl: ledger line 1: no LF at the end of "
        "the file (a torn write); the ledger is never repaired"
    )
    assert run_ledger(capsys, "show")[0] == 1


def test_record_order_through_the_command(capsys, monkeypatch, shown):
    good = ["--scope", "open_sealed", "--candidate", T_B, "--text", lf.ORDER_TEXT]
    for argv, message in [
        (["--scope", "open_sealed", "--candidate", "T-D-0ee0d123", "--text", "x"], "test id"),
        (["--scope", "open_sealed", "--candidate", "T-B-0EE0D123", "--text", "x"], "test id"),
        (["--scope", "open_sealed", "--candidate", T_B], "--text"),
        (["--candidate", T_B, "--text", "x"], "--scope"),
        (["--scope", "close_sealed", "--candidate", T_B, "--text", "x"], "invalid choice"),
        (["--scope", "open_sealed", "--candidate", T_B, "--text", NON_ASCII], "printable"),
        (["--scope", "open_sealed", "--candidate", T_B, "--text", ""], "printable ASCII"),
        (["--scope", "open_sealed", "--candidate", T_B, "--text", "x" * 1001], "1 to 1,000"),
    ]:
        assert message in usage_error(capsys, "record-order", *argv)
    assert not paths.ledger_dir().exists()  # nothing ran
    code, output = run_ledger(capsys, "record-order", *good)
    assert code == 2
    assert output.startswith(f"REFUSED: no T-B test entry for the candidate {T_B}")
    lf.register_bundle(lf.R1, now=lf.MONDAY)
    budget.spend_test("T-A", recipe_id=lf.R1, now=lf.MONDAY)
    budget.spend_test("T-B", recipe_id=lf.R1, now=lf.MONDAY + lf.WEEK)
    monkeypatch.setattr(ledger_cli, "_now", lambda: lf.MONDAY + lf.WEEK - timedelta(seconds=1))
    code, output = run_ledger(capsys, "record-order", *good)  # the command's clock went back
    assert code == 2 and "the ledger's times never decrease" in output
    monkeypatch.setattr(ledger_cli, "_now", lambda: lf.MONDAY + lf.WEEK)
    code, output = run_ledger(capsys, "record-order", *good)
    entry = ledger.read_ledger()[-1]
    assert code == 0 and entry.kind == "owner_order" and entry.seq == 4
    assert output.splitlines() == [
        (
            f"recorded order-0001 (open_sealed, candidate {T_B}): ledger entry #4, line sha256 "
            f"{entry.line_sha256}"
        ),
        f"ledger {shown}: 4 entries",
    ]
    code, output = run_ledger(capsys, "record-order", *good)
    assert code == 2 and "has an order already (order-0001, ledger entry #4)" in output
    second = lf.recipe_id("second")
    lf.register_bundle(second, now=lf.MONDAY + lf.WEEK)
    budget.spend_test("T-A", recipe_id=second, now=lf.MONDAY + lf.WEEK)
    budget.spend_test("T-B", recipe_id=second, now=lf.MONDAY + 2 * lf.WEEK)
    monkeypatch.setattr(ledger_cli, "_now", lambda: lf.MONDAY + 2 * lf.WEEK)
    other = ["--scope", "open_sealed", "--candidate", ledger.id_for_test("T-B", second)]
    monkeypatch.setattr(getpass, "getuser", lambda: "zq-synthetic-user")
    code, output = run_ledger(capsys, "record-order", *other, "--text", "zq-synthetic-user said")
    assert code == 2 and "personal data (username) at payload.text" in output
    for text, problem in [  # the hygiene scan of reports/ would flag these forever
        (f"approved by the owner, {EMAIL}", "an email address"),
        (f"mirror on {CLOUD_HOST}", "a cloud host name"),
        (f"notes in {HOME_AFTER_A_WORD}", "a personal path"),
        (f"set {CREDENTIAL}", "a credential assignment"),
    ]:
        code, output = run_ledger(capsys, "record-order", *other, "--text", text)
        assert code == 2 and output.startswith(f"REFUSED: {problem} at payload.text: "), output
    with exclusive_lock(paths.ledger_dir() / ledger.LOCK_NAME):
        code, output = run_ledger(capsys, "record-order", *other, "--text", lf.ORDER_TEXT)
    assert code == 2 and output.startswith("REFUSED: ledger/.lock is locked by another run")
    before = ledger_bytes()
    ledger.ledger_path().write_bytes(before.replace(b'"seq":2,', b'"seq":7,'))
    code, output = run_ledger(capsys, "record-order", *other, "--text", lf.ORDER_TEXT)
    assert code == 1 and "ledger line 2: seq 7 is not the line number 2" in output
    ledger.ledger_path().write_bytes(before)
    code, output = run_ledger(capsys, "record-order", *other, "--text", lf.ORDER_TEXT)
    assert code == 0 and output.startswith("recorded order-0002")
    assert [item.payload["order_id"] for item in ledger.entries_of("owner_order")] == [
        "order-0001",
        "order-0002",
    ]
