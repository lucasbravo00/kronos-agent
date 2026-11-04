"""``kronos-agent finetune fold | pilot-check | tokenizer-diag`` through ``cli.main``: the wiring of
the commands to ``finetune.walkforward``, ``finetune.pilot_check``, ``finetune.tokenizer_diag`` and
the experiment ledger, their refusals and failures (D-039), the bundle check every one of them
runs first (design 7.1) and their output (no absolute path outside the test's own roots).

Every test runs on the test's own roots (``conftest.isolated_roots``) and on stand-ins: the fold
command's steps are ``fold_fakes`` (``walkforward.real_steps`` is replaced, D-055 (6)), and the
diagnostics' ``prepare_*``, ``estimate_*``, ``run_*`` and ``record_*_config`` are recording
fakes: these tests check the wiring, the studies have their own tests. Nothing trains, times,
forecasts or reads a real root.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import fold_fakes
import pytest

from kronos_agent import budget, cli, ledger, paths
from kronos_agent.cli import finetune as finetune_cli
from kronos_agent.finetune import pilot_check, tokenizer_diag
from kronos_agent.finetune import walkforward as wf
from kronos_agent.finetune.export_csv import fold_export_dir
from kronos_agent.finetune.recipe import R1, recipe8, recipe_id
from kronos_agent.finetune.registry import runs_dir
from kronos_agent.paths import data_dir, reports_dir, weights_dir

GIVEN_AT = datetime(2026, 11, 3, 9, 30, tzinfo=UTC)
PILOT_STEM = f"pilot-check-{recipe8(R1)}"
# an absolute path token, as the ledger and the timing record refuse one
ABSOLUTE = re.compile(r"(?<![\w.~/-])(?:~/|/)[^\s'\"`<>,;:()\[\]{}]+")


@pytest.fixture
def world(monkeypatch) -> fold_fakes.World:
    """The fold command on the stand-ins, with a fresh timing record in the test's reports root
    and the command's clock fixed."""
    fold_fakes.write_timing_record()
    world = fold_fakes.World()
    monkeypatch.setattr(wf, "real_steps", world.steps)
    monkeypatch.setattr(finetune_cli, "_now", lambda: GIVEN_AT)
    return world


def main(capsys, *argv: str) -> tuple[int, str]:
    code = cli.main(["finetune", *argv])
    out = capsys.readouterr().out
    assert_no_absolute_path(out)
    return code, out


def assert_no_absolute_path(text: str) -> None:
    """No absolute path but the test's own roots, which a real run shows repository-relative
    (``cli.display_path``)."""
    for root in (reports_dir(), data_dir(), weights_dir()):
        text = text.replace(str(root), "ROOT")
    assert ABSOLUTE.search(text) is None, ABSOLUTE.search(text)


def kinds() -> list[str]:
    return fold_fakes.ledger_kinds()


def train_pilot(capsys, world: fold_fakes.World, *, selected: int = 3) -> None:
    world.script.append(fold_fakes.Script(selected=selected))
    code, out = main(capsys, "fold", "--fold", "q-2024-07", "--approved-by-owner", "10")
    assert code == 0, out


def test_the_literal_choices_and_names_agree_with_the_libraries():
    assert finetune_cli.TOKENIZER_DIAG_PARTS == tokenizer_diag.PARTS
    assert finetune_cli.MISMATCH_FLAG in tokenizer_diag.FLAGS
    assert finetune_cli.PILOT_CHECK_COMMAND in ledger.APPROVAL_COMMANDS
    assert wf.COMMAND in ledger.APPROVAL_COMMANDS
    assert set(finetune_cli.SUBCOMMANDS) == {"timing", "fold", "pilot-check", "tokenizer-diag"}


# -- fold ------------------------------------------------------------------------------------------


def test_fold_estimate_prints_the_plan_and_writes_nothing(world, capsys):
    code, out = main(capsys, "fold", "--fold", "q-2024-07", "--estimate")
    assert code == 0
    assert out.startswith("finetune fold q-2024-07: quarterly layout (D-056), the pilot")
    assert "= 17801.8 s = 4.945 MPS-hours" in out and "= 35204.3 s = 9.779 MPS-hours" in out
    assert "gate 3: 4.945 MPS-hours is above 1 hour" in out
    assert out.rstrip().endswith("--estimate: nothing ran")
    assert not ledger.ledger_path().exists() and world.fits == [] and world.exports == []
    assert not fold_export_dir(recipe8(R1), "q-2024-07").exists()
    assert not runs_dir().exists()


def test_fold_is_refused_without_the_approval_and_runs_with_it(world, capsys):
    code, out = main(capsys, "fold", "--fold", "q-2024-07")
    assert code == 2 and "REFUSED: fold q-2024-07 needs about 4.945 MPS-hours" in out
    assert not ledger.ledger_path().exists() and world.fits == []
    world.script.append(fold_fakes.Script(selected=3))
    code, out = main(capsys, "fold", "--fold", "q-2024-07", "--approved-by-owner", "10")
    assert code == 0, out
    assert kinds() == ["gate3_approval", "exploration", "fit", "exploration"]
    stages = [entry.payload["stage"] for entry in ledger.entries_of("exploration")]
    assert stages == ["planned", "done"]
    approval = ledger.entries_of("gate3_approval")[0].payload
    assert approval["approved"] == 10.0 and approval["given_at"] == "2026-11-03T09:30:00Z"
    assert "gate 3: the owner approved 10 MPS-hours" in out and "pilot rule" in out
    assert out.rstrip().endswith("registered ks-btc-0ee0d123-q-2024-07")
    code, out = main(capsys, "fold", "--fold", "q-2024-07", "--approved-by-owner", "10")
    assert code == 0 and "nothing to run" in out and len(world.fits) == 1
    assert kinds() == ["gate3_approval", "exploration", "fit", "exploration"]


@pytest.mark.parametrize(
    "argv, code, prefix",
    [
        (["--fold", "wf-2024-07"], 2, "REFUSED: wf-2024-07: the walk-forward runs the quarterly"),
        (["--fold", "final-2026-08-01"], 2, "REFUSED: final-2026-08-01: the model to date"),
        (["--fold", "q-2024-10"], 2, "REFUSED: fold q-2024-10 trains after the pre-registration"),
        (["--fold", "q-2024-07", "--order", "order-0001"], 2, "REFUSED: --order belongs"),
        (["--fold", "q-2024-08"], 1, "ERROR: FoldError:"),
        (["--fold", "q-2024-07", "--replicate", "1"], 1, "ERROR: FoldError:"),
    ],
)
def test_fold_refusals_and_failures(world, capsys, argv, code, prefix):
    found, out = main(capsys, "fold", *argv, "--estimate")
    assert found == code and out.startswith(prefix), out
    assert world.fits == [] and not ledger.ledger_path().exists()


def test_a_fold_that_fails_twice_exits_1_after_its_entries(world, capsys):
    world.script += [fold_fakes.Script(status="failed")] * 2
    code, out = main(capsys, "fold", "--fold", "q-2024-07", "--approved-by-owner", "10")
    assert code == 1 and "ERROR: FoldFailed:" in out and "after its retry" in out
    assert kinds() == ["gate3_approval", "exploration", "fit", "fit"]


@pytest.mark.skipif(
    not (paths.REPO_ROOT / "data").is_dir(), reason="the repository has no data/ to protect"
)
def test_a_sealed_fold_on_a_relocated_ledgers_order_is_refused_for_the_repositorys_data(
    world, capsys, monkeypatch
):
    """The reviewer's command, ``fold --fold s-2026-09 --order order-0001 --estimate``, with
    the order in the test's (relocated) ledger and the data root left at the repository's
    data/: exit 2 before any plan is printed (only metadata of data/ is read)."""
    monday = datetime(2026, 10, 5, 12, tzinfo=UTC)
    fold_fakes.write_bundle(R1, now=monday)
    budget.spend_test("T-A", recipe_id=recipe_id(R1), now=monday)
    budget.spend_test("T-B", recipe_id=recipe_id(R1), now=monday + timedelta(days=7))
    ledger.record_order(
        scope="open_sealed",
        candidate=ledger.id_for_test("T-B", recipe_id(R1)),
        text="open it once",
        now=monday + timedelta(days=7),
    )
    argv = ("fold", "--fold", "s-2026-09", "--order", "order-0001", "--estimate")
    code, out = main(capsys, *argv)
    assert code == 0 and "order order-0001" in out  # its own data root: planned
    monkeypatch.delenv(paths.DATA_DIR_ENV)
    code, out = main(capsys, *argv)
    assert code == 2 and "\nREFUSED: the experiment ledger in use is not the repository's" in out
    assert "finetune fold s-2026-09" not in out and world.exports == []


def test_a_stale_timing_record_is_a_refusal(world, capsys):
    record = fold_fakes.timing_record()
    record["environment"]["torch_version"] = "2.0.0"
    fold_fakes.write_timing_record(record)
    code, out = main(capsys, "fold", "--fold", "q-2024-07", "--estimate")
    assert code == 2 and out.startswith("REFUSED: the timing record") and "stale" in out


# -- pilot-check -----------------------------------------------------------------------------------


class PilotCheckFakes:
    """Recording fakes of the pilot check's public functions."""

    def __init__(
        self,
        monkeypatch,
        *,
        cpu_seconds: float = 3200.0,
        reading: str = "GO",
        stop: bool = False,
    ):
        self.calls: list[tuple[str, object]] = []
        self.cpu_seconds = cpu_seconds
        self.reading = reading
        self.stop = stop
        self.approved: list[float | None] = []
        self.prior: list[float] = []
        for name in ("real_steps", "record_check_config", "prepare_check", "estimate_check"):
            monkeypatch.setattr(pilot_check, name, getattr(self, name))
        monkeypatch.setattr(pilot_check, "run_check", self.run_check)

    def real_steps(self):
        return "pilot-check steps"

    def record_check_config(self, recipe, fold, *, directory=None):
        self.calls.append(("record_check_config", (recipe, fold.label)))
        return "c" * 64

    def prepare_check(self, inputs, *, params, steps, root=None, directory=None):
        self.calls.append(("prepare_check", inputs))
        return "prepared"

    def estimate_check(self, prepared, *, params, steps):
        self.calls.append(("estimate_check", prepared))
        return {
            "cpu_seconds": self.cpu_seconds,
            "cpu_hours": self.cpu_seconds / 3600,
            "pending": 504,
            "needs_owner_approval": self.cpu_seconds > 3600,
        }

    def run_check(
        self,
        prepared,
        *,
        params,
        steps,
        root=None,
        directory=None,
        progress=print,
        approved_cpu_hours=None,
        prior_cpu_seconds=0.0,
    ):
        self.calls.append(("run_check", ledger_kinds_now()))
        self.approved.append(approved_cpu_hours)
        self.prior.append(prior_cpu_seconds)
        if self.stop:
            limit = pilot_check.spend_limit(approved_cpu_hours)
            raise pilot_check.PilotCheckStopped(
                f"{PILOT_STEM}: the run's spend, {limit + 10:.0f} CPU-seconds, passed its limit of "
                f"{limit:.0f}",
                spend={
                    "cpu_seconds": limit + 10.0,
                    "run_cpu_seconds": limit + 10.0 - prior_cpu_seconds,
                    "prior_cpu_seconds": prior_cpu_seconds,
                    "limit_cpu_seconds": limit,
                    "approved_cpu_hours": approved_cpu_hours,
                    "stopped": "after 100 of the 168 pilot fans",
                },
            )
        path = reports_dir() / "finetune" / f"{PILOT_STEM}.json"
        return pilot_check.CheckResult(
            report={},
            json_path=path,
            markdown_path=path.with_suffix(".md"),
            details={"reading": self.reading, "a_same": 0.5, "decisions": 168},
            reading=self.reading,
            summary_lines=("A_same 0.5 (pilot and shipped, common seeds)",),
        )

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def ledger_kinds_now() -> list[str]:
    return fold_fakes.ledger_kinds()


def test_pilot_check_record_config_runs_before_the_pilot_exists(world, capsys, monkeypatch):
    fakes = PilotCheckFakes(monkeypatch)
    code, out = main(capsys, "pilot-check", "--record-config")
    assert code == 0 and f"{PILOT_STEM}.config.json recorded (sha256 {'c' * 64})" in out
    assert fakes.calls == [("record_check_config", (R1, "q-2024-07"))]
    assert not ledger.ledger_path().exists()


def test_pilot_check_needs_the_pilot(world, capsys, monkeypatch):
    fakes = PilotCheckFakes(monkeypatch)
    code, out = main(capsys, "pilot-check", "--estimate")
    assert code == 2 and "REFUSED: train the pilot first" in out and fakes.calls == []


def test_pilot_check_estimate_gate3_approval_and_reading(world, capsys, monkeypatch):
    train_pilot(capsys, world)
    before = kinds()
    fakes = PilotCheckFakes(monkeypatch, cpu_seconds=4000.0)
    code, out = main(capsys, "pilot-check", "--estimate")
    assert code == 0 and "4000 CPU-seconds (1.11 CPU-hours) for 504 pending fans" in out
    assert "--estimate: nothing ran" in out and kinds() == before
    (_, inputs) = fakes.calls[0]
    fit = ledger.entries_of("fit")[0].payload
    assert inputs.checkpoint.checkpoint_id == "ks-btc-0ee0d123-q-2024-07"
    assert inputs.checkpoint.line_sha256 == fit["registry_line_sha256"]
    assert inputs.fit_record_sha256 == fit["fit_record_sha256"]
    assert (inputs.recipe, inputs.fold.label) == (R1, "q-2024-07")
    code, out = main(capsys, "pilot-check")
    assert code == 2 and out.startswith("pilot check pilot-check-0ee0d123 of ks-btc-")
    assert "\nREFUSED: the batch needs about 4000 CPU-seconds" in out
    assert "run_check" not in fakes.names() and kinds() == before
    code, out = main(capsys, "pilot-check", "--approved-by-owner", "1.5")
    assert code == 0, out
    assert kinds() == [*before, "gate3_approval", "exploration"]
    assert fakes.calls[-1] == ("run_check", [*before, "gate3_approval"])  # approval first
    approval = ledger.entries_of("gate3_approval")[-1].payload
    assert (approval["command"], approval["subject"], approval["unit"]) == (
        "finetune pilot-check",
        PILOT_STEM,
        "cpu_hours",
    )
    assert approval["approved"] == 1.5 and approval["estimate_detail"]["pending"] == 504
    done = ledger.entries_of("exploration")[-1].payload
    assert (done["activity"], done["subject"], done["stage"]) == ("pilot-check", PILOT_STEM, "done")
    assert (done["recipe_id"], done["inputs"], done["outcomes"]) == (
        recipe_id(R1),
        ["training"],
        ["training"],
    )
    assert done["details"]["a_same"] == 0.5
    assert "A_same 0.5" in out and out.rstrip().endswith("READING: GO")


def test_pilot_check_no_go_is_a_result_and_exits_0(world, capsys, monkeypatch):
    train_pilot(capsys, world)
    fakes = PilotCheckFakes(monkeypatch, reading="NO-GO")
    code, out = main(capsys, "pilot-check")
    assert code == 0 and "gate3_approval" not in kinds()[4:]
    assert out.rstrip().endswith(f"READING: NO-GO: {pilot_check.NO_GO_OPTIONS}")
    assert fakes.approved == [None]  # below gate 3: the run's limit is gate 3 itself
    assert "spend control: the run stops once it passes 3600 CPU-seconds" in out


def test_a_pilot_check_its_spend_stopped_is_a_refusal_without_a_done_entry(
    world, capsys, monkeypatch
):
    train_pilot(capsys, world)
    before = kinds()
    fakes = PilotCheckFakes(monkeypatch, cpu_seconds=4000.0, stop=True)
    code, out = main(capsys, "pilot-check", "--approved-by-owner", "1.5")
    assert fakes.approved == [1.5] and fakes.prior == [0.0]
    assert "spend control: the run stops once it passes 6750 CPU-seconds" in out
    assert code == 2 and "REFUSED: pilot-check-0ee0d123: the run's spend" in out
    # the approval stays, and the stopped run is recorded with what it spent; no done entry
    assert kinds() == [*before, "gate3_approval", "exploration"]
    stopped = ledger.entries_of("exploration")[-1]
    assert (stopped.payload["subject"], stopped.payload["stage"]) == (PILOT_STEM, "stopped")
    details = stopped.payload["details"]
    assert details["run_cpu_seconds"] == 6760.0 and details["approval_seq"] == len(kinds()) - 1
    assert f"ledger: stopped entry #{stopped.seq} (line" in out and "6760 CPU-seconds" in out


def test_a_rerun_after_a_stop_counts_what_the_stopped_run_spent(world, capsys, monkeypatch):
    """Design 3.5: a lost run's compute counts against the approval. A run stopped at gate 3's
    3,600 CPU-seconds is recorded; the rerun's estimate (3,200 for the pending fans) is shown
    with the stopped run's 3,610: 6,810 in all, above gate 3, so it needs a figure covering the
    total, and it runs from that spend."""
    train_pilot(capsys, world)
    fakes = PilotCheckFakes(monkeypatch, cpu_seconds=3200.0, stop=True)
    code, out = main(capsys, "pilot-check")  # below gate 3: no figure, stopped at 3,600
    assert code == 2 and fakes.prior == [0.0]
    stopped = ledger.entries_of("exploration")[-1]
    assert stopped.payload["details"]["run_cpu_seconds"] == 3610.0
    fakes.stop = False
    code, out = main(capsys, "pilot-check", "--estimate")
    assert (
        code == 0
        and (
            f"plus the 3610 CPU-seconds its stopped runs spent (ledger entry #{stopped.seq}): 6810 "
            "CPU-seconds (1.89 CPU-hours) in all; above gate 3"
        )
        in out
    )
    code, out = main(capsys, "pilot-check")
    assert code == 2 and "REFUSED: the batch needs about 6810 CPU-seconds" in out
    assert fakes.prior == [0.0]  # nothing ran
    code, out = main(capsys, "pilot-check", "--approved-by-owner", "2")
    assert code == 0 and fakes.prior == [0.0, 3610.0] and fakes.approved[-1] == 2.0
    approval = ledger.entries_of("gate3_approval")[-1].payload
    assert approval["estimate"] == pytest.approx(6810.0 / 3600)
    assert approval["estimate_detail"]["prior_cpu_seconds"] == 3610.0
    assert "counted from the 3610 its stopped runs spent" in out
    done = ledger.entries_of("exploration")[-1]
    assert done.payload["stage"] == "done"
    assert pilot_check.prior_spend(PILOT_STEM, entries=ledger.read_ledger()) == {
        "cpu_seconds": 0.0,
        "entries": [],
    }


# -- tokenizer-diag --------------------------------------------------------------------------------


class DiagFakes:
    """Recording fakes of the tokenizer diagnostic's public functions."""

    def __init__(self, monkeypatch, *, cpu_seconds: float = 120.0):
        self.calls: list[tuple[str, object]] = []
        self.cpu_seconds = cpu_seconds
        monkeypatch.setattr(tokenizer_diag, "real_steps", lambda: "diag steps")
        for name in ("record_diag_config", "prepare_diag", "estimate_diag", "run_diag"):
            monkeypatch.setattr(tokenizer_diag, name, getattr(self, name))

    def record_diag_config(self, part, *, directory=None):
        self.calls.append(("record_diag_config", part))
        return "d" * 64

    def prepare_diag(self, part, *, params, steps, root=None, directory=None):
        self.calls.append(("prepare_diag", part))
        return part

    def estimate_diag(self, prepared, *, steps):
        self.calls.append(("estimate_diag", prepared))
        return {
            "windows": 1634,
            "seconds_per_window": 0.05,
            "cpu_seconds": self.cpu_seconds,
            "gate3_cpu_seconds": 3600,
        }

    def run_diag(self, prepared, *, params, steps, directory=None, progress=print):
        self.calls.append(("run_diag", prepared))
        stem = tokenizer_diag.STEMS[prepared]
        path = reports_dir() / "finetune" / f"{stem}.json"
        flags = (
            {
                "flag": "r1_token_mismatch",
                "message": "binance: R1's context-token mismatch above 5 %",
            },
            {"flag": "rho", "message": "binance: the return resolution rho above 0.25"},
        )
        return tokenizer_diag.DiagResult(
            report={},
            json_path=path,
            markdown_path=path.with_suffix(".md"),
            details={"windows_kept": {"binance": 1534}, "flags_raised": ["rho/binance"]},
            flags=flags if prepared == "training" else (),
            summary_lines=(f"{stem}: summary",),
        )


def test_tokenizer_diag_part_a_runs_and_records_its_exploration(world, capsys, monkeypatch):
    fakes = DiagFakes(monkeypatch)
    code, out = main(capsys, "tokenizer-diag", "--part", "training", "--record-config")
    assert code == 0 and f"tokenizer-training.config.json recorded (sha256 {'d' * 64})" in out
    assert fakes.calls == [("record_diag_config", "training")]
    assert not ledger.ledger_path().exists()
    code, out = main(capsys, "tokenizer-diag", "--part", "training")
    assert code == 0, out
    (entry,) = ledger.entries_of("exploration")
    payload = entry.payload
    assert (payload["activity"], payload["subject"], payload["stage"]) == (
        "tokenizer-diag",
        "tokenizer-training",
        "done",
    )
    assert (payload["recipe_id"], payload["inputs"], payload["outcomes"]) == (
        None,
        ["training"],
        ["training"],
    )
    assert payload["details"]["windows_kept"] == {"binance": 1534}
    assert (
        "WARNING: r1_token_mismatch: binance: R1's context-token mismatch above 5 %; tell the "
        "owner before the pre-registration bundle (D-048 (5))\n" in out
    )
    assert "WARNING: rho: binance: the return resolution rho above 0.25\n" in out
    assert "about 120 CPU-seconds for 1634 windows" in out


def test_tokenizer_diag_above_gate3_is_refused(world, capsys, monkeypatch):
    fakes = DiagFakes(monkeypatch, cpu_seconds=5000.0)
    code, out = main(capsys, "tokenizer-diag", "--part", "training")
    assert code == 2 and "REFUSED:" in out and "no approval flag" in out
    assert "run_diag" not in [name for name, _ in fakes.calls]
    assert not ledger.ledger_path().exists()


def test_tokenizer_diag_part_b_runs_only_after_the_bundle(world, capsys, monkeypatch):
    fakes = DiagFakes(monkeypatch)
    code, out = main(capsys, "tokenizer-diag", "--part", "validation", "--record-config")
    assert code == 0 and fakes.calls == [("record_diag_config", "validation")]
    code, out = main(capsys, "tokenizer-diag", "--part", "validation")
    assert code == 2 and out.startswith("REFUSED: Part B runs after the pre-registration bundle")
    assert [name for name, _ in fakes.calls] == ["record_diag_config"]
    fold_fakes.write_bundle(R1)
    code, out = main(capsys, "tokenizer-diag", "--part", "validation")
    assert code == 0, out
    payload = ledger.entries_of("exploration")[-1].payload
    assert (payload["subject"], payload["inputs"], payload["outcomes"]) == (
        "tokenizer-validation",
        ["validation"],
        [],
    )
    assert "ledger: 1 pre-registration bundle(s) verified" in out


def test_part_b_and_the_blinded_folds_never_read_the_repositorys_data_under_another_ledger(
    world, capsys, monkeypatch, tmp_path
):
    """The reviewer's setup: a stand-in repository whose data/ is the data root (as an unset
    KRONOS_AGENT_DATA_DIR makes the real one) and the test's own, relocated reports root, whose
    ledger records a bundle. Part B and a fold after the pilot are refused (exit 2) before any
    window is built or any plan is printed; with a data root of its own, both run."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "reports" / "ledger").mkdir(parents=True)
    monkeypatch.setattr(ledger, "REPO_ROOT", repo)
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(repo / "data"))
    diag = DiagFakes(monkeypatch)
    fold_fakes.write_bundle(R1)
    before = kinds()
    code, out = main(capsys, "tokenizer-diag", "--part", "validation")
    assert code == 2 and "\nREFUSED: the experiment ledger in use is not the repository's" in out
    assert diag.calls == [] and kinds() == before
    code, out = main(capsys, "fold", "--fold", "q-2024-10", "--estimate")
    assert code == 2 and out.startswith("ledger: 1 pre-registration bundle(s) verified\nREFUSED:")
    assert "finetune fold q-2024-10" not in out and world.exports == []
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "own-data"))
    code, out = main(capsys, "tokenizer-diag", "--part", "validation")
    assert code == 0 and [name for name, _ in diag.calls][:1] == ["prepare_diag"]
    code, out = main(capsys, "fold", "--fold", "q-2024-10", "--estimate")
    assert code == 0 and "finetune fold q-2024-10: quarterly layout (D-056), blinded" in out


def test_a_training_child_left_running_makes_the_fold_command_exit_2(world, capsys):
    world.running.append((4242, str(runs_dir() / "q-2024-07-20261103T120000Z")))
    code, out = main(capsys, "fold", "--fold", "q-2024-07", "--estimate")
    assert code == 2 and out.startswith("REFUSED: a training child is still running")
    assert "pid 4242, run q-2024-07-20261103T120000Z (this fine-tune root)" in out
    assert not ledger.ledger_path().exists() and world.fits == []


# -- the bundle check ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["fold", "--fold", "q-2024-07", "--estimate"],
        ["pilot-check", "--record-config"],
        ["pilot-check", "--estimate"],
        ["tokenizer-diag", "--part", "training", "--record-config"],
        ["tokenizer-diag", "--part", "validation"],
    ],
)
def test_a_changed_bundle_makes_every_subcommand_exit_2(world, capsys, monkeypatch, argv):
    pilot = PilotCheckFakes(monkeypatch)
    diag = DiagFakes(monkeypatch)
    fold_fakes.write_bundle(R1)
    path: Path = fold_fakes.bundle_path(R1)
    path.write_bytes(path.read_bytes() + b"\n")
    code, out = main(capsys, *argv)
    assert code == 2 and out.startswith("REFUSED: the pre-registration bundle"), out
    assert "changed after ledger entry #1" in out
    assert pilot.calls == [] and diag.calls == [] and world.fits == []
