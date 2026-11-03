"""The parent of one training child (``finetune.train``; ``docs/RETRAINING.md`` 3.1, 3.5, 3.6):
the child's command line and environment allow-list, the run inputs (``run.json``, ``run.yaml``),
the run directory, waiting, timeouts and the result, ``fit`` with and without registration (only
a recipe run is registered, and a taken checkpoint id is refused before the child starts), the
registry payload, a gapped export without the ``gap_aware`` wrapper, and tests T19 (the parent's
timeout) and T20 (two CPU runs of a tiny fold give the same checkpoint). The integration clauses
of T2 and T3 (the exporter's CSVs as the vendored dataset and the child read them) are here too;
T8's is in ``test_finetune_runner.py``.

Children are real (tiny models, tiny synthetic folds from the exporter) except where a test
replaces the command line with a stand-in process to reach a path a real child never takes.
Registration runs against a temporary fine-tune root; nothing touches ``weights/``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

import finetune_fakes as ff
import numpy as np
import pandas as pd
import pytest
import step2_fakes
import yaml

from kronos_agent.data.timeutil import HOUR_MS
from kronos_agent.finetune import registry, runner, train
from kronos_agent.finetune.export_csv import ExportError, export_fold
from kronos_agent.finetune.recipe import recipe8, recipe_id, recipe_mapping, training_seeds
from kronos_agent.finetune.train import (
    ChildFailed,
    Compute,
    ModelFiles,
    TimingSpec,
    TrainError,
    Wrappers,
)
from kronos_agent.hashing import sha256_bytes, sha256_file
from kronos_agent.kronos import vendored as vendored_module
from kronos_agent.kronos.vendored import VendoredTreeError
from kronos_agent.kronos.weights import load_registry, repos_for_model
from kronos_agent.manifests import VerifyReport
from kronos_agent.models import spec_for
from kronos_agent.paths import weights_dir
from kronos_agent.runtime import THREAD_ENV_VARS

FAST_LR = 1e-3
CREDENTIALS = ("BINGX_API_" + "KEY", "BINGX_API_" + "SECRET")  # assembled: hygiene rule


@pytest.fixture
def tiny(tmp_path):
    return ff.tiny_run(tmp_path)


# -- command line and environment ----------------------------------------------------------------


def test_child_command_runs_the_runner_module_without_bytecode_or_working_directory(tmp_path):
    run_dir = tmp_path / "run"
    assert train.child_command(run_dir) == [
        sys.executable,
        "-B",
        "-P",
        "-m",
        "kronos_agent.finetune.runner",
        "--run-dir",
        str(run_dir.absolute()),
    ]
    assert train.CHILD_MODULE == runner.__name__


def test_child_environment_is_an_allow_list():
    environ = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-home",
        "TMPDIR": "/tmp/x",
        "LANG": "C.UTF-8",
        "KRONOS_AGENT_WEIGHTS_DIR": "/w",
        "PYTHONPATH": "/elsewhere",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "VIRTUAL_ENV": "/venv",
        "OMP_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "0",
        "SOME_TOKEN": "x",
        **dict.fromkeys(CREDENTIALS, "not-a-real-value"),
    }
    environment = train.child_environment(Compute(threads=3), environ=environ)
    expected = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-home",
        "TMPDIR": "/tmp/x",
        "LANG": "C.UTF-8",
        "KRONOS_AGENT_WEIGHTS_DIR": "/w",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        **dict.fromkeys(THREAD_ENV_VARS, "3"),
    }
    assert environment == expected
    assert not set(CREDENTIALS) & set(environment)
    mps = train.child_environment(Compute(device="mps"), environ={})
    assert mps[train.MPS_WATERMARK_ENV] == "1.0" and mps[train.MPS_LOW_WATERMARK_ENV] == "1.0"
    high = train.child_environment(Compute(device="mps", mps_watermark_ratio=1.7), environ={})
    assert high[train.MPS_WATERMARK_ENV] == "1.7" and high[train.MPS_LOW_WATERMARK_ENV] == "1.4"
    low = train.child_environment(Compute(device="mps", mps_watermark_ratio=0.5), environ={})
    assert low[train.MPS_WATERMARK_ENV] == "0.5" and low[train.MPS_LOW_WATERMARK_ENV] == "0.5"
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in mps


def test_compute_timing_and_request_validation(tiny):
    for arguments in (
        {"device": "cuda"},
        {"threads": 0},
        {"mps_watermark_ratio": 1.0},  # MPS only
        {"device": "mps", "mps_watermark_ratio": 1.8},
        {"memory_guard": "yes"},
    ):
        with pytest.raises(TrainError):
            Compute(**arguments)
    with pytest.raises(TrainError, match="exactly one"):
        TimingSpec(warmup_steps=1)
    with pytest.raises(TrainError, match="exactly one"):
        TimingSpec(warmup_steps=1, timed_steps=2, timed_seconds=3.0)
    assert Wrappers.none() == Wrappers(**dict.fromkeys(runner.WRAPPERS, False))
    assert tuple(asdict(Wrappers())) == runner.WRAPPERS
    with pytest.raises(TrainError, match="mode"):
        tiny.request(mode="fly")
    with pytest.raises(TrainError, match="TimingSpec"):
        tiny.request(mode="timing", wrappers=Wrappers(early_stopping=False))
    with pytest.raises(TrainError, match="early_stopping"):
        tiny.request(mode="timing", timing=TimingSpec(warmup_steps=0, timed_steps=1))
    with pytest.raises(TrainError, match="distinct checks"):
        tiny.request(mode="preflight")
    with pytest.raises(TrainError, match="unknown preflight"):
        tiny.request(mode="preflight", preflight_checks=("parity", "speed"))
    with pytest.raises(TrainError, match="device wrapper"):
        tiny.request(compute=Compute(device="mps"), wrappers=Wrappers(device=False))
    with pytest.raises(TrainError, match="deadline"):
        tiny.request(deadline_seconds=0)
    with pytest.raises(TrainError, match="absolute"):
        ModelFiles(Path("model"), Path("/t"), tiny.model_files.predictor_sha256, {})


def test_shipped_model_files_name_the_registered_files_without_reading_them(tmp_path):
    files = train.shipped_model_files("small", base=tmp_path)
    model_repo, tokenizer_repo = repos_for_model("small")
    assert files.predictor_dir == (tmp_path / spec_for("small").repo_id).absolute()
    assert files.tokenizer_dir == (tmp_path / spec_for("small").tokenizer_repo_id).absolute()
    assert files.predictor_sha256 == {item.name: item.sha256 for item in model_repo.files}
    assert files.tokenizer_sha256 == {item.name: item.sha256 for item in tokenizer_repo.files}
    assert train.shipped_model_files().predictor_dir == (weights_dir() / model_repo.repo_id)


# -- the run directory and its inputs ------------------------------------------------------------


def test_new_run_dir_is_exclusive_and_named_after_the_label(tmp_path):
    first = train.new_run_dir("wf-2024-07", root=tmp_path)
    second = train.new_run_dir("wf-2024-07", root=tmp_path)
    assert first.parent == second.parent == tmp_path.absolute() / registry.RUNS_DIRNAME
    assert first.name.startswith("wf-2024-07-") and first != second
    assert first.is_dir() and second.is_dir()
    if first.name[:-2] == second.name[:-2]:  # same second: the suffix keeps them apart
        assert second.name.endswith("-1")
    with pytest.raises(ValueError, match="label"):
        train.new_run_dir("bad/label", root=tmp_path)


def test_run_inputs_follow_the_interfaces(tiny):
    request = tiny.request()
    run_dir = train.new_run_dir(tiny.fold.run_label, root=tiny.root)
    hashes = train.write_run_inputs(run_dir, request)
    run_json = run_dir / runner.RUN_JSON
    text = run_json.read_text(encoding="ascii")
    document = json.loads(text)
    assert text == json.dumps(document, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    assert hashes == {
        runner.RUN_YAML: sha256_file(run_dir / runner.RUN_YAML),
        runner.RUN_JSON: sha256_bytes(text.encode("ascii")),
    }
    assert sorted(document) == sorted(runner._RUN_KEYS)
    identifier = recipe_id(tiny.recipe)
    assert document["recipe"] == json.loads(json.dumps(recipe_mapping(tiny.recipe)))
    assert document["recipe_id"] == identifier and document["run_id"] == run_dir.name
    assert document["seeds"] == asdict(training_seeds(identifier, "synthetic", 0))
    assert document["yaml"] == {"name": "run.yaml", "sha256": hashes[runner.RUN_YAML]}
    assert document["train_csv"] == {
        "path": str(Path(tiny.export.csv_path).absolute()),
        "sha256": tiny.export.csv_sha256,
        "rows": 1060,
        "n_fit": 544,
        "n_val": 516,
    }
    assert document["nll_inputs"]["decisions"] == 129
    assert document["windows"] == {
        "gap_aware": False,
        "fit_windows_kept": 32,
        "val_windows_kept": 4,
    }
    assert document["compute"] == asdict(request.compute)
    assert document["wrappers"] == dict.fromkeys(runner.WRAPPERS, True)
    assert document["nll_batch_size"] == 16 and document["memory"]["every_steps"] == 50
    assert document["timing"] is None and document["preflight"] is None
    assert document["deadline_seconds"] is None and document["blinded"] is False
    loaded = yaml.safe_load((run_dir / runner.RUN_YAML).read_text(encoding="ascii"))
    assert loaded["training"]["seed"] == document["seeds"]["yaml_seed"]
    assert loaded["model_paths"]["base_save_path"] == str(run_dir / "vendored")
    assert loaded["model_paths"]["finetuned_tokenizer"] == str(tiny.model_files.tokenizer_dir)
    spec = runner._read_run_spec(run_dir)  # the child accepts what the parent writes
    assert runner._verify_inputs(spec)["run_json_sha256"] == hashes[runner.RUN_JSON]


def test_timing_and_preflight_inputs(tiny):
    timing = tiny.request(
        mode="timing",
        wrappers=Wrappers(early_stopping=False),
        timing=TimingSpec(warmup_steps=3, timed_seconds=5),
        deadline_seconds=60,
    )
    run_dir = ff.prepare_run_dir(tiny, timing)
    document = json.loads((run_dir / runner.RUN_JSON).read_text(encoding="ascii"))
    assert document["nll_inputs"] is None and document["deadline_seconds"] == 60.0
    assert document["timing"] == {"warmup_steps": 3, "timed_steps": None, "timed_seconds": 5.0}
    assert runner._read_run_spec(run_dir).timing == document["timing"]
    preflight = tiny.request(mode="preflight", preflight_checks=("parity", "multinomial"))
    run_dir = ff.prepare_run_dir(tiny, preflight)
    document = json.loads((run_dir / runner.RUN_JSON).read_text(encoding="ascii"))
    assert document["preflight"] == {
        "checks": ["parity", "multinomial"],
        "multinomial_rows": 16384,
        "multinomial_repetitions": 64,
        "steps": 20,
    }


# -- running children ----------------------------------------------------------------------------


def test_t20_two_cpu_runs_of_a_tiny_fold_give_the_same_checkpoint(tmp_path):
    recipe = ff.tiny_recipe(predictor_learning_rate=FAST_LR)
    tiny = ff.tiny_run(tmp_path, recipe=recipe)
    first = train.fit(tiny.request(), root=tiny.root, register=False)
    second = train.fit(tiny.request(), root=tiny.root, register=False)
    assert first.run_dir != second.run_dir
    assert first.status == second.status == "completed" and first.finite and second.finite
    assert first.selected_sha256 == second.selected_sha256
    assert first.checkpoint is None and second.checkpoint is None
    records = [ff.read_result(outcome.run_dir) for outcome in (first, second)]
    assert records[0]["epochs"] and [e["nll4_tf"] for e in records[0]["epochs"]] == [
        e["nll4_tf"] for e in records[1]["epochs"]
    ]
    assert records[0]["selected_pseudo_epoch"] == records[1]["selected_pseudo_epoch"]
    initial = sha256_file(tiny.model_files.predictor_dir / "model.safetensors")
    assert first.selected_sha256 != initial  # training moved the weights
    assert first.wall_seconds > 0 and first.cpu_seconds > 0


def test_printable_holds_no_selection_value(tiny):
    outcome = train.fit(tiny.request(blinded=True), root=tiny.root, register=False)
    assert outcome.blinded
    assert outcome.printable() == {
        "run_id": outcome.run_id,
        "status": "completed",
        "wall_seconds": outcome.wall_seconds,
        "finite": True,
        "selected_sha256": outcome.selected_sha256,
    }
    assert ff.read_result(outcome.run_dir)["epochs"]  # the record keeps them, blinded


def test_fit_reports_a_failed_child_and_hides_its_message_when_blinded(tiny):
    files = ModelFiles(
        predictor_dir=tiny.model_files.predictor_dir,
        tokenizer_dir=tiny.model_files.tokenizer_dir,
        predictor_sha256={"config.json": "0" * 64, "model.safetensors": "0" * 64},
        tokenizer_sha256=tiny.model_files.tokenizer_sha256,
    )
    with pytest.raises(ChildFailed, match="sha256 mismatch: predictor config.json") as raised:
        train.fit(tiny.request(model_files=files), root=tiny.root, register=False)
    assert "ended failed (exit 1)" in str(raised.value) and "RunnerError" in str(raised.value)
    with pytest.raises(ChildFailed) as blinded:
        train.fit(tiny.request(model_files=files, blinded=True), root=tiny.root, register=False)
    assert "RunnerError" in str(blinded.value) and "sha256" not in str(blinded.value)
    with pytest.raises(TrainError, match="train request"):
        train.fit(
            tiny.request(mode="preflight", preflight_checks=("parity",)),
            root=tiny.root,
            register=False,
        )


def wait_for_log(run_dir: Path, text: str, seconds: float = 120.0) -> None:
    for _ in range(int(seconds / 0.25)):
        if text in ff.child_log(run_dir, tail=None):
            return
        time.sleep(0.25)
    raise AssertionError(f"{text!r} never appeared in the child's log")


def test_t19_the_parent_timeout_terminates_the_child(tmp_path, monkeypatch):
    """A child still measuring when the parent's timeout fires gets SIGTERM: it writes a failed
    result and exits 128 + 15. The timeout is armed once the child is training (its first logged
    step), so a slow start cannot turn the test into a kill before the handlers exist."""
    monkeypatch.setattr(train, "KILL_GRACE_SECONDS", 60.0)
    recipe = ff.tiny_recipe(pseudo_epoch_steps=4096)
    tiny = ff.tiny_run(tmp_path, recipe=recipe)
    request = tiny.request(
        mode="timing",
        wrappers=Wrappers(early_stopping=False),
        timing=TimingSpec(warmup_steps=1, timed_seconds=600),
    )
    running = train.start_run(request, root=tiny.root)
    wait_for_log(running.run_dir, "[Epoch 1/3, Step 2/4096]")
    result = running.wait(timeout_seconds=1)
    assert result.timed_out
    assert result.status == "failed", ff.child_log(result.run_dir)
    assert result.exit_code == 128 + signal.SIGTERM
    assert result.document["error"]["message"] == f"stopped by a signal (exit {result.exit_code})"
    assert result.document["steps"] > 1  # it was training when it was stopped


STUBBORN_CHILD = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("ready", flush=True)
time.sleep(120)
"""


def test_a_child_that_ignores_sigterm_is_killed(tiny, monkeypatch):
    monkeypatch.setattr(train, "KILL_GRACE_SECONDS", 1.0)
    monkeypatch.setattr(train, "child_command", lambda run_dir: ff.python_command(STUBBORN_CHILD))
    running = train.start_run(tiny.request(), root=tiny.root)
    wait_for_log(running.run_dir, "ready")
    result = running.wait(timeout_seconds=1)
    assert result.timed_out and result.status == "killed"
    assert result.exit_code == -signal.SIGKILL and result.document == {}


def test_a_child_that_writes_no_result(tiny, monkeypatch):
    monkeypatch.setattr(train, "child_command", lambda run_dir: ff.python_command("pass"))
    result = train.run(tiny.request(), root=tiny.root)
    assert result.status == "no_result" and result.exit_code == 0 and not result.timed_out
    assert result.document == {}
    with pytest.raises(ChildFailed, match="ended no_result"):
        train.fit(tiny.request(), root=tiny.root, register=False)


def test_the_vendored_tree_is_verified_before_and_after_the_child(tiny, monkeypatch):
    bad = VerifyReport(problems=["unlisted file: third_party/kronos/model/__pycache__/x.pyc"])
    monkeypatch.setattr(vendored_module, "verify_vendored_tree", lambda **kwargs: bad)
    with pytest.raises(VendoredTreeError, match="__pycache__"):
        train.run(tiny.request(), root=tiny.root)
    assert not (tiny.root / registry.RUNS_DIRNAME).exists()  # nothing was started
    calls = []

    def verify_once(**kwargs):
        calls.append(kwargs)
        return VerifyReport() if len(calls) == 1 else bad

    monkeypatch.setattr(vendored_module, "verify_vendored_tree", verify_once)
    monkeypatch.setattr(train, "child_command", lambda run_dir: ff.python_command("pass"))
    with pytest.raises(VendoredTreeError):
        train.run(tiny.request(), root=tiny.root)
    assert len(calls) == 2


def test_an_export_of_another_fold_is_refused(tiny):
    other = ff.tiny_fold(tiny.recipe, rows=ff.TINY_ROWS + 4)
    with pytest.raises(TrainError, match=r"\['csv_end_ms', 'inner_validation_start_ms'\]"):
        train.run(tiny.request(fold=other), root=tiny.root)
    with pytest.raises(TrainError, match="recipe must be a Recipe"):
        tiny.request(recipe=object())
    assert not (tiny.root / registry.RUNS_DIRNAME).exists()


def test_a_changed_export_is_refused_before_anything_starts(tiny):
    csv_path = Path(tiny.export.csv_path)
    data = bytearray(csv_path.read_bytes())
    data[-3] = ord("9") if data[-3] != ord("9") else ord("8")
    os.chmod(csv_path, 0o644)
    csv_path.write_bytes(bytes(data))
    with pytest.raises(ExportError):
        train.run(tiny.request(), root=tiny.root)
    assert not (tiny.root / registry.RUNS_DIRNAME).exists()


# -- registration --------------------------------------------------------------------------------


def fake_result(run_dir: Path, document: dict) -> train.RunnerResult:
    run_dir.mkdir(parents=True)
    (run_dir / runner.RESULT_JSON).write_text(json.dumps(document), encoding="ascii")
    return train.RunnerResult(
        run_id=run_dir.name,
        run_dir=run_dir,
        mode="train",
        status="completed",
        exit_code=0,
        timed_out=False,
        document=document,
    )


def test_registry_payload_is_a_valid_fold_entry(tiny, tmp_path):
    request = tiny.request(model_files=train.shipped_model_files("small"))
    best = {"files": [{"name": "model.safetensors", "size": 1, "sha256": "a" * 64}]}
    document = {"torch_version": "2.14.1", "platform": "darwin-arm64", "vendored_best_model": best}
    result = fake_result(tmp_path / "runs" / "synthetic-run", document)
    checkpoint_id = registry.checkpoint_id_for(recipe8(tiny.recipe), "synthetic")
    payload = train.registry_payload(request, result, checkpoint_id=checkpoint_id)
    assert sorted(payload) == sorted(registry.PAYLOAD_KEYS)
    model_repo, tokenizer_repo = repos_for_model("small")
    assert payload["kind"] == "fold" and payload["checkpoint_id"] == checkpoint_id
    assert payload["base"] == {
        "repo_id": model_repo.repo_id,
        "sha256": {item.name: item.sha256 for item in model_repo.files},
    }
    assert payload["tokenizer"]["repo_id"] == tokenizer_repo.repo_id
    assert payload["expected_config"] == load_registry()[model_repo.repo_id].expected_config
    assert payload["data_end_ms"] == tiny.fold.csv_end_ms
    assert payload["fit_range_ms"] == list(tiny.fold.fit_range_ms)
    assert payload["seeds"] == asdict(training_seeds(recipe_id(tiny.recipe), "synthetic", 0))
    assert payload["fit_record_sha256"] == sha256_file(result.run_dir / runner.RESULT_JSON)
    assert payload["vendored_best_model_sha256"] == "a" * 64
    assert payload["training_code_sha256"] == train.training_code_sha256()
    assert payload["store_meta_sha256"] is None  # synthetic data
    assert (payload["device"], payload["threads"], payload["deterministic"]) == ("cpu", 2, True)
    assert payload["activation_checkpointing"] is False
    checkpointing = replace(
        request, compute=Compute(memory_guard=False, activation_checkpointing=True)
    )
    assert train.registry_payload(checkpointing, result, checkpoint_id=checkpoint_id)[
        "activation_checkpointing"
    ]
    # the registry accepts it: install a stand-in checkpoint with the shipped configuration
    source = tmp_path / "source"
    source.mkdir()
    config = load_registry()[model_repo.repo_id].expected_config
    (source / "config.json").write_text(json.dumps(config), encoding="ascii")
    (source / "model.safetensors").write_bytes(b"stand-in")
    entry = registry.install_checkpoint(
        source, payload=payload, check=lambda directory: [], root=tmp_path / "ft"
    )
    assert entry.checkpoint_id == checkpoint_id and entry.data_end_ms == tiny.fold.csv_end_ms
    with pytest.raises(TrainError, match="shipped weights"):
        train.registry_payload(tiny.request(), result, checkpoint_id=checkpoint_id)


@pytest.fixture
def tiny_registration(monkeypatch):
    """Stand-ins that let ``fit(register=True)`` register a tiny model, which is not a
    Kronos-small: its files are accepted as the start (they are not the registered shipped
    ones; the payload records the shipped hashes), the payload's expected configuration is the
    tiny one and the loadability check finds nothing. Everything else is the real path."""
    from kronos_agent.kronos import weights

    real_payload = train.registry_payload

    def tiny_payload(request, result, *, checkpoint_id):
        payload = real_payload(request, result, checkpoint_id=checkpoint_id)
        payload["expected_config"] = dict(ff.TINY_MODEL_CONFIG)
        return payload

    monkeypatch.setattr(train, "_shipped_start", lambda request: repos_for_model("small"))
    monkeypatch.setattr(train, "registry_payload", tiny_payload)
    monkeypatch.setattr(weights, "checkpoint_dir_problems", lambda directory: [])


def never_start(*args, **kwargs):
    raise AssertionError("the child must not start: the registration is refused first")


def test_fit_registers_the_selected_checkpoint_and_cleans_its_run(
    tiny, tiny_registration, monkeypatch
):
    """``fit(register=True)`` with the real registry on a tiny model (``tiny_registration``);
    the same fold is then refused before a second child starts."""
    outcome = train.fit(tiny.request(), root=tiny.root, register=True)
    entry = outcome.checkpoint
    assert entry is not None
    assert entry.checkpoint_id == registry.checkpoint_id_for(recipe8(tiny.recipe), "synthetic")
    files = {item.name: item for item in entry.files}
    assert files["model.safetensors"].sha256 == outcome.selected_sha256
    installed = registry.checkpoint_dir(registry.CheckpointRef(entry.checkpoint_id, tiny.root))
    assert sha256_file(installed / "model.safetensors") == outcome.selected_sha256
    assert not (outcome.run_dir / runner.SELECTED_DIRNAME).exists()
    assert not (outcome.run_dir / runner.VENDORED_BEST_RELPATH).exists()
    record = outcome.run_dir / runner.RESULT_JSON
    assert entry.document["fit_record_sha256"] == sha256_file(record)
    assert entry.document["run_id"] == outcome.run_id
    assert entry.document["activation_checkpointing"] is False  # the run's compute choice
    assert registry.read_registry(tiny.root)[-1].line_sha256 == entry.line_sha256
    assert "nll4_tf" not in entry.line.decode("ascii")  # selection values stay in the record
    # a registered fold is never trained again: refused before a second child starts
    runs = sorted(path.name for path in (tiny.root / registry.RUNS_DIRNAME).iterdir())
    monkeypatch.setattr(train, "start_run", never_start)
    with pytest.raises(registry.CheckpointExists, match="registered already"):
        train.fit(tiny.request(), root=tiny.root, register=True)
    assert sorted(path.name for path in (tiny.root / registry.RUNS_DIRNAME).iterdir()) == runs
    assert [item.line for item in registry.read_registry(tiny.root)] == [entry.line]


@pytest.mark.parametrize("off", [*runner.WRAPPERS, "all"])
def test_only_a_recipe_run_is_registered(tiny, tiny_registration, monkeypatch, off):
    """A wrapper off is a diagnostic configuration (with early stopping off, the vendored
    keep-best on the vendored validation loss, which reads the target in eval mode, M:387): it
    is refused before any child starts and never takes the recipe's checkpoint id."""
    wrappers = Wrappers.none() if off == "all" else Wrappers(**{off: False})
    monkeypatch.setattr(train, "start_run", never_start)
    with pytest.raises(TrainError, match="only a recipe run, every wrapper on") as refused:
        train.fit(tiny.request(wrappers=wrappers), root=tiny.root, register=True)
    assert (off if off != "all" else "device") in str(refused.value)
    assert not tiny.root.exists()  # no run directory, no registry, no lock


def test_registration_needs_the_shipped_weights_as_the_start(tiny, monkeypatch):
    monkeypatch.setattr(train, "start_run", never_start)
    with pytest.raises(TrainError, match="registered shipped weights"):
        train.fit(tiny.request(), root=tiny.root, register=True)  # tiny models
    assert not tiny.root.exists()


def test_a_taken_checkpoint_id_is_refused_before_the_child_starts(tiny, tmp_path, monkeypatch):
    """The fold's id registered already, its directory left by an interrupted install, or a
    registry that does not verify: refused before any compute is spent (design 3.5)."""
    monkeypatch.setattr(train, "start_run", never_start)
    request = tiny.request(model_files=train.shipped_model_files("small"))
    checkpoint_id = registry.checkpoint_id_for(recipe8(tiny.recipe), tiny.fold.run_label)
    entry = ff.install_stand_in(tiny.root, checkpoint_id, data_end_ms=tiny.fold.csv_end_ms)
    with pytest.raises(registry.CheckpointExists, match="registered already"):
        train.fit(request, root=tiny.root, register=True)
    leftover = tmp_path / "leftover"
    (leftover / checkpoint_id).mkdir(parents=True)
    with pytest.raises(registry.CheckpointExists, match="exists without a registry entry"):
        train.fit(request, root=leftover, register=True)
    torn = tmp_path / "torn"
    torn.mkdir()
    (torn / registry.REGISTRY_NAME).write_bytes(entry.line)  # no LF: a torn write
    with pytest.raises(registry.RegistryChainError, match="torn write"):
        train.fit(request, root=torn, register=True)
    assert not (tiny.root / registry.RUNS_DIRNAME).exists()
    assert sorted(path.name for path in leftover.iterdir()) == [checkpoint_id]
    # a fine-tune root without that id accepts the same request
    assert train._registration_id(request, root=tmp_path / "free") == checkpoint_id
    assert not (tmp_path / "free").exists()


def completed_result(run_dir: Path, *, wrappers: dict, source: str) -> train.RunnerResult:
    """A completed train run's result whose ``selected`` directory verifies on disk."""
    selected = run_dir / ("selected" if source == "nll4_tf" else runner.VENDORED_BEST_RELPATH)
    selected.mkdir(parents=True)
    files = []
    for name, data in (("config.json", b"{}"), ("model.safetensors", b"weights")):
        (selected / name).write_bytes(data)
        files.append({"name": name, "size": len(data), "sha256": sha256_bytes(data)})
    document = {
        "status": "completed",
        "finite": True,
        "wrappers": wrappers,
        "selected": {
            "dir": selected.relative_to(run_dir).as_posix(),
            "source": source,
            "files": files,
        },
    }
    (run_dir / runner.RESULT_JSON).write_text(json.dumps(document), encoding="ascii")
    return train.RunnerResult(
        run_id=run_dir.name,
        run_dir=run_dir,
        mode="train",
        status="completed",
        exit_code=0,
        timed_out=False,
        document=document,
    )


@pytest.mark.parametrize(
    "wrappers, source",
    [
        (dict.fromkeys(runner.WRAPPERS, True), "vendored_best"),
        ({**dict.fromkeys(runner.WRAPPERS, True), "clean_validation": False}, "nll4_tf"),
        (None, "nll4_tf"),
    ],
)
def test_a_record_that_is_not_a_recipe_run_is_never_registered(
    tiny, tmp_path, monkeypatch, wrappers, source
):
    """After the child: its record must show every wrapper on and the state NLL4-TF selected;
    anything else (a run.json changed under the parent, say) is refused and nothing installed."""
    result = completed_result(tmp_path / "run-x", wrappers=wrappers, source=source)
    monkeypatch.setattr(train, "run", lambda request, **kwargs: result)
    request = tiny.request(model_files=train.shipped_model_files("small"))
    with pytest.raises(TrainError, match="is not a recipe run") as refused:
        train.fit(request, root=tiny.root, register=True)
    assert "not registered" in str(refused.value)
    assert registry.read_registry(tiny.root) == ()
    assert (result.run_dir / result.document["selected"]["dir"]).is_dir()  # nothing removed


def test_a_gapped_export_needs_the_gap_aware_wrapper(tmp_path):
    """The vendored dataset would build windows across the missing hours (design 1.6): the
    request is refused before any run directory exists (the child refuses such a run.json
    too: ``test_finetune_runner.py``)."""
    recipe = ff.tiny_recipe(max_dropped_fit_share=0.5)
    tiny = ff.tiny_run(tmp_path, recipe=recipe, missing_hours=(5, 1058))
    assert tiny.export.gap_aware
    tiny.request()  # every wrapper on: accepted
    for wrappers in (Wrappers(gap_aware=False), Wrappers.none()):
        with pytest.raises(TrainError, match="the export has a gap"):
            tiny.request(wrappers=wrappers)
    assert not tiny.root.exists()
    ungapped = ff.tiny_run(tmp_path / "ungapped")
    assert not ungapped.export.gap_aware
    ungapped.request(wrappers=Wrappers.none())  # without a gap the equivalence run is allowed


HEAVY_IMPORTS = """
import json, sys
import kronos_agent.finetune.nll, kronos_agent.finetune.runner, kronos_agent.finetune.train
heavy = ("torch", "model", "train_sequential")
print(json.dumps(sorted(name for name in heavy if name in sys.modules)))
"""


def test_the_training_modules_import_neither_torch_nor_the_vendored_code():
    completed = subprocess.run(
        ff.python_command(HEAVY_IMPORTS), capture_output=True, text=True, timeout=300, check=False
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


# -- integration with the exporter (T2, T3) ------------------------------------------------------


def test_t2_the_vendored_dataset_splits_an_export_exactly_at_the_inner_validation(tiny):
    """The vendored ``CustomKlineDataset`` on an exported CSV, with the export's
    ``train_ratio`` and ``val_ratio`` 1.0, gives the fit rows to training and exactly the inner
    validation to validation (FB:75-97)."""
    vendored = runner.import_vendored_finetune()
    fold, record = tiny.fold, tiny.export
    opens = {}
    for data_type in ("train", "val"):
        dataset = vendored.CustomKlineDataset(
            data_path=str(record.csv_path),
            data_type=data_type,
            lookback_window=512,
            predict_window=0,
            clip=5.0,
            seed=1,
            train_ratio=record.train_ratio,
            val_ratio=1.0,
            test_ratio=0.0,
        )
        stamps = dataset.timestamps.to_numpy(dtype="datetime64[ms]").astype(np.int64)
        opens[data_type] = stamps
    hours = np.arange(fold.csv_start_ms, fold.csv_end_ms, HOUR_MS, dtype=np.int64)
    inner = fold.inner_validation_start_ms
    assert np.array_equal(opens["train"], hours[hours < inner])
    assert np.array_equal(opens["val"], hours[hours >= inner])
    assert opens["train"][-1] < inner <= opens["val"][0] and opens["val"][-1] < fold.csv_end_ms


def altered_after(frame: pd.DataFrame, cut_ms: int) -> pd.DataFrame:
    """Every price after ``cut_ms`` times 10 and every volume times 0.1 (plain decimals)."""
    out = frame.copy()
    later = out["open_ms"] >= cut_ms
    for name in ("open", "high", "low", "close"):
        out.loc[later, name] = [format(Decimal(v) * 10, "f") for v in out.loc[later, name]]
    out.loc[later, "volume"] = [format(Decimal(v) / 10, "f") for v in out.loc[later, "volume"]]
    return out


def test_t3_invented_future_candles_leave_a_tiny_checkpoint_bit_identical(tmp_path):
    """Two Binance stores that differ only after the CSV end (prices x10, volumes x0.1): their
    exports are identical, and so is the tiny CPU checkpoint trained on each."""
    recipe = ff.tiny_recipe(predictor_learning_rate=FAST_LR)
    fold = ff.tiny_fold(recipe)
    count = ff.TINY_ROWS + 200
    frame = step2_fakes.binance_frame(fold.csv_start_ms, count, interval="1h", seed=4)
    model_files = ff.save_tiny_models(tmp_path / "models")
    outcomes, exports = [], []
    for name, candles in (("base", frame), ("altered", altered_after(frame, fold.csv_end_ms))):
        store_root = tmp_path / name
        step2_fakes.write_binance_store(store_root, candles)
        record = export_fold(fold, recipe, root=store_root, out_dir=tmp_path / f"export-{name}")
        exports.append(record)
        tiny = ff.TinyRun(
            root=(tmp_path / f"ft-{name}").absolute(),
            recipe=recipe,
            fold=fold,
            export=record,
            model_files=model_files,
        )
        outcomes.append(train.fit(tiny.request(), root=tiny.root, register=False))
    base, altered = exports
    assert base.csv_path.read_bytes() == altered.csv_path.read_bytes()
    assert base.content_sha256 == altered.content_sha256
    assert base.store_meta_sha256 != altered.store_meta_sha256  # the stores did differ
    assert outcomes[0].selected_sha256 == outcomes[1].selected_sha256
    initial = sha256_file(model_files.predictor_dir / "model.safetensors")
    assert outcomes[0].selected_sha256 != initial
