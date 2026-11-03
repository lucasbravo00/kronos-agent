"""The training child (``finetune.runner``; ``docs/RETRAINING.md`` 3.1-3.5): tests T1, T4, T11
to T15, T17 to T19 of the design's plan, on tiny random Kronos models and tiny synthetic folds
exported by the real exporter (``finetune_fakes``).

T1 runs the child with every wrapper off and the vendored ``train_sequential.py`` itself on the
same YAML, and compares the vendored keep-best ``model.safetensors`` byte for byte. The other
training tests drive the vendored ``train_model`` in this process through the child's own
session (the wrappers, the gate, the monitor), or start the child through ``finetune.train``.
No test needs MPS (the MPS smoke test skips without it) or the real weights.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import finetune_fakes as ff
import numpy as np
import pytest
import torch
import yaml

from kronos_agent.finetune import runner, train
from kronos_agent.finetune.export_csv import vendored_float32
from kronos_agent.finetune.runner import (
    CleanValidation,
    MemoryAbort,
    MemoryGuard,
    MemoryReading,
    MultinomialRouter,
    RunnerError,
    TrainLoaderGate,
)
from kronos_agent.finetune.train import Compute, TimingSpec, Wrappers
from kronos_agent.hashing import sha256_file
from kronos_agent.kronos.vendored import import_kronos, verify_vendored_tree
from kronos_agent.paths import vendored_kronos_dir

# the vendored loaders ask for pinned memory (FB:218, FB:228), which torch declines with a warning
# on a machine whose accelerator is MPS; the design keeps the vendored loader as it is
pytestmark = pytest.mark.filterwarnings("ignore:'pin_memory' argument is set as true:UserWarning")
LOGGER = logging.getLogger("kronos_agent.tests.finetune_runner")
WINDOW = 513
FAST_LR = 1e-3  # large enough for the tiny weights to move visibly in a few steps


@pytest.fixture(scope="module")
def vendored():
    return runner.import_vendored_finetune()


@pytest.fixture
def tiny(tmp_path):
    return ff.tiny_run(tmp_path)


@pytest.fixture
def restore_signals():
    """``runner.run`` installs SIGTERM and SIGHUP handlers: put the test process's back."""
    saved = {name: signal.getsignal(getattr(signal, name)) for name in ("SIGTERM", "SIGHUP")}
    yield
    for name, handler in saved.items():
        signal.signal(getattr(signal, name), handler)


def same_state(first: dict, second: dict) -> bool:
    return first.keys() == second.keys() and all(
        runner._same_bits(first[name], second[name]) for name in first
    )


def train_in_process(
    tiny: ff.TinyRun,
    request: train.RunRequest,
    *,
    evaluate=None,
    early_stopping: bool = False,
    patience: int | None = None,
    without_validation: bool = False,
) -> SimpleNamespace:
    """The child's train mode in this process: the vendored phase's seeding and model loading
    (TS:165-225), then the session's ``train_model`` with its wrappers installed.

    ``evaluate(monitor, epoch)`` stands for NLL4-TF; ``without_validation`` hands the vendored
    loop an empty validation loader instead of the diagnostic.
    """
    vendored = runner.import_vendored_finetune()
    api = import_kronos()
    run_dir = ff.prepare_run_dir(tiny, request)
    spec = runner._read_run_spec(run_dir)
    holder = SimpleNamespace()
    monitor = runner._Monitor(
        evaluate=None if evaluate is None else (lambda epoch: evaluate(holder.monitor, epoch)),
        early_stopping=early_stopping,
        patience=patience or spec.recipe.patience,
    )
    holder.monitor = monitor
    router = MultinomialRouter(spec.seeds["router_seed"]) if spec.wrappers["router"] else None
    session = runner._Session(spec, vendored, monitor, router=router)
    if without_validation:
        create = session.create_dataloaders

        def no_validation(config):
            gate, _, train_ds, val_ds, train_sampler, val_sampler = create(config)
            return gate, [], train_ds, val_ds, train_sampler, val_sampler

        session.create_dataloaders = no_validation
    config = vendored.config_loader.CustomFinetuneConfig(str(run_dir / runner.RUN_YAML))
    vendored.train_sequential.set_seed(config.seed)
    tokenizer = api.KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
    model = api.Kronos.from_pretrained(config.pretrained_predictor_path)
    outcome = "max_epochs"
    with session.installed():
        try:
            session.train_model(
                model, tokenizer, torch.device("cpu"), config, config.basemodel_save_path, LOGGER
            )
        except runner.EarlyStop:
            outcome = "early_stop"
        else:
            monitor.finish()
    return SimpleNamespace(
        monitor=monitor, model=model, session=session, outcome=outcome, spec=spec, router=router
    )


def nll_evaluator(tiny: ff.TinyRun, decisions: int = 8):
    """A real NLL4-TF evaluation on CPU copies, as the child's (first ``decisions`` of the fold)."""
    from kronos_agent.finetune.nll import load_nll_inputs, nll4_tf

    inputs = load_nll_inputs(tiny.export.nll_inputs_path).head(decisions)
    tokenizer, model = ff.load_saved(tiny.model_files)

    def evaluate(monitor, epoch):
        model.load_state_dict(monitor.model.state_dict())
        return nll4_tf(tokenizer, model, inputs, batch_size=4).value

    return evaluate


# -- the vendored modules ------------------------------------------------------------------------

IMPORT_IN_CHILD = """
import json, sys
from kronos_agent import runtime
runtime.configure_runtime(2)
from kronos_agent.finetune import runner
path_before = list(sys.path)
first = runner.import_vendored_finetune()
second = runner.import_vendored_finetune()
print(json.dumps({
    "cached": first is second,
    "path_unchanged": sys.path == path_before,
    "files": {name: sys.modules[name].__file__ for name in runner.VENDORED_FINETUNE_MODULES},
    "trainer_is_vendored": first.SequentialTrainer.__module__ == "train_sequential",
}))
"""

FOREIGN_IN_CHILD = """
import sys, types
from kronos_agent import runtime
runtime.configure_runtime(2)
sys.modules["config_loader"] = types.ModuleType("config_loader")
from kronos_agent.finetune import runner
from kronos_agent.kronos.vendored import VendoredTreeError
try:
    runner.import_vendored_finetune()
except VendoredTreeError as exc:
    print("refused:", exc)
"""


def test_vendored_finetune_modules_come_from_the_verified_tree():
    completed = subprocess.run(
        ff.python_command(IMPORT_IN_CHILD), capture_output=True, text=True, timeout=300, check=False
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    outcome = json.loads(completed.stdout.strip().splitlines()[-1])
    assert outcome["cached"] and outcome["path_unchanged"] and outcome["trainer_is_vendored"]
    directory = (vendored_kronos_dir() / "finetune_csv").resolve()
    for name, location in outcome["files"].items():
        assert Path(location).resolve().parent == directory, name
    assert verify_vendored_tree(strict=True).ok  # no bytecode was written (D-014)


def test_a_foreign_module_of_a_vendored_name_is_refused():
    completed = subprocess.run(
        ff.python_command(FOREIGN_IN_CHILD),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "refused: a foreign module named 'config_loader'" in completed.stdout


# -- T1: equivalence with the vendored script ----------------------------------------------------


def test_t1_every_wrapper_off_reproduces_the_vendored_script_bit_for_bit(tmp_path):
    recipe = ff.tiny_recipe(max_epochs=2, predictor_learning_rate=FAST_LR)
    tiny = ff.tiny_run(tmp_path, recipe=recipe)
    compute = Compute(device="cpu", threads=2, deterministic=False, memory_guard=False)
    request = tiny.request(wrappers=Wrappers.none(), compute=compute)
    result = train.run(request, root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "completed", ff.child_log(result.run_dir)
    document = result.document
    assert document["selected"]["source"] == "vendored_best"
    assert document["dataset"]["class"] == "CustomKlineDataset"
    assert document["steps"] == 2 * (ff.TINY_FIT_WINDOWS // recipe.batch_size)
    # the vendored script on the same YAML, saving elsewhere, with the same thread variables
    reference_yaml = yaml.safe_load((result.run_dir / runner.RUN_YAML).read_text("ascii"))
    reference_yaml["model_paths"]["base_save_path"] = str(tmp_path / "reference")
    reference_path = tmp_path / "reference.yaml"
    reference_path.write_text(yaml.safe_dump(reference_yaml, sort_keys=True), encoding="ascii")
    completed = subprocess.run(
        [sys.executable, "-B", "train_sequential.py", "--config", str(reference_path)]
        + ["--skip-tokenizer"],
        cwd=vendored_kronos_dir() / "finetune_csv",
        env=train.child_environment(compute),
        capture_output=True,
        text=True,
        timeout=ff.CHILD_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    child = result.run_dir / runner.VENDORED_BEST_RELPATH / "model.safetensors"
    reference = tmp_path / "reference" / "basemodel" / "best_model" / "model.safetensors"
    assert child.read_bytes() == reference.read_bytes()
    # positive control: training moved the weights
    initial = tiny.model_files.predictor_dir / "model.safetensors"
    assert sha256_file(child) != sha256_file(initial)
    assert verify_vendored_tree(strict=True).ok


# -- T4 and the gap-aware dataset ----------------------------------------------------------------


def make_dataset(vendored, csv_path, train_ratio, *, normalisation=None, data_type="train"):
    base = vendored.CustomKlineDataset
    dataset_class = (
        base
        if normalisation is None
        else runner.make_dataset_class(base, normalisation=normalisation)
    )
    return dataset_class(
        data_path=str(csv_path),
        data_type=data_type,
        lookback_window=512,
        predict_window=0,
        clip=5.0,
        seed=1,
        train_ratio=train_ratio,
        val_ratio=1.0,
        test_ratio=0.0,
    )


def expected_window(floats: np.ndarray, start: int, *, context: int | None = None) -> np.ndarray:
    """The vendored normalisation (FB:122-127) of rows ``[start, start + 513)``, statistics over
    all of them, or over the first ``context`` rows."""
    x = np.asfortranarray(floats[start : start + WINDOW])
    stats = x if context is None else np.asfortranarray(x[:context])
    mean, std = np.mean(stats, axis=0), np.std(stats, axis=0)
    return np.clip((x - mean) / (std + 1e-5), -5.0, 5.0)


def train_start(index: int, samples: int, epoch: int = 0) -> int:
    return (index * 9973 + (epoch + 1) * 104729) % samples  # FB:114


def perturbed_csv(source: Path, target: Path, row: int, factor: float) -> Path:
    """A copy of the fold CSV whose data row ``row`` has its prices multiplied by ``factor``."""
    lines = source.read_text(encoding="ascii").splitlines()
    fields = lines[row + 1].split(",")
    for column in (1, 2, 3, 4):
        fields[column] = repr(float(fields[column]) * factor)
    lines[row + 1] = ",".join(fields)
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target


def test_t4_each_training_window_is_the_normalisation_of_its_own_rows(vendored, tiny):
    record = tiny.export
    floats = vendored_float32(record.csv_path)
    for normalisation in (None, "vendored"):
        dataset = make_dataset(
            vendored, record.csv_path, record.train_ratio, normalisation=normalisation
        )
        assert len(dataset) == ff.TINY_FIT_WINDOWS
        starts = set()
        for index in range(len(dataset)):
            start = train_start(index, len(dataset))
            starts.add(start)
            x, stamp = dataset[index]
            assert np.array_equal(
                x.numpy().view(np.uint32), expected_window(floats, start).view(np.uint32)
            )
            assert stamp.shape == (WINDOW, 5)
        assert starts == set(range(ff.TINY_FIT_WINDOWS))  # the mapping is a bijection here
    # rows outside [s, s + 513) do not enter the window
    start = 7
    index = next(i for i in range(32) if train_start(i, 32) == start)
    changed = Path(tiny.root.parent / "outside.csv")
    perturbed_csv(Path(record.csv_path), changed, start + WINDOW, 10.0)
    perturbed_csv(changed, changed, start - 1, 10.0)
    base = make_dataset(vendored, record.csv_path, record.train_ratio)[index][0]
    moved = make_dataset(vendored, changed, record.train_ratio)[index][0]
    assert torch.equal(base, moved)


def test_t4_positive_control_the_vendored_look_ahead_is_detected(vendored, tiny):
    """Changing the window's last row (the target of its last position) moves the vendored
    normalisation of every context row; context-only statistics leave rows 0..511 alone."""
    record = tiny.export
    start = 3
    index = next(i for i in range(32) if train_start(i, 32) == start)
    changed = perturbed_csv(Path(record.csv_path), tiny.root.parent / "last.csv", start + 512, 10.0)
    vendored_base = make_dataset(vendored, record.csv_path, record.train_ratio)[index][0]
    vendored_moved = make_dataset(vendored, changed, record.train_ratio)[index][0]
    assert not torch.equal(vendored_base[0], vendored_moved[0])  # the look-ahead
    context_base = make_dataset(
        vendored, record.csv_path, record.train_ratio, normalisation="context_only"
    )[index][0]
    context_moved = make_dataset(
        vendored, changed, record.train_ratio, normalisation="context_only"
    )[index][0]
    assert torch.equal(context_base[:512], context_moved[:512])
    assert not torch.equal(context_base[512], context_moved[512])
    floats = vendored_float32(record.csv_path)
    expected = expected_window(floats, start, context=512)
    assert np.array_equal(context_base.numpy().view(np.uint32), expected.view(np.uint32))


def test_t4_context_only_makes_every_position_invariant_to_the_last_row(vendored, tiny):
    """On a tiny model, the s1 logits of positions 0..511 do not move when row 512 does, under
    context-only statistics; under the vendored ones they do (positive control)."""
    record = tiny.export
    tokenizer, model = ff.load_saved(tiny.model_files)
    start = 11
    index = next(i for i in range(32) if train_start(i, 32) == start)
    changed = perturbed_csv(
        Path(record.csv_path), tiny.root.parent / "row512.csv", start + 512, 10.0
    )

    def logits(csv_path, normalisation):
        x, stamp = make_dataset(
            vendored, csv_path, record.train_ratio, normalisation=normalisation
        )[index]
        with torch.no_grad():
            s1, s2 = tokenizer.encode(x[None], half=True)
            s1_logits, _ = model.decode_s1(s1[:, :-1], s2[:, :-1], stamp[None, :-1])
        return s1[:, :-1], s1_logits

    tokens_base, base = logits(record.csv_path, "context_only")
    tokens_moved, moved = logits(changed, "context_only")
    assert torch.equal(tokens_base, tokens_moved)
    assert torch.equal(base, moved)
    tokens_base, base = logits(record.csv_path, "vendored")
    tokens_moved, moved = logits(changed, "vendored")
    assert not torch.equal(tokens_base, tokens_moved)
    assert not torch.equal(base, moved)


def test_gap_aware_dataset_keeps_exactly_the_exporters_windows(vendored, tmp_path):
    """Windows spanning a missing hour are dropped (nothing is filled), the counts are the
    exporter's, and a child run on the gapped fold agrees with them."""
    recipe = ff.tiny_recipe(max_dropped_fit_share=0.5, max_epochs=1, patience=1)
    tiny = ff.tiny_run(tmp_path, recipe=recipe, missing_hours=(5, 1058))
    record = tiny.export
    assert record.gap_aware
    assert (record.fit_windows_kept, record.val_windows_kept) == (26, 2)
    for data_type, kept in (("train", 26), ("val", 2)):
        dataset = make_dataset(
            vendored,
            record.csv_path,
            record.train_ratio,
            normalisation="vendored",
            data_type=data_type,
        )
        assert len(dataset) == kept < dataset.vendored_samples
        stamps = dataset.timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        hour = 3_600 * 10**9
        for start in range(dataset.vendored_samples):
            consecutive = stamps[start + WINDOW - 1] - stamps[start] == (WINDOW - 1) * hour
            assert consecutive == (start in set(dataset.kept_starts.tolist()))
        for index in range(len(dataset)):
            x, _ = dataset[index]
            assert x.shape == (WINDOW, 6)
    result = train.run(tiny.request(), root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "completed", ff.child_log(result.run_dir)
    assert result.document["dataset"] == {
        "class": "GapAwareKlineDataset",
        "fit_windows": 26,
        "val_windows": 2,
    }


def test_the_window_counts_are_the_exports_whichever_dataset_runs(vendored, tiny):
    """Without a gap the vendored dataset counts every window, as the exporter does; counts that
    differ from the export's are refused before the vendored loop starts, with the gap_aware
    wrapper off (the vendored dataset) as with it on."""
    request = tiny.request(wrappers=Wrappers(gap_aware=False, clean_validation=False))
    run_dir = ff.prepare_run_dir(tiny, request)
    config = vendored.config_loader.CustomFinetuneConfig(str(run_dir / runner.RUN_YAML))

    def loaders() -> runner._Session:
        session = runner._Session(runner._read_run_spec(run_dir), vendored, runner._Monitor())
        session.create_dataloaders(config)
        return session

    assert loaders().dataset == {
        "class": "CustomKlineDataset",
        "fit_windows": ff.TINY_FIT_WINDOWS,
        "val_windows": ff.TINY_VAL_WINDOWS,
    }
    rewrite(run_dir, lambda d: d["windows"].update(fit_windows_kept=ff.TINY_FIT_WINDOWS - 1))
    with pytest.raises(RunnerError, match="the datasets hold 32 fit and 4 validation windows"):
        loaders()
    rewrite(run_dir, lambda d: d["wrappers"].update(gap_aware=True))  # the wrapper on
    with pytest.raises(RunnerError, match="the datasets hold 32 fit and 4 validation windows"):
        loaders()


# -- T11, T12: the validation diagnostic ---------------------------------------------------------


def teacher_forced_s2(model, s1, s2, stamp):
    """s2 logits of a forward with the true s1 (deterministic: no multinomial)."""
    with torch.no_grad():
        _, s2_logits = model(s1, s2, stamp, use_teacher_forcing=True, s1_targets=s1)
    return s2_logits


def test_t11_the_validation_diagnostic_is_causal_with_dropout_off():
    _, model = ff.build_tiny_models(seed=4)
    model.eval()
    generator = torch.Generator().manual_seed(0)
    s1 = torch.randint(0, 16, (2, 40), generator=generator)
    s2 = torch.randint(0, 16, (2, 40), generator=generator)
    stamp = torch.zeros(2, 40, 5)
    stamp[..., 1] = torch.arange(40) % 24
    position = 20
    s1_changed, s2_changed = s1.clone(), s2.clone()
    s1_changed[:, position + 1] = (s1[:, position + 1] + 5) % 16
    s2_changed[:, position + 1] = (s2[:, position + 1] + 5) % 16
    # eval mode without the hook: position 20 reads position 21 (M:387), the leak exists
    leaky = teacher_forced_s2(model, s1, s2, stamp)
    leaky_changed = teacher_forced_s2(model, s1_changed, s2_changed, stamp)
    assert not torch.equal(leaky[:, position], leaky_changed[:, position])
    clean = CleanValidation([(s1, s2), (s1_changed, s2_changed)], model=model, val_seed=7)
    hooked = []
    for batch_s1, batch_s2 in clean:
        cross_attn = model.dep_layer.cross_attn
        assert cross_attn.training
        assert cross_attn.attn_dropout_p == 0.0 and cross_attn.resid_dropout.p == 0.0
        others = [
            name
            for name, module in model.named_modules()
            if module.training and not name.startswith("dep_layer.cross_attn")
        ]
        assert others == []  # every dropout stays off
        hooked.append(teacher_forced_s2(model, batch_s1, batch_s2, stamp))
    assert not model.dep_layer.cross_attn.training
    assert torch.equal(hooked[0][:, : position + 1], hooked[1][:, : position + 1])
    assert not torch.equal(hooked[0][:, position + 1], hooked[1][:, position + 1])
    # a cross-attention with dropout would apply it in train mode: refused
    model.dep_layer.cross_attn.resid_dropout.p = 0.1
    with pytest.raises(RunnerError, match="dropout"):
        CleanValidation([], model=model, val_seed=7)


def vendored_validation_loss(model, tokenizer, loader) -> float:
    """The vendored validation pass (FB:313-343) over ``loader``."""
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_x_stamp in loader:
            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            logits = model(token_seq_0[:, :-1], token_seq_1[:, :-1], batch_x_stamp[:, :-1, :])
            loss, _, _ = model.head.compute_loss(
                logits[0], logits[1], token_seq_0[:, 1:], token_seq_1[:, 1:]
            )
            total += loss.item()
            batches += 1
    return total / batches


def rng_snapshot():
    numpy_state = np.random.get_state()
    return torch.get_rng_state(), (numpy_state[0], numpy_state[1].copy()), random.getstate()


def same_rng(first, second) -> bool:
    return (
        torch.equal(first[0], second[0])
        and first[1][0] == second[1][0]
        and np.array_equal(first[1][1], second[1][1])
        and first[2] == second[2]
    )


def test_t12_the_same_diagnostic_loss_twice_and_every_rng_state_restored(vendored, tiny):
    record = tiny.export
    tokenizer, model = ff.load_saved(tiny.model_files)
    dataset = make_dataset(vendored, record.csv_path, record.train_ratio, data_type="val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    clean = CleanValidation(loader, model=model, val_seed=99)
    before = rng_snapshot()
    first = vendored_validation_loss(model, tokenizer, clean)
    assert same_rng(before, rng_snapshot())  # the loader's base-seed draw is undone too
    torch.rand(17)
    second = vendored_validation_loss(model, tokenizer, clean)
    assert first == second and clean.passes == 2
    # positive control: the vendored pass samples s1 from the global RNG and is random
    torch.manual_seed(1)
    plain_first = vendored_validation_loss(model, tokenizer, loader)
    torch.manual_seed(2)
    plain_second = vendored_validation_loss(model, tokenizer, loader)
    assert plain_first != plain_second


def test_t12_with_the_router_the_pass_draws_from_its_own_generator(vendored, tiny):
    record = tiny.export
    tokenizer, model = ff.load_saved(tiny.model_files)
    dataset = make_dataset(vendored, record.csv_path, record.train_ratio, data_type="val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    router = MultinomialRouter(seed=5)
    clean = CleanValidation(loader, model=model, val_seed=99, router=router)
    with router.installed():
        training_state = router.generator.get_state()
        before = rng_snapshot()
        first = vendored_validation_loss(model, tokenizer, clean)
        second = vendored_validation_loss(model, tokenizer, clean)
        assert torch.equal(router.generator.get_state(), training_state)
        assert same_rng(before, rng_snapshot())
    assert first == second and router.calls == 4  # two batches per pass


def test_t12_training_is_identical_with_and_without_the_diagnostic_and_nll4_tf(tmp_path):
    """The validation diagnostic and the NLL4-TF evaluations draw no random number that training
    uses: the trained weights equal those of a run without any validation at all."""
    recipe = ff.tiny_recipe(predictor_learning_rate=FAST_LR, patience=3)
    tiny = ff.tiny_run(tmp_path, recipe=recipe)
    hooked = train_in_process(
        tiny, tiny.request(), evaluate=nll_evaluator(tiny), early_stopping=True
    )
    assert hooked.outcome == "max_epochs"
    assert [entry["nll4_tf"] is not None for entry in hooked.monitor.epochs] == [True] * 4
    assert all(entry["vendored_val_loss"] is not None for entry in hooked.monitor.epochs[1:])
    bare = train_in_process(tiny, tiny.request(), without_validation=True)
    assert all(entry["vendored_val_loss"] is None for entry in bare.monitor.epochs)
    assert same_state(hooked.model.state_dict(), bare.model.state_dict())
    _, initial = ff.load_saved(tiny.model_files)
    assert not same_state(initial.state_dict(), bare.model.state_dict())


# -- T13: early stopping -------------------------------------------------------------------------


def scripted(values: dict[int, float], snapshots: dict):
    def evaluate(monitor, epoch):
        snapshots[epoch] = runner._snapshot(monitor.model)
        return values[epoch]

    return evaluate


@pytest.fixture
def recorded_schedulers(monkeypatch):
    """Every OneCycleLR the vendored loop builds (FB:253 looks the class up at call time)."""
    built = []

    class Recorded(torch.optim.lr_scheduler.OneCycleLR):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(torch.optim.lr_scheduler, "OneCycleLR", Recorded)
    return built


def test_t13_early_stopping_on_scripted_metrics(tmp_path, recorded_schedulers):
    recipe = ff.tiny_recipe(max_epochs=6, patience=2, predictor_learning_rate=FAST_LR)
    tiny = ff.tiny_run(tmp_path, recipe=recipe)
    snapshots: dict = {}
    values = {0: 1.0, 1: 5.0, 2: 4.0, 3: 4.5, 4: 4.2, 5: 3.0, 6: 2.0}
    run = train_in_process(
        tiny, tiny.request(), evaluate=scripted(values, snapshots), early_stopping=True
    )
    monitor = run.monitor
    assert run.outcome == "early_stop" and monitor.stopped_at == 4
    assert [entry["pseudo_epoch"] for entry in monitor.epochs] == [0, 1, 2, 3, 4]
    assert [entry["nll4_tf"] for entry in monitor.epochs] == [1.0, 5.0, 4.0, 4.5, 4.2]
    assert monitor.best_epoch == 2  # pseudo-epoch 0 (the shipped weights) is never a candidate
    assert same_state(monitor.best_state, snapshots[2])
    assert not same_state(monitor.best_state, snapshots[4])
    assert monitor.optimizer_steps == 4 * recipe.pseudo_epoch_steps  # pseudo-epoch 5 never ran
    (scheduler,) = recorded_schedulers
    assert scheduler.total_steps == recipe.max_epochs * recipe.pseudo_epoch_steps == 24
    assert monitor.onecycle_total_steps == 24
    assert scheduler.last_epoch == monitor.optimizer_steps <= scheduler.total_steps


def test_t13_ties_go_to_the_earlier_pseudo_epoch_and_the_span_is_never_passed(
    tmp_path, recorded_schedulers
):
    recipe = ff.tiny_recipe(max_epochs=3, patience=3)
    tiny = ff.tiny_run(tmp_path, recipe=recipe)
    snapshots: dict = {}
    run = train_in_process(
        tiny,
        tiny.request(),
        evaluate=scripted({0: 0.5, 1: 5.0, 2: 5.0, 3: 5.0}, snapshots),
        early_stopping=True,
    )
    assert run.outcome == "max_epochs" and run.monitor.best_epoch == 1
    assert same_state(run.monitor.best_state, snapshots[1])
    (scheduler,) = recorded_schedulers
    assert scheduler.last_epoch == scheduler.total_steps == run.monitor.optimizer_steps == 12


def test_selection_rule_unit():
    monitor = runner._Monitor(evaluate=lambda epoch: 0.0, early_stopping=True, patience=2)
    monitor.model = torch.nn.Linear(2, 2)
    monitor._select(1, 3.0, may_stop=True)
    monitor._select(2, 2.0, may_stop=True)
    monitor._select(3, 2.0, may_stop=True)  # a tie is no improvement
    with pytest.raises(runner.EarlyStop):
        monitor._select(4, 2.5, may_stop=True)
    assert (monitor.best_epoch, monitor.stopped_at) == (2, 4)
    with pytest.raises(RunnerError, match="NLL4-TF evaluation"):
        runner._Monitor(early_stopping=True)


# -- T14: the pseudo-epoch sampler ---------------------------------------------------------------


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(index)


def test_t14_pseudo_epochs_draw_without_replacement_from_their_own_generator():
    loader = runner.pseudo_epoch_loader(IndexDataset(100), batch_size=4, steps=8, seed=11)
    assert len(loader) == 8
    before = torch.get_rng_state()
    first = [batch.tolist() for batch in loader]
    second = [batch.tolist() for batch in loader]
    assert torch.equal(torch.get_rng_state(), before)  # the global RNG is never drawn from
    for epoch in (first, second):
        assert len(epoch) == 8 and all(len(batch) == 4 for batch in epoch)
        flat = [index for batch in epoch for index in batch]
        assert len(set(flat)) == 32  # no repeat inside a pseudo-epoch
    assert first != second  # the generator moves on
    torch.manual_seed(1234)
    torch.rand(100)
    again = runner.pseudo_epoch_loader(IndexDataset(100), batch_size=4, steps=8, seed=11)
    assert [batch.tolist() for batch in again] == first  # order from its own seed only
    other = runner.pseudo_epoch_loader(IndexDataset(100), batch_size=4, steps=8, seed=12)
    assert [batch.tolist() for batch in other] != first


def test_t14_the_gate_hands_the_vendored_loop_the_rebuilt_loader(tmp_path):
    tiny = ff.tiny_run(tmp_path)
    run = train_in_process(tiny, tiny.request(), without_validation=True)
    assert run.monitor.optimizer_steps == 3 * 4
    assert run.monitor.onecycle_total_steps == 12
    assert len(run.monitor.step_durations) == 12
    vendored_loop = train_in_process(
        tiny, tiny.request(wrappers=Wrappers(pseudo_epochs=False)), without_validation=True
    )
    assert vendored_loop.monitor.optimizer_steps == 3 * (ff.TINY_FIT_WINDOWS // 4)


def test_gate_forwards_length_and_batches():
    events = []

    class Recorder:
        def epoch_boundary(self):
            events.append("boundary")

        def before_batch(self):
            events.append("before")

        def batch_given(self):
            events.append("given")

        def after_last_batch(self):
            events.append("end")

    gate = TrainLoaderGate([1, 2], Recorder())
    assert len(gate) == 2
    assert list(gate) == [1, 2]
    assert events == ["boundary", "before", "given", "before", "given", "before", "end"]


# -- T15: the multinomial router -----------------------------------------------------------------


def test_t15_router_draws_on_the_cpu_from_its_generator_and_is_removed_after():
    probabilities = torch.softmax(
        torch.randn(50, 16, generator=torch.Generator().manual_seed(3)), -1
    )
    original = torch.multinomial
    router = MultinomialRouter(seed=123)
    before = torch.get_rng_state()
    with router.installed():
        assert torch.multinomial is router
        draws = torch.multinomial(probabilities.view(-1, 16), 1)  # K:271's call
        with pytest.raises(RunnerError, match="generator"):
            torch.multinomial(probabilities, 1, generator=torch.Generator())
        with pytest.raises(RunnerError, match="already replaced"), MultinomialRouter(1).installed():
            pass
        other = torch.Generator().manual_seed(77)
        with router.using(other):
            switched = torch.multinomial(probabilities, 1)
    assert torch.multinomial is original
    assert torch.equal(torch.get_rng_state(), before)
    assert draws.device == probabilities.device and draws.dtype == torch.int64
    expected = original(probabilities, 1, generator=torch.Generator().manual_seed(123))
    assert torch.equal(draws, expected)
    assert torch.equal(
        switched, original(probabilities, 1, generator=torch.Generator().manual_seed(77))
    )
    # inference untouched: the vendored sampler draws from the global RNG again
    sample = import_kronos().module.kronos.sample_from_logits
    torch.manual_seed(5)
    first = sample(torch.log(probabilities), top_k=0, top_p=1.0)
    torch.manual_seed(5)
    assert torch.equal(first, sample(torch.log(probabilities), top_k=0, top_p=1.0))
    assert router.calls == 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_t15_router_returns_indices_on_the_inputs_device_mps():
    probabilities = torch.softmax(torch.randn(8, 16), -1).to("mps")
    router = MultinomialRouter(seed=9)
    with router.installed():
        draws = torch.multinomial(probabilities, 1)
    assert draws.device.type == "mps"
    expected = torch.multinomial(probabilities.cpu(), 1, generator=torch.Generator().manual_seed(9))
    assert torch.equal(draws.cpu(), expected)


# -- T17, T18: activation checkpointing and the weight check ------------------------------------


def test_t17_activation_checkpointing_keeps_cpu_gradients_bit_identical():
    _, model = ff.build_tiny_models(seed=2)
    wrapped = copy.deepcopy(model)
    runner.enable_activation_checkpointing(wrapped)
    with pytest.raises(RunnerError, match="already enabled"):
        runner.enable_activation_checkpointing(wrapped)
    generator = torch.Generator().manual_seed(1)
    s1 = torch.randint(0, 16, (3, 41), generator=generator)
    s2 = torch.randint(0, 16, (3, 41), generator=generator)
    stamp = torch.zeros(3, 41, 5)
    stamp[..., 1] = torch.arange(41) % 24
    gradients, losses = [], []
    for module in (model, wrapped):
        module.train()
        torch.manual_seed(21)  # dropout and the s1 draw consume the same stream in both
        logits = module(s1[:, :-1], s2[:, :-1], stamp[:, :-1])
        loss, _, _ = module.head.compute_loss(logits[0], logits[1], s1[:, 1:], s2[:, 1:])
        loss.backward()
        losses.append(loss.detach())
        gradients.append({name: p.grad.clone() for name, p in module.named_parameters()})
    assert torch.equal(losses[0], losses[1])
    assert same_state(gradients[0], gradients[1])


def test_t18_the_bitwise_weight_check_refuses_a_perturbed_tensor(tmp_path):
    files = ff.save_tiny_models(tmp_path)
    tokenizer, model = ff.load_saved(files)
    weights = files.predictor_dir / "model.safetensors"
    assert runner.state_mismatches(model, weights) == []
    assert runner.state_mismatches(tokenizer, files.tokenizer_dir / "model.safetensors") == []
    name = "head.proj_s1.bias"
    with torch.no_grad():
        tensor = model.state_dict()[name]
        tensor[0] = torch.nextafter(tensor[0], torch.tensor(math.inf))  # one ulp
    assert runner.state_mismatches(model, weights) == [f"{name}: values differ"]
    problems = runner.state_mismatches(tokenizer, weights)
    assert any("in the file, not in the module" in problem for problem in problems)
    assert any("in the module, not in the file" in problem for problem in problems)


def test_t18_the_child_refuses_weights_the_vendored_loader_only_partly_loads(tmp_path):
    """The vendored ``from_pretrained`` loads with ``strict=False`` (HM:785): a file without one
    tensor loads silently with that tensor at its random initial value; the check refuses it.
    The file is the one ``run.json`` names, so its sha256 verifies."""
    from safetensors.torch import load_file, save_file

    tiny = ff.tiny_run(tmp_path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_bytes(
        (tiny.model_files.predictor_dir / "config.json").read_bytes()
    )
    state = load_file(str(tiny.model_files.predictor_dir / "model.safetensors"))
    del state["head.proj_s1.bias"]
    save_file(state, str(partial / "model.safetensors"), metadata={"format": "pt"})
    files = train.ModelFiles(
        predictor_dir=partial.absolute(),
        tokenizer_dir=tiny.model_files.tokenizer_dir,
        predictor_sha256={
            name: sha256_file(partial / name) for name in ("config.json", "model.safetensors")
        },
        tokenizer_sha256=tiny.model_files.tokenizer_sha256,
    )
    # a timing run: in a train run the strict load of the NLL4-TF copy refuses the file first
    request = tiny.request(
        mode="timing",
        model_files=files,
        wrappers=Wrappers(early_stopping=False),
        timing=TimingSpec(warmup_steps=1, timed_steps=1),
    )
    result = train.run(request, root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "failed" and result.exit_code == runner.EXIT_FAILED
    assert result.document["error"]["type"] == "RunnerError"
    assert "differ from the verified files" in result.document["error"]["message"]
    assert "head.proj_s1.bias" in result.document["error"]["message"]
    assert result.document["selected"] is None


# -- T19: memory guard and deadline --------------------------------------------------------------


def test_t19_memory_guard_on_simulated_readings():
    readings = iter([MemoryReading(1, 100), MemoryReading(2, 300), MemoryReading(4, 300)])
    guard = MemoryGuard(lambda: next(readings), every_steps=50, max_swap_growth_bytes=1000)
    guard.start()
    guard.check(1)  # not a multiple of every_steps: no reading
    guard.check(50)
    with pytest.raises(MemoryAbort, match="pressure level 4"):
        guard.check(100)
    assert guard.summary() == {
        "swap_start_bytes": 100,
        "swap_growth_bytes": 200,
        "max_pressure_level": 4,
        "checks": 3,
    }
    growth = iter([MemoryReading(1, 0), MemoryReading(1, 1000), MemoryReading(1, 1001)])
    guard = MemoryGuard(lambda: next(growth), every_steps=1, max_swap_growth_bytes=1000)
    guard.check(0)  # the first reading is the baseline
    guard.check(1)  # exactly the limit is not above it
    with pytest.raises(MemoryAbort, match="swap use grew by 1001"):
        guard.check(2)
    with pytest.raises(RunnerError, match="unavailable"):
        MemoryGuard(lambda: None).start()


def test_t19_swap_parsing_and_the_real_readings():
    text = "total = 9216.00M  used = 7962.44M  free = 1253.56M  (encrypted)"
    assert runner.parse_swap_used(text) == round(7962.44 * 2**20)
    assert runner.parse_swap_used("total = 2.00G  used = 1.50G  free = 0.50G") == 3 * 2**29
    assert runner.parse_swap_used("vm.swapusage: unavailable") is None
    reading = runner.read_memory()
    if sys.platform != "darwin":
        assert reading is None
        return
    assert reading is not None
    assert reading.pressure_level in (1, 2, 4) and reading.swap_used_bytes >= 0


def test_t19_a_passed_deadline_is_a_time_abort(tiny):
    request = tiny.request(deadline_seconds=0.05)
    result = train.run(request, root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "time_abort" and result.exit_code == runner.EXIT_TIME_ABORT
    assert result.document["error"]["type"] == "TimeAbort"
    assert not result.timed_out


def test_t19_a_memory_abort_is_its_own_status_and_exit_code(tiny, monkeypatch, restore_signals):
    readings = iter([MemoryReading(1, 0), MemoryReading(4, 0)])
    monkeypatch.setattr(runner, "read_memory", lambda: next(readings))
    request = tiny.request(compute=Compute(threads=2, memory_guard=True))
    run_dir = ff.prepare_run_dir(tiny, request)
    assert runner.run(run_dir) == runner.EXIT_MEMORY_ABORT
    document = ff.read_result(run_dir)
    assert document["status"] == "memory_abort" and document["error"]["type"] == "MemoryAbort"
    assert document["memory"]["max_pressure_level"] == 4 and document["memory"]["checks"] == 2
    assert document["steps"] == 0


# -- modes: train, timing, preflight -------------------------------------------------------------


def test_train_mode_record(tiny):
    result = train.run(tiny.request(), root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "completed", ff.child_log(result.run_dir)
    document = result.document
    assert sorted(document) == sorted(runner._blank_result(result.run_dir))
    assert document["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert document["stop_reason"] in ("max_epochs", "early_stop") and document["finite"]
    assert document["device"] == "cpu" and document["threads"] == 2 and document["deterministic"]
    assert document["torch_version"] == torch.__version__
    assert document["wrappers"] == dict.fromkeys(runner.WRAPPERS, True)
    assert document["dataset"] == {
        "class": "CustomKlineDataset",
        "fit_windows": ff.TINY_FIT_WINDOWS,
        "val_windows": ff.TINY_VAL_WINDOWS,
    }
    assert document["onecycle_total_steps"] == 12 and document["steps"] <= 12
    assert document["epochs"][0]["pseudo_epoch"] == 0
    assert document["epochs"][0]["train_loss"] is None and document["epochs"][0]["nll4_tf"] > 0
    assert document["selected"]["source"] == "nll4_tf" and document["selected"]["dir"] == "selected"
    assert document["selected_pseudo_epoch"] >= 1
    for item in document["selected"]["files"]:
        assert sha256_file(result.run_dir / "selected" / item["name"]) == item["sha256"]
    assert document["nll"]["decisions"] == 129 and document["nll"]["batch_size"] == 16
    assert len(document["nll"]["evaluation_seconds"]) == len(document["epochs"])
    inputs = document["inputs"]
    assert inputs["train_csv_sha256"] == tiny.export.csv_sha256
    assert inputs["nll_inputs_sha256"] == tiny.export.nll_inputs_sha256
    assert inputs["model_files"]["predictor"] == dict(tiny.model_files.predictor_sha256)
    text = (result.run_dir / runner.RESULT_JSON).read_text(encoding="ascii")
    assert str(Path.home()) not in text and str(tiny.root) not in text


def test_timing_mode_measures_steps_only(tmp_path):
    recipe = ff.tiny_recipe(pseudo_epoch_steps=64)
    tiny = ff.tiny_run(tmp_path, recipe=recipe)
    request = tiny.request(
        mode="timing",
        wrappers=Wrappers(early_stopping=False),
        timing=TimingSpec(warmup_steps=2, timed_steps=3),
    )
    result = train.run(request, root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "completed", ff.child_log(result.run_dir)
    document = result.document
    assert document["stop_reason"] == "timing_done"
    assert document["steps"] == 5 and document["onecycle_total_steps"] == 192
    assert len(document["step_times"]) == 3 and document["step_offsets"][0] == 0.0
    assert document["step_offsets"] == sorted(document["step_offsets"])
    assert document["step_seconds"]["count"] == 3
    assert document["epochs"] == [] and document["selected"] is None and document["nll"] is None


def test_preflight_mode_on_the_cpu(tiny):
    request = tiny.request(mode="preflight", preflight_checks=runner.PREFLIGHT_CHECKS)
    result = train.run(request, root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "completed", ff.child_log(result.run_dir)
    assert result.document["stop_reason"] == "preflight_done"
    checks = result.document["preflight"]
    assert sorted(checks) == sorted(runner.PREFLIGHT_CHECKS)
    for name, outcome in checks.items():
        assert outcome["passed"] is True, (name, outcome)
    multinomial = checks["multinomial"]
    assert multinomial["rows"] == 4 * 512 and multinomial["categories"] == 16
    assert multinomial["device"]["zero_probability_draws"] == 0
    assert multinomial["cpu"]["p_value"] >= runner.PREFLIGHT_ALPHA
    assert checks["parity"]["max_relative_logit_difference"] == 0.0
    assert checks["steps"]["relative_difference"] == 0.0 and checks["steps"]["steps"] == 20
    assert checks["deterministic"] == {"passed": True, "refused": False, "op": None}
    assert checks["checkpoint_gradients"]["mismatches"] == []


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_device_path_smoke(tmp_path):
    """The device wrapper on MPS: a timing run and the preflight complete and record their
    measurements (whether MPS passes is the measurement step's question, not this test's)."""
    recipe = ff.tiny_recipe(pseudo_epoch_steps=64)
    tiny = ff.tiny_run(tmp_path, recipe=recipe)
    compute = Compute(device="mps", threads=2, deterministic=False, memory_guard=False)
    timing = tiny.request(
        mode="timing",
        compute=compute,
        wrappers=Wrappers(early_stopping=False),
        timing=TimingSpec(warmup_steps=1, timed_steps=2),
    )
    result = train.run(timing, root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "completed", ff.child_log(result.run_dir)
    assert result.document["device"] == "mps" and result.document["steps"] == 3
    preflight = tiny.request(
        mode="preflight",
        compute=compute,
        preflight_checks=("multinomial", "parity", "deterministic"),
    )
    result = train.run(preflight, root=tiny.root, timeout_seconds=ff.CHILD_TIMEOUT_SECONDS)
    assert result.status == "completed", ff.child_log(result.run_dir)
    checks = result.document["preflight"]
    for outcome in checks.values():
        assert isinstance(outcome["passed"], bool) and "error" not in outcome, outcome
    # the measurements are numbers: MPS tensors reach the CPU before any float64 conversion
    assert checks["multinomial"]["device"]["dof"] > 0
    assert checks["multinomial"]["device"]["p_value"] is not None
    assert checks["parity"]["max_relative_logit_difference"] is not None
    assert checks["parity"]["gradient_cosine"] is not None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_cpu_float64_of_an_mps_tensor_is_exact():
    tensor = torch.randn(64, 16).to("mps")
    assert torch.equal(runner._cpu_float64(tensor), tensor.cpu().double())


# -- run.json, inputs and the result ------------------------------------------------------------


def rewrite(run_dir: Path, change) -> None:
    path = run_dir / runner.RUN_JSON
    document = json.loads(path.read_text(encoding="ascii"))
    change(document)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="ascii")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(extra=1), "unknown keys"),
        (lambda d: d.pop("blinded"), "missing keys"),
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d["seeds"].update(sampler_seed=d["seeds"]["sampler_seed"] + 1), "seeds"),
        (lambda d: d.update(recipe_id="0" * 64), "recipe_id"),
        (lambda d: d["fold"].update(run_label="other"), "run_label"),
        (
            lambda d: d.update(timing={"warmup_steps": 1, "timed_steps": 1, "timed_seconds": None}),
            "timing",
        ),
        (lambda d: d["compute"].update(device="cuda"), "compute.device"),
        (lambda d: d["wrappers"].update(router="yes"), "boolean"),
        (lambda d: d["recipe"].update(batch_size=0), "invalid recipe"),
        (  # an export with a gap and the vendored dataset (design 1.6)
            lambda d: (d["windows"].update(gap_aware=True), d["wrappers"].update(gap_aware=False)),
            "the export has a gap: a run on it needs the gap_aware wrapper",
        ),
    ],
)
def test_run_json_is_validated(tiny, change, message):
    run_dir = ff.prepare_run_dir(tiny, tiny.request())
    assert runner._read_run_spec(run_dir).mode == "train"
    rewrite(run_dir, change)
    with pytest.raises(RunnerError, match=message):
        runner._read_run_spec(run_dir)


def test_inputs_are_verified_against_run_json(tiny):
    run_dir = ff.prepare_run_dir(tiny, tiny.request())
    spec = runner._read_run_spec(run_dir)
    inputs = runner._verify_inputs(spec)
    assert inputs["run_yaml_sha256"] == sha256_file(run_dir / runner.RUN_YAML)
    yaml_path = run_dir / runner.RUN_YAML
    document = yaml.safe_load(yaml_path.read_text(encoding="ascii"))
    document["training"]["batch_size"] = 8
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="ascii")
    rewrite(run_dir, lambda d: d["yaml"].update(sha256=sha256_file(yaml_path)))
    with pytest.raises(RunnerError, match="is not the run YAML"):
        runner._verify_inputs(runner._read_run_spec(run_dir))


def test_the_result_is_written_even_when_the_run_cannot_start(tiny, restore_signals):
    run_dir = ff.prepare_run_dir(tiny, tiny.request())
    rewrite(run_dir, lambda d: d.update(mode="fly"))
    assert runner.run(run_dir) == runner.EXIT_FAILED
    document = ff.read_result(run_dir)
    assert document["status"] == "failed" and document["error"]["type"] == "RunnerError"
    assert document["run_id"] == run_dir.name and document["finished_at"] is not None
    empty = tiny.root / "empty"
    empty.mkdir()
    assert runner.run(empty) == runner.EXIT_FAILED
    assert ff.read_result(empty)["error"]["message"].startswith("cannot read run.json")


def test_json_safe_marks_non_finite_values():
    document, replaced = runner._json_safe({"a": [1.0, float("nan")], "b": {"c": float("inf")}})
    assert replaced and document == {"a": [1.0, None], "b": {"c": None}}
    assert runner._json_safe({"a": 1.0}) == ({"a": 1.0}, False)


def test_the_vendored_tree_is_still_clean():
    """Nothing in these tests wrote bytecode or anything else into third_party/kronos (D-014)."""
    assert verify_vendored_tree(strict=True).ok
