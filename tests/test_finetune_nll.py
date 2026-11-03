"""NLL4-TF (``finetune.nll``; ``docs/RETRAINING.md`` 3.3, test T16): the teacher-forced
next-4-candle NLL in the exact inference geometry, on tiny random Kronos models.

T16: with the vendored ``sample_from_logits`` patched to return the true tokens, the logits that
``auto_regressive_inference`` computes (through ``KronosPredictor.predict_batch``, the forecast
path) equal NLL4-TF's bit for bit at every step; and NLL4-TF gives the same values in two
processes with the same thread count. The inputs file is the exporter's (``nll_inputs.npz``).
"""

from __future__ import annotations

import io
import json
import random
import subprocess
import zipfile
from pathlib import Path

import finetune_fakes as ff
import numpy as np
import pandas as pd
import pytest
import torch

from kronos_agent.data.timeutil import HOUR_MS, to_ms
from kronos_agent.data.windows import naive_timestamps
from kronos_agent.finetune.export_csv import nll_inputs_bytes
from kronos_agent.finetune.nll import (
    NLL_CONTEXT,
    NLL_ROWS,
    NLL_STEPS,
    NllError,
    NllInputs,
    load_nll_inputs,
    nll4_tf,
    normalise,
    teacher_forced_logits,
    time_stamps,
)
from kronos_agent.kronos.features import FEATURE_COLUMNS
from kronos_agent.kronos.vendored import import_kronos


@pytest.fixture(scope="module")
def tiny_export(tmp_path_factory):
    directory = tmp_path_factory.mktemp("nll")
    recipe = ff.tiny_recipe()
    _, record = ff.tiny_export(directory / "export", recipe)
    files = ff.save_tiny_models(directory / "models", seed=3)
    return record, files


@pytest.fixture(scope="module")
def inputs(tiny_export) -> NllInputs:
    record, _ = tiny_export
    return load_nll_inputs(record.nll_inputs_path, expected_sha256=record.nll_inputs_sha256)


@pytest.fixture(scope="module")
def models(tiny_export):
    _, files = tiny_export
    return ff.load_saved(files)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int32)


def _predictor_frame(values: np.ndarray) -> pd.DataFrame:
    """The context as the forecast path hands it to the predictor: one float64 column per
    channel (``data.windows.context_frame`` builds its frame the same way)."""
    return pd.DataFrame({name: values[:, index] for index, name in enumerate(FEATURE_COLUMNS)})


def test_inputs_are_the_exporters_decisions(tiny_export, inputs):
    record, _ = tiny_export
    assert inputs.decisions == len(record.nll_decisions_ms) == 129
    assert tuple(int(value) for value in inputs.decision_ms) == record.nll_decisions_ms
    assert inputs.values.shape == (129, NLL_ROWS, 6)
    assert not inputs.values.flags.writeable and not inputs.decision_ms.flags.writeable


def test_normalise_reproduces_the_predictor_input_bit_for_bit(inputs):
    """Rows 0..511 equal ``predict_batch``'s normalised context (K:620-631), whose float32
    statistics are reduced on a Fortran-ordered array; rows 512..515 use the same statistics."""
    normalised = normalise(inputs.values)
    assert normalised.dtype == np.float32 and normalised.shape == inputs.values.shape
    for index in (0, 57, inputs.decisions - 1):
        frame = _predictor_frame(inputs.values[index, :NLL_CONTEXT]).copy()
        x = frame[list(FEATURE_COLUMNS)].values.astype(np.float32)
        assert x.flags["F_CONTIGUOUS"]  # the layout the predictor reduces
        mean, std = np.mean(x, axis=0), np.std(x, axis=0)
        expected = np.clip((x - mean) / (std + 1e-5), -5, 5)
        assert np.array_equal(
            normalised[index, :NLL_CONTEXT].view(np.uint32), expected.view(np.uint32)
        )
        targets = inputs.values[index, NLL_CONTEXT:].astype(np.float32)
        expected_targets = np.clip((targets - mean) / (std + 1e-5), -5, 5)
        assert np.array_equal(
            normalised[index, NLL_CONTEXT:].view(np.uint32), expected_targets.view(np.uint32)
        )


def test_time_stamps_are_the_vendored_time_features_of_the_open_times():
    decision = to_ms(pd.Timestamp("2024-06-09T16:00Z").to_pydatetime())
    stamps = time_stamps(np.array([decision], dtype=np.int64))
    assert stamps.shape == (1, NLL_ROWS, 5) and stamps.dtype == np.float32
    # row 512 opens at D (minute, hour, weekday, day, month): a Sunday
    assert stamps[0, NLL_CONTEXT].tolist() == [0.0, 16.0, 6.0, 9.0, 6.0]
    # row 0 opens 512 h (21 days and 8 hours) earlier: 2024-05-19T08:00Z, a Sunday as well
    assert stamps[0, 0].tolist() == [0.0, 8.0, 6.0, 19.0, 5.0]
    expected = import_kronos().module.kronos.calc_time_stamps(
        naive_timestamps(decision + (np.arange(NLL_ROWS) - NLL_CONTEXT) * HOUR_MS)
    )
    assert np.array_equal(stamps[0], expected.values.astype(np.float32))


def _true_token_sampler(sequence: list[torch.Tensor], seen: list[torch.Tensor]):
    """A stand-in for the vendored ``sample_from_logits`` (K:373-386): records the logits it is
    given and returns the next true token (s1 and s2 alternate, as the loop calls them)."""

    def sampler(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
        seen.append(logits.detach().clone())
        return sequence[len(seen) - 1].reshape(1, 1).clone()

    return sampler


def test_logits_equal_auto_regressive_inference_with_the_true_tokens(models, inputs, monkeypatch):
    """T16: the predictor's own loop, its sampler patched to return the true tokens, computes
    the same logits as NLL4-TF at every step, bit for bit (one decision per batch)."""
    tokenizer, model = models
    api = import_kronos()
    subset = inputs.head(3)
    forced = teacher_forced_logits(tokenizer, model, subset, batch_size=1)
    assert forced.s1_logits.shape == (3, NLL_STEPS, 16)
    assert forced.s2_logits.shape == (3, NLL_STEPS, 16)
    predictor = api.KronosPredictor(model, tokenizer, device="cpu", max_context=512, clip=5)
    kronos_module = api.module.kronos
    for index in range(subset.decisions):
        recorded: list[torch.Tensor] = []
        true_sequence = [
            token
            for step in range(NLL_STEPS)
            for token in (forced.s1_targets[index, step], forced.s2_targets[index, step])
        ]

        monkeypatch.setattr(
            kronos_module, "sample_from_logits", _true_token_sampler(true_sequence, recorded)
        )
        decision = int(subset.decision_ms[index])
        opens = decision + (np.arange(NLL_ROWS, dtype=np.int64) - NLL_CONTEXT) * HOUR_MS
        predictor.predict_batch(
            [_predictor_frame(subset.values[index, :NLL_CONTEXT])],
            [naive_timestamps(opens[:NLL_CONTEXT])],
            [naive_timestamps(opens[NLL_CONTEXT:])],
            pred_len=NLL_STEPS,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        assert len(recorded) == 2 * NLL_STEPS
        for step in range(NLL_STEPS):
            s1, s2 = recorded[2 * step], recorded[2 * step + 1]
            assert torch.equal(_bits(s1[0]), _bits(forced.s1_logits[index, step])), (index, step)
            assert torch.equal(_bits(s2[0]), _bits(forced.s2_logits[index, step])), (index, step)


def test_value_is_the_mean_of_s1_plus_s2_over_decisions_and_steps(models, inputs):
    tokenizer, model = models
    subset = inputs.head(20)
    result = nll4_tf(tokenizer, model, subset, batch_size=8)
    assert result.decisions == 20 and result.batch_size == 8
    assert result.threads == torch.get_num_threads()
    assert result.s1.shape == result.s2.shape == (20, NLL_STEPS)
    assert result.s1.dtype == np.float64 and (result.s1 > 0).all() and (result.s2 > 0).all()
    assert np.array_equal(result.per_decision, (result.s1 + result.s2).mean(axis=1))
    assert result.value == pytest.approx(float((result.s1 + result.s2).mean()), rel=1e-15)
    forced = teacher_forced_logits(tokenizer, model, subset, batch_size=8)
    log_probabilities = torch.log_softmax(forced.s1_logits.double(), dim=-1)
    first = -float(log_probabilities[0, 0, int(forced.s1_targets[0, 0])])
    assert result.s1[0, 0] == first


def test_deterministic_and_draws_no_random_number(models, inputs):
    tokenizer, model = models
    subset = inputs.head(10)
    torch_state, numpy_state, python_state = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    )
    first = nll4_tf(tokenizer, model, subset, batch_size=4)
    second = nll4_tf(tokenizer, model, subset, batch_size=4)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0] and np.array_equal(numpy_after[1], numpy_state[1])
    assert np.array_equal(first.per_decision, second.per_decision) and first.value == second.value
    single = nll4_tf(tokenizer, model, subset, batch_size=1)
    assert np.allclose(single.per_decision, first.per_decision, rtol=1e-5, atol=1e-5)


NLL_IN_CHILD = """
import hashlib, json, sys
from kronos_agent import runtime
runtime.configure_runtime(2)
from kronos_agent.finetune.nll import load_nll_inputs, nll4_tf
from kronos_agent.kronos.vendored import import_kronos
api = import_kronos()
tokenizer = api.KronosTokenizer.from_pretrained(sys.argv[1], strict=True)
model = api.Kronos.from_pretrained(sys.argv[2], strict=True)
for module in (tokenizer, model):
    module.eval()
    module.requires_grad_(False)
inputs = load_nll_inputs(sys.argv[3]).head(16)
result = nll4_tf(tokenizer, model, inputs, batch_size=4)
print(json.dumps({"value": result.value.hex(), "threads": result.threads,
                  "sha256": hashlib.sha256(result.per_decision.tobytes()).hexdigest()}))
"""


def test_same_values_in_two_processes_with_the_same_threads(tiny_export, models, inputs):
    """T16: NLL4-TF is a function of the weights, the inputs and the thread count."""
    import hashlib

    record, files = tiny_export
    tokenizer, model = models
    here = nll4_tf(tokenizer, model, inputs.head(16), batch_size=4)
    assert here.threads == 2
    outcomes = []
    for _ in range(2):
        completed = subprocess.run(
            ff.python_command(
                NLL_IN_CHILD,
                str(files.tokenizer_dir),
                str(files.predictor_dir),
                str(record.nll_inputs_path),
            ),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        outcomes.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0]["threads"] == 2
    assert outcomes[0]["value"] == here.value.hex()
    assert outcomes[0]["sha256"] == hashlib.sha256(here.per_decision.tobytes()).hexdigest()


def test_models_must_be_in_eval_mode_on_the_cpu(models, inputs):
    tokenizer, model = models
    subset = inputs.head(1)
    try:
        model.dep_layer.train()
        with pytest.raises(NllError, match="eval mode"):
            nll4_tf(tokenizer, model, subset)
    finally:
        model.eval()
    _, fresh = ff.build_tiny_models(seed=5)
    fresh = ff.eval_copy(fresh).to("meta")
    with pytest.raises(NllError, match="on the CPU"):
        nll4_tf(tokenizer, fresh, subset)
    with pytest.raises(NllError, match="batch_size"):
        nll4_tf(tokenizer, model, subset, batch_size=0)


def test_load_refuses_another_file_or_a_changed_byte(tiny_export, tmp_path):
    record, _ = tiny_export
    data = Path(record.nll_inputs_path).read_bytes()
    with pytest.raises(NllError, match="sha256 mismatch"):
        load_nll_inputs(record.nll_inputs_path, expected_sha256="0" * 64)
    changed = bytearray(data)
    changed[len(changed) // 2] ^= 0x01
    path = tmp_path / "changed.npz"
    path.write_bytes(bytes(changed))
    with pytest.raises(NllError, match="sha256 mismatch"):
        load_nll_inputs(path, expected_sha256=record.nll_inputs_sha256)


def _npz(members: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, array in members.items():
            with archive.open(f"{name}.npy", "w") as handle:
                np.lib.format.write_array(handle, array, allow_pickle=False)
    return buffer.getvalue()


def test_load_refuses_malformed_inputs(inputs, tmp_path):
    decisions = np.asarray(inputs.decision_ms[:2])
    values = np.asarray(inputs.values[:2])
    valid = nll_inputs_bytes(decisions, values)
    path = tmp_path / "inputs.npz"
    path.write_bytes(valid)
    assert load_nll_inputs(path).decisions == 2
    with zipfile.ZipFile(io.BytesIO(valid)) as archive:
        meta = np.lib.format.read_array(io.BytesIO(archive.read("meta.npy")))
    reordered = _npz({"values": values, "decision_ms": decisions, "meta": meta})
    path.write_bytes(reordered)
    with pytest.raises(NllError, match="exactly"):
        load_nll_inputs(path)
    document = json.loads(str(meta[()]))
    document["decisions"] = 3
    wrong_meta = np.array(json.dumps(document, sort_keys=True, separators=(",", ":")))
    path.write_bytes(_npz({"decision_ms": decisions, "values": values, "meta": wrong_meta}))
    with pytest.raises(NllError, match="decisions"):
        load_nll_inputs(path)
    path.write_bytes(_npz({"decision_ms": decisions, "values": values[:, :500], "meta": meta}))
    with pytest.raises(NllError, match="shape"):
        load_nll_inputs(path)


def test_inputs_validation():
    decisions = np.array([0, 4 * HOUR_MS], dtype=np.int64)
    values = np.ones((2, NLL_ROWS, 6))
    assert NllInputs(decisions, values).decisions == 2
    with pytest.raises(NllError, match="strictly increasing"):
        NllInputs(decisions[::-1].copy(), values)
    with pytest.raises(NllError, match="exact hour"):
        NllInputs(decisions + 1, values)
    with pytest.raises(NllError, match="int64"):
        NllInputs(decisions.astype(np.int32), values)
    bad = values.copy()
    bad[1, 7, 3] = np.nan
    with pytest.raises(NllError, match="finite"):
        NllInputs(decisions, bad)
    with pytest.raises(NllError, match="shape"):
        NllInputs(decisions, values[:, :NLL_CONTEXT])
    with pytest.raises(NllError, match="count"):
        NllInputs(decisions, values).head(3)
