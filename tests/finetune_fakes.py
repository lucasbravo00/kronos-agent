"""Shared fixtures of the step-6 training tests: tiny random Kronos models, tiny synthetic folds
exported by the real exporter, and helpers that start the training child.

The tiny models follow the shapes checked in the interfaces of phase 1 against the vendored
constructors (``group_size`` divides ``s1_bits + s2_bits``; the tokenizer builds
``n_enc_layers - 1`` and ``n_dec_layers - 1`` blocks; the dependency layer has 4 heads). The
predictor keeps Kronos-small's dropouts (``ffn 0.25, attn 0.1, resid 0.25, token 0.1``) so that
every RNG path of training runs. Both are built under a fixed ``torch.manual_seed`` and saved
with ``save_pretrained``. A tiny fold is 1,060 synthetic hourly candles from 2020-01-01T00:00Z
(``export_csv.synthetic_frame``): 544 fit rows (32 windows of 513 rows) and an inner validation
of 516 rows (4 windows, 129 NLL4-TF decisions), exported by ``export_csv.export_frame`` into a
temporary directory. A tiny recipe draws 4 batches of 4 windows per pseudo-epoch.

Nothing here touches the repository's ``data/``, ``reports/`` or ``weights/``: every path is a
pytest temporary directory, and the child's environment comes from
``finetune.train.child_environment`` (an allow-list).

The registry and batch tests use stand-in checkpoints: a valid fold payload of a Kronos-small
checkpoint (:func:`checkpoint_payload`) installed in a temporary fine-tune root with the
registered ``config.json`` and a few bytes for ``model.safetensors`` (:func:`install_stand_in`;
nothing can load it, the registry and the batch preparation read only its entry), and stand-in
models that forecast a seeded random walk with the hashes, id and data end of such an entry
(:func:`walk_models`).
"""

from __future__ import annotations

import copy
import functools
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kronos_agent.data.partition import TRAINING_START
from kronos_agent.data.timeutil import HOUR_MS, to_ms, utc_text
from kronos_agent.hashing import sha256_file

TINY_TOKENIZER_CONFIG: dict[str, Any] = {
    "d_in": 6,
    "d_model": 16,
    "n_heads": 2,
    "ff_dim": 32,
    "n_enc_layers": 2,
    "n_dec_layers": 2,
    "ffn_dropout_p": 0.0,
    "attn_dropout_p": 0.0,
    "resid_dropout_p": 0.0,
    "s1_bits": 4,
    "s2_bits": 4,
    "beta": 0.05,
    "gamma0": 1.0,
    "gamma": 1.1,
    "zeta": 0.05,
    "group_size": 4,
}
TINY_MODEL_CONFIG: dict[str, Any] = {
    "s1_bits": 4,
    "s2_bits": 4,
    "n_layers": 1,
    "d_model": 16,
    "n_heads": 2,
    "ff_dim": 32,
    "ffn_dropout_p": 0.25,
    "attn_dropout_p": 0.1,
    "resid_dropout_p": 0.25,
    "token_dropout_p": 0.1,
    "learn_te": True,
}
TINY_CSV_START_MS = to_ms(TRAINING_START)
TINY_ROWS = 1060
TINY_INNER_VALIDATION_HOURS = 516
TINY_FIT_WINDOWS = 32
TINY_VAL_WINDOWS = 4
CHILD_TIMEOUT_SECONDS = 300.0


def tiny_recipe(**overrides: Any) -> Any:
    """A valid test-scale recipe: batch 4, pseudo-epochs of 4 steps, 3 of them, patience 2."""
    from kronos_agent.finetune.recipe import Recipe

    values: dict[str, Any] = {
        "name": "tiny",
        "batch_size": 4,
        "pseudo_epoch_steps": 4,
        "max_epochs": 3,
        "patience": 2,
        "inner_validation_hours": TINY_INNER_VALIDATION_HOURS,
        "min_nll_decisions": 1,
        "log_interval": 1,
    }
    values.update(overrides)
    return Recipe(**values)


def build_tiny_models(seed: int = 0) -> tuple[Any, Any]:
    """A tiny tokenizer and predictor built under ``torch.manual_seed(seed)`` (train mode)."""
    import torch

    from kronos_agent.kronos.vendored import import_kronos

    api = import_kronos()
    torch.manual_seed(seed)
    tokenizer = api.KronosTokenizer(**TINY_TOKENIZER_CONFIG)
    model = api.Kronos(**TINY_MODEL_CONFIG)
    return tokenizer, model


def eval_copy(module: Any) -> Any:
    """``module`` in eval mode without gradients, on the CPU (what NLL4-TF requires)."""
    module.eval()
    module.requires_grad_(False)
    return module.to("cpu")


def save_tiny_models(directory: Path, *, seed: int = 0) -> Any:
    """``save_pretrained`` of the tiny models into ``directory/tokenizer`` and
    ``directory/model``; returns their ``finetune.train.ModelFiles``."""
    from kronos_agent.finetune.train import ModelFiles

    tokenizer, model = build_tiny_models(seed)
    tokenizer_dir = (Path(directory) / "tokenizer").absolute()
    model_dir = (Path(directory) / "model").absolute()
    tokenizer.save_pretrained(str(tokenizer_dir))
    model.save_pretrained(str(model_dir))
    names = ("config.json", "model.safetensors")
    return ModelFiles(
        predictor_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        predictor_sha256={name: sha256_file(model_dir / name) for name in names},
        tokenizer_sha256={name: sha256_file(tokenizer_dir / name) for name in names},
    )


def load_saved(files: Any) -> tuple[Any, Any]:
    """The tokenizer and predictor of ``files`` (``ModelFiles``), loaded strictly, in eval mode."""
    from kronos_agent.kronos.vendored import import_kronos

    api = import_kronos()
    tokenizer = api.KronosTokenizer.from_pretrained(str(files.tokenizer_dir), strict=True)
    model = api.Kronos.from_pretrained(str(files.predictor_dir), strict=True)
    return eval_copy(tokenizer), eval_copy(model)


def tiny_fold(recipe: Any, *, rows: int = TINY_ROWS, csv_start_ms: int = TINY_CSV_START_MS) -> Any:
    from kronos_agent.finetune.folds import synthetic_fold

    return synthetic_fold(csv_start_ms, csv_start_ms + rows * HOUR_MS, recipe)


def tiny_frame(fold: Any, *, seed: int = 0, missing_hours: tuple[int, ...] = ()) -> Any:
    """The fold's synthetic candles (store-like strings); ``missing_hours`` (offsets from the
    CSV start) are removed, never filled."""
    from kronos_agent.finetune.export_csv import synthetic_frame

    rows = (fold.csv_end_ms - fold.csv_start_ms) // HOUR_MS
    frame = synthetic_frame(fold.csv_start_ms, rows, seed=seed)
    if missing_hours:
        drop = {fold.csv_start_ms + hour * HOUR_MS for hour in missing_hours}
        frame = frame[~frame["open_ms"].isin(drop)].reset_index(drop=True)
    return frame


def tiny_export(
    directory: Path,
    recipe: Any,
    *,
    rows: int = TINY_ROWS,
    seed: int = 0,
    missing_hours: tuple[int, ...] = (),
) -> tuple[Any, Any]:
    """``(fold, export record)`` of a tiny synthetic fold exported into ``directory``."""
    from kronos_agent.finetune.export_csv import export_frame

    fold = tiny_fold(recipe, rows=rows)
    frame = tiny_frame(fold, seed=seed, missing_hours=missing_hours)
    record = export_frame(frame, fold, recipe, Path(directory))
    return fold, record


@dataclass(frozen=True)
class TinyRun:
    """Everything a test needs to build requests for one tiny fold."""

    root: Path  # the fine-tune root (runs under root/.runs)
    recipe: Any
    fold: Any
    export: Any
    model_files: Any

    def request(self, **overrides: Any) -> Any:
        """A ``train`` request on the CPU (2 threads, deterministic, no memory guard) with every
        wrapper on, unless ``overrides`` says otherwise."""
        from kronos_agent.finetune.train import Compute, RunRequest

        values: dict[str, Any] = {
            "mode": "train",
            "recipe": self.recipe,
            "fold": self.fold,
            "export": self.export,
            "compute": Compute(device="cpu", threads=2, deterministic=True, memory_guard=False),
            "model_files": self.model_files,
        }
        values.update(overrides)
        return RunRequest(**values)


def tiny_run(
    tmp_path: Path, *, recipe: Any = None, seed: int = 0, **export_options: Any
) -> TinyRun:
    """Tiny models and a tiny export under ``tmp_path``; runs go to ``tmp_path/ft/.runs``."""
    recipe = recipe if recipe is not None else tiny_recipe()
    model_files = save_tiny_models(tmp_path / "models", seed=seed)
    fold, export = tiny_export(tmp_path / "export", recipe, **export_options)
    return TinyRun(
        root=(tmp_path / "ft").absolute(),
        recipe=recipe,
        fold=fold,
        export=export,
        model_files=model_files,
    )


def prepare_run_dir(tiny: TinyRun, request: Any) -> Path:
    """A run directory with ``run.yaml`` and ``run.json`` written for ``request`` (no child)."""
    from kronos_agent.finetune.train import new_run_dir, write_run_inputs

    run_dir = new_run_dir(request.fold.run_label, root=tiny.root)
    write_run_inputs(run_dir, request)
    return run_dir


def read_result(run_dir: Path) -> dict[str, Any]:
    from kronos_agent.finetune.runner import RESULT_JSON

    return json.loads((Path(run_dir) / RESULT_JSON).read_text(encoding="ascii"))


def child_log(run_dir: Path, *, tail: int | None = 4000) -> str:
    """The end of the child's log (``tail=None``: all of it); for assertions and their messages
    only, never for a report (it holds absolute paths)."""
    from kronos_agent.finetune.runner import CHILD_LOG

    path = Path(run_dir) / CHILD_LOG
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return text if tail is None else text[-tail:]


def python_command(code: str, *args: str) -> list[str]:
    """``[python, -B, -c, code, *args]`` with the test interpreter."""
    return [sys.executable, "-B", "-c", code, *args]


# -- stand-in checkpoints and models (registry and batch tests) ----------------------------------

STAND_IN_WEIGHTS = b"stand-in model.safetensors: never loaded"
STAND_IN_RECIPE_ID = "0123abcd" * 8


@functools.cache
def _shipped_payload() -> dict[str, Any]:
    """The identity payload of the shipped Kronos-small (``batch._identity_payload``), computed
    once: its code provenance hashes the source tree."""
    from kronos_agent.kronos.batch import _identity_payload
    from kronos_agent.params import load_params

    return _identity_payload(load_params())


def checkpoint_payload(checkpoint_id: str, *, data_end_ms: int, **changes: Any) -> dict[str, Any]:
    """A valid fold payload (interfaces 5.8) of a Kronos-small checkpoint trained on candles
    before ``data_end_ms``: the registered shipped repositories, configuration and element
    count, an inner validation of 1,344 hours, test months from ``data_end_ms + 512 h`` and
    made-up hashes for the run's files. ``changes`` replace keys (for refusal tests)."""
    payload = copy.deepcopy(_shipped_payload())
    inner_start = data_end_ms - 1344 * HOUR_MS
    test_start = data_end_ms + 512 * HOUR_MS
    payload.update(
        kind="fold",
        checkpoint_id=checkpoint_id,
        data_end_ms=data_end_ms,
        data_end=utc_text(data_end_ms),
        recipe_id=STAND_IN_RECIPE_ID,
        fold="stand-in",
        replicate=0,
        fit_range_ms=[min(TINY_CSV_START_MS, inner_start), inner_start],
        inner_validation_range_ms=[inner_start, data_end_ms],
        test_start_ms=test_start,
        test_end_ms=test_start + 720 * HOUR_MS,
        train_csv_sha256="1" * 64,
        nll_inputs_sha256="2" * 64,
        export_content_sha256="3" * 64,
        run_id="stand-in-run",
        fit_record_sha256="4" * 64,
        seeds={"yaml_seed": 1},
        training_code_sha256="5" * 64,
    )
    payload.update(changes)
    return payload


def stand_in_source(directory: Path, payload: dict[str, Any]) -> Path:
    """A new directory under ``directory`` holding what a run's ``selected/`` holds: the
    payload's ``config.json``, stand-in ``model.safetensors`` bytes and a ``README.md``."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    source = Path(tempfile.mkdtemp(prefix="stand-in-", dir=directory))
    (source / "config.json").write_text(json.dumps(payload["expected_config"]), encoding="ascii")
    (source / "model.safetensors").write_bytes(STAND_IN_WEIGHTS)
    (source / "README.md").write_text("generated by save_pretrained\n", encoding="ascii")
    return source


def earlier_data_end(entry: Any, hours: int) -> dict[str, Any]:
    """The changes that move a registry ``entry``'s data end ``hours`` earlier and keep it a
    valid fold entry (its inner validation ends at the data end)."""
    end = entry.data_end_ms - hours * HOUR_MS
    inner = [entry.document["inner_validation_range_ms"][0], end]
    return {"data_end_ms": end, "data_end": utc_text(end), "inner_validation_range_ms": inner}


def edit_last_line(root: Path, **changes: Any) -> bytes:
    """Rewrite the last line of the registry of ``root`` with ``changes``, its ``prev_sha256``
    kept: a valid entry and a valid chain (only a recorded line sha256 can tell); returns the
    new line."""
    from kronos_agent.finetune import registry
    from kronos_agent.hashing import canonical_json

    path = registry.registry_path(root)
    *earlier, last, tail = path.read_bytes().split(b"\n")
    assert tail == b"", "a registry ends with LF"
    line = canonical_json({**json.loads(last), **changes})
    path.write_bytes(b"".join(item + b"\n" for item in (*earlier, line)))
    return line


def install_stand_in(root: Path, checkpoint_id: str, *, data_end_ms: int, **changes: Any) -> Any:
    """Register a stand-in checkpoint ``checkpoint_id`` in the fine-tune root ``root`` (with a
    check that finds nothing; :func:`checkpoint_payload` and :func:`stand_in_source`); returns its
    ``RegistryEntry``."""
    from kronos_agent.finetune import registry

    payload = checkpoint_payload(checkpoint_id, data_end_ms=data_end_ms, **changes)
    source = stand_in_source(Path(root).parent / "sources", payload)
    return registry.install_checkpoint(
        source, payload=payload, check=lambda directory: [], root=root
    )


class WalkPredictor:
    """``predict_batch`` stand-in: a random walk from the last close of each context, drawn from
    numpy's global generator, which ``seed_everything`` seeds before every fan."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict_batch(
        self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p, **_
    ):
        import numpy as np
        import pandas as pd

        from kronos_agent.kronos.features import FEATURE_COLUMNS

        self.calls.append({"batch": len(df_list), "rows": len(df_list[0]), "pred_len": pred_len})
        frames = []
        for df, y in zip(df_list, y_timestamp_list, strict=True):
            last = float(df["close"].iloc[-1])
            close = last * np.exp(np.cumsum(np.random.normal(0.0, 0.01, size=pred_len)))
            values = np.column_stack(
                [close, close * 1.001, close * 0.999, close, np.ones(pred_len), close]
            )
            frames.append(pd.DataFrame(values, columns=list(FEATURE_COLUMNS), index=y))
        return frames


def walk_models(entry: Any = None, *, predictor: Any = None) -> Any:
    """Stand-in ``LoadedModels`` forecasting with a :class:`WalkPredictor`: the shipped
    Kronos-small's registered weight hashes, or the hashes, id and data end of the registry
    ``entry`` (what ``load_models(checkpoint=...)`` reports for it)."""
    from kronos_agent.kronos.batch import registered_weights_sha256
    from kronos_agent.kronos.weights import LoadedModels
    from kronos_agent.models import spec_for

    return LoadedModels(
        spec=spec_for("small"),
        tokenizer=None,
        model=None,
        predictor=predictor if predictor is not None else WalkPredictor(),
        weights_sha256=(
            registered_weights_sha256("small") if entry is None else entry.weights_sha256()
        ),
        load_seconds=0.0,
        checkpoint=None if entry is None else entry.checkpoint_id,
        data_end_ms=None if entry is None else entry.data_end_ms,
    )
