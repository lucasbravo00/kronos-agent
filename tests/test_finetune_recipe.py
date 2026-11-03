"""The fine-tuning recipe: schema, id, training seeds, split arithmetic (T5) and the run YAML.

The vendored files are read as text only (never imported): the loader's keys and the hard-coded
optimiser values are pinned against the pinned source.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import re
from collections import defaultdict
from dataclasses import fields, replace
from pathlib import Path

import pytest
import yaml

from kronos_agent.finetune import recipe as rc
from kronos_agent.finetune.recipe import (
    R1,
    TEMPLATE_UNUSED,
    VENDORED_HARD_CODED,
    YAML_SECTIONS,
    Recipe,
    RecipeError,
    YamlPaths,
    recipe8,
    recipe_from_mapping,
    recipe_id,
    recipe_mapping,
    run_yaml_bytes,
    run_yaml_document,
    train_ratio,
    training_seeds,
    write_run_yaml,
)
from kronos_agent.hashing import canonical_json, sha256_bytes
from kronos_agent.paths import vendored_kronos_dir
from kronos_agent.seeds import TORCH_SEED_MASK

# Test vectors: a change of R1, of its mapping or of the seed scheme must be deliberate.
R1_ID = "0ee0d123268b2c2fbca2b42e4b0f5b5e23023807105fcb666d159841bee5fdfa"
PILOT_SEEDS = {
    "digest": "26d7e2adf2d7dfe49f24ab04fbe6303ecbc9b6d1896ac5d4b388032054a27e6f",
    "yaml_seed": 651682477,
    "sampler_seed": 8275329013332093700,
    "router_seed": 8927876357573686993,
    "val_seed": 8799807249948651451,
}
FINETUNE_CSV = vendored_kronos_dir() / "finetune_csv"
TINY = replace(
    R1,
    name="tiny",
    batch_size=4,
    pseudo_epoch_steps=2,
    max_epochs=2,
    patience=1,
    inner_validation_hours=516,
    min_nll_decisions=8,
)


def yaml_paths(tmp_path: Path) -> YamlPaths:
    return YamlPaths(
        data_path=tmp_path / "export" / "train.csv",
        predictor_dir=tmp_path / "weights" / "Kronos-small",
        tokenizer_dir=tmp_path / "weights" / "Kronos-Tokenizer-base",
        save_dir=tmp_path / "run" / "vendored",
        base_path=tmp_path / "run",
    )


def document(tmp_path: Path, recipe: Recipe = R1, **overrides) -> dict:
    kwargs = {
        "paths": yaml_paths(tmp_path),
        "n_rows": 38_920,
        "n_fit": 37_576,
        "seed": PILOT_SEEDS["yaml_seed"],
        "exp_name": "wf-2024-07-20261102T120000Z",
        "description": "R1 wf-2024-07 train",
    }
    kwargs.update(overrides)
    return run_yaml_document(recipe, **kwargs)


# -- the recipe --------------------------------------------------------------------------------


def test_r1_holds_the_values_of_the_design():
    assert Recipe() == R1
    assert R1.predictor_learning_rate == 1e-06
    assert (R1.lookback_window, R1.predict_window, R1.window_rows) == (512, 0, 513)
    assert (R1.batch_size, R1.pseudo_epoch_steps, R1.max_epochs) == (32, 512, 6)
    assert R1.onecycle_total_steps == 3072
    assert (R1.patience, R1.early_stopping, R1.nll_steps, R1.nll_grid_hours) == (
        2,
        "nll4_tf",
        4,
        4,
    )
    assert (R1.inner_validation_hours, R1.test_embargo_hours) == (1344, 512)
    assert (R1.min_nll_decisions, R1.max_dropped_fit_share) == (168, 0.05)
    assert (R1.normalisation, R1.clip, R1.num_workers) == ("vendored", 5.0, 0)
    assert (R1.adam_beta1, R1.adam_beta2, R1.adam_weight_decay) == (0.9, 0.95, 0.1)
    assert (R1.source, R1.interval, R1.history_start) == ("binance", "1h", "2020-01-01T00:00Z")


def test_recipe_id_is_the_sha256_of_the_canonical_mapping():
    assert recipe_id(R1) == R1_ID
    assert recipe8(R1) == R1_ID[:8]
    assert recipe_id(R1) == sha256_bytes(canonical_json(recipe_mapping(R1)))
    assert recipe_id(TINY) != R1_ID
    assert recipe_id(replace(R1, batch_size=16)) != R1_ID  # a batch change is a recipe change


def test_mapping_holds_every_field_and_the_hard_coded_values():
    mapping = recipe_mapping(R1)
    names = {field_.name for field_ in fields(Recipe)}
    assert set(mapping) == names | {"schema_version", "vendored_hard_coded"}
    assert mapping["schema_version"] == rc.RECIPE_SCHEMA_VERSION == 1
    assert mapping["vendored_hard_coded"]["onecycle"]["pct_start"] == 0.03
    assert type(mapping["vendored_hard_coded"]) is dict  # plain JSON, not a read-only view
    mapping["vendored_hard_coded"]["grad_clip_max_norm"] = 99.0
    assert VENDORED_HARD_CODED["grad_clip_max_norm"] == 3.0  # the constant cannot change
    with pytest.raises(TypeError):
        VENDORED_HARD_CODED["onecycle"]["pct_start"] = 0.5  # type: ignore[index]


@pytest.mark.parametrize(
    "recipe", [R1, TINY, replace(R1, budget="template", pseudo_epoch_steps=None)]
)
def test_mapping_round_trips(recipe):
    assert recipe_from_mapping(recipe_mapping(recipe)) == recipe
    assert recipe_id(recipe_from_mapping(recipe_mapping(recipe))) == recipe_id(recipe)
    assert recipe_from_mapping(yaml.safe_load(yaml.safe_dump(recipe_mapping(recipe)))) == recipe


def test_integers_for_float_fields_give_the_same_recipe_and_id():
    recipe = Recipe(clip=5, adam_weight_decay=0, max_dropped_fit_share=0)
    assert isinstance(recipe.clip, float) and isinstance(recipe.adam_weight_decay, float)
    assert Recipe(clip=5) == R1 and recipe_id(Recipe(clip=5)) == R1_ID
    mapping = recipe_mapping(R1)
    mapping["clip"] = 5
    assert recipe_from_mapping(mapping) == R1


def changed_mapping(**changes) -> dict:
    mapping = recipe_mapping(R1)
    mapping.update(changes)
    return mapping


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(changed_mapping(extra=1), id="unknown-key"),
        pytest.param(
            {k: v for k, v in recipe_mapping(R1).items() if k != "patience"}, id="missing"
        ),
        pytest.param(changed_mapping(batch_size=True), id="bool-for-int"),
        pytest.param(changed_mapping(batch_size=32.0), id="float-for-int"),
        pytest.param(changed_mapping(clip="5.0"), id="string-for-float"),
        pytest.param(changed_mapping(clip=True), id="bool-for-float"),
        pytest.param(changed_mapping(name=1), id="int-for-string"),
        pytest.param(changed_mapping(pseudo_epoch_steps=512.0), id="float-for-optional-int"),
        pytest.param(changed_mapping(schema_version=2), id="schema-2"),
        pytest.param(changed_mapping(schema_version=True), id="schema-bool"),
        pytest.param(
            changed_mapping(
                vendored_hard_coded={
                    **recipe_mapping(R1)["vendored_hard_coded"],
                    "grad_clip_max_norm": 3,
                }
            ),
            id="hard-coded-int-for-float",
        ),
        pytest.param(changed_mapping(vendored_hard_coded={}), id="hard-coded-empty"),
        pytest.param(changed_mapping(vendored_hard_coded={1, 2}), id="hard-coded-not-json"),
        pytest.param(changed_mapping(max_epochs=0), id="invalid-value"),
        pytest.param([("name", "R1")], id="not-a-mapping"),
    ],
)
def test_mapping_is_strict(raw):
    with pytest.raises(RecipeError):
        recipe_from_mapping(raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "bad name"},
        {"name": ""},
        {"model": "base"},
        {"tokenizer": "trained"},
        {"source": "bingx"},
        {"interval": "15m"},
        {"amount_policy": "zeros"},
        {"history_start": "2019-12-31T00:00Z"},
        {"history_start": "2020-01-01T00:00:00Z"},  # the hour, but not its one text form
        {"history_start": "2020-01-01T00:30Z"},
        {"history_start": "2020-01-01"},
        {"lookback_window": 256},
        {"predict_window": -1},
        {"predict_window": 49},
        {"clip": 4.0},
        {"clip": "5.0"},
        {"normalisation": "context"},
        {"predictor_learning_rate": 0.0},
        {"predictor_learning_rate": 2e-3},
        {"predictor_learning_rate": math.nan},
        {"adam_beta1": 1.0},
        {"adam_beta2": -0.1},
        {"adam_weight_decay": -0.1},
        {"adam_weight_decay": math.inf},
        {"batch_size": 0},
        {"batch_size": 257},
        {"batch_size": True},
        {"batch_size": 32.0},
        {"budget": "epochs"},
        {"pseudo_epoch_steps": 0},
        {"pseudo_epoch_steps": None},
        {"budget": "template"},  # the template budget has no pseudo-epoch steps
        {"max_epochs": 0},
        {"max_epochs": 51},
        {"patience": 0},
        {"patience": 7},
        {"early_stopping": "vendored_val_loss"},
        {"nll_steps": 8},
        {"nll_grid_hours": 0},
        {"nll_grid_hours": 5},
        {"inner_validation_hours": 513},  # a vendored dataset needs more rows than a window
        {"inner_validation_hours": 1346},  # not on the 4-hour NLL grid
        {"test_embargo_hours": -1},
        {"min_nll_decisions": 0},
        {"max_dropped_fit_share": 1.0},
        {"max_dropped_fit_share": -0.01},
        {"log_interval": 0},
        {"num_workers": 6},
    ],
    ids=lambda changes: "-".join(f"{k}={v!r}" for k, v in changes.items()),
)
def test_invalid_recipes_are_refused(changes):
    with pytest.raises(RecipeError):
        replace(R1, **changes)


def test_test_scale_and_variant_recipes_are_valid():
    assert TINY.onecycle_total_steps == 4
    template = replace(R1, budget="template", pseudo_epoch_steps=None, max_epochs=20)
    assert template.onecycle_total_steps is None
    assert replace(R1, predict_window=48).window_rows == 561  # the template's window
    assert replace(R1, normalisation="context_only").normalisation == "context_only"
    assert replace(R1, test_embargo_hours=0).test_embargo_hours == 0


def test_hard_coded_values_match_the_pinned_vendored_code():
    import torch

    base = (FINETUNE_CSV / "finetune_base_model.py").read_text(encoding="utf-8")
    assert "pct_start=0.03" in base and "div_factor=10\n" in base
    assert "max_norm=3.0" in base
    assert "torch.optim.AdamW(" in base and "OneCycleLR(" in base
    assert "drop_last=True" in base
    # what OneCycleLR leaves at torch's defaults (torch 2.14.1)
    defaults = inspect.signature(torch.optim.lr_scheduler.OneCycleLR).parameters
    onecycle = VENDORED_HARD_CODED["onecycle"]
    for name in ("final_div_factor", "anneal_strategy", "cycle_momentum", "base_momentum"):
        assert defaults[name].default == onecycle[name], name
    assert defaults["max_momentum"].default == onecycle["max_momentum"]
    assert defaults["three_phase"].default == onecycle["three_phase"]
    template = yaml.safe_load(
        (FINETUNE_CSV / "configs" / "config_ali09988_candle-5min.yaml").read_text("utf-8")
    )
    for key, value in TEMPLATE_UNUSED.items():
        assert template["training"][key] == value, key
    assert template["training"]["predictor_learning_rate"] == R1.predictor_learning_rate


# -- training seeds ------------------------------------------------------------------------------


def test_training_seeds_of_the_pilot():
    seeds = training_seeds(R1_ID, "wf-2024-07", 0)
    assert seeds.canonical == f"kronos-agent/train-seed/v1/{R1_ID}/wf-2024-07/r0"
    assert {name: getattr(seeds, name) for name in PILOT_SEEDS} == PILOT_SEEDS
    digest = hashlib.sha256(seeds.canonical.encode("ascii")).digest()
    assert seeds.digest == digest.hex()
    assert seeds.yaml_seed == int.from_bytes(digest[:4], "big") < 2**32
    assert seeds.sampler_seed == int.from_bytes(digest[4:12], "big") & TORCH_SEED_MASK
    assert seeds.router_seed == int.from_bytes(digest[12:20], "big") & TORCH_SEED_MASK
    val = hashlib.sha256(b"kronos-agent/finetune/val-seed/v1").digest()
    assert seeds.val_seed == int.from_bytes(val[:8], "big") & TORCH_SEED_MASK


def test_training_seeds_differ_per_fold_and_replicate_and_share_the_validation_seed():
    every = [
        training_seeds(R1_ID, label, replicate)
        for label in ("wf-2024-07", "wf-2025-06", "wf-2026-06", "q-2024-07", "synthetic")
        for replicate in (0, 1, 2)
    ]
    assert len({seeds.digest for seeds in every}) == len(every)
    assert len({seeds.yaml_seed for seeds in every}) == len(every)
    assert len({seeds.val_seed for seeds in every}) == 1
    other = training_seeds(recipe_id(TINY), "wf-2024-07", 0)
    assert other.yaml_seed != PILOT_SEEDS["yaml_seed"]


@pytest.mark.parametrize(
    ("identifier", "label", "replicate"),
    [
        (R1_ID[:8], "wf-2024-07", 0),
        (R1_ID.upper(), "wf-2024-07", 0),
        (R1_ID, "wf 2024-07", 0),
        (R1_ID, "wf-2024-07/x", 0),
        (R1_ID, "wf-2024-07-r1", 1),  # the run label, not the fold label
        (R1_ID, "wf-2024-07", -1),
        (R1_ID, "wf-2024-07", True),
        (R1_ID, "wf-2024-07", 1.0),
    ],
)
def test_training_seeds_refuse_bad_requests(identifier, label, replicate):
    with pytest.raises(RecipeError):
        training_seeds(identifier, label, replicate)


# -- T5: the vendored split arithmetic -----------------------------------------------------------


def vendored_split(n_rows: int, ratio: float) -> tuple[int, int]:
    """``train_end`` and ``val_end`` as FB:75-97 compute them with ``val_ratio = 1.0``."""
    return int(n_rows * ratio), int(n_rows * (ratio + 1.0))


def test_train_ratio_lands_on_every_row_of_small_csvs():
    for n_rows in range(2, 400):
        for n_fit in range(1, n_rows):
            ratio = train_ratio(n_rows, n_fit)
            train_end, val_end = vendored_split(n_rows, ratio)
            assert train_end == n_fit and val_end >= n_rows, (n_rows, n_fit)


def test_train_ratio_lands_on_the_planned_and_large_counts():
    planned = [(38_920, 37_576), (55_720, 54_376), (57_696, 56_352), (4096, 2752)]
    large = [(n, n - 1344) for n in range(40_000, 2_000_000, 9_973)]
    for n_rows, n_fit in planned + large + [(10**7, 1), (10**7, 10**7 - 1)]:
        ratio = train_ratio(n_rows, n_fit)
        assert ratio == (n_fit + 0.5) / n_rows
        train_end, val_end = vendored_split(n_rows, ratio)
        assert train_end == n_fit and val_end >= n_rows


@pytest.mark.parametrize(
    ("n_rows", "n_fit"),
    [(10, 0), (10, 10), (10, 11), (0, 0), (10.0, 5), (10, True), ("10", 5)],
)
def test_train_ratio_refuses_impossible_splits(n_rows, n_fit):
    with pytest.raises(RecipeError):
        train_ratio(n_rows, n_fit)


# -- the run YAML ------------------------------------------------------------------------------


def loader_keys() -> dict[str, set[str]]:
    """Every key the pinned vendored ``config_loader`` reads, per section (from its source)."""
    text = (FINETUNE_CSV / "config_loader.py").read_text(encoding="utf-8")
    sections = {
        "data_config": "data",
        "training_config": "training",
        "model_paths": "model_paths",
        "experiment_config": "experiment",
        "device_config": "device",
        "distributed_config": "distributed",
    }
    keys: dict[str, set[str]] = defaultdict(set)
    for variable, key in re.findall(r"(\w+)\.get\('(\w+)'", text):
        if variable in sections:
            keys[sections[variable]].add(key)
    for key in re.findall(r"config\.get\('model_paths', \{\}\)\.get\('(\w+)'", text):
        keys["model_paths"].add(key)
    return dict(keys)


def test_yaml_sections_are_every_key_the_vendored_loader_reads():
    keys = loader_keys()
    assert "epochs" in keys["training"]
    keys["training"].discard("epochs")  # the alias, read only when an epoch key is absent
    assert {section: set(names) for section, names in YAML_SECTIONS.items()} == keys


def test_run_yaml_document_is_every_key_explicit(tmp_path):
    doc = document(tmp_path)
    assert {section: set(values) for section, values in doc.items()} == {
        section: set(names) for section, names in YAML_SECTIONS.items()
    }
    data, training, model_paths = doc["data"], doc["training"], doc["model_paths"]
    assert data["train_ratio"] == train_ratio(38_920, 37_576)
    assert (data["val_ratio"], data["test_ratio"]) == (1.0, 0.0)
    assert (data["lookback_window"], data["predict_window"], data["max_context"]) == (512, 0, 512)
    assert data["data_path"] == str(tmp_path / "export" / "train.csv")
    assert training["basemodel_epochs"] == 6 and training["batch_size"] == 32
    assert training["seed"] == PILOT_SEEDS["yaml_seed"]
    assert training["predictor_learning_rate"] == 1e-06
    assert training["num_workers"] == 0
    for key, value in TEMPLATE_UNUSED.items():
        assert training[key] == value
    # the predictor phase loads the tokenizer from finetuned_tokenizer: the shipped one, frozen
    assert model_paths["finetuned_tokenizer"] == model_paths["pretrained_tokenizer"]
    assert model_paths["base_save_path"] == str(tmp_path / "run" / "vendored")
    assert doc["experiment"]["train_tokenizer"] is False
    assert doc["experiment"]["train_basemodel"] is True
    assert doc["device"] == {"use_cuda": False, "device_id": 0}


def test_run_yaml_bytes_are_deterministic_ascii_and_load_back(tmp_path):
    doc = document(tmp_path)
    data = run_yaml_bytes(doc)
    assert data == run_yaml_bytes(document(tmp_path))
    assert data.decode("ascii").endswith("\n") and "{" not in data.decode("ascii")
    assert yaml.safe_load(data) == doc
    assert b"predictor_learning_rate: 1.0e-06\n" in data
    path = tmp_path / "run" / "run.yaml"
    assert write_run_yaml(path, doc) == sha256_bytes(data)
    assert path.read_bytes() == data
    assert write_run_yaml(path, doc) == sha256_bytes(data)  # rewritten atomically, same bytes


def test_run_yaml_keeps_odd_but_valid_texts(tmp_path):
    doc = document(tmp_path, exp_name="2024", description="null true 1e5 yes")
    assert yaml.safe_load(run_yaml_bytes(doc)) == doc


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(extra={}),
        lambda doc: doc.pop("distributed"),
        lambda doc: doc["training"].update(epochs=6),
        lambda doc: doc["training"].pop("accumulation_steps"),
        lambda doc: doc["data"].update(clip=None),
        lambda doc: doc["data"].update(clip=[5.0]),
        lambda doc: doc["data"].update(train_ratio=math.nan),
        lambda doc: doc["model_paths"].update(base_save_path="/runs/{exp_name}"),
    ],
)
def test_run_yaml_bytes_refuse_a_document_that_is_not_the_exact_shape(tmp_path, mutate):
    doc = document(tmp_path)
    mutate(doc)
    with pytest.raises(RecipeError):
        run_yaml_bytes(doc)


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"exp_name": "run id"},
        {"exp_name": "{exp_name}"},
        {"description": "R1 {fold}"},
        {"description": "caf" + chr(0xE9)},  # not ASCII
        {"description": "two\nlines"},
        {"description": ""},
        {"n_fit": 38_920},
    ],
)
def test_run_yaml_document_refuses_bad_values(tmp_path, overrides):
    with pytest.raises(RecipeError):
        document(tmp_path, **overrides)


def test_yaml_paths_are_absolute_and_free_of_braces(tmp_path):
    paths = yaml_paths(tmp_path)
    assert all(isinstance(getattr(paths, f.name), Path) for f in fields(paths))
    assert YamlPaths(*(str(getattr(paths, f.name)) for f in fields(paths))) == paths
    with pytest.raises(RecipeError):
        replace(paths, data_path=Path("export/train.csv"))
    with pytest.raises(RecipeError):
        replace(paths, save_dir=tmp_path / "{exp_name}")
    with pytest.raises(RecipeError):
        replace(paths, base_path=None)
