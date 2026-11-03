"""The fine-tuning recipe (step 6; ``docs/RETRAINING.md`` 3.2): schema, id, seeds and run YAML.

A :class:`Recipe` names every value that shapes a fine-tune: what the vendored predictor phase of
``finetune_csv`` reads from its YAML, and what the project adds around it (the pseudo-epoch
budget, early stopping on NLL4-TF, the fold geometry and the data rules of the exporter).
:data:`R1` is the first recipe of the design. Its id, :func:`recipe_id`, is the sha256 of the
canonical JSON of :func:`recipe_mapping`, which also records the values the vendored code
hard-codes (:data:`VENDORED_HARD_CODED`; FB:246-260, FB:292-296, K:267-272), so they are part of
what a recipe is even though nothing can change them. The compute choices (device, threads,
determinism, activation checkpointing) stay outside the id: every run records them next to it.
The batch size is inside it (a batch change is a recipe change, design section (a)).

Training seeds come from ``sha256("kronos-agent/train-seed/v1/<recipe_id>/<fold>/r<k>")``
(design 3.4): bytes 0-3 are the YAML ``seed`` (32 bits: the vendored ``set_seed`` calls
``np.random.seed``), bytes 4-11 the sampler's generator and bytes 12-19 the multinomial router's,
both masked to 63 bits as :mod:`kronos_agent.seeds` masks torch seeds.

The run YAML is the file the vendored ``CustomFinetuneConfig`` reads (CL:119-182). Every key it
reads is written explicitly (:data:`YAML_SECTIONS`), the unused tokenizer keys included
(:data:`TEMPLATE_UNUSED`), so a fine-tune never depends on a code default. No value may contain
``{`` or ``}`` (the loader formats ``{exp_name}`` placeholders, CL:25-49), and the bytes written
must load back to the same document.

Standard library and ``yaml`` only at module level: no numpy, pandas or torch.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ..data.partition import TRAINING_START
from ..data.timeutil import parse_utc_hour, to_ms, utc_text
from ..hashing import canonical_json, sha256_bytes
from ..models import spec_for
from ..paths import LABEL_RE
from ..seeds import TORCH_SEED_MASK

RECIPE_SCHEMA_VERSION = 1
BUDGETS: tuple[str, ...] = ("pseudo_epochs", "template")
NORMALISATIONS: tuple[str, ...] = ("vendored", "context_only")
EARLY_STOPPING_METRICS: tuple[str, ...] = ("nll4_tf",)
NLL_STEPS = 4
TRAIN_SEED_SCHEME = "kronos-agent/train-seed/v1"
VAL_SEED_STRING = "kronos-agent/finetune/val-seed/v1"
MAX_PREDICT_WINDOW = 48
MAX_LEARNING_RATE = 1e-3
MAX_BATCH_SIZE = 256
MAX_EPOCHS = 50
YAML_SEED_LIMIT = 1 << 32  # np.random.seed accepts 0 <= seed < 2**32
YAML_EXPERIMENT_NAME = "kronos-agent-finetune"


def _read_only(value: Any) -> Any:
    """``value`` with every mapping wrapped in a read-only view (lists become tuples)."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _read_only(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_read_only(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    """``value`` as plain JSON types: mappings become dicts, tuples become lists."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# Recorded, never configurable: what the pinned vendored predictor phase hard-codes.
VENDORED_HARD_CODED: Mapping[str, Any] = _read_only(
    {
        "grad_clip_max_norm": 3.0,
        "optimizer": "AdamW",
        "weight_decay_applies_to": "every parameter",
        "scheduler": "OneCycleLR",
        "scheduler_step": "every batch",
        "onecycle": {
            "pct_start": 0.03,
            "div_factor": 10.0,
            "final_div_factor": 10000.0,
            "anneal_strategy": "cos",
            "cycle_momentum": True,
            "base_momentum": 0.85,
            "max_momentum": 0.95,
            "three_phase": False,
        },
        "loss": "(ce_s1 + ce_s2) / 2 over every position",
        "s2_conditioning": "s1 sampled by multinomial at T = 1 without top-p",
        "drop_last": True,
        "validation_loss": "mean of per-batch means",
    }
)
# The template's values (CFG) of the keys the predictor phase never reads (train_tokenizer false).
TEMPLATE_UNUSED: Mapping[str, Any] = MappingProxyType(
    {"tokenizer_epochs": 30, "tokenizer_learning_rate": 0.0002, "accumulation_steps": 1}
)
# Every key CustomFinetuneConfig reads (CL:119-182), plus base_path (CL:31), per section.
YAML_SECTIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "data": (
            "data_path",
            "lookback_window",
            "predict_window",
            "max_context",
            "clip",
            "train_ratio",
            "val_ratio",
            "test_ratio",
        ),
        "training": (
            "tokenizer_epochs",
            "basemodel_epochs",
            "batch_size",
            "log_interval",
            "num_workers",
            "seed",
            "tokenizer_learning_rate",
            "predictor_learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_weight_decay",
            "accumulation_steps",
        ),
        "model_paths": (
            "exp_name",
            "base_path",
            "pretrained_tokenizer",
            "pretrained_predictor",
            "finetuned_tokenizer",
            "base_save_path",
            "tokenizer_save_name",
            "basemodel_save_name",
        ),
        "experiment": (
            "name",
            "description",
            "use_comet",
            "train_tokenizer",
            "train_basemodel",
            "skip_existing",
            "pre_trained",
            "pre_trained_tokenizer",
            "pre_trained_predictor",
        ),
        "device": ("use_cuda", "device_id"),
        "distributed": ("use_ddp", "backend"),
    }
)

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_REPLICATE_SUFFIX = re.compile(r"-r[0-9]+$")

_STR_FIELDS = frozenset(
    {
        "name",
        "model",
        "tokenizer",
        "source",
        "interval",
        "history_start",
        "amount_policy",
        "normalisation",
        "budget",
        "early_stopping",
    }
)
_INT_FIELDS = frozenset(
    {
        "lookback_window",
        "predict_window",
        "batch_size",
        "max_epochs",
        "patience",
        "nll_steps",
        "nll_grid_hours",
        "inner_validation_hours",
        "test_embargo_hours",
        "min_nll_decisions",
        "log_interval",
        "num_workers",
    }
)
_OPTIONAL_INT_FIELDS = frozenset({"pseudo_epoch_steps"})
_FLOAT_FIELDS = frozenset(
    {
        "clip",
        "predictor_learning_rate",
        "adam_beta1",
        "adam_beta2",
        "adam_weight_decay",
        "max_dropped_fit_share",
    }
)
_FIXED_STRINGS: Mapping[str, str] = MappingProxyType(
    {
        "model": "small",
        "tokenizer": "frozen",
        "source": "binance",
        "interval": "1h",
        "amount_policy": "volume_x_mean_ohlc",
    }
)


class RecipeError(ValueError):
    """A recipe, a recipe mapping, a seed request or a run YAML that is not valid."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Recipe:
    """One fine-tuning recipe; :meth:`__post_init__` refuses an invalid one (:class:`RecipeError`).

    Integers given for float fields are stored as floats, so equal recipes have equal ids.
    Test-scale recipes (few steps, a short inner validation) are valid; :data:`R1` is
    ``Recipe()``.
    """

    name: str = "R1"
    model: str = "small"
    tokenizer: str = "frozen"
    source: str = "binance"
    interval: str = "1h"
    history_start: str = "2020-01-01T00:00Z"
    amount_policy: str = "volume_x_mean_ohlc"
    lookback_window: int = 512
    predict_window: int = 0
    clip: float = 5.0
    normalisation: str = "vendored"
    predictor_learning_rate: float = 1e-06
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.1
    batch_size: int = 32
    budget: str = "pseudo_epochs"
    pseudo_epoch_steps: int | None = 512
    max_epochs: int = 6  # E_max = basemodel_epochs
    early_stopping: str = "nll4_tf"
    patience: int = 2
    nll_steps: int = 4
    nll_grid_hours: int = 4
    inner_validation_hours: int = 1344
    test_embargo_hours: int = 512
    min_nll_decisions: int = 168
    max_dropped_fit_share: float = 0.05
    log_interval: int = 50
    num_workers: int = 0

    def __post_init__(self) -> None:
        self._check_types()
        self._check_values()

    def _fail(self, message: str) -> RecipeError:
        return RecipeError(f"recipe {self.name!r}: {message}")

    def _check_types(self) -> None:
        for field_ in fields(self):
            name = field_.name
            value = getattr(self, name)
            if name in _STR_FIELDS and not isinstance(value, str):
                raise self._fail(f"{name} must be a string, got {value!r}")
            if name in _INT_FIELDS and not _is_int(value):
                raise self._fail(f"{name} must be an integer, got {value!r}")
            if name in _OPTIONAL_INT_FIELDS and value is not None and not _is_int(value):
                raise self._fail(f"{name} must be an integer or None, got {value!r}")
            if name in _FLOAT_FIELDS:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise self._fail(f"{name} must be a number, got {value!r}")
                if not math.isfinite(value):
                    raise self._fail(f"{name} must be finite, got {value!r}")
                object.__setattr__(self, name, float(value))

    def _check_values(self) -> None:
        if not LABEL_RE.fullmatch(self.name):
            raise RecipeError(f"recipe name must match {LABEL_RE.pattern}, got {self.name!r}")
        for name, expected in _FIXED_STRINGS.items():
            if getattr(self, name) != expected:
                raise self._fail(f"{name} must be {expected!r}, got {getattr(self, name)!r}")
        try:
            start = parse_utc_hour(self.history_start)
        except ValueError as exc:
            raise self._fail(f"history_start: {exc}") from None
        if self.history_start != utc_text(to_ms(start)):
            raise self._fail(
                f"history_start must be written {utc_text(to_ms(start))!r} (one text per hour, "
                f"so one id per recipe), got {self.history_start!r}"
            )
        if start < TRAINING_START:
            raise self._fail(
                f"history_start {self.history_start} is before the training segment "
                f"({utc_text(to_ms(TRAINING_START))})"
            )
        max_context = spec_for(self.model).max_context
        if self.lookback_window != max_context:
            raise self._fail(
                f"lookback_window must be the model's context of {max_context}, got "
                f"{self.lookback_window}"
            )
        if not 0 <= self.predict_window <= MAX_PREDICT_WINDOW:
            raise self._fail(f"predict_window must be in [0, {MAX_PREDICT_WINDOW}]")
        if self.clip != 5.0:
            raise self._fail(f"clip must be 5.0 (the inference clip), got {self.clip}")
        if self.normalisation not in NORMALISATIONS:
            raise self._fail(f"normalisation {self.normalisation!r} not in {list(NORMALISATIONS)}")
        if not 0.0 < self.predictor_learning_rate <= MAX_LEARNING_RATE:
            raise self._fail(f"predictor_learning_rate must be in (0, {MAX_LEARNING_RATE}]")
        for name in ("adam_beta1", "adam_beta2"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise self._fail(f"{name} must be in [0, 1)")
        if self.adam_weight_decay < 0.0:
            raise self._fail("adam_weight_decay must be >= 0")
        if not 1 <= self.batch_size <= MAX_BATCH_SIZE:
            raise self._fail(f"batch_size must be in [1, {MAX_BATCH_SIZE}]")
        self._check_budget()
        if self.nll_steps != NLL_STEPS:
            raise self._fail(f"nll_steps must be {NLL_STEPS} (NLL4-TF), got {self.nll_steps}")
        if self.nll_grid_hours < 1 or 24 % self.nll_grid_hours:
            raise self._fail(f"nll_grid_hours must divide 24, got {self.nll_grid_hours}")
        if self.inner_validation_hours < self.window_rows + 1:
            raise self._fail(
                f"inner_validation_hours must be at least window_rows + 1 = "
                f"{self.window_rows + 1} (the vendored dataset needs more rows than a window)"
            )
        if self.inner_validation_hours % self.nll_grid_hours:
            raise self._fail("inner_validation_hours must be a multiple of nll_grid_hours")
        if self.test_embargo_hours < 0:
            raise self._fail("test_embargo_hours must be >= 0")
        if self.min_nll_decisions < 1:
            raise self._fail("min_nll_decisions must be >= 1")
        if not 0.0 <= self.max_dropped_fit_share < 1.0:
            raise self._fail("max_dropped_fit_share must be in [0, 1)")
        if self.log_interval < 1:
            raise self._fail("log_interval must be >= 1")
        if self.num_workers != 0:
            raise self._fail(
                "num_workers must be 0: loader workers would import the vendored tree outside "
                "import_kronos() (D-014)"
            )

    def _check_budget(self) -> None:
        if self.budget not in BUDGETS:
            raise self._fail(f"budget {self.budget!r} not in {list(BUDGETS)}")
        if self.budget == "pseudo_epochs":
            if self.pseudo_epoch_steps is None or self.pseudo_epoch_steps < 1:
                raise self._fail("the pseudo_epochs budget needs pseudo_epoch_steps >= 1")
        elif self.pseudo_epoch_steps is not None:
            raise self._fail("the template budget has no pseudo_epoch_steps (None)")
        if not 1 <= self.max_epochs <= MAX_EPOCHS:
            raise self._fail(f"max_epochs must be in [1, {MAX_EPOCHS}]")
        if not 1 <= self.patience <= self.max_epochs:
            raise self._fail("patience must be in [1, max_epochs]")
        if self.early_stopping not in EARLY_STOPPING_METRICS:
            raise self._fail(
                f"early_stopping {self.early_stopping!r} not in {list(EARLY_STOPPING_METRICS)}"
            )

    @property
    def window_rows(self) -> int:
        """Rows of a vendored window, ``lookback + predict + 1`` (FB:33): 513 for R1."""
        return self.lookback_window + self.predict_window + 1

    @property
    def onecycle_total_steps(self) -> int | None:
        """``max_epochs * pseudo_epoch_steps``, the OneCycle span; ``None`` for the template."""
        if self.pseudo_epoch_steps is None:
            return None
        return self.max_epochs * self.pseudo_epoch_steps


R1 = Recipe()


@dataclass(frozen=True)
class TrainingSeeds:
    """The seeds of one fit (design 3.4), derived from :attr:`canonical`."""

    canonical: str
    digest: str
    yaml_seed: int
    sampler_seed: int
    router_seed: int
    val_seed: int


@dataclass(frozen=True)
class YamlPaths:
    """The paths of a run YAML; every one absolute and free of ``{`` and ``}``."""

    data_path: Path
    predictor_dir: Path
    tokenizer_dir: Path
    save_dir: Path
    base_path: Path

    def __post_init__(self) -> None:
        for field_ in fields(self):
            value = getattr(self, field_.name)
            if not isinstance(value, (str, Path)):
                raise RecipeError(f"{field_.name} must be a path, got {type(value).__name__}")
            path = Path(value)
            if not path.is_absolute():
                raise RecipeError(f"{field_.name} must be an absolute path")
            if "{" in str(path) or "}" in str(path):
                raise RecipeError(
                    f"{field_.name} must not contain '{{' or '}}' (the vendored loader formats "
                    "placeholders)"
                )
            object.__setattr__(self, field_.name, path)


def recipe_mapping(recipe: Recipe) -> dict[str, Any]:
    """Every field of ``recipe``, its schema version and the hard-coded vendored values (JSON)."""
    if not isinstance(recipe, Recipe):
        raise TypeError(f"expected a Recipe, got {type(recipe).__name__}")
    mapping: dict[str, Any] = {
        field_.name: getattr(recipe, field_.name) for field_ in fields(recipe)
    }
    mapping["schema_version"] = RECIPE_SCHEMA_VERSION
    mapping["vendored_hard_coded"] = _plain(VENDORED_HARD_CODED)
    return mapping


def _field_value(name: str, value: object) -> object:
    """``value`` of recipe field ``name`` from a mapping; ints are accepted for floats."""
    if name in _INT_FIELDS and not _is_int(value):
        raise RecipeError(f"{name}: expected an integer, got {value!r}")
    if name in _OPTIONAL_INT_FIELDS and value is not None and not _is_int(value):
        raise RecipeError(f"{name}: expected an integer or null, got {value!r}")
    if name in _FLOAT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RecipeError(f"{name}: expected a number, got {value!r}")
        return float(value)
    if name in _STR_FIELDS and not isinstance(value, str):
        raise RecipeError(f"{name}: expected a string, got {value!r}")
    return value


def recipe_from_mapping(raw: Mapping[str, Any]) -> Recipe:
    """The :class:`Recipe` of a :func:`recipe_mapping` (strict: keys, types, schema, constants)."""
    if not isinstance(raw, Mapping):
        raise RecipeError(f"a recipe mapping must be a mapping, got {type(raw).__name__}")
    names = {field_.name for field_ in fields(Recipe)}
    expected = names | {"schema_version", "vendored_hard_coded"}
    unknown = sorted(str(key) for key in set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown:
        raise RecipeError(f"recipe mapping: unknown keys {unknown}")
    if missing:
        raise RecipeError(f"recipe mapping: missing keys {missing}")
    version = raw["schema_version"]
    if not _is_int(version) or version != RECIPE_SCHEMA_VERSION:
        raise RecipeError(
            f"recipe mapping: schema_version must be {RECIPE_SCHEMA_VERSION}, got {version!r}"
        )
    try:
        recorded = canonical_json(_plain(raw["vendored_hard_coded"]))
    except (TypeError, ValueError) as exc:
        raise RecipeError(f"recipe mapping: vendored_hard_coded is not JSON ({exc})") from None
    if recorded != canonical_json(_plain(VENDORED_HARD_CODED)):
        raise RecipeError(
            "recipe mapping: vendored_hard_coded differs from the pinned vendored code's values"
        )
    values = {name: _field_value(name, raw[name]) for name in sorted(names)}
    return Recipe(**values)


def recipe_id(recipe: Recipe) -> str:
    """sha256 hex of the canonical JSON of :func:`recipe_mapping`."""
    return sha256_bytes(canonical_json(recipe_mapping(recipe)))


def recipe8(recipe: Recipe) -> str:
    """The first 8 hex characters of :func:`recipe_id` (directory and checkpoint names)."""
    return recipe_id(recipe)[:8]


def training_seeds(recipe_id: str, fold_label: str, replicate: int) -> TrainingSeeds:
    """The seeds of fold ``fold_label`` (no replicate suffix), replicate ``replicate`` (design 3.4).

    ``canonical = "kronos-agent/train-seed/v1/<recipe_id>/<fold_label>/r<replicate>"``; bytes 0-3
    of its sha256 (big-endian) are the YAML seed, bytes 4-11 and 12-19 the sampler and router
    seeds (masked to 63 bits); the validation seed is bytes 0-7 of
    ``sha256("kronos-agent/finetune/val-seed/v1")``, the same for every fit.
    """
    if not isinstance(recipe_id, str) or not _SHA256_HEX.fullmatch(recipe_id):
        raise RecipeError(f"recipe_id must be 64 lowercase hex characters, got {recipe_id!r}")
    if not isinstance(fold_label, str) or not LABEL_RE.fullmatch(fold_label):
        raise RecipeError(f"fold label must match {LABEL_RE.pattern}, got {fold_label!r}")
    if _REPLICATE_SUFFIX.search(fold_label):
        raise RecipeError(f"pass the fold label without its replicate suffix, got {fold_label!r}")
    if not _is_int(replicate) or replicate < 0:
        raise RecipeError(f"replicate must be an integer >= 0, got {replicate!r}")
    canonical = f"{TRAIN_SEED_SCHEME}/{recipe_id}/{fold_label}/r{replicate}"
    digest = hashlib.sha256(canonical.encode("ascii")).digest()
    val_digest = hashlib.sha256(VAL_SEED_STRING.encode("ascii")).digest()
    return TrainingSeeds(
        canonical=canonical,
        digest=digest.hex(),
        yaml_seed=int.from_bytes(digest[0:4], "big"),
        sampler_seed=int.from_bytes(digest[4:12], "big") & TORCH_SEED_MASK,
        router_seed=int.from_bytes(digest[12:20], "big") & TORCH_SEED_MASK,
        val_seed=int.from_bytes(val_digest[0:8], "big") & TORCH_SEED_MASK,
    )


def train_ratio(n_rows: int, n_fit: int) -> float:
    """``(n_fit + 0.5) / n_rows``, checked against the vendored split arithmetic (FB:75-97).

    With that ratio and ``val_ratio = 1.0`` the vendored split gives rows ``[0, n_fit)`` to
    training and ``[n_fit, n_rows)`` to validation: ``int(n_rows * r) == n_fit`` and
    ``int(n_rows * (r + 1.0)) >= n_rows`` are checked with this exact float (:class:`RecipeError`
    otherwise).
    """
    if not _is_int(n_rows) or not _is_int(n_fit):
        raise RecipeError(f"row counts must be integers, got {n_rows!r} and {n_fit!r}")
    if not 0 < n_fit < n_rows:
        raise RecipeError(f"need 0 < n_fit < n_rows, got n_fit={n_fit}, n_rows={n_rows}")
    ratio = (n_fit + 0.5) / n_rows
    if int(n_rows * ratio) != n_fit or int(n_rows * (ratio + 1.0)) < n_rows:
        raise RecipeError(
            f"the vendored split of {n_rows} rows at train_ratio {ratio!r} does not land on row "
            f"{n_fit}"
        )
    return ratio


def _yaml_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecipeError(f"{name} must be a non-empty string")
    if "{" in value or "}" in value:
        raise RecipeError(f"{name} must not contain '{{' or '}}'")
    if not value.isascii() or not value.isprintable():
        raise RecipeError(f"{name} must be printable ASCII")
    return value


def run_yaml_document(
    recipe: Recipe,
    *,
    paths: YamlPaths,
    n_rows: int,
    n_fit: int,
    seed: int,
    exp_name: str,
    description: str,
) -> dict[str, Any]:
    """The run YAML document of one fit: every key the vendored loader reads, explicitly.

    The predictor phase loads the tokenizer from ``finetuned_tokenizer`` (TS:171), which points
    at the shipped tokenizer (frozen, ``train_tokenizer: false``); ``train_ratio`` is
    :func:`train_ratio` of the export's row counts with ``val_ratio`` 1.0 and ``test_ratio`` 0.0.
    """
    if not isinstance(recipe, Recipe):
        raise TypeError(f"expected a Recipe, got {type(recipe).__name__}")
    if not isinstance(paths, YamlPaths):
        raise TypeError(f"expected YamlPaths, got {type(paths).__name__}")
    if not _is_int(seed) or not 0 <= seed < YAML_SEED_LIMIT:
        raise RecipeError(f"seed must be an integer in [0, 2**32), got {seed!r}")
    if not isinstance(exp_name, str) or not LABEL_RE.fullmatch(exp_name):
        raise RecipeError(f"exp_name must match {LABEL_RE.pattern}, got {exp_name!r}")
    description = _yaml_text(description, "description")
    ratio = train_ratio(n_rows, n_fit)
    tokenizer_dir = str(paths.tokenizer_dir)
    return {
        "data": {
            "data_path": str(paths.data_path),
            "lookback_window": recipe.lookback_window,
            "predict_window": recipe.predict_window,
            "max_context": spec_for(recipe.model).max_context,
            "clip": recipe.clip,
            "train_ratio": ratio,
            "val_ratio": 1.0,
            "test_ratio": 0.0,
        },
        "training": {
            "tokenizer_epochs": TEMPLATE_UNUSED["tokenizer_epochs"],
            "basemodel_epochs": recipe.max_epochs,
            "batch_size": recipe.batch_size,
            "log_interval": recipe.log_interval,
            "num_workers": recipe.num_workers,
            "seed": seed,
            "tokenizer_learning_rate": TEMPLATE_UNUSED["tokenizer_learning_rate"],
            "predictor_learning_rate": recipe.predictor_learning_rate,
            "adam_beta1": recipe.adam_beta1,
            "adam_beta2": recipe.adam_beta2,
            "adam_weight_decay": recipe.adam_weight_decay,
            "accumulation_steps": TEMPLATE_UNUSED["accumulation_steps"],
        },
        "model_paths": {
            "exp_name": exp_name,
            "base_path": str(paths.base_path),
            "pretrained_tokenizer": tokenizer_dir,
            "pretrained_predictor": str(paths.predictor_dir),
            "finetuned_tokenizer": tokenizer_dir,
            "base_save_path": str(paths.save_dir),
            "tokenizer_save_name": "tokenizer",
            "basemodel_save_name": "basemodel",
        },
        "experiment": {
            "name": YAML_EXPERIMENT_NAME,
            "description": description,
            "use_comet": False,
            "train_tokenizer": False,
            "train_basemodel": True,
            "skip_existing": False,
            "pre_trained": True,
            "pre_trained_tokenizer": True,
            "pre_trained_predictor": True,
        },
        "device": {"use_cuda": False, "device_id": 0},
        "distributed": {"use_ddp": False, "backend": "nccl"},
    }


def _check_document(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Plain copy of a run YAML document after checking its exact key set and value types."""
    if not isinstance(document, Mapping):
        raise RecipeError(f"a run YAML document is a mapping, got {type(document).__name__}")
    if set(document) != set(YAML_SECTIONS):
        raise RecipeError(
            f"run YAML sections {sorted(map(str, document))} != {sorted(YAML_SECTIONS)}"
        )
    plain: dict[str, dict[str, Any]] = {}
    for section, keys in YAML_SECTIONS.items():
        values = document[section]
        if not isinstance(values, Mapping) or set(values) != set(keys):
            raise RecipeError(f"run YAML section {section!r} must hold exactly {sorted(keys)}")
        for key, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise RecipeError(f"run YAML {section}.{key} must be finite")
            if not isinstance(value, (str, int, float)):  # bool is an int
                raise RecipeError(f"run YAML {section}.{key} must be a scalar, got {value!r}")
            if isinstance(value, str) and ("{" in value or "}" in value):
                raise RecipeError(f"run YAML {section}.{key} must not contain '{{' or '}}'")
        plain[section] = dict(values)
    return plain


def run_yaml_bytes(document: Mapping[str, Any]) -> bytes:
    """``yaml.safe_dump`` of ``document`` (sorted keys, block style, ASCII), checked to load back.

    The document must hold exactly :data:`YAML_SECTIONS` with scalar values; the bytes must load
    with ``yaml.safe_load`` to an equal document (:class:`RecipeError` otherwise).
    """
    plain = _check_document(document)
    text = yaml.safe_dump(plain, sort_keys=True, default_flow_style=False, allow_unicode=False)
    data = text.encode("ascii")
    loaded = yaml.safe_load(data)
    if loaded != plain or canonical_json(loaded) != canonical_json(plain):
        raise RecipeError("the run YAML does not load back to the same document")
    return data


def write_run_yaml(path: Path, document: Mapping[str, Any]) -> str:
    """Write :func:`run_yaml_bytes` of ``document`` to ``path`` (durable); return their sha256."""
    from ..data.fsutil import atomic_write_bytes

    data = run_yaml_bytes(document)
    atomic_write_bytes(Path(path), data)
    return sha256_bytes(data)


__all__ = [
    "BUDGETS",
    "EARLY_STOPPING_METRICS",
    "MAX_BATCH_SIZE",
    "MAX_EPOCHS",
    "MAX_LEARNING_RATE",
    "MAX_PREDICT_WINDOW",
    "NLL_STEPS",
    "NORMALISATIONS",
    "R1",
    "RECIPE_SCHEMA_VERSION",
    "TEMPLATE_UNUSED",
    "TRAIN_SEED_SCHEME",
    "VAL_SEED_STRING",
    "VENDORED_HARD_CODED",
    "YAML_EXPERIMENT_NAME",
    "YAML_SECTIONS",
    "YAML_SEED_LIMIT",
    "Recipe",
    "RecipeError",
    "TrainingSeeds",
    "YamlPaths",
    "recipe8",
    "recipe_from_mapping",
    "recipe_id",
    "recipe_mapping",
    "run_yaml_bytes",
    "run_yaml_document",
    "train_ratio",
    "training_seeds",
    "write_run_yaml",
]
