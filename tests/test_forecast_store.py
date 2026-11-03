"""Stored fans: the configuration id, the layout, the file format, never overwriting, reuse.

Fans come from :func:`kronos_agent.kronos.decision.forecast_window` with a stand-in model (a
seeded random walk; no weights needed), so they carry real seeds, timestamps and provenance.
Models in the state ``load_models`` leaves Kronos in are imitated with small torch modules (eval
mode, no gradients) shared with a stand-in predictor that carries the clip, context length and
device ``load_models`` builds a ``KronosPredictor`` with; ``load_models`` itself runs once with the
vendored package replaced by such modules. The test marked ``weights`` checks the real models.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes
import torch

from kronos_agent.data import forecast_store as fs
from kronos_agent.data import windows as data_windows
from kronos_agent.data.forecast_store import (
    CONFIG_KEYS,
    META_KEYS,
    ForecastStoreError,
    StoredFan,
    check_reusable,
    config_id,
    context_sha256,
    fan_meta,
    fan_path,
    fan_relpath,
    forecast_config,
    load_fan,
    paths_sha256,
    save_fan,
)
from kronos_agent.data.store import float_values
from kronos_agent.data.timeutil import to_ms
from kronos_agent.data.windows import build_window
from kronos_agent.envinfo import assert_no_personal_data
from kronos_agent.hashing import canonical_json
from kronos_agent.kronos import weights as kronos_weights
from kronos_agent.kronos.batch import (
    FanTask,
    prepare_tasks,
    registered_weights_sha256,
    registry_config,
    run_fan_batch,
)
from kronos_agent.kronos.decision import forecast_window
from kronos_agent.kronos.features import FEATURE_COLUMNS
from kronos_agent.kronos.weights import PREDICTOR_CLIP, LoadedModels, inference_problems
from kronos_agent.manifests import VerifyReport
from kronos_agent.models import spec_for
from kronos_agent.params import load_params
from kronos_agent.paths import REPO_ROOT, default_label
from kronos_agent.provenance import FORECASTING_FILES, forecasting_code_sha256
from kronos_agent.runtime import require_configured
from kronos_agent.seeds import SEED_SCHEME

HOUR_MS = 3_600_000
D = datetime(2025, 3, 3, 12, tzinfo=UTC)
D_MS = to_ms(D)
FIRST_MS = D_MS - 600 * HOUR_MS
BINGX = fakes.synthetic_candles(FIRST_MS, 700, seed=3)
BINANCE = fakes.frame_to_candles(fakes.binance_frame(FIRST_MS, 700, interval="1h", seed=4))
PARAMS = load_params()
KRONOS = PARAMS.kronos
WEIGHTS = {
    "NeoQuasar/Kronos-small/model.safetensors": "a" * 64,
    "NeoQuasar/Kronos-Tokenizer-base/model.safetensors": "b" * 64,
}
# 17 significant digits: the default pandas parser reads it one ulp off; a store must not
VECTOR = float("465655238.20240235")


class WalkPredictor:
    """``predict_batch`` stand-in: a seeded random walk from the last close of each context."""

    def predict_batch(
        self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p, **_
    ):
        frames = []
        for df, y in zip(df_list, y_timestamp_list, strict=True):
            last = float(df["close"].iloc[-1])
            close = last * np.exp(np.cumsum(np.random.normal(0.0, 0.01, size=pred_len)))
            values = np.column_stack(
                [close, close * 1.001, close * 0.999, close, np.ones(pred_len), close]
            )
            frames.append(pd.DataFrame(values, columns=list(FEATURE_COLUMNS), index=y))
        return frames


MODELS = LoadedModels(
    spec=spec_for("small"),
    tokenizer=None,
    model=None,
    predictor=WalkPredictor(),
    weights_sha256=dict(WEIGHTS),
    load_seconds=0.0,
)
CONFIG = forecast_config(MODELS, KRONOS)
CID = config_id(CONFIG)


class BuiltPredictor(WalkPredictor):
    """The stand-in walk, carrying what ``load_models`` builds a ``KronosPredictor`` with."""

    def __init__(self, tokenizer, model, *, clip=PREDICTOR_CLIP, max_context=512, device="cpu"):
        self.tokenizer = tokenizer
        self.model = model
        self.clip = clip
        self.max_context = max_context
        self.device = device


def frozen_module() -> torch.nn.Module:
    """A small torch module in the state ``load_models`` leaves Kronos in: eval, no gradients."""
    module = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Dropout(0.1))
    return module.eval().requires_grad_(False)


def built_models(**predictor_changes) -> LoadedModels:
    """Stand-in models whose tokenizer and model are torch modules shared with the predictor."""
    tokenizer, model = frozen_module(), frozen_module()
    predictor = BuiltPredictor(tokenizer, model, **predictor_changes)
    return replace(MODELS, tokenizer=tokenizer, model=model, predictor=predictor)


def unlike_loaded(case: str) -> LoadedModels:
    """Models that would not forecast as ``load_models`` leaves them, one difference each."""
    if case == "clip":
        return built_models(clip=3)
    if case == "predictor context":
        return built_models(max_context=256)
    if case == "predictor device":
        return built_models(device="mps")
    models = built_models()
    if case == "predictor without a clip":  # it runs torch modules: it must say how it clips
        del models.predictor.clip
    elif case == "tokenizer in training mode":
        models.tokenizer.train()
    elif case == "one submodule in training mode":
        models.model[1].train()  # the dropout layer only
    elif case == "gradients enabled":
        models.model[0].weight.requires_grad_(True)
    elif case == "the predictor's own model":
        models.predictor.model = frozen_module().train()
    elif case == "not a torch module":
        models.tokenizer = object()
    else:
        raise AssertionError(case)
    return models


UNLIKE_LOADED = [
    ("clip", "the predictor's clip is 3, load_models builds it with 5"),
    ("predictor context", "the predictor's max_context is 256"),
    ("predictor device", "the predictor's device is 'mps'"),
    ("predictor without a clip", "the predictor's clip is None, load_models builds it with 5"),
    ("tokenizer in training mode", "the tokenizer is in training mode"),
    ("one submodule in training mode", "the model is in training mode"),
    ("gradients enabled", "the model has parameters with gradients enabled"),
    ("the predictor's own model", "the predictor's model is in training mode"),
    ("not a torch module", "the tokenizer is not a torch module (object)"),
]


def float32_values(strings) -> np.ndarray:
    """Another string-to-float conversion: correctly rounded, then rounded again to float32."""
    return float_values(strings).astype(np.float32).astype(np.float64)


def window_of(candles=BINGX, hour=D, *, source="bingx", **kwargs):
    kwargs.setdefault("horizon", KRONOS.horizon_n)
    return build_window(candles, hour, source=source, **kwargs)


WINDOW = window_of()


def make_fan(window=WINDOW, stream="forecast"):
    return forecast_window(MODELS, window, PARAMS, stream=stream)


def store_fan(root: Path, fan=None, window=WINDOW, stream="forecast", config=CONFIG) -> Path:
    fan = make_fan(window, stream) if fan is None else fan
    path = fan_path(window.source, config_id(config), stream, window.decision_hour, root)
    save_fan(path, fan, window, stream=stream, config=config, params_version=PARAMS.version)
    return path


def write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


UNPICKLED: list[str] = []


def _tripwire() -> str:
    UNPICKLED.append("a pickled object was loaded")
    return "tripwire"


class Tripwire:
    """Unpickling an instance runs :func:`_tripwire`: proof that a file was unpickled."""

    def __reduce__(self):
        return (_tripwire, ())


def arrays_of(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# -- configuration ---------------------------------------------------------------------------------


def test_forecast_config_holds_every_input_that_changes_a_fan():
    assert tuple(CONFIG) == CONFIG_KEYS
    assert CONFIG == {
        "model": "small",
        "weights_sha256": dict(sorted(WEIGHTS.items())),
        "device": "cpu",
        "max_context": 512,
        "predictor_clip": 5,
        "horizon_n": KRONOS.horizon_n,
        "temperature": KRONOS.temperature,
        "top_p": KRONOS.top_p,
        "top_k": KRONOS.top_k,
        "trajectories": KRONOS.trajectories,
        "threads": KRONOS.threads,
        "amount_policy": KRONOS.amount_policy,
        "seed_salt": KRONOS.seed_salt,
        "seed_scheme": SEED_SCHEME,
        "seed_symbol": "BTC-USDT",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": default_label(),
        "forecasting_code_sha256": forecasting_code_sha256(),
    }
    assert list(CONFIG["weights_sha256"]) == sorted(WEIGHTS)
    assert PREDICTOR_CLIP == 5  # upstream's default, the +-5 of SPEC section 3


def test_config_id_is_the_first_16_hex_of_the_canonical_sha256():
    assert CID == hashlib.sha256(canonical_json(CONFIG)).hexdigest()[:16]
    assert re.fullmatch(r"[0-9a-f]{16}", CID)
    assert config_id(dict(reversed(list(CONFIG.items())))) == CID  # key order is irrelevant
    with pytest.raises(ForecastStoreError):
        config_id({"x": object()})
    with pytest.raises(ForecastStoreError):
        config_id([("model", "small")])


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0.8},
        {"top_p": 0.95},
        {"top_k": 1, "top_p": 1.0},
        {"trajectories": 21},
        {"horizon_n": 8},
        {"threads": 4},
        {"amount_policy": "zeros"},
        {"seed_salt": "s1"},
    ],
)
def test_config_id_changes_with_every_governed_input(change):
    changed = forecast_config(MODELS, replace(KRONOS, **change))
    assert config_id(changed) != CID


def test_config_id_changes_with_the_weights_device_libraries_platform_clip_and_code(monkeypatch):
    other_weights = replace(MODELS, weights_sha256={**WEIGHTS, next(iter(WEIGHTS)): "c" * 64})
    assert config_id(forecast_config(other_weights, KRONOS)) != CID
    assert config_id(forecast_config(replace(MODELS, device="mps"), KRONOS)) != CID
    monkeypatch.setattr(fs, "default_label", lambda: "linux-x86_64")
    assert config_id(forecast_config(MODELS, KRONOS)) != CID
    monkeypatch.undo()
    info = replace(require_configured(), torch_version="0.0.0")
    monkeypatch.setattr(fs, "require_configured", lambda: info)
    assert config_id(forecast_config(MODELS, KRONOS)) != CID
    monkeypatch.undo()
    for library in (np, pd):  # the predictor normalises in numpy, its time features are pandas
        monkeypatch.setattr(library, "__version__", "0.0.0")
        assert config_id(forecast_config(MODELS, KRONOS)) != CID
        monkeypatch.undo()
    monkeypatch.setattr(fs, "PREDICTOR_CLIP", 3)
    assert config_id(forecast_config(MODELS, KRONOS)) != CID
    monkeypatch.undo()
    monkeypatch.setattr(fs, "_process_forecasting_code_sha256", lambda: "0" * 64)
    assert config_id(forecast_config(MODELS, KRONOS)) != CID
    monkeypatch.undo()
    assert config_id(forecast_config(MODELS, KRONOS)) == CID


# Edits of the models' construction that no configuration key and no refusal of
# ``inference_problems`` can see: (file, the text replaced, its replacement)
CONSTRUCTION_EDITS = [
    (  # a dynamic int8 quantisation of the model's Linear layers (a CPU speed-up, say)
        "src/kronos_agent/kronos/weights.py",
        "    predictor = api.KronosPredictor(\n",
        (
            "    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, torch.qint8)"
            "\n    predictor = api.KronosPredictor(\n"
        ),
    ),
    (  # other tensors (a fine-tuned checkpoint) under the registry's weight hashes
        "src/kronos_agent/kronos/weights.py",
        "api.Kronos.from_pretrained(str(model_dir), strict=True)",
        'api.Kronos.from_pretrained(str(model_dir / "tuned"), strict=True)',
    ),
    (  # another class than the vendored predictor
        "src/kronos_agent/kronos/vendored.py",
        "KronosPredictor=model.KronosPredictor,",
        "KronosPredictor=getattr(model, 'FastPredictor', model.KronosPredictor),",
    ),
]


@pytest.mark.parametrize(
    "relpath, old, new", CONSTRUCTION_EDITS, ids=["quantised", "other tensors", "other class"]
)
def test_an_edit_of_the_models_construction_gives_a_new_config_id(
    tmp_path, monkeypatch, relpath, old, new
):
    """foundation-1: ``kronos/weights.py`` and ``kronos/vendored.py`` are hashed whole into
    ``config_id``, so an edit of ``load_models`` or ``import_kronos`` never reuses fans of the
    models as they were built before."""
    for item in {*FORECASTING_FILES, relpath}:
        (tmp_path / item).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / item).write_bytes((REPO_ROOT / item).read_bytes())
    code_hash = forecasting_code_sha256(tmp_path)
    assert code_hash == forecasting_code_sha256()  # the copy is the repository's code
    monkeypatch.setattr(fs, "_process_forecasting_code_sha256", lambda: code_hash)
    assert config_id(forecast_config(MODELS, KRONOS)) == CID
    path = tmp_path / relpath
    source = path.read_text(encoding="utf-8")
    assert source.count(old) == 1
    path.write_text(source.replace(old, new), encoding="utf-8")
    edited = forecasting_code_sha256(tmp_path)
    assert edited != code_hash
    monkeypatch.setattr(fs, "_process_forecasting_code_sha256", lambda: edited)
    assert config_id(forecast_config(MODELS, KRONOS)) != CID


def test_forecast_config_refuses_models_that_are_not_the_params_model():
    with pytest.raises(ForecastStoreError, match="differs"):
        forecast_config(MODELS, replace(KRONOS, model="base"))
    with pytest.raises(ForecastStoreError, match="reads 512"):
        forecast_config(MODELS, replace(KRONOS, max_context=256))
    with pytest.raises(ForecastStoreError, match="weights_sha256"):
        forecast_config(replace(MODELS, weights_sha256={}), KRONOS)
    with pytest.raises(TypeError):
        forecast_config(MODELS, PARAMS)


def test_models_as_load_models_leaves_them_have_the_stand_in_configuration():
    models = built_models()
    assert inference_problems(models) == []
    assert forecast_config(models, KRONOS) == CONFIG
    assert inference_problems(MODELS) == []  # no module, no Kronos code: nothing to ask for


@pytest.mark.parametrize("case, named", UNLIKE_LOADED)
def test_forecast_config_refuses_models_that_would_not_forecast_as_loaded(case, named):
    with pytest.raises(ForecastStoreError, match="would not forecast as load_models") as info:
        forecast_config(unlike_loaded(case), KRONOS)
    assert named in str(info.value)


def test_load_models_builds_the_predictor_the_configuration_records(monkeypatch):
    """``load_models`` without weights: the vendored package replaced by small torch modules (in
    training mode with gradients, as freshly built modules are) and a predictor that records its
    arguments."""
    spec = spec_for("small")
    numel: dict[int, int] = {}
    built: list[dict] = []

    def from_pretrained_with(count: int):
        def from_pretrained(path, strict):
            assert strict is True
            module = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Dropout(0.1))
            numel[id(module)] = count
            return module

        return from_pretrained

    class RecordingPredictor:
        def __init__(self, model, tokenizer, **kwargs):
            built.append(kwargs)
            self.model = model
            self.tokenizer = tokenizer
            self.device = kwargs["device"]
            self.max_context = kwargs["max_context"]
            self.clip = kwargs["clip"]

    api = SimpleNamespace(
        KronosTokenizer=SimpleNamespace(
            from_pretrained=from_pretrained_with(spec.tokenizer_state_dict_numel)
        ),
        Kronos=SimpleNamespace(from_pretrained=from_pretrained_with(spec.state_dict_numel)),
        KronosPredictor=RecordingPredictor,
    )
    monkeypatch.setattr(kronos_weights, "verify_model", lambda name, base=None: VerifyReport(4))
    monkeypatch.setattr(kronos_weights, "import_kronos", lambda: api)
    monkeypatch.setattr(kronos_weights, "_check_config", lambda directory, repo: None)
    monkeypatch.setattr(kronos_weights, "_state_dict_numel", lambda module: numel[id(module)])
    models = kronos_weights.load_models("small")
    assert built == [{"device": "cpu", "max_context": 512, "clip": PREDICTOR_CLIP}]
    assert models.predictor.tokenizer is models.tokenizer and models.predictor.model is models.model
    assert inference_problems(models) == []  # eval mode, no gradients
    assert forecast_config(models, KRONOS) == registry_config(KRONOS)  # what a batch prepares


@pytest.mark.weights
def test_the_loaded_kronos_small_forecasts_as_its_configuration_says(loaded_small):
    assert loaded_small.predictor.clip == PREDICTOR_CLIP
    assert inference_problems(loaded_small) == []
    assert forecast_config(loaded_small, KRONOS)["predictor_clip"] == PREDICTOR_CLIP


# -- layout ----------------------------------------------------------------------------------------


def test_fan_path_layout(tmp_path, monkeypatch):
    relpath = f"forecasts/bingx/{CID}/forecast/2025-03-03T12Z.npz"
    assert fan_relpath("bingx", CID, "forecast", D) == relpath
    assert fan_relpath("bingx", CID, "forecast", "2025-03-03T12:00Z") == relpath
    assert fan_path("bingx", CID, "forecast", D, tmp_path) == tmp_path / relpath
    assert fan_path("binance", CID, "mismatch-floor", D, tmp_path) == (
        tmp_path / "forecasts" / "binance" / CID / "mismatch-floor" / "2025-03-03T12Z.npz"
    )
    monkeypatch.setenv("KRONOS_AGENT_DATA_DIR", str(tmp_path / "elsewhere"))
    assert fan_path("bingx", CID, "forecast", D) == (tmp_path / "elsewhere" / relpath).resolve()


@pytest.mark.parametrize(
    "source, cid, stream, hour",
    [
        ("demo", CID, "forecast", D),
        ("bingx", CID.upper(), "forecast", D),
        ("bingx", CID[:15], "forecast", D),
        ("bingx", CID + "0", "forecast", D),
        ("bingx", CID, "a/b", D),
        ("bingx", CID, "..", D),
        ("bingx", CID, "", D),
        ("bingx", CID, "with space", D),
        ("bingx", CID, "forecast", D + timedelta(minutes=30)),
        ("bingx", CID, "forecast", datetime(2025, 3, 3, 12)),  # naive
    ],
)
def test_fan_path_refuses_what_is_not_a_plain_layout_component(source, cid, stream, hour):
    with pytest.raises(ForecastStoreError):
        fan_relpath(source, cid, stream, hour)


# -- saving and loading ----------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    fan = make_fan()
    paths = fan.paths.copy()
    paths[0, 0, 4] = VECTOR
    fan = replace(fan, paths=paths)
    path = store_fan(tmp_path, fan)
    assert path == tmp_path / f"forecasts/bingx/{CID}/forecast/2025-03-03T12Z.npz"
    stored = load_fan(path)
    assert isinstance(stored, StoredFan)
    assert stored.paths.dtype == np.float64 and stored.paths.shape == (20, KRONOS.horizon_n, 6)
    assert stored.paths.tobytes() == fan.paths.tobytes()  # bit for bit
    assert stored.paths[0, 0, 4] == VECTOR
    assert not stored.paths.flags.writeable and not stored.y_open_ms.flags.writeable
    assert stored.y_open_ms.dtype == np.int64
    assert stored.y_open_ms.tolist() == [D_MS + k * HOUR_MS for k in range(KRONOS.horizon_n)]
    meta = stored.meta
    assert tuple(sorted(meta)) == tuple(sorted(META_KEYS))
    assert meta["schema_version"] == 2
    assert (meta["source"], meta["symbol"], meta["stream"]) == ("bingx", "BTC-USDT", "forecast")
    assert (meta["decision_hour"], meta["decision_ms"]) == ("2025-03-03T12:00Z", D_MS)
    assert (meta["window_sha256"], meta["inputs_sha256"]) == (WINDOW.sha256, WINDOW.inputs_sha256)
    assert meta["context_sha256"] == context_sha256(WINDOW)
    assert meta["amount_policy"] == "volume_x_mean_ohlc" and meta["max_context"] == 512
    assert (meta["horizon"], meta["trajectories"]) == (KRONOS.horizon_n, 20)
    assert meta["columns"] == list(FEATURE_COLUMNS)
    assert meta["last_close"] == BINGX[599].close  # the candle opening at D - 1h, as a string
    assert meta["fetch_pairs"] is None
    assert meta["seeds"] == {
        "canonical": fan.seeds.canonical,
        "digest": fan.seeds.digest,
        "torch_seed": fan.seeds.torch_seed,
        "numpy_seed": fan.seeds.numpy_seed,
    }
    assert meta["model"] == "small" and meta["weights_sha256"] == dict(sorted(WEIGHTS.items()))
    assert meta["threads"] == 2 and meta["torch_version"] == torch.__version__
    assert meta["platform"] == default_label()
    assert meta["params_version"] == PARAMS.version
    assert (meta["config_id"], meta["config"]) == (CID, CONFIG)
    assert set(meta["code"]) == {"commit", "dirty", "source_tree_sha256"}
    assert meta["paths_sha256"] == paths_sha256(fan.paths)
    assert meta["elapsed_s"] == fan.elapsed_s
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta["created_at"])
    assert_no_personal_data(meta)
    raw = arrays_of(path)
    assert sorted(raw) == ["meta", "paths", "y_open_ms"]
    assert raw["meta"].dtype.kind == "U" and raw["meta"].ndim == 0
    assert str(raw["meta"][()]) == canonical_json(meta).decode("ascii")


def test_fan_meta_describes_the_fan_without_writing_anything(tmp_path):
    fan = make_fan()
    meta = fan_meta(fan, WINDOW, stream="forecast", config=CONFIG, params_version="abc")
    assert meta["params_version"] == "abc" and meta["config_id"] == CID
    assert list(tmp_path.iterdir()) == []


def test_a_stored_fan_is_never_overwritten(tmp_path):
    path = store_fan(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ForecastStoreError, match="never overwritten") as info:
        store_fan(tmp_path)
    assert f"forecasts/bingx/{CID}/forecast/2025-03-03T12Z.npz" in str(info.value)
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temporary file left


def test_save_puts_a_fan_only_where_its_metadata_says(tmp_path):
    fan = make_fan()
    elsewhere = tmp_path / "fan.npz"
    with pytest.raises(ForecastStoreError, match="stored as"):
        save_fan(elsewhere, fan, WINDOW, stream="forecast", config=CONFIG, params_version="v")
    floor_path = fan_path("bingx", CID, "mismatch-floor", D, tmp_path)
    floor = make_fan(stream="mismatch-floor")
    other_hour = fan_path("bingx", CID, "mismatch-floor", D + timedelta(hours=1), tmp_path)
    with pytest.raises(ForecastStoreError, match="stored as"):
        save_fan(
            other_hour, floor, WINDOW, stream="mismatch-floor", config=CONFIG, params_version="v"
        )
    save_fan(floor_path, floor, WINDOW, stream="mismatch-floor", config=CONFIG, params_version="v")
    assert load_fan(floor_path).meta["stream"] == "mismatch-floor"


def _foreign(case: str):
    """(fan, window, stream, config) where the fan does not belong to the window or config."""
    fan = make_fan()
    if case == "trajectories":
        return fan, WINDOW, "forecast", forecast_config(MODELS, replace(KRONOS, trajectories=21))
    if case == "another hour":
        return fan, window_of(hour=D + timedelta(hours=1)), "forecast", CONFIG
    if case == "another stream":
        return fan, WINDOW, "mismatch-floor", CONFIG
    if case == "seed salt":
        return fan, WINDOW, "forecast", forecast_config(MODELS, replace(KRONOS, seed_salt="s"))
    if case == "model":
        return replace(fan, model_name="base"), WINDOW, "forecast", CONFIG
    if case == "weights":
        return replace(fan, weights_sha256={"x": "0" * 64}), WINDOW, "forecast", CONFIG
    if case == "threads":
        return replace(fan, threads=4), WINDOW, "forecast", CONFIG
    if case == "torch":
        return replace(fan, torch_version="0.0.0"), WINDOW, "forecast", CONFIG
    if case == "nan":
        paths = fan.paths.copy()
        paths[1, 1, 1] = np.nan
        return replace(fan, paths=paths), WINDOW, "forecast", CONFIG
    if case == "float32":
        return replace(fan, paths=fan.paths.astype(np.float32)), WINDOW, "forecast", CONFIG
    if case == "targets":
        later = fan.y_timestamps + pd.Timedelta(hours=1)
        return replace(fan, y_timestamps=later), WINDOW, "forecast", CONFIG
    if case == "zeros window":
        return fan, window_of(amount_policy="zeros"), "forecast", CONFIG
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "trajectories",
        "another hour",
        "another stream",
        "seed salt",
        "model",
        "weights",
        "threads",
        "torch",
        "nan",
        "float32",
        "targets",
        "zeros window",
    ],
)
def test_save_refuses_a_fan_that_does_not_belong_to_its_window_and_config(tmp_path, case):
    fan, window, stream, config = _foreign(case)
    path = fan_path(window.source, config_id(config), stream, window.decision_hour, tmp_path)
    with pytest.raises(ForecastStoreError):
        save_fan(path, fan, window, stream=stream, config=config, params_version="v")
    assert not path.exists()


def test_save_needs_a_params_version_and_a_plain_stream(tmp_path):
    path = fan_path("bingx", CID, "forecast", D, tmp_path)
    with pytest.raises(ForecastStoreError, match="params_version"):
        save_fan(path, make_fan(), WINDOW, stream="forecast", config=CONFIG, params_version="")
    with pytest.raises(ForecastStoreError, match="stream"):
        save_fan(path, make_fan(), WINDOW, stream="a/b", config=CONFIG, params_version="v")


def _damaged(case: str, good: dict[str, np.ndarray]) -> dict[str, np.ndarray] | bytes:
    meta = json.loads(str(good["meta"][()]))

    def with_meta(**changes):
        return {**good, "meta": np.array(canonical_json({**meta, **changes}).decode("ascii"))}

    if case == "pickled object array":
        return {**good, "meta": np.array([Tripwire()], dtype=object)}
    if case == "pickled paths":
        return {**good, "paths": np.array([Tripwire()], dtype=object)}
    if case == "extra array":
        return {**good, "extra": np.zeros(1)}
    if case == "missing array":
        return {"paths": good["paths"], "meta": good["meta"]}
    if case == "tampered paths":
        return {**good, "paths": good["paths"] + 1e-9}
    if case == "float32 paths":
        return {**good, "paths": good["paths"].astype(np.float32)}
    if case == "shifted targets":
        return {**good, "y_open_ms": good["y_open_ms"] + HOUR_MS}
    if case == "int32 targets":
        return {**good, "y_open_ms": good["y_open_ms"].astype(np.int32)}
    if case == "unknown schema":
        return with_meta(schema_version=3)
    if case == "schema 1":  # the format before context_sha256
        return with_meta(schema_version=1)
    if case == "config id":
        return with_meta(config_id="0" * 16)
    if case in ("missing key", "missing context hash"):
        dropped = "seeds" if case == "missing key" else "context_sha256"
        return {
            **good,
            "meta": np.array(
                canonical_json({k: v for k, v in meta.items() if k != dropped}).decode("ascii")
            ),
        }
    if case == "not canonical":
        return {**good, "meta": np.array(json.dumps(meta, indent=1))}
    if case == "meta bytes":
        return {**good, "meta": np.frombuffer(canonical_json(meta), dtype=np.uint8)}
    if case == "truncated":
        buffer = io.BytesIO()
        np.savez(buffer, **good)
        return buffer.getvalue()[: len(buffer.getvalue()) // 2]
    if case == "not a zip":
        return b"\x93NUMPY not really a stored fan"
    if case == "plain npy":
        buffer = io.BytesIO()
        np.save(buffer, good["paths"])
        return buffer.getvalue()
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "pickled object array",
        "pickled paths",
        "extra array",
        "missing array",
        "tampered paths",
        "float32 paths",
        "shifted targets",
        "int32 targets",
        "unknown schema",
        "schema 1",
        "config id",
        "missing key",
        "missing context hash",
        "not canonical",
        "meta bytes",
        "truncated",
        "not a zip",
        "plain npy",
    ],
)
def test_load_fan_refuses_damaged_or_foreign_files(tmp_path, case):
    path = store_fan(tmp_path)
    good = arrays_of(path)
    damaged = _damaged(case, good)
    path.unlink()
    if isinstance(damaged, bytes):
        path.write_bytes(damaged)
    else:
        write_npz(path, **damaged)
    with pytest.raises(ForecastStoreError) as info:
        load_fan(path)
    assert "2025-03-03T12Z.npz" in str(info.value)
    assert UNPICKLED == []  # allow_pickle=False: nothing in a file is ever unpickled


def test_load_fan_of_a_missing_file(tmp_path):
    with pytest.raises(ForecastStoreError, match="cannot read"):
        load_fan(fan_path("bingx", CID, "forecast", D, tmp_path))


# -- reuse -----------------------------------------------------------------------------------------


def test_a_stored_fan_is_reusable_for_the_same_window(tmp_path):
    stored = load_fan(store_fan(tmp_path))
    check_reusable(stored, window_of(list(reversed(BINGX))), stream="forecast")  # same candles
    later_changed = [replace(c, close="1.0") if c.open_ms >= D_MS else c for c in BINGX]
    check_reusable(stored, window_of(later_changed), stream="forecast")  # nothing at or after D


def _changed_window(case: str):
    if case == "context candle":
        candles = [
            replace(c, volume="1.0") if c.open_ms == D_MS - 100 * HOUR_MS else c for c in BINGX
        ]
        return window_of(candles), "forecast", ["window sha256", "inputs_sha256"]
    if case == "first context candle":
        candles = [replace(c, low="1.0") if c.open_ms == D_MS - 512 * HOUR_MS else c for c in BINGX]
        return window_of(candles), "forecast", ["window sha256", "inputs_sha256"]
    if case == "stream":
        return WINDOW, "mismatch-floor", ["stream"]
    if case == "hour":
        return window_of(hour=D + timedelta(hours=1)), "forecast", ["decision hour", "target opens"]
    if case == "source":
        return window_of(BINANCE, source="binance"), "forecast", ["source"]
    if case == "amount policy":
        return window_of(amount_policy="zeros"), "forecast", ["inputs_sha256", "amount_policy"]
    if case == "horizon":
        return window_of(horizon=8), "forecast", ["horizon", "target opens"]
    if case == "context length":
        return window_of(max_context=256), "forecast", ["window sha256", "max_context"]
    if case == "one context float":  # the same strings, one float one ulp away
        context = WINDOW.context.copy()
        context.loc[0, "close"] = np.nextafter(context.loc[0, "close"], np.inf)
        return replace(WINDOW, context=context), "forecast", ["context_sha256"]
    if case == "context timestamps":
        later = WINDOW.x_timestamps + pd.Timedelta(hours=1)
        return replace(WINDOW, x_timestamps=later), "forecast", ["context_sha256"]
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "context candle",
        "first context candle",
        "stream",
        "hour",
        "source",
        "amount policy",
        "horizon",
        "context length",
        "one context float",
        "context timestamps",
    ],
)
def test_check_reusable_refuses_a_changed_window_and_names_the_file(tmp_path, case):
    path = store_fan(tmp_path)
    before = path.read_bytes()
    stored = load_fan(path)
    window, stream, fields = _changed_window(case)
    with pytest.raises(ForecastStoreError) as info:
        check_reusable(stored, window, stream=stream)
    message = str(info.value)
    assert f"forecasts/bingx/{CID}/forecast/2025-03-03T12Z.npz" in message
    assert "never overwritten" in message
    for field in fields:
        assert field in message
    if case == "amount policy":
        assert "window sha256" not in message  # the same candles
    if case in ("one context float", "context timestamps"):
        assert "window sha256" not in message and "inputs_sha256" not in message
    assert path.read_bytes() == before


def test_context_sha256_covers_exactly_the_floats_and_timestamps_the_model_reads():
    header = {
        "columns": list(FEATURE_COLUMNS),
        "shape": [512, 6],
        "x_timestamps": 512,
        "y_timestamps": KRONOS.horizon_n,
    }
    expected = hashlib.sha256(
        canonical_json(header)
        + np.ascontiguousarray(WINDOW.context.to_numpy(dtype=np.float64), dtype="<f8").tobytes()
        + WINDOW.x_timestamps.to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes()
        + WINDOW.y_timestamps.to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes()
    ).hexdigest()
    assert context_sha256(WINDOW) == expected
    assert context_sha256(window_of(list(reversed(BINGX)))) == expected  # the same candles
    later_changed = [replace(c, close="1.0") if c.open_ms >= D_MS else c for c in BINGX]
    assert context_sha256(window_of(later_changed)) == expected  # nothing at or after D
    renamed = WINDOW.context.rename(columns={"amount": "quote_volume"})  # the same bytes
    later_targets = WINDOW.y_timestamps + pd.Timedelta(hours=1)
    for changed in (replace(WINDOW, context=renamed), replace(WINDOW, y_timestamps=later_targets)):
        assert context_sha256(changed) != expected
    with pytest.raises(ForecastStoreError):
        context_sha256(WINDOW.context)
    with pytest.raises(ForecastStoreError, match="tz-naive"):
        context_sha256(replace(WINDOW, x_timestamps=WINDOW.x_timestamps.dt.tz_localize("UTC")))


def test_check_reusable_refuses_a_fan_made_from_other_floats_of_the_same_strings(
    tmp_path, monkeypatch
):
    path = store_fan(tmp_path)
    before = path.read_bytes()
    stored = load_fan(path)
    monkeypatch.setattr(data_windows, "float_values", float32_values)
    window = window_of()  # the same candles through another string-to-float conversion
    assert (window.sha256, window.inputs_sha256) == (WINDOW.sha256, WINDOW.inputs_sha256)
    assert not window.context.equals(WINDOW.context)
    with pytest.raises(ForecastStoreError, match="context_sha256") as info:
        check_reusable(stored, window, stream="forecast")
    assert "window sha256" not in str(info.value) and "inputs_sha256" not in str(info.value)
    assert path.read_bytes() == before
    monkeypatch.undo()
    check_reusable(stored, window_of(), stream="forecast")  # the conversion the fan was made with


def test_a_batch_refuses_a_stored_fan_made_otherwise_under_the_same_config_id(
    tmp_path, monkeypatch
):
    """The silent reuse the configuration used to allow: a stored fan, then the same batch with
    another predictor clip, a module in training mode or another string-to-float conversion.
    Each is refused before any fan runs; the stored fan is never reused or overwritten."""
    fakes.write_bingx_store(tmp_path, BINGX)
    tasks = [FanTask("bingx", D)]
    registered = registered_weights_sha256("small")
    models = replace(MODELS, weights_sha256=registered)
    prepared = prepare_tasks(tasks, params=PARAMS, root=tmp_path)
    (summary,) = run_fan_batch(prepared, params=PARAMS, root=tmp_path, models=models)
    path = tmp_path / summary.relpath
    before = path.read_bytes()
    prepared = prepare_tasks(tasks, params=PARAMS, root=tmp_path)
    assert prepared.pending == 0  # reusable as it is
    clipped = replace(unlike_loaded("clip"), weights_sha256=registered)
    training = replace(unlike_loaded("tokenizer in training mode"), weights_sha256=registered)
    for other, named in ((clipped, "clip"), (training, "training mode")):
        with pytest.raises(ForecastStoreError, match=named):
            run_fan_batch(prepared, params=PARAMS, root=tmp_path, models=other)
    monkeypatch.setattr(data_windows, "float_values", float32_values)
    with pytest.raises(ForecastStoreError, match="context_sha256") as info:
        prepare_tasks(tasks, params=PARAMS, root=tmp_path)
    assert summary.relpath in str(info.value)
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


@pytest.mark.parametrize("change", ["numpy", "pandas", "clip"])
def test_a_batch_under_other_libraries_or_another_clip_computes_into_another_directory(
    tmp_path, monkeypatch, change
):
    fakes.write_bingx_store(tmp_path, BINGX)
    tasks = [FanTask("bingx", D)]
    models = replace(MODELS, weights_sha256=registered_weights_sha256("small"))
    first = prepare_tasks(tasks, params=PARAMS, root=tmp_path)
    (stored,) = run_fan_batch(first, params=PARAMS, root=tmp_path, models=models)
    before = (tmp_path / stored.relpath).read_bytes()
    if change == "clip":
        monkeypatch.setattr(fs, "PREDICTOR_CLIP", 3)
    else:
        monkeypatch.setattr(np if change == "numpy" else pd, "__version__", "0.0.0")
    again = prepare_tasks(tasks, params=PARAMS, root=tmp_path)
    assert again.config_id != first.config_id and again.pending == 1  # nothing to reuse
    (computed,) = run_fan_batch(again, params=PARAMS, root=tmp_path, models=models)
    assert computed.computed
    assert computed.relpath == fan_relpath("bingx", again.config_id, "forecast", D)
    assert (tmp_path / stored.relpath).read_bytes() == before


def test_check_reusable_argument_types(tmp_path):
    stored = load_fan(store_fan(tmp_path))
    with pytest.raises(ForecastStoreError):
        check_reusable(stored.meta, WINDOW, stream="forecast")
    with pytest.raises(ForecastStoreError):
        check_reusable(stored, WINDOW.context, stream="forecast")
    with pytest.raises(ForecastStoreError):
        check_reusable(stored, WINDOW, stream="../forecast")
