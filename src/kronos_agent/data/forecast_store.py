"""Stored Kronos fans: one ``.npz`` per source, configuration, stream and decision hour.

Layout under the data root (``docs/DATA.md``, "Forecasts"):
``forecasts/<source>/<config_id>/<stream>/<YYYY-MM-DDTHH>Z.npz``. ``config_id`` is the first 16
hex characters of the sha256 of the canonical JSON of :func:`forecast_config`, which holds every
input of a configuration that changes a fan: the model and its weight hashes, the device,
``max_context``, the predictor's clip, ``horizon_n``, the sampling parameters, the number of
trajectories, the thread count, the ``amount`` policy, the seed salt, scheme and symbol, the
torch, numpy and pandas versions, the platform and the sha256 of the forecasting code (the
construction of the models in ``kronos/weights.py`` and ``kronos/vendored.py`` and the
vendored-source manifest included, :func:`kronos_agent.provenance.forecasting_code_sha256`).
Models that would not forecast as :func:`~kronos_agent.kronos.weights.load_models` leaves them
(another clip, a module in training mode or with gradients enabled) are refused, not described.
Fans of different configurations therefore never share a directory; what each fan read of its
window is recorded with it and compared on every reuse (below).

A file holds three arrays and nothing pickled (it is read with ``allow_pickle=False``):
``paths`` (``K x N x 6`` float64, channels in ``FEATURE_COLUMNS`` order), ``y_open_ms`` (int64,
the target opens ``D, ..., D + (N - 1) h``) and ``meta``, one canonical-JSON string: source,
``D``, stream, the window's ``sha256`` and ``inputs_sha256``, the ``context_sha256`` of the
floats and timestamps the model read (:func:`context_sha256`), seeds, model, weight hashes,
threads, torch version, platform, params version, ``amount`` policy, the config and its id, the
code, the sha256 of ``paths``, elapsed seconds and creation time. :func:`save_fan` refuses a fan
that does not belong under its config (another model, thread count, torch build, seeds or
target hours) and publishes with ``os.link`` (:func:`kronos_agent.data.fsutil.publish_new`), so
an existing file is never replaced. A stored fan is reused only when its metadata matches the
current window's hashes, ``D``, source, stream and ``amount`` policy (:func:`check_reusable`);
the window hashes cover the candle strings and ``context_sha256`` the floats and timestamps made
from them, so a change in how strings become floats (``data/store.py``, outside the hashed
forecasting code) is refused on reuse wherever its code lives, never reused silently. Otherwise
:class:`ForecastStoreError` names the file and leaves it in place.

The code provenance and the forecasting-code hash are read once per process: a running process
executes the code it imported, whatever happens to the files afterwards.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..hashing import canonical_json, sha256_bytes
from ..kronos.features import FEATURE_COLUMNS
from ..kronos.forecast import Fan
from ..kronos.weights import PREDICTOR_CLIP, LoadedModels, inference_problems
from ..params import KronosParams
from ..paths import LABEL_RE, data_dir, default_label
from ..provenance import code_version, forecasting_code_sha256
from ..runtime import require_configured
from ..seeds import SEED_SCHEME, seed_for_decision
from .fsutil import publish_new
from .store import SOURCE_SYMBOLS
from .timeutil import HOUR_MS, hour_key, parse_utc_hour, to_ms, utc_text
from .windows import DecisionWindow

FAN_SCHEMA_VERSION = 2
FAN_ARRAYS: tuple[str, ...] = ("paths", "y_open_ms", "meta")
FORECASTS_DIR = "forecasts"
CONFIG_ID_LENGTH = 16
# params.yaml pins ``symbol`` to BTC-USDT; seeds use it for every source (docs/DATA.md)
SEED_SYMBOL = "BTC-USDT"
CONFIG_KEYS: tuple[str, ...] = (
    "model",
    "weights_sha256",
    "device",
    "max_context",
    "predictor_clip",
    "horizon_n",
    "temperature",
    "top_p",
    "top_k",
    "trajectories",
    "threads",
    "amount_policy",
    "seed_salt",
    "seed_scheme",
    "seed_symbol",
    "torch_version",
    "numpy_version",
    "pandas_version",
    "platform",
    "forecasting_code_sha256",
)
META_KEYS: tuple[str, ...] = (
    "schema_version",
    "source",
    "symbol",
    "decision_hour",
    "decision_ms",
    "stream",
    "window_sha256",
    "inputs_sha256",
    "context_sha256",
    "amount_policy",
    "max_context",
    "horizon",
    "trajectories",
    "columns",
    "last_close",
    "fetch_pairs",
    "seeds",
    "model",
    "weights_sha256",
    "threads",
    "torch_version",
    "platform",
    "params_version",
    "config_id",
    "config",
    "code",
    "paths_sha256",
    "elapsed_s",
    "created_at",
)

_CONFIG_ID_RE = re.compile(r"[0-9a-f]{16}")
_NS_PER_MS = 1_000_000


class ForecastStoreError(Exception):
    """A fan that cannot be stored, or a stored fan that cannot be read or reused."""


@dataclass(frozen=True, eq=False)
class StoredFan:
    """A fan read back from its file: read-only arrays and the parsed metadata.

    ``eq=False``: arrays have no truth value, so equality is identity (as for ``DecisionWindow``).
    """

    paths: np.ndarray
    y_open_ms: np.ndarray
    meta: dict[str, Any]


@functools.cache
def _process_forecasting_code_sha256() -> str:
    return forecasting_code_sha256()


@functools.cache
def _process_code_version() -> dict[str, Any]:
    return code_version()


# -- configuration ---------------------------------------------------------------------------------


def forecast_config(models: LoadedModels, kronos: KronosParams) -> dict[str, Any]:
    """Every input that changes a fan made with ``models`` under ``kronos`` (:data:`CONFIG_KEYS`).

    The torch version is the runtime's (``configure_runtime`` must have run), the numpy and
    pandas versions are the imported ones (the predictor normalises in numpy and derives its time
    features in pandas) and the platform is ``<sys.platform>-<machine>``. ``models`` needs
    ``spec``, ``weights_sha256`` and ``device``; its model and context length must be the ones
    ``kronos`` names. ``predictor_clip`` is
    :data:`~kronos_agent.kronos.weights.PREDICTOR_CLIP`, the clip ``load_models`` builds every
    predictor with; a loaded predictor's own clip is read back and must equal it. Models whose
    predictor, tokenizer or model would forecast otherwise (another clip, context length or
    device, training mode, gradients enabled) are refused
    (:func:`~kronos_agent.kronos.weights.inference_problems`), so the configuration describes
    every fan they make. Models without any torch module run no Kronos code (the registry
    description of :mod:`kronos_agent.kronos.batch`, the stand-ins of the tests): their
    predictor is not asked for a clip it does not have.
    """
    if not isinstance(kronos, KronosParams):
        raise TypeError(f"expected KronosParams, got {type(kronos).__name__}")
    spec = models.spec
    if spec.name != kronos.model:
        raise ForecastStoreError(f"loaded model {spec.name!r} differs from params {kronos.model!r}")
    if spec.max_context != kronos.max_context:
        raise ForecastStoreError(
            f"model {spec.name!r} reads {spec.max_context} candles, params say {kronos.max_context}"
        )
    weights = dict(models.weights_sha256)
    if not weights or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in weights.items()
    ):
        raise ForecastStoreError("weights_sha256 must map every weight file to its sha256 hex")
    problems = inference_problems(models)
    if problems:
        raise ForecastStoreError(
            "the models would not forecast as load_models leaves them: " + "; ".join(problems)
        )
    return {
        "model": kronos.model,
        "weights_sha256": {key: weights[key] for key in sorted(weights)},
        "device": str(models.device),
        "max_context": kronos.max_context,
        "predictor_clip": PREDICTOR_CLIP,
        "horizon_n": kronos.horizon_n,
        "temperature": kronos.temperature,
        "top_p": kronos.top_p,
        "top_k": kronos.top_k,
        "trajectories": kronos.trajectories,
        "threads": kronos.threads,
        "amount_policy": kronos.amount_policy,
        "seed_salt": kronos.seed_salt,
        "seed_scheme": SEED_SCHEME,
        "seed_symbol": SEED_SYMBOL,
        "torch_version": require_configured().torch_version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": default_label(),
        "forecasting_code_sha256": _process_forecasting_code_sha256(),
    }


def config_id(config: Mapping[str, Any]) -> str:
    """First 16 hex characters of ``sha256(canonical_json(config))``."""
    if not isinstance(config, Mapping):
        raise ForecastStoreError(f"a forecast config is a mapping, got {type(config).__name__}")
    try:
        encoded = canonical_json(dict(config))
    except (TypeError, ValueError) as exc:
        raise ForecastStoreError(f"forecast config is not JSON-serialisable: {exc}") from exc
    return sha256_bytes(encoded)[:CONFIG_ID_LENGTH]


# -- layout ----------------------------------------------------------------------------------------


def check_stream(stream: str) -> str:
    """A stream names a seed family and a directory: a plain label such as ``mismatch-floor``."""
    if not isinstance(stream, str) or not LABEL_RE.fullmatch(stream):
        raise ForecastStoreError(f"stream must match {LABEL_RE.pattern}, got {stream!r}")
    return stream


def fan_relpath(source: str, config_id: str, stream: str, decision_hour: datetime) -> str:
    """``forecasts/<source>/<config_id>/<stream>/<YYYY-MM-DDTHH>Z.npz`` (relative, POSIX)."""
    if source not in SOURCE_SYMBOLS:
        raise ForecastStoreError(
            f"unknown source {source!r}; expected one of {list(SOURCE_SYMBOLS)}"
        )
    if not isinstance(config_id, str) or not _CONFIG_ID_RE.fullmatch(config_id):
        raise ForecastStoreError(
            f"config_id must be {CONFIG_ID_LENGTH} lowercase hex characters, got {config_id!r}"
        )
    check_stream(stream)
    try:
        key = hour_key(decision_hour)
    except ValueError as exc:
        raise ForecastStoreError(f"invalid decision hour: {exc}") from exc
    return f"{FORECASTS_DIR}/{source}/{config_id}/{stream}/{key}.npz"


def fan_path(
    source: str,
    config_id: str,
    stream: str,
    decision_hour: datetime,
    root: Path | None = None,
) -> Path:
    """The file of one fan under the data root (``root``, default ``data_dir()``)."""
    base = Path(root) if root is not None else data_dir()
    return base / fan_relpath(source, config_id, stream, decision_hour)


def _file_label(path: Path) -> str:
    """The data-root-relative name of a fan file when it follows the layout, else its name."""
    parts = Path(path).parts
    if len(parts) >= 5 and parts[-5] == FORECASTS_DIR:
        return "/".join(parts[-5:])
    return Path(path).name


def _meta_label(meta: Mapping[str, Any]) -> str:
    try:
        return fan_relpath(
            meta["source"], meta["config_id"], meta["stream"], parse_utc_hour(meta["decision_hour"])
        )
    except (KeyError, TypeError, ValueError, ForecastStoreError):
        return "stored fan (unreadable location)"


# -- metadata --------------------------------------------------------------------------------------


def paths_sha256(paths: np.ndarray) -> str:
    """sha256 of a fan's shape and little-endian float64 bytes (equal digests: identical fans)."""
    array = np.ascontiguousarray(paths, dtype="<f8")
    digest = hashlib.sha256(repr(tuple(int(n) for n in array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _nanoseconds(timestamps: pd.Series, label: str) -> np.ndarray:
    """Little-endian int64 nanoseconds of a tz-naive ``datetime64`` Series."""
    if not isinstance(timestamps, pd.Series) or not pd.api.types.is_datetime64_dtype(
        timestamps.dtype
    ):
        raise ForecastStoreError(f"{label} must be a tz-naive datetime64 Series")
    return timestamps.to_numpy(dtype="datetime64[ns]").astype("<i8")


def context_sha256(window: DecisionWindow) -> str:
    """sha256 of the floats and timestamps the model reads for ``window``.

    One sha256 fed with the canonical JSON of the context's column names and shape and of the
    timestamp counts, then the context as C-ordered little-endian float64 bytes, then
    ``x_timestamps`` and ``y_timestamps`` as little-endian int64 nanoseconds. The window's
    ``sha256`` and ``inputs_sha256`` cover the candle strings; this digest covers the floats made
    from them, whatever code made them.
    """
    if not isinstance(window, DecisionWindow):
        raise ForecastStoreError(f"expected a DecisionWindow, got {type(window).__name__}")
    context = window.context
    if not isinstance(context, pd.DataFrame):
        raise ForecastStoreError("a window's context must be a DataFrame")
    values = np.ascontiguousarray(context.to_numpy(dtype=np.float64), dtype="<f8")
    x_ns = _nanoseconds(window.x_timestamps, "window x_timestamps")
    y_ns = _nanoseconds(window.y_timestamps, "window y_timestamps")
    header = {
        "columns": [str(name) for name in context.columns],
        "shape": [int(n) for n in values.shape],
        "x_timestamps": int(x_ns.size),
        "y_timestamps": int(y_ns.size),
    }
    digest = hashlib.sha256(canonical_json(header))
    for array in (values, x_ns, y_ns):
        digest.update(array.tobytes())
    return digest.hexdigest()


def _y_open_ms(y_timestamps: pd.Series) -> np.ndarray:
    nanoseconds = _nanoseconds(y_timestamps, "fan y_timestamps")
    if (nanoseconds % _NS_PER_MS).any():
        raise ForecastStoreError("fan y_timestamps are not whole milliseconds")
    return nanoseconds // _NS_PER_MS


def _target_opens(decision_ms: int, horizon: int) -> list[int]:
    return [decision_ms + step * HOUR_MS for step in range(horizon)]


def _check_fan(
    fan: Fan, window: DecisionWindow, *, stream: str, config: Mapping[str, Any]
) -> np.ndarray:
    """Refuse a fan that does not belong to ``window``, ``stream`` and ``config``; return its
    target opens."""
    if not isinstance(fan, Fan):
        raise ForecastStoreError(f"expected a Fan, got {type(fan).__name__}")
    if not isinstance(window, DecisionWindow):
        raise ForecastStoreError(f"expected a DecisionWindow, got {type(window).__name__}")
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise ForecastStoreError(f"forecast config misses {missing}")
    paths = fan.paths
    if (
        not isinstance(paths, np.ndarray)
        or paths.dtype != np.float64
        or paths.ndim != 3
        or paths.shape[2] != len(FEATURE_COLUMNS)
    ):
        raise ForecastStoreError("fan paths must be a float64 array of shape (K, N, 6)")
    if not np.isfinite(paths).all():
        raise ForecastStoreError("fan paths contain NaN or infinite values")
    if tuple(fan.columns) != FEATURE_COLUMNS:
        raise ForecastStoreError(f"fan columns {fan.columns} != {FEATURE_COLUMNS}")
    decision_ms = to_ms(window.decision_hour)
    y_open = _y_open_ms(fan.y_timestamps)
    expectations: dict[str, tuple[Any, Any]] = {
        "trajectories": (int(paths.shape[0]), config["trajectories"]),
        "horizon": (int(paths.shape[1]), config["horizon_n"]),
        "window horizon": (window.horizon, config["horizon_n"]),
        "max_context": (window.max_context, config["max_context"]),
        "amount_policy": (window.amount_policy, config["amount_policy"]),
        "model": (fan.model_name, config["model"]),
        "weights_sha256": (dict(fan.weights_sha256), dict(config["weights_sha256"])),
        "threads": (fan.threads, config["threads"]),
        "torch_version": (fan.torch_version, config["torch_version"]),
        "target opens": (y_open.tolist(), _target_opens(decision_ms, window.horizon)),
    }
    problems = [name for name, (actual, expected) in expectations.items() if actual != expected]
    try:
        expected_seeds = seed_for_decision(
            window.decision_hour,
            stream=stream,
            symbol=config["seed_symbol"],
            salt=config["seed_salt"],
        )
    except ValueError:
        expected_seeds = None
    if fan.seeds != expected_seeds:
        problems.append("seeds")
    if problems:
        raise ForecastStoreError(
            f"the fan of the {window.source} window of {utc_text(decision_ms)} (stream {stream}) "
            f"does not belong to forecast config {config_id(config)}: {', '.join(problems)} differ"
        )
    return y_open


def fan_meta(
    fan: Fan,
    window: DecisionWindow,
    *,
    stream: str,
    config: Mapping[str, Any],
    params_version: str,
) -> dict[str, Any]:
    """The metadata stored with a fan (:data:`META_KEYS`); the fan must belong to ``config``."""
    stream = check_stream(stream)
    if not isinstance(params_version, str) or not params_version:
        raise ForecastStoreError("params_version must be a non-empty string")
    _check_fan(fan, window, stream=stream, config=config)
    decision_ms = to_ms(window.decision_hour)
    seeds = fan.seeds
    return {
        "schema_version": FAN_SCHEMA_VERSION,
        "source": window.source,
        "symbol": window.symbol,
        "decision_hour": utc_text(decision_ms),
        "decision_ms": decision_ms,
        "stream": stream,
        "window_sha256": window.sha256,
        "inputs_sha256": window.inputs_sha256,
        "context_sha256": context_sha256(window),
        "amount_policy": window.amount_policy,
        "max_context": window.max_context,
        "horizon": fan.horizon,
        "trajectories": fan.trajectories,
        "columns": list(fan.columns),
        "last_close": window.candles[-1].close,
        "fetch_pairs": window.fetch_pairs,
        "seeds": {
            "canonical": seeds.canonical,
            "digest": seeds.digest,
            "torch_seed": seeds.torch_seed,
            "numpy_seed": seeds.numpy_seed,
        },
        "model": fan.model_name,
        "weights_sha256": {key: fan.weights_sha256[key] for key in sorted(fan.weights_sha256)},
        "threads": fan.threads,
        "torch_version": fan.torch_version,
        "platform": config["platform"],
        "params_version": params_version,
        "config_id": config_id(config),
        "config": dict(config),
        "code": dict(_process_code_version()),
        "paths_sha256": paths_sha256(fan.paths),
        "elapsed_s": float(fan.elapsed_s),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# -- files -----------------------------------------------------------------------------------------


def save_fan(
    path: Path,
    fan: Fan,
    window: DecisionWindow,
    *,
    stream: str,
    config: Mapping[str, Any],
    params_version: str,
) -> None:
    """Publish ``fan`` at ``path`` (``np.savez`` into an open in-memory handle, then
    :func:`~kronos_agent.data.fsutil.publish_new`); an existing file is never overwritten.

    ``path`` must be the layout's file for this fan (:func:`fan_path`), so that a file always
    sits where its metadata says.
    """
    path = Path(path)
    meta = fan_meta(fan, window, stream=stream, config=config, params_version=params_version)
    expected = fan_relpath(meta["source"], meta["config_id"], stream, window.decision_hour)
    if _file_label(path) != expected:
        raise ForecastStoreError(
            f"a fan of this window, stream and config is stored as {expected}, not at "
            f"{_file_label(path)}"
        )
    buffer = io.BytesIO()
    np.savez(
        buffer,
        paths=np.ascontiguousarray(fan.paths, dtype=np.float64),
        y_open_ms=np.asarray(_target_opens(meta["decision_ms"], fan.horizon), dtype=np.int64),
        meta=np.array(canonical_json(meta).decode("ascii")),
    )
    try:
        publish_new(path, buffer.getvalue())
    except FileExistsError:
        raise ForecastStoreError(
            f"{expected} already exists; stored fans are never overwritten"
        ) from None


def _parse_meta(array: np.ndarray, label: str) -> dict[str, Any]:
    if array.ndim != 0 or array.dtype.kind != "U":
        raise ForecastStoreError(f"{label}: meta is not a JSON string")
    text = str(array[()])
    try:
        meta = json.loads(text)
    except ValueError as exc:
        raise ForecastStoreError(f"{label}: meta is not valid JSON") from exc
    if not isinstance(meta, dict):
        raise ForecastStoreError(f"{label}: meta is not a JSON object")
    if canonical_json(meta).decode("ascii") != text:
        raise ForecastStoreError(f"{label}: meta is not canonical JSON")
    if meta.get("schema_version") != FAN_SCHEMA_VERSION:
        raise ForecastStoreError(f"{label}: unknown fan schema {meta.get('schema_version')!r}")
    missing = [key for key in META_KEYS if key not in meta]
    if missing:
        raise ForecastStoreError(f"{label}: meta misses {missing}")
    return meta


def load_fan(path: Path) -> StoredFan:
    """Read a stored fan (``np.load(allow_pickle=False)``) and check it against its metadata.

    Checked: exactly the arrays ``paths``, ``y_open_ms`` and ``meta``; ``paths`` float64 of the
    recorded shape, finite and matching the recorded sha256; ``y_open_ms`` int64 equal to
    ``D, ..., D + (N - 1) h``; the recorded ``config_id`` equal to the id of the recorded config.
    Any problem raises :class:`ForecastStoreError` naming the file.
    """
    path = Path(path)
    label = _file_label(path)
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ForecastStoreError(f"cannot read stored fan {label}: {exc}") from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ForecastStoreError(f"{label} is not a stored fan (.npz archive)")
    try:
        with loaded:
            names = sorted(loaded.files)
            if names != sorted(FAN_ARRAYS):
                raise ForecastStoreError(f"{label}: arrays {names} != {sorted(FAN_ARRAYS)}")
            paths = loaded["paths"]
            y_open = loaded["y_open_ms"]
            meta = _parse_meta(loaded["meta"], label)
    except ForecastStoreError:
        raise
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ForecastStoreError(f"cannot read stored fan {label}: {exc}") from exc
    shape = (meta["trajectories"], meta["horizon"], len(FEATURE_COLUMNS))
    if paths.dtype != np.float64 or paths.shape != shape:
        raise ForecastStoreError(f"{label}: paths are not float64 of shape {shape}")
    if not np.isfinite(paths).all():
        raise ForecastStoreError(f"{label}: paths contain NaN or infinite values")
    if paths_sha256(paths) != meta["paths_sha256"]:
        raise ForecastStoreError(f"{label}: paths do not match the recorded sha256")
    if meta["columns"] != list(FEATURE_COLUMNS):
        raise ForecastStoreError(f"{label}: unexpected columns {meta['columns']}")
    expected_open = _target_opens(meta["decision_ms"], meta["horizon"])
    if y_open.dtype != np.int64 or y_open.tolist() != expected_open:
        raise ForecastStoreError(f"{label}: y_open_ms are not the {meta['horizon']} target opens")
    if config_id(meta["config"]) != meta["config_id"]:
        raise ForecastStoreError(f"{label}: config_id does not match the recorded config")
    paths.setflags(write=False)
    y_open.setflags(write=False)
    return StoredFan(paths=paths, y_open_ms=y_open, meta=meta)


def check_reusable(stored: StoredFan, window: DecisionWindow, *, stream: str) -> None:
    """Refuse to reuse ``stored`` for ``window`` and ``stream`` unless its metadata matches.

    Compared: source, ``D``, stream, the window's ``sha256`` and ``inputs_sha256``, the
    :func:`context_sha256` of its floats and timestamps (equal strings made into other floats are
    refused), the ``amount`` policy, ``max_context``, the horizon and the target opens. A
    mismatch raises :class:`ForecastStoreError` naming the file and the differing fields; the
    file is kept.
    """
    if not isinstance(stored, StoredFan):
        raise ForecastStoreError(f"expected a StoredFan, got {type(stored).__name__}")
    if not isinstance(window, DecisionWindow):
        raise ForecastStoreError(f"expected a DecisionWindow, got {type(window).__name__}")
    stream = check_stream(stream)
    meta = stored.meta
    decision_ms = to_ms(window.decision_hour)
    comparisons: dict[str, tuple[Any, Any]] = {
        "source": (meta.get("source"), window.source),
        "decision hour": (meta.get("decision_ms"), decision_ms),
        "stream": (meta.get("stream"), stream),
        "window sha256": (meta.get("window_sha256"), window.sha256),
        "inputs_sha256": (meta.get("inputs_sha256"), window.inputs_sha256),
        "context_sha256": (meta.get("context_sha256"), context_sha256(window)),
        "amount_policy": (meta.get("amount_policy"), window.amount_policy),
        "max_context": (meta.get("max_context"), window.max_context),
        "horizon": (int(stored.paths.shape[1]), window.horizon),
        "target opens": (
            np.asarray(stored.y_open_ms).tolist(),
            _target_opens(decision_ms, window.horizon),
        ),
    }
    differing = [name for name, (have, want) in comparisons.items() if have != want]
    if differing:
        raise ForecastStoreError(
            f"stored fan {_meta_label(meta)} cannot be reused for the {window.source} window of "
            f"{utc_text(decision_ms)} (stream {stream}): {', '.join(differing)} differ; the file "
            "is never overwritten"
        )


__all__ = [
    "CONFIG_ID_LENGTH",
    "CONFIG_KEYS",
    "FAN_ARRAYS",
    "FAN_SCHEMA_VERSION",
    "META_KEYS",
    "SEED_SYMBOL",
    "ForecastStoreError",
    "StoredFan",
    "check_reusable",
    "check_stream",
    "config_id",
    "context_sha256",
    "fan_meta",
    "fan_path",
    "fan_relpath",
    "forecast_config",
    "load_fan",
    "paths_sha256",
    "save_fan",
]
