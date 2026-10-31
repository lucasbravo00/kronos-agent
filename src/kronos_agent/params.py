"""Strict loader for ``params.yaml`` (SPEC sections 4, 9 and 12d).

Unknown or missing keys, wrong types and out-of-range values raise ``ParamsError``. The model
<-> tokenizer <-> ``max_context`` pairing and the ``top_k``/``top_p`` interaction of the pinned
Kronos code are validated here as well as in the loader. ``kronos.seed_salt`` follows the
seed-string grammar of :mod:`kronos_agent.seeds` (ASCII, no ``/`` or newline), so a bad salt
fails at load time instead of at the first forecast. The ceilings on ``trajectories`` and
``horizon_n`` are sanity bounds (twice the SPEC's largest values), not the server's memory budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .hashing import sha256_file
from .models import MODEL_SPECS
from .paths import REPO_ROOT

DEFAULT_PARAMS_PATH = REPO_ROOT / "params.yaml"


class ParamsError(ValueError):
    """Invalid ``params.yaml``."""


@dataclass(frozen=True)
class CriterionParams:
    atr_period: int
    atr_method: str
    profile_window_candles: int
    profile_rows: int
    profile_row_mode: str
    profile_smoothing_sigma_rows: float
    profile_min_prominence_pct: float
    profile_source_timeframe: str
    profile_split_up_down: bool
    value_area_pct: float
    level_poc: bool
    level_hvn: bool
    level_vah_val: bool
    level_lvn: bool
    target_level_types: str
    arrival_k_atr: float
    arrival_by: str
    decel_n_candles: int
    decel_rule: str
    swing_window_candles: int
    divergence_rule: str
    prior_move_m_candles: int
    stop_s_atr: float
    time_stop_t_candles: int
    min_rr_r: float
    side: str
    session_filter: str
    volatility_filter: str
    entry: str


@dataclass(frozen=True)
class KronosParams:
    model: str
    max_context: int
    horizon_n: int
    temperature: float
    top_p: float
    top_k: int
    trajectories: int
    device: str
    threads: int
    amount_policy: str
    seed_salt: str
    finetune_recipe: Any
    retrain_frequency: str
    history: str
    train_timeframe: str


@dataclass(frozen=True)
class CostParams:
    taker_fee_bp: float
    spread_bp: float
    slippage_bp: float
    funding_interval_hours: int


@dataclass(frozen=True)
class Params:
    schema_version: int
    symbol: str
    timeframe: str
    criterion: CriterionParams
    kronos: KronosParams
    costs: CostParams
    version: str


ENUMS: dict[str, set[str]] = {
    "atr_method": {"wilder", "sma"},
    "profile_row_mode": {"range_rows", "fixed_bp"},
    "profile_source_timeframe": {"1h", "5m", "1m"},
    "target_level_types": {"active", "all"},
    "arrival_by": {"close", "wick"},
    "decel_rule": {"strictly_decreasing", "net_decrease"},
    "divergence_rule": {"extreme_candle_volume", "three_candle_sum"},
    "side": {"long", "short", "both"},
    "session_filter": {"off", "utc_sessions"},
    "volatility_filter": {"off", "atr_percentile"},
    "entry": {"market", "limit"},
    "amount_policy": {"volume_x_mean_ohlc", "binance_quote_volume", "zeros"},
    "retrain_frequency": {"weekly", "biweekly", "monthly", "quarterly"},
    "history": {"all", "2y"},
    "train_timeframe": {"1h", "15m"},
}


def _section(raw: Mapping[str, Any], name: str, cls: type) -> dict[str, Any]:
    if name not in raw or not isinstance(raw[name], Mapping):
        raise ParamsError(f"section {name!r} is missing or not a mapping")
    data = dict(raw[name])
    expected = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - expected)
    missing = sorted(expected - set(data))
    if unknown:
        raise ParamsError(f"{name}: unknown keys {unknown}")
    if missing:
        raise ParamsError(f"{name}: missing keys {missing}")
    for field_ in fields(cls):
        value = data[field_.name]
        if field_.type == "bool" and not isinstance(value, bool):
            raise ParamsError(f"{name}.{field_.name}: expected a boolean")
        if field_.type == "int" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ParamsError(f"{name}.{field_.name}: expected an integer")
        if field_.type == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ParamsError(f"{name}.{field_.name}: expected a number")
            data[field_.name] = float(value)
        if field_.type == "str" and not isinstance(value, str):
            raise ParamsError(f"{name}.{field_.name}: expected a string")
        if field_.name in ENUMS and value not in ENUMS[field_.name]:
            raise ParamsError(
                f"{name}.{field_.name}: {value!r} not in {sorted(ENUMS[field_.name])}"
            )
    return data


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ParamsError(message)


def validate(raw: Mapping[str, Any], *, version: str) -> Params:
    top_expected = {"schema_version", "symbol", "timeframe", "criterion", "kronos", "costs"}
    unknown = sorted(set(raw) - top_expected)
    missing = sorted(top_expected - set(raw))
    _check(not unknown, f"unknown top-level keys {unknown}")
    _check(not missing, f"missing top-level keys {missing}")
    schema_version = raw["schema_version"]
    _check(
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == 1,
        "schema_version must be 1",
    )
    _check(
        isinstance(raw["symbol"], str) and raw["symbol"] == "BTC-USDT", "symbol must be BTC-USDT"
    )
    _check(raw["timeframe"] == "1h", "timeframe must be 1h")

    criterion = CriterionParams(**_section(raw, "criterion", CriterionParams))
    kronos = KronosParams(**_section(raw, "kronos", KronosParams))
    costs = CostParams(**_section(raw, "costs", CostParams))

    c = criterion
    _check(c.atr_period >= 1, "criterion.atr_period must be >= 1")
    _check(10 <= c.profile_window_candles <= 2000, "criterion.profile_window_candles out of range")
    _check(2 <= c.profile_rows <= 1000, "criterion.profile_rows out of range")
    _check(
        c.profile_smoothing_sigma_rows >= 0, "criterion.profile_smoothing_sigma_rows must be >= 0"
    )
    _check(
        0 < c.profile_min_prominence_pct <= 100, "criterion.profile_min_prominence_pct out of range"
    )
    _check(0 < c.value_area_pct < 100, "criterion.value_area_pct must be in (0, 100)")
    _check(c.arrival_k_atr > 0, "criterion.arrival_k_atr must be > 0")
    _check(c.decel_n_candles >= 1, "criterion.decel_n_candles must be >= 1")
    _check(c.swing_window_candles >= 1, "criterion.swing_window_candles must be >= 1")
    _check(c.prior_move_m_candles >= 1, "criterion.prior_move_m_candles must be >= 1")
    _check(c.stop_s_atr > 0, "criterion.stop_s_atr must be > 0")
    _check(c.time_stop_t_candles >= 1, "criterion.time_stop_t_candles must be >= 1")
    _check(c.min_rr_r > 0, "criterion.min_rr_r must be > 0")

    k = kronos
    _check(k.model in MODEL_SPECS, f"kronos.model must be one of {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[k.model]
    _check(
        k.max_context == spec.max_context,
        f"kronos.max_context must be {spec.max_context} for model {k.model!r}",
    )
    _check(k.horizon_n >= 1, "kronos.horizon_n must be >= 1")
    _check(
        k.horizon_n <= 48,
        "kronos.horizon_n must be <= 48 (SPEC: N = 4, variant 8; skill curve to h = 24)",
    )
    _check(0 < k.temperature <= 3, "kronos.temperature out of range")
    _check(0 < k.top_p <= 1, "kronos.top_p must be in (0, 1]")
    _check(k.top_k >= 0, "kronos.top_k must be >= 0")
    _check(
        k.top_k == 0 or k.top_p == 1.0,
        "kronos.top_k > 0 disables top_p in the pinned Kronos code; keep top_k = 0",
    )
    _check(k.trajectories >= 1, "kronos.trajectories must be >= 1")
    _check(
        k.trajectories <= 100,
        "kronos.trajectories must be <= 100 (SPEC range 20-50; memory grows with the fan)",
    )
    _check(k.device == "cpu", "kronos.device must be cpu (comparable inference is CPU only)")
    _check(k.threads >= 1, "kronos.threads must be >= 1")
    _check(
        k.seed_salt.isascii() and "/" not in k.seed_salt and "\n" not in k.seed_salt,
        "kronos.seed_salt must be ASCII without '/' or newlines (grammar of kronos_agent.seeds)",
    )
    _check(
        k.finetune_recipe is None or isinstance(k.finetune_recipe, Mapping),
        "kronos.finetune_recipe must be null or a mapping",
    )

    _check(
        costs.taker_fee_bp >= 0 and costs.spread_bp >= 0 and costs.slippage_bp >= 0,
        "costs must be >= 0",
    )
    _check(
        costs.funding_interval_hours in {1, 2, 4, 8},
        "costs.funding_interval_hours must be 1, 2, 4 or 8",
    )

    return Params(
        schema_version=1,
        symbol=raw["symbol"],
        timeframe=raw["timeframe"],
        criterion=criterion,
        kronos=kronos,
        costs=costs,
        version=version,
    )


def params_version(path: Path | str = DEFAULT_PARAMS_PATH) -> str:
    """First 12 hex digits of the sha256 of the file bytes; recorded in ledgers and reports."""
    return sha256_file(path)[:12]


def load_params(path: Path | str = DEFAULT_PARAMS_PATH) -> Params:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ParamsError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ParamsError(f"{path.name}: top level must be a mapping")
    return validate(raw, version=params_version(path))
