"""params.yaml loader: the SPEC's initial values, a stable version and every rejection asked for."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from kronos_agent.hashing import sha256_file
from kronos_agent.params import (
    DEFAULT_PARAMS_PATH,
    ParamsError,
    load_params,
    params_version,
    validate,
)
from kronos_agent.seeds import seed_for_decision

SALT_MESSAGE = "kronos.seed_salt must be ASCII without '/' or newlines"


def variant(tmp_path: Path, mutate: Callable[[dict[str, Any]], Any]) -> Path:
    """A copy of params.yaml with ``mutate`` applied to the parsed mapping (strings stay quoted)."""
    raw = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def set_kronos(**values: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(raw: dict[str, Any]) -> None:
        raw["kronos"].update(values)

    return mutate


def set_criterion(**values: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(raw: dict[str, Any]) -> None:
        raw["criterion"].update(values)

    return mutate


def test_initial_values_match_the_spec():
    params = load_params()
    assert (params.schema_version, params.symbol, params.timeframe) == (1, "BTC-USDT", "1h")

    c = params.criterion
    assert (c.profile_window_candles, c.profile_rows) == (100, 75)
    assert c.profile_smoothing_sigma_rows == 2.0
    assert c.profile_min_prominence_pct == 10.0
    assert (c.level_poc, c.level_hvn, c.level_vah_val, c.level_lvn) == (True, True, False, False)
    assert (c.arrival_k_atr, c.arrival_by) == (0.5, "close")
    assert c.decel_n_candles == 3
    assert c.swing_window_candles == 12
    assert c.prior_move_m_candles == 6
    assert c.stop_s_atr == 0.5
    assert c.time_stop_t_candles == 8
    assert c.min_rr_r == 1.0
    assert (c.side, c.entry) == ("both", "market")
    assert (c.session_filter, c.volatility_filter) == ("off", "off")

    k = params.kronos
    assert (k.model, k.max_context, k.horizon_n) == ("small", 512, 4)
    assert (k.temperature, k.top_p, k.top_k) == (1.0, 0.9, 0)
    assert (k.trajectories, k.device, k.threads) == (20, "cpu", 2)
    assert k.amount_policy == "volume_x_mean_ohlc"
    assert k.seed_salt == ""
    assert k.finetune_recipe is None

    costs = params.costs
    assert (costs.taker_fee_bp, costs.spread_bp, costs.slippage_bp) == (5.0, 0.5, 0.5)
    assert costs.funding_interval_hours == 8


def test_params_version_is_12_hex_and_stable():
    first, second = params_version(), params_version()
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{12}", first)
    assert first == sha256_file(DEFAULT_PARAMS_PATH)[:12]
    assert load_params().version == first


def test_params_version_follows_the_file_bytes(tmp_path):
    copy = tmp_path / "params.yaml"
    copy.write_bytes(DEFAULT_PARAMS_PATH.read_bytes())
    assert params_version(copy) == params_version()
    copy.write_bytes(DEFAULT_PARAMS_PATH.read_bytes() + b"# trailing comment\n")
    assert params_version(copy) != params_version()
    assert load_params(copy).version == params_version(copy)


REJECTIONS = [
    pytest.param(
        set_kronos(model="mini", max_context=512), "max_context must be 2048", id="mini-512"
    ),
    pytest.param(
        set_kronos(model="small", max_context=2048), "max_context must be 512", id="small-2048"
    ),
    pytest.param(
        set_kronos(model="base", max_context=2048), "max_context must be 512", id="base-2048"
    ),
    pytest.param(set_kronos(top_k=1, top_p=0.9), "top_k > 0 disables top_p", id="top_k-with-top_p"),
    pytest.param(set_kronos(device="mps"), "device must be cpu", id="device-mps"),
    pytest.param(set_kronos(model="large"), "kronos.model must be one of", id="unknown-model"),
    pytest.param(set_kronos(threads=0), "threads must be >= 1", id="threads-zero"),
    pytest.param(set_kronos(top_p=1.5), "top_p must be in (0, 1]", id="top_p-range"),
    pytest.param(set_kronos(horizon_n=0), "kronos.horizon_n must be >= 1", id="horizon-zero"),
    pytest.param(set_kronos(horizon_n=49), "kronos.horizon_n must be <= 48", id="horizon-ceiling"),
    pytest.param(
        set_kronos(trajectories=0), "kronos.trajectories must be >= 1", id="trajectories-zero"
    ),
    pytest.param(
        set_kronos(trajectories=101),
        "kronos.trajectories must be <= 100",
        id="trajectories-ceiling",
    ),
    pytest.param(set_kronos(seed_salt="a/b"), SALT_MESSAGE, id="slash-salt"),
    pytest.param(set_kronos(seed_salt="a\nb"), SALT_MESSAGE, id="newline-salt"),
    pytest.param(set_kronos(seed_salt="\u00f1"), SALT_MESSAGE, id="non-ascii-salt"),
    pytest.param(
        set_criterion(unknown_knob=1), "criterion: unknown keys", id="unknown-criterion-key"
    ),
    pytest.param(
        lambda raw: raw["criterion"].pop("side"), "criterion: missing keys", id="missing-key"
    ),
    pytest.param(
        lambda raw: raw.update(extra_section={}), "unknown top-level keys", id="unknown-section"
    ),
    pytest.param(lambda raw: raw.pop("costs"), "missing top-level keys", id="missing-section"),
    pytest.param(set_criterion(side="up"), "criterion.side", id="bad-enum"),
    pytest.param(set_criterion(profile_rows=1), "profile_rows out of range", id="rows-range"),
    pytest.param(set_criterion(level_poc="yes"), "expected a boolean", id="string-for-bool"),
    pytest.param(set_criterion(decel_n_candles=2.5), "expected an integer", id="float-for-int"),
    pytest.param(
        lambda raw: raw["costs"].update(funding_interval_hours=6),
        "funding_interval_hours",
        id="funding",
    ),
    pytest.param(lambda raw: raw.update(symbol="ETH-USDT"), "symbol must be BTC-USDT", id="symbol"),
    pytest.param(lambda raw: raw.update(schema_version=2), "schema_version must be 1", id="schema"),
    pytest.param(
        lambda raw: raw.update(schema_version=True), "schema_version must be 1", id="schema-bool"
    ),
    pytest.param(
        lambda raw: raw.update(schema_version=1.0), "schema_version must be 1", id="schema-float"
    ),
]


@pytest.mark.parametrize("mutate, message", REJECTIONS)
def test_invalid_variants_are_rejected(tmp_path, mutate, message):
    with pytest.raises(ParamsError, match=re.escape(message)):
        load_params(variant(tmp_path, mutate))


def test_top_k_is_allowed_only_with_top_p_one(tmp_path):
    params = load_params(variant(tmp_path, set_kronos(top_k=5, top_p=1.0)))
    assert (params.kronos.top_k, params.kronos.top_p) == (5, 1.0)


def test_spec_kronos_ranges_and_the_ceilings_are_accepted(tmp_path):
    """SPEC: 20-50 trajectories, N = 4 (variant 8), skill curve to h = 24; ceilings 100 and 48."""
    for trajectories, horizon_n in [(50, 24), (100, 48)]:
        mutate = set_kronos(trajectories=trajectories, horizon_n=horizon_n)
        k = load_params(variant(tmp_path, mutate)).kronos
        assert (k.trajectories, k.horizon_n) == (trajectories, horizon_n)


def seeds_accept(salt: str) -> bool:
    try:
        seed_for_decision(datetime(2026, 10, 29, 16, tzinfo=UTC), salt=salt)
    except ValueError:
        return False
    return True


def test_seed_salt_grammar_matches_seeds():
    """A salt loads exactly when kronos_agent.seeds can hash it, so no forecast fails later."""
    raw = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
    salts = ["", "v2", "a b", "\u00f1", "a\u2028b", *(f"a{chr(code)}b" for code in range(128))]
    rejected = {salt for salt in salts if not seeds_accept(salt)}
    assert {"a/b", "a\nb", "\u00f1", "a\u2028b"} <= rejected
    for salt in salts:
        raw["kronos"]["seed_salt"] = salt
        if salt in rejected:
            with pytest.raises(ParamsError, match=re.escape(SALT_MESSAGE)):
                validate(raw, version="test")
        else:
            assert validate(raw, version="test").kronos.seed_salt == salt


def test_unquoted_off_is_a_yaml_boolean_and_rejected(tmp_path):
    text = DEFAULT_PARAMS_PATH.read_text(encoding="utf-8")
    assert 'session_filter: "off"' in text
    assert (
        yaml.safe_load("value: off")["value"] is False
    )  # YAML 1.1 boolean: the trap the quotes avoid
    path = tmp_path / "params.yaml"
    path.write_text(text.replace('session_filter: "off"', "session_filter: off"), encoding="utf-8")
    with pytest.raises(ParamsError, match=re.escape("criterion.session_filter: expected a string")):
        load_params(path)


def test_invalid_yaml_and_non_mapping_are_params_errors(tmp_path):
    broken = tmp_path / "broken.yaml"
    broken.write_text("criterion: [unclosed\n", encoding="utf-8")
    with pytest.raises(ParamsError, match="invalid YAML"):
        load_params(broken)
    listed = tmp_path / "list.yaml"
    listed.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ParamsError, match="top level must be a mapping"):
        load_params(listed)
    assert issubclass(ParamsError, ValueError)
