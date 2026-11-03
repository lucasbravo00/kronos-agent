"""The tokenizer diagnostic (``docs/RETRAINING.md`` section 4; interfaces 10 and 12).

Tiny tokenizers (``finetune_fakes.build_tiny_models``, 4 + 4 bits) and synthetic stores
(``step2_fakes``) under temporary roots; the decision lists and the reference seeds are shortened
by monkeypatching the module, so a whole run takes about a second. One ``weights`` test runs the
per-window computations with the shipped tokenizer on synthetic windows (the session's
``loaded_small``): the context-only positive control and the ``z_pre`` path hold there too.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import finetune_fakes as ff
import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes
import torch

from kronos_agent.data.partition import (
    TRAINING_START,
    VALIDATION_END,
    VALIDATION_START,
)
from kronos_agent.data.timeutil import HOUR_MS, to_ms, utc_text
from kronos_agent.data.windows import StoreWindowSource
from kronos_agent.finetune import tokenizer_diag as td
from kronos_agent.finetune.folds import on_grid
from kronos_agent.finetune.nll import normalise
from kronos_agent.hashing import sha256_file
from kronos_agent.kronos.features import FEATURE_COLUMNS, synthetic_window
from kronos_agent.params import load_params
from kronos_agent.paths import REPO_ROOT
from kronos_agent.report.writer import ConfigChangedError

PARAMS = load_params()
CLOSE = FEATURE_COLUMNS.index("close")


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer, _ = ff.build_tiny_models(0)
    return ff.eval_copy(tokenizer)


def steps_for(tokenizer, *, clock=None, window_source=StoreWindowSource):
    ticks = iter(clock) if clock is not None else None
    return td.TokenizerSteps(
        load_tokenizer=lambda: tokenizer,
        window_source=window_source,
        clock=(lambda: next(ticks)) if ticks is not None else (lambda: 0.0),
    )


def windows_of(count, rows, *, seed=0):
    """``(count, rows, 6)`` float64 synthetic GBM windows (the reference's generator)."""
    return np.stack(
        [
            synthetic_window(rows, seed=seed + index)[0][list(FEATURE_COLUMNS)].to_numpy(np.float64)
            for index in range(count)
        ]
    )


def g25_after(first_ms, count, *, spacing_hours=25):
    """``count`` G25 hours from ``first_ms`` on, at least ``spacing_hours`` apart."""
    chosen = []
    hour = first_ms
    while len(chosen) < count:
        if on_grid(hour, "G25") and (not chosen or hour - chosen[-1] >= spacing_hours * HOUR_MS):
            chosen.append(hour)
        hour += HOUR_MS
    return tuple(chosen)


def binance_store(root, start_ms, end_ms, *, seed=3, missing=()):
    frame = fakes.binance_frame(start_ms, (end_ms - start_ms) // HOUR_MS, interval="1h", seed=seed)
    if missing:
        frame = frame[~frame["open_ms"].isin(set(missing))].reset_index(drop=True)
    fakes.write_binance_store(root, frame)


@pytest.fixture
def short(monkeypatch):
    """Shortened decision lists and reference (the definition changes with them)."""
    chosen: dict[str, tuple[int, ...]] = {
        # 600 h apart: the 561 candles of one window never reach another window
        "training": g25_after(to_ms(td.PART_A_FIRST), 6, spacing_hours=600),
        "validation": g25_after(to_ms(VALIDATION_START), 5),
    }
    monkeypatch.setattr(td, "decisions_ms", lambda part: chosen[td._check_part(part)])
    monkeypatch.setattr(td, "REFERENCE_SEEDS", (0, 1, 2, 3))
    return chosen


def part_a_world(tmp_path, chosen, **store):
    data = tmp_path / "data"
    binance_store(data, to_ms(TRAINING_START), chosen[-1] + td.NEXT_ROWS * HOUR_MS, **store)
    return data


# -- the definition ---------------------------------------------------------------------------


def test_decision_hours_counts_ends_and_candle_boundaries():
    part_a = td.decisions_ms("training")
    part_b = td.decisions_ms("validation")
    assert len(part_a) == 1534 and len(part_b) == 710
    assert (utc_text(part_a[0]), utc_text(part_a[-1])) == ("2020-01-22T21:00Z", "2024-06-06T18:00Z")
    assert (utc_text(part_b[0]), utc_text(part_b[-1])) == ("2024-07-01T18:00Z", "2026-07-10T07:00Z")
    assert all(on_grid(d, "G25") for d in (*part_a, *part_b))
    assert part_a[0] - 512 * HOUR_MS >= to_ms(TRAINING_START)
    assert to_ms(td.PART_A_FIRST) == to_ms(TRAINING_START) + 512 * HOUR_MS
    # every candle Part A reads opens before 2024-06-08T19:00Z, inside the training boundary
    plan = td.diag_plan("training")
    source = plan["sources"]["binance"]
    assert (source["start"], source["end"]) == ("2020-01-01T00:00Z", "2024-06-08T19:00Z")
    assert part_a[-1] + 48 * HOUR_MS < to_ms(td.TRAINING_CANDLE_END)
    assert utc_text(to_ms(td.TRAINING_CANDLE_END)) == "2024-06-09T16:00Z"
    for name in ("binance", "bingx"):
        source = td.diag_plan("validation")["sources"][name]
        assert (source["start"], source["end"]) == ("2024-06-09T16:00Z", "2026-07-10T08:00Z")
    # a decision whose next rows would cross the boundary is refused
    late = to_ms(datetime(2024, 6, 8, tzinfo=UTC))
    with pytest.raises(td.TokenizerDiagError, match="outside the training segment"):
        td._source_range("training", (late,))
    with pytest.raises(td.TokenizerDiagError, match="part must be one of"):
        td.decisions_ms("test")


def test_the_plan_holds_the_definition_and_no_data():
    plan = td.diag_plan("training")
    assert plan["decisions"]["planned"] == 1534 and plan["window"]["next_rows"] == 49
    assert set(plan["normalisation"]["variants"]) == {"inference", "r1", "template", "context_only"}
    assert plan["normalisation"]["variants"]["context_only"] == {
        "statistics_rows": 512,
        "rows": 513,
    }
    assert plan["tests"] == 0 and plan["tokenizer"]["repo_id"].endswith("Kronos-Tokenizer-base")
    assert "r1_token_mismatch" in plan["flags"]
    assert "validation_perplexity_ratio" not in plan["flags"]
    assert plan["token_mismatch_statistic"] == "mismatch_r1_any"
    assert "s1 or s2 token" in plan["flags"]["r1_token_mismatch"]["rule"]
    assert "mismatch_<variant>_<s1|s2|any>[_last24]" in plan["statistics"]
    part_b = td.diag_plan("validation")
    assert set(part_b["normalisation"]["variants"]) == {"inference"}
    assert part_b["window"]["next_rows"] == 0 and "inputs" in part_b
    assert "validation_perplexity_ratio" in part_b["flags"]
    assert "r1_token_mismatch" not in part_b["flags"]
    assert part_b["token_mismatch_statistic"] is None
    text = json.dumps(plan)
    assert "params_version" not in text and "source_tree_sha256" not in text


def test_the_config_is_recorded_once_and_a_changed_one_is_refused(tmp_path, tokenizer):
    first = td.record_diag_config("training", directory=tmp_path)
    assert td.record_diag_config("training", directory=tmp_path) == first
    path = tmp_path / "tokenizer-training.config.json"
    assert path.is_file()
    path.write_bytes(path.read_bytes().replace(b'"next_rows":49', b'"next_rows":48'))

    def no_store(*args, **kwargs):
        raise AssertionError("a store was read before the definition was verified")

    steps = steps_for(tokenizer, window_source=no_store)
    with pytest.raises(ConfigChangedError):
        td.prepare_diag("training", params=PARAMS, steps=steps, directory=tmp_path)


def test_a_definition_changed_after_the_preparation_stops_the_run(tmp_path, short, tokenizer):
    data = part_a_world(tmp_path, short["training"])
    steps = steps_for(tokenizer, clock=[0.0, 1.0])
    prepared = td.prepare_diag(
        "training", params=PARAMS, steps=steps, root=data, directory=tmp_path
    )
    path = tmp_path / "tokenizer-training.config.json"
    path.write_bytes(path.read_bytes().replace(b'"last_rows":24', b'"last_rows":12'))
    with pytest.raises(ConfigChangedError):
        td.run_diag(prepared, params=PARAMS, steps=steps, directory=tmp_path)
    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "tokenizer-training.json").exists()


def test_part_b_without_part_a_report_fails_before_reading_a_store(tmp_path, tokenizer):
    def no_store(*args, **kwargs):
        raise AssertionError("a store was read")

    steps = steps_for(tokenizer, window_source=no_store)
    with pytest.raises(td.TokenizerDiagError, match="Part A's report"):
        td.prepare_diag("validation", params=PARAMS, steps=steps, directory=tmp_path)


def test_prepare_refuses_another_thread_count_or_model(tmp_path, tokenizer):
    steps = steps_for(tokenizer)
    for kronos in (replace(PARAMS.kronos, threads=8), replace(PARAMS.kronos, max_context=256)):
        with pytest.raises(td.TokenizerDiagError):
            td.prepare_diag("training", params=replace(PARAMS, kronos=kronos), steps=steps)


# -- normalisation ----------------------------------------------------------------------------


def test_normalise_rows_with_the_context_statistics_is_nll_normalise_bit_for_bit():
    values = windows_of(3, 516, seed=7)
    expected = normalise(values)
    actual = td.normalise_rows(values, stats_rows=512)
    assert actual.dtype == np.float32 and actual.shape == values.shape
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    single = td.normalise_rows(values[1], stats_rows=512)
    assert np.array_equal(single.view(np.uint32), expected[1].view(np.uint32))


def test_the_r1_variant_is_the_vendored_window_normalisation_bit_for_bit():
    values = windows_of(1, 513, seed=11)[0]
    # FB:122-127: window_data[feature_list].values.astype(np.float32), mean and std on axis 0
    x = pd.DataFrame(values, columns=list(FEATURE_COLUMNS)).values.astype(np.float32)
    mean, std = np.mean(x, axis=0), np.std(x, axis=0)
    vendored = np.clip((x - mean) / (std + 1e-5), -5.0, 5.0)
    actual = td.normalise_rows(values, stats_rows=513)
    assert np.array_equal(actual.view(np.uint32), vendored.view(np.uint32))
    # the context-only variant: the context statistics on all 513 rows
    context_only = td.normalise_rows(values, stats_rows=512)
    inference = td.normalise_rows(values[:512], stats_rows=512)
    assert np.array_equal(context_only[:512].view(np.uint32), inference.view(np.uint32))


def test_normalise_rows_refuses_bad_shapes_and_arguments():
    values = windows_of(1, 512)[0]
    for bad in (values[:, :5], values[None, None]):
        with pytest.raises(td.TokenizerDiagError):
            td.normalise_rows(bad, stats_rows=10)
    for stats_rows in (0, 513, True, 1.5):
        with pytest.raises(td.TokenizerDiagError):
            td.normalise_rows(values, stats_rows=stats_rows)
    with pytest.raises(td.TokenizerDiagError):
        td.normalise_rows(values, stats_rows=512, clip=0)


# -- per-window computations ------------------------------------------------------------------


def test_the_context_only_control_is_zero_and_a_large_row_512_is_detected(tokenizer):
    values = windows_of(6, 561, seed=3)
    out = td._diagnose_batch(tokenizer, values)
    assert np.count_nonzero(out["mismatch_context_only"]) == 0
    td._check_control(out)
    leaky = values.copy()
    leaky[:, 512, :4] *= 50.0  # one huge future price row enters the r1 statistics
    leak = td._diagnose_batch(tokenizer, leaky)
    assert np.count_nonzero(leak["mismatch_context_only"]) == 0  # the control ignores row 512
    assert leak["mismatch_r1"][:, 0].mean() > 0 and leak["mismatch_template"][:, 0].mean() > 0
    assert np.array_equal(leak["s1"], out["s1"])  # the inference encoding never sees row 512
    assert (np.abs(leak["next_z"][:, CLOSE]) > 5).all()
    with pytest.raises(td.TokenizerDiagError, match="positive control"):
        td._check_control({"mismatch_context_only": np.array([[0.0, 0, 0, 0], [0.002, 0, 0, 0]])})
    with pytest.raises(td.TokenizerDiagError, match="positive control"):  # an s2 token only
        td._check_control({"mismatch_context_only": np.array([[0.0, 0, 0.002, 0, 0.002, 0]])})


class SignTokenizer:
    """A stand-in tokenizer: per row, s1 is the sign of the normalised close and s2 the sign of
    the normalised volume, so that a variant can change the s2 tokens of rows whose s1 tokens it
    keeps."""

    def encode(self, x, half=True):
        values = x.numpy()
        close, volume = FEATURE_COLUMNS.index("close"), FEATURE_COLUMNS.index("volume")
        return (values[..., close] > 0).astype(np.int64), (values[..., volume] > 0).astype(np.int64)


def test_the_context_token_mismatch_counts_rows_whose_s1_or_s2_token_changed():
    """A huge future volume in row 512 moves R1's volume statistics only: no s1 token changes,
    some s2 tokens do, and the share of rows with either token changed (the flag's statistic)
    sees them; the context-only control changes nothing."""
    generator = np.random.default_rng(8)
    values = 100.0 + generator.normal(0.0, 1.0, size=(3, 561, 6))
    close, volume = FEATURE_COLUMNS.index("close"), FEATURE_COLUMNS.index("volume")
    values[:, :512, close] = 100.0 + np.tile([1.0, -1.0], 256)  # every row 1 from the mean
    values[:, 512, close] = 100.0  # the mean: R1's close statistics keep every sign
    values[:, 512, volume] = 1e4
    tokenizer = SignTokenizer()
    zc = td.normalise_rows(values[:, :512], stats_rows=512)
    s1, s2 = tokenizer.encode(torch.from_numpy(zc))
    out = td._mismatches(tokenizer, values, s1, s2)
    assert td.MISMATCH_COLUMNS == ("s1", "s1_last24", "s2", "s2_last24", "any", "any_last24")
    r1 = out["mismatch_r1"]
    assert r1.shape == (3, len(td.MISMATCH_COLUMNS))
    assert (r1[:, 0] == 0).all() and (r1[:, 2] > 0).all()  # s1 kept, s2 changed
    assert np.array_equal(r1[:, 4], r1[:, 2]) and np.array_equal(r1[:, 5], r1[:, 3])
    assert np.count_nonzero(out["mismatch_context_only"]) == 0
    td._check_control(out)
    both = values.copy()
    both[:, 512, close] = 1e4  # now the s1 tokens change as well
    union = td._mismatches(tokenizer, both, s1, s2)["mismatch_r1"]
    assert (union[:, 4] >= np.maximum(union[:, 0], union[:, 2])).all()
    assert (union[:, 4] <= union[:, 0] + union[:, 2]).all() and (union[:, 0] > 0).all()


def test_z_pre_and_the_reconstruction_are_the_forward_pass(tokenizer):
    zc = torch.from_numpy(td.normalise_rows(windows_of(4, 512, seed=5), stats_rows=512))
    with torch.no_grad():
        (z_pre, z), _, _, _ = tokenizer(zc)
        s1, s2 = tokenizer.encode(zc, half=True)
        assert torch.equal(td._z_pre(tokenizer, s1, s2), z_pre)
        assert torch.equal(tokenizer.decode([s1, s2], half=True), z)


def test_the_bit_margin_reads_the_quantiser_latent(tokenizer):
    values = windows_of(4, 512, seed=9)
    zc = torch.from_numpy(td.normalise_rows(values, stats_rows=512))
    with torch.no_grad():
        latent = td._latent(tokenizer, zc)
        s1, s2 = tokenizer.encode(zc, half=True)
    bits = tokenizer.s1_bits
    weights = 2 ** torch.arange(bits)
    assert torch.equal(((latent[..., :bits] > 0).long() * weights).sum(-1), s1)
    assert torch.equal(((latent[..., bits:] > 0).long() * weights).sum(-1), s2)
    assert torch.allclose(latent.norm(dim=-1), torch.ones(latent.shape[:-1]))
    out = td._diagnose_batch(tokenizer, values)
    margins = latent.abs().numpy()
    for column, margin in enumerate(td.BIT_MARGINS):
        assert out["margins"][:, column] == pytest.approx((margins < margin).mean(axis=(1, 2)))


def test_saturation_of_a_crafted_window(tokenizer):
    generator = np.random.default_rng(0)
    values = 100.0 + generator.normal(0.0, 1.0, size=(1, 561, 6))
    values[0, 100, CLOSE] = 1e4  # one context outlier: |z| about sqrt(511) > 5
    values[0, 512, :] = 1e4  # the next row beyond +-5 in every channel
    out = td._diagnose_batch(tokenizer, values)
    expected = np.zeros(6, dtype=np.int64)
    expected[CLOSE] = 1
    assert out["saturated"][0].tolist() == expected.tolist()
    assert out["max_abs_z"][0, CLOSE] == pytest.approx(math.sqrt(511), rel=1e-3)
    assert (np.abs(out["next_z"][0]) > 5).all()
    windows = td.DiagWindows(
        "binance", np.array([to_ms(td.PART_A_FIRST)], dtype=np.int64), values, ()
    )
    result = td._set_results("training", "binance", windows, out, bits=4, matched=512)
    statistics = result["statistics"]
    assert statistics["saturation_context_close"]["point"] == pytest.approx(1 / 512)
    assert statistics["saturation_windows_close"]["point"] == 1.0
    assert statistics["saturation_windows_open"]["point"] == 0.0
    assert statistics["saturation_next_volume"]["point"] == 1.0
    assert "fewer than 20 blocks" in statistics["saturation_context_close"]["reason"]


# -- codebook and divergences -----------------------------------------------------------------


def test_codebook_statistics_on_crafted_counts():
    uniform = td.codebook_statistics(np.full(16, 10), 4)
    assert uniform["entropy"] == pytest.approx(math.log(16))
    assert uniform["perplexity"] == pytest.approx(16.0)
    assert uniform["normalised_entropy"] == pytest.approx(1.0)
    assert (uniform["distinct"], uniform["unused_share"]) == (16, 0.0)
    assert uniform["coverage"] == {"50%": 8, "90%": 15, "99%": 16}
    assert uniform["bit_ones"] == [0.5] * 4 and uniform["dead_bits"] == []
    one = np.zeros(16, dtype=np.int64)
    one[5] = 7  # 0b0101
    single = td.codebook_statistics(one, 4)
    assert (single["entropy"], single["perplexity"], single["normalised_entropy"]) == (
        0.0,
        1.0,
        0.0,
    )
    assert single["unused_share"] == 15 / 16 and single["coverage"] == {
        "50%": 1,
        "90%": 1,
        "99%": 1,
    }
    assert single["bit_ones"] == [1.0, 0.0, 1.0, 0.0] and single["dead_bits"] == [0, 1, 2, 3]
    skewed = np.zeros(16, dtype=np.int64)
    skewed[[3, 1, 7, 2]] = [50, 40, 9, 1]  # exact boundaries: 50, 90 and 99 of 100 rows
    assert td.codebook_statistics(skewed, 4)["coverage"] == {"50%": 1, "90%": 2, "99%": 3}
    near = np.zeros(16, dtype=np.int64)
    near[[0, 1]] = [995, 5]  # bit 0 set in 0.5 % of rows: dead below 1 %
    assert td.codebook_statistics(near, 4)["dead_bits"] == [0, 1, 2, 3]
    for bad in (np.full(15, 1), np.zeros(16, dtype=np.int64), np.full(16, -1), np.full(16, 0.5)):
        with pytest.raises(td.TokenizerDiagError):
            td.codebook_statistics(bad, 4)


def test_js_divergence_known_values():
    assert td.js_divergence(np.array([3, 1, 0]), np.array([6, 2, 0])) == pytest.approx(0.0)
    assert td.js_divergence(np.array([1, 0]), np.array([0, 4])) == pytest.approx(1.0)
    expected = 0.5 * math.log2(1 / 0.75) + 0.5 * (
        0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25)
    )
    assert td.js_divergence(np.array([1, 0]), np.array([1, 1])) == pytest.approx(expected)
    with pytest.raises(td.TokenizerDiagError):
        td.js_divergence(np.array([1, 0]), np.array([0, 0]))


def test_matched_indices_spread_rows_in_time_order():
    assert td.matched_indices(10, 4).tolist() == [0, 3, 6, 9]
    assert td.matched_indices(5, 5).tolist() == [0, 1, 2, 3, 4]
    assert td.matched_indices(7, 1).tolist() == [0]
    for rows, matched in ((5, 6), (5, 0), (0, 0)):
        with pytest.raises(td.TokenizerDiagError):
            td.matched_indices(rows, matched)


def test_flags_follow_the_pre_registered_thresholds():
    def results(**points):
        statistics = {
            name: {"point": points.get(name, 0.0)}
            for name in (
                *(f"saturation_context_{channel}" for channel in FEATURE_COLUMNS),
                *(f"saturation_next_{channel}" for channel in FEATURE_COLUMNS),
                "rho",
                "mismatch_r1_s1",
                "mismatch_r1_s2",
                "mismatch_r1_any",
            )
        }
        codebook = {
            "s1": {
                "normalised_entropy": points.get("entropy", 0.9),
                "unused_share": points.get("unused", 0.1),
                "perplexity": points.get("perplexity", 500.0),
            }
        }
        return {"statistics": statistics, "codebook": codebook}

    quiet = td._raise_flags("training", {"binance": results()}, None)
    assert quiet == []
    at_thresholds = results(
        saturation_context_close=0.001,
        saturation_next_open=0.01,
        saturation_context_volume=0.01,
        rho=0.25,
        mismatch_r1_s1=0.05,
        mismatch_r1_s2=0.05,
        mismatch_r1_any=0.05,
        entropy=0.5,
        unused=0.5,
    )
    assert td._raise_flags("training", {"binance": at_thresholds}, None) == []
    loud = results(
        saturation_context_close=0.0011,
        saturation_next_open=0.011,
        saturation_context_amount=0.011,
        rho=0.26,
        mismatch_r1_s1=0.051,
        mismatch_r1_any=0.051,
        entropy=0.49,
        unused=0.51,
    )
    flags = td._raise_flags("training", {"binance": loud}, None)
    assert {flag["flag"] for flag in flags} == {
        "price_saturation_context",
        "price_saturation_next",
        "volume_amount_saturation",
        "rho",
        "r1_token_mismatch",
        "s1_normalised_entropy",
        "s1_unused_share",
    }
    mismatch = next(flag for flag in flags if flag["flag"] == "r1_token_mismatch")
    assert "D-048 (5)" in mismatch["message"] and mismatch["source"] == "binance"
    assert mismatch["statistic"] == td.TOKEN_MISMATCH_STATISTIC == "mismatch_r1_any"
    # the owner's "context tokens": below 5 % for s1 and for s2 alone, above it together
    split = results(mismatch_r1_s1=0.03, mismatch_r1_s2=0.04, mismatch_r1_any=0.06)
    (flag,) = td._raise_flags("training", {"binance": split}, None)
    assert (flag["flag"], flag["value"]) == ("r1_token_mismatch", 0.06)
    assert "mismatch_r1_any = 0.06, threshold 0.05; s1 0.03; s2 0.04)" in flag["message"]
    validation = {"binance": results(perplexity=249.0), "bingx": results(perplexity=250.0)}
    flags = td._raise_flags("validation", validation, 500.0)
    assert [(flag["flag"], flag["source"]) for flag in flags] == [
        ("validation_perplexity_ratio", "binance")
    ]


# -- windows and gaps -------------------------------------------------------------------------


def test_gaps_drop_and_list_windows(tmp_path, short, tokenizer):
    chosen = short["training"]
    gap_context = chosen[2] - 3 * HOUR_MS  # in the context of the third decision
    gap_next = chosen[4] + 10 * HOUR_MS  # in the next rows of the fifth only
    data = part_a_world(tmp_path, chosen, missing=(gap_context, gap_next))
    prepared = td.prepare_diag(
        "training", params=PARAMS, steps=steps_for(tokenizer), root=data, directory=tmp_path
    )
    binance = prepared.sources["binance"]
    kept = set(binance.decision_ms.tolist())
    assert kept == {chosen[0], chosen[1], chosen[3], chosen[5]}
    dropped = {item["decision"]: item for item in binance.dropped}
    third = dropped[utc_text(chosen[2])]
    assert third["missing"] == [[utc_text(gap_context), utc_text(gap_context + HOUR_MS)]]
    assert third["missing_candles"] == 1
    assert dropped[utc_text(chosen[4])]["missing"] == [
        [utc_text(gap_next), utc_text(gap_next + HOUR_MS)]
    ]
    assert binance.values.shape[1:] == (561, 6)
    assert prepared.reference.windows == 4 and prepared.reference.values.shape[1:] == (512, 6)
    # the kept windows hold the inference floats of the store's candles
    first = StoreWindowSource("binance", root=data, end_ms=chosen[-1] + 49 * HOUR_MS)
    window = first.window(datetime.fromtimestamp(chosen[0] / 1000, UTC), horizon=1)
    assert np.array_equal(binance.values[0, :512], window.context.to_numpy(np.float64))


def test_diag_windows_are_checked_and_read_only():
    values = windows_of(2, 512)
    decisions = np.array([5 * HOUR_MS, 6 * HOUR_MS], dtype=np.int64)
    windows = td.DiagWindows("binance", decisions, values, ())
    assert not windows.values.flags.writeable and windows.windows == 2
    for bad in (
        {"decision_ms": decisions[::-1].copy()},
        {"decision_ms": decisions.astype(np.int32)},
        {"values": values[:, :500].copy()},
        {"values": np.where(values > 0, np.nan, values)},
    ):
        arguments = {"decision_ms": decisions, "values": values, **bad}
        with pytest.raises(td.TokenizerDiagError):
            td.DiagWindows("binance", dropped=(), **arguments)


# -- the estimate -----------------------------------------------------------------------------


def test_estimate_diag_on_a_stand_in_clock_and_gate_3(tokenizer):
    decisions = np.array([to_ms(td.PART_A_FIRST) + 25 * HOUR_MS * k for k in range(40)])
    prepared = td.PreparedDiag(
        part="training",
        config_sha256="0" * 64,
        sources={"binance": td.DiagWindows("binance", decisions, windows_of(40, 561), ())},
        reference=td.DiagWindows("reference", np.zeros(3, dtype=np.int64), windows_of(3, 512), ()),
        training_report=None,
    )
    estimate = td.estimate_diag(prepared, steps=steps_for(tokenizer, clock=[10.0, 13.2]))
    assert estimate["windows"] == 43 and estimate["batch_windows"] == td.BATCH_WINDOWS
    assert estimate["seconds_per_window"] == pytest.approx(3.2 / 32)
    assert estimate["statistics_seconds"] == 0.0  # 40 windows of 25 h: fewer than 20 blocks
    assert estimate["threads"] == 2
    assert estimate["cpu_seconds"] == pytest.approx(43 * 0.1 * 2)
    assert estimate["gate3_cpu_seconds"] == 3600 and estimate["needs_owner_approval"] is False
    td.require_within_gate3(estimate)
    td.require_within_gate3({**estimate, "cpu_seconds": 3600.0})
    with pytest.raises(td.TokenizerDiagRefused, match="gate 3"):
        td.require_within_gate3({**estimate, "cpu_seconds": 3600.5})
    with pytest.raises(td.TokenizerDiagError):
        td.require_within_gate3({"cpu_seconds": float("nan")})
    with pytest.raises(td.TokenizerDiagError, match="clock"):
        td.estimate_diag(prepared, steps=steps_for(tokenizer, clock=[5.0, 5.0]))


def test_the_estimate_adds_the_measured_cost_of_the_pooled_intervals(tokenizer):
    # 24 windows four weeks apart: 24 blocks, enough for intervals
    decisions = np.array(
        [to_ms(td.PART_A_FIRST) + 4 * 7 * 24 * HOUR_MS * k for k in range(24)], dtype=np.int64
    )
    prepared = td.PreparedDiag(
        part="validation",
        config_sha256="0" * 64,
        sources={
            "binance": td.DiagWindows("binance", decisions, windows_of(24, 512), ()),
            "bingx": td.DiagWindows("bingx", decisions, windows_of(24, 512, seed=50), ()),
        },
        reference=td.DiagWindows("reference", np.zeros(2, dtype=np.int64), windows_of(2, 512), ()),
        training_report=None,
    )
    clock = [0.0, 2.4, 100.0, 100.3]  # the batch, then the probe's resamples
    estimate = td.estimate_diag(prepared, steps=steps_for(tokenizer, clock=clock))
    probe = 0.3 / td.PROBE_RESAMPLES * td.RESAMPLES * td.POOLED_STATISTICS * 2
    assert estimate["statistics_seconds"] == pytest.approx(probe)
    assert estimate["seconds_per_window"] == pytest.approx(2.4 / 24)
    assert estimate["cpu_seconds"] == pytest.approx((50 * 0.1 + probe) * 2)
    assert estimate["needs_owner_approval"] is False


# -- whole runs -------------------------------------------------------------------------------


def run_part_a(tmp_path, chosen, tokenizer, name="reports"):
    data = part_a_world(tmp_path, chosen)
    directory = tmp_path / name
    steps = steps_for(tokenizer, clock=[0.0, 1.0])
    prepared = td.prepare_diag(
        "training", params=PARAMS, steps=steps, root=data, directory=directory
    )
    lines: list[str] = []
    result = td.run_diag(
        prepared, params=PARAMS, steps=steps, directory=directory, progress=lines.append
    )
    return prepared, result, lines


def test_part_a_writes_its_report_with_deterministic_figures_and_no_path(
    tmp_path, short, tokenizer
):
    prepared, result, lines = run_part_a(tmp_path / "one", short["training"], tokenizer)
    report = result.report
    for key in (
        "retraining_md_sha256",
        "part",
        "tokenizer",
        "windows",
        "results",
        "flags",
        "s1_counts",
        "s2_counts",
        "figures",
        "runtime",
    ):
        assert key in report, key
    assert report["study"] == "tokenizer-training" and report["tests_consumed"] == 0
    assert report["config_sha256"] == prepared.config_sha256
    assert report["runtime"] == {**report["runtime"], "threads": 2, "seconds": 1.0}
    assert len(report["s1_counts"]) == 16 and sum(report["s1_counts"]) == 6 * 512
    assert set(report["results"]) == {"binance", "reference"}
    binance = report["results"]["binance"]
    for column in td.MISMATCH_COLUMNS:
        assert binance["descriptive"][f"mismatch_context_only_{column}"] == 0.0
        assert 0.0 <= binance["statistics"][f"mismatch_r1_{column}"]["point"] <= 1.0
    r1 = {
        column: binance["statistics"][f"mismatch_r1_{column}"]["point"] for column in ("s1", "s2")
    }
    assert binance["statistics"]["mismatch_r1_any"]["point"] >= max(r1.values())
    assert binance["codebook"]["matched_rows"] == 4 * 512  # the reference's rows
    with_intervals = sum(
        result.get("low") is not None
        for item in report["results"].values()
        for result in item["statistics"].values()
    )
    assert report["statistics_with_intervals"] == with_intervals == 0  # 6 windows: 5 blocks
    assert report["divergences"]["js_first_second_half"] >= 0.0
    for token in ("any", "s1", "s2"):
        assert (
            result.details[f"r1_{token}_mismatch_share"]
            == binance["statistics"][f"mismatch_r1_{token}"]["point"]
        )
    assert result.details["windows_kept"] == {"binance": 6}
    text = result.json_path.read_text(encoding="ascii") + result.markdown_path.read_text("ascii")
    assert str(tmp_path) not in text and str(REPO_ROOT) not in text
    assert json.loads(result.json_path.read_text("ascii"))["study"] == "tokenizer-training"
    assert result.details["report_sha256"] == sha256_file(result.json_path)
    assert any("encoding 6 binance windows" in line for line in lines)
    assert result.summary_lines[0].startswith("tokenizer-training: windows kept binance 6/6")
    # the same inputs give the same figures, byte for byte
    _, again, _ = run_part_a(tmp_path / "two", short["training"], tokenizer)
    assert [f["sha256"] for f in again.report["figures"]] == [
        f["sha256"] for f in report["figures"]
    ]
    for figure in report["figures"]:
        assert (result.json_path.parent / figure["relpath"]).is_file()
        assert figure["relpath"].startswith("figures/tokenizer-training-")


def test_intervals_with_enough_blocks(tmp_path, monkeypatch, tokenizer):
    chosen = g25_after(to_ms(td.PART_A_FIRST), 24, spacing_hours=28 * 24)
    monkeypatch.setattr(td, "decisions_ms", lambda part: chosen)
    monkeypatch.setattr(td, "REFERENCE_SEEDS", (0, 1))
    _, result, _ = run_part_a(tmp_path, chosen, tokenizer)
    statistics = result.report["results"]["binance"]["statistics"]
    rho = statistics["rho"]
    assert rho["method"] == "percentile_block_bootstrap" and rho["n_blocks"] == 24
    assert rho["low"] <= rho["point"] <= rho["high"]
    assert rho["key"] == "tokenizer-training/binance/rho"
    close = statistics["close_error_median_abs_bp"]
    assert close["n"] == 24 * 512 and close["low"] is not None
    assert result.report["statistics_with_intervals"] > 40
    assert result.report["expected_chance_misses"] == pytest.approx(
        result.report["statistics_with_intervals"] * 0.05
    )
    assert result.report["results"]["reference"]["statistics"]["rho"]["low"] is None


def test_part_b_reads_part_a_and_compares_the_sources(tmp_path, short, tokenizer):
    _, part_a, _ = run_part_a(tmp_path, short["training"], tokenizer)
    directory = part_a.json_path.parent
    data = tmp_path / "data-b"
    chosen = short["validation"]
    start = to_ms(VALIDATION_START) - 512 * HOUR_MS
    end = chosen[-1] + HOUR_MS
    binance_store(data, start, end, seed=8)
    fakes.write_bingx_store(data, fakes.synthetic_candles(start, (end - start) // HOUR_MS, seed=9))
    steps = steps_for(tokenizer, clock=[0.0, 2.0])
    prepared = td.prepare_diag(
        "validation", params=PARAMS, steps=steps, root=data, directory=directory
    )
    assert set(prepared.sources) == {"binance", "bingx"}
    assert prepared.training_report["s1_counts"] == part_a.report["s1_counts"]
    assert all(windows.values.shape[1] == 512 for windows in prepared.sources.values())
    result = td.run_diag(prepared, params=PARAMS, steps=steps, directory=directory)
    report = result.report
    assert report["study"] == "tokenizer-validation" and "s1_counts" not in report
    divergences = report["divergences"]
    for key in (
        "js_training_validation_binance",
        "js_training_validation_bingx",
        "js_binance_bingx",
        "perplexity_ratio_binance",
        "perplexity_ratio_bingx",
    ):
        assert 0.0 <= divergences[key] < 10.0, key
    assert (
        divergences["js_first_second_half_part_a"]
        == part_a.report["divergences"]["js_first_second_half"]
    )
    assert "mismatch_r1_s1" not in report["results"]["binance"]["statistics"]
    assert "mismatch_r1_any" not in report["results"]["binance"]["statistics"]
    assert "r1_s1_mismatch_share" not in result.details
    assert "r1_any_mismatch_share" not in result.details
    assert result.details["windows_kept"] == {"binance": 5, "bingx": 5}
    # an edited Part A report is not read
    path = directory / "tokenizer-training.config.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(td.TokenizerDiagError, match="recorded Part A definition"):
        td._training_report(directory)


def test_run_refuses_a_tokenizer_in_training_mode():
    tokenizer, _ = ff.build_tiny_models(0)
    tokenizer.train()
    with pytest.raises(td.TokenizerDiagError, match="eval mode"):
        td._check_tokenizer(tokenizer)
    with pytest.raises(td.TokenizerDiagError, match="torch module"):
        td._check_tokenizer(object())


def test_real_steps_are_the_store_window_source_and_a_clock():
    steps = td.real_steps()
    assert steps.window_source is StoreWindowSource
    assert steps.load_tokenizer is td._load_shipped_tokenizer
    assert isinstance(steps.clock(), float)


@pytest.mark.weights
def test_the_shipped_tokenizer_keeps_the_control_and_the_z_pre_path(loaded_small):
    tokenizer = loaded_small.tokenizer
    assert td._check_tokenizer(tokenizer) == (10, 10)
    values = windows_of(3, 561, seed=21)
    out = td._diagnose_batch(tokenizer, values)
    assert np.count_nonzero(out["mismatch_context_only"]) == 0
    assert out["s1"].shape == (3, 512) and int(out["s1"].max()) < 2**10
    zc = torch.from_numpy(td.normalise_rows(values[:, :512], stats_rows=512))
    with torch.no_grad():
        (z_pre, _), _, _, _ = tokenizer(zc)
        s1, s2 = tokenizer.encode(zc, half=True)
        assert torch.equal(td._z_pre(tokenizer, s1, s2), z_pre)


def test_the_validation_range_ends_before_the_sealed_candles():
    assert td.VALIDATION_CANDLE_END - timedelta(hours=8) == VALIDATION_END
    assert to_ms(VALIDATION_END) <= to_ms(td.VALIDATION_CANDLE_END)
