"""No lookahead (SPEC section 5; the step-2 "done when" of docs/DATA.md).

Candles opening at or after the decision hour ``D`` must never change what is computed for
``D``: a test invents future candles, alters them (the 1h candle opening exactly at ``D``
included) and demands, bit for bit, the same windows (``sha256``, ``inputs_sha256``, the float
context and its timestamps; BingX, Binance, and Binance under the ``binance_quote_volume``
policy) and the same fan (paths, seeds, signal, ``p_up``, computed with ``forecast_window``
directly), on the store path and on the live path (a BingX-shaped mock answering at ``D + 90 s``
whose unclosed candle at ``D`` changes between the two fetches). Positive controls show that the
comparisons are not vacuous: the candle at ``D - 1h`` changes them, and the candle at
``D - 512h`` changes the window while ``D - 513h`` does not.

Every variant is written to a fresh data root, read through fresh sources, and the store it alters
is checked to have another CSV sha256. The fans of the plain tests come from a stand-in model
whose paths depend on every context value; the test marked ``weights`` repeats them with the real
Kronos-small and the governed parameters (it runs whenever the verified weights are present).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes

from kronos_agent.data.store import BASE_COLUMNS, BINANCE_COLUMNS, StoreId, write_store
from kronos_agent.data.timeutil import HOUR_MS, to_ms
from kronos_agent.data.windows import (
    LIVE_MIN_DELAY_MS,
    DecisionWindow,
    StoreWindowSource,
    build_window,
    live_decision_window,
)
from kronos_agent.exchange.public_market import Candle
from kronos_agent.kronos.decision import fan_p_up, fan_signal, forecast_window
from kronos_agent.kronos.features import FEATURE_COLUMNS
from kronos_agent.kronos.forecast import Fan
from kronos_agent.kronos.weights import LoadedModels
from kronos_agent.models import spec_for
from kronos_agent.params import load_params

PARAMS = load_params()
KRONOS = PARAMS.kronos
D = datetime(2025, 6, 2, 12, tzinfo=UTC)  # validation, after the Kronos cutoff
D_MS = to_ms(D)
CONTEXT = KRONOS.max_context  # 512
SOURCES: tuple[str, ...] = ("bingx", "binance")
FIRST_MS = D_MS - 600 * HOUR_MS
FUTURE_MS = 30 * HOUR_MS
WINDOWS: tuple[str, ...] = ("bingx", "binance", "binance_quote")
PROVENANCE = {"kind": "no-lookahead test candles"}

Data = dict[str, pd.DataFrame]  # the 1h store of each source


# -- candles --------------------------------------------------------------------------------------


def truth() -> Data:
    """The 1h store of each source with 30 hours of candles at and after D."""
    count = (D_MS + FUTURE_MS - FIRST_MS) // HOUR_MS
    return {
        "bingx": fakes.candles_to_frame(fakes.synthetic_candles(FIRST_MS, count, seed=10)),
        "binance": fakes.binance_frame(
            FIRST_MS, count, interval="1h", seed=20, start_price=60_000.0
        ),
    }


TRUTH = truth()


def wild_row(source: str, open_ms: int, index: int) -> list:
    """An invented candle far outside the real ones (valid store strings, OHLC relations kept)."""
    if source == "bingx":
        return [
            open_ms,
            "1.5",
            f"{999_999 - index}.9",
            "0.5",
            f"{500_000 + index}.0",
            "123456.7891",
        ]
    return [
        open_ms,
        "1.50",
        f"{999_999 - index}.90",
        "0.50",
        f"{500_000 + index}.00",
        "123456.789",
        "61728394500.12345",
        "1",
        "1.000",
        "1.00000",
    ]


def columns(source: str) -> tuple[str, ...]:
    return BASE_COLUMNS if source == "bingx" else BINANCE_COLUMNS


def frame_of(source: str, rows: list[list]) -> pd.DataFrame:
    names = columns(source)
    frame = pd.DataFrame(
        {name: pd.Series([row[i] for row in rows], dtype=object) for i, name in enumerate(names)}
    )
    frame["open_ms"] = frame["open_ms"].astype(np.int64)
    return frame


def past_only(data: Data = TRUTH) -> Data:
    return {
        source: frame[frame["open_ms"] < D_MS].reset_index(drop=True)
        for source, frame in data.items()
    }


def appended(data: Data) -> Data:
    """Invented candles opening at D and after appended to every store."""
    out: Data = {}
    for source, frame in data.items():
        opens = range(D_MS, D_MS + FUTURE_MS, HOUR_MS)
        rows = [wild_row(source, open_ms, index) for index, open_ms in enumerate(opens)]
        out[source] = pd.concat([frame, frame_of(source, rows)], ignore_index=True)
    return out


def altered(data: Data, *, at_d_only: bool) -> Data:
    """Every candle opening at or after D (or only the one opening at D) replaced by a wild one."""
    out: Data = {}
    for source, frame in data.items():
        frame = frame.copy()
        mask = frame["open_ms"] == D_MS if at_d_only else frame["open_ms"] >= D_MS
        for index, position in enumerate(np.flatnonzero(mask.to_numpy())):
            row = wild_row(source, int(frame.at[position, "open_ms"]), index)
            for name, value in zip(columns(source)[1:], row[1:], strict=True):
                frame.at[position, name] = value
        assert mask.any()
        out[source] = frame
    return out


def with_new_extreme(data: Data, changes: dict[str, int]) -> Data:
    """The candle opening at ``changes[source]`` of each store gets a new extreme high (far above
    every price of the window) and a doubled volume."""
    out = dict(data)
    for source, open_ms in changes.items():
        frame = data[source].copy()
        (position,) = np.flatnonzero(frame["open_ms"].to_numpy() == open_ms)
        frame.at[position, "high"] = "999999.9" if source == "bingx" else "999999.90"
        volume = frame.at[position, "volume"]
        frame.at[position, "volume"] = format(2 * float(volume) + 1.0, ".4f")
        out[source] = frame
    return out


def every(offset_ms: int) -> dict[str, int]:
    """``{source: D - offset}`` for both sources."""
    return {source: D_MS - offset_ms for source in SOURCES}


# -- outputs --------------------------------------------------------------------------------------


def write(root: Path, data: Data) -> dict[str, str]:
    """Write every 1h store under a fresh ``root``; return each store's CSV sha256."""
    assert not root.exists()
    return {
        source: write_store(StoreId(source, "1h"), frame, provenance=PROVENANCE, root=root)[
            "csv_sha256"
        ]
        for source, frame in data.items()
    }


def windows_of(root: Path) -> dict[str, DecisionWindow]:
    """The windows of D from the stores under ``root`` (fresh sources)."""
    bingx = StoreWindowSource("bingx", root=root)
    binance = StoreWindowSource("binance", root=root)
    policy = KRONOS.amount_policy
    return {
        "bingx": bingx.window(
            D, horizon=KRONOS.horizon_n, max_context=CONTEXT, amount_policy=policy
        ),
        "binance": binance.window(
            D, horizon=KRONOS.horizon_n, max_context=CONTEXT, amount_policy=policy
        ),
        "binance_quote": binance.window(
            D, horizon=KRONOS.horizon_n, max_context=CONTEXT, amount_policy="binance_quote_volume"
        ),
    }


def window_facts(window: DecisionWindow) -> tuple:
    return (
        window.sha256,
        window.inputs_sha256,
        window.candles,
        tuple(window.context.columns),
        window.context.to_numpy(dtype=np.float64).tobytes(),
        window.x_timestamps.to_numpy().tobytes(),
        window.y_timestamps.to_numpy().tobytes(),
        window.last_close,
    )


def assert_same(a: dict[str, DecisionWindow], b: dict[str, DecisionWindow]) -> None:
    assert set(a) == set(b) == set(WINDOWS)
    for name in WINDOWS:
        assert window_facts(a[name]) == window_facts(b[name]), name


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict[str, DecisionWindow]:
    """The windows of D from stores that hold no candle at or after D."""
    root = tmp_path_factory.mktemp("reference") / "data"
    write(root, past_only())
    return windows_of(root)


# -- the store path -------------------------------------------------------------------------------


FUTURE_VARIANTS: dict[str, tuple[Callable[[], Data], Callable[[], Data]]] = {
    # name: (the data it alters, the variant)
    "appended_invented": (past_only, lambda: appended(past_only())),
    "true_future": (past_only, lambda: TRUTH),
    "altered_future": (lambda: TRUTH, lambda: altered(TRUTH, at_d_only=False)),
    "altered_at_d": (lambda: TRUTH, lambda: altered(TRUTH, at_d_only=True)),
}


@pytest.mark.parametrize("name", list(FUTURE_VARIANTS))
def test_candles_at_or_after_d_change_nothing(name, reference, tmp_path):
    base, variant = FUTURE_VARIANTS[name]
    base_sha = write(tmp_path / "base", base())
    sha = write(tmp_path / name, variant())
    for source in SOURCES:  # every store was altered or extended at or after D
        assert sha[source] != base_sha[source], source
        # the window source holds the candle opening at D and later ones
        loaded = StoreWindowSource(source, root=tmp_path / name)
        assert len(loaded.candles_between(D_MS, D_MS + HOUR_MS)) == 1, source
        assert loaded.last_open_ms > D_MS, source
    assert_same(windows_of(tmp_path / name), reference)


CONTROLS: dict[str, tuple[int, bool]] = {
    # name: (offset of the altered candle before D, whether the windows change)
    "d_minus_1h": (HOUR_MS, True),
    "context_start": (CONTEXT * HOUR_MS, True),
    "before_context_start": ((CONTEXT + 1) * HOUR_MS, False),
}


@pytest.mark.parametrize("name", list(CONTROLS))
def test_positive_controls(name, reference, tmp_path):
    offset_ms, changed = CONTROLS[name]
    base_sha = write(tmp_path / "base", past_only())
    sha = write(tmp_path / name, with_new_extreme(past_only(), every(offset_ms)))
    for source in SOURCES:
        assert sha[source] != base_sha[source], source
    result = windows_of(tmp_path / name)
    for window in WINDOWS:
        facts, expected = window_facts(result[window]), window_facts(reference[window])
        assert (facts[0] != expected[0]) == changed, window  # sha256
        assert (facts[1] != expected[1]) == changed, window  # inputs_sha256
        assert (facts[4] != expected[4]) == changed, window  # the float context


def test_the_quote_volume_reaches_only_the_inputs_hash(reference, tmp_path):
    data = past_only()
    frame = data["binance"].copy()
    (position,) = np.flatnonzero(frame["open_ms"].to_numpy() == D_MS - HOUR_MS)
    frame.at[position, "quote_volume"] = "1.00000"
    data["binance"] = frame
    write(tmp_path / "quote", data)
    result = windows_of(tmp_path / "quote")
    quote, plain = result["binance_quote"], result["binance"]
    assert quote.sha256 == reference["binance_quote"].sha256  # the same candles
    assert quote.inputs_sha256 != reference["binance_quote"].inputs_sha256
    assert quote.context["amount"].iloc[-1] == 1.0
    assert window_facts(plain) == window_facts(reference["binance"])


# -- the live path --------------------------------------------------------------------------------


def live_window(data: Data, *, record: list | None = None) -> DecisionWindow:
    """The live window of D from a BingX-shaped mock at D + 90 s holding ``data``'s 1h candles
    (those after the server time are never served) and answering every request with the forming
    candle of D first, its volume different on every fetch (recorded in ``record``)."""
    candles = fakes.frame_to_candles(data["bingx"])
    fake = fakes.FakeBingX(D_MS + LIVE_MIN_DELAY_MS, {"1h": candles})

    def forming_first(call, rows):
        forming = Candle(D_MS, "60000.0", "60150.0", "59850.0", "60010.0", f"{call.index + 1}.5")
        if record is not None:
            record.append(forming.volume)
        return [forming, *(row for row in rows if row.open_ms != D_MS)]

    fake.mutator = forming_first
    client, _ = fakes.bingx_client(fake)
    window = live_decision_window(
        client, D, horizon=KRONOS.horizon_n, max_context=CONTEXT, amount_policy=KRONOS.amount_policy
    )
    assert [call.end_ms for call in fake.klines_calls] == [D_MS - 1, D_MS - 1]
    assert fake.served("1h", D_MS - HOUR_MS, D_MS + 20 * HOUR_MS)[0].open_ms == D_MS  # held
    return window


@pytest.mark.parametrize(
    "name, make",
    [
        ("true_future", lambda: TRUTH),
        ("appended_invented", lambda: appended(past_only())),
        ("altered_future", lambda: altered(TRUTH, at_d_only=False)),
    ],
)
def test_live_window_equals_the_store_window(name, make, reference):
    seen: list[str] = []
    window = live_window(make(), record=seen)
    assert window.fetch_pairs == 1  # the two fetches agree once the forming candle is dropped
    assert seen == ["1.5", "2.5"]  # the forming candle of D reached both fetches, changed
    assert window_facts(window) == window_facts(reference["bingx"])


def test_live_window_positive_control(reference):
    data = with_new_extreme(TRUTH, {"bingx": D_MS - HOUR_MS})
    window = live_window(data)
    assert window.sha256 != reference["bingx"].sha256
    assert window.context.to_numpy().tobytes() != reference["bingx"].context.to_numpy().tobytes()
    later = with_new_extreme(TRUTH, {"bingx": D_MS - (CONTEXT + 1) * HOUR_MS})
    assert window_facts(live_window(later)) == window_facts(reference["bingx"])


# -- fans -----------------------------------------------------------------------------------------


class ContextPredictor:
    """``predict_batch`` stand-in whose paths depend on every context value and timestamp (a
    sha256 of them tilts the walk) and on numpy's global generator, which ``seed_everything``
    seeds from the decision hour before every fan."""

    def predict_batch(
        self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p, **_
    ):
        frames = []
        for df, x_ts, y_ts in zip(df_list, x_timestamp_list, y_timestamp_list, strict=True):
            digest = hashlib.sha256(
                df.to_numpy(dtype=np.float64).tobytes()
                + x_ts.to_numpy().tobytes()
                + y_ts.to_numpy().tobytes()
            ).digest()
            tilt = int.from_bytes(digest[:4], "big") / 2**32 - 0.5
            last = float(df["close"].iloc[-1])
            close = last * np.exp(np.cumsum(np.random.normal(tilt * 0.01, 0.01, size=pred_len)))
            values = np.column_stack(
                [close, close * 1.001, close * 0.999, close, np.ones(pred_len), close]
            )
            frames.append(pd.DataFrame(values, columns=list(FEATURE_COLUMNS), index=y_ts))
        return frames


def stand_in_models() -> LoadedModels:
    return LoadedModels(
        spec=spec_for(KRONOS.model),
        tokenizer=None,
        model=None,
        predictor=ContextPredictor(),
        weights_sha256={"stand-in/model.safetensors": "0" * 64},
        load_seconds=0.0,
    )


def fan_facts(fan: Fan, window: DecisionWindow) -> tuple:
    return (
        fan.paths.tobytes(),
        fan.seeds,
        fan_signal(fan.paths, window.last_close),
        fan_p_up(fan.paths, window.last_close),
    )


def fan_windows(tmp_path: Path) -> dict[str, DecisionWindow]:
    """The BingX windows of D: store path (reference, invented and altered future), live path,
    and the positive control (the candle at D - 1h)."""
    windows: dict[str, DecisionWindow] = {}
    for name, data in (
        ("reference", past_only()),
        ("appended_invented", appended(past_only())),
        ("altered_future", altered(TRUTH, at_d_only=False)),
        ("d_minus_1h", with_new_extreme(past_only(), {"bingx": D_MS - HOUR_MS})),
    ):
        root = tmp_path / name
        write(root, {"bingx": data["bingx"]})
        windows[name] = StoreWindowSource("bingx", root=root).window(
            D, horizon=KRONOS.horizon_n, max_context=CONTEXT, amount_policy=KRONOS.amount_policy
        )
    windows["live"] = live_window(TRUTH)
    return windows


def check_fans(models: LoadedModels, windows: dict[str, DecisionWindow]) -> None:
    fans = {name: forecast_window(models, window, PARAMS) for name, window in windows.items()}
    reference = fan_facts(fans["reference"], windows["reference"])
    assert fans["reference"].paths.shape == (KRONOS.trajectories, KRONOS.horizon_n, 6)
    for name in ("appended_invented", "altered_future", "live"):
        assert fan_facts(fans[name], windows[name]) == reference, name
    control = fan_facts(fans["d_minus_1h"], windows["d_minus_1h"])
    assert control[0] != reference[0]  # other paths
    assert control[1] == reference[1]  # the same seeds: they come from D alone


def test_fans_ignore_candles_at_or_after_d(tmp_path):
    check_fans(stand_in_models(), fan_windows(tmp_path))


@pytest.mark.weights
def test_real_kronos_fans_ignore_candles_at_or_after_d(loaded_small, tmp_path):
    check_fans(loaded_small, fan_windows(tmp_path))


def test_the_stand_in_model_is_not_blind():
    window = build_window(
        fakes.frame_to_candles(TRUTH["bingx"]),
        D,
        source="bingx",
        horizon=KRONOS.horizon_n,
        amount_policy=KRONOS.amount_policy,
    )
    predictor = stand_in_models().predictor

    def paths(context: pd.DataFrame) -> bytes:
        np.random.seed(7)
        frames = predictor.predict_batch(
            [context], [window.x_timestamps], [window.y_timestamps], KRONOS.horizon_n, 1.0, 0, 0.9
        )
        return frames[0].to_numpy().tobytes()

    oldest = window.context.copy()
    oldest.loc[0, "volume"] += 1.0  # the candle opening at D - 512h
    assert paths(window.context) == paths(window.context.copy())
    assert paths(oldest) != paths(window.context)
