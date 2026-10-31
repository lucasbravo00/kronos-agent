"""Kronos input features: the six channels, the ``amount`` proxy and window validation.

Kronos reads six channels per candle, ``open, high, low, close, volume, amount``, and its
normalisation statistics use every row of the DataFrame it receives, so a forecast context must
hold exactly ``max_context`` rows (512 for Kronos-small and Kronos-base). When ``amount`` is
absent the pinned predictor fills it with ``volume * mean(open, high, low, close)`` per row;
:func:`add_amount_proxy` evaluates the same pandas expression so that a pre-computed proxy (the
form training CSVs need) and the implicit one agree bit for bit.

Timestamps are tz-naive pandas Series, UTC by convention: the predictor derives its temporal
embeddings from ``.dt.minute/hour/weekday/day/month`` and never looks at a timezone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import vendored_kronos_dir

FEATURE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")
PRICE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS[:4]
REQUIRED_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS[:5]
AMOUNT_COLUMN = "amount"

UPSTREAM_FIXTURE_RELPATH = "tests/data/regression_input.csv"
UPSTREAM_FIXTURE_STEP = pd.Timedelta(minutes=5)

SYNTHETIC_DEFAULT_STEP = pd.Timedelta(hours=1)
SYNTHETIC_DEFAULT_END = pd.Timestamp("2026-10-29 16:00:00")
SYNTHETIC_START_PRICE = 60_000.0
SYNTHETIC_RETURN_SIGMA = 0.01
SYNTHETIC_WICK_SIGMA = 0.003
SYNTHETIC_VOLUME_MEDIAN = 500.0
SYNTHETIC_VOLUME_SIGMA = 0.6


def add_amount_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``amount = volume * mean(open, high, low, close)`` when absent.

    The expression is the one the pinned ``KronosPredictor`` applies to a context without
    ``amount`` (``df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)``), so a fan
    built from the returned frame equals the fan built from the original one. A frame that
    already carries ``amount`` comes back as an unchanged copy.
    """
    if not isinstance(df, pd.DataFrame):
        # ValueError (not TypeError) is this module's contract
        raise ValueError("context must be a pandas DataFrame")  # noqa: TRY004
    out = df.copy()
    if AMOUNT_COLUMN in out.columns:
        return out
    missing = [column for column in REQUIRED_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(f"cannot build the amount proxy: missing columns {missing}")
    out[AMOUNT_COLUMN] = out["volume"] * out[list(PRICE_COLUMNS)].mean(axis=1)
    return out


def feature_columns_present(df: pd.DataFrame) -> list[str]:
    """The Kronos channels present in ``df``, in :data:`FEATURE_COLUMNS` order."""
    return [column for column in FEATURE_COLUMNS if column in df.columns]


def _check_timestamp_series(series: object, *, name: str, length: int) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ValueError(  # noqa: TRY004 - ValueError is this module's contract
            f"{name} must be a pandas Series, got {type(series).__name__}"
        )
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(  # noqa: TRY004 - ValueError is this module's contract
            f"{name} must be tz-naive (UTC by convention), got dtype {series.dtype}"
        )
    if not pd.api.types.is_datetime64_dtype(series.dtype):
        raise ValueError(f"{name} must have a datetime64 dtype, got {series.dtype}")
    if len(series) != length:
        raise ValueError(f"{name} has {len(series)} entries, expected {length}")
    if series.isna().any():
        raise ValueError(f"{name} contains NaT")
    return series


def window_step(x_timestamps: pd.Series) -> pd.Timedelta:
    """The constant positive step of a strictly increasing timestamp Series.

    Raises ``ValueError`` when the series has fewer than two entries, is not strictly increasing
    or does not advance by one constant step (gaps and irregular bars are rejected).
    """
    if len(x_timestamps) < 2:
        raise ValueError("at least two timestamps are needed to define a step")
    diffs = x_timestamps.diff().iloc[1:]
    if (diffs <= pd.Timedelta(0)).any():
        raise ValueError("timestamps must be strictly increasing")
    step = diffs.iloc[0]
    if not (diffs == step).all():
        raise ValueError(
            "timestamps must advance by a constant step (gaps or irregular bars found)"
        )
    return step


def validate_window(
    df: pd.DataFrame,
    x_timestamps: pd.Series,
    *,
    max_context: int,
    horizon: int | None = None,
    y_timestamps: pd.Series | None = None,
) -> None:
    """Validate a forecast context; raise ``ValueError`` on the first problem found.

    Checks: ``df`` is a DataFrame with exactly ``max_context`` rows (the predictor normalises
    over every row it receives); the columns ``open, high, low, close, volume`` are present
    (``amount`` is optional); the channels present are numeric without NaN or infinities;
    ``x_timestamps`` is a tz-naive ``datetime64`` Series with one entry per row, strictly
    increasing with a constant step; ``horizon``, when given, is a positive integer; and
    ``y_timestamps``, when given, is a Series of ``horizon`` entries continuing that step
    right after the last context timestamp.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(  # noqa: TRY004 - ValueError is this module's contract
            f"context must be a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(max_context, int) or isinstance(max_context, bool) or max_context < 2:
        raise ValueError(f"max_context must be an integer >= 2, got {max_context!r}")
    if len(df) != max_context:
        raise ValueError(
            f"context must have exactly {max_context} rows (normalisation uses every row), "
            f"got {len(df)}"
        )
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"context is missing columns {missing}")
    columns = feature_columns_present(df)
    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(df[column].dtype)
        or pd.api.types.is_bool_dtype(df[column].dtype)
    ]
    if non_numeric:
        raise ValueError(f"context columns must be numeric, got {non_numeric}")
    values = df[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = [column for column, ok in zip(columns, np.isfinite(values).all(axis=0)) if not ok]
        raise ValueError(f"context contains NaN or infinite values in columns {bad}")
    _check_timestamp_series(x_timestamps, name="x_timestamps", length=len(df))
    step = window_step(x_timestamps)
    if horizon is not None and (
        not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1
    ):
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    if y_timestamps is None:
        return
    if horizon is None:
        raise ValueError("horizon is required to validate y_timestamps")
    _check_timestamp_series(y_timestamps, name="y_timestamps", length=horizon)
    expected_first = x_timestamps.iloc[-1] + step
    if y_timestamps.iloc[0] != expected_first:
        raise ValueError(
            f"y_timestamps must start one step after the context ({expected_first}), "
            f"got {y_timestamps.iloc[0]}"
        )
    if horizon > 1 and not (y_timestamps.diff().iloc[1:] == step).all():
        raise ValueError(f"y_timestamps must continue the context step of {step}")


def future_timestamps(x_timestamps: pd.Series, step: pd.Timedelta, horizon: int) -> pd.Series:
    """``horizon`` timestamps continuing ``x_timestamps`` with ``step`` (tz-naive Series)."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    start = x_timestamps.iloc[-1] + step
    return pd.Series(pd.date_range(start=start, periods=horizon, freq=step))


def upstream_fixture_frame() -> pd.DataFrame:
    """The upstream regression input as shipped at the pinned commit.

    Columns ``timestamps`` (parsed, tz-naive) and the six channels: real 5-minute bars of a
    session-based market with ``amount`` present, 2500 rows.
    """
    path = vendored_kronos_dir() / UPSTREAM_FIXTURE_RELPATH
    frame = pd.read_csv(path, parse_dates=["timestamps"])
    expected = ["timestamps", *FEATURE_COLUMNS]
    if list(frame.columns) != expected:
        raise ValueError(f"unexpected columns in {UPSTREAM_FIXTURE_RELPATH}: {list(frame.columns)}")
    return frame


def upstream_fixture_window(n: int = 512) -> tuple[pd.DataFrame, pd.Series, pd.Timedelta]:
    """First ``n`` candles of the upstream fixture on a regular 5-minute timestamp grid.

    The fixture holds real bars of a market with trading sessions, so its own timestamps carry
    session gaps (lunch breaks, nights, weekends) and would fail :func:`validate_window`. The
    returned ``x_timestamps`` is a regular 5-minute grid starting at the fixture's first
    timestamp; the channel values are untouched. The upstream regression replica
    (``kronos_agent.bench``) uses :func:`upstream_fixture_frame` with the raw timestamps.
    """
    frame = upstream_fixture_frame()
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= len(frame):
        raise ValueError(f"n must be an integer between 1 and {len(frame)}, got {n!r}")
    df = frame.iloc[:n][list(FEATURE_COLUMNS)].reset_index(drop=True)
    x_timestamps = pd.Series(
        pd.date_range(start=frame["timestamps"].iloc[0], periods=n, freq=UPSTREAM_FIXTURE_STEP)
    )
    return df, x_timestamps, UPSTREAM_FIXTURE_STEP


def synthetic_window(
    n: int = 512,
    *,
    step: pd.Timedelta = SYNTHETIC_DEFAULT_STEP,
    end: pd.Timestamp = SYNTHETIC_DEFAULT_END,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Timedelta]:
    """Deterministic geometric-Brownian-motion candles with positive volume and the amount proxy.

    Log-returns are N(0, 1 %) per candle from a 60 000 start, wicks a few tenths of a percent
    beyond the body, volume log-normal around 500 and ``amount`` the proxy of
    :func:`add_amount_proxy`. Timestamps are a regular grid of ``step`` (default 1 h) ending
    at ``end`` (default 2026-10-29 16:00, tz-naive). The same ``seed`` always yields the same
    frame.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    step = pd.Timedelta(step)
    if step <= pd.Timedelta(0):
        raise ValueError(f"step must be positive, got {step}")
    end = pd.Timestamp(end)
    if end.tzinfo is not None:
        raise ValueError("end must be tz-naive (UTC by convention)")
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, SYNTHETIC_RETURN_SIGMA, size=n)
    close = SYNTHETIC_START_PRICE * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate(([SYNTHETIC_START_PRICE], close[:-1]))
    upper_wick = np.abs(rng.normal(0.0, SYNTHETIC_WICK_SIGMA, size=n))
    lower_wick = np.abs(rng.normal(0.0, SYNTHETIC_WICK_SIGMA, size=n))
    high = np.maximum(open_, close) * (1.0 + upper_wick)
    low = np.minimum(open_, close) * (1.0 - lower_wick)
    volume = rng.lognormal(
        mean=np.log(SYNTHETIC_VOLUME_MEDIAN), sigma=SYNTHETIC_VOLUME_SIGMA, size=n
    )
    df = add_amount_proxy(
        pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
    )
    x_timestamps = pd.Series(pd.date_range(end=end, periods=n, freq=step))
    return df, x_timestamps, step


__all__ = [
    "AMOUNT_COLUMN",
    "FEATURE_COLUMNS",
    "PRICE_COLUMNS",
    "REQUIRED_COLUMNS",
    "UPSTREAM_FIXTURE_STEP",
    "add_amount_proxy",
    "feature_columns_present",
    "future_timestamps",
    "synthetic_window",
    "upstream_fixture_frame",
    "upstream_fixture_window",
    "validate_window",
    "window_step",
]
