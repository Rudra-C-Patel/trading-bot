"""Shared indicator helpers -- pure pandas, trailing-only."""

import numpy as np
import pandas as pd

ATR_PERIOD = 14


def atr(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    """Wilder's ATR on Open/High/Low/Close bars."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()
