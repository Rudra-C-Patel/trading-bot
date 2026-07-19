"""Mean reversion on 15-minute bars (SPY / QQQ).

Spec: 20-period SMA and stddev of closes; enter when price stretches
entry_z standard deviations from the mean (SPY 1.5 / QQQ 1.8), exit when
price reverts to the mean. Trades both sides (long a downside stretch,
short an upside stretch). Hard stop at 2 ATR from entry, never moved;
no trailing stop -- the exit IS the reversion to the mean.
"""

import pandas as pd

from bot.strategies.indicators import atr

NAME = "meanrev"
STOP_ATR = 2.0
TRAIL_ATR = None

# Pre-registered walk-forward grid: entry stretch only. Selection rule
# lives in bot/walkforward.py and was fixed before any test window ran.
GRID = [dict(entry_z=1.5), dict(entry_z=1.8), dict(entry_z=2.2)]

LOOKBACK = 20


def default_params(symbol: str) -> dict:
    return {"SPY": dict(entry_z=1.5)}.get(symbol, dict(entry_z=1.8))


def prepare(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    df["sma"] = df["Close"].rolling(LOOKBACK).mean()
    sd = df["Close"].rolling(LOOKBACK).std()
    df["z"] = (df["Close"] - df["sma"]) / sd.where(sd > 0)
    df["atr"] = atr(df)
    return df


def entry(df: pd.DataFrame, i: int, params: dict) -> str | None:
    z = df["z"].iloc[i]
    if pd.isna(z) or pd.isna(df["atr"].iloc[i]):
        return None
    if z <= -params["entry_z"]:
        return "long"
    if z >= params["entry_z"]:
        return "short"
    return None


def exit(df: pd.DataFrame, i: int, params: dict, side: str) -> bool:
    z = df["z"].iloc[i]
    if pd.isna(z):
        return False
    return z >= 0 if side == "long" else z <= 0
