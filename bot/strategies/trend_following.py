"""Trend following on session-aligned 4-hour bars (GLD / USO).

Spec: long when the 50-period EMA crosses above the 200-period EMA,
exit on the cross back down or on the 3x ATR trailing stop, whichever
comes first. Long-only. Hard stop 3 ATR below entry, never moved.
"""

import pandas as pd

from bot.strategies.indicators import atr, ema

NAME = "trend"
STOP_ATR = 3.0
TRAIL_ATR = 3.0

# Pre-registered walk-forward grid: EMA pair only.
GRID = [dict(fast=50, slow=200), dict(fast=30, slow=120),
        dict(fast=100, slow=300)]


def default_params(symbol: str) -> dict:
    return dict(fast=50, slow=200)


def prepare(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    df["ema_f"] = ema(df["Close"], params["fast"])
    df["ema_s"] = ema(df["Close"], params["slow"])
    df["atr"] = atr(df)
    return df


def _cross(df: pd.DataFrame, i: int) -> str | None:
    f0, s0 = df["ema_f"].iloc[i - 1], df["ema_s"].iloc[i - 1]
    f1, s1 = df["ema_f"].iloc[i], df["ema_s"].iloc[i]
    if pd.isna(f0) or pd.isna(s0) or pd.isna(f1) or pd.isna(s1):
        return None
    if f0 <= s0 and f1 > s1:
        return "up"
    if f0 >= s0 and f1 < s1:
        return "down"
    return None


def entry(df: pd.DataFrame, i: int, params: dict) -> str | None:
    if i < 1 or pd.isna(df["atr"].iloc[i]):
        return None
    return "long" if _cross(df, i) == "up" else None


def exit(df: pd.DataFrame, i: int, params: dict, side: str) -> bool:
    if i < 1:
        return False
    f1, s1 = df["ema_f"].iloc[i], df["ema_s"].iloc[i]
    return bool(not pd.isna(f1) and not pd.isna(s1) and f1 < s1)
