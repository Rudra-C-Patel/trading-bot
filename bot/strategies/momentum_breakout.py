"""Momentum breakout on 1-hour bars (BTC/USD).

Spec: buy when the close breaks the prior 20-period high with volume at
least 1.5x its 20-period average; trail a 2x ATR stop. Long-only --
Alpaca spot crypto cannot be shorted, so a break of the 20-period low is
an exit signal, not a short entry. Hard stop 2 ATR below entry, never
moved; trailing 2x ATR stop ratchets behind the highest high since entry.
"""

import pandas as pd

from bot.strategies.indicators import atr

NAME = "breakout"
STOP_ATR = 2.0
TRAIL_ATR = 2.0

VOL_MULT = 1.5
VOL_LOOKBACK = 20

# Pre-registered walk-forward grid: breakout lookback only.
GRID = [dict(lookback=20), dict(lookback=40), dict(lookback=55)]


def default_params(symbol: str) -> dict:
    return dict(lookback=20)


def prepare(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    n = params["lookback"]
    df["hh"] = df["High"].rolling(n).max().shift(1)   # prior N-bar high
    df["ll"] = df["Low"].rolling(n).min().shift(1)    # prior N-bar low
    df["vol_avg"] = df["Volume"].rolling(VOL_LOOKBACK).mean().shift(1)
    df["atr"] = atr(df)
    return df


def entry(df: pd.DataFrame, i: int, params: dict) -> str | None:
    row = df.iloc[i]
    if pd.isna(row["hh"]) or pd.isna(row["vol_avg"]) or pd.isna(row["atr"]):
        return None
    if row["Close"] > row["hh"] and row["Volume"] >= VOL_MULT * row["vol_avg"]:
        return "long"
    return None


def exit(df: pd.DataFrame, i: int, params: dict, side: str) -> bool:
    row = df.iloc[i]
    return bool(not pd.isna(row["ll"]) and row["Close"] < row["ll"])
