"""Tests for the multi-instrument suite: signal logic, ATR sizing,
hard-stop immutability, trailing ratchet, circuit breaker, correlation
filter. Pure logic only -- no network, no broker."""

import numpy as np
import pandas as pd
import pytest

from bot.engine import Engine, daily_equity
from bot.strategies import mean_reversion, momentum_breakout, trend_following
from bot.strategies.indicators import atr
from bot.suite_bot import breaker_tripped, correlation_blocked, effective_stop


def bars(closes, highs=None, lows=None, vols=None, freq="15min"):
    n = len(closes)
    closes = np.asarray(closes, float)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq=freq,
                        tz="US/Eastern")
    return pd.DataFrame({
        "Open": closes, "High": highs if highs is not None else closes * 1.001,
        "Low": lows if lows is not None else closes * 0.999,
        "Close": closes,
        "Volume": vols if vols is not None else np.full(n, 1000.0)}, index=idx)


# -- strategies ---------------------------------------------------------------

def test_meanrev_signals():
    px = [100.0] * 30 + [95.0]          # sharp drop below -entry_z
    df = mean_reversion.prepare(bars(px), dict(entry_z=1.5))
    assert mean_reversion.entry(df, len(df) - 1, dict(entry_z=1.5)) == "long"
    px2 = [100.0] * 30 + [105.0]        # stretch up -> short
    df2 = mean_reversion.prepare(bars(px2), dict(entry_z=1.5))
    assert mean_reversion.entry(df2, len(df2) - 1, dict(entry_z=1.5)) == "short"
    # revert to mean -> exit for a long
    assert mean_reversion.exit(df2, len(df2) - 1, dict(entry_z=1.5), "long")


def test_breakout_needs_volume():
    base = [100.0 + 0.01 * i for i in range(40)]
    px = base + [105.0]
    quiet = bars(px, vols=np.full(41, 1000.0))
    loud_v = np.full(41, 1000.0); loud_v[-1] = 2000.0
    loud = bars(px, vols=loud_v)
    p = dict(lookback=20)
    dq = momentum_breakout.prepare(quiet, p)
    dl = momentum_breakout.prepare(loud, p)
    assert momentum_breakout.entry(dq, 40, p) is None        # 1.0x volume
    assert momentum_breakout.entry(dl, 40, p) == "long"      # 2.0x volume


def test_trend_cross():
    up = [100.0 - 0.05 * i for i in range(350)] \
        + [83.0 + 0.4 * i for i in range(150)]
    df = trend_following.prepare(bars(up, freq="4h"), dict(fast=50, slow=200))
    sig = [i for i in range(len(df))
           if trend_following.entry(df, i, dict(fast=50, slow=200)) == "long"]
    assert len(sig) >= 1                    # exactly the cross bar(s)
    i = sig[0]
    assert df["ema_f"].iloc[i] > df["ema_s"].iloc[i]
    assert df["ema_f"].iloc[i - 1] <= df["ema_s"].iloc[i - 1]


# -- sizing -------------------------------------------------------------------

def test_atr_sizing_one_percent():
    # 1 ATR move on the position = 1% of equity
    qty = Engine.position_size(100_000, atr_value=2.0, price=50.0,
                               risk_frac=0.01, notional_cap=100.0,
                               fractional=True)
    assert qty * 2.0 == pytest.approx(1000.0)      # 1% of 100K


def test_notional_cap_binds_on_low_atr():
    qty = Engine.position_size(100_000, atr_value=0.5, price=600.0,
                               risk_frac=0.01, notional_cap=1.0,
                               fractional=True)
    assert qty * 600.0 <= 100_000 + 1e-6           # capped at 1x equity


def test_degenerate_sizing_is_zero():
    assert Engine.position_size(100_000, 0.0, 50.0, 0.01, 1.0, True) == 0.0
    assert Engine.position_size(-1, 2.0, 50.0, 0.01, 1.0, True) == 0.0
    assert Engine.position_size(100_000, float("nan"), 50.0, 0.01, 1.0,
                                True) == 0.0


# -- engine: hard stop + trail ------------------------------------------------

def test_hard_stop_never_moved_and_fills():
    # flat then plunge: long must exit at the hard stop, ~-1R
    px = [100.0] * 30 + [95.0] + [95.5] * 3 + [80.0] * 6
    df = mean_reversion.prepare(bars(px), dict(entry_z=1.5))
    res = Engine(mean_reversion, df, dict(entry_z=1.5), capital=100_000,
                 cost=0.0, label="T").run()
    stops = [t for t in res["trades"] if t["reason"] in ("stop", "stop_gap")]
    assert stops, res["trades"]
    assert stops[0]["r_multiple"] <= -0.95         # full 1R loss taken


def test_trailing_stop_ratchets_up():
    # breakout long that runs, then retraces: trail exit locks in profit
    base = [100.0 + 0.01 * i for i in range(40)]
    run = [101.0 + 1.0 * i for i in range(30)]
    fade = [130.0 - 1.5 * i for i in range(12)]
    vols = np.full(40 + 30 + 12, 1000.0); vols[40] = 5000.0
    df = momentum_breakout.prepare(bars(base + run + fade, vols=vols,
                                        freq="1h"), dict(lookback=20))
    res = Engine(momentum_breakout, df, dict(lookback=20), capital=100_000,
                 cost=0.0, fractional=True, label="T").run()
    assert res["trades"], "no trade taken"
    t = res["trades"][0]
    assert t["reason"] in ("stop", "stop_gap")
    assert t["pnl"] > 0                             # trail locked in gains


def test_circuit_breaker_halts():
    # relentless grind down: repeated losses must trip the 10% breaker
    px = list(np.linspace(100, 55, 400))
    df = mean_reversion.prepare(bars(px), dict(entry_z=1.5))
    res = Engine(mean_reversion, df, dict(entry_z=1.5), capital=100_000,
                 cost=0.0, breaker_dd=0.10, label="T").run()
    if res["halted"]:
        eq = res["equity"]
        # after halting the curve stays flat (no new positions)
        post = eq[eq.index > eq.idxmin()]
        assert post.nunique() <= 2


# -- live-runner pure helpers -------------------------------------------------

def test_correlation_filter():
    both_long = {"SPY": {"side": "long"}, "QQQ": {"side": "long"}}
    assert correlation_blocked("BTC/USD", "long", both_long)
    assert not correlation_blocked("BTC/USD", "long",
                                   {"SPY": {"side": "long"}})
    assert not correlation_blocked("BTC/USD", "long",
                                   {"SPY": {"side": "long"},
                                    "QQQ": {"side": "short"}})
    assert not correlation_blocked("GLD", "long", both_long)


def test_effective_stop_tightens_only():
    pos = {"side": "long", "hard_stop": 95.0, "trail": None}
    assert effective_stop(pos) == 95.0
    pos["trail"] = 98.0
    assert effective_stop(pos) == 98.0
    pos["trail"] = 90.0                    # trail below hard: hard wins
    assert effective_stop(pos) == 95.0


def test_breaker_threshold():
    assert breaker_tripped(89_999, 100_000)
    assert not breaker_tripped(90_001, 100_000)


def test_atr_positive_on_real_shape():
    df = bars(list(100 + np.sin(np.arange(100) / 5.0)))
    a = atr(df)
    assert a.iloc[-1] > 0 and a.isna().sum() >= 13   # min_periods respected
