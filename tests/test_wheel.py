"""Wheel bot pure logic + backtest pricing layer."""

import os
import sys
from math import log, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config            # noqa: E402
import wheel_backtest    # noqa: E402
import wheel_bot         # noqa: E402


# ---------------------------------------------------------------------------
# Black-Scholes layer
# ---------------------------------------------------------------------------

def bs_delta(S, K, T, sigma, right, r=config.RISK_FREE_RATE):
    d1 = (log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * sqrt(T))
    n = NormalDist().cdf
    return n(d1) if right == "C" else n(d1) - 1


def test_strike_for_delta_round_trip():
    S, T = 87.0, 30 / 365
    for sigma in (0.18, 0.30, 0.55):
        Kp = wheel_backtest.strike_for_delta(S, T, sigma, 0.30, "P")
        Kc = wheel_backtest.strike_for_delta(S, T, sigma, 0.20, "C")
        assert Kp < S < Kc
        # rounding to listed increments allows a small delta gap
        assert abs(abs(bs_delta(S, Kp, T, sigma, "P")) - 0.30) < 0.06
        assert abs(bs_delta(S, Kc, T, sigma, "C") - 0.20) < 0.06


def test_bs_price_floors_at_intrinsic():
    assert wheel_backtest.bs_price(90, 100, 0.0, 0.3, "P") == 10.0
    assert wheel_backtest.bs_price(110, 100, 0.0, 0.3, "C") == 10.0
    assert wheel_backtest.bs_price(100, 100, 30 / 365, 0.0, "P") == 0.0


def test_put_call_parity():
    S, K, T, sigma, r = 100.0, 97.0, 30 / 365, 0.3, config.RISK_FREE_RATE
    c = wheel_backtest.bs_price(S, K, T, sigma, "C")
    p = wheel_backtest.bs_price(S, K, T, sigma, "P")
    assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-9


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------

def test_put_management_trigger():
    assert wheel_bot.decide_put_action(96, 97, 3) == "manage"      # ITM near expiry
    assert wheel_bot.decide_put_action(98.5, 97, 3) == "manage"    # within 2%
    assert wheel_bot.decide_put_action(105, 97, 3) == "hold"       # safely OTM
    assert wheel_bot.decide_put_action(90, 97, 20) == "hold"       # deep ITM but far out


def test_call_assignment_vs_roll():
    # profitable (strike >= basis): let it be called away
    assert wheel_bot.decide_call_action(105, 100, 2, basis=95) == "allow_assignment"
    # unprofitable (strike < basis): try to roll
    assert wheel_bot.decide_call_action(105, 100, 2, basis=110) == "try_roll"
    assert wheel_bot.decide_call_action(95, 100, 2, basis=95) == "hold"   # OTM
    assert wheel_bot.decide_call_action(105, 100, 20, basis=95) == "hold" # far out


def test_review_trigger():
    assert wheel_bot.needs_review(84.9, 100)
    assert not wheel_bot.needs_review(85.1, 100)


def test_concentration_caps():
    # 10% of 100K = 10K per ticker; K=50 -> cap 2 contracts
    assert wheel_bot.max_new_contracts(100_000, 50, active_tickers=3) == 2
    # but only 1 until MIN_CONCURRENT_TICKERS names are active
    assert wheel_bot.max_new_contracts(100_000, 50, active_tickers=0) == 1
    # strike too big for the cap -> no trade
    assert wheel_bot.max_new_contracts(100_000, 120, active_tickers=3) == 0
    # pyramiding respects existing contracts
    assert wheel_bot.max_new_contracts(100_000, 50, 3, existing_ticker_contracts=2) == 0


def test_spread_and_vol_filters():
    assert wheel_bot.spread_ok(1.00, 1.04)
    assert not wheel_bot.spread_ok(1.00, 1.11)
    assert not wheel_bot.spread_ok(0.0, 0.05)          # no real bid
    assert wheel_bot.vol_ok(0.35)
    assert not wheel_bot.vol_ok(0.85)
    assert not wheel_bot.vol_ok(float("nan"))


def test_delta_band_guard():
    # target 0.30, band +/- MAX_DELTA_DISTANCE (0.08): 0.22-0.38 inclusive
    assert wheel_bot.delta_ok(-0.29, 0.30)             # puts quote negative
    assert wheel_bot.delta_ok(0.38, 0.30)
    assert wheel_bot.delta_ok(0.22, 0.30)
    assert not wheel_bot.delta_ok(-0.42, 0.30)         # the live 2026-07-18 case
    assert not wheel_bot.delta_ok(0.21, 0.30)
    assert not wheel_bot.delta_ok(None, 0.30)
    assert not wheel_bot.delta_ok(float("nan"), 0.30)


# Regression for the 2026-07-18 live finding: pick_by_delta returned the
# closest spread-passing delta with no cap on distance from target, handing
# back 0.41-0.46-delta puts for a 0.30 target. Exercises the real Alpaca
# selection loop over canned chain snapshots -- no credentials, no network.

def _fake_snap(delta, bid, ask):
    return {"greeks": {"delta": delta},
            "latestQuote": {"bp": bid, "ap": ask}}


def _alpaca_pick(snapshots, right="P", target=0.30):
    import alpaca_broker
    b = object.__new__(alpaca_broker.AlpacaBroker)   # skip __init__: no creds
    b._paged = lambda *a, **k: iter(snapshots.items())
    return b.pick_by_delta("KO", "2026-08-14", 80.0, right, target)


def test_pick_rejects_out_of_band_delta(capsys):
    # acceptable spread (2.5% of mid) but delta 0.42: must be refused even
    # though it is the closest -- and only -- spread-passing candidate
    q = _alpaca_pick({"KO260814P00079000": _fake_snap(-0.42, 2.00, 2.05)})
    assert q is None
    assert "DELTA-BAND REJECT" in capsys.readouterr().out


def test_pick_accepts_in_band_delta():
    q = _alpaca_pick({"KO260814P00075000": _fake_snap(-0.29, 1.00, 1.04)})
    assert q is not None
    assert q.strike == 75.0 and q.delta == -0.29 and q.handle.endswith("75000")


def test_pick_still_selects_normally_with_guard():
    # in-band 0.29 and out-of-band 0.42 both pass spread: normal
    # closest-delta selection picks 0.29 and the guard stays silent
    both = {"KO260814P00079000": _fake_snap(-0.42, 2.00, 2.05),
            "KO260814P00075000": _fake_snap(-0.29, 1.00, 1.04)}
    q = _alpaca_pick(both)
    assert q is not None and q.strike == 75.0


def test_earnings_crossing():
    earnings = [pd.Timestamp("2024-04-25")]
    assert wheel_bot.expiry_crosses_earnings(
        "2024-05-17", earnings, asof="2024-04-20")
    assert not wheel_bot.expiry_crosses_earnings(
        "2024-04-24", earnings, asof="2024-04-20")
    assert not wheel_bot.expiry_crosses_earnings(
        "2024-05-17", [], asof="2024-04-20")   # unknown = not blocked (disclosed)


# ---------------------------------------------------------------------------
# Entry signal: hand-built cases + parity with the vectorized backtest signal
# ---------------------------------------------------------------------------

def _trend_frame(n=120, dip_day=None, close_above=True):
    """Rising tape; optionally one bar that wicks to the 21 EMA."""
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(np.linspace(100, 130, n), index=idx)
    low = close - 0.5
    high = close + 0.5
    if dip_day is not None:
        e = wheel_bot.ema(close, config.EMA_PERIOD)
        low.iloc[dip_day] = e.iloc[dip_day] - 0.10          # wick through the EMA
        if not close_above:
            close.iloc[dip_day] = e.iloc[dip_day] - 0.05    # ...and close below it
    return pd.DataFrame({"Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": 1_000_000}, index=idx)


def test_entry_signal_fires_only_on_wick_and_hold():
    assert not wheel_bot.entry_signal(_trend_frame())                    # no touch
    assert wheel_bot.entry_signal(_trend_frame(dip_day=119))             # wick + hold
    assert not wheel_bot.entry_signal(_trend_frame(dip_day=119,
                                                   close_above=False))   # closed below


def test_entry_signal_matches_vectorized_backtest_signal():
    df = _trend_frame(dip_day=119)
    enriched = wheel_backtest.enrich(df)
    for i in (80, 100, 119):
        assert bool(enriched["signal"].iloc[i]) == \
            wheel_bot.entry_signal(df.iloc[: i + 1])
