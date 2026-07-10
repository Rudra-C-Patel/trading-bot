"""Wheel strategy backtest -- HONEST APPROXIMATION, read the caveats.

Real historical options quotes are not freely available the way stock
prices are. This backtest therefore prices every option with
Black-Scholes using trailing 30-day realized volatility, a flat 4%
risk-free rate, and a 10% premium haircut for spread crossing (all in
config.py). That means:

  * Premiums are ESTIMATES, not fills anyone could have had. Realized
    vol chronically understates implied vol in calm markets (backtest
    premiums too low) and overstates it right after crashes (premiums
    too high at exactly the moments the wheel gets assigned).
  * The bid-ask liquidity filter cannot be simulated (no historical
    quotes) -- it exists only in the live bot.
  * yfinance adjusted prices fold dividends into the price series, so
    dividend income isn't separately credited and historical strikes
    are approximate.
  * Earnings-date avoidance uses yfinance's historical earnings
    calendar, which has gaps; per-ticker coverage is printed in the
    report rather than assumed.

Treat the output as a sanity check of the MECHANICS (assignment
frequency, drawdown shape, premium-vs-loss balance), not as a return
forecast. Do not quote the CAGR without this paragraph attached.

Usage:
    python wheel_backtest.py                          # 2020-2024, $100K
    python wheel_backtest.py --start 2022-01-01 --end 2024-12-31
"""

import argparse
import os
import pickle
from datetime import datetime
from math import erf, exp, log, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

import config
import wheel_bot
from backtest import perf_metrics, monthly_distribution_lines

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_N = NormalDist()


# ---------------------------------------------------------------------------
# Black-Scholes approximation layer
# ---------------------------------------------------------------------------

def _cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float, right: str,
             r: float = config.RISK_FREE_RATE) -> float:
    """European BS price; floors at intrinsic for degenerate inputs."""
    intrinsic = max(S - K, 0.0) if right == "C" else max(K - S, 0.0)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return intrinsic
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if right == "C":
        return S * _cdf(d1) - K * exp(-r * T) * _cdf(d2)
    return K * exp(-r * T) * _cdf(-d2) - S * _cdf(-d1)


def strike_for_delta(S: float, T: float, sigma: float, target_delta: float,
                     right: str, r: float = config.RISK_FREE_RATE) -> float:
    """Invert BS delta for the strike hitting the target (put delta is the
    absolute value). Rounded to a plausible listing increment."""
    z = _N.inv_cdf(target_delta if right == "C" else 1.0 - target_delta)
    K = S * exp((r + 0.5 * sigma * sigma) * T - z * sigma * sqrt(T))
    step = 0.5 if K < 25 else (1.0 if K < 100 else 2.5)
    return round(K / step) * step


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"wheel_prices_{start}_{end}_{len(tickers)}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    dl_start = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=dl_start, end=end, interval="1d",
                       group_by="ticker", auto_adjust=True, threads=True,
                       progress=False)
    frames = {}
    for t in tickers:
        try:
            df = data[t].dropna(subset=["Close"])
        except KeyError:
            continue
        if len(df) > 100:
            frames[t] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    with open(cache, "wb") as f:
        pickle.dump(frames, f)
    return frames


def load_earnings(tickers: list[str]) -> tuple[dict[str, list], dict[str, int]]:
    """Historical + upcoming earnings dates per ticker via yfinance.
    Returns (dates, per-ticker count) so coverage is reported, not assumed."""
    import yfinance as yf
    out, counts = {}, {}
    for t in tickers:
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=40)
            dates = ([pd.Timestamp(d).tz_localize(None).normalize()
                      for d in df.index] if df is not None and len(df) else [])
        except Exception:
            dates = []
        out[t], counts[t] = sorted(set(dates)), len(set(dates))
    return out, counts


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["ema21"] = wheel_bot.ema(close, config.EMA_PERIOD)
    out["sma50"] = close.rolling(config.TREND_SMA).mean()
    out["rv30"] = close.pct_change().rolling(
        config.REALIZED_VOL_WINDOW).std() * np.sqrt(252)
    ema_rising = out["ema21"] > out["ema21"].shift(3)
    out["signal"] = ((out["Low"] <= out["ema21"]) & (close >= out["ema21"])
                     & ema_rising & (close > out["sma50"]))
    return out


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class WheelBacktest:
    def __init__(self, frames: dict[str, pd.DataFrame], earnings: dict[str, list],
                 start: str, end: str, capital: float):
        self.frames = {t: enrich(df) for t, df in frames.items()}
        self.earnings = earnings
        self.capital0 = capital
        self.cash = capital
        self.short_puts: dict[str, dict] = {}
        self.stock: dict[str, dict] = {}
        self.short_calls: dict[str, dict] = {}
        self.review_flags: dict[str, str] = {}
        self.trades: list[dict] = []
        self.stats = {"csp_opened": 0, "csp_expired": 0, "assignments": 0,
                      "cc_opened": 0, "called_away": 0, "put_rolls": 0,
                      "cc_rolls": 0, "review_flags": 0, "earnings_skips": 0,
                      "premium_collected": 0.0}
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        cal = sorted(set().union(*[set(df.index) for df in self.frames.values()]))
        self.days = [d for d in cal
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        self.locs = {t: pd.Series(np.arange(len(df)), index=df.index)
                     for t, df in self.frames.items()}

    # -- helpers --------------------------------------------------------------
    def _bar(self, t, date):
        loc = self.locs[t]
        if date not in loc.index:
            return None
        return self.frames[t].iloc[int(loc[date])]

    def _log(self, date, **row):
        self.trades.append({"date": date.date(), **row})

    def _reserved(self) -> float:
        return sum(p["strike"] * config.CONTRACT_MULTIPLIER * p["contracts"]
                   for p in self.short_puts.values())

    def _active(self) -> set:
        return set(self.short_puts) | set(self.stock)

    def _sell_premium(self, S, K, T, sigma, right) -> float:
        return bs_price(S, K, T, sigma, right) * config.PREMIUM_HAIRCUT

    def _earnings_blocked(self, t, date, expiry) -> bool:
        if not config.AVOID_EARNINGS:
            return False
        blocked = wheel_bot.expiry_crosses_earnings(
            expiry, self.earnings.get(t, []), asof=date)
        if blocked:
            self.stats["earnings_skips"] += 1
        return blocked

    def _expiry_for(self, date) -> pd.Timestamp:
        return date + pd.Timedelta(days=config.TARGET_DTE)

    # -- daily steps ------------------------------------------------------------
    def settle_and_manage(self, date):
        # short puts
        for t in list(self.short_puts):
            p = self.short_puts[t]
            bar = self._bar(t, date)
            if bar is None:
                continue
            S = float(bar["Close"])
            dte = (p["expiry"] - date).days
            if dte <= 0:
                if S < p["strike"]:   # assigned
                    shares = p["contracts"] * config.CONTRACT_MULTIPLIER
                    self.cash -= p["strike"] * shares
                    basis = p["strike"] - p["credit"]
                    self.stock[t] = {"shares": shares, "basis": basis}
                    self.stats["assignments"] += 1
                    self._log(date, ticker=t, action="ASSIGNED",
                              strike=p["strike"], basis=round(basis, 2))
                else:
                    self.stats["csp_expired"] += 1
                    self._log(date, ticker=t, action="PUT_EXPIRED",
                              strike=p["strike"])
                del self.short_puts[t]
                continue
            if wheel_bot.decide_put_action(S, p["strike"], dte) == "manage":
                sigma = float(bar["rv30"]) if np.isfinite(bar["rv30"]) else 0.5
                close_cost = bs_price(S, p["strike"], dte / 365, sigma, "P")
                new_expiry = self._expiry_for(date)
                if self._earnings_blocked(t, date, new_expiry):
                    continue
                new_K = strike_for_delta(S, config.TARGET_DTE / 365, sigma,
                                         config.CSP_TARGET_DELTA, "P")
                new_credit = self._sell_premium(S, new_K, config.TARGET_DTE / 365,
                                                sigma, "P")
                if new_credit - close_cost >= config.ROLL_MIN_CREDIT:
                    n = p["contracts"]
                    self.cash += (new_credit - close_cost) * n * config.CONTRACT_MULTIPLIER
                    self.stats["put_rolls"] += 1
                    self.stats["premium_collected"] += (new_credit - close_cost) * n * config.CONTRACT_MULTIPLIER
                    p.update(strike=new_K, expiry=new_expiry,
                             credit=p["credit"] + new_credit - close_cost)
                    self._log(date, ticker=t, action="PUT_ROLLED",
                              strike=new_K, credit=round(new_credit - close_cost, 2))

        # stock + covered calls
        for t in list(self.stock):
            s = self.stock[t]
            bar = self._bar(t, date)
            if bar is None:
                continue
            S = float(bar["Close"])
            if t not in self.review_flags and wheel_bot.needs_review(S, s["basis"]):
                self.review_flags[t] = str(date.date())
                self.stats["review_flags"] += 1
                self._log(date, ticker=t, action="REVIEW_FLAG",
                          spot=round(S, 2), basis=round(s["basis"], 2))
            call = self.short_calls.get(t)
            if call:
                dte = (call["expiry"] - date).days
                if dte <= 0:
                    if S >= call["strike"]:   # called away
                        self.cash += call["strike"] * s["shares"]
                        self.stats["called_away"] += 1
                        pnl = (call["strike"] - s["basis"]) * s["shares"]
                        self._log(date, ticker=t, action="CALLED_AWAY",
                                  strike=call["strike"], pnl=round(pnl, 2))
                        del self.stock[t]
                        self.review_flags.pop(t, None)
                    else:
                        self._log(date, ticker=t, action="CC_EXPIRED",
                                  strike=call["strike"])
                    del self.short_calls[t]
                    continue
                action = wheel_bot.decide_call_action(S, call["strike"], dte,
                                                      s["basis"])
                if action == "try_roll":
                    sigma = float(bar["rv30"]) if np.isfinite(bar["rv30"]) else 0.5
                    close_cost = bs_price(S, call["strike"], dte / 365, sigma, "C")
                    new_expiry = self._expiry_for(date)
                    if self._earnings_blocked(t, date, new_expiry):
                        continue
                    new_K = strike_for_delta(S, config.TARGET_DTE / 365, sigma,
                                             config.CC_TARGET_DELTA, "C")
                    new_credit = self._sell_premium(S, new_K, config.TARGET_DTE / 365,
                                                    sigma, "C")
                    if new_K > call["strike"] and \
                            new_credit - close_cost >= config.ROLL_MIN_CREDIT:
                        self.cash += (new_credit - close_cost) * call["contracts"] \
                            * config.CONTRACT_MULTIPLIER
                        self.stats["cc_rolls"] += 1
                        self.stats["premium_collected"] += (new_credit - close_cost) \
                            * call["contracts"] * config.CONTRACT_MULTIPLIER
                        call.update(strike=new_K, expiry=new_expiry)
                        self._log(date, ticker=t, action="CC_ROLLED", strike=new_K)
            elif t not in self.review_flags:
                sigma = float(bar["rv30"]) if np.isfinite(bar["rv30"]) else float("nan")
                if not wheel_bot.vol_ok(sigma):
                    continue
                expiry = self._expiry_for(date)
                if self._earnings_blocked(t, date, expiry):
                    continue
                K = strike_for_delta(S, config.TARGET_DTE / 365, sigma,
                                     config.CC_TARGET_DELTA, "C")
                credit = self._sell_premium(S, K, config.TARGET_DTE / 365, sigma, "C")
                n = s["shares"] // config.CONTRACT_MULTIPLIER
                self.cash += credit * n * config.CONTRACT_MULTIPLIER
                self.stats["cc_opened"] += 1
                self.stats["premium_collected"] += credit * n * config.CONTRACT_MULTIPLIER
                self.short_calls[t] = {"strike": K, "expiry": expiry,
                                       "contracts": n, "credit": credit}
                self._log(date, ticker=t, action="CC_OPENED", strike=K,
                          credit=round(credit, 2))

    def open_new_csps(self, date):
        equity = self.mark_to_market(date)
        for t, df in self.frames.items():
            if t in self._active() or t in config.OPTIONS_BLACKLIST:
                continue
            bar = self._bar(t, date)
            if bar is None or not bool(bar["signal"]):
                continue
            sigma = float(bar["rv30"]) if np.isfinite(bar["rv30"]) else float("nan")
            if not wheel_bot.vol_ok(sigma):
                continue
            expiry = self._expiry_for(date)
            if self._earnings_blocked(t, date, expiry):
                continue
            S = float(bar["Close"])
            K = strike_for_delta(S, config.TARGET_DTE / 365, sigma,
                                 config.CSP_TARGET_DELTA, "P")
            n = wheel_bot.max_new_contracts(equity, K, len(self._active()))
            if n <= 0:
                continue
            collateral = K * n * config.CONTRACT_MULTIPLIER
            if self.cash - self._reserved() < collateral:
                continue
            credit = self._sell_premium(S, K, config.TARGET_DTE / 365, sigma, "P")
            self.cash += credit * n * config.CONTRACT_MULTIPLIER
            self.stats["csp_opened"] += 1
            self.stats["premium_collected"] += credit * n * config.CONTRACT_MULTIPLIER
            self.short_puts[t] = {"strike": K, "expiry": expiry,
                                  "contracts": n, "credit": credit}
            self._log(date, ticker=t, action="CSP_OPENED", strike=K,
                      credit=round(credit, 2), contracts=n)

    def mark_to_market(self, date) -> float:
        eq = self.cash
        for t, s in self.stock.items():
            bar = self._bar(t, date)
            S = float(bar["Close"]) if bar is not None else s["basis"]
            eq += s["shares"] * S
        for book, right in ((self.short_puts, "P"), (self.short_calls, "C")):
            for t, o in book.items():
                bar = self._bar(t, date)
                if bar is None:
                    continue
                S = float(bar["Close"])
                sigma = float(bar["rv30"]) if np.isfinite(bar["rv30"]) else 0.5
                T = max((o["expiry"] - date).days, 0) / 365
                eq -= bs_price(S, o["strike"], T, sigma, right) \
                    * o["contracts"] * config.CONTRACT_MULTIPLIER
        return eq

    def run(self) -> pd.DataFrame:
        for k, date in enumerate(self.days):
            self.settle_and_manage(date)
            self.open_new_csps(date)
            self.equity_curve.append((date, self.mark_to_market(date)))
            if k % 250 == 0:
                print(f"  {date.date()}  equity ${self.equity_curve[-1][1]:,.0f}  "
                      f"puts {len(self.short_puts)} stock {len(self.stock)}")
        return pd.DataFrame(self.equity_curve,
                            columns=["date", "equity"]).set_index("date")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(equity: pd.DataFrame, bt: WheelBacktest, capital: float,
           start: str, end: str, earnings_counts: dict[str, int]) -> str:
    eq = equity["equity"]
    m = perf_metrics(eq, capital)

    def f2(v):
        return f"{v:.2f}" if np.isfinite(v) else "n/a"

    st = bt.stats
    lines = [
        "=" * 64,
        "WHEEL STRATEGY BACKTEST (Black-Scholes APPROXIMATION -- see caveats)",
        f"Period: {start} -> {end}   Capital: ${capital:,.0f}   "
        f"Watchlist: {len(bt.frames)} names",
        "=" * 64,
        f"Final equity:        ${eq.iloc[-1]:>14,.0f}",
        f"Total return:        {m['total_return']:>14.1%}",
        f"CAGR:                {m['cagr']:>14.1%}",
        f"Max drawdown:        {m['max_dd']:>14.1%}",
        f"Sharpe (daily,rf=0): {f2(m['sharpe']):>14}",
        f"Sortino (rf=0):      {f2(m['sortino']):>14}",
        f"Calmar:              {f2(m['calmar']):>14}",
        "",
    ]
    lines += monthly_distribution_lines(m["monthly"])
    lines += [
        "",
        "Wheel mechanics:",
        f"  CSPs opened:          {st['csp_opened']}",
        f"  expired worthless:    {st['csp_expired']}",
        f"  assignments:          {st['assignments']}",
        f"  covered calls opened: {st['cc_opened']}",
        f"  called away:          {st['called_away']}",
        f"  put rolls / CC rolls: {st['put_rolls']} / {st['cc_rolls']}",
        f"  earnings skips:       {st['earnings_skips']}",
        f"  REVIEW FLAGS raised:  {st['review_flags']}"
        + (f"  ({', '.join(bt.review_flags)})" if bt.review_flags else ""),
        f"  premium collected:    ${st['premium_collected']:,.0f}",
        "",
        "Earnings-data coverage (dates known per ticker; 20 quarters span",
        "2020-2024 -- anything lower means avoidance ran partially blind):",
        "  " + "  ".join(f"{t}:{earnings_counts.get(t, 0)}"
                         for t in sorted(bt.frames)),
        "",
        "APPROXIMATION CAVEATS (do not quote returns without these):",
        "  * Option premiums are Black-Scholes on 30d realized vol with a",
        f"    {1 - config.PREMIUM_HAIRCUT:.0%} haircut -- estimates, not historical quotes.",
        "    Realized vol underprices premium in calm tape and overprices",
        "    it post-crash, exactly when assignments cluster.",
        "  * No bid-ask spread data exists historically; the live bot's",
        "    liquidity filter is NOT simulated here.",
        "  * Adjusted prices fold dividends into the series; dividend",
        "    income on assigned stock is not separately credited.",
        "  * European exercise assumed (no early assignment).",
        "=" * 64,
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Wheel strategy backtest (BS approximation)")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--capital", type=float, default=config.ACCOUNT_SIZE_FALLBACK)
    args = p.parse_args()

    tickers = [t for t in config.WATCHLIST if t not in config.OPTIONS_BLACKLIST]
    print(f"Loading prices for {len(tickers)} watchlist names...")
    frames = load_prices(tickers, args.start, args.end)
    print(f"Got {len(frames)}. Loading earnings calendars...")
    earnings, counts = load_earnings(list(frames))
    bt = WheelBacktest(frames, earnings, args.start, args.end, args.capital)
    print(f"Simulating {len(bt.days)} sessions...")
    equity = bt.run()

    text = report(equity, bt, args.capital, args.start, args.end, counts)
    print()
    print(text)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "wheel_backtest_report.txt"), "w") as f:
        f.write(text + "\n")
    pd.DataFrame(bt.trades).to_csv(
        os.path.join(DATA_DIR, "wheel_backtest_trades.csv"), index=False)
    equity.to_csv(os.path.join(DATA_DIR, "wheel_backtest_equity.csv"))
    print("\nSaved: data/wheel_backtest_report.txt, _trades.csv, _equity.csv")


if __name__ == "__main__":
    main()
