"""Morgan Tradez -- backtester (2020-2024 by default).

Event-driven daily simulation of the strategy in strategy.py over the
current S&P 500 + NASDAQ 100 universe using yfinance data.

Mechanics per simulated day:
  1. Manage open positions against today's OHLC (stops, 5R partials,
     20 SMA trail).
  2. Check yesterday's watchlist: if a setup's trigger prints today on
     breakout-grade volume, enter at max(open, trigger) with slippage.
  3. After the "close", rebuild the watchlist: rank RS percentiles across
     the universe, keep the top 2%, run the full setup check.

Honesty notes (also in the final report):
  * The universe is TODAY'S index membership -- survivorship bias inflates
    results. Delisted 2020-2024 winners/losers are absent.
  * Fills assume you get the breakout price +0.1% slippage; fast movers
    fill worse in practice.
  * yfinance adjusted data can differ slightly from broker prints.

Usage:
    python backtest.py                        # full 2020-2024 run
    python backtest.py --start 2022-01-01 --end 2023-12-31
    python backtest.py --capital 25000 --max-positions 4
"""

import argparse
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

import strategy
from scanner import get_universe, MIN_UNIVERSE_PRICE, MIN_AVG_DOLLAR_VOLUME

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SLIPPAGE = 0.001          # 0.1% paid on entries and stop exits
WATCHLIST_TTL = 3         # a setup stays actionable for this many sessions
WARMUP_DAYS = 320         # calendar days of extra data before `start` for indicators


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------

def load_price_data(tickers: list[str], start: str, end: str,
                    cache_dir: str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Download (or load cached) daily OHLCV for the whole universe."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f"prices_{start}_{end}_{len(tickers)}.pkl")
    if os.path.exists(cache_file):
        print(f"Loading cached prices: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    dl_start = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    # resume support: big universes take a while, so partial progress is
    # checkpointed and an interrupted download picks up where it left off
    partial_file = cache_file + ".partial"
    frames: dict[str, pd.DataFrame] = {}
    done: set[str] = set()
    if os.path.exists(partial_file):
        with open(partial_file, "rb") as f:
            saved = pickle.load(f)
        frames, done = saved["frames"], set(saved["attempted"])
        print(f"Resuming download: {len(done)} tickers already attempted")

    def fetch_batch(batch: list[str]) -> int:
        """Download one batch into `frames`; returns how many came back non-empty."""
        got = 0
        data = yf.download(batch, start=dl_start, end=end, interval="1d",
                           group_by="ticker", auto_adjust=True, threads=True,
                           progress=False)
        if data is None or data.empty:
            return 0
        for t in batch:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                continue
            df = df.dropna(subset=["Close"])
            if len(df):
                got += 1
            if len(df) >= 250:
                frames[t] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        return got

    todo = [t for t in tickers if t not in done]
    batch_size = 100
    for b in range(0, len(todo), batch_size):
        batch = todo[b:b + batch_size]
        print(f"  downloading {b + 1}-{min(b + batch_size, len(todo))} of {len(todo)}...")
        got = fetch_batch(batch)
        if got == 0 and len(batch) >= 20:
            # a whole batch of real tickers returning empty = Yahoo throttling,
            # not delistings; back off hard and retry the same batch once
            print("  [WARN] empty batch (likely rate-limited); backing off 30s...")
            time.sleep(30)
            got = fetch_batch(batch)
        done.update(batch)
        if (b // batch_size) % 10 == 9:
            with open(partial_file, "wb") as f:
                pickle.dump({"frames": frames, "attempted": sorted(done)}, f)
        time.sleep(1.5)

    # retry rounds: anything still missing is either delisted/short-history
    # (legitimately absent) or a casualty of throttling -- retry to find out
    for round_no in (1, 2):
        missing = [t for t in tickers if t not in frames]
        if not missing:
            break
        print(f"  retry round {round_no}: {len(missing)} tickers still empty...")
        recovered = 0
        for b in range(0, len(missing), 50):
            recovered += fetch_batch(missing[b:b + 50])
            time.sleep(3.0)
        print(f"  retry round {round_no} recovered data for {recovered} tickers")
        if recovered == 0:
            break

    with open(cache_file, "wb") as f:
        pickle.dump(frames, f)
    if os.path.exists(partial_file):
        os.remove(partial_file)
    print(f"Cached {len(frames)} tickers -> {cache_file}")
    return frames


def load_spy_regime(start: str, end: str, cache_dir: str = DATA_DIR) -> pd.Series:
    """Daily bool: was SPY above its own 200-day SMA at the PRIOR close?

    Used as an optional entry gate (--spy-filter). Evaluated at the prior
    close so the gate uses no same-day information: an intraday breakout
    entry cannot know where SPY will close tonight.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"spy_{start}_{end}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            spy = pickle.load(f)
    else:
        dl_start = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
        spy = yf.download("SPY", start=dl_start, end=end, interval="1d",
                          auto_adjust=True, progress=False)
        if spy is None or spy.empty:
            raise RuntimeError("SPY download failed; cannot build the regime gate")
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        with open(cache_file, "wb") as f:
            pickle.dump(spy, f)
    close = spy["Close"].dropna()
    sma200 = close.rolling(200).mean()
    return (close > sma200).shift(1).fillna(False).astype(bool)


def precompute(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Indicator frames per ticker + a dates-x-tickers RS score matrix."""
    enriched, scores = {}, {}
    for t, df in frames.items():
        e = strategy.add_indicators(df)
        enriched[t] = e
        w = sum(strategy.RS_WEIGHTS)
        s = sum(e[f"ret{n}"] * wt for n, wt in zip(strategy.RS_WINDOWS, strategy.RS_WEIGHTS)) / w
        # only rank names passing the universe + liquidity filters that day
        # (price/volume checked per-day, so these two are point-in-time correct)
        ok = (
            (e["Close"] > strategy.MIN_PRICE)
            & (e["adr_pct"] > strategy.MIN_ADR_PCT)
            & (e["Close"] > e["sma50"])
            & (e["Close"] > e["sma200"])
            & (e["Close"] > MIN_UNIVERSE_PRICE)
            & (e["dollar_vol50"] > MIN_AVG_DOLLAR_VOLUME)
        )
        scores[t] = s.where(ok)
    score_matrix = pd.DataFrame(scores)
    rs_pct = score_matrix.rank(axis=1, pct=True)  # cross-sectional percentile per day
    return enriched, rs_pct


# ---------------------------------------------------------------------------
# Portfolio objects
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry: float
    stop: float
    initial_stop: float
    shares: int
    risk_per_share: float
    partial_done: bool = False
    realized: float = 0.0  # P&L banked from partial sells


@dataclass
class WatchItem:
    setup: strategy.Setup
    born: pd.Timestamp
    age: int = 0


class Backtest:
    def __init__(self, enriched: dict[str, pd.DataFrame], rs_pct: pd.DataFrame,
                 start: str, end: str, capital: float, max_positions: int,
                 top_pct: float, regime: pd.Series | None = None):
        self.enriched = enriched
        self.rs_pct = rs_pct
        self.capital0 = capital
        self.cash = capital
        self.max_positions = max_positions
        self.top_pct = top_pct
        self.positions: dict[str, Position] = {}
        self.watchlist: dict[str, WatchItem] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        # positional index per ticker for O(1) date lookup
        self.locs = {t: pd.Series(np.arange(len(df)), index=df.index)
                     for t, df in enriched.items()}
        cal = rs_pct.index
        self.days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
        # optional entry gate: bool per session (already prior-close-shifted
        # by load_spy_regime); None = gate disabled, behavior unchanged
        if regime is not None:
            self.regime = regime.sort_index().reindex(
                self.days, method="ffill").fillna(False).astype(bool)
        else:
            self.regime = None

    # -- helpers ------------------------------------------------------------
    def _bar(self, ticker: str, date: pd.Timestamp):
        loc = self.locs[ticker]
        if date not in loc.index:
            return None, None
        i = int(loc[date])
        return self.enriched[ticker], i

    def _equity(self, date: pd.Timestamp) -> float:
        eq = self.cash
        for t, pos in self.positions.items():
            df, i = self._bar(t, date)
            px = float(df["Close"].iloc[i]) if df is not None else pos.entry
            eq += pos.shares * px
        return eq

    def _record_exit(self, pos: Position, date: pd.Timestamp, price: float,
                     shares: int, reason: str, final: bool):
        pnl = shares * (price - pos.entry)
        self.cash += shares * price
        pos.realized += pnl
        row = {
            "ticker": pos.ticker,
            "entry_date": pos.entry_date.date(),
            "exit_date": date.date(),
            "entry": round(pos.entry, 4),
            "exit": round(price, 4),
            "shares": shares,
            "pnl": round(pnl, 2),
            "reason": reason,
            "final": final,
        }
        if final:
            total_shares = sum(tr["shares"] for tr in self.trades
                               if tr["ticker"] == pos.ticker
                               and tr["entry_date"] == pos.entry_date.date()) + shares
            row["r_multiple"] = round(
                pos.realized / (pos.risk_per_share * total_shares), 2)
        self.trades.append(row)

    # -- daily steps ----------------------------------------------------------
    def manage_positions(self, date: pd.Timestamp):
        for t in list(self.positions):
            pos = self.positions[t]
            df, i = self._bar(t, date)
            if df is None:
                continue
            o, h, l, c = (float(df[k].iloc[i]) for k in ("Open", "High", "Low", "Close"))

            # 1) protective stop (gap-down fills at the open)
            if l <= pos.stop:
                fill = min(pos.stop, o) * (1 - SLIPPAGE)
                self._record_exit(pos, date, fill, pos.shares,
                                  "stop" if pos.stop <= pos.initial_stop else "trail_stop",
                                  final=True)
                del self.positions[t]
                continue

            # 2) 5R partial: sell a slice into strength, stop -> breakeven
            target = pos.entry + strategy.PARTIAL_R_MULTIPLE * pos.risk_per_share
            if not pos.partial_done and h >= target:
                sell = max(1, int(pos.shares * strategy.PARTIAL_SELL_FRACTION))
                fill = max(target, o)  # gap-ups fill better
                self._record_exit(pos, date, fill, sell, "partial_5R", final=False)
                pos.shares -= sell
                pos.partial_done = True
                pos.stop = max(pos.stop, pos.entry)  # never a loser after 5R
                if pos.shares <= 0:
                    del self.positions[t]
                    continue

            # 3) trail the remainder: close below the 20 SMA ends the trade
            sma = df[f"sma{strategy.TRAIL_SMA}"].iloc[i]
            if pos.partial_done and not pd.isna(sma) and c < float(sma):
                self._record_exit(pos, date, c, pos.shares, "sma_trail", final=True)
                del self.positions[t]

    def process_entries(self, date: pd.Timestamp):
        # regime gate (optional): entries may not fire while SPY sat below its
        # 200-day SMA at the prior close; setups still age and expire normally
        regime_ok = self.regime is None or bool(self.regime.loc[date])
        for t in list(self.watchlist):
            item = self.watchlist[t]
            item.age += 1
            if item.age > WATCHLIST_TTL:
                del self.watchlist[t]
                continue
            if not regime_ok:
                continue
            if t in self.positions or len(self.positions) >= self.max_positions:
                continue
            df, i = self._bar(t, date)
            if df is None:
                continue
            s = item.setup
            if not strategy.breakout_confirmed(df, i, s.trigger):
                continue
            o = float(df["Open"].iloc[i])
            fill = max(o, s.trigger) * (1 + SLIPPAGE)
            stop = strategy.compute_stop(fill, s.stop)  # re-clamp 2-5% vs actual fill
            equity = self._equity(date)
            shares = strategy.position_size(equity, fill, stop)
            cost = shares * fill
            if shares <= 0 or cost > self.cash:
                del self.watchlist[t]
                continue
            self.cash -= cost
            self.positions[t] = Position(
                ticker=t, entry_date=date, entry=fill, stop=stop,
                initial_stop=stop, shares=shares,
                risk_per_share=fill - stop)
            del self.watchlist[t]

    def rebuild_watchlist(self, date: pd.Timestamp):
        if date not in self.rs_pct.index:
            return
        row = self.rs_pct.loc[date].dropna()
        if row.empty:
            return
        # strongest RS first, so they get entry priority when slots are scarce
        leaders = row[row >= 1.0 - self.top_pct].sort_values(ascending=False).index
        for t in leaders:
            if t in self.positions or t in self.watchlist:
                continue
            df, i = self._bar(t, date)
            if df is None:
                continue
            setup = strategy.find_setup(df, i, ticker=t)
            if setup is not None:
                setup.rs_percentile = float(row[t]) * 100
                self.watchlist[t] = WatchItem(setup=setup, born=date)

    def run(self) -> pd.DataFrame:
        n = len(self.days)
        for k, date in enumerate(self.days):
            self.manage_positions(date)
            self.process_entries(date)
            self.rebuild_watchlist(date)
            self.equity_curve.append((date, self._equity(date)))
            if k % 125 == 0:
                print(f"  {date.date()}  equity ${self._equity(date):,.0f}  "
                      f"open {len(self.positions)}  ({k}/{n})")
        # liquidate leftovers at the last close so the report is complete
        last = self.days[-1]
        for t in list(self.positions):
            pos = self.positions[t]
            df, i = self._bar(t, last)
            c = float(df["Close"].iloc[i]) if df is not None else pos.entry
            self._record_exit(pos, last, c, pos.shares, "end_of_backtest", final=True)
            del self.positions[t]
        return pd.DataFrame(self.equity_curve, columns=["date", "equity"]).set_index("date")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def perf_metrics(eq: pd.Series, capital: float) -> dict:
    """Extended performance metrics for an equity curve.

    Shared by report() below and walkforward.py so every window is scored
    with identical math. Sortino uses downside deviation vs a 0% target;
    Calmar is CAGR / |max drawdown|. Monthly returns are compounded from
    daily equity changes.
    """
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1 / years) - 1 if years > 0 else 0.0
    max_dd = float((eq / eq.cummax() - 1.0).min())
    std = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    downside = float(np.sqrt((rets.clip(upper=0.0) ** 2).mean())) if len(rets) else 0.0
    if downside > 0:
        sortino = float(rets.mean() / downside * np.sqrt(252))
    else:
        sortino = float("inf") if len(rets) and rets.mean() > 0 else 0.0
    if max_dd < 0:
        calmar = cagr / abs(max_dd)
    else:
        calmar = float("inf") if cagr > 0 else 0.0
    monthly = ((1.0 + rets).resample("ME").prod() - 1.0) if len(rets) \
        else pd.Series(dtype=float)
    return {
        "total_return": eq.iloc[-1] / capital - 1.0,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "monthly": monthly,
    }


def monthly_distribution_lines(monthly: pd.Series) -> list[str]:
    """Report block for the monthly return distribution (shared with walkforward)."""
    lines = [f"Monthly return distribution ({len(monthly)} months):"]
    if not len(monthly):
        lines.append("  (no complete months)")
        return lines
    pos = (monthly > 0).mean()
    lines += [
        f"  positive months:     {pos:>10.1%}",
        f"  mean / median:       {monthly.mean():>10.2%} / {monthly.median():.2%}",
        f"  std dev:             {monthly.std():>10.2%}",
        f"  best / worst:        {monthly.max():>10.2%} / {monthly.min():.2%}",
    ]
    return lines


def report(equity: pd.DataFrame, trades: list[dict], capital: float,
           start: str, end: str) -> str:
    eq = equity["equity"]
    rets = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / capital) ** (1 / years) - 1 if years > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    m = perf_metrics(eq, capital)

    def _f2(v: float) -> str:
        return f"{v:>14.2f}" if np.isfinite(v) else f"{'n/a':>14}"

    tdf = pd.DataFrame(trades)
    lines = [
        "=" * 64,
        "MORGAN TRADEZ BACKTEST REPORT",
        f"Period: {start} -> {end}   Starting capital: ${capital:,.0f}",
        "=" * 64,
        f"Final equity:        ${eq.iloc[-1]:>14,.0f}",
        f"Total return:        {eq.iloc[-1] / capital - 1:>14.1%}",
        f"CAGR:                {cagr:>14.1%}",
        f"Max drawdown:        {dd.min():>14.1%}",
        f"Sharpe (daily, rf=0):{sharpe:>14.2f}",
        f"Sortino (rf=0):      {_f2(m['sortino'])}",
        f"Calmar (CAGR/maxDD): {_f2(m['calmar'])}",
    ]
    lines.append("")
    lines += monthly_distribution_lines(m["monthly"])
    lines.append("")
    lines.append("Annual returns:")
    yearly = eq.resample("YE").last()
    prev = capital
    for ts, v in yearly.items():
        lines.append(f"  {ts.year}: {v / prev - 1:>8.1%}   (year-end equity ${v:,.0f})")
        prev = v

    closed = tdf[tdf["final"]] if not tdf.empty else pd.DataFrame()
    if len(closed):
        rr = closed["r_multiple"].dropna()
        # win/loss on completed trades (partials fold into the final R)
        wins = int((rr > 0).sum())
        lines += ["", f"Round-trip trades:   {len(closed):>10d}",
                  f"Win rate:            {wins / len(closed):>13.1%}"]
        if len(rr):
            lines.append(f"Average R multiple:  {rr.mean():>13.2f}R")
            lines.append(f"Best / worst R:      {rr.max():>6.2f}R / {rr.min():.2f}R")
            if wins:
                lines.append(f"Avg winner:          {rr[rr > 0].mean():>13.2f}R")
            if len(rr) - wins:
                lines.append(f"Avg loser:           {rr[rr <= 0].mean():>13.2f}R")
        lines += ["", "Exit reasons (final exits):"]
        for reason, cnt in closed["reason"].value_counts().items():
            lines.append(f"  {reason:<18} {cnt}")

        # per-year trade breakdown (entry-dated) to expose regime dependence
        by_year = closed.copy()
        by_year["year"] = pd.to_datetime(by_year["entry_date"]).dt.year
        lines += ["", "Trades by entry year:",
                  f"  {'year':<6}{'trades':>7}{'win rate':>10}{'avg R':>8}{'total R':>9}"]
        for yr, grp in by_year.groupby("year"):
            g_rr = grp["r_multiple"].dropna()
            g_win = (g_rr > 0).mean() if len(g_rr) else float("nan")
            lines.append(f"  {yr:<6}{len(grp):>7d}{g_win:>10.1%}"
                         f"{g_rr.mean():>8.2f}{g_rr.sum():>9.1f}")
    else:
        lines += ["", "No trades were taken. Loosen --top-pct or widen the dates."]

    # Survivorship-bias adjustment: the universe is TODAY'S listings, so
    # names delisted during the test are invisible. Published estimates put
    # survivor-only inflation at 1-4 percentage points of annual return
    # (momentum strategies sit at the sensitive end). Stated, not hidden:
    adj_lo, adj_hi = cagr - 0.04, cagr - 0.01
    lines += [
        "",
        "SURVIVORSHIP-BIAS ADJUSTMENT (universe = today's listings only):",
        f"  Raw CAGR {cagr:.1%}; studies estimate survivor-only backtests",
        "  overstate annual returns by 1-4 percentage points, so a",
        f"  defensible expectation is roughly {adj_lo:.1%} to {adj_hi:.1%} CAGR.",
        "  Point-in-time membership + delisted price data (the real fix)",
        "  requires a paid feed (e.g. Norgate Platinum, ~$787/yr).",
    ]

    lines += [
        "",
        "CAVEATS: current index membership only (survivorship bias inflates",
        "results), 0.1% slippage assumption, adjusted Yahoo data, no",
        "commissions, breakout fills at max(open, trigger). Treat the",
        "absolute numbers as optimistic; the relative behavior (drawdowns,",
        "win rate, R distribution) is the honest signal.",
        "=" * 64,
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Backtest the Morgan Tradez strategy")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--top-pct", type=float, default=0.02)
    p.add_argument("--universe", choices=["all", "sp500"], default="all",
                   help="all US common stocks (default) or S&P 500 + NDX only")
    p.add_argument("--universe-limit", type=int, default=0,
                   help="cap universe size for a quick smoke test")
    p.add_argument("--spy-filter", action="store_true",
                   help="entries only fire when SPY closed above its 200-day "
                        "SMA the prior session (regime gate)")
    args = p.parse_args()

    universe = get_universe(source=args.universe)
    if args.universe_limit:
        universe = universe[:args.universe_limit]
    print(f"Universe: {len(universe)} tickers")

    frames = load_price_data(universe, args.start, args.end)
    print(f"Usable tickers: {len(frames)}. Precomputing indicators + RS ranks...")
    enriched, rs_pct = precompute(frames)
    regime = load_spy_regime(args.start, args.end) if args.spy_filter else None

    bt = Backtest(enriched, rs_pct, args.start, args.end, args.capital,
                  args.max_positions, args.top_pct, regime=regime)
    print(f"Simulating {len(bt.days)} sessions...")
    equity = bt.run()

    os.makedirs(DATA_DIR, exist_ok=True)
    trades_path = os.path.join(DATA_DIR, "backtest_trades.csv")
    equity_path = os.path.join(DATA_DIR, "backtest_equity.csv")
    pd.DataFrame(bt.trades).to_csv(trades_path, index=False)
    equity.to_csv(equity_path)

    text = report(equity, bt.trades, args.capital, args.start, args.end)
    print()
    print(text)
    with open(os.path.join(DATA_DIR, "backtest_report.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nSaved: {trades_path}, {equity_path}, data/backtest_report.txt")


if __name__ == "__main__":
    main()
