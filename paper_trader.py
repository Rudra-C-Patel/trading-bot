"""Morgan Tradez -- paper trading engine (NO live brokerage, ever).

Keeps a simulated account in data/paper_state.json and logs every fill to
data/paper_trades.csv. Run it on a schedule:

    python paper_trader.py scan     # after the close: rebuild the watchlist
    python paper_trader.py update   # during/after market: fills + exits
    python paper_trader.py status   # print account, positions, watchlist

Typical cron (weekdays, US/Eastern):
    35 9  * * 1-5  python paper_trader.py update   # right after the open
    05 16 * * 1-5  python paper_trader.py update   # right after the close
    30 16 * * 1-5  python paper_trader.py scan     # evening watchlist

`update` uses the latest daily bar, so intraday runs act on today's
running OHLC and the post-close run settles the day.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

import strategy
from risk_manager import RiskManager
from scanner import get_universe, download_history, rank_and_scan

RM = RiskManager()  # shared risk layer: combined ceiling + ticker exclusivity

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATE_FILE = os.path.join(DATA_DIR, "paper_state.json")
TRADES_CSV = os.path.join(DATA_DIR, "paper_trades.csv")
STARTING_CASH = float(os.environ.get("ACCOUNT_SIZE", 100_000))
WATCHLIST_TTL = 3  # sessions a setup stays actionable


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": STARTING_CASH, "positions": {}, "watchlist": {},
            "created": datetime.now().isoformat()}


def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, STATE_FILE)


def log_trade(row: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    fields = ["timestamp", "ticker", "action", "shares", "price", "stop",
              "reason", "pnl", "r_multiple", "cash_after"]
    new = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def fetch_bars(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Recent daily bars (with today's running bar during market hours)."""
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    data = yf.download(tickers, period="3mo", interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    if data is None or data.empty:
        return out
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            continue
        df = df.dropna(subset=["Close"])
        if len(df):
            out[t] = df
    return out


def equity_now(state: dict, bars: dict[str, pd.DataFrame]) -> float:
    eq = state["cash"]
    for t, pos in state["positions"].items():
        px = float(bars[t]["Close"].iloc[-1]) if t in bars else pos["entry"]
        eq += pos["shares"] * px
    return eq


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_scan(args):
    """Evening job: rebuild the watchlist from a full universe scan."""
    state = load_state()
    universe = get_universe()
    print(f"Scanning {len(universe)} tickers...")
    frames = download_history(universe)
    setups = rank_and_scan(frames, top_pct=args.top_pct)

    today = datetime.now().strftime("%Y-%m-%d")
    kept = {t: w for t, w in state["watchlist"].items()
            if w.get("age", 0) + 1 <= WATCHLIST_TTL}
    for t, w in kept.items():
        w["age"] = w.get("age", 0) + 1
    for s in setups:
        if s.ticker in state["positions"]:
            continue
        kept[s.ticker] = {
            "trigger": round(s.trigger, 4), "stop": round(s.stop, 4),
            "close": round(s.close, 4), "adr_pct": round(s.adr_pct, 2),
            "rs_percentile": s.rs_percentile, "born": today, "age": 0,
        }
    state["watchlist"] = kept
    save_state(state)
    print(f"Watchlist: {len(kept)} names -> {STATE_FILE}")
    for t, w in kept.items():
        print(f"  {t:<7} trigger {w['trigger']:.2f}  stop {w['stop']:.2f}  "
              f"RS {w.get('rs_percentile', '?')}")
    return setups


def cmd_update(args):
    """Fill pending breakouts and manage open positions on the latest bar."""
    state = load_state()
    tickers = sorted(set(state["positions"]) | set(state["watchlist"]))
    if not tickers:
        print("Nothing to do: no positions and empty watchlist. Run `scan` first.")
        return
    bars = fetch_bars(tickers)
    now = datetime.now().isoformat(timespec="seconds")
    equity = equity_now(state, bars)

    # --- manage open positions ------------------------------------------------
    for t in list(state["positions"]):
        if t not in bars:
            continue
        pos = state["positions"][t]
        bar = bars[t].iloc[-1]
        o, h, l, c = (float(bar[k]) for k in ("Open", "High", "Low", "Close"))

        if l <= pos["stop"]:
            fill = min(pos["stop"], o)
            pnl = pos["shares"] * (fill - pos["entry"])
            pos["realized"] = pos.get("realized", 0.0) + pnl
            state["cash"] += pos["shares"] * fill
            r_mult = pos["realized"] / (pos["risk_per_share"] * pos["initial_shares"])
            log_trade({"timestamp": now, "ticker": t, "action": "SELL",
                       "shares": pos["shares"], "price": round(fill, 4),
                       "reason": "stop", "pnl": round(pnl, 2),
                       "r_multiple": round(r_mult, 2),
                       "cash_after": round(state["cash"], 2)})
            print(f"STOPPED OUT {t}: {pos['shares']} @ {fill:.2f} ({r_mult:+.2f}R)")
            del state["positions"][t]
            RM.release("momentum", t, realized_pnl=pos["realized"])
            continue

        target = pos["entry"] + strategy.PARTIAL_R_MULTIPLE * pos["risk_per_share"]
        if not pos.get("partial_done") and h >= target:
            sell = max(1, int(pos["shares"] * strategy.PARTIAL_SELL_FRACTION))
            fill = max(target, o)
            pnl = sell * (fill - pos["entry"])
            pos["realized"] = pos.get("realized", 0.0) + pnl
            state["cash"] += sell * fill
            pos["shares"] -= sell
            pos["partial_done"] = True
            pos["stop"] = max(pos["stop"], pos["entry"])
            log_trade({"timestamp": now, "ticker": t, "action": "SELL",
                       "shares": sell, "price": round(fill, 4),
                       "reason": "partial_5R", "pnl": round(pnl, 2),
                       "cash_after": round(state["cash"], 2)})
            print(f"5R PARTIAL {t}: sold {sell} @ {fill:.2f}, stop -> breakeven")
            if pos["shares"] <= 0:
                del state["positions"][t]
                RM.release("momentum", t, realized_pnl=pos["realized"])
                continue

        if pos.get("partial_done"):
            closes = bars[t]["Close"]
            sma = closes.rolling(strategy.TRAIL_SMA).mean().iloc[-1]
            if not pd.isna(sma) and c < float(sma):
                pnl = pos["shares"] * (c - pos["entry"])
                pos["realized"] = pos.get("realized", 0.0) + pnl
                state["cash"] += pos["shares"] * c
                r_mult = pos["realized"] / (pos["risk_per_share"] * pos["initial_shares"])
                log_trade({"timestamp": now, "ticker": t, "action": "SELL",
                           "shares": pos["shares"], "price": round(c, 4),
                           "reason": "sma_trail", "pnl": round(pnl, 2),
                           "r_multiple": round(r_mult, 2),
                           "cash_after": round(state["cash"], 2)})
                print(f"TRAIL EXIT {t}: {pos['shares']} @ {c:.2f} ({r_mult:+.2f}R)")
                del state["positions"][t]
                RM.release("momentum", t, realized_pnl=pos["realized"])

    # --- fill pending breakouts -------------------------------------------------
    max_positions = args.max_positions
    for t in list(state["watchlist"]):
        if t in state["positions"] or t not in bars:
            continue
        if len(state["positions"]) >= max_positions:
            break
        w = state["watchlist"][t]
        df = bars[t]
        i = len(df) - 1
        enriched = strategy.add_indicators(df) if len(df) >= 50 else None
        vol_ok = (enriched is not None
                  and strategy.breakout_confirmed(enriched, i, w["trigger"]))
        if not vol_ok:
            continue
        o = float(df["Open"].iloc[-1])
        fill = max(o, w["trigger"])
        stop = strategy.compute_stop(fill, w["stop"])
        shares = strategy.position_size(equity, fill, stop)
        cost = shares * fill
        if shares <= 0 or cost > state["cash"]:
            continue
        ok, why = RM.reserve("momentum", t, shares * (fill - stop), equity)
        if not ok:
            print(f"RISK BLOCKED {t}: {why}")
            continue
        state["cash"] -= cost
        state["positions"][t] = {
            "entry": round(fill, 4), "stop": round(stop, 4),
            "initial_stop": round(stop, 4), "shares": shares,
            "initial_shares": shares, "risk_per_share": round(fill - stop, 4),
            "partial_done": False, "realized": 0.0,
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
        }
        log_trade({"timestamp": now, "ticker": t, "action": "BUY",
                   "shares": shares, "price": round(fill, 4),
                   "stop": round(stop, 4), "reason": "breakout",
                   "cash_after": round(state["cash"], 2)})
        print(f"ENTERED {t}: {shares} @ {fill:.2f}, stop {stop:.2f} "
              f"(risk ${shares * (fill - stop):,.0f})")
        del state["watchlist"][t]

    save_state(state)
    print(f"\nEquity: ${equity_now(state, bars):,.2f}  cash ${state['cash']:,.2f}  "
          f"positions {len(state['positions'])}  watchlist {len(state['watchlist'])}")


def cmd_status(args):
    state = load_state()
    tickers = sorted(set(state["positions"]) | set(state["watchlist"]))
    bars = fetch_bars(tickers) if tickers else {}
    equity = equity_now(state, bars)
    print(f"Paper account (started ${STARTING_CASH:,.0f})")
    print(f"  Equity: ${equity:,.2f}   Cash: ${state['cash']:,.2f}   "
          f"P&L: {equity / STARTING_CASH - 1:+.1%}")
    if state["positions"]:
        print("\nOpen positions:")
        for t, p in state["positions"].items():
            px = float(bars[t]["Close"].iloc[-1]) if t in bars else p["entry"]
            upnl = p["shares"] * (px - p["entry"])
            r = (px - p["entry"]) / p["risk_per_share"]
            print(f"  {t:<7} {p['shares']} @ {p['entry']:.2f}  now {px:.2f}  "
                  f"stop {p['stop']:.2f}  {upnl:+,.0f} ({r:+.1f}R)"
                  f"{'  [partial taken]' if p.get('partial_done') else ''}")
    if state["watchlist"]:
        print("\nWatchlist:")
        for t, w in state["watchlist"].items():
            print(f"  {t:<7} trigger {w['trigger']:.2f}  stop {w['stop']:.2f}  "
                  f"age {w.get('age', 0)}/{WATCHLIST_TTL}")
    if not state["positions"] and not state["watchlist"]:
        print("\nFlat, empty watchlist. Run `python paper_trader.py scan`.")


def main():
    p = argparse.ArgumentParser(description="Morgan Tradez paper trader (simulation only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("scan", help="rebuild the watchlist (run after the close)")
    # Paper account runs the top-DECILE experiment (see README): the
    # literal top-2% spec produced 3 trades in 5 backtest years.
    ps.add_argument("--top-pct", type=float, default=0.10)
    pu = sub.add_parser("update", help="process fills and exits on the latest bar")
    pu.add_argument("--max-positions", type=int, default=5)
    sub.add_parser("status", help="print account, positions, watchlist")
    args = p.parse_args()
    {"scan": cmd_scan, "update": cmd_update, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
