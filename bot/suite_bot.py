"""Multi-instrument suite paper-trading runner -- ALPACA PAPER ONLY.

Runs the three strategies (mean reversion SPY/QQQ 15-min, momentum
breakout BTC/USD 1-hour, trend following GLD/USO 4-hour) against the
same Alpaca paper account and broker adapter the wheel bot uses. Every
order goes through alpaca_broker.AlpacaBroker, whose connect() hard-
refuses anything that is not the paper host + a PA* account -- the same
enforcement pattern as wheel_bot.

Risk rules (see also bot/engine.py, whose sizing math is reused):
  * 1 ATR move = 1% of equity; per-position notional capped at 1x equity.
  * Hard stop placed broker-side (GTC) at entry -/+ STOP_ATR x ATR and
    NEVER moved; trailing stops only tighten (cancel/replace).
  * Shared risk layer (risk_manager.py): combined 5% open-risk ceiling
    across ALL strategies incl. wheel, ticker exclusivity, per-strategy
    P&L attribution.
  * Correlation filter: no new BTC/USD long while SPY and QQQ suite
    positions are both long.
  * Circuit breaker: account equity 10% below its stored peak -> close
    every suite position, cancel suite stops, halt. Halt persists in
    state until `python -m bot.suite_bot reset-halt` is run deliberately.
    (The wheel bot shares the account; the breaker closes only suite
    positions but is computed on whole-account equity.)

Only instruments that SURVIVED walk-forward validation are traded:
ENABLED below. Trades log to data/trades.csv, daily P&L to
data/daily_pnl.csv, state to data/suite_state.json.

Usage:
    python -m bot.suite_bot scan               # signals only, no broker
    python -m bot.suite_bot update --dry-run   # decide all, place nothing
    python -m bot.suite_bot update             # trade on Alpaca paper
    python -m bot.suite_bot status
    python -m bot.suite_bot reset-halt         # deliberate breaker reset
"""

import argparse
import csv
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from risk_manager import RiskManager
from bot.data import fetch_crypto_bars, fetch_stock_bars, resample_4h_session, rth_only
from bot.engine import Engine
from bot.strategies import INSTRUMENTS

RM = RiskManager()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(DATA_DIR, "suite_state.json")
TRADES_CSV = os.path.join(DATA_DIR, "trades.csv")
DAILY_PNL_CSV = os.path.join(DATA_DIR, "daily_pnl.csv")

# Set after walk-forward validation (Phase 2): only survivors trade.
# 2026-07-13 result: ALL FIVE pairs failed out-of-sample (see the suite
# section of WALKFORWARD_RESULTS.md) -- nothing is enabled.
# Override per-run: SUITE_INSTRUMENTS="SPY,QQQ" python -m bot.suite_bot ...
ENABLED: list[str] = []
if os.environ.get("SUITE_INSTRUMENTS"):
    ENABLED = [s.strip() for s in os.environ["SUITE_INSTRUMENTS"].split(",")]

RISK_FRAC = 0.01          # 1 ATR move = 1% equity
NOTIONAL_CAP = 1.0        # per-position notional <= 1x equity
BREAKER_DD = 0.10         # halt at 10% below peak account equity
WARMUP = {"15Min": 15, "1Hour": 30, "4HourRTH": 460}   # calendar days
TF_MINUTES = {"15Min": 15, "1Hour": 60, "4HourRTH": 240}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no broker, no state)
# ---------------------------------------------------------------------------

def correlation_blocked(symbol: str, side: str, positions: dict) -> bool:
    """No new BTC/USD long while SPY and QQQ suite positions are both
    long. `positions` is the state dict {symbol: {"side": ...}}."""
    if symbol == "BTC/USD" and side == "long":
        return all(positions.get(s, {}).get("side") == "long"
                   for s in ("SPY", "QQQ"))
    return False


def effective_stop(pos: dict) -> float:
    """Tighter of hard stop and trail; hard stop itself never moves."""
    hard, trail = pos["hard_stop"], pos.get("trail")
    if trail is None:
        return hard
    return max(hard, trail) if pos["side"] == "long" else min(hard, trail)


def breaker_tripped(equity: float, peak: float,
                    dd: float = BREAKER_DD) -> bool:
    return peak > 0 and equity < peak * (1 - dd)


# ---------------------------------------------------------------------------
# State + logs
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"positions": {}, "peak_equity": 0.0, "halted": False,
            "created": datetime.now().isoformat(timespec="seconds")}


def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, STATE_FILE)


def log_trade(row: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    fields = ["timestamp", "strategy", "symbol", "action", "side", "qty",
              "price", "stop", "atr", "reason", "pnl", "dry_run"]
    new = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def log_daily_pnl(equity: float, state: dict):
    """One row per calendar day: overwrite today's row on later runs."""
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    if os.path.exists(DAILY_PNL_CSV):
        with open(DAILY_PNL_CSV) as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != today]
    prev_eq = float(rows[-1]["equity"]) if rows else equity
    rows.append({"date": today, "equity": f"{equity:.2f}",
                 "pnl": f"{equity - prev_eq:.2f}",
                 "open_positions": len(state["positions"]),
                 "halted": state["halted"]})
    with open(DAILY_PNL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "equity", "pnl",
                                          "open_positions", "halted"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Market data for the live loop (fresh, uncached)
# ---------------------------------------------------------------------------

def recent_bars(symbol: str, spec: dict) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    start = str((now - timedelta(days=WARMUP[spec["timeframe"]])).date())
    # free tier refuses SIP queries ending within the last 15 minutes;
    # cap `end` 16 minutes back (signals use completed bars anyway)
    end = (now - timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
    if spec["asset"] == "crypto":
        df = fetch_crypto_bars(symbol, spec["timeframe"], start)
    elif spec["timeframe"] == "4HourRTH":
        df = resample_4h_session(fetch_stock_bars(symbol, "1Hour", start,
                                                  end=end))
    else:
        df = rth_only(fetch_stock_bars(symbol, spec["timeframe"], start,
                                       end=end))
    # drop the still-forming bar so decisions use completed bars only
    if len(df):
        cutoff = datetime.now(timezone.utc) \
            - timedelta(minutes=TF_MINUTES[spec["timeframe"]])
        df = df[df.index <= pd.Timestamp(cutoff)]
    return df


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_scan(args):
    """Signal check on the last completed bar of every instrument."""
    state = load_state()
    print(f"Suite scan {datetime.now():%Y-%m-%d %H:%M}  "
          f"(enabled: {ENABLED or 'NONE -- set after walk-forward'})")
    for sym, spec in INSTRUMENTS.items():
        strat = spec["strategy"]
        df = recent_bars(sym, spec)
        if df.empty:
            print(f"  {sym:<8} no data")
            continue
        d = strat.prepare(df, strat.default_params(sym))
        i = len(d) - 1
        held = state["positions"].get(sym)
        if held:
            sig = f"holding {held['side']} (stop {effective_stop(held):.2f})"
        else:
            side = strat.entry(d, i, strat.default_params(sym))
            sig = side or "-"
            if side and correlation_blocked(sym, side, state["positions"]):
                sig += "  [BLOCKED: correlation filter]"
        print(f"  {sym:<8} {strat.NAME:<9} last bar {d.index[i]}  "
              f"close {d['Close'].iloc[i]:,.2f}  atr {d['atr'].iloc[i]:,.3f}  "
              f"signal: {sig}")


def _close_suite_position(broker, state, sym, spot, reason, dry_run):
    pos = state["positions"][sym]
    spec = INSTRUMENTS[sym]
    if not dry_run:
        if pos.get("stop_order_id"):
            broker.cancel_order(pos["stop_order_id"])
        broker.place_market(sym, pos["qty"],
                            "sell" if pos["side"] == "long" else "buy",
                            spec["asset"])
    px = spot
    pnl = pos["qty"] * ((px - pos["entry"]) if pos["side"] == "long"
                        else (pos["entry"] - px))
    log_trade({"timestamp": datetime.now().isoformat(timespec="seconds"),
               "strategy": pos["strategy"], "symbol": sym, "action": "EXIT",
               "side": pos["side"], "qty": pos["qty"], "price": round(px, 4),
               "reason": reason, "pnl": round(pnl, 2), "dry_run": dry_run})
    print(f"  {'DRY-RUN ' if dry_run else ''}EXIT {pos['side']} {pos['qty']} "
          f"{sym} @ ~{px:,.2f} ({reason})  pnl ~${pnl:,.0f}")
    if not dry_run:
        RM.release(pos["strategy"], sym, realized_pnl=pnl)
        del state["positions"][sym]


def _reconcile(broker, state):
    """Detect suite stops that filled between runs: state position gone
    (or shrunk) at the broker -> book the exit."""
    held = broker.all_positions()
    for sym in list(state["positions"]):
        pos = state["positions"][sym]
        b = held.get(sym) or held.get(sym.replace("/", ""))
        have = abs(b["qty"]) if b else 0.0
        if have >= pos["qty"] * 0.999:
            continue
        fill = None
        if pos.get("stop_order_id"):
            try:
                o = broker.get_order(pos["stop_order_id"])
                if o.get("status") == "filled":
                    fill = float(o.get("filled_avg_price") or 0) or None
            except RuntimeError:
                pass
        px = fill or effective_stop(pos)
        pnl = pos["qty"] * ((px - pos["entry"]) if pos["side"] == "long"
                            else (pos["entry"] - px))
        log_trade({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "strategy": pos["strategy"], "symbol": sym,
                   "action": "EXIT", "side": pos["side"], "qty": pos["qty"],
                   "price": round(px, 4), "reason": "stop_filled",
                   "pnl": round(pnl, 2), "dry_run": False})
        print(f"  {sym}: protective stop filled @ ~{px:,.2f} "
              f"(pnl ~${pnl:,.0f})")
        RM.release(pos["strategy"], sym, realized_pnl=pnl)
        del state["positions"][sym]


def cmd_update(args):
    from alpaca_broker import AlpacaBroker
    state = load_state()
    broker = AlpacaBroker().connect()
    try:
        equity = broker.equity()
        state["peak_equity"] = max(state.get("peak_equity", 0.0), equity)
        print(f"Alpaca paper equity ${equity:,.0f}  "
              f"(peak ${state['peak_equity']:,.0f})  "
              f"enabled: {ENABLED or 'NONE'}")

        if state["halted"]:
            print("  HALTED by circuit breaker -- no trading. "
                  "`python -m bot.suite_bot reset-halt` to resume.")
            log_daily_pnl(equity, state)
            save_state(state)
            return

        if not args.dry_run:
            _reconcile(broker, state)

        # -- circuit breaker ---------------------------------------------------
        if breaker_tripped(equity, state["peak_equity"]):
            print(f"  CIRCUIT BREAKER: equity ${equity:,.0f} is >10% below "
                  f"peak ${state['peak_equity']:,.0f} -- closing all suite "
                  "positions and halting.")
            for sym in list(state["positions"]):
                spec = INSTRUMENTS[sym]
                df = recent_bars(sym, spec)
                spot = float(df["Close"].iloc[-1]) if len(df) \
                    else state["positions"][sym]["entry"]
                _close_suite_position(broker, state, sym, spot,
                                      "circuit_breaker", args.dry_run)
            state["halted"] = not args.dry_run
            log_daily_pnl(equity, state)
            save_state(state)
            return

        # -- per-instrument management + entries -------------------------------
        for sym in ENABLED:
            spec = INSTRUMENTS[sym]
            strat = spec["strategy"]
            params = strat.default_params(sym)
            df = strat.prepare(recent_bars(sym, spec), params)
            if len(df) < 5 or pd.isna(df["atr"].iloc[-1]):
                print(f"  {sym}: insufficient data; skipping")
                continue
            i = len(df) - 1
            close = float(df["Close"].iloc[i])
            atr_now = float(df["atr"].iloc[i])
            pos = state["positions"].get(sym)

            if pos is not None:
                if strat.exit(df, i, params, pos["side"]):
                    _close_suite_position(broker, state, sym, close,
                                          "signal_exit", args.dry_run)
                    continue
                if strat.TRAIL_ATR is not None:
                    since = df[df.index >= pd.Timestamp(pos["opened"])]
                    wm = float(since["High"].max()) if pos["side"] == "long" \
                        else float(since["Low"].min())
                    cand = wm - strat.TRAIL_ATR * atr_now \
                        if pos["side"] == "long" \
                        else wm + strat.TRAIL_ATR * atr_now
                    old = effective_stop(pos)
                    tighter = cand > old if pos["side"] == "long" \
                        else cand < old
                    if tighter:
                        pos["trail"] = cand
                        new_stop = effective_stop(pos)
                        print(f"  {sym}: trail -> {new_stop:,.2f}")
                        if not args.dry_run:
                            if pos.get("stop_order_id"):
                                broker.cancel_order(pos["stop_order_id"])
                            o = broker.place_stop(
                                sym.replace("/", ""), pos["qty"],
                                "sell" if pos["side"] == "long" else "buy",
                                new_stop, spec["asset"])
                            pos["stop_order_id"] = o.get("id")
                continue

            # -- new entry ------------------------------------------------------
            side = strat.entry(df, i, params)
            if side is None:
                continue
            if side == "short" and spec["asset"] == "crypto":
                continue
            if correlation_blocked(sym, side, state["positions"]):
                print(f"  {sym}: {side} signal BLOCKED by correlation filter "
                      "(SPY + QQQ both long)")
                continue
            qty = Engine.position_size(equity, atr_now, close, RISK_FRAC,
                                       NOTIONAL_CAP,
                                       fractional=spec["asset"] == "crypto")
            if spec["asset"] == "crypto":
                qty = round(qty, 4)
            if qty <= 0:
                continue
            hard = close - strat.STOP_ATR * atr_now if side == "long" \
                else close + strat.STOP_ATR * atr_now
            risk = qty * abs(close - hard)
            check = RM.can_reserve if args.dry_run else RM.reserve
            ok, why = check(strat.NAME, sym, risk, equity)
            if not ok:
                print(f"  RISK BLOCKED {sym}: {why}")
                continue
            print(f"  {'DRY-RUN ' if args.dry_run else ''}ENTER {side} "
                  f"{qty} {sym} @ ~{close:,.2f}  hard stop {hard:,.2f}  "
                  f"(risk ${risk:,.0f})")
            stop_id = None
            fill_px = close
            if not args.dry_run:
                o = broker.place_market(sym, qty,
                                        "buy" if side == "long" else "sell",
                                        spec["asset"])
                done = broker.get_order(o["id"])
                fill_px = float(done.get("filled_avg_price") or close)
                so = broker.place_stop(
                    sym.replace("/", ""), qty,
                    "sell" if side == "long" else "buy", hard, spec["asset"])
                stop_id = so.get("id")
                state["positions"][sym] = {
                    "strategy": strat.NAME, "side": side, "qty": qty,
                    "entry": fill_px, "atr_entry": atr_now,
                    "hard_stop": hard, "trail": None,
                    "stop_order_id": stop_id,
                    "opened": str(df.index[i])}
            log_trade({"timestamp": datetime.now()
                       .isoformat(timespec="seconds"),
                       "strategy": strat.NAME, "symbol": sym,
                       "action": "ENTRY", "side": side, "qty": qty,
                       "price": round(fill_px, 4), "stop": round(hard, 4),
                       "atr": round(atr_now, 4), "reason": "signal",
                       "dry_run": args.dry_run})

        log_daily_pnl(equity, state)
        save_state(state)
        print(f"State saved. Open: {list(state['positions']) or 'none'}")
    finally:
        broker.disconnect()


def cmd_status(args):
    state = load_state()
    print(f"Suite state (created {state.get('created', '?')})  "
          f"halted={state['halted']}  peak ${state.get('peak_equity', 0):,.0f}")
    for sym, p in state["positions"].items():
        print(f"  {sym}: {p['side']} {p['qty']} @ {p['entry']} "
              f"stop {effective_stop(p):,.2f} ({p['strategy']})")
    if not state["positions"]:
        print("  no open positions")
    print(RM.report())


def cmd_reset_halt(args):
    state = load_state()
    state["halted"] = False
    state["peak_equity"] = 0.0     # re-anchors to current equity next run
    save_state(state)
    print("Halt cleared and peak re-anchored. Review data/trades.csv before "
          "resuming.")


def main():
    p = argparse.ArgumentParser(description="Multi-instrument suite bot "
                                            "(Alpaca PAPER only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", help="signal check, no broker")
    pu = sub.add_parser("update", help="manage + trade via Alpaca paper")
    pu.add_argument("--dry-run", action="store_true")
    sub.add_parser("status", help="print state + risk report")
    sub.add_parser("reset-halt", help="deliberately clear the circuit breaker")
    args = p.parse_args()
    {"scan": cmd_scan, "update": cmd_update, "status": cmd_status,
     "reset-halt": cmd_reset_halt}[args.cmd](args)


if __name__ == "__main__":
    main()
