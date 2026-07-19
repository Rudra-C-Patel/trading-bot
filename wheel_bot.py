"""Wheel strategy paper-trading bot -- IBKR TWS PAPER ONLY.

Sells ~0.30-delta cash-secured puts (~30 DTE) on watchlist names that
print the 21 EMA wick+close signal; after assignment, sells ~0.20-delta
covered calls. Risk management (earnings avoidance, concentration caps,
liquidity + volatility filters, credit-only rolls, drawdown review
flags) lives in config.py and decide_* below -- these are ADDITIONS the
original setup doc did not include.

Layout: everything above the "Broker layer" marker is pure logic shared
with wheel_backtest.py and the test suite; ib_async is imported lazily
so those imports never require TWS or the package.

Broker backends (config.BROKER / WHEEL_BROKER env, default "ibkr"):
IBKRBroker below wraps the original ib_async layer; AlpacaBroker in
alpaca_broker.py speaks the same interface (connect, equity,
stock_positions, spot, expirations, pick_by_delta, quote, place_limit,
order_status, disconnect) against Alpaca paper.

Usage (TWS paper session running, or WHEEL_BROKER=alpaca + .env keys):
    python wheel_bot.py scan               # signal check on the watchlist (no broker needed)
    python wheel_bot.py update             # manage positions + open new ones
    python wheel_bot.py update --dry-run   # decide everything, place no orders
    python wheel_bot.py status             # account + state snapshot

THIS BOT NEVER CONNECTS TO A LIVE ACCOUNT: on IBKR any port outside
config.PAPER_PORTS or any account id not starting with "DU" aborts; on
Alpaca any host other than paper-api.alpaca.markets or any account not
starting with "PA" aborts.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import config
from risk_manager import RiskManager

RM = RiskManager()  # shared risk layer: combined ceiling + ticker exclusivity

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATE_FILE = os.path.join(DATA_DIR, "wheel_state.json")
TRADES_CSV = os.path.join(DATA_DIR, "wheel_trades.csv")


# ===========================================================================
# Pure logic -- no IB, no network. Shared with wheel_backtest.py and tests.
# ===========================================================================

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def realized_vol(close: pd.Series, window: int = config.REALIZED_VOL_WINDOW) -> float:
    """Annualized close-to-close volatility over the trailing window."""
    rets = close.pct_change().dropna().iloc[-window:]
    if len(rets) < window // 2:
        return float("nan")
    return float(rets.std() * np.sqrt(252))


def entry_signal(df: pd.DataFrame) -> bool:
    """21 EMA wick+close: bar dips below a rising 21 EMA and closes above
    it, in an uptrend (close > 50 SMA). `df` needs Open/High/Low/Close."""
    if len(df) < config.TREND_SMA + 5:
        return False
    close = df["Close"]
    e = ema(close, config.EMA_PERIOD)
    sma = close.rolling(config.TREND_SMA).mean()
    i = len(df) - 1
    if pd.isna(sma.iloc[i]) or pd.isna(e.iloc[i - 3]):
        return False
    return bool(
        df["Low"].iloc[i] <= e.iloc[i]          # wick tested the EMA...
        and close.iloc[i] >= e.iloc[i]          # ...and the close held it
        and e.iloc[i] > e.iloc[i - 3]           # EMA rising
        and close.iloc[i] > sma.iloc[i]         # uptrend context
    )


def spread_ok(bid: float, ask: float) -> bool:
    """Liquidity filter: bid-ask spread <= MAX_SPREAD_PCT of the mid."""
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return False
    mid = (bid + ask) / 2
    return (ask - bid) <= config.MAX_SPREAD_PCT * mid


@dataclass
class OptionQuote:
    """Broker-agnostic option quote: everything the orchestration needs to
    compare, price and book a contract. `handle` is the broker's own token
    (ib_async Contract / OCC symbol) passed back verbatim to place_limit."""
    ticker: str
    expiry: str          # YYYY-MM-DD
    strike: float
    right: str           # "P" | "C"
    bid: float
    ask: float
    delta: float | None = None
    handle: object = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


def delta_ok(delta: float | None, target_delta: float) -> bool:
    """Delta-band guard: |delta| must land within MAX_DELTA_DISTANCE of
    the target. A 0.42-delta put is not an acceptable stand-in for a
    0.30 target even when it is the closest strike passing the spread
    filter -- better no trade than near-ATM assignment risk."""
    if delta is None or not np.isfinite(delta):
        return False
    # epsilon keeps the band boundary inclusive under float error
    # (abs(0.38 - 0.30) > 0.08 in IEEE-754)
    return abs(abs(delta) - target_delta) <= config.MAX_DELTA_DISTANCE + 1e-9


def vol_ok(annualized_vol: float) -> bool:
    """Premium-trap guard: reject names above the realized-vol ceiling."""
    return bool(np.isfinite(annualized_vol)
                and annualized_vol <= config.MAX_REALIZED_VOL)


def expiry_crosses_earnings(expiry, earnings_dates, asof=None) -> bool:
    """True if any known earnings date falls in (asof, expiry]. Unknown
    (empty) earnings data returns False -- callers must disclose that."""
    if not earnings_dates:
        return False
    asof = pd.Timestamp(asof or datetime.now().date())
    expiry = pd.Timestamp(expiry)
    return any(asof < pd.Timestamp(d) <= expiry for d in earnings_dates)


def max_new_contracts(equity: float, strike: float, active_tickers: int,
                      existing_ticker_contracts: int = 0) -> int:
    """Concentration rules: per-ticker assigned-stock exposure capped at
    MAX_TICKER_EXPOSURE_PCT of equity, and no pyramiding a name past one
    contract until MIN_CONCURRENT_TICKERS names are active."""
    if equity <= 0 or strike <= 0:
        return 0
    cap = int((equity * config.MAX_TICKER_EXPOSURE_PCT)
              // (strike * config.CONTRACT_MULTIPLIER))
    cap -= existing_ticker_contracts
    if cap <= 0:
        return 0
    if active_tickers < config.MIN_CONCURRENT_TICKERS:
        cap = min(cap, 1 - existing_ticker_contracts)
    return max(cap, 0)


def decide_put_action(spot: float, strike: float, dte: int) -> str:
    """'hold' | 'manage'. 'manage' = near expiry and ITM-or-close: the
    caller attempts a down-and-out roll for >= ROLL_MIN_CREDIT, otherwise
    accepts assignment."""
    if dte > config.ROLL_DTE:
        return "hold"
    if spot <= strike * (1 + config.PUT_ROLL_TRIGGER_PCT):
        return "manage"
    return "hold"


def decide_call_action(spot: float, strike: float, dte: int, basis: float) -> str:
    """'hold' | 'allow_assignment' | 'try_roll'. A profitable call about
    to be assigned is let go; an unprofitable one is rolled up-and-out
    for credit where possible."""
    if dte > config.ROLL_DTE or spot < strike:
        return "hold"
    return "allow_assignment" if strike >= basis else "try_roll"


def needs_review(spot: float, basis: float) -> bool:
    """Reassessment trigger: assigned stock REVIEW_DRAWDOWN_PCT below cost
    basis -> manual review; stop mechanically selling calls below basis."""
    return spot <= basis * (1 - config.REVIEW_DRAWDOWN_PCT)


def eligible_for_new_position(ticker: str, df: pd.DataFrame,
                              active: dict) -> tuple[bool, str]:
    """Watchlist gatekeeper for a new CSP: blacklist, not already active,
    vol ceiling, entry signal. Returns (ok, reason_if_not)."""
    if ticker in config.OPTIONS_BLACKLIST:
        return False, "blacklisted"
    if ticker in active:
        return False, "already active"
    rv = realized_vol(df["Close"])
    if not vol_ok(rv):
        return False, f"realized vol {rv:.0%} above ceiling"
    if not entry_signal(df):
        return False, "no 21 EMA wick+close signal"
    return True, ""


# ===========================================================================
# State + logging
# ===========================================================================

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"short_puts": {}, "stock": {}, "short_calls": {},
            "review_flags": {}, "realized_pnl": 0.0,
            "created": datetime.now().isoformat()}


def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, STATE_FILE)


def log_trade(row: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    fields = ["timestamp", "ticker", "action", "right", "strike", "expiry",
              "contracts", "price", "reason", "cash_impact"]
    new = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def active_tickers(state: dict) -> dict:
    """Every ticker with any wheel exposure (short put, stock, or call)."""
    out = {}
    for bucket in ("short_puts", "stock", "short_calls"):
        for t in state.get(bucket, {}):
            out[t] = bucket
    return out


# ===========================================================================
# Market data helpers (yfinance for bars/earnings; IB only for chains/orders)
# ===========================================================================

def fetch_daily_bars(tickers: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    data = yf.download(tickers, period=period, interval="1d", group_by="ticker",
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


def next_earnings_dates(ticker: str) -> list:
    """Upcoming earnings dates via yfinance; [] when unknown (and the
    caller logs that earnings avoidance ran blind for this name)."""
    import yfinance as yf
    try:
        cal = yf.Ticker(ticker).calendar or {}
        dates = cal.get("Earnings Date") or []
        return [pd.Timestamp(d) for d in dates]
    except Exception as e:
        print(f"  [WARN] {ticker}: earnings lookup failed ({e}); "
              "avoidance running blind", file=sys.stderr)
        return []


# ===========================================================================
# Broker layer -- PAPER ONLY. Two backends behind one interface:
# IBKRBroker (ib_async, imported lazily) and alpaca_broker.AlpacaBroker.
# ===========================================================================

def get_broker():
    """Instantiate the backend named by config.BROKER (default 'ibkr')."""
    if config.BROKER == "alpaca":
        from alpaca_broker import AlpacaBroker
        return AlpacaBroker()
    if config.BROKER != "ibkr":
        sys.exit(f"REFUSED: unknown BROKER '{config.BROKER}' "
                 "(expected 'ibkr' or 'alpaca')")
    return IBKRBroker()


def connect_paper():
    """Connect to TWS/Gateway and hard-verify it is a paper session."""
    if config.IB_PORT not in config.PAPER_PORTS:
        sys.exit(f"REFUSED: port {config.IB_PORT} is not a known paper port "
                 f"{sorted(config.PAPER_PORTS)}. This bot never trades live.")
    from ib_async import IB
    ib = IB()
    try:
        ib.connect(config.IB_HOST, config.IB_PORT,
                   clientId=config.IB_CLIENT_ID, timeout=10)
    except Exception as e:
        sys.exit(f"Could not connect to TWS paper on "
                 f"{config.IB_HOST}:{config.IB_PORT} ({e}). Is TWS running "
                 "with API enabled?")
    accounts = ib.managedAccounts()
    if not accounts or not all(a.startswith(config.PAPER_ACCOUNT_PREFIX)
                               for a in accounts):
        ib.disconnect()
        sys.exit(f"REFUSED: account(s) {accounts} do not look like IB paper "
                 f"accounts ({config.PAPER_ACCOUNT_PREFIX}*). Aborting.")
    ib.reqMarketDataType(4)  # delayed-frozen: works without data subscriptions
    return ib


def ib_equity(ib) -> float:
    for row in ib.accountSummary():
        if row.tag == "NetLiquidation":
            return float(row.value)
    return config.ACCOUNT_SIZE_FALLBACK


def pick_expiry(expirations, asof=None) -> str | None:
    """Listed expiry closest to TARGET_DTE within tolerance, else None."""
    asof = pd.Timestamp(asof or datetime.now().date())
    best, best_gap = None, None
    for exp in expirations:
        dte = (pd.Timestamp(exp) - asof).days
        gap = abs(dte - config.TARGET_DTE)
        if gap <= config.DTE_TOLERANCE and (best_gap is None or gap < best_gap):
            best, best_gap = exp, gap
    return best


def pick_contract_by_delta(ib, ticker: str, expiry: str, spot: float,
                           right: str, target_delta: float):
    """Qualify strikes around spot and return (contract, ticker_data) whose
    model delta is closest to target. Skips quotes failing the spread test."""
    from ib_async import Option
    lo, hi = (0.70 * spot, 1.02 * spot) if right == "P" else (0.98 * spot, 1.35 * spot)
    [stk] = ib.qualifyContracts(_stock(ticker))
    chains = ib.reqSecDefOptParams(stk.symbol, "", stk.secType, stk.conId)
    chain = next((c for c in chains if c.exchange == "SMART"), None)
    if chain is None:
        return None, None
    strikes = [k for k in chain.strikes if lo <= k <= hi]
    contracts = ib.qualifyContracts(*[
        Option(ticker, expiry.replace("-", ""), k, right, "SMART",
               tradingClass=chain.tradingClass) for k in strikes])
    if not contracts:
        return None, None
    tickers = ib.reqTickers(*contracts)
    best, best_td, best_gap = None, None, None
    for c, td in zip(contracts, tickers):
        g = td.modelGreeks
        if g is None or g.delta is None:
            continue
        if not spread_ok(td.bid, td.ask):
            continue
        gap = abs(abs(g.delta) - target_delta)
        if best_gap is None or gap < best_gap:
            best, best_td, best_gap = c, td, gap
    # Delta-band guard: the closest spread-passing strike can still sit far
    # from target when nothing near it quotes tight enough. Refuse it.
    if best is not None and not delta_ok(best_td.modelGreeks.delta,
                                         target_delta):
        print(f"  {ticker}: DELTA-BAND REJECT {expiry} {best.strike}{right} "
              f"-- delta {abs(best_td.modelGreeks.delta):.3f} is the closest "
              f"spread-passing strike but lies outside target "
              f"{target_delta:.2f} +/- {config.MAX_DELTA_DISTANCE:.2f}; "
              "no trade")
        return None, None
    return best, best_td


def _stock(ticker: str):
    from ib_async import Stock
    return Stock(ticker, "SMART", "USD")


class IBKRBroker:
    """Thin adapter over the original ib_async layer above -- the functions
    themselves are unchanged; this just gives them the shared broker shape."""
    name = "ibkr"

    def __init__(self):
        self.ib = None

    def connect(self):
        self.ib = connect_paper()
        return self

    def disconnect(self):
        if self.ib is not None:
            self.ib.disconnect()

    def equity(self) -> float:
        return ib_equity(self.ib)

    def stock_positions(self) -> dict[str, float]:
        return {p.contract.symbol: p.position for p in self.ib.positions()
                if p.contract.secType == "STK"}

    def spot(self, ticker: str) -> float | None:
        [stk] = self.ib.qualifyContracts(_stock(ticker))
        px = self.ib.reqTickers(stk)[0].marketPrice()
        return float(px) if px and px > 0 and np.isfinite(px) else None

    def expirations(self, ticker: str) -> list[str]:
        [stk] = self.ib.qualifyContracts(_stock(ticker))
        chains = self.ib.reqSecDefOptParams(stk.symbol, "", stk.secType,
                                            stk.conId)
        chain = next((c for c in chains if c.exchange == "SMART"), None)
        if chain is None:
            return []
        return sorted(str(pd.Timestamp(e).date()) for e in chain.expirations)

    def pick_by_delta(self, ticker, expiry, spot, right,
                      target_delta) -> OptionQuote | None:
        c, td = pick_contract_by_delta(self.ib, ticker, expiry, spot, right,
                                       target_delta)
        if c is None:
            return None
        g = td.modelGreeks
        return OptionQuote(ticker,
                           str(pd.Timestamp(c.lastTradeDateOrContractMonth).date()),
                           c.strike, right, td.bid, td.ask,
                           g.delta if g else None, handle=c)

    def quote(self, ticker, expiry, strike, right) -> OptionQuote | None:
        from ib_async import Option
        cs = self.ib.qualifyContracts(Option(
            ticker, expiry.replace("-", ""), strike, right, "SMART"))
        if not cs:
            return None
        td = self.ib.reqTickers(cs[0])[0]
        return OptionQuote(ticker, expiry, strike, right, td.bid, td.ask,
                           handle=cs[0])

    def place_limit(self, q: OptionQuote, side: str, contracts_n: int,
                    limit_price: float) -> str:
        from ib_async import LimitOrder
        trade = self.ib.placeOrder(q.handle,
                                   LimitOrder(side, contracts_n, limit_price))
        return str(trade.order.orderId)

    def order_status(self, order_id: str) -> str:
        for t in self.ib.trades():
            if str(t.order.orderId) == order_id:
                return t.orderStatus.status
        return "unknown"


def sell_option(broker, q: OptionQuote, contracts_n: int, reason: str,
                dry_run: bool) -> float:
    """Place a SELL limit at the mid. Returns credit per share (mid)."""
    mid = round(q.mid, 2)
    if not dry_run:
        broker.place_limit(q, "SELL", contracts_n, mid)
    log_trade({"timestamp": datetime.now().isoformat(timespec="seconds"),
               "ticker": q.ticker, "action": "SELL_TO_OPEN" if "open" in reason else "SELL",
               "right": q.right, "strike": q.strike,
               "expiry": q.expiry,
               "contracts": contracts_n, "price": mid, "reason": reason,
               "cash_impact": round(mid * contracts_n * config.CONTRACT_MULTIPLIER, 2)})
    print(f"  {'DRY-RUN ' if dry_run else ''}SELL {contracts_n}x {q.ticker} "
          f"{q.expiry} {q.strike}{q.right} @ ~{mid} ({reason})")
    return mid


def buy_to_close(broker, q: OptionQuote, contracts_n: int, reason: str,
                 dry_run: bool) -> float:
    mid = round(q.mid, 2)
    if not dry_run:
        broker.place_limit(q, "BUY", contracts_n, mid)
    log_trade({"timestamp": datetime.now().isoformat(timespec="seconds"),
               "ticker": q.ticker, "action": "BUY_TO_CLOSE",
               "right": q.right, "strike": q.strike,
               "expiry": q.expiry,
               "contracts": contracts_n, "price": mid, "reason": reason,
               "cash_impact": round(-mid * contracts_n * config.CONTRACT_MULTIPLIER, 2)})
    print(f"  {'DRY-RUN ' if dry_run else ''}BUY-TO-CLOSE {contracts_n}x "
          f"{q.ticker} {q.strike}{q.right} @ ~{mid} ({reason})")
    return mid


# ===========================================================================
# Commands
# ===========================================================================

def cmd_scan(args):
    """Signal check across the watchlist. Pure yfinance -- no TWS needed."""
    state = load_state()
    act = active_tickers(state)
    bars = fetch_daily_bars(config.WATCHLIST)
    print(f"Wheel scan {datetime.now():%Y-%m-%d %H:%M} -- "
          f"{len(config.WATCHLIST)} names, {len(act)} already active")
    candidates = []
    for t in config.WATCHLIST:
        if t not in bars:
            print(f"  {t:<6} no data")
            continue
        ok, why = eligible_for_new_position(t, bars[t], act)
        if ok:
            rv = realized_vol(bars[t]["Close"])
            print(f"  {t:<6} SIGNAL  (30d realized vol {rv:.0%})")
            candidates.append(t)
        else:
            print(f"  {t:<6} -       {why}")
    state["candidates"] = {"date": datetime.now().strftime("%Y-%m-%d"),
                           "tickers": candidates}
    save_state(state)
    print(f"{len(candidates)} candidate(s) saved to state.")


def cmd_update(args):
    """Manage open wheel positions and open new CSPs via the paper broker."""
    state = load_state()
    broker = get_broker()
    broker.connect()
    try:
        equity = broker.equity()
        print(f"Paper account equity: ${equity:,.0f}  (broker: {broker.name})")
        today = pd.Timestamp(datetime.now().date())

        _reconcile_assignments(broker, state)

        # -- manage short puts ------------------------------------------------
        for t in list(state["short_puts"]):
            p = state["short_puts"][t]
            dte = (pd.Timestamp(p["expiry"]) - today).days
            bars = fetch_daily_bars([t], period="1mo")
            if t not in bars:
                continue
            spot = float(bars[t]["Close"].iloc[-1])
            if dte < 0:      # expired OTM (assignment shows up in reconcile)
                print(f"  {t}: short put expired (spot {spot:.2f} vs strike "
                      f"{p['strike']}) -- premium kept")
                pnl = p["credit"] * p["contracts"] * config.CONTRACT_MULTIPLIER
                state["realized_pnl"] += pnl
                del state["short_puts"][t]
                RM.release("wheel", t, realized_pnl=pnl)
                continue
            if decide_put_action(spot, p["strike"], dte) == "manage":
                _roll_or_assign_put(broker, state, t, p, spot, args.dry_run)

        # -- manage covered calls / assigned stock ----------------------------
        for t in list(state["stock"]):
            s = state["stock"][t]
            bars = fetch_daily_bars([t], period="1mo")
            if t not in bars:
                continue
            spot = float(bars[t]["Close"].iloc[-1])
            if needs_review(spot, s["basis"]):
                if t not in state["review_flags"]:
                    state["review_flags"][t] = {
                        "flagged": datetime.now().strftime("%Y-%m-%d"),
                        "spot": spot, "basis": s["basis"]}
                    print(f"  [REVIEW] {t}: spot {spot:.2f} is "
                          f"{spot / s['basis'] - 1:.0%} vs basis {s['basis']:.2f} "
                          "-- manual review required; no calls will be sold")
                continue
            call = state["short_calls"].get(t)
            if call:
                dte = (pd.Timestamp(call["expiry"]) - today).days
                action = decide_call_action(spot, call["strike"], dte, s["basis"])
                if action == "allow_assignment":
                    print(f"  {t}: CC {call['strike']} likely assigned above "
                          f"basis {s['basis']:.2f} -- letting it be called away")
                elif action == "try_roll":
                    _roll_call_up_out(broker, state, t, call, s, spot,
                                      args.dry_run)
            else:
                _open_covered_call(broker, state, t, s, spot, args.dry_run)

        # -- open new CSPs from scan candidates --------------------------------
        cand = state.get("candidates", {}).get("tickers", [])
        for t in cand:
            if t in active_tickers(state):
                continue
            _open_csp(broker, state, t, equity, args.dry_run)

        save_state(state)
        print(f"State saved. Active: {list(active_tickers(state)) or 'none'}  "
              f"review flags: {list(state['review_flags']) or 'none'}")
    finally:
        broker.disconnect()


def _reconcile_assignments(broker, state):
    """Detect stock that appeared via assignment (or vanished via call-away)
    by comparing broker positions to state."""
    ib_stock = broker.stock_positions()
    for t in list(state["short_puts"]):
        p = state["short_puts"][t]
        if ib_stock.get(t, 0) >= p["contracts"] * config.CONTRACT_MULTIPLIER \
                and t not in state["stock"]:
            basis = p["strike"] - p["credit"]
            state["stock"][t] = {"shares": p["contracts"] * config.CONTRACT_MULTIPLIER,
                                 "basis": basis,
                                 "assigned": datetime.now().strftime("%Y-%m-%d")}
            print(f"  {t}: ASSIGNED {state['stock'][t]['shares']} sh, "
                  f"basis {basis:.2f} (strike - premium)")
            del state["short_puts"][t]
    for t in list(state["stock"]):
        if ib_stock.get(t, 0) <= 0 and t in state["short_calls"]:
            c = state["short_calls"][t]
            s = state["stock"][t]
            pnl = (c["strike"] - s["basis"] + c["credit"]) * s["shares"]
            state["realized_pnl"] += pnl
            print(f"  {t}: CALLED AWAY at {c['strike']} -- wheel cycle P&L "
                  f"${pnl:,.0f}")
            del state["stock"][t]
            del state["short_calls"][t]
            state["review_flags"].pop(t, None)
            RM.release("wheel", t, realized_pnl=pnl)


def _open_csp(broker, state, ticker, equity, dry_run):
    earnings = next_earnings_dates(ticker) if config.AVOID_EARNINGS else []
    spot = broker.spot(ticker)
    if not spot or spot <= 0 or not np.isfinite(spot):
        print(f"  {ticker}: no usable spot price; skipping")
        return
    expirations = broker.expirations(ticker)
    if not expirations:
        print(f"  {ticker}: no option chain; skipping")
        return
    expiry = pick_expiry(expirations)
    if expiry is None:
        print(f"  {ticker}: no expiry near {config.TARGET_DTE} DTE; skipping")
        return
    if config.AVOID_EARNINGS and expiry_crosses_earnings(expiry, earnings):
        print(f"  {ticker}: expiry {expiry} crosses earnings; skipping")
        return
    q = broker.pick_by_delta(ticker, expiry, spot, "P",
                             config.CSP_TARGET_DELTA)
    if q is None:
        print(f"  {ticker}: no liquid strike near {config.CSP_TARGET_DELTA} "
              "delta (spread/greeks/delta-band filters); skipping")
        return
    n = max_new_contracts(equity, q.strike,
                          len(active_tickers(state)))
    if n <= 0:
        print(f"  {ticker}: concentration cap leaves no room "
              f"(strike {q.strike}); skipping")
        return
    # shared risk layer: risk = loss at the manual-review trigger
    risk = config.REVIEW_DRAWDOWN_PCT * q.strike \
        * config.CONTRACT_MULTIPLIER * n
    check = RM.can_reserve if dry_run else RM.reserve
    ok, why = check("wheel", ticker, risk, equity)
    if not ok:
        print(f"  RISK BLOCKED {ticker}: {why}")
        return
    credit = sell_option(broker, q, n, "open_csp", dry_run)
    if not dry_run:
        state["short_puts"][ticker] = {
            "strike": q.strike, "expiry": q.expiry,
            "contracts": n, "credit": credit,
            "opened": datetime.now().strftime("%Y-%m-%d")}


def _roll_or_assign_put(broker, state, ticker, p, spot, dry_run):
    """Near-expiry ITM-ish short put: roll down+out for net credit, else
    accept assignment."""
    old_q = broker.quote(ticker, p["expiry"], p["strike"], "P")
    new_expiry = pick_expiry(broker.expirations(ticker))
    new_q = (broker.pick_by_delta(ticker, new_expiry, spot, "P",
                                  config.CSP_TARGET_DELTA)
             if new_expiry else None)
    if old_q and new_q is not None:
        close_cost = old_q.mid
        new_credit = new_q.mid
        if new_credit - close_cost >= config.ROLL_MIN_CREDIT:
            buy_to_close(broker, old_q, p["contracts"], "roll_put", dry_run)
            credit = sell_option(broker, new_q, p["contracts"],
                                 "roll_put_open", dry_run)
            if not dry_run:
                p.update(strike=new_q.strike, expiry=new_q.expiry,
                         credit=p["credit"] + credit - close_cost)
            return
    print(f"  {ticker}: no roll available for >= {config.ROLL_MIN_CREDIT} "
          "credit -- accepting assignment")


def _open_covered_call(broker, state, ticker, s, spot, dry_run):
    earnings = next_earnings_dates(ticker) if config.AVOID_EARNINGS else []
    expiry = pick_expiry(broker.expirations(ticker))
    if expiry is None:
        return
    if config.AVOID_EARNINGS and expiry_crosses_earnings(expiry, earnings):
        print(f"  {ticker}: CC expiry {expiry} crosses earnings; waiting")
        return
    q = broker.pick_by_delta(ticker, expiry, spot, "C",
                             config.CC_TARGET_DELTA)
    if q is None:
        print(f"  {ticker}: no liquid CC strike; waiting")
        return
    n = s["shares"] // config.CONTRACT_MULTIPLIER
    credit = sell_option(broker, q, n, "open_cc", dry_run)
    if not dry_run:
        state["short_calls"][ticker] = {
            "strike": q.strike, "expiry": q.expiry,
            "contracts": n, "credit": credit}


def _roll_call_up_out(broker, state, ticker, call, s, spot, dry_run):
    old_q = broker.quote(ticker, call["expiry"], call["strike"], "C")
    new_expiry = pick_expiry(broker.expirations(ticker))
    new_q = (broker.pick_by_delta(ticker, new_expiry, spot, "C",
                                  config.CC_TARGET_DELTA)
             if new_expiry else None)
    if old_q and new_q is not None and new_q.strike > call["strike"]:
        close_cost = old_q.mid
        new_credit = new_q.mid
        if new_credit - close_cost >= config.ROLL_MIN_CREDIT:
            buy_to_close(broker, old_q, call["contracts"], "roll_cc", dry_run)
            credit = sell_option(broker, new_q, call["contracts"],
                                 "roll_cc_open", dry_run)
            if not dry_run:
                call.update(strike=new_q.strike, expiry=new_q.expiry,
                            credit=call["credit"] + credit - close_cost)
            return
    print(f"  {ticker}: CC below basis and no credit roll available -- "
          "holding; assignment would realize a loss (review manually)")


def cmd_status(args):
    state = load_state()
    print(f"Wheel state (created {state.get('created', '?')})")
    print(f"  realized P&L booked: ${state.get('realized_pnl', 0):,.2f}")
    for name, bucket in (("Short puts", "short_puts"), ("Stock", "stock"),
                         ("Short calls", "short_calls")):
        items = state.get(bucket, {})
        print(f"  {name}: {len(items)}")
        for t, v in items.items():
            print(f"    {t}: {v}")
    flags = state.get("review_flags", {})
    if flags:
        print(f"  REVIEW FLAGS: {flags}")
    cand = state.get("candidates", {})
    if cand:
        print(f"  Candidates ({cand.get('date')}): {cand.get('tickers')}")


def main():
    p = argparse.ArgumentParser(
        description="Wheel strategy bot (paper ONLY; broker = config.BROKER "
                    "/ WHEEL_BROKER env: ibkr | alpaca)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", help="watchlist signal check (no broker needed)")
    pu = sub.add_parser("update", help="manage positions + open trades via paper broker")
    pu.add_argument("--dry-run", action="store_true",
                    help="decide everything, place no orders")
    sub.add_parser("status", help="print wheel state")
    args = p.parse_args()
    {"scan": cmd_scan, "update": cmd_update, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
