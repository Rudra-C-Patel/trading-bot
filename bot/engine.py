"""Event-driven engine for the multi-instrument suite -- one instrument,
one strategy, one parameter set per run. Shared by the walk-forward
validator and by nothing else that trades: the live runner reuses the
same sizing/stop math via the helpers at the bottom, but never this loop.

Execution model (identical assumptions across all three strategies):
  * Decisions are taken on bar CLOSES and filled at the NEXT bar's open.
  * Stops are checked intra-bar: a bar opening through the stop fills at
    the open (gap), otherwise at the stop price.
  * Hard stop = entry -/+ STOP_ATR x ATR(signal bar). NEVER moved.
  * Trailing stop (where the strategy defines one) ratchets behind the
    best price since entry minus TRAIL_ATR x current ATR; it only
    tightens. Effective stop = the tighter of hard and trail.
  * Costs: `cost` fraction per side, applied to every fill.
  * Circuit breaker: if mark-to-market equity drops `breaker_dd` (10%)
    below its peak, the position is closed at the next open and the run
    halts flat for the remainder of the window.

Sizing: 1 ATR move = risk_frac (1%) of equity -> qty = 0.01 x equity /
ATR. Notional is capped at notional_cap (1.0x) of equity; on low-ATR
intraday instruments (SPY 15-min ATR is ~0.1% of price) the cap, not
the ATR rule, is what binds -- disclosed in the README.
"""

import numpy as np
import pandas as pd


class Engine:
    def __init__(self, strategy, df: pd.DataFrame, params: dict,
                 capital: float = 100_000.0, cost: float = 0.0002,
                 fractional: bool = False, risk_frac: float = 0.01,
                 notional_cap: float = 1.0, breaker_dd: float = 0.10,
                 label: str = ""):
        self.s = strategy
        self.df = df
        self.params = params
        self.capital = capital
        self.cost = cost
        self.fractional = fractional
        self.risk_frac = risk_frac
        self.notional_cap = notional_cap
        self.breaker_dd = breaker_dd
        self.label = label
        self.trades: list[dict] = []

    # -- sizing (shared with the live runner) --------------------------------
    @staticmethod
    def position_size(equity: float, atr_value: float, price: float,
                      risk_frac: float, notional_cap: float,
                      fractional: bool) -> float:
        """1 ATR move = risk_frac of equity, capped at notional_cap x
        equity of notional. Returns 0 when inputs are degenerate."""
        if equity <= 0 or atr_value <= 0 or price <= 0 \
                or not np.isfinite(atr_value) or not np.isfinite(price):
            return 0.0
        qty = (risk_frac * equity) / atr_value
        qty = min(qty, (notional_cap * equity) / price)
        if not fractional:
            qty = float(int(qty))
        return max(qty, 0.0)

    # -- main loop -------------------------------------------------------------
    def run(self) -> dict:
        df, s, p = self.df, self.s, self.params
        o = df["Open"].to_numpy(float)
        h = df["High"].to_numpy(float)
        l = df["Low"].to_numpy(float)
        c = df["Close"].to_numpy(float)
        a = df["atr"].to_numpy(float)
        n = len(df)

        cash = self.capital
        eq = self.capital
        peak = self.capital
        eq_curve = np.full(n, np.nan)
        pos = None                    # dict when a position is open
        pending = None                # ("enter", side, atr_sig) | ("exit", reason)
        halted = False

        def close_position(i, px, reason):
            nonlocal cash, pos
            fill = px * (1 - self.cost) if pos["side"] == "long" \
                else px * (1 + self.cost)
            if pos["side"] == "long":
                cash += pos["qty"] * fill
                pnl = pos["qty"] * (fill - pos["entry"])
            else:
                cash -= pos["qty"] * fill
                pnl = pos["qty"] * (pos["entry"] - fill)
            risk = pos["qty"] * abs(pos["entry"] - pos["hard_stop"])
            self.trades.append({
                "symbol": self.label, "side": pos["side"],
                "entry_time": pos["entry_time"], "exit_time": df.index[i],
                "entry": round(pos["entry"], 4), "exit": round(fill, 4),
                "qty": pos["qty"], "pnl": round(pnl, 2),
                "r_multiple": round(pnl / risk, 3) if risk > 0 else np.nan,
                "reason": reason, "bars_held": i - pos["entry_idx"],
            })
            pos = None

        for i in range(n):
            # 1. fills queued at the previous close
            if pending is not None:
                kind = pending[0]
                if kind == "exit" and pos is not None:
                    close_position(i, o[i], pending[1])
                elif kind == "enter" and pos is None and not halted:
                    side, atr_sig = pending[1], pending[2]
                    fill = o[i] * (1 + self.cost) if side == "long" \
                        else o[i] * (1 - self.cost)
                    qty = self.position_size(eq, atr_sig, fill, self.risk_frac,
                                             self.notional_cap, self.fractional)
                    if qty > 0:
                        stop_mult = s.STOP_ATR
                        hard = fill - stop_mult * atr_sig if side == "long" \
                            else fill + stop_mult * atr_sig
                        if side == "long":
                            cash -= qty * fill
                        else:
                            cash += qty * fill
                        pos = {"side": side, "qty": qty, "entry": fill,
                               "hard_stop": hard, "trail": None,
                               "watermark": fill, "entry_time": df.index[i],
                               "entry_idx": i}
                pending = None

            # 2. intra-bar stop check
            if pos is not None:
                if pos["side"] == "long":
                    stop = pos["hard_stop"] if pos["trail"] is None \
                        else max(pos["hard_stop"], pos["trail"])
                    if o[i] <= stop:
                        close_position(i, o[i], "stop_gap")
                    elif l[i] <= stop:
                        close_position(i, stop, "stop")
                else:
                    stop = pos["hard_stop"] if pos["trail"] is None \
                        else min(pos["hard_stop"], pos["trail"])
                    if o[i] >= stop:
                        close_position(i, o[i], "stop_gap")
                    elif h[i] >= stop:
                        close_position(i, stop, "stop")

            # 3. trailing-stop ratchet (never loosens)
            if pos is not None and s.TRAIL_ATR is not None \
                    and np.isfinite(a[i]):
                if pos["side"] == "long":
                    pos["watermark"] = max(pos["watermark"], h[i])
                    cand = pos["watermark"] - s.TRAIL_ATR * a[i]
                    pos["trail"] = cand if pos["trail"] is None \
                        else max(pos["trail"], cand)
                else:
                    pos["watermark"] = min(pos["watermark"], l[i])
                    cand = pos["watermark"] + s.TRAIL_ATR * a[i]
                    pos["trail"] = cand if pos["trail"] is None \
                        else min(pos["trail"], cand)

            # 4. mark to market + circuit breaker
            if pos is None:
                eq = cash
            elif pos["side"] == "long":
                eq = cash + pos["qty"] * c[i]
            else:
                eq = cash - pos["qty"] * c[i]
            eq_curve[i] = eq
            peak = max(peak, eq)
            if not halted and eq < peak * (1 - self.breaker_dd):
                halted = True
                if pos is not None:
                    pending = ("exit", "circuit_breaker")
                continue    # no new signals once halted

            # 5. signals at the close (fill next open)
            if halted:
                continue
            if pos is not None:
                if s.exit(df, i, p, pos["side"]):
                    pending = ("exit", "signal")
            else:
                side = s.entry(df, i, p)
                if side is not None and np.isfinite(a[i]) and a[i] > 0:
                    pending = ("enter", side, a[i])

        if pos is not None:
            close_position(n - 1, c[n - 1], "end_of_window")
            eq_curve[n - 1] = cash

        equity = pd.Series(eq_curve, index=df.index).dropna()
        return {"equity": equity, "trades": self.trades, "halted": halted}


def daily_equity(equity: pd.Series) -> pd.Series:
    """Bar-level equity -> daily closes (what perf_metrics expects)."""
    if equity.empty:
        return equity
    return equity.resample("1D").last().dropna()
