"""Shared risk layer for both strategies (momentum + wheel). Paper only.

Three jobs, per the design brief:

1. ONE combined account-level risk ceiling. Every strategy must reserve
   risk dollars here before opening a position, and the sum of all open
   reservations across BOTH strategies may not exceed
   MAX_TOTAL_OPEN_RISK_PCT of account equity. Two independent budgets
   that stack is exactly what this prevents.
2. Ticker exclusivity. A symbol with an open reservation in one
   strategy cannot be opened by the other (or doubled within the same
   one) until released.
3. Separate, labeled P&L attribution. Realized P&L is booked per
   strategy so performance can be attributed and debugged independently
   even though risk is shared.

Risk definitions (what "risk dollars" means per strategy):
  * momentum: shares x (entry - stop) -- the 1R stop distance.
  * wheel:    REVIEW_DRAWDOWN_PCT x collateral (strike x 100 x
              contracts) -- the loss at the manual-review trigger. Full
              collateral would be the theoretical max loss but would
              also make the two strategies incomparable; the review
              trigger is where the bot stops acting mechanically, so it
              is the honest working stop-equivalent.

State lives in data/risk_state.json guarded by a lock file so the two
bots' cron jobs cannot race each other.
"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RISK_STATE = os.path.join(DATA_DIR, "risk_state.json")
LOCK_FILE = RISK_STATE + ".lock"

MAX_TOTAL_OPEN_RISK_PCT = 0.05     # combined open risk across BOTH strategies
STRATEGIES = ("momentum", "wheel")
LOCK_TIMEOUT_S = 10.0


@contextmanager
def _locked():
    """Exclusive lock via O_CREAT|O_EXCL; stale locks (>60s) are broken."""
    os.makedirs(DATA_DIR, exist_ok=True)
    deadline = time.time() + LOCK_TIMEOUT_S
    while True:
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK_FILE) > 60:
                    os.remove(LOCK_FILE)   # stale: holder died
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                raise TimeoutError(f"risk state lock held too long: {LOCK_FILE}")
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            os.remove(LOCK_FILE)
        except FileNotFoundError:
            pass


class RiskManager:
    def __init__(self, state_file: str = RISK_STATE,
                 ceiling_pct: float = MAX_TOTAL_OPEN_RISK_PCT):
        self.state_file = state_file
        self.ceiling_pct = ceiling_pct

    # -- state ----------------------------------------------------------------
    def _load(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)
        return {"reservations": {},
                "pnl": {s: 0.0 for s in STRATEGIES},
                "created": datetime.now().isoformat()}

    def _save(self, state: dict):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_file)

    # -- queries ----------------------------------------------------------------
    def open_risk(self) -> float:
        state = self._load()
        return sum(r["risk"] for r in state["reservations"].values())

    def holder_of(self, ticker: str) -> str | None:
        """Which strategy (if any) currently holds this ticker."""
        r = self._load()["reservations"].get(ticker)
        return r["strategy"] if r else None

    def can_reserve(self, strategy: str, ticker: str, risk_dollars: float,
                    equity: float) -> tuple[bool, str]:
        """Read-only preview of reserve() for dry runs -- same checks, no
        state change, no lock (advisory only; the real reserve re-checks)."""
        if strategy not in STRATEGIES:
            return False, f"unknown strategy '{strategy}'"
        if risk_dollars <= 0 or equity <= 0:
            return False, "degenerate risk/equity"
        state = self._load()
        held = state["reservations"].get(ticker)
        if held:
            return False, f"ticker {ticker} already active in '{held['strategy']}'"
        open_risk = sum(r["risk"] for r in state["reservations"].values())
        if open_risk + risk_dollars > self.ceiling_pct * equity:
            return False, "combined risk ceiling would be exceeded"
        return True, "ok"

    # -- core API ----------------------------------------------------------------
    def reserve(self, strategy: str, ticker: str, risk_dollars: float,
                equity: float) -> tuple[bool, str]:
        """Ask for permission to open. Atomically checks ticker exclusivity
        and the combined ceiling; on success the reservation is recorded."""
        if strategy not in STRATEGIES:
            return False, f"unknown strategy '{strategy}'"
        if risk_dollars <= 0 or equity <= 0:
            return False, "degenerate risk/equity"
        with _locked():
            state = self._load()
            held = state["reservations"].get(ticker)
            if held:
                return False, (f"ticker {ticker} already active in "
                               f"'{held['strategy']}'")
            open_risk = sum(r["risk"] for r in state["reservations"].values())
            ceiling = self.ceiling_pct * equity
            if open_risk + risk_dollars > ceiling:
                return False, (f"combined risk ceiling: open ${open_risk:,.0f} "
                               f"+ new ${risk_dollars:,.0f} > "
                               f"${ceiling:,.0f} ({self.ceiling_pct:.0%} of equity)")
            state["reservations"][ticker] = {
                "strategy": strategy, "risk": float(risk_dollars),
                "opened": datetime.now().isoformat(timespec="seconds")}
            self._save(state)
            return True, "ok"

    def release(self, strategy: str, ticker: str,
                realized_pnl: float = 0.0) -> tuple[bool, str]:
        """Close a reservation and book realized P&L to the strategy.
        A strategy cannot release the other strategy's reservation; P&L on
        a missing reservation is still booked (exits must never be lost)."""
        if strategy not in STRATEGIES:
            return False, f"unknown strategy '{strategy}'"
        with _locked():
            state = self._load()
            held = state["reservations"].get(ticker)
            msg = "ok"
            if held is None:
                msg = f"no reservation for {ticker}; P&L booked anyway"
            elif held["strategy"] != strategy:
                return False, (f"{ticker} is held by '{held['strategy']}', "
                               f"not '{strategy}' -- refusing cross-release")
            else:
                del state["reservations"][ticker]
            state["pnl"][strategy] = state["pnl"].get(strategy, 0.0) \
                + float(realized_pnl)
            self._save(state)
            return True, msg

    # -- reporting ----------------------------------------------------------------
    def report(self) -> str:
        state = self._load()
        res = state["reservations"]
        lines = [f"Shared risk layer (ceiling {self.ceiling_pct:.0%} of equity)",
                 f"  open reservations: {len(res)}  "
                 f"(total risk ${sum(r['risk'] for r in res.values()):,.0f})"]
        for t, r in sorted(res.items()):
            lines.append(f"    {t:<7} {r['strategy']:<9} "
                         f"${r['risk']:>9,.0f}  since {r['opened']}")
        lines.append("  realized P&L by strategy:")
        for s in STRATEGIES:
            lines.append(f"    {s:<9} ${state['pnl'].get(s, 0.0):>12,.2f}")
        return "\n".join(lines)


if __name__ == "__main__":
    print(RiskManager().report())
