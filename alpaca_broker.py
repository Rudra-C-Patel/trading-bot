"""Alpaca paper-trading adapter for wheel_bot -- PAPER ONLY.

Implements the same broker interface as wheel_bot.IBKRBroker (connect,
equity, stock_positions, spot, expirations, pick_by_delta, quote,
place_limit, order_status, disconnect) over plain REST -- no Alpaca SDK,
just `requests`, which is already pinned.

Endpoints: Trading API v2 (account / positions / orders / option
contracts) on ALPACA_BASE_URL, and Market Data on data.alpaca.markets
using the free tiers: `iex` feed for stock snapshots, `indicative` feed
for option chain snapshots (quotes + greeks) -- the greeks drive the
same delta-targeting the IBKR path gets from modelGreeks.

Credentials come from the environment or the git-ignored .env beside
this file (ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL).

THIS ADAPTER NEVER CONNECTS TO A LIVE ACCOUNT: any host other than
paper-api.alpaca.markets, any account number not starting with "PA",
or an options_trading_level below 1 aborts.

Standalone verification (no orders placed):
    python alpaca_broker.py        # account + options level + positions
"""

import os
import sys
from datetime import datetime, timedelta
from urllib.parse import urlparse

import requests

import config

DATA_URL = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
PAPER_HOST = "paper-api.alpaca.markets"
PAPER_ACCOUNT_PREFIX = "PA"     # Alpaca paper account numbers start with PA
STOCK_FEED = "iex"              # free stock data feed
OPTIONS_FEED = "indicative"     # free options feed; includes quotes + greeks
TIMEOUT_S = 20

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _load_env():
    """Fill os.environ from .env without overriding real env vars."""
    if not os.path.exists(ENV_FILE):
        return
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def occ_symbol(ticker: str, expiry: str, strike: float, right: str) -> str:
    """OCC option symbol, e.g. KO260814P00062500 (strike x1000, 8 digits)."""
    d = datetime.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
    return f"{ticker}{d}{right}{int(round(strike * 1000)):08d}"


class AlpacaBroker:
    name = "alpaca"

    def __init__(self):
        _load_env()
        key = os.environ.get("ALPACA_API_KEY")
        sec = os.environ.get("ALPACA_SECRET_KEY")
        base = (os.environ.get("ALPACA_BASE_URL") or "").rstrip("/")
        if not key or not sec or not base:
            sys.exit("Alpaca credentials missing: set ALPACA_API_KEY / "
                     "ALPACA_SECRET_KEY / ALPACA_BASE_URL (env or .env)")
        host = urlparse(base).hostname
        if host != PAPER_HOST:
            sys.exit(f"REFUSED: ALPACA_BASE_URL host '{host}' is not "
                     f"{PAPER_HOST}. This bot never trades live.")
        self.base = base
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key,
                                     "APCA-API-SECRET-KEY": sec})
        self.account = None

    # -- plumbing ---------------------------------------------------------
    def _get(self, url: str, **params):
        r = self.session.get(url, params=params or None, timeout=TIMEOUT_S)
        if r.status_code != 200:
            raise RuntimeError(f"Alpaca GET {url} -> {r.status_code}: "
                               f"{r.text[:200]}")
        return r.json()

    def _paged(self, url: str, item_key: str, **params):
        """Yield items across next_page_token pages. Yields list elements
        for list payloads (option contracts) and (key, value) pairs for
        dict payloads (chain snapshots)."""
        while True:
            data = self._get(url, **params)
            items = data.get(item_key) or {}
            yield from (items.items() if isinstance(items, dict) else items)
            token = data.get("next_page_token")
            if not token:
                return
            params["page_token"] = token

    # -- broker interface ---------------------------------------------------
    def connect(self):
        """Fetch the account and hard-verify paper + options entitlement."""
        try:
            acct = self._get(self.base + "/v2/account")
        except RuntimeError as e:
            sys.exit(f"Could not connect to Alpaca paper: {e}")
        num = acct.get("account_number", "")
        if not num.startswith(PAPER_ACCOUNT_PREFIX):
            sys.exit(f"REFUSED: account '{num}' does not look like an Alpaca "
                     f"paper account ({PAPER_ACCOUNT_PREFIX}*). Aborting.")
        if acct.get("status") != "ACTIVE" or acct.get("trading_blocked") \
                or acct.get("account_blocked"):
            sys.exit(f"REFUSED: account {num} not tradable "
                     f"(status {acct.get('status')}, trading_blocked="
                     f"{acct.get('trading_blocked')})")
        level = int(acct.get("options_trading_level") or 0)
        if level < 1:
            sys.exit(f"REFUSED: options_trading_level={level} -- level 1+ "
                     "(covered calls / cash-secured puts) required")
        self.account = acct
        return self

    def disconnect(self):
        self.session.close()

    def equity(self) -> float:
        self.account = self._get(self.base + "/v2/account")
        try:
            return float(self.account["equity"])
        except (KeyError, TypeError, ValueError):
            return config.ACCOUNT_SIZE_FALLBACK

    def stock_positions(self) -> dict[str, float]:
        return {p["symbol"]: float(p["qty"])
                for p in self._get(self.base + "/v2/positions")
                if p.get("asset_class") == "us_equity"}

    def spot(self, ticker: str) -> float | None:
        s = self._get(f"{DATA_URL}/v2/stocks/{ticker}/snapshot",
                      feed=STOCK_FEED)
        px = (s.get("latestTrade") or {}).get("p")
        if not px:
            q = s.get("latestQuote") or {}
            bid, ask = q.get("bp"), q.get("ap")
            px = (bid + ask) / 2 if bid and ask \
                else (s.get("dailyBar") or {}).get("c")
        return float(px) if px and px > 0 else None

    def expirations(self, ticker: str) -> list[str]:
        today = datetime.now().date()
        horizon = today + timedelta(days=config.TARGET_DTE
                                    + config.DTE_TOLERANCE)
        return sorted({c["expiration_date"] for c in self._paged(
            self.base + "/v2/options/contracts", "option_contracts",
            underlying_symbols=ticker, type="put",
            expiration_date_gte=str(today),
            expiration_date_lte=str(horizon), limit=1000)})

    def pick_by_delta(self, ticker, expiry, spot, right, target_delta):
        """Chain snapshot filtered to the same strike window the IBKR path
        scans; return the spread-passing quote with delta closest to
        target -- but only if that delta lands inside the
        MAX_DELTA_DISTANCE band around target -- else None."""
        from wheel_bot import OptionQuote, delta_ok, spread_ok
        lo, hi = (0.70 * spot, 1.02 * spot) if right == "P" \
            else (0.98 * spot, 1.35 * spot)
        best, best_gap = None, None
        for occ, snap in self._paged(
                f"{DATA_URL}/v1beta1/options/snapshots/{ticker}", "snapshots",
                feed=OPTIONS_FEED, type="put" if right == "P" else "call",
                expiration_date=expiry, strike_price_gte=round(lo, 2),
                strike_price_lte=round(hi, 2), limit=1000):
            delta = (snap.get("greeks") or {}).get("delta")
            if delta is None:
                continue
            q = snap.get("latestQuote") or {}
            bid, ask = q.get("bp"), q.get("ap")
            if not spread_ok(bid, ask):
                continue
            gap = abs(abs(delta) - target_delta)
            if best_gap is None or gap < best_gap:
                best = OptionQuote(ticker, expiry, int(occ[-8:]) / 1000.0,
                                   right, float(bid), float(ask),
                                   float(delta), handle=occ)
                best_gap = gap
        # Delta-band guard (live 2026-07-18 finding): the closest
        # spread-passing strike can still be 0.41-0.46 delta. Refuse it.
        if best is not None and not delta_ok(best.delta, target_delta):
            print(f"  {ticker}: DELTA-BAND REJECT {expiry} "
                  f"{best.strike}{right} -- delta {abs(best.delta):.3f} is "
                  f"the closest spread-passing strike but lies outside "
                  f"target {target_delta:.2f} +/- "
                  f"{config.MAX_DELTA_DISTANCE:.2f}; no trade")
            return None
        return best

    def quote(self, ticker, expiry, strike, right):
        from wheel_bot import OptionQuote
        occ = occ_symbol(ticker, expiry, strike, right)
        data = self._get(f"{DATA_URL}/v1beta1/options/quotes/latest",
                         feed=OPTIONS_FEED, symbols=occ)
        q = (data.get("quotes") or {}).get(occ)
        if not q or q.get("bp") is None or q.get("ap") is None:
            return None
        return OptionQuote(ticker, expiry, float(strike), right,
                           float(q["bp"]), float(q["ap"]), handle=occ)

    def place_limit(self, q, side: str, contracts_n: int,
                    limit_price: float) -> str:
        occ = q.handle or occ_symbol(q.ticker, q.expiry, q.strike, q.right)
        r = self.session.post(self.base + "/v2/orders", timeout=TIMEOUT_S,
                              json={"symbol": occ, "qty": str(int(contracts_n)),
                                    "side": side.lower(), "type": "limit",
                                    "time_in_force": "day",
                                    "limit_price": str(limit_price)})
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Alpaca order rejected ({r.status_code}): "
                               f"{r.text[:300]}")
        order = r.json()
        print(f"    alpaca order {order.get('id')} "
              f"status={order.get('status')}")
        return order.get("id", "")

    def order_status(self, order_id: str) -> str:
        return self._get(f"{self.base}/v2/orders/{order_id}") \
            .get("status", "unknown")

    # -- multi-instrument suite extensions (stocks + crypto) -----------------
    # Same session, same paper-only guarantees: every call below goes to
    # self.base, which connect() has already verified is the paper host.

    def _post(self, path: str, payload: dict) -> dict:
        r = self.session.post(self.base + path, json=payload,
                              timeout=TIMEOUT_S)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Alpaca POST {path} -> {r.status_code}: "
                               f"{r.text[:300]}")
        return r.json()

    def all_positions(self) -> dict[str, dict]:
        """symbol -> {qty (signed float), asset_class, avg_entry_price}.
        Crypto symbols come back slashless (BTCUSD); callers should match
        on both forms."""
        return {p["symbol"]: {"qty": float(p["qty"]),
                              "asset_class": p.get("asset_class"),
                              "avg_entry_price": float(p["avg_entry_price"])}
                for p in self._get(self.base + "/v2/positions")}

    def get_order(self, order_id: str) -> dict:
        return self._get(f"{self.base}/v2/orders/{order_id}")

    def cancel_order(self, order_id: str):
        r = self.session.delete(f"{self.base}/v2/orders/{order_id}",
                                timeout=TIMEOUT_S)
        if r.status_code not in (200, 204, 404):
            raise RuntimeError(f"Alpaca cancel {order_id} -> {r.status_code}: "
                               f"{r.text[:200]}")

    def place_market(self, symbol: str, qty: float, side: str,
                     asset: str = "stock") -> dict:
        """Market order; crypto uses gtc (day is rejected for crypto)."""
        return self._post("/v2/orders", {
            "symbol": symbol, "qty": str(qty), "side": side.lower(),
            "type": "market",
            "time_in_force": "gtc" if asset == "crypto" else "day"})

    def place_stop(self, symbol: str, qty: float, side: str,
                   stop_price: float, asset: str = "stock") -> dict:
        """Protective stop, GTC. Crypto only supports stop_limit, so the
        limit is set 0.5% through the stop to behave like a stop-market."""
        payload = {"symbol": symbol, "qty": str(qty), "side": side.lower(),
                   "time_in_force": "gtc",
                   "stop_price": str(round(stop_price, 2))}
        if asset == "crypto":
            slip = 0.995 if side.lower() == "sell" else 1.005
            payload.update(type="stop_limit",
                           limit_price=str(round(stop_price * slip, 2)))
        else:
            payload["type"] = "stop"
        return self._post("/v2/orders", payload)

    def close_position(self, symbol: str) -> bool:
        """Market-close an open position (used by the circuit breaker)."""
        r = self.session.delete(
            f"{self.base}/v2/positions/{symbol.replace('/', '')}",
            timeout=TIMEOUT_S)
        return r.status_code in (200, 204, 207)


def main():
    """Connect-and-verify report (Phase 4 check). Places no orders."""
    b = AlpacaBroker().connect()
    a = b.account
    print(f"Alpaca PAPER account {a['account_number']} -- {a['status']}")
    print(f"  equity ${float(a['equity']):,.2f}  cash ${float(a['cash']):,.2f}  "
          f"options buying power ${float(a['options_buying_power']):,.2f}")
    print(f"  options level: approved {a.get('options_approved_level')} / "
          f"trading {a.get('options_trading_level')}  "
          f"(wheel needs level 1+)")
    print(f"  blocked flags: trading={a['trading_blocked']} "
          f"account={a['account_blocked']}  created {a['created_at']}")
    positions = b._get(b.base + "/v2/positions")
    print(f"  positions: {len(positions)}")
    for p in positions:
        print(f"    {p['symbol']}: {p['qty']} ({p['asset_class']})")
    orders = b._get(b.base + "/v2/orders", status="open")
    print(f"  open orders: {len(orders)}")
    for o in orders:
        print(f"    {o['id'][:8]} {o['side']} {o['qty']}x {o['symbol']} "
              f"({o['status']})")
    b.disconnect()


if __name__ == "__main__":
    main()
