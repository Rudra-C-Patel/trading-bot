# Project status — as of 2026-07-18

One-paragraph version: **every directional strategy tested so far has failed
walk-forward validation and is parked.** The only strategy still standing is
the **wheel bot** — not because it passed anything, but because it has never
been forward-tested: it is *untested, not failed*. Its Alpaca paper path is
verified end-to-end read-only (including live-market strike selection), a
delta-band safety guard was added 2026-07-18 after live data exposed a gap,
and as of the last live check exactly **1 of 12 watchlist names is genuinely
tradeable (INTC)**. **No orders — paper or otherwise — have ever been placed
by any bot in this repo.** The user gate on the first order is still in
effect.

Run `python validate_config.py` before any session that touches this bot
(pre-flight: config consistency + Alpaca paper connection + options level).

---

## 1. Strategy scoreboard

| Strategy | Instruments / timeframe | Walk-forward OOS result | Status |
|---|---|---|---|
| Momentum RS (top-decile cutoff) | US equities, daily | **-3.9% CAGR** (corrected filters, 2021-07→2024-12); original mis-coded filters -0.8%; fixed top-10% benchmark +1.3% (noise) | **FAILED — parked** |
| Momentum RS + SPY>200SMA regime gate | same | **-0.9% CAGR**, 7/7 folds ≤ 0; gate only reduced trading, per-trade edge still negative (-0.61R avg) | **FAILED — closes the book on regime-timing** |
| Mean reversion (z-score) | SPY & QQQ, 15-min | SPY **-80.3%** / QQQ **-73.2%** total OOS; circuit breaker tripped 13/18 folds each | **FAILED — parked** |
| Momentum breakout | BTC/USD, 1-hour | **-50.8%** total OOS, 22% win rate, breaker tripped 8/8 folds; zero edge even at zero cost | **FAILED — parked** |
| Trend following (EMA cross) | GLD & USO, session 4-hour | GLD **-6.5%** / USO **-10.4%** total; GLD lost money over 9 years in which GLD itself ~tripled | **FAILED (thin sample — "no edge detectable") — parked** |
| **Wheel (CSP → covered call)** | 12-name watchlist, ~30 DTE options | **Never walk-forward tested** (backtest is a BS-approximation only, see README honesty section) | **UNTESTED — the only live candidate** |

Evidence for every failed row is in `WALKFORWARD_RESULTS.md` (protocols were
pre-registered; fold tables, trade logs and raw CSV paths are all there). The
multi-instrument suite infrastructure (`bot/`) is built and tested but
`ENABLED = []` in `bot/suite_bot.py` — it trades nothing.

Standing rule (user's): **only walk-forward survivors get paper-traded;
negatives are documented, never re-tuned into passes.** Any strategy revival
requires a new pre-registered hypothesis, justified before running.

## 2. The wheel bot (`wheel_bot.py` + `config.py` + `alpaca_broker.py`)

Mechanics: on a 21-EMA wick+close pullback signal (uptrend only, close >
50-SMA), sell a ~0.30-delta cash-secured put at ~30 DTE; after assignment,
sell ~0.20-delta covered calls. Risk additions beyond the source doc:
earnings avoidance, 10% per-ticker concentration cap, 5% max bid-ask spread,
60% realized-vol ceiling, credit-only rolls, -15% drawdown manual-review
flag, and the shared risk layer (`risk_manager.py`: 5% combined open-risk
ceiling + ticker exclusivity across all strategies).

Two broker backends behind one interface, selected by `WHEEL_BROKER` env
(default `ibkr`):

- **Alpaca paper** — the verified path. Account PA3W75UVGHFQ, ACTIVE, $100K,
  `options_trading_level=3` (wheel needs 1+). Raw REST, no SDK; free `iex`
  stock feed + `indicative` options feed (greeks included). Verified
  end-to-end read-only 2026-07-18 during market hours.
- **IBKR paper** — code intact but blocked: TWS/Gateway has never been
  installed on this machine. User expects it may unblock ~2026-08-10.

Paper-only guards (verified 2026-07-18 by code re-read — see §5): IBKR
refuses any port outside {7497, 4002} and any account not starting `DU`;
Alpaca refuses any host other than `paper-api.alpaca.markets`, any account
not starting `PA`, and options level < 1. Going live would require
deliberately editing both `config.py` and the guard code — intentional
friction.

## 3. Delta-band guard (added 2026-07-18)

**The gap:** live-market testing on 2026-07-18 showed `pick_by_delta`
returned the strike *closest* to 0.30 delta among spread-passing quotes —
with no cap on how far away that could be. When no true ~0.30-delta strike
quoted tightly enough, it would happily return a **0.41–0.46 delta** put
(PFE 25P at 0.458, WMT 113P at 0.408) — near-ATM assignment risk a
"~0.30-delta" strategy never intended.

**The fix:** `config.MAX_DELTA_DISTANCE = 0.08` + pure helper
`wheel_bot.delta_ok()` (epsilon-inclusive boundary, since
`abs(0.38-0.30) > 0.08` in IEEE-754), enforced as a post-loop guard in
**both** broker paths (`pick_contract_by_delta` for IBKR,
`AlpacaBroker.pick_by_delta` for Alpaca), logging `DELTA-BAND REJECT` and
returning no trade. 4 regression tests added; suite 35/35 passing at the
time of the fix.

## 4. Last verified watchlist state (live market hours, 2026-07-18, read-only)

All 12 names resolved expiry **2026-08-14** (~30 DTE). Result: **1 of 12
genuinely tradeable.**

| Outcome | Names | Detail |
|---|---|---|
| **PASS** | **INTC** | 85P, delta -0.306, spread 3.8% of mid — the only true ~0.30-delta, liquid strike |
| Delta-band reject (post-fix) | PFE, WMT | PFE 25P at -0.458 and WMT 113P at -0.408 passed the spread filter but sit outside 0.30 ± 0.08 — rejected with clear log lines |
| Spread / greeks fail | remaining 9 | Dominated by indicative-feed spreads >5% of mid at the nearest-0.30d strikes (e.g. CSCO 15.3%, MRK 19.4%); some strikes missing greeks entirely |

Note the structural implication: on the free `indicative` options feed, the
5% spread filter is the binding constraint for most of the watchlist. Whether
that reflects true illiquidity or feed quality is unresolved.

## 5. Safety posture

- **No orders ever placed** (paper or live) by wheel, momentum, or the
  suite. First-order user gate is still in effect.
- Paper-only guards re-verified by code read 2026-07-18 (this session):
  - `wheel_bot.py` `connect_paper()` — hard-refuses ports outside
    `PAPER_PORTS` {7497, 4002} and accounts not prefixed `DU`.
  - `alpaca_broker.py` — `__init__` refuses any `ALPACA_BASE_URL` whose
    parsed hostname ≠ `paper-api.alpaca.markets`; `connect()` refuses
    accounts not prefixed `PA`, non-ACTIVE/blocked accounts, and options
    level < 1. All order endpoints post to that verified base URL only.
  - `get_broker()` refuses unknown `BROKER` values.
- `validate_config.py` (added 2026-07-18) is the standing pre-flight check:
  config internal consistency + Alpaca paper connection + options level,
  PASS/FAIL summary, read-only.

## 6. Open threads

1. **Wheel forward paper test** — the next milestone, pending explicit user
   go-ahead for a first (paper) order. INTC is the only current candidate,
   and it must also print the 21-EMA entry signal on the day.
2. **IBKR path** — retest after ~2026-08-10 if TWS/Gateway gets installed.
3. **Momentum forward paper account** — reset 2026-07-10 under corrected
   rules; accrues genuinely unseen data. Realistic prior per walk-forward:
   ~flat.
4. **Uncommitted work** — the delta-band fix, `alpaca_broker.py`, `bot/`,
   and doc updates are uncommitted as of 2026-07-18.
