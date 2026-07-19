# Morgan Tradez — swing trading bot (paper only)

A mechanical implementation of a Qullamaggie-style momentum breakout
strategy: find the strongest stocks in the market, wait for an orderly
bull-flag pullback to a rising 10/20 SMA, buy the high-volume breakout,
risk 1% of the account, and don't sell before 5x risk.

**This bot never touches a brokerage. It scans, backtests, paper-trades,
and sends alerts. Nothing here is financial advice.**

---

## Strategy rules (what the code actually does)

### Stock selection (universe filter)
- Universe: **all US-listed common stocks** — NASDAQ + NYSE + NYSE
  American, from the official [NASDAQ Trader symbol directory](https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt)
  (free, refreshed nightly; ~5,300 names after dropping ETFs, warrants,
  rights, units, preferreds, notes and funds). Cached for 7 days in
  `data/universe_all.json`. `--universe sp500` restores the old narrow
  S&P 500 + NASDAQ 100 universe for comparison.
- Liquidity gate (universe-level, checked per-day in the backtest):
  price above **$5** and 50-day average **dollar volume above $3.5M**
  (corrected 2026-07-10 from the earlier 1M-shares floor — the source
  spec uses dollar volume, which is a different metric, not just a
  different number)
- Price above $1 (strategy rule)
- Average Daily Range (ADR%) above 5% — 20-day mean of `High/Low − 1`
  (corrected 2026-07-10 from 4%, per the source spec)
- Above the 50-day **and** 200-day SMA

### Entry
1. **Relative strength**: top 2% of the universe by weighted momentum
   across 1-month (40%), 3-month (35%) and 6-month (25%) returns,
   percentile-ranked cross-sectionally each day
2. **Prior move**: 30%+ gain from swing low to swing high, spanning at
   least 5 sessions, and no single day contributing more than 60% of the
   move (rejects one-day pumps)
3. **Rising 10 & 20 SMA** (both above their value 5 sessions ago)
4. **Orderly pullback**: 3–25 sessions off the high, giving back at most
   25%, with the pullback low touching (within 4% of) the 10 or 20 SMA
   and the close still holding the 20 SMA
5. **Tightening range**: last-3-day average range below 80% of ADR
6. **Volume dry-up**: 5-day average volume below the 50-day average
7. **Trigger**: buy stop 0.1% above the 5-day range high, taken only if
   the breakout bar prints **1.5x+ the 50-day average volume**

### Risk management
- Stop under the pullback low, clamped to **2–5% below entry**
- `Shares = Risk_dollars / (Entry − Stop)` with risk = **1% of equity**
- Position value capped at 25% of equity (rules allow 10–40%; the cap is
  configurable in `strategy.py`), floor of 10% — smaller positions are
  skipped
- Max 5 concurrent positions (configurable)

### Exit
- Hold until **5R minimum**; at 5R sell 25% into strength and move the
  stop to breakeven
- Trail the remainder: close below the **20 SMA** exits the trade
- Hard stop always active; gap-downs fill at the open

All thresholds live at the top of `strategy.py` as named constants.

---

## Files

| File | Purpose |
|------|---------|
| `strategy.py` | All entry/exit/sizing rules — shared by everything below |
| `scanner.py` | Daily scan: RS ranking → top 2% → setup check → watchlist |
| `backtest.py` | Event-driven daily backtest with report + trades CSV |
| `walkforward.py` | Walk-forward validation of the RS cutoff (see `WALKFORWARD_RESULTS.md`) |
| `paper_trader.py` | Stateful paper trading engine (JSON state, CSV log) |
| `telegram_alerts.py` | Morning scan → Telegram message |
| `config.py` | Wheel strategy thresholds + watchlist/blacklist + broker flag (paper-only enforced) |
| `wheel_bot.py` | Wheel bot: pure decision logic + broker layer (IBKR TWS **paper** default) |
| `alpaca_broker.py` | Alpaca **paper** adapter (same broker interface; select with `WHEEL_BROKER=alpaca`) |
| `wheel_backtest.py` | Wheel backtest (Black-Scholes approximation — see honesty section) |
| `risk_manager.py` | Shared risk layer: one combined ceiling, ticker exclusivity, per-strategy P&L |
| `bot/` | Multi-instrument suite (added 2026-07-13): `data.py` Alpaca history, `engine.py` sizing/stops/breaker, `strategies/` (mean reversion, breakout, trend), `walkforward.py`, `suite_bot.py` paper runner |
| `tests/` | pytest suite for wheel logic + risk layer + suite (`python -m pytest tests/`) |
| `requirements.txt` | Pinned dependencies (Python 3.11) |
| `data/` | Caches, state, logs (created on first run) |

Both trading engines consult `risk_manager.py` before opening anything:
combined open risk across the two strategies is capped at 5% of equity
(one shared budget, not two stacking ones), a ticker active in one
strategy is blocked in the other, and realized P&L is booked per
strategy in `data/risk_state.json` for independent attribution.

---

## Setup

```bash
cd trading-bot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Why the `ta` library and not `pandas-ta`? The original `pandas-ta` is
> unmaintained and crashes on NumPy 2.x (`np.NaN` removal). `ta` is the
> maintained pure-pandas alternative and covers everything this strategy
> needs (SMAs, ATR); the rest is plain pandas.

### Telegram (optional, for morning alerts)
1. Create a bot with [@BotFather](https://t.me/botfather), copy the token
2. Message your bot once, then get your chat id from
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. ```bash
   export TRADEZ_TELEGRAM_BOT_TOKEN="123456:ABC..."
   export TRADEZ_TELEGRAM_CHAT_ID="123456789"
   export ACCOUNT_SIZE="100000"   # used for position sizing in alerts
   ```

---

## Usage

```bash
# 1. Backtest first (first run downloads ~6y x ~5,300 tickers, 20-40 min,
#    checkpointed + cached; subsequent runs take ~2 min from cache)
python backtest.py                          # 2020-2024, $100K, 5 positions
python backtest.py --universe sp500         # old narrow universe, for comparison
python backtest.py --start 2022-01-01 --end 2023-12-31 --capital 25000

# 2. Daily scanner (run after the close; full-market scan downloads ~10 min)
python scanner.py                           # top 2% + setup check
python scanner.py --universe sp500          # faster, large caps only
python scanner.py --top-pct 0.05            # wider net for eyeballing

# 3. Paper trading loop
#    NOTE: the paper account runs the top-DECILE experiment by default
#    (scan --top-pct defaults to 0.10 here; see the experiment section).
python paper_trader.py scan                 # evening: build watchlist
python paper_trader.py update               # morning + close: fills/exits
python paper_trader.py status               # account snapshot

# 4. Morning Telegram alert
python telegram_alerts.py --dry-run         # preview
python telegram_alerts.py                   # send
```

Installed cron (this machine, US/Pacific — market is 6:30-13:00 PT;
logs to `data/cron.log`; the Mac must be awake at these times):

```cron
35 6  * * 1-5  cd trading-bot && .venv/bin/python paper_trader.py update
5  13 * * 1-5  cd trading-bot && .venv/bin/python paper_trader.py update
30 13 * * 1-5  cd trading-bot && .venv/bin/python paper_trader.py scan
```

Add a `telegram_alerts.py` line (e.g. `30 5 * * 1-5`) once the
`TRADEZ_TELEGRAM_*` env vars exist — cron jobs need them exported in the
crontab or a wrapper script, not just your shell profile.

---

## Backtest results & honest expectations

Two reference runs, 2020-01-01 → 2024-12-31, $100K, top 2% RS, max 5
positions, same strategy rules (executed 2026-07-08; regenerate with
`python backtest.py [--universe sp500]`):

```
                       FULL MARKET (4,317)     S&P 500 + NDX (508)
Total return                 -0.8%                   +40.7%
CAGR                         -0.2%                    +7.1%
Max drawdown                 -4.4%                    -6.8%
Sharpe (daily, rf=0)         -0.05                     0.88
Trades                        3                        6
Win rate                     33.3%                    66.7%
Average R                    -0.27R                   +6.11R
Bias-adjusted CAGR       -4.2% to -1.2%           +3.1% to +6.1%
```

**The wider universe made the strategy worse, and the diagnostic shows
why it isn't a bug.** With ~88 eligible names per day (median), the top
2% RS cut selects ~2.5 candidates/day — and in the full market those
are parabolic movers (NVAX, BBBY, RVP, CLSK...), not orderly trends.
157 flag setups formed over 5 years but only 3 ever confirmed a
1.5x-volume breakout within the 3-session window; the entries were GOGO
(-1R), HDSN (-1R) and NVAX 2024 (+1.2R after the 5R partial).

The narrow universe's +40.7% was a small-pool artifact, not alpha: on
their historical entry dates, the "top 2%" names it bought mostly were
NOT top 2% of the real market — MRNA Jul-2021 sat at the 69.6th RS
percentile, FSLR 78.7th, MSTR 88.3rd, RKLB 90.6th. Only CVNA (98.8th)
and MRNA May-2020 (97.9th) genuinely qualified. In a 508-name pool the
98th-percentile cut effectively selects the market's top ~20% RS —
where quality momentum lives — while in the full market a literal top-2%
cut selects lottery tickets that don't form tradeable flags.

**Bottom line: implemented literally, this ruleset does not make money
on 2020-2024 US equities.** The famous discretionary results add trade
selection, episodic-pivot entries and concentration that these rules
don't encode.

### Top-decile experiment (one deliberate deviation from the spec)

Changing ONE parameter — the RS cut from top 2% to top 10%
(`python backtest.py --top-pct 0.10`) — so the full-market scan selects
"strong momentum" instead of "parabolic outliers":

```
Total return   +64.5%      CAGR      10.5%
Max drawdown   -12.3%      Sharpe    0.90
Trades         26          Win rate  46.2%
Average R      +2.06R      Avg win/loss  +5.67R / -1.02R
Bias-adjusted CAGR: 6.5% to 9.5%
Annual: 2020 +16.7% | 2021 -1.2% | 2022 -1.8% | 2023 +19.6% | 2024 +21.5%
```

The trade list is exactly this strategy's canon — SHOP, MRNA, CELH
(+10.7R), MGNI, GRWG, CVNA (+8.7R), AFRM (+10.3R), SMR, RKLB (+19.6R) —
with every loser capped near -1R by the stop discipline. Returns did
NOT depend on the 2020-21 bull specifically: the chop years (2021/2022)
were merely flat (-1 to -2%), while every trending year (2020/2023/2024)
made +16-22%. Saved reports: `data/backtest_report_top2pct.txt` and
`data/backtest_report_top10pct.txt`.

Honesty check before extrapolating: (1) 26 trades still isn't a
statistically robust sample; (2) the 10% figure was chosen AFTER seeing
the 2%-cut fail — that's in-sample selection, so treat it as a
hypothesis for forward paper trading, not a validated edge; (3)
survivorship bias hits hardest exactly in the small-cap pool these
winners came from — dead momentum names from 2020-22 are invisible,
so the true expectation sits at or below the adjusted band.

**Update (2026-07-10): the walk-forward test in `walkforward.py` was run
and the top-decile edge did NOT survive.** Selecting the RS cutoff on
rolling 18-month train windows and trading the following 6 months
out-of-sample produced -0.8% CAGR over 2021-07 → 2024-12 (5 trades) —
the top-decile cutoff never won a training window once 2020 rolled out
of view. Full fold-by-fold breakdown and verdict:
[`WALKFORWARD_RESULTS.md`](WALKFORWARD_RESULTS.md). The forward
paper-trading account remains the only live test; expect roughly flat,
not 10.5%.

**Second update (same day): all backtest numbers above were produced
under mis-coded filters** (ADR > 4 instead of the spec's > 5; 1M-share
volume floor instead of $3.5M average daily dollar volume). With the
source-accurate filters the walk-forward result worsens to -3.9% CAGR
and the fixed top-decile benchmark collapses to +1.3% CAGR — the 10.5%
was partly a filter-bug artifact. See the corrected-filters section of
`WALKFORWARD_RESULTS.md`. The paper account was reset the same day, so
its forward record reflects corrected rules only.

See `data/backtest_report.txt` after a run. Read the numbers with these
caveats — they matter more than the headline return:

1. **Survivorship bias.** The universe is *today's* index membership.
   Stocks that were delisted or dropped 2020–2024 are invisible, which
   inflates returns.
2. **The top-2% rule is universe-sensitive.** A percentile cut means
   opposite things in a 508-name pool (≈ market top ~20% RS: quality
   momentum) and a 4,300-name pool (true top 2%: parabolic lottery
   tickets). Tested both — see the comparison table above. Neither
   interpretation produced statistically meaningful trade counts.
3. **Fill assumptions.** Entries at `max(open, trigger)` + 0.1% slippage;
   fast breakouts fill worse in real life. No commissions modeled.
4. **This style is regime-dependent.** Breakouts feast in 2020-21-style
   momentum markets and starve in chop (2022). A realistic expectation
   for a mechanical version is a **25–45% win rate** carried by a small
   number of large-R winners — not a smooth equity curve. Published
   mechanical replications of this strategy show roughly this profile;
   the multi-hundred-percent years the strategy is famous for came from
   discretionary trade selection, a full-market universe, and
   concentration that a rules-based bot deliberately doesn't replicate.

If the backtest shows a modest CAGR with a double-digit drawdown and a
sub-50% win rate, that *is* the realistic result. Distrust anything that
looks too good.

---

## Wheel strategy bot (second strategy, added 2026-07-10)

`wheel_bot.py` + `config.py` + `wheel_backtest.py` implement a wheel
(cash-secured puts → assignment → covered calls) against **paper
trading only**, on either of two brokers behind one interface
(`connect`, `equity`, `stock_positions`, `spot`, `expirations`,
`pick_by_delta`, `quote`, `place_limit`, `order_status`):

- **IBKR TWS paper** (default) — port 7497; live ports and non-`DU*`
  accounts are hard-refused in `connect_paper()`.
- **Alpaca paper** (`WHEEL_BROKER=alpaca`, added 2026-07-12) —
  `alpaca_broker.py`, raw REST with the already-pinned `requests` (no
  SDK). Credentials in a git-ignored `.env` (`ALPACA_API_KEY` /
  `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL`). Hard-refuses any host other
  than `paper-api.alpaca.markets`, account numbers not starting with
  `PA`, and options trading level below 1 (covered calls / CSPs).
  Market data uses the free tiers: `iex` stock feed, `indicative`
  options feed (quotes + greeks for the delta targeting). Note the 5%
  spread filter rejects most quotes while the market is closed —
  indicative weekend/overnight spreads are wide; scan results are only
  meaningful during market hours. `python alpaca_broker.py` prints a
  connect-and-verify account report (never places orders).

The backend is chosen by `config.BROKER`, overridable per-run with the
`WHEEL_BROKER` env var (`ibkr` | `alpaca`); unset means IBKR. Decision
logic is broker-independent — the same rules run against either.

Mechanics from the setup doc: 21 EMA wick+close entry signal (bar wicks
below a rising daily 21 EMA and closes above it, in an uptrend), sell a
~0.30-delta cash-secured put at ~30 DTE, and after assignment sell
~0.20-delta covered calls. **Risk management the original doc did not
include** (all thresholds in `config.py`): earnings avoidance (no
option whose expiry crosses a known earnings date), a 10% per-ticker
assigned-exposure cap with capital spread across 3+ names before any
name pyramids, a 5% bid-ask spread ceiling (live only — see below), a
60% realized-vol ceiling plus `OPTIONS_BLACKLIST` for premium-trap
names, credit-only roll logic (puts roll down-and-out near expiry;
profitable calls get called away, unprofitable ones roll up-and-out for
credit or hold), and a manual-review flag when assigned stock falls 15%
below cost basis (the bot stops selling calls below basis on flagged
names).

```bash
python wheel_bot.py scan                             # signal check (no broker needed)
python wheel_bot.py update --dry-run                 # decide everything, place nothing
python wheel_bot.py update                           # trade via TWS paper (must be running)
WHEEL_BROKER=alpaca python wheel_bot.py update       # same, via Alpaca paper (.env keys)
python alpaca_broker.py                              # Alpaca account/options-level report
python wheel_backtest.py                             # BS-approximation backtest
```

### Wheel backtest honesty (read before quoting any number)

Historical options quotes are not freely available the way stock prices
are, so `wheel_backtest.py` **approximates every premium with
Black-Scholes on trailing 30-day realized volatility** (flat 4% rate,
10% premium haircut for spread crossing — same disclosure-first
approach as the momentum backtest). Realized vol systematically
underprices options in calm markets and overprices them right after
crashes — exactly when the wheel gets assigned — so the backtest CAGR
is an estimate of mechanics, not a return anyone could have earned.
The live liquidity filter (bid-ask spread) cannot be simulated at all
(no historical quote data), dividends aren't separately credited
(adjusted prices), and early assignment isn't modeled. Reference run
(2020-2024, $100K, 12-name watchlist): +4.4% CAGR, -16.6% max DD,
Sharpe 0.50, 100 CSPs, 29 assignments, 8 manual-review flags — i.e.
roughly T-bill-grade returns with equity-grade drawdowns under these
assumptions. Full report: `data/wheel_backtest_report.txt`.

## Multi-instrument suite (third strategy set, added 2026-07-13)

`bot/` implements three intraday/swing strategies over five instruments
against the same Alpaca paper account and `alpaca_broker.py` adapter
(same paper-only hard guards):

- **Mean reversion** — SPY/QQQ, 15-min bars, 20-period SMA/stddev
  z-score entry (1.5/1.8), exit at the mean, 2 ATR hard stop.
- **Momentum breakout** — BTC/USD, 1-hour bars, 20-period high breakout
  with 1.5x volume confirmation, 2x ATR trailing stop, long-only.
- **Trend following** — GLD/USO, session-anchored 4-hour bars, 50/200
  EMA cross, 3x ATR trailing stop, long-only.

Shared risk rules: 1 ATR move = 1% of equity sizing (notional capped at
1x equity — the cap binds on 15-min index ETFs); broker-side GTC hard
stop placed at entry, **never moved**; trailing stops only tighten; the
`risk_manager.py` combined 5% ceiling + ticker exclusivity applies
across ALL strategies including the wheel; no new BTC long while SPY
and QQQ are both long (correlation filter); a 10% drawdown from peak
equity closes every suite position and halts until a deliberate
`reset-halt`. Trades log to `data/trades.csv`, daily P&L to
`data/daily_pnl.csv`.

```bash
python -m bot.suite_bot scan               # signals only, no broker
python -m bot.suite_bot update --dry-run   # decide everything, place nothing
python -m bot.suite_bot update             # trade enabled instruments (Alpaca paper)
python -m bot.walkforward                  # per-instrument walk-forward validation
```

**Validation result (2026-07-13): all five instrument/strategy pairs
failed walk-forward — zero out-of-sample edge, confirmed robust to a
zero-cost re-run. Nothing is enabled for paper trading (`ENABLED = []`
in `bot/suite_bot.py`); the suite is parked infrastructure until a new
pre-registered hypothesis earns its way in.** Full protocol, per-fold
tables and post-mortem: the suite section of
[`WALKFORWARD_RESULTS.md`](WALKFORWARD_RESULTS.md).

## Paper → live checklist (deliberately not implemented)

This repo stops at paper trading by design. Before even thinking about
live capital: 6+ months of forward paper results matching the backtest's
R-distribution, a broker API sandbox, order-type handling (buy-stop
entries, OCO stops), partial-fill logic, and a kill switch. None of that
is here, on purpose.
