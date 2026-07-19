# Walk-forward validation of the RS cutoff — results

> **Three runs live in this file.** The original run below used the filters as
> first coded (ADR% > 4, 1M-share volume floor). A second run with the
> source-accurate filters (ADR% > 5, $3.5M average daily **dollar** volume)
> was added 2026-07-10 — see
> [the corrected-filter section](#corrected-filters-run-adr--5-35m-dollar-volume--2026-07-10).
> Spoiler: the correction made the result **worse**, and it also gutted the
> fixed top-decile benchmark. A third run testing one pre-registered
> regime-filter hypothesis (SPY > 200-day SMA entry gate) was added later
> the same day — see
> [the SPY regime-gate section](#spy--200-day-sma-regime-gate--2026-07-10)
> at the bottom. It also fails out-of-sample, which closes the book on
> regime-timing as the missing ingredient.

**Run:** 2026-07-10, `python walkforward.py` (defaults: 2020-01-01 → 2024-12-31,
$100K per window, max 5 positions, full-market universe, cached prices).
Raw outputs (renamed after the corrected run took the default filenames):
`data/walkforward_report_adr4_sharevol.txt`, `…_folds_adr4_sharevol.csv`,
`…_trades_adr4_sharevol.csv`, `…_oos_equity_adr4_sharevol.csv`.

## Verdict, up front

**The top-decile edge does not survive contact with data it never saw.**
Selecting the RS cutoff on 18 months of prior data and trading the next
6 months — repeated across 7 folds, 2021-07 → 2024-12 — produced
**-2.8% total return (-0.8% CAGR), 5 trades, 20% win rate** over 3.5
out-of-sample years. The in-sample 10.5% CAGR was never achievable in
real time: no training window after 2020 dropped out of view ever
pointed to the top-decile cutoff again. The 10.5% figure should be
retired as an expectation and treated as what it was — a parameter
picked after seeing the answer.

## Protocol

- Rolling folds: **18 months train / 6 months test, stepped 6 months** → 7 folds.
- On each train window only, the full backtest engine (unchanged
  `strategy.py` rules) ran once per RS cutoff in **{2%, 5%, 10%, 15%}**.
- Selection rule, fixed before any test window was run: **highest train
  CAGR**; ties → higher Sharpe, then more closed trades, then the
  earlier cutoff in the list.
- The chosen cutoff — and only it — then ran on the following 6-month
  test window with a fresh $100K. Test windows are contiguous and
  non-overlapping, so they stitch into one out-of-sample equity curve.
- Nothing was re-run or re-picked after seeing test results. Every fold
  is reported below, including the zero-trade ones.

## Fold-by-fold

| Fold | Train window | Train leaderboard (CAGR / trades) | Chosen | Test window | OOS return | OOS trades |
|------|--------------|-----------------------------------|--------|-------------|-----------|------------|
| 1 | 2020-01 → 2021-06 | 2%: -0.7%/1 · 5%: -0.6%/4 · **10%: +10.8%/13** · 15%: +9.7%/20 | **10%** | 2021-07 → 2021-12 | **-1.0%** (Sharpe -0.21, DD -4.4%) | 1 (ASAN, -1.0R) |
| 2 | 2020-07 → 2021-12 | 2%: -0.7%/1 · 5%: -2.0%/3 · **10%: +8.1%/10** · 15%: +7.0%/17 | **10%** | 2022-01 → 2022-06 | **-3.0%** (Sharpe -3.20, DD -3.0%) | 3 (RES, LNTH, HDSN — all -1.0R) |
| 3 | 2021-01 → 2022-06 | **2%: -0.7%/1** · 5%: -2.0%/3 · 10%: -2.1%/6 · 15%: -4.0%/11 | **2%** | 2022-07 → 2022-12 | **0.0%** | 0 |
| 4 | 2021-07 → 2022-12 | **2%: -0.7%/1** · 5%: -2.0%/3 · 10%: -1.9%/5 · 15%: -3.9%/8 | **2%** | 2023-01 → 2023-06 | **0.0%** | 0 |
| 5 | 2022-01 → 2023-06 | **2%: -0.7%/1** · 5%: -2.7%/4 · 10%: -2.6%/6 · 15%: -4.6%/9 | **2%** | 2023-07 → 2023-12 | **0.0%** | 0 |
| 6 | 2022-07 → 2023-12 | 2%: 0.0%/0 · **5%: +14.3%/3** · 10%: +13.6%/6 · 15%: +11.4%/9 | **5%** | 2024-01 → 2024-06 | **+1.2%** (Sharpe 0.37) | 1 (NVAX, +1.2R) |
| 7 | 2023-01 → 2024-06 | 2%: +0.8%/1 · **5%: +13.2%/4** · 10%: +11.6%/6 · 15%: +8.5%/10 | **5%** | 2024-07 → 2024-12 | **0.0%** | 0 |

Full per-fold Sharpe/Sortino/Calmar/max-DD are in
`data/walkforward_report.txt`; the complete out-of-sample trade log
(all 5 trades) is in `data/walkforward_trades.csv`.

## Stitched out-of-sample result (2021-07 → 2024-12)

Every dollar below was traded on a cutoff chosen from prior data only.

| Metric | Walk-forward OOS | Fixed top-10% on same span (in-sample pick, for reference) |
|--------|------------------|------------------------------------------------------------|
| Total return | **-2.8%** | +41.2% |
| CAGR | **-0.8%** | +10.4% |
| Max drawdown | -7.3% | -8.1% |
| Sharpe (daily, rf=0) | -0.17 | 0.92 |
| Sortino (rf=0) | -0.24 | — |
| Calmar | -0.11 | — |
| Closed trades | 5 (win rate 20%, avg -0.57R) | 13 |

Monthly return distribution (42 OOS months): **4.8% positive months**
(2 of 42 — the account was flat-in-cash most months), mean -0.07%,
median 0.00%, std 0.74%, best +2.46%, worst -3.39%.

## What actually failed

1. **The selection is unstable, which is the tell.** The "best" cutoff
   flip-flopped 10% → 2% → 5% across folds. Top decile only wins
   training while 2020 is inside the train window; the moment 2020
   rolls out, it never wins again. A parameter that can only be found
   by including the period that motivated it is the textbook signature
   of in-sample selection.
2. **Train-window CAGR is noise at these trade counts.** Train windows
   produced 0–20 trades (usually ≤13), so ranking cutoffs by train CAGR
   is largely ranking luck. In the 2021–2023 chop folds every cutoff
   was negative, the rule picked the least-bad (2%), and that cutoff
   then took literally zero trades for three consecutive test windows —
   18 straight months without a position.
3. **The OOS trade log is 4 stop-outs and 1 modest winner.** ASAN, RES,
   LNTH, HDSN all -1.0R; NVAX +1.2R. The stop discipline worked exactly
   as designed (no loser worse than -1.02R) — there was simply no edge
   for it to protect.

## What this does and does not prove

- It **does** prove the *process* — "pick the RS cutoff that worked
  recently" — has no out-of-sample value on 2020-2024. Real-time
  expectation for that process: ~0% minus costs, before the 1–4pt
  survivorship haircut that still applies to everything here.
- It does **not** prove a fixed top-10% cutoff can never work: held
  fixed over the same OOS span it returned +10.4% CAGR (13 trades). But
  that number is unusable as validation, because "hold 10% fixed" is
  precisely the choice made after seeing 2020-2024 — the thing this
  test exists to catch. With 13 trades it also remains statistically
  thin.
- The one honest test of top-decile left standing is the **forward
  paper-trading account** (which runs top-decile by default). Its
  results accrue on genuinely unseen data. Judged against this
  walk-forward, the realistic prior for it is roughly flat, not 10.5%.
- Tempting next step to avoid: swapping the selection metric (Sharpe
  instead of CAGR, more cutoffs, different window lengths) until the
  walk-forward looks good. That would just move the in-sample selection
  up one level. Any protocol change must be justified before seeing its
  results.

## Reproduce

```bash
cd trading-bot && source .venv/bin/activate
python walkforward.py                 # uses the cached 2020-2024 prices
python walkforward.py --cutoffs 0.02,0.05,0.10,0.15 --train-months 18 --test-months 6
```

Inherited caveats (unchanged from `backtest.py`): survivorship-biased
universe (today's listings only), 0.1% slippage, adjusted Yahoo data,
no commissions. Walk-forward removes the parameter-selection bias; it
does not fix the data bias, which only flatters these numbers.

---

# Corrected-filters run (ADR > 5, $3.5M dollar volume) — 2026-07-10

The original implementation deviated from the actual source spec
(verified against sartrading.io/strategy on 2026-07-10) in two places:
ADR% floor was coded as > 4 instead of > 5, and the liquidity gate was
1M **shares**/day instead of $3.5M average daily **dollar** volume — a
different metric, not just a different number. Both were fixed in
`strategy.py` / `scanner.py` / `backtest.py` and the identical
walk-forward protocol was re-run (same folds, same cutoffs, same
selection rule, same cached price data). Raw outputs:
`data/walkforward_report.txt`, `…_folds.csv`, `…_trades.csv`,
`…_oos_equity.csv`.

## Answer to the question asked

**The corrected filters changed the result — for the worse.** The
out-of-sample walk-forward went from -0.8% CAGR to **-3.9% CAGR**, and
the fixed top-decile benchmark collapsed from +10.4% to **+1.3% CAGR**
on the same span. The source-accurate rules do not rescue the strategy;
they remove what was left of it.

| Metric (stitched OOS, 2021-07 → 2024-12) | Original filters | Corrected filters |
|---|---|---|
| Total return | -2.8% | **-13.1%** |
| CAGR | -0.8% | **-3.9%** |
| Max drawdown | -7.3% | **-15.5%** |
| Sharpe / Sortino / Calmar | -0.17 / -0.24 / -0.11 | **-0.65 / -0.80 / -0.26** |
| Closed trades | 5 (win rate 20%, avg -0.57R) | **13 (win rate 15.4%, avg -1.07R)** |
| Positive months | 2 of 42 | 2 of 42 |
| Benchmark: fixed top 10%, same span | +41.2% total / +10.4% CAGR / Sharpe 0.92 | **+4.7% total / +1.3% CAGR / Sharpe 0.17** |

Fold-by-fold (details in `data/walkforward_report.txt`): chosen cutoffs
were 15%, 2%, 2%, 10%, 2%, 15%, 5% — even less stable than before
(original run: 10, 10, 2, 2, 2, 5, 5) — and six of seven test windows
were negative or flat. Fold 4 was the worst: the rule picked top 10%
off a marginally positive train window and lost -6.8% in six months
(3 trades, all losers, avg -2.31R).

## What the correction actually did

The stricter ADR floor (>5) and the dollar-volume gate shift the
eligible pool toward **more volatile names and cheaper high-turnover
stocks** (a $6 stock trading 700K shares/day now qualifies; a slow $80
large cap trading 800K shares/day now qualifies too, where the share
floor excluded it). Trade counts went up (13 vs 5 OOS; benchmark 25 vs
13) while quality went down. The +10.5% in-sample top-decile result
documented in the README was therefore **partly an artifact of the
mis-coded filters**: run under the rules the source actually specifies,
its configuration earns +1.3% CAGR on 2021-2024 — inside the noise band
of zero, and before the 1-4pt survivorship haircut.

## Combined verdict

Under the source-accurate filters, there is no configuration left
standing: literal top-2% barely trades, walk-forward cutoff selection
loses money (-3.9% CAGR), and the after-the-fact top-decile pick is
flat. The mechanical strategy, implemented faithfully, does not make
money on 2020-2024 US equities in this framework. The paper account was
reset on 2026-07-10 (fresh $100K, no fills had occurred) so its forward
record reflects the corrected rules only.

---

# SPY > 200-day SMA regime gate — 2026-07-10

**One pre-registered hypothesis, stated before running:** the losses
above come from taking momentum breakouts in hostile tape, so gating
entries on SPY being above its own 200-day SMA should rescue the edge.
This section tests exactly that variant and nothing else. Design was
fixed in advance:

- **Gate**: an entry may fire only if SPY closed above its 200-day SMA
  at the **prior close** (no same-day lookahead — an intraday breakout
  can't know tonight's SPY close). Setups still form and age on the
  watchlist; open positions are managed normally. Only the entry fires
  are gated. No entry/exit rule was retuned.
- **Implementation**: optional `--spy-filter` flag on `backtest.py` /
  `walkforward.py` (`load_spy_regime()` + a `regime` parameter on
  `Backtest`). Flag off is the untouched code path — verified by
  re-running the unfiltered walk-forward after the refactor:
  trades and equity CSVs came back **byte-identical** to the
  corrected-filters run.
- **Protocol**: identical to both prior runs (same 7 folds, same
  cutoffs, same selection rule, same cached prices). With the flag on,
  the gate applies to train windows, test windows and the fixed
  top-10% benchmark alike, so training selects over the same gated
  strategy the test window trades.
- Gate coverage on 2020-2024: ON 77.8% of sessions; OFF Feb–May 2020
  (COVID crash), briefly Jan 2022, Apr 2022 → Jan 2023 (the bear), and
  a few short 2023 wobbles. Exactly the periods the hypothesis wants
  to skip.
- Raw outputs: `data/walkforward_report_spy200.txt`, `…_folds_spy200.csv`,
  `…_trades_spy200.csv`, `…_oos_equity_spy200.csv`.

## Side-by-side (stitched OOS, 2021-07 → 2024-12, corrected filters both sides)

| Metric | Gate OFF | Gate ON |
|---|---|---|
| Total return | -13.1% | **-3.0%** |
| CAGR | -3.9% | **-0.9%** |
| Max drawdown | -15.5% | **-5.4%** |
| Sharpe / Sortino / Calmar | -0.65 / -0.80 / -0.26 | **-0.21 / -0.31 / -0.16** |
| Closed trades | 13 (win rate 15.4%, avg -1.07R) | **5 (win rate 20.0%, avg -0.61R)** |
| Positive months (of 42) | 2 | **1** |
| Benchmark: fixed top 10%, same span | +4.7% total / +1.3% CAGR / Sharpe 0.17 / DD -15.9% / 25 trades | **+0.8% total / +0.2% CAGR / Sharpe 0.08 / DD -19.0% / 19 trades** |

Fold-by-fold with the gate on: chosen cutoffs 2, 2, 2, 2, 2, 15, 5;
OOS returns 0.0%, 0.0%, -1.2%, -1.0%, 0.0%, -0.8%, 0.0%. Seven folds,
zero positive. The five trades that fired in-regime were FDMT -1.22R,
VKTX -1.02R, ROOT -1.02R, CRBP -1.02R and NVAX +1.23R — four stop-outs
and one modest winner, average -0.61R.

## The hypothesis fails

**The gate lost less money only because it traded less; the trades it
allowed were still losers.** -0.9% CAGR vs -3.9% is not an edge
appearing — it is the account sitting in cash more (1 positive month
in 42, monthly std 0.29%). Per trade, in-regime entries averaged
-0.61R with a 20% win rate: the breakouts that fire while SPY is above
its 200-day SMA fail at essentially the same rate as the rest.

Two details make the failure unambiguous rather than borderline:

1. **The benchmark got worse, not better.** Fixed top-10% with the
   gate fell from +1.3% CAGR to +0.2%, with a deeper drawdown (-19.0%
   vs -15.9%). If bad-regime entries were the problem, filtering them
   out of the benchmark should have helped. It didn't.
2. **Training windows show the entries lose even in-regime.** With the
   gate on, folds 1–3 chose top 2% because it took *zero* training
   trades — 0.0% CAGR beat every active cutoff, all of which were
   negative (e.g. fold 1: 5% → -11.2%, 10% → -5.4%, 15% → -0.8%).
   Under the gate, "never trade" outperformed every configuration that
   traded, across three consecutive 18-month training windows.

## What this means — plainly

This was the pre-registered regime-timing hypothesis, and it failed
out-of-sample. That is a real, informative result: **the problem is
not that the strategy trades in bad regimes — the entries themselves
have no edge in this data, in good regimes or bad.** Regime-gating a
losing entry signal produces a smaller loss, not a profit.

Consequently, further parameter tuning — different SMA lengths,
breadth gates, VIX filters, other cutoff grids — is unlikely to help
and should not be attempted piecemeal. Each additional variant tested
against the same 2020-2024 data is another draw from the multiple-
comparisons well, and anything that "works" at this point is far more
likely to be selection than signal. What's left standing is what was
left standing before this run: the forward paper account (corrected
rules, reset 2026-07-10) accruing genuinely unseen data, or a
fundamentally different approach — different entry logic, different
data (point-in-time universe with delistings), or a different strategy
class — justified in advance, not discovered by iterating on this
window.

## Reproduce

```bash
cd trading-bot && source .venv/bin/activate
python walkforward.py               # gate off, default filenames
python walkforward.py --spy-filter  # gate on, *_spy200 filenames
```

---

# Multi-instrument suite walk-forward — 2026-07-13

Three new strategies across five instruments were built (`bot/`) and
walk-forward validated on maximum available Alpaca history **before any
paper trading**. Protocol identical in spirit to the momentum runs
above and pre-registered in full (grids, selection rule, costs,
execution model) before any test window was scored.

| Instrument | Strategy | Bars | History | Folds |
|---|---|---|---|---|
| SPY | mean reversion (20-SMA/stddev z-score, exit at mean, 2 ATR hard stop) | 15-min RTH | 2016-01 → 2026-07 | 18 |
| QQQ | same, entry_z 1.8 default | 15-min RTH | 2016-01 → 2026-07 | 18 |
| BTC/USD | momentum breakout (20-bar high + 1.5x volume, 2x ATR trail), long-only | 1-hour | 2021-01 → 2026-07 | 8 |
| GLD | trend following (50/200 EMA cross, 3x ATR trail), long-only | session 4-hour | 2016-01 → 2026-07 | 18 |
| USO | same | session 4-hour | 2016-01 → 2026-07 | 18 |

## Verdict, up front

**Zero of the five instrument/strategy pairs show any out-of-sample
edge. Nothing is flagged paper-trade-ready; the entire suite is parked
(`ENABLED = []` in `bot/suite_bot.py`).**

| Stitched OOS (params chosen on prior data only) | Total | CAGR | Max DD | Sharpe | Trades | Win rate | Avg R | Breaker folds |
|---|---|---|---|---|---|---|---|---|
| SPY meanrev (2017-07 → 2026-07) | **-80.3%** | -16.5% | -81.1% | -1.6 | 2,746 | 47% | -0.16R | 13 / 18 |
| QQQ meanrev (2017-07 → 2026-07) | **-73.2%** | -13.6% | -73.6% | -1.4 | 2,256 | 48% | -0.12R | 13 / 18 |
| BTC/USD breakout (2022-07 → 2026-06) | **-50.8%** | -16.3% | -51.4% | -1.6 | 126 | 22% | -0.46R | 8 / 8 |
| GLD trend (2017-07 → 2026-07) | **-6.5%** | -0.7% | -6.7% | -0.28 | 12 | 33% | -0.29R | 0 / 18 |
| USO trend (2017-07 → 2026-07) | **-10.4%** | -1.2% | -17.1% | -0.53 | 18 | 39% | -0.20R | 0 / 18 |

(Stitched OOS compounds 6-month test windows that each restart with a
fresh $100K; folds that hit the 10% in-window circuit breaker halt flat
for the rest of that window. Fixed-default benchmarks over the same
spans — one continuous run each — are similarly negative: SPY -9.4%,
QQQ -10.1%, BTC -9.7%, GLD -7.0%, USO -9.2% total.)

## Protocol (fixed before any test window ran)

- **Folds**: 18m train / 6m test, stepped 6m, anchored at each
  instrument's own data start. Fresh $100K per window; positions open
  at a window's end force-closed and counted.
- **Grids** (3 parameter sets each, in the strategy modules):
  meanrev entry_z {1.5, 1.8, 2.2}; breakout lookback {20, 40, 55};
  trend EMA pair {50/200, 30/120, 100/300}. Selection rule identical to
  the momentum walk-forward: highest train CAGR, ties → Sharpe →
  trades → earlier grid entry.
- **Execution model** (`bot/engine.py`): decisions on bar closes, fills
  at next open; stops intra-bar (gap → open fill); hard stop at entry
  ∓ STOP_ATR × ATR(signal bar), never moved; trailing stops only
  tighten. Sizing: 1 ATR move = 1% of equity, notional capped at 1x
  equity (the cap, not the ATR rule, binds on 15-min SPY/QQQ). 10%
  circuit breaker inside every window.
- **Costs per side**: stocks 0.02%, BTC 0.30% (Alpaca 25bp taker fee
  + 5bp slippage).
- **Data**: Alpaca SIP consolidated-tape bars (full real volume — the
  free `iex` feed only carries ~2-3% of volume and would have poisoned
  the 1.5x-volume confirmation); crypto bars from the crypto feed.
  Stock bars RTH-filtered; "4-hour" bars are session-anchored
  (09:30-13:30 / 13:30-16:00 ET). No survivorship issue: the five
  instruments were fixed by the design brief, not screened.

## Cost-sensitivity check (run after, labeled as such)

To rule out "the cost model killed it" as the explanation, the spec
defaults were re-run over full history with the circuit breaker off, at
the modeled cost and at **zero** cost:

| Full history, defaults, no breaker | Modeled cost | Zero cost |
|---|---|---|
| SPY meanrev | -25.1% CAGR | **-8.1% CAGR** |
| QQQ meanrev | -19.7% CAGR | **-4.2% CAGR** |
| BTC/USD breakout | -53.6% CAGR | **-4.7% CAGR** (avg +0.003R/trade) |

Even free execution loses. The entries have no edge; costs only deepen
the bleed. (BTC is the textbook case: exactly zero per-trade edge
before costs, then 0.6% round-trip fees on ~130 trades/year.)

## What failed, per strategy

1. **Mean reversion (SPY/QQQ)**: ~48% win rate with losers slightly
   larger than winners = steady bleed at ~1-2 trades/day, and the 10%
   breaker tripped in 13 of 18 test windows for both symbols. The
   selected entry_z flip-flopped across folds (1.5 → 2.2 → 1.5 → 1.8…)
   — the same selection-instability tell as the momentum RS cutoff.
   A 20-bar z-score on 15-min index ETFs is too weak a signal to clear
   even zero costs on 2016-2026 data.
2. **Momentum breakout (BTC/USD)**: 22% OOS win rate, avg -0.46R, and
   the breaker tripped in **all eight** test windows. Volume-confirmed
   1-hour breakouts on BTC after 2021 resolve overwhelmingly into
   chop; the 2x ATR trail systematically sells the retrace.
3. **Trend following (GLD/USO)**: structurally too few signals — a
   50/200 EMA cross on 4-hour bars fires ~1-2 times a year, so most
   6-month test windows contained zero trades. The loophole-closing
   check is the continuous 9-year benchmark run: GLD -7.0% total
   **while GLD itself roughly tripled** — every cross either whipsawed
   or gave its open profit back to the 3x ATR trail. USO the same
   (-9.2%).

## What this does and does not prove

- It **does** prove none of these five pairs, as specified, earned
  anything out-of-sample on the maximum history Alpaca serves, under
  honest costs and honest fills — and (via the zero-cost check) that
  this is signal failure, not friction.
- It does **not** prove mean reversion / breakout / trend are dead as
  strategy classes — only these parameterizations on these timeframes
  and instruments. The trend result in particular is thin (12-18
  trades); "no edge detectable" is the right reading, not "negative
  edge proven".
- The temptation to resist is identical to last time: swapping
  z-thresholds, lookbacks, EMA pairs, timeframes or instruments until
  a fold passes is multiple-comparisons harvesting. Any revival must
  be a new pre-registered hypothesis, justified before running.

## Disposition

- **All five pairs parked.** `ENABLED` in `bot/suite_bot.py` stays
  empty; the runner will not open positions for non-enabled
  instruments. Nothing trades Monday.
- The infrastructure (data layer, engine, three strategy modules,
  live runner with shared risk layer, correlation filter, broker-side
  hard stops, circuit breaker, trades.csv / daily_pnl.csv logging) is
  built, tested (31 passing) and stays — it is strategy-agnostic and
  ready for the next pre-registered hypothesis.
- Raw outputs: `data/wf_suite_report.txt`,
  `data/wf_suite_folds_<SYM>.csv`, `data/wf_suite_trades_<SYM>.csv`,
  `data/wf_suite_oos_equity_<SYM>.csv`.

## Reproduce

```bash
cd trading-bot && source .venv/bin/activate
python -m bot.walkforward                          # all five instruments
python -m bot.walkforward --instruments "BTC/USD"  # one at a time
```
