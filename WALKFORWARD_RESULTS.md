# Walk-forward validation of the RS cutoff — results

> **Two runs live in this file.** The original run below used the filters as
> first coded (ADR% > 4, 1M-share volume floor). A second run with the
> source-accurate filters (ADR% > 5, $3.5M average daily **dollar** volume)
> was added 2026-07-10 — see
> [the corrected-filter section](#corrected-filters-run-adr--5-35m-dollar-volume--2026-07-10)
> at the bottom. Spoiler: the correction made the result **worse**, and it
> also gutted the fixed top-decile benchmark.

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
