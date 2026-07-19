"""Morgan Tradez -- walk-forward validation of the RS-cutoff parameter.

Why this exists: the top-decile result (10.5% CAGR on 2020-2024, 26
trades) was chosen AFTER watching the top-2% cut fail on the same
window. That is in-sample parameter selection, not a validated edge.
This script asks the only question that matters: does picking the RS
cutoff on data you HAVE seen keep working on data you HAVEN'T?

Protocol (rolling walk-forward):
  * Split the period into rolling windows: TRAIN_MONTHS of training
    followed by TEST_MONTHS of testing, rolled forward STEP_MONTHS at a
    time (defaults 18 / 6 / 6 -> 7 folds on 2020-2024).
  * On each TRAIN window only, run the full backtest engine
    (backtest.Backtest, strategy.py rules untouched) once per RS cutoff
    in CUTOFFS and pick the best performer. "Best" = highest CAGR;
    ties break to higher Sharpe, then more closed trades, then the
    earlier cutoff in the list. The rule is fixed here, in advance.
  * Run ONLY the chosen cutoff on the following TEST window with a
    fresh $100K and record what actually happened out of sample.
  * Stitch the test windows (contiguous, non-overlapping) into one
    out-of-sample equity curve and score it with the same perf_metrics
    used everywhere else.

Honesty rules baked in:
  * Nothing is re-run, re-picked, or dropped after seeing test results;
    every fold appears in the report regardless of outcome.
  * Zero-trade windows are reported as zero-trade windows (flat equity).
  * Positions still open at a window's end are force-closed at the last
    close (same behavior as backtest.py) and counted separately.
  * Each fold's test starts flat with fresh capital: no positions or
    P&L carry across the train/test boundary or between folds.

Caveats inherited from backtest.py: survivorship-biased universe
(today's listings), 0.1% slippage, adjusted Yahoo data, no commissions.
Walk-forward fixes the parameter-selection bias, not the data bias.

Usage:
    python walkforward.py                        # 2020-2024, defaults above
    python walkforward.py --cutoffs 0.02,0.05,0.10,0.15
    python walkforward.py --train-months 18 --test-months 6 --step-months 6
"""

import argparse
import os

import numpy as np
import pandas as pd

from backtest import (DATA_DIR, Backtest, load_price_data, load_spy_regime,
                      monthly_distribution_lines, perf_metrics, precompute)
from scanner import get_universe

DEFAULT_CUTOFFS = "0.02,0.05,0.10,0.15"


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def build_folds(start: str, end: str, train_months: int, test_months: int,
                step_months: int) -> list[dict]:
    """Rolling train/test windows; only folds whose test window fits fully."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    folds = []
    k = 0
    while True:
        tr_s = start_ts + pd.DateOffset(months=step_months * k)
        tr_e = tr_s + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        te_s = tr_e + pd.Timedelta(days=1)
        te_e = te_s + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if te_e > end_ts:
            break
        folds.append({"fold": k + 1, "train": (tr_s, tr_e), "test": (te_s, te_e)})
        k += 1
    return folds


# ---------------------------------------------------------------------------
# Single-window run
# ---------------------------------------------------------------------------

def run_window(enriched: dict, rs_pct: pd.DataFrame, start: pd.Timestamp,
               end: pd.Timestamp, capital: float, max_positions: int,
               top_pct: float, regime: pd.Series | None = None) -> dict:
    """One independent backtest over [start, end]; returns metrics + trades."""
    bt = Backtest(enriched, rs_pct, str(start.date()), str(end.date()),
                  capital, max_positions, top_pct, regime=regime)
    equity = bt.run()
    eq = equity["equity"]
    m = perf_metrics(eq, capital)
    closed = [t for t in bt.trades if t["final"]]
    rr = pd.Series([t.get("r_multiple") for t in closed], dtype=float).dropna()
    m["trades"] = len(closed)
    m["forced_close"] = sum(1 for t in closed if t["reason"] == "end_of_backtest")
    m["avg_r"] = float(rr.mean()) if len(rr) else float("nan")
    m["win_rate"] = float((rr > 0).mean()) if len(rr) else float("nan")
    m["equity"] = eq
    m["trade_rows"] = bt.trades
    return m


def choose_cutoff(train_results: dict[float, dict]) -> float:
    """Highest train CAGR; ties -> higher Sharpe, more trades, earlier cutoff."""
    best_c, best_key = None, None
    for c, m in train_results.items():
        key = (round(m["cagr"], 10), round(m["sharpe"], 10), m["trades"])
        if best_key is None or key > best_key:
            best_c, best_key = c, key
    return best_c


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _f2(v: float) -> str:
    return f"{v:.2f}" if np.isfinite(v) else "n/a"


def fold_report_lines(row: dict, cutoffs: list[float]) -> list[str]:
    tr = row["train_results"]
    lines = [
        f"FOLD {row['fold']}  train {row['train_start']} -> {row['train_end']}"
        f"   test {row['test_start']} -> {row['test_end']}",
        f"  train leaderboard ({'cutoff: CAGR / Sharpe / trades'}):",
    ]
    for c in cutoffs:
        m = tr[c]
        mark = "  <-- chosen" if c == row["chosen_cutoff"] else ""
        lines.append(f"    top {c:>4.0%}:  {m['cagr']:>7.1%} / {_f2(m['sharpe']):>5} "
                     f"/ {m['trades']:>2d}{mark}")
    t = row["test_metrics"]
    wr = f"{t['win_rate']:.0%}" if np.isfinite(t["win_rate"]) else "n/a"
    ar = f"{t['avg_r']:+.2f}R" if np.isfinite(t["avg_r"]) else "n/a"
    lines += [
        f"  OUT-OF-SAMPLE (top {row['chosen_cutoff']:.0%} applied to unseen test window):",
        f"    return {t['total_return']:+7.1%} (annualized {t['cagr']:+7.1%})   "
        f"max DD {t['max_dd']:>6.1%}",
        f"    Sharpe {_f2(t['sharpe'])}   Sortino {_f2(t['sortino'])}   "
        f"Calmar {_f2(t['calmar'])}",
        f"    trades {t['trades']} (win rate {wr}, avg {ar}, "
        f"{t['forced_close']} force-closed at window end)",
        "",
    ]
    return lines


def main():
    p = argparse.ArgumentParser(
        description="Walk-forward validation of the RS cutoff parameter")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--train-months", type=int, default=18)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--step-months", type=int, default=6)
    p.add_argument("--cutoffs", default=DEFAULT_CUTOFFS,
                   help="comma-separated RS cutoffs to compete in training")
    p.add_argument("--universe", choices=["all", "sp500"], default="all")
    p.add_argument("--universe-limit", type=int, default=0,
                   help="cap universe size for a quick smoke test")
    p.add_argument("--spy-filter", action="store_true",
                   help="entries only fire when SPY closed above its 200-day "
                        "SMA the prior session; applied identically to train "
                        "windows, test windows and the benchmark; output "
                        "files get a _spy200 suffix")
    args = p.parse_args()
    cutoffs = [float(x) for x in args.cutoffs.split(",")]
    regime = load_spy_regime(args.start, args.end) if args.spy_filter else None
    suffix = "_spy200" if args.spy_filter else ""

    universe = get_universe(source=args.universe)
    if args.universe_limit:
        universe = universe[:args.universe_limit]
    print(f"Universe: {len(universe)} tickers")
    frames = load_price_data(universe, args.start, args.end)
    print(f"Usable tickers: {len(frames)}. Precomputing indicators + RS ranks...")
    enriched, rs_pct = precompute(frames)

    folds = build_folds(args.start, args.end, args.train_months,
                        args.test_months, args.step_months)
    print(f"\n{len(folds)} folds "
          f"({args.train_months}m train / {args.test_months}m test, "
          f"{args.step_months}m step):")
    for f in folds:
        print(f"  fold {f['fold']}: train {f['train'][0].date()} -> "
              f"{f['train'][1].date()}, test {f['test'][0].date()} -> "
              f"{f['test'][1].date()}")

    fold_rows, oos_ret_chunks, oos_trades = [], [], []
    for f in folds:
        tr_s, tr_e = f["train"]
        te_s, te_e = f["test"]
        print(f"\n=== FOLD {f['fold']}: training {tr_s.date()} -> {tr_e.date()} ===")
        train_results = {}
        for c in cutoffs:
            m = run_window(enriched, rs_pct, tr_s, tr_e, args.capital,
                           args.max_positions, c, regime=regime)
            train_results[c] = m
            print(f"  train top {c:>4.0%}: CAGR {m['cagr']:+7.1%}  "
                  f"Sharpe {_f2(m['sharpe'])}  trades {m['trades']}")
        chosen = choose_cutoff(train_results)
        print(f"  chosen cutoff: top {chosen:.0%} -> testing "
              f"{te_s.date()} -> {te_e.date()}")
        tm = run_window(enriched, rs_pct, te_s, te_e, args.capital,
                        args.max_positions, chosen, regime=regime)
        print(f"  OOS: return {tm['total_return']:+7.1%}  "
              f"Sharpe {_f2(tm['sharpe'])}  trades {tm['trades']}")

        fold_rows.append({
            "fold": f["fold"],
            "train_start": tr_s.date(), "train_end": tr_e.date(),
            "test_start": te_s.date(), "test_end": te_e.date(),
            "chosen_cutoff": chosen,
            "train_results": train_results,
            "test_metrics": tm,
        })
        oos_ret_chunks.append(tm["equity"].pct_change().dropna())
        for t in tm["trade_rows"]:
            oos_trades.append({**t, "fold": f["fold"], "cutoff": chosen})

    # ---- stitched out-of-sample curve (contiguous test windows) ------------
    oos_rets = pd.concat(oos_ret_chunks).sort_index()
    oos_eq = args.capital * (1.0 + oos_rets).cumprod()
    anchor = pd.Series([args.capital],
                       index=[folds[0]["test"][0] - pd.Timedelta(days=1)])
    oos_eq = pd.concat([anchor, oos_eq])
    om = perf_metrics(oos_eq, args.capital)
    oos_closed = [t for t in oos_trades if t["final"]]
    oos_rr = pd.Series([t.get("r_multiple") for t in oos_closed],
                       dtype=float).dropna()

    # Benchmark on the same span: the in-sample-selected top-decile cutoff
    # held fixed across the whole OOS period, for an apples-to-apples line.
    bench_s, bench_e = folds[0]["test"][0], folds[-1]["test"][1]
    print(f"\nBenchmark: fixed top 10% over {bench_s.date()} -> {bench_e.date()}")
    bench = run_window(enriched, rs_pct, bench_s, bench_e, args.capital,
                       args.max_positions, 0.10, regime=regime)

    # ---- report -------------------------------------------------------------
    lines = [
        "=" * 72,
        "MORGAN TRADEZ WALK-FORWARD REPORT (RS cutoff selection)",
        f"Period: {args.start} -> {args.end}   Capital per window: "
        f"${args.capital:,.0f}",
        f"Folds: {len(folds)}  ({args.train_months}m train / "
        f"{args.test_months}m test, {args.step_months}m step)   "
        f"Cutoffs tested: {', '.join(f'{c:.0%}' for c in cutoffs)}",
        "Selection rule (fixed in advance): highest train CAGR; ties ->",
        "higher Sharpe, then more trades, then the earlier cutoff.",
        f"SPY>200SMA entry gate (prior close): "
        f"{'ON' if args.spy_filter else 'OFF'}",
        "=" * 72,
        "",
    ]
    for row in fold_rows:
        lines += fold_report_lines(row, cutoffs)

    wr = f"{(oos_rr > 0).mean():.1%}" if len(oos_rr) else "n/a"
    ar = f"{oos_rr.mean():+.2f}R" if len(oos_rr) else "n/a"
    lines += [
        "-" * 72,
        f"STITCHED OUT-OF-SAMPLE RESULT "
        f"({folds[0]['test'][0].date()} -> {folds[-1]['test'][1].date()}, "
        f"every dollar traded on a cutoff chosen from prior data only):",
        f"  Total return:        {om['total_return']:>10.1%}",
        f"  CAGR:                {om['cagr']:>10.1%}",
        f"  Max drawdown:        {om['max_dd']:>10.1%}",
        f"  Sharpe (daily,rf=0): {_f2(om['sharpe']):>10}",
        f"  Sortino (rf=0):      {_f2(om['sortino']):>10}",
        f"  Calmar (CAGR/maxDD): {_f2(om['calmar']):>10}",
        f"  Closed trades:       {len(oos_closed):>10d}   "
        f"(win rate {wr}, avg {ar})",
        "",
    ]
    lines += ["  " + ln for ln in monthly_distribution_lines(om["monthly"])]
    lines += [
        "",
        f"BENCHMARK -- fixed top 10% (the in-sample pick) on the same span:",
        f"  Total return {bench['total_return']:+7.1%}   CAGR "
        f"{bench['cagr']:+7.1%}   Sharpe {_f2(bench['sharpe'])}   "
        f"max DD {bench['max_dd']:.1%}   trades {bench['trades']}",
        "",
        "READ THIS BEFORE THE HEADLINE NUMBER:",
        "  * The walk-forward CAGR above is the defensible estimate of what",
        "    cutoff-selection-on-past-data would actually have earned. The",
        "    in-sample 10.5% (2020-2024, top decile) is NOT comparable: it",
        "    includes the 2020 training-only period and was picked after the",
        "    fact.",
        "  * Survivorship bias still applies (today's listings only); knock",
        "    1-4 points off the CAGR for a defensible band.",
        "  * Trade counts per fold are tiny; single trades can swing a fold.",
        "=" * 72,
    ]
    text = "\n".join(lines)
    print()
    print(text)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f"walkforward_report{suffix}.txt"), "w") as fh:
        fh.write(text + "\n")

    csv_rows = []
    for row in fold_rows:
        flat = {k: row[k] for k in ("fold", "train_start", "train_end",
                                    "test_start", "test_end", "chosen_cutoff")}
        for c, m in row["train_results"].items():
            flat[f"train_cagr_{c:.2f}"] = round(m["cagr"], 4)
            flat[f"train_trades_{c:.2f}"] = m["trades"]
        t = row["test_metrics"]
        flat.update(test_return=round(t["total_return"], 4),
                    test_cagr_ann=round(t["cagr"], 4),
                    test_sharpe=round(t["sharpe"], 3),
                    test_sortino=round(t["sortino"], 3) if np.isfinite(t["sortino"]) else "",
                    test_max_dd=round(t["max_dd"], 4),
                    test_trades=t["trades"],
                    test_forced_close=t["forced_close"])
        csv_rows.append(flat)
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(DATA_DIR, f"walkforward_folds{suffix}.csv"), index=False)
    pd.DataFrame(oos_trades).to_csv(
        os.path.join(DATA_DIR, f"walkforward_trades{suffix}.csv"), index=False)
    oos_eq.rename("equity").to_csv(
        os.path.join(DATA_DIR, f"walkforward_oos_equity{suffix}.csv"))
    print(f"\nSaved: data/walkforward_report{suffix}.txt, "
          f"data/walkforward_folds{suffix}.csv, "
          f"data/walkforward_trades{suffix}.csv, "
          f"data/walkforward_oos_equity{suffix}.csv")


if __name__ == "__main__":
    main()
