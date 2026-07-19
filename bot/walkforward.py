"""Walk-forward validation of the multi-instrument suite -- one report
per instrument/strategy pair, same protocol as the momentum walk-forward
in walkforward.py (whose build_folds and scoring math are imported and
reused, not reimplemented).

Protocol, fixed before any test window was run:
  * Rolling folds: 18 months train / 6 months test, stepped 6 months,
    anchored at each instrument's own data start (stocks 2016-01,
    BTC/USD 2021-01).
  * On each TRAIN window only, the engine runs once per parameter set in
    the strategy's pre-registered GRID (3 sets each -- see the strategy
    modules). Selection rule: highest train CAGR; ties -> higher Sharpe,
    then more closed trades, then the earlier entry in the grid. This is
    the identical rule walkforward.py pre-registered.
  * The chosen parameters -- and only they -- run on the following
    6-month test window with a fresh $100K. Test windows are contiguous
    and non-overlapping and stitch into one OOS equity curve per
    instrument, scored by backtest.perf_metrics on daily equity.
  * BENCHMARK per instrument: the spec's fixed default parameters run
    once over the whole stitched OOS span. This is the "traded the spec
    as written" line; it is reported, not selected on.
  * Nothing is re-run, re-picked or dropped after seeing test results.
    Zero-trade windows are reported as zero-trade windows.

Indicators are computed on full history once per parameter set and the
windows sliced afterwards -- all indicators are trailing, so this is the
same no-lookahead precompute pattern walkforward.py uses. Positions
open at a window's end are force-closed at the last close and counted.

Costs and the 10% circuit breaker apply inside every window exactly as
in the engine (see bot/engine.py docstring).

Usage:
    python -m bot.walkforward                      # all 5 instruments
    python -m bot.walkforward --instruments SPY,BTC/USD
"""

import argparse
import os

import numpy as np
import pandas as pd

from backtest import DATA_DIR, monthly_distribution_lines, perf_metrics
from walkforward import build_folds

from bot.data import load_bars
from bot.engine import Engine, daily_equity
from bot.strategies import INSTRUMENTS


def _f2(v) -> str:
    return f"{v:.2f}" if np.isfinite(v) else "n/a"


def _ptag(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items())


def run_window(strategy, df_slice, params, capital, cost, fractional,
               label) -> dict:
    eng = Engine(strategy, df_slice, params, capital=capital, cost=cost,
                 fractional=fractional, label=label)
    res = eng.run()
    deq = daily_equity(res["equity"])
    if len(deq) < 2:
        m = {"total_return": 0.0, "cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0,
             "sortino": 0.0, "calmar": 0.0, "monthly": pd.Series(dtype=float)}
    else:
        m = perf_metrics(deq, capital)
    closed = res["trades"]
    rr = pd.Series([t["r_multiple"] for t in closed], dtype=float).dropna()
    m.update(trades=len(closed),
             halted=res["halted"],
             forced_close=sum(1 for t in closed
                              if t["reason"] == "end_of_window"),
             avg_r=float(rr.mean()) if len(rr) else float("nan"),
             win_rate=float((rr > 0).mean()) if len(rr) else float("nan"),
             equity=deq, trade_rows=closed)
    return m


def choose(train_results: list[tuple[dict, dict]]) -> dict:
    """Highest train CAGR; ties -> Sharpe, trades, earlier grid entry."""
    best_p, best_key = None, None
    for params, m in train_results:
        key = (round(m["cagr"], 10), round(m["sharpe"], 10), m["trades"])
        if best_key is None or key > best_key:
            best_p, best_key = params, key
    return best_p


def validate_instrument(symbol: str, spec: dict, args) -> dict:
    strategy = spec["strategy"]
    fractional = spec["asset"] == "crypto"
    print(f"\n{'=' * 68}\n{symbol}  ({strategy.NAME}, {spec['timeframe']}, "
          f"cost {spec['cost']:.2%}/side)\n{'=' * 68}")
    raw = load_bars(symbol, spec["timeframe"], spec["history_start"],
                    spec["asset"], end=args.end)
    if raw.empty:
        return {"symbol": symbol, "error": "no data"}
    print(f"  bars: {len(raw)}  ({raw.index[0]} -> {raw.index[-1]})")

    enriched = {_ptag(p): strategy.prepare(raw, p) for p in strategy.GRID}
    d0, d1 = str(raw.index[0].date()), str(raw.index[-1].date())
    folds = build_folds(d0, d1, args.train_months, args.test_months,
                        args.step_months)
    if not folds:
        return {"symbol": symbol, "error": "not enough history for one fold"}
    print(f"  {len(folds)} folds ({args.train_months}m train / "
          f"{args.test_months}m test / {args.step_months}m step)")

    fold_rows, oos_ret_chunks, oos_trades = [], [], []
    for f in folds:
        tr_s, tr_e = f["train"]
        te_s, te_e = f["test"]
        train_results = []
        for p in strategy.GRID:
            df = enriched[_ptag(p)].loc[str(tr_s.date()):str(tr_e.date())]
            m = run_window(strategy, df, p, args.capital, spec["cost"],
                           fractional, symbol)
            train_results.append((p, m))
        chosen = choose(train_results)
        df = enriched[_ptag(chosen)].loc[str(te_s.date()):str(te_e.date())]
        tm = run_window(strategy, df, chosen, args.capital, spec["cost"],
                        fractional, symbol)
        print(f"  fold {f['fold']}: chose [{_ptag(chosen)}] -> OOS "
              f"{tm['total_return']:+6.1%}  {tm['trades']} trades"
              f"{'  [BREAKER]' if tm['halted'] else ''}")
        fold_rows.append({"fold": f["fold"],
                          "train_start": tr_s.date(), "train_end": tr_e.date(),
                          "test_start": te_s.date(), "test_end": te_e.date(),
                          "chosen": _ptag(chosen),
                          "train_results": train_results, "test_metrics": tm})
        if len(tm["equity"]) >= 2:
            oos_ret_chunks.append(tm["equity"].pct_change().dropna())
        for t in tm["trade_rows"]:
            oos_trades.append({**t, "fold": f["fold"], "params": _ptag(chosen)})

    # stitched OOS curve
    oos_rets = (pd.concat(oos_ret_chunks).sort_index()
                if oos_ret_chunks else pd.Series(dtype=float))
    if len(oos_rets):
        oos_eq = args.capital * (1.0 + oos_rets).cumprod()
        om = perf_metrics(oos_eq, args.capital)
    else:
        om = {"total_return": 0.0, "cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0,
              "sortino": 0.0, "calmar": 0.0, "monthly": pd.Series(dtype=float)}
        oos_eq = pd.Series(dtype=float)

    # benchmark: spec defaults, one run over the whole stitched OOS span
    dflt = strategy.default_params(symbol)
    bs, be = folds[0]["test"][0], folds[-1]["test"][1]
    bdf = strategy.prepare(raw, dflt).loc[str(bs.date()):str(be.date())]
    bench = run_window(strategy, bdf, dflt, args.capital, spec["cost"],
                       fractional, symbol)

    rr = pd.Series([t["r_multiple"] for t in oos_trades], dtype=float).dropna()
    return {"symbol": symbol, "strategy": strategy.NAME,
            "timeframe": spec["timeframe"], "grid": strategy.GRID,
            "folds": fold_rows, "oos_metrics": om, "oos_equity": oos_eq,
            "oos_trades": oos_trades, "oos_rr": rr,
            "oos_span": (bs.date(), be.date()),
            "bench_params": dflt, "bench": bench,
            "breaker_folds": sum(1 for r in fold_rows
                                 if r["test_metrics"]["halted"])}


def report_lines(r: dict) -> list[str]:
    if "error" in r:
        return [f"{r['symbol']}: SKIPPED -- {r['error']}", ""]
    om, bench = r["oos_metrics"], r["bench"]
    rr = r["oos_rr"]
    wr = f"{(rr > 0).mean():.0%}" if len(rr) else "n/a"
    ar = f"{rr.mean():+.2f}R" if len(rr) else "n/a"
    lines = [
        "=" * 68,
        f"{r['symbol']}  --  {r['strategy']} on {r['timeframe']} bars",
        "=" * 68,
        f"OOS span {r['oos_span'][0]} -> {r['oos_span'][1]}  "
        f"({len(r['folds'])} folds; circuit breaker tripped in "
        f"{r['breaker_folds']})",
        "",
        "Fold detail (train leaderboard: params CAGR/Sharpe/trades):",
    ]
    for row in r["folds"]:
        board = " | ".join(
            f"{_ptag(p)}: {m['cagr']:+.1%}/{_f2(m['sharpe'])}/{m['trades']}"
            for p, m in row["train_results"])
        t = row["test_metrics"]
        lines += [
            f"  fold {row['fold']}  train {row['train_start']} -> "
            f"{row['train_end']}   test {row['test_start']} -> {row['test_end']}",
            f"    {board}",
            f"    chosen [{row['chosen']}] -> OOS {t['total_return']:+7.1%}  "
            f"Sharpe {_f2(t['sharpe'])}  DD {t['max_dd']:.1%}  "
            f"{t['trades']} trades"
            f"{'  [CIRCUIT BREAKER]' if t['halted'] else ''}",
        ]
    lines += [
        "",
        "STITCHED OUT-OF-SAMPLE (params always chosen on prior data only):",
        f"  Total return {om['total_return']:+8.1%}   CAGR {om['cagr']:+7.1%}   "
        f"max DD {om['max_dd']:.1%}",
        f"  Sharpe {_f2(om['sharpe'])}   Sortino {_f2(om['sortino'])}   "
        f"Calmar {_f2(om['calmar'])}",
        f"  Closed trades {len(r['oos_trades'])}   win rate {wr}   avg {ar}",
    ]
    lines += ["  " + ln for ln in monthly_distribution_lines(om["monthly"])]
    lines += [
        "",
        f"BENCHMARK -- spec defaults [{_ptag(r['bench_params'])}] held fixed "
        "over the same span:",
        f"  Total return {bench['total_return']:+8.1%}   CAGR "
        f"{bench['cagr']:+7.1%}   Sharpe {_f2(bench['sharpe'])}   "
        f"max DD {bench['max_dd']:.1%}   trades {bench['trades']}"
        f"{'   [CIRCUIT BREAKER]' if bench['halted'] else ''}",
        "",
    ]
    return lines


def main():
    p = argparse.ArgumentParser(
        description="Per-instrument walk-forward validation of the suite")
    p.add_argument("--instruments", default=",".join(INSTRUMENTS),
                   help="comma-separated subset of " + ", ".join(INSTRUMENTS))
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--train-months", type=int, default=18)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--step-months", type=int, default=6)
    p.add_argument("--end", default=None, help="last data date (YYYY-MM-DD)")
    args = p.parse_args()

    results = []
    for sym in [s.strip() for s in args.instruments.split(",") if s.strip()]:
        if sym not in INSTRUMENTS:
            print(f"unknown instrument {sym}; skipping")
            continue
        results.append(validate_instrument(sym, INSTRUMENTS[sym], args))

    lines = [
        "=" * 68,
        "MULTI-INSTRUMENT SUITE -- WALK-FORWARD REPORT",
        f"Protocol: {args.train_months}m train / {args.test_months}m test / "
        f"{args.step_months}m step; fresh ${args.capital:,.0f} per window;",
        "selection rule fixed in advance: highest train CAGR, ties ->",
        "Sharpe, trades, earlier grid entry. Costs and 10% circuit breaker",
        "applied inside every window. See bot/engine.py for execution model.",
        "=" * 68, "",
    ]
    for r in results:
        lines += report_lines(r)

    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "wf_suite_report.txt"), "w") as fh:
        fh.write(text + "\n")
    for r in results:
        if "error" in r:
            continue
        tag = r["symbol"].replace("/", "")
        pd.DataFrame(r["oos_trades"]).to_csv(
            os.path.join(DATA_DIR, f"wf_suite_trades_{tag}.csv"), index=False)
        if len(r["oos_equity"]):
            r["oos_equity"].rename("equity").to_csv(
                os.path.join(DATA_DIR, f"wf_suite_oos_equity_{tag}.csv"))
        rows = []
        for row in r["folds"]:
            flat = {k: row[k] for k in ("fold", "train_start", "train_end",
                                        "test_start", "test_end", "chosen")}
            for p_, m in row["train_results"]:
                flat[f"train_cagr[{_ptag(p_)}]"] = round(m["cagr"], 4)
                flat[f"train_trades[{_ptag(p_)}]"] = m["trades"]
            t = row["test_metrics"]
            flat.update(test_return=round(t["total_return"], 4),
                        test_sharpe=round(t["sharpe"], 3),
                        test_max_dd=round(t["max_dd"], 4),
                        test_trades=t["trades"], test_breaker=t["halted"])
            rows.append(flat)
        pd.DataFrame(rows).to_csv(
            os.path.join(DATA_DIR, f"wf_suite_folds_{tag}.csv"), index=False)
    print(f"\nSaved: data/wf_suite_report.txt + per-instrument folds/trades/"
          f"equity CSVs")


if __name__ == "__main__":
    main()
