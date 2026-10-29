"""Like-for-like in-sample backtest: corrected filters, 2020-2024, top 10% RS,
on the same price cache the walk-forward used (data/prices_..._5309.pkl).

`python backtest.py` alone re-resolves today's universe and re-downloads
prices when the ticker count changes, so its numbers drift with the data
snapshot. This wrapper pins the walk-forward's cache instead.

    python scripts/run_insample_backtest.py [--cache PATH] [--top-pct 0.10]
"""
import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import backtest  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--cache",
               default=str(ROOT / "data" / "prices_2020-01-01_2024-12-31_5309.pkl"))
p.add_argument("--top-pct", default="0.10")
args = p.parse_args()
cache = Path(args.cache)
if not cache.exists():
    sys.exit(f"price cache not found: {cache} (run walkforward.py once to build it)")


def load_pinned_cache(tickers, start, end, cache_dir=None):
    print(f"Loading cached prices: {cache.name}")
    with open(cache, "rb") as f:
        return pickle.load(f)


backtest.get_universe = lambda source="all": []  # prices come from the cache
backtest.load_price_data = load_pinned_cache
sys.argv = ["backtest.py", "--top-pct", args.top_pct]
backtest.main()
