"""One-off cache repair: re-fetch tickers Yahoo throttled out of the
price cache, looping until no new data comes back. Tickers that return
bars but fewer than 250 are recorded as resolved-short so they are not
retried forever. Usage:

    python repair_cache.py data/prices_2020-01-01_2024-12-31_5309.pkl \
        2020-01-01 2024-12-31
"""

import pickle
import sys
import time

import pandas as pd
import yfinance as yf

from scanner import get_universe

WARMUP_DAYS = 320


def main():
    cache_file, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    dl_start = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    with open(cache_file, "rb") as f:
        frames = pickle.load(f)
    universe = get_universe()
    short: set[str] = set()

    for round_no in range(1, 15):
        missing = [t for t in universe if t not in frames and t not in short]
        if not missing:
            break
        print(f"round {round_no}: {len(missing)} tickers to try")
        stored = shorted = 0
        for b in range(0, len(missing), 50):
            batch = missing[b:b + 50]
            data = yf.download(batch, start=dl_start, end=end, interval="1d",
                               group_by="ticker", auto_adjust=True,
                               threads=True, progress=False)
            if data is not None and not data.empty:
                for t in batch:
                    try:
                        df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                    except KeyError:
                        continue
                    df = df.dropna(subset=["Close"])
                    if len(df) >= 250:
                        frames[t] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                        stored += 1
                    elif len(df) > 0:
                        short.add(t)  # real data, too little history: resolved
                        shorted += 1
            time.sleep(2.0)
        print(f"  stored {stored}, resolved-short {shorted}")
        with open(cache_file, "wb") as f:
            pickle.dump(frames, f)
        if stored == 0 and shorted == 0:
            print("  nothing new came back; stopping")
            break
        time.sleep(10)

    remaining = [t for t in universe if t not in frames and t not in short]
    print(f"done: {len(frames)} cached, {len(short)} short-history, "
          f"{len(remaining)} permanently empty (delisted/renamed on Yahoo)")


if __name__ == "__main__":
    main()
