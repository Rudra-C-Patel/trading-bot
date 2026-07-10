"""Morgan Tradez -- daily momentum scanner.

Ranks all US-listed common stocks (NASDAQ + NYSE + NYSE American, from
the official NASDAQ Trader symbol directory) by relative strength across
1/3/6-month lookbacks, keeps the top 2%, then checks each survivor for a
bull-flag setup per strategy.py. Run after the close (or pre-market) to
build the day's watchlist.

A liquidity gate (price > $5, 50-day avg dollar volume > $3.5M) is
applied on top of the strategy's own filters so the watchlist is
tradeable.

Usage:
    python scanner.py                    # full-market scan
    python scanner.py --universe sp500   # old narrow S&P 500 + NDX universe
    python scanner.py --top-pct 0.05     # loosen the RS cut for eyeballing
    python scanner.py --equity 25000     # size positions for your account
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

import strategy

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
UNIVERSE_CACHE_DAYS = 7
HISTORY_PERIOD = "2y"  # enough for the 200 SMA + 6-month RS with runway

# Liquidity gate for the tradeable universe (applied on top of the
# strategy's own price>$1 rule -- these are universe filters, not
# strategy rules, so they live here and not in strategy.py).
# Source spec: average daily DOLLAR volume > $3.5M (a share-count floor
# is the wrong metric: $2 x 1M sh = $2M/day should fail, $50 x 500K sh
# = $25M/day should pass).
MIN_UNIVERSE_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 3_500_000  # vs the 50-day avg of Close x Volume

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"
# Official NASDAQ Trader symbol directory (free, refreshed nightly):
# every NASDAQ-listed and every other-exchange-listed US security.
NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
_HEADERS = {"User-Agent": "Mozilla/5.0 (MorganTradez research scanner)"}

# Non-common-stock security types to exclude, matched against the name.
_EXCLUDE_NAME = (
    "warrant", "right", " unit", "units ", "preferred", "preference",
    "depositary", "notes", "debenture", "bond", " etn", "fund", "trust, inc. pfd",
)

# Minimal offline fallback so the scanner still runs if Wikipedia is
# unreachable and no cache exists. The live fetch replaces this ASAP.
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
    "NFLX", "CRM", "ORCL", "ADBE", "COST", "PEP", "LIN", "MRK", "ABBV",
    "LLY", "UNH", "JPM", "V", "MA", "HD", "PG", "XOM", "CVX", "KO",
    "PLTR", "SMCI", "PANW", "CRWD", "NOW", "UBER", "ABNB", "MELI", "SHOP",
    "DE", "CAT", "GE", "BA", "MMM", "HON", "UNP", "GS", "MS", "AXP",
    "ISRG", "REGN", "VRTX", "GILD", "AMGN", "TMO", "DHR", "SYK", "BSX",
    "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "INTC",
]


def _fetch_wikipedia_tickers() -> list[str]:
    """Scrape constituent tables from Wikipedia; raises on any failure."""
    tickers: set[str] = set()
    for url in (WIKI_SP500, WIKI_NDX):
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        for table in tables:
            for col in ("Symbol", "Ticker"):
                if col in table.columns:
                    vals = table[col].astype(str).str.strip()
                    # Yahoo uses '-' where filings use '.' (BRK.B -> BRK-B)
                    tickers.update(v.replace(".", "-") for v in vals if v and v != "nan")
                    break
    tickers = {t for t in tickers if t.isascii() and 0 < len(t) <= 6}
    if len(tickers) < 400:
        raise RuntimeError(f"only {len(tickers)} tickers scraped; refusing partial universe")
    return sorted(tickers)


def _fetch_broad_tickers() -> list[str]:
    """All US-listed common stocks from the NASDAQ Trader symbol directory.

    nasdaqlisted.txt = every NASDAQ security; otherlisted.txt = NYSE /
    NYSE American / Arca / BATS / IEX. Both are pipe-delimited with ETF
    and test-issue flags. We keep plain common stock on the three main
    stock exchanges and drop everything structured (ETFs, warrants,
    rights, units, preferreds, notes, funds).
    """
    tickers: set[str] = set()

    resp = requests.get(NASDAQ_LISTED, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    nq = pd.read_csv(io.StringIO(resp.text), sep="|")
    nq = nq[nq["Symbol"].notna() & (nq["Test Issue"] == "N") & (nq["ETF"] == "N")]
    for _, r in nq.iterrows():
        name = str(r["Security Name"]).lower()
        if any(x in name for x in _EXCLUDE_NAME):
            continue
        sym = str(r["Symbol"]).strip()
        if sym.isalpha():
            tickers.add(sym)

    resp = requests.get(OTHER_LISTED, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    ot = pd.read_csv(io.StringIO(resp.text), sep="|")
    # Exchange: N = NYSE, A = NYSE American (P/Z/V are ETF-heavy venues)
    ot = ot[ot["ACT Symbol"].notna() & (ot["Test Issue"] == "N")
            & (ot["ETF"] == "N") & (ot["Exchange"].isin(["N", "A"]))]
    for _, r in ot.iterrows():
        name = str(r["Security Name"]).lower()
        if any(x in name for x in _EXCLUDE_NAME):
            continue
        sym = str(r["ACT Symbol"]).strip()
        if "$" in sym:          # preferred-share convention
            continue
        sym = sym.replace(".", "-")  # class shares: BRK.B -> BRK-B for Yahoo
        if sym.replace("-", "").isalpha():
            tickers.add(sym)

    if len(tickers) < 2000:
        raise RuntimeError(f"only {len(tickers)} tickers parsed; refusing partial universe")
    return sorted(tickers)


def get_universe(refresh: bool = False, source: str = "all") -> list[str]:
    """Ticker universe, cached for a week per source, with fallbacks.

    source="all"   -- every US-listed common stock (default)
    source="sp500" -- the old S&P 500 + NASDAQ 100 universe
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "universe.json" if source == "sp500"
                         else "universe_all.json")
    if not refresh and os.path.exists(cache):
        try:
            with open(cache) as f:
                cached = json.load(f)
            fetched = datetime.fromisoformat(cached["fetched"])
            if datetime.now() - fetched < timedelta(days=UNIVERSE_CACHE_DAYS):
                return cached["tickers"]
        except Exception as e:
            print(f"[WARN] universe cache unreadable ({e}); refetching", file=sys.stderr)
    try:
        tickers = (_fetch_wikipedia_tickers() if source == "sp500"
                   else _fetch_broad_tickers())
        with open(cache, "w") as f:
            json.dump({"fetched": datetime.now().isoformat(), "source": source,
                       "tickers": tickers}, f)
        return tickers
    except Exception as e:
        print(f"[WARN] universe fetch failed ({e}); trying stale cache/fallback",
              file=sys.stderr)
        if os.path.exists(cache):
            try:
                with open(cache) as f:
                    return json.load(f)["tickers"]
            except Exception:
                pass
        return FALLBACK_TICKERS


def download_history(tickers: list[str], period: str = HISTORY_PERIOD,
                     batch_size: int = 100) -> dict[str, pd.DataFrame]:
    """Batched yfinance download -> {ticker: OHLCV frame}. Skips empty/broken.

    Yahoo silently rate-limits large downloads by returning empty batches,
    so a fully-empty batch triggers a backoff-and-retry, and tickers still
    missing at the end get two slower retry rounds. Without this, a daily
    full-market scan quietly ranks a gutted universe.
    """
    frames: dict[str, pd.DataFrame] = {}
    attempted_short: set[str] = set()

    def fetch(batch: list[str]) -> int:
        got = 0
        data = yf.download(batch, period=period, interval="1d", group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
        if data is None or data.empty:
            return 0
        for t in batch:
            try:
                df = data[t].dropna(subset=["Close"]) if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                continue
            if len(df):
                got += 1
            if len(df) >= 220:  # need 200 SMA + runway
                frames[t] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            elif len(df):
                attempted_short.add(t)  # real data, short history: resolved
        return got

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        got = fetch(batch)
        if got == 0 and len(batch) >= 20:
            print(f"  [WARN] empty batch at {start + 1} (rate-limited?); backing off 30s...",
                  file=sys.stderr)
            time.sleep(30)
            fetch(batch)
        time.sleep(1.5)

    for round_no in (1, 2):
        missing = [t for t in tickers if t not in frames and t not in attempted_short]
        if not missing:
            break
        print(f"  retry round {round_no}: {len(missing)} tickers empty...", file=sys.stderr)
        recovered = 0
        for b in range(0, len(missing), 50):
            recovered += fetch(missing[b:b + 50])
            time.sleep(3.0)
        if recovered == 0:
            break
    return frames


def rank_and_scan(frames: dict[str, pd.DataFrame], top_pct: float = 0.02,
                  require_breakout: bool = False) -> list[strategy.Setup]:
    """Cross-sectional RS ranking, then setup checks on the top slice."""
    enriched = {t: strategy.add_indicators(df) for t, df in frames.items()}

    scores = {}
    for t, df in enriched.items():
        row = df.iloc[-1]
        if not strategy.passes_universe_filter(row):
            continue
        # liquidity gate: tradeable price and real dollar volume
        if (row["Close"] < MIN_UNIVERSE_PRICE
                or pd.isna(row["dollar_vol50"])
                or row["dollar_vol50"] < MIN_AVG_DOLLAR_VOLUME):
            continue
        s = strategy.rs_raw_score(row)
        if not pd.isna(s):
            scores[t] = s
    if not scores:
        return []

    ranked = pd.Series(scores).rank(pct=True)  # percentile among today's survivors
    cutoff = 1.0 - top_pct
    leaders = ranked[ranked >= cutoff].sort_values(ascending=False)

    setups: list[strategy.Setup] = []
    for t in leaders.index:
        setup = strategy.find_setup(enriched[t], ticker=t)
        if setup is None:
            continue
        setup.rs_percentile = round(float(ranked[t]) * 100, 1)
        if require_breakout and not strategy.breakout_confirmed(
                enriched[t], len(enriched[t]) - 1, setup.trigger):
            continue
        setups.append(setup)
    return setups


def format_watchlist(setups: list[strategy.Setup], equity: float) -> str:
    if not setups:
        return "No setups today. The top-2% RS leaders are extended or not yet tight."
    lines = [
        f"{'TICKER':<8}{'RS%':>6}{'CLOSE':>10}{'ENTRY':>10}{'STOP':>10}"
        f"{'5R TGT':>10}{'SHARES':>8}{'POS $':>12}{'ADR%':>7}"
    ]
    for s in setups:
        shares = strategy.position_size(equity, s.trigger, s.stop)
        lines.append(
            f"{s.ticker:<8}{s.rs_percentile:>6.1f}{s.close:>10.2f}{s.trigger:>10.2f}"
            f"{s.stop:>10.2f}{s.target_5r:>10.2f}{shares:>8d}{shares * s.trigger:>12,.0f}"
            f"{s.adr_pct:>7.2f}"
        )
    lines.append("")
    lines.append("Entry = buy stop above the range high; only take it on 1.5x+ avg volume.")
    return "\n".join(lines)


def run_scan(top_pct: float = 0.02, equity: float = 100_000.0,
             refresh_universe: bool = False, source: str = "all") -> list[strategy.Setup]:
    universe = get_universe(refresh=refresh_universe, source=source)
    print(f"Universe: {len(universe)} tickers ({source}). "
          f"Downloading {HISTORY_PERIOD} of daily bars...")
    frames = download_history(universe)
    print(f"Got usable history for {len(frames)} tickers. Ranking...")
    setups = rank_and_scan(frames, top_pct=top_pct)
    print()
    print(f"=== Morgan Tradez watchlist -- {datetime.now():%Y-%m-%d %H:%M} ===")
    print(format_watchlist(setups, equity))
    return setups


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Daily RS + bull-flag scanner")
    p.add_argument("--top-pct", type=float, default=0.02,
                   help="RS slice to keep (default 0.02 = top 2%%)")
    p.add_argument("--equity", type=float, default=float(os.environ.get("ACCOUNT_SIZE", 100_000)),
                   help="account size for position sizing (or ACCOUNT_SIZE env var)")
    p.add_argument("--universe", choices=["all", "sp500"], default="all",
                   help="all US common stocks (default) or S&P 500 + NDX only")
    p.add_argument("--refresh-universe", action="store_true",
                   help="refetch the constituent lists even if cached")
    args = p.parse_args()
    run_scan(top_pct=args.top_pct, equity=args.equity,
             refresh_universe=args.refresh_universe, source=args.universe)
