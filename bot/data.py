"""Alpaca historical bars for the multi-instrument suite -- data API only.

Read-only market data (data.alpaca.markets); never touches trading
endpoints. Credentials come from the same .env alpaca_broker.py uses.

Feed choice (probed 2026-07-13 on this account): `sip` historical bars
work on the free tier and reach back to 2016-01 for stocks with full
consolidated-tape volume; `iex` only starts 2020-07 and carries ~2-3%
of real volume, which would poison any volume-confirmation rule. Crypto
bars start 2021-01 and have no feed distinction.

Stock intraday bars are filtered to regular trading hours (09:30-16:00
ET). "4-hour" stock bars are built here by resampling RTH 1-hour bars
into session-anchored buckets (09:30-13:30 and 13:30-16:00 ET) --
Alpaca's native 4Hour buckets are midnight-anchored and straddle the
pre-market, which makes indicator values depend on overnight noise.
"""

import os
import pickle
from datetime import date, datetime

import pandas as pd
import requests

from alpaca_broker import _load_env

DATA_URL = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")
STOCK_FEED = "sip"
TIMEOUT_S = 30
PAGE_LIMIT = 10000

RTH_START = "09:30"
RTH_END = "15:59"          # bar *start* times inside the session
ET = "US/Eastern"


def _headers():
    _load_env()
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise SystemExit("Alpaca credentials missing (ALPACA_API_KEY / "
                         "ALPACA_SECRET_KEY in env or .env)")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def _paged_bars(url: str, symbol_key: str | None, **params) -> list[dict]:
    """Collect bars across next_page_token pages. Stock endpoint returns
    {"bars": [...]}; crypto returns {"bars": {"BTC/USD": [...]}}."""
    headers = _headers()
    out: list[dict] = []
    while True:
        r = requests.get(url, headers=headers, params=params,
                         timeout=TIMEOUT_S)
        if r.status_code != 200:
            raise RuntimeError(f"Alpaca GET {url} -> {r.status_code}: "
                               f"{r.text[:200]}")
        data = r.json()
        bars = data.get("bars") or ([] if symbol_key is None else {})
        out.extend(bars.get(symbol_key, []) if symbol_key else bars)
        token = data.get("next_page_token")
        if not token:
            return out
        params["page_token"] = token


def _to_frame(bars: list[dict], tz: str) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(bars).rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df.index = pd.DatetimeIndex(pd.to_datetime(df["t"], utc=True)).tz_convert(tz)
    df.index.name = "time"
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def fetch_stock_bars(symbol: str, timeframe: str, start: str,
                     end: str | None = None) -> pd.DataFrame:
    bars = _paged_bars(f"{DATA_URL}/v2/stocks/{symbol}/bars", None,
                       timeframe=timeframe, start=start,
                       end=end or str(date.today()), feed=STOCK_FEED,
                       adjustment="split", limit=PAGE_LIMIT)
    return _to_frame(bars, ET)


def fetch_crypto_bars(symbol: str, timeframe: str, start: str,
                      end: str | None = None) -> pd.DataFrame:
    bars = _paged_bars(f"{DATA_URL}/v1beta3/crypto/us/bars", symbol,
                       symbols=symbol, timeframe=timeframe, start=start,
                       end=end or str(date.today()), limit=PAGE_LIMIT)
    return _to_frame(bars, "UTC")


def rth_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars whose start time falls inside regular trading hours."""
    return df.between_time(RTH_START, RTH_END)


def resample_4h_session(df_1h: pd.DataFrame) -> pd.DataFrame:
    """RTH 1-hour bars -> two session-anchored '4-hour' bars per day
    (09:30-13:30 and 13:30-16:00 ET; the second is 2.5h -- documented)."""
    df = rth_only(df_1h).copy()
    minutes = df.index.hour * 60 + df.index.minute - (9 * 60 + 30)
    bucket = (minutes // 240).astype(int)
    key = [f"{d}_{b}" for d, b in zip(df.index.date, bucket)]
    g = df.groupby(key, sort=False)
    out = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(),
        "Low": g["Low"].min(), "Close": g["Close"].last(),
        "Volume": g["Volume"].sum(),
    })
    out.index = g.apply(lambda x: x.index[0])
    return out.sort_index()


def load_bars(symbol: str, timeframe: str, start: str, asset: str,
              end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Cached fetch. Stock intraday bars come back RTH-filtered; the
    '4HourRTH' pseudo-timeframe fetches 1Hour and session-resamples."""
    end = end or str(date.today())
    tag = symbol.replace("/", "")
    cache = os.path.join(DATA_DIR, f"suite_bars_{tag}_{timeframe}_{start}_{end}.pkl")
    if os.path.exists(cache) and not refresh:
        with open(cache, "rb") as f:
            return pickle.load(f)
    if asset == "crypto":
        df = fetch_crypto_bars(symbol, timeframe, start, end)
    elif timeframe == "4HourRTH":
        df = resample_4h_session(fetch_stock_bars(symbol, "1Hour", start, end))
    else:
        df = rth_only(fetch_stock_bars(symbol, timeframe, start, end))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(df, f)
    return df
