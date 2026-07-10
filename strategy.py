"""Morgan Tradez -- swing trading strategy rules.

Qullamaggie-style momentum breakout:
  1. Universe filter: price > $1, ADR% > 5, above 50 & 200 SMA.
  2. Momentum: top 2% relative strength across 1m / 3m / 6m lookbacks.
  3. Setup: 30%+ prior move over weeks (not a one-day pump), then an
     orderly multi-day pullback to a rising 10/20 SMA with tightening
     range and drying volume (bull flag).
  4. Trigger: breakout above the consolidation high on strong volume.
  5. Risk: stop 2-5% below entry, position sized so risk <= 1% of the
     account, position value 10-40% of the account.
  6. Exit: hold until 5x risk minimum, take a partial there, trail the
     rest on the 20 SMA with the stop moved to breakeven.

All functions are pure and operate on a single ticker's daily OHLCV
DataFrame (columns: Open, High, Low, Close, Volume) so the scanner,
backtester and paper trader share one implementation.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ta.trend import sma_indicator
from ta.volatility import AverageTrueRange

# ---------------------------------------------------------------------------
# Tunables (defaults follow the written strategy rules)
# ---------------------------------------------------------------------------
MIN_PRICE = 1.0                # ignore sub-$1 stocks
MIN_ADR_PCT = 5.0              # average daily range % floor (source spec: ADR% > 5)
MIN_PRIOR_MOVE = 0.30          # 30%+ move up before the flag
PRIOR_MOVE_LOOKBACK = 90       # bars to search for the low of the move
HIGH_LOOKBACK = 60             # bars to search for the swing high
MIN_MOVE_DAYS = 5              # low -> high must span >= this many bars (no one-day pumps)
MAX_ONE_DAY_SHARE = 0.60       # biggest single up-day may be at most 60% of the move
MIN_PULLBACK_DAYS = 3          # orderly pullback lasts days...
MAX_PULLBACK_DAYS = 25         # ...to a few weeks
MAX_PULLBACK_DEPTH = 0.25      # give back at most 25% from the swing high
SMA_TOUCH_TOLERANCE = 1.04     # pullback low within 4% above the 10/20 SMA counts as a touch
TIGHT_RANGE_RATIO = 0.80       # last-3-day avg range must shrink below 80% of ADR
VOLUME_DRYUP_RATIO = 1.00      # 5-day avg volume below the 50-day average
BREAKOUT_VOLUME_MULT = 1.5     # breakout day needs volume >= 1.5x the 50-day average
TRIGGER_WINDOW = 5             # consolidation high = highest high of the last N bars
TRIGGER_PAD = 0.001            # buy 0.1% above the range high

STOP_MIN_PCT = 0.02            # stop at least 2% below entry
STOP_MAX_PCT = 0.05            # and never more than 5% below entry
RISK_PER_TRADE = 0.01          # never risk more than 1% of the account
MIN_POSITION_PCT = 0.10        # position must be worth taking (10% of account)...
MAX_POSITION_PCT = 0.25        # ...but capped (rules allow 10-40%; default conservative)

PARTIAL_R_MULTIPLE = 5.0       # never sell before 5x risk
PARTIAL_SELL_FRACTION = 0.25   # take 10-30% off at 5R (default 25%)
TRAIL_SMA = 20                 # after the partial, trail a close below this SMA

RS_WINDOWS = (21, 63, 126)     # ~1 month, 3 months, 6 months of trading days
RS_WEIGHTS = (0.4, 0.35, 0.25) # favor recent strength slightly


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the OHLCV frame with every column the rules need."""
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]

    for n in (10, 20, 50, 200):
        out[f"sma{n}"] = sma_indicator(close, window=n)

    # Average Daily Range % -- 20-day mean of (High/Low - 1) * 100
    out["adr_pct"] = ((high / low) - 1.0).rolling(20).mean() * 100.0

    out["atr14"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    out["vol50"] = out["Volume"].rolling(50).mean()
    # average daily DOLLAR volume -- the liquidity metric the source spec
    # actually uses (share count alone lets $2 stocks through)
    out["dollar_vol50"] = (close * out["Volume"]).rolling(50).mean()
    out["range_pct"] = (high - low) / close * 100.0

    for n in RS_WINDOWS:
        out[f"ret{n}"] = close.pct_change(n)
    return out


def rs_raw_score(row: pd.Series) -> float:
    """Weighted momentum score for one ticker on one day (higher = stronger).

    The scanner converts these to cross-sectional percentile ranks; the raw
    score only needs to order tickers consistently.
    """
    score, total_w = 0.0, 0.0
    for n, w in zip(RS_WINDOWS, RS_WEIGHTS):
        r = row.get(f"ret{n}")
        if r is not None and not pd.isna(r):
            score += w * r
            total_w += w
    return score / total_w if total_w else float("nan")


# ---------------------------------------------------------------------------
# Setup detection
# ---------------------------------------------------------------------------

@dataclass
class Setup:
    """A stock that passed every entry rule as of `date`, awaiting breakout."""
    ticker: str
    date: pd.Timestamp          # bar the setup was detected on (enter after this)
    close: float
    trigger: float              # buy stop: consolidation high + pad
    stop: float                 # initial protective stop
    adr_pct: float
    prior_move: float           # size of the run-up that built the flag
    pullback_days: int
    rs_percentile: float = field(default=float("nan"))

    @property
    def risk_per_share(self) -> float:
        return self.trigger - self.stop

    @property
    def target_5r(self) -> float:
        return self.trigger + PARTIAL_R_MULTIPLE * self.risk_per_share


def passes_universe_filter(row: pd.Series) -> bool:
    """Price > $1, ADR% > 5, above both the 50 and 200 SMA."""
    needed = ("Close", "adr_pct", "sma50", "sma200")
    if any(pd.isna(row.get(k)) for k in needed):
        return False
    return (
        row["Close"] > MIN_PRICE
        and row["adr_pct"] > MIN_ADR_PCT
        and row["Close"] > row["sma50"]
        and row["Close"] > row["sma200"]
    )


def find_setup(df: pd.DataFrame, i: int | None = None, ticker: str = "") -> Setup | None:
    """Check every entry rule at bar position `i` (default: last bar).

    `df` must already have indicators (see add_indicators). Only data up to
    and including bar `i` is used, so the same code is safe for backtesting.
    Returns a Setup (with trigger + stop) or None.
    """
    if i is None:
        i = len(df) - 1
    if i < 210:  # need the 200 SMA plus some runway
        return None

    row = df.iloc[i]
    if not passes_universe_filter(row):
        return None

    # --- rising 10 & 20 SMAs -------------------------------------------------
    sma10_now, sma10_then = df["sma10"].iloc[i], df["sma10"].iloc[i - 5]
    sma20_now, sma20_then = df["sma20"].iloc[i], df["sma20"].iloc[i - 5]
    if pd.isna(sma10_then) or pd.isna(sma20_then):
        return None
    if not (sma10_now > sma10_then and sma20_now > sma20_then):
        return None

    # --- 30%+ prior move over weeks, not a one-day pump ----------------------
    win_hi = df["High"].iloc[i - HIGH_LOOKBACK + 1: i + 1]
    hi_pos = int(np.argmax(win_hi.values))
    idx_high = i - HIGH_LOOKBACK + 1 + hi_pos
    swing_high = float(df["High"].iloc[idx_high])

    lo_start = max(0, idx_high - PRIOR_MOVE_LOOKBACK)
    if idx_high <= lo_start:
        return None
    win_lo = df["Low"].iloc[lo_start:idx_high]
    lo_pos = int(np.argmin(win_lo.values))
    idx_low = lo_start + lo_pos
    swing_low = float(df["Low"].iloc[idx_low])
    if swing_low <= 0:
        return None

    prior_move = swing_high / swing_low - 1.0
    if prior_move < MIN_PRIOR_MOVE:
        return None
    if idx_high - idx_low < MIN_MOVE_DAYS:
        return None
    # reject pump-style moves: one close-to-close jump doing most of the work
    run_rets = df["Close"].iloc[idx_low: idx_high + 1].pct_change().dropna()
    if len(run_rets) and run_rets.max() > MAX_ONE_DAY_SHARE * prior_move:
        return None

    # --- orderly pullback to the 10/20 SMA -----------------------------------
    pullback_days = i - idx_high
    if not (MIN_PULLBACK_DAYS <= pullback_days <= MAX_PULLBACK_DAYS):
        return None

    pull_low = float(df["Low"].iloc[idx_high: i + 1].min())
    depth = (swing_high - pull_low) / swing_high
    if depth > MAX_PULLBACK_DEPTH:
        return None
    # the low must come into the rising 10 or 20 SMA (within tolerance)...
    if pull_low > max(sma10_now, sma20_now) * SMA_TOUCH_TOLERANCE:
        return None
    # ...but the stock must be holding it, not knifing through
    if row["Close"] < sma20_now * 0.97:
        return None

    # --- tightening range ----------------------------------------------------
    recent_range = df["range_pct"].iloc[i - 2: i + 1].mean()
    if pd.isna(recent_range) or recent_range > TIGHT_RANGE_RATIO * row["adr_pct"]:
        return None

    # --- volume dry-up -------------------------------------------------------
    vol5 = df["Volume"].iloc[i - 4: i + 1].mean()
    if pd.isna(row["vol50"]) or row["vol50"] <= 0 or vol5 >= VOLUME_DRYUP_RATIO * row["vol50"]:
        return None

    # --- trigger & stop ------------------------------------------------------
    range_high = float(df["High"].iloc[i - TRIGGER_WINDOW + 1: i + 1].max())
    trigger = range_high * (1.0 + TRIGGER_PAD)
    stop = compute_stop(trigger, pull_low)

    return Setup(
        ticker=ticker,
        date=df.index[i],
        close=float(row["Close"]),
        trigger=trigger,
        stop=stop,
        adr_pct=float(row["adr_pct"]),
        prior_move=prior_move,
        pullback_days=pullback_days,
    )


def breakout_confirmed(df: pd.DataFrame, i: int, trigger: float) -> bool:
    """True if bar `i` broke the trigger on breakout-grade volume."""
    row = df.iloc[i]
    if pd.isna(row["vol50"]) or row["vol50"] <= 0:
        return False
    return row["High"] >= trigger and row["Volume"] >= BREAKOUT_VOLUME_MULT * row["vol50"]


# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------

def compute_stop(entry: float, structural_low: float) -> float:
    """Stop below the pullback low, clamped to 2-5% under the entry."""
    if entry <= 0:
        return 0.0
    pct = (entry - structural_low) / entry
    pct = min(max(pct, STOP_MIN_PCT), STOP_MAX_PCT)
    return entry * (1.0 - pct)


def position_size(equity: float, entry: float, stop: float,
                  risk_pct: float = RISK_PER_TRADE,
                  min_pos_pct: float = MIN_POSITION_PCT,
                  max_pos_pct: float = MAX_POSITION_PCT) -> int:
    """Shares = risk dollars / (entry - stop), capped by max position value.

    Returns 0 when the trade cannot satisfy the rules (position would be
    below the minimum size after capping, or inputs are degenerate).
    """
    risk_per_share = entry - stop
    if equity <= 0 or entry <= 0 or risk_per_share <= 0:
        return 0
    shares = int((equity * risk_pct) // risk_per_share)
    max_shares = int((equity * max_pos_pct) // entry)
    shares = min(shares, max_shares)
    if shares <= 0 or shares * entry < equity * min_pos_pct:
        # After capping, a position under the 10% floor isn't worth carrying.
        return 0
    return shares
