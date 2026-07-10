"""Wheel strategy configuration -- every threshold in one place.

PAPER TRADING ONLY. wheel_bot.py hard-refuses any port not in
PAPER_PORTS and any account id that does not look like an IB paper
account. Going live requires deliberately editing this file AND
wheel_bot.py's guard -- that is intentional friction.

Mechanics come from the setup doc (21 EMA wick+close entry signal,
~0.30-delta cash-secured put at ~30 DTE, ~0.20-delta covered call after
assignment). Everything under "risk management" below is an ADDITION
the original doc did not include, sourced from standard wheel-strategy
practice.
"""

# ---------------------------------------------------------------------------
# IBKR connection (PAPER ONLY)
# ---------------------------------------------------------------------------
IB_HOST = "127.0.0.1"
IB_PORT = 7497                  # TWS paper port. IB Gateway paper = 4002.
IB_CLIENT_ID = 42
PAPER_PORTS = {7497, 4002}      # live ports (7496 TWS / 4001 Gateway) are refused
PAPER_ACCOUNT_PREFIX = "DU"     # IB paper accounts start with DU; anything else is refused

ACCOUNT_SIZE_FALLBACK = 100_000.0   # used when IB equity is unavailable (backtest, dry-run)

# ---------------------------------------------------------------------------
# Entry signal (from the setup doc)
# ---------------------------------------------------------------------------
EMA_PERIOD = 21                 # daily 21 EMA
TREND_SMA = 50                  # signal only counts in an uptrend: close > 50 SMA
# Signal: today's bar wicks below the 21 EMA (Low <= EMA) but closes at or
# above it (Close >= EMA), with the EMA itself rising -- a tested-and-held
# pullback, not a breakdown.

# ---------------------------------------------------------------------------
# Option selection (from the setup doc)
# ---------------------------------------------------------------------------
CSP_TARGET_DELTA = 0.30         # cash-secured put: sell ~0.30 delta
CC_TARGET_DELTA = 0.20          # covered call after assignment: sell ~0.20 delta
TARGET_DTE = 30                 # aim ~30 days to expiration
DTE_TOLERANCE = 12              # accept listed expiries within TARGET_DTE +/- this
CONTRACT_MULTIPLIER = 100

# ---------------------------------------------------------------------------
# Risk management (ADDITIONS -- not in the original doc)
# ---------------------------------------------------------------------------
# Earnings/catalyst avoidance: never open a CSP or CC whose expiration
# crosses a known earnings date; positions already open get rolled or
# closed before the event where possible.
AVOID_EARNINGS = True

# Concentration: no single ticker's assigned-stock exposure (strike x 100
# x contracts) may exceed this fraction of account equity, and capital is
# spread across at least MIN_CONCURRENT_TICKERS names before any name gets
# a second contract.
MAX_TICKER_EXPOSURE_PCT = 0.10
MIN_CONCURRENT_TICKERS = 3

# Option liquidity: skip any option whose bid-ask spread exceeds this
# fraction of the mid price.
MAX_SPREAD_PCT = 0.05

# Volatility ceiling (premium-trap guard): skip names whose 30-day
# realized volatility exceeds this (annualized). High premium on these
# names is compensation for gap risk, not free income.
MAX_REALIZED_VOL = 0.60

# Rolling: manage a short option once it is within ROLL_DTE days of
# expiry. A short put gets rolled down-and-out if spot is ITM or within
# PUT_ROLL_TRIGGER_PCT of the strike -- but only if the roll collects at
# least ROLL_MIN_CREDIT per share net; otherwise take assignment.
# A covered call about to be assigned is allowed to exercise when the
# strike is at or above cost basis (profitable exit); below basis, roll
# up-and-out for net credit if possible.
ROLL_DTE = 5
PUT_ROLL_TRIGGER_PCT = 0.02
ROLL_MIN_CREDIT = 0.05

# Reassessment trigger: assigned stock trading this far below cost basis
# is flagged for MANUAL REVIEW; the bot stops mechanically selling calls
# below basis on a flagged name.
REVIEW_DRAWDOWN_PCT = 0.15

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# Liquid, fundamentally established names with moderate share prices --
# a $100K account with a 10% per-ticker cap can only cash-secure strikes
# up to ~$100, which is itself a real small-account constraint of the
# wheel, so the list skews to quality names under that level.
WATCHLIST = [
    "KO", "PFE", "CSCO", "INTC", "BAC", "XOM",
    "MRK", "GILD", "VZ", "GM", "SCHW", "WMT",
]

# Meme / chronically high-IV names: rich premium that prices in gap risk
# ("premium traps"). Never wheeled, regardless of signal. The realized-vol
# ceiling above is the systematic version of this list.
OPTIONS_BLACKLIST = [
    "GME", "AMC", "MSTR", "COIN", "MARA", "RIOT",
    "SMCI", "DJT", "NVAX", "HOOD", "TSLA",
]

# ---------------------------------------------------------------------------
# Backtest approximation knobs (see README honesty section)
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.04           # flat r for Black-Scholes premium approximation
REALIZED_VOL_WINDOW = 30        # trading days used to estimate sigma
PREMIUM_HAIRCUT = 0.90          # collect only 90% of theoretical premium
                                # (crude stand-in for spread crossing + the
                                # fact that BS-with-realized-vol is not a
                                # market quote; see README honesty section)
