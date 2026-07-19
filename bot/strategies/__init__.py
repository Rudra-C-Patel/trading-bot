"""Strategy registry for the multi-instrument suite.

Every strategy module exposes the same pure-logic interface (no network,
no broker -- same layout discipline as wheel_bot's pure section):

    NAME            risk_manager strategy label
    STOP_ATR        hard-stop distance in entry-bar ATRs; NEVER moved
    TRAIL_ATR       trailing-stop ATR multiple (None = no trail)
    GRID            pre-registered walk-forward parameter grid
    default_params(symbol) -> dict     the spec parameters
    prepare(df, params) -> df          adds indicator columns (trailing only)
    entry(df, i, params) -> "long" | "short" | None    decided at bar i close
    exit(df, i, params, side) -> bool                  strategy exit at close

Decisions are made on bar closes and filled at the NEXT bar open by the
engine; stops are handled intra-bar by the engine. All indicators are
trailing, so prepare() on full history introduces no lookahead when a
window is sliced afterwards (same precompute pattern as walkforward.py).
"""

from bot.strategies import mean_reversion, momentum_breakout, trend_following

# Instrument -> strategy wiring, timeframes and per-side trading costs.
# Cost notes (pre-registered, per side, applied to fill price):
#   stocks: 0.02% -- SPY/QQQ/GLD spreads are ~1bp, plus slippage;
#           commission-free at Alpaca. USO is wider; same figure used,
#           disclosed as flattering to USO.
#   crypto: 0.30% -- Alpaca crypto taker fee (25bp) + 5bp slippage.
INSTRUMENTS = {
    "SPY":     dict(strategy=mean_reversion,    timeframe="15Min",
                    asset="stock",  cost=0.0002, history_start="2016-01-01"),
    "QQQ":     dict(strategy=mean_reversion,    timeframe="15Min",
                    asset="stock",  cost=0.0002, history_start="2016-01-01"),
    "BTC/USD": dict(strategy=momentum_breakout, timeframe="1Hour",
                    asset="crypto", cost=0.0030, history_start="2021-01-01"),
    "GLD":     dict(strategy=trend_following,   timeframe="4HourRTH",
                    asset="stock",  cost=0.0002, history_start="2016-01-01"),
    "USO":     dict(strategy=trend_following,   timeframe="4HourRTH",
                    asset="stock",  cost=0.0002, history_start="2016-01-01"),
}
