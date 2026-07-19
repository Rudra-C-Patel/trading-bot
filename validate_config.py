"""Pre-flight validation for the wheel bot -- run before any session
touches this bot. READ-ONLY: the only network call is GET /v2/account
(via AlpacaBroker.connect()); no orders are ever placed.

Checks three layers and prints a PASS/FAIL summary (exit 0/1):

1. config.py internal consistency -- every threshold sane on its own and
   compatible with the others (e.g. MAX_DELTA_DISTANCE cannot make
   CSP_TARGET_DELTA impossible to hit, ROLL_DTE cannot overlap the entry
   DTE window).
2. Paper-only guard constants -- the values the runtime guards compare
   against have not drifted (live ports absent, DU/PA prefixes intact).
3. Alpaca paper connection -- credentials resolve, host guard passes,
   account is ACTIVE with options_trading_level >= 1. Skipped with
   --offline (config checks still run).

Usage:
    python validate_config.py             # full pre-flight
    python validate_config.py --offline   # config checks only, no network
"""

import argparse
import sys

import config
import risk_manager

RESULTS: list[tuple[bool, str, str]] = []   # (ok, name, detail)


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((bool(ok), name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def check_config():
    print("Config consistency (config.py / risk_manager.py):")

    check("BROKER is a known backend",
          config.BROKER in ("ibkr", "alpaca"),
          f"BROKER={config.BROKER!r}")

    # -- delta targeting: the band around each target must be reachable ----
    for label, target in (("CSP_TARGET_DELTA", config.CSP_TARGET_DELTA),
                          ("CC_TARGET_DELTA", config.CC_TARGET_DELTA)):
        check(f"{label} in (0, 1)", 0 < target < 1, f"{target}")
        lo = target - config.MAX_DELTA_DISTANCE
        hi = target + config.MAX_DELTA_DISTANCE
        check(f"{label} band is hittable",
              config.MAX_DELTA_DISTANCE > 0 and hi < 1,
              f"accepts |delta| in [{max(lo, 0):.2f}, {hi:.2f}] "
              f"(MAX_DELTA_DISTANCE={config.MAX_DELTA_DISTANCE})")
        check(f"{label} band excludes 0-delta junk", lo > 0,
              f"band floor {lo:.2f}")

    # -- DTE geometry ------------------------------------------------------
    check("TARGET_DTE positive", config.TARGET_DTE > 0, f"{config.TARGET_DTE}")
    check("DTE_TOLERANCE within [0, TARGET_DTE)",
          0 <= config.DTE_TOLERANCE < config.TARGET_DTE,
          f"{config.DTE_TOLERANCE}")
    check("new positions open outside the roll-management zone",
          config.TARGET_DTE - config.DTE_TOLERANCE > config.ROLL_DTE,
          f"earliest accepted expiry {config.TARGET_DTE - config.DTE_TOLERANCE} "
          f"DTE > ROLL_DTE {config.ROLL_DTE}")

    # -- rolls -------------------------------------------------------------
    check("ROLL_MIN_CREDIT positive", config.ROLL_MIN_CREDIT > 0,
          f"{config.ROLL_MIN_CREDIT}")
    check("PUT_ROLL_TRIGGER_PCT non-negative",
          config.PUT_ROLL_TRIGGER_PCT >= 0, f"{config.PUT_ROLL_TRIGGER_PCT}")

    # -- risk fractions ----------------------------------------------------
    check("MAX_TICKER_EXPOSURE_PCT in (0, 1]",
          0 < config.MAX_TICKER_EXPOSURE_PCT <= 1,
          f"{config.MAX_TICKER_EXPOSURE_PCT}")
    check("MIN_CONCURRENT_TICKERS >= 1",
          config.MIN_CONCURRENT_TICKERS >= 1,
          f"{config.MIN_CONCURRENT_TICKERS}")
    check("MAX_SPREAD_PCT in (0, 1)", 0 < config.MAX_SPREAD_PCT < 1,
          f"{config.MAX_SPREAD_PCT}")
    check("MAX_REALIZED_VOL positive", config.MAX_REALIZED_VOL > 0,
          f"{config.MAX_REALIZED_VOL}")
    check("REVIEW_DRAWDOWN_PCT in (0, 1)",
          0 < config.REVIEW_DRAWDOWN_PCT < 1, f"{config.REVIEW_DRAWDOWN_PCT}")
    check("CONTRACT_MULTIPLIER is 100 (US equity options)",
          config.CONTRACT_MULTIPLIER == 100, f"{config.CONTRACT_MULTIPLIER}")
    check("ACCOUNT_SIZE_FALLBACK positive",
          config.ACCOUNT_SIZE_FALLBACK > 0, f"{config.ACCOUNT_SIZE_FALLBACK}")

    # A single max-size wheel position (risk = REVIEW_DRAWDOWN_PCT x
    # collateral at the per-ticker cap) must fit under the shared ceiling,
    # or the risk layer silently blocks every trade the sizing rules allow.
    per_pos = config.REVIEW_DRAWDOWN_PCT * config.MAX_TICKER_EXPOSURE_PCT
    check("one max-size wheel position fits under the shared risk ceiling",
          per_pos <= risk_manager.MAX_TOTAL_OPEN_RISK_PCT,
          f"per-position {per_pos:.1%} of equity <= ceiling "
          f"{risk_manager.MAX_TOTAL_OPEN_RISK_PCT:.0%}")
    check("'wheel' registered in risk layer STRATEGIES",
          "wheel" in risk_manager.STRATEGIES)

    # -- universe ----------------------------------------------------------
    check("WATCHLIST non-empty", len(config.WATCHLIST) > 0,
          f"{len(config.WATCHLIST)} names")
    check("WATCHLIST has no duplicates",
          len(config.WATCHLIST) == len(set(config.WATCHLIST)))
    overlap = set(config.WATCHLIST) & set(config.OPTIONS_BLACKLIST)
    check("WATCHLIST disjoint from OPTIONS_BLACKLIST", not overlap,
          f"overlap: {sorted(overlap)}" if overlap else "no overlap")


def check_paper_guards():
    print("Paper-only guard constants:")
    live_ports = {7496, 4001}
    check("IB_PORT is a paper port", config.IB_PORT in config.PAPER_PORTS,
          f"{config.IB_PORT} in {sorted(config.PAPER_PORTS)}")
    check("no live IB ports in PAPER_PORTS",
          not (set(config.PAPER_PORTS) & live_ports),
          f"PAPER_PORTS={sorted(config.PAPER_PORTS)}, live={sorted(live_ports)}")
    check("IB paper account prefix is 'DU'",
          config.PAPER_ACCOUNT_PREFIX == "DU",
          f"{config.PAPER_ACCOUNT_PREFIX!r}")
    check("IB_HOST is localhost", config.IB_HOST in ("127.0.0.1", "localhost"),
          f"{config.IB_HOST!r}")

    import alpaca_broker
    check("Alpaca guard host is the paper API",
          alpaca_broker.PAPER_HOST == "paper-api.alpaca.markets",
          f"{alpaca_broker.PAPER_HOST!r}")
    check("Alpaca paper account prefix is 'PA'",
          alpaca_broker.PAPER_ACCOUNT_PREFIX == "PA",
          f"{alpaca_broker.PAPER_ACCOUNT_PREFIX!r}")


def check_alpaca_connection():
    """Read-only: AlpacaBroker.__init__ (host guard) + connect() (GET
    /v2/account, PA-prefix + status + options-level guards). No orders."""
    print("Alpaca paper connection (read-only):")
    from alpaca_broker import AlpacaBroker
    try:
        b = AlpacaBroker().connect()
    except SystemExit as e:      # the broker guards refuse via sys.exit
        check("Alpaca connect + paper guards", False, str(e))
        return
    except Exception as e:
        check("Alpaca connect + paper guards", False, f"{type(e).__name__}: {e}")
        return
    try:
        a = b.account
        check("connected to paper account", True,
              f"{a['account_number']} ({a['status']})")
        check("account number has PA prefix",
              a["account_number"].startswith("PA"), a["account_number"])
        level = int(a.get("options_trading_level") or 0)
        check("options_trading_level >= 1 (CSP/CC entitlement)", level >= 1,
              f"level {level}")
        check("trading not blocked",
              not a.get("trading_blocked") and not a.get("account_blocked"))
        equity = float(a["equity"])
        check("equity is positive", equity > 0, f"${equity:,.2f}")
    finally:
        b.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true",
                    help="skip the Alpaca connection check (no network)")
    args = ap.parse_args()

    check_config()
    check_paper_guards()
    if args.offline:
        print("Alpaca connection: SKIPPED (--offline)")
    else:
        check_alpaca_connection()

    failed = [(n, d) for ok, n, d in RESULTS if not ok]
    print(f"\n{'=' * 60}")
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("OVERALL: FAIL")
        for n, d in failed:
            print(f"  FAIL: {n}" + (f" -- {d}" if d else ""))
        sys.exit(1)
    print("OVERALL: PASS -- config consistent, paper guards intact"
          + ("" if args.offline else ", Alpaca paper reachable"))
    sys.exit(0)


if __name__ == "__main__":
    main()
