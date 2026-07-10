"""Morgan Tradez -- Telegram morning scan alerts.

Runs the daily scanner and pushes the watchlist (ticker, entry, stop, 5R
target, position size) to a Telegram chat.

Setup:
  1. Create a bot with @BotFather, copy the token.
  2. Message the bot once, then read your chat id from
     https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Export env vars (TRADEZ_* preferred so they never collide with other
     projects on this machine):
         export TRADEZ_TELEGRAM_BOT_TOKEN="123456:ABC..."
         export TRADEZ_TELEGRAM_CHAT_ID="123456789"
         export ACCOUNT_SIZE="100000"        # optional, for sizing

Usage:
    python telegram_alerts.py            # scan + send
    python telegram_alerts.py --dry-run  # scan + print, no send

Cron (07:30 US/Eastern, weekdays):
    30 7 * * 1-5  cd /path/to/trading-bot && .venv/bin/python telegram_alerts.py
"""

import argparse
import os
import sys
from datetime import datetime

import requests

import strategy
from scanner import run_scan

BOT_TOKEN = (os.environ.get("TRADEZ_TELEGRAM_BOT_TOKEN")
             or os.environ.get("TELEGRAM_BOT_TOKEN", ""))
CHAT_ID = (os.environ.get("TRADEZ_TELEGRAM_CHAT_ID")
           or os.environ.get("TELEGRAM_CHAT_ID", ""))
EQUITY = float(os.environ.get("ACCOUNT_SIZE", 100_000))


def send_telegram(text: str) -> bool:
    """Send one message; returns True on success. Never raises."""
    if not BOT_TOKEN or not CHAT_ID:
        print("[ERR] TRADEZ_TELEGRAM_BOT_TOKEN / TRADEZ_TELEGRAM_CHAT_ID not set",
              file=sys.stderr)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=20,
        )
        if r.status_code != 200:
            print(f"[ERR] Telegram API {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return False
        return True
    except requests.RequestException as e:
        print(f"[ERR] Telegram send failed: {e}", file=sys.stderr)
        return False


def format_alert(setups, equity: float) -> str:
    """HTML message: one block per candidate with all the numbers to act on."""
    date = datetime.now().strftime("%a %b %d, %Y")
    if not setups:
        return (f"<b>Morgan Tradez</b> - {date}\n\n"
                "No qualifying setups this morning. Leaders are extended or "
                "still consolidating. Sit on hands.")
    lines = [f"<b>Morgan Tradez</b> - {date}",
             f"{len(setups)} setup(s) in the top-2% RS leaders:\n"]
    for s in setups:
        shares = strategy.position_size(equity, s.trigger, s.stop)
        risk = shares * s.risk_per_share
        lines += [
            f"<b>{s.ticker}</b>  (RS {s.rs_percentile:.0f}th pct, ADR {s.adr_pct:.1f}%)",
            f"  Entry (buy stop): ${s.trigger:,.2f}",
            f"  Stop: ${s.stop:,.2f} ({(1 - s.stop / s.trigger):.1%} below)",
            f"  5R target: ${s.target_5r:,.2f}",
            f"  Size: {shares} sh = ${shares * s.trigger:,.0f} "
            f"({shares * s.trigger / equity:.0%} of acct, risk ${risk:,.0f})",
            "",
        ]
    lines += ["Rules: enter only on a volume breakout (1.5x+ 50d avg). "
              "Hold to 5R minimum; partial there; trail the 20 SMA. Paper only."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Morning scan -> Telegram alert")
    p.add_argument("--dry-run", action="store_true", help="print instead of sending")
    p.add_argument("--top-pct", type=float, default=0.02)
    args = p.parse_args()

    setups = run_scan(top_pct=args.top_pct, equity=EQUITY)
    msg = format_alert(setups, EQUITY)
    if args.dry_run:
        print("\n--- message preview ---\n" + msg)
        return
    ok = send_telegram(msg)
    print("Alert sent." if ok else "Alert NOT sent (see error above).")


if __name__ == "__main__":
    main()
