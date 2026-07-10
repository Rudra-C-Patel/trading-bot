"""Shared risk layer: the properties both bots rely on."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_manager import RiskManager  # noqa: E402

EQUITY = 100_000.0


def rm(tmp_path) -> RiskManager:
    return RiskManager(state_file=str(tmp_path / "risk_state.json"))


def test_ticker_exclusivity_across_strategies(tmp_path):
    """THE core guarantee: a symbol active in one strategy cannot be
    opened in the other simultaneously."""
    r = rm(tmp_path)
    ok, _ = r.reserve("momentum", "NVDA", 1_000, EQUITY)
    assert ok
    blocked, why = r.reserve("wheel", "NVDA", 1_000, EQUITY)
    assert not blocked
    assert "already active in 'momentum'" in why
    # and the same strategy cannot double-open it either
    blocked2, _ = r.reserve("momentum", "NVDA", 1_000, EQUITY)
    assert not blocked2
    # after release, the other strategy may take it
    r.release("momentum", "NVDA", realized_pnl=250.0)
    ok2, _ = r.reserve("wheel", "NVDA", 1_000, EQUITY)
    assert ok2


def test_combined_ceiling_not_two_stacked_budgets(tmp_path):
    """Both strategies draw from ONE ceiling (5% of equity = $5,000)."""
    r = rm(tmp_path)
    assert r.reserve("momentum", "AAA", 2_000, EQUITY)[0]
    assert r.reserve("wheel", "BBB", 2_500, EQUITY)[0]
    # 4,500 reserved; another 1,000 would cross 5,000 combined
    blocked, why = r.reserve("momentum", "CCC", 1_000, EQUITY)
    assert not blocked and "ceiling" in why
    # but 500 still fits
    assert r.reserve("wheel", "DDD", 500, EQUITY)[0]
    assert r.open_risk() == 5_000


def test_pnl_attribution_is_per_strategy(tmp_path):
    r = rm(tmp_path)
    r.reserve("momentum", "AAA", 1_000, EQUITY)
    r.reserve("wheel", "BBB", 1_000, EQUITY)
    r.release("momentum", "AAA", realized_pnl=-500.0)
    r.release("wheel", "BBB", realized_pnl=+300.0)
    state = r._load()
    assert state["pnl"]["momentum"] == -500.0
    assert state["pnl"]["wheel"] == 300.0
    report = r.report()
    assert "momentum" in report and "wheel" in report


def test_cross_release_refused(tmp_path):
    """One strategy cannot release (and free up) the other's ticker."""
    r = rm(tmp_path)
    r.reserve("momentum", "AAA", 1_000, EQUITY)
    ok, why = r.release("wheel", "AAA", realized_pnl=999.0)
    assert not ok and "refusing cross-release" in why
    assert r.holder_of("AAA") == "momentum"          # still held
    assert r._load()["pnl"]["wheel"] == 0.0          # nothing booked


def test_release_without_reservation_still_books_pnl(tmp_path):
    """An exit must never lose its P&L even if the reservation is gone."""
    r = rm(tmp_path)
    ok, why = r.release("momentum", "GHOST", realized_pnl=-123.0)
    assert ok and "booked anyway" in why
    assert r._load()["pnl"]["momentum"] == -123.0


def test_can_reserve_is_read_only(tmp_path):
    r = rm(tmp_path)
    ok, _ = r.can_reserve("wheel", "AAA", 1_000, EQUITY)
    assert ok
    assert r.holder_of("AAA") is None                # nothing was recorded
    assert r.open_risk() == 0.0


def test_degenerate_inputs_rejected(tmp_path):
    r = rm(tmp_path)
    assert not r.reserve("momentum", "AAA", 0, EQUITY)[0]
    assert not r.reserve("momentum", "AAA", 1_000, 0)[0]
    assert not r.reserve("hedge_fund", "AAA", 1_000, EQUITY)[0]
