"""Circuit breakers, the pullback style, and the stale-data / bad-account guards."""
import json
import random
from datetime import datetime, timezone

import pytest

from core.fx_trend_broker import Position
from strategy.fx_risk import (
    RiskLimits, in_loss_pause, lock_reason, loss_streak, risk_multiplier, streak_has_room,
)
from strategy.fx_trend import TrendParams, compute_indicators
from strategy.fx_trend_backtest import BtConfig, run_backtest
from tests.test_fx_trend_runner import NOW, FakeBroker, bars, runner

LIMITS = RiskLimits(max_consecutive_losses=7, pause_after_losses=4, pause_days=5, max_drawdown_halt_pct=8.0)
BREAKERS = dict(max_consecutive_losses=7, pause_after_losses=4, max_drawdown_halt_pct=8.0)


def losses(n, start_ticket=1, day="2026-02-01"):
    return [{"ticket": start_ticket + i, "pair": "GBPUSD", "side": 1, "volume": 0.01,
             "open_time": f"{day}T00:00:00+00:00", "close_time": f"{day}T01:{i:02d}:00+00:00",
             "entry": 1.2, "exit": 1.19, "profit": -10.0, "commission": 0.0, "swap": 0.0}
            for i in range(n)]


# ---- pure rules -----------------------------------------------------------
def test_loss_streak_counts_only_the_tail():
    assert loss_streak([-1, -1, 2, -1, -1, -1]) == 3
    assert loss_streak([-1, 0]) == 0 and loss_streak([]) == 0


def test_lock_pause_and_reduced_risk():
    assert lock_reason(6, 0.0, LIMITS) is None and lock_reason(7, 0.0, LIMITS)
    assert lock_reason(0, -7.9, LIMITS) is None and lock_reason(0, -8.0, LIMITS)
    assert in_loss_pause(4, datetime(2026, 3, 1), datetime(2026, 3, 4), LIMITS)
    assert not in_loss_pause(4, datetime(2026, 3, 1), datetime(2026, 3, 6), LIMITS)
    assert risk_multiplier(3, LIMITS) == 1.0 and risk_multiplier(4, LIMITS) == 0.5


def test_streak_can_never_pass_the_limit_even_if_everything_loses():
    """Random mix of opening and closing, worst case biased: the realised streak stays <= 7."""
    rng = random.Random(3)
    for _ in range(500):
        streak = open_count = 0
        for _ in range(300):
            if streak >= LIMITS.max_consecutive_losses:
                break                                   # locked: nothing new is opened
            if rng.random() < 0.6 and streak_has_room(streak, open_count, LIMITS):
                open_count += 1
            elif open_count:
                open_count -= 1
                streak = streak + 1 if rng.random() < 0.9 else 0
            assert streak + open_count <= 7
        assert streak + open_count <= 7                 # still true if all the rest lose


# ---- live runner ------------------------------------------------------------
def test_runner_locks_after_seven_losses_and_stays_locked_until_reset(tmp_path):
    b = FakeBroker({"EURUSD": bars()})
    b.history = losses(7)
    rn = runner(tmp_path, b, **BREAKERS)
    rn.run_once(NOW)
    assert not b.opened
    b.history += losses(1, 100, "2026-02-02") + [{**losses(1, 200, "2026-02-03")[0], "profit": 50.0}]
    b.data["EURUSD"] = bars(n=301, end="2026-03-04")
    rn.run_once(datetime(2026, 3, 5, tzinfo=timezone.utc))
    assert not b.opened                                  # a later win does NOT unlock it

    (tmp_path / "data/fx_trend.RESET_BREAKER").write_text("x")
    b.data["EURUSD"] = bars(n=302, end="2026-03-05")
    rn.run_once(datetime(2026, 3, 6, tzinfo=timezone.utc))
    assert len(b.opened) == 1 and not (tmp_path / "data/fx_trend.RESET_BREAKER").exists()


def test_runner_pauses_after_four_recent_losses(tmp_path):
    b = FakeBroker({"EURUSD": bars()})
    b.history = losses(4, day="2026-03-02")
    runner(tmp_path, b, **BREAKERS).run_once(NOW)
    assert not b.opened


def test_runner_streak_guard_counts_open_positions(tmp_path):
    b = FakeBroker({"EURUSD": bars()})
    b.history = losses(3)
    b.pos = [Position(i, f"XX{i}YY{i}"[:6], 1, 0.01, 1.0, 0.9) for i in range(4)]  # 3 + 4 + 1 > 7
    runner(tmp_path, b, max_open_positions=10, **BREAKERS).run_once(NOW)
    assert not b.opened


def test_runner_drawdown_lock(tmp_path):
    b = FakeBroker({"EURUSD": bars()}, lots=0.0)
    rn = runner(tmp_path, b, **BREAKERS)
    rn.run_once(NOW)
    b._equity, b._lots = 9_100, 0.1                      # -9% from the peak
    b.data["EURUSD"] = bars(n=301, end="2026-03-04")
    rn.run_once(datetime(2026, 3, 5, tzinfo=timezone.utc))
    assert not b.opened
    state = json.loads((tmp_path / "data/fx_trend_state.json").read_text())
    assert state["breaker_locked"].startswith("drawdown")   # not just the daily-loss halt


def test_runner_blocks_entries_if_trade_history_unreadable(tmp_path):
    b = FakeBroker({"EURUSD": bars()})

    def boom(since):
        raise RuntimeError("terminal busy")
    b.closed_trades = boom
    runner(tmp_path, b, **BREAKERS).run_once(NOW)
    assert not b.opened


def test_stale_prices_do_nothing_and_do_not_consume_the_bar(tmp_path):
    b = FakeBroker({"EURUSD": bars(end="2023-06-07")}, equity=0.0)   # what the dry run really saw
    b.pos = [Position(1, "EURUSD", -1, 0.1, 1.2, 1.5)]
    rn = runner(tmp_path, b)
    r = rn.run_once(NOW)
    assert r["skipped"] == "no_data" and not b.opened and not b.closed
    assert not (tmp_path / "data/fx_trend_state.json").exists()


def test_unreadable_equity_does_nothing(tmp_path):
    b = FakeBroker({"EURUSD": bars()}, equity=0.0)
    assert runner(tmp_path, b).run_once(NOW)["skipped"] == "bad_account"
    assert not b.opened


def test_missing_data_for_a_held_pair_waits(tmp_path):
    b = FakeBroker({"EURUSD": bars(), "GBPUSD": bars().iloc[:0]})
    b.pos = [Position(1, "GBPUSD", 1, 0.1, 1.2, 1.1)]
    assert runner(tmp_path, b).run_once(NOW)["skipped"] == "incomplete_data"
    assert not b.opened and not b.closed


# ---- pullback style ---------------------------------------------------------
PULLBACK = TrendParams(style="pullback", sma_len=200, max_hold_days=5)


def dipped():
    """Long uptrend, then two sharp down days that stay above the 200-day average."""
    df = bars(n=300, drift=0.0015)
    for i, k in ((-2, 0.980), (-1, 0.960)):
        c = df["close"].iloc[-3] * k
        df.iloc[i] = [c / k ** 0.1, c * 1.004, c * 0.999, c]
    return df


def test_pullback_buys_the_dip_with_a_stop(tmp_path):
    assert compute_indicators(dipped(), PULLBACK)["signal"].iloc[-1] == 1
    b = FakeBroker({"EURUSD": dipped()})
    runner(tmp_path, b, params=PULLBACK).run_once(NOW)
    assert b.opened[0][1] == 1 and b.opened[0][3] < dipped()["close"].iloc[-1]


def test_pullback_time_exit_and_no_trailing(tmp_path):
    b = FakeBroker({"EURUSD": dipped()})
    b.pos = [Position(1, "EURUSD", 1, 0.1, 1.2, 0.5, open_time="2026-02-20T00:00:00+00:00")]
    runner(tmp_path, b, params=PULLBACK).run_once(NOW)
    assert b.closed == ["EURUSD"] and not b.opened and not b.modified


def test_pullback_holds_young_trade_without_moving_stop(tmp_path):
    b = FakeBroker({"EURUSD": dipped()})
    b.pos = [Position(1, "EURUSD", 1, 0.1, 1.2, 0.5, open_time="2026-03-02T00:00:00+00:00")]
    runner(tmp_path, b, params=PULLBACK).run_once(NOW)
    assert not b.closed and not b.modified


def test_pullback_signals_do_not_look_ahead():
    df = bars(n=400, drift=0.0003, seed=5)
    full, part = compute_indicators(df, PULLBACK), compute_indicators(df.iloc[:350], PULLBACK)
    cols = ["signal", "exit_long", "exit_short"]
    assert (full[cols].iloc[:350] == part[cols]).all().all()


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError):
        TrendParams(style="martingale")


def test_backtest_respects_hold_limit_and_counts_locks():
    frames = {f"P{i}": bars(n=900, drift=0.0, seed=i) for i in range(6)}
    p = TrendParams(style="pullback", sma_len=100, rsi_entry=15, max_hold_days=5)
    res = run_backtest(frames, p, BtConfig(max_consecutive_losses=3))
    assert res["trades"] and res["stats"]["breaker_locks"] >= 1
    assert all((t.exit_time - t.entry_time).days <= 5 for t in res["trades"])
