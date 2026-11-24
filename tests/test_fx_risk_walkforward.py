from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.fx_trend_broker import Position
from core.fx_trend_journal import Journal
from core.fx_trend_runner import FxTrendRunner, RunnerConfig
from strategy.fx_risk import (
    RiskLimits, entry_block_reason, halt_reason, in_cooldown, net_exposure,
)
from strategy.fx_trend import TrendParams, compute_indicators
from strategy.fx_trend_backtest import BtConfig, run_backtest
from strategy.fx_trend_walkforward import (
    baseline_grid, compare, filter_grid, walk_forward,
)
from tests.test_fx_trend import make_df, trending
from tests.test_fx_trend_runner import FakeBroker, NOW, bars


def regime_df(n=3000, seed=0, start="2012-01-02"):
    """Alternating trend regimes so a trend follower has something to find."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate([np.full(250, d) for d in
                            np.tile([0.0012, -0.0010, 0.0008, -0.0012], n // 1000 + 1)])[:n]
    c = 1.1 * np.cumprod(1 + drift + rng.normal(0, 0.004, n))
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001,
                         "low": np.minimum(o, c) * 0.999, "close": c}, index=idx)


# ---------------- shared gates ----------------
def test_net_exposure_and_caps():
    held = [("EURUSD", 1), ("GBPUSD", 1), ("AUDUSD", 1)]
    assert net_exposure(held)["USD"] == -3
    lim = RiskLimits(max_open_positions=6, max_net_per_currency=3)
    assert entry_block_reason("NZDUSD", 1, held, lim) == "currency exposure cap"
    assert entry_block_reason("NZDUSD", -1, held, lim) is None          # opposite side reduces USD bet
    assert entry_block_reason("EURUSD", 1, held * 2, RiskLimits(max_open_positions=3)) == "max open positions reached"


def test_halt_reason_and_cooldown():
    lim = RiskLimits(daily_loss_halt_pct=2, weekly_loss_halt_pct=4)
    assert halt_reason(-1.9, -3.9, lim) is None
    assert "daily" in halt_reason(-2.0, 0, lim) and "weekly" in halt_reason(0, -4.5, lim)
    assert in_cooldown(date(2026, 3, 1), date(2026, 3, 4), 5) is True
    assert in_cooldown(date(2026, 3, 1), date(2026, 3, 6), 5) is False
    assert in_cooldown(None, date(2026, 3, 6), 5) is False and in_cooldown(date(2026, 3, 1), date(2026, 3, 2), 0) is False
    assert in_cooldown(datetime(2026, 3, 1, 12), datetime(2026, 3, 3, 1), 5) is True


# ---------------- entry-strength filter ----------------
def test_entry_ok_is_stricter_than_signal_only_when_enabled():
    df = make_df(trending(300))
    off = compute_indicators(df, TrendParams())
    on = compute_indicators(df, TrendParams(min_entry_strength_atr=50.0))
    assert ((off["signal"] != 0) == off["entry_ok"]).all()               # off: identical to signal
    assert not on["entry_ok"].any() and (on["signal"] == off["signal"]).all()   # signal untouched


def test_strength_filter_reduces_entries_in_backtest():
    frames = {"EURUSD": regime_df(1500)}
    base = run_backtest(frames, TrendParams())
    strict = run_backtest(frames, TrendParams(min_entry_strength_atr=1.5))
    assert strict["stats"]["n_trades"] <= base["stats"]["n_trades"]


def test_cooldown_blocks_reentry_in_backtest():
    frames = {"EURUSD": regime_df(1500, seed=3)}
    gaps = lambda r: [(b.entry_time - a.exit_time).days  # noqa: E731
                      for a, b in zip(r["trades"], r["trades"][1:]) if a.pnl < 0]
    base = run_backtest(frames, TrendParams())
    cool = run_backtest(frames, TrendParams(cooldown_days=10))
    assert min(gaps(cool), default=99) >= 10
    assert cool["stats"]["n_trades"] <= base["stats"]["n_trades"]


# ---------------- live runner uses the same filters ----------------
def _runner(tmp_path, broker, **p):
    cfg = RunnerConfig(pairs=list(broker.data), dry_run=False, params=TrendParams(**p))
    return FxTrendRunner(broker, cfg, base_dir=tmp_path)


def test_runner_skips_weak_signal(tmp_path):
    b = FakeBroker({"EURUSD": bars()})
    rn = _runner(tmp_path, b, min_entry_strength_atr=99.0)
    rn.run_once(NOW)
    assert not b.opened
    assert any("too weak" in e["message"] for e in rn.journal.events())


def test_runner_cooldown_after_loss(tmp_path):
    b = FakeBroker({"EURUSD": bars()})
    rn = _runner(tmp_path, b, cooldown_days=10)
    rn.journal.upsert_trades([{"ticket": 1, "pair": "EURUSD", "side": 1, "volume": 0.1,
                               "open_time": "2026-02-20T00:00:00", "close_time": "2026-03-02T00:00:00",
                               "entry": 1.1, "exit": 1.09, "profit": -10.0, "commission": 0.0, "swap": 0.0}])
    rn.run_once(NOW)                                                      # NOW = 2026-03-04
    assert not b.opened and any("cooling down" in e["message"] for e in rn.journal.events())


# ---------------- walk-forward ----------------
def test_grids_have_expected_shape():
    assert len(baseline_grid()) == 12 and len(filter_grid()) == 48
    assert all(g.min_entry_strength_atr == 0 and g.cooldown_days == 0 for g in baseline_grid())


def test_walk_forward_is_out_of_sample_and_stitched():
    frames = {"EURUSD": regime_df(3300, 1), "GBPUSD": regime_df(3300, 2)}
    grid = baseline_grid()[:4]
    wf = walk_forward(frames, grid, BtConfig(), train_years=4, test_years=1)
    assert wf["ok"] and wf["n_folds"] >= 3
    for f in wf["folds"]:
        assert f["train"][1] == f["test"][0]                              # test starts where train ends
    tests = [f["test"][0] for f in wf["folds"]]
    assert tests == sorted(tests) and len(set(tests)) == len(tests)       # disjoint, ordered
    assert 0 < wf["stability"]["share"] <= 1 and wf["oos"]["n_trades"] > 0


def test_walk_forward_handles_too_little_history():
    wf = walk_forward({"EURUSD": regime_df(500)}, baseline_grid()[:2])
    assert wf["ok"] is False and "history" in wf["reason"]


def test_compare_verdict_is_cautious(monkeypatch):
    import strategy.fx_trend_walkforward as w
    mk = lambda sharpe, prof: {"ok": True, "oos": {"sharpe": sharpe}, "profitable_folds": prof}  # noqa: E731
    seq = iter([mk(0.5, 4), mk(0.6, 5)])
    monkeypatch.setattr(w, "walk_forward", lambda *a, **k: next(seq))
    assert "no meaningful difference" in w.compare({"x": None})["verdict"]
    seq = iter([mk(0.5, 4), mk(1.2, 5)])
    assert "improved" in w.compare({"x": None})["verdict"]
    seq = iter([mk(0.5, 4), mk(0.2, 3)])
    assert "did NOT help" in w.compare({"x": None})["verdict"]
