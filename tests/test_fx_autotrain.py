from datetime import timedelta

import pytest

from core.fx_trend_autotrain import AutoTrainConfig, AutoTrainer
from core.fx_trend_broker import Position
from strategy.fx_autotrain import (
    TrainSettings, apply_signal_params, candidate_pool, signal_params, train,
)
from strategy.fx_trend import STYLES, TrendParams, compute_indicators
from strategy.fx_trend_backtest import BtConfig
from tests.test_fx_trend_runner import NOW, FakeBroker, bars, runner

SHORT = TrainSettings(train_years=2, validation_years=1)
LONG_BARS = dict(n=1600)          # > train + validation + 1 years of daily rows
NEW = signal_params(TrendParams(style="breakout"))
PAIRS = ["EURUSD", "GBPUSD"]


def noise():
    return {p: bars(n=1600, drift=0.0, seed=i) for i, p in enumerate(PAIRS, 1)}


def result(decision, champion_passes=True):
    return {"decision": decision, "reason": "because",
            "champion": {"passes": champion_passes}, "challenger": {"params": NEW}}


def make(tmp_path, decision="keep", broker=None, dry_run=False, champion_passes=True, **cfg):
    b = broker or FakeBroker({"EURUSD": bars(**LONG_BARS)})
    rn = runner(tmp_path, b, dry_run=dry_run)
    calls = []

    def fake_train(*a, **k):
        calls.append(k)
        return result(decision, champion_passes)

    t = AutoTrainer(rn, AutoTrainConfig(enabled=True, settings=SHORT, **cfg),
                    train_fn=fake_train, background=False)
    return t, rn, b, calls


@pytest.mark.parametrize("style", STYLES)
def test_every_style_has_no_lookahead(style):
    df, p = bars(n=500, drift=0.0), TrendParams(style=style, max_vol_ratio=1.5)
    cols = ["signal", "entry_ok", "exit_long", "exit_short"]
    full = compute_indicators(df, p)[cols].iloc[:400]
    assert full.equals(compute_indicators(df.iloc[:400], p)[cols])


def test_train_only_changes_signal_settings_and_respects_may_promote():
    base = TrendParams(style="pullback", risk_pct=0.5)
    r = train(noise(), base, BtConfig(), SHORT, pool=candidate_pool(base)[:3], may_promote=False)
    assert r["decision"] in ("keep", "no_edge")
    assert "risk_pct" not in r["champion"]["params"]
    assert apply_signal_params(base, {"risk_pct": 9.0}).risk_pct == 0.5


def test_noise_with_heavy_costs_is_not_promoted():
    base = TrendParams(style="pullback")
    r = train(noise(), base, BtConfig(spread_pips={}, default_spread_pips=20.0), SHORT,
              pool=candidate_pool(base)[:3])
    assert r["decision"] == "no_edge"


def test_runs_once_per_interval(tmp_path):
    t, _, _, calls = make(tmp_path)
    t.poll(NOW)
    t.poll(NOW + timedelta(days=1))
    assert len(calls) == 1


def test_no_edge_pauses_entries_and_survives_restart(tmp_path):
    t, rn, b, _ = make(tmp_path, "no_edge", champion_passes=False)
    t.poll(NOW)
    rn.run_once(NOW)
    assert not b.opened
    _, rn2, _, _ = make(tmp_path)
    assert rn2.entry_hold


def test_on_no_edge_keep_does_not_pause(tmp_path):
    t, rn, _, _ = make(tmp_path, "no_edge", on_no_edge="keep")
    t.poll(NOW)
    assert rn.entry_hold is None


def test_switch_waits_until_flat(tmp_path):
    b = FakeBroker({"EURUSD": bars(**LONG_BARS)})
    b.pos.append(Position(1, "EURUSD", 1, 0.1, 1.0, 0.9))
    t, rn, _, _ = make(tmp_path, "promote", broker=b)
    t.poll(NOW)
    assert rn.cfg.params.style == "trend" and rn.entry_hold
    b.pos.clear()
    t.poll(NOW)
    assert rn.cfg.params.style == "breakout" and rn.entry_hold is None


def test_live_account_gets_a_proposal_only(tmp_path):
    b = FakeBroker({"EURUSD": bars(**LONG_BARS)})
    b.is_demo = lambda: False
    t, rn, _, _ = make(tmp_path, "promote", broker=b, champion_passes=False)
    t.poll(NOW)
    assert rn.cfg.params.style == "trend" and t.model["proposal"] == NEW
    assert rn.entry_hold                      # a failing champion must not keep opening trades


def test_thin_history_fails_safely_and_waits_before_retrying(tmp_path):
    t, rn, _, calls = make(tmp_path, broker=FakeBroker({"EURUSD": bars(n=300)}))
    t.poll(NOW)
    assert not calls and t.model["failed_at"] and rn.cfg.params == TrendParams()
    assert not t._due(NOW + timedelta(hours=1)) and t._due(NOW + timedelta(hours=25))


def test_editing_the_yaml_strategy_discards_the_saved_model(tmp_path):
    t, _, b, _ = make(tmp_path, "promote", dry_run=True)
    t.poll(NOW)
    rn = runner(tmp_path, b)
    rn.cfg.params = TrendParams(sma_len=150)
    AutoTrainer(rn, AutoTrainConfig(enabled=True))
    assert rn.cfg.params == TrendParams(sma_len=150)


def test_failed_close_keeps_kill_switch_running(tmp_path):
    b = FakeBroker({"EURUSD": bars()})
    b.pos.append(Position(1, "EURUSD", 1, 0.1, 1.0, 0.9))
    b.close = lambda pos: False
    (tmp_path / "data").mkdir()
    (tmp_path / "data/fx_trend.KILL").write_text("x")
    assert runner(tmp_path, b).run_once(NOW)["skipped"] == "kill_incomplete"
