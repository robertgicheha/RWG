"""
Walk-forward retraining with a champion / challenger gate. PURE: no I/O, no clock.

Each training run:
  1. Split history: a TRAIN window, then the most recent VALIDATION window.
  2. Pick ONE challenger from the candidate pool using the train window only.
  3. Judge the challenger and the current champion on the validation window, at
     normal and at stressed costs.
  4. Walk-forward the whole "pick the best on train" procedure over all history.
     If picking has not made money out-of-sample, nothing may be promoted.

Decisions: "promote" (use the challenger), "keep" (champion still passes) or
"no_edge" (nothing passes: stop opening trades). Only signal settings are ever
changed. Risk per trade, loss limits, pairs and breakers are never touched.

Passing these gates is NOT a promise of profit. Repeating the test every month
gives luck more chances to get through, which is why promotion also needs a
margin over the champion and a minimum time between changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import pandas as pd

from strategy.fx_trend import TrendParams
from strategy.fx_trend_backtest import BtConfig, run_backtest
from strategy.fx_trend_walkforward import MIN_TRADES_TO_SCORE, walk_forward

# The only settings retraining may change.
SIGNAL_KEYS = ("style", "sma_len", "mom_len", "atr_stop_mult", "rsi_len", "rsi_entry",
               "exit_sma_len", "max_hold_days", "breakout_len", "breakout_exit_len",
               "band_len", "band_k", "max_vol_ratio", "min_entry_strength_atr", "cooldown_days")


@dataclass(frozen=True)
class TrainSettings:
    train_years: int = 6
    validation_years: int = 2
    min_validation_trades: int = 30
    max_validation_drawdown_pct: float = 8.0
    min_profit_factor: float = 1.05
    min_sharpe_gain: float = 0.2          # challenger must beat a healthy champion by this
    stress_slippage_pips: float = 1.0
    stress_swap_multiplier: float = 2.0
    min_walk_forward_return_pct: float = 0.0


def candidate_pool(base: TrendParams) -> list[TrendParams]:
    """Deliberately small: every extra candidate is another chance to fit noise."""
    pool: list[TrendParams] = []
    pull = dict(style="pullback", max_hold_days=5, rsi_len=2, exit_sma_len=5)
    for rsi_entry in (5.0, 10.0):
        for stop in (2.0, 3.0):
            for vol in (0.0, 1.5):
                pool.append(replace(base, sma_len=200, rsi_entry=rsi_entry,
                                    atr_stop_mult=stop, max_vol_ratio=vol, **pull))
    for k in (1.5, 2.0):
        for stop in (2.0, 3.0):
            pool.append(replace(base, style="band", sma_len=200, band_len=20, band_k=k,
                                atr_stop_mult=stop, max_hold_days=10, max_vol_ratio=0.0))
    for n_in, n_out in ((20, 10), (55, 20), (100, 50)):
        for stop in (2.0, 3.0):
            pool.append(replace(base, style="breakout", sma_len=100, breakout_len=n_in,
                                breakout_exit_len=n_out, atr_stop_mult=stop,
                                max_hold_days=0, max_vol_ratio=0.0))
    for sma in (100, 200):
        pool.append(replace(base, style="trend", sma_len=sma, mom_len=int(sma * 0.9),
                            atr_stop_mult=3.0, max_hold_days=0, max_vol_ratio=0.0))
    return pool


def signal_params(p: TrendParams) -> dict:
    return {k: v for k, v in asdict(p).items() if k in SIGNAL_KEYS}


def apply_signal_params(base: TrendParams, params: dict) -> TrendParams:
    return replace(base, **{k: v for k, v in params.items() if k in SIGNAL_KEYS})


def label(p: TrendParams) -> str:
    detail = {"pullback": f"rsi<{p.rsi_entry:g} sma{p.sma_len}",
              "band": f"{p.band_k:g}sd/{p.band_len} sma{p.sma_len}",
              "breakout": f"{p.breakout_len}/{p.breakout_exit_len}",
              "trend": f"sma{p.sma_len}"}[p.style]
    vol = f" vol<{p.max_vol_ratio:g}" if p.max_vol_ratio else ""
    return f"{p.style} {detail} stop {p.atr_stop_mult:g}ATR{vol}"


def _window(frames, start, end=None, warmup_bars: int = 320) -> dict:
    """Bars in [start, end) plus warm-up bars before `start`."""
    out = {}
    for pair, df in frames.items():
        if end is not None:
            df = df[df.index < end]
        first = max(int(df.index.searchsorted(start)) - warmup_bars, 0)
        if len(df) > first:
            out[pair] = df.iloc[first:]
    return out


def evaluate(frames, p: TrendParams, cfg: BtConfig, start, s: TrainSettings) -> dict:
    """Validation stats at normal and stressed costs, plus the reasons it fails (if any)."""
    data = _window(frames, start)
    run = replace(cfg, trading_enabled_from=start)
    normal = run_backtest(data, p, run)["stats"]
    stressed = run_backtest(data, p, replace(
        run, slippage_pips=s.stress_slippage_pips,
        swap_bps_per_day=cfg.swap_bps_per_day * s.stress_swap_multiplier))["stats"]
    why = []
    if normal.get("n_trades", 0) < s.min_validation_trades:
        why.append(f"only {normal.get('n_trades', 0)} trades (need {s.min_validation_trades})")
    if normal.get("total_return_pct", 0) <= 0:
        why.append(f"lost money ({normal.get('total_return_pct', 0)}%)")
    pf = normal.get("profit_factor")
    if pf is not None and pf < s.min_profit_factor:
        why.append(f"profit factor {pf} below {s.min_profit_factor}")
    if -normal.get("max_drawdown_pct", 0) > s.max_validation_drawdown_pct:
        why.append(f"drawdown {normal.get('max_drawdown_pct')}% too deep")
    if stressed.get("total_return_pct", 0) < 0:
        why.append(f"loses at higher costs ({stressed.get('total_return_pct')}%)")
    return {"label": label(p), "params": signal_params(p), "normal": normal,
            "stressed": stressed, "passes": not why, "fail_reasons": why}


def train(frames: dict[str, pd.DataFrame], champion: TrendParams, cfg: BtConfig,
          s: TrainSettings = TrainSettings(), pool: list[TrendParams] | None = None,
          may_promote: bool = True) -> dict:
    """One training run. `may_promote=False` still reports the challenger's results."""
    pool = pool if pool is not None else candidate_pool(champion)
    last = max(df.index[-1] for df in frames.values())
    val_start = last - pd.DateOffset(years=s.validation_years)
    train_start = val_start - pd.DateOffset(years=s.train_years)

    train_frames = _window(frames, train_start, val_start)
    train_cfg = replace(cfg, trading_enabled_from=train_start)
    scored = []
    for p in pool:
        st = run_backtest(train_frames, p, train_cfg)["stats"]
        if st.get("n_trades", 0) >= MIN_TRADES_TO_SCORE and st.get("total_return_pct", 0) > 0:
            scored.append((st.get("sharpe", 0.0), p, st))

    champ = evaluate(frames, champion, cfg, val_start, s)
    challenger = None
    if scored:
        _, best, train_stats = max(scored, key=lambda x: x[0])
        challenger = {**evaluate(frames, best, cfg, val_start, s), "train": train_stats}

    wf = walk_forward(frames, pool, cfg, train_years=s.train_years, test_years=1)
    wf_return = wf["oos"].get("total_return_pct") if wf.get("ok") else None
    procedure_ok = wf_return is not None and wf_return > s.min_walk_forward_return_pct

    better = (
        challenger is not None and challenger["passes"]
        and challenger["params"] != champ["params"]
        and (not champ["passes"]
             or challenger["normal"].get("sharpe", 0) >= champ["normal"].get("sharpe", 0) + s.min_sharpe_gain)
    )
    if better and procedure_ok and may_promote:
        decision = "promote"
        reason = f"{challenger['label']} passed on unseen data and beat the current settings"
    elif champ["passes"]:
        decision = "keep"
        reason = "current settings still pass"
        if better:
            reason += ("; challenger held back: " +
                       ("too soon since the last change" if procedure_ok else
                        "picking settings this way has not made money out-of-sample"))
    else:
        decision = "no_edge"
        reason = "current settings fail: " + ", ".join(champ["fail_reasons"])

    return {
        "decision": decision, "reason": reason,
        "validation_from": str(val_start.date()), "data_to": str(last.date()),
        "pairs": sorted(frames), "candidates": len(pool), "profitable_in_training": len(scored),
        "champion": champ, "challenger": challenger,
        "walk_forward": {"ok": bool(wf.get("ok")), "oos": wf.get("oos"),
                         "profitable_folds": wf.get("profitable_folds"), "n_folds": wf.get("n_folds"),
                         "efficiency": wf.get("efficiency"), "procedure_ok": procedure_ok},
    }
