"""
Walk-forward analysis for the FX trend strategy.

For each fold: pick the best parameter set on a TRAIN window, then evaluate that
single choice on the TEST window that follows it, which it never saw. Stitching
the test windows gives an out-of-sample equity curve: the honest estimate.

Reading the output:
  * oos      -> stats of the stitched out-of-sample curve (trust this)
  * efficiency = mean OOS Sharpe / mean in-sample Sharpe. Near 1 is healthy;
    well below ~0.5 means the optimiser is mostly fitting noise.
  * stability -> how often the same parameters win. If the winner jumps around
    every fold, the "edge" is not tied to any particular setting.

Caveat: searching a grid and keeping the best is itself a mild form of
overfitting. Keep grids small, and compare a baseline grid against one with
extra filters on the SAME folds before believing a filter helps.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd

from strategy.fx_trend import TrendParams
from strategy.fx_trend_backtest import BtConfig, run_backtest, stats

MIN_TRADES_TO_SCORE = 15


def baseline_grid(base: TrendParams = TrendParams()) -> list[TrendParams]:
    """Trend lookback x stop width. 12 candidates, filters off."""
    return [replace(base, sma_len=s, mom_len=int(s * 0.9), atr_stop_mult=m)
            for s in (50, 100, 150, 200) for m in (2.5, 3.0, 3.5)]


def filter_grid(base: TrendParams = TrendParams()) -> list[TrendParams]:
    """Baseline grid x whipsaw filters. 12 x 4 = 48 candidates."""
    return [replace(g, min_entry_strength_atr=st, cooldown_days=cd)
            for g in baseline_grid(base) for st in (0.0, 0.5) for cd in (0, 5)]


def _slice(frames: dict[str, pd.DataFrame], start, end, pre_days: int) -> dict[str, pd.DataFrame]:
    lo = start - pd.Timedelta(days=pre_days)
    out = {}
    for pair, df in frames.items():
        s = df[(df.index >= lo) & (df.index < end)]
        if len(s):
            out[pair] = s
    return out


def _score(st: dict) -> float:
    if st.get("n_trades", 0) < MIN_TRADES_TO_SCORE:
        return float("-inf")
    return float(st.get("sharpe", float("-inf")))


def _label(p: TrendParams) -> str:
    return f"sma{p.sma_len}/atr{p.atr_stop_mult}/str{p.min_entry_strength_atr}/cd{p.cooldown_days}"


def walk_forward(frames: dict[str, pd.DataFrame], grid: list[TrendParams],
                 cfg: BtConfig | None = None, train_years: int = 4, test_years: int = 1) -> dict:
    cfg = cfg or BtConfig()
    if not frames or not grid:
        return {"ok": False, "reason": "no data or empty grid"}
    pre_days = int(max(g.warmup for g in grid) * 1.6) + 10
    first = min(df.index[0] for df in frames.values())
    last = max(df.index[-1] for df in frames.values())
    t0 = (first + pd.Timedelta(days=pre_days)).normalize()
    train, test = pd.DateOffset(years=train_years), pd.DateOffset(years=test_years)

    folds, oos_returns, oos_trades = [], [], []
    while t0 + train + test <= last + pd.Timedelta(days=1):
        tr_s, tr_e = t0, t0 + train
        te_s, te_e = tr_e, tr_e + test
        tr_frames = _slice(frames, tr_s, tr_e, pre_days)
        scored = []
        for g in grid:
            res = run_backtest(tr_frames, g, replace(cfg, trading_enabled_from=tr_s))
            scored.append((_score(res["stats"]), g, res["stats"]))
        best_score, best, best_stats = max(scored, key=lambda x: x[0])
        if best_score == float("-inf"):
            t0 += test
            continue                      # nothing tradable/scorable in this window
        te = run_backtest(_slice(frames, te_s, te_e, pre_days), best,
                          replace(cfg, trading_enabled_from=te_s))
        if len(te["equity"]) > 1:
            oos_returns.append(te["equity"].pct_change().dropna())
            oos_trades.extend(te["trades"])
        folds.append({
            "train": [str(tr_s.date()), str(tr_e.date())], "test": [str(te_s.date()), str(te_e.date())],
            "params": _label(best), "is_sharpe": best_stats.get("sharpe"),
            "oos": te["stats"],
        })
        t0 += test

    if not folds or not oos_returns:
        return {"ok": False, "reason": "not enough history for even one train+test fold "
                                       f"({train_years}y + {test_years}y after warm-up)"}
    rets = pd.concat(oos_returns)
    curve = (1 + rets).cumprod() * cfg.initial_equity
    oos = stats(curve, oos_trades)
    is_mean = float(np.mean([f["is_sharpe"] for f in folds if f["is_sharpe"] is not None]))
    oos_mean = float(np.mean([f["oos"].get("sharpe", 0.0) for f in folds]))
    counts = Counter(f["params"] for f in folds)
    top, n_top = counts.most_common(1)[0]
    return {
        "ok": True, "n_folds": len(folds), "grid_size": len(grid),
        "train_years": train_years, "test_years": test_years,
        "oos": oos, "folds": folds,
        "efficiency": round(oos_mean / is_mean, 2) if is_mean > 0 else None,
        "is_sharpe_mean": round(is_mean, 2), "oos_sharpe_mean": round(oos_mean, 2),
        "profitable_folds": sum(1 for f in folds if f["oos"].get("total_return_pct", 0) > 0),
        "stability": {"most_common": top, "share": round(n_top / len(folds), 2),
                      "distinct_choices": len(counts)},
        "oos_equity": [{"t": str(i.date()), "v": round(float(v), 2)}
                       for i, v in curve.iloc[::max(len(curve) // 300, 1)].items()],
    }


def compare(frames, cfg: BtConfig | None = None, base: TrendParams = TrendParams(),
            train_years: int = 4, test_years: int = 1) -> dict:
    """Same folds, baseline grid vs grid with filters. Verdict is deliberately cautious."""
    a = walk_forward(frames, baseline_grid(base), cfg, train_years, test_years)
    b = walk_forward(frames, filter_grid(base), cfg, train_years, test_years)
    verdict = "inconclusive"
    if a.get("ok") and b.get("ok"):
        da = b["oos"].get("sharpe", 0) - a["oos"].get("sharpe", 0)
        if da > 0.3 and b["profitable_folds"] >= a["profitable_folds"]:
            verdict = "filters improved out-of-sample Sharpe; still confirm on a second, later period"
        elif da < -0.1:
            verdict = "filters did NOT help out-of-sample: leave them off"
        else:
            verdict = "no meaningful difference: keep it simple, leave filters off"
    return {"baseline": a, "with_filters": b, "verdict": verdict}
