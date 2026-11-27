"""
Portfolio event-loop backtest for strategy/fx_trend.py (daily bars).

Timing: signals are computed at bar t's close and filled at bar t+1's open.
Stops are checked on each bar's high/low (gap-aware: a gap through the stop
fills at the open, not at the stop). A position stopped out intrabar is not
re-entered on the same bar.

Costs modelled: spread (per pair, in pips, paid half on each side), slippage
(pips, always against you, also on stop fills) and swap (bps of notional per
day held; default is a flat COST, i.e. carry is not credited).

Risk controls modelled: risk_pct sizing from the ATR stop, leverage cap per
position, max concurrent positions, max net exposure per single currency,
daily-loss and weekly-loss entry halts, and the losing-streak / drawdown circuit
breakers. Live, a lock stays until a human resets it; here the reset is assumed
to happen `lock_review_days` later, and every lock is counted.

Simplification: P&L is computed as return on notional (account-currency
notional * price return), exact for USD-quoted pairs and a close approximation
for others. The live bot sizes with MT5's exact loss-per-lot instead.

Efficiency: per-bar rows are pre-extracted into plain dicts, so the main loop
does O(1) lookups instead of DataFrame slicing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy.fx_risk import (
    RiskLimits, entry_block_reason, halt_reason, in_cooldown, in_loss_pause, lock_reason,
    net_exposure, risk_multiplier, sizing_equity, streak_has_room,
)
from strategy.fx_trend import (
    TrendParams, compute_indicators, currency_legs, initial_stop,
    position_notional, trail_stop,
)

# Rough all-in typical spreads (pips). Override with your broker's real numbers.
DEFAULT_SPREAD_PIPS = {
    "EURUSD": 0.8, "GBPUSD": 1.0, "USDJPY": 0.9, "AUDUSD": 1.0, "USDCAD": 1.2,
    "USDCHF": 1.2, "NZDUSD": 1.4, "EURGBP": 1.2, "EURJPY": 1.5, "GBPJPY": 2.0,
    "AUDJPY": 1.8, "EURCHF": 1.8, "AUDNZD": 2.5, "EURAUD": 2.0,
}


def pip_size(pair: str) -> float:
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


@dataclass
class BtConfig:
    initial_equity: float = 10_000.0
    reinvest_pct: float = 100.0          # % of profit above initial equity that is reinvested in sizing
    slippage_pips: float = 0.3
    swap_bps_per_day: float = 0.3        # cost, per day per notional (calendar days held)
    spread_pips: dict = field(default_factory=lambda: dict(DEFAULT_SPREAD_PIPS))
    default_spread_pips: float = 1.5
    max_open_positions: int = 6
    max_net_per_currency: int = 3        # e.g. at most 3 net long-USD legs open at once
    daily_loss_halt_pct: float = 2.0     # no new entries next bar after a day down this much
    weekly_loss_halt_pct: float = 4.0    # no new entries for rest of ISO week
    trading_enabled_from: pd.Timestamp | None = None  # evaluate only from here (warm-up before)
    max_consecutive_losses: int = 0
    pause_after_losses: int = 0
    pause_days: int = 5
    pause_risk_factor: float = 0.5
    max_drawdown_halt_pct: float = 0.0
    lock_review_days: int = 30

    def risk_limits(self) -> RiskLimits:
        return RiskLimits(self.max_open_positions, self.max_net_per_currency,
                          self.daily_loss_halt_pct, self.weekly_loss_halt_pct,
                          self.max_consecutive_losses, self.pause_after_losses, self.pause_days,
                          self.pause_risk_factor, self.max_drawdown_halt_pct)


@dataclass
class Trade:
    pair: str
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    notional: float
    pnl: float
    equity_at_entry: float
    reason: str

    @property
    def ret_on_equity(self) -> float:
        return self.pnl / self.equity_at_entry if self.equity_at_entry else 0.0


def _fill(price: float, side_of_order: int, pair: str, cfg: BtConfig) -> float:
    """Price actually paid. side_of_order +1 = buying, -1 = selling."""
    half_spread = 0.5 * cfg.spread_pips.get(pair, cfg.default_spread_pips)
    adverse = (half_spread + cfg.slippage_pips) * pip_size(pair)
    return price + side_of_order * adverse


def run_backtest(frames: dict[str, pd.DataFrame], p: TrendParams = TrendParams(),
                 cfg: BtConfig | None = None) -> dict:
    cfg = cfg or BtConfig()
    ind = {pair: compute_indicators(df, p) for pair, df in frames.items()}
    rows = {pair: df[["open", "high", "low", "close", "atr", "signal", "entry_ok",
                      "exit_long", "exit_short"]].to_dict("index")
            for pair, df in ind.items()}
    limits = cfg.risk_limits()
    dates = sorted(set().union(*[set(r) for r in rows.values()]))
    if not dates:
        return {"equity": pd.Series(dtype=float), "trades": [], "stats": {"n_trades": 0}}

    cash = cfg.initial_equity
    pos: dict[str, dict] = {}
    pending: dict[str, int] = {}          # signal from previous close, per pair
    pending_ok: dict[str, bool] = {}      # whether that signal also passes the entry filter
    pending_exit: dict[str, tuple[bool, bool]] = {}   # (leave a long?, leave a short?)
    streak, locks, locked_since, last_any_loss = 0, 0, None, None
    peak_equity = cfg.initial_equity
    last_close: dict[str, float] = {}
    last_atr: dict[str, float] = {}
    trades: list[Trade] = []
    equity_curve: dict[pd.Timestamp, float] = {}
    prev_equity = cfg.initial_equity
    week_key, week_start_equity = None, cfg.initial_equity
    last_daily_pct = 0.0
    last_loss_close: dict[str, pd.Timestamp] = {}
    prev_date = None

    def mark() -> float:
        """Equity at the most recent close seen."""
        return cash + sum(ps["notional"] * ps["side"] * (last_close[pair] / ps["entry"] - 1.0)
                          for pair, ps in pos.items())

    def close_position(pair: str, date, price: float, reason: str):
        nonlocal cash, streak, last_any_loss
        ps = pos.pop(pair)
        px = _fill(price, -ps["side"], pair, cfg)
        gross = ps["notional"] * ps["side"] * (px / ps["entry"] - 1.0)
        cash += gross  # swap was already debited from cash day by day
        net_pnl = gross - ps["swap_total"]
        if net_pnl < 0:
            last_loss_close[pair] = date
            streak, last_any_loss = streak + 1, date
        else:
            streak = 0
        trades.append(Trade(pair, ps["side"], ps["entry_time"], date, ps["entry"], px,
                            ps["notional"], net_pnl, ps["equity_at_entry"], reason))

    def check_stop(pair: str, date) -> bool:
        """Close `pair` if today's bar touched its stop. Returns True if closed."""
        bar, ps = rows[pair].get(date), pos.get(pair)
        if bar is None or ps is None:
            return False
        if ps["side"] > 0 and bar["open"] <= ps["stop"]:
            close_position(pair, date, bar["open"], "stop_gap")
        elif ps["side"] < 0 and bar["open"] >= ps["stop"]:
            close_position(pair, date, bar["open"], "stop_gap")
        elif ps["side"] > 0 and bar["low"] <= ps["stop"]:
            close_position(pair, date, ps["stop"], "stop")
        elif ps["side"] < 0 and bar["high"] >= ps["stop"]:
            close_position(pair, date, ps["stop"], "stop")
        else:
            return False
        return True

    for date in dates:
        # ---- 1. stops (level set at the previous close) ----
        stopped_today = {pair for pair in list(pos) if check_stop(pair, date)}

        # ---- 2. signal / time exits, then entries, at today's open ----
        timed_out_today = set()
        for pair, (leave_long, leave_short) in pending_exit.items():
            bar, ps = rows[pair].get(date), pos.get(pair)
            if bar is None or ps is None:
                continue
            timed_out = bool(p.max_hold_days) and ps["bars_held"] >= p.max_hold_days
            if timed_out or (leave_long if ps["side"] > 0 else leave_short):
                close_position(pair, date, bar["open"], "time" if timed_out else "signal")
                if timed_out:
                    timed_out_today.add(pair)

        tradable = cfg.trading_enabled_from is None or date >= cfg.trading_enabled_from
        iso = date.isocalendar()
        wk = (iso.year, iso.week)
        weekly_pct = (mark() / week_start_equity - 1.0) * 100 if week_key == wk else 0.0
        halted = halt_reason(last_daily_pct, weekly_pct, limits) is not None

        eq_now = mark()
        peak_equity = max(peak_equity, eq_now)
        if locked_since is not None and (date - locked_since).days >= cfg.lock_review_days:
            locked_since, streak, peak_equity = None, 0, eq_now      # simulated manual reset
        if locked_since is None and lock_reason(streak, (eq_now / peak_equity - 1) * 100, limits):
            locked_since, locks = date, locks + 1
        halted = halted or locked_since is not None or in_loss_pause(streak, last_any_loss, date, limits)

        if tradable and not halted:
            net = net_exposure((q, ps["side"]) for q, ps in pos.items())
            for pair, want in pending.items():
                bar = rows[pair].get(date)
                # no same-bar re-entry after an intrabar stop-out (the stop happened
                # AFTER the open we would be filling at)
                if want == 0 or bar is None or pair in pos or pair in stopped_today or pair in timed_out_today:
                    continue
                if not streak_has_room(streak, len(pos), limits):
                    break
                if not pending_ok.get(pair):          # entry filter (hold logic is unaffected)
                    continue
                if in_cooldown(last_loss_close.get(pair), date, p.cooldown_days):
                    continue
                if len(pos) >= limits.max_open_positions:
                    break
                if entry_block_reason(pair, want, [(q, ps["side"]) for q, ps in pos.items()],
                                      limits, net):
                    continue
                legs = currency_legs(pair, want)
                atr_prev = last_atr.get(pair)
                if atr_prev is None or atr_prev != atr_prev:   # none / NaN
                    continue
                entry = _fill(bar["open"], want, pair, cfg)
                stop = initial_stop(want, entry, atr_prev, p.atr_stop_mult)
                eq = mark()
                notional = position_notional(sizing_equity(eq, cfg.initial_equity, cfg.reinvest_pct),
                                             p.risk_pct * risk_multiplier(streak, limits),
                                             entry, stop, p.max_leverage_per_pos)
                if notional <= 0:
                    continue
                pos[pair] = dict(side=want, entry=entry, stop=stop, notional=notional,
                                 entry_time=date, equity_at_entry=eq, swap_total=0.0, bars_held=0)
                for c, s in legs.items():
                    net[c] = net.get(c, 0) + s
                if check_stop(pair, date):   # the entry bar itself can hit the new stop
                    stopped_today.add(pair)
                    for c, s in legs.items():
                        net[c] -= s

        # ---- 3. end of day: roll closes, trail stops, swap, equity, new signals ----
        for pair, r in rows.items():
            bar = r.get(date)
            if bar is not None:
                last_close[pair], last_atr[pair] = bar["close"], bar["atr"]
        days = max((date - prev_date).days, 1) if prev_date is not None else 1
        for pair, ps in pos.items():
            bar = rows[pair].get(date)
            if bar is None:
                continue
            ps["bars_held"] += 1
            if p.trails:
                ps["stop"] = trail_stop(ps["side"], ps["stop"], bar["high"], bar["low"],
                                        bar["atr"], p.atr_stop_mult)
            swap = ps["notional"] * cfg.swap_bps_per_day / 1e4 * days
            ps["swap_total"] += swap
            cash -= swap

        eq = mark()
        equity_curve[date] = eq
        if week_key != wk:
            week_key, week_start_equity = wk, prev_equity
        last_daily_pct = (eq / prev_equity - 1.0) * 100
        prev_equity, prev_date = eq, date

        for pair, r in rows.items():
            bar = r.get(date)
            if bar is not None:
                pending[pair], pending_ok[pair] = int(bar["signal"]), bool(bar["entry_ok"])
                pending_exit[pair] = (bool(bar["exit_long"]), bool(bar["exit_short"]))

    # close anything still open at the last close, for reporting
    for pair in list(pos):
        close_position(pair, dates[-1], last_close[pair], "end")
    curve = pd.Series(equity_curve)
    if cfg.trading_enabled_from is not None:
        curve = curve[curve.index >= cfg.trading_enabled_from]
    return {"equity": curve, "trades": trades,
            "stats": {**stats(curve, trades), "max_loss_streak": longest_loss_streak(trades),
                      "breaker_locks": locks}}


def longest_loss_streak(trades: list[Trade]) -> int:
    best = cur = 0
    for t in trades:
        cur = cur + 1 if t.pnl < 0 else 0
        best = max(best, cur)
    return best


def stats(curve: pd.Series, trades: list[Trade]) -> dict:
    if len(curve) < 2:
        return {"n_trades": len(trades)}
    rets = curve.pct_change().dropna()
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    total = float(curve.iloc[-1] / curve.iloc[0] - 1)
    dd = float((curve / curve.cummax() - 1).min())
    pnl = np.array([t.pnl for t in trades]) if trades else np.array([])
    wins, losses = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    return {
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(((1 + total) ** (1 / years) - 1) * 100, 2) if total > -1 else -100.0,
        "max_drawdown_pct": round(dd * 100, 2),
        "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(252)), 2) if rets.std() > 0 else 0.0,
        "n_trades": len(trades),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 1) if len(pnl) else 0.0,
        "profit_factor": round(wins / losses, 2) if losses > 0 else None,
        "avg_trade_pct_of_equity": round(float(np.mean([t.ret_on_equity for t in trades]) * 100), 3) if trades else 0.0,
    }


def monte_carlo(trades: list[Trade], n_sims: int = 2000, seed: int = 7) -> dict:
    """Bootstrap trade-level returns-on-equity (with replacement) to see the
    spread of outcomes the same edge could have produced in a different order."""
    if len(trades) < 10:
        return {"note": "too few trades for Monte Carlo"}
    r = np.array([t.ret_on_equity for t in trades])
    rng = np.random.default_rng(seed)
    paths = np.cumprod(1 + rng.choice(r, size=(n_sims, len(r)), replace=True), axis=1)
    peaks = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), paths], axis=1), axis=1)[:, 1:]
    dds = (paths / peaks - 1).min(axis=1)
    finals = paths[:, -1] - 1
    return {
        "final_return_pct_p5": round(float(np.percentile(finals, 5) * 100), 1),
        "final_return_pct_median": round(float(np.percentile(finals, 50) * 100), 1),
        "max_dd_pct_p95_worst": round(float(np.percentile(dds, 5) * 100), 1),
        "prob_loss_pct": round(float((finals < 0).mean() * 100), 1),
    }
