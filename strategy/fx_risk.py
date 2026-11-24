"""
Entry gates shared by the backtest and the live runner.

Both call these exact functions, so "it passed in the backtest" and "it was
allowed live" can never silently diverge. Pure: no I/O, no clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

from strategy.fx_trend import currency_legs


@dataclass(frozen=True)
class RiskLimits:
    max_open_positions: int = 6
    max_net_per_currency: int = 3
    daily_loss_halt_pct: float = 2.0
    weekly_loss_halt_pct: float = 4.0


def net_exposure(open_positions: Iterable[tuple[str, int]]) -> dict[str, int]:
    """[('EURUSD', +1), ('GBPUSD', +1)] -> {'EUR': 1, 'USD': -2, 'GBP': 1}"""
    net: dict[str, int] = {}
    for pair, side in open_positions:
        for ccy, leg in currency_legs(pair, side).items():
            net[ccy] = net.get(ccy, 0) + leg
    return net


def halt_reason(daily_pct: float, weekly_pct: float, limits: RiskLimits) -> str | None:
    """Percent changes, negative = loss. Returns why new entries are blocked."""
    if daily_pct <= -limits.daily_loss_halt_pct:
        return f"daily loss {daily_pct:.2f}% hit the -{limits.daily_loss_halt_pct}% limit"
    if weekly_pct <= -limits.weekly_loss_halt_pct:
        return f"weekly loss {weekly_pct:.2f}% hit the -{limits.weekly_loss_halt_pct}% limit"
    return None


def entry_block_reason(pair: str, side: int, open_positions: list[tuple[str, int]],
                       limits: RiskLimits, net: dict[str, int] | None = None) -> str | None:
    """Why a new position in `pair` must be refused, or None if allowed.
    Pass a precomputed `net` (from net_exposure) to avoid recomputing in a loop."""
    if len(open_positions) >= limits.max_open_positions:
        return "max open positions reached"
    net = net if net is not None else net_exposure(open_positions)
    for ccy, leg in currency_legs(pair, side).items():
        if abs(net.get(ccy, 0) + leg) > limits.max_net_per_currency:
            return "currency exposure cap"
    return None


def sizing_equity(equity: float, starting_capital: float, reinvest_pct: float = 100.0) -> float:
    """Equity the position sizer is allowed to risk against.

    Profits above `starting_capital` are reinvested at `reinvest_pct` (100 = full
    compounding, 0 = always size off the starting capital and treat all profit
    as banked). Below starting capital, size off real equity: losses always
    shrink position sizes, never the other way round.
    """
    if starting_capital <= 0 or equity <= starting_capital:
        return equity
    r = min(max(reinvest_pct, 0.0), 100.0) / 100.0
    return starting_capital + r * (equity - starting_capital)


def in_cooldown(last_loss_close: date | datetime | None, today: date | datetime,
                cooldown_days: int) -> bool:
    """True if the pair closed at a loss less than `cooldown_days` ago."""
    if cooldown_days <= 0 or last_loss_close is None:
        return False
    a = last_loss_close.date() if isinstance(last_loss_close, datetime) else last_loss_close
    b = today.date() if isinstance(today, datetime) else today
    return (b - a).days < cooldown_days
