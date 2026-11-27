"""
Daily time-series-momentum (trend-following) strategy for FX majors/crosses.

PURE FUNCTIONS ONLY: no broker calls, no I/O, no clock. The same functions feed
strategy/fx_trend_backtest.py and (later) a live runner, so a backtest describes
what live trading would do.

Rules (all evaluated on a CLOSED daily bar; trades are filled at the NEXT open):
  long  when close > SMA(sma_len)  AND  close / close[mom_len] - 1 > 0
  short when close < SMA(sma_len)  AND  close / close[mom_len] - 1 < 0
  flat  otherwise (the two filters disagree)
Risk: initial stop = atr_stop_mult * ATR(atr_len) from entry, ratcheting
(chandelier) in the trade's favour; size so that a stop-out loses risk_pct of
equity.

Nothing here is a promise of profit. Trend following loses in long chop and
earns its money in a few large trends.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendParams:
    sma_len: int = 100
    mom_len: int = 90
    atr_len: int = 14
    atr_stop_mult: float = 3.0
    risk_pct: float = 0.5          # % of equity lost if the initial stop is hit
    allow_short: bool = True
    max_leverage_per_pos: float = 5.0   # cap on notional / equity for one position
    # Whipsaw controls. Both default OFF, so behaviour is unchanged until a
    # walk-forward test (strategy/fx_trend_walkforward.py) shows they help.
    min_entry_strength_atr: float = 0.0  # open only if |close - SMA| >= this many ATRs (holding is unaffected)
    cooldown_days: int = 0               # after a LOSING close, don't re-enter that pair for N calendar days
    # Entry style. "trend" = the rules in the module docstring. "pullback" = with
    # close above the SMA(sma_len), buy when RSI(rsi_len) < rsi_entry (mirrored for
    # shorts); leave when close crosses back over SMA(exit_sma_len).
    style: str = "trend"
    rsi_len: int = 2
    rsi_entry: float = 5.0
    exit_sma_len: int = 5
    max_hold_days: int = 0               # leave after N closed daily bars (0 = no time limit)

    def __post_init__(self):
        if self.style not in ("trend", "pullback"):
            raise ValueError(f"unknown strategy style {self.style!r}")

    @property
    def trails(self) -> bool:
        """Only trend following ratchets its stop; a pullback trade keeps the initial one."""
        return self.style == "trend"

    @property
    def warmup(self) -> int:
        return max(self.sma_len, self.mom_len, self.atr_len) + 1


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR. Row t uses data up to and including bar t only."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    """Wilder RSI. Row t uses data up to and including bar t only."""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / down)


def compute_indicators(df: pd.DataFrame, p: TrendParams = TrendParams()) -> pd.DataFrame:
    """Return a copy with sma, mom, atr, strength, signal (+1/-1/0), entry_ok,
    exit_long and exit_short.

    Every column at row t depends only on bars <= t (no look-ahead). It is
    NaN-safe: during warm-up the signal is 0. `signal` is the side to OPEN,
    `entry_ok` the extra condition for opening it (see min_entry_strength_atr),
    and exit_long / exit_short say when to LEAVE a position already held.
    """
    out = df.copy()
    out["sma"] = out["close"].rolling(p.sma_len, min_periods=p.sma_len).mean()
    out["mom"] = out["close"] / out["close"].shift(p.mom_len) - 1.0
    out["atr"] = atr(out, p.atr_len)

    long_ = (out["close"] > out["sma"]) & (out["mom"] > 0)
    short_ = (out["close"] < out["sma"]) & (out["mom"] < 0) if p.allow_short else False
    ready = out[["sma", "mom", "atr"]].notna().all(axis=1)
    if p.style == "pullback":
        out["rsi"] = rsi(out["close"], p.rsi_len)
        exit_sma = out["close"].rolling(p.exit_sma_len, min_periods=p.exit_sma_len).mean()
        long_ = (out["close"] > out["sma"]) & (out["rsi"] < p.rsi_entry)
        short_ = (out["close"] < out["sma"]) & (out["rsi"] > 100.0 - p.rsi_entry) if p.allow_short else False
    sig = np.where(long_, 1, np.where(short_, -1, 0))
    out["signal"] = pd.Series(sig, index=out.index).where(ready, 0).astype(int)
    if p.style == "pullback":
        out["exit_long"] = out["close"] > exit_sma
        out["exit_short"] = out["close"] < exit_sma
    else:
        out["exit_long"] = out["signal"] != 1
        out["exit_short"] = out["signal"] != -1
    out["strength"] = (out["close"] - out["sma"]) / out["atr"]
    out["entry_ok"] = (out["signal"] != 0) & (out["strength"].abs() >= p.min_entry_strength_atr)
    return out


def initial_stop(side: int, entry: float, atr_value: float, mult: float) -> float:
    return entry - side * mult * atr_value


def trail_stop(side: int, current_stop: float, bar_high: float, bar_low: float,
               atr_value: float, mult: float) -> float:
    """Chandelier trail: only ever tightens, never loosens."""
    if side > 0:
        return max(current_stop, bar_high - mult * atr_value)
    return min(current_stop, bar_low + mult * atr_value)


def position_notional(equity: float, risk_pct: float, entry: float, stop: float,
                      max_leverage: float) -> float:
    """Notional (account currency) such that hitting `stop` loses risk_pct% of equity."""
    stop_frac = abs(entry - stop) / entry if entry > 0 else 0.0
    if equity <= 0 or stop_frac <= 0:
        return 0.0
    notional = equity * (risk_pct / 100.0) / stop_frac
    return min(notional, equity * max_leverage)


def currency_legs(pair: str, side: int) -> dict[str, int]:
    """'EURUSD' long -> {'EUR': +1, 'USD': -1}. Used to cap one-sided currency bets."""
    pair = pair.upper().replace("/", "").replace("_", "")[:6]
    base, quote = pair[:3], pair[3:6]
    return {base: side, quote: -side}


def latest_signals(frames: dict[str, pd.DataFrame], p: TrendParams = TrendParams()) -> dict[str, dict]:
    """Live helper: call with CLOSED daily bars only (drop the forming bar).

    Returns {pair: {"side": +1/-1/0, "atr": float, "close": float, "stop_dist": float}}.
    """
    out = {}
    for pair, df in frames.items():
        if len(df) < p.warmup:
            continue
        ind = compute_indicators(df, p)
        last = ind.iloc[-1]
        out[pair] = {
            "side": int(last["signal"]),
            "atr": float(last["atr"]),
            "close": float(last["close"]),
            "stop_dist": float(p.atr_stop_mult * last["atr"]),
        }
    return out
