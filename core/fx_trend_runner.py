"""
Live runner for the daily FX trend-following bot.

tick()      -> called every poll interval (cheap): refreshes the dashboard
               snapshot, syncs closed trades, then run_once().
run_once()  -> one decision cycle, acting at most once per new closed daily bar.
               Idempotent: restarts and polling can never double-trade.

Pipeline (mirrors the backtest, strategy/fx_trend_backtest.py):
  closed daily bars -> compute_indicators (pure) -> exits / stop trailing
  -> risk gates -> sizing from the stop -> orders (with SL attached)

Safety rails (all on by default):
  * dry_run=True: logs what it WOULD do, sends nothing
  * every order carries a stop-loss at send time
  * only touches positions with its own magic number
  * kill switch file: next cycle flattens and stops
  * circuit breakers: pause after a short losing streak; lock new entries at the
    streak limit or max drawdown until a human resets them
  * stale prices or an unreadable account: do nothing and retry
  * daily / weekly loss halts, max positions, per-currency exposure cap,
    spread guard, lots rounded DOWN so risk is never exceeded
  * broker positions are the source of truth each cycle (reconciliation)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from core.fx_trend_analytics import flip_analysis, running_stats, signal_levels, strength
from core.fx_trend_broker import Broker, Position
from core.fx_trend_journal import Journal
from strategy.fx_risk import (
    RiskLimits, entry_block_reason, halt_reason, in_cooldown, in_loss_pause, lock_reason,
    loss_streak, net_exposure, risk_multiplier, sizing_equity, streak_has_room,
)
from strategy.fx_trend import (
    TrendParams, compute_indicators, currency_legs, initial_stop, trail_stop,
)

logger = logging.getLogger("fx_trend_runner")

TRADE_SYNC_SECONDS = 300


@dataclass
class RunnerConfig:
    pairs: list[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "EURJPY", "GBPJPY"])
    params: TrendParams = field(default_factory=TrendParams)
    dry_run: bool = True
    history_bars: int = 400
    max_open_positions: int = 6
    max_net_per_currency: int = 3
    daily_loss_halt_pct: float = 2.0
    weekly_loss_halt_pct: float = 4.0
    max_consecutive_losses: int = 0
    pause_after_losses: int = 0
    pause_days: int = 5
    pause_risk_factor: float = 0.5
    max_drawdown_halt_pct: float = 0.0
    max_data_age_days: int = 5       # newest closed daily bar older than this = stale (covers holidays)
    breaker_reset_file: str = "data/fx_trend.RESET_BREAKER"
    max_spread_pips: float = 3.0
    max_lots_per_order: float = 1.0
    reinvest_pct: float = 100.0      # % of profit above starting capital that is reinvested in sizing
    starting_capital: float = 0.0    # 0 = use account equity at the first cycle
    kill_file: str = "data/fx_trend.KILL"
    state_file: str = "data/fx_trend_state.json"
    snapshot_file: str = "data/fx_trend_snapshot.json"
    journal_file: str = "data/fx_trend.db"


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    for attempt in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:      # Windows: a reader has the file open for a moment
            if attempt == 4:
                raise
            time.sleep(0.1)


class FxTrendRunner:
    def __init__(self, broker: Broker, cfg: RunnerConfig,
                 notify: Callable[[str], None] = lambda msg: None,
                 base_dir: Path | str = ".", journal: Journal | None = None):
        self.broker, self.cfg, self.notify = broker, cfg, notify
        self.base = Path(base_dir)
        self.journal = journal if journal is not None else Journal(self.base / cfg.journal_file)
        self._last_trade_sync: datetime | None = None
        self._halted = False

    # ------------------------------------------------------------------
    def _event(self, kind: str, msg: str, pair: str = "", level: str = "INFO") -> None:
        try:
            self.journal.add_event(kind, msg, pair, level)
        except Exception:
            logger.exception("journal write failed")

    def _act(self, kind: str, pair: str, msg: str) -> None:
        tag = "[DRY] " if self.cfg.dry_run else ""
        logger.info(tag + msg)
        self.notify(tag + msg)
        self._event(kind, tag + msg, pair)

    def _flatten(self, reason: str) -> None:
        for pos in self.broker.positions():
            self._act("close", pos.pair, f"CLOSE {pos.pair} ({reason})")
            if not self.cfg.dry_run:
                self.broker.close(pos)

    # ------------------------------------------------------------------
    def snapshot(self, now: datetime | None = None) -> dict:
        """Cheap live view for the dashboard (account + positions)."""
        now = now or datetime.now(timezone.utc)
        state = _load(self.base / self.cfg.state_file)
        positions = []
        for p in self.broker.positions():
            q = self.broker.quote(p.pair)
            px = (q.bid if p.side > 0 else q.ask) if q else None
            positions.append({
                "ticket": p.ticket, "pair": p.pair, "side": p.side, "volume": p.volume,
                "entry": p.entry, "sl": p.sl, "price": px, "profit": p.profit,
                "open_time": p.open_time,
                "dist_to_sl_pct": round(abs(px - p.sl) / px * 100, 3) if px and p.sl else None,
            })
        acct = self.broker.account_summary()
        base = state.get("starting_capital") or 0.0
        profit = (acct.get("equity", 0.0) - base) if base else 0.0
        r = min(max(self.cfg.reinvest_pct, 0.0), 100.0) / 100.0
        capital = {"starting_capital": base, "profit": round(profit, 2),
                   "reinvest_pct": self.cfg.reinvest_pct,
                   "reinvested": round(max(profit, 0.0) * r, 2),
                   "banked": round(max(profit, 0.0) * (1 - r), 2)}
        snap = {
            "capital": capital,
            "ts": now.isoformat(timespec="seconds"),
            "mode": "dry_run" if self.cfg.dry_run else "orders",
            "account": acct,
            "positions": positions,
            "floating_pnl": round(sum(p["profit"] for p in positions), 2),
            "last_bar": state.get("last_bar"),
            "halted": self._halted,
            "breaker": {"loss_streak": state.get("loss_streak", 0),
                        "locked": state.get("breaker_locked")},
            "pairs": self.cfg.pairs,
        }
        _save(self.base / self.cfg.snapshot_file, snap)
        return snap

    def sync_trades(self, now: datetime, force: bool = False) -> int | None:
        """Returns the number of trades read, or None if the history could not be read."""
        if not force and self._last_trade_sync and (now - self._last_trade_sync).total_seconds() < TRADE_SYNC_SECONDS:
            return 0
        self._last_trade_sync = now
        try:
            trades = self.broker.closed_trades(now - timedelta(days=400))
            known = {t["ticket"] for t in self.journal.trades(5000)}
            self.journal.upsert_trades(trades)
            self._announce_closed([t for t in trades if t["ticket"] not in known], now)
            return len(trades)
        except Exception:
            logger.exception("trade sync failed")
            return None

    def _announce_closed(self, new: list[dict], now: datetime) -> None:
        """After each newly closed trade: its P&L, total money made so far, and win rate."""
        if not new:
            return
        fresh = {t["ticket"] for t in new
                 if (t.get("close_time") or "") >= (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")}
        for r in running_stats(self.journal.trades(5000)):       # oldest first
            if r["ticket"] in fresh:
                self._act("trade", r["pair"],
                          f"TRADE CLOSED {r['pair']} {'BUY' if r['side'] > 0 else 'SELL'} net {r['net']:+.2f} "
                          f"| total made {r['cum_pnl']:+.2f} | win rate {r['win_rate_pct']:.1f}% "
                          f"({r['wins']}/{r['n']})")

    def tick(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        try:
            self.snapshot(now)
            self.sync_trades(now)
        except Exception:
            logger.exception("snapshot/sync failed")
        return self.run_once(now)

    def _write_candles(self, inds: dict, n: int = 150) -> None:
        """Recent daily OHLC + SMA per pair for the dashboard's candlestick charts."""
        try:
            out = {}
            for pair, ind in inds.items():
                t = ind.tail(n)
                out[pair] = [{"d": i.strftime("%Y-%m-%d"), "o": float(r["open"]), "h": float(r["high"]),
                              "l": float(r["low"]), "c": float(r["close"]),
                              "sma": None if r["sma"] != r["sma"] else float(r["sma"])}
                             for i, r in t.iterrows()]
            path = self.base / "data" / "fx_trend_candles.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(out))
            tmp.replace(path)
        except Exception:
            logger.exception("candle export failed")

    # ------------------------------------------------------------------
    def _apply_breaker_reset(self, state: dict, state_path: Path) -> None:
        """A human asked to unlock: forget the streak and restart the drawdown peak."""
        flag = self.base / self.cfg.breaker_reset_file
        if not flag.exists():
            return
        self.sync_trades(datetime.now(timezone.utc), force=True)
        state["streak_since"] = max((t.get("close_time") or "" for t in self.journal.trades(5000)), default="")
        for key in ("breaker_locked", "peak_equity", "loss_streak"):
            state.pop(key, None)
        _save(state_path, state)
        flag.unlink(missing_ok=True)
        self._act("breaker", "", "circuit breaker RESET by operator")

    def _loss_record(self, state: dict) -> tuple[int, datetime | None]:
        """(current losing streak, close time of the latest loss) since the last reset."""
        since = state.get("streak_since", "")
        done = [r for r in running_stats(self.journal.trades(5000)) if (r.get("close_time") or "") > since]
        streak = loss_streak(r["net"] for r in done)
        try:
            last = datetime.fromisoformat(done[-1]["close_time"]) if streak else None
        except (ValueError, TypeError):
            last = None
        return streak, last

    def _skip_cycle(self, summary: dict, why: str, msg: str) -> dict:
        """Do nothing and leave last_bar alone, so the next poll tries again."""
        logger.warning(msg)
        self._event("data", msg, level="WARNING")
        summary["skipped"] = why
        return summary

    # ------------------------------------------------------------------
    def run_once(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        cfg, p = self.cfg, self.cfg.params
        state_path = self.base / cfg.state_file
        state = _load(state_path)
        summary = {"actions": [], "skipped": None}

        # ---- kill switch -------------------------------------------------
        if (self.base / cfg.kill_file).exists():
            self._flatten("KILL SWITCH")
            self._event("kill", "Kill switch active: flattened and stopped.", level="WARNING")
            self.notify("FX trend bot: kill switch active, flattened and stopped.")
            summary["skipped"] = "kill_switch"
            return summary

        self._apply_breaker_reset(state, state_path)

        # ---- cheap new-bar probe (avoids downloading history every poll) --
        probe = self.broker.latest_bar_date(cfg.pairs[0]) if cfg.pairs else None
        if probe and probe == state.get("last_bar"):
            summary["skipped"] = "already_processed"
            return summary

        # ---- data --------------------------------------------------------
        frames = {}
        oldest_ok = (now - timedelta(days=cfg.max_data_age_days)).strftime("%Y-%m-%d")
        for pair in cfg.pairs:
            df = self.broker.daily_bars(pair, cfg.history_bars)
            if len(df) and df.index[-1].strftime("%Y-%m-%d") < oldest_ok:
                msg = f"{pair}: STALE data (newest bar {df.index[-1]:%Y-%m-%d}), skipped"
                logger.warning(msg)
                self._event("data", msg, pair, "WARNING")
            elif len(df) >= p.warmup + 5:
                frames[pair] = df
            else:
                logger.warning("%s: only %d bars, skipped", pair, len(df))
                self._event("data", f"{pair}: only {len(df)} bars, skipped", pair, "WARNING")
        if not frames:
            summary["skipped"] = "no_data"
            self._event("data", "NO DATA from broker, nothing done", level="ERROR")
            self.notify("FX trend bot: NO DATA from broker, nothing done.")
            return summary

        latest_bar = max(df.index[-1] for df in frames.values()).strftime("%Y-%m-%d")
        if state.get("last_bar") == latest_bar:
            summary["skipped"] = "already_processed"
            return summary
        # a held pair without data cannot be managed: wait, do not consume the bar
        unmanaged = [pos.pair for pos in self.broker.positions() if pos.pair in cfg.pairs and pos.pair not in frames]
        if unmanaged:
            return self._skip_cycle(summary, "incomplete_data", f"no usable data for held pair(s) {unmanaged}")

        # ---- loss halts --------------------------------------------------
        equity = self.broker.equity()
        if not equity or equity <= 0:
            return self._skip_cycle(summary, "bad_account", f"account equity unreadable ({equity})")
        if not state.get("starting_capital"):
            state["starting_capital"] = cfg.starting_capital or equity
            self._event("capital", f"starting capital set to {state['starting_capital']:.2f}")
        size_eq = sizing_equity(equity, state["starting_capital"], cfg.reinvest_pct)
        iso = now.isocalendar()
        wk = f"{iso.year}-W{iso.week:02d}"
        if state.get("week") != wk:
            state["week"], state["week_start_equity"] = wk, equity
        prev_eq = state.get("prev_equity", equity)
        daily_pct = (equity / prev_eq - 1) * 100 if prev_eq else 0.0
        weekly_pct = (equity / state["week_start_equity"] - 1) * 100 if state.get("week_start_equity") else 0.0
        limits = RiskLimits(cfg.max_open_positions, cfg.max_net_per_currency,
                            cfg.daily_loss_halt_pct, cfg.weekly_loss_halt_pct,
                            cfg.max_consecutive_losses, cfg.pause_after_losses, cfg.pause_days,
                            cfg.pause_risk_factor, cfg.max_drawdown_halt_pct)
        why_halted = halt_reason(daily_pct, weekly_pct, limits)

        # ---- circuit breakers ----------------------------------------------
        history_ok = self.sync_trades(now, force=True) is not None
        streak, last_loss = self._loss_record(state)
        peak = max(state.get("peak_equity") or equity, equity)
        state.update(peak_equity=peak, loss_streak=streak)
        if not state.get("breaker_locked"):
            locked = lock_reason(streak, (equity / peak - 1) * 100, limits)
            if locked:
                state["breaker_locked"] = locked
                self._act("breaker", "", f"CIRCUIT BREAKER LOCKED: {locked}. No new entries until reset.")
        if state.get("breaker_locked"):
            why_halted = f"circuit breaker locked ({state['breaker_locked']}); reset required"
        elif in_loss_pause(streak, last_loss, now, limits):
            why_halted = f"pausing {limits.pause_days}d after {streak} losses in a row"
        elif not history_ok and (limits.max_consecutive_losses or limits.pause_after_losses):
            why_halted = "trade history unreadable, cannot check the losing streak"
        risk_pct = p.risk_pct * risk_multiplier(streak, limits)
        halted = why_halted is not None
        self._halted = halted
        if halted:
            msg = f"entries HALTED: {why_halted}"
            self.notify("FX trend bot: " + msg)
            self._event("halt", msg, level="WARNING")

        inds = {pair: compute_indicators(df, p) for pair, df in frames.items()}
        rows = {pair: ind.iloc[-1] for pair, ind in inds.items()}
        self._write_candles(inds)

        # ---- journal the analysis for the dashboard ------------------------
        sig_rows = []
        for pair, row in rows.items():
            up, dn = signal_levels(frames[pair]["close"], float(row["sma"]), p.mom_len)
            sig_rows.append({
                "pair": pair, "side": int(row["signal"]), "close": float(row["close"]),
                "sma": float(row["sma"]), "mom": float(row["mom"]), "atr": float(row["atr"]),
                "strength": strength(float(row["close"]), float(row["sma"]), float(row["atr"])),
                "up_level": up, "dn_level": dn,
                **flip_analysis(int(row["signal"]), float(row["close"]), float(row["atr"]), up, dn),
            })
        self.journal.add_signals(latest_bar, sig_rows)

        # ---- 1. manage existing positions ---------------------------------
        held = self.broker.positions()
        # positions in pairs no longer configured still count against every limit
        remaining: list[Position] = [pos for pos in held if pos.pair not in frames]
        timed_out_pairs: set[str] = set()
        for pos in (x for x in held if x.pair in frames):
            row = rows[pos.pair]
            bars_held = int((inds[pos.pair].index >= pos.open_time[:10]).sum()) if pos.open_time else 0
            timed_out = bool(p.max_hold_days) and bars_held >= p.max_hold_days
            if timed_out or bool(row["exit_long" if pos.side > 0 else "exit_short"]):
                why = f"held {bars_held} days" if timed_out else "exit rule"
                if timed_out:
                    timed_out_pairs.add(pos.pair)
                self._act("close", pos.pair, f"CLOSE {pos.pair} {'LONG' if pos.side > 0 else 'SHORT'} ({why})")
                summary["actions"].append(("close", pos.pair))
                if not cfg.dry_run and not self.broker.close(pos):
                    self._event("order", f"CLOSE FAILED {pos.pair}", pos.pair, "ERROR")
                    remaining.append(pos)  # still exposed: keep counting it
                continue
            base_sl = pos.sl or initial_stop(pos.side, pos.entry, row["atr"], p.atr_stop_mult)
            new_sl = (trail_stop(pos.side, base_sl, row["high"], row["low"], row["atr"], p.atr_stop_mult)
                      if p.trails else base_sl)      # a missing stop is always restored
            improved = (pos.sl == 0) or (new_sl > pos.sl if pos.side > 0 else new_sl < pos.sl)
            if improved:
                self._act("trail", pos.pair, f"TRAIL {pos.pair} SL {pos.sl:.5f} -> {new_sl:.5f}")
                summary["actions"].append(("trail", pos.pair))
                if not cfg.dry_run:
                    if self.broker.modify_sl(pos, new_sl):
                        pos.sl = new_sl
                    else:
                        self._event("order", f"SL MODIFY FAILED {pos.pair}", pos.pair, "ERROR")
            remaining.append(pos)

        # ---- 2. entries ----------------------------------------------------
        open_list = [(pos.pair, pos.side) for pos in remaining]
        open_pairs = {pair for pair, _ in open_list}
        net = net_exposure(open_list)

        def skip(pair, why):
            logger.info("%s skipped: %s", pair, why)
            self._event("skip", f"{pair} entry skipped: {why}", pair)

        if not halted:
            for pair, row in rows.items():
                want = int(row["signal"])
                if want == 0 or pair in open_pairs or pair in timed_out_pairs:
                    continue
                if not streak_has_room(streak, len(open_list), limits):
                    skip(pair, f"streak guard: {streak} losses + {len(open_list)} open positions, one more "
                               f"could pass {limits.max_consecutive_losses} losses in a row")
                    continue
                if not bool(row["entry_ok"]):
                    skip(pair, f"signal too weak ({row['strength']:+.2f} ATR < "
                               f"{p.min_entry_strength_atr} required)")
                    continue
                if in_cooldown(self.journal.last_loss_close(pair), now, p.cooldown_days):
                    skip(pair, f"cooling down after a loss ({p.cooldown_days}d)")
                    continue
                blocked = entry_block_reason(pair, want, open_list, limits, net)
                if blocked:
                    skip(pair, blocked)
                    continue
                legs = currency_legs(pair, want)
                q = self.broker.quote(pair)
                if q is None:
                    skip(pair, "no quote")
                    continue
                if q.spread_pips > cfg.max_spread_pips:
                    skip(pair, f"spread {q.spread_pips:.1f} pips > {cfg.max_spread_pips}")
                    continue
                entry = q.ask if want > 0 else q.bid
                stop = initial_stop(want, entry, row["atr"], p.atr_stop_mult)
                lots = min(self.broker.lots_for_risk(pair, want, size_eq * risk_pct / 100.0, entry, stop),
                           cfg.max_lots_per_order)
                if lots <= 0:
                    skip(pair, "risk-sized lot below broker minimum")
                    continue
                self._act("open", pair, f"OPEN {pair} {'LONG' if want > 0 else 'SHORT'} {lots} lots "
                                        f"@~{entry:.5f} SL {stop:.5f} (risk {risk_pct}%)")
                if not cfg.dry_run and self.broker.open(pair, want, lots, stop) is None:
                    self.notify(f"FX trend bot: ORDER FAILED {pair}")
                    self._event("order", f"OPEN FAILED {pair}", pair, "ERROR")
                    summary.setdefault("failed", []).append(pair)
                    continue
                summary["actions"].append(("open", pair))
                open_pairs.add(pair)
                open_list.append((pair, want))
                for c, s in legs.items():
                    net[c] = net.get(c, 0) + s

        # ---- persist ------------------------------------------------------
        self.journal.add_cycle(latest_bar, equity, daily_pct, weekly_pct, halted, len(open_pairs),
                               "halted" if halted else "ok")
        # A failed order (e.g. AutoTrading switched off in MT5) must not consume the bar:
        # leave last_bar unset so the next poll retries. Already-open pairs are not re-sent.
        state.update(prev_equity=equity, last_run=now.isoformat(timespec="seconds"))
        if not summary.get("failed"):
            state["last_bar"] = latest_bar
        else:
            state.pop("last_bar", None)
            summary["retry"] = True
        _save(state_path, state)
        self.sync_trades(now, force=True)
        summary["equity"] = equity
        return summary
