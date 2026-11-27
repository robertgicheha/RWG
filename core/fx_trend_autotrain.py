"""
Schedules strategy/fx_autotrain.py inside the live bot and applies its decision.

  * Bars are downloaded on the bot's own thread (MT5 calls stay on one thread);
    the number crunching runs in a background thread so polling continues.
  * The result is saved to data/fx_trend_model.json. config/fx_trend.yaml is
    never rewritten. Editing the strategy section there discards the saved model.
  * New settings take over only when the bot holds NO positions, so a trade is
    always closed by the rules that opened it. Entries wait until then.
  * On a real-money account with orders on, a new strategy is only PROPOSED
    unless auto_apply_live is set. Pausing entries applies everywhere.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from core.fx_trend_runner import FxTrendRunner, _load, _save
from strategy.fx_autotrain import (
    TrainSettings, apply_signal_params, label, signal_params, train,
)
from strategy.fx_trend_backtest import BtConfig

logger = logging.getLogger("fx_trend_autotrain")


@dataclass
class AutoTrainConfig:
    enabled: bool = False
    every_days: int = 30
    retry_hours: int = 24                 # after a failed run
    history_bars: int = 5000
    min_days_between_changes: int = 90
    auto_apply: bool = True               # dry-run and demo accounts
    auto_apply_live: bool = False         # real money: off unless you turn it on
    on_no_edge: str = "pause"             # "pause" new entries, or "keep" trading
    model_file: str = "data/fx_trend_model.json"
    settings: TrainSettings = field(default_factory=TrainSettings)

    def __post_init__(self):
        if self.on_no_edge not in ("pause", "keep"):
            raise ValueError(f"autotrain.on_no_edge must be 'pause' or 'keep', not {self.on_no_edge!r}")


class AutoTrainer:
    def __init__(self, runner: FxTrendRunner, cfg: AutoTrainConfig,
                 bt: BtConfig | None = None, train_fn=train, background: bool = True):
        self.runner, self.cfg, self.train_fn, self.background = runner, cfg, train_fn, background
        self.bt = bt or BtConfig()
        self.base_params = runner.cfg.params          # what the YAML says
        self.path = Path(runner.base) / cfg.model_file
        self._thread: threading.Thread | None = None
        self._result: dict | None = None
        self.model = _load(self.path)
        # round-trip through JSON so the comparison matches what was saved
        self._fingerprint = json.loads(json.dumps(asdict(self.base_params)))
        if self.model.get("base") != self._fingerprint:
            self.model = {"base": self._fingerprint}
        self._apply()

    # ------------------------------------------------------------------
    def _apply(self) -> None:
        """Push the saved model into the runner."""
        if self.model.get("params"):
            self.runner.cfg.params = apply_signal_params(self.base_params, self.model["params"])
        hold = None
        if self.model.get("pending"):
            hold = "new settings waiting for open positions to close"
        elif self.model.get("no_edge") and self.cfg.on_no_edge == "pause":
            hold = "retraining found no strategy that passes: " + self.model.get("reason", "")
        self.runner.entry_hold = hold

    def _say(self, msg: str, level: str = "INFO") -> None:
        logger.log(getattr(logging, level), msg)
        self.runner._event("train", msg, level=level)
        self.runner.notify("FX trend bot: " + msg)

    def _switch_if_flat(self, now: datetime) -> None:
        pending = self.model.get("pending")
        if not pending:
            return
        mine = self.runner.broker.positions()
        if mine and not self.runner.cfg.dry_run:
            return
        self.model.update(params=pending, pending=None, no_edge=False,
                          changed_at=now.isoformat(timespec="seconds"))
        _save(self.path, self.model)
        self._apply()
        self._say(f"now trading: {label(self.runner.cfg.params)}")

    def _due(self, now: datetime) -> bool:
        def since(key):
            return now - datetime.fromisoformat(self.model[key]) if self.model.get(key) else None
        failed, done = since("failed_at"), since("trained_at")
        if failed is not None and failed < timedelta(hours=self.cfg.retry_hours):
            return False
        return done is None or done >= timedelta(days=self.cfg.every_days)

    def _may_promote(self, now: datetime) -> bool:
        changed = self.model.get("changed_at")
        if not changed or self.model.get("no_edge"):
            return True
        return now - datetime.fromisoformat(changed) >= timedelta(days=self.cfg.min_days_between_changes)

    def _download(self, now: datetime) -> dict:
        """Enough fresh history for most pairs, or raise: never train on a thin sample."""
        s = self.cfg.settings
        need_days = (s.train_years + s.validation_years + 1) * 365
        fresh_after = (now - timedelta(days=self.runner.cfg.max_data_age_days)).strftime("%Y-%m-%d")
        frames = {}
        for pair in self.runner.cfg.pairs:
            df = self.runner.broker.daily_bars(pair, self.cfg.history_bars)
            if (len(df) and (df.index[-1] - df.index[0]).days >= need_days
                    and df.index[-1].strftime("%Y-%m-%d") >= fresh_after):
                frames[pair] = df
        if len(frames) < 0.75 * len(self.runner.cfg.pairs):
            raise RuntimeError(f"only {len(frames)}/{len(self.runner.cfg.pairs)} pairs have {need_days // 365} "
                               "years of fresh daily history (raise Max bars in MT5 options)")
        return frames

    def _work(self, frames, champion, may_promote) -> None:
        try:
            self._result = self.train_fn(frames, champion, self.bt, self.cfg.settings,
                                         may_promote=may_promote)
        except Exception as e:
            logger.exception("training failed")
            self._result = {"error": str(e)}

    def _fail(self, now: datetime, why: str) -> None:
        self.model["failed_at"] = now.isoformat(timespec="seconds")
        _save(self.path, self.model)
        self._say(f"retraining failed, current settings unchanged: {why}", "WARNING")

    def _finish(self, now: datetime) -> None:
        result, self._result, self._thread = self._result, None, None
        if "error" in result:
            self._fail(now, result["error"])
            return
        decision = result["decision"]
        self.model.update(trained_at=now.isoformat(timespec="seconds"), failed_at=None,
                          reason=result["reason"], report=result, proposal=None, pending=None,
                          no_edge=decision == "no_edge")
        if decision == "promote":
            new = result["challenger"]["params"]
            broker = self.runner.broker
            live = not self.runner.cfg.dry_run and not broker.is_demo()
            if self.cfg.auto_apply_live if live else self.cfg.auto_apply:
                self.model["pending"] = new
            else:
                self.model["proposal"] = new
                # the champion may itself be failing: do not call that healthy
                self.model["no_edge"] = not result["champion"]["passes"]
        _save(self.path, self.model)
        self._apply()
        note = " (proposal only, not applied)" if self.model.get("proposal") else ""
        self._say(f"retraining: {decision.upper()}{note}. {result['reason']}",
                  "WARNING" if decision == "no_edge" else "INFO")

    # ------------------------------------------------------------------
    def poll(self, now: datetime) -> None:
        """Call once per bot poll, before runner.tick(). Never raises."""
        if not self.cfg.enabled:
            return
        try:
            if self._thread is not None and not self._thread.is_alive():
                self._finish(now)
            if self._thread is None and self._due(now):
                frames = self._download(now)
                logger.info("retraining started on %d pairs", len(frames))
                self._thread = threading.Thread(
                    target=self._work, daemon=True,
                    args=(frames, self.runner.cfg.params, self._may_promote(now)))
                self._thread.start()
                if not self.background:
                    self._thread.join()
                    self._finish(now)
            self._switch_if_flat(now)
        except Exception as e:
            logger.exception("auto-train error")
            self._thread = None
            try:
                self._fail(now, str(e))
            except Exception:
                logger.exception("could not record the training failure")

    def current(self) -> dict:
        return signal_params(self.runner.cfg.params)
