# RWG

Daily trend-following FX bot for MetaTrader 5, with a backtester and an admin dashboard.

No strategy is guaranteed to make money, and most retail FX accounts lose. The bot is
**dry-run by default**, refuses live accounts unless you pass explicit CLI flags, and
should be validated on a demo account for 2-3 months first.

See [docs/FX_TREND_BOT.md](docs/FX_TREND_BOT.md) for the file map, workflow and control files.

## Quick start
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (MetaTrader5 package is Windows-only)
pip install -r requirements.txt
cp .env.example .env                                  # set FX_ADMIN_TOKEN, MT5_* creds

python scripts/export_mt5_history.py --out data/fx_daily
python scripts/run_fx_trend_backtest.py --csv-dir data/fx_daily --holdout-start 2022-01-01 --walk-forward
python -m dashboard.fx_admin                          # http://127.0.0.1:8090
python scripts/run_fx_trend_bot.py                    # dry-run
pytest
```

## Layout
- `strategy/` pure strategy + backtester
- `core/` MT5 adapter, runner, journal, analytics, process supervisor
- `dashboard/` FastAPI admin dashboard (token-protected, localhost by default)
- `scripts/` bot and backtest entrypoints
- `config/fx_trend.yaml` all settings
