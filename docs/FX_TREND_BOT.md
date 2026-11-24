# FX Trend Bot

Daily time-series-momentum bot for FX majors/crosses on MT5, with an admin dashboard.
No strategy is guaranteed to profit. Validate on a demo account first.

## Files
| File | Role |
|---|---|
| `strategy/fx_trend.py` | Pure strategy functions (signals, stops, sizing) |
| `strategy/fx_trend_backtest.py` | Portfolio backtest with costs, risk gates, Monte Carlo |
| `core/fx_trend_runner.py` | Live decision cycle (idempotent, once per closed daily bar) |
| `core/fx_trend_broker.py` | MT5 adapter (own magic number only) |
| `core/fx_trend_journal.py` | SQLite journal: cycles, signals, events, trades |
| `core/fx_trend_analytics.py` | Performance + predictive analytics |
| `core/fx_trend_supervisor.py` | Start / stop / restart / flatten the bot process |
| `dashboard/fx_admin.py` + `templates/fx_admin.html` | Admin dashboard |
| `config/fx_trend.yaml` | All settings |

## Workflow
1. `python scripts/run_fx_trend_backtest.py --csv-dir <dir> --holdout-start 2022-01-01` (or `--mt5`).
   Writes `data/fx_trend_backtest.json` (shown on the dashboard).
2. Set `FX_ADMIN_TOKEN` in `.env`, then `python -m dashboard.fx_admin` -> http://127.0.0.1:8090
   (binds to localhost; use an SSH tunnel or a TLS reverse proxy for remote access).
3. Start the bot from the dashboard's Control tab (dry-run), or:
   `python scripts/run_fx_trend_bot.py` (dry-run) / `--send-orders` (demo accounts only).
4. After 2-3 months of demo results that match the backtest, consider live:
   `--send-orders --allow-live --i-understand-the-risk` (CLI only; the dashboard cannot do this).

## Control files (in `data/`)
`fx_trend.STOP` graceful exit, positions stay open. `fx_trend.KILL` flatten this bot's positions, then exit
(stays until cleared). Logs: `data/logs/fx_trend.log`, `fx_trend_errors.log`, `fx_trend_console.log`.

## Tests
`pytest tests/test_fx_trend.py tests/test_fx_trend_runner.py tests/test_fx_admin.py`

## Validating the strategy honestly
1. Export real history from the broker you will trade:
   `python scripts/export_mt5_history.py --out data/fx_daily --bars 5000`
2. Backtest with a holdout and walk-forward:
   `python scripts/run_fx_trend_backtest.py --csv-dir data/fx_daily --holdout-start 2022-01-01 --walk-forward`
   - **Holdout:** look at it once; re-tuning afterwards makes it in-sample.
   - **Walk-forward:** re-optimises on 4 years, tests on the next unseen year, rolling. Read the
     out-of-sample numbers, the *efficiency* (OOS/in-sample Sharpe; below ~0.5 = curve-fitting) and
     *parameter stability* (a winner that changes every fold means no real edge).
   - It compares the baseline grid against one with the whipsaw filters on the same folds.
3. Only if the filters win out-of-sample, enable them in `config/fx_trend.yaml`
   (`min_entry_strength_atr`, `cooldown_days`). They are OFF by default.

## One source of truth for risk rules
`strategy/fx_risk.py` holds the entry gates (max positions, per-currency exposure cap, daily/weekly
loss halts, cooldown). The backtest and the live runner both call it, so they cannot drift apart.
