# FX Trend Bot

Daily FX bot for majors/crosses on MT5, with an admin dashboard.
No strategy is guaranteed to profit. Validate on a demo account first.

## Strategy and evidence (measured 2026-11-26)
Active style: `pullback` (config/fx_trend.yaml). Long when close > SMA200 and RSI(2) < 5, short
mirrored. Exit when close crosses back over SMA5, after 5 daily bars, or at the 2 x ATR stop.

Portfolio backtest, 8 configured pairs, about 18 years, 0.5% risk, modelled costs:

| | old `trend` rules | `pullback` + breakers |
|---|---|---|
| Total return | -36.1% | +7.3% |
| Per year (CAGR) | -2.2% | +0.35% |
| Max drawdown | -42.6% | -7.9% |
| Profit factor | 0.85 | 1.07 |
| Win rate | 31% | 61% |

**Do not read this as a reliable profit source.**
- With 1.0 pip slippage and double swap, the result is break-even (-0.3%).
- It was picked from 36 variants using both halves of the history, so those numbers flatter it.
- On four pairs not used in that selection (NZDUSD, EURGBP, AUDJPY, EURCHF) per-trade profit
  factor was 1.10 at stop 2 (t = 0.8: not distinguishable from zero), 0.98 at double costs.
- Doubling risk to 1% barely raised the return and about doubled the drawdown.

## Automatic retraining (`autotrain:` in the config)
Every 30 days the running bot downloads daily history and:
1. Picks one challenger from 28 candidates (pullback, band reversion, Donchian breakout, trend;
   some pullbacks with a volatility filter) using a 6-year train window.
2. Tests the challenger and the current settings on the following 2 years, at normal and higher costs.
3. Walk-forwards the whole picking method. If it lost money out-of-sample, nothing is promoted.

| Decision | Effect |
|---|---|
| `promote` | New settings apply once the bot holds no positions (entries wait until then) |
| `keep` | No change |
| `no_edge` | Current settings fail and nothing replaces them: new entries pause, exits keep running |

On a real-money account with orders on, promotion is only a proposal unless `auto_apply_live: true`.
Results: `data/fx_trend_model.json`, the dashboard events list and `fx_trend.log`.
A failed run changes nothing and is retried after 24 hours. To retrain now, delete the model file
and restart (this also forgets any adopted settings). Editing `strategy:` also discards them.

**First measured run (2026-11-27, `data/fx_daily`): `keep`.**
- Current settings, 2024-11-25 to 2026-11-25: +4.3%, max drawdown -2.1%, PF 1.58, 86 trades.
  This period is not truly unseen for them: they were chosen using all history.
- Walk-forward of the picking method: **-23.0%**, PF 0.69, 3 of 13 years profitable.
  Choosing recent winners has not worked, so promotion is currently blocked. It stays blocked
  while this number is not positive. Passing the gates would still not promise profit.

## Circuit breakers (`risk:` in the config)
- Every order carries a stop-loss. Gaps and slippage can make a loss larger than planned.
- 4 losses in a row: 5-day pause, then half risk until a win.
- 7 losses in a row, or 8% below peak equity: entries LOCK until
  `python scripts/run_fx_trend_bot.py --reset-breaker`. Exits and stops keep working.
- Streak guard: an entry is refused if losing it plus all open positions would exceed 7.
  So 8 in a row cannot happen without a reset. A reset starts a new count, so a loss right
  after one can be the 8th in a row in the account history.
- Breakers count closed trades, so they never trigger in dry-run (nothing closes there).
- The daily, weekly and drawdown limits use whole-account equity, including other bots' trades.
- Stale prices (older than 5 days), unreadable equity or missing data for a held pair: no action.

## Files
| File | Role |
|---|---|
| `strategy/fx_trend.py` | Pure strategy functions (signals, stops, sizing) |
| `strategy/fx_trend_backtest.py` | Portfolio backtest with costs, risk gates, Monte Carlo |
| `strategy/fx_autotrain.py` | Retraining and champion/challenger gates (pure) |
| `core/fx_trend_autotrain.py` | Schedules retraining in the live bot and applies the decision |
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
`pytest`

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
The backtest script reads the strategy and risk settings from `config/fx_trend.yaml`.

3. Only if the filters win out-of-sample, enable them in `config/fx_trend.yaml`
   (`min_entry_strength_atr`, `cooldown_days`). They are OFF by default.

## One source of truth for risk rules
`strategy/fx_risk.py` holds the entry gates (max positions, per-currency exposure cap, daily/weekly
loss halts, cooldown). The backtest and the live runner both call it, so they cannot drift apart.
