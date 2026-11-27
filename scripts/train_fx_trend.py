"""
Run one retraining by hand and print the result. Changes nothing the bot uses.

  python scripts/train_fx_trend.py --csv-dir data/fx_daily
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from run_fx_trend_backtest import DEFAULT_PAIRS, live_settings, load_csv_dir
from strategy.fx_autotrain import TrainSettings, train
from strategy.fx_trend import TrendParams
from strategy.fx_trend_backtest import BtConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, default=ROOT / "data" / "fx_daily")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "fx_trend.yaml")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    args = ap.parse_args()

    frames = load_csv_dir(args.csv_dir, args.pairs)
    if not frames:
        sys.exit("No data loaded.")
    strategy, risk = live_settings(args.config)
    auto = yaml.safe_load(args.config.read_text(encoding="utf-8"))["fx_trend"].get("autotrain", {})
    r = train(frames, TrendParams(**strategy), BtConfig(**risk),
              TrainSettings(**auto.get("gates", {})))

    print(f"\nDecision: {r['decision'].upper()}\n{r['reason']}")
    print(f"Unseen validation data: {r['validation_from']} to {r['data_to']}")
    for name in ("champion", "challenger"):
        e = r[name]
        if not e:
            print(f"\n{name}: none (no candidate made money in training)")
            continue
        n = e["normal"]
        print(f"\n{name}: {e['label']}\n  return {n.get('total_return_pct')}%  maxDD {n.get('max_drawdown_pct')}%"
              f"  PF {n.get('profit_factor')}  trades {n.get('n_trades')}"
              f"  | higher costs: {e['stressed'].get('total_return_pct')}%")
        print("  PASSES" if e["passes"] else "  FAILS: " + ", ".join(e["fail_reasons"]))
    wf = r["walk_forward"]
    if wf["ok"]:
        o = wf["oos"]
        print(f"\nWalk-forward of the selection method: return {o.get('total_return_pct')}%  "
              f"maxDD {o.get('max_drawdown_pct')}%  PF {o.get('profit_factor')}  "
              f"profitable years {wf['profitable_folds']}/{wf['n_folds']}")
    out = ROOT / "data" / "fx_trend_training.json"
    out.write_text(json.dumps(r, indent=2, default=float))
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
