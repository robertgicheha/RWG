"""
Export daily bars from your MT5 terminal to CSV, for the backtest.

  python scripts/export_mt5_history.py --out data/fx_daily --bars 5000
  python scripts/run_fx_trend_backtest.py --csv-dir data/fx_daily --holdout-start 2022-01-01 --walk-forward

Test against the SAME broker you will trade, because spreads and rollover times
differ between brokers. In MT5 you may need to scroll the chart back (or raise
Tools > Options > Charts > Max bars) so the terminal holds enough history.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from data_feeds.mt5_feed import MT5Feed

DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "EURJPY", "GBPJPY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "fx_daily")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--suffix", default="", help="broker symbol suffix, e.g. m or .r")
    args = ap.parse_args()

    feed = MT5Feed()
    if not feed.connect():
        sys.exit("Could not connect to the MT5 terminal (is it running and logged in?).")
    args.out.mkdir(parents=True, exist_ok=True)
    for pair in args.pairs:
        df = feed.get_ohlcv(pair + args.suffix, "1d", args.bars)
        if df.empty:
            print(f"  {pair}: no data (check the symbol name / suffix, and that it is in Market Watch)")
            continue
        out = df[["open", "high", "low", "close"]].copy()
        out.index.name = "time"
        out.to_csv(args.out / f"{pair}.csv")
        print(f"  {pair}: {len(out)} bars  {out.index[0].date()} -> {out.index[-1].date()}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
