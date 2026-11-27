"""
Real trading costs vs what the backtest assumes. PURE: no broker, no I/O, no clock.

Three costs, each compared on exactly the same trades:
  price      what a fill cost against the mid price quoted when the order was sent
             (half the spread + slippage). Needs the quote the bot records at send
             time, so orders sent before that was added cannot be measured.
  stops      broker-side stop-loss fills: how far past the stop level they filled.
  commission commission + fees. The backtest assumes none.
  overnight  swap per calendar night held. Positive numbers are a cost.

Sign convention everywhere: positive = costs you money.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from strategy.fx_trend_backtest import BtConfig

MIN_FILLS = 30            # price cost
MIN_STOP_FILLS = 10
MIN_COMMISSION_FILLS = 5  # a fixed tariff: a few fills show it
MIN_NIGHTS = 100          # position-nights of swap
TOLERANCE = 0.10          # within 10% of the assumption counts as "in line"
_STOP_LEVEL = re.compile(r"\[sl\s+([0-9.]+)\]", re.I)


def _day(ts: int):
    return datetime.fromtimestamp(ts, timezone.utc).date()


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _money(pips: float, f: dict) -> float:
    """A cost in pips on one fill, as account currency."""
    return pips * f["pip"] / f["price"] * f["notional"]


def _to_pips(money: float, f: dict) -> float:
    return money / f["notional"] / (f["pip"] / f["price"])


def _half_spread(pair: str, model: BtConfig) -> float:
    return 0.5 * model.spread_pips.get(pair, model.default_spread_pips)


def _component(name, unit, n, needed, actual, assumed, actual_money, assumed_money) -> dict:
    if not n:
        status = "no data"
    elif n < needed:
        status = "too little data"
    elif actual > assumed + max(abs(assumed) * TOLERANCE, 1e-9):
        status = "higher than assumed"
    elif actual < assumed - max(abs(assumed) * TOLERANCE, 1e-9):
        status = "lower than assumed"
    else:
        status = "in line"

    def r(x, d=3):
        return None if x is None else round(x, d)

    return {"name": name, "unit": unit, "n": n, "needed": needed, "status": status,
            "enough": n >= needed, "actual": r(actual), "assumed": r(assumed),
            "actual_money": r(actual_money, 2), "assumed_money": r(assumed_money, 2)}


def build_report(fills: list[dict], executions: list[dict], open_positions: list[dict],
                 model: BtConfig | None = None, server_now: int = 0) -> dict:
    """
    fills:      {order, position_id, pair, side (+1 bought / -1 sold), is_entry, is_stop,
                 volume, price, pip, notional, commission, swap, fee, time, comment}
                commission / swap / fee keep MT5's sign (negative = charged).
    executions: {order, pair, side, kind, bid, ask, day_open} recorded at send time.
    open_positions: {position_id, swap, notional, time}
    """
    model = model or BtConfig()
    sent = {e["order"]: e for e in executions if e.get("order")}
    rows, price, stops = [], [], []

    for f in sorted(fills, key=lambda f: f["time"]):
        if f["notional"] <= 0 or f["price"] <= 0 or f["pip"] <= 0:
            continue
        assumed = _half_spread(f["pair"], model) + model.slippage_pips
        row = {"time": datetime.fromtimestamp(f["time"], timezone.utc).strftime("%Y-%m-%d %H:%M"),
               "pair": f["pair"], "side": f["side"], "action": "open" if f["is_entry"] else "close",
               "volume": f["volume"], "price": f["price"], "price_cost_pips": None,
               "spread_pips": None, "since_day_open_pips": None, "past_stop_pips": None,
               "commission_pips": round(_to_pips(-(f["commission"] + f["fee"]), f), 3)}
        e = sent.get(f["order"])
        level = _STOP_LEVEL.search(f.get("comment") or "") if f["is_stop"] else None
        if e and e["bid"] > 0 and e["ask"] > 0:
            mid = (e["bid"] + e["ask"]) / 2
            cost = f["side"] * (f["price"] - mid) / f["pip"]
            row["price_cost_pips"] = round(cost, 3)
            row["spread_pips"] = round((e["ask"] - e["bid"]) / f["pip"], 3)
            if e.get("day_open"):   # MT5 bars are BID prices, so compare bid with bid
                row["since_day_open_pips"] = round(f["side"] * (e["bid"] - e["day_open"]) / f["pip"], 2)
            price.append((f, cost, assumed, row["spread_pips"]))
        elif level:
            past = f["side"] * (f["price"] - float(level.group(1))) / f["pip"]
            row["past_stop_pips"] = round(past, 3)
            stops.append((f, past, assumed))
        rows.append(row)

    usable = [f for f in fills if f["notional"] > 0 and f["price"] > 0 and f["pip"] > 0]
    commission_money = sum(-(f["commission"] + f["fee"]) for f in usable)

    # ---- overnight: one entry per position, closed or still open ----
    positions: dict[int, list[dict]] = {}
    for f in usable:
        positions.setdefault(f["position_id"], []).append(f)
    held = []                                 # (swap cost, notional, nights)
    open_ids = {p["position_id"] for p in open_positions}
    for pid, group in positions.items():
        ins = [f for f in group if f["is_entry"]]
        outs = [f for f in group if not f["is_entry"]]
        if pid in open_ids or not ins or not outs:
            continue
        nights = (_day(max(f["time"] for f in outs)) - _day(min(f["time"] for f in ins))).days
        if nights > 0:
            held.append((-sum(f["swap"] for f in group), sum(f["notional"] for f in ins), nights))
    for p in open_positions:
        nights = (_day(server_now) - _day(p["time"])).days if server_now else 0
        if nights > 0 and p["notional"] > 0:
            held.append((-p["swap"], p["notional"], nights))
    nights = sum(h[2] for h in held)
    exposure = sum(h[1] * h[2] for h in held)      # notional x nights
    swap_money = sum(h[0] for h in held)

    components = [
        _component("Spread + slippage on bot orders", "pips per fill", len(price), MIN_FILLS,
                   _mean(c for _, c, _, _ in price), _mean(a for _, _, a, _ in price),
                   sum(_money(c, f) for f, c, _, _ in price) if price else None,
                   sum(_money(a, f) for f, _, a, _ in price) if price else None),
        _component("Stop-loss fills past the stop", "pips per fill", len(stops), MIN_STOP_FILLS,
                   _mean(c for _, c, _ in stops), _mean(a for _, _, a in stops),
                   sum(_money(c, f) for f, c, _ in stops) if stops else None,
                   sum(_money(a, f) for f, _, a in stops) if stops else None),
        _component("Commission and fees", "pips per fill", len(usable), MIN_COMMISSION_FILLS,
                   _mean(_to_pips(-(f["commission"] + f["fee"]), f) for f in usable),
                   0.0 if usable else None,
                   commission_money if usable else None, 0.0 if usable else None),
        _component("Overnight swap", "bps of position per night", nights, MIN_NIGHTS,
                   swap_money / exposure * 1e4 if exposure else None,
                   model.swap_bps_per_day if exposure else None,
                   swap_money if exposure else None,
                   exposure * model.swap_bps_per_day / 1e4 if exposure else None),
    ]
    measured = [c for c in components if c["n"]]
    actual = sum(c["actual_money"] for c in measured)
    assumed = sum(c["assumed_money"] for c in measured)

    higher = [c["name"] for c in components if c["status"] == "higher than assumed"]
    waiting = [c["name"] for c in components if not c["enough"]]
    if higher:
        verdict = "Costs above the backtest's assumption: " + ", ".join(higher) + "."
    elif waiting:
        verdict = "Not enough data to judge yet."
    else:
        verdict = "No measured cost is above the backtest's assumption."
    if waiting:
        verdict += " Still collecting: " + ", ".join(waiting) + "."

    spreads = {}
    for f, _, _, spread in price:
        spreads.setdefault(f["pair"], []).append(spread)
    drift = [r["since_day_open_pips"] for r in rows if r["since_day_open_pips"] is not None]
    return {
        "verdict": verdict, "components": components,
        "total": {"actual_money": round(actual, 2), "assumed_money": round(assumed, 2)},
        "spreads": [{"pair": p, "n": len(v), "actual_pips": round(_mean(v), 2), "max_pips": round(max(v), 2),
                     "assumed_pips": round(2 * _half_spread(p, model), 2)}
                    for p, v in sorted(spreads.items())],
        "since_day_open": {"n": len(drift), "mean_pips": round(_mean(drift), 2) if drift else None},
        "price_slippage_pips": (None if not price else round(
            _mean(c - _half_spread(f["pair"], model) for f, c, _, _ in price), 3)),
        "fills": rows[::-1],
        "assumptions": {"slippage_pips": model.slippage_pips, "swap_bps_per_day": model.swap_bps_per_day,
                        "commission": 0.0},
    }


def measured_model(report: dict, model: BtConfig | None = None) -> tuple[BtConfig, list[str]]:
    """The backtest's costs with measured values swapped in wherever there is ENOUGH data.
    Commission is added as extra pips per fill, which costs the same as charging it."""
    from dataclasses import replace
    model = model or BtConfig()
    price, _, commission, swap = report["components"]
    slippage, used = model.slippage_pips, []
    if price["enough"]:
        slippage = report["price_slippage_pips"]
        used.append(f"spread and slippage ({price['n']} fills)")
    if commission["enough"]:
        slippage += commission["actual"]
        used.append(f"commission {commission['actual']} pips per fill ({commission['n']} fills)")
    bps = model.swap_bps_per_day
    if swap["enough"]:
        bps = swap["actual"]
        used.append(f"swap {bps} bps per night ({swap['n']} nights)")
    return replace(model, slippage_pips=slippage, swap_bps_per_day=bps), used


def to_markdown(r: dict) -> str:
    def v(x):
        return "-" if x is None else f"{x:g}"

    out = ["# FX bot: real costs vs backtest assumptions", "",
           f"Generated {r.get('generated', '-')} from account {r.get('server', '-')}. "
           "Positive numbers cost money.", "", f"**{r['verdict']}**", "",
           "| Cost | Actual | Assumed | Unit | Sample | Actual money | Assumed money | Status |",
           "|---|---|---|---|---|---|---|---|"]
    for c in r["components"]:
        out.append(f"| {c['name']} | {v(c['actual'])} | {v(c['assumed'])} | {c['unit']} | "
                   f"{c['n']} (need {c['needed']}) | {v(c['actual_money'])} | {v(c['assumed_money'])} | {c['status']} |")
    t = r["total"]
    out += ["", f"Total over what could be measured: **{t['actual_money']:.2f} actual** vs "
                f"{t['assumed_money']:.2f} assumed ({r.get('currency', 'account currency')}). "
                "Components cover different samples, so this is not the cost of one set of trades."]
    if r["spreads"]:
        out += ["", "## Spread when the bot sent orders (pips)", "",
                "| Pair | Orders | Average | Widest | Assumed |", "|---|---|---|---|---|"]
        out += [f"| {s['pair']} | {s['n']} | {s['actual_pips']} | {s['max_pips']} | {s['assumed_pips']} |"
                for s in r["spreads"]]
    d = r["since_day_open"]
    if d["n"]:
        out += ["", f"**Delay:** the backtest fills at the daily open. When the bot sent its orders, price had "
                    f"moved {d['mean_pips']} pips against it on average since that open ({d['n']} orders; "
                    "negative = in its favour). This includes ordinary market movement and is not in the "
                    "table above."]
    if r.get("backtest"):
        b = r["backtest"]
        out += ["", "## Backtest rerun with measured costs", "",
                "Replaced: " + (", ".join(b["replaced"]) or "nothing (not enough data)") + ".", "",
                "| | Assumed costs | Measured costs |", "|---|---|---|"]
        out += [f"| {k} | {b['assumed'].get(k)} | {b['measured'].get(k)} |"
                for k in ("total_return_pct", "cagr_pct", "max_drawdown_pct", "profit_factor", "n_trades")]
    if r["fills"]:
        out += ["", "## Latest fills (broker server time)", "",
                "| Time | Pair | Order | Lots | Price | Spread + slippage | Past stop | Commission |",
                "|---|---|---|---|---|---|---|---|"]
        out += [f"| {f['time']} | {f['pair']} | {f['action']} {'buy' if f['side'] > 0 else 'sell'} | "
                f"{f['volume']} | {f['price']} | {v(f['price_cost_pips'])} | {v(f['past_stop_pips'])} | "
                f"{v(f['commission_pips'])} |" for f in r["fills"][:25]]
    out += ["", "Notes: pip costs are per fill. Money values use today's exchange rates, so they are "
                "approximate for non-USD pairs. Commission is rounded to cents, which distorts very small "
                "lots. Demo fills and swap rates can differ from a real-money account."]
    return "\n".join(out) + "\n"
