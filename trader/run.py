"""Ciclo completo: datos → Kronos → comité → riesgo → broker → panel.

Uso:
  python -m trader.run                 # ciclo de hoy (tras el cierre de EE. UU.)
  python -m trader.run --demo          # datos sintéticos + motor bootstrap, sin red
  python -m trader.run --replay 60     # rehace los últimos 60 días sin mirar el futuro
                                       # (útil para estrenar la cartera paper y probar el panel)
  python -m trader.run --reset         # borra la cartera paper y empieza de cero
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from datetime import datetime, timezone

import yaml

from .agents import decide
from .broker import IBKRBroker, PaperBroker
from .data import get_candles
from .forecast import load_engine, run_forecast
from .risk import RiskManager

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_cfg(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def emit(log: list[str], line: str):
    """Guarda la línea para el panel y la imprime al momento (progreso visible en GitHub Actions)."""
    log.append(line)
    print(line, flush=True)


def cycle(cfg, engine, broker, risk, frames: dict, upto: int | None, demo: bool, log: list[str]) -> dict:
    """Un ciclo de decisión usando solo velas hasta `upto` (None = todas)."""
    universe = cfg["universe"]
    view = {t: (df.iloc[: upto] if upto is not None else df) for t, df in frames.items()}
    date = view[universe[0]].index[-1].strftime("%Y-%m-%d")
    run_id = f"{date}-{int(time.time()) % 100000}"
    prices = {t: float(df["close"].iloc[-1]) for t, df in view.items()}
    equity = broker.equity(prices) if hasattr(broker, "equity") else None

    per = {}
    signals = {}
    for i, t in enumerate(universe):
        df = view[t].tail(cfg["data"]["lookback"])
        t0 = time.time()
        fc = run_forecast(engine, t, df, cfg["forecast"], seed=zlib.crc32(f"{t}{date}".encode()))
        cur_w = (broker.positions.get(t, {}).get("qty", 0) * prices[t] / equity) if equity else 0.0
        sig, feats = decide(cfg["decision"]["mode"], t, df, fc, cur_w,
                            cfg["risk"]["max_position_pct"], cfg["decision"], date)
        signals[t] = sig
        emit(log, f"[{date}] {t:<5} {fc.engine} {fc.stats['n_up']:>2}/{fc.stats['samples']} ↑ "
                   f"E[r] {fc.stats['exp_return']:+.2%} → {sig.action} {sig.target_weight:.0%} "
                   f"conf {sig.confidence:.0%}"
             + (f" · desfase {fc.stats['recon_gap']:+.1%}" if fc.stats.get("recon_gap") is not None else "")
             + f" ({time.time() - t0:.1f}s)")
        per[t] = {"forecast": fc, "signal": sig, "features": feats}

    orders, rlog = risk.build_orders(signals, broker.positions, broker.cash, prices, broker.last_equity()
                                     if hasattr(broker, "last_equity") else None)
    for line in rlog:
        emit(log, f"[{date}] RISK  {line}")
    fills = broker.execute(orders, run_id)
    for f in fills:
        emit(log, f"[{date}] FILL  {f['side']} {f['qty']:g} {f['ticker']} @ {f['price']} · {f['reason']}")
    if hasattr(broker, "mark"):
        broker.mark(prices, prices[cfg["benchmark"]], date)
    return {"date": date, "run_id": run_id, "per": per, "orders": orders, "fills": fills,
            "risk_log": rlog, "prices": prices}


def build_state(cfg, broker, res, frames, engine_name, demo, log) -> dict:
    n = cfg["output"]["candles_in_dashboard"]
    tickers = {}
    for t, d in res["per"].items():
        df = frames[t].loc[: res["date"]].tail(n)
        tickers[t] = {
            "candles": [[i.strftime("%Y-%m-%d"), *(round(float(x), 4) for x in row)]
                        for i, row in zip(df.index, df[["open", "high", "low", "close", "volume"]].values)],
            "forecast": d["forecast"].to_json(),
            "signal": d["signal"].to_json(),
            "features": {k: round(v, 5) for k, v in d["features"].items()},
        }
    prices = res["prices"]
    eq = broker.equity(prices)
    positions = []
    for t, p in broker.positions.items():
        px = prices.get(t, p["avg_price"])
        positions.append({"ticker": t, "qty": p["qty"], "avg_price": round(p["avg_price"], 4), "price": px,
                          "value": round(p["qty"] * px, 2), "pnl_pct": round(px / p["avg_price"] - 1, 4),
                          "weight": round(p["qty"] * px / eq, 4)})
    trades = broker.book["trades"] if hasattr(broker, "book") else []
    closed = [x for x in trades if x.get("pnl") is not None]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": res["date"],
        "demo": demo,
        "engine": engine_name,
        "decision_mode": cfg["decision"]["mode"],
        "broker": cfg["broker"]["kind"],
        "benchmark": cfg["benchmark"],
        "universe": cfg["universe"],
        "tickers": tickers,
        "portfolio": {
            "starting_cash": broker.book.get("starting_cash") if hasattr(broker, "book") else None,
            "cash": round(broker.cash, 2),
            "equity": round(eq, 2),
            "positions": sorted(positions, key=lambda x: -x["value"]),
            "equity_history": broker.book["equity_history"][-400:] if hasattr(broker, "book") else [],
            "n_trades": len(trades),
            "win_rate": round(sum(1 for x in closed if x["pnl"] > 0) / len(closed), 3) if closed else None,
        },
        "last_orders": [o.to_json() for o in res["orders"]],
        "trades": trades[-60:],
        "risk": {k: v for k, v in cfg["risk"].items()},
        "log": log[-120:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--demo", action="store_true", help="datos sintéticos, sin red")
    ap.add_argument("--replay", type=int, default=0, help="rehacer N días pasados")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    out_state = os.path.join(ROOT, cfg["output"]["state_file"])
    book_path = os.path.join(os.path.dirname(out_state), "book.json")
    if a.reset and os.path.exists(book_path):
        os.remove(book_path)

    universe = list(dict.fromkeys(cfg["universe"] + [cfg["benchmark"]]))
    cfg["universe"] = universe
    lookback = cfg["data"]["lookback"]
    frames = {t: get_candles(t, lookback + a.replay + 5, cfg["data"]["interval"], demo=a.demo) for t in universe}
    common = sorted(set.intersection(*(set(df.index) for df in frames.values())))
    frames = {t: df.loc[common] for t, df in frames.items()}

    engine = load_engine(cfg["forecast"], demo=a.demo)
    if cfg["broker"]["kind"] == "ibkr":
        broker = IBKRBroker(cfg["broker"]["ibkr"])
    else:
        broker = PaperBroker(book_path, cfg["risk"]["starting_cash"], cfg["broker"])
    risk = RiskManager(cfg["risk"], root=ROOT)

    log: list[str] = []
    steps = range(len(common) - a.replay + 1, len(common) + 1) if a.replay else [None]
    res = None
    total = len(steps)
    t_start = time.time()
    for k, upto in enumerate(steps, 1):
        res = cycle(cfg, engine, broker, risk, frames, upto, a.demo, log)
        if total > 1:
            el = time.time() - t_start
            print(f"── día {k}/{total} · {el / 60:.1f} min · faltan ~{el / k * (total - k) / 60:.1f} min", flush=True)
        if hasattr(broker, "save") and k % 10 == 0:
            broker.save()          # guardado intermedio por si GitHub corta la ejecución

    if hasattr(broker, "save"):
        broker.save()
    state = build_state(cfg, broker, res, frames, engine.name, a.demo, log)
    os.makedirs(os.path.dirname(out_state), exist_ok=True)
    with open(out_state, "w") as fh:
        json.dump(state, fh, ensure_ascii=False)
    with open(os.path.join(ROOT, cfg["output"]["journal_file"]), "a") as fh:
        for t, d in state["tickers"].items():
            fh.write(json.dumps({"date": state["as_of"], "ticker": t, "signal": d["signal"],
                                 "stats": {k: v for k, v in d["forecast"]["stats"].items() if k != "mean_path"}},
                                ensure_ascii=False) + "\n")
    print(f"\nPatrimonio: {state['portfolio']['equity']:,.2f} USD · estado → {out_state}")


if __name__ == "__main__":
    main()