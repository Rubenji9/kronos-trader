"""Revalorización en directo. NO compra ni vende.

Se ejecuta cada 30 minutos con la bolsa de Nueva York abierta: baja el último precio
de cada valor, recalcula cuánto vale la cartera, cuánto se gana o pierde hoy y guarda
un punto en el gráfico intradía del panel.

Uso:
  python -m trader.mark            # solo actúa con el mercado abierto
  python -m trader.mark --force    # actúa siempre (pruebas)
  python -m trader.mark --demo     # precios simulados, sin red
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def market_open(now: pd.Timestamp | None = None) -> bool:
    now = now if now is not None else pd.Timestamp.now(tz="America/New_York")
    if now.weekday() >= 5:
        return False
    return (9, 30) <= (now.hour, now.minute) <= (16, 10)


def last_prices(tickers: list[str]) -> dict[str, float]:
    import yfinance as yf

    out = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="1d", interval="5m", auto_adjust=True)
            if h is not None and not h.empty:
                out[t] = float(h["Close"].dropna().iloc[-1])
        except Exception as e:  # noqa: BLE001
            print(f"[mark] sin precio para {t}: {e!r}")
    return out


def demo_prices(prev: dict[str, float], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {t: round(p * float(np.exp(0.006 * rng.standard_normal())), 4) for t, p in prev.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if not (a.force or a.demo or market_open()):
        print("Mercado cerrado: nada que revalorizar.")
        return

    cfg = yaml.safe_load(open(a.config))
    data_dir = os.path.join(ROOT, os.path.dirname(cfg["output"]["state_file"]))
    state_path = os.path.join(ROOT, cfg["output"]["state_file"])
    book_path = os.path.join(data_dir, "book.json")
    intraday_path = os.path.join(data_dir, "intraday.json")
    if not (os.path.exists(state_path) and os.path.exists(book_path)):
        print("Aún no hay cartera: ejecuta primero el ciclo diario.")
        return

    state = json.load(open(state_path))
    book = json.load(open(book_path))
    bench = cfg["benchmark"]
    tickers = list(dict.fromkeys(state["universe"] + [bench] + list(book["positions"])))
    close = {t: d["candles"][-1][4] for t, d in state["tickers"].items()}   # último cierre conocido

    if a.demo:
        prev_live = (state.get("live") or {}).get("prices") or close
        prices = demo_prices({t: prev_live.get(t, close.get(t)) for t in tickers},
                             seed=int(datetime.now().timestamp()) % 2**31)
    else:
        prices = last_prices(tickers)
        if not prices:
            print("No se pudo obtener ningún precio en directo: no guardo nada.")
            return
    for t in tickers:                                   # sin precio → último cierre
        prices.setdefault(t, close.get(t))

    cash = float(book["cash"])
    positions = []
    eq = cash
    for t, p in book["positions"].items():
        px = prices.get(t) or p["avg_price"]
        val = p["qty"] * px
        eq += val
        positions.append({"ticker": t, "qty": p["qty"], "avg_price": round(p["avg_price"], 4), "price": round(px, 4),
                          "value": round(val, 2), "pnl_pct": round(px / p["avg_price"] - 1, 4),
                          "pnl_usd": round((px - p["avg_price"]) * p["qty"], 2),
                          "day_pct": round(px / close[t] - 1, 4) if close.get(t) else None})
    for x in positions:
        x["weight"] = round(x["value"] / eq, 4) if eq else 0
    hist = book.get("equity_history") or []
    prev_eq = hist[-1]["equity"] if hist else book.get("starting_cash", eq)
    prev_bench = hist[-1]["bench"] if hist else close.get(bench)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    live = {
        "ts": now,
        "equity": round(eq, 2),
        "cash": round(cash, 2),
        "invested": round(eq - cash, 2),
        "prev_close_equity": prev_eq,
        "day_pnl": round(eq - prev_eq, 2),
        "day_pnl_pct": round(eq / prev_eq - 1, 5) if prev_eq else None,
        "bench_day_pct": round(prices[bench] / prev_bench - 1, 5) if prev_bench and prices.get(bench) else None,
        "prices": {t: round(v, 4) for t, v in prices.items() if v},
        "changes": {t: round(prices[t] / close[t] - 1, 5) for t in prices if close.get(t) and prices.get(t)},
    }
    state["live"] = live
    state["portfolio"]["positions"] = sorted(positions, key=lambda x: -x["value"])
    state["portfolio"]["equity"] = live["equity"]

    intraday = json.load(open(intraday_path)) if os.path.exists(intraday_path) else {"points": []}
    intraday["points"].append({"ts": now, "equity": live["equity"], "bench": round(prices.get(bench) or 0, 4),
                               "prev_close_equity": prev_eq, "prev_bench": prev_bench})
    intraday["points"] = intraday["points"][-500:]       # ~5 semanas de puntos cada 30 min

    with open(state_path, "w") as fh:
        json.dump(state, fh, ensure_ascii=False)
    with open(intraday_path, "w") as fh:
        json.dump(intraday, fh, ensure_ascii=False)
    print(f"{now} · patrimonio {eq:,.2f} $ · hoy {live['day_pnl']:+,.2f} $ "
          f"({(live['day_pnl_pct'] or 0):+.2%}) · {bench} {(live['bench_day_pct'] or 0):+.2%}")


if __name__ == "__main__":
    main()
