"""¿Predice Kronos algo útil? Prueba honesta sobre historia real.

Para varias fechas pasadas, hace la previsión viendo SOLO los datos hasta ese día y la
compara con lo que pasó de verdad `horizon` sesiones después. Repite con varias
longitudes de contexto (lookback), porque Kronos normaliza sobre esa ventana y eso
cambia mucho su sesgo.

Métricas por lookback (todas sobre el rendimiento a `horizon` sesiones):
  sesgo     media(previsto) − media(real). Muy negativo = pesimista sistemático.
  acierto   % de veces que acierta la dirección (sube/baja).
  base      % de veces que el precio subió. Si "acierto" no supera "base", da igual
            hacer caso a Kronos que comprar siempre.
  IC        correlación de rangos entre previsto y real. ~0 = no ordena nada;
            > 0,05 sostenido ya sería interesante en finanzas.
  IC_tk     media del IC calculado dentro de cada ticker (quita el efecto de que unos
            valores suben más que otros en el periodo).

Uso:
  python tools/evaluate_kronos.py --lookbacks 64,128,256 --dates 20 --step 3
  python tools/evaluate_kronos.py --demo      # prueba rápida offline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from trader.data import get_candles  # noqa: E402
from trader.forecast import load_engine, run_forecast  # noqa: E402


def spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--lookbacks", default="64,128,256")
    ap.add_argument("--dates", type=int, default=20, help="nº de fechas pasadas a evaluar")
    ap.add_argument("--step", type=int, default=3, help="sesiones entre fechas evaluadas")
    ap.add_argument("--samples", type=int, default=8, help="escenarios por previsión (menos = más rápido)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "eval.json"))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    fcfg = dict(cfg["forecast"], samples=a.samples)
    h = int(fcfg["horizon"])
    lbs = [int(x) for x in a.lookbacks.split(",")]
    tickers = list(dict.fromkeys(cfg["universe"] + [cfg["benchmark"]]))
    need = max(lbs) + (a.dates - 1) * a.step + h + 5

    frames = {t: get_candles(t, need, cfg["data"]["interval"], demo=a.demo) for t in tickers}
    engine = load_engine(fcfg, demo=a.demo)
    total = len(lbs) * len(tickers) * a.dates
    print(f"Motor {engine.name} · {len(tickers)} tickers × {a.dates} fechas × {len(lbs)} lookbacks "
          f"= {total} previsiones · horizonte {h} sesiones", flush=True)

    rows, done, t0 = [], 0, time.time()
    for lb in lbs:
        for t, df in frames.items():
            n = len(df)
            ends = [n - h - k * a.step for k in range(a.dates)]       # índice exclusivo del contexto
            for e in ends:
                ctx = df.iloc[e - lb:e]
                if len(ctx) < lb:
                    continue
                fc = run_forecast(engine, t, ctx, fcfg, seed=e)
                real = float(df["close"].iloc[e - 1 + h] / df["close"].iloc[e - 1] - 1)
                rows.append({"lookback": lb, "ticker": t, "date": ctx.index[-1].strftime("%Y-%m-%d"),
                             "pred": fc.stats["exp_return"], "p_up": fc.stats["p_up"], "real": real,
                             "vs_sma": float(ctx["close"].iloc[-1] / ctx["close"].mean() - 1)})
                done += 1
                if done % 10 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{total} · {el/60:.1f} min · faltan ~{el/done*(total-done)/60:.1f} min", flush=True)

    r = pd.DataFrame(rows)
    summary = []
    print(f"\n{'lookback':>8} {'n':>4} {'previsto':>9} {'real':>8} {'sesgo':>8} {'acierto':>8} {'base':>6} {'IC':>6} {'IC_tk':>6} {'IC_reversión':>13}")
    for lb, g in r.groupby("lookback"):
        hit = float((np.sign(g["pred"]) == np.sign(g["real"])).mean())
        base = float((g["real"] > 0).mean())
        ic = spearman(g["pred"], g["real"])
        ic_tk = float(np.nanmean([spearman(x["pred"], x["real"]) for _, x in g.groupby("ticker")]))
        # ¿el sesgo viene de estar lejos de la media de la ventana?
        rev = spearman(g["vs_sma"], g["pred"])
        s = {"lookback": int(lb), "n": int(len(g)), "pred_mean": round(float(g["pred"].mean()), 5),
             "real_mean": round(float(g["real"].mean()), 5),
             "bias": round(float(g["pred"].mean() - g["real"].mean()), 5),
             "hit_rate": round(hit, 4), "base_up_rate": round(base, 4),
             "ic": round(ic, 4), "ic_per_ticker": round(ic_tk, 4), "corr_dist_to_mean_vs_pred": round(rev, 4)}
        summary.append(s)
        print(f"{lb:>8} {len(g):>4} {g['pred'].mean():>+9.2%} {g['real'].mean():>+8.2%} {s['bias']:>+8.2%} "
              f"{hit:>8.0%} {base:>6.0%} {ic:>+6.2f} {ic_tk:>+6.2f} {rev:>+13.2f}")

    print("\nPor ticker (lookback con mejor IC):")
    best = max(summary, key=lambda s: (s["ic"] if s["ic"] == s["ic"] else -9))["lookback"]
    for t, g in r[r.lookback == best].groupby("ticker"):
        print(f"  {t:<5} previsto {g['pred'].mean():+.2%} · real {g['real'].mean():+.2%} · "
              f"acierto {(np.sign(g['pred']) == np.sign(g['real'])).mean():.0%} · IC {spearman(g['pred'], g['real']):+.2f}")
    print("\nIC_reversión ≈ −1 significa: cuanto más por encima de su media está el precio, más bajada predice.")
    print(f"Ojo: {a.dates} fechas por ticker con horizontes solapados → muestra pequeña, tómalo como orientación.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"engine": engine.name, "horizon": h, "samples": a.samples, "step": a.step,
               "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
               "summary": summary, "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nDetalle → {a.out}")


if __name__ == "__main__":
    main()