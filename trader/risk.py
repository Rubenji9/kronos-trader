"""Reglas de riesgo deterministas. Ningún agente puede saltárselas.

Orden de aplicación:
  1. Kill switch (fichero KILL) → no se opera nada.
  2. Stop-loss por posición → venta forzosa, aunque el comité diga HOLD/BUY.
  3. Freno de pérdida diaria → solo se permiten ventas.
  4. Confianza mínima, tope por posición, exposición bruta, caja mínima,
     tamaño mínimo de orden y nº máximo de órdenes.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

from .agents import Signal


@dataclass
class Order:
    ticker: str
    side: str        # BUY | SELL
    qty: float
    ref_price: float
    reason: str

    def to_json(self) -> dict:
        return asdict(self)


class RiskManager:
    def __init__(self, cfg: dict, root: str = "."):
        self.c = cfg
        self.root = root

    def kill_switch(self) -> bool:
        return os.path.exists(os.path.join(self.root, self.c.get("kill_switch_file", "KILL")))

    def build_orders(self, signals: dict[str, Signal], positions: dict, cash: float,
                     prices: dict[str, float], prev_equity: float | None) -> tuple[list[Order], list[str]]:
        c = self.c
        log: list[str] = []
        if self.kill_switch():
            return [], ["KILL SWITCH activo — no se envía ninguna orden"]

        equity = cash + sum(p["qty"] * prices.get(t, p["avg_price"]) for t, p in positions.items())
        orders: list[Order] = []
        forced: set[str] = set()

        # 2) stop-loss
        for t, p in positions.items():
            px = prices.get(t)
            if px and p["qty"] > 0 and px <= p["avg_price"] * (1 - c["stop_loss_pct"]):
                orders.append(Order(t, "SELL", p["qty"], px, f"STOP-LOSS {px / p['avg_price'] - 1:+.1%}"))
                forced.add(t)
                log.append(f"{t}: stop-loss disparado")

        # 3) freno diario
        sells_only = False
        if prev_equity and equity / prev_equity - 1 <= -c["max_daily_loss_pct"]:
            sells_only = True
            log.append(f"Freno diario: cartera {equity / prev_equity - 1:+.2%} → solo ventas")

        # 4) señales del comité → órdenes, primero ventas (liberan caja)
        sim_cash = cash
        gross = equity - cash
        ordered = sorted(signals.values(), key=lambda s: 0 if s.action == "SELL" else 1)
        for s in ordered:
            t = s.ticker
            if t in forced or t not in prices or s.action == "HOLD":
                continue
            px = prices[t]
            cur_qty = positions.get(t, {}).get("qty", 0.0)
            cur_val = cur_qty * px
            if s.confidence < c["min_confidence"] and s.action != "SELL":
                log.append(f"{t}: {s.action} descartado (confianza {s.confidence:.0%} < {c['min_confidence']:.0%})")
                continue
            target_w = min(s.target_weight, c["max_position_pct"])
            delta_val = target_w * equity - cur_val
            # banda de rebalanceo: no tocar si el cambio es pequeño (evita comisiones por ruido),
            # salvo que la orden cierre la posición entera
            if abs(delta_val) < c.get("rebalance_band", 0.0) * equity and not (target_w == 0 and cur_qty > 0):
                continue

            if delta_val < 0:   # vender
                qty = min(cur_qty, math.floor(-delta_val / px))
                if qty * px >= c["min_trade_usd"]:
                    orders.append(Order(t, "SELL", qty, px, f"comité → {target_w:.0%}"))
                    sim_cash += qty * px
                    gross -= qty * px
            elif delta_val > 0 and not sells_only:
                room_gross = c["max_gross_exposure_pct"] * equity - gross
                room_cash = sim_cash - c["min_cash_pct"] * equity
                buy_val = min(delta_val, room_gross, room_cash)
                qty = math.floor(max(buy_val, 0) / px)
                if qty * px >= c["min_trade_usd"]:
                    orders.append(Order(t, "BUY", qty, px, f"comité → {target_w:.0%}"))
                    sim_cash -= qty * px
                    gross += qty * px
                elif delta_val >= c["min_trade_usd"]:
                    log.append(f"{t}: compra limitada por caja/exposición")

        if len(orders) > c["max_trades_per_run"]:
            log.append(f"Recortadas {len(orders) - c['max_trades_per_run']} órdenes por límite diario")
            orders = [o for o in orders if o.reason.startswith("STOP")] + \
                     [o for o in orders if not o.reason.startswith("STOP")]
            orders = orders[: c["max_trades_per_run"]]
        return orders, log
