"""Reglas de riesgo deterministas. Ningún agente puede saltárselas.

Orden de aplicación:
  1. Kill switch (fichero KILL) → no se opera nada.
  2. Stop-loss por posición → venta forzosa, aunque el comité diga HOLD/BUY.
  3. Freno de pérdida diaria → solo se permiten ventas.
  4. Confianza mínima, tope por posición, exposición bruta, caja mínima,
     banda de rebalanceo, tamaño mínimo de orden y nº máximo de órdenes.

Cada vez que una regla frena algo se guarda un evento con el motivo, para que el
panel muestre también lo que el sistema decidió NO hacer.
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
        self.events: list[dict] = []     # bloqueos del último ciclo: {ticker, rule, text}

    def kill_switch(self) -> bool:
        return os.path.exists(os.path.join(self.root, self.c.get("kill_switch_file", "KILL")))

    def _block(self, log: list[str], ticker: str | None, rule: str, text: str):
        log.append(f"{ticker + ': ' if ticker else ''}{text}")
        self.events.append({"ticker": ticker, "rule": rule, "text": text})

    def build_orders(self, signals: dict[str, Signal], positions: dict, cash: float,
                     prices: dict[str, float], prev_equity: float | None) -> tuple[list[Order], list[str]]:
        c = self.c
        log: list[str] = []
        self.events = []
        if self.kill_switch():
            self._block(log, None, "kill", "KILL SWITCH activo — no se envía ninguna orden")
            return [], log

        equity = cash + sum(p["qty"] * prices.get(t, p["avg_price"]) for t, p in positions.items())
        orders: list[Order] = []
        forced: set[str] = set()

        # 2) stop-loss
        for t, p in positions.items():
            px = prices.get(t)
            if px and p["qty"] > 0 and px <= p["avg_price"] * (1 - c["stop_loss_pct"]):
                orders.append(Order(t, "SELL", p["qty"], px, f"STOP-LOSS {px / p['avg_price'] - 1:+.1%}"))
                forced.add(t)
                self._block(log, t, "stop", f"stop-loss disparado ({px / p['avg_price'] - 1:+.1%} desde compra) → venta forzosa")

        # 3) freno diario
        sells_only = False
        if prev_equity and equity / prev_equity - 1 <= -c["max_daily_loss_pct"]:
            sells_only = True
            self._block(log, None, "freno", f"Freno diario: cartera {equity / prev_equity - 1:+.2%} → solo ventas")

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
                self._block(log, t, "confianza",
                            f"{s.action} descartado: confianza {s.confidence:.1%} < mínimo {c['min_confidence']:.0%}")
                continue
            target_w = min(s.target_weight, c["max_position_pct"])
            delta_val = target_w * equity - cur_val
            # banda de rebalanceo: no tocar si el cambio es pequeño (evita comisiones por ruido),
            # salvo que la orden cierre la posición entera
            if abs(delta_val) < c.get("rebalance_band", 0.0) * equity and not (target_w == 0 and cur_qty > 0):
                self._block(log, t, "banda",
                            f"{s.action} no ejecutado: cambio de peso {delta_val / equity:+.1%} dentro de la banda "
                            f"de ±{c.get('rebalance_band', 0):.0%}")
                continue

            if delta_val < 0:   # vender
                qty = min(cur_qty, math.floor(-delta_val / px))
                if qty * px >= c["min_trade_usd"]:
                    orders.append(Order(t, "SELL", qty, px, f"comité → {target_w:.0%}"))
                    sim_cash += qty * px
                    gross -= qty * px
                elif cur_qty > 0:
                    self._block(log, t, "minimo", f"venta de {qty * px:,.0f} $ por debajo del mínimo de {c['min_trade_usd']} $")
            elif delta_val > 0:
                if sells_only:
                    self._block(log, t, "freno", "compra bloqueada por el freno de pérdida diaria")
                    continue
                room_gross = c["max_gross_exposure_pct"] * equity - gross
                room_cash = sim_cash - c["min_cash_pct"] * equity
                buy_val = min(delta_val, room_gross, room_cash)
                qty = math.floor(max(buy_val, 0) / px)
                if qty * px >= c["min_trade_usd"]:
                    if buy_val < delta_val - 1:
                        self._block(log, t, "limite", f"compra recortada de {delta_val:,.0f} $ a {qty * px:,.0f} $ "
                                                      f"por caja mínima / exposición máxima")
                    orders.append(Order(t, "BUY", qty, px, f"comité → {target_w:.0%}"))
                    sim_cash -= qty * px
                    gross += qty * px
                elif delta_val >= c["min_trade_usd"]:
                    self._block(log, t, "limite", "compra bloqueada: sin caja libre o exposición máxima alcanzada")

        if len(orders) > c["max_trades_per_run"]:
            extra = len(orders) - c["max_trades_per_run"]
            orders = [o for o in orders if o.reason.startswith("STOP")] + \
                     [o for o in orders if not o.reason.startswith("STOP")]
            for o in orders[c["max_trades_per_run"]:]:
                self._block(log, o.ticker, "max_ordenes", f"{o.side} aplazado: límite de {c['max_trades_per_run']} órdenes por día")
            orders = orders[: c["max_trades_per_run"]]
            log.append(f"Recortadas {extra} órdenes por límite diario")
        return orders, log
