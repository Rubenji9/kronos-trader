"""Brokers.

PaperBroker   Libro simulado propio, persistido en JSON dentro del repo. Rellena
              al último cierre + slippage + comisión estilo IBKR. Es lo que corre
              en la fase 1 (GitHub Actions, gratis).
IBKRBroker    Interactive Brokers vía IB Gateway + ib_async. Fase 2: necesita un
              servidor siempre encendido con IB Gateway (p. ej. imagen Docker
              gnzsnz/ib-gateway). Por defecto apunta al puerto paper 4002 y se
              niega a operar en real salvo `allow_live: true`.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .risk import Order


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperBroker:
    def __init__(self, state_path: str, starting_cash: float, cfg: dict):
        self.path = state_path
        self.cfg = cfg
        if os.path.exists(state_path):
            with open(state_path) as fh:
                self.book = json.load(fh)
        else:
            self.book = {"cash": starting_cash, "starting_cash": starting_cash,
                         "positions": {}, "trades": [], "equity_history": []}

    # ---- lectura ----
    @property
    def cash(self) -> float:
        return float(self.book["cash"])

    @property
    def positions(self) -> dict:
        return self.book["positions"]

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p["qty"] * prices.get(t, p["avg_price"]) for t, p in self.positions.items())

    def last_equity(self) -> float | None:
        h = self.book["equity_history"]
        return h[-1]["equity"] if h else None

    # ---- ejecución ----
    def _commission(self, qty: float) -> float:
        return max(self.cfg["min_commission"], qty * self.cfg["commission_per_share"])

    def execute(self, orders: list[Order], run_id: str) -> list[dict]:
        fills = []
        slip = self.cfg["slippage_bps"] / 1e4
        for o in orders:
            px = o.ref_price * (1 + slip if o.side == "BUY" else 1 - slip)
            fee = self._commission(o.qty)
            pos = self.positions.get(o.ticker, {"qty": 0.0, "avg_price": 0.0})
            if o.side == "BUY":
                cost = o.qty * px + fee
                if cost > self.cash:
                    continue
                new_qty = pos["qty"] + o.qty
                pos["avg_price"] = (pos["qty"] * pos["avg_price"] + o.qty * px) / new_qty
                pos["qty"] = new_qty
                self.book["cash"] = self.cash - cost
                pnl = None
            else:
                q = min(o.qty, pos["qty"])
                if q <= 0:
                    continue
                pnl = (px - pos["avg_price"]) * q - fee
                pos["qty"] -= q
                self.book["cash"] = self.cash + q * px - fee
            if pos["qty"] > 1e-9:
                self.positions[o.ticker] = pos
            else:
                self.positions.pop(o.ticker, None)
            fill = {"ts": _now(), "run": run_id, "ticker": o.ticker, "side": o.side, "qty": o.qty,
                    "price": round(px, 4), "fee": round(fee, 2), "reason": o.reason,
                    "pnl": None if pnl is None else round(pnl, 2)}
            self.book["trades"].append(fill)
            fills.append(fill)
        return fills

    def mark(self, prices: dict[str, float], bench_price: float, date: str):
        eq = self.equity(prices)
        h = self.book["equity_history"]
        entry = {"date": date, "equity": round(eq, 2), "cash": round(self.cash, 2), "bench": round(bench_price, 4)}
        if h and h[-1]["date"] == date:
            h[-1] = entry
        else:
            h.append(entry)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(self.book, fh, indent=1)


class IBKRBroker:
    """Esqueleto para la fase 2. Requiere `pip install ib_async` e IB Gateway corriendo."""

    def __init__(self, cfg: dict):
        from ib_async import IB  # type: ignore

        self.cfg = cfg
        if cfg["port"] in (4001, 7496) and not cfg.get("allow_live"):
            raise RuntimeError("Puerto de cuenta REAL y allow_live=false → me niego a conectar")
        self.ib = IB()
        self.ib.connect(cfg["host"], cfg["port"], clientId=cfg["client_id"], timeout=20)

    @property
    def cash(self) -> float:
        vals = {v.tag: v.value for v in self.ib.accountSummary() if v.currency == "USD"}
        return float(vals.get("TotalCashValue", 0))

    @property
    def positions(self) -> dict:
        return {p.contract.symbol: {"qty": float(p.position), "avg_price": float(p.avgCost)}
                for p in self.ib.positions() if p.position}

    def execute(self, orders: list[Order], run_id: str) -> list[dict]:
        from ib_async import MarketOrder, Stock  # type: ignore

        fills = []
        for o in orders:
            c = Stock(o.ticker, "SMART", "USD")
            self.ib.qualifyContracts(c)
            # Tras el cierre: orden a mercado para la apertura (MOO) → tif="OPG"
            tr = self.ib.placeOrder(c, MarketOrder(o.side, o.qty, tif="OPG"))
            fills.append({"ts": _now(), "run": run_id, "ticker": o.ticker, "side": o.side,
                          "qty": o.qty, "price": None, "status": tr.orderStatus.status, "reason": o.reason})
        return fills
