"""Las reglas de riesgo son lo que protege el dinero: se prueban aparte."""
import os, sys, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trader.agents import Signal
from trader.risk import RiskManager

CFG = dict(long_only=True, max_position_pct=0.20, max_gross_exposure_pct=0.90, min_cash_pct=0.10,
           stop_loss_pct=0.08, max_daily_loss_pct=0.03, max_trades_per_run=6, min_confidence=0.6,
           min_trade_usd=250, rebalance_band=0.04, kill_switch_file="KILL")

def sig(t, a, w, c=0.8): return Signal(t, a, w, c, "", [])

def test_cap_por_posicion():
    o, _ = RiskManager(CFG, tempfile.mkdtemp()).build_orders({"A": sig("A", "BUY", 0.9)}, {}, 100_000, {"A": 100}, None)
    assert o[0].qty * 100 <= 20_000

def test_stop_loss_forzado_aunque_comite_compre():
    pos = {"A": {"qty": 100, "avg_price": 100}}
    o, _ = RiskManager(CFG, tempfile.mkdtemp()).build_orders({"A": sig("A", "BUY", 0.2)}, pos, 50_000, {"A": 91}, None)
    assert len(o) == 1 and o[0].side == "SELL" and o[0].qty == 100 and o[0].reason.startswith("STOP")

def test_confianza_baja_no_compra():
    o, _ = RiskManager(CFG, tempfile.mkdtemp()).build_orders({"A": sig("A", "BUY", 0.2, 0.4)}, {}, 100_000, {"A": 100}, None)
    assert o == []

def test_freno_diario_solo_ventas():
    pos = {"B": {"qty": 100, "avg_price": 100}}
    s = {"A": sig("A", "BUY", 0.2), "B": sig("B", "SELL", 0.0)}
    o, log = RiskManager(CFG, tempfile.mkdtemp()).build_orders(s, pos, 80_000, {"A": 100, "B": 99}, prev_equity=95_000)
    assert [x.side for x in o] == ["SELL"] and any("Freno" in l for l in log)

def test_caja_minima_y_exposicion():
    s = {t: sig(t, "BUY", 0.2) for t in "ABCDEF"}
    o, _ = RiskManager(CFG, tempfile.mkdtemp()).build_orders(s, {}, 100_000, {t: 100 for t in "ABCDEF"}, None)
    assert sum(x.qty * 100 for x in o) <= 90_000

def test_kill_switch():
    d = tempfile.mkdtemp(); open(os.path.join(d, "KILL"), "w").close()
    o, log = RiskManager(CFG, d).build_orders({"A": sig("A", "BUY", 0.2)}, {}, 100_000, {"A": 100}, None)
    assert o == [] and "KILL" in log[0]

def test_banda_rebalanceo():
    pos = {"A": {"qty": 180, "avg_price": 100}}   # 18 % → objetivo 20 %: dentro de la banda
    o, _ = RiskManager(CFG, tempfile.mkdtemp()).build_orders({"A": sig("A", "BUY", 0.2)}, pos, 82_000, {"A": 100}, None)
    assert o == []

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("✓", k)
