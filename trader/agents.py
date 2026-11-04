"""Comité de agentes: convierte datos + escenarios de Kronos en una señal por ticker.

Tres modos (config.decision.mode):
  rules          Analistas deterministas. Gratis y reproducible; es la línea base
                 contra la que hay que medir si los LLMs aportan algo.
  claude         Una llamada a Claude por ticker que hace de comité (alcista,
                 bajista, riesgo, gestor) y devuelve JSON estructurado.
  tradingagents  TauricResearch/TradingAgents (LangGraph multi-agente). Su
                 veredicto se combina con la señal de Kronos.

La salida del comité NUNCA va directa al broker: pasa por risk.RiskManager.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .forecast import Forecast


@dataclass
class Vote:
    agent: str
    stance: str          # bull | bear | neutral
    score: float         # -1 … +1
    note: str


@dataclass
class Signal:
    ticker: str
    action: str          # BUY | SELL | HOLD
    target_weight: float # peso deseado en cartera (0 … 1)
    confidence: float    # 0 … 1
    rationale: str
    votes: list[Vote] = field(default_factory=list)
    source: str = "rules"

    def to_json(self) -> dict:
        d = asdict(self)
        d["target_weight"] = round(self.target_weight, 4)
        d["confidence"] = round(self.confidence, 3)
        return d


# ───────────────────────── features ─────────────────────────
def features(df: pd.DataFrame) -> dict:
    c = df["close"]
    r = np.log(c).diff()
    return {
        "close": float(c.iloc[-1]),
        "ret_5d": float(c.iloc[-1] / c.iloc[-6] - 1),
        "ret_20d": float(c.iloc[-1] / c.iloc[-21] - 1),
        "ret_60d": float(c.iloc[-1] / c.iloc[-61] - 1),
        "sma50": float(c.tail(50).mean()),
        "sma200": float(c.tail(200).mean()) if len(c) >= 200 else float(c.mean()),
        "vol_20d_ann": float(r.tail(20).std() * np.sqrt(252)),
        "rsi14": _rsi(c, 14),
        "drawdown_from_high": float(c.iloc[-1] / c.tail(252).max() - 1),
    }


def _rsi(c: pd.Series, n: int) -> float:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up.iloc[-1] / (dn.iloc[-1] + 1e-12)
    return float(100 - 100 / (1 + rs))


def _stance(s: float) -> str:
    return "bull" if s > 0.15 else "bear" if s < -0.15 else "neutral"


# ───────────────────────── modo rules ─────────────────────────
def rules_committee(ticker: str, f: dict, fc: Forecast, current_weight: float, max_w: float) -> Signal:
    st = fc.stats
    votes: list[Vote] = []

    # 1) Analista Kronos: probabilidad de subida y rendimiento esperado ajustado por dispersión
    edge = (st["p_up"] - 0.5) * 2
    sharpe_like = st["exp_return"] / (st["dispersion"] + 1e-6)
    s1 = float(np.clip(0.6 * edge + 0.4 * np.tanh(sharpe_like * 2), -1, 1))
    votes.append(Vote("Kronos", _stance(s1), s1,
                      f"{st['n_up']}/{st['samples']} escenarios al alza · E[r]={st['exp_return']:+.2%} "
                      f"· rango p10–p90 {st['p10']:.2f}–{st['p90']:.2f}"))

    # 2) Analista técnico: tendencia (SMA) + momentum
    trend = (1 if f["close"] > f["sma50"] else -1) * 0.5 + (1 if f["sma50"] > f["sma200"] else -1) * 0.5
    mom = np.tanh(f["ret_60d"] * 5)
    s2 = float(np.clip(0.6 * trend + 0.4 * mom, -1, 1))
    votes.append(Vote("Técnico", _stance(s2), s2,
                      f"precio {'>' if f['close'] > f['sma50'] else '<'} SMA50 · SMA50 "
                      f"{'>' if f['sma50'] > f['sma200'] else '<'} SMA200 · 60d {f['ret_60d']:+.1%}"))

    # 3) Analista contrarian: sobrecompra/sobreventa
    s3 = float(np.clip((50 - f["rsi14"]) / 30, -1, 1)) * 0.6
    votes.append(Vote("Contrarian", _stance(s3), s3, f"RSI14 {f['rsi14']:.0f}"))

    # 4) Riesgo: penaliza volatilidad alta y caídas fuertes
    s4 = -float(np.clip((f["vol_20d_ann"] - 0.30) / 0.4, 0, 1)) - (0.3 if f["drawdown_from_high"] < -0.2 else 0)
    votes.append(Vote("Riesgo", _stance(s4), s4,
                      f"vol anual {f['vol_20d_ann']:.0%} · drawdown {f['drawdown_from_high']:.0%}"))

    w = {"Kronos": 0.45, "Técnico": 0.30, "Contrarian": 0.10, "Riesgo": 0.15}
    score = sum(v.score * w[v.agent] for v in votes)
    agreement = 1 - np.std([v.score for v in votes]) / 1.0
    confidence = float(np.clip(0.5 + abs(score) * 0.6 + 0.15 * (agreement - 0.5), 0, 1))

    target = float(np.clip(score, 0, 1) * max_w) if score > 0.1 else 0.0
    action = "BUY" if target > current_weight + 0.02 else "SELL" if target < current_weight - 0.02 else "HOLD"
    why = f"score {score:+.2f} → peso objetivo {target:.0%} (actual {current_weight:.0%})"
    return Signal(ticker, action, target, confidence, why, votes, "rules")


# ───────────────────────── modo claude ─────────────────────────
CLAUDE_PROMPT = """Eres un comité de inversión de cuatro miembros (analista alcista, analista bajista,
gestor de riesgos y gestor de cartera) que decide UNA posición en {ticker} para una cartera
long-only de largo plazo con horizonte de ~{horizon} sesiones.

Datos técnicos (velas diarias):
{features}

Modelo Kronos ({engine}) — {samples} escenarios simulados a {horizon} sesiones:
- escenarios al alza: {n_up}/{samples}  (p_up = {p_up:.0%})
- rendimiento esperado: {exp_return:+.2%} · mediana {median_return:+.2%}
- percentiles del cierre final: p10 {p10:.2f} · p50 {p50:.2f} · p90 {p90:.2f}  (último {last:.2f})
- dispersión entre escenarios: {dispersion:.2%}

Peso actual en cartera: {current:.0%}. Peso máximo permitido: {max_w:.0%}.

Ten en cuenta que Kronos es un modelo estadístico de series temporales, no conoce noticias, y que
la opción por defecto ante señales débiles o contradictorias es no operar.

Responde SOLO con JSON válido, sin texto alrededor:
{{"bull": "...", "bear": "...", "risk": "...",
  "action": "BUY|SELL|HOLD", "target_weight": 0.0, "confidence": 0.0, "rationale": "..."}}"""


def claude_committee(ticker: str, f: dict, fc: Forecast, current_weight: float, max_w: float, model: str) -> Signal:
    import anthropic

    client = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY
    st = fc.stats
    prompt = CLAUDE_PROMPT.format(
        ticker=ticker, horizon=st["horizon"], engine=fc.engine, samples=st["samples"],
        features=json.dumps({k: round(v, 4) for k, v in f.items()}, ensure_ascii=False),
        n_up=st["n_up"], p_up=st["p_up"], exp_return=st["exp_return"], median_return=st["median_return"],
        p10=st["p10"], p50=st["p50"], p90=st["p90"], last=fc.last_close, dispersion=st["dispersion"],
        current=current_weight, max_w=max_w)
    msg = client.messages.create(model=model, max_tokens=900, messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))

    tw = float(np.clip(data.get("target_weight", 0), 0, max_w))
    votes = [
        Vote("Alcista", "bull", 0.5, data.get("bull", "")),
        Vote("Bajista", "bear", -0.5, data.get("bear", "")),
        Vote("Riesgo", "neutral", 0.0, data.get("risk", "")),
    ]
    return Signal(ticker, str(data.get("action", "HOLD")).upper(), tw,
                  float(np.clip(data.get("confidence", 0), 0, 1)), data.get("rationale", ""), votes, f"claude:{model}")


# ─────────────────────── modo tradingagents ───────────────────────
def tradingagents_committee(ticker: str, f: dict, fc: Forecast, current_weight: float, max_w: float,
                            ta_cfg: dict, date: str) -> Signal:
    from tradingagents.default_config import DEFAULT_CONFIG  # type: ignore
    from tradingagents.graph.trading_graph import TradingAgentsGraph  # type: ignore

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({k: v for k, v in ta_cfg.items()})
    ta = TradingAgentsGraph(debug=False, config=cfg)
    _, decision = ta.propagate(ticker, date)
    text = str(decision)
    up = text.upper()
    verdict = 1.0 if re.search(r"\b(BUY|OVERWEIGHT)\b", up) else -1.0 if re.search(r"\b(SELL|UNDERWEIGHT)\b", up) else 0.0

    base = rules_committee(ticker, f, fc, current_weight, max_w)   # Kronos + técnico como contrapeso
    kron = next(v.score for v in base.votes if v.agent == "Kronos")
    score = 0.6 * verdict + 0.4 * kron
    target = float(np.clip(score, 0, 1) * max_w) if score > 0.1 else 0.0
    action = "BUY" if target > current_weight + 0.02 else "SELL" if target < current_weight - 0.02 else "HOLD"
    votes = [Vote("TradingAgents", _stance(verdict), verdict, text[:400])] + base.votes[:1]
    conf = float(np.clip(0.5 + abs(score) * 0.5, 0, 1))
    return Signal(ticker, action, target, conf, f"TA={verdict:+.0f} · Kronos={kron:+.2f}", votes, "tradingagents")


def decide(mode: str, ticker: str, df: pd.DataFrame, fc: Forecast, current_weight: float,
           max_w: float, dcfg: dict, date: str) -> tuple[Signal, dict]:
    f = features(df)
    try:
        if mode == "claude" and os.environ.get("ANTHROPIC_API_KEY"):
            return claude_committee(ticker, f, fc, current_weight, max_w, dcfg.get("claude_model")), f
        if mode == "tradingagents":
            return tradingagents_committee(ticker, f, fc, current_weight, max_w, dcfg.get("tradingagents", {}), date), f
    except Exception as e:  # noqa: BLE001 — si el LLM falla, nunca bloqueamos: caemos a reglas
        print(f"[agents] {mode} falló para {ticker}: {e!r} → reglas")
    return rules_committee(ticker, f, fc, current_weight, max_w), f
