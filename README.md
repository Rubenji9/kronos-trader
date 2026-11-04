# Kronos Trader

Sistema personal y autónomo en el que agentes de IA gestionan una cartera de acciones.
Empieza con **dinero ficticio** (paper trading) y está preparado para pasar a Interactive Brokers
cuando los resultados lo justifiquen.

```
 velas diarias ──► Kronos (32 escenarios) ──► comité de agentes ──► REGLAS DE RIESGO ──► broker ──► panel
   yfinance         probabilidad de subida      rules / claude /       en código, no        paper /     GitHub
                    percentiles, dispersión     tradingagents          negociables          IBKR        Pages
```

## Piezas

| Módulo | Qué hace |
|---|---|
| `trader/data.py` | Velas OHLCV de Yahoo Finance (o sintéticas con `--demo`) |
| `trader/forecast.py` | Kronos-small genera N caminos de precio **por separado** (el `predict` original los promedia) y calcula P(↑), E[r], p10/p50/p90 y dispersión |
| `trader/agents.py` | Comité: analista Kronos, técnico, contrarian y riesgo (`rules`), o Claude como comité (`claude`), o TradingAgents completo (`tradingagents`) |
| `trader/risk.py` | Tope por posición, exposición máxima, caja mínima, stop-loss, freno de pérdida diaria, banda de rebalanceo, confianza mínima y kill switch |
| `trader/broker.py` | `PaperBroker` (libro propio en JSON, con slippage y comisiones estilo IBKR) e `IBKRBroker` (fase 2) |
| `trader/run.py` | Orquesta el ciclo y escribe `docs/data/state.json` |
| `docs/index.html` | Panel visual (abanico de escenarios, comité, log, cartera vs SPY) |

## Probarlo en local

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
git clone --depth 1 https://github.com/shiyu-coder/Kronos.git vendor/Kronos

python -m trader.run --demo --replay 90 --reset   # sin red, para ver el panel
python -m trader.run --replay 60 --reset          # datos reales + Kronos real, 60 días rehechos
python -m http.server -d docs 8000                # abre http://localhost:8000
```

`--replay N` rehace los últimos N días **viendo solo los datos disponibles en cada día**: es una
mini-prueba histórica y sirve para estrenar la cartera paper con historial.

## Desplegarlo gratis (fase 1: paper)

1. Crea un repo en GitHub y sube esta carpeta.
2. *Settings → Actions → General → Workflow permissions* → **Read and write**.
3. *Settings → Pages* → Source: `main` / carpeta `/docs`. El panel queda en `https://<usuario>.github.io/<repo>/`.
   (Pages en repos privados requiere GitHub Pro; si el repo es privado, Cloudflare Pages conectado al repo es gratis.)
4. *Actions → ciclo-diario → Run workflow* con `replay = 60` para la primera ejecución.
5. A partir de ahí corre solo cada día laborable a las 22:20 UTC. No necesitas tener nada encendido.

Para usar Claude como comité: `decision.mode: claude` en `config.yaml` y un secreto `ANTHROPIC_API_KEY`
en *Settings → Secrets and variables → Actions*.

**Parar todo:** crea un fichero `KILL` en la raíz del repo. El siguiente ciclo no enviará ninguna orden.

## Fases

1. **Paper en GitHub Actions** (ahora). Mínimo 3 meses. Criterio para seguir: batir a SPY después de comisiones,
   con un drawdown que aceptarías con dinero real, y que el modo `claude` aporte algo frente a `rules`.
2. **IBKR paper**: VM siempre encendida (Oracle Cloud Always Free, 2 OCPU / 12 GB, o un VPS de ~5 €/mes) con
   IB Gateway en Docker (`gnzsnz/ib-gateway`) y `broker.kind: ibkr`, puerto 4002.
3. **IBKR real con poco capital**: puerto 4001 y `allow_live: true` (doble llave a propósito). Las órdenes salen
   como *market-on-open* tras el cierre. Cuenta con aprobar el 2FA una vez por semana.

## Limitaciones conocidas

- En paper, las órdenes se rellenan al cierre del día de la decisión (+5 pb de slippage). En real se rellenarán
  en la apertura siguiente, así que habrá diferencias.
- Kronos solo ve precio y volumen: no conoce noticias ni resultados. El modo `claude` tampoco recibe noticias
  todavía (es la siguiente mejora natural).
- El modo `rules` es la línea base. Si los LLMs no la baten en paper, no aportan valor.
- Esto es un proyecto personal de experimentación, no asesoramiento financiero.
