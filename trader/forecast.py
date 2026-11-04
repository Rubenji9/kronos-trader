"""Escenarios de precio con Kronos.

`KronosPredictor.predict()` promedia todas las muestras en una sola línea. Aquí
replicamos su normalización pero lanzamos N caminos en el mismo batch y los
conservamos por separado: eso es el "abanico" del panel y, sobre todo, nos da
una distribución (probabilidad de subida, percentiles) en vez de un único número.

Si Kronos no está disponible (sin torch / sin pesos) se usa un bootstrap de
rendimientos históricos, marcado como `engine = "bootstrap-demo"`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PRICE = ["open", "high", "low", "close"]
FEATS = PRICE + ["volume", "amount"]


@dataclass
class Forecast:
    ticker: str
    engine: str
    last_close: float
    dates: list[str]
    paths: np.ndarray                  # (samples, horizon) cierres simulados
    stats: dict = field(default_factory=dict)

    def to_json(self, max_paths: int = 64) -> dict:
        return {
            "engine": self.engine,
            "last_close": round(self.last_close, 4),
            "dates": self.dates,
            "paths": np.round(self.paths[:max_paths], 4).tolist(),
            "stats": self.stats,
        }


def _future_dates(last: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)


def _time_feats(ts: pd.Series) -> np.ndarray:
    ts = pd.Series(pd.to_datetime(ts))
    return np.stack([ts.dt.minute, ts.dt.hour, ts.dt.weekday, ts.dt.day, ts.dt.month], axis=1).astype(np.float32)


def summarize(paths: np.ndarray, last_close: float) -> dict:
    term = paths[:, -1]
    rets = term / last_close - 1
    p10, p50, p90 = np.percentile(term, [10, 50, 90])
    return {
        "samples": int(paths.shape[0]),
        "horizon": int(paths.shape[1]),
        "p_up": round(float((term > last_close).mean()), 4),
        "n_up": int((term > last_close).sum()),
        "n_down": int((term <= last_close).sum()),
        "exp_return": round(float(rets.mean()), 5),
        "median_return": round(float(np.median(rets)), 5),
        "p10": round(float(p10), 4),
        "p50": round(float(p50), 4),
        "p90": round(float(p90), 4),
        "dispersion": round(float(rets.std()), 5),        # incertidumbre del modelo
        "mean_path": np.round(paths.mean(axis=0), 4).tolist(),
    }


# ───────────────────────── Kronos ─────────────────────────
class KronosEngine:
    name = "kronos"

    def __init__(self, model_id: str, tokenizer_id: str, max_context: int = 512, device: str | None = None):
        kronos_path = os.environ.get("KRONOS_PATH", os.path.join(os.path.dirname(__file__), "..", "vendor", "Kronos"))
        sys.path.insert(0, os.path.abspath(kronos_path))
        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

        tok = KronosTokenizer.from_pretrained(tokenizer_id)
        mdl = Kronos.from_pretrained(model_id)
        tok.eval(); mdl.eval()
        self.predictor = KronosPredictor(mdl, tok, device=device, max_context=max_context)
        self.name = f"kronos:{model_id.split('/')[-1]}"

    @classmethod
    def from_modules(cls, predictor, name="kronos:custom"):
        obj = cls.__new__(cls)
        obj.predictor = predictor
        obj.name = name
        return obj

    def paths(self, df: pd.DataFrame, horizon: int, samples: int, T: float, top_p: float, seed: int = 0) -> np.ndarray:
        import torch

        torch.manual_seed(seed)
        p = self.predictor
        x = df[FEATS].values.astype(np.float32)
        x_stamp = _time_feats(df.index.to_series())
        y_idx = _future_dates(df.index[-1], horizon)
        y_stamp = _time_feats(y_idx.to_series())

        mean, std = x.mean(axis=0), x.std(axis=0)
        xn = np.clip((x - mean) / (std + 1e-5), -p.clip, p.clip)

        xb = np.repeat(xn[None], samples, axis=0)
        xsb = np.repeat(x_stamp[None], samples, axis=0)
        ysb = np.repeat(y_stamp[None], samples, axis=0)
        # sample_count=1 por fila → cada fila del batch es un escenario independiente
        preds = p.generate(xb, xsb, ysb, horizon, T, 0, top_p, 1, False)   # (samples, horizon, 6)
        preds = preds * (std + 1e-5) + mean
        return preds[:, :, FEATS.index("close")]


# ───────────────────── Fallback demo ─────────────────────
class BootstrapEngine:
    """Remuestrea rendimientos diarios recientes. NO es una predicción: sirve para
    probar el circuito y el panel cuando Kronos no está instalado."""
    name = "bootstrap-demo"

    def paths(self, df: pd.DataFrame, horizon: int, samples: int, T: float, top_p: float, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        r = np.log(df["close"]).diff().dropna().values[-250:]
        draws = rng.choice(r, size=(samples, horizon), replace=True) * T
        return float(df["close"].iloc[-1]) * np.exp(np.cumsum(draws, axis=1))


def load_engine(cfg: dict, demo: bool = False):
    if demo:
        return BootstrapEngine()
    try:
        return KronosEngine(cfg["model"], cfg["tokenizer"])
    except Exception as e:  # noqa: BLE001
        print(f"[forecast] Kronos no disponible ({e!r}); uso bootstrap-demo")
        return BootstrapEngine()


def run_forecast(engine, ticker: str, df: pd.DataFrame, cfg: dict, seed: int = 0) -> Forecast:
    horizon, samples = int(cfg["horizon"]), int(cfg["samples"])
    paths = engine.paths(df, horizon, samples, float(cfg.get("temperature", 1.0)), float(cfg.get("top_p", 0.9)), seed)
    last = float(df["close"].iloc[-1])
    dates = [d.strftime("%Y-%m-%d") for d in _future_dates(df.index[-1], horizon)]
    return Forecast(ticker, engine.name, last, dates, paths, summarize(paths, last))
