"""Descarga de velas OHLCV.

Fuente real: Yahoo Finance vía yfinance (gratis, sin API key).
Modo demo: series sintéticas (movimiento browniano con régimen de volatilidad)
para probar todo el circuito sin conexión. Nunca operes con datos demo.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close", "volume", "amount"]


def fetch_yahoo(ticker: str, lookback: int, interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf

    period = "5y" if interval == "1d" else "60d"
    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Sin datos para {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):          # yfinance >= 0.2.40
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "timestamps"
    return df.tail(lookback)


def synthetic(ticker: str, lookback: int, seed: int | None = None) -> pd.DataFrame:
    """Serie diaria realista-ish para pruebas offline."""
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(ticker.encode()))
    start_price = {"AAPL": 190, "MSFT": 410, "NVDA": 120, "AMZN": 185, "SPY": 560}.get(ticker, 100)
    n = lookback
    vol = np.empty(n)
    v = 0.018
    for i in range(n):                       # volatilidad con memoria (tipo GARCH light)
        v = 0.94 * v + 0.06 * 0.018 + 0.004 * rng.standard_normal()
        vol[i] = max(v, 0.006)
    drift = 0.0004
    rets = drift + vol * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]] * np.exp(0.003 * rng.standard_normal(n))
    spread = np.abs(rng.standard_normal(n)) * vol * close
    high = np.maximum(open_, close) + spread * 0.6
    low = np.minimum(open_, close) - spread * 0.6
    volume = (5e7 * np.exp(0.35 * rng.standard_normal(n)) * (1 + 8 * np.abs(rets))).round()
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": volume}, index=idx)
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df.index.name = "timestamps"
    return df


def get_candles(ticker: str, lookback: int, interval: str = "1d", demo: bool = False) -> pd.DataFrame:
    if demo:
        return synthetic(ticker, lookback)
    return fetch_yahoo(ticker, lookback, interval)
