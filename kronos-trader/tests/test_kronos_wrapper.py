"""Valida el envoltorio de Kronos con un modelo de pesos aleatorios (sin descargar nada).
Comprueba formas, desnormalización y que cada escenario es distinto."""
import os, sys, time
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.environ.get("KRONOS_PATH", os.path.join(ROOT, "vendor", "Kronos")))
from model import Kronos, KronosTokenizer, KronosPredictor
from trader.data import synthetic
from trader.forecast import KronosEngine, run_forecast

torch.manual_seed(0)
tok = KronosTokenizer(d_in=6, d_model=64, n_heads=4, ff_dim=128, n_enc_layers=2, n_dec_layers=2,
                      ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=6, s2_bits=6,
                      beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4).eval()
mdl = Kronos(s1_bits=6, s2_bits=6, n_layers=2, d_model=64, n_heads=4, ff_dim=128, ffn_dropout_p=0,
             attn_dropout_p=0, resid_dropout_p=0, token_dropout_p=0, learn_te=True).eval()
eng = KronosEngine.from_modules(KronosPredictor(mdl, tok, device="cpu", max_context=512), "kronos:random-test")

df = synthetic("AAPL", 400)
t = time.time()
fc = run_forecast(eng, "AAPL", df, {"horizon": 10, "samples": 32, "temperature": 1.0, "top_p": 0.9})
print(f"{time.time()-t:.1f}s  paths={fc.paths.shape}  last={fc.last_close:.2f}")
assert fc.paths.shape == (32, 10)
assert np.isfinite(fc.paths).all()
assert np.unique(fc.paths[:, -1].round(4)).size > 16, "los escenarios deberían diferir entre sí"
lo, hi = df["close"].min(), df["close"].max()
assert (fc.paths > lo * 0.3).all() and (fc.paths < hi * 3).all(), "desnormalización fuera de rango"
print("stats:", {k: v for k, v in fc.stats.items() if k != "mean_path"})
print("OK")
