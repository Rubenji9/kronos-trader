"""Genera un HTML autocontenido (panel + datos incrustados) para compartir o abrir sin servidor."""
import json, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
data = root / "docs" / "data"
html = (root / "docs/index.html").read_text(encoding="utf-8")
state = (data / "state.json").read_text(encoding="utf-8")
act = (data / "activity.json").read_text(encoding="utf-8") if (data / "activity.json").exists() else "[]"
intra = (data / "intraday.json").read_text(encoding="utf-8") if (data / "intraday.json").exists() else '{"points":[]}'
inject = f"<script>window.__STATE__={state};window.__ACTIVITY__={act};window.__INTRADAY__={intra};</script>\n"
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else root / "preview.html")
out.write_text(html.replace("<script>\n(() => {", inject + "<script>\n(() => {", 1), encoding="utf-8")
print(out)
