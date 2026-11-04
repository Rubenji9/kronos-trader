"""Genera un HTML autocontenido (panel + datos incrustados) para compartir o abrir sin servidor."""
import json, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
html = (root / "docs/index.html").read_text()
state = (root / "docs/data/state.json").read_text()
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else root / "preview.html")
out.write_text(html.replace("<script>\n(() => {", f"<script>window.__STATE__={state};</script>\n<script>\n(() => {{", 1))
print(out)
