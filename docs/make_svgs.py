"""Compile the LaTeX equations in docs/ into the light and dark SVGs used by the README.

Each docs/<name>.tex gives docs/<name>-light.svg and docs/<name>-dark.svg.
Requires `latex` and `dvisvgm` (both included in TeX Live / MacTeX).

Usage: python docs/make_svgs.py
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent

# Background and text colors, matching GitHub's light and dark themes
THEMES = {"light": ("#ffffff", "#1f2328"),
          "dark":  ("#0d1117", "#e6edf3")}

PX_PER_PT = 2.0  # size in px per TeX pt
MARGIN = 5.0     # margin around the equation, in TeX pt


def tex_to_svg(tex):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(tex, tmp)
        subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                       cwd=tmp, check=True, capture_output=True)
        result = subprocess.run(["dvisvgm", "--no-fonts", "--optimize", "--stdout", tex.stem + ".dvi"],
                                cwd=tmp, check=True, capture_output=True, text=True)
    return result.stdout


def themed_svgs(raw):
    x, y, w, h = map(float, re.search(r"viewBox=['\"]([^'\"]+)['\"]", raw).group(1).split())
    x, y, w, h = x - MARGIN, y - MARGIN, w + 2 * MARGIN, h + 2 * MARGIN
    body = re.sub(r"<!--.*?-->\n?", "", raw)
    for theme, (bg, fg) in THEMES.items():
        header = (f"<svg version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' "
                  f"width='{w * PX_PER_PT:.2f}px' height='{h * PX_PER_PT:.2f}px' viewBox='{x:.3f} {y:.3f} {w:.3f} {h:.3f}' fill='{fg}'>"
                  f"<rect x='{x:.3f}' y='{y:.3f}' width='{w:.3f}' height='{h:.3f}' fill='{bg}'/>")
        yield theme, re.sub(r"<svg[^>]*>", lambda _: header, body, count=1)


if __name__ == "__main__":
    for tex in sorted(DOCS.glob("*.tex")):
        for theme, svg in themed_svgs(tex_to_svg(tex)):
            out = DOCS / f"{tex.stem}-{theme}.svg"
            out.write_text(svg)
            print(f"{out.relative_to(DOCS.parent)}")
