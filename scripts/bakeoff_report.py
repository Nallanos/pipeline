"""Page de comparaison du bake-off : pour chaque scène, Klein (images du
pipeline), Qwen-Image-Edit-2511 (qwen_test/results.json) et chaque modèle
de qwen_test/bakeoff/results.json, avec un filtre par modèle.

    uv run python scripts/bakeoff_report.py
"""

from __future__ import annotations

import html
import json
import statistics
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "qwen_test"
OUT = BENCH / "bakeoff" / "report"
IMG = OUT / "img"

# Proportion de scènes avec référence de décor (video5 + escobart).
DECOR_SHARE = 0.16
IMAGES_PER_VIDEO = 405
GPU_PRICE = 1.09  # L40S secure, $/h facturés

# Ordre d'affichage, libellés, licence, nombre de références accepté, réglages.
MODELS = {
    "klein": ("FLUX.2 Klein 4B", "Apache 2.0", "4 pas · local RX 9060 XT", "chaîne ReferenceLatent"),
    "qwen": ("Qwen-Image-Edit-2511", "Apache 2.0", "Lightning 4 pas", "3 réf."),
    "firered": ("FireRed-Image-Edit 1.1", "Apache 2.0", "Lightning 8 pas", "3 réf."),
    "mage": ("Mage-Flow Edit Turbo", "MIT", "4 pas", "3 réf."),
    "hidream": ("HiDream-O1 Dev", "MIT", "28 pas", "10 réf."),
    "joyai": ("JoyAI-Image-Edit", "Apache 2.0", "40 pas, cfg 4", "6 réf."),
    "boogu": ("Boogu-Image Edit Turbo", "Apache 2.0", "4 pas", "1 réf. (costume seul)"),
    "longcat": ("LongCat-Image-Edit Turbo", "Apache 2.0", "8 pas", "1 réf. (costume seul)"),
}


def thumb(src: str, name: str, width: int = 720) -> str:
    target = IMG / f"{name}.jpg"
    if not target.exists():
        with Image.open(src) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((width, width))
            rgb.save(target, quality=80, optimize=True)
    return f"img/{target.name}"


def e(text: object) -> str:
    return html.escape(str(text))


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((BENCH / "results.json").read_text(encoding="utf-8"))
    bake = json.loads((BENCH / "bakeoff" / "results.json").read_text(encoding="utf-8"))

    images: dict[tuple[str, str, int], str] = {}
    timings: dict[str, list[dict]] = {}
    errors: dict[str, list[str]] = {}
    for s in scenes:
        images[("klein", s["project"], s["scene"])] = s["klein"]
        images[("qwen", s["project"], s["scene"])] = s["qwen"]
        if not s["warmup"] and (s.get("server_s") or 0) > 1:
            timings.setdefault("qwen", []).append({"wall": s["wall_s"], "server": s["server_s"], "decor": bool(s["decor"])})
    decor_of = {(s["project"], s["scene"]): bool(s["decor"]) for s in scenes}
    for r in bake:
        if "error" in r:
            errors.setdefault(r["model"], []).append(r["error"])
            continue
        images[(r["model"], r["project"], r["scene"])] = r["image"]
        if not r["warmup"] and (r.get("server_s") or 0) > 0.5:
            timings.setdefault(r["model"], []).append(
                {"wall": r["wall_s"], "server": r["server_s"], "decor": r["refs"] > 1}
            )

    present = [m for m in MODELS if any(k[0] == m for k in images)]

    rows = []
    for m in present:
        label, licence, steps, refs = MODELS[m]
        t = timings.get(m, [])
        done = sum(1 for k in images if k[0] == m)
        if m == "klein":
            rows.append(f"<tr><th scope='row'>{e(label)}</th><td>{e(licence)}</td><td>{e(steps)}</td><td>{e(refs)}</td>"
                        "<td class='n'>~18 s (local)</td><td class='n'>0 $ (électricité)</td><td class='n'>20/20</td></tr>")
            continue
        if t:
            solo = [x for x in t if not x["decor"]] or t
            duo = [x for x in t if x["decor"]] or solo
            w = (1 - DECOR_SHARE) * statistics.median(x["wall"] for x in solo) + DECOR_SHARE * statistics.median(x["wall"] for x in duo)
            s_ = (1 - DECOR_SHARE) * statistics.median(x["server"] for x in solo) + DECOR_SHARE * statistics.median(x["server"] for x in duo)
            cost = IMAGES_PER_VIDEO * w / 3600 * GPU_PRICE
            time_cell = f"{fmt(s_)} s <small>/ {fmt(w)} s mur</small>"
            cost_cell = f"≈ {fmt(cost, 2)} $"
        else:
            time_cell, cost_cell = "—", "—"
        err = f" <small>({len(errors[m])} échec)</small>" if m in errors else ""
        rows.append(f"<tr><th scope='row'>{e(label)}</th><td>{e(licence)}</td><td>{e(steps)}</td><td>{e(refs)}</td>"
                    f"<td class='n'>{time_cell}</td><td class='n'>{cost_cell}</td><td class='n'>{done}/20{err}</td></tr>")

    chips = "".join(
        f'<label class="chip"><input type="checkbox" id="m-{m}" data-m="{m}" checked> {e(MODELS[m][0])}</label>'
        for m in present
    )

    cards = []
    for s in scenes:
        key = (s["project"], s["scene"])
        cells = []
        for m in present:
            src = images.get((m, *key))
            if not src:
                continue
            path = thumb(src, f"{m}_{s['project']}_{s['scene']:04d}")
            cells.append(
                f'<figure data-m="{m}"><img src="{path}" alt="{e(MODELS[m][0])}, {e(s["project"])} scène {s["scene"]}" loading="lazy">'
                f"<figcaption>{e(MODELS[m][0])}</figcaption></figure>"
            )
        c = thumb(s["costume"], f"ref_{s['project']}_c_{Path(s['costume']).stem}", 320)
        d = thumb(s["decor"], f"ref_{s['project']}_d_{Path(s['decor']).stem}", 320) if s["decor"] else None
        refs = f'<img src="{c}" alt="Référence de costume">' + (f'<img src="{d}" alt="Référence de décor">' if d else "")
        cards.append(f"""
<article class="scene">
  <header><h3>{e(s['project'])} · scène {s['scene']}</h3><span class="meta">{'costume + décor' if decor_of[key] else 'costume seul'}</span></header>
  <div class="foot"><div class="refs">{refs}</div><p class="prompt"><span>prompt</span> {e(s['subject'])}</p></div>
  <div class="grid">{''.join(cells)}</div>
</article>""")

    page = (Path(__file__).parent / "bakeoff_report_template.html").read_text(encoding="utf-8")
    page = page.replace("{{ROWS}}", "".join(rows)).replace("{{CHIPS}}", chips).replace("{{CARDS}}", "".join(cards))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html", len(list(IMG.iterdir())), "vignettes")
    for m in present:
        print(m, len(timings.get(m, [])), "temps", errors.get(m, [])[:1])


if __name__ == "__main__":
    main()
