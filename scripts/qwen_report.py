"""Page de comparaison Klein vs Qwen à partir de qwen_test/results.json et
variants.json. Écrit qwen_test/report/index.html + les vignettes JPEG
(les PNG pleine taille dépasseraient la limite de publication).

    uv run python scripts/qwen_report.py
"""

from __future__ import annotations

import html
import json
import statistics
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "qwen_test"
REPORT = OUT / "report"
IMG = REPORT / "img"

# Lecture à l'œil des 20 paires (07/10/2026). « qwen » / « klein » = qui
# l'emporte sur cette scène, « = » = équivalent.
NOTES: dict[tuple[str, int], tuple[str, str]] = {
    ("video5", 15): ("=", "Les deux suivent le décor de référence."),
    ("video5", 37): ("=", "Bulles de dialogue en charabia des deux côtés : le prompt dit « confessing words »."),
    ("video5", 83): ("klein", "Qwen ignore le décor (salle ronde) et invente un palais avec une enseigne illisible."),
    ("video5", 105): ("=", "Équivalent."),
    ("video5", 171): ("klein", "Qwen donne le visage du héros à « Hwang » : deux bldguy identiques."),
    ("video5", 181): ("klein", "Klein reprend le bureau de référence, Qwen le simplifie en plan frontal."),
    ("video5", 240): ("=", "Équivalent ; Qwen écrit un faux mot sur la feuille."),
    ("video5", 248): ("=", "Équivalent."),
    ("video5", 305): ("=", "Deux bldguy des deux côtés : le prompt dit « side by side »."),
    ("video5", 310): ("qwen", "Qwen : vraie marche de profil, sol pavé, plus vivant."),
    ("escobart", 32): ("=", "Équivalent (temps exclu : image servie par le cache de mon essai)."),
    ("escobart", 35): ("=", "Équivalent ; Qwen garde le sac à dos de la référence."),
    ("escobart", 93): ("klein", "Qwen recopie le revolver et la pose frontale de la référence de costume."),
    ("escobart", 99): ("qwen", "Qwen fidèle au décor et ajoute l'interlocuteur."),
    ("escobart", 150): ("qwen", "Klein fait repousser des cheveux ; Qwen garde le crâne rasé (mais copie le pistolet)."),
    ("escobart", 162): ("=", "Les deux littéralisent mal « Avianca bomb » (logo écrit / bombe dessinée)."),
    ("escobart", 220): ("=", "Équivalent."),
    ("escobart", 223): ("qwen", "Klein oublie le protagoniste ; Qwen le place dans le plan large."),
    ("escobart", 271): ("qwen", "Qwen suit le prompt (dans la voiture, radio) ; Klein le met à moto."),
    ("escobart", 284): ("=", "Équivalent, intérieur de voiture plus propre chez Qwen."),
}


def thumb(src: str, name: str, width: int = 960) -> str:
    target = IMG / f"{name}.jpg"
    if not target.exists():
        with Image.open(src) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((width, width))
            rgb.save(target, quality=82, optimize=True)
    return f"img/{target.name}"


def e(text: object) -> str:
    return html.escape(str(text))


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    variants = json.loads((OUT / "variants.json").read_text(encoding="utf-8"))
    by_key = {(r["project"], r["scene"]): r for r in results}

    timed = [r for r in results if not r["warmup"] and (r.get("server_s") or 0) > 1]
    solo = [r for r in timed if not r["decor"]]
    duo = [r for r in timed if r["decor"]]
    med = statistics.median
    stats = {
        "solo_wall": med(r["wall_s"] for r in solo), "solo_srv": med(r["server_s"] for r in solo),
        "duo_wall": med(r["wall_s"] for r in duo), "duo_srv": med(r["server_s"] for r in duo),
    }
    print(stats)

    cards = []
    for r in results:
        key = (r["project"], r["scene"])
        verdict, note = NOTES.get(key, ("=", ""))
        k = thumb(r["klein"], f"{r['project']}_{r['scene']:04d}_klein")
        q = thumb(r["qwen"], f"{r['project']}_{r['scene']:04d}_qwen")
        c = thumb(r["costume"], f"ref_{r['project']}_costume_{Path(r['costume']).stem}", 320)
        d = thumb(r["decor"], f"ref_{r['project']}_decor_{Path(r['decor']).stem}", 320) if r["decor"] else None
        timing = (
            "cache" if (r.get("server_s") or 0) < 1
            else f"{r['server_s']:.1f} s serveur · {r['wall_s']:.1f} s mur"
        )
        badge = {"qwen": "Qwen mieux", "klein": "Klein mieux", "=": "Équivalent"}[verdict]
        refs = f'<img src="{c}" alt="Référence de costume" loading="lazy">' + (
            f'<img src="{d}" alt="Référence de décor" loading="lazy">' if d else ""
        )
        cards.append(f"""
<article class="pair" data-verdict="{verdict}">
  <header>
    <h3>{e(r['project'])} · scène {r['scene']}</h3>
    <span class="chip v-{'eq' if verdict == '=' else verdict}">{badge}</span>
    <span class="meta">{'costume + décor' if d else 'costume seul'} · {timing}</span>
  </header>
  <div class="imgs">
    <figure><img src="{k}" alt="Klein, scène {r['scene']}" loading="lazy"><figcaption>FLUX.2 Klein 4B</figcaption></figure>
    <figure><img src="{q}" alt="Qwen, scène {r['scene']}" loading="lazy"><figcaption>Qwen-Image-Edit-2511</figcaption></figure>
  </div>
  <div class="foot">
    <div class="refs">{refs}</div>
    <div class="txt"><p class="note">{e(note)}</p><p class="prompt"><span>prompt</span> {e(r['subject'])}</p></div>
  </div>
</article>""")

    var_rows = []
    for v in variants:
        r = by_key[(v["project"], v["scene"])]
        k = thumb(r["klein"], f"{r['project']}_{r['scene']:04d}_klein")
        q = thumb(r["qwen"], f"{r['project']}_{r['scene']:04d}_qwen")
        x = thumb(v["image"], f"{v['variant']}_{r['project']}_{r['scene']:04d}")
        label = "Qwen, style revu" if v["variant"] == "style" else "Qwen sans Lightning (40 pas)"
        var_rows.append(f"""
<div class="vrow" data-variant="{v['variant']}">
  <p class="vlabel">{e(r['project'])} · scène {r['scene']} <span class="meta">{label} : {v['server_s']:.0f} s serveur</span></p>
  <div class="imgs three">
    <figure><img src="{k}" alt="" loading="lazy"><figcaption>Klein</figcaption></figure>
    <figure><img src="{q}" alt="" loading="lazy"><figcaption>Qwen, style actuel</figcaption></figure>
    <figure><img src="{x}" alt="" loading="lazy"><figcaption>{label}</figcaption></figure>
  </div>
</div>""")

    page = (Path(__file__).parent / "qwen_report_template.html").read_text(encoding="utf-8")
    page = page.replace("{{CARDS}}", "".join(cards))
    page = page.replace("{{VARIANTS_STYLE}}", "".join(r for r in var_rows if 'data-variant="style"' in r))
    page = page.replace("{{VARIANTS_NOLIGHT}}", "".join(r for r in var_rows if 'data-variant="nolightning"' in r))
    for key, value in stats.items():
        page = page.replace("{{" + key + "}}", f"{value:.1f}".replace(".", ","))
    (REPORT / "index.html").write_text(page, encoding="utf-8")
    print(REPORT / "index.html", len(list(IMG.iterdir())), "images")


if __name__ == "__main__":
    main()
