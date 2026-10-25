"""Provider bouchon — PNG généré localement, instantané et gratuit.

Il n'existe pas pour faire joli : c'est lui qui rend le pipeline exécutable de
bout en bout sans dépendre du GPU. Tant que ComfyUI n'est pas branché, ou s'il
tombe, le rendu complet reste vérifiable — durées, synchronisation audio,
enchaînement des coupes. Il remplace le provider payant que le SPEC prévoyait
en S1.

Chaque image porte son index, son seed et son prompt, ce qui permet de lire un
défaut de synchronisation directement à l'écran.
"""

from __future__ import annotations

import colorsys
import io
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

from ..style import StyleConfig

# Clé tEXt posée dans le PNG quand ce provider sert de REPLI (et pas de
# provider principal). C'est le seul moyen fiable de reconnaître un bouchon
# plus tard : il s'écrit sur le même chemin `NNNN.png` qu'une vraie image, et
# `images.json` ne suffit pas — une scène interrompue par le coupe-circuit
# n'y a aucune entrée, et rien ne distinguait alors son vieux bouchon d'une
# image réussie. Vu en pratique (16/09, video5) : 49 bouchons du run
# précédent auraient été gardés tels quels par la reprise.
FALLBACK_MARKER = "pipeline-fallback"


class StubProvider:
    name = "stub"

    def __init__(self, fallback: bool = False) -> None:
        # `fallback=True` seulement quand images.py s'en sert comme repli
        # d'un vrai provider. En provider principal (PIPELINE_PROVIDER=stub,
        # le défaut, utilisé pour vérifier durées et montage sans GPU) les
        # images ne sont PAS marquées : elles sont le résultat attendu du
        # run, pas des trous, et doivent survivre à une reprise.
        self.fallback = fallback

    def generate(
        self,
        prompt: str,
        style: StyleConfig,
        seed: int,
        reference: Path | None = None,
        decor_reference: Path | None = None,
    ) -> bytes:
        # Teinte dérivée du seed : deux scènes voisines sont visuellement
        # distinctes, donc une coupe manquée se voit immédiatement.
        hue = (seed % 997) / 997.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.22, 0.93)
        background = (int(r * 255), int(g * 255), int(b * 255))
        ink = (30, 30, 46)

        image = Image.new("RGB", (style.width, style.height), background)
        draw = ImageDraw.Draw(image)

        margin = style.width // 20
        draw.rectangle(
            [margin, margin, style.width - margin, style.height - margin],
            outline=ink,
            width=max(2, style.width // 400),
        )

        header = f"#{seed % 10000:04d}  seed {seed}"
        draw.text((margin * 2, margin * 2), header, fill=ink)

        wrapped = textwrap.fill(prompt, width=max(20, style.width // 14))
        draw.multiline_text((margin * 2, margin * 3), wrapped, fill=ink, spacing=6)

        metadata = PngInfo()
        if self.fallback:
            metadata.add_text(FALLBACK_MARKER, "1")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", pnginfo=metadata)
        return buffer.getvalue()
