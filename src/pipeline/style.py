"""Chargement de la style bible (`config/style.yaml`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from .project import CONFIG_DIR


@dataclass(frozen=True)
class StyleConfig:
    name: str
    medium: str
    palette: str
    framing: str
    lighting: str
    suffix: str
    negative: str
    width: int
    height: int
    steps: int
    cfg: float

    def bible(self) -> str:
        """La partie descriptive, telle qu'elle est donnée au LLM à l'étape 4."""
        return (
            f"Trait     : {self.medium}\n"
            f"Palette   : {self.palette}\n"
            f"Cadrage   : {self.framing}\n"
            f"Lumière   : {self.lighting}"
        )

    def full_prompt(self, subject: str) -> str:
        """Le prompt envoyé au générateur d'images.

        `framing` était défini dans `style.yaml` et donné au LLM (via `bible()`)
        mais jamais transmis à SDXL — bug resté invisible parce que le prompt
        restait syntaxiquement valide sans lui. C'est probablement ce qui a
        laissé SDXL improviser un cadrage différent de « plan moyen » sur
        certaines scènes (constaté sur un run réel, 14/08).

        `suffix` n'est délibérément pas concaténé ici : il ne fait que répéter
        des interdictions déjà dans `negative` (texte, filigrane, signature),
        et n'apporterait rien.

        Correction (15/08) : l'affirmation précédente disait que dépasser la
        limite ~75 tokens de l'encodeur CLIP tronquait la fin du prompt en
        silence. Faux pour ComfyUI — vérifié dans `comfy/sd1_clip.py` (fonction
        de tokenisation) : au-delà de 77 tokens, ComfyUI découpe le prompt en
        plusieurs segments encodés séparément (chunking), il ne coupe rien.
        `suffix` reste exclu pour la raison ci-dessus, pas par crainte d'une
        troncature qui n'existe pas.
        """
        return f"{subject}. {self.medium}. {self.palette}. {self.framing}. {self.lighting}"


def load(draft: bool = False) -> StyleConfig:
    raw: dict[str, Any] = yaml.safe_load((CONFIG_DIR / "style.yaml").read_text(encoding="utf-8"))
    overrides: dict[str, Any] = raw.get("draft", {}) if draft else {}

    def get(key: str, default: Any = None) -> Any:
        return overrides.get(key, raw.get(key, default))

    return StyleConfig(
        name=raw["name"],
        medium=raw["medium"].strip(),
        palette=raw["palette"].strip(),
        framing=raw["framing"].strip(),
        lighting=raw["lighting"].strip(),
        suffix=raw["suffix"].strip(),
        negative=raw["negative"].strip(),
        width=int(get("width")),
        height=int(get("height")),
        steps=int(get("steps")),
        cfg=float(get("cfg")),
    )
