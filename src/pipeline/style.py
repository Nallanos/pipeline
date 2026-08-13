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
        """Le prompt envoyé au générateur d'images."""
        return f"{subject}. {self.medium}. {self.palette}. {self.lighting}. {self.suffix}"


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
