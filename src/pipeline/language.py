"""Configuration par langue (`config/languages.yaml`).

La langue est fixée à la création du projet et persistée à côté du script :
un projet a un script, donc une langue, et elle ne change plus. Les étapes la
relisent plutôt que de la recevoir en option, pour qu'un `run --step segment`
lancé six mois plus tard reste cohérent avec le TTS qui l'a précédé.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .project import CONFIG_DIR, MODELS_DIR


@dataclass(frozen=True)
class LanguageConfig:
    code: str
    voice: str
    whisper: str
    connectives: frozenset[str]

    @property
    def voice_path(self) -> Path:
        return MODELS_DIR / "piper" / f"{self.voice}.onnx"


def _raw() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(
        (CONFIG_DIR / "languages.yaml").read_text(encoding="utf-8")
    )
    return data


def default_code() -> str:
    return str(_raw().get("default", "fr"))


def available() -> list[str]:
    return sorted(k for k, v in _raw().items() if isinstance(v, dict))


def load(code: str) -> LanguageConfig:
    data = _raw()
    entry = data.get(code)
    if not isinstance(entry, dict):
        raise ValueError(f"langue inconnue : {code!r} (parmi {', '.join(available())})")
    return LanguageConfig(
        code=code,
        voice=entry["voice"],
        whisper=entry["whisper"],
        connectives=frozenset(entry.get("connectives", [])),
    )


def for_project(project_dir: Path) -> LanguageConfig:
    """Relit la langue persistée du projet."""
    marker = project_dir / "language.txt"
    code = marker.read_text(encoding="utf-8").strip() if marker.exists() else default_code()
    return load(code)
