"""Emplacements des artefacts et contrat d'étape.

Le contrat d'étape (SPEC §1) est une fonction, pas un framework : une étape lit
des fichiers sous `projects/<id>/`, écrit exactement un artefact, et ne fait
rien si cet artefact existe déjà. Il n'y a pas d'autre abstraction dans le
système et il ne faut pas en introduire.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = ROOT / "projects"
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class Project:
    """Les chemins d'un projet. Aucune logique, seulement des emplacements."""

    id: str

    @property
    def dir(self) -> Path:
        return PROJECTS_DIR / self.id

    @property
    def script(self) -> Path:
        return self.dir / "script.txt"

    @property
    def script_normalized(self) -> Path:
        """Source de vérité de tout le pipeline (SPEC §4, étape 1).

        Les deux invariants se mesurent contre ce fichier, jamais contre
        `script.txt` brut.
        """
        return self.dir / "script_normalized.txt"

    @property
    def audio(self) -> Path:
        return self.dir / "audio.wav"

    @property
    def alignment(self) -> Path:
        return self.dir / "alignment.json"

    @property
    def boundaries(self) -> Path:
        return self.dir / "boundaries.json"

    @property
    def scenes(self) -> Path:
        return self.dir / "scenes.json"

    @property
    def images(self) -> Path:
        return self.dir / "images"

    @property
    def segments(self) -> Path:
        return self.dir / "segments"

    @property
    def out(self) -> Path:
        return self.dir / "out.mp4"


@dataclass(frozen=True)
class Options:
    """Options de run, passées telles quelles à chaque étape."""

    force: bool = False
    draft: bool = False
    sample: bool = False


def normalize(text: str) -> str:
    """La normalisation du script. Une seule définition pour tout le système.

    Toute divergence entre la normalisation appliquée au TTS et celle appliquée
    à la vérification des invariants produit un décalage silencieux (SPEC §7).
    """
    return " ".join(text.split())


def write_atomic(path: Path, data: bytes | str) -> None:
    """Écrit via un fichier temporaire puis renomme.

    Sans ça, une interruption pendant l'écriture laisse un artefact tronqué que
    le contrat d'étape considérera comme fait, et l'étape ne sera jamais rejouée.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(data, str):
        tmp.write_text(data, encoding="utf-8")
    else:
        tmp.write_bytes(data)
    tmp.replace(path)
