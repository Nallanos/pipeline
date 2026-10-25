"""Emplacements des artefacts et contrat d'étape.

Le contrat d'étape (SPEC §1) est une fonction, pas un framework : une étape lit
des fichiers sous `projects/<id>/`, écrit exactement un artefact, et ne fait
rien si cet artefact existe déjà. Il n'y a pas d'autre abstraction dans le
système et il ne faut pas en introduire.
"""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = ROOT / "projects"
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
MODELS_DIR = ROOT / "models"


def safe_wav_frames(path: Path) -> int:
    """Nombre de frames d'un WAV, borné par la taille réelle du fichier.

    `wave` fait confiance à `nframes` déclaré dans l'en-tête sans jamais le
    confronter à la taille du fichier. Vu deux fois en pratique avec l'API
    Fish Audio (`steps/align.py`, `steps/tts.py`) et une troisième en la
    corrigeant à moitié (`steps/render.py::_card_audio`, 04/09) : un en-tête
    déclarant nframes proche de 2**31 (dépassement d'entier 32 bits côté
    serveur) sur un fichier bien plus court — sans ce plafond, tout ce qui
    consomme cette durée (segmentation, ffmpeg -d=...) part sur des heures
    au lieu de secondes. Centralisé ici après l'avoir corrigé séparément
    deux fois : la duplication avait déjà laissé passer l'oubli une fois.
    """
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        frame_bytes = wav.getsampwidth() * wav.getnchannels()
    if not frame_bytes:
        return frames
    return min(frames, path.stat().st_size // frame_bytes)


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
    def levels(self) -> Path:
        """Cartons de palier ("Niveau 1 : ..."), extraits du script à l'import
        par `clean_source()`. Absent si le script n'a pas de marqueur `NIVEAU:`
        — pas toutes les vidéos ne sont un format « paliers »."""
        return self.dir / "levels.json"

    @property
    def section_bible(self) -> Path:
        """Descriptif costume/décor par palier, dérivé de `levels.json` à
        l'étape prompts (SPEC : voir `section_bible.py`). Artefact de
        diagnostic, pas suivi par le contrat d'étape — recalculé à chaque
        exécution réelle de `prompts.run()`. Absent si `levels.json` l'est."""
        return self.dir / "section_bible.json"

    @property
    def images(self) -> Path:
        return self.dir / "images"

    @property
    def section_refs(self) -> Path:
        """Une image de référence par palier (`NIVEAU:`), costume de la
        section sur fond neutre — voir `steps/refs.py`. `section_refs.json`
        fait foi (`{"refs": {}}` pour un projet sans paliers)."""
        return self.dir / "section_refs"

    @property
    def location_bible(self) -> Path:
        """Lieux récurrents détectés par palier (SPEC : voir
        `location_bible.py`). Artefact de diagnostic, comme `section_bible`
        — recalculé à chaque exécution réelle de `prompts.run()`."""
        return self.dir / "location_bible.json"

    @property
    def cast_bible(self) -> Path:
        """Personnages secondaires récurrents et leur description figée
        (voir `cast_bible.py`). Calculé une fois à l'étape prompts, relu en
        reprise, avec ou sans paliers."""
        return self.dir / "cast_bible.json"

    @property
    def location_refs(self) -> Path:
        """Un plan d'établissement SANS personnage par lieu récurrent — voir
        `steps/location_refs.py`. `location_refs.json` fait foi (`{"refs":
        {}}` si aucun lieu n'a franchi le seuil de récurrence)."""
        return self.dir / "location_refs"

    @property
    def segments(self) -> Path:
        return self.dir / "segments"

    @property
    def out(self) -> Path:
        return self.dir / "out.mp4"

    @property
    def out_sample(self) -> Path:
        """Sortie du rendu `--sample` (SPEC §5.1) — jamais `out.mp4`.

        Le mode `--sample` produit un montage disjoint (3 fenêtres de 30 s) dont
        l'audio ne correspond qu'à un fragment continu, jamais aux trois
        fenêtres réellement montées. C'est un artefact de jugement, pas un
        livrable : il ne doit jamais pouvoir écraser la vidéo complète.
        """
        return self.dir / "out.sample.mp4"


@dataclass(frozen=True)
class Options:
    """Options de run, passées telles quelles à chaque étape."""

    force: bool = False
    draft: bool = False
    sample: bool = False


# Une ligne à elle seule, écrite à la main dans le script importé, avant le
# paragraphe où commence un nouveau palier :
#
#     NIVEAU: Komono / Chūgen
#
# `render.py` insère un carton statique portant ce texte juste avant la scène
# qui commence à cet endroit. Majuscule volontaire : improbable qu'une phrase
# de narration commence une ligne par "NIVEAU:", donc pas d'ambiguïté à lever.
_LEVEL_LINE = re.compile(r"^NIVEAU\s*:\s*(.+?)\s*$")


def clean_source(text: str) -> tuple[str, int, list[tuple[int, str]]]:
    """Ôte les horodatages et les marqueurs `NIVEAU:` d'un script importé.

    Retourne (texte nettoyé, nb horodatages ôtés, [(index de mot, libellé)]) —
    l'index de mot pointe le premier mot du palier dans le texte nettoyé, dans
    le même référentiel que `Scene.word_start` (SPEC : les deux comptent sur
    `normalize(texte).split()`), pour que `render.py` puisse retrouver la
    scène en face sans dépendre d'un LLM ni d'un matching approximatif.

    Les horodatages sont ôtés d'abord : un marqueur `NIVEAU:` sur une ligne
    qui en portait un doit rester détecté après leur suppression.

    Appliqué **à l'import**, pas à la lecture : `script.txt` dans le projet est
    toujours de la prose (plus de marqueurs), et `normalize()` reste la seule
    définition de la normalisation.
    """
    text, stripped = re.subn(r"\[\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\]\s*", "", text)

    kept_lines: list[str] = []
    levels: list[tuple[int, str]] = []
    word_count = 0
    for line in text.split("\n"):
        match = _LEVEL_LINE.match(line.strip())
        if match:
            levels.append((word_count, match.group(1)))
            continue
        kept_lines.append(line)
        word_count += len(line.split())

    return "\n".join(kept_lines), stripped, levels


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
