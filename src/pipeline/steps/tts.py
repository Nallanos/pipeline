"""Étape 1 — TTS. `script.txt` → `audio.wav` (SPEC §4).

Deux fournisseurs, choisis par `PIPELINE_TTS_PROVIDER` (défaut `piper`) :

- `piper` : local, CPU, gratuit, toujours disponible — le filet de sécurité.
- `chatterbox` : serveur HTTP (Chatterbox TTS Server, CPU, sur la machine
  Windows où tourne aussi ComfyUI — voir `TTS-LOCAL-MODEL.md`). Bien meilleure
  qualité de voix, mais dépend d'un serveur externe.

Piper 1.3 n'expose aucun réglage de silence inter-phrases : il découpe le texte
lui-même et concatène les chunks bout à bout, ce qui donne une narration sans
respiration. On synthétise donc segment par segment et on insère les silences
nous-mêmes.

Ce n'est pas qu'une question de confort d'écoute : les silences sont le signal
de premier ordre de la segmentation (SPEC §4.3.a), et les durées ci-dessous
sont calibrées sur le barème des scores. Seules les fins de phrase franchissent
le seuil des 400 ms — les virgules et les points-virgules restent en dessous,
pour que leur score vienne de la ponctuation seule et ne soit pas compté deux
fois. Chatterbox gère sa propre prosodie (modèle de clonage, pas de synthèse
concaténative) : ce découpage ne s'applique qu'au provider `piper`.
"""

from __future__ import annotations

import hashlib
import os
import re
import wave

import numpy as np

from ..language import LanguageConfig, for_project
from ..project import Options, Project, normalize, write_atomic

SENTENCE_SILENCE_MS = 450  # . ? ! — franchit le seuil « pause > 400 ms » (score 1.0)
CLAUSE_SILENCE_MS = 220  # : ; — et … — sous le seuil des 250 ms, délibérément
COMMA_SILENCE_MS = 120  # ,

# Découpe en gardant le délimiteur : les groupes de capture sont conservés par
# re.split, ce qui permet de savoir quel silence insérer après chaque segment.
_SPLIT = re.compile(r"([.!?]+|[:;—…]|,)")

_SILENCE_FOR = {
    ".": SENTENCE_SILENCE_MS,
    "!": SENTENCE_SILENCE_MS,
    "?": SENTENCE_SILENCE_MS,
    ":": CLAUSE_SILENCE_MS,
    ";": CLAUSE_SILENCE_MS,
    "—": CLAUSE_SILENCE_MS,
    "…": CLAUSE_SILENCE_MS,
    ",": COMMA_SILENCE_MS,
}


def _split_with_silences(text: str) -> list[tuple[str, int]]:
    """Découpe le texte en (segment prononçable, silence en ms à ajouter après)."""
    parts = _SPLIT.split(text)
    out: list[tuple[str, int]] = []
    # re.split alterne : texte, délimiteur, texte, délimiteur, …
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        if not chunk:
            continue
        # La ponctuation est réinjectée dans le texte prononcé : elle porte la
        # prosodie (une question ne s'intonne pas comme une affirmation).
        spoken = f"{chunk}{delim}"
        silence = _SILENCE_FOR.get(delim[:1], 0) if delim else SENTENCE_SILENCE_MS
        out.append((spoken, silence))
    return out


TTS_PROVIDER = os.environ.get("PIPELINE_TTS_PROVIDER", "piper")

CHATTERBOX_URL = os.environ.get("CHATTERBOX_URL", "http://127.0.0.1:8189")
CHATTERBOX_VOICE = os.environ.get("CHATTERBOX_VOICE", "Adrian.wav")
# ~2× le temps réel mesuré en CPU (voir TTS-LOCAL-MODEL.md) : le serveur doit
# tenir toute la durée du script, pas juste une phrase — sur 24 min de cible,
# ça peut prendre 45-50 min. Une marge large plutôt qu'un timeout qui coupe un
# appel de plusieurs dizaines de minutes à la dernière minute.
CHATTERBOX_TIMEOUT_S = float(os.environ.get("CHATTERBOX_TIMEOUT_S", "7200"))


def _seed_for_project(project_id: str) -> int:
    """Même construction que `scenes.seed_for` : déterministe, pas `hash()`."""
    digest = hashlib.sha256(f"{project_id}:tts".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _run_piper(project: Project, language: LanguageConfig, text: str) -> None:
    from piper import PiperVoice

    voice = PiperVoice.load(str(language.voice_path))
    rate = voice.config.sample_rate

    audio: list[np.ndarray] = []
    for spoken, silence_ms in _split_with_silences(text):
        for chunk in voice.synthesize(spoken):
            audio.append(chunk.audio_float_array)
        if silence_ms:
            audio.append(np.zeros(rate * silence_ms // 1000, dtype=np.float32))

    samples = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    project.audio.parent.mkdir(parents=True, exist_ok=True)
    tmp = project.audio.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    tmp.replace(project.audio)

    print(
        f"  audio : {len(samples) / rate / 60:.1f} min, {rate} Hz mono "
        f"— voix {language.voice}"
    )


def _run_chatterbox(project: Project, language: LanguageConfig, text: str) -> None:
    import httpx

    # `language.whisper` est déjà un code ISO 639-1 (fr, en) — le même que le
    # serveur Chatterbox attend, testé directement (SPEC : une seule
    # définition de la langue par projet, pas un champ redondant de plus).
    payload = {
        "text": text,
        "predefined_voice_id": CHATTERBOX_VOICE,
        "language": language.whisper,
        "output_format": "wav",
        "seed": _seed_for_project(project.id),
        # Chunking géré côté serveur : un seul appel pour tout le script
        # (~20 000 caractères), pas un appel par phrase.
        "split_text": True,
    }
    try:
        response = httpx.post(
            f"{CHATTERBOX_URL}/tts", json=payload, timeout=CHATTERBOX_TIMEOUT_S
        )
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"serveur Chatterbox injoignable sur {CHATTERBOX_URL} — "
            f"vérifier qu'il tourne (voir TTS-LOCAL-MODEL.md) : {exc}"
        ) from None
    if response.status_code != 200:
        raise RuntimeError(
            f"Chatterbox a répondu {response.status_code} : {response.text[:300]}"
        )

    project.audio.parent.mkdir(parents=True, exist_ok=True)
    tmp = project.audio.with_suffix(".wav.tmp")
    tmp.write_bytes(response.content)
    tmp.replace(project.audio)

    with wave.open(str(project.audio), "rb") as wav:
        minutes = wav.getnframes() / wav.getframerate() / 60
        rate = wav.getframerate()
    print(
        f"  audio : {minutes:.1f} min, {rate} Hz — Chatterbox ({CHATTERBOX_VOICE}, "
        f"langue {language.whisper})"
    )


def run(project: Project, opts: Options) -> None:
    script = project.script.read_text(encoding="utf-8")
    text = normalize(script)
    # Persisté avant toute synthèse : c'est la source de vérité du pipeline
    # entier, y compris des deux invariants (SPEC §7).
    write_atomic(project.script_normalized, text)

    language = for_project(project.dir)

    if TTS_PROVIDER == "chatterbox":
        _run_chatterbox(project, language, text)
    else:
        _run_piper(project, language, text)
