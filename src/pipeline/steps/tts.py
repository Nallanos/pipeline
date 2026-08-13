"""Étape 1 — TTS. `script.txt` → `audio.wav` (SPEC §4).

Piper 1.3 n'expose aucun réglage de silence inter-phrases : il découpe le texte
lui-même et concatène les chunks bout à bout, ce qui donne une narration sans
respiration. On synthétise donc segment par segment et on insère les silences.

Ce n'est pas qu'une question de confort d'écoute : les silences sont le signal
de premier ordre de la segmentation (SPEC §4.3.a), et les durées ci-dessous
sont calibrées sur le barème des scores. Seules les fins de phrase franchissent
le seuil des 400 ms — les virgules et les points-virgules restent en dessous,
pour que leur score vienne de la ponctuation seule et ne soit pas compté deux
fois.
"""

from __future__ import annotations

import re
import wave

import numpy as np
from piper import PiperVoice

from ..project import MODELS_DIR, Options, Project, normalize, write_atomic

VOICE = MODELS_DIR / "piper" / "fr_FR-siwis-medium.onnx"

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


def run(project: Project, opts: Options) -> None:
    script = project.script.read_text(encoding="utf-8")
    text = normalize(script)
    # Persisté avant toute synthèse : c'est la source de vérité du pipeline
    # entier, y compris des deux invariants (SPEC §7).
    write_atomic(project.script_normalized, text)

    voice = PiperVoice.load(str(VOICE))
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

    print(f"  audio : {len(samples) / rate / 60:.1f} min, {rate} Hz mono")
