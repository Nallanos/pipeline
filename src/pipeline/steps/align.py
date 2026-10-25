"""Étape 2 — Alignement. `audio.wav` + `script.txt` → `alignment.json` (SPEC §4).

C'est du forced alignment, pas de la transcription : le texte est connu, seuls
les temps manquent. Whisper sert de source de timings, et de rien d'autre.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

from ..alignment import WhisperWord, check_invariant_a, check_plausible, reconcile
from ..language import for_project
from ..project import Options, Project, safe_wav_frames

MODEL = "small"
DRAFT_MODEL = "tiny"  # pour itérer ; jamais pour la production


def _audio_ms(path: str) -> int:
    with wave.open(path, "rb") as wav:
        rate = wav.getframerate()
    return round(safe_wav_frames(Path(path)) * 1000 / rate)


def run(project: Project, opts: Options) -> None:
    from faster_whisper import WhisperModel

    script_normalized = project.script_normalized.read_text(encoding="utf-8")
    script_words = script_normalized.split()

    model = WhisperModel(
        DRAFT_MODEL if opts.draft else MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=os.cpu_count() or 4,
    )
    segments, _info = model.transcribe(
        str(project.audio),
        language=for_project(project.dir).whisper,
        word_timestamps=True,
        # L'audio contient des silences insérés volontairement (steps/tts.py).
        # Le VAD les découperait et décalerait les temps rapportés.
        vad_filter=False,
        # Sur 24 minutes, le conditionnement sur le texte précédent finit par
        # produire des boucles de répétition. On ne veut que des timings.
        condition_on_previous_text=False,
    )

    audio_ms = _audio_ms(str(project.audio))

    # Whisper rapporte parfois des temps au-delà de la fin du fichier, en
    # particulier quand il hallucine sur le silence final. Non borné, ça donne
    # une dernière scène plus longue que l'audio et une vidéo qui déborde.
    whisper_words = [
        WhisperWord(
            text=word.word.strip(),
            start_ms=min(round(word.start * 1000), audio_ms),
            end_ms=min(round(word.end * 1000), audio_ms),
        )
        for segment in segments
        for word in (segment.words or [])
        if word.word.strip()
    ]

    alignment = reconcile(script_words, whisper_words, audio_ms)
    # Vérifiés avant écriture : un alignement faux ne doit jamais atteindre le
    # disque. L'invariant A contrôle les mots, `check_plausible` les temps.
    check_invariant_a(alignment, script_normalized)
    check_plausible(alignment)
    alignment.save(project.alignment)

    drift = len(whisper_words) - len(script_words)
    print(
        f"  {len(script_words)} mots, {alignment.total_ms / 60000:.1f} min "
        f"— Whisper en a produit {len(whisper_words)} ({drift:+d})"
    )
