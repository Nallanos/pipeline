"""Étape 3 — Segmentation. `alignment.json` → `boundaries.json` → `scenes.json`.

Sélection sous contrainte (SPEC §4.3.b) : on accumule la durée depuis la
dernière coupe et, à l'approche de la cible, on prend dans la fenêtre le
candidat de meilleur score. Si rien n'atteint le seuil, on étend la recherche ;
au-delà du maximum, on force la coupe.

Rationale du forçage : rester 9 secondes sur une image est plus dommageable
qu'une coupe imparfaite.
"""

from __future__ import annotations

import json

from .. import scenes as scenes_mod
from ..alignment import Alignment
from ..boundaries import Candidate, detect, pause_thresholds, score_histogram
from ..project import Options, Project, write_atomic
from ..scenes import Scene

TARGET_MS = 4500
MIN_MS = 3000
MAX_MS = 6000
WINDOW_MS = 1500
MIN_SCORE = 0.3


def _pick(
    candidates: list[Candidate],
    scene_start_ms: int,
    limit_ms: int,
) -> tuple[Candidate | None, bool]:
    """Choisit la coupe suivante. Retourne (candidat, forcé)."""
    target = scene_start_ms + TARGET_MS

    def usable(lo: int, hi: int) -> list[Candidate]:
        return [
            c
            for c in candidates
            if lo <= c.cut_ms <= min(hi, limit_ms) and c.score >= MIN_SCORE
        ]

    # 1. Dans la fenêtre de tolérance autour de la cible.
    window = usable(target - WINDOW_MS, target + WINDOW_MS)
    # 2. Sinon, étendre jusqu'au maximum autorisé.
    if not window:
        window = usable(scene_start_ms + MIN_MS, scene_start_ms + MAX_MS)

    if window:
        # Meilleur score ; à score égal, le plus proche de la cible.
        return max(window, key=lambda c: (c.score, -abs(c.cut_ms - target))), False

    # 3. Rien d'acceptable : coupe forcée à la meilleure position disponible.
    forced_pool = [c for c in candidates if scene_start_ms < c.cut_ms <= limit_ms]
    if not forced_pool:
        return None, True
    return min(forced_pool, key=lambda c: abs(c.cut_ms - target)), True


def build_scenes(alignment: Alignment, project_id: str) -> list[Scene]:
    """Le parcours glouton du §4.3.b. Fonction pure, testable sans E/S."""
    candidates = detect(alignment)
    words = alignment.words
    built: list[Scene] = []
    word_cursor = 0
    time_cursor = 0
    remaining = list(candidates)

    while word_cursor < len(words):
        # Candidats encore devant nous, en indices de mots.
        remaining = [c for c in remaining if c.word_index > word_cursor]

        # Fin de l'audio : ce qui reste tient dans une scène. Couper ici
        # produirait un croupion — la cible tombe au-delà de la fin du fichier,
        # donc la sélection forcerait la coupe sur le dernier candidat venu,
        # quelle que soit sa position. La dernière scène absorbe la queue du
        # script et le silence final, et la vidéo dure exactement l'audio.
        if alignment.audio_ms - time_cursor <= MAX_MS:
            picked, forced = None, False
        else:
            picked, forced = _pick(remaining, time_cursor, alignment.audio_ms - MIN_MS)

        if picked is None:
            word_end, cut_ms, score, reasons = len(words), alignment.audio_ms, 0.0, []
        else:
            word_end, cut_ms, score, reasons = (
                picked.word_index,
                picked.cut_ms,
                picked.score,
                picked.reasons,
            )

        index = len(built)
        built.append(
            Scene(
                index=index,
                text=" ".join(words[word_cursor:word_end]),
                word_start=word_cursor,
                word_end=word_end,
                start_ms=time_cursor,
                end_ms=cut_ms,
                boundary_score=round(score, 2),
                seed=scenes_mod.seed_for(project_id, index),
                forced=forced,
                reasons=reasons,
            )
        )
        word_cursor, time_cursor = word_end, cut_ms

    return built


def run(project: Project, opts: Options) -> None:
    alignment = Alignment.load(project.alignment)
    script_normalized = project.script_normalized.read_text(encoding="utf-8")

    candidates = detect(alignment)
    write_atomic(
        project.boundaries,
        json.dumps(
            {
                "candidates": [
                    {
                        "word_index": c.word_index,
                        "score": round(c.score, 2),
                        "reasons": c.reasons,
                        "cut_ms": c.cut_ms,
                    }
                    for c in candidates
                ]
            },
            ensure_ascii=False,
            indent=1,
        ),
    )

    built = build_scenes(alignment, project.id)
    scenes_mod.check_invariant_b(built, script_normalized)
    scenes_mod.save(project.scenes, built)

    pause_long, pause_short = pause_thresholds(alignment)
    print(f"  seuils de pause : {pause_short} / {pause_long} ms")
    print(f"  {score_histogram(candidates)}")
    print(f"  {scenes_mod.duration_report(built)}")
