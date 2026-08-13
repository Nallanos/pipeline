"""Détection et scoring des frontières (SPEC §4.3.a).

Fonction pure : `Alignment` → candidats. Aucune E/S, donc directement testable
et rejouable sans rien relancer.

Principe : les coupes tombent sur des ruptures réelles ; la durée est une
contrainte, pas le critère. On change d'image parce qu'une idée se termine, une
phrase se ferme, ou la voix marque une pause.

Une frontière d'indice `i` signifie : la coupe a lieu **avant** le mot `i`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .alignment import Alignment

PAUSE_LONG_MS = 400
PAUSE_SHORT_MS = 250

SCORE_PAUSE_LONG = 1.0
SCORE_SENTENCE_END = 0.9
SCORE_PAUSE_SHORT = 0.6
SCORE_CONNECTIVE = 0.5
SCORE_CLAUSE = 0.4
SCORE_COMMA = 0.3

SENTENCE_END = (".", "!", "?", "…")
CLAUSE_END = (":", ";", "—", "–")

# Les connecteurs logiques dépendent de la langue et vivent dans
# config/languages.yaml. Un `detect()` appelé sans eux perd un des cinq signaux
# du barème — silencieusement, puisque le score reste calculable.


@dataclass(frozen=True)
class Candidate:
    word_index: int
    score: float
    reasons: list[str]
    cut_ms: int


def _strip_closing(word: str) -> str:
    """Ôte les guillemets et parenthèses fermantes pour voir la ponctuation réelle."""
    return word.rstrip("»\"')]…").rstrip() or word


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def pause_thresholds(alignment: Alignment) -> tuple[int, int]:
    """Seuils de pause, calibrés sur la distribution réellement observée.

    Les seuils absolus du SPEC (400 / 250 ms) supposent une mesure fine des
    silences. En pratique la granularité des timestamps Whisper gonfle tous les
    écarts : sur un TTS local, une virgule mesure 380–640 ms là où une fin de
    phrase mesure 660–880. Avec les seuils absolus, les deux scorent 1.0, la
    hiérarchie du §4.3.a s'effondre et la sélection dégénère en « le plus
    proche de la cible ».

    On prend donc le maximum entre le seuil du SPEC et un percentile de la
    distribution. Sur un TTS expressif finement mesuré, les seuils absolus
    l'emportent et le comportement est exactement celui du SPEC ; sur une mesure
    grossière, les percentiles restaurent la discrimination.
    """
    gaps = [g for g in alignment.gaps_ms if g > 0]
    return (
        max(PAUSE_LONG_MS, _percentile(gaps, 0.80)),
        max(PAUSE_SHORT_MS, _percentile(gaps, 0.55)),
    )


def detect(
    alignment: Alignment,
    connectives: frozenset[str] = frozenset(),
) -> list[Candidate]:
    """Calcule le score cumulatif de chaque position inter-mots.

    Les signaux se cumulent et sont plafonnés à 1.0 : une pause longue sur une
    fin de phrase est la meilleure coupe possible.
    """
    words = alignment.words
    starts, ends = alignment.starts_ms(), alignment.ends_ms()
    pause_long, pause_short = pause_thresholds(alignment)
    candidates: list[Candidate] = []

    for i in range(1, len(words)):
        score = 0.0
        reasons: list[str] = []

        gap = alignment.gaps_ms[i]
        if gap >= pause_long:
            score += SCORE_PAUSE_LONG
            reasons.append(f"pause_{gap}ms")
        elif gap >= pause_short:
            score += SCORE_PAUSE_SHORT
            reasons.append(f"pause_{gap}ms")

        previous = _strip_closing(words[i - 1])
        if previous.endswith(SENTENCE_END):
            score += SCORE_SENTENCE_END
            reasons.append("sentence_end")
        elif previous.endswith(CLAUSE_END):
            score += SCORE_CLAUSE
            reasons.append("clause_end")
        elif previous.endswith(","):
            score += SCORE_COMMA
            reasons.append("comma")

        head = words[i].lower().strip("«\"'([").rstrip(",")
        if head in connectives:
            score += SCORE_CONNECTIVE
            reasons.append(f"connective:{head}")

        if not reasons:
            continue

        # La coupe tombe au milieu du silence : l'image change pendant que la
        # voix respire, pas sur l'attaque du mot suivant.
        cut_ms = ends[i - 1] + gap // 2 if gap else starts[i]

        candidates.append(
            Candidate(
                word_index=i,
                score=min(score, 1.0),
                reasons=reasons,
                cut_ms=cut_ms,
            )
        )

    return candidates


def score_histogram(candidates: list[Candidate]) -> str:
    """Diagnostic du §4.3 : la hiérarchie des scores tient-elle encore ?

    Si la quasi-totalité des candidats est à 1.0, le barème ne discrimine plus
    rien et la sélection dégénère en « le plus proche de la cible ». C'est le
    signal qu'il faut recalibrer les seuils de pause ou activer le §4.3.c.
    """
    if not candidates:
        return "aucun candidat"
    buckets = {"1.0": 0, "0.7–0.9": 0, "0.4–0.6": 0, "<0.4": 0}
    for c in candidates:
        if c.score >= 1.0:
            buckets["1.0"] += 1
        elif c.score >= 0.7:
            buckets["0.7–0.9"] += 1
        elif c.score >= 0.4:
            buckets["0.4–0.6"] += 1
        else:
            buckets["<0.4"] += 1
    total = len(candidates)
    parts = [f"{k}: {v} ({100 * v // total}%)" for k, v in buckets.items() if v]
    return f"{total} candidats — " + ", ".join(parts)
