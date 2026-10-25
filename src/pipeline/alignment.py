"""Modèle temporel — le cœur du système (SPEC §3.1, §7).

Une erreur ici ne lève aucune exception et ne plante rien. Elle se manifeste
trois semaines plus tard par des images en retard d'une demi-phrase, et le bug
sera cherché dans le rendu. Relire ligne à ligne.

Deux règles portent tout le fichier :

1. **Des durées, jamais des timestamps absolus.** Les temps de début sont
   dérivés par somme cumulée à la lecture, ce qui évite toute dérive d'arrondi
   sur 24 minutes.
2. **Le script est la vérité, Whisper ne fournit que les temps.** Whisper
   réécrit les chiffres, les acronymes et les noms propres ; on garde le mot du
   script et le timing de Whisper.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .project import write_atomic

# Plancher de durée d'un mot, appliqué par la passe de réparation monotone.
# Un mot de durée nulle ou négative produirait une scène de durée nulle plus
# loin dans le pipeline, donc une image jamais affichée.
PONCTUATION_MAX_MS = 400  # au-delà, un token non prononcé a capté le temps d'un voisin
MIN_WORD_MS = 40


@dataclass(frozen=True)
class Alignment:
    """`alignment.json` (SPEC §3.1).

    `gaps_ms[i]` est le silence **avant** le mot `i` — donc `gaps_ms[0]` est le
    silence de tête. C'est le signal de premier ordre de la segmentation : il
    reflète les pauses réellement produites par le TTS.
    """

    words: list[str]
    durations_ms: list[int]
    gaps_ms: list[int]
    total_ms: int
    audio_ms: int

    def starts_ms(self) -> list[int]:
        """Temps de début de chaque mot, par somme cumulée."""
        starts: list[int] = []
        t = 0
        for gap, dur in zip(self.gaps_ms, self.durations_ms, strict=True):
            t += gap
            starts.append(t)
            t += dur
        return starts

    def ends_ms(self) -> list[int]:
        return [s + d for s, d in zip(self.starts_ms(), self.durations_ms, strict=True)]

    def text(self) -> str:
        return " ".join(self.words)

    def save(self, path: Path) -> None:
        write_atomic(
            path,
            json.dumps(
                {
                    "words": self.words,
                    "durations_ms": self.durations_ms,
                    "gaps_ms": self.gaps_ms,
                    "total_ms": self.total_ms,
                    "audio_ms": self.audio_ms,
                },
                ensure_ascii=False,
                indent=1,
            ),
        )

    @staticmethod
    def load(path: Path) -> Alignment:
        d = json.loads(path.read_text(encoding="utf-8"))
        return Alignment(
            words=d["words"],
            durations_ms=d["durations_ms"],
            gaps_ms=d["gaps_ms"],
            total_ms=d["total_ms"],
            audio_ms=d.get("audio_ms", d["total_ms"]),
        )


def fold(word: str) -> str:
    """Forme repliée servant **uniquement** à la comparaison.

    Accents, casse et ponctuation ôtés : « L'Été, » et « lete » se comparent
    égaux. Le mot conservé reste toujours celui du script — cette forme ne
    sort jamais de la réconciliation.
    """
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in decomposed if c.isalnum() and not unicodedata.combining(c))


@dataclass(frozen=True)
class WhisperWord:
    text: str
    start_ms: int
    end_ms: int


def reconcile(
    script_words: list[str],
    whisper_words: list[WhisperWord],
    audio_ms: int,
) -> Alignment:
    """Aligne les timings de Whisper sur les mots du script.

    Retourne un `Alignment` dont les mots sont **exactement** `script_words` —
    l'invariant A est donc garanti par construction ici, et vérifié en plus par
    l'étape appelante.
    """
    if not script_words:
        raise ValueError("script vide")
    if not whisper_words:
        raise ValueError("Whisper n'a produit aucun mot — audio muet ou illisible ?")

    folded_script = [fold(w) for w in script_words]
    folded_whisper = [fold(w.text) for w in whisper_words]

    # Intervalles bruts, un par mot du script. Remplis bloc par bloc.
    spans: list[list[int]] = [[0, 0] for _ in script_words]

    matcher = SequenceMatcher(None, folded_script, folded_whisper, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if i1 == i2:
            # Whisper a inventé des mots absents du script : on les ignore.
            # Leur temps est absorbé par les blocs voisins.
            continue

        if tag == "equal":
            for k in range(i2 - i1):
                w = whisper_words[j1 + k]
                spans[i1 + k] = [w.start_ms, w.end_ms]
            continue

        # 'replace' ou 'delete' : les mots du script du bloc n'ont pas de
        # correspondant fiable. On leur répartit l'intervalle couvert par le
        # bloc Whisper correspondant, au prorata de leur longueur.
        if j1 < j2:
            t0, t1 = whisper_words[j1].start_ms, whisper_words[j2 - 1].end_ms
        else:
            # 'delete' pur : rien côté Whisper. L'intervalle est le trou entre
            # le mot précédent et le suivant.
            t0 = whisper_words[j1 - 1].end_ms if j1 > 0 else 0
            t1 = whisper_words[j2].start_ms if j2 < len(whisper_words) else t0

        weights = [len(folded_script[i]) + 1 for i in range(i1, i2)]
        total_weight = sum(weights)
        cursor = t0
        for k, weight in enumerate(weights):
            share = (t1 - t0) * weight // total_weight
            spans[i1 + k] = [cursor, cursor + share]
            cursor += share

    # Passe de réparation : rendre la suite strictement monotone et les durées
    # positives. Les cas dégénérés (bloc 'delete' sur un intervalle nul) sont
    # rares mais produiraient sinon des scènes de durée nulle.
    prev_end = 0
    for span in spans:
        span[0] = max(span[0], prev_end)
        span[1] = max(span[1], span[0] + MIN_WORD_MS)
        prev_end = span[1]

    _rendre_le_temps_de_la_ponctuation(spans, script_words)

    durations = [end - start for start, end in spans]
    gaps = [spans[0][0]] + [spans[i][0] - spans[i - 1][1] for i in range(1, len(spans))]
    total = spans[-1][1]

    return Alignment(
        words=list(script_words),
        durations_ms=durations,
        gaps_ms=gaps,
        total_ms=total,
        audio_ms=max(audio_ms, total),
    )


def _rendre_le_temps_de_la_ponctuation(
    spans: list[list[int]], script_words: list[str]
) -> None:
    """Rend aux mots voisins le temps capté par un token non prononcé.

    `«`, `»`, `?`, `:` sont des mots pour le script mais n'existent dans aucune
    transcription : Whisper ne les produit jamais. Ils ne peuvent donc pas être
    ancrés correctement, et il arrive qu'un d'eux tombe sur un long segment et
    absorbe plusieurs secondes pendant que ses voisins, faute d'ancrage, sont
    tassés à `MIN_WORD_MS`.

    Vu en pratique (19/09) sur un script dialogué : un `»` portait 4 840 ms
    quand les 28 mots qui le précédaient avaient 40 ms chacun — soit 97
    caractères/seconde, quatre fois le débit réel, ce que `check_plausible`
    refusait à juste titre.

    On borne donc la durée d'un token non prononcé et on redonne l'excédent
    aux mots collés au plancher qui le précèdent, au prorata de leur longueur.
    Les bornes du bloc ne bougent pas : rien n'est décalé en dehors.
    """
    for i, mot in enumerate(script_words):
        if any(c.isalnum() for c in mot):
            continue  # un vrai mot, il a le droit de durer
        duree = spans[i][1] - spans[i][0]
        if duree <= PONCTUATION_MAX_MS:
            continue

        # Les mots juste avant qui sont au plancher : ce sont eux qu'on a
        # privés de leur temps.
        debut = i
        while debut > 0 and spans[debut - 1][1] - spans[debut - 1][0] <= MIN_WORD_MS:
            debut -= 1
        if debut == i:
            continue

        excedent = duree - PONCTUATION_MAX_MS
        poids = [len(script_words[k]) + 1 for k in range(debut, i)]
        total = sum(poids)
        curseur = spans[debut][0]
        for k, poids_k in enumerate(poids):
            part = MIN_WORD_MS + excedent * poids_k // total
            spans[debut + k] = [curseur, curseur + part]
            curseur += part
        spans[i] = [curseur, curseur + PONCTUATION_MAX_MS]


def check_plausible(alignment: Alignment, window: int = 25, tolerance: float = 4.0) -> None:
    """Garde-fou contre les décalages que l'invariant A ne peut pas voir.

    L'invariant A vérifie les **mots**, jamais les **temps**. Une réconciliation
    qui apparie la première occurrence d'une phrase avec la troisième produit un
    alignement dont tous les mots sont corrects et dont tous les temps sont
    faux : il passe l'invariant A, ne lève rien, et se manifeste trois semaines
    plus tard par des images en retard d'une demi-phrase (SPEC §7).

    `difflib.SequenceMatcher` choisit ses blocs par longueur, sans notion de
    position : tout passage répété mot pour mot peut donc être apparié à la
    mauvaise occurrence, ce qui étire un bout de l'alignement et en comprime un
    autre. On mesure ici le débit de parole sur une fenêtre glissante : il varie
    peu dans une narration réelle, donc un écart d'un facteur `tolerance` par
    rapport à la médiane signale un appariement erroné.
    """
    words = alignment.words
    if len(words) < window * 2:
        return

    starts, ends = alignment.starts_ms(), alignment.ends_ms()
    rates: list[tuple[float, int]] = []
    for i in range(0, len(words) - window):
        chars = sum(len(w) for w in words[i : i + window])
        span = ends[i + window - 1] - starts[i]
        if span > 0:
            rates.append((1000.0 * chars / span, i))

    if not rates:
        return

    median = sorted(r for r, _ in rates)[len(rates) // 2]
    for rate, i in rates:
        if rate > median * tolerance or rate < median / tolerance:
            excerpt = " ".join(words[i : i + 6])
            raise AssertionError(
                f"alignement implausible au mot {i} : {rate:.0f} caractères/s "
                f"contre une médiane de {median:.0f} — « {excerpt} … ». "
                "La réconciliation Whisper a probablement apparié une répétition "
                "du script à la mauvaise occurrence."
            )


def check_invariant_a(alignment: Alignment, script_normalized: str) -> None:
    """Invariant A (SPEC §7.1) — échec dur.

    Un décalage d'alignement est silencieux ; c'est le seul endroit où on peut
    encore l'attraper.
    """
    joined = alignment.text()
    if joined != script_normalized:
        for i, (a, b) in enumerate(zip(joined.split(), script_normalized.split(), strict=False)):
            if a != b:
                raise AssertionError(
                    f"invariant A violé au mot {i} : alignement={a!r} script={b!r}"
                )
        raise AssertionError(
            f"invariant A violé : {len(joined.split())} mots alignés "
            f"contre {len(script_normalized.split())} dans le script"
        )
