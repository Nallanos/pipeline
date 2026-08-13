"""Les deux tests non négociables (SPEC §7).

Ce sont les deux seuls endroits où une erreur est **silencieuse** : rien ne
plante, et le symptôme apparaît des semaines plus tard sous forme d'images
décalées d'une demi-phrase. Ne pas ajouter d'autres tests ici (SPEC §8).
"""

from __future__ import annotations

import pytest

from pipeline.alignment import Alignment, WhisperWord, check_invariant_a, reconcile
from pipeline.project import normalize

SCRIPT = (
    "Le projet commence par une observation simple. En 2024, l'équipe SNCF "
    "a mesuré 3 200 trajets ; mais personne n'a publié les chiffres."
)


def _whisper_like(text: str, ms_per_word: int = 300, gap_ms: int = 50) -> list[WhisperWord]:
    """Simule une sortie Whisper : un mot toutes les `ms_per_word`, séparés par `gap_ms`."""
    words = []
    t = 0
    for token in text.split():
        words.append(WhisperWord(text=token, start_ms=t, end_ms=t + ms_per_word))
        t += ms_per_word + gap_ms
    return words


# ---------------------------------------------------------------- invariant A


def test_invariant_a_alignement_parfait() -> None:
    script = normalize(SCRIPT)
    alignment = reconcile(script.split(), _whisper_like(script), audio_ms=60_000)
    check_invariant_a(alignment, script)


def test_invariant_a_whisper_reecrit_chiffres_et_acronymes() -> None:
    """Le cas réel : Whisper réécrit les nombres et les sigles.

    Le script reste la vérité, Whisper ne fournit que les temps.
    """
    script = normalize(SCRIPT)
    transcription = (
        "Le projet commence par une observation simple. En deux mille vingt-quatre, "
        "l'équipe S.N.C.F. a mesuré trois mille deux cents trajets, "
        "mais personne n'a publié les chiffres."
    )
    alignment = reconcile(script.split(), _whisper_like(transcription), audio_ms=60_000)

    check_invariant_a(alignment, script)
    assert alignment.words == script.split()


def test_invariant_a_whisper_saute_des_mots() -> None:
    script = normalize(SCRIPT)
    partiel = " ".join(w for i, w in enumerate(script.split()) if i % 5 != 0)
    alignment = reconcile(script.split(), _whisper_like(partiel), audio_ms=60_000)
    check_invariant_a(alignment, script)


def test_invariant_a_detecte_un_decalage() -> None:
    script = normalize(SCRIPT)
    alignment = reconcile(script.split(), _whisper_like(script), audio_ms=60_000)
    decale = Alignment(
        words=alignment.words[1:],
        durations_ms=alignment.durations_ms[1:],
        gaps_ms=alignment.gaps_ms[1:],
        total_ms=alignment.total_ms,
        audio_ms=alignment.audio_ms,
    )
    with pytest.raises(AssertionError, match="invariant A"):
        check_invariant_a(decale, script)


# ------------------------------------------------- propriétés du modèle temporel


@pytest.mark.parametrize(
    "transcription",
    [
        SCRIPT,
        "Le projet commence par une observation simple.",  # Whisper s'arrête tôt
        "En deux mille vingt-quatre l'équipe a mesuré trois mille deux cents trajets",
        "totalement différent et sans aucun rapport avec le script d'origine",
    ],
)
def test_temps_monotones_et_durees_positives(transcription: str) -> None:
    """Quelle que soit la divergence, la suite temporelle reste exploitable.

    Une durée nulle produirait une scène jamais affichée ; un temps qui recule
    produirait un segment vidéo de durée négative au rendu.
    """
    script = normalize(SCRIPT)
    alignment = reconcile(script.split(), _whisper_like(transcription), audio_ms=60_000)

    starts, ends = alignment.starts_ms(), alignment.ends_ms()
    assert all(d > 0 for d in alignment.durations_ms)
    assert all(g >= 0 for g in alignment.gaps_ms)
    assert all(starts[i] >= ends[i - 1] for i in range(1, len(starts)))
    assert ends[-1] == alignment.total_ms


def test_somme_cumulee_coherente_avec_total() -> None:
    """`total_ms` doit être exactement reconstructible depuis les durées et les gaps."""
    script = normalize(SCRIPT)
    alignment = reconcile(script.split(), _whisper_like(script), audio_ms=60_000)
    assert sum(alignment.durations_ms) + sum(alignment.gaps_ms) == alignment.total_ms


def test_decalage_de_timings_invisible_pour_l_invariant_a() -> None:
    """Le cas que l'invariant A ne peut pas voir : bons mots, mauvais temps.

    C'est la panne silencieuse du SPEC §7. Sur un script contenant une phrase
    répétée, la réconciliation peut apparier la mauvaise occurrence : tous les
    mots restent corrects, donc l'invariant A passe, mais une partie de
    l'alignement est étirée et l'autre comprimée.
    """
    from pipeline.alignment import check_plausible

    repete = normalize(" ".join(["La question revient toujours au même point de départ."] * 12))
    words = repete.split()

    # Whisper n'a « entendu » que la seconde moitié : la première moitié du
    # script se retrouve écrasée sur un intervalle minuscule.
    transcrit = _whisper_like(" ".join(words[len(words) // 2 :]))
    alignment = reconcile(words, transcrit, audio_ms=transcrit[-1].end_ms)

    check_invariant_a(alignment, repete)  # passe : les mots sont intacts
    with pytest.raises(AssertionError, match="implausible"):
        check_plausible(alignment)


def test_alignement_regulier_juge_plausible() -> None:
    from pipeline.alignment import check_plausible

    script = normalize(LONG_SCRIPT)
    check_plausible(_realistic_alignment(script))


# ---------------------------------------------------------------- invariant B

LONG_SCRIPT = normalize(
    " ".join(
        [
            "Le projet commence par une observation simple.",
            "Pendant des années, on a cru que la difficulté était technique.",
            "Mais elle ne l'était pas ; elle était structurelle, et nul ne voulait l'admettre.",
            "Alors une question se pose : que se passe-t-il quand on retire l'hypothèse ?",
            "La réponse surprend. Tout tient encore debout, mais pour des raisons différentes.",
            "Ensuite les choses s'accélèrent. Trois équipes, deux continents, un seul objectif.",
            "Personne n'y croyait vraiment. Aujourd'hui le résultat est visible partout.",
        ]
        * 6
    )
)


def _realistic_alignment(script: str) -> Alignment:
    """Alignement plausible : mots de longueur variable, silences sur la ponctuation."""
    words = script.split()
    starts, ends = [], []
    t = 0
    for word in words:
        pause = 800 if word.endswith((".", "?", "!")) else 400 if word.endswith(",;:") else 60
        starts.append(t)
        duration = 90 + 45 * len(word)
        ends.append(t + duration)
        t = ends[-1] + pause
    return reconcile(
        words,
        [WhisperWord(w, s, e) for w, s, e in zip(words, starts, ends, strict=True)],
        audio_ms=t,
    )


def test_invariant_b_concatenation_reproduit_le_script() -> None:
    from pipeline.scenes import check_invariant_b
    from pipeline.steps.segment import build_scenes

    scenes = build_scenes(_realistic_alignment(LONG_SCRIPT), "demo")
    check_invariant_b(scenes, LONG_SCRIPT)


def test_invariant_b_scenes_couvrent_les_mots_sans_trou_ni_recouvrement() -> None:
    """L'invariant B tient par construction : les scènes sont des tranches d'indices."""
    from pipeline.steps.segment import build_scenes

    alignment = _realistic_alignment(LONG_SCRIPT)
    scenes = build_scenes(alignment, "demo")

    assert scenes[0].word_start == 0
    assert scenes[-1].word_end == len(alignment.words)
    for previous, following in zip(scenes[:-1], scenes[1:], strict=True):
        assert previous.word_end == following.word_start
        assert previous.end_ms == following.start_ms


def test_duree_des_scenes_bornee() -> None:
    """Le rythme ne doit jamais s'effondrer, même sur une coupe forcée."""
    from pipeline.steps.segment import MAX_MS, build_scenes

    alignment = _realistic_alignment(LONG_SCRIPT)
    scenes = build_scenes(alignment, "demo")

    assert scenes[0].start_ms == 0
    assert scenes[-1].end_ms == alignment.audio_ms
    # La dernière scène absorbe la queue, les autres respectent le plafond.
    assert all(s.duration_ms <= MAX_MS for s in scenes[:-1])
    assert all(s.duration_ms > 0 for s in scenes)


def test_seed_deterministe_entre_processus() -> None:
    """Un seed non reproductible casserait la reprise et la comparaison d'itérations."""
    from pipeline.scenes import seed_for

    assert seed_for("demo", 0) == seed_for("demo", 0)
    assert seed_for("demo", 0) != seed_for("demo", 1)
    assert seed_for("demo", 0) != seed_for("autre", 0)
