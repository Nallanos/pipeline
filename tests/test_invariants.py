"""Les deux tests non négociables (SPEC §7).

Ce sont les deux seuls endroits où une erreur est **silencieuse** : rien ne
plante, et le symptôme apparaît des semaines plus tard sous forme d'images
décalées d'une demi-phrase. Ne pas ajouter d'autres tests ici (SPEC §8).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline.alignment import (
    PONCTUATION_MAX_MS,
    Alignment,
    WhisperWord,
    check_invariant_a,
    check_plausible,
    reconcile,
)
from pipeline.ffmpeg import title_card_command
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


def test_ponctuation_ne_capte_pas_le_temps_des_voisins() -> None:
    """Cas réel (19/09) : un `»` portait 4,8 s, ses 28 voisins 40 ms chacun.

    Les tokens de ponctuation n'existent dans aucune transcription — Whisper
    ne les produit jamais — donc ils ne peuvent pas être ancrés. Quand l'un
    d'eux tombe sur un long segment, il absorbe le temps de tout le bloc et
    `check_plausible` refuse l'alignement, à juste titre.
    """
    script = "il pose la montre « votre moyenne elle est sur combien » silence il repart"
    mots = script.split()
    # Whisper ne transcrit ni les guillemets ni « moyenne », et place un long
    # silence là où tombe le guillemet fermant.
    transcription = [
        WhisperWord(text="il", start_ms=0, end_ms=200),
        WhisperWord(text="pose", start_ms=200, end_ms=500),
        WhisperWord(text="la", start_ms=500, end_ms=650),
        WhisperWord(text="montre", start_ms=650, end_ms=1000),
        WhisperWord(text="silence", start_ms=6000, end_ms=6500),
        WhisperWord(text="il", start_ms=6500, end_ms=6700),
        WhisperWord(text="repart", start_ms=6700, end_ms=7000),
    ]
    alignment = reconcile(mots, transcription, audio_ms=7000)

    check_invariant_a(alignment, script)
    check_plausible(alignment)  # ne doit pas lever

    for mot, duree in zip(alignment.words, alignment.durations_ms, strict=True):
        if not any(c.isalnum() for c in mot):
            assert duree <= PONCTUATION_MAX_MS, (
                f"le token {mot!r} capte {duree} ms, il devrait être borné"
            )


def test_carton_supporte_les_apostrophes() -> None:
    """Cas réel (19/09) : « Niveau six : tireur d'élite » cassait le rendu.

    Chaque caractère du carton est un `text='X'` séparé dans une chaîne de
    drawtext, et ffmpeg n'autorise AUCUN échappement à l'intérieur de quotes
    simples : `\'`, `'\''` et le texte sans quotes échouent tous. La commande
    ne doit donc plus contenir d'apostrophe droite dans un `text=`.
    """
    commande = title_card_command("Niveau six : tireur d'élite", Path("x.mp4"), 35)
    filtre = next(a for a in commande if "drawtext" in a)

    textes = re.findall(r":text='([^']*)'", filtre)
    assert textes, "aucun texte trouvé dans le filtre"
    assert "'" not in "".join(textes), "une apostrophe droite subsiste dans le filtre"
    assert "\u2019" in filtre, "l'apostrophe doit être remplacée par sa forme typographique"


def test_remap_for_render_sans_troncature_ni_carton() -> None:
    """Cas simple : une seule scène, aucun carton — juste la division par SPEED."""
    from pipeline.subtitles import Ligne, remap_for_render

    lignes = [Ligne(1000, 2000, "bonjour")]
    breakpoints = [(0, 5000, 0)]
    out = remap_for_render(lignes, breakpoints, speed=1.25)
    assert out == [Ligne(800, 1600, "bonjour")]


def test_remap_for_render_avec_troncature() -> None:
    """L'intro est coupée (vidéo commence au 1er palier) : une ligne avant la
    coupure disparaît, une ligne après est recalée sur le zéro du fichier."""
    from pipeline.subtitles import Ligne, remap_for_render

    lignes = [
        Ligne(0, 500, "intro coupée"),
        Ligne(10_000, 11_000, "premier mot du palier 1"),
    ]
    # Seule la 2e scène (à partir de 10000ms absolu) est retenue, et elle
    # atterrit au tout début du fichier rendu (position 0).
    breakpoints = [(10_000, 20_000, 0)]
    out = remap_for_render(lignes, breakpoints, speed=1.0)
    assert len(out) == 1
    assert out[0].texte == "premier mot du palier 1"
    assert out[0].start_ms == 0
    assert out[0].end_ms == 1000


def test_remap_for_render_avec_carton_insere() -> None:
    """Un carton parlé de 1500ms est inséré avant la 2e scène : tout ce qui
    la suit doit être décalé d'autant dans le fichier rendu, alors que rien
    ne bouge à l'échelle ABSOLUE du script (alignment.json)."""
    from pipeline.subtitles import Ligne, remap_for_render

    lignes = [
        Ligne(0, 1000, "scène 1"),
        Ligne(5000, 6000, "scène 2, juste après le carton"),
    ]
    # scène 1 : absolu [0, 5000) -> position vidéo 0
    # scène 2 : absolu [5000, 10000) -> position vidéo 5000 + 1500 (carton)
    breakpoints = [(0, 5000, 0), (5000, 10_000, 6500)]
    out = remap_for_render(lignes, breakpoints, speed=1.0)
    assert out[0] == Ligne(0, 1000, "scène 1")
    assert out[1] == Ligne(6500, 7500, "scène 2, juste après le carton")


def test_remap_for_render_ligne_vide() -> None:
    """Aucune scène retenue -> liste vide, pas d'exception."""
    from pipeline.subtitles import Ligne, remap_for_render

    assert remap_for_render([Ligne(0, 100, "x")], [], speed=1.0) == []
