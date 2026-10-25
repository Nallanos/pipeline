"""Sous-titres, dérivés de l'alignement mot à mot (`alignment.json`).

Aucune reconnaissance vocale ici : l'étape `align` a déjà produit la durée et
le silence qui suit chaque mot, donc les bornes exactes de chacun. Ce module
ne fait que les regrouper en lignes lisibles et les écrire dans les deux
formats attendus :

- **SRT** (`out.srt`), à côté de la vidéo — pour l'envoi à YouTube, qui gère
  alors l'affichage, l'accessibilité et l'indexation de la transcription.
- **ASS**, un fichier par segment — incrusté à l'encodage du segment, donc
  sans le moindre ré-encodage supplémentaire.

Le facteur de vitesse du montage (`ffmpeg.SPEED`) est appliqué ICI, une fois :
l'alignement est mesuré sur l'audio d'origine, alors que la vidéo est montée
accélérée. Sans cette division, les sous-titres dériveraient de 25 % — près de
quatre minutes d'écart en fin de vidéo.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Une ligne de sous-titre. Au-delà, ça déborde ou ça force à lire trop vite.
MAX_CARACTERES = 42
MAX_MOTS = 7
# En dessous, l'affichage clignote ; au-delà, la ligne traîne après la voix.
DUREE_MIN_MS = 700
DUREE_MAX_MS = 6000
# Une ponctuation forte termine une ligne : on suit la respiration du texte.
_FIN_DE_SOUFFLE = re.compile(r"[.!?…:;]$")

# Mots sur lesquels une ligne ne doit pas se terminer : ils annoncent la suite
# et laissent le lecteur en suspens (« chef-lieu du » / « Hwanghae du Nord »).
# Le mot est alors reporté sur la ligne suivante, avec son timing.
_MOTS_OUTILS = frozenset("""
le la les un une des du de d au aux a l
ce cet cette ces mon ton son ma ta sa mes tes ses notre votre leur nos vos leurs
dans sur sous par pour avec sans vers chez entre contre depuis pendant jusqu
et ou mais donc or ni car que qu qui quoi dont ou si comme quand lorsque
je tu il elle on nous vous ils elles se me te lui y en
est es suis sont ai as ont avait etait plus tres bien tout tous toute toutes
""".split())


def _est_mot_outil(mot: str) -> bool:
    nu = re.sub(r"[^\w']", "", mot.lower()).rstrip("'")
    nu = "".join(
        c for c in unicodedata.normalize("NFD", nu) if unicodedata.category(c) != "Mn"
    )
    return nu in _MOTS_OUTILS


@dataclass(frozen=True)
class Ligne:
    start_ms: int
    end_ms: int
    texte: str


def lignes(
    mots: list[str], durees_ms: list[int], silences_ms: list[int], speed: float = 1.0
) -> list[Ligne]:
    """Regroupe les mots en lignes affichables, à l'échelle de temps de SORTIE.

    Le regroupement suit la ponctuation d'abord, la longueur ensuite : couper
    au milieu d'une proposition se lit mal, alors qu'une ligne courte terminée
    par un point se lit très bien.
    """
    # (mot, debut_ms, fin_ms) : garder les bornes de chaque mot est ce qui
    # permet de reporter le dernier mot sur la ligne suivante SANS fausser les
    # timings — c'est tout l'intérêt par rapport à un découpage sur le texte.
    horloge = 0
    marques: list[tuple[str, int, int]] = []
    for mot, duree, silence in zip(mots, durees_ms, silences_ms, strict=False):
        marques.append((mot, horloge, horloge + duree))
        horloge += duree + silence

    out: list[Ligne] = []
    courant: list[tuple[str, int, int]] = []

    def fermer(sur_ponctuation: bool = False) -> None:
        """Ferme la ligne courante, en reportant éventuellement son dernier mot.

        Une ligne qui se termine par « chef-lieu du » laisse le lecteur en
        suspens : le mot-outil part sur la ligne suivante, avec son timing.
        Sauf si ce mot porte déjà une ponctuation (la pause est réelle) ou si
        la ligne deviendrait trop courte pour être lisible.
        """
        nonlocal courant
        if not courant:
            return
        # Boucle et non un seul retrait : enlever « un » de « ... et un »
        # expose « et », qui est lui aussi un mot-outil. Sur un texte réel,
        # deux à trois mots d'affilée sont courants (« et que le », « dans le
        # but de »). La ligne garde toujours au moins trois mots.
        reportes: list[tuple[str, int, int]] = []
        while (
            not sur_ponctuation
            and len(courant) >= 4
            and _est_mot_outil(courant[-1][0])
            and not re.search(r"[,;:.!?…]$", courant[-1][0])
        ):
            reportes.insert(0, courant.pop())
        texte = " ".join(m for m, _, _ in courant)
        out.append(Ligne(round(courant[0][1] / speed), round(courant[-1][2] / speed), texte))
        courant = reportes

    for marque in marques:
        mot = marque[0]
        # Longueur testée AVANT d'ajouter : tester après laissait passer des
        # lignes bien au-delà de la limite (54 caractères mesurés sur un
        # script réel pour un plafond à 42).
        if courant and (
            len(" ".join(m for m, _, _ in courant)) + 1 + len(mot) > MAX_CARACTERES
            or len(courant) >= MAX_MOTS
        ):
            fermer()
        courant.append(marque)
        # Couper sur la ponctuation, mais pas au point de produire une ligne
        # qui clignote : un « Oui. » isolé reste collé à la phrase suivante.
        if _FIN_DE_SOUFFLE.search(mot) and (courant[-1][2] - courant[0][1]) >= DUREE_MIN_MS * speed:
            fermer(sur_ponctuation=True)

    fermer(sur_ponctuation=True)

    # Bornes : une ligne ne doit ni clignoter ni survivre à la suivante.
    ajustees: list[Ligne] = []
    for i, ligne in enumerate(out):
        # On étire jusqu'à la ligne suivante quand il reste du silence : la
        # ligne se lit plus confortablement, et l'écran ne se vide pas entre
        # deux phrases. On ne mord jamais sur la suivante.
        plafond = out[i + 1].start_ms if i + 1 < len(out) else ligne.end_ms + DUREE_MIN_MS
        fin = min(max(ligne.end_ms, min(plafond, ligne.start_ms + DUREE_MIN_MS)), plafond)
        fin = min(fin, ligne.start_ms + DUREE_MAX_MS)
        ajustees.append(Ligne(ligne.start_ms, max(fin, ligne.start_ms + 1), ligne.texte))
    return ajustees


def remap_for_render(
    lignes_: list[Ligne], breakpoints: list[tuple[int, int, int]], speed: float
) -> list[Ligne]:
    """Convertit des lignes en temps ABSOLU (échelle du script entier, telle
    que produite par `lignes(..., speed=1.0)`) vers le temps du fichier
    RENDU, en tenant compte des scènes exclues (intro tronquée avant le
    premier palier) et des blocs insérés avant certaines scènes (cartons de
    palier parlés, qui ajoutent du temps réel).

    `breakpoints` : liste ORDONNÉE de (début_scène_ms, fin_scène_ms,
    position_ms), une entrée par scène retenue dans le montage —
    `position_ms` est l'instant, dans le fichier rendu, où le contenu de
    cette scène commence. Construite par `render.py` au fil de l'assemblage :
    lui seul sait combien de temps chaque carton a réellement inséré avant
    telle ou telle scène ; ce module n'a aucune raison de le recalculer.

    Une ligne dont l'intervalle ne recoupe aucune scène retenue (silence
    dans l'intro exclue, ou entre deux scènes coupées en --sample) est
    abandonnée. Une ligne qui chevauche exactement la frontière entre deux
    scènes retenues est rattachée à la première des deux — limite acceptée :
    ça ne peut décaler qu'une ligne isolée à la jonction d'un palier, jamais
    accumuler sur le reste.
    """
    out: list[Ligne] = []
    for ligne in lignes_:
        for debut_scene, fin_scene, position in breakpoints:
            if ligne.start_ms < fin_scene and ligne.end_ms > debut_scene:
                decalage = position - round(debut_scene / speed)
                start = max(position, round(ligne.start_ms / speed) + decalage)
                end = round(ligne.end_ms / speed) + decalage
                out.append(Ligne(start, max(end, start + 1), ligne.texte))
                break
    return out


def _horodatage_srt(ms: int) -> str:
    h, reste = divmod(max(0, ms), 3_600_000)
    m, reste = divmod(reste, 60_000)
    s, ms = divmod(reste, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(lignes_: list[Ligne]) -> str:
    blocs = [
        f"{i}\n{_horodatage_srt(l.start_ms)} --> {_horodatage_srt(l.end_ms)}\n{l.texte}\n"
        for i, l in enumerate(lignes_, 1)
    ]
    return "\n".join(blocs)


def _horodatage_ass(ms: int) -> str:
    h, reste = divmod(max(0, ms), 3_600_000)
    m, reste = divmod(reste, 60_000)
    s, cs = divmod(reste, 1000)
    return f"{h:d}:{m:02d}:{s:02d}.{cs // 10:02d}"


def to_ass(lignes_: list[Ligne], width: int, height: int, decalage_ms: int = 0) -> str:
    """Fichier ASS pour un segment, timings ramenés à son origine.

    Blanc plein, contour noir épais, sans ombre ni fond : lisible sur les
    aplats clairs comme sur les fonds sombres du style. `PlayResX/Y` doit
    valoir la résolution FINALE du segment, sinon libass met le texte à
    l'échelle et le contour change d'épaisseur d'un rendu à l'autre.
    """
    # height/20 (54 px en 1080p) était trop petit à l'usage — comparé le
    # 19/09 à height/15 (72 px), nettement plus lisible sans déborder.
    taille = max(32, round(height / 15))
    contour = max(3, round(taille / 12))
    marge = round(height / 12)
    entete = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{taille},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{contour},0,2,80,80,{marge},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    evenements = [
        f"Dialogue: 0,{_horodatage_ass(l.start_ms - decalage_ms)},"
        f"{_horodatage_ass(l.end_ms - decalage_ms)},Default,,0,0,0,,"
        + l.texte.replace("\n", " ").replace("{", "(").replace("}", ")")
        for l in lignes_
    ]
    return entete + "\n".join(evenements) + "\n"
