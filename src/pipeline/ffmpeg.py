"""Construction des commandes ffmpeg (SPEC §2).

Isolé du reste pour être lisible et vérifiable à sec : ces commandes sont le
poste de temps dominant du rendu et la principale source de bizarreries.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

FPS = 24
WIDTH = 1920
HEIGHT = 1080
DRAFT_WIDTH = 1138
DRAFT_HEIGHT = 640

# zoompan calcule sa position de recadrage en pixels entiers, à sa résolution
# de SORTIE — indépendamment de la résolution d'entrée. Suréchantillonner
# l'entrée (comme le faisait la première version) ne corrige donc rien : le
# même arrondi frappe à 1080p qu'à 4K en entrée, et le résultat est un
# tremblement caractéristique, un saut d'un pixel toutes les quelques frames.
#
# Le correctif standard pour ce défaut connu de zoompan : le faire sortir à une
# résolution plus grande que la cible, puis redescendre avec un filtre de
# qualité (lanczos). Un saut d'un pixel à 2× la résolution finale devient un
# demi-pixel après la réduction — sous le seuil de perception, lissé par
# l'interpolation plutôt que visible comme un décrochage.
OVERSAMPLE = 2

# Vitesse de lecture du montage final. 1.25 par défaut : validé à l'oreille le
# 17/09 sur un extrait de video5, c'est le rythme du format — les silences se
# resserrent et la narration gagne du nerf, sans effet « accéléré ».
#
# Appliqué ici plutôt qu'en post-traitement sur out.mp4 : les segments sont
# rendus directement à la bonne longueur (frame_span et title_card_command
# divisent par SPEED), donc la vidéo n'est jamais ré-encodée une deuxième
# fois. Seul l'audio passe par `atempo`, ce qui ne coûte rien puisque
# concat_command le ré-encode déjà en AAC. `atempo` corrige le pitch : pas de
# voix de canard.
#
# PIPELINE_SPEED=1.0 rétablit la vitesse d'origine.
SPEED = float(os.environ.get("PIPELINE_SPEED", "1.25"))


def _atempo_chain(speed: float) -> str:
    """`atempo` n'accepte qu'un facteur entre 0.5 et 2.0 — au-delà il faut
    chaîner plusieurs filtres. Inutile à 1.25, mais une valeur exotique passée
    par PIPELINE_SPEED ne doit pas produire un filtre silencieusement refusé."""
    facteurs, reste = [], speed
    while reste > 2.0:
        facteurs.append(2.0)
        reste /= 2.0
    while reste < 0.5:
        facteurs.append(0.5)
        reste /= 0.5
    facteurs.append(reste)
    return ",".join(f"atempo={f:.6f}" for f in facteurs)


# Durée par défaut d'un carton de palier quand il n'est pas prononcé (pas de
# TTS disponible pour l'annonce) — sinon la durée réelle de l'audio synthétisé
# prime (voir render.py). Positive.md : aucun mouvement de caméra dessus,
# juste le texte qui s'affiche puis passe.
LEVEL_CARD_S = 1.8

# Police bulle/arrondie (Chewy, Google Fonts, OFL) — pas une police système :
# c'est le style demandé (cf. capture de référence, 04/09), un arial gras ne
# le rendait pas. Statique (pas de police variable) : ffmpeg/freetype ne sait
# pas choisir un axe de graisse à l'affichage, une police variable rendrait
# à sa graisse par défaut, pas forcément grasse.
LEVEL_CARD_FONT = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "Chewy-Regular.ttf"

# Tremblement des lettres — amplitude en pixels et période en secondes de
# l'oscillation verticale par caractère.
#
# Désactivé (amplitude 0) le 17/09 : l'effet venait d'une capture de référence
# (04/09), mais sur les vidéos réelles il distrait plus qu'il n'apporte. Le
# carton gagne à rester parfaitement fixe. Remettre 3 pour le retrouver.
WOBBLE_AMPLITUDE_PX = 0
WOBBLE_PERIOD_S = 1.6


def binary(name: str = "ffmpeg") -> str:
    found = shutil.which(name) or str(Path.home() / ".local" / "bin" / name)
    if not Path(found).exists():
        raise RuntimeError(f"{name} introuvable — installer ffmpeg")
    return found


def _escape_filter_arg(path: Path) -> str:
    """Échappe un chemin pour l'utiliser comme valeur d'un filtre ffmpeg.

    Les deux-points d'un chemin Windows (`C:\\...`) sont le séparateur
    d'options du filtre lui-même — sans échappement, `fontfile=C:/Windows/...`
    coupe l'argument à la lettre `C`. Passer par `/` évite en plus tout
    problème avec les antislashs.
    """
    return str(path).replace("\\", "/").replace(":", "\\:")


def ken_burns(seed: int) -> tuple[float, float, float, float, float, float]:
    """Cadrage, dérivé du seed (SPEC §4, étape 6) — désormais figé, pas animé.

    Retourne (z0, z1, x0, x1, y0, y1) où z est le facteur de zoom et x/y sont
    des positions relatives dans [0, 1] du cadre à l'intérieur de l'image.

    z0==z1 en plus de x0==x1 et y0==y1 : aucune animation du tout, ni
    panoramique ni zoom. Confirmé sur la première vidéo complète (01/09) —
    « aucun mouvement » voulait dire aucun mouvement, zoom compris, pas
    seulement le panoramique retiré plus tôt. Chaque plan garde un recadrage
    légèrement différent selon le seed pour la variété visuelle, mais tenu
    fixe sur toute sa durée — le montage vient de la coupe, pas d'un
    mouvement dans le plan.
    """
    amplitude = 1.06 + (seed % 9) / 100.0  # 1.06 → 1.14, variété de cadrage seule
    return amplitude, amplitude, 0.5, 0.5, 0.5, 0.5


def frame_span(start_ms: int, end_ms: int) -> int:
    """Nombre de frames d'une scène, dérivé de bornes **absolues**.

    Arrondir chaque durée indépendamment fait accumuler l'erreur segment après
    segment : sur 300 scènes, la vidéo dérive de l'audio d'autant. En arrondissant
    les bornes puis en soustrayant, les erreurs se télescopent et l'écart total
    reste borné à une frame, quelle que soit la longueur de la vidéo.
    """
    # Les bornes sont d'abord ramenées à l'échelle de temps de SORTIE (donc
    # divisées par SPEED) AVANT l'arrondi : c'est ce qui préserve le
    # télescopage des erreurs décrit ci-dessus. Diviser le résultat après coup
    # ré-introduirait une erreur d'arrondi par segment.
    return max(
        1,
        round(end_ms * FPS / (1000 * SPEED)) - round(start_ms * FPS / (1000 * SPEED)),
    )


def segment_command(
    image: Path,
    output: Path,
    frames: int,
    seed: int,
    draft: bool = False,
    subtitles: Path | None = None,
) -> list[str]:
    """Rend une scène en un MP4 autonome, prêt pour le concat en copie de flux."""
    width, height = (DRAFT_WIDTH, DRAFT_HEIGHT) if draft else (WIDTH, HEIGHT)
    z0, z1, x0, x1, y0, y1 = ken_burns(seed)

    # Progression linéaire sur le compteur de frames de sortie.
    t = f"(on/{max(frames - 1, 1)})"
    z = f"{z0:.4f}+({z1 - z0:.4f})*{t}"
    x = f"(iw-iw/zoom)*({x0:.3f}+({x1 - x0:.3f})*{t})"
    y = f"(ih-ih/zoom)*({y0:.3f}+({y1 - y0:.3f})*{t})"

    vf = (
        # Recadrage au ratio cible : on remplit puis on rogne, jamais de bandes.
        f"scale={width * OVERSAMPLE}:{height * OVERSAMPLE}"
        f":force_original_aspect_ratio=increase,"
        f"crop={width * OVERSAMPLE}:{height * OVERSAMPLE},"
        # zoompan sort à la même résolution suréchantillonnée que son entrée —
        # c'est cette sortie-là qui doit être plus grande que la cible finale
        # pour que le lissage ait quelque chose à lisser.
        f"zoompan=z='{z}':x='{x}':y='{y}':d=1"
        f":s={width * OVERSAMPLE}x{height * OVERSAMPLE}:fps={FPS},"
        # Redescente à la résolution finale : c'est elle qui absorbe le
        # tremblement en le transformant en flou sous-pixel.
        f"scale={width}:{height}:flags=lanczos,"
        f"format=yuv420p"
    )
    if subtitles is not None:
        # APRÈS la redescente à la résolution finale : incrusté avant, le
        # texte serait dessiné à 2x puis réduit avec l'image, et son contour
        # perdrait la moitié de son épaisseur.
        vf += f",subtitles='{_escape_filter_arg(subtitles)}'"

    return [
        binary(),
        "-y", "-loglevel", "error", "-nostdin",
        "-loop", "1", "-framerate", str(FPS), "-i", str(image),
        "-frames:v", str(frames),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "ultrafast" if draft else "medium",
        "-crf", "28" if draft else "20",
        # Un GOP fixe et identique partout : sans ça, le concat en copie de flux
        # produit des sauts de timing aux jointures.
        "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        str(output),
    ]


def _escape_drawtext_text(text: str) -> str:
    """Échappe une valeur `text=` inline de drawtext.

    Deux-points, apostrophes, antislashs et pourcentages ont un sens spécial
    dans le graphe de filtres — nécessaire ici puisque chaque caractère du
    carton est un `text=` inline séparé (pour le positionner et l'animer
    individuellement), pas un `textfile=` global comme avant.
    """
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )


def _split_label(label: str) -> list[str]:
    """« Niveau 1 : Esclave » → 2 lignes, comme la référence (« Level One: »
    / « The Farmer »). Sans « : », une seule ligne — pas de retour à la
    ligne automatique au-delà, les libellés restent courts par construction
    (SPEC : NIVEAU: est écrit à la main, pas généré)."""
    if " : " in label:
        head, tail = label.split(" : ", 1)
        return [f"{head} :", tail]
    return [label]


def _char_positions(
    line: str, font_path: Path, fontsize: int, canvas_width: int
) -> list[tuple[str, int]]:
    """Position x de chaque caractère d'une ligne centrée.

    Mesuré avec la police réelle (PIL, même moteur freetype que ffmpeg) pour
    que les caractères se touchent sans trou ni chevauchement une fois
    rendus individuellement — un seul `drawtext` par caractère est ce qui
    permet à chacun de trembler indépendamment.
    """
    from PIL import ImageFont

    font = ImageFont.truetype(str(font_path), fontsize)
    widths = [font.getlength(ch) for ch in line]
    total = sum(widths)
    x = (canvas_width - total) / 2
    positions = []
    for ch, w in zip(line, widths, strict=True):
        positions.append((ch, round(x)))
        x += w
    return positions


def title_card_command(
    label: str,
    output: Path,
    frames: int,
    draft: bool = False,
) -> list[str]:
    """Rend le carton d'un palier — fond noir, police bulle, lettres qui
    tremblent légèrement (référence 04/09), aucun mouvement de caméra.

    `frames` vient de l'appelant, exprimé dans la même unité que
    `segment_command` : le carton est un plan comme un autre, découpé DANS la
    durée de sa scène et non ajouté par-dessus, donc il n'y a plus de durée à
    convertir ni de vitesse à appliquer ici. Chaque
    caractère est un `drawtext` positionné et animé séparément — c'est ce qui
    permet le tremblement individuel, `textfile=` global ne le permettait pas.
    """
    # L'apostrophe droite est IMPOSSIBLE à passer dans une chaîne de drawtext :
    # ffmpeg n'autorise aucun échappement à l'intérieur de quotes simples, et
    # chaque caractère du carton est un `text='X'` séparé (pour le positionner
    # individuellement). Vérifié le 19/09 sur « Niveau six : tireur d'élite » —
    # `'`, `'''` et le texte sans quotes échouent tous de la même façon, le
    # parseur perd le fil et prend le chemin de la police pour une option.
    # L'apostrophe typographique passe sans problème, et c'est de toute façon
    # la forme correcte en français.
    label = label.replace("'", "’")
    width, height = (DRAFT_WIDTH, DRAFT_HEIGHT) if draft else (WIDTH, HEIGHT)
    fontsize = max(28, width // 16)
    lines = _split_label(label)
    line_height = int(fontsize * 1.15)
    top = (height - line_height * len(lines)) / 2

    filters = []
    for li, line in enumerate(lines):
        y_base = top + li * line_height
        for ch, x in _char_positions(line, LEVEL_CARD_FONT, fontsize, width):
            if ch == " ":
                continue
            if WOBBLE_AMPLITUDE_PX:
                phase = x / 40.0  # décale la phase par caractère selon sa position
                y_expr = (
                    f"{y_base:.1f}+{WOBBLE_AMPLITUDE_PX}*"
                    f"sin(2*PI*(t/{WOBBLE_PERIOD_S}+{phase:.3f}))"
                )
            else:
                # Position constante : évite de faire évaluer à ffmpeg une
                # expression temporelle par caractère et par frame pour rien.
                y_expr = f"{y_base:.1f}"
            filters.append(
                f"drawtext=fontfile='{_escape_filter_arg(LEVEL_CARD_FONT)}'"
                f":text='{_escape_drawtext_text(ch)}'"
                # expansion=none : sans ça, `%` ouvre une séquence
                # d'expansion de drawtext (`%{...}`) et casse le filtre.
                # Seul caractère restant à poser problème après le
                # remplacement des apostrophes (vérifié le 19/09 sur tout le
                # jeu de ponctuation et d'accents français).
                f":fontsize={fontsize}:fontcolor=white:expansion=none:x={x}:y='{y_expr}'"
            )

    return [
        binary(),
        "-y", "-loglevel", "error", "-nostdin",
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:r={FPS}",
        "-frames:v", str(frames),
        "-vf", ",".join(filters),
        "-c:v", "libx264",
        "-preset", "ultrafast" if draft else "medium",
        "-crf", "28" if draft else "20",
        "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        str(output),
    ]


def audio_slice_command(audio: Path, start_ms: int, end_ms: int, output: Path) -> list[str]:
    """Extrait la tranche [start_ms, end_ms) de l'audio, sans ré-encoder.

    Utilisé par le rendu `--sample` : les scènes sélectionnées viennent de
    fenêtres disjointes, donc l'audio ne peut pas être une simple tranche
    continue — il faut découper puis recoller dans le même ordre que la vidéo.
    """
    return [
        binary(),
        "-y", "-loglevel", "error", "-nostdin",
        "-i", str(audio),
        "-ss", f"{start_ms / 1000:.3f}",
        "-to", f"{end_ms / 1000:.3f}",
        "-c", "copy",
        str(output),
    ]


def concat_audio_command(list_file: Path, output: Path) -> list[str]:
    """Concatène des tranches audio (mêmes format/débit — copie de flux)."""
    return [
        binary(),
        "-y", "-loglevel", "error", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy",
        str(output),
    ]


def concat_command(list_file: Path, audio: Path, output: Path) -> list[str]:
    """Concatène les segments sans ré-encoder, et muxe l'audio."""
    return [
        binary(),
        "-y", "-loglevel", "error", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-i", str(audio),
        "-c:v", "copy",
        # L'audio est de toute façon ré-encodé ici : appliquer atempo au
        # passage ne coûte rien de plus.
        *(["-filter:a", _atempo_chain(SPEED)] if SPEED != 1.0 else []),
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(output),
    ]


def duration_ms(path: Path) -> int:
    """Durée réelle d'un média, via ffprobe."""
    out = subprocess.run(
        [
            binary("ffprobe"), "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return round(float(out.stdout.strip()) * 1000)
