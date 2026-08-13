"""Construction des commandes ffmpeg (SPEC §2).

Isolé du reste pour être lisible et vérifiable à sec : ces commandes sont le
poste de temps dominant du rendu et la principale source de bizarreries.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FPS = 24
WIDTH = 1920
HEIGHT = 1080
DRAFT_WIDTH = 1138
DRAFT_HEIGHT = 640

# zoompan calcule x et y en entiers sur la résolution d'**entrée**. Sur un zoom
# lent, l'arrondi fait sauter l'image d'un pixel toutes les quelques frames —
# le tremblement caractéristique du Ken Burns fait à la va-vite. Suréchantillonner
# l'entrée avant le filtre divise l'erreur d'autant. C'est aussi l'agrandissement
# dont on a besoin de toute façon, SDXL produisant plus petit que la cible.
OVERSAMPLE = 2


def binary(name: str = "ffmpeg") -> str:
    found = shutil.which(name) or str(Path.home() / ".local" / "bin" / name)
    if not Path(found).exists():
        raise RuntimeError(f"{name} introuvable — installer ffmpeg")
    return found


def ken_burns(seed: int) -> tuple[float, float, float, float, float, float]:
    """Sens et amplitude du mouvement, dérivés du seed (SPEC §4, étape 6).

    Retourne (z0, z1, x0, x1, y0, y1) où z est le facteur de zoom et x/y sont
    des positions relatives dans [0, 1] du cadre à l'intérieur de l'image.
    """
    amplitude = 1.06 + (seed % 9) / 100.0  # 1.06 → 1.14
    zoom_in = bool((seed >> 4) & 1)
    z0, z1 = (1.0, amplitude) if zoom_in else (amplitude, 1.0)

    # Quatre directions de travelling, pour ne pas répéter le même mouvement
    # sur deux scènes voisines.
    direction = (seed >> 5) & 3
    x0, x1 = [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5), (0.35, 0.65)][direction]
    y0, y1 = [(0.5, 0.5), (0.35, 0.65), (0.0, 1.0), (1.0, 0.0)][direction]
    return z0, z1, x0, x1, y0, y1


def frame_span(start_ms: int, end_ms: int) -> int:
    """Nombre de frames d'une scène, dérivé de bornes **absolues**.

    Arrondir chaque durée indépendamment fait accumuler l'erreur segment après
    segment : sur 300 scènes, la vidéo dérive de l'audio d'autant. En arrondissant
    les bornes puis en soustrayant, les erreurs se télescopent et l'écart total
    reste borné à une frame, quelle que soit la longueur de la vidéo.
    """
    return max(1, round(end_ms * FPS / 1000) - round(start_ms * FPS / 1000))


def segment_command(
    image: Path,
    output: Path,
    frames: int,
    seed: int,
    draft: bool = False,
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
        f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={width}x{height}:fps={FPS},"
        f"format=yuv420p"
    )

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


def concat_command(list_file: Path, audio: Path, output: Path) -> list[str]:
    """Concatène les segments sans ré-encoder, et muxe l'audio."""
    return [
        binary(),
        "-y", "-loglevel", "error", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-i", str(audio),
        "-c:v", "copy",
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
