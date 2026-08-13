"""Étape 6 — Rendu. `scenes.json` + `audio.wav` + `images/` → `out.mp4` (SPEC §4).

À 24 min et 24 fps, c'est ~34 000 frames de zoompan — souvent plus long que la
génération d'images. D'où le découpage en un segment par scène, rendus en
parallèle sur les cœurs CPU, puis concaténés en copie de flux.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..ffmpeg import concat_command, duration_ms, frame_span, segment_command
from ..project import Options, Project
from ..scenes import Scene, load, sample_indices


def _render_one(args: tuple[Path, Path, int, int, bool]) -> tuple[Path, str | None]:
    image, output, frames, seed, draft = args
    command = segment_command(image, output, frames, seed, draft)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return output, result.stderr.strip()[:300]
    return output, None


def run(project: Project, opts: Options) -> None:
    scenes = load(project.scenes)
    selected = set(sample_indices(scenes, scenes[-1].end_ms)) if opts.sample else None
    todo: list[Scene] = [s for s in scenes if selected is None or s.index in selected]

    project.segments.mkdir(parents=True, exist_ok=True)

    jobs = []
    for scene in todo:
        image = project.images / f"{scene.index:04d}.png"
        if not image.exists():
            raise RuntimeError(
                f"image manquante pour la scène {scene.index} — lancer l'étape images"
            )
        jobs.append(
            (
                image,
                project.segments / f"{scene.index:04d}.mp4",
                frame_span(scene.start_ms, scene.end_ms),
                scene.seed,
                opts.draft,
            )
        )

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(_render_one, jobs))

    failures = [(p, e) for p, e in results if e]
    if failures:
        for path, error in failures[:3]:
            print(f"    {path.name}: {error}")
        raise RuntimeError(f"{len(failures)} segments en échec")

    # Le concat demuxer veut des chemins relatifs au fichier de liste.
    list_file = project.segments / "concat.txt"
    list_file.write_text(
        "".join(f"file '{path.name}'\n" for path, _ in results),
        encoding="utf-8",
    )

    subprocess.run(concat_command(list_file, project.audio, project.out), check=True)

    video_ms = duration_ms(project.out)
    audio_ms = duration_ms(project.audio)
    expected = sum(s.duration_ms for s in todo)

    # Purge des segments : en --sample on les garde, l'itération est le but.
    if not opts.sample:
        shutil.rmtree(project.segments, ignore_errors=True)

    drift = video_ms - (audio_ms if selected is None else expected)
    print(f"  {len(todo)} segments — vidéo {video_ms / 1000:.1f} s, audio {audio_ms / 1000:.1f} s")
    print(f"  dérive vidéo/audio : {drift:+d} ms")
