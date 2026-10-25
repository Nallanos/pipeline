"""Étape 6 — Rendu. `scenes.json` + `audio.wav` + `images/` → `out.mp4` (SPEC §4).

À 24 min et 24 fps, c'est ~34 000 frames de zoompan — souvent plus long que la
génération d'images. D'où le découpage en un segment par scène, rendus en
parallèle sur les cœurs CPU, puis concaténés en copie de flux.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import wave

from ..ffmpeg import (
    DRAFT_HEIGHT,
    binary,
    FPS,
    DRAFT_WIDTH,
    HEIGHT,
    LEVEL_CARD_S,
    SPEED,
    WIDTH,
    audio_slice_command,
    concat_audio_command,
    concat_command,
    duration_ms,
    frame_span,
    segment_command,
    title_card_command,
)
from .. import subtitles
from ..language import LanguageConfig, for_project
from ..project import Options, Project, safe_wav_frames, write_atomic
from ..scenes import Scene, load, sample_indices
from . import tts
from .tts import synthesize_snippet


def _render_one(args: tuple[Path, Path, int, int, bool, Path | None]) -> tuple[Path, str | None]:
    image, output, frames, seed, draft, subs = args
    command = segment_command(image, output, frames, seed, draft, subs)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return output, result.stderr.strip()[:300]
    return output, None


def _render_title_card(label: str, video_path: Path, frames: int, draft: bool) -> None:
    command = title_card_command(label, video_path, frames, draft)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"carton de palier {label!r} : {result.stderr.strip()[:300]}")




def _load_level_cards(project: Project, scenes: list[Scene]) -> dict[int, list[str]]:
    """`levels.json` (marqueurs `NIVEAU:` extraits à l'import) → scène cible.

    Chaque marqueur pointe un index de mot (SPEC project.clean_source) ; la
    scène cible est la première dont `word_start` l'atteint ou le dépasse —
    aucun besoin de correspondance floue, l'index est exact par construction.
    """
    cards: dict[int, list[str]] = defaultdict(list)
    if not project.levels.exists():
        return cards
    raw = json.loads(project.levels.read_text(encoding="utf-8"))["levels"]
    for entry in raw:
        target = next((s.index for s in scenes if s.word_start >= entry["word_index"]), None)
        if target is None:
            print(f"    palier ignoré (au-delà de la fin du script) : {entry['label']!r}")
            continue
        cards[target].append(entry["label"])
    return cards


def _write_silence(reference_wav: Path, output: Path, duration_s: float) -> None:
    """Repli quand la synthèse de l'annonce échoue : du silence, à la place
    d'un carton muet-par-accident qui romprait le rythme sans le signaler."""
    with wave.open(str(reference_wav), "rb") as ref:
        rate, channels, width = ref.getframerate(), ref.getnchannels(), ref.getsampwidth()
    with wave.open(str(output), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00" * (round(rate * duration_s) * channels * width))


def _match_audio_format(path: Path, reference_wav: Path) -> None:
    """Reformate l'annonce synthétisée sur le débit/canal de l'audio
    principal : `concat_audio_command` colle les tranches en copie de flux,
    ce qui exige un format identique sur tout le fichier de liste. Piper
    (22 050 Hz) et l'audio principal (44 100 Hz) ne matchent jamais tels
    quels."""
    with wave.open(str(reference_wav), "rb") as ref:
        rate, channels, width = ref.getframerate(), ref.getnchannels(), ref.getsampwidth()
    tmp = path.with_suffix(".reformat.wav")
    subprocess.run(
        [
            binary(), "-y", "-loglevel", "error", "-i", str(path),
            "-ar", str(rate), "-ac", str(channels),
            "-sample_fmt", "s16" if width == 2 else "s32",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(path)


def _card_audio(
    language: LanguageConfig,
    label: str,
    output: Path,
    project_audio: Path,
    provider: str | None,
    voice_id: str | None = None,
) -> float:
    """Audio de l'annonce d'un palier — l'appel TTS est SÉPARÉ du script
    principal (voir `synthesize_snippet`), donc rien en amont (`align`,
    `segment`, `prompts`, `images`) n'a besoin de rejouer pour l'obtenir.
    Renvoie la durée réelle en secondes, repli sur `LEVEL_CARD_S` de silence
    si la synthèse échoue — même tolérance que le reste du pipeline face à un
    aléa de génération (voir la docstring du module `images.py`).

    `provider` vient de `tts.load_provider()` : LA MÊME voix que la narration
    principale, jamais une devinée depuis l'environnement courant (voir
    `synthesize_snippet`).
    """
    try:
        synthesize_snippet(language, label, output, provider=provider, voice_id=voice_id)
        _match_audio_format(output, project_audio)
    except Exception as exc:  # noqa: BLE001 — défensif par conception, voir ci-dessus
        print(f"    annonce non synthétisée ({exc}) — repli sur {LEVEL_CARD_S}s de silence")
        _write_silence(project_audio, output, LEVEL_CARD_S)
        return LEVEL_CARD_S
    with wave.open(str(output), "rb") as w:
        rate = w.getframerate()
    return safe_wav_frames(output) / rate


def run(project: Project, opts: Options) -> None:
    scenes = load(project.scenes)
    selected = set(sample_indices(scenes, scenes[-1].end_ms)) if opts.sample else None

    # Cartons de palier : seulement sur un rendu complet — le --sample est un
    # artefact de jugement (3 fenêtres de 30 s, cf. sample_indices), rien ne
    # garantit qu'un palier tombe dedans, et ce n'est pas ce qu'il sert à juger.
    cards = _load_level_cards(project, scenes) if selected is None else {}
    language = for_project(project.dir) if cards else None
    # Voir `tts.load_provider()` : la même voix — et pour Fish Audio, le même
    # `voice_id` — que la narration principale, jamais devinés depuis
    # l'environnement au moment du rendu.
    tts_provider, tts_voice_id = tts.load_provider(project) if cards else (None, None)
    if cards and tts_provider is None:
        print(
            "    tts.json absent (projet antérieur au 20/09) — l'annonce du "
            "carton peut sortir dans une voix différente de la narration ; "
            "PIPELINE_TTS_PROVIDER doit être réglé à la main pour ce run"
        )

    todo: list[Scene] = [s for s in scenes if selected is None or s.index in selected]
    if cards:
        # La vidéo commence au premier palier, pas avant (demandé
        # explicitement, 04/09) : le premier plan doit être « Niveau 1 »,
        # pas une introduction. Rien n'invalide le script/l'audio/les images
        # d'avant — ils restent générés, simplement pas montés.
        first_card_scene = min(cards)
        todo = [s for s in todo if s.index >= first_card_scene]

    project.segments.mkdir(parents=True, exist_ok=True)

    # Sous-titres : dérivés de l'alignement mot à mot déjà calculé, donc
    # gratuits. Absents si `align` n'a pas tourné — le rendu se fait alors
    # sans, exactement comme avant leur ajout.
    #
    # `speed=1.0` volontairement : gardés en temps ABSOLU (échelle du script
    # entier, comme `scene.start_ms`) le temps de la boucle ci-dessous, qui a
    # besoin de comparer des grandeurs homogènes. `subtitles.remap_for_render`
    # applique `SPEED` et les décalages (intro tronquée, cartons parlés
    # insérés) en une seule passe, à la fin.
    sous_titres_abs: list[subtitles.Ligne] = []
    if project.alignment.exists():
        brut = json.loads(project.alignment.read_text(encoding="utf-8"))
        sous_titres_abs = subtitles.lignes(
            brut["words"], brut["durations_ms"], brut["gaps_ms"], speed=1.0
        )

    # Le carton parlé insère du temps RÉEL (annonce TTS jouée avant la scène,
    # voir `_card_audio`) — décalage inévitable puisque quelqu'un doit avoir
    # fini de parler avant que la scène commence. `breakpoints` trace, pour
    # chaque scène retenue, où elle atterrit dans le fichier final : c'est la
    # seule source qui sait combien de temps les cartons précédents ont
    # réellement ajouté, donc la seule qui puisse recaler les sous-titres.
    breakpoints: list[tuple[int, int, int]] = []
    video_time_ms = 0

    jobs = []
    audio_parts: list[Path] = []
    video_paths: list[Path] = []
    for scene in todo:
        image = project.images / f"{scene.index:04d}.png"
        if not image.exists():
            raise RuntimeError(
                f"image manquante pour la scène {scene.index} — lancer l'étape images"
            )

        # Cartons de CETTE scène, rendus et insérés AVANT elle — vidéo et
        # audio ensemble, pour que les deux fichiers restent une seule source
        # de vérité (jamais deux pistes assemblées séparément en espérant
        # qu'elles restent synchrones).
        for i, label in enumerate(cards.get(scene.index, [])):
            assert language is not None  # cards non vide => language chargée plus haut
            audio_path = project.segments / f"level-{scene.index:04d}-{i}.wav"
            duree_s = _card_audio(language, label, audio_path, project.audio, tts_provider, tts_voice_id)
            # Toute la piste audio finale passe par `atempo=SPEED` au mux
            # (voir `concat_command`), l'annonce y compris : sa vidéo doit
            # donc durer `duree_s / SPEED`, pas `duree_s`, pour rester calée
            # sur son propre audio une fois celui-ci accéléré.
            frames_carton = max(1, round(duree_s * FPS / SPEED))
            card_path = project.segments / f"level-{scene.index:04d}-{i}.mp4"
            _render_title_card(label, card_path, frames_carton, opts.draft)
            video_paths.append(card_path)
            audio_parts.append(audio_path)
            video_time_ms += round(frames_carton * 1000 / FPS)

        frames_scene = frame_span(scene.start_ms, scene.end_ms)
        breakpoints.append((scene.start_ms, scene.end_ms, video_time_ms))

        subs_path: Path | None = None
        if sous_titres_abs:
            dans_scene = [
                l for l in sous_titres_abs
                if l.start_ms < scene.end_ms and l.end_ms > scene.start_ms
            ]
            if dans_scene:
                subs_path = project.segments / f"{scene.index:04d}.ass"
                largeur, hauteur = (DRAFT_WIDTH, DRAFT_HEIGHT) if opts.draft else (WIDTH, HEIGHT)
                remap = subtitles.remap_for_render(
                    dans_scene, [(scene.start_ms, scene.end_ms, video_time_ms)], SPEED
                )
                subs_path.write_text(
                    subtitles.to_ass(remap, largeur, hauteur, decalage_ms=video_time_ms),
                    encoding="utf-8",
                )

        part = project.segments / f"{scene.index:04d}.mp4"
        video_paths.append(part)
        jobs.append((image, part, frames_scene, scene.seed, opts.draft, subs_path))
        audio_parts.append(project.segments / f"audio-{scene.index:04d}.wav")
        video_time_ms += round(frames_scene * 1000 / FPS)

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(_render_one, jobs))

    failures = [(p, e) for p, e in results if e]
    if failures:
        for path, error in failures[:3]:
            print(f"    {path.name}: {error}")
        raise RuntimeError(f"{len(failures)} segments en échec")

    # Sous-titres globaux (.srt) : une seule passe sur la liste ABSOLUE
    # complète, via les mêmes `breakpoints` que ceux qui ont servi aux .ass
    # incrustés juste au-dessus — même source, elles ne peuvent pas diverger
    # l'une de l'autre comme c'était le cas avant ce correctif (20/09).
    if sous_titres_abs:
        srt_lignes = subtitles.remap_for_render(sous_titres_abs, breakpoints, SPEED)
        srt_path = project.out.with_suffix(".srt")
        write_atomic(srt_path, subtitles.to_srt(srt_lignes))
        print(f"  {len(srt_lignes)} sous-titres -> {srt_path.name}")

    # Le concat demuxer veut des chemins relatifs au fichier de liste.
    list_file = project.segments / "concat.txt"
    list_file.write_text(
        "".join(f"file '{p.name}'\n" for p in video_paths),
        encoding="utf-8",
    )

    out_path = project.out_sample if opts.sample else project.out

    if selected is None and not cards:
        # Scènes contiguës depuis 0, aucun carton à insérer : l'audio complet
        # correspond déjà exactement à la vidéo, une seule tranche suffit.
        audio_for_mux = project.audio
    else:
        # Scènes disjointes (§5.1, mode --sample) et/ou cartons parlés à
        # insérer : l'audio doit être découpé (et les annonces glissées dedans
        # au bon endroit) puis recollé dans le même ordre que la vidéo, sinon
        # le mux désynchronise narration et image.
        for scene in todo:
            part = project.segments / f"audio-{scene.index:04d}.wav"
            if not part.exists():
                subprocess.run(
                    audio_slice_command(project.audio, scene.start_ms, scene.end_ms, part),
                    check=True,
                )
        audio_list = project.segments / "concat_audio.txt"
        audio_list.write_text(
            "".join(f"file '{p.name}'\n" for p in audio_parts), encoding="utf-8"
        )
        audio_name = "sample_audio.wav" if opts.sample else "spliced_audio.wav"
        audio_for_mux = project.segments / audio_name
        subprocess.run(concat_audio_command(audio_list, audio_for_mux), check=True)

    subprocess.run(concat_command(list_file, audio_for_mux, out_path), check=True)
    # ffmpeg rend la main avant que Windows ait vidé son cache d'écriture.
    # Vu le 30/09 (escobart) : écran bleu une minute après un rendu réussi,
    # out.mp4 retrouvé à la bonne taille mais rempli de zéros, sans atome
    # moov — et render.json disait le montage à jour. Forcer l'écriture sur
    # disque ici, avant de déclarer l'étape terminée.
    with open(out_path, "rb+") as f:
        os.fsync(f.fileno())

    video_ms = duration_ms(out_path)
    card_count = sum(len(v) for v in cards.values())
    # `video_time_ms` porte déjà la somme exacte des cartons ET des scènes
    # (accumulée pendant la boucle ci-dessus) : la seule source de vérité,
    # pas la peine de la recalculer séparément à partir des durées de scène.
    reference_ms = video_time_ms
    if card_count:
        print(f"  {card_count} carton(s) de palier, {round(video_time_ms - sum(round(s.duration_ms / SPEED) for s in todo))}ms d'annonces insérées")

    # Purge des segments : en --sample on les garde, l'itération est le but.
    if not opts.sample:
        shutil.rmtree(project.segments, ignore_errors=True)

    # Trace de la vitesse employée : sans elle, changer PIPELINE_SPEED ne
    # rendrait pas out.mp4 périmé (les images n'ont pas bougé), et l'étape
    # serait sautée — on croirait tester une nouvelle vitesse en regardant
    # l'ancienne vidéo. Voir _render_up_to_date dans __main__.py.
    write_atomic(
        project.dir / "render.json",
        json.dumps({"speed": SPEED, "video_ms": video_ms}, ensure_ascii=False, indent=1),
    )

    drift = video_ms - reference_ms
    print(
        f"  {len(todo)} segments -> {out_path.name} - "
        f"vidéo {video_ms / 1000:.1f} s, audio {reference_ms / 1000:.1f} s"
    )
    print(f"  dérive vidéo/audio : {drift:+d} ms")
