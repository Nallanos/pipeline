"""CLI (SPEC §5).

    python -m pipeline new <id> --script script.txt
    python -m pipeline run <id> [--step X] [--force X] [--draft] [--sample]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import script_check
from .project import PROJECTS_DIR, Options, Project, clean_source, normalize

# Chargé ici, pas seulement par webui/server.py : sans ça, un `pipeline run`
# lancé depuis un terminal nu ne voit ni FISH_API_KEY ni FISH_VOICE_ID tant
# qu'on ne les a pas exportés à la main, alors que le webui les a toujours vus
# via son propre load_dotenv() — deux comportements différents pour la même
# commande selon d'où elle part. Vu en pratique (20/09) : l'annonce du carton
# de palier retombait sur Piper faute de credentials Fish Audio chargés.
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True)
class Step:
    """Une étape du pipeline (SPEC §1).

    `output` désigne l'artefact produit : s'il existe et que `--force` n'a pas
    été passé, l'étape ne fait rien. C'est tout le mécanisme de reprise.

    `done` raffine ce critère pour les étapes dont l'artefact peut exister
    tout en étant incomplet ou périmé — la présence du fichier ne dit alors
    pas la vérité. Absent, le comportement est exactement l'ancien.
    """

    name: str
    output: Callable[[Project], Path]
    run: Callable[[Project, Options], None]
    done: Callable[[Project, Options], bool] | None = None


def _render_up_to_date(project: Project, opts: Options) -> bool:
    del opts  # le rendu ne dépend pas du mode d'échantillonnage
    images_file = project.dir / "images.json"
    if not project.out.exists():
        return False

    # Vitesse : elle ne laisse aucune trace dans les images, donc sans cette
    # comparaison un changement de PIPELINE_SPEED serait invisible ici et
    # l'étape sautée — on croirait tester une nouvelle vitesse en regardant
    # l'ancien montage. `render.json` est écrit par steps/render.py.
    from .ffmpeg import SPEED

    marker = project.dir / "render.json"
    if marker.exists():
        try:
            rendu = json.loads(marker.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if abs(float(rendu.get("speed", 1.0)) - SPEED) > 1e-9:
            print(f"[render] vitesse changee: {rendu.get('speed')} -> {SPEED}, montage a refaire")
            return False

    if not images_file.exists():
        return True  # rien pour dater le montage : on garde l'existant
    return project.out.stat().st_mtime >= images_file.stat().st_mtime


def _steps() -> list[Step]:
    # Import différé : charger faster-whisper ou Pillow au démarrage du CLI
    # rendrait `--help` lent pour rien.
    from .steps import align, images, location_refs, prompts, refs, render, segment, tts

    return [
        Step("tts", lambda p: p.audio, tts.run),
        Step("align", lambda p: p.alignment, align.run),
        Step("segment", lambda p: p.scenes, segment.run),
        # Même raison que pour `images` : le fichier peut exister en ne
        # couvrant qu'une partie des scènes (lot abandonné).
        Step("prompts", lambda p: p.dir / "prompts.json", prompts.run, prompts.is_complete),
        # Présence de l'index insuffisante là aussi : il peut pointer vers
        # un PNG disparu ou illisible, que l'étape `images` transformerait
        # en échec sur toutes les scènes du palier (voir `refs.is_complete`).
        Step("refs", lambda p: p.dir / "section_refs.json", refs.run, refs.is_complete),
        Step(
            "location_refs",
            lambda p: p.dir / "location_refs.json",
            location_refs.run,
            location_refs.is_complete,
        ),
        # `images.json` peut exister en ne couvrant qu'une partie des scènes
        # (étape interrompue) ou en gardant des replis à retenter : c'est
        # `images.is_complete` qui tranche, pas la présence du fichier.
        Step("images", lambda p: p.dir / "images.json", images.run, images.is_complete),
        # Une vidéo plus ancienne que les images qu'elle monte est périmée —
        # sinon un run qui complète les images laisse en place l'out.mp4 de
        # la tentative précédente, et l'UI affiche l'ancienne vidéo.
        Step("render", lambda p: p.out, render.run, _render_up_to_date),
    ]


def cmd_new(args: argparse.Namespace) -> int:
    project = Project(args.id)
    if project.dir.exists():
        print(f"projet '{args.id}' existe déjà : {project.dir}", file=sys.stderr)
        return 1
    src = Path(args.script)
    if not src.is_file():
        print(f"script introuvable : {src}", file=sys.stderr)
        return 1
    from .language import default_code, load

    try:
        language = load(args.lang or default_code())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not language.voice_path.exists():
        print(
            f"voix Piper absente : {language.voice_path.name}\n"
            f"  uv run python -m piper.download_voices {language.voice} "
            f"--data-dir models/piper",
            file=sys.stderr,
        )
        return 1

    cleaned, stripped, levels = clean_source(src.read_text(encoding="utf-8"))
    project.dir.mkdir(parents=True)
    project.script.write_text(cleaned, encoding="utf-8")
    (project.dir / "language.txt").write_text(language.code, encoding="utf-8")
    if levels:
        import json

        project.levels.write_text(
            json.dumps(
                {"levels": [{"word_index": w, "label": label} for w, label in levels]},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    chars = len(normalize(cleaned))
    print(f"projet '{args.id}' créé — {chars} caractères, langue '{language.code}'")
    script_check.report(cleaned, levels)
    if stripped:
        print(f"  {stripped} horodatages ôtés du script importé")
    if levels:
        print(f"  {len(levels)} paliers détectés (NIVEAU:)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    project = Project(args.id)
    if not project.script.is_file():
        print(f"projet '{args.id}' introuvable ou sans script.txt", file=sys.stderr)
        return 1

    # Relu à chaque run, pas seulement à l'import : le script peut avoir été
    # édité entre-temps, et un défaut de forme coûte des heures de génération
    # avant d'être visible à l'écran.
    levels_raw: list[tuple[int, str]] = []
    if project.levels.exists():
        levels_raw = [
            (entry["word_index"], entry["label"])
            for entry in json.loads(project.levels.read_text(encoding="utf-8"))["levels"]
        ]
    script_check.report(project.script.read_text(encoding="utf-8"), levels_raw)

    steps = _steps()
    names = [s.name for s in steps]

    if args.step and args.step not in names:
        print(f"étape inconnue : {args.step} (parmi {', '.join(names)})", file=sys.stderr)
        return 1
    if args.force and args.force not in names:
        print(f"étape inconnue : {args.force} (parmi {', '.join(names)})", file=sys.stderr)
        return 1

    # --force <étape> invalide à partir de cette étape : tout ce qui en dépend
    # est nécessairement périmé, on ne peut pas régénérer la segmentation en
    # gardant les images de l'ancienne.
    forced_from = names.index(args.force) if args.force else len(names)
    selected = {args.step} if args.step else set(names)

    opts = Options(draft=args.draft, sample=args.sample)
    timings: list[tuple[str, float]] = []

    for i, step in enumerate(steps):
        if step.name not in selected:
            continue
        force = i >= forced_from or (args.step == step.name and args.force is not None)
        out = step.output(project)
        try:
            already = step.done(project, opts) if step.done else out.exists()
        except Exception:  # noqa: BLE001 — un prédicat qui rate = étape à refaire
            already = False
        if already and not force:
            print(f"[{step.name}] déjà fait — ignoré")
            continue
        print(f"[{step.name}] …")
        start = time.monotonic()
        step.run(project, Options(force=force, draft=opts.draft, sample=opts.sample))
        elapsed = time.monotonic() - start
        timings.append((step.name, elapsed))
        print(f"[{step.name}] {elapsed:.1f} s")

    if timings:
        total = sum(t for _, t in timings)
        print("\n— run —")
        for name, t in timings:
            print(f"  {name:<8} {t:>7.1f} s")
        print(f"  {'total':<8} {total:>7.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="créer un projet")
    p_new.add_argument("id")
    p_new.add_argument("--script", required=True)
    p_new.add_argument("--lang", help="langue du script (voir config/languages.yaml)")
    p_new.set_defaults(func=cmd_new)

    p_run = sub.add_parser("run", help="exécuter les étapes manquantes")
    p_run.add_argument("id")
    p_run.add_argument("--step", help="n'exécuter que cette étape")
    p_run.add_argument("--force", help="invalider à partir de cette étape")
    p_run.add_argument("--draft", action="store_true", help="basse résolution, rendu rapide")
    p_run.add_argument("--sample", action="store_true", help="3 fenêtres de 30 s (SPEC §5.1)")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    PROJECTS_DIR.mkdir(exist_ok=True)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
