"""CLI (SPEC §5).

    python -m pipeline new <id> --script script.txt
    python -m pipeline run <id> [--step X] [--force X] [--draft] [--sample]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .project import PROJECTS_DIR, Options, Project


@dataclass(frozen=True)
class Step:
    """Une étape du pipeline (SPEC §1).

    `output` désigne l'artefact produit : s'il existe et que `--force` n'a pas
    été passé, l'étape ne fait rien. C'est tout le mécanisme de reprise.
    """

    name: str
    output: Callable[[Project], Path]
    run: Callable[[Project, Options], None]


def _steps() -> list[Step]:
    # Import différé : charger faster-whisper ou Pillow au démarrage du CLI
    # rendrait `--help` lent pour rien.
    from .steps import align, images, prompts, render, segment, tts

    return [
        Step("tts", lambda p: p.audio, tts.run),
        Step("align", lambda p: p.alignment, align.run),
        Step("segment", lambda p: p.scenes, segment.run),
        Step("prompts", lambda p: p.dir / "prompts.json", prompts.run),
        Step("images", lambda p: p.dir / "images.json", images.run),
        Step("render", lambda p: p.out, render.run),
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
    project.dir.mkdir(parents=True)
    shutil.copyfile(src, project.script)
    chars = len(project.script.read_text(encoding="utf-8"))
    print(f"projet '{args.id}' créé — {chars} caractères → {project.dir}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    project = Project(args.id)
    if not project.script.is_file():
        print(f"projet '{args.id}' introuvable ou sans script.txt", file=sys.stderr)
        return 1

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
        if out.exists() and not force:
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
