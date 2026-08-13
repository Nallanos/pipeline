"""Étape 5 — Génération des images. `scenes.json` → `images/NNNN.png` (SPEC §4).

Règle absolue : **jamais d'exception qui tue le run**. Une image qui échoue
stocke son `error_code`, reçoit un repli, et le run continue. Sur 300 images,
abandonner à la 287e parce qu'une seule a été refusée est le pire comportement
possible.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

from ..project import Options, Project, write_atomic
from ..providers.base import ContentRefused, ImageProvider
from ..scenes import Scene, load, sample_indices
from ..style import StyleConfig
from ..style import load as load_style

PROVIDER = os.environ.get("PIPELINE_PROVIDER", "stub")


def _provider() -> ImageProvider:
    if PROVIDER == "comfy":
        from ..providers.local import ComfyProvider

        return ComfyProvider()
    from ..providers.stub import StubProvider

    return StubProvider()


def _fallback(scene: Scene, style: StyleConfig) -> bytes:
    """Image de repli : le run continue, le trou est visible au montage."""
    from ..providers.stub import StubProvider

    return StubProvider().generate(f"[repli] {scene.text}", style, scene.seed)


def run(project: Project, opts: Options) -> None:
    scenes = load(project.scenes)
    style = load_style(draft=opts.draft)
    provider = _provider()

    prompts_file = project.dir / "prompts.json"
    prompts: dict[str, str] = {}
    if prompts_file.exists():
        prompts = json.loads(prompts_file.read_text(encoding="utf-8"))["prompts"]

    selected = set(sample_indices(scenes, scenes[-1].end_ms)) if opts.sample else None
    todo = [s for s in scenes if selected is None or s.index in selected]

    project.images.mkdir(parents=True, exist_ok=True)

    def generate(scene: Scene) -> tuple[int, str, str | None]:
        path = project.images / f"{scene.index:04d}.png"
        # Reprise : une image déjà sur le disque n'est pas régénérée.
        if path.exists() and not opts.force:
            return scene.index, str(path.relative_to(project.dir)), None

        subject = prompts.get(str(scene.index))
        error: str | None = None
        if not subject:
            subject, error = scene.text, "no_prompt"

        try:
            data = provider.generate(style.full_prompt(subject), style, scene.seed)
        except ContentRefused:
            error = "content_refused"
            data = _fallback(scene, style)
        except Exception as exc:  # noqa: BLE001 — aucune exception ne tue le run
            error = f"provider_error:{type(exc).__name__}"
            data = _fallback(scene, style)

        path.write_bytes(data)
        return scene.index, str(path.relative_to(project.dir)), error

    # Le provider local sérialise sur le GPU ; le bouchon n'a pas de contention.
    workers = 1 if PROVIDER == "comfy" else min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(generate, todo))

    index = {
        str(i): {"image_path": path, "error_code": error} for i, path, error in results
    }
    write_atomic(
        project.dir / "images.json",
        json.dumps({"provider": PROVIDER, "images": index}, ensure_ascii=False, indent=1),
    )

    errors = [(i, e) for i, _, e in results if e]
    suffix = f" — {len(errors)} en erreur" if errors else ""
    print(f"  {len(results)} images via '{PROVIDER}'{suffix}")
    for i, error in errors[:5]:
        print(f"    scène {i}: {error}")
