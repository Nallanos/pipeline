"""Banc d'essai Klein vs Qwen-Image-Edit sur des scènes réelles.

Prend N scènes réparties sur chaque projet (prompt écrit, image Klein déjà
produite sans erreur, référence de costume de palier), les régénère via
ComfyProvider avec COMFY_MODEL=qwen sur un ComfyUI distant, et consigne le
temps de chaque image dans qwen_test/results.json.

    COMFY_URL=https://<pod>-8188.proxy.runpod.net uv run python scripts/qwen_bench.py

La première image sert de chauffe et est exclue des temps : à froid ComfyUI
charge ~30 Go de poids, ce qui n'a rien à voir avec le régime établi.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PIPELINE_PROVIDER", "comfy")
os.environ["COMFY_MODEL"] = "qwen"
os.environ.setdefault("COMFY_AUTOSTART", "0")

from pipeline import location_bible, section_bible  # noqa: E402
from pipeline.project import ROOT, Project  # noqa: E402
from pipeline.providers.local import ComfyProvider  # noqa: E402
from pipeline.scenes import load  # noqa: E402
from pipeline.style import load as load_style  # noqa: E402

OUT = ROOT / "qwen_test"
PROJECTS = sys.argv[1:] or ["video5", "escobart"]
PER_PROJECT = int(os.environ.get("BENCH_PER_PROJECT", "10"))


def pick(project: Project) -> list[dict]:
    scenes = load(project.scenes)
    prompts = json.loads((project.dir / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    images = json.loads((project.dir / "images.json").read_text(encoding="utf-8"))["images"]
    costume = section_bible.scene_refs(project, scenes)
    decor = location_bible.scene_refs(project, scenes)

    candidates = [
        s for s in scenes
        if prompts.get(str(s.index), "").strip()
        and images.get(str(s.index), {}).get("error_code") is None
        and s.index in costume
    ]
    # Moitié avec référence de décor, moitié sans : seule une scène sur trois
    # en a une, et un tirage uniforme sous-testerait le cas à deux références.
    def spread(pool: list, count: int) -> list:
        step = len(pool) / count
        return [pool[int(i * step + step / 2)] for i in range(count)]

    with_decor = [s for s in candidates if s.index in decor]
    half = min(PER_PROJECT // 2, len(with_decor))
    chosen = spread(with_decor, half) + spread(
        [s for s in candidates if s.index not in decor], PER_PROJECT - half
    )
    chosen.sort(key=lambda s: s.index)
    return [
        {
            "project": project.id,
            "scene": s.index,
            "seed": s.seed,
            "subject": prompts[str(s.index)],
            "klein": str(project.images / f"{s.index:04d}.png"),
            "costume": str(costume[s.index]),
            "decor": str(decor[s.index]) if s.index in decor else None,
        }
        for s in chosen
    ]


def server_seconds(provider: ComfyProvider, prompt_id: str) -> float | None:
    """Durée d'exécution vue par ComfyUI (hors upload et allers-retours)."""
    entry = provider._client.get(f"/history/{prompt_id}").json().get(prompt_id, {})
    stamps = {kind: data.get("timestamp") for kind, data in entry.get("status", {}).get("messages", [])}
    start, end = stamps.get("execution_start"), stamps.get("execution_success")
    return (end - start) / 1000 if start and end else None


def main() -> None:
    style = load_style()
    jobs = [job for pid in PROJECTS for job in pick(Project(pid))]
    provider = ComfyProvider()
    results: list[dict] = []
    results_file = OUT / "results.json"

    for n, job in enumerate(jobs):
        out = OUT / job["project"] / f"{job['scene']:04d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            data = provider.generate(
                style.full_prompt(job["subject"]), style, job["seed"],
                reference=Path(job["costume"]),
                decor_reference=Path(job["decor"]) if job["decor"] else None,
            )
        except Exception as exc:  # noqa: BLE001 — on consigne et on continue
            job["error"] = str(exc)[:400]
            print(f"[{n + 1}/{len(jobs)}] {job['project']} {job['scene']} ÉCHEC {exc}", flush=True)
            results.append(job)
            continue
        job["wall_s"] = round(time.perf_counter() - started, 2)
        job["server_s"] = server_seconds(provider, provider.last_prompt_id)
        job["warmup"] = n == 0
        job["qwen"] = str(out)
        out.write_bytes(data)
        results.append(job)
        print(
            f"[{n + 1}/{len(jobs)}] {job['project']} {job['scene']} "
            f"wall {job['wall_s']} s / serveur {job['server_s']} s",
            flush=True,
        )
        results_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    timed = [r for r in results if "wall_s" in r and not r["warmup"]]
    if timed:
        walls = sorted(r["wall_s"] for r in timed)
        servers = sorted(r["server_s"] for r in timed if r["server_s"])
        print(f"régime établi sur {len(timed)} images — wall médian {walls[len(walls) // 2]} s, "
              f"serveur médian {servers[len(servers) // 2] if servers else '?'} s")


if __name__ == "__main__":
    main()
