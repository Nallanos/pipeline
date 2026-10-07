"""Contre-épreuves du banc Qwen, sur quelques scènes déjà passées par
scripts/qwen_bench.py (lit qwen_test/results.json).

    VARIANT=nolightning COMFY_QWEN_LORA= COMFY_QWEN_STEPS=40 COMFY_QWEN_CFG=4 \
        uv run python scripts/qwen_variants.py video5:171 escobart:93
    VARIANT=style uv run python scripts/qwen_variants.py escobart:150 ...

`style` remplace les champs de style.yaml qui brident la composition
(« children's picture book », « single subject, simple uncluttered
composition », « minimal shading ») par une consigne plus cinématographique,
pour séparer ce qui tient au modèle de ce qui tient à la style bible.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PIPELINE_PROVIDER", "comfy")
os.environ["COMFY_MODEL"] = "qwen"
os.environ.setdefault("COMFY_AUTOSTART", "0")

sys.path.insert(0, str(Path(__file__).parent))

from qwen_bench import OUT, server_seconds  # noqa: E402

from pipeline.providers.local import ComfyProvider  # noqa: E402
from pipeline.style import load as load_style  # noqa: E402

VARIANT = os.environ["VARIANT"]

STYLE_OVERRIDE = {
    "medium": (
        "detailed 2D cartoon illustration in the style of a modern animated series, "
        "clean bold black outlines, cel shading with soft shadows, rich detailed background"
    ),
    "framing": "cinematic composition, dynamic camera angle, characters interacting naturally with the scene",
    "lighting": "atmospheric lighting with clear light direction",
}


def main() -> None:
    style = load_style()
    if VARIANT == "style":
        style = dataclasses.replace(style, **STYLE_OVERRIDE)
    wanted = {tuple(arg.split(":")) for arg in sys.argv[1:]}
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    jobs = [r for r in results if (r["project"], str(r["scene"])) in wanted]

    provider = ComfyProvider()
    log_file = OUT / "variants.json"
    log = json.loads(log_file.read_text(encoding="utf-8")) if log_file.exists() else []
    for job in jobs:
        out = OUT / VARIANT / job["project"] / f"{job['scene']:04d}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        data = provider.generate(
            style.full_prompt(job["subject"]), style, job["seed"],
            reference=Path(job["costume"]),
            decor_reference=Path(job["decor"]) if job["decor"] else None,
        )
        wall = round(time.perf_counter() - started, 2)
        server = server_seconds(provider, provider.last_prompt_id)
        out.write_bytes(data)
        log.append({"variant": VARIANT, "project": job["project"], "scene": job["scene"],
                    "image": str(out), "wall_s": wall, "server_s": server})
        print(f"{VARIANT} {job['project']} {job['scene']} wall {wall} s / serveur {server} s", flush=True)
        log_file.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
