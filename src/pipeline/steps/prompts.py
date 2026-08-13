"""Étape 4 — Prompts image. `scenes.json` → `prompts.json` (SPEC §4).

Appels LLM par lots de ~20 scènes, en parallèle : un seul appel pour 300 scènes
dérive en qualité et en format. Chaque lot reçoit la style bible, un résumé
global du script (cohérence narrative) et les 2 scènes précédentes (continuité
locale).
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from ..project import PROMPTS_DIR, Options, Project, write_atomic
from ..scenes import Scene, load, sample_indices
from ..style import StyleConfig
from ..style import load as load_style

MODEL = "claude-sonnet-5"
BATCH_SIZE = 20
MAX_WORKERS = 8
RETRIES = 2

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _client() -> object:
    from anthropic import Anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY non défini. Le reste du pipeline reste utilisable : "
            "`run <id> --step images` puis `--step render` génèrent la vidéo en "
            "prenant le texte de la scène comme prompt (error_code=no_prompt)."
        )
    return Anthropic()


def _ask(client: object, prompt: str, max_tokens: int) -> str:
    message = client.messages.create(  # type: ignore[attr-defined]
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _parse(raw: str) -> dict[int, str]:
    """Parsing défensif : le modèle encadre parfois le JSON de prose ou de ```."""
    match = _JSON_OBJECT.search(raw)
    if not match:
        raise ValueError(f"aucun objet JSON dans la réponse : {raw[:200]!r}")
    data = json.loads(match.group(0))
    return {int(item["index"]): str(item["prompt"]).strip() for item in data["prompts"]}


def _summarize(client: object, script: str) -> str:
    excerpt = script if len(script) < 12_000 else f"{script[:6000]}\n[…]\n{script[-6000:]}"
    return _ask(
        client,
        "Résume ce script de vidéo en un paragraphe de 80 mots maximum : sujet, "
        "ton, et fil narratif. Ce résumé servira à garder une cohérence visuelle "
        "entre 300 illustrations. Réponds par le seul paragraphe.\n\n" + excerpt,
        max_tokens=400,
    )


def _batch_prompt(
    style: StyleConfig,
    summary: str,
    previous: list[Scene],
    batch: list[Scene],
) -> str:
    template = (PROMPTS_DIR / "image_prompt.txt").read_text(encoding="utf-8")
    return template.format(
        style_bible=style.bible(),
        summary=summary,
        previous="\n".join(f"- {s.text}" for s in previous) or "(début du script)",
        scenes="\n".join(f"{s.index}. {s.text}" for s in batch),
    )


def run(project: Project, opts: Options) -> None:
    scenes = load(project.scenes)
    style = load_style(draft=opts.draft)

    selected = set(sample_indices(scenes, scenes[-1].end_ms)) if opts.sample else None
    todo = [s for s in scenes if selected is None or s.index in selected]

    client = _client()
    summary = _summarize(client, project.script_normalized.read_text(encoding="utf-8"))

    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

    def handle(batch: list[Scene]) -> dict[int, str]:
        # Les 2 scènes précédentes viennent de la liste complète, pas du lot :
        # en --sample les scènes ne sont pas contiguës.
        first = batch[0].index
        previous = [s for s in scenes if first - 2 <= s.index < first]
        prompt = _batch_prompt(style, summary, previous, batch)
        for attempt in range(RETRIES + 1):
            try:
                parsed = _parse(_ask(client, prompt, max_tokens=200 * len(batch) + 500))
                missing = {s.index for s in batch} - parsed.keys()
                if missing and attempt < RETRIES:
                    raise ValueError(f"index manquants : {sorted(missing)}")
                return parsed
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                if attempt == RETRIES:
                    print(f"  lot {first}: abandon après {RETRIES + 1} essais — {exc}")
                    return {}
        return {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(handle, batches))

    prompts: dict[int, str] = {}
    for result in results:
        prompts.update(result)

    write_atomic(
        project.dir / "prompts.json",
        json.dumps(
            {
                "model": MODEL,
                "summary": summary,
                "prompts": {str(k): v for k, v in prompts.items()},
            },
            ensure_ascii=False,
            indent=1,
        ),
    )

    failed = len(todo) - len(prompts)
    suffix = f" — {failed} manquants" if failed else ""
    print(f"  {len(prompts)} prompts en {len(batches)} lots{suffix}")
