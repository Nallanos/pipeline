"""Génère `variants` alternatives du prompt (édité ou non) de chaque scène
marquée pour régénération dans la page de revue — appelé en sous-processus
par server.py, une ligne JSON imprimée par image terminée.

    python _gen_review.py <jobs.json> <dossier_sortie> <variants> <seed_base> <project_id>

<jobs.json> : [{"index": <scène>, "prompt": "<texte, possiblement édité>"}, ...]
— passé par fichier plutôt qu'en argv : un prompt édité à la main est du texte
libre (guillemets, retours à la ligne), pas sûr à faire traverser un argv.

Contrairement à `_gen_thumbnail.py` (exploration libre, prompt brut), le
prompt passe ici par `style.full_prompt()` avant `provider.generate()`, comme
`steps/images.py` — une variante candidate à remplacer une image du projet
doit porter le même medium/palette/cadrage/lumière que le reste de la vidéo,
sinon le remplacement choisi jurerait visuellement avec ses voisines.

Même raison pour la référence de personnage et pour celle de décor : sans
`project_id`, une régénération depuis la revue retomberait sur
`COMFY_CHARACTER_REFS` (le repli global) au lieu de la référence de palier/lieu
de la scène, et réintroduirait exactement les bugs de costume et de décor que
la revue sert justement à corriger. Résolues via `section_bible.scene_refs()`
et `location_bible.scene_refs()`, partagées avec `steps/images.py` pour que
les deux choisissent toujours les mêmes références pour une scène donnée.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline import location_bible, section_bible  # noqa: E402
from pipeline.project import Project  # noqa: E402
from pipeline.providers.local import ComfyProvider  # noqa: E402
from pipeline.scenes import load as load_scenes  # noqa: E402
from pipeline.style import load as load_style  # noqa: E402


def main() -> None:
    jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_dir = Path(sys.argv[2])
    variants = int(sys.argv[3])
    seed_base = int(sys.argv[4])
    project = Project(sys.argv[5])

    style = load_style(draft=False)
    provider = ComfyProvider()
    scenes = load_scenes(project.scenes)
    section_ref = section_bible.scene_refs(project, scenes)
    decor_ref = location_bible.scene_refs(project, scenes)

    for job in jobs:
        index = int(job["index"])
        prompt = str(job["prompt"])
        job_dir = out_dir / f"{index:04d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        for v in range(variants):
            # Écarté à dessein de seed_for(project_id, index) (scenes.py) : une
            # variante candidate ne doit jamais reproduire pixel pour pixel
            # l'image déjà en place.
            seed = seed_base + index * variants + v
            try:
                reference = section_ref.get(index)
                decor_reference = decor_ref.get(index)
                data = provider.generate(
                    style.full_prompt(prompt), style, seed=seed,
                    reference=reference, decor_reference=decor_reference,
                )
                path = job_dir / f"{v:03d}.png"
                path.write_bytes(data)
                print(
                    json.dumps(
                        {"type": "image", "scene_index": index, "variant": v, "file": f"{index:04d}/{v:03d}.png"}
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — une variante ratée ne doit pas tuer le lot
                print(
                    json.dumps(
                        {"type": "error", "scene_index": index, "variant": v, "message": str(exc)[:300]}
                    ),
                    flush=True,
                )

    print(json.dumps({"type": "done"}), flush=True)


if __name__ == "__main__":
    main()
