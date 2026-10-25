"""Génère un lot de variantes d'un même prompt via ComfyUI — appelé en
sous-processus par server.py, une ligne JSON imprimée par image terminée
pour que le serveur streame la progression sans attendre le lot entier.

    python _gen_thumbnail.py <prompt> <dossier_sortie> <nombre> <seed_base>

Un seul process pour tout le lot (pas un par image) : ComfyProvider uploade
les références de personnage une fois à l'ouverture — les réuploader à
chaque image serait un aller-retour perdu à chaque fois pour rien.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.providers.local import ComfyProvider  # noqa: E402
from pipeline.style import load as load_style  # noqa: E402


def main() -> None:
    prompt = sys.argv[1]
    out_dir = Path(sys.argv[2])
    count = int(sys.argv[3])
    seed_base = int(sys.argv[4])

    out_dir.mkdir(parents=True, exist_ok=True)
    style = load_style(draft=False)
    provider = ComfyProvider()

    for i in range(count):
        try:
            data = provider.generate(prompt, style, seed=seed_base + i)
            path = out_dir / f"{i:03d}.png"
            path.write_bytes(data)
            print(json.dumps({"type": "image", "index": i, "file": path.name}), flush=True)
        except Exception as exc:  # noqa: BLE001 — une image ratée ne doit pas tuer le lot
            print(json.dumps({"type": "error", "index": i, "message": str(exc)[:300]}), flush=True)

    print(json.dumps({"type": "done"}), flush=True)


if __name__ == "__main__":
    main()
