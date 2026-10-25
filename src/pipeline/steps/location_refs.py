"""Étape — Références de lieu. `location_bible.json` → `location_refs.json`
(voir plan « Cohérence des décors — une référence par lieu récurrent »).

Contrairement au costume (constant sur tout un palier, une seule référence
suffit — voir `steps/refs.py`), le décor change *à l'intérieur* d'un même
palier. Cette étape génère donc une image de référence par LIEU récurrent
détecté (`location_bible.py`), pas par palier : un plan d'établissement SANS
personnage, sur le seul texte de sa description — validé en test manuel
(09/09) qu'aucune ancre d'identité n'est nécessaire pour un décor propre,
sans fuite de visage.

`steps/images.py` (et `webui/_gen_review.py`) font ensuite tourner ces
références en plus de la référence de costume, chaînées dans cet ordre
[costume, décor] (voir `providers/local.py::ComfyProvider.generate`).

Sans `levels.json`/`location_bible.json`, ou sans aucun lieu détecté : artefact
vide, rien ne change pour `images.py`.
"""

from __future__ import annotations

import json
import os
import random

from ..project import Options, Project, write_atomic
from ..providers.base import ImageProvider
from ..style import load as load_style

PROVIDER = os.environ.get("PIPELINE_PROVIDER", "stub")


def _provider() -> ImageProvider:
    # Même fabrique que steps/images.py et steps/refs.py, dupliquée à dessein
    # (SPEC project.py : pas d'abstraction au-delà du contrat d'étape).
    if PROVIDER == "comfy":
        from ..providers.local import REFS_MODEL, REFS_URL, ComfyProvider

        # Moteur des références, qui peut différer de celui des scènes :
        # avec COMFY_MODEL=hidream, les références restent à Klein en local
        # (voir REFS_MODEL dans providers/local.py).
        print(f"  [location_refs] références générées par {REFS_MODEL} sur {REFS_URL}")
        return ComfyProvider(model=REFS_MODEL, url=REFS_URL)
    from ..providers.stub import StubProvider

    return StubProvider()


def _era_guardrail(project: Project) -> str:
    """Relit `prompts.json::era_guardrail` (calculé par `steps/prompts.py`).

    Sans ça, vérifié sur un run réel (10/09) : « Noble's Manor » (whitewashed
    brick mansion, sycamore garden) rendait une maison coloniale américaine
    avec piscine, et « Royal Palace » rendait des colonnes néoclassiques avec
    du texte de façade halluciné — précisément la dérive que ce mécanisme
    corrige déjà pour le texte des prompts de scène
    (`_batch_prompt`/`image_prompt.txt`), mais qui n'était jamais lu ici :
    ce module compose son sujet directement, sans passer par le lot LLM qui
    injecte `{era_guardrail}`."""
    prompts_file = project.dir / "prompts.json"
    if not prompts_file.exists():
        return ""
    return json.loads(prompts_file.read_text(encoding="utf-8")).get("era_guardrail", "")



def _is_readable(path: Path) -> bool:
    """Ce PNG est-il réellement décodable ?

    `path.exists()` ne suffit pas comme pas de reprise : `write_bytes` n'est
    pas atomique et NTFS journalise la taille avant les données, donc une
    coupure brutale laisse un fichier de la bonne taille rempli de zéros.
    Réutilisé tel quel, il fait échouer côté ComfyUI TOUTES les scènes du
    palier, run après run, sans que rien ne le régénère. Vu en pratique
    (22/09, japon) : deux références à 100 % de zéros, 73 scènes en bouchons.
    Illisible = à refaire, jamais à réutiliser.
    """
    try:
        from PIL import Image

        with Image.open(path) as probe:
            probe.verify()
        return True
    except Exception:  # noqa: BLE001 — PNG vide, tronqué, illisible...
        return False



def is_complete(project: Project, opts: Options) -> bool:
    """L'étape est-elle RÉELLEMENT terminée ?

    « `location_refs.json` existe » ne suffit pas : l'index peut pointer vers un PNG
    disparu, ou — le cas vu le 22/09 sur `japon` — vers un PNG de la bonne
    taille entièrement rempli de zéros (`write_bytes` non atomique + coupure
    brutale). L'étape était alors sautée run après run, `_is_readable` dans
    `run()` n'avait jamais l'occasion de s'exécuter, et l'étape `images`
    échouait sur TOUTES les scènes du palier concerné. Même raison d'être que
    `prompts.is_complete` / `images.is_complete` : un artefact présent mais
    incomplet doit rejouer l'étape, pas la déclarer faite.
    """
    del opts  # ces références ne dépendent pas du mode d'échantillonnage
    out_file = project.dir / "location_refs.json"
    if not out_file.exists():
        return False
    try:
        refs = json.loads(out_file.read_text(encoding="utf-8"))["refs"]
    except Exception:  # noqa: BLE001 — artefact illisible = étape à refaire
        return False
    return all(_is_readable(project.dir / path) for path in refs.values())


def run(project: Project, opts: Options) -> None:
    out_file = project.dir / "location_refs.json"

    bible_file = project.location_bible
    if not project.levels.exists() or not bible_file.exists():
        write_atomic(out_file, json.dumps({"refs": {}}, ensure_ascii=False, indent=1))
        return

    locations = json.loads(bible_file.read_text(encoding="utf-8"))["locations"]
    if not locations:
        write_atomic(out_file, json.dumps({"refs": {}}, ensure_ascii=False, indent=1))
        return

    if PROVIDER == "comfy" and os.environ.get("COMFY_MODEL", "sdxl") not in ("flux2", "qwen", "hidream"):
        # Même garde-fou que steps/refs.py : le graphe SDXL ignore
        # totalement les références, générer quand même produirait des
        # images inutiles en silence.
        print("  [location_refs] COMFY_MODEL sans références (ni flux2, ni qwen, ni hidream) — les références de lieu n'ont aucun effet, ignoré")
        write_atomic(out_file, json.dumps({"refs": {}}, ensure_ascii=False, indent=1))
        return

    style = load_style(draft=opts.draft)
    try:
        provider = _provider()
    except Exception as exc:  # noqa: BLE001 — ComfyUI injoignable ne doit pas tuer le run
        # Même bug que steps/refs.py, vu en pratique (14/09) : ComfyUI down
        # pile à l'instant de la construction du provider faisait planter
        # tout `pipeline run` — voir le commentaire équivalent dans refs.py.
        print(f"  [location_refs] ComfyUI injoignable — étape abandonnée sans rien écraser ({exc})")
        return
    era_guardrail = _era_guardrail(project)

    project.location_refs.mkdir(parents=True, exist_ok=True)

    refs_index: dict[str, str] = {}
    for location in locations:
        index = location["index"]
        path = project.location_refs / f"{index:03d}.png"
        if path.exists() and not opts.force:
            if _is_readable(path):
                refs_index[str(index)] = str(path.relative_to(project.dir))
                continue
            print(f"  [location_refs] {path.name} illisible (écriture interrompue) — régénéré")

        description = location.get("description", "")
        if not description:
            print(f"  [location_refs] lieu {location.get('name', index)!r} sans description — ignoré")
            continue

        # Aucun personnage : un plan d'établissement, pas un portrait — d'où
        # aucune référence d'identité ici (contrairement à steps/refs.py).
        # « architectural photography » retiré : ça tirait vers des clichés
        # d'immeuble de bureaux modernes (enseignes, texte de façade) plutôt
        # que l'illustration plate voulue par la style bible.
        subject = (
            f"empty establishing shot of {description}, no people, no "
            "characters, no vehicles, no signage, no readable text"
        )
        if era_guardrail:
            subject += f". {era_guardrail.replace(chr(10), '. ')}"
        # Aléatoire à chaque génération (pas seed_for), même raison que
        # steps/refs.py : `--force location_refs` doit donner un vrai
        # nouveau tirage.
        seed = random.randint(0, 2**31 - 1)
        try:
            data = provider.generate(style.full_prompt(subject), style, seed=seed)
        except Exception as exc:  # noqa: BLE001 — un lieu raté ne doit pas bloquer les autres
            print(f"  [location_refs] lieu {location.get('name', index)!r} abandonné — {exc}")
            continue

        write_atomic(path, data)
        refs_index[str(index)] = str(path.relative_to(project.dir))

    if locations and not refs_index:
        # Même protection que steps/refs.py : un échec sur tous les lieux ne
        # doit pas effacer l'index d'un run antérieur réussi.
        print(f"  [location_refs] {len(locations)} lieux tous en échec — out_file laissé tel quel")
        return

    write_atomic(out_file, json.dumps({"refs": refs_index}, ensure_ascii=False, indent=1))
    print(f"  {len(refs_index)}/{len(locations)} références de lieu")
