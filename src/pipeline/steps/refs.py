"""Étape — Références de palier. `section_bible.json` → `section_refs.json`
(voir plan « Fix costume : une référence de personnage par palier »).

Une vidéo à paliers a un costume différent par section (Esclave, Paysan,
Artisan, ... Pharaon), mais `providers/local.py` n'utilisait jusqu'ici qu'une
seule référence de personnage fixe pour tout le run (`COMFY_CHARACTER_REFS`).
FLUX.2 encode l'image de référence entière — vêtements et décor compris, pas
seulement le visage (`ReferenceLatent`, voir `providers/local.py`) — donc une
référence qui ne correspond pas au palier en cours contamine chaque image de
son costume, quoi que dise le prompt texte.

Cette étape génère une image de référence par palier : `bldguy` dans le
costume de la section (`section_bible.json::costume`), sur fond neutre,
ancrée sur l'identité via une image de tête seule (voir `_identity_ref()`).
`steps/images.py` fait ensuite tourner ces références selon le palier de
chaque scène.

Sans `levels.json` (script sans marqueur `NIVEAU:`) ou sans `section_bible.json`
(l'étape `prompts` ne l'écrit que si des paliers existent) : artefact vide,
rien ne change pour `images.py`, qui retombe sur `COMFY_CHARACTER_REFS`.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from ..project import ROOT, Options, Project, write_atomic
from ..providers.base import ImageProvider
from ..style import load as load_style

PROVIDER = os.environ.get("PIPELINE_PROVIDER", "stub")

# Tête seule de bldguy, sans costume ni décor — voir plan §0. Recadrée à la
# main depuis characters/refs/protagonist_adult_close.png : les deux
# références historiques mélangeaient un costume (cowboy) au visage, ce qui
# aurait reproduit le même bug sur les 8 références de palier.
_DEFAULT_IDENTITY_REF = ROOT / "characters" / "refs" / "bldguy_head.png"


def _provider() -> ImageProvider:
    # Même fabrique que steps/images.py, dupliquée à dessein plutôt
    # qu'importée ou partagée via une abstraction commune (SPEC project.py :
    # pas d'abstraction au-delà du contrat d'étape).
    if PROVIDER == "comfy":
        from ..providers.local import REFS_MODEL, REFS_URL, ComfyProvider

        # Moteur des références, qui peut différer de celui des scènes :
        # avec COMFY_MODEL=hidream, les références restent à Klein en local
        # (voir REFS_MODEL dans providers/local.py).
        print(f"  [refs] références générées par {REFS_MODEL} sur {REFS_URL}")
        return ComfyProvider(model=REFS_MODEL, url=REFS_URL)
    from ..providers.stub import StubProvider

    return StubProvider()


def _identity_ref() -> Path:
    override = os.environ.get("PIPELINE_IDENTITY_REF")
    return Path(override) if override else _DEFAULT_IDENTITY_REF



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

    « `section_refs.json` existe » ne suffit pas : l'index peut pointer vers un PNG
    disparu, ou — le cas vu le 22/09 sur `japon` — vers un PNG de la bonne
    taille entièrement rempli de zéros (`write_bytes` non atomique + coupure
    brutale). L'étape était alors sautée run après run, `_is_readable` dans
    `run()` n'avait jamais l'occasion de s'exécuter, et l'étape `images`
    échouait sur TOUTES les scènes du palier concerné. Même raison d'être que
    `prompts.is_complete` / `images.is_complete` : un artefact présent mais
    incomplet doit rejouer l'étape, pas la déclarer faite.
    """
    del opts  # ces références ne dépendent pas du mode d'échantillonnage
    out_file = project.dir / "section_refs.json"
    if not out_file.exists():
        return False
    try:
        refs = json.loads(out_file.read_text(encoding="utf-8"))["refs"]
    except Exception:  # noqa: BLE001 — artefact illisible = étape à refaire
        return False
    return all(_is_readable(project.dir / path) for path in refs.values())


def run(project: Project, opts: Options) -> None:
    out_file = project.dir / "section_refs.json"

    bible_file = project.section_bible
    if not project.levels.exists() or not bible_file.exists():
        write_atomic(out_file, json.dumps({"refs": {}}, ensure_ascii=False, indent=1))
        return

    if PROVIDER == "comfy" and os.environ.get("COMFY_MODEL", "sdxl") not in ("flux2", "qwen", "hidream"):
        # Le graphe SDXL (providers/local.py::_workflow_sdxl) ignore
        # totalement `reference` : générer quand même produirait 8 images
        # inutiles en silence — la classe d'échec que ce dépôt évite partout
        # ailleurs (voir ex. le garde-fou Chatterbox Turbo anglais-seul).
        print("  [refs] COMFY_MODEL sans références (ni flux2, ni qwen, ni hidream) — les références de palier n'ont aucun effet, ignoré")
        write_atomic(out_file, json.dumps({"refs": {}}, ensure_ascii=False, indent=1))
        return

    sections = json.loads(bible_file.read_text(encoding="utf-8"))["sections"]
    style = load_style(draft=opts.draft)
    try:
        provider = _provider()
    except Exception as exc:  # noqa: BLE001 — ComfyUI injoignable ne doit pas tuer le run
        # Vu en pratique (14/09) : ComfyProvider.__init__ tente d'uploader la
        # référence d'identité tout de suite — si ComfyUI est down pile à cet
        # instant, ça levait une exception non rattrapée qui faisait planter
        # tout `pipeline run` (refs, images, render compris), à l'exact
        # opposé de la règle absolue du dépôt (« jamais d'exception qui tue
        # le run », voir steps/images.py). On laisse `out_file` tel quel
        # (ne pas écraser un résultat précédent) plutôt que d'écrire un
        # `{"refs": {}}` qui effacerait un run antérieur réussi.
        print(f"  [refs] ComfyUI injoignable — étape abandonnée sans rien écraser ({exc})")
        return
    identity_ref = _identity_ref()

    project.section_refs.mkdir(parents=True, exist_ok=True)

    refs_index: dict[str, str] = {}
    for section in sections:
        index = section["index"]
        path = project.section_refs / f"{index:03d}.png"
        if path.exists() and not opts.force:
            if _is_readable(path):
                refs_index[str(index)] = str(path.relative_to(project.dir))
                continue
            print(f"  [refs] {path.name} illisible (écriture interrompue) — régénéré")

        costume = section.get("costume", "")
        if not costume:
            # Descriptif rejeté à l'étape prompts (réponse non exploitable,
            # voir section_bible.py::_looks_like_refusal) — pas de costume à
            # illustrer, cette section retombe sur la référence globale.
            print(f"  [refs] palier {section['label']!r} sans costume — ignoré")
            continue

        # "bald, shaved head" explicite : sans ça, vérifié sur un run réel
        # (09/09) le crâne rasé — pourtant l'identité même de `bldguy`,
        # portée par l'ancre de tête — se perdait sur les 8 références,
        # remplacé par des cheveux. `image_prompt.txt` n'a jamais ce problème
        # parce qu'il ne décrit jamais le visage/la tête (juste le mot
        # `bldguy`) ; ici le sujet est un portrait corps entier, donc il faut
        # le dire.
        subject = (
            f"bldguy, bald shaved head, {costume}, standing, full body, "
            "front view, plain flat neutral background"
        )
        # Aléatoire à chaque génération (pas seed_for) : `--force refs` doit
        # donner un vrai nouveau tirage, pas reproduire la même image.
        seed = random.randint(0, 2**31 - 1)
        try:
            data = provider.generate(style.full_prompt(subject), style, seed=seed, reference=identity_ref)
        except Exception as exc:  # noqa: BLE001 — un palier raté ne doit pas bloquer les autres
            print(f"  [refs] palier {section['label']!r} abandonné — {exc}")
            continue

        write_atomic(path, data)
        refs_index[str(index)] = str(path.relative_to(project.dir))

    if sections and not refs_index:
        # Tous les paliers ont échoué : écrire l'index vide effacerait un run
        # antérieur réussi alors même que les PNG sont encore sur le disque —
        # exactement ce que le garde-fou « ComfyUI injoignable » ci-dessus
        # cherche à éviter, mais il ne couvrait que l'exception à la
        # construction du provider, pas l'échec de chaque palier pris un par
        # un. Vu en pratique (16/09, video5) : section_refs.json réduit à
        # {"refs": {}} avec ses 10 images intactes à côté.
        print(f"  [refs] {len(sections)} paliers tous en échec — out_file laissé tel quel")
        return

    write_atomic(out_file, json.dumps({"refs": refs_index}, ensure_ascii=False, indent=1))
    print(f"  {len(refs_index)}/{len(sections)} références de palier")
