"""Étape 5 — Génération des images. `scenes.json` → `images/NNNN.png` (SPEC §4).

Règle absolue : **jamais d'exception qui tue le run**. Une image qui échoue
stocke son `error_code`, reçoit un repli, et le run continue. Sur 300 images,
abandonner à la 287e parce qu'une seule a été refusée est le pire comportement
possible.

Le repli s'écrit sur le même chemin `NNNN.png` que l'image réelle — pas de
suffixe distinct, pour que `render.py` n'ait rien à connaître de la
distinction. La reprise (§ ci-dessous) s'appuie donc sur `images.json`, pas
sur la seule présence du fichier, pour savoir quelles images sont de vrais
replis à retenter."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import location_bible, section_bible
from ..project import Options, Project, write_atomic
from ..providers.base import ContentRefused, ImageProvider
from ..scenes import Scene, load, sample_indices
from ..style import StyleConfig
from ..style import load as load_style

PROVIDER = os.environ.get("PIPELINE_PROVIDER", "stub")

# Nombre d'échecs provider CONSÉCUTIFS au-delà duquel l'étape s'arrête au lieu
# de continuer à écrire des bouchons. La règle absolue de ce fichier protège
# contre l'abandon sur un échec isolé ; elle n'a jamais voulu dire « continuer
# 300 fois contre un serveur mort ». Vu en pratique (15/09, video5) : ComfyUI
# tombé à l'image 39, les 304 suivantes écrites en bouchons en 10 minutes,
# aucun signal — et le rendu final est parti dessus.
CONSECUTIVE_FAILURE_LIMIT = int(os.environ.get("PIPELINE_IMAGES_FAILURE_LIMIT", "5"))


def _provider() -> ImageProvider:
    if PROVIDER == "comfy":
        from ..providers.local import ComfyProvider

        return ComfyProvider()
    from ..providers.stub import StubProvider

    return StubProvider()


def _fallback(scene: Scene, style: StyleConfig) -> bytes:
    """Image de repli : le run continue, le trou est visible au montage."""
    from ..providers.stub import StubProvider

    return StubProvider(fallback=True).generate(f"[repli] {scene.text}", style, scene.seed)


def _a_regenerer(
    project: Project,
    scene_index: int,
    previous_errors: dict[str, str | None],
    force: bool,
    prompts: dict[str, str] | None = None,
) -> bool:
    """Cette scène doit-elle (re)passer par le provider ?

    Seule source de vérité : `generate()` l'applique pour décider, et `run()`
    la rejoue pour annoncer le volume réel de travail. Deux copies de cette
    logique divergeraient au premier changement, et le journal annoncerait
    alors autre chose que ce qui se passe.
    """
    path = project.images / f"{scene_index:04d}.png"
    if force or not path.exists():
        return True
    # Absente de images.json : origine inconnue (scène laissée de côté par le
    # coupe-circuit, index effacé...) — on ne parie pas, on régénère.
    if str(scene_index) not in previous_errors:
        return True
    error = previous_errors.get(str(scene_index)) or ""
    if error == "content_refused" or error.startswith("provider_error:"):
        return True
    # `no_prompt` : l'image a bien été produite par le provider, mais à partir
    # du texte français brut de la scène faute de prompt — donc hors sujet. On
    # la refait dès qu'un prompt existe enfin pour elle (typiquement après une
    # reprise de l'étape prompts qui a rattrapé un lot abandonné).
    if error == "no_prompt" and prompts is not None and prompts.get(str(scene_index), "").strip():
        return True
    return _is_fallback(path)


def _is_fallback(path: Path) -> bool:
    """Ce PNG est-il un repli (marqueur posé par StubProvider) ?

    Lu dans le fichier plutôt que déduit d'`images.json` : une scène
    interrompue par le coupe-circuit n'a aucune entrée dans cet index, et son
    bouchon serait sinon indiscernable d'une image réussie. Pillow lit les
    chunks tEXt à l'ouverture, sans décoder l'image — négligeable face aux
    ~64 s que coûte une génération.

    Un fichier illisible est traité comme un repli : le régénérer est la
    réaction sûre, le garder ne l'est pas."""
    from ..providers.stub import FALLBACK_MARKER

    try:
        from PIL import Image

        with Image.open(path) as image:
            return FALLBACK_MARKER in image.info
    except Exception:  # noqa: BLE001 — PNG tronqué, illisible, vide...
        return True


def is_complete(project: Project, opts: Options) -> bool:
    """L'étape est-elle RÉELLEMENT terminée ?

    « `images.json` existe » ne suffit pas comme critère de reprise : une
    étape interrompue par le coupe-circuit en écrit un qui ne couvre qu'une
    partie des scènes, et une scène peut y figurer avec un repli à retenter.
    Sans ce prédicat, `pipeline run` affichait « images — déjà fait, ignoré »
    et aucune relance ne rattrapait jamais les images manquantes : il fallait
    `--force images`, qui régénère les 343 au lieu des 54 restantes. Vu en
    pratique (16/09, video5) : run interrompu à la scène 289 sur 343.
    """
    images_file = project.dir / "images.json"
    if not images_file.exists() or not project.scenes.exists():
        return False
    try:
        index = json.loads(images_file.read_text(encoding="utf-8"))["images"]
        scenes = load(project.scenes)
        prompts_file = project.dir / "prompts.json"
        prompts_ecrits: dict[str, str] = (
            json.loads(prompts_file.read_text(encoding="utf-8"))["prompts"]
            if prompts_file.exists()
            else {}
        )
    except Exception:  # noqa: BLE001 — artefact illisible = étape à refaire
        return False

    # En --sample seules quelques fenêtres sont générées : exiger les 343
    # scènes rendrait l'étape éternellement incomplète.
    wanted = set(sample_indices(scenes, scenes[-1].end_ms)) if opts.sample else {s.index for s in scenes}

    for scene_index in wanted:
        entry = index.get(str(scene_index))
        if entry is None:
            return False
        error = entry.get("error_code")
        if error == "content_refused" or (error and str(error).startswith("provider_error:")):
            return False
        if error == "no_prompt" and prompts_ecrits.get(str(scene_index), "").strip():
            return False  # un prompt existe désormais : l'image est à refaire
    return True


def run(project: Project, opts: Options) -> None:
    scenes = load(project.scenes)
    style = load_style(draft=opts.draft)
    try:
        provider: ImageProvider = _provider()
    except Exception as exc:  # noqa: BLE001 — seul cas où l'étape s'arrête, voir ci-dessous
        # Exception à la LIMITE de la règle absolue de ce fichier, assumée :
        # cette règle protège contre « abandonner à la 287e image parce
        # qu'une seule a été refusée ». Un provider injoignable dès le
        # démarrage n'est pas cet accident-là — c'est la certitude que les
        # 343 scènes vont toutes échouer. L'ancien comportement
        # (`provider = None`, repli par scène) écrivait 343 bouchons en
        # 2 secondes puis laissait `render` en faire un out.mp4 de 39 Mo :
        # un run entier consommé pour une vidéo inutilisable, alors que la
        # cause tient en une ligne. Vu en pratique le 16/09 sur video5,
        # ComfyUI simplement pas lancé.
        raise RuntimeError(
            f"provider '{PROVIDER}' injoignable au démarrage de l'étape images : {exc}\n"
            "  Rien n'a été généré, aucune image écrasée.\n"
            "  Le démarrage automatique a déjà été tenté (COMFY_AUTOSTART) : "
            "vérifier COMFY_DIR, COMFY_URL, ou lancer ComfyUI à la main."
        ) from exc
    section_ref = section_bible.scene_refs(project, scenes)
    decor_ref = location_bible.scene_refs(project, scenes)

    prompts_file = project.dir / "prompts.json"
    prompts: dict[str, str] = {}
    if prompts_file.exists():
        prompts = json.loads(prompts_file.read_text(encoding="utf-8"))["prompts"]

    # error_code du run précédent, lu AVANT que ce run écrase images.json.
    # Sans ça, la reprise (`path.exists()`) traitait un repli déjà sur le
    # disque comme une image faite — vu en pratique (15/09, video5) : 304
    # images sur 343 étaient des replis (~8 Ko, ComfyUI tombé en cours de
    # run), et le run suivant les a toutes recomptées "faites" plutôt que de
    # les retenter, tout en réécrivant images.json avec error_code=null
    # partout — la trace de l'échec disparaissait avec lui.
    images_file = project.dir / "images.json"
    previous_errors: dict[str, str | None] = {}
    if images_file.exists():
        previous_errors = {
            k: v.get("error_code") for k, v in json.loads(images_file.read_text(encoding="utf-8"))["images"].items()
        }

    selected = set(sample_indices(scenes, scenes[-1].end_ms)) if opts.sample else None
    todo = [s for s in scenes if selected is None or s.index in selected]

    project.images.mkdir(parents=True, exist_ok=True)

    # Listes plutôt que des int : `generate` est une closure, et une liste
    # mutable évite un `nonlocal` par compteur. `failure_lock` les protège —
    # le pool a plusieurs workers dès que le provider n'est pas comfy.
    consecutive: list[int] = []
    aborted: list[bool] = []
    failure_lock = threading.Lock()
    # Progression lisible par le webui (estimation du temps restant) : une
    # ligne `[progress] images k/n` par image réellement générée. Le webui ne
    # l'affiche pas dans le journal, il s'en sert pour mesurer le débit.
    progress: list[int] = []
    progress_total: list[int] = [0]

    def generate(scene: Scene) -> tuple[int, str, str | None]:
        path = project.images / f"{scene.index:04d}.png"
        # Reprise : une image déjà sur le disque n'est régénérée que si le
        # run précédent y a écrit un repli (_fallback) — `content_refused` ou
        # `provider_error:*`. `no_prompt` n'en fait PAS partie : cette
        # scène a reçu une vraie image du provider (texte brut de la scène
        # comme sujet, pas le bouchon StubProvider), pas un placeholder — la
        # retenter à chaque run coûterait du GPU sans rien changer tant que
        # `prompts.json` ne comble pas l'entrée manquante.
        previous_error = previous_errors.get(str(scene.index))
        if not _a_regenerer(project, scene.index, previous_errors, opts.force, prompts):
            return scene.index, str(path.relative_to(project.dir)), previous_error

        subject = prompts.get(str(scene.index))
        error: str | None = None
        if not subject:
            subject, error = scene.text, "no_prompt"

        # Provider déjà déclaré mort par une rafale d'échecs : ne rien
        # tenter, ne RIEN écrire. Pas de bouchon sur le disque => `path`
        # reste absent => le prochain run reprend naturellement cette scène.
        if aborted:
            return scene.index, "", "aborted"

        try:
            reference = section_ref.get(scene.index)
            decor_reference = decor_ref.get(scene.index)
            data = provider.generate(
                style.full_prompt(subject), style, scene.seed,
                reference=reference, decor_reference=decor_reference,
            )
        except ContentRefused:
            # Un refus de contenu est un verdict sur CETTE scène, pas un
            # symptôme de serveur mort : il ne compte pas dans la rafale.
            error = "content_refused"
            data = _fallback(scene, style)
        except Exception as exc:  # noqa: BLE001 — une image ratée ne tue pas le run
            error = f"provider_error:{type(exc).__name__}"
            data = _fallback(scene, style)
            with failure_lock:
                consecutive.append(scene.index)
                if len(consecutive) >= CONSECUTIVE_FAILURE_LIMIT:
                    aborted.append(True)
                    print(
                        f"  {len(consecutive)} échecs provider consécutifs "
                        f"(scènes {consecutive[0]}-{scene.index}) — étape interrompue"
                    )
        else:
            with failure_lock:
                consecutive.clear()

        # Atomique : `write_bytes` laisse, sur coupure brutale, un PNG de la
        # bonne taille rempli de zéros que `_is_fallback` (Image.open) juge
        # illisible donc à refaire — mais le pas de reprise des étapes refs
        # n'avait pas cette chance (voir steps/refs.py::_is_readable).
        write_atomic(path, data)
        with failure_lock:
            progress.append(scene.index)
            print(f"[progress] images {len(progress)}/{progress_total[0]}", flush=True)
        return scene.index, str(path.relative_to(project.dir)), error

    # Annonce de ce qui va réellement être fait : sans ça le journal affiche
    # « [images] … » aussi bien pour 343 images que pour une seule, et un run
    # qui ne vient compléter qu'une scène manquante ressemble à s'y méprendre
    # à une régénération complète de plusieurs heures.
    a_faire = [s.index for s in todo if _a_regenerer(project, s.index, previous_errors, opts.force, prompts)]
    if len(a_faire) < len(todo):
        apercu = ", ".join(str(i) for i in a_faire[:8]) + ("…" if len(a_faire) > 8 else "")
        print(f"  {len(a_faire)}/{len(todo)} images à (re)générer{' : ' + apercu if a_faire else ''}")
    else:
        print(f"  {len(todo)} images à générer")
    progress_total[0] = len(a_faire)

    # Le provider local sérialise sur le GPU ; le bouchon n'a pas de contention.
    workers = 1 if PROVIDER == "comfy" else min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(generate, todo))

    # Les scènes "aborted" n'ont rien sur le disque : les inscrire avec un
    # image_path vide donnerait à render/webui un chemin qui ne mène nulle
    # part. Absentes de l'index, elles seront simplement reprises au prochain
    # run, comme des scènes jamais traitées.
    index = {
        str(i): {"image_path": path, "error_code": error}
        for i, path, error in results
        if error != "aborted"
    }
    write_atomic(
        project.dir / "images.json",
        json.dumps({"provider": PROVIDER, "images": index}, ensure_ascii=False, indent=1),
    )

    errors = [(i, e) for i, _, e in results if e and e != "aborted"]
    skipped = [i for i, _, e in results if e == "aborted"]
    suffix = f" — {len(errors)} en erreur" if errors else ""
    print(f"  {len(index)} images via '{PROVIDER}'{suffix}")
    for i, error in errors[:5]:
        print(f"    scène {i}: {error}")

    if skipped:
        # images.json est déjà écrit : la trace de ce qui a été fait survit.
        # L'exception empêche `render` de fabriquer une vidéo à partir d'un
        # lot d'images majoritairement absentes ou bouchonnées.
        raise RuntimeError(
            f"provider '{PROVIDER}' tombé en cours d'étape — "
            f"{len(skipped)} scènes non traitées (à partir de la {skipped[0]}).\n"
            "  Les images déjà produites sont conservées ; relancer reprendra "
            "où ça s'est arrêté.\n"
            "  ComfyUI a répondu au démarrage puis s'est tu : vérifier sa "
            "fenêtre (VRAM saturée, modèle absent, process tué)."
        )
