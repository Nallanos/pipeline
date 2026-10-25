"""Provider local — SDXL ou FLUX.2 via l'API HTTP de ComfyUI.

Le GPU (RX 9060 XT, RDNA4) n'est visible que depuis Windows : WSL n'expose que
`/dev/dxg`, ni `/dev/kfd` ni `/dev/dri`. ComfyUI tourne donc côté Windows et on
lui parle en HTTP. Ce n'est pas un pis-aller : toute la douleur ROCm reste hors
du dépôt, il n'y a aucune dépendance PyTorch ici, et déplacer ComfyUI sur une
autre machine ne coûte qu'une variable d'environnement.

    PIPELINE_PROVIDER=comfy
    COMFY_URL=http://<ip-hôte-windows>:8188

ComfyUI doit écouter sur 0.0.0.0 (`--listen`) et le port être ouvert au
pare-feu Windows. En mode réseau « mirrored » de WSL, http://127.0.0.1:8188
suffit.

Le GPU est sérialisé par la queue de ComfyUI elle-même : l'étape images n'ouvre
qu'un seul worker pour ce provider, il n'y a donc rien à ordonnancer ici.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from io import BytesIO
from pathlib import Path

import httpx

from ..style import StyleConfig

URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
CHECKPOINT = os.environ.get("COMFY_CHECKPOINT", "sd_xl_base_1.0.safetensors")
TIMEOUT_S = float(os.environ.get("COMFY_TIMEOUT", "300"))
POLL_S = 0.5

# LoRA de personnage — désactivé tant que COMFY_CHARACTER_LORA n'est pas
# renseigné, donc inerte par défaut : ce brouillon a été écrit pendant
# l'entraînement de bldguy-lora-v2 (voir lora-training/), avant validation du
# résultat. Ne pas activer en production avant d'avoir confirmé la sortie sur
# smith/chef/lighthouse (voir character-consistency-report.md).
#   COMFY_CHARACTER_LORA=bldguy_lora.safetensors
#   COMFY_CHARACTER_LORA_STRENGTH=1.0   (0.6-1.2 testé ; 1.0 est le point retenu)
#   COMFY_CHARACTER_LORA_TRIGGER=bldguy
CHARACTER_LORA = os.environ.get("COMFY_CHARACTER_LORA", "")
CHARACTER_LORA_STRENGTH = float(os.environ.get("COMFY_CHARACTER_LORA_STRENGTH", "1.0"))
CHARACTER_LORA_TRIGGER = os.environ.get("COMFY_CHARACTER_LORA_TRIGGER", "bldguy")

# Moteur d'images : "sdxl" (défaut, comportement historique) ou "flux2".
#
# FLUX.2 Klein 4B a été retenu après comparaison sur les prompts réels du
# pipeline (voir character-consistency-report.md) : SDXL perdait les scènes à
# deux personnages et les relations spatiales — « emerges onto the iron
# gallery » devenait des rochers, « his small son » disparaissait. C'est de
# l'adhérence au prompt, que ni un LoRA ni un réglage de sampler ne corrigent.
#
# Klein tient en outre la cohérence de personnage via des images de référence
# (ReferenceLatent), sans entraînement : deux références ont fait mieux que le
# LoRA entraîné sur 104 images, notamment sur les âges (enfance, vieillesse)
# que le LoRA ne savait pas rendre.
#
#   COMFY_MODEL=flux2
#   COMFY_CHARACTER_REFS=C:\...\protagonist_full_bald_sq.png,C:\...\head_sq.png
IMAGE_MODEL = os.environ.get("COMFY_MODEL", "sdxl")
FLUX2_UNET = os.environ.get("COMFY_FLUX2_UNET", "flux-2-klein-4b.safetensors")
# Encodeur texte en fp4 par défaut, PAS la version fp16 de 7,7 Go : les deux
# modèles ne tiennent pas ensemble dans les ~11,5 Go utilisables de la carte
# (7,7 encodeur + 7,4 UNet = 15,1), et ComfyUI passe alors son temps à les
# charger/décharger — « Unloaded partially » trois fois par image dans ses
# logs. Mesuré le 16/09 sur un vrai prompt du projet, mêmes références, même
# seed : fp16 erratique entre 21 et 91 s l'image (selon ce qui est résident à
# l'instant T), fp4 stable à 19 s. Sortie visuellement identique au même seed
# — c'est l'encodeur qu'on quantifie, et l'adhérence au prompt ne bouge pas.
# Repasser à "qwen_3_4b.safetensors" si un doute apparaît sur un prompt
# complexe.
FLUX2_CLIP = os.environ.get("COMFY_FLUX2_CLIP", "qwen_3_4b_fp4_flux2.safetensors")
FLUX2_VAE = os.environ.get("COMFY_FLUX2_VAE", "flux2-vae.safetensors")
# FLUX.2 est distillé : cfg 1.0 / euler-simple, rien à voir avec les 50 pas et
# cfg 6.5 de style.yaml, qui restent justes pour SDXL.
#
# 4 pas, et la note d'origine (« 8 pas ne donnent pas de gain visible sur 4 »)
# était encore optimiste : testé le 17/09 sur un gros plan, même seed, 6 ET
# 8 pas font DÉRIVER L'IDENTITÉ — bldguy ressort avec des cheveux (bruns à 6,
# noirs avec moustache à 8) là où 4 pas le garde chauve. Ne pas augmenter
# cette valeur en croyant gagner en qualité : c'est la cohérence du
# personnage qu'on perd.
# Plus de pas = plus de liberté pour s'éloigner du conditionnement de
# référence, et sur un gros plan où le visage occupe tout le cadre, cette
# dérive se paie en identité. Le surcoût en temps n'est plus le problème
# (17 -> 26 s l'image) ; le risque sur la cohérence du personnage l'est.
# COMFY_FLUX2_STEPS permet d'essayer autre chose ponctuellement.
FLUX2_STEPS = int(os.environ.get("COMFY_FLUX2_STEPS", "4"))
FLUX2_CFG = float(os.environ.get("COMFY_FLUX2_CFG", "1.0"))
# Images de référence du protagoniste, séparées par des virgules. Uploadées
# une fois à l'ouverture du provider ; ComfyUI met en cache leur encodage VAE
# tant que le graphe ne change pas, donc le coût est payé une seule fois.
# Facteur de réduction appliqué aux seules références de DÉCOR avant upload.
#
# Chaque ReferenceLatent ajoute ses tokens au conditioning et le coût de
# l'attention est quadratique : une référence en 1344x768 pèse ~4032 tokens,
# autant que l'image générée elle-même. Avec costume + décor pleine taille le
# conditioning atteint ~12 000 tokens, dépasse le seuil mémoire de la carte, et
# ComfyUI se met à décharger les modèles en plein sampling.
#
# Mesuré le 17/09 sur video5, 6 prompts différents alternés, GPU libre :
#   décor 1344x768 : 59 / 58 / 22 s  (instable — le seuil est franchi ou non)
#   décor  448x256 : 17 / 17 / 17 s  (stable, = le temps d'une scène sans décor)
#
# UNIQUEMENT le décor : réduire la référence de COSTUME fait perdre l'identité
# du personnage — vérifié à l'œil le 17/09, bldguy y perd son crâne rasé et
# ressort avec des cheveux dès la demi-taille. Le décor, lui, ne porte ni
# visage ni identité, seulement des matériaux et une ambiance : il supporte la
# réduction sans que rien de visible ne bouge.
#
# COMFY_DECOR_REF_SCALE=1.0 désactive le mécanisme.
#
# Qwen-Image-Edit n'est pas concerné : TextEncodeQwenImageEditPlus ramène
# lui-même chaque référence à ~1 Mpx avant de l'encoder, et le seuil mémoire
# en cause est celui de la carte 16 Go. Le décor y passe donc à pleine taille.
def _decor_ref_scale(model: str) -> float:
    return float(os.environ.get("COMFY_DECOR_REF_SCALE", "1.0" if model in ("qwen", "hidream") else "0.34"))


DECOR_REF_SCALE = _decor_ref_scale(IMAGE_MODEL)

# Qwen-Image-Edit-2511 (Apache 2.0) — COMFY_MODEL=qwen. Testé sur un pod
# Runpod (RTX 5090, 32 Go), pas sur la carte locale : 20 Go d'UNet fp8 +
# 9 Go d'encodeur ne tiennent pas dans 16 Go.
#
# Différence de fond avec Klein : les références ne passent pas par une
# chaîne ReferenceLatent muette, elles sont NOMMÉES dans le prompt (« image
# 1 », « image 2 »). Le graphe réécrit donc le prompt (voir _qwen_prompt).
#
# Lightning 4 pas + cfg 1.0 : le LoRA de distillation officiel de lightx2v
# pour 2511. COMFY_QWEN_LORA="" le désactive (il faut alors ~40 pas, cfg 4).
QWEN_UNET = os.environ.get("COMFY_QWEN_UNET", "qwen_image_edit_2511_fp8mixed.safetensors")
QWEN_CLIP = os.environ.get("COMFY_QWEN_CLIP", "qwen_2.5_vl_7b_fp8_scaled.safetensors")
QWEN_VAE = os.environ.get("COMFY_QWEN_VAE", "qwen_image_vae.safetensors")
QWEN_LORA = os.environ.get("COMFY_QWEN_LORA", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
QWEN_STEPS = int(os.environ.get("COMFY_QWEN_STEPS", "4"))
QWEN_CFG = float(os.environ.get("COMFY_QWEN_CFG", "1.0"))
QWEN_SHIFT = float(os.environ.get("COMFY_QWEN_SHIFT", "3.1"))
# HiDream-O1-Image Dev (MIT) — COMFY_MODEL=hidream. Retenu le 07/10 après le
# bake-off de 8 modèles (scripts/bakeoff.py) : poses les plus naturelles
# (s'agenouiller, marcher, s'effondrer sur un volant là où les autres restent
# debout face caméra) et décors de référence suivis. ~8 s l'image sur L40S.
#
# Un seul fichier de 8 Go (modèle, encodeur et VAE ensemble), d'où
# CheckpointLoaderSimple. Réglages du template officiel ComfyUI, ceux du
# bake-off : 28 pas, échantillonneur « flash » (SamplerLCM) avec bruit
# d'entraînement 7,6, cfg 1. Les références sont nommées dans le prompt
# comme pour Qwen (voir _qwen_prompt).
HIDREAM_CKPT = os.environ.get("COMFY_HIDREAM_CKPT", "hidream_o1_image_dev_fp8_scaled.safetensors")
HIDREAM_STEPS = int(os.environ.get("COMFY_HIDREAM_STEPS", "28"))
HIDREAM_NOISE_SCALE = float(os.environ.get("COMFY_HIDREAM_NOISE_SCALE", "7.6"))
# Les moteurs qui exploitent `reference` / `decor_reference`. Les étapes refs
# et location_refs ne génèrent rien pour les autres (le graphe SDXL les
# ignorerait en silence).
REFERENCE_MODELS = ("flux2", "qwen", "hidream")

# Moteur et ComfyUI des étapes de RÉFÉRENCES (steps/refs.py, costumes ;
# steps/location_refs.py, décors), distincts de ceux des scènes.
#
# Par défaut les mêmes, sauf avec hidream : ses propres références sont
# inutilisables (vérifié le 07/10 sur une copie de video5). La planche de
# costume perd le crâne rasé, le décor « no people » se remplit de monde,
# et surtout HiDream s'effondre en texture grise sur toute scène qui
# réutilise une image qu'il a lui-même produite (11 scènes sur 11). Les
# références restent donc à Klein, sur le ComfyUI local, et seules les scènes
# partent sur le GPU loué — exactement la configuration du bake-off.
#   COMFY_REFS_MODEL=flux2   COMFY_REFS_URL=http://127.0.0.1:8188
REFS_MODEL = os.environ.get("COMFY_REFS_MODEL") or ("flux2" if IMAGE_MODEL == "hidream" else IMAGE_MODEL)
REFS_URL = (
    os.environ.get("COMFY_REFS_URL") or ("http://127.0.0.1:8188" if IMAGE_MODEL == "hidream" else URL)
).rstrip("/")

# En dessous de cet écart-type de luminance (vignette 336x192), une image est
# une plage uniforme : le modèle a échoué sans que ComfyUI le signale. Les
# 2 016 images réelles des projets mesurées le 07/10 vont de 17 (nuit) à 80,
# médiane 50 ; les échecs HiDream de la même date sortaient entre 2 et 8.
FLAT_IMAGE_STD = float(os.environ.get("COMFY_FLAT_IMAGE_STD", "12"))

CHARACTER_REFS = [
    r.strip() for r in os.environ.get("COMFY_CHARACTER_REFS", "").split(",") if r.strip()
]


# Démarrage automatique de ComfyUI quand il ne répond pas (voir _boot).
#
# Motivation : `start.ps1` lance déjà ComfyUI, mais rien n'oblige à passer par
# lui — un `pipeline run` direct, un run programmé depuis le webui la nuit, ou
# simplement l'oubli, et l'étape images échoue alors que tout le reste est
# prêt. Vu en pratique deux fois (15/09 et 16/09) sur le même projet.
#
# Chemin d'installation par défaut aligné sur start.ps1 ; à changer là-bas
# aussi si l'installation bouge.
COMFY_DIR = os.environ.get("COMFY_DIR", r"C:\ComfyUI")
AUTOSTART = os.environ.get("COMFY_AUTOSTART", "1").strip().lower() not in ("0", "false", "no")
# ComfyUI répond sur /system_stats dès que son serveur HTTP écoute, bien avant
# d'avoir chargé le moindre modèle : ce délai couvre l'import de torch et
# l'énumération des noeuds, pas le chargement de FLUX.2 (qui se fait à la
# première image, dans le TIMEOUT_S de la génération).
BOOT_TIMEOUT_S = float(os.environ.get("COMFY_BOOT_TIMEOUT", "180"))
BOOT_POLL_S = 1.0

# Sérialise les tentatives de démarrage : `ComfyProvider` est construit une
# fois par étape GPU (refs, location_refs, images), et le webui peut en
# construire un autre en parallèle pour une régénération depuis la revue.
# Sans ce verrou, deux constructions simultanées lanceraient deux ComfyUI sur
# le même port, dont un qui meurt aussitôt sur « address already in use ».
_boot_lock = threading.Lock()


def _is_local(url: str) -> bool:
    """Un ComfyUI distant (COMFY_URL vers une autre machine) ne se lance pas
    d'ici : on ne peut que constater qu'il est absent."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "")


def _comfy_python() -> Path | None:
    """Interpréteur du venv de ComfyUI. Surtout pas `sys.executable` : c'est
    le venv du pipeline, qui n'a ni torch ni les dépendances de ComfyUI."""
    for relative in ("venv/Scripts/python.exe", "venv/bin/python", ".venv/Scripts/python.exe", ".venv/bin/python"):
        candidate = Path(COMFY_DIR) / relative
        if candidate.is_file():
            return candidate
    return None


def _ping(client: httpx.Client) -> bool:
    try:
        client.get("/system_stats", timeout=5.0).raise_for_status()
    except Exception:  # noqa: BLE001 — toute erreur = pas (encore) prêt
        return False
    return True


def _spawn() -> str | None:
    """Lance ComfyUI détaché, dans sa propre console. Renvoie None si OK,
    sinon la raison de l'abandon.

    Détaché volontairement : le serveur doit survivre à la fin du run, comme
    celui que lance `start.ps1` — l'étape suivante, un run ultérieur ou une
    régénération depuis le webui le retrouvent chaud, modèles déjà en VRAM.
    """
    if not Path(COMFY_DIR).is_dir():
        return f"dossier ComfyUI introuvable : {COMFY_DIR} (définir COMFY_DIR)"
    python = _comfy_python()
    if python is None:
        return f"venv de ComfyUI introuvable sous {COMFY_DIR}"

    # Sur Windows : sa propre fenêtre, comme start.ps1, pour que les logs de
    # ComfyUI restent lisibles au lieu de se mélanger à ceux du pipeline.
    # Ailleurs : simple détachement de session.
    if sys.platform == "win32":
        kwargs: dict[str, object] = {"creationflags": subprocess.CREATE_NEW_CONSOLE}
    else:
        kwargs = {"start_new_session": True, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}

    try:
        subprocess.Popen([str(python), "main.py"], cwd=COMFY_DIR, close_fds=True, **kwargs)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 — remonté comme raison d'abandon
        return f"lancement impossible : {exc}"
    return None


def _boot(client: httpx.Client, url: str) -> str | None:
    """Démarre ComfyUI et attend qu'il réponde. None si prêt, sinon la raison.

    Re-teste le port sous le verrou avant de lancer quoi que ce soit : entre
    le constat d'absence et l'obtention du verrou, une autre étape a pu
    démarrer le serveur, et il ne faut surtout pas en lancer un deuxième.
    """
    with _boot_lock:
        if _ping(client):
            return None
        if not _is_local(url):
            return f"ComfyUI est distant ({url}) — impossible de le démarrer d'ici"

        print(f"  ComfyUI ne répond pas — démarrage automatique depuis {COMFY_DIR}")
        failure = _spawn()
        if failure:
            return failure

        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if _ping(client):
                waited = BOOT_TIMEOUT_S - (deadline - time.monotonic())
                print(f"  ComfyUI prêt en {waited:.0f} s")
                return None
            time.sleep(BOOT_POLL_S)
        return f"démarré mais toujours muet après {BOOT_TIMEOUT_S:.0f} s"


def _workflow(
    prompt: str,
    style: StyleConfig,
    seed: int,
    refs: list[str] | None = None,
    model: str = IMAGE_MODEL,
) -> dict[str, dict[str, object]]:
    """Aiguille vers le graphe du moteur choisi (voir IMAGE_MODEL).

    Qwen et HiDream ne passent pas par ici : ils ont besoin de distinguer le
    costume du décor pour les nommer dans le prompt, voir
    ComfyProvider.generate."""
    if model == "flux2":
        return _workflow_flux2(prompt, style, seed, refs or [])
    return _workflow_sdxl(prompt, style, seed)


def _workflow_flux2(
    prompt: str,
    style: StyleConfig,
    seed: int,
    refs: list[str],
) -> dict[str, dict[str, object]]:
    """Graphe FLUX.2 : unet + clip + vae séparés, puis chaîne de références.

    FLUX.2 ne se charge pas comme SDXL : pas de checkpoint unique, trois
    fichiers distincts. L'encodeur texte est un Qwen3-4B, pas un CLIP.

    Chaque image de référence passe par LoadImage → VAEEncode → ReferenceLatent,
    et les ReferenceLatent se chaînent : le conditioning de l'un alimente le
    suivant. C'est ce qui tient le personnage sans LoRA.

    Pas de négatif utile ici : à cfg 1.0 la branche négative n'est pas
    échantillonnée, on la laisse vide plutôt que d'y mettre style.negative qui
    ne servirait à rien tout en coûtant un encodage de plus.
    """
    graph: dict[str, dict[str, object]] = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": FLUX2_UNET, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": FLUX2_CLIP, "type": "flux2"},
        },
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_VAE}},
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["clip", 0]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["clip", 0]},
        },
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": style.width, "height": style.height, "batch_size": 1},
        },
    }

    conditioning: list[object] = ["positive", 0]
    for i, name in enumerate(refs):
        graph[f"ref_img{i}"] = {
            "class_type": "LoadImage",
            "inputs": {"image": name},
        }
        graph[f"ref_enc{i}"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [f"ref_img{i}", 0], "vae": ["vae", 0]},
        }
        graph[f"ref_lat{i}"] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": conditioning, "latent": [f"ref_enc{i}", 0]},
        }
        conditioning = [f"ref_lat{i}", 0]

    graph["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": FLUX2_STEPS,
            "cfg": FLUX2_CFG,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["unet", 0],
            "positive": conditioning,
            "negative": ["negative", 0],
            "latent_image": ["latent", 0],
        },
    }
    graph["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
    }
    graph["save"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "pipeline", "images": ["decode", 0]},
    }
    return graph


_PROTAGONIST = re.compile(r"\bbldguy\b", re.IGNORECASE)


def _qwen_prompt(prompt: str, has_character: bool, has_decor: bool) -> str:
    """Réécrit un prompt Klein pour Qwen-Image-Edit.

    Klein apprenait `bldguy` par la référence elle-même ; Qwen ne connaît pas
    ce mot, mais il sait lire « image 1 ». Le personnage est donc toujours
    image 1, le décor image 2 (ou image 1 s'il est seul).

    La consigne finale « nouvelle scène » évite le piège propre aux modèles
    d'édition : la référence de costume est un personnage debout sur fond
    neutre, et sans elle Qwen la recopie telle quelle au lieu de composer.
    """
    if not has_character and not has_decor:
        return prompt  # aucune image à nommer (plans de steps/location_refs.py)
    text = prompt
    notes: list[str] = []
    if has_character:
        seen: list[bool] = []

        def name(_: re.Match[str]) -> str:
            first = not seen
            seen.append(True)
            return "the bald man from image 1" if first else "the same man"

        text = _PROTAGONIST.sub(name, text)
        notes.append(
            "Keep the man from image 1 exactly: same face, same bald head, "
            "same body proportions, same outfit and same drawing style"
        )
    if has_decor:
        notes.append(
            f"The scene takes place in the location shown in image {2 if has_character else 1}, "
            "keep its architecture, materials and colours"
        )
    # Sauf pour la planche de costume de steps/refs.py, qui DEMANDE justement
    # un personnage debout sur fond neutre : la consigne la contredirait.
    if "neutral background" not in prompt:
        notes.append(
            "Compose a new illustration of this scene with the camera framing described; "
            "do not copy the plain background or the standing pose of the reference images"
        )
    return f"{text}. {'. '.join(notes)}."


def _workflow_qwen(
    prompt: str,
    style: StyleConfig,
    seed: int,
    refs: list[str],
) -> dict[str, dict[str, object]]:
    """Graphe Qwen-Image-Edit-2511, calqué sur le template officiel ComfyUI.

    `refs` est dans l'ordre où le prompt les nomme (image 1, image 2...) ;
    TextEncodeQwenImageEditPlus en accepte trois au plus. Les références
    entrent à la fois dans l'encodeur vision-langage (ce qui permet de les
    désigner) et, via `vae`, comme latents de référence.

    Latent vide plutôt que l'image 1 encodée comme dans le template : on ne
    retouche pas une image, on compose une scène neuve au format du pipeline.

    Pas de négatif : à cfg 1.0 ComfyUI n'échantillonne pas la branche
    négative, un ConditioningZeroOut suffit à remplir l'entrée.
    """
    graph: dict[str, dict[str, object]] = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"},
        },
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
    }
    model: list[object] = ["unet", 0]
    if QWEN_LORA:
        graph["lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model, "lora_name": QWEN_LORA, "strength_model": 1.0},
        }
        model = ["lora", 0]
    graph["shift"] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": model, "shift": QWEN_SHIFT},
    }
    graph["cfgnorm"] = {
        "class_type": "CFGNorm",
        "inputs": {"model": ["shift", 0], "strength": 1.0},
    }

    encode: dict[str, object] = {"clip": ["clip", 0], "prompt": prompt, "vae": ["vae", 0]}
    for i, name in enumerate(refs[:3]):
        graph[f"ref_img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        encode[f"image{i + 1}"] = [f"ref_img{i}", 0]
    graph["encode"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": encode}
    graph["positive"] = {
        "class_type": "FluxKontextMultiReferenceLatentMethod",
        "inputs": {"conditioning": ["encode", 0], "reference_latents_method": "index_timestep_zero"},
    }
    graph["negative"] = {
        "class_type": "ConditioningZeroOut",
        "inputs": {"conditioning": ["encode", 0]},
    }
    graph["latent"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": style.width, "height": style.height, "batch_size": 1},
    }
    graph["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": QWEN_STEPS,
            "cfg": QWEN_CFG,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["cfgnorm", 0],
            "positive": ["positive", 0],
            "negative": ["negative", 0],
            "latent_image": ["latent", 0],
        },
    }
    graph["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
    }
    graph["save"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "pipeline", "images": ["decode", 0]},
    }
    return graph


def _workflow_hidream(
    prompt: str,
    style: StyleConfig,
    seed: int,
    refs: list[str],
) -> dict[str, dict[str, object]]:
    """Graphe HiDream-O1-Image Dev, calqué sur le template officiel ComfyUI.

    `refs` dans l'ordre où le prompt les nomme (image 1, image 2...), comme
    pour Qwen. Sans aucune référence (plans d'établissement de
    steps/location_refs.py), HiDreamO1ReferenceImages est omis : il exige au
    moins une image, et le template fait de même via son interrupteur.
    """
    graph: dict[str, dict[str, object]] = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": HIDREAM_CKPT}},
        "noise": {
            "class_type": "ModelNoiseScale",
            "inputs": {"model": ["ckpt", 0], "noise_scale": HIDREAM_NOISE_SCALE},
        },
        "pos_text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": prompt}},
        "neg_text": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": ""}},
        "latent": {
            "class_type": "EmptyHiDreamO1LatentImage",
            "inputs": {"width": style.width, "height": style.height, "batch_size": 1},
        },
        "sigmas": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["noise", 0], "scheduler": "normal", "steps": HIDREAM_STEPS, "denoise": 1.0},
        },
        "lcm": {
            "class_type": "SamplerLCM",
            "inputs": {"s_noise": 1.0, "s_noise_end": 1.0, "noise_clip_std": 2.5},
        },
    }
    positive: list[object] = ["pos_text", 0]
    negative: list[object] = ["neg_text", 0]
    if refs:
        ref_inputs: dict[str, object] = {"positive": positive, "negative": negative}
        for i, name in enumerate(refs):
            graph[f"ref_img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            ref_inputs[f"images.image_{i + 1}"] = [f"ref_img{i}", 0]
        graph["refs"] = {"class_type": "HiDreamO1ReferenceImages", "inputs": ref_inputs}
        positive, negative = ["refs", 0], ["refs", 1]
    graph["sampler"] = {
        "class_type": "SamplerCustom",
        "inputs": {
            "model": ["noise", 0],
            "add_noise": True,
            "noise_seed": seed,
            "cfg": 1.0,
            "positive": positive,
            "negative": negative,
            "sampler": ["lcm", 0],
            "sigmas": ["sigmas", 0],
            "latent_image": ["latent", 0],
        },
    }
    graph["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["ckpt", 2]},
    }
    graph["save"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "pipeline", "images": ["decode", 0]},
    }
    return graph


def _workflow_sdxl(
    prompt: str,
    style: StyleConfig,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Graphe ComfyUI minimal : checkpoint → (LoRA) → CLIP ×2 → latent → sampler → image.

    Écrit en API format (le JSON que produit « Save (API Format) » dans
    ComfyUI). Volontairement plat : tout ajout — LoRA de style, ControlNet —
    s'insère entre le checkpoint et le sampler sans toucher au reste.

    Le LoRA de personnage s'insère ici de la même façon que dans
    character-consistency/lora_workflow.api.json (même graphe, testé
    séparément) : un noeud LoraLoader entre le checkpoint et tout le reste.
    Le trigger est préfixé automatiquement plutôt que laissé à la charge de
    l'appelant — l'oublier produirait une génération silencieusement non
    conditionnée, indiscernable à l'oeil d'un LoRA qui n'aurait rien appris
    (même classe d'échec silencieux que le mauvais format de clés attrapé
    pendant l'entraînement, voir convert_lora_for_comfy.py).
    """
    model_source = "ckpt"
    text = prompt

    graph: dict[str, dict[str, object]] = {
        "ckpt": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
    }

    if CHARACTER_LORA:
        graph["lora"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["ckpt", 0],
                "clip": ["ckpt", 1],
                "lora_name": CHARACTER_LORA,
                "strength_model": CHARACTER_LORA_STRENGTH,
                "strength_clip": CHARACTER_LORA_STRENGTH,
            },
        }
        model_source = "lora"
        if CHARACTER_LORA_TRIGGER and CHARACTER_LORA_TRIGGER not in prompt:
            text = f"{CHARACTER_LORA_TRIGGER}, {prompt}"

    clip_source: list[object] = [model_source, 1]
    graph["positive"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": text, "clip": clip_source},
    }
    graph["negative"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": style.negative, "clip": clip_source},
    }
    graph["latent"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": style.width, "height": style.height, "batch_size": 1},
    }
    graph["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            # ComfyUI attend un seed sur 64 bits ; le nôtre tient sur 32.
            "seed": seed,
            "steps": style.steps,
            "cfg": style.cfg,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0,
            "model": [model_source, 0],
            "positive": ["positive", 0],
            "negative": ["negative", 0],
            "latent_image": ["latent", 0],
        },
    }
    graph["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["ckpt", 2]},
    }
    graph["save"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "pipeline", "images": ["decode", 0]},
    }
    return graph



def _execution_error(status: dict) -> str:
    """Message lisible d'un `status` ComfyUI en erreur.

    ComfyUI range la cause dans le dernier message `execution_error` de
    `status["messages"]` : type et message de l'exception, plus le nœud
    fautif. Sans cette extraction, l'appelant ne voyait qu'un `str(status)`
    tronqué dont les 300 premiers caractères sont la liste des nœuds mis en
    cache — la cause réelle arrivait bien après la troncature.
    """
    for kind, payload in reversed(status.get("messages", [])):
        if kind != "execution_error":
            continue
        node = payload.get("node_type") or payload.get("node_id") or "?"
        exc_type = payload.get("exception_type", "?")
        message = str(payload.get("exception_message", "")).strip()
        return f"{node} a échoué — {exc_type}: {message[:300]}"
    return str(status)[:300]


class ComfyProvider:
    name = "comfy"

    def __init__(self, model: str | None = None, url: str | None = None) -> None:
        # Par défaut le moteur et le ComfyUI des scènes (COMFY_MODEL,
        # COMFY_URL) ; les étapes de références passent REFS_MODEL/REFS_URL.
        self.model = model or IMAGE_MODEL
        self.url = (url or URL).rstrip("/")
        self._decor_scale = _decor_ref_scale(self.model)
        self._client = httpx.Client(base_url=self.url, timeout=30.0)
        # Joignabilité vérifiée ICI, explicitement, et pas laissée au hasard
        # du premier appel réseau. Avant, le seul appel de __init__ était
        # l'upload des références : sans COMFY_CHARACTER_REFS, la
        # construction ne touchait pas le réseau et réussissait alors même
        # que ComfyUI était éteint. Tous les garde-fous « ComfyUI injoignable
        # au démarrage » (steps/refs.py, steps/location_refs.py,
        # steps/images.py) s'appuient sur une exception à la construction :
        # ils ne se déclenchaient donc jamais dans ce cas. Vu en pratique
        # (16/09, video5) : run lancé ComfyUI éteint, 10 paliers échoués un
        # par un, section_refs.json écrasé par un index vide, puis 343
        # images de repli en 2 secondes et un out.mp4 de bouchons.
        if not _ping(self._client):
            if not AUTOSTART:
                raise RuntimeError(
                    f"ComfyUI injoignable sur {self.url} et COMFY_AUTOSTART=0"
                )
            failure = _boot(self._client, self.url)
            if failure:
                raise RuntimeError(f"ComfyUI injoignable sur {self.url} — {failure}")
        # Chemin local -> nom déjà uploadé côté ComfyUI, pour ne jamais
        # ré-uploader une même référence de palier plus d'une fois par run
        # (voir _upload) même si `reference` varie d'appel en appel.
        self._uploaded: dict[str, str] = {}
        self._upload_lock = threading.Lock()
        self._refs: list[str] = []
        if self.model in REFERENCE_MODELS and CHARACTER_REFS:
            self._refs = [self._upload(Path(r)) for r in CHARACTER_REFS]

    def _upload(self, path: Path, scale: float = 1.0) -> str:
        """Envoie une référence dans le dossier input de ComfyUI, retourne son nom.

        Fait une fois à l'ouverture, pas à chaque image : LoadImage travaille sur
        des noms de fichiers côté serveur, et ré-uploader à chaque scène ferait
        varier le graphe, ce qui invaliderait le cache d'encodage VAE des
        références et rajouterait ce coût à chacune des ~490 images.

        Le nom distant est préfixé par le dossier parent (`section_refs`,
        `location_refs`...), jamais `path.name` seul : `section_refs/000.png`
        et `location_refs/000.png` partagent le même nom de fichier (index
        zero-paddé à 3 chiffres, remis à zéro dans chaque dossier), et
        `overwrite=true` fait qu'un upload écrase silencieusement l'autre côté
        ComfyUI. Vu en pratique sur un run réel (15/09, video5) : costume
        (palier 0) et décor (lieu 0) uploadaient tous deux vers `000.png`,
        l'un écrasant l'autre au fil des scènes — le graphe demandait deux
        références différentes par leur nom mais recevait deux fois la même
        image, tantôt le costume, tantôt le décor selon l'ordre d'upload.
        Résultat : personnage remplacé par le décor sur des dizaines de
        scènes, sans qu'aucune exception ne le signale.
        """
        if not path.is_file():
            raise RuntimeError(f"référence de personnage introuvable : {path}")
        # Existe ≠ lisible. `path.write_bytes()` n'est pas atomique et NTFS
        # journalise la taille avant les données : une coupure brutale laisse
        # un PNG de la bonne taille entièrement rempli de zéros, que
        # `is_file()` accepte et que le pas de reprise des étapes refs
        # (`if path.exists()`) réutilise indéfiniment. ComfyUI, lui, le refuse
        # à chaque scène — une image illisible ici, c'est TOUTES les scènes du
        # palier en bouchons. Vu en pratique (22/09, japon) :
        # `section_refs/000.png` et `001.png` à 100 % de zéros, 73 scènes
        # perdues. Décodé ici pour que l'erreur nomme le fichier à refaire.
        try:
            from PIL import Image

            with Image.open(path) as probe:
                probe.verify()
        except Exception as exc:  # noqa: BLE001 — PNG vide, tronqué, illisible...
            raise RuntimeError(
                f"référence illisible : {path} ({exc}) — la supprimer et "
                "rejouer l'étape qui la produit (`--force refs` / "
                "`--force location_refs`)"
            ) from exc
        # Préfixé aussi par le PROJET (dossier grand-parent) : `section_refs/
        # 002.png` existe dans chaque projet, et un même ComfyUI servant deux
        # projets d'affilée recevait le costume de l'un sous le nom de l'autre
        # — `_ref_name` croyait l'ancien encore en place. Vu le 07/10 sur le
        # bake-off : costume d'escobart dessiné sur des scènes de video5.
        remote_name = f"{path.parent.parent.name}_{path.parent.name}_{path.name}"
        payload: bytes | None = None
        if scale < 1.0:
            # Dimensions ramenées à un multiple de 16 : c'est le pas de la
            # grille latente de FLUX.2, un reste produirait un padding inutile.
            from io import BytesIO

            from PIL import Image

            with Image.open(path) as image:
                source = image.convert("RGB")
                width = max(16, int(source.width * scale) // 16 * 16)
                height = max(16, int(source.height * scale) // 16 * 16)
                buffer = BytesIO()
                source.resize((width, height), Image.LANCZOS).save(buffer, format="PNG")
            payload = buffer.getvalue()
            # Nom distinct : la version réduite ne doit pas écraser l'originale
            # côté ComfyUI, qui peut être utilisée ailleurs à pleine taille.
            remote_name = f"{path.parent.parent.name}_{path.parent.name}_{path.stem}_s{int(scale * 100)}{path.suffix}"

        with (BytesIO(payload) if payload is not None else path.open("rb")) as handle:
            response = self._client.post(
                "/upload/image",
                files={"image": (remote_name, handle, "image/png")},
                data={"overwrite": "true"},
            )
        response.raise_for_status()
        name: str = response.json()["name"]
        return name

    def _ref_name(self, path: Path, scale: float = 1.0) -> str:
        # Verrouillé : `steps/images.py` fait tourner plusieurs workers pour
        # garder la queue de ComfyUI pleine, et deux scènes voisines
        # partagent souvent la même référence. Sans ça, deux threads
        # uploaderaient le même fichier en même temps et écriraient
        # concurremment dans `_uploaded`.
        key = f"{path}|{scale}"
        with self._upload_lock:
            cached = self._uploaded.get(key)
            if cached is None:
                cached = self._upload(path, scale)
                self._uploaded[key] = cached
        return cached

    def generate(
        self,
        prompt: str,
        style: StyleConfig,
        seed: int,
        reference: Path | None = None,
        decor_reference: Path | None = None,
    ) -> bytes:
        # `reference` remplace `self._refs` plutôt que s'y ajouter : elle
        # porte déjà l'identité du personnage (générée à partir de l'ancre
        # d'identité, voir steps/refs.py) — empiler les deux références
        # mettrait deux costumes en concurrence dans la même chaîne
        # ReferenceLatent, précisément le bug que ce mécanisme corrige.
        #
        # `decor_reference`, lui, S'AJOUTE, et passe EN PREMIER dans la
        # chaîne — costume/identité en dernier, pas en second comme avant.
        # Chaque maillon ReferenceLatent conditionne sur le précédent, donc
        # le dernier de la chaîne a la dernière main sur la composition ;
        # l'identité du personnage doit gagner ce bras de fer, pas le décor.
        # Revu le 15/09 après un run réel (video5) où des scènes de dialogue
        # avec `reference` ET `decor_reference` rendaient le décor pur, sans
        # personnage — la validation manuelle du 09/09 qui avait retenu
        # l'ordre inverse a très probablement tourné sous le bug de
        # collision de noms d'upload corrigé aujourd'hui dans `_upload()`
        # (costume et décor s'écrasaient l'un l'autre côté ComfyUI), pas
        # sous ce graphe-ci tel qu'il est réellement exécuté.
        refs = [self._ref_name(reference)] if reference is not None else self._refs
        if self.model in ("qwen", "hidream"):
            # Ordre inverse de Klein, et pour une autre raison : Qwen et
            # HiDream n'ont pas de « dernier maillon qui gagne », ils lisent
            # les références par leur numéro dans le prompt. Personnage =
            # image 1, décor = image 2.
            # Une scène sans `bldguy` (plan d'établissement) ne reçoit pas la
            # référence de costume : nommée nulle part, Qwen l'insérerait
            # quand même dans le plan.
            has_character = bool(refs) and _PROTAGONIST.search(prompt) is not None
            qwen_refs = refs[:1] if has_character else []
            if decor_reference is not None:
                qwen_refs.append(self._ref_name(decor_reference, self._decor_scale))
            build = _workflow_hidream if self.model == "hidream" else _workflow_qwen
            graph = build(
                _qwen_prompt(prompt, has_character, decor_reference is not None),
                style, seed, qwen_refs,
            )
            return self._run(graph)
        if decor_reference is not None:
            # Préfixé, pas ajouté à la fin : le dernier maillon de la chaîne
            # ReferenceLatent a la dernière main sur la composition (voir
            # commentaire ci-dessus), et l'identité doit toujours être ce
            # dernier maillon — que ce soit `reference` (costume de palier)
            # ou, à défaut, `self._refs` (identité globale du protagoniste).
            refs = [self._ref_name(decor_reference, self._decor_scale)] + refs
        return self._run(_workflow(prompt, style, seed, refs, self.model))

    def _run(self, graph: dict[str, dict[str, object]]) -> bytes:
        """Soumet un graphe à ComfyUI et attend l'image produite."""
        response = self._client.post(
            "/prompt",
            content=json.dumps({"prompt": graph}),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 400:
            # ComfyUI renvoie 400 sur un graphe invalide : sans checkpoint
            # chargeable, chaque image échouerait de la même façon. Ce n'est pas
            # un refus de contenu, c'est une erreur de configuration.
            raise RuntimeError(f"ComfyUI a rejeté le graphe : {response.text[:300]}")
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        # Exposé pour les mesures (temps d'exécution côté serveur, lu dans
        # /history) — voir scripts/qwen_bench.py.
        self.last_prompt_id = prompt_id

        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline:
            history = self._client.get(f"/history/{prompt_id}").json()
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    # PAS un ContentRefused : ComfyUI n'a aucun filtre de
                    # contenu, un `status_str == "error"` est TOUJOURS une
                    # erreur d'exécution du graphe (référence illisible, VRAM,
                    # nœud manquant...). Le classer en refus de contenu le
                    # sortait du coupe-circuit de `steps/images.py` — « verdict
                    # sur CETTE scène », donc pas une rafale — et noyait le
                    # message réel dans un `str(status)` tronqué à 300
                    # caractères où la trace Python passait avant lui. Vu en
                    # pratique (22/09, japon) : deux `section_refs/*.png`
                    # remplies de zéros (voir `_upload`), 73 scènes de suite en
                    # bouchons, journal affichant `content_refused` sans autre
                    # indice, et un out.mp4 rendu par-dessus.
                    raise RuntimeError(f"ComfyUI : {_execution_error(status)}")
                for output in entry.get("outputs", {}).values():
                    for image in output.get("images", []):
                        data = self._client.get(
                            "/view",
                            params={
                                "filename": image["filename"],
                                "subfolder": image.get("subfolder", ""),
                                "type": image.get("type", "output"),
                            },
                        ).content
                        _reject_flat(data)
                        return data
            time.sleep(POLL_S)

        raise TimeoutError(f"ComfyUI n'a rien rendu en {TIMEOUT_S:.0f} s")


def _reject_flat(data: bytes) -> None:
    """Lève si l'image est une plage uniforme (voir FLAT_IMAGE_STD).

    Un modèle peut « réussir » côté ComfyUI en rendant une texture grise sans
    rien dessus. Sans ce contrôle, ces images partaient dans images.json avec
    error_code=None, donc jamais retentées ni signalées. Vu le 07/10 avec
    HiDream : 11 scènes de suite. Levée ici, l'image devient un
    provider_error, et le coupe-circuit de steps/images.py arrête l'étape au
    bout de 5 d'affilée au lieu de remplir une vidéo de gris."""
    from PIL import Image, ImageStat

    with Image.open(BytesIO(data)) as image:
        gray = image.convert("L")
        gray.thumbnail((336, 192))
        std = ImageStat.Stat(gray).stddev[0]
    if std < FLAT_IMAGE_STD:
        raise RuntimeError(f"image uniforme (écart-type {std:.1f} < {FLAT_IMAGE_STD:g}) : échec silencieux du modèle")
