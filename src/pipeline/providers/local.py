"""Provider local — SDXL via l'API HTTP de ComfyUI.

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
import time
from pathlib import Path

import httpx

from ..style import StyleConfig
from .base import ContentRefused

URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
CHECKPOINT = os.environ.get("COMFY_CHECKPOINT", "sd_xl_base_1.0.safetensors")
TIMEOUT_S = float(os.environ.get("COMFY_TIMEOUT", "300"))
POLL_S = 0.5


def _workflow(
    prompt: str,
    style: StyleConfig,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Graphe ComfyUI minimal : checkpoint → CLIP ×2 → latent → sampler → image.

    Écrit en API format (le JSON que produit « Save (API Format) » dans
    ComfyUI). Volontairement plat : tout ajout — LoRA de style, ControlNet —
    s'insère entre le checkpoint et le sampler sans toucher au reste.
    """
    return {
        "ckpt": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["ckpt", 1]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": style.negative, "clip": ["ckpt", 1]},
        },
        "latent": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": style.width, "height": style.height, "batch_size": 1},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                # ComfyUI attend un seed sur 64 bits ; le nôtre tient sur 32.
                "seed": seed,
                "steps": style.steps,
                "cfg": style.cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["ckpt", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["latent", 0],
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sampler", 0], "vae": ["ckpt", 2]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "pipeline", "images": ["decode", 0]},
        },
    }


class ComfyProvider:
    name = "comfy"

    def __init__(self) -> None:
        self._client = httpx.Client(base_url=URL, timeout=30.0)

    def generate(
        self,
        prompt: str,
        style: StyleConfig,
        seed: int,
        reference: Path | None = None,
    ) -> bytes:
        response = self._client.post(
            "/prompt",
            content=json.dumps({"prompt": _workflow(prompt, style, seed)}),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 400:
            # ComfyUI renvoie 400 sur un graphe invalide : sans checkpoint
            # chargeable, chaque image échouerait de la même façon. Ce n'est pas
            # un refus de contenu, c'est une erreur de configuration.
            raise RuntimeError(f"ComfyUI a rejeté le graphe : {response.text[:300]}")
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]

        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline:
            history = self._client.get(f"/history/{prompt_id}").json()
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ContentRefused(str(status)[:300])
                for output in entry.get("outputs", {}).values():
                    for image in output.get("images", []):
                        return self._client.get(
                            "/view",
                            params={
                                "filename": image["filename"],
                                "subfolder": image.get("subfolder", ""),
                                "type": image.get("type", "output"),
                            },
                        ).content
            time.sleep(POLL_S)

        raise TimeoutError(f"ComfyUI n'a rien rendu en {TIMEOUT_S:.0f} s")
