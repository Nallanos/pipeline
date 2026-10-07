"""Comparatif de modèles d'image open source (licence commerciale) sur les
mêmes 20 scènes que scripts/qwen_bench.py, via un ComfyUI distant.

    uv run python scripts/bakeoff.py --downloads        # commande de démarrage du pod
    COMFY_URL=https://<pod>-8188.proxy.runpod.net uv run python scripts/bakeoff.py mage boogu ...

Les graphes vivent ici et pas dans providers/local.py : seul le gagnant y
entrera. Chacun suit le template officiel ComfyUI du modèle, en remplaçant
l'image éditée par un latent vide au format du pipeline (on compose une scène,
on ne retouche pas une image).

Deux modèles (boogu, longcat) n'acceptent qu'UNE référence : ils reçoivent
le costume, le décor ne passe que par le texte.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

os.environ.setdefault("PIPELINE_PROVIDER", "comfy")
os.environ["COMFY_MODEL"] = "qwen"  # décor envoyé à pleine taille (voir DECOR_REF_SCALE)
os.environ.setdefault("COMFY_AUTOSTART", "0")
os.environ.setdefault("COMFY_TIMEOUT", "900")

sys.path.insert(0, str(Path(__file__).parent))

from qwen_bench import OUT as BENCH_DIR  # noqa: E402
from qwen_bench import server_seconds  # noqa: E402

from pipeline.providers import local  # noqa: E402
from pipeline.style import load as load_style  # noqa: E402

OUT = BENCH_DIR / "bakeoff"
HF = "https://huggingface.co"
W, H = 1344, 768

Graph = dict[str, dict[str, object]]


def _tail(graph: Graph, samples: list[object], vae: list[object]) -> Graph:
    graph["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": samples, "vae": vae}}
    graph["save"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "bakeoff", "images": ["decode", 0]}}
    return graph


def _load_refs(graph: Graph, refs: list[str]) -> list[list[object]]:
    for i, name in enumerate(refs):
        graph[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
    return [[f"ref{i}", 0] for i in range(len(refs))]


def _ksampler(model: list[object], pos: list[object], neg: list[object], latent: list[object],
              seed: int, steps: int, cfg: float, sampler: str = "euler", scheduler: str = "simple") -> dict:
    return {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler,
        "denoise": 1.0, "model": model, "positive": pos, "negative": neg, "latent_image": latent}}


def firered(prompt: str, refs: list[str], seed: int) -> Graph:
    """FireRed-Image-Edit 1.1 : un fine-tune de Qwen-Image-Edit, même graphe.
    Seul le bf16 de 41 Go existe : chargé en fp8 à la volée."""
    saved = (local.QWEN_UNET, local.QWEN_LORA, local.QWEN_STEPS)
    local.QWEN_UNET = "FireRed-Image-Edit-1.1-transformer.safetensors"
    local.QWEN_LORA = "FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors"
    local.QWEN_STEPS = 8
    try:
        graph = local._workflow_qwen(prompt, load_style(), seed, refs)
    finally:
        local.QWEN_UNET, local.QWEN_LORA, local.QWEN_STEPS = saved
    graph["unet"]["inputs"]["weight_dtype"] = "fp8_e4m3fn"
    return graph


def mage(prompt: str, refs: list[str], seed: int) -> Graph:
    """Mage-Flow Edit Turbo (Microsoft, MIT) : 4 pas, jusqu'à 3 références.
    L'encodeur produit lui-même le latent à la taille demandée."""
    g: Graph = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "mage_flow_edit_turbo_int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "mage", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "mage_flow_vae_bf16.safetensors"}},
    }
    enc: dict[str, object] = {"clip": ["clip", 0], "prompt": prompt, "negative_prompt": "", "vae": ["vae", 0],
                              "width": W, "height": H, "batch_size": 1}
    for i, ref in enumerate(_load_refs(g, refs)):
        enc[f"images.image_{i + 1}"] = ref
    g["enc"] = {"class_type": "TextEncodeMageFlowEdit", "inputs": enc}
    g["sampler"] = _ksampler(["unet", 0], ["enc", 0], ["enc", 1], ["enc", 2], seed, 4, 1.0)
    return _tail(g, ["sampler", 0], ["vae", 0])


def boogu(prompt: str, refs: list[str], seed: int) -> Graph:
    """Boogu-Image 0.1 Edit Turbo (Apache 2.0) : 4 pas, cfg 1, une référence."""
    g: Graph = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "boogu_image_edit_turbo_hotfix_1k_20260708_int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_fp8_scaled.safetensors", "type": "boogu", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux1_vae_bf16.safetensors"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.16}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
    }
    enc: dict[str, object] = {"clip": ["clip", 0], "prompt": prompt, "negative_prompt": "", "vae": ["vae", 0]}
    for i, ref in enumerate(_load_refs(g, refs)):
        enc[f"images.image_{i + 1}"] = ref
    g["enc"] = {"class_type": "TextEncodeBooguEdit", "inputs": enc}
    g["sampler"] = _ksampler(["shift", 0], ["enc", 0], ["enc", 1], ["latent", 0], seed, 4, 1.0)
    return _tail(g, ["sampler", 0], ["vae", 0])


def longcat(prompt: str, refs: list[str], seed: int) -> Graph:
    """LongCat-Image-Edit Turbo (Meituan, Apache 2.0) : 8 pas, cfg 1, une référence."""
    g: Graph = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "longcat_image_edit_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "longcat_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
    }
    (ref,) = _load_refs(g, refs[:1])
    for name, text in (("pos", prompt), ("neg", "")):
        g[f"{name}_enc"] = {"class_type": "TextEncodeQwenImageEdit", "inputs": {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], "image": ref}}
        g[f"{name}_guid"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": [f"{name}_enc", 0], "guidance": 4.5}}
        g[name] = {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": [f"{name}_guid", 0], "reference_latents_method": "index"}}
    g["sampler"] = _ksampler(["unet", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, 8, 1.0)
    return _tail(g, ["sampler", 0], ["vae", 0])


def joyai(prompt: str, refs: list[str], seed: int) -> Graph:
    """JoyAI-Image-Edit (JD, Apache 2.0) : pas de version distillée, 40 pas cfg 4."""
    g: Graph = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "joyai_image_edit_int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_joyimage_edit_int8_convrot.safetensors", "type": "joyimage", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["unet", 0], "strength": 1.0, "pre_cfg": True}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
    }
    loaded = _load_refs(g, refs)
    for name, text in (("pos", prompt), ("neg", "")):
        enc: dict[str, object] = {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0]}
        for i, ref in enumerate(loaded):
            enc[f"images.image{i}"] = ref
        g[name] = {"class_type": "TextEncodeJoyImageEdit", "inputs": enc}
    g["sampler"] = _ksampler(["norm", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, 40, 4.0, scheduler="normal")
    return _tail(g, ["sampler", 0], ["vae", 0])


# Paramètres de SamplerLCM lus dans /object_info au démarrage du pod (ses
# noms d'entrée ne figurent pas dans le template) — voir hidream().
LCM_INPUTS: list[str] = []


def hidream(prompt: str, refs: list[str], seed: int) -> Graph:
    """HiDream-O1-Image Dev (MIT) : 28 pas, échantillonneur « flash » du
    template (SamplerLCM + ModelNoiseScale 7,6), références multiples."""
    g: Graph = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "hidream_o1_image_dev_fp8_scaled.safetensors"}},
        "noise": {"class_type": "ModelNoiseScale", "inputs": {"model": ["ckpt", 0], "noise_scale": 7.6}},
        "pos_txt": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": prompt}},
        "neg_txt": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["ckpt", 1], "text": ""}},
        "latent": {"class_type": "EmptyHiDreamO1LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": ["noise", 0], "scheduler": "normal", "steps": 28, "denoise": 1.0}},
        "lcm": {"class_type": "SamplerLCM", "inputs": dict(zip(LCM_INPUTS, [1.0, 1.0, 2.5]))},
    }
    refin: dict[str, object] = {"positive": ["pos_txt", 0], "negative": ["neg_txt", 0]}
    for i, ref in enumerate(_load_refs(g, refs)):
        refin[f"images.image_{i + 1}"] = ref
    g["refs"] = {"class_type": "HiDreamO1ReferenceImages", "inputs": refin}
    g["sampler"] = {"class_type": "SamplerCustom", "inputs": {
        "model": ["noise", 0], "add_noise": True, "noise_seed": seed, "cfg": 1.0,
        "positive": ["refs", 0], "negative": ["refs", 1], "sampler": ["lcm", 0],
        "sigmas": ["sched", 0], "latent_image": ["latent", 0]}}
    return _tail(g, ["sampler", 0], ["ckpt", 2])


# nom -> (graphe, accepte plusieurs références, fichiers à télécharger)
MODELS: dict[str, tuple[Callable[[str, list[str], int], Graph], bool, list[tuple[str, str]]]] = {
    "mage": (mage, True, [
        ("diffusion_models", f"{HF}/Comfy-Org/Mage-Flow/resolve/main/diffusion_models/mage_flow_edit_turbo_int8_convrot.safetensors"),
        ("text_encoders", f"{HF}/Comfy-Org/Mage-Flow/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors"),
        ("vae", f"{HF}/Comfy-Org/Mage-Flow/resolve/main/vae/mage_flow_vae_bf16.safetensors"),
    ]),
    "boogu": (boogu, False, [
        ("diffusion_models", f"{HF}/Comfy-Org/Boogu-Image/resolve/main/diffusion_models/boogu_image_edit_turbo_hotfix_1k_20260708_int8_convrot.safetensors"),
        ("text_encoders", f"{HF}/Comfy-Org/Boogu-Image/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"),
        ("vae", f"{HF}/Comfy-Org/Boogu-Image/resolve/main/vae/flux1_vae_bf16.safetensors"),
    ]),
    "longcat": (longcat, False, [
        ("diffusion_models", f"{HF}/Comfy-Org/LongCat-Image/resolve/main/split_files/diffusion_models/longcat_image_edit_turbo_bf16.safetensors"),
        ("text_encoders", f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
        ("vae", f"{HF}/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"),
    ]),
    "firered": (firered, True, [
        ("diffusion_models", f"{HF}/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/resolve/main/FireRed-Image-Edit-1.1-transformer.safetensors"),
        ("loras", f"{HF}/FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI/resolve/main/FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors"),
        ("text_encoders", f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
        ("vae", f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"),
    ]),
    "hidream": (hidream, True, [
        ("checkpoints", f"{HF}/Comfy-Org/HiDream-O1-Image/resolve/main/checkpoints/hidream_o1_image_dev_fp8_scaled.safetensors"),
    ]),
    "joyai": (joyai, True, [
        ("diffusion_models", f"{HF}/Comfy-Org/JoyAI-Image-Edit/resolve/main/diffusion_models/joyai_image_edit_int8_convrot.safetensors"),
        ("text_encoders", f"{HF}/Comfy-Org/JoyAI-Image-Edit/resolve/main/text_encoders/qwen3vl_8b_joyimage_edit_int8_convrot.safetensors"),
        ("vae", f"{HF}/Comfy-Org/JoyAI-Image-Edit/resolve/main/vae/wan_2.1_vae.safetensors"),
    ]),
}


def downloads_command() -> str:
    """Commande de démarrage du pod : télécharge tout en parallèle (dans
    l'ordre de MODELS, pour pouvoir commencer par les premiers), puis lance
    le start.sh normal du template ComfyUI."""
    seen: set[str] = set()
    lines = []
    for _, _, files in MODELS.values():
        for folder, url in files:
            if url not in seen:
                seen.add(url)
                lines.append(f"dl {folder} {url} &")
    return (
        'D=/workspace/runpod-slim/ComfyUI/models; dl() { mkdir -p "$D/$1"; f="$D/$1/$(basename "$2")"; '
        '[ -f "$f" ] && return; t0=$(date +%s); curl -L --fail --retry 5 -s -o "$f.part" "$2" '
        '&& mv "$f.part" "$f" && echo "DL_OK $f $(( $(date +%s) - t0 ))s" || echo "DL_FAIL $2"; }; '
        '( until [ -f /workspace/runpod-slim/ComfyUI/main.py ] && [ -d "$D" ]; do sleep 2; done; echo DL_START; '
        + " ".join(lines)
        + " wait; echo DL_DONE ) & nvidia-smi --query-gpu=name,memory.total --format=csv; exec /start.sh"
    )


def main(names: list[str]) -> None:
    style = load_style()
    jobs = json.loads((BENCH_DIR / "results.json").read_text(encoding="utf-8"))
    provider = local.ComfyProvider()

    lcm = provider._client.get("/object_info/SamplerLCM").json()["SamplerLCM"]["input"]
    LCM_INPUTS[:] = list(lcm.get("required", {}))

    results_file = OUT / "results.json"
    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else []
    done = {(r["model"], r["project"], r["scene"]) for r in results if "error" not in r}

    for name in names:
        build, multi, _ = MODELS[name]
        first = True
        for job in jobs:
            key = (name, job["project"], job["scene"])
            if key in done:
                continue
            refs = [provider._ref_name(Path(job["costume"]))]
            has_decor = bool(job["decor"]) and multi
            if has_decor:
                refs.append(provider._ref_name(Path(job["decor"]), local.DECOR_REF_SCALE))
            prompt = local._qwen_prompt(style.full_prompt(job["subject"]), True, has_decor)
            entry = {"model": name, "project": job["project"], "scene": job["scene"],
                     "refs": len(refs), "warmup": first}
            started = time.perf_counter()
            try:
                data = provider._run(build(prompt, refs, job["seed"]))
            except Exception as exc:  # noqa: BLE001 — on consigne et on passe à la suite
                entry["error"] = str(exc)[:400]
                print(f"{name} {job['project']} {job['scene']} ÉCHEC {exc}", flush=True)
                results.append(entry)
                results_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
                if first:
                    break  # graphe cassé : inutile d'insister sur les 19 autres
                continue
            entry["wall_s"] = round(time.perf_counter() - started, 2)
            entry["server_s"] = server_seconds(provider, provider.last_prompt_id)
            out = OUT / name / job["project"] / f"{job['scene']:04d}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            entry["image"] = str(out)
            results.append(entry)
            results_file.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{name} {job['project']} {job['scene']} wall {entry['wall_s']} s / serveur {entry['server_s']} s", flush=True)
            first = False


if __name__ == "__main__":
    if sys.argv[1:] == ["--downloads"]:
        print(downloads_command())
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        main(sys.argv[1:] or list(MODELS))
