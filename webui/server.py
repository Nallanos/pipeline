"""Panneau de contrôle local pour le pipeline — ne tourne jamais que sur cette
machine (127.0.0.1). Pas un service publiable : il lance des sous-processus
locaux, lit/écrit `projects/`, et attend ComfyUI/Chatterbox sur leurs ports
habituels. Aucune raison de l'exposer au-delà de localhost.

    C:/ComfyUI/venv/Scripts/python.exe main.py            (si FLUX/SDXL)
    C:/ChatterboxTTS/venv/Scripts/python.exe server.py    (si voix Chatterbox)
    .venv/Scripts/python.exe webui/server.py

Puis http://127.0.0.1:8765
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# Utilisé comme composant de chemin (projects/<id>/) : caractères sûrs
# seulement, sinon un id malicieux pourrait s'échapper du dossier projects/.
_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Même ordre que _steps() dans src/pipeline/__main__.py — dupliqué ici plutôt
# qu'importé : le webui ne dépend d'aucun module interne du pipeline, il ne
# fait que lancer `python -m pipeline` en sous-processus (voir _build_env).
_STEPS = ("tts", "align", "segment", "prompts", "refs", "location_refs", "images", "render")

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CHARACTERS_DIR = ROOT / "characters" / "refs"
THUMBNAILS_DIR = ROOT / "thumbnails"
# Repli quand aucune référence n'est cochée dans l'UI, pour un run comme pour
# la revue. Les scènes qui précèdent le premier `NIVEAU:` n'ont pas de
# référence de palier et retombent sur COMFY_CHARACTER_REFS : vide, Klein
# n'a plus que le mot `bldguy`, qui ne lui dit rien sans le LoRA, et dessine
# un garçon générique avec des cheveux. Vu le 30/09 sur escobart, scènes 0-2.
DEFAULT_CHARACTER_REF = "bldguy_head.png"
# Hors de projects/<id>/ volontairement : ce sont des candidats, pas un
# artefact du pipeline — les y mélanger risquerait qu'un futur code y
# glisse par erreur (glob trop large sur images/, par ex.).
REVIEW_DIR = ROOT / "review_batches"

# Charge FISH_API_KEY / FISH_VOICE_ID (etc.) dans l'environnement du serveur —
# repli utilisé par _build_env() quand le formulaire les laisse vides. Le
# fichier n'est jamais lu ni exposé ailleurs ; .env reste dans .gitignore.
load_dotenv(ROOT / ".env")

app = FastAPI(title="Pipeline — panneau de contrôle")


class RunState:
    """Un run en cours (ou fini), détaché de toute connexion WebSocket.

    Lancé par `_execute_run` en tâche de fond via `asyncio.create_task` — il
    continue même si personne ne l'observe. Le journal produit jusqu'ici est
    gardé en mémoire (`log`) pour qu'un client qui se (re)connecte en cours de
    route puisse rattraper ce qu'il a manqué avant de rejoindre le flux live
    via `subscribers`.
    """

    def __init__(self, project_id: str, opts: dict | None = None) -> None:
        self.project_id = project_id
        self.opts = opts or {}
        self.process: subprocess.Popen | None = None
        self.log: list[str] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.done = False
        self.code: int | None = None
        # Suivi pour l'estimation du temps restant (voir _estimate) : étape
        # en cours, son heure de début, et la progression de l'étape images.
        self.step: str | None = None
        self.step_started = time.time()
        self.images_done = 0
        self.images_total = 0

    def publish(self, msg: dict) -> None:
        if msg.get("type") == "log":
            self.log.append(msg["line"])
            self._track(msg["line"])
        for queue in list(self.subscribers):
            queue.put_nowait(msg)

    def _track(self, line: str) -> None:
        m = _STEP_LINE_RE.match(line)
        if not m:
            return
        name, rest = m.groups()
        if name == "progress":
            p = _PROGRESS_RE.match(rest)
            if p:
                self.images_done, self.images_total = int(p.group(1)), int(p.group(2))
            return
        if name not in _STEPS:
            return
        if rest == "…":
            self.step = name
            self.step_started = time.time()
            if name == "images":
                self.images_done = self.images_total = 0
            return
        d = _DURATION_RE.match(rest)
        if d:
            try:
                _timings_learn(self, name, float(d.group(1)))
            except Exception as exc:  # noqa: BLE001 — l'estimation ne doit jamais casser un run
                print(f"estimation : {exc}")


# Un seul run à la fois : le pipeline sérialise déjà le GPU côté ComfyUI, et
# faire tourner deux `pipeline run` en parallèle corromprait leurs artefacts.
# `_run_state` sert à la fois de verrou (posé de façon synchrone dans
# `api_start_run`, donc sans fenêtre de course) et de tableau d'affichage
# pour quiconque se connecte ensuite.
_run_state: RunState | None = None


class ScheduledRun:
    """Un run programmé pour plus tard — pas encore lancé, juste en attente.

    Vit uniquement en mémoire : redémarrer le serveur avant l'heure prévue
    perd la programmation. Assumé pour un outil local mono-poste (RUN.md) —
    la machine doit rester allumée jusqu'à l'heure dite de toute façon."""

    def __init__(self, project_id: str, opts: dict, when: datetime) -> None:
        self.project_id = project_id
        self.opts = opts
        self.when = when


# Une seule programmation à la fois, même logique que _run_state : un seul
# GPU, pas la peine d'en empiler plusieurs qui se marcheraient dessus à
# l'heure dite.
_scheduled: ScheduledRun | None = None


# ---------- Estimation du temps restant ----------
# Durées apprises au fil des runs (moyenne glissante), dans `timings.json`.
# Les valeurs par défaut ne servent qu'au tout premier run : elles viennent
# des mesures de RUN.md (≈ 18 s/image en fp4 + surcoût) ou sont prudentes.
# Unités : secondes par étape, sauf `image` (par image générée) et
# `*_per_scene` (par scène du projet).
TIMINGS_FILE = ROOT / "timings.json"
_DEFAULT_TIMINGS = {
    "tts": 120.0,
    "align": 60.0,
    "segment": 5.0,
    "prompts_per_scene": 1.0,
    "refs": 250.0,
    "location_refs": 320.0,
    "image": 20.0,
    "render_per_scene": 1.5,
}
_STEP_LINE_RE = re.compile(r"^\[(\w+)\] (.+)$")
_PROGRESS_RE = re.compile(r"^images (\d+)/(\d+)$")
_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?) s$")
# `--sample` ne garde que ~3×30 s de vidéo (scenes.sample_indices).
_SAMPLE_SCENES = 22


def _timings() -> dict:
    try:
        return {**_DEFAULT_TIMINGS, **json.loads(TIMINGS_FILE.read_text(encoding="utf-8"))}
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULT_TIMINGS)


def _scene_count(project_dir: Path) -> int:
    """Nombre de scènes : `scenes.json` s'il existe, sinon estimé depuis le
    script (≈ 8 mots par scène : un plan / 3,3 s, ~2,5 mots/s)."""
    try:
        data = json.loads((project_dir / "scenes.json").read_text(encoding="utf-8"))
        scenes = data["scenes"] if isinstance(data, dict) else data
        return len(scenes)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    try:
        words = len((project_dir / "script.txt").read_text(encoding="utf-8").split())
        return max(1, round(words / 8))
    except OSError:
        return 300


def _timings_learn(state: "RunState", step: str, seconds: float) -> None:
    """Intègre la durée mesurée d'une étape terminée dans `timings.json`."""
    t = _timings()
    n = _scene_count(PROJECTS_DIR / state.project_id)
    sample = bool(state.opts.get("sample"))
    if step == "images":
        if state.images_done <= 0:
            return
        key, value = "image", seconds / state.images_done
    elif step in ("prompts", "render"):
        if sample or seconds < 5:
            return  # reprise partielle ou sample : ne dit rien du coût par scène
        key, value = f"{step}_per_scene", seconds / n
    else:
        if seconds < 1:
            return
        key, value = step, seconds
    t[key] = round(0.5 * t[key] + 0.5 * value, 3)
    _write_atomic(TIMINGS_FILE, json.dumps(t, indent=1))


def _missing_images(project_dir: Path) -> int | None:
    try:
        data = json.loads((project_dir / "scenes.json").read_text(encoding="utf-8"))
        scenes = data["scenes"] if isinstance(data, dict) else data
        indices = [int(s["index"]) for s in scenes]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return sum(1 for i in indices if not (project_dir / "images" / f"{i:04d}.png").exists())


def _step_needed(project_dir: Path, step: str, sample: bool) -> bool:
    """Heuristique sur les artefacts — le vrai verdict appartient au
    pipeline (`is_complete` de chaque étape), ceci ne sert qu'à estimer."""
    has_levels = (project_dir / "levels.json").exists()
    if step == "tts":
        return not (project_dir / "audio.wav").exists()
    if step == "align":
        return not (project_dir / "alignment.json").exists()
    if step == "segment":
        return not (project_dir / "scenes.json").exists()
    if step == "prompts":
        return not (project_dir / "prompts.json").exists()
    if step == "refs":
        return has_levels and not (project_dir / "section_refs.json").exists()
    if step == "location_refs":
        return has_levels and not (project_dir / "location_refs.json").exists()
    if step == "images":
        missing = _missing_images(project_dir)
        return missing is None or missing > 0
    out = project_dir / ("out.sample.mp4" if sample else "out.mp4")
    images_file = project_dir / "images.json"
    return not out.exists() or not images_file.exists() or out.stat().st_mtime < images_file.stat().st_mtime


def _estimate(project_id: str, opts: dict, state: "RunState | None" = None) -> dict | None:
    """Temps restant estimé pour un run (en cours si `state`, sinon à venir)."""
    try:
        d = PROJECTS_DIR / project_id
        t = _timings()
        sample = bool(opts.get("sample"))
        n = _scene_count(d)
        n_eff = min(n, _SAMPLE_SCENES) if sample else n
        force = opts.get("force") or ""
        forced_from = _STEPS.index(force) if force in _STEPS else len(_STEPS)

        def cost(step: str) -> float:
            if step == "prompts":
                return t["prompts_per_scene"] * n_eff
            if step == "render":
                return t["render_per_scene"] * n_eff
            if step == "images":
                missing = _missing_images(d)
                if _STEPS.index("images") >= forced_from or missing is None:
                    units = n_eff
                else:
                    units = min(missing, n_eff)
                return t["image"] * units
            return t[step]

        remaining = 0.0
        current = state.step if state is not None else None
        start_i = _STEPS.index(current) if current in _STEPS else 0
        if current is not None:
            elapsed = time.time() - state.step_started
            if current == "images" and state.images_done > 0:
                rate = elapsed / state.images_done
                remaining += rate * max(0, state.images_total - state.images_done)
            elif current == "images" and state.images_total:
                remaining += t["image"] * state.images_total
            else:
                remaining += max(0.0, cost(current) - elapsed)
            start_i += 1
        for i, step in enumerate(_STEPS[start_i:], start=start_i):
            if i >= forced_from or step == "render" or _step_needed(d, step, sample):
                remaining += cost(step)
        return {
            "remaining_s": round(remaining),
            "step": current,
            "images": [state.images_done, state.images_total] if state is not None and state.images_total else None,
        }
    except Exception as exc:  # noqa: BLE001 — une estimation ratée n'est jamais bloquante
        print(f"estimation {project_id} : {exc}")
        return None


# ---------- File de nuit ----------
# Plusieurs projets enchaînés l'un après l'autre, à partir d'une heure donnée.
# Contrairement à `_scheduled`, la file est PERSISTÉE (`queue.json`) : un
# redémarrage du serveur — ou un écran bleu, vu deux fois en pleine génération
# — ne la perd pas. Au redémarrage, l'élément qui était « running » repasse
# « pending » et est relancé : le pipeline reprend de lui-même à l'étape où il
# s'était arrêté.
#
# Toutes les mutations passent par des handlers `async def` et par
# `_scheduler_loop`, donc par le seul thread de la boucle asyncio : pas de
# verrou nécessaire.
QUEUE_FILE = ROOT / "queue.json"
# Jamais écrite sur disque : la clé retombe sur `.env` au lancement (voir
# _build_env), comme quand le champ du formulaire est laissé vide.
_QUEUE_SECRET_OPTS = ("fish_api_key",)


def _queue_empty() -> dict:
    return {"start_at": None, "paused": True, "items": []}


def _queue_load() -> dict:
    try:
        data = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        data.setdefault("items", [])
        data.setdefault("paused", True)
        data.setdefault("start_at", None)
        return data
    except (OSError, json.JSONDecodeError):
        return _queue_empty()


_queue: dict = _queue_empty()
# Id de l'élément de file dont le run occupe `_run_state`, None sinon — pour
# savoir, quand ce run se termine, quel élément marquer terminé/en échec.
_queue_current: str | None = None


def _queue_save() -> None:
    _write_atomic(QUEUE_FILE, json.dumps(_queue, ensure_ascii=False, indent=1))


def _queue_item(item_id: str) -> dict | None:
    return next((it for it in _queue["items"] if it["id"] == item_id), None)


def _queue_due() -> bool:
    if _queue["paused"]:
        return False
    start_at = _queue.get("start_at")
    return not start_at or datetime.now() >= datetime.fromisoformat(start_at)


# Empêche la mise en veille tant qu'un run tourne (file ou run manuel) — la
# veille a déjà provoqué un bugcheck en pleine génération (RUN.md, 15/08).
# `SetThreadExecutionState` vaut pour le thread appelant : toujours appelé
# depuis `_scheduler_loop`, donc depuis le même thread, qui vit tant que le
# serveur tourne.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_keep_awake_on = False


def _keep_awake(on: bool) -> None:
    global _keep_awake_on
    if on == _keep_awake_on or os.name != "nt":
        return
    import ctypes

    flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
    _keep_awake_on = on


def _queue_tick() -> None:
    """Un pas de la file : clôt l'élément dont le run vient de finir, puis
    lance le suivant si la file est due et que le GPU est libre."""
    global _queue_current, _run_state
    if _queue_current is not None and _run_state is not None and _run_state.done:
        item = _queue_item(_queue_current)
        if item is not None:
            item["status"] = "done" if _run_state.code == 0 else "failed"
            item["code"] = _run_state.code
            item["ended_at"] = datetime.now().isoformat(timespec="seconds")
            _queue_save()
        _queue_current = None

    if _run_state is not None and not _run_state.done:
        return
    if not _queue_due():
        return
    item = next((it for it in _queue["items"] if it["status"] == "pending"), None)
    if item is None:
        return
    if not (PROJECTS_DIR / item["project_id"]).is_dir():
        item["status"] = "failed"
        item["code"] = None
        item["error"] = "projet introuvable"
        _queue_save()
        return

    item["status"] = "running"
    item["started_at"] = datetime.now().isoformat(timespec="seconds")
    item.pop("ended_at", None)
    item.pop("code", None)
    _queue_save()
    _queue_current = item["id"]
    state = RunState(item["project_id"], item["opts"])
    _run_state = state
    asyncio.create_task(_execute_run(state, item["opts"]))


async def _scheduler_loop() -> None:
    """Vérifie toutes les 15 s si une programmation est arrivée à échéance,
    puis fait avancer la file de nuit.

    N'écrase jamais un run déjà en cours : si l'heure prévue tombe pendant
    qu'un autre run tourne encore, elle attend simplement que `_run_state` se
    libère avant de démarrer, plutôt que de le couper ou d'échouer. La
    programmation simple passe avant la file quand les deux sont dues.
    """
    global _scheduled, _run_state
    while True:
        await asyncio.sleep(15)
        run_active = _run_state is not None and not _run_state.done
        if (
            _scheduled is not None
            and datetime.now() >= _scheduled.when
            and not run_active
            and _queue_current is None
        ):
            due = _scheduled
            _scheduled = None
            state = RunState(due.project_id, due.opts)
            _run_state = state
            asyncio.create_task(_execute_run(state, due.opts))
        else:
            try:
                _queue_tick()
            except Exception as exc:  # noqa: BLE001 — la boucle ne doit jamais mourir
                print(f"file de nuit : {exc}")
        _keep_awake(_run_state is not None and not _run_state.done)


@app.on_event("startup")
async def _on_startup() -> None:
    global _queue
    _queue = _queue_load()
    # Un élément « running » au démarrage = le serveur (ou la machine) est
    # tombé pendant son run : on le relance, le pipeline reprend où il était.
    for item in _queue["items"]:
        if item["status"] == "running":
            item["status"] = "pending"
    if _queue["items"]:
        _queue_save()
    asyncio.create_task(_scheduler_loop())


class ThumbState:
    """Un lot de miniatures en cours (ou fini) — même principe que RunState :
    détaché de toute connexion, un abonné qui arrive en cours de route
    rattrape les images déjà produites avant de rejoindre le flux live."""

    def __init__(self, batch_id: str, prompt: str, count: int) -> None:
        self.batch_id = batch_id
        self.prompt = prompt
        self.count = count
        self.images: list[str] = []
        self.process: asyncio.subprocess.Process | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.done = False

    def publish(self, msg: dict) -> None:
        if msg.get("type") == "image":
            self.images.append(msg["file"])
        for queue in list(self.subscribers):
            queue.put_nowait(msg)


# Même contrainte que les runs : ComfyUI sérialise le GPU, deux lots en
# parallèle se marcheraient dessus. Un seul lot de miniatures à la fois,
# indépendant de _run_state (générer des miniatures et lancer un vrai projet
# en même temps saturerait juste le même GPU un peu plus fort — accepté).
_thumb_state: ThumbState | None = None


class ReviewState:
    """Un lot de régénération (page « revue ») en cours ou fini — même
    principe que `ThumbState`, mais groupé par scène : chaque scène marquée
    reçoit `variants` alternatives, pas une liste plate."""

    def __init__(self, batch_id: str, project_id: str, jobs: list[dict], variants: int) -> None:
        self.batch_id = batch_id
        self.project_id = project_id
        self.jobs = jobs  # [{"index": int, "prompt": str}, ...]
        self.variants = variants
        self.images: dict[int, list[str]] = defaultdict(list)
        self.process: asyncio.subprocess.Process | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.done = False

    def publish(self, msg: dict) -> None:
        if msg.get("type") == "image":
            self.images[msg["scene_index"]].append(msg["file"])
        for queue in list(self.subscribers):
            queue.put_nowait(msg)


# Un seul lot de régénération à la fois — même GPU que _run_state/_thumb_state,
# mais indépendant des deux : les faire tourner en même temps ralentit juste
# tout le monde, ComfyUI ne corrompt rien (sa propre file interne suffit).
_review_state: ReviewState | None = None


def _project_status(project_id: str, has_video: bool) -> tuple[str, str | None]:
    """(statut, heure programmée ISO ou None) — dérivé de l'état global, pas
    stocké sur disque : `_run_state`/`_scheduled` sont la seule source de
    vérité, un projet ne "sait" pas lui-même qu'il tourne ou est programmé."""
    if _run_state is not None and _run_state.project_id == project_id and not _run_state.done:
        return "running", None
    if _scheduled is not None and _scheduled.project_id == project_id:
        return "scheduled", _scheduled.when.isoformat()
    if any(it["project_id"] == project_id and it["status"] == "pending" for it in _queue["items"]):
        return "queued", _queue.get("start_at")
    if has_video:
        return "done", None
    return "idle", None


def _list_projects() -> list[dict]:
    if not PROJECTS_DIR.exists():
        return []
    out = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        images_dir = d / "images"
        has_video = (d / "out.mp4").exists()
        status, scheduled_at = _project_status(d.name, has_video)

        cover = None
        if images_dir.exists():
            first = next(iter(sorted(images_dir.glob("*.png"))), None)
            if first:
                cover = first.name

        out.append(
            {
                "id": d.name,
                "has_script": (d / "script.txt").exists(),
                "has_audio": (d / "audio.wav").exists(),
                "has_images": images_dir.exists() and any(images_dir.iterdir()),
                "has_video": has_video,
                "has_sample": (d / "out.sample.mp4").exists(),
                "status": status,
                "scheduled_at": scheduled_at,
                "cover_image": cover,
            }
        )
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/api/projects")
def api_projects() -> JSONResponse:
    return JSONResponse(_list_projects())


@app.get("/api/characters")
def api_characters() -> JSONResponse:
    """Références de personnage disponibles pour FLUX.2, pour le menu déroulant."""
    if not CHARACTERS_DIR.exists():
        return JSONResponse([])
    return JSONResponse(sorted(p.name for p in CHARACTERS_DIR.glob("*.png")))


def _port_open(port: int) -> bool:
    """Un simple connect TCP : ComfyUI et Chatterbox n'exposent pas de route
    de santé standard, mais un port qui accepte la connexion suffit à dire
    « le serveur tourne » pour ce qu'on en fait ici — pas besoin de plus."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


@app.get("/api/status")
def api_status() -> JSONResponse:
    """Vue d'ensemble pour la page « Statut » : quels serveurs locaux
    répondent, et quel projet a un run en cours, si un a été démarré depuis
    n'importe quel onglet — `_run_state` est partagé par le process serveur,
    pas par connexion."""
    return JSONResponse(
        {
            "servers": {
                "comfyui": _port_open(8188),
                "chatterbox": _port_open(8189),
            },
            "run": (
                {"project_id": _run_state.project_id, "done": _run_state.done}
                if _run_state is not None
                else None
            ),
        }
    )


@app.post("/api/projects")
async def api_new_project(
    id: str = Form(...),
    lang: str = Form("en"),
    script: UploadFile = File(...),
) -> JSONResponse:
    project_id = id.strip()
    if not _ID_RE.match(project_id):
        return JSONResponse(
            {"error": "id : lettres, chiffres, - et _ seulement"}, status_code=400
        )
    if (PROJECTS_DIR / project_id).exists():
        return JSONResponse(
            {"error": f"le projet '{project_id}' existe déjà"}, status_code=400
        )

    content = (await script.read()).decode("utf-8", errors="replace").strip()
    if not content:
        return JSONResponse({"error": "fichier .txt vide"}, status_code=400)

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    script_path = PROJECTS_DIR / f"_tmp_{project_id}.txt"
    script_path.write_text(content, encoding="utf-8")
    try:
        result = subprocess.run(
            [str(PYTHON), "-m", "pipeline", "new", project_id,
             "--script", str(script_path), "--lang", lang],
            cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        )
    finally:
        script_path.unlink(missing_ok=True)

    if result.returncode != 0:
        return JSONResponse({"error": result.stderr[-2000:]}, status_code=400)
    return JSONResponse({"ok": True, "output": result.stdout})


@app.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str) -> JSONResponse:
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.is_dir():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    # Irréversible : la confirmation vit côté UI (confirm() avant l'appel),
    # pas ici — cet endpoint fait ce qu'on lui demande, sans redemander.
    shutil.rmtree(project_dir)
    return JSONResponse({"ok": True})


@app.get("/api/projects/{project_id}/images")
def api_images(project_id: str) -> JSONResponse:
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    img_dir = PROJECTS_DIR / project_id / "images"
    if not img_dir.exists():
        return JSONResponse([])
    return JSONResponse(sorted(p.name for p in img_dir.glob("*.png")))


def _write_atomic(path: Path, data: bytes | str) -> None:
    """Même logique que `pipeline.project.write_atomic`, dupliquée ici : le
    webui ne dépend d'aucun module interne du pipeline (voir _STEPS ci-dessus)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(data, str):
        tmp.write_text(data, encoding="utf-8")
    else:
        tmp.write_bytes(data)
    tmp.replace(path)


def _review_state_file(project_id: str) -> Path:
    return PROJECTS_DIR / project_id / "review.json"


@app.get("/api/projects/{project_id}/review/flags")
def api_review_flags_get(project_id: str) -> JSONResponse:
    """État de revue persisté : scènes marquées et prompts réécrits à la main.

    Sans ça, tout se perdait au rechargement de l'onglet — `reviewFlagged` et
    `reviewEditedPrompts` ne vivaient qu'en mémoire du navigateur, et seules
    les scènes DÉJÀ régénérées étaient restaurées (depuis /api/review/status).
    Marquer 40 images puis fermer la page par mégarde effaçait le travail.

    Un artefact JSON dans le dossier du projet, comme scenes.json ou
    images.json : inspectable, versionnable, et il survit à tout le reste.
    """
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    path = _review_state_file(project_id)
    if not path.exists():
        return JSONResponse({"flagged": [], "prompts": {}})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Fichier corrompu : repartir vide plutôt que casser la page de revue.
        return JSONResponse({"flagged": [], "prompts": {}})
    return JSONResponse({
        "flagged": sorted({int(i) for i in data.get("flagged", [])}),
        "prompts": {str(k): str(v) for k, v in (data.get("prompts") or {}).items()},
    })


@app.put("/api/projects/{project_id}/review/flags")
async def api_review_flags_put(project_id: str, request: Request) -> JSONResponse:
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    if not (PROJECTS_DIR / project_id).is_dir():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    body = await request.json()
    try:
        flagged = sorted({int(i) for i in body.get("flagged", [])})
    except (TypeError, ValueError):
        return JSONResponse({"error": "flagged doit être une liste d'entiers"}, status_code=400)
    prompts = {str(k): str(v) for k, v in (body.get("prompts") or {}).items()}
    _write_atomic(
        _review_state_file(project_id),
        json.dumps(
            {"flagged": flagged, "prompts": prompts, "updated_at": datetime.now().isoformat(timespec="seconds")},
            ensure_ascii=False,
            indent=1,
        ).encode("utf-8"),
    )
    return JSONResponse({"ok": True, "flagged": len(flagged)})


@app.get("/api/projects/{project_id}/review")
def api_review_data(project_id: str) -> JSONResponse:
    """Une entrée par scène qui a déjà une image — assez pour la page de
    revue : image, éventuelle erreur de génération, et le prompt actuel
    (éditable avant régénération). `api_images` reste tel quel, pour la
    galerie en lecture seule de la page projet."""
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    project_dir = PROJECTS_DIR / project_id
    images_file = project_dir / "images.json"
    if not images_file.exists():
        return JSONResponse([])
    images = json.loads(images_file.read_text(encoding="utf-8"))["images"]

    prompts: dict[str, str] = {}
    prompts_file = project_dir / "prompts.json"
    if prompts_file.exists():
        prompts = json.loads(prompts_file.read_text(encoding="utf-8"))["prompts"]

    texts: dict[str, str] = {}
    scenes_file = project_dir / "scenes.json"
    if scenes_file.exists():
        for s in json.loads(scenes_file.read_text(encoding="utf-8"))["scenes"]:
            texts[str(s["index"])] = s["text"]

    out = [
        {
            "index": int(idx),
            "image_path": info["image_path"],
            "error_code": info.get("error_code"),
            "prompt": prompts.get(idx, ""),
            "text": texts.get(idx, ""),
        }
        for idx, info in images.items()
        if info.get("image_path")
    ]
    out.sort(key=lambda s: s["index"])
    return JSONResponse(out)


@app.api_route("/media_file/{project_id}/{filename:path}", methods=["GET", "HEAD"])
def media_file(project_id: str, filename: str):
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    project_dir = (PROJECTS_DIR / project_id).resolve()
    path = (project_dir / filename).resolve()
    # filename vient de l'URL : sans ce garde-fou, un "../../.." remonterait
    # hors du dossier du projet et lirait n'importe quel fichier du disque.
    if project_dir not in path.parents and path != project_dir:
        return JSONResponse({"error": "chemin invalide"}, status_code=400)
    if not path.is_file():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    return FileResponse(path)


def _character_refs_env(opts: dict) -> str | None:
    """Valeur de COMFY_CHARACTER_REFS : les références cochées, sinon le
    repli DEFAULT_CHARACTER_REF s'il existe."""
    refs = opts.get("character_refs") or []
    if not refs and (CHARACTERS_DIR / DEFAULT_CHARACTER_REF).exists():
        refs = [DEFAULT_CHARACTER_REF]
    return ",".join(str(CHARACTERS_DIR / r) for r in refs) or None


def _route_image_model(env: dict, image_model: str) -> None:
    """Avec HiDream, les scènes partent sur le GPU loué (COMFY_HIDREAM_URL,
    à mettre dans .env quand le pod tourne) ; les références restent à Klein
    sur le ComfyUI local (voir REFS_MODEL dans providers/local.py). COMFY_URL
    n'est pas touché dans .env : il reste celui du ComfyUI local pour les
    runs Klein."""
    if image_model != "hidream":
        return
    pod = os.environ.get("COMFY_HIDREAM_URL", "").strip()
    if pod:
        env["COMFY_REFS_URL"] = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
        env["COMFY_URL"] = pod


def _build_env(opts: dict) -> dict:
    """Traduit les options du formulaire en variables d'environnement du pipeline.

    Toutes les variables lues par `steps/tts.py` et `providers/local.py`, au
    même endroit que la doc `RUN.md` — pas de nouvelle convention, juste un
    formulaire par-dessus ce qui existe déjà.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    tts_provider = opts.get("tts_provider", "piper")
    env["PIPELINE_TTS_PROVIDER"] = tts_provider
    if tts_provider == "chatterbox":
        env["CHATTERBOX_LANGUAGES"] = opts.get("chatterbox_languages", "en")
    elif tts_provider == "fishaudio":
        if opts.get("fish_api_key"):
            env["FISH_API_KEY"] = opts["fish_api_key"]
        if opts.get("fish_voice_id"):
            env["FISH_VOICE_ID"] = opts["fish_voice_id"]

    # Vitesse du montage final (ffmpeg.py::SPEED). Bornée : en dessous de 0.5
    # ou au-dessus de 2.0 le rendu devient inécoutable, et une valeur absurde
    # saisie par erreur ne doit pas coûter un rendu complet pour rien.
    try:
        speed = float(opts.get("speed", 1.25))
    except (TypeError, ValueError):
        speed = 1.25
    env["PIPELINE_SPEED"] = str(min(2.0, max(0.5, speed)))

    image_model = opts.get("image_model", "sdxl")
    env["COMFY_MODEL"] = image_model
    if image_model in ("flux2", "hidream"):
        refs = _character_refs_env(opts)
        if refs:
            env["COMFY_CHARACTER_REFS"] = refs
        env["COMFY_TIMEOUT"] = "600"
    _route_image_model(env, image_model)

    return env


async def _execute_run(state: RunState, opts: dict) -> None:
    """Fait tourner `pipeline run` jusqu'au bout, quoi qu'il arrive côté client.

    Ni un onglet fermé, ni une mise en veille, ni une coupure réseau n'arrête
    ce sous-processus : rien ici n'observe l'état d'une connexion WebSocket.
    """
    env = _build_env(opts)
    args = [str(PYTHON), "-u", "-m", "pipeline", "run", state.project_id]
    if opts.get("sample"):
        args.append("--sample")
    if opts.get("draft"):
        args.append("--draft")
    if opts.get("force") in _STEPS:
        args += ["--force", opts["force"]]

    try:
        state.process = subprocess.Popen(
            args, cwd=str(ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, state.process.stdout.readline)
            if not line:
                break
            state.publish({"type": "log", "line": line.rstrip()})
        code = state.process.wait()
    except Exception as exc:  # défensif : ex. python.exe introuvable
        state.publish({"type": "error", "line": str(exc)})
        code = -1

    state.process = None
    state.done = True
    state.code = code
    state.publish({"type": "done", "code": code})


@app.post("/api/projects/{project_id}/run")
async def api_start_run(project_id: str, request: Request) -> JSONResponse:
    """Démarre un run en tâche de fond, détaché de cette requête HTTP.

    Le corps de la réponse ne contient que l'accusé de lancement — pour
    suivre la progression, se connecter à `/ws/run/{project_id}` séparément.
    """
    global _run_state
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    if not (PROJECTS_DIR / project_id).is_dir():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    # Posé avant tout `await` : aucune fenêtre de course où deux requêtes
    # concurrentes passeraient toutes les deux ce garde-fou.
    if _run_state is not None and not _run_state.done:
        return JSONResponse({"error": "un run est déjà en cours"}, status_code=409)

    opts = await request.json()
    if opts.get("force") and opts["force"] not in _STEPS:
        return JSONResponse(
            {"error": f"étape de forçage inconnue : {opts['force']!r} (parmi {', '.join(_STEPS)})"},
            status_code=400,
        )
    state = RunState(project_id, opts)
    _run_state = state
    asyncio.create_task(_execute_run(state, opts))
    return JSONResponse({"ok": True})


@app.post("/api/projects/{project_id}/schedule")
async def api_schedule_run(project_id: str, request: Request) -> JSONResponse:
    """Programme un run pour plus tard — mêmes options que `.../run`, plus
    `when` (date/heure locale, format `datetime-local` du navigateur)."""
    global _scheduled
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    if not (PROJECTS_DIR / project_id).is_dir():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    if _scheduled is not None:
        return JSONResponse(
            {"error": f"une génération est déjà programmée pour '{_scheduled.project_id}'"},
            status_code=409,
        )

    body = await request.json()
    try:
        when = datetime.fromisoformat(body.get("when", ""))
    except ValueError:
        return JSONResponse({"error": "date/heure invalide"}, status_code=400)
    if when <= datetime.now():
        return JSONResponse({"error": "l'heure programmée doit être dans le futur"}, status_code=400)
    if body.get("force") and body["force"] not in _STEPS:
        return JSONResponse(
            {"error": f"étape de forçage inconnue : {body['force']!r} (parmi {', '.join(_STEPS)})"},
            status_code=400,
        )

    opts = {k: v for k, v in body.items() if k != "when"}
    _scheduled = ScheduledRun(project_id, opts, when)
    return JSONResponse({"ok": True, "when": when.isoformat()})


@app.delete("/api/projects/{project_id}/schedule")
def api_cancel_schedule(project_id: str) -> JSONResponse:
    global _scheduled
    if _scheduled is None or _scheduled.project_id != project_id:
        return JSONResponse({"error": "rien de programmé pour ce projet"}, status_code=404)
    _scheduled = None
    return JSONResponse({"ok": True})


# ---------- File de nuit : API ----------
# Handlers `async def` à dessein : ils tournent sur le thread de la boucle,
# comme `_queue_tick`, donc jamais en même temps que lui (voir QUEUE_FILE).

def _queue_payload() -> dict:
    """La file + une estimation par élément (`eta_s`, `eta_end`), calculée à
    la volée et jamais écrite dans `queue.json`. Les heures de fin sont
    cumulées dans l'ordre de la file, à partir de maintenant ou de l'heure
    de départ si elle est dans le futur."""
    now = datetime.now()
    t = now
    if not _queue["paused"] and _queue.get("start_at"):
        t = max(now, datetime.fromisoformat(_queue["start_at"]))
    items = []
    for it in _queue["items"]:
        it = dict(it)
        if it["status"] in ("pending", "running"):
            state = _run_state if it["status"] == "running" and it["id"] == _queue_current else None
            est = _estimate(it["project_id"], it["opts"], state)
            if est is not None:
                t = t + timedelta(seconds=est["remaining_s"])
                it["eta_s"] = est["remaining_s"]
                it["eta_end"] = t.isoformat(timespec="minutes")
                it["progress"] = est["images"]
        items.append(it)
    return {**_queue, "items": items, "current": _queue_current, "eta_end": t.isoformat(timespec="minutes")}


@app.get("/api/projects/{project_id}/eta")
async def api_project_eta(project_id: str) -> JSONResponse:
    """Temps restant du run en cours sur ce projet, s'il y en a un."""
    if _run_state is None or _run_state.done or _run_state.project_id != project_id:
        return JSONResponse({"active": False})
    est = _estimate(project_id, _run_state.opts, _run_state) or {}
    return JSONResponse({"active": True, **est})


@app.get("/api/queue")
async def api_queue_get() -> JSONResponse:
    return JSONResponse(_queue_payload())


@app.post("/api/queue")
async def api_queue_add(request: Request) -> JSONResponse:
    """Ajoute un projet en fin de file — mêmes options que `.../run`."""
    body = await request.json()
    project_id = str(body.get("project_id", ""))
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    if not (PROJECTS_DIR / project_id).is_dir():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    if any(it["project_id"] == project_id and it["status"] in ("pending", "running") for it in _queue["items"]):
        return JSONResponse({"error": f"'{project_id}' est déjà dans la file"}, status_code=409)
    opts = {k: v for k, v in body.items() if k != "project_id" and k not in _QUEUE_SECRET_OPTS}
    if opts.get("force") and opts["force"] not in _STEPS:
        return JSONResponse({"error": f"étape de forçage inconnue : {opts['force']!r}"}, status_code=400)
    _queue["items"].append(
        {
            "id": os.urandom(6).hex(),
            "project_id": project_id,
            "opts": opts,
            "status": "pending",
            "added_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _queue_save()
    return JSONResponse(_queue_payload())


@app.delete("/api/queue/{item_id}")
async def api_queue_remove(item_id: str) -> JSONResponse:
    item = _queue_item(item_id)
    if item is None:
        return JSONResponse({"error": "élément introuvable"}, status_code=404)
    if item["status"] == "running":
        return JSONResponse({"error": "en cours — l'arrêter depuis sa page projet"}, status_code=409)
    _queue["items"].remove(item)
    _queue_save()
    return JSONResponse(_queue_payload())


@app.post("/api/queue/{item_id}/move")
async def api_queue_move(item_id: str, request: Request) -> JSONResponse:
    """`{"delta": -1}` remonte l'élément d'un cran, `1` le descend."""
    item = _queue_item(item_id)
    if item is None:
        return JSONResponse({"error": "élément introuvable"}, status_code=404)
    delta = int((await request.json()).get("delta", 0))
    items = _queue["items"]
    i = items.index(item)
    j = max(0, min(len(items) - 1, i + delta))
    items.insert(j, items.pop(i))
    _queue_save()
    return JSONResponse(_queue_payload())


@app.post("/api/queue/retry/{item_id}")
async def api_queue_retry(item_id: str) -> JSONResponse:
    item = _queue_item(item_id)
    if item is None or item["status"] not in ("done", "failed"):
        return JSONResponse({"error": "seul un élément terminé ou en échec se relance"}, status_code=409)
    item["status"] = "pending"
    for k in ("code", "error", "started_at", "ended_at"):
        item.pop(k, None)
    _queue_save()
    return JSONResponse(_queue_payload())


@app.post("/api/queue/start")
async def api_queue_start(request: Request) -> JSONResponse:
    """Démarre la file : `when` vide = tout de suite, sinon date/heure locale
    au format `datetime-local` du navigateur."""
    when_raw = str((await request.json()).get("when", "") or "")
    if when_raw:
        try:
            when = datetime.fromisoformat(when_raw)
        except ValueError:
            return JSONResponse({"error": "date/heure invalide"}, status_code=400)
        _queue["start_at"] = when.isoformat(timespec="minutes")
    else:
        _queue["start_at"] = None
    _queue["paused"] = False
    _queue_save()
    return JSONResponse(_queue_payload())


@app.post("/api/queue/pause")
async def api_queue_pause() -> JSONResponse:
    """Met la file en pause : le run en cours va à son terme, aucun suivant
    ne démarre."""
    _queue["paused"] = True
    _queue_save()
    return JSONResponse(_queue_payload())


@app.post("/api/queue/clear")
async def api_queue_clear() -> JSONResponse:
    """Retire les éléments terminés ou en échec."""
    _queue["items"] = [it for it in _queue["items"] if it["status"] in ("pending", "running")]
    _queue_save()
    return JSONResponse(_queue_payload())


@app.post("/api/projects/{project_id}/stop")
def api_stop_run(project_id: str) -> JSONResponse:
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    if _run_state is None or _run_state.done or _run_state.project_id != project_id:
        return JSONResponse({"error": "aucun run en cours pour ce projet"}, status_code=404)
    if _run_state.process is not None:
        # `Popen.terminate()` ne tue que ce process-là — pas les workers que
        # `render.py` lance lui-même via ProcessPoolExecutor. Sur Windows ça
        # laisse une dizaine de python.exe orphelins par run interrompu (vu en
        # pratique : 22 processus zombies accumulés en une session). `taskkill
        # /T` tue l'arbre entier au lieu du seul process direct.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(_run_state.process.pid)],
            capture_output=True,
        )
    return JSONResponse({"ok": True})


@app.get("/api/projects/{project_id}/run")
def api_run_status(project_id: str) -> JSONResponse:
    """Permet à un client qui (re)charge la page de savoir s'il doit se
    rattacher à un run déjà en cours plutôt que d'en proposer un nouveau."""
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    active = _run_state is not None and not _run_state.done and _run_state.project_id == project_id
    return JSONResponse({"active": active})


@app.websocket("/ws/run/{project_id}")
async def ws_run(websocket: WebSocket, project_id: str) -> None:
    """Observe le run en cours pour ce projet — ne le lance ni ne l'arrête.

    Se déconnecter (onglet fermé, veille, coupure réseau) n'a plus aucun
    effet sur le sous-processus, lancé et suivi par `_execute_run`
    indépendamment de cette connexion. Une reconnexion rejoue d'abord le
    journal déjà produit, puis continue en direct.
    """
    await websocket.accept()

    if not _ID_RE.match(project_id):
        await websocket.send_json({"type": "error", "line": "id invalide"})
        await websocket.close()
        return

    state = _run_state
    if state is None or state.project_id != project_id:
        await websocket.send_json({"type": "error", "line": "aucun run en cours pour ce projet"})
        await websocket.close()
        return

    for line in state.log:
        await websocket.send_json({"type": "log", "line": line})
    if state.done:
        await websocket.send_json({"type": "done", "code": state.code})
        await websocket.close()
        return

    queue: asyncio.Queue = asyncio.Queue()
    state.subscribers.add(queue)
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("type") == "done":
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.subscribers.discard(queue)


async def _execute_thumbnails(state: ThumbState, opts: dict) -> None:
    """Lance _gen_thumbnail.py en sous-processus et republie chaque ligne
    JSON qu'il imprime — une par image terminée, plus un "done" final."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["COMFY_MODEL"] = opts.get("image_model", "flux2")
    _route_image_model(env, env["COMFY_MODEL"])
    refs = opts.get("character_refs") or []
    if refs:
        env["COMFY_CHARACTER_REFS"] = ",".join(str(CHARACTERS_DIR / r) for r in refs)
    env["COMFY_TIMEOUT"] = "600"

    batch_dir = THUMBNAILS_DIR / state.batch_id
    seed_base = int.from_bytes(os.urandom(4), "big")

    try:
        state.process = await asyncio.create_subprocess_exec(
            str(PYTHON), "-u", str(ROOT / "webui" / "_gen_thumbnail.py"),
            state.prompt, str(batch_dir), str(state.count), str(seed_base),
            cwd=str(ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        assert state.process.stdout is not None
        while True:
            line = await state.process.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            state.publish(msg)
        await state.process.wait()
    except Exception as exc:  # défensif : ex. python.exe introuvable
        state.publish({"type": "error", "message": str(exc)})

    state.process = None
    state.done = True
    state.publish({"type": "done"})


@app.post("/api/thumbnails")
async def api_start_thumbnails(request: Request) -> JSONResponse:
    """Démarre un lot de miniatures en tâche de fond — même détachement que
    `/api/projects/{id}/run` : la requête HTTP ne fait que le lancer."""
    global _thumb_state
    if _thumb_state is not None and not _thumb_state.done:
        return JSONResponse({"error": "une génération de miniatures est déjà en cours"}, status_code=409)

    opts = await request.json()
    prompt = str(opts.get("prompt", "")).strip()
    if not prompt:
        return JSONResponse({"error": "prompt vide"}, status_code=400)
    try:
        count = int(opts.get("count", 30))
    except (TypeError, ValueError):
        return JSONResponse({"error": "count invalide"}, status_code=400)
    count = max(1, min(count, 60))  # plafond raisonnable : 60 images ~ 1h en FLUX.2

    batch_id = time.strftime("%Y%m%d-%H%M%S")
    state = ThumbState(batch_id, prompt, count)
    _thumb_state = state
    asyncio.create_task(_execute_thumbnails(state, opts))
    return JSONResponse({"ok": True, "batch_id": batch_id, "count": count})


@app.post("/api/thumbnails/stop")
def api_stop_thumbnails() -> JSONResponse:
    if _thumb_state is None or _thumb_state.done:
        return JSONResponse({"error": "aucune génération en cours"}, status_code=404)
    if _thumb_state.process is not None:
        # Même tree-kill que api_stop_run : un simple .terminate() laisserait
        # filer l'appel HTTP en cours vers ComfyUI sans le couper proprement.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(_thumb_state.process.pid)],
            capture_output=True,
        )
    return JSONResponse({"ok": True})


@app.get("/api/thumbnails/status")
def api_thumbnails_status() -> JSONResponse:
    if _thumb_state is None:
        return JSONResponse({"active": False, "batch_id": None, "images": [], "count": 0})
    return JSONResponse(
        {
            "active": not _thumb_state.done,
            "batch_id": _thumb_state.batch_id,
            "prompt": _thumb_state.prompt,
            "images": _thumb_state.images,
            "count": _thumb_state.count,
        }
    )


@app.websocket("/ws/thumbnails")
async def ws_thumbnails(websocket: WebSocket) -> None:
    """Observe le lot de miniatures en cours — même principe que /ws/run."""
    await websocket.accept()

    state = _thumb_state
    if state is None:
        await websocket.send_json({"type": "error", "message": "aucune génération en cours"})
        await websocket.close()
        return

    for i, fname in enumerate(state.images):
        await websocket.send_json({"type": "image", "index": i, "file": fname})
    if state.done:
        await websocket.send_json({"type": "done"})
        await websocket.close()
        return

    queue: asyncio.Queue = asyncio.Queue()
    state.subscribers.add(queue)
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("type") == "done":
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.subscribers.discard(queue)


@app.api_route("/thumb_file/{batch_id}/{filename}", methods=["GET", "HEAD"])
def thumb_file(batch_id: str, filename: str):
    if not _ID_RE.match(batch_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    batch_dir = (THUMBNAILS_DIR / batch_id).resolve()
    path = (batch_dir / filename).resolve()
    if batch_dir not in path.parents and path != batch_dir:
        return JSONResponse({"error": "chemin invalide"}, status_code=400)
    if not path.is_file():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    return FileResponse(path)


async def _execute_review(state: ReviewState, opts: dict) -> None:
    """Lance _gen_review.py en sous-processus et republie chaque ligne JSON
    qu'il imprime — une par variante terminée, plus un "done" final. Même
    principe que _execute_thumbnails."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["COMFY_MODEL"] = opts.get("image_model", "flux2")
    _route_image_model(env, env["COMFY_MODEL"])
    refs = _character_refs_env(opts)
    if refs:
        env["COMFY_CHARACTER_REFS"] = refs
    env["COMFY_TIMEOUT"] = "600"

    batch_dir = REVIEW_DIR / state.project_id / state.batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = batch_dir / "jobs.json"
    _write_atomic(jobs_file, json.dumps(state.jobs, ensure_ascii=False))
    seed_base = int.from_bytes(os.urandom(4), "big")

    try:
        state.process = await asyncio.create_subprocess_exec(
            str(PYTHON), "-u", str(ROOT / "webui" / "_gen_review.py"),
            str(jobs_file), str(batch_dir), str(state.variants), str(seed_base), state.project_id,
            cwd=str(ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        assert state.process.stdout is not None
        while True:
            line = await state.process.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            state.publish(msg)
        await state.process.wait()
    except Exception as exc:  # défensif : ex. python.exe introuvable
        state.publish({"type": "error", "message": str(exc)})

    state.process = None
    state.done = True
    state.publish({"type": "done"})


@app.post("/api/projects/{project_id}/review/regenerate")
async def api_review_regenerate(project_id: str, request: Request) -> JSONResponse:
    """Démarre la régénération des scènes marquées — même détachement que
    `/api/thumbnails` : la requête HTTP ne fait que lancer le lot."""
    global _review_state
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    if not (PROJECTS_DIR / project_id).is_dir():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    if _review_state is not None and not _review_state.done:
        return JSONResponse({"error": "une régénération est déjà en cours"}, status_code=409)

    body = await request.json()
    raw_jobs = body.get("jobs") or []
    if not raw_jobs:
        return JSONResponse({"error": "aucune scène marquée"}, status_code=400)
    jobs: list[dict] = []
    for job in raw_jobs:
        try:
            index = int(job["index"])
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"error": "index de scène invalide"}, status_code=400)
        prompt = str(job.get("prompt", "")).strip()
        if not prompt:
            return JSONResponse({"error": f"prompt vide pour la scène {index}"}, status_code=400)
        jobs.append({"index": index, "prompt": prompt})

    try:
        variants = max(1, min(int(body.get("variants", 4)), 8))
    except (TypeError, ValueError):
        return JSONResponse({"error": "nombre de variantes invalide"}, status_code=400)

    batch_id = time.strftime("%Y%m%d-%H%M%S")
    state = ReviewState(batch_id, project_id, jobs, variants)
    _review_state = state
    asyncio.create_task(_execute_review(state, body))
    return JSONResponse({"ok": True, "batch_id": batch_id, "variants": variants})


@app.post("/api/review/stop")
def api_review_stop() -> JSONResponse:
    if _review_state is None or _review_state.done:
        return JSONResponse({"error": "aucune régénération en cours"}, status_code=404)
    if _review_state.process is not None:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(_review_state.process.pid)],
            capture_output=True,
        )
    return JSONResponse({"ok": True})


@app.get("/api/review/status")
def api_review_status() -> JSONResponse:
    """Permet à la page de revue, rechargée pendant qu'un lot tourne, de s'y
    rattacher plutôt que d'en proposer un nouveau — même rôle que
    `/api/thumbnails/status`."""
    if _review_state is None:
        return JSONResponse({"active": False, "batch_id": None, "project_id": None, "images": {}, "variants": 0})
    return JSONResponse(
        {
            "active": not _review_state.done,
            "batch_id": _review_state.batch_id,
            "project_id": _review_state.project_id,
            "images": _review_state.images,
            "variants": _review_state.variants,
        }
    )


@app.websocket("/ws/review/{batch_id}")
async def ws_review(websocket: WebSocket, batch_id: str) -> None:
    """Observe le lot de régénération en cours — même principe que /ws/thumbnails."""
    await websocket.accept()

    state = _review_state
    if state is None or state.batch_id != batch_id:
        await websocket.send_json({"type": "error", "message": "lot introuvable"})
        await websocket.close()
        return

    for scene_index, files in state.images.items():
        for variant, fname in enumerate(files):
            await websocket.send_json(
                {"type": "image", "scene_index": scene_index, "variant": variant, "file": fname}
            )
    if state.done:
        await websocket.send_json({"type": "done"})
        await websocket.close()
        return

    queue: asyncio.Queue = asyncio.Queue()
    state.subscribers.add(queue)
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("type") == "done":
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.subscribers.discard(queue)


@app.api_route("/review_file/{project_id}/{batch_id}/{filename:path}", methods=["GET", "HEAD"])
def review_file(project_id: str, batch_id: str, filename: str):
    if not _ID_RE.match(project_id) or not _ID_RE.match(batch_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    batch_dir = (REVIEW_DIR / project_id / batch_id).resolve()
    path = (batch_dir / filename).resolve()
    if batch_dir not in path.parents and path != batch_dir:
        return JSONResponse({"error": "chemin invalide"}, status_code=400)
    if not path.is_file():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    return FileResponse(path)


@app.post("/api/projects/{project_id}/review/apply")
async def api_review_apply(project_id: str, request: Request) -> JSONResponse:
    """Fait d'une variante candidate l'image officielle de la scène : copiée
    dans `images/NNNN.png`, `images.json` mis à jour, et `prompts.json` aussi
    si le prompt a été édité — sinon la correction ne survivrait pas à un
    simple rechargement de la page projet."""
    if not _ID_RE.match(project_id):
        return JSONResponse({"error": "id invalide"}, status_code=400)
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.is_dir():
        return JSONResponse({"error": "introuvable"}, status_code=404)

    body = await request.json()
    try:
        index = int(body["index"])
        variant = int(body["variant"])
    except (KeyError, TypeError, ValueError):
        return JSONResponse({"error": "index/variant invalide"}, status_code=400)
    batch_id = str(body.get("batch_id", ""))
    if not _ID_RE.match(batch_id):
        return JSONResponse({"error": "batch_id invalide"}, status_code=400)

    review_root = (REVIEW_DIR / project_id).resolve()
    source = (review_root / batch_id / f"{index:04d}" / f"{variant:03d}.png").resolve()
    # Même garde-fou que media_file/thumb_file : batch_id/index/variant
    # viennent tous du corps de la requête, jamais fiables tels quels.
    if review_root not in source.parents or not source.is_file():
        return JSONResponse({"error": "variante introuvable"}, status_code=404)

    dest = project_dir / "images" / f"{index:04d}.png"
    _write_atomic(dest, source.read_bytes())

    images_file = project_dir / "images.json"
    images_data = (
        json.loads(images_file.read_text(encoding="utf-8"))
        if images_file.exists()
        else {"provider": "comfy", "images": {}}
    )
    images_data.setdefault("images", {})[str(index)] = {
        "image_path": f"images/{index:04d}.png",
        "error_code": None,
    }
    _write_atomic(images_file, json.dumps(images_data, ensure_ascii=False, indent=1))

    prompt = str(body.get("prompt", "")).strip()
    if prompt:
        prompts_file = project_dir / "prompts.json"
        prompts_data = (
            json.loads(prompts_file.read_text(encoding="utf-8"))
            if prompts_file.exists()
            else {"via": "webui-review", "summary": "", "prompts": {}}
        )
        prompts_data.setdefault("prompts", {})[str(index)] = prompt
        _write_atomic(prompts_file, json.dumps(prompts_data, ensure_ascii=False, indent=1))

    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn

    print(f"racine du dépôt : {ROOT}")
    print("http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765)
