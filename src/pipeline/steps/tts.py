"""Étape 1 — TTS. `script.txt` → `audio.wav` (SPEC §4).

Trois fournisseurs, choisis par `PIPELINE_TTS_PROVIDER` (défaut `piper`) :

- `piper` : local, CPU, gratuit, toujours disponible — le filet de sécurité.
- `chatterbox` : serveur HTTP (Chatterbox TTS Server, CPU, sur la machine
  Windows où tourne aussi ComfyUI — voir `TTS-LOCAL-MODEL.md`). Bien meilleure
  qualité de voix, mais dépend d'un serveur externe. Anglais seul sur ce
  serveur (voir CHATTERBOX_LANGUAGES) — pas de modèle français disponible.
- `fishaudio` : API hébergée (S2 Pro). Payante (~0,015 $/1000 caractères),
  clonage de voix inclus, multilingue sans configuration — utilisée pour le
  français, faute d'équivalent local viable trouvé le 31/08 (Zonos : 24× temps
  réel sur cette carte AMD faute de torch.compile fonctionnel ; Chatterbox :
  pas de modèle français). Testée en direct via l'API le 01/09 : format
  msgpack obligatoire, le JSON+base64 documenté dans leur schéma OpenAPI est
  rejeté par le serveur (« Reference Audio is not valid ») malgré un schéma
  qui semble l'autoriser.

Piper 1.3 n'expose aucun réglage de silence inter-phrases : il découpe le texte
lui-même et concatène les chunks bout à bout, ce qui donne une narration sans
respiration. On synthétise donc segment par segment et on insère les silences
nous-mêmes.

Ce n'est pas qu'une question de confort d'écoute : les silences sont le signal
de premier ordre de la segmentation (SPEC §4.3.a), et les durées ci-dessous
sont calibrées sur le barème des scores. Seules les fins de phrase franchissent
le seuil des 400 ms — les virgules et les points-virgules restent en dessous,
pour que leur score vienne de la ponctuation seule et ne soit pas compté deux
fois. Chatterbox gère sa propre prosodie (modèle de clonage, pas de synthèse
concaténative) : ce découpage ne s'applique qu'au provider `piper`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import wave
from pathlib import Path

import numpy as np

from ..language import LanguageConfig, for_project
from ..project import Options, Project, normalize, safe_wav_frames, write_atomic


def _wav_minutes_and_rate(path: Path) -> tuple[float, int]:
    """Durée réelle d'un WAV reçu d'un serveur tiers, pas celle déclarée —
    voir `project.safe_wav_frames`. Pour que ce qui s'affiche ici corresponde
    à ce que `align` mesurera juste après."""
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
    return safe_wav_frames(path) / rate / 60, rate

SENTENCE_SILENCE_MS = 450  # . ? ! — franchit le seuil « pause > 400 ms » (score 1.0)
CLAUSE_SILENCE_MS = 220  # : ; — et … — sous le seuil des 250 ms, délibérément
COMMA_SILENCE_MS = 120  # ,

# Découpe en gardant le délimiteur : les groupes de capture sont conservés par
# re.split, ce qui permet de savoir quel silence insérer après chaque segment.
_SPLIT = re.compile(r"([.!?]+|[:;—…]|,)")

_SILENCE_FOR = {
    ".": SENTENCE_SILENCE_MS,
    "!": SENTENCE_SILENCE_MS,
    "?": SENTENCE_SILENCE_MS,
    ":": CLAUSE_SILENCE_MS,
    ";": CLAUSE_SILENCE_MS,
    "—": CLAUSE_SILENCE_MS,
    "…": CLAUSE_SILENCE_MS,
    ",": COMMA_SILENCE_MS,
}


def _split_with_silences(text: str) -> list[tuple[str, int]]:
    """Découpe le texte en (segment prononçable, silence en ms à ajouter après)."""
    parts = _SPLIT.split(text)
    out: list[tuple[str, int]] = []
    # re.split alterne : texte, délimiteur, texte, délimiteur, …
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        if not chunk:
            continue
        # La ponctuation est réinjectée dans le texte prononcé : elle porte la
        # prosodie (une question ne s'intonne pas comme une affirmation).
        spoken = f"{chunk}{delim}"
        silence = _SILENCE_FOR.get(delim[:1], 0) if delim else SENTENCE_SILENCE_MS
        out.append((spoken, silence))
    return out


TTS_PROVIDER = os.environ.get("PIPELINE_TTS_PROVIDER", "piper")

CHATTERBOX_URL = os.environ.get("CHATTERBOX_URL", "http://127.0.0.1:8189")
CHATTERBOX_VOICE = os.environ.get("CHATTERBOX_VOICE", "Adrian.wav")
# ~2× le temps réel mesuré en CPU (voir TTS-LOCAL-MODEL.md) : le serveur doit
# tenir toute la durée du script, pas juste une phrase — sur 24 min de cible,
# ça peut prendre 45-50 min. Une marge large plutôt qu'un timeout qui coupe un
# appel de plusieurs dizaines de minutes à la dernière minute.
CHATTERBOX_TIMEOUT_S = float(os.environ.get("CHATTERBOX_TIMEOUT_S", "7200"))

# Le serveur ne renvoie AUCUNE erreur quand on lui demande une langue que le
# modèle chargé ne sait pas faire : il « réussit » silencieusement, dans la
# mauvaise langue (constaté à l'écoute — un script français lu en anglais,
# vérifié après coup : Chatterbox Turbo est anglais seul, le multilingue est
# un modèle distinct, jamais chargé sur ce serveur). Pas d'erreur serveur à
# intercepter, donc le garde-fou est ici : les langues que le modèle
# actuellement chargé sait vraiment faire, à mettre à jour si le modèle change.
CHATTERBOX_LANGUAGES = frozenset(
    os.environ.get("CHATTERBOX_LANGUAGES", "en").split(",")
)

FISHAUDIO_API_KEY = os.environ.get("FISH_API_KEY", "")
FISHAUDIO_VOICE_ID = os.environ.get("FISH_VOICE_ID", "")
FISHAUDIO_URL = os.environ.get("FISHAUDIO_URL", "https://api.fish.audio/v1/tts")
FISHAUDIO_TIMEOUT_S = float(os.environ.get("FISHAUDIO_TIMEOUT_S", "600"))


def _seed_for_project(project_id: str) -> int:
    """Même construction que `scenes.seed_for` : déterministe, pas `hash()`."""
    digest = hashlib.sha256(f"{project_id}:tts".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def synthesize_snippet(
    language: LanguageConfig,
    text: str,
    output: Path,
    provider: str | None = None,
    voice_id: str | None = None,
) -> None:
    """Synthétise un texte court en dehors du script principal — utilisé par
    `render.py` pour que le carton de palier ("Niveau 1 : Esclave") soit
    vraiment prononcé plutôt que posé sur du silence.

    `provider` : à passer explicitement quand l'appelant le connaît (voir
    `load_provider()`) — c'est la SEULE façon de garantir la même voix que la
    narration principale, `PIPELINE_TTS_PROVIDER` au moment de l'appel n'étant
    pas fiable (peut différer de celui utilisé pour générer `audio.wav`,
    voir `load_provider`). `None` retombe sur l'environnement, pour les
    appelants qui n'ont pas cette information.

    `voice_id` : même raisonnement que `provider`, pour Fish Audio. Le webui
    expose un champ « voix Fish » PAR RUN qui surcharge `FISH_VOICE_ID` du
    `.env` sans laisser de trace sur le disque ; sans cet argument l'annonce
    sortait dans la voix du `.env`, pas celle de la narration. `None` retombe
    sur l'environnement.

    Ne couvre que Fish Audio et Piper : Chatterbox retombe sur Piper (anglais
    seul sur ce serveur de toute façon, peu pertinent pour une annonce
    française). N'écrit jamais dans `project.audio` — fichier séparé, appelé
    hors du contrat d'étape normal (pas de reprise, pas d'invariant à tenir :
    c'est un aparté, pas une portion du script).
    """
    if provider is None:
        provider = os.environ.get("PIPELINE_TTS_PROVIDER", "piper")
    if provider == "fishaudio":
        _snippet_fishaudio(text, output, voice_id)
    else:
        _snippet_piper(language, text, output)


def _snippet_piper(language: LanguageConfig, text: str, output: Path) -> None:
    from piper import PiperVoice

    voice = PiperVoice.load(str(language.voice_path))
    rate = voice.config.sample_rate
    audio: list[np.ndarray] = [c.audio_float_array for c in voice.synthesize(text)]
    samples = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def _snippet_fishaudio(text: str, output: Path, voice_id: str | None = None) -> None:
    import httpx
    import ormsgpack

    # `voice_id` (celui qui a réellement généré `audio.wav`, relu de
    # `tts.json`) prime sur l'environnement — voir `synthesize_snippet`.
    reference_id = voice_id or FISHAUDIO_VOICE_ID
    if not FISHAUDIO_API_KEY or not reference_id:
        raise RuntimeError(
            "FISH_API_KEY/FISH_VOICE_ID manquant — impossible de faire "
            "prononcer l'annonce de palier par Fish Audio."
        )
    payload = {
        "text": text,
        "reference_id": reference_id,
        "format": "wav",
        "normalize": True,
    }
    response = httpx.post(
        FISHAUDIO_URL,
        content=ormsgpack.packb(payload),
        headers={
            "Authorization": f"Bearer {FISHAUDIO_API_KEY}",
            "Content-Type": "application/msgpack",
        },
        timeout=FISHAUDIO_TIMEOUT_S,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Fish Audio (annonce de palier) a répondu {response.status_code} : "
            f"{response.text[:300]}"
        )
    output.write_bytes(response.content)


def _run_piper(project: Project, language: LanguageConfig, text: str) -> None:
    from piper import PiperVoice

    voice = PiperVoice.load(str(language.voice_path))
    rate = voice.config.sample_rate

    audio: list[np.ndarray] = []
    for spoken, silence_ms in _split_with_silences(text):
        for chunk in voice.synthesize(spoken):
            audio.append(chunk.audio_float_array)
        if silence_ms:
            audio.append(np.zeros(rate * silence_ms // 1000, dtype=np.float32))

    samples = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    project.audio.parent.mkdir(parents=True, exist_ok=True)
    tmp = project.audio.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    tmp.replace(project.audio)

    print(
        f"  audio : {len(samples) / rate / 60:.1f} min, {rate} Hz mono "
        f"— voix {language.voice}"
    )


def _run_chatterbox(project: Project, language: LanguageConfig, text: str) -> None:
    import httpx

    if language.whisper not in CHATTERBOX_LANGUAGES:
        raise RuntimeError(
            f"le modèle Chatterbox actuellement chargé ne fait que "
            f"{sorted(CHATTERBOX_LANGUAGES)}, pas '{language.whisper}'. Le serveur "
            f"ne signale pas cette incompatibilité lui-même — il génère de "
            f"l'audio dans la mauvaise langue sans erreur. Charger un modèle "
            f"multilingue et mettre à jour CHATTERBOX_LANGUAGES, ou repasser "
            f"sur Piper pour ce projet (PIPELINE_TTS_PROVIDER=piper)."
        )

    # `language.whisper` est déjà un code ISO 639-1 (fr, en) — le même que le
    # serveur Chatterbox attend, testé directement (SPEC : une seule
    # définition de la langue par projet, pas un champ redondant de plus).
    payload = {
        "text": text,
        "predefined_voice_id": CHATTERBOX_VOICE,
        "language": language.whisper,
        "output_format": "wav",
        "seed": _seed_for_project(project.id),
        # Chunking géré côté serveur : un seul appel pour tout le script
        # (~20 000 caractères), pas un appel par phrase.
        "split_text": True,
    }
    try:
        response = httpx.post(
            f"{CHATTERBOX_URL}/tts", json=payload, timeout=CHATTERBOX_TIMEOUT_S
        )
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"serveur Chatterbox injoignable sur {CHATTERBOX_URL} — "
            f"vérifier qu'il tourne (voir TTS-LOCAL-MODEL.md) : {exc}"
        ) from None
    if response.status_code != 200:
        raise RuntimeError(
            f"Chatterbox a répondu {response.status_code} : {response.text[:300]}"
        )

    project.audio.parent.mkdir(parents=True, exist_ok=True)
    tmp = project.audio.with_suffix(".wav.tmp")
    tmp.write_bytes(response.content)
    tmp.replace(project.audio)

    minutes, rate = _wav_minutes_and_rate(project.audio)
    print(
        f"  audio : {minutes:.1f} min, {rate} Hz — Chatterbox ({CHATTERBOX_VOICE}, "
        f"langue {language.whisper})"
    )


def _run_fishaudio(project: Project, language: LanguageConfig, text: str) -> None:
    import httpx
    import ormsgpack

    if not FISHAUDIO_API_KEY:
        raise RuntimeError(
            "FISH_API_KEY manquant — générer une clé sur "
            "https://fish.audio/app/api-keys/ et créditer le solde API "
            "(distinct de l'abonnement) sur https://fish.audio/app/developers."
        )
    if not FISHAUDIO_VOICE_ID:
        raise RuntimeError(
            "FISH_VOICE_ID manquant — copier l'ID d'une voix depuis leur "
            "interface (paramètre reference_id de leur API)."
        )

    payload = {
        "text": text,
        "reference_id": FISHAUDIO_VOICE_ID,
        "format": "wav",
        "normalize": True,
    }
    # msgpack, pas JSON : le champ `audio` de leur schéma de référence est
    # déclaré en bytes côté serveur (format "binary" dans l'OpenAPI). Le
    # JSON+base64 que le schéma semble pourtant autoriser est rejeté en
    # pratique ("Reference Audio is not valid"), vérifié le 01/09. On ne
    # passe pas de référence audio ici (reference_id suffit), mais on garde
    # msgpack pour tout le corps par cohérence avec ce qui est prouvé marcher.
    try:
        response = httpx.post(
            FISHAUDIO_URL,
            content=ormsgpack.packb(payload),
            headers={
                "Authorization": f"Bearer {FISHAUDIO_API_KEY}",
                "Content-Type": "application/msgpack",
            },
            timeout=FISHAUDIO_TIMEOUT_S,
        )
    except httpx.ConnectError as exc:
        raise RuntimeError(f"API Fish Audio injoignable : {exc}") from None
    if response.status_code == 402:
        raise RuntimeError(
            "Solde API Fish Audio insuffisant — le crédit API est distinct du "
            "crédit d'abonnement, à recharger sur "
            "https://fish.audio/app/developers."
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"API Fish Audio a répondu {response.status_code} : "
            f"{response.text[:300]}"
        )

    project.audio.parent.mkdir(parents=True, exist_ok=True)
    tmp = project.audio.with_suffix(".wav.tmp")
    tmp.write_bytes(response.content)
    tmp.replace(project.audio)

    minutes, rate = _wav_minutes_and_rate(project.audio)
    print(
        f"  audio : {minutes:.1f} min, {rate} Hz — Fish Audio API "
        f"(voix {FISHAUDIO_VOICE_ID[:8]}…, langue {language.whisper})"
    )


def load_provider(project: Project) -> tuple[str | None, str | None]:
    """(provider, voice_id) ayant réellement généré `audio.wav`, relus depuis
    `tts.json`. `voice_id` n'est renseigné que pour Fish Audio.

    Sert à `render.py` pour l'annonce parlée du carton de palier : sans ça,
    elle retombait sur l'environnement AU MOMENT DU RENDU — `PIPELINE_TTS_
    PROVIDER`, mais aussi `FISH_VOICE_ID`, qui peuvent très bien différer de
    ceux utilisés pour la narration principale. Le webui expose un champ
    « voix Fish » PAR RUN (mémorisé côté navigateur, invisible depuis le
    disque) qui surcharge le `.env` : sans cette trace persistée, aucun moyen
    de savoir après coup quel identifiant a servi. Vu en pratique (20/09) :
    l'annonce du carton est sortie dans une voix Fish différente de celle de
    toute la narration.
    """
    meta_file = project.dir / "tts.json"
    if not meta_file.exists():
        return None, None
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    return data.get("provider"), data.get("fish_voice_id")


def run(project: Project, opts: Options) -> None:
    script = project.script.read_text(encoding="utf-8")
    text = normalize(script)
    # Persisté avant toute synthèse : c'est la source de vérité du pipeline
    # entier, y compris des deux invariants (SPEC §7).
    write_atomic(project.script_normalized, text)

    language = for_project(project.dir)

    if TTS_PROVIDER == "chatterbox":
        _run_chatterbox(project, language, text)
    elif TTS_PROVIDER == "fishaudio":
        _run_fishaudio(project, language, text)
    else:
        _run_piper(project, language, text)

    # Voir `load_provider()` : sans cette trace, l'annonce du carton de
    # palier (render.py) n'a aucun moyen fiable de savoir quelle voix a
    # servi à la narration principale — y compris l'ID Fish exact, que le
    # webui peut faire varier d'un run à l'autre indépendamment du `.env`.
    metadata: dict[str, str] = {"provider": TTS_PROVIDER}
    if TTS_PROVIDER == "fishaudio":
        metadata["fish_voice_id"] = FISHAUDIO_VOICE_ID
    write_atomic(
        project.dir / "tts.json",
        json.dumps(metadata, ensure_ascii=False, indent=1),
    )
