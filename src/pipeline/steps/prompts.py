"""Étape 4 — Prompts image. `scenes.json` → `prompts.json` (SPEC §4).

Appels LLM par lots de ~20 scènes, en parallèle : un seul appel pour 300 scènes
dérive en qualité et en format. Chaque lot reçoit la style bible, un résumé
global du script (cohérence narrative) et les 2 scènes précédentes (continuité
locale).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import cast_bible, location_bible, section_bible
from ..project import PROMPTS_DIR, Options, Project, write_atomic
from ..scenes import Scene, load, sample_indices
from ..style import StyleConfig
from ..style import load as load_style

BATCH_SIZE = 20
# Décrire une scène en 15-30 mots n'est pas une tâche qui a besoin d'Opus —
# c'est de la description factuelle courte, pas du raisonnement. Haiku va
# nettement plus vite pour ce travail, sur les ~15 lots d'un run complet.
MODEL = os.environ.get("PIPELINE_PROMPTS_MODEL", "haiku")
# Chaque appel lance un processus CLI complet (~300 Mo de binaire), pas une
# requête HTTP : on parallélise beaucoup moins qu'avec le SDK.
MAX_WORKERS = 3
RETRIES = 2
TIMEOUT_S = 300

# Repli exponentiel pour un plafond d'abonnement — distinct de RETRIES : une
# panne transitoire (CLI qui hoquette) mérite d'abandonner vite, un plafond
# se lève tout seul et mérite d'attendre, pas de compter comme un échec.
BACKOFF_START_S = 60
BACKOFF_MAX_S = 900
BACKOFF_FACTOR = 2

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Tout ce qui peut rater sur un lot sans que le run entier doive s'arrêter.
_RECOVERABLE = (
    ValueError,
    KeyError,
    json.JSONDecodeError,
    subprocess.SubprocessError,
    RuntimeError,
)

# Vu dans `payload["result"]` ou `stderr` quand le CLI signale un plafond —
# pas de format garanti, donc on reconnaît le vocabulaire plutôt qu'un code.
_RATE_LIMIT_MARKERS = ("usage limit", "rate limit", "overloaded", "credit balance too low")
_RATE_LIMIT_STATUSES = {429, 529}


class RateLimited(RuntimeError):
    """Le CLI a signalé un plafond d'abonnement — une pause à respecter, pas
    un échec à compter dans RETRIES."""


# Les lots tournent en parallèle (ThreadPoolExecutor) : sans ce verrou
# partagé, chaque thread qui détecte le plafond retenterait à son propre
# rythme et les re-déclencherait tous ensemble. Avec, ils convergent sur la
# même échéance de réveil.
_cooldown_lock = threading.Lock()
_cooldown_until = 0.0


def _enter_cooldown(delay_s: float) -> float:
    global _cooldown_until
    with _cooldown_lock:
        _cooldown_until = max(_cooldown_until, time.monotonic() + delay_s)
        return _cooldown_until


def _wait_out_cooldown() -> None:
    while True:
        with _cooldown_lock:
            remaining = _cooldown_until - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5.0))


def _looks_rate_limited(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _RATE_LIMIT_MARKERS)


def _binary() -> str:
    """Localise le CLI Claude Code.

    On passe par le CLI plutôt que par le SDK : il est couvert par l'abonnement
    Claude Pro, là où l'API se facture séparément. En contrepartie la
    parallélisation est limitée et les quotas sont ceux de l'abonnement.
    """
    override = os.environ.get("CLAUDE_BIN")
    if override:
        return override
    found = shutil.which("claude")
    if found:
        return found
    # Le binaire livré avec l'extension VS Code, quand le CLI n'est pas dans le PATH.
    candidates = sorted(
        (Path.home() / ".vscode-server" / "extensions").glob(
            "anthropic.claude-code-*/resources/native-binary/claude"
        )
    )
    if candidates:
        return str(candidates[-1])
    raise RuntimeError(
        "CLI Claude introuvable. Installer Claude Code, ou pointer CLAUDE_BIN "
        "sur le binaire. Le reste du pipeline reste utilisable : "
        "`run <id> --step images` puis `--step render` génèrent la vidéo en "
        "prenant le texte de la scène comme prompt (error_code=no_prompt)."
    )


def _ask(binary: str, prompt: str, max_tokens: int) -> str:
    del max_tokens  # le CLI ne l'expose pas ; la longueur est bornée par le prompt
    result = subprocess.run(
        [binary, "--model", MODEL, "-p", prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_S,
    )

    # Le code de sortie seul ne dit rien de fiable : une erreur *proprement*
    # rapportée (mauvais modèle, plafond atteint...) ressort quand même en
    # JSON structuré sur stdout, avec un code de sortie parfois non nul et
    # parfois pas. On fait donc confiance à ce payload dès qu'il est
    # exploitable, et on ne retombe sur le code de sortie brut / stderr que
    # s'il n'y en a pas (crash avant la moindre sortie, binaire introuvable).
    payload = None
    if result.stdout.strip():
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            payload = None

    if isinstance(payload, dict) and "is_error" in payload:
        if not payload["is_error"]:
            return str(payload.get("result", ""))
        message = str(payload.get("result") or payload.get("subtype") or payload)[:300]
        if payload.get("api_error_status") in _RATE_LIMIT_STATUSES or _looks_rate_limited(message):
            raise RateLimited(message)
        raise RuntimeError(f"claude -p : {message}")

    message = (result.stderr or result.stdout)[:300]
    if _looks_rate_limited(message):
        raise RateLimited(message)
    raise RuntimeError(f"claude -p a échoué ({result.returncode}) : {message}")


def _call_with_retry[T](label: str, call: Callable[[], T]) -> T:
    """Politique commune aux appels du CLI : pannes transitoires (RETRIES
    tentatives immédiates, même comportement qu'avant), plafond d'abonnement
    (attente avec repli exponentiel, sans limite de tentatives — le run
    reprend tout seul quand la fenêtre se libère, ou reste arrêtable depuis
    l'UI via /stop).
    """
    attempt = 0
    backoff = BACKOFF_START_S
    while True:
        _wait_out_cooldown()
        try:
            return call()
        except RateLimited as exc:
            until = _enter_cooldown(backoff)
            wait = max(0.0, until - time.monotonic())
            print(
                f"[prompts] en pause — {label} : plafond d'abonnement atteint, "
                f"nouvelle tentative dans {wait:.0f}s ({exc})"
            )
            time.sleep(wait)
            backoff = min(backoff * BACKOFF_FACTOR, BACKOFF_MAX_S)
            print("[prompts] …")  # reprise : redonne son état actif à la barre de progression
        except _RECOVERABLE:
            attempt += 1
            if attempt > RETRIES:
                raise


def _parse(raw: str) -> list[tuple[int, str]]:
    """Parsing défensif : le modèle encadre parfois le JSON de prose ou de ```.

    Renvoie une LISTE ordonnée, pas un dict : `handle()` doit pouvoir
    comparer position par position l'ordre reçu à l'ordre envoyé (voir son
    commentaire). Convertir ici en dict par index perdrait cette info — un
    lot où le modèle a décalé les index d'un cran (prompt de la scène N+1
    étiqueté N) contient quand même exactement le bon ensemble de clés, donc
    passerait inaperçu si on ne regardait que `dict.keys()`."""
    match = _JSON_OBJECT.search(raw)
    if not match:
        raise ValueError(f"aucun objet JSON dans la réponse : {raw[:200]!r}")
    data = json.loads(match.group(0))
    return [(int(item["index"]), str(item["prompt"]).strip()) for item in data["prompts"]]


def _summarize(binary: str, script: str) -> str:
    excerpt = script if len(script) < 12_000 else f"{script[:6000]}\n[…]\n{script[-6000:]}"
    prompt = (
        "Résume ce script de vidéo en un paragraphe de 80 mots maximum : sujet, "
        "ton, et fil narratif. Ce résumé servira à garder une cohérence visuelle "
        "entre 300 illustrations. Réponds par le seul paragraphe.\n\n" + excerpt
    )
    # Un seul appel, contrairement aux lots ci-dessous — mais une panne
    # transitoire du CLI ici tue tout le run avant même le premier lot.
    try:
        return _call_with_retry("résumé du script", lambda: _ask(binary, prompt, max_tokens=400))
    except _RECOVERABLE as exc:
        raise RuntimeError(
            f"résumé du script : abandon après {RETRIES + 1} essais — {exc}"
        ) from exc


_ERA_LINE = re.compile(r"^(AVOID|EXPECTED)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _era_prompt(summary: str) -> str:
    return (
        "Voici le résumé d'une vidéo narrée :\n\n"
        f"{summary}\n\n"
        "Identifie factuellement l'époque et le registre technologique de cette "
        "histoire. En anglais, deux lignes étiquetées, chacune 15 à 30 mots, "
        "phrase nominale sans ponctuation finale :\n\n"
        "AVOID: <objets/architecture/technologie anachroniques à ne jamais montrer>\n"
        "EXPECTED: <objets/architecture/technologie normaux et attendus dans ce contexte>\n\n"
        "Réponds uniquement par ces deux lignes, rien d'autre."
    )


def _era_guardrail(binary: str, summary: str) -> str:
    """Un appel de plus, une fois par run (comme `_summarize`) — déduit du
    résumé global ce qui est anachronique ou attendu, plutôt que d'écrire en
    dur dans `image_prompt.txt` une règle propre à UN thème (« jamais
    d'architecture moderne ») qui casserait un futur projet différent (un
    « Hacking » où les écrans sont justement ce qu'on veut voir). Validé
    empiriquement sur Égypte antique / hacking / trading avant d'écrire ce
    code : le même prompt donne « AVOID: dial-up... EXPECTED: cloud,
    ransomware... » sur un synopsis hacking, l'exact inverse d'Égypte.

    Chaîne vide si l'appel échoue ou si la réponse n'a pas le format attendu
    — `image_prompt.txt` retombe alors sur « non déterminé », jamais sur un
    refus injecté tel quel dans un vrai prompt de lot (même précaution que
    `section_bible.py::_looks_like_refusal`)."""
    try:
        reply = _call_with_retry(
            "registre historique/technologique",
            lambda: _ask(binary, _era_prompt(summary), max_tokens=200),
        )
    except _RECOVERABLE as exc:
        print(f"  registre historique/technologique : abandon — {exc}")
        return ""

    lines = [ln.strip() for ln in reply.strip().splitlines() if _ERA_LINE.match(ln.strip())]
    if len(lines) < 2:
        print("  registre historique/technologique : réponse non exploitable, ignoré")
        return ""
    return "\n".join(lines)


def _batch_prompt(
    style: StyleConfig,
    summary: str,
    era_guardrail: str,
    previous: list[Scene],
    batch: list[Scene],
    section_ctx: section_bible.SectionContext | None = None,
    cast: list[cast_bible.CastMember] | None = None,
) -> str:
    template = (PROMPTS_DIR / "image_prompt.txt").read_text(encoding="utf-8")
    if section_ctx is None:
        # Chemin inchangé pour les scripts sans paliers (SPEC section_bible.py) :
        # identique bit à bit à avant l'ajout du mécanisme de paliers.
        previous_str = "\n".join(f"- {s.text}" for s in previous) or "(début du script)"
        scenes_str = "\n".join(f"{s.index}. {s.text}" for s in batch)
    else:
        previous_str = section_ctx.render_previous(previous)
        scenes_str = section_ctx.render_scenes(batch)
    return template.format(
        style_bible=style.bible(),
        summary=summary,
        era_guardrail=era_guardrail or "(non déterminé)",
        cast=cast_bible.render(cast or []),
        previous=previous_str,
        scenes=scenes_str,
    )


def is_complete(project: Project, opts: Options) -> bool:
    """L'étape a-t-elle un prompt pour CHAQUE scène ?

    « `prompts.json` existe » ne suffit pas : un lot abandonné laisse ses
    vingt scènes sans prompt, et `images.py` les illustre alors à partir du
    texte français brut de la scène (`error_code=no_prompt`) — ce qui donne
    une image hors sujet. Vu en pratique (19/09) : deux lots perdus sur un
    run, 40 scènes sur 305.
    """
    prompts_file = project.dir / "prompts.json"
    if not prompts_file.exists() or not project.scenes.exists():
        return False
    try:
        ecrits = json.loads(prompts_file.read_text(encoding="utf-8"))["prompts"]
        scenes = load(project.scenes)
    except Exception:  # noqa: BLE001 — artefact illisible = étape à refaire
        return False

    voulues = (
        set(sample_indices(scenes, scenes[-1].end_ms))
        if opts.sample
        else {s.index for s in scenes}
    )
    return all(str(i) in ecrits and ecrits[str(i)].strip() for i in voulues)


def run(project: Project, opts: Options) -> None:
    scenes = load(project.scenes)
    style = load_style(draft=opts.draft)

    selected = set(sample_indices(scenes, scenes[-1].end_ms)) if opts.sample else None
    todo = [s for s in scenes if selected is None or s.index in selected]

    # Reprise : les prompts déjà écrits sont conservés, seules les scènes
    # absentes sont redemandées. Sans ça, rattraper quarante scènes manquantes
    # imposait de refaire les trois cents autres — donc de payer quinze lots
    # de LLM et de changer des prompts qui convenaient.
    prompts_file = project.dir / "prompts.json"
    existants: dict[int, str] = {}
    resume: dict[str, str] = {}
    if prompts_file.exists() and not opts.force:
        ancien = json.loads(prompts_file.read_text(encoding="utf-8"))
        existants = {int(k): v for k, v in ancien.get("prompts", {}).items()}
        resume = {k: ancien.get(k, "") for k in ("summary", "era_guardrail")}
        manquants = [s for s in todo if s.index not in existants]
        if existants and manquants:
            print(f"  reprise : {len(existants)} prompts conservés, {len(manquants)} à générer")
            todo = manquants
        elif existants and not manquants:
            print(f"  {len(existants)} prompts déjà complets")
            return

    binary = _binary()
    # En reprise, le résumé et le registre d'époque sont relus plutôt que
    # recalculés : ce sont deux appels LLM, et surtout deux valeurs qui
    # doivent rester IDENTIQUES à celles des prompts déjà écrits, sinon les
    # scènes rattrapées seraient décrites dans un autre registre que leurs
    # voisines.
    summary = resume.get("summary") or _summarize(
        binary, project.script_normalized.read_text(encoding="utf-8")
    )
    # Registre historique/technologique — déduit du résumé, pas codé en dur
    # (voir docstring de `_era_guardrail`) : un run sans paliers en profite
    # aussi, c'est un problème général, pas propre au format « paliers ».
    era_guardrail = resume.get("era_guardrail") or _era_guardrail(binary, summary)

    # Descriptif costume/décor par palier (SPEC section_bible.py) — seulement
    # si le script a des marqueurs NIVEAU:, calculé une fois, avant les lots
    # (peu de paliers, pas besoin du pool parallèle ci-dessous). Absent, tout
    # reste identique à avant l'ajout du mécanisme.
    section_ctx: section_bible.SectionContext | None = None
    levels_raw = section_bible.load_levels(project)
    if levels_raw:
        sections = section_bible.build_sections(levels_raw, scenes)
        scene_to_section = section_bible.assign_scenes(sections, scenes)

        def ask_descriptor(label: str, prompt: str) -> str:
            return _call_with_retry(label, lambda: _ask(binary, prompt, max_tokens=150))

        # En reprise, les bibles sont RELUES, jamais recalculées : une nouvelle
        # détection donnerait d'autres lieux, avec d'autres index, alors que
        # les images de référence déjà générées (`location_refs/NNN.png`)
        # portent les anciens. On remapperait des décors sur les mauvaises
        # scènes pour rattraper quelques prompts.
        bibles_existantes = existants and project.section_bible.exists()
        if bibles_existantes:
            sections = section_bible.load(project) or sections
            print(f"  {len(sections)} paliers — descriptif relu (reprise)")
        else:
            section_bible.compute_descriptors(ask_descriptor, sections, scenes, scene_to_section)
            section_bible.save(project, sections)
            print(f"  {len(sections)} paliers — descriptif costume/décor calculé")

        def ask_location(label: str, prompt: str) -> str:
            return _call_with_retry(label, lambda: _ask(binary, prompt, max_tokens=300))

        locations = location_bible.load(project) if bibles_existantes else None
        if locations is None:
            locations = location_bible.detect_locations(
                ask_location, sections, scenes, scene_to_section
            )
            location_bible.save(project, locations)
        location_ctx = location_bible.LocationContext(locations, location_bible.build_scene_to_location(locations))
        print(f"  {len(locations)} lieux récurrents détectés")

        section_ctx = section_bible.SectionContext(
            sections, scene_to_section, location_label=location_ctx.label_for
        )

    # Personnages secondaires récurrents (voir `cast_bible.py`). Relus dès
    # que le fichier existe, sauf `--force` : en reprise, les scènes
    # rattrapées doivent décrire les personnages comme leurs voisines, et le
    # fichier a pu être corrigé à la main — le recalculer effacerait la
    # correction.
    cast = None if opts.force else cast_bible.load(project)
    if cast is None:
        cast = cast_bible.detect_cast(
            lambda label, prompt: _call_with_retry(label, lambda: _ask(binary, prompt, max_tokens=1500)),
            project.script_normalized.read_text(encoding="utf-8"),
        )
        cast_bible.save(project, cast)
    print(f"  {len(cast)} personnages secondaires récurrents")

    batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

    def handle(batch: list[Scene]) -> dict[int, str]:
        # Les 2 scènes précédentes viennent de la liste complète, pas du lot :
        # en --sample les scènes ne sont pas contiguës.
        first = batch[0].index
        previous = [s for s in scenes if first - 2 <= s.index < first]
        prompt = _batch_prompt(style, summary, era_guardrail, previous, batch, section_ctx, cast)

        # Gardé même si une tentative ULTÉRIEURE finit par être abandonnée :
        # mieux vaut les prompts d'un essai VALIDÉ que rien — `images.py` ne
        # comble que les scènes réellement absentes de `prompts.json`. Mis à
        # jour seulement après validation de l'ordre ci-dessous : un lot
        # désaligné ne doit jamais écraser un essai précédent qui, lui,
        # avait le bon alignement.
        last_parsed: dict[int, str] = {}

        def call() -> dict[int, str]:
            nonlocal last_parsed
            parsed_list = _parse(_ask(binary, prompt, max_tokens=200 * len(batch) + 500))
            expected = [s.index for s in batch]
            got = [index for index, _ in parsed_list]
            # Comparaison stricte, position par position, pas seulement un
            # ensemble de clés : le lot est envoyé dans l'ordre des scènes et
            # le prompt (image_prompt.txt) exige la même réponse "dans le
            # même ordre, avec exactement les mêmes index". Un modèle qui
            # décale ses index d'un cran (prompt de la scène N+1 étiqueté N)
            # fournit quand même le bon ENSEMBLE d'index — seul un contrôle
            # d'ordre le détecte. Vu en pratique (15/09, video5) : plusieurs
            # scènes d'un lot illustraient systématiquement la scène
            # suivante, sans qu'aucune vérification existante ne le
            # remarque.
            if got != expected:
                raise ValueError(f"lot désaligné : attendu {expected}, reçu {got}")
            parsed = dict(parsed_list)
            last_parsed = parsed
            return parsed

        try:
            return _call_with_retry(f"lot {first}", call)
        except _RECOVERABLE as exc:
            if len(batch) > 1:
                # Plutôt que de perdre le lot entier, on le coupe en deux et on
                # réessaie chaque moitié : un lot plus court laisse moins de
                # place à la dérive d'index, et un lot d'une seule scène ne
                # peut pas être désaligné du tout. Vu en pratique (19/09) :
                # deux lots de 20 abandonnés sur un run, soit 40 scènes sans
                # prompt illustrées à partir du texte français brut.
                milieu = len(batch) // 2
                print(f"  lot {first}: échec, découpage en {milieu} + {len(batch) - milieu}")
                gauche = handle(batch[:milieu])
                droite = handle(batch[milieu:])
                return {**gauche, **droite}
            print(f"  scène {first}: abandon après {RETRIES + 1} essais — {exc}")
            return last_parsed

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(handle, batches))

    prompts: dict[int, str] = dict(existants)
    for result in results:
        prompts.update(result)

    write_atomic(
        project.dir / "prompts.json",
        json.dumps(
            {
                "via": "claude-code-cli",
                "summary": summary,
                "era_guardrail": era_guardrail,
                "prompts": {str(k): v for k, v in prompts.items()},
            },
            ensure_ascii=False,
            indent=1,
        ),
    )

    # Compté sur les scènes demandées à CE run (todo), pas sur le fichier
    # entier : en reprise, `prompts` contient aussi les anciens, et comparer
    # les deux donnait des « -265 manquants ».
    restants = [s.index for s in todo if s.index not in prompts]
    suffix = f" — {len(restants)} encore manquants" if restants else ""
    print(f"  {len(todo)} scènes traitées en {len(batches)} lots{suffix}")
    if restants:
        print(f"    scènes sans prompt : {restants[:10]}")
