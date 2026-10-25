"""Descriptif costume/décor par palier (`NIVEAU:` / `levels.json`).

`steps/prompts.py` ne donne à chaque lot de scènes qu'un résumé global (80
mots) et les 2 scènes précédentes : aucun état persistant de type « costume
et décor actuels » ne traverse les lots, qui tournent en plus en parallèle.
Résultat observé en pratique sur une vidéo à paliers (rangs de l'Égypte
antique) : costume et décor dérivent au hasard d'un lot à l'autre au lieu de
suivre le palier réellement en cours.

Ce module comble ce trou pour les scripts qui ont des marqueurs `NIVEAU:` :
un appel LLM par palier (pas par lot), avant que les lots ne démarrent,
décrit une fois pour toutes le costume du protagoniste et le décor dominant
de la section — `steps/prompts.py` injecte ensuite ce descriptif dans le
prompt de chaque scène qui appartient à ce palier.

N'active rien pour les scripts sans `levels.json` (SPEC `project.py::levels`
— pas toutes les vidéos ne sont un format « paliers ») : `load_levels` rend
`None` et tout le reste du module n'est jamais appelé.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .project import Project, write_atomic
from .scenes import Scene

# Même seuil que `_summarize` dans prompts.py : au-delà, un extrait tête+queue
# suffit à décrire un palier sans faire exploser le prompt du descriptif.
EXCERPT_THRESHOLD = 12_000
EXCERPT_HEAD = 6_000
EXCERPT_TAIL = 6_000

# Filet de sécurité derrière le prompt de `_descriptor_prompt` : vu en
# pratique (04/09, avant correctif du prompt) un Haiku qui refuse et demande
# une capture d'écran plutôt que de répondre. Un faux négatif ici ne fait que
# priver une section de descriptif (retombe sur le comportement d'avant ce
# module) ; laisser passer un refus, en revanche, l'injecterait tel quel dans
# un vrai prompt de lot — bien pire.
_REFUSAL_MARKERS = (
    "je n'ai pas",
    "je n'ai pas accès",
    "i don't have",
    "i need",
    "i don't have access",
    "pourriez-vous",
    "pouvez-vous",
    "capture d'écran",
    "screenshot",
    "could you",
    "n'ai pas d'informations",
)
_MAX_DESCRIPTOR_WORDS = 60


def _looks_like_refusal(text: str) -> bool:
    low = text.lower()
    if "?" in text:
        return True
    if len(text.split()) > _MAX_DESCRIPTOR_WORDS:
        return True
    return any(marker in low for marker in _REFUSAL_MARKERS)


@dataclass
class Section:
    """Un palier, dans l'ordre du script. `costume`/`setting` sont vides tant
    que `compute_descriptors` n'a pas tourné (ou si son appel LLM a échoué ou
    a été rejeté — la section reste alors sans consigne plutôt que de faire
    échouer tout le run, même tolérance que le reste du pipeline face à un
    aléa de génération).

    Séparés plutôt qu'un seul `descriptor` : `steps/refs.py` a besoin du
    costume SEUL (sans le lieu) pour générer une image de référence de
    palier — un décor mélangé dedans s'y retrouverait figé sur toutes les
    scènes du palier, alors que le lieu change à l'intérieur d'un même palier
    (ex. "Artisan" : atelier, village, tombe, maison)."""

    index: int
    label: str
    word_start: int
    costume: str = ""
    setting: str = ""


@dataclass
class SectionContext:
    """Ce dont `steps/prompts.py` a besoin pour construire les blocs
    `{previous}`/`{scenes}` d'un lot en tenant compte des paliers."""

    sections: list[Section]
    scene_to_section: dict[int, int] = field(default_factory=dict)
    # Injecté par callable plutôt que par import de `location_bible.Location`
    # (ce module importerait alors `location_bible`, qui importe déjà
    # `section_bible` pour `Section` — cycle). `None` par défaut : tout
    # appelant qui ne le passe pas garde le rendu d'avant ce mécanisme.
    location_label: Callable[[int], str | None] | None = None

    def _label_for(self, scene_index: int) -> str | None:
        ordinal = self.scene_to_section.get(scene_index, -1)
        if ordinal < 0:
            return None
        section = self.sections[ordinal]
        parts = []
        if section.costume:
            parts.append(f"costume : {section.costume}")
        if section.setting:
            parts.append(f"décor : {section.setting}")
        if parts:
            return f"{section.label} — {' — '.join(parts)}"
        return section.label

    def render_scenes(self, batch: list[Scene]) -> str:
        """Un en-tête `[SECTION: ...]` à chaque changement de palier à
        l'intérieur du lot — un lot peut chevaucher plus de deux paliers si
        l'un d'eux ne couvre que quelques scènes. Sous chaque section, un
        sous-en-tête `[LOCATION: ...]` optionnel à chaque changement de lieu
        récurrent (voir `location_bible.py`) — réinitialisé à chaque
        changement de palier pour qu'un lieu ne « fuie » jamais dans le
        palier suivant."""
        lines: list[str] = []
        last_ordinal: int | None = None
        last_location: str | None = None
        for scene in batch:
            ordinal = self.scene_to_section.get(scene.index, -1)
            if ordinal != last_ordinal:
                label = self._label_for(scene.index)
                if label is not None:
                    if lines:
                        lines.append("")
                    lines.append(f"[SECTION: {label}]")
                last_ordinal = ordinal
                last_location = None
            if self.location_label is not None:
                loc_label = self.location_label(scene.index)
                if loc_label != last_location:
                    if loc_label is not None:
                        lines.append(f"  [LOCATION: {loc_label}]")
                    last_location = loc_label
            lines.append(f"{scene.index}. {scene.text}")
        return "\n".join(lines)

    def render_previous(self, previous: list[Scene]) -> str:
        """Étiquette chaque scène précédente de son palier — évite qu'à une
        frontière de palier le lot hérite à tort du costume de la section
        d'avant via ce bloc de contexte local."""
        if not previous:
            return "(début du script)"
        lines = []
        for scene in previous:
            ordinal = self.scene_to_section.get(scene.index, -1)
            prefix = f"[{self.sections[ordinal].label}] " if ordinal >= 0 else ""
            lines.append(f"- {prefix}{scene.text}")
        return "\n".join(lines)


def load_levels(project: Project) -> list[dict] | None:
    """`None` si le projet n'a pas de paliers — désactive tout le mécanisme."""
    if not project.levels.exists():
        return None
    return json.loads(project.levels.read_text(encoding="utf-8"))["levels"]


def build_sections(levels_raw: list[dict], scenes: list[Scene]) -> list[Section]:
    """Un palier dont le `word_index` dépasse la fin du script n'a aucune
    scène à décrire : ignoré, même tolérance que
    `steps/render.py::_load_level_cards`."""
    max_word_start = max((s.word_start for s in scenes), default=-1)
    sections: list[Section] = []
    for entry in sorted(levels_raw, key=lambda e: e["word_index"]):
        if entry["word_index"] > max_word_start:
            print(f"  palier ignoré (au-delà de la fin du script) : {entry['label']!r}")
            continue
        sections.append(
            Section(index=len(sections), label=entry["label"], word_start=entry["word_index"])
        )
    return sections


def assign_scenes(sections: list[Section], scenes: list[Scene]) -> dict[int, int]:
    """scene.index -> ordinal de son palier (le dernier dont `word_start` est
    atteint), ou -1 pour les scènes avant le premier palier. Suppose `scenes`
    trié par `word_start` croissant, comme le produit `steps/segment.py`."""
    assignment: dict[int, int] = {}
    ordinal = -1
    next_idx = 0
    for scene in scenes:
        while next_idx < len(sections) and sections[next_idx].word_start <= scene.word_start:
            ordinal = next_idx
            next_idx += 1
        assignment[scene.index] = ordinal
    return assignment


def _excerpt(text: str) -> str:
    if len(text) < EXCERPT_THRESHOLD:
        return text
    return f"{text[:EXCERPT_HEAD]}\n[…]\n{text[-EXCERPT_TAIL:]}"


_FIELD_LINE = re.compile(r"^(COSTUME|DECOR)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _descriptor_prompt(label: str, excerpt: str) -> str:
    # Sans la précision sur `bldguy` ci-dessous, Haiku a refusé de répondre
    # sur la moitié des paliers d'un run réel (04/09) : il demandait une
    # capture d'écran du personnage, croyant devoir décrire un visage/corps
    # qu'il n'a jamais vu. `bldguy` n'est qu'un mot-clé de design pour le
    # générateur d'images, pas un personnage à observer — il n'y a rien à
    # voir, seulement à déduire du texte narratif ci-dessous. Même précaution
    # que `image_prompt.txt` (qui l'explique déjà au LLM des lots), reprise
    # ici parce que ce prompt-ci n'a pas ce contexte par ailleurs.
    #
    # Deux lignes étiquetées plutôt qu'une phrase unique : `steps/refs.py`
    # a besoin du costume seul, sans le lieu, pour générer l'image de
    # référence du palier (voir docstring de `Section`).
    return (
        f"Tu écris des prompts d'illustration pour une vidéo narrée, format "
        "« une vie racontée niveau par niveau ». Le protagoniste s'écrit "
        "toujours `bldguy` : c'est un mot-clé de design pour le générateur "
        "d'images, pas un personnage que tu dois voir ou avoir déjà vu — "
        "n'essaie jamais de décrire son visage ou son corps, et ne demande "
        "jamais d'image ou de capture d'écran, tu n'y as pas accès et ce "
        "n'est pas nécessaire.\n\n"
        f"Palier « {label} ». Voici le texte des scènes de cette section :\n\n"
        f"{excerpt}\n\n"
        "À partir de ce seul texte, déduis factuellement en anglais :\n"
        "- le costume que porte `bldguy` pendant cette section (10-20 mots,\n"
        "  vêtements et accessoires seulement, pas le lieu)\n"
        "- le lieu/décor dominant de cette section (10-20 mots)\n\n"
        "Pas de nom propre inventé, pas de mise en forme, aucune question ni "
        "commentaire. Réponds par exactement ces deux lignes, rien d'autre :\n\n"
        "COSTUME: <phrase nominale anglaise, sans ponctuation finale>\n"
        "DECOR: <phrase nominale anglaise, sans ponctuation finale>"
    )


def _parse_descriptor(reply: str) -> tuple[str, str]:
    """(costume, décor) — chaîne vide pour un champ absent, mal formé, ou qui
    ressemble à un refus (`_looks_like_refusal`, appliqué à chaque champ
    séparément : un refus sur le costume ne doit pas invalider un décor
    valide, et inversement)."""
    costume = ""
    setting = ""
    for line in reply.splitlines():
        match = _FIELD_LINE.match(line.strip())
        if not match:
            continue
        field, value = match.group(1).upper(), match.group(2).strip()
        if _looks_like_refusal(value):
            continue
        if field == "COSTUME":
            costume = value
        else:
            setting = value
    return costume, setting


def compute_descriptors(
    ask: Callable[[str, str], str],
    sections: list[Section],
    scenes: list[Scene],
    scene_to_section: dict[int, int],
) -> None:
    """Remplit `section.costume`/`section.setting` en place, un appel LLM par palier,
    séquentiel — peu de paliers (~8 sur une vidéo complète), pas besoin du
    pool parallèle des lots, et ça évite d'entrelacer ces appels avec ceux
    des lots dans le même cooldown de plafond d'abonnement.

    `ask(label, prompt) -> str` porte la politique de retry/plafond de
    `steps/prompts.py` (`_call_with_retry` + `_ask`) — injectée plutôt
    qu'importée pour éviter un cycle (`prompts.py` importe ce module, pas
    l'inverse).
    """
    texts_by_section: dict[int, list[str]] = defaultdict(list)
    for scene in scenes:
        ordinal = scene_to_section.get(scene.index, -1)
        if ordinal >= 0:
            texts_by_section[ordinal].append(scene.text)

    for section in sections:
        excerpt = _excerpt(" ".join(texts_by_section.get(section.index, [])))
        prompt = _descriptor_prompt(section.label, excerpt)
        try:
            reply = ask(f"palier {section.index} ({section.label})", prompt).strip()
        except Exception as exc:  # noqa: BLE001 — un palier raté ne doit pas bloquer les autres
            print(f"  descriptif du palier {section.label!r} abandonné — {exc}")
            reply = ""
        costume, setting = _parse_descriptor(reply) if reply else ("", "")
        if reply and not costume and not setting:
            print(f"  descriptif du palier {section.label!r} rejeté (réponse non exploitable)")
        section.costume = costume
        section.setting = setting


def scene_refs(project: Project, scenes: list[Scene]) -> dict[int, Path]:
    """scene.index -> chemin de la référence de palier de cette scène (voir
    `steps/refs.py`). Partagée entre `steps/images.py` (run normal) et
    `webui/_gen_review.py` (régénération ciblée) plutôt que dupliquée : les
    deux doivent choisir exactement la même référence pour une scène donnée,
    sinon une image « appliquée » depuis la revue pourrait jurer avec le
    reste de son palier généré par un run normal.

    Vide pour un projet sans paliers, ou si `refs` n'a rien généré pour une
    section (descriptif de costume rejeté à l'étape prompts) — l'appelant
    retombe alors sur `COMFY_CHARACTER_REFS`, comme pour les scènes qui
    précèdent le premier palier."""
    levels_raw = load_levels(project)
    refs_file = project.dir / "section_refs.json"
    if not levels_raw or not refs_file.exists():
        return {}

    refs_by_section = json.loads(refs_file.read_text(encoding="utf-8"))["refs"]
    if not refs_by_section:
        return {}

    sections = build_sections(levels_raw, scenes)
    scene_to_section = assign_scenes(sections, scenes)

    return {
        scene_index: project.dir / refs_by_section[str(ordinal)]
        for scene_index, ordinal in scene_to_section.items()
        if str(ordinal) in refs_by_section
    }


def load(project: Project) -> list[Section] | None:
    """Relit `section_bible.json`, symétrique de `save()`.

    Sert à la reprise de `steps/prompts.py` : quand on ne rattrape que
    quelques prompts manquants, il ne faut surtout pas relancer
    `compute_descriptors` — les costumes changeraient, alors que les images de
    référence de palier déjà générées portent les anciens.
    """
    if not project.section_bible.exists():
        return None
    try:
        data = json.loads(project.section_bible.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return [
        Section(
            index=e["index"],
            label=e["label"],
            word_start=e["word_start"],
            costume=e.get("costume", ""),
            setting=e.get("setting", ""),
        )
        for e in data.get("sections", [])
    ]


def save(project: Project, sections: list[Section]) -> None:
    """Artefact de diagnostic, pas suivi par le contrat d'étape : recalculé
    à chaque exécution réelle de `steps/prompts.py::run()`."""
    write_atomic(
        project.section_bible,
        json.dumps(
            {
                "sections": [
                    {
                        "index": s.index,
                        "label": s.label,
                        "word_start": s.word_start,
                        "costume": s.costume,
                        "setting": s.setting,
                    }
                    for s in sections
                ]
            },
            ensure_ascii=False,
            indent=1,
        ),
    )
