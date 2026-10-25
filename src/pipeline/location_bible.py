"""Lieux récurrents par palier (`NIVEAU:` / `levels.json`).

Le costume est constant sur tout un palier — une seule référence par palier
suffit (`section_bible.py`, `steps/refs.py`). Le décor, lui, change *à
l'intérieur* d'un même palier : le palier « Noble » a une demeure familiale,
une cour administrative, des chantiers. Une référence par palier figerait
tout le palier sur un seul lieu ; ce module détecte à la place les lieux qui
reviennent réellement plusieurs fois dans un palier, et laisse le reste
(narration sans décor concret) au texte seul, comme avant.

Un appel LLM par palier (pas par scène) propose des lieux candidats ; le
seuil de récurrence et la résolution des chevauchements (une scène ne peut
appartenir qu'à un seul lieu) sont tranchés ici en code, pas seulement dans
le prompt — validé en pratique (09/09) : sur 41 scènes réelles, une scène
mentionnait à la fois des greniers et des chantiers et s'est retrouvée
proposée dans deux lieux à la fois.

N'active rien pour les scripts sans `levels.json`, exactement comme
`section_bible.py` : `steps/prompts.py` ne construit `Location`s que dans le
même bloc `if levels_raw:` qui pilote déjà `compute_descriptors`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import section_bible
from .project import Project, write_atomic
from .scenes import Scene

# Un lieu qui ne garde pas au moins ce nombre de scènes (après résolution des
# chevauchements) ne mérite pas sa propre image de référence — validé sur le
# palier Noble réel : 10 et 7 scènes retenues, 2 scènes exclues à raison.
MIN_SCENES = 3

# La référence de décor n'est passée qu'aux premières scènes d'un lieu, pour
# l'établir ; les suivantes n'en gardent que la description texte (injectée
# dans les prompts par `LocationContext`). Au-delà, Klein ne s'en sert plus
# comme ambiance : il RECOPIE le cadrage de la référence. Vu le 30/09 sur
# escobart : « The corner » couvrait 15 scènes et a produit 14 fois le même
# angle de rue, même immeuble, même moto, bldguy minuscule ou absent.
# Regénérées sans la référence, les mêmes scènes ont retrouvé des cadrages
# variés sans perdre le lieu.
MAX_REF_SCENES = 3

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Location:
    """Un lieu récurrent, après résolution des chevauchements. `index` est
    global à toute la vidéo (pas remis à zéro par palier) : il nomme
    `location_refs/NNN.png`, qui doit être unique sur tout le projet."""

    index: int
    section: int  # palier d'origine — traçabilité seulement, pas relu ailleurs
    name: str
    description: str
    scene_indices: list[int] = field(default_factory=list)


@dataclass
class LocationContext:
    """Injecté dans `section_bible.SectionContext` par callable plutôt que
    par import de `Location` — évite un cycle `section_bible` ↔
    `location_bible` (celui-ci importe déjà `section_bible` pour `Section`)."""

    locations: list[Location]
    scene_to_location: dict[int, int] = field(default_factory=dict)

    def label_for(self, scene_index: int) -> str | None:
        loc_index = self.scene_to_location.get(scene_index)
        if loc_index is None:
            return None
        loc = next((location for location in self.locations if location.index == loc_index), None)
        return f"{loc.name} — {loc.description}" if loc else None


def _location_prompt(label: str, scenes: list[Scene]) -> str:
    numbered = "\n".join(f"{s.index}. {s.text}" for s in scenes)
    return (
        f"Voici les scènes du palier « {label} » d'une vidéo narrée, format "
        "« une vie racontée niveau par niveau », avec leur index :\n\n"
        f"{numbered}\n\n"
        "Ce palier illustre le quotidien de ce personnage. Identifie 2 à 5 "
        "lieux physiques RÉCURRENTS qui reviendraient plusieurs fois si on "
        "illustrait chaque scène — un lieu concerné par une seule scène "
        "isolée ne compte pas, et la plupart des scènes n'ont pas de lieu "
        "dédié (narration abstraite) : ne les force dans aucun lieu. Chaque "
        "scène ne doit appartenir qu'à un seul lieu.\n\n"
        "Pour chaque lieu retenu : un nom court, une description factuelle "
        "en anglais (matériau, style architectural, végétation — jamais de "
        "personnage, 10 à 20 mots), et la liste des index de scènes qui s'y "
        "déroulent plausiblement.\n\n"
        "Réponds en JSON strict, rien d'autre :\n"
        '{"locations": [{"name": "...", "description": "...", "scenes": [278, 282]}]}'
    )


def _parse_locations(reply: str, valid_indices: set[int]) -> list[tuple[str, str, list[int]]]:
    """Extraction défensive, même tolérance que `prompts.py::_parse` (le
    modèle encadre parfois le JSON de prose ou de ```). Une réponse
    inexploitable donne une liste vide — le palier retombe sur le texte
    seul, comme aujourd'hui, plutôt que de faire échouer le run."""
    match = _JSON_OBJECT.search(reply)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        raw = data["locations"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

    out: list[tuple[str, str, list[int]]] = []
    for entry in raw:
        try:
            name = str(entry["name"]).strip()
            description = str(entry["description"]).strip()
            scenes = [int(i) for i in entry["scenes"] if int(i) in valid_indices]
        except (KeyError, TypeError, ValueError):
            continue
        if name and description and scenes:
            out.append((name, description, scenes))
    return out


def detect_locations(
    ask: Callable[[str, str], str],
    sections: list[section_bible.Section],
    scenes: list[Scene],
    scene_to_section: dict[int, int],
) -> list[Location]:
    """Un appel LLM par palier, séquentiel (même raison que
    `section_bible.compute_descriptors` : peu de paliers, pas besoin du pool
    parallèle des lots, évite d'entrelacer ces appels avec eux dans le même
    cooldown de plafond d'abonnement).

    Seuil de récurrence et chevauchements tranchés ICI, pas seulement dans le
    prompt : les lieux proposés sont parcourus dans l'ordre reçu, le premier
    à réclamer une scène la garde, et un lieu qui ne garde pas au moins
    `MIN_SCENES` scènes après ce dédoublonnage est abandonné silencieusement.
    """
    scenes_by_section: dict[int, list[Scene]] = {}
    for scene in scenes:
        ordinal = scene_to_section.get(scene.index, -1)
        if ordinal >= 0:
            scenes_by_section.setdefault(ordinal, []).append(scene)

    locations: list[Location] = []
    next_index = 0
    for section in sections:
        section_scenes = scenes_by_section.get(section.index, [])
        if len(section_scenes) < MIN_SCENES:
            continue  # même un seul lieu ne pourrait pas franchir le seuil

        valid_indices = {s.index for s in section_scenes}
        try:
            reply = ask(f"lieux du palier {section.index} ({section.label})", _location_prompt(section.label, section_scenes))
        except Exception as exc:  # noqa: BLE001 — un palier raté ne doit pas bloquer les autres
            print(f"  lieux du palier {section.label!r} abandonnés — {exc}")
            continue

        raw = _parse_locations(reply, valid_indices)
        claimed: set[int] = set()
        for name, description, scene_idx in raw:
            deduped = sorted(i for i in scene_idx if i not in claimed)
            claimed.update(deduped)
            if len(deduped) < MIN_SCENES:
                continue
            locations.append(
                Location(index=next_index, section=section.index, name=name, description=description, scene_indices=deduped)
            )
            next_index += 1

    return locations


def build_scene_to_location(locations: list[Location]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for location in locations:
        for scene_index in location.scene_indices:
            mapping[scene_index] = location.index
    return mapping


def save(project: Project, locations: list[Location]) -> None:
    """Artefact de diagnostic, comme `section_bible.json` : recalculé à
    chaque exécution réelle de `steps/prompts.py::run()`."""
    write_atomic(
        project.location_bible,
        json.dumps(
            {
                "locations": [
                    {
                        "index": loc.index,
                        "section": loc.section,
                        "name": loc.name,
                        "description": loc.description,
                        "scenes": loc.scene_indices,
                    }
                    for loc in locations
                ]
            },
            ensure_ascii=False,
            indent=1,
        ),
    )


def load(project: Project) -> list[Location] | None:
    if not project.location_bible.exists():
        return None
    data = json.loads(project.location_bible.read_text(encoding="utf-8"))
    return [
        Location(index=e["index"], section=e["section"], name=e["name"], description=e["description"], scene_indices=e["scenes"])
        for e in data["locations"]
    ]


def scene_refs(project: Project, scenes: list[Scene]) -> dict[int, Path]:
    """scene.index -> chemin de la référence de décor de cette scène (voir
    `steps/location_refs.py`). Appelée identiquement par `steps/images.py`
    et `webui/_gen_review.py`, comme `section_bible.scene_refs()` pour le
    costume : un run normal et une régénération depuis la revue ne doivent
    jamais choisir un décor différent pour la même scène.

    Vide pour un projet sans paliers, sans lieu détecté, ou si `refs` n'a
    rien généré pour un lieu (échec ponctuel) — la scène retombe alors sur
    le texte seul, sans décor de référence, exactement comme avant ce
    mécanisme. Limité aux `MAX_REF_SCENES` premières scènes de chaque lieu
    (voir la constante)."""
    refs_file = project.dir / "location_refs.json"
    if not project.levels.exists() or not refs_file.exists():
        return {}

    refs_by_location = json.loads(refs_file.read_text(encoding="utf-8"))["refs"]
    if not refs_by_location:
        return {}

    locations = load(project)
    if not locations:
        return {}

    # Filtre défensif : scenes.json a pu changer depuis la détection
    # (segment re-forcé, par ex.) — une scène qui n'existe plus ne doit pas
    # ressortir une référence de décor périmée.
    valid_indices = {s.index for s in scenes}
    refs: dict[int, Path] = {}
    for location in locations:
        ref = refs_by_location.get(str(location.index))
        if ref is None:
            continue
        kept = sorted(i for i in location.scene_indices if i in valid_indices)
        for scene_index in kept[:MAX_REF_SCENES]:
            refs[scene_index] = project.dir / ref
    return refs
