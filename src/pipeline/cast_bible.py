"""Personnages secondaires récurrents — une description visuelle FIGÉE par
personnage, reprise mot pour mot dans chaque prompt où il apparaît.

Le protagoniste a sa référence d'image (par palier, `section_bible.py`), les
lieux récurrents aussi (`location_bible.py`) ; les personnages secondaires
n'avaient rien. Les prompts étant écrits par lots de 20 scènes, chaque lot
réinventait « El Mono » à sa façon — « motorcyclist with bleached hair » ici,
« El Mono » tout court là, nom que Klein ignore — et le personnage changeait
de visage à chaque image. Vu le 30/09 sur escobart, scènes 16-19 : quatre
motards différents. Avec la même description répétée à l'identique dans les
quatre prompts, le personnage devient stable.

Texte seul, pas d'image de référence : deux références de personnage dans
une même scène, c'est précisément le cas où `ReferenceLatent` attribue le
design au mauvais personnage (voir RUN.md).

Un seul appel LLM sur le script entier (pas par palier) : un personnage
secondaire traverse souvent plusieurs paliers, et sa description doit être
la même d'un bout à l'autre de la vidéo. Actif avec ou sans `NIVEAU:`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .project import Project, write_atomic

# Au-delà, le bloc injecté dans chaque lot devient du bruit : les figurants
# d'une seule scène n'ont pas besoin de description figée.
MAX_MEMBERS = 12

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Même liste que les interdits d'`image_prompt.txt` : une mention d'âge sur
# N'IMPORTE QUEL personnage déteint sur le protagoniste et lui fait pousser
# des cheveux. Et « bald/shaved » ferait un second chauve, que
# `ReferenceLatent` confondrait avec `bldguy`. Une description qui en
# contient est rejetée plutôt que nettoyée : l'amputer d'un mot peut la
# rendre incohérente.
_FORBIDDEN = re.compile(
    r"\b(\d+\s*(years?|yrs?)|aged?|young(er)?|old(er)?|elderly|teen(age|ager)?s?|"
    r"middle-aged|boyish|youthful|bald|shaved|shaven)\b",
    re.IGNORECASE,
)


@dataclass
class CastMember:
    name: str
    description: str
    aliases: list[str] = field(default_factory=list)


def _cast_prompt(script: str) -> str:
    return (
        "Voici le script d'une vidéo narrée à la deuxième personne (« tu ») :\n\n"
        f"{script}\n\n"
        "Liste les personnages SECONDAIRES qui reviennent dans au moins deux "
        "passages distincts du script — pas le protagoniste (« tu »), pas les "
        "figurants d'une seule scène, pas les groupes anonymes (« les "
        f"soldats »). {MAX_MEMBERS} au maximum, les plus présents d'abord.\n\n"
        "Pour chacun :\n"
        "- `name` : le nom tel qu'écrit dans le script ;\n"
        "- `aliases` : les autres façons dont le script le désigne (surnom, "
        "fonction : « le professeur », « son oncle ») ;\n"
        "- `description` : une description visuelle en anglais, 10 à 15 "
        "mots, phrase nominale commençant par « a man », « a woman » ou « a "
        "boy/girl » selon le cas, puis : silhouette, UN trait distinctif "
        "(couleur de cheveux, lunettes, cicatrice...), une tenue signature "
        "avec sa couleur. Elle servira telle quelle dans des centaines de "
        "prompts d'image, qui ne dessinent que ce qui se VOIT : aucun trait "
        "de caractère, d'humeur, d'habitude ou de rôle (« stern », « quiet "
        "authority », « calculating demeanor », « chalk always in pocket »). "
        "Deux personnages ne doivent JAMAIS partager la même tenue ni la même "
        "couleur dominante : chacun doit être reconnaissable à côté des "
        "autres. Un groupe de plusieurs personnes n'est pas UN personnage : "
        "ne les fusionne jamais en une seule entrée. INTERDIT : âge ou tout "
        "mot d'âge (young, old, teenage, 40 years...), crâne rasé ou chauve, "
        "nom propre, marque, nom d'équipe ou texte écrit sur un vêtement (le "
        "générateur l'écrirait en lettres illisibles).\n\n"
        "Réponds en JSON strict, rien d'autre :\n"
        '{"cast": [{"name": "...", "aliases": ["..."], "description": "a ..."}]}'
    )


def _parse_cast(reply: str) -> list[CastMember]:
    """Même tolérance que `location_bible._parse_locations` : une réponse
    inexploitable donne une liste vide, le run continue sans bible."""
    match = _JSON_OBJECT.search(reply)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))["cast"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []

    out: list[CastMember] = []
    for entry in raw[:MAX_MEMBERS]:
        try:
            name = str(entry["name"]).strip()
            description = str(entry["description"]).strip().rstrip(".")
            aliases = list(dict.fromkeys(
                a for a in (str(a).strip() for a in entry.get("aliases", []))
                if a and a.lower() != name.lower()
            ))
        except (KeyError, TypeError, AttributeError):
            continue
        if not name or not description:
            continue
        # Insérée en milieu de phrase dans les prompts : « A man » → « a man ».
        description = description[:1].lower() + description[1:]
        if _FORBIDDEN.search(description):
            print(f"  personnage {name!r} écarté — description avec âge/crâne : {description!r}")
            continue
        out.append(CastMember(name=name, description=description, aliases=aliases))
    return out


def detect_cast(ask: Callable[[str, str], str], script: str) -> list[CastMember]:
    try:
        reply = ask("personnages récurrents", _cast_prompt(script))
    except Exception as exc:  # noqa: BLE001 — sans bible, le run continue comme avant
        print(f"  personnages récurrents abandonnés — {exc}")
        return []
    return _parse_cast(reply)


def render(cast: list[CastMember]) -> str:
    """Bloc injecté dans `image_prompt.txt` (`{cast}`)."""
    if not cast:
        return "(aucun personnage secondaire récurrent)"
    lines = []
    for member in cast:
        names = " / ".join([member.name, *member.aliases])
        lines.append(f"- {names} → `{member.description}`")
    return "\n".join(lines)


def save(project: Project, cast: list[CastMember]) -> None:
    write_atomic(
        project.cast_bible,
        json.dumps(
            {
                "cast": [
                    {"name": m.name, "aliases": m.aliases, "description": m.description}
                    for m in cast
                ]
            },
            ensure_ascii=False,
            indent=1,
        ),
    )


def load(project: Project) -> list[CastMember] | None:
    if not project.cast_bible.exists():
        return None
    data = json.loads(project.cast_bible.read_text(encoding="utf-8"))
    return [
        CastMember(name=e["name"], description=e["description"], aliases=e.get("aliases", []))
        for e in data["cast"]
    ]
