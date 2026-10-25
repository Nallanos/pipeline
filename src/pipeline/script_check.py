"""Contrôle de conformité du script, avant toute génération.

Un script mal formé ne casse rien tout de suite : il produit une vidéo bizarre
trois heures plus tard. Ce module regarde le texte AVANT le TTS et signale ce
qui a déjà surpris en pratique, par ordre de coût.

Ce sont des AVERTISSEMENTS, pas des erreurs : le script reste générable. Le but
est qu'on voie le problème quand il est encore corrigeable en dix secondes,
pas après un run complet.
"""

from __future__ import annotations

import re
import unicodedata

# Horodatages [MM:SS] du format « dix niveaux » : `clean_source` les retire à
# l'import, mais un script réinjecté à la main peut en contenir encore.
_HORODATAGE = re.compile(r"\[\s*\d{1,2}\s*:\s*\d{2}\s*\]")

# Un marqueur de palier est un intitulé court. Au-delà, c'est probablement une
# phrase de narration écrite par erreur sur une ligne `NIVEAU:`.
_LABEL_MOTS_MAX = 8

# Déséquilibre entre paliers : un palier qui pèse moins que cette fraction de
# la moyenne donnera une poignée de plans, et l'inverse écrase le rythme.
_DESEQUILIBRE = 0.4


def _normalise(texte: str) -> str:
    """Minuscules, sans accents ni ponctuation — pour comparer un label de
    palier au texte qui le suit sans se faire piéger par « chŏnsa » / « chonsa »
    ou par « : » contre « , »."""
    sans_accent = "".join(
        c for c in unicodedata.normalize("NFD", texte.lower())
        if unicodedata.category(c) != "Mn"
    )
    return " ".join(re.sub(r"[^\w\s]", " ", sans_accent).split())


def _repetition(label: str, suite: str) -> bool:
    """Le paragraphe qui suit le carton redit-il ce que le carton affiche ?

    Vu sur video5 : le carton « Niveau un : chŏnsa » précédait « Niveau un, le
    chŏnsa. Tu as 17 ans... ». Le spectateur lisait puis entendait la même
    chose. Comparaison sur les premiers mots seulement — la narration
    poursuit ensuite sur autre chose."""
    mots_label = _normalise(label).split()
    if not mots_label:
        return False
    debut = _normalise(suite).split()[: len(mots_label) + 3]
    if not debut:
        return False
    communs = sum(1 for m in mots_label if m in debut)
    return communs >= max(2, len(mots_label) - 1)


def check(script: str, levels: list[tuple[int, str]] | None = None) -> list[str]:
    """Renvoie la liste des avertissements, vide si le script est propre."""
    avertissements: list[str] = []
    levels = levels or []
    mots = script.split()

    if not mots:
        return ["script vide"]

    horodatages = _HORODATAGE.findall(script)
    if horodatages:
        avertissements.append(
            f"{len(horodatages)} horodatage(s) [MM:SS] encore dans le texte "
            f"(ex. {horodatages[0]}) — ils seront lus par le TTS"
        )

    if not levels and re.search(r"\bniveau\s+(un|deux|trois|\d)\b", script, re.I):
        avertissements.append(
            "le script parle de « niveau » mais ne contient aucun marqueur "
            "`NIVEAU:` — aucun carton ne sera inséré, et les références de "
            "costume par palier ne seront pas générées"
        )

    for i, (mot_index, label) in enumerate(levels, 1):
        if len(label.split()) > _LABEL_MOTS_MAX:
            avertissements.append(
                f"palier {i} : intitulé de {len(label.split())} mots "
                f"({label[:50]}...) — un carton n'affiche bien qu'un titre court"
            )
        suite = " ".join(mots[mot_index : mot_index + 25])
        if _repetition(label, suite):
            avertissements.append(
                f"palier {i} : le texte redit le carton — carton « {label} », "
                f"narration « {suite[:45]}... »"
            )

    if len(levels) >= 2:
        bornes = [m for m, _ in levels] + [len(mots)]
        tailles = [b - a for a, b in zip(bornes, bornes[1:])]
        moyenne = sum(tailles) / len(tailles)
        for i, taille in enumerate(tailles, 1):
            if taille < moyenne * _DESEQUILIBRE:
                avertissements.append(
                    f"palier {i} : {taille} mots contre {moyenne:.0f} en moyenne "
                    "— il passera très vite à l'écran"
                )

    return avertissements


def report(script: str, levels: list[tuple[int, str]] | None = None) -> None:
    """Affiche le contrôle. Silencieux si le script est propre."""
    for a in check(script, levels):
        print(f"  [script] {a}")
