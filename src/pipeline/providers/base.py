"""Protocole provider (SPEC §3.4).

Ne pas sur-concevoir cette interface : la forme définitive émergera au
branchement de ComfyUI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..style import StyleConfig


class ContentRefused(Exception):
    """Le provider a refusé le prompt. Réécrire et réessayer (SPEC §4, étape 5)."""


class ImageProvider(Protocol):
    def generate(
        self,
        prompt: str,
        style: StyleConfig,
        seed: int,
        reference: Path | None = None,
        decor_reference: Path | None = None,
    ) -> bytes: ...
