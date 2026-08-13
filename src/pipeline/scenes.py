"""`scenes.json` (SPEC §3.3).

Une scène est une **tranche d'indices** sur la liste de mots de l'alignement.
Aucun LLM ne produit ce texte : l'invariant B est garanti par construction, pas
par vérification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .project import write_atomic

SAMPLE_WINDOW_MS = 30_000


@dataclass
class Scene:
    index: int
    text: str
    word_start: int
    word_end: int  # exclusif
    start_ms: int
    end_ms: int
    boundary_score: float
    seed: int
    forced: bool = False
    reasons: list[str] = field(default_factory=list)
    image_prompt: str | None = None
    image_path: str | None = None
    error_code: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def seed_for(project_id: str, index: int) -> int:
    """Seed déterministe et rejouable (SPEC §4.3).

    `hash()` de Python est randomisé par processus : il donnerait des images
    différentes à chaque run, ce qui casserait à la fois la reprise et la
    comparaison entre deux itérations.
    """
    digest = hashlib.sha256(f"{project_id}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def save(path: Path, scenes: list[Scene]) -> None:
    write_atomic(
        path,
        json.dumps({"scenes": [asdict(s) for s in scenes]}, ensure_ascii=False, indent=1),
    )


def load(path: Path) -> list[Scene]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Scene(**s) for s in data["scenes"]]


def check_invariant_b(scenes: list[Scene], script_normalized: str) -> None:
    """Invariant B (SPEC §7.2) — échec dur."""
    joined = " ".join(s.text for s in scenes)
    if joined != script_normalized:
        raise AssertionError(
            f"invariant B violé : {len(joined.split())} mots dans les scènes "
            f"contre {len(script_normalized.split())} dans le script"
        )


def sample_indices(scenes: list[Scene], audio_ms: int) -> list[int]:
    """Les scènes retenues par `--sample` (SPEC §5.1).

    Les 30 premières secondes en entier (elles décident de la rétention), une
    fenêtre de 30 s au milieu, une vers la fin. Soit ~20 images au lieu de 300.
    """
    windows = [
        (0, SAMPLE_WINDOW_MS),
        (audio_ms // 2, audio_ms // 2 + SAMPLE_WINDOW_MS),
        (max(0, audio_ms - SAMPLE_WINDOW_MS), audio_ms),
    ]
    keep = {
        s.index
        for s in scenes
        for w0, w1 in windows
        if s.start_ms < w1 and s.end_ms > w0
    }
    return sorted(keep)


def duration_report(scenes: list[Scene]) -> str:
    """Indicateur de santé de la segmentation (SPEC §5.2).

    Trop de coupes forcées signale un script mal ponctué ou un TTS qui ne
    respire pas.
    """
    if not scenes:
        return "aucune scène"
    durations = sorted(s.duration_ms for s in scenes)
    median = durations[len(durations) // 2]
    forced = sum(1 for s in scenes if s.forced)
    return (
        f"{len(scenes)} scènes — durées min {durations[0] / 1000:.1f} s / "
        f"médiane {median / 1000:.1f} s / max {durations[-1] / 1000:.1f} s "
        f"— {forced} coupes forcées ({100 * forced // len(scenes)}%)"
    )
