"""Versioned elite-formula archives for baseline search backends."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from .candidates import CandidateScore
from .ir import FormulaSyntaxError, canonical_tokens
from .vocab import FORMULA_VOCAB, FormulaVocab

ELITE_ARCHIVE_VERSION = 1
DEFAULT_ELITE_ARCHIVE_SIZE = 64
_BASELINE_ORDER = ("gp", "tpe", "random")


@dataclass(frozen=True)
class EliteFormula:
    tokens: tuple[int, ...]
    canonical_tokens: tuple[int, ...]
    formula_text: str
    source: str
    direction: int
    val_reward: float
    train_reward: float
    complexity_cost: float

    def __post_init__(self) -> None:
        if not self.tokens or not self.canonical_tokens:
            raise ValueError("elite formula tokens must not be empty")
        if not math.isfinite(float(self.val_reward)) or not math.isfinite(
            float(self.train_reward)
        ):
            raise ValueError("elite formula rewards must be finite")

    @property
    def deterministic_key(self) -> str:
        return ",".join(f"{int(token):08d}" for token in self.tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": list(self.tokens),
            "canonical_tokens": list(self.canonical_tokens),
            "formula_text": self.formula_text,
            "source": self.source,
            "direction": self.direction,
            "val_reward": self.val_reward,
            "train_reward": self.train_reward,
            "complexity_cost": self.complexity_cost,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EliteFormula":
        return cls(
            tokens=tuple(int(token) for token in payload["tokens"]),
            canonical_tokens=tuple(
                int(token) for token in payload["canonical_tokens"]
            ),
            formula_text=str(payload["formula_text"]),
            source=str(payload["source"]),
            direction=int(payload["direction"]),
            val_reward=float(payload["val_reward"]),
            train_reward=float(payload["train_reward"]),
            complexity_cost=float(payload["complexity_cost"]),
        )


@dataclass(frozen=True)
class EliteArchive:
    source_backends: tuple[str, ...]
    entries: tuple[EliteFormula, ...]
    version: int = ELITE_ARCHIVE_VERSION

    def __post_init__(self) -> None:
        if int(self.version) != ELITE_ARCHIVE_VERSION:
            raise ValueError(
                f"unsupported elite archive version {self.version}; "
                f"current is {ELITE_ARCHIVE_VERSION}"
            )
        if len(set(self.source_backends)) != len(self.source_backends):
            raise ValueError("source_backends must be unique")
        canonical = [entry.canonical_tokens for entry in self.entries]
        if len(set(canonical)) != len(canonical):
            raise ValueError("elite archive entries must be canonically unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_backends": list(self.source_backends),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EliteArchive":
        version = int(payload.get("version", -1))
        if version != ELITE_ARCHIVE_VERSION:
            raise ValueError(
                f"unsupported elite archive version {version}; "
                f"current is {ELITE_ARCHIVE_VERSION}"
            )
        entries = tuple(
            EliteFormula.from_dict(entry) for entry in payload.get("entries", [])
        )
        return cls(
            source_backends=tuple(
                str(source) for source in payload.get("source_backends", [])
            ),
            entries=entries,
            version=version,
        )


def _rank_key(entry: EliteFormula) -> tuple[float, float, float, str]:
    return (
        -float(entry.val_reward),
        -float(entry.train_reward),
        float(entry.complexity_cost),
        entry.deterministic_key,
    )


def _ordered_sources(sources: Iterable[str]) -> tuple[str, ...]:
    unique = set(str(source) for source in sources)
    ordered = [source for source in _BASELINE_ORDER if source in unique]
    ordered.extend(sorted(unique - set(_BASELINE_ORDER)))
    return tuple(ordered)


def build_elite_archive(
    backend: str,
    scores: Iterable[CandidateScore],
    *,
    max_size: int = DEFAULT_ELITE_ARCHIVE_SIZE,
    vocab: FormulaVocab | None = None,
) -> EliteArchive:
    """Select the deterministic top eligible canonical formulas."""

    max_size = int(max_size)
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    vocab = vocab or FORMULA_VOCAB
    candidates: list[EliteFormula] = []
    for score in scores:
        if (
            not score.eligible
            or score.tokens is None
            or not math.isfinite(float(score.val_reward))
            or not math.isfinite(float(score.train_reward))
        ):
            continue
        try:
            canonical = canonical_tokens(score.tokens, vocab)
        except FormulaSyntaxError:
            continue
        if canonical is None:
            continue
        candidates.append(
            EliteFormula(
                tokens=tuple(int(token) for token in score.tokens),
                canonical_tokens=tuple(int(token) for token in canonical),
                formula_text=score.formula_text,
                source=str(backend),
                direction=int(score.direction),
                val_reward=float(score.val_reward),
                train_reward=float(score.train_reward),
                complexity_cost=float(score.complexity_cost),
            )
        )
    candidates.sort(key=_rank_key)
    unique: list[EliteFormula] = []
    seen: set[tuple[int, ...]] = set()
    for entry in candidates:
        if entry.canonical_tokens in seen:
            continue
        seen.add(entry.canonical_tokens)
        unique.append(entry)
        if len(unique) >= max_size:
            break
    return EliteArchive(source_backends=(str(backend),), entries=tuple(unique))


def merge_elite_archives(
    archives: Iterable[EliteArchive],
    *,
    max_size: int = DEFAULT_ELITE_ARCHIVE_SIZE,
) -> EliteArchive:
    """Merge, re-rank and canonically deduplicate baseline archives."""

    max_size = int(max_size)
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    archives = tuple(archives)
    sources = _ordered_sources(
        source for archive in archives for source in archive.source_backends
    )
    entries = sorted(
        (entry for archive in archives for entry in archive.entries),
        key=_rank_key,
    )
    unique: list[EliteFormula] = []
    seen: set[tuple[int, ...]] = set()
    for entry in entries:
        if entry.canonical_tokens in seen:
            continue
        seen.add(entry.canonical_tokens)
        unique.append(entry)
        if len(unique) >= max_size:
            break
    return EliteArchive(source_backends=sources, entries=tuple(unique))


def write_elite_archive(path: str | Path, archive: EliteArchive) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(archive.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def load_elite_archive(path: str | Path) -> EliteArchive:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("elite archive payload must be an object")
    return EliteArchive.from_dict(payload)
