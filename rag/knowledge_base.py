"""Curated career knowledge base with BM25 retrieval.

Each chunk describes one aspect of one role (overview, skills, interview topics,
portfolio projects, roadmap). Agents retrieve the chunks that match the target
role and cite their ids, so advice is grounded in reviewable reference text
instead of the model's memory alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rag.bm25 import BM25, tokenize

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"
KINDS = ("overview", "skills", "interview", "projects", "roadmap")
DEFAULT_MIN_SCORE = 5.0
TITLE_BOOST = 3
# Words too common to identify a role on their own ("Data Engineer" must not resolve to Data Scientist).
GENERIC_ROLE_TOKENS = frozenset(
    {"engineer", "developer", "specialist", "manager", "lead", "senior", "junior", "associate", "data"}
)


@dataclass(frozen=True)
class Chunk:
    id: str
    role: str
    kind: str
    title: str
    text: str
    aliases: str = ""


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]) -> None:
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError("knowledge base chunk ids must be unique")
        self.chunks = chunks
        self._bm25 = BM25([self._index_tokens(c) for c in chunks])
        self._role_tokens = self._build_role_tokens(chunks)

    @staticmethod
    def _index_tokens(chunk: Chunk) -> list[str]:
        # Repeating role, aliases and title makes "Cloud Engineer ..." queries land on cloud chunks.
        header = f"{chunk.role} {chunk.aliases} {chunk.title} " * TITLE_BOOST
        return tokenize(f"{header}{chunk.text}")

    @staticmethod
    def _build_role_tokens(chunks: list[Chunk]) -> dict[str, set[str]]:
        tokens: dict[str, set[str]] = {}
        for chunk in chunks:
            tokens.setdefault(chunk.role, set()).update(tokenize(f"{chunk.role} {chunk.aliases}"))
        return {role: found - GENERIC_ROLE_TOKENS for role, found in tokens.items()}

    @classmethod
    def from_json(cls, path: Path = DATA_PATH) -> KnowledgeBase:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Chunk(**row) for row in rows])

    @property
    def roles(self) -> list[str]:
        return sorted({c.role for c in self.chunks})

    def retrieve(
        self,
        query: str,
        *,
        k: int = 3,
        kinds: tuple[str, ...] | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[Hit]:
        """Top-k chunks for `query`, optionally restricted to some kinds.

        Returns an empty list when nothing scores above `min_score`, so callers can
        say "no reference material" instead of grounding on an irrelevant chunk.
        """
        scores = self._bm25.scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        hits = [
            Hit(self.chunks[i], scores[i])
            for i in ranked
            if scores[i] >= min_score and (kinds is None or self.chunks[i].kind in kinds)
        ]
        return hits[:k]

    def matching_roles(self, role: str) -> list[str]:
        """The best-matching role(s) for `role`, judged by shared distinctive words.

        Generic words ("engineer", "data") never count, so "Quantum Engineer" matches nothing
        instead of borrowing an unrelated role. Only roles with the highest overlap are kept:
        "Security Analyst" resolves to Cybersecurity Engineer, not also to Data Analyst.
        """
        distinct = set(tokenize(role)) - GENERIC_ROLE_TOKENS
        overlaps = {name: len(tokens & distinct) for name, tokens in self._role_tokens.items()}
        best = max(overlaps.values(), default=0)
        return sorted(name for name, overlap in overlaps.items() if best and overlap == best)

    def retrieve_for_role(
        self,
        role: str,
        topic: str,
        *,
        k: int = 2,
        kinds: tuple[str, ...] | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[Hit]:
        """BM25 retrieval restricted to the roles that match `role`; empty if none match."""
        candidates = set(self.matching_roles(role))
        if not candidates:
            return []
        hits = self.retrieve(f"{role} {topic}", k=len(self.chunks), kinds=kinds, min_score=min_score)
        return [hit for hit in hits if hit.chunk.role in candidates][:k]


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.from_json()


def format_context(hits: list[Hit]) -> str:
    """Render hits as a prompt block with citable ids."""
    if not hits:
        return (
            "REFERENCE MATERIAL: none available for this role. Use general industry knowledge and keep claims modest."
        )
    blocks = [f"[{h.chunk.id}] {h.chunk.role} - {h.chunk.title}\n{h.chunk.text}" for h in hits]
    return "REFERENCE MATERIAL (trusted, curated; cite the ids you rely on in `sources`):\n\n" + "\n\n".join(blocks)


def retrieval_summary(hits: list[Hit]) -> list[dict[str, object]]:
    return [{"id": h.chunk.id, "title": f"{h.chunk.role} - {h.chunk.title}", "score": round(h.score, 2)} for h in hits]


def cited_sources(cited: list[str], hits: list[Hit]) -> list[str]:
    """Keep only citations that point at chunks that were actually retrieved."""
    allowed = {h.chunk.id for h in hits}
    seen: list[str] = []
    for source in cited:
        source = source.strip().strip("[]")
        if source in allowed and source not in seen:
            seen.append(source)
    return seen
