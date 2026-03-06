from __future__ import annotations
from abc import ABC, abstractmethod

from typemem.types import MemoryEntry, SearchResult


class MemoryStore(ABC):
    """Abstract memory store. Dumb key-value + vector search. No policy."""

    @abstractmethod
    def add(self, text: str, metadata: dict | None = None, id: str | None = None) -> str:
        """Store a memory. Returns its ID (generated if not provided)."""
        ...

    def add_batch(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> list[str]:
        """Store multiple memories in one call. Default falls back to add() loop."""
        result_ids = []
        for i, text in enumerate(texts):
            md = metadatas[i] if metadatas else None
            mid = ids[i] if ids else None
            result_ids.append(self.add(text, metadata=md, id=mid))
        return result_ids

    @abstractmethod
    def search(self, query: str, n: int = 10, filters: dict | None = None) -> list[SearchResult]:
        """Semantic search. Returns results sorted by relevance (lowest distance first)."""
        ...

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a memory by ID."""
        ...

    @abstractmethod
    def update(self, id: str, text: str | None = None, metadata: dict | None = None) -> None:
        """Update a memory's text and/or metadata."""
        ...

    @abstractmethod
    def get(self, id: str) -> MemoryEntry | None:
        """Get a single memory by ID."""
        ...

    @abstractmethod
    def get_all(self, filters: dict | None = None) -> list[MemoryEntry]:
        """Get all memories, optionally filtered by metadata."""
        ...

    @abstractmethod
    def count(self, filters: dict | None = None) -> int:
        """Count memories, optionally filtered by metadata."""
        ...
