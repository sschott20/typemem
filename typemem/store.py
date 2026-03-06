from __future__ import annotations
from abc import ABC, abstractmethod

from typemem.types import MemoryEntry, SearchResult


class MemoryStore(ABC):
    """Abstract memory store. Dumb key-value + vector search. No policy."""

    @abstractmethod
    def add(self, text: str, metadata: dict | None = None, id: str | None = None) -> str:
        """Store a memory. Returns its ID (generated if not provided)."""
        ...

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
