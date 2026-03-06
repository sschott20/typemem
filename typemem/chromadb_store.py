from __future__ import annotations

import chromadb

from typemem.store import MemoryStore
from typemem.types import MemoryEntry, SearchResult, make_id


class ChromaDBStore(MemoryStore):
    """MemoryStore backed by ChromaDB with cosine similarity."""

    def __init__(self, persist_dir: str, collection_name: str = "typemem"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, text: str, metadata: dict | None = None, id: str | None = None) -> str:
        mid = id or make_id()
        kwargs = {"ids": [mid], "documents": [text]}
        if metadata:
            kwargs["metadatas"] = [metadata]
        self._collection.add(**kwargs)
        return mid

    def search(self, query: str, n: int = 10, filters: dict | None = None) -> list[SearchResult]:
        count = self._collection.count()
        if count == 0:
            return []
        actual_n = min(n, count)
        kwargs = {"query_texts": [query], "n_results": actual_n, "include": ["documents", "metadatas", "distances"]}
        if filters:
            kwargs["where"] = filters
        try:
            result = self._collection.query(**kwargs)
        except Exception:
            return []
        results = []
        for i, doc_id in enumerate(result["ids"][0]):
            entry = MemoryEntry(id=doc_id, text=result["documents"][0][i], metadata=result["metadatas"][0][i] or {})
            results.append(SearchResult(entry=entry, distance=result["distances"][0][i]))
        return results

    def delete(self, id: str) -> None:
        try:
            self._collection.delete(ids=[id])
        except Exception:
            pass

    def update(self, id: str, text: str | None = None, metadata: dict | None = None) -> None:
        kwargs = {"ids": [id]}
        if text is not None:
            kwargs["documents"] = [text]
        if metadata is not None:
            kwargs["metadatas"] = [metadata]
        self._collection.update(**kwargs)

    def get(self, id: str) -> MemoryEntry | None:
        try:
            result = self._collection.get(ids=[id], include=["documents", "metadatas"])
        except Exception:
            return None
        if not result["ids"]:
            return None
        return MemoryEntry(id=result["ids"][0], text=result["documents"][0], metadata=result["metadatas"][0] or {})

    def get_all(self, filters: dict | None = None) -> list[MemoryEntry]:
        kwargs = {"include": ["documents", "metadatas"]}
        if filters:
            kwargs["where"] = filters
        result = self._collection.get(**kwargs)
        return [
            MemoryEntry(id=result["ids"][i], text=result["documents"][i], metadata=result["metadatas"][i] or {})
            for i in range(len(result["ids"]))
        ]

    def count(self, filters: dict | None = None) -> int:
        if filters is None:
            return self._collection.count()
        result = self._collection.get(where=filters, include=[])
        return len(result["ids"])
