from __future__ import annotations

import logging
import time

import chromadb

from typemem.store import MemoryStore
from typemem.types import MemoryEntry, SearchResult, make_id

logger = logging.getLogger(__name__)


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
        meta = dict(metadata) if metadata else {}
        if "_timestamp" not in meta:
            meta["_timestamp"] = time.time()
        kwargs = {"ids": [mid], "documents": [text], "metadatas": [meta]}
        self._collection.add(**kwargs)
        return mid

    def add_batch(self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None) -> list[str]:
        batch_ids = ids or [make_id() for _ in texts]
        now = time.time()
        batch_metas = [dict(m) for m in metadatas] if metadatas else [{} for _ in texts]
        for meta in batch_metas:
            if "_timestamp" not in meta:
                meta["_timestamp"] = now
        kwargs: dict = {"ids": batch_ids, "documents": texts, "metadatas": batch_metas}
        self._collection.add(**kwargs)
        return batch_ids

    def search(self, query: str, n: int = 10, filters: dict | None = None) -> list[SearchResult]:
        kwargs = {"query_texts": [query], "n_results": n, "include": ["documents", "metadatas", "distances"]}
        if filters:
            kwargs["where"] = filters
        try:
            result = self._collection.query(**kwargs)
        except Exception:
            logger.warning("ChromaDB query failed", exc_info=True)
            return []
        results = []
        for i, doc_id in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i] or {}
            ts = meta.get("_timestamp", 0.0)
            entry = MemoryEntry(id=doc_id, text=result["documents"][0][i], metadata=meta, timestamp=ts)
            results.append(SearchResult(entry=entry, distance=result["distances"][0][i]))
        return results

    def delete(self, id: str) -> None:
        try:
            self._collection.delete(ids=[id])
        except Exception:
            logger.warning("ChromaDB delete failed for id=%s", id, exc_info=True)

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
            logger.warning("ChromaDB get failed for id=%s", id, exc_info=True)
            return None
        if not result["ids"]:
            return None
        meta = result["metadatas"][0] or {}
        ts = meta.get("_timestamp", 0.0)
        return MemoryEntry(id=result["ids"][0], text=result["documents"][0], metadata=meta, timestamp=ts)

    def get_all(self, filters: dict | None = None) -> list[MemoryEntry]:
        kwargs = {"include": ["documents", "metadatas"]}
        if filters:
            kwargs["where"] = filters
        result = self._collection.get(**kwargs)
        entries = []
        for i in range(len(result["ids"])):
            meta = result["metadatas"][i] or {}
            ts = meta.get("_timestamp", 0.0)
            entries.append(MemoryEntry(id=result["ids"][i], text=result["documents"][i], metadata=meta, timestamp=ts))
        return entries

    def count(self, filters: dict | None = None) -> int:
        if filters is None:
            return self._collection.count()
        result = self._collection.get(where=filters, include=[])
        return len(result["ids"])
