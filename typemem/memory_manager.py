import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from typemem.memory_item import MemoryItem, MemoryTier
from typemem.link_index import LinkIndex

logger = logging.getLogger(__name__)

LINK_TOP_K = 3
LINK_DISTANCE_THRESHOLD = 1.0
DEDUP_DISTANCE_THRESHOLD = 0.05


class MemoryManager:
    """Core memory store backed by ChromaDB with Zettelkasten linking."""

    def __init__(
        self,
        persist_dir: str,
        robot_id: str,
        collection_name: str = "typemem",
        link_top_k: int = LINK_TOP_K,
        link_distance_threshold: float = LINK_DISTANCE_THRESHOLD,
        dedup_distance_threshold: float = DEDUP_DISTANCE_THRESHOLD,
    ):
        self._persist_dir = persist_dir
        self._robot_id = robot_id
        self._link_top_k = link_top_k
        self._link_distance_threshold = link_distance_threshold
        self._dedup_distance_threshold = dedup_distance_threshold
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = ONNXMiniLM_L6_V2()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )
        link_path = os.path.join(persist_dir, "links.json")
        self.links = LinkIndex(link_path)
        self._recorder = None
        self._on_add = None  # callback: (item) -> None, called after each add

    @property
    def persist_dir(self) -> str:
        return self._persist_dir

    def all_ids(self) -> set:
        """Return set of all item IDs in the collection."""
        return set(self._collection.get(include=[])["ids"])

    def set_recorder(self, recorder):
        self._recorder = recorder

    def set_on_add(self, callback):
        """Set a callback invoked after each successful add: callback(item)."""
        self._on_add = callback

    def add(self, item: MemoryItem, auto_link: bool = True, skip_dedup: bool = False) -> str:
        if not skip_dedup:
            existing_id = self._dedup_check(item)
            if existing_id is not None:
                return existing_id

        self._collection.add(
            ids=[item.id],
            documents=[item.document],
            metadatas=[item.to_metadata()],
        )
        self._post_insert(item, auto_link)
        return item.id

    def add_batch(self, items: List[MemoryItem], auto_link: bool = True) -> List[str]:
        if not items:
            return []

        result_ids: List[str] = []
        to_insert: List[MemoryItem] = []

        for item in items:
            existing_id = self._dedup_check(item)
            if existing_id is not None:
                result_ids.append(existing_id)
            else:
                to_insert.append(item)

        if to_insert:
            self._collection.add(
                ids=[item.id for item in to_insert],
                documents=[item.document for item in to_insert],
                metadatas=[item.to_metadata() for item in to_insert],
            )
            collection_count = self._collection.count()
            for item in to_insert:
                result_ids.append(item.id)
                self._post_insert(item, auto_link, collection_count)

        return result_ids

    def _dedup_check(self, item: MemoryItem) -> Optional[str]:
        if item.tier == MemoryTier.M0:
            return None
        existing_id = self._find_near_duplicate(item)
        if existing_id is not None:
            self._update_timestamp(existing_id, item.timestamp)
            logger.debug("Memory dedup: [%s] matched existing %s", item.tier.label, existing_id[:8])
            return existing_id
        return None

    def _post_insert(self, item: MemoryItem, auto_link: bool, collection_count: Optional[int] = None):
        logger.debug("Memory add: [%s] %s", item.tier.label, item.document[:80])
        if auto_link and item.tier != MemoryTier.M0:
            count = collection_count if collection_count is not None else self._collection.count()
            if count > 1:
                self._create_links(item)
        if self._recorder:
            self._recorder.record_write(
                tier=item.tier.label, document=item.document,
                metadata=item.to_metadata(), item_id=item.id,
            )
        if self._on_add:
            try:
                self._on_add(item)
            except Exception:
                logger.exception("on_add callback failed")

    def get(self, item_id: str) -> Optional[MemoryItem]:
        try:
            result = self._collection.get(ids=[item_id], include=["documents", "metadatas"])
        except Exception as e:
            logger.error("Failed to get memory item %s: %s", item_id, e)
            return None
        if not result["ids"]:
            return None
        return MemoryItem.from_chromadb(
            result["ids"][0], result["documents"][0], result["metadatas"][0],
        )

    def search(
        self,
        query: str,
        tiers: Optional[List[MemoryTier]] = None,
        n_results: int = 10,
    ) -> List[MemoryItem]:
        items, _ = self.search_with_distances(query, tiers=tiers, n_results=n_results)
        return items

    def search_with_distances(
        self,
        query: str,
        tiers: Optional[List[MemoryTier]] = None,
        n_results: int = 10,
    ) -> Tuple[List[MemoryItem], List[float]]:
        where = None
        if tiers:
            if len(tiers) == 1:
                where = {"tier": tiers[0].label}
            else:
                where = {"tier": {"$in": [t.label for t in tiers]}}

        if self._collection.count() == 0:
            return [], []

        try:
            result = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Memory search failed for query='%s': %s", query[:50], e)
            return [], []

        items = []
        distances = []
        for i, doc_id in enumerate(result["ids"][0]):
            item = MemoryItem.from_chromadb(
                doc_id, result["documents"][0][i], result["metadatas"][0][i],
            )
            items.append(item)
            distances.append(result["distances"][0][i])
        return items, distances

    def delete(self, item_id: str):
        try:
            self._collection.delete(ids=[item_id])
        except Exception as e:
            logger.error("Failed to delete memory item %s: %s", item_id, e)
        self.links.remove_node(item_id)

    def update_document(self, item_id: str, new_document: str):
        self._collection.update(ids=[item_id], documents=[new_document])

    def expire_tier(self, tier: MemoryTier, retention_seconds: Optional[int] = None) -> List[str]:
        """Delete items in `tier` older than `retention_seconds`. Returns deleted IDs.

        If retention_seconds is None, uses DEFAULT_TIER_RETENTION for backward compat.
        Prefer explicit retention arg — the default is just a suggestion.
        """
        if retention_seconds is None:
            from typemem.memory_item import DEFAULT_TIER_RETENTION
            retention_seconds = DEFAULT_TIER_RETENTION.get(tier)
        if retention_seconds is None:
            return []

        cutoff = time.time() - retention_seconds
        results = self._collection.get(
            where={"$and": [{"tier": tier.label}, {"timestamp": {"$lt": cutoff}}]},
            include=[],
        )
        expired_ids = results["ids"]

        if expired_ids:
            self._collection.delete(ids=expired_ids)
            for eid in expired_ids:
                self.links.remove_node(eid)
            logger.debug("Expired %d items from %s", len(expired_ids), tier.label)

        return expired_ids

    def get_by_tier(self, tier: MemoryTier) -> List[MemoryItem]:
        results = self._collection.get(
            where={"tier": tier.label},
            include=["documents", "metadatas"],
        )
        items = []
        for i, doc_id in enumerate(results["ids"]):
            items.append(MemoryItem.from_chromadb(
                doc_id, results["documents"][i], results["metadatas"][i],
            ))
        return items

    def get_by_source(self, source: str, tier: Optional[MemoryTier] = None) -> List[MemoryItem]:
        """Return items whose tags contain 'source:<source>'. Convenience wrapper
        around get_by_tag — maintained for readability; source is just a tag
        convention, not a core field."""
        return self.get_by_tag(f"source:{source}", tier=tier)

    def get_by_tag(
        self,
        tag: str,
        tier: Optional[MemoryTier] = None,
        exclude: bool = False,
    ) -> List[MemoryItem]:
        """Return items whose tags field contains the given tag.

        Args:
            tag: tag to match (e.g. "unprocessed").
            tier: optional tier filter.
            exclude: if True, return items that do NOT have the tag.
        """
        if tier is not None:
            results = self._collection.get(
                where={"tier": tier.label},
                include=["documents", "metadatas"],
            )
        else:
            results = self._collection.get(include=["documents", "metadatas"])
        items = []
        for i, doc_id in enumerate(results["ids"]):
            item = MemoryItem.from_chromadb(
                doc_id, results["documents"][i], results["metadatas"][i],
            )
            has_tag = tag in item.tags
            if has_tag != exclude:
                items.append(item)
        return items

    def update_metadata(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Merge ``updates`` into the item's metadata. Returns True if applied.

        Use for plugin-private cached state (e.g. parsed/compiled artifacts).
        Core fields (tier, memory_type, robot_id, timestamp, frame_ref, tags)
        are protected — pass them through tags or document edits instead.
        """
        try:
            result = self._collection.get(ids=[item_id], include=["metadatas"])
            if not result["ids"]:
                return False
            metadata = dict(result["metadatas"][0])
            from typemem.memory_item import _CORE_META_KEYS
            for k, v in updates.items():
                if k in _CORE_META_KEYS:
                    continue
                if v is None:
                    metadata.pop(k, None)
                else:
                    metadata[k] = v
            self._collection.update(ids=[item_id], metadatas=[metadata])
            return True
        except Exception as e:
            logger.error("update_metadata failed for %s: %s", item_id, e)
            return False

    def add_tag(self, item_id: str, tag: str) -> bool:
        """Add a tag to an item's tags set. Returns True if modified."""
        try:
            result = self._collection.get(ids=[item_id], include=["metadatas"])
            if not result["ids"]:
                return False
            metadata = result["metadatas"][0]
            tags_str = metadata.get("tags", "")
            tags = set(t for t in tags_str.split(",") if t) if tags_str else set()
            if tag in tags:
                return False
            tags.add(tag)
            metadata["tags"] = ",".join(sorted(tags))
            self._collection.update(ids=[item_id], metadatas=[metadata])
            return True
        except Exception as e:
            logger.error("add_tag failed for %s: %s", item_id, e)
            return False

    def remove_tag(self, item_id: str, tag: str) -> bool:
        """Remove a tag from an item's tags set. Returns True if modified."""
        try:
            result = self._collection.get(ids=[item_id], include=["metadatas"])
            if not result["ids"]:
                return False
            metadata = result["metadatas"][0]
            tags_str = metadata.get("tags", "")
            tags = set(t for t in tags_str.split(",") if t) if tags_str else set()
            if tag not in tags:
                return False
            tags.discard(tag)
            metadata["tags"] = ",".join(sorted(tags)) if tags else ""
            self._collection.update(ids=[item_id], metadatas=[metadata])
            return True
        except Exception as e:
            logger.error("remove_tag failed for %s: %s", item_id, e)
            return False

    def count(self, tier: Optional[MemoryTier] = None) -> int:
        if tier is None:
            return self._collection.count()
        results = self._collection.get(where={"tier": tier.label}, include=[])
        return len(results["ids"])

    def get_embeddings(self, item_ids: List[str]) -> Optional[List[List[float]]]:
        if not item_ids:
            return None
        try:
            result = self._collection.get(ids=item_ids, include=["embeddings"])
            return result.get("embeddings")
        except Exception as e:
            logger.error("Failed to get embeddings for %d items: %s", len(item_ids), e)
            return None

    def save(self):
        self.links.save()

    def _find_near_duplicate(self, item: MemoryItem) -> Optional[str]:
        try:
            result = self._collection.query(
                query_texts=[item.document],
                n_results=1,
                where={"tier": item.tier.label},
                include=["distances"],
            )
        except Exception:
            return None

        if not result["ids"] or not result["ids"][0]:
            return None

        distance = result["distances"][0][0]
        candidate_id = result["ids"][0][0]
        if distance < self._dedup_distance_threshold:
            return candidate_id
        return None

    def _update_timestamp(self, item_id: str, new_timestamp: float):
        try:
            self._collection.update(ids=[item_id], metadatas=[{"timestamp": new_timestamp}])
        except Exception as e:
            logger.error("Failed to update timestamp for %s: %s", item_id, e)

    def _create_links(self, item: MemoryItem):
        try:
            results = self._collection.query(
                query_texts=[item.document],
                n_results=min(self._link_top_k + 1, self._collection.count()),
                include=["distances"],
            )
        except Exception as e:
            logger.debug("Link creation query failed for item %s: %s", item.id, e)
            return

        for i, doc_id in enumerate(results["ids"][0]):
            if doc_id == item.id:
                continue
            distance = results["distances"][0][i]
            if distance <= self._link_distance_threshold:
                self.links.add_link(item.id, doc_id)
