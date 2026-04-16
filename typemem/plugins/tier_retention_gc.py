"""Tier retention garbage collector plugin.

Deletes memory items older than their tier's retention window. Default behavior
preserves the previous hardcoded expiry policy (M0: 60s, M1: 10min, M2: 1hr, M3: never).

Typemem ships this as a default plugin but does not register it automatically —
applications opt in by including it in their plugin list. Applications with
different retention policies can subclass or write a custom GC plugin instead.
"""

import logging
from typing import List, Optional

from typemem.memory_item import MemoryTier
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.plugins.base import ConsolidationPlugin

logger = logging.getLogger(__name__)


class TierRetentionGC(ConsolidationPlugin):
    """Periodically deletes items whose age exceeds their tier's retention.

    Uses the retention values defined on MemoryTier (M0: 60s, M1: 600s, M2: 3600s,
    M3: None = persistent). Also prunes the processed_index to drop entries for
    deleted items.
    """

    def __init__(self, interval_seconds: float = 60.0,
                 tiers: Optional[List[MemoryTier]] = None):
        """
        Args:
            interval_seconds: how often to run GC (default 60s).
            tiers: which tiers to check. None = all tiers with non-None retention.
        """
        self._interval = interval_seconds
        self._tiers = tiers if tiers is not None else [
            t for t in MemoryTier if t.retention is not None
        ]

    @property
    def name(self) -> str:
        return "tier_retention_gc"

    @property
    def interval_seconds(self) -> float:
        return self._interval

    def run(
        self,
        manager: MemoryManager,
        llm=None,
        processed_index: Optional[ProcessedIndex] = None,
    ) -> List[str]:
        total_expired = 0
        for tier in self._tiers:
            expired = manager.expire_tier(tier)
            total_expired += len(expired)

        if total_expired > 0 and processed_index is not None:
            all_ids = manager.all_ids()
            processed_index.prune(all_ids)
            processed_index.save()
            logger.info("TierRetentionGC: expired %d items", total_expired)

        return []
