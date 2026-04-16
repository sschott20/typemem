"""Tier retention garbage collector plugin.

Deletes memory items older than a configured retention window. Retention
is a plugin-level policy — applications can pass custom retention values
per tier, or rely on the defaults in DEFAULT_TIER_RETENTION.

Typemem does not register this automatically — applications opt in.
"""

import logging
import time
from typing import Dict, List, Optional

from typemem.memory_item import MemoryTier, DEFAULT_TIER_RETENTION
from typemem.memory_manager import MemoryManager
from typemem.plugins.base import ConsolidationPlugin

logger = logging.getLogger(__name__)


class TierRetentionGC(ConsolidationPlugin):
    """Periodically deletes items whose age exceeds a per-tier retention.

    Retention is configured via the `retention` dict passed to __init__
    (tier → seconds, None = persistent). If None is passed, uses the
    framework default DEFAULT_TIER_RETENTION.
    """

    def __init__(self, interval_seconds: float = 60.0,
                 retention: Optional[Dict[MemoryTier, Optional[int]]] = None):
        """
        Args:
            interval_seconds: how often to run GC (default 60s).
            retention: per-tier retention in seconds. None value = persistent.
                If the whole arg is None, uses DEFAULT_TIER_RETENTION.
        """
        self._interval = interval_seconds
        self._retention = (
            dict(retention) if retention is not None else dict(DEFAULT_TIER_RETENTION)
        )

    @property
    def name(self) -> str:
        return "tier_retention_gc"

    @property
    def interval_seconds(self) -> float:
        return self._interval

    def run(self, manager: MemoryManager, llm=None) -> List[str]:
        now = time.time()
        total_expired = 0
        for tier, seconds in self._retention.items():
            if seconds is None:
                continue
            cutoff = now - seconds
            # Find items in this tier older than cutoff
            items = manager.get_by_tier(tier)
            expired_ids = [it.id for it in items if it.timestamp < cutoff]
            if expired_ids:
                for iid in expired_ids:
                    manager.delete(iid)
                total_expired += len(expired_ids)

        if total_expired > 0:
            logger.info("TierRetentionGC: expired %d items", total_expired)
        return []
