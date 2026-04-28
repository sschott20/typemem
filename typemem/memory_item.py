import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class MemoryTier(Enum):
    """Memory tier levels. Tiers are opaque levels — typemem core does not
    interpret them. Retention policy, consolidation flow, and tier semantics
    are all plugin-level concerns.

    The retention attribute is kept for backward compatibility and as a
    *default* that TierRetentionGC uses if no override is provided, but it
    is no longer a framework guarantee. Applications can ignore tiers entirely
    or use their own tier conventions on top of these levels.
    """
    M0 = ("M0", 0)
    M1 = ("M1", 1)
    M2 = ("M2", 2)
    M3 = ("M3", 3)

    def __init__(self, label: str, level: int):
        self.label = label
        self.level = level

    def __str__(self):
        return self.label


# Default retention suggestions — used by TierRetentionGC if no override given.
# NOT a framework guarantee; plugins decide their own retention policies.
DEFAULT_TIER_RETENTION: Dict["MemoryTier", Optional[int]] = {}  # populated below


class MemoryType(Enum):
    OBSERVATION = "observation"
    SUMMARY = "summary"
    INSTRUCTION = "instruction"
    ACTION = "action"
    LESSON = "lesson"


_TIER_MAP = {t.label: t for t in MemoryTier}
_TYPE_MAP = {t.value: t for t in MemoryType}

# Populate default retention suggestions (in seconds; None = persistent)
DEFAULT_TIER_RETENTION.update({
    MemoryTier.M0: 60,
    MemoryTier.M1: 600,
    MemoryTier.M2: 3600,
    MemoryTier.M3: None,
})


_CORE_META_KEYS = {"tier", "memory_type", "robot_id", "timestamp",
                   "frame_ref", "tags"}


@dataclass
class MemoryItem:
    document: str
    tier: MemoryTier
    memory_type: MemoryType
    robot_id: str
    timestamp: float = field(default_factory=time.time)
    frame_ref: Optional[str] = None
    # Tags carry all provenance/state info. "source:plugin_name" identifies
    # origin (set automatically when `source=` kwarg is used at construction),
    # "processed:plugin_name" marks handling state, etc.
    tags: Set[str] = field(default_factory=set)
    # Domain-specific metadata. typemem core doesn't interpret these.
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    # Compat: kept as a regular field for ergonomic construction
    # (MemoryItem(..., source="X")). Stored form of truth is the "source:X"
    # tag — __post_init__ mirrors this field into tags.
    source: str = ""

    def __post_init__(self):
        # Mirror source field into tags
        if self.source:
            self.tags.add(f"source:{self.source}")
        else:
            for t in self.tags:
                if t.startswith("source:"):
                    self.source = t[len("source:"):]
                    break
        # Mirror memory_type into tags (type:observation, type:lesson, etc.)
        if self.memory_type is not None:
            self.tags.add(f"type:{self.memory_type.value}")
        # Mirror tier into tags (tier:M0, tier:M1, etc.) so it's filterable
        # alongside other tags. The structured `tier` field is kept for
        # ergonomics, retention queries, and ChromaDB indexed lookups.
        if self.tier is not None:
            self.tags.add(f"tier:{self.tier.label}")

    def to_metadata(self) -> Dict[str, Any]:
        meta = {
            "tier": self.tier.label,
            "memory_type": self.memory_type.value,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
        }
        if self.frame_ref is not None:
            meta["frame_ref"] = self.frame_ref
        if self.tags:
            meta["tags"] = ",".join(sorted(self.tags))
        for k, v in self.metadata.items():
            if v is None or k in _CORE_META_KEYS:
                continue
            meta[k] = v
        return meta

    @classmethod
    def from_chromadb(cls, doc_id: str, document: str, metadata: Dict[str, Any]) -> "MemoryItem":
        tags_str = metadata.get("tags", "")
        tags = set(t for t in tags_str.split(",") if t) if tags_str else set()
        # Legacy: old chromadb rows may have a `source` metadata field
        legacy_source = metadata.get("source")
        if legacy_source:
            tags.add(f"source:{legacy_source}")
        domain_meta = {k: v for k, v in metadata.items()
                       if k not in _CORE_META_KEYS and k != "source"}
        return cls(
            id=doc_id,
            document=document,
            tier=_TIER_MAP[metadata["tier"]],
            memory_type=_TYPE_MAP[metadata["memory_type"]],
            robot_id=metadata.get("robot_id", ""),
            timestamp=metadata.get("timestamp", 0.0),
            frame_ref=metadata.get("frame_ref"),
            tags=tags,
            metadata=domain_meta,
        )
