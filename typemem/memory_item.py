import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class MemoryTier(Enum):
    M0 = ("M0", 0, 60)       # raw observations, 60s retention
    M1 = ("M1", 1, 600)      # NL observations, 10min retention
    M2 = ("M2", 2, 3600)     # summaries, 1hr retention
    M3 = ("M3", 3, None)     # long-term knowledge, persistent

    def __init__(self, label: str, level: int, retention: Optional[int]):
        self.label = label
        self.level = level
        self.retention = retention

    def __str__(self):
        return self.label


class MemoryType(Enum):
    OBSERVATION = "observation"
    SUMMARY = "summary"
    INSTRUCTION = "instruction"
    ACTION = "action"
    LESSON = "lesson"


_TIER_MAP = {t.label: t for t in MemoryTier}
_TYPE_MAP = {t.value: t for t in MemoryType}


_CORE_META_KEYS = {"tier", "memory_type", "robot_id", "timestamp",
                   "source", "frame_ref", "tags"}


@dataclass
class MemoryItem:
    document: str
    tier: MemoryTier
    memory_type: MemoryType
    robot_id: str
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    frame_ref: Optional[str] = None
    # Tags are free-form labels for filtering/state tracking. Used for both
    # processing state ("processed:plugin_name") and classification/keywords
    # ("about:person", "topic:waypoint_0"). Replaces the old `keywords` field.
    tags: Set[str] = field(default_factory=set)
    # Domain-specific metadata (waypoint, task_id, or anything else a plugin
    # needs to attach). typemem core doesn't interpret these — they round-trip
    # through ChromaDB metadata transparently.
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))

    def to_metadata(self) -> Dict[str, Any]:
        meta = {
            "tier": self.tier.label,
            "memory_type": self.memory_type.value,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "source": self.source,
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
        domain_meta = {k: v for k, v in metadata.items() if k not in _CORE_META_KEYS}
        return cls(
            id=doc_id,
            document=document,
            tier=_TIER_MAP[metadata["tier"]],
            memory_type=_TYPE_MAP[metadata["memory_type"]],
            robot_id=metadata.get("robot_id", ""),
            timestamp=metadata.get("timestamp", 0.0),
            source=metadata.get("source", ""),
            frame_ref=metadata.get("frame_ref"),
            tags=tags,
            metadata=domain_meta,
        )
