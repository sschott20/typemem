import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


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


@dataclass
class MemoryItem:
    document: str
    tier: MemoryTier
    memory_type: MemoryType
    robot_id: str
    timestamp: float = field(default_factory=time.time)
    waypoint: Optional[int] = None
    task_id: Optional[int] = None
    keywords: str = ""
    source: str = ""
    id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))

    def to_metadata(self) -> Dict[str, Any]:
        meta = {
            "tier": self.tier.label,
            "memory_type": self.memory_type.value,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "keywords": self.keywords,
            "source": self.source,
        }
        if self.waypoint is not None:
            meta["waypoint"] = self.waypoint
        if self.task_id is not None:
            meta["task_id"] = self.task_id
        return meta

    @classmethod
    def from_chromadb(cls, doc_id: str, document: str, metadata: Dict[str, Any]) -> "MemoryItem":
        return cls(
            id=doc_id,
            document=document,
            tier=_TIER_MAP[metadata["tier"]],
            memory_type=_TYPE_MAP[metadata["memory_type"]],
            robot_id=metadata.get("robot_id", ""),
            timestamp=metadata.get("timestamp", 0.0),
            waypoint=metadata.get("waypoint"),
            task_id=metadata.get("task_id"),
            keywords=metadata.get("keywords", ""),
            source=metadata.get("source", ""),
        )
