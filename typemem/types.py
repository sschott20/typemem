from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING
import time
import uuid

if TYPE_CHECKING:
    from typemem.store import MemoryStore


@dataclass
class MemoryEntry:
    """A single memory in the store."""
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    """A memory entry with its search distance."""
    entry: MemoryEntry
    distance: float


def make_id() -> str:
    """Generate a unique memory ID."""
    return uuid.uuid4().hex[:16]


ObservationFn = Callable[["dict", "MemoryStore"], "list[str]"]
ConsolidationFn = Callable[["MemoryStore"], "list[str]"]
InjectionFn = Callable[["str", "MemoryStore", "int"], "str"]
