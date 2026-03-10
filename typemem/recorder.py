import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

logger = logging.getLogger(__name__)


@dataclass
class RecordedEvent:
    event: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"event": self.event, "data": self.data, "ts": self.timestamp}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RecordedEvent":
        return cls(event=d["event"], data=d["data"], timestamp=d["ts"])


class SessionRecorder:
    """Records memory system events for replay and benchmarking."""

    def __init__(self, path: str, enabled: bool = True, flush_interval: int = 50):
        self._path = path
        self._enabled = enabled
        self._events: List[RecordedEvent] = []
        self._flush_interval = flush_interval
        self._since_flush = 0

    def _auto_flush(self):
        self._since_flush += 1
        if self._since_flush >= self._flush_interval:
            self.flush()

    def record_write(self, tier: str, document: str, metadata: Dict, item_id: str):
        if not self._enabled:
            return
        self._events.append(RecordedEvent(
            event="write",
            data={"tier": tier, "document": document, "metadata": metadata, "item_id": item_id},
        ))
        self._auto_flush()

    def record_consolidation(self, strategy: str, source_ids: List[str], result_id: str):
        if not self._enabled:
            return
        self._events.append(RecordedEvent(
            event="consolidate",
            data={"strategy": strategy, "source_ids": source_ids, "result_id": result_id},
        ))
        self._auto_flush()

    def record_injection(self, stage: str, memory_ids: List[str], scores: List[float]):
        if not self._enabled:
            return
        self._events.append(RecordedEvent(
            event="inject",
            data={"stage": stage, "memory_ids": memory_ids, "scores": scores},
        ))
        self._auto_flush()

    def get_events(self) -> List[RecordedEvent]:
        return list(self._events)

    def flush(self):
        if not self._enabled or not self._events:
            return
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "a") as f:
            for event in self._events:
                f.write(json.dumps(event.to_dict()) + "\n")
        self._events.clear()
        self._since_flush = 0

    def load_events(self) -> List[RecordedEvent]:
        events = []
        if os.path.exists(self._path):
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(RecordedEvent.from_dict(json.loads(line)))
        return events

    def replay(self) -> Iterator[RecordedEvent]:
        yield from self._events
