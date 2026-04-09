"""Replay recorded event-bus traffic into the typemem event bus.

Reads a JSONL file produced by TypeGo's EventRecorder and pushes events
into typemem's own event bus, optionally respecting original timing.
Frames are loaded from a companion directory and stored into a FrameStore.

Usage::

    from typemem.replay import EventReplay

    replay = EventReplay("path/to/recording_dir")
    replay.play()            # realtime
    replay.play(speed=0)     # instant (no delays)
    replay.play(speed=10)    # 10x faster
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

from typemem import events
from typemem.frame_store import FrameStore

logger = logging.getLogger(__name__)


@dataclass
class RawEvent:
    """A single recorded event bus event."""
    channel: str
    data: dict
    timestamp: float

    def to_dict(self) -> dict:
        return {"ch": self.channel, "data": self.data, "ts": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict) -> "RawEvent":
        return cls(channel=d["ch"], data=d["data"], timestamp=d["ts"])


def load_events(jsonl_path: str) -> List[RawEvent]:
    """Load events from a JSONL file, sorted by timestamp."""
    out: List[RawEvent] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(RawEvent.from_dict(json.loads(line)))
    out.sort(key=lambda e: e.timestamp)
    return out


class EventReplay:
    """Replay a TypeGo event recording into the typemem event bus.

    Parameters
    ----------
    recording_dir : str
        Directory containing ``events.jsonl`` and optionally ``frames/``.
    frame_store : FrameStore | None
        If provided, frame images from the recording are copied into this
        store so that observation plugins (e.g. CaptionObserver) can load
        them during replay.
    """

    def __init__(
        self,
        recording_dir: str,
        frame_store: Optional[FrameStore] = None,
    ):
        self._recording_dir = recording_dir
        self._jsonl_path = os.path.join(recording_dir, "events.jsonl")
        self._frames_dir = os.path.join(recording_dir, "frames")
        self._frame_store = frame_store

        if not os.path.isfile(self._jsonl_path):
            raise FileNotFoundError(f"No events.jsonl in {recording_dir}")

        self._events = load_events(self._jsonl_path)
        logger.info("Loaded %d events from %s", len(self._events), self._jsonl_path)

    @property
    def events(self) -> List[RawEvent]:
        return list(self._events)

    @property
    def duration(self) -> float:
        """Wall-clock duration of the recording in seconds."""
        if len(self._events) < 2:
            return 0.0
        return self._events[-1].timestamp - self._events[0].timestamp

    @property
    def channels(self) -> List[str]:
        """Unique channels in the recording."""
        return sorted(set(e.channel for e in self._events))

    def play(
        self,
        speed: float = 1.0,
        channels: Optional[List[str]] = None,
    ) -> int:
        """Push events into the typemem event bus.

        Parameters
        ----------
        speed : float
            Playback speed multiplier.  ``1.0`` = realtime, ``0`` = instant
            (no delays), ``10.0`` = 10x faster.
        channels : list[str] | None
            If set, only replay events from these channels.

        Returns
        -------
        int
            Number of events pushed.
        """
        pushed = 0
        prev_ts: Optional[float] = None

        for ev in self._events:
            if channels and ev.channel not in channels:
                continue

            # Timing
            if speed > 0 and prev_ts is not None:
                delta = (ev.timestamp - prev_ts) / speed
                if delta > 0:
                    time.sleep(delta)
            prev_ts = ev.timestamp

            # Load frame into frame store if available
            if ev.channel == "frame" and self._frame_store is not None:
                self._load_frame(ev.data)

            events.push(ev.channel, ev.data)
            pushed += 1

        logger.info("Replayed %d events", pushed)
        return pushed

    def iter_events(
        self,
        channels: Optional[List[str]] = None,
    ) -> Iterator[RawEvent]:
        """Yield events without pushing or sleeping.

        Useful for custom replay loops or analysis.
        """
        for ev in self._events:
            if channels and ev.channel not in channels:
                continue
            yield ev

    def _load_frame(self, data: dict):
        """Copy a recorded frame JPEG into the live FrameStore."""
        frame_id = data.get("frame_id")
        if not frame_id:
            return

        src = os.path.join(self._frames_dir, f"{frame_id}.jpg")
        if not os.path.isfile(src):
            return

        # Copy JPEG directly into FrameStore's directory
        dst = os.path.join(self._frame_store._store_dir, f"{frame_id}.jpg")
        shutil.copy2(src, dst)

        # Write metadata so FrameStore.load() works
        meta_path = os.path.join(self._frame_store._store_dir, f"{frame_id}.json")
        meta = {
            "timestamp": data.get("timestamp", time.time()),
            "frame_id": frame_id,
        }
        if "waypoint" in data:
            meta["waypoint"] = data["waypoint"]
        with open(meta_path, "w") as f:
            json.dump(meta, f)
