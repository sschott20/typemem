"""Standalone event bus for decoupled data flow.

Hosts push events to named channels. Plugins subscribe and drain their own deques.
Thread-safe: deque operations are atomic under CPython GIL. The subscriber registry
is guarded by a threading.Lock.
"""

import collections
import threading
from typing import Dict, List

_lock = threading.Lock()
_registry: Dict[str, List[collections.deque]] = {}


def push(channel: str, data: dict) -> None:
    """Push an event to all subscribers of a channel. Fire-and-forget."""
    with _lock:
        subscribers = list(_registry.get(channel, []))
    for dq in subscribers:
        dq.append(data)


def subscribe(channel: str, maxlen: int = 1000) -> collections.deque:
    """Subscribe to a channel. Returns a deque to drain.
    Multiple subscribers to the same channel each get their own deque (fan-out).
    """
    dq = collections.deque(maxlen=maxlen)
    with _lock:
        if channel not in _registry:
            _registry[channel] = []
        _registry[channel].append(dq)
    return dq


def channels() -> List[str]:
    """Return list of channels that have subscribers."""
    with _lock:
        return list(_registry.keys())


def drain(dq: collections.deque) -> list:
    """Drain all items from a deque. Thread-safe via deque's atomic popleft."""
    items = []
    while dq:
        try:
            items.append(dq.popleft())
        except IndexError:
            break
    return items


def reset() -> None:
    """Clear all subscriptions. For testing only."""
    with _lock:
        _registry.clear()
