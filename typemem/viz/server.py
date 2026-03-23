"""Zero-dependency HTTP server for memory visualization."""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, List, Optional

from typemem.memory_item import MemoryTier
from typemem.memory_manager import MemoryManager


@dataclass
class VizEvent:
    """An event pushed to SSE subscribers."""
    event_type: str  # "add", "consolidation", "injection"
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class _EventBus:
    """Thread-safe event queue with SSE fan-out."""

    def __init__(self, max_events: int = 10_000):
        self._events: list[VizEvent] = []
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._max_events = max_events

    def push(self, event: VizEvent):
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def get_events(self) -> list[VizEvent]:
        with self._lock:
            return list(self._events)


def _serialize_event(event: VizEvent) -> dict:
    d = asdict(event)
    d["type"] = event.event_type
    return d


def _entry_to_dict(item) -> dict:
    return {
        "id": item.id,
        "text": item.document,
        "tier": item.tier.label,
        "memory_type": item.memory_type.value,
        "robot_id": item.robot_id,
        "timestamp": item.timestamp,
        "keywords": item.keywords,
        "source": item.source,
    }


class VizHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _json_response(self, data: Any, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        manager: MemoryManager = self.server.manager
        bus: _EventBus = self.server.event_bus
        path = self.path.split("?")[0]

        if path == "/":
            from typemem.viz.frontend import FRONTEND_HTML
            self._html_response(FRONTEND_HTML)

        elif path == "/api/stats":
            tiers = {}
            total = 0
            for tier in MemoryTier:
                count = manager.count(tier=tier)
                if count:
                    tiers[tier.label] = count
                total += count
            consol_events = [e for e in bus.get_events() if e.event_type == "consolidation"]
            inject_events = [e for e in bus.get_events() if e.event_type == "injection"]
            self._json_response({
                "total": total,
                "tiers": tiers,
                "events_count": len(bus.get_events()),
                "consolidations_count": len(consol_events),
                "injections_count": len(inject_events),
            })

        elif path == "/api/entries":
            entries = []
            for tier in MemoryTier:
                for item in manager.get_by_tier(tier):
                    d = _entry_to_dict(item)
                    d["links"] = len(manager.links.get_links(item.id))
                    entries.append(d)
            entries.sort(key=lambda e: e["timestamp"], reverse=True)
            self._json_response(entries)

        elif path == "/api/events":
            self._json_response([_serialize_event(e) for e in bus.get_events()])

        elif path == "/api/consolidations":
            events = [e for e in bus.get_events() if e.event_type == "consolidation"]
            self._json_response([_serialize_event(e) for e in events])

        elif path == "/api/injections":
            events = [e for e in bus.get_events() if e.event_type == "injection"]
            self._json_response([_serialize_event(e) for e in events])

        elif path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            sub = bus.subscribe()
            try:
                while True:
                    try:
                        event = sub.get(timeout=30)
                        data = json.dumps(_serialize_event(event), default=str)
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                bus.unsubscribe(sub)

        else:
            self.send_error(404)


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class VizServer:
    """Threaded HTTP server for memory visualization."""

    def __init__(self, manager: MemoryManager, engine=None, injector=None, port: int = 8811):
        self._manager = manager
        self._engine = engine
        self._injector = injector
        self.event_bus = _EventBus()
        self._server = _ThreadedHTTPServer(("127.0.0.1", port), VizHandler)
        self._server.manager = manager
        self._server.event_bus = self.event_bus
        self.port = self._server.server_address[1]
        self._thread: Optional[threading.Thread] = None

    def notify(self, event_type: str, **details):
        """Push an event to all SSE subscribers."""
        self.event_bus.push(VizEvent(event_type=event_type, details=details))

    def notify_add(self, item):
        """Notify that a memory item was added."""
        self.notify("add", id=item.id, text=item.document, tier=item.tier.label)

    def notify_consolidation(self, strategy_name: str,
                              inputs: Optional[list[dict]] = None,
                              outputs: Optional[list[dict]] = None,
                              duration_ms: float = 0):
        """Notify that consolidation ran.

        inputs/outputs: list of {"id": ..., "text": ..., "tier": ...}
        """
        inputs = inputs or []
        outputs = outputs or []
        self.notify("consolidation", name=strategy_name,
                    inputs=inputs, outputs=outputs,
                    input_count=len(inputs), output_count=len(outputs),
                    duration_ms=round(duration_ms, 1))

    def notify_injection(self, stage: str, query: str, context: str,
                          results: Optional[list[dict]] = None,
                          duration_ms: float = 0):
        """Notify that injection occurred.

        results: list of {"id": ..., "text": ..., "tier": ..., "score": ...}
        """
        results = results or []
        self.notify("injection", stage=stage, query=query,
                    context=context, results=results,
                    result_count=len(results),
                    duration_ms=round(duration_ms, 1))

    def start(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)
