"""Zero-dependency HTTP server for memory visualization."""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from typemem.viz.tracing import (
    TracingStore, TracingSystem,
    StoreEvent, ConsolidationEvent, InjectionEvent,
)


def _serialize_event(event: StoreEvent | ConsolidationEvent | InjectionEvent) -> dict:
    """Convert event dataclass to JSON-safe dict."""
    d = asdict(event)
    if isinstance(event, ConsolidationEvent):
        d["type"] = "consolidation"
    elif isinstance(event, InjectionEvent):
        d["type"] = "injection"
    else:
        d["type"] = "store"
    return d


class VizHandler(BaseHTTPRequestHandler):
    """HTTP request handler for viz endpoints."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

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
        tracing: TracingStore = self.server.tracing_store
        tracing_system: TracingSystem = self.server.tracing_system

        path = self.path.split("?")[0]

        if path == "/":
            from typemem.viz.frontend import FRONTEND_HTML
            self._html_response(FRONTEND_HTML)

        elif path == "/api/stats":
            entries = tracing.inner.get_all()
            tiers = {}
            for e in entries:
                tier = e.metadata.get("_tier", "none")
                tiers[tier] = tiers.get(tier, 0) + 1
            self._json_response({
                "total": len(entries),
                "tiers": tiers,
                "events_count": len(tracing.get_events()),
                "consolidations_count": len(tracing_system.get_consolidations()),
                "injections_count": len(tracing_system.get_injections()),
            })

        elif path == "/api/entries":
            entries = tracing.inner.get_all()
            self._json_response([
                {"id": e.id, "text": e.text, "metadata": e.metadata, "timestamp": e.timestamp}
                for e in entries
            ])

        elif path == "/api/events":
            events = tracing.get_events()
            self._json_response([_serialize_event(e) for e in events])

        elif path == "/api/consolidations":
            events = tracing_system.get_consolidations()
            self._json_response([_serialize_event(e) for e in events])

        elif path == "/api/injections":
            events = tracing_system.get_injections()
            self._json_response([_serialize_event(e) for e in events])

        elif path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            sub = tracing.subscribe()
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
                tracing.unsubscribe(sub)

        else:
            self.send_error(404)


class VizServer:
    """Threaded HTTP server for memory visualization."""

    def __init__(self, tracing_store: TracingStore, tracing_system: TracingSystem, port: int = 8811):
        self._server = HTTPServer(("127.0.0.1", port), VizHandler)
        self._server.tracing_store = tracing_store
        self._server.tracing_system = tracing_system
        self.port = self._server.server_address[1]
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)
