"""Zero-dependency HTTP server for memory visualization."""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, List, Optional
from urllib.parse import urlparse, parse_qs

from typemem.memory_item import MemoryTier
from typemem.memory_manager import MemoryManager

_TIER_ORDER = {t.label: t.level for t in MemoryTier}

_DEFAULT_DB_PRESETS = {
    "TypeGo": "/home/gc635/Documents/TypeGo/install/typego/share/typego/resource/memory_db",
}


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

    def _parse_query(self) -> dict[str, str]:
        """Parse URL query parameters, returning first value for each key."""
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        return {k: v[0] for k, v in qs.items()}

    def do_GET(self):
        manager: MemoryManager = self.server.manager
        bus: _EventBus = self.server.event_bus
        path = self.path.split("?")[0]

        if path == "/":
            from typemem.viz.frontend import FRONTEND_HTML
            self._html_response(FRONTEND_HTML)

        elif path == "/api/stats":
            # Gather all items once for aggregated stats
            all_items = []
            tiers = {}
            total = 0
            for tier in MemoryTier:
                tier_items = manager.get_by_tier(tier)
                count = len(tier_items)
                if count:
                    tiers[tier.label] = count
                total += count
                all_items.extend(tier_items)

            consol_events = [e for e in bus.get_events() if e.event_type == "consolidation"]
            inject_events = [e for e in bus.get_events() if e.event_type == "injection"]

            # type_counts: memory_type.value -> count
            type_counter: Counter = Counter()
            source_counter: Counter = Counter()
            keyword_counter: Counter = Counter()
            timestamps: list[float] = []

            for item in all_items:
                type_counter[item.memory_type.value] += 1
                if item.source:
                    source_counter[item.source] += 1
                if item.keywords:
                    for kw in item.keywords.split(","):
                        kw = kw.strip().lower()
                        if kw:
                            keyword_counter[kw] += 1
                timestamps.append(item.timestamp)

            # keywords: top 20
            keywords = [{"keyword": kw, "count": c}
                        for kw, c in keyword_counter.most_common(20)]

            # timeline: 30 bins spanning all items
            timeline: list[dict] = []
            if timestamps:
                t_min = min(timestamps)
                t_max = max(timestamps)
                n_bins = 30
                if t_max == t_min:
                    # All items share the same timestamp — single bin
                    timeline = [{"t": t_min, "count": len(timestamps)}]
                else:
                    bin_width = (t_max - t_min) / n_bins
                    bins = [0] * n_bins
                    for ts in timestamps:
                        idx = int((ts - t_min) / bin_width)
                        if idx >= n_bins:
                            idx = n_bins - 1
                        bins[idx] += 1
                    timeline = [{"t": t_min + i * bin_width, "count": bins[i]}
                                for i in range(n_bins)]

            self._json_response({
                "total": total,
                "tiers": tiers,
                "events_count": len(bus.get_events()),
                "consolidations_count": len(consol_events),
                "injections_count": len(inject_events),
                "type_counts": dict(type_counter),
                "source_counts": dict(source_counter),
                "keywords": keywords,
                "timeline": timeline,
            })

        elif path == "/api/entries":
            params = self._parse_query()

            # Fetch all items from all tiers
            entries = []
            for tier in MemoryTier:
                for item in manager.get_by_tier(tier):
                    d = _entry_to_dict(item)
                    d["links"] = len(manager.links.get_links(item.id))
                    entries.append(d)

            # Filter by tier
            if "tier" in params:
                tier_filter = params["tier"]
                entries = [e for e in entries if e["tier"] == tier_filter]

            # Filter by memory_type
            if "type" in params:
                type_filter = params["type"]
                entries = [e for e in entries if e["memory_type"] == type_filter]

            # Filter by source
            if "source" in params:
                source_filter = params["source"]
                entries = [e for e in entries if e["source"] == source_filter]

            # Filter by since (unix timestamp)
            if "since" in params:
                try:
                    since_ts = float(params["since"])
                    entries = [e for e in entries if e["timestamp"] >= since_ts]
                except (ValueError, TypeError):
                    pass

            # Filter by search (substring in text)
            if "search" in params:
                search_str = params["search"].lower()
                entries = [e for e in entries if search_str in (e.get("text") or "").lower()]

            # Total after filtering, before pagination
            total = len(entries)

            # Sort
            sort_mode = params.get("sort", "newest")
            if sort_mode == "oldest":
                entries.sort(key=lambda e: e["timestamp"])
            elif sort_mode == "tier_asc":
                entries.sort(key=lambda e: _TIER_ORDER.get(e["tier"], 0))
            elif sort_mode == "tier_desc":
                entries.sort(key=lambda e: _TIER_ORDER.get(e["tier"], 0), reverse=True)
            elif sort_mode == "source_asc":
                entries.sort(key=lambda e: (e.get("source") or "").lower())
            elif sort_mode == "source_desc":
                entries.sort(key=lambda e: (e.get("source") or "").lower(), reverse=True)
            else:  # "newest" (default)
                entries.sort(key=lambda e: e["timestamp"], reverse=True)

            # Pagination
            try:
                offset = max(0, int(params.get("offset", 0)))
            except (ValueError, TypeError):
                offset = 0
            try:
                limit = max(1, int(params.get("limit", 100)))
            except (ValueError, TypeError):
                limit = 100

            entries = entries[offset:offset + limit]

            self._json_response({
                "items": entries,
                "total": total,
                "offset": offset,
                "limit": limit,
            })

        elif path == "/api/events":
            self._json_response([_serialize_event(e) for e in bus.get_events()])

        elif path == "/api/consolidations":
            events = [e for e in bus.get_events() if e.event_type == "consolidation"]
            self._json_response([_serialize_event(e) for e in events])

        elif path == "/api/injections":
            events = [e for e in bus.get_events() if e.event_type == "injection"]
            self._json_response([_serialize_event(e) for e in events])

        elif path.startswith("/api/links/"):
            item_id = path[len("/api/links/"):]
            if not item_id:
                self._json_response([], status=400)
            else:
                linked_ids = manager.links.get_links(item_id)
                linked_items = []
                for lid in linked_ids:
                    linked_item = manager.get(lid)
                    if linked_item is not None:
                        linked_items.append(_entry_to_dict(linked_item))
                self._json_response(linked_items)

        elif path == "/api/search":
            params = self._parse_query()
            q = params.get("q", "")
            if not q:
                self._json_response([])
            else:
                try:
                    n = max(1, int(params.get("n", 10)))
                except (ValueError, TypeError):
                    n = 10
                items, distances = manager.search_with_distances(q, n_results=n)
                results = []
                for item, dist in zip(items, distances):
                    d = _entry_to_dict(item)
                    d["distance"] = dist
                    results.append(d)
                self._json_response(results)

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

        elif path == "/api/databases":
            presets = getattr(self.server, "db_presets", {})
            current = getattr(self.server.manager, "persist_dir", "")
            self._json_response({
                "current": current,
                "presets": presets,
            })

        else:
            self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/api/database":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                self._json_response({"error": "invalid JSON"}, status=400)
                return

            db_path = data.get("path", "").strip()
            if not db_path:
                self._json_response({"error": "path is required"}, status=400)
                return

            import os
            if not os.path.isdir(db_path):
                self._json_response({"error": f"directory not found: {db_path}"}, status=400)
                return

            try:
                new_manager = MemoryManager(persist_dir=db_path, robot_id="viz")
                self.server.manager = new_manager
                self.server.event_bus = _EventBus()
                total = 0
                for tier in MemoryTier:
                    total += len(new_manager.get_by_tier(tier))
                self._json_response({"ok": True, "path": db_path, "total": total})
            except Exception as ex:
                self._json_response({"error": str(ex)}, status=500)
        else:
            self.send_error(404)


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class VizServer:
    """Threaded HTTP server for memory visualization."""

    def __init__(self, manager: MemoryManager, engine=None, injector=None,
                 port: int = 8811, db_presets: Optional[dict] = None):
        self._manager = manager
        self._engine = engine
        self._injector = injector
        self.event_bus = _EventBus()
        self._server = _ThreadedHTTPServer(("127.0.0.1", port), VizHandler)
        self._server.manager = manager
        self._server.event_bus = self.event_bus
        self._server.db_presets = db_presets if db_presets is not None else dict(_DEFAULT_DB_PRESETS)
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
