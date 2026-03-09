"""Tests for viz HTTP server."""
import json
import time
import urllib.request

import pytest

from typemem.chromadb_store import ChromaDBStore
from typemem.baselines import make_tiered_memory
from typemem.viz.tracing import TracingStore, TracingSystem
from typemem.viz.server import VizServer


@pytest.fixture
def setup(tmp_path):
    store = ChromaDBStore(persist_dir=str(tmp_path / "chroma"))
    tracing = TracingStore(store)
    system = make_tiered_memory(tracing)
    ts = TracingSystem(system, tracing)
    return tracing, ts


@pytest.fixture
def server(setup):
    tracing, ts = setup
    srv = VizServer(tracing, ts, port=0)  # port=0 picks a free port
    srv.start()
    yield srv
    srv.stop()


def _get(server, path):
    url = f"http://localhost:{server.port}{path}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode())


def test_server_starts_and_serves_index(server):
    url = f"http://localhost:{server.port}/"
    with urllib.request.urlopen(url, timeout=5) as resp:
        html = resp.read().decode()
    assert "typemem" in html.lower()
    assert resp.status == 200


def test_api_stats(server, setup):
    tracing, ts = setup
    tracing.add("test entry", metadata={"_tier": "raw"})
    data = _get(server, "/api/stats")
    assert data["total"] >= 1


def test_api_entries(server, setup):
    tracing, ts = setup
    tracing.add("hello world")
    data = _get(server, "/api/entries")
    assert len(data) >= 1
    assert data[0]["text"] == "hello world"


def test_api_events(server, setup):
    tracing, ts = setup
    tracing.add("event test")
    data = _get(server, "/api/events")
    assert len(data) >= 1
    assert data[0]["operation"] == "add"


def test_api_consolidations(server, setup):
    tracing, ts = setup
    for i in range(5):
        ts.observe({"text": f"obs {i}"})
    ts.consolidate()
    data = _get(server, "/api/consolidations")
    assert len(data) >= 1


def test_api_injections(server, setup):
    tracing, ts = setup
    ts.observe({"text": "cup on table"})
    ts.inject("tiered", "where is cup", 500)
    data = _get(server, "/api/injections")
    assert len(data) >= 1
    assert data[0]["query"] == "where is cup"
