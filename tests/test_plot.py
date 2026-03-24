"""Tests for benchmark plotting."""

import json
from benchmarks.latency import LatencyResult, latency_results_to_json


class TestLatencyJson:
    def test_latency_results_to_json_structure(self):
        r = LatencyResult(strategy_name="test", store_size=100)
        r.observation_latencies_ms = [10.0, 20.0, 30.0]
        r.injection_latencies_ms = [5.0, 15.0, 25.0]
        data = latency_results_to_json([r])
        assert len(data) == 1
        assert data[0]["strategy"] == "test"
        assert data[0]["store_size"] == 100
        assert data[0]["obs_p50"] == r.obs_p50
        assert data[0]["obs_p99"] == r.obs_p99
        assert data[0]["inj_p50"] == r.inj_p50
        assert data[0]["inj_p99"] == r.inj_p99
        # Verify JSON-serializable
        json.dumps(data)
