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


from benchmarks.plot_utils import apply_style, STRATEGY_COLORS, STRATEGY_LINESTYLES


class TestPlotUtils:
    def test_strategy_colors_has_all_strategies(self):
        expected = {"full_context", "monolithic_rag", "tiered_memory",
                    "rag_with_recency", "tiered_no_consol"}
        assert expected.issubset(set(STRATEGY_COLORS.keys()))

    def test_strategy_linestyles_has_all_strategies(self):
        expected = {"full_context", "monolithic_rag", "tiered_memory",
                    "rag_with_recency", "tiered_no_consol"}
        assert expected.issubset(set(STRATEGY_LINESTYLES.keys()))

    def test_apply_style_returns_fig_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        apply_style()
        assert fig is not None
        assert ax is not None
        plt.close(fig)


import os
import tempfile

from benchmarks.plot import plot_strategy_comparison


class TestStrategyComparison:
    def _make_synthetic_json(self):
        return [
            {
                "strategy": "full_context",
                "scenario": "kitchen_patrol",
                "total_memories": 7,
                "avg_precision": 0.67,
                "avg_injection_latency_ms": 1.5,
                "avg_token_count": 200.0,
                "avg_llm_judge_score": None,
                "queries": [],
            },
            {
                "strategy": "monolithic_rag",
                "scenario": "kitchen_patrol",
                "total_memories": 7,
                "avg_precision": 0.50,
                "avg_injection_latency_ms": 140.0,
                "avg_token_count": 100.0,
                "avg_llm_judge_score": None,
                "queries": [],
            },
        ]

    def test_creates_files(self):
        import matplotlib
        matplotlib.use("Agg")
        data = self._make_synthetic_json()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_strategy_comparison(data, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "strategy_comparison.pdf"))
            assert os.path.exists(os.path.join(tmpdir, "strategy_comparison.png"))


from benchmarks.plot import plot_quality_vs_size


class TestQualityVsSize:
    def _make_scaling_json(self):
        results = []
        for strat in ["full_context", "tiered_memory"]:
            for count in [0, 50, 200]:
                results.append({
                    "strategy": strat,
                    "scenario": "kitchen_patrol",
                    "mode": "noise_padding",
                    "scale_param": count,
                    "store_size": 7 + count,
                    "avg_precision": max(0.0, 0.8 - count * 0.001),
                    "avg_llm_judge_score": None,
                    "avg_injection_latency_ms": 100.0 + count * 0.5,
                })
        return results

    def test_creates_files(self):
        import matplotlib
        matplotlib.use("Agg")
        data = self._make_scaling_json()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_quality_vs_size(data, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "quality_vs_size.pdf"))
            assert os.path.exists(os.path.join(tmpdir, "quality_vs_size.png"))


from benchmarks.plot import plot_latency_vs_size


class TestLatencyVsSize:
    def _make_latency_json(self):
        results = []
        for strat in ["full_context", "monolithic_rag"]:
            for size in [100, 500, 1000]:
                results.append({
                    "strategy": strat,
                    "store_size": size,
                    "obs_p50": 400.0,
                    "obs_p99": 800.0,
                    "inj_p50": 2.0 if strat == "full_context" else 140.0,
                    "inj_p99": 5.0 if strat == "full_context" else 200.0,
                    "observation_latencies_ms": [400.0] * 10,
                    "injection_latencies_ms": [140.0] * 10,
                })
        return results

    def test_creates_files(self):
        import matplotlib
        matplotlib.use("Agg")
        data = self._make_latency_json()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_latency_vs_size(data, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "latency_vs_size.pdf"))
            assert os.path.exists(os.path.join(tmpdir, "latency_vs_size.png"))


from benchmarks.plot import plot_latency_distribution


class TestLatencyDistribution:
    def _make_latency_json(self):
        return [
            {
                "strategy": "full_context",
                "store_size": 1000,
                "obs_p50": 400.0, "obs_p99": 800.0,
                "inj_p50": 2.0, "inj_p99": 5.0,
                "observation_latencies_ms": [400.0] * 10,
                "injection_latencies_ms": [1.5, 2.0, 2.5, 1.8, 3.0, 2.2, 1.9, 2.1, 2.8, 2.0],
            },
            {
                "strategy": "monolithic_rag",
                "store_size": 1000,
                "obs_p50": 450.0, "obs_p99": 850.0,
                "inj_p50": 140.0, "inj_p99": 200.0,
                "observation_latencies_ms": [450.0] * 10,
                "injection_latencies_ms": [130.0, 140.0, 150.0, 145.0, 160.0, 135.0, 142.0, 155.0, 138.0, 148.0],
            },
        ]

    def test_creates_files(self):
        import matplotlib
        matplotlib.use("Agg")
        data = self._make_latency_json()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_latency_distribution(data, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "latency_distribution.pdf"))
            assert os.path.exists(os.path.join(tmpdir, "latency_distribution.png"))
