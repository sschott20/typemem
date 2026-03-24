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
