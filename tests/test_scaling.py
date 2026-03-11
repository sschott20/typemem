"""Tests for the quality-vs-store-size scaling benchmark."""

from benchmarks.scaling import (
    ScalingResult,
    run_replay_fraction,
    run_noise_padding,
    DISTRACTOR_POOL,
)
from benchmarks.synthetic import _default_strategies


SCENARIO = "benchmarks/scenarios/kitchen_patrol.yaml"


class TestReplayFraction:
    def test_returns_results_for_each_fraction(self):
        strategy = _default_strategies()[1]  # monolithic_rag
        fractions = [0.5, 1.0]
        results = run_replay_fraction(SCENARIO, strategy, fractions=fractions)
        assert len(results) == 2
        assert results[0].mode == "replay_fraction"
        assert results[0].scale_param == 0.5
        assert results[1].scale_param == 1.0

    def test_more_events_means_more_memories(self):
        strategy = _default_strategies()[1]
        results = run_replay_fraction(SCENARIO, strategy, fractions=[0.25, 1.0])
        assert results[0].store_size <= results[1].store_size

    def test_result_has_precision(self):
        strategy = _default_strategies()[1]
        results = run_replay_fraction(SCENARIO, strategy, fractions=[1.0])
        assert results[0].avg_precision >= 0.0
        assert results[0].avg_llm_judge_score is None


class TestNoisePadding:
    def test_returns_results_for_each_count(self):
        strategy = _default_strategies()[1]
        counts = [0, 10]
        results = run_noise_padding(SCENARIO, strategy, distractor_counts=counts)
        assert len(results) == 2
        assert results[0].mode == "noise_padding"
        assert results[0].scale_param == 0
        assert results[1].scale_param == 10

    def test_more_distractors_means_more_memories(self):
        strategy = _default_strategies()[1]
        results = run_noise_padding(SCENARIO, strategy, distractor_counts=[0, 50])
        assert results[1].store_size > results[0].store_size

    def test_distractor_pool_exists(self):
        assert len(DISTRACTOR_POOL) >= 20


class TestScalingResult:
    def test_fields(self):
        r = ScalingResult(
            strategy_name="test", scenario_name="test",
            mode="replay_fraction", scale_param=0.5,
            store_size=10, avg_precision=0.8,
            avg_llm_judge_score=None, avg_injection_latency_ms=5.0,
        )
        assert r.mode == "replay_fraction"
        assert r.avg_llm_judge_score is None
