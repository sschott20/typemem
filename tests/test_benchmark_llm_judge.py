"""Tests for LLM judge integration in the synthetic benchmark."""

from benchmarks.synthetic import run_benchmark, _default_strategies, results_to_json


def _mock_judge_llm(prompt: str) -> str:
    return "3"


class TestBenchmarkLLMJudge:
    def test_run_benchmark_without_llm_has_none_scores(self):
        strategies = _default_strategies()
        result = run_benchmark(
            "benchmarks/scenarios/kitchen_patrol.yaml",
            strategies[1],
        )
        for qr in result.query_results:
            assert qr.llm_judge_score is None
        assert result.avg_llm_judge_score is None

    def test_run_benchmark_with_llm_has_scores(self):
        strategies = _default_strategies()
        result = run_benchmark(
            "benchmarks/scenarios/kitchen_patrol.yaml",
            strategies[1],
            llm=_mock_judge_llm,
        )
        for qr in result.query_results:
            assert qr.llm_judge_score == 3.0
        assert result.avg_llm_judge_score == 3.0

    def test_results_to_json_includes_judge_score(self):
        strategies = _default_strategies()
        result = run_benchmark(
            "benchmarks/scenarios/kitchen_patrol.yaml",
            strategies[1],
            llm=_mock_judge_llm,
        )
        json_out = results_to_json([result])
        assert "avg_llm_judge_score" in json_out[0]
        assert json_out[0]["avg_llm_judge_score"] == 3.0
        for q in json_out[0]["queries"]:
            assert "llm_judge_score" in q
