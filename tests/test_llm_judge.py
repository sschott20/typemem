"""Tests for the LLM judge evaluation metric."""

from benchmarks.llm_judge import llm_judge_score


def _mock_llm_score(score: int):
    def mock(prompt: str) -> str:
        return str(score)
    return mock


class TestLLMJudge:
    def test_returns_valid_score(self):
        result = llm_judge_score(
            llm=_mock_llm_score(4),
            query="Where is the cup?",
            context="[M1] Red cup on dining table",
            ground_truth=["cup", "dining table"],
        )
        assert result == 4.0

    def test_score_zero(self):
        result = llm_judge_score(
            llm=_mock_llm_score(0),
            query="Where is the cat?",
            context="[M1] Cup on counter",
            ground_truth=["cat", "counter"],
        )
        assert result == 0.0

    def test_score_five(self):
        result = llm_judge_score(
            llm=_mock_llm_score(5),
            query="What happened?",
            context="[M1] Person entered kitchen",
            ground_truth=["person", "entered"],
        )
        assert result == 5.0

    def test_handles_non_integer_response(self):
        def bad_llm(prompt: str) -> str:
            return "I think the score is about three"

        result = llm_judge_score(
            llm=bad_llm,
            query="Where is the cup?",
            context="[M1] Cup on table",
            ground_truth=["cup"],
        )
        assert result == 0.0

    def test_handles_llm_exception(self):
        def failing_llm(prompt: str) -> str:
            raise RuntimeError("API error")

        result = llm_judge_score(
            llm=failing_llm,
            query="Where is the cup?",
            context="[M1] Cup on table",
            ground_truth=["cup"],
        )
        assert result == 0.0

    def test_handles_out_of_range(self):
        result = llm_judge_score(
            llm=_mock_llm_score(9),
            query="test",
            context="test",
            ground_truth=["test"],
        )
        assert result == 5.0

    def test_handles_negative(self):
        def neg_llm(prompt: str) -> str:
            return "-1"

        result = llm_judge_score(
            llm=neg_llm,
            query="test",
            context="test",
            ground_truth=["test"],
        )
        assert result == 0.0

    def test_prompt_contains_query_and_context(self):
        prompts_seen = []

        def capturing_llm(prompt: str) -> str:
            prompts_seen.append(prompt)
            return "3"

        llm_judge_score(
            llm=capturing_llm,
            query="Where is the red cup?",
            context="[M1] Red cup on dining table near window",
            ground_truth=["red cup", "dining table"],
        )

        assert len(prompts_seen) == 1
        assert "Where is the red cup?" in prompts_seen[0]
        assert "Red cup on dining table near window" in prompts_seen[0]
        assert "red cup" in prompts_seen[0]
        assert "dining table" in prompts_seen[0]
