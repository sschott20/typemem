"""Tests for the LLM callable interface."""

from typemem.llm import make_anthropic_llm, LLMCallable


class TestLLMCallable:
    def test_make_anthropic_llm_returns_callable(self):
        """Verify factory returns a callable (don't actually call the API)."""
        fn = make_anthropic_llm()
        assert callable(fn)

    def test_mock_llm_matches_protocol(self):
        """A simple lambda satisfies LLMCallable."""
        mock_llm: LLMCallable = lambda prompt: "mock response"
        assert mock_llm("hello") == "mock response"

    def test_make_anthropic_llm_custom_model(self):
        fn = make_anthropic_llm(model="claude-haiku-4-5-20251001")
        assert callable(fn)
