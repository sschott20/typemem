"""LLM callable interface — decouples plugins from specific providers."""

from typing import Callable

LLMCallable = Callable[[str], str]


def make_anthropic_llm(model: str = "claude-haiku-4-5-20251001") -> LLMCallable:
    """Create an LLM callable using the Anthropic SDK.

    Requires ANTHROPIC_API_KEY environment variable.
    """
    import anthropic

    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    return call
