"""LLM-based judge for evaluating memory retrieval quality."""

import logging
import re

from typemem.llm import LLMCallable

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """Rate how well this retrieved memory context answers the robot's query.

Query: {query}
Ground truth facts that should be present: {ground_truth}

Retrieved context:
{context}

Score 0-5:
- 0: No relevant information
- 1: Mentions topic but missing key facts
- 2: Some relevant facts, significant gaps
- 3: Most key facts present, minor gaps
- 4: All key facts present, minimal noise
- 5: Perfect — all facts present, concise, no irrelevant content

Reply with ONLY a single integer 0-5."""


def llm_judge_score(
    llm: LLMCallable,
    query: str,
    context: str,
    ground_truth: list[str],
) -> float:
    """Score how well retrieved context answers a query. Returns 0.0-5.0."""
    prompt = _JUDGE_PROMPT.format(
        query=query,
        ground_truth=", ".join(ground_truth),
        context=context,
    )

    try:
        response = llm(prompt)
    except Exception as e:
        logger.error("LLM judge call failed: %s", e)
        return 0.0

    match = re.search(r"-?\d+", response.strip())
    if match is None:
        return 0.0

    score = int(match.group())
    return float(max(0, min(5, score)))
