"""Quick end-to-end test: run one benchmark scenario, judge results with Claude Haiku."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from benchmarks.synthetic import run_benchmark, _default_strategies, print_results


def judge_query(client, query: str, context: str, ground_truth: list[str]) -> dict:
    """Ask Claude Haiku to evaluate a single (query, context, ground_truth) triple."""
    prompt = f"""You are evaluating a robot memory system. A robot was asked a question, and its memory system injected context to help answer it.

Score the injected context on three dimensions (each 0.0 to 1.0):

1. **Relevance**: Does the context contain information that addresses the query?
2. **Completeness**: What fraction of the ground truth facts are conveyed (even via paraphrase)?
3. **Precision**: Is the context focused and free of irrelevant noise?

Query: {query}

Ground truth facts that should be present: {json.dumps(ground_truth)}

Injected context:
---
{context}
---

Respond in EXACTLY this JSON format, nothing else:
{{"relevance": 0.0, "completeness": 0.0, "precision": 0.0, "explanation": "one sentence"}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Handle markdown-wrapped JSON (```json ... ```)
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Pick two strategies to compare: full_context vs monolithic_rag
    strategies = _default_strategies()
    selected = [s for s in strategies if s.name in ("full_context", "monolithic_rag")]

    print("Running benchmark on kitchen_patrol...")
    results = []
    for s in selected:
        r = run_benchmark("benchmarks/scenarios/kitchen_patrol.yaml", s)
        results.append(r)

    print_results(results)

    # Judge each query result
    print("\n\n=== LLM-Judge Evaluation (Claude Haiku) ===\n")
    for r in results:
        print(f"--- {r.strategy_name} ---")
        for qr in r.query_results:
            scores = judge_query(client, qr.query, qr.context, qr.ground_truth)
            composite = 0.4 * scores["relevance"] + 0.4 * scores["completeness"] + 0.2 * scores["precision"]
            print(f"  query: {qr.query!r}")
            print(f"    relevance={scores['relevance']:.1f}  completeness={scores['completeness']:.1f}  precision={scores['precision']:.1f}  composite={composite:.2f}")
            print(f"    explanation: {scores['explanation']}")
        print()

    print("Done! Total judge calls:", sum(len(r.query_results) for r in results))


if __name__ == "__main__":
    main()
