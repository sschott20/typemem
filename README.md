# typemem

An abstract memory framework for LLM-controlled robots.

typemem decomposes robot memory into three function types — **observation**, **consolidation**, and **injection** — operating over a generic vector store. This decomposition enables controlled comparison of memory strategies, from naive baselines to cognitively-inspired multi-tier architectures.

## Research Question

> Does a cognitively-inspired memory hierarchy (observation → consolidation → injection) outperform simpler alternatives for LLM-controlled robots in task performance, retrieval relevance, and latency?

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Memory Store (swappable)                │
│  add / search / delete / update / get_all / count   │
└──────────┬──────────────────┬──────────────┬────────┘
           │                  │              │
    Observation(s)     Consolidation(s)   Injection(s)
    sensor → store     store → store      store → prompt
    [background]       [background]       [on-demand]
```

- **Observation functions** run in the background, converting raw sensor data into text memories.
- **Consolidation functions** run in the background, reading existing memories to create summaries, extract patterns, or prune redundancy — optionally using LLM calls.
- **Injection functions** are called synchronously before LLM inference, searching memory and returning relevant context within a token budget.

All three baselines (full-context, monolithic RAG, tiered memory) are expressed as implementations of the same interface.

## LLM-Generated Functions

Given a natural language personality description, typemem generates starter observation, consolidation, and injection functions (Python code + LLM prompts) that users can review and refine.

## Origin

Abstracted from the memory system of [TypeGo](https://github.com/sschott20/TypeGo), an LLM-controlled robotics platform using a 4-tier ChromaDB-backed memory hierarchy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/ -v
```

## Benchmarks

```bash
# Synthetic scenario benchmarks
python -m benchmarks.synthetic

# Latency benchmarks across store sizes
python -m benchmarks.latency
```

## License

MIT
