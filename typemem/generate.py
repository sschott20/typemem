from __future__ import annotations

import yaml
from typing import Any

_OPENAI_CLIENT = None

def _get_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


def _llm_call(prompt: str, model: str = "gpt-4o") -> str:
    """Make an LLM call. Separated for easy mocking in tests."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


_CONFIG_PROMPT = """You are designing a memory system for an LLM-controlled robot.

Robot personality/use-case: {personality}

Generate a YAML config defining observation, consolidation, and injection functions.
Each function has a name, interval (seconds, for observation/consolidation only), and description.

Rules:
- Observation functions convert raw sensor data into text memories
- Consolidation functions read existing memories, create summaries/patterns/refined knowledge
- Injection functions search memory and return relevant context for LLM prompts
- Be specific to the robot's use case
- 2-4 observation functions, 1-3 consolidation functions, 1-2 injection functions

Output ONLY valid YAML, no markdown fences, with this structure:
observation:
  - name: ...
    interval: ...
    description: "..."
consolidation:
  - name: ...
    interval: ...
    description: "..."
injection:
  - name: ...
    description: "..."
"""

_FN_PROMPT = """Write a Python function for a robot memory system.

Robot personality: {personality}
Function type: {fn_type}
Function name: {name}
Description: {description}

The function must be named `fn` with this signature:
{signature}

Available store methods:
- store.add(text, metadata=None, id=None) -> str  (returns ID)
- store.search(query, n=10, filters=None) -> list[SearchResult]  (SearchResult has .entry.text, .entry.metadata, .distance)
- store.delete(id) -> None
- store.update(id, text=None, metadata=None) -> None
- store.get_all(filters=None) -> list[MemoryEntry]  (MemoryEntry has .id, .text, .metadata)
- store.count(filters=None) -> int

Output ONLY the Python function definition, no markdown fences, no imports.
"""

_SIGNATURES = {
    "observation": "def fn(raw_data: dict, store) -> list[str]:",
    "consolidation": "def fn(store) -> list[str]:",
    "injection": "def fn(query: str, store, token_budget: int) -> str:",
}


def generate_config(personality: str, model: str = "gpt-4o") -> dict:
    """Generate a typemem YAML config from a personality description."""
    prompt = _CONFIG_PROMPT.format(personality=personality)
    response = _llm_call(prompt, model=model)
    response = response.strip()
    if response.startswith("```"):
        response = "\n".join(response.split("\n")[1:])
    if response.endswith("```"):
        response = "\n".join(response.split("\n")[:-1])
    config = yaml.safe_load(response)
    config["personality"] = personality
    return config


def generate_functions(fn_type: str, fn_def: dict, personality: str, model: str = "gpt-4o") -> dict:
    """Generate Python code for a single function definition."""
    prompt = _FN_PROMPT.format(
        personality=personality,
        fn_type=fn_type,
        name=fn_def["name"],
        description=fn_def["description"],
        signature=_SIGNATURES[fn_type],
    )
    response = _llm_call(prompt, model=model)
    code = response.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])
    result = dict(fn_def)
    result["code"] = code
    return result


def generate_full(personality: str, model: str = "gpt-4o") -> dict:
    """Generate a complete config with code for all functions."""
    config = generate_config(personality, model=model)
    for fn_type in ("observation", "consolidation", "injection"):
        for i, fn_def in enumerate(config.get(fn_type, [])):
            config[fn_type][i] = generate_functions(fn_type, fn_def, personality, model=model)
    return config
