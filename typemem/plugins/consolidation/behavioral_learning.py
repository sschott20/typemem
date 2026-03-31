"""Consolidation plugin: extract behavioral rules from repeated conversations (M2 -> M3)."""

import logging
import re
from typing import List, Optional, Tuple

from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.memory_manager import MemoryManager
from typemem.processed_index import ProcessedIndex
from typemem.plugins.base import ConsolidationPlugin

logger = logging.getLogger(__name__)

_MIN_CONVERSATIONS = 2


def _parse_behavioral_response(response: str) -> Tuple[str, str, str]:
    """Parse Rule/Trigger/Keywords from LLM response. Returns (rule, trigger, keywords)."""
    rule = ""
    trigger = ""
    keywords = ""
    for line in response.strip().splitlines():
        line = line.strip()
        m = re.match(r'^Rule\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            rule = m.group(1).strip()
            continue
        m = re.match(r'^Trigger\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            trigger = m.group(1).strip()
            continue
        m = re.match(r'^Keywords?\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            keywords = m.group(1).strip().rstrip(".")
            continue
    return rule, trigger, keywords


class BehavioralLearningPlugin(ConsolidationPlugin):
    """Detect repeated interaction patterns and extract behavioral rules as M3 LESSON items."""

    @property
    def name(self) -> str:
        return "behavioral_learning"

    @property
    def source_tier(self) -> MemoryTier:
        return MemoryTier.M2

    @property
    def target_tier(self) -> MemoryTier:
        return MemoryTier.M3

    @property
    def interval_seconds(self) -> float:
        return 60.0

    def run(self, manager: MemoryManager, llm=None, processed_index: Optional[ProcessedIndex] = None) -> List[str]:
        if llm is None:
            return []

        unprocessed = self.get_unprocessed(manager, processed_index) if processed_index else manager.get_by_tier(self.source_tier)

        # Filter to only chat history archives
        conversations = [item for item in unprocessed if item.source == "chat_history"]

        if len(conversations) < _MIN_CONVERSATIONS:
            return []

        conversations_text = "\n\n".join(
            f"--- Conversation {i+1} ---\n{conv.document}"
            for i, conv in enumerate(conversations)
        )

        prompt = (
            "Here are recent completed conversations between the user and robot:\n\n"
            f"{conversations_text}\n\n"
            "Are there any repeated patterns where the user consistently gives "
            "the same instruction in response to a similar situation? If so, "
            "extract a behavioral rule the robot should follow autonomously.\n\n"
            "If no clear pattern exists, respond with exactly: NO_PATTERN\n\n"
            "Otherwise respond with:\n"
            "Rule: <one sentence behavioral rule>\n"
            "Trigger: <what triggers this behavior>\n"
            "Keywords: <comma-separated>"
        )

        try:
            response = llm(prompt)
        except Exception as e:
            logger.error("BehavioralLearning LLM call failed: %s", e)
            return []

        if "NO_PATTERN" in response:
            if processed_index:
                self.mark_done(processed_index, [c.id for c in conversations])
            return []

        rule, trigger, keywords = _parse_behavioral_response(response)
        if not rule:
            if processed_index:
                self.mark_done(processed_index, [c.id for c in conversations])
            return []

        # Check redundancy against existing lessons
        existing = manager.get_by_source("behavioral_learning", tier=MemoryTier.M3)
        if existing:
            existing_rules = "\n".join(f"- {item.document}" for item in existing)
            check_prompt = (
                f"Existing behavioral rules:\n{existing_rules}\n\n"
                f"Proposed new rule: {rule}\n\n"
                "Is this new rule redundant with any existing rule? "
                "Answer YES or NO only."
            )
            try:
                check = llm(check_prompt).strip().upper()
                if "YES" in check:
                    logger.info("BehavioralLearning: skipping redundant rule: %s", rule[:60])
                    if processed_index:
                        self.mark_done(processed_index, [c.id for c in conversations])
                    return []
            except Exception:
                pass  # proceed with adding if check fails

        doc = f"{rule} (Trigger: {trigger})" if trigger else rule
        lesson = MemoryItem(
            document=doc,
            tier=MemoryTier.M3,
            memory_type=MemoryType.LESSON,
            robot_id=conversations[0].robot_id,
            source="behavioral_learning",
            keywords=keywords or "behavior,learned",
        )
        new_id = manager.add(lesson)

        if processed_index:
            self.mark_done(processed_index, [c.id for c in conversations])

        logger.info("BehavioralLearning: extracted rule from %d conversations: %s", len(conversations), rule[:80])
        return [new_id]
