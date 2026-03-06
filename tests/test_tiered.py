"""Tests for LLM-powered tiered memory strategy. All LLM calls are mocked."""
import time

import pytest
from unittest.mock import patch

from typemem.tiered import make_tiered_llm, _parse_response, _group_similar


MOCK_SUMMARY_RESPONSE = (
    "Summary: Routine patrol of Lobby and East Wing shows all areas secure "
    "with normal temperatures and no activity.\n"
    "Keywords: patrol,routine,secure,lobby,east_wing"
)

MOCK_LESSON_RESPONSE = (
    "Lesson: Server Room cooling system requires monitoring as temperature "
    "has shown a gradual upward trend over the shift.\n"
    "Keywords: server_room,cooling,temperature,monitoring"
)


class TestParseResponse:
    def test_parse_summary(self):
        content, keywords = _parse_response(MOCK_SUMMARY_RESPONSE, "Summary")
        assert "patrol" in content.lower()
        assert "patrol" in keywords

    def test_parse_lesson(self):
        content, keywords = _parse_response(MOCK_LESSON_RESPONSE, "Lesson")
        assert "cooling" in content.lower()
        assert "server_room" in keywords

    def test_parse_no_label_returns_whole_text(self):
        content, keywords = _parse_response("Just plain text", "Summary")
        assert content == "Just plain text"
        assert keywords == ""


class TestGroupSimilar:
    def test_groups_similar_entries(self, store):
        # Add several similar observations
        ids = []
        for text in [
            "Lobby: Front entrance secured, all clear",
            "Lobby: Front entrance locked, quiet night",
            "Lobby: Reception area clear, entrance secure",
            "Server Room: Temperature 72F, all systems normal",
            "Server Room: Temperature 73F, cooling active",
        ]:
            ids.append(store.add(text, metadata={"_tier": "raw"}))

        from typemem.types import MemoryEntry
        entries = store.get_all(filters={"_tier": "raw"})
        groups = _group_similar(store, entries, distance_threshold=0.5)
        # Should have at least 2 groups (lobby vs server room)
        assert len(groups) >= 2

    def test_empty_entries_returns_empty(self, store):
        groups = _group_similar(store, [], distance_threshold=0.5)
        assert groups == []


class TestTieredObservation:
    def test_observe_tags_raw_tier(self, store):
        system = make_tiered_llm(store)
        ids = system.observe({"text": "Lobby: Front entrance secured"})
        assert len(ids) == 1
        entry = store.get(ids[0])
        assert entry.metadata["_tier"] == "raw"

    def test_observe_empty_text_returns_empty(self, store):
        system = make_tiered_llm(store)
        ids = system.observe({"text": ""})
        assert ids == []

    def test_observe_no_text_key_returns_empty(self, store):
        system = make_tiered_llm(store)
        ids = system.observe({"sensor": "data"})
        assert ids == []


class TestTieredConsolidation:
    @patch("typemem.tiered._llm_call")
    def test_m1_to_m2_creates_summary(self, mock_llm, store):
        mock_llm.return_value = MOCK_SUMMARY_RESPONSE
        system = make_tiered_llm(store, min_group_size=3)

        # Add enough similar observations to trigger grouping
        for i in range(5):
            store.add(
                f"Lobby: Entrance secure, patrol check #{i}",
                metadata={"_tier": "raw"},
            )

        created = system.consolidate()
        assert len(created) >= 1

        # Verify a summary was created
        summaries = store.get_all(filters={"_tier": "summary"})
        assert len(summaries) >= 1
        assert "patrol" in summaries[0].text.lower()
        assert mock_llm.called

    @patch("typemem.tiered._llm_call")
    def test_m2_to_m3_creates_lesson(self, mock_llm, store):
        mock_llm.return_value = MOCK_LESSON_RESPONSE
        system = make_tiered_llm(store, min_group_size=3)

        # Add summaries directly (bypassing M1->M2)
        for i in range(4):
            store.add(
                f"Summary of patrol round {i}: all areas secure",
                metadata={"_tier": "summary"},
            )

        created = system.consolidate()
        knowledge = store.get_all(filters={"_tier": "knowledge"})
        assert len(knowledge) >= 1
        assert "cooling" in knowledge[0].text.lower()

    @patch("typemem.tiered._llm_call")
    def test_processed_tracking_prevents_duplicates(self, mock_llm, store):
        mock_llm.return_value = MOCK_SUMMARY_RESPONSE
        system = make_tiered_llm(store, min_group_size=3)

        for i in range(5):
            store.add(f"Lobby: patrol check #{i}", metadata={"_tier": "raw"})

        # First consolidation creates summaries
        first = system.consolidate()
        summary_count_1 = store.count(filters={"_tier": "summary"})

        # Second consolidation should NOT create duplicates
        second = system.consolidate()
        summary_count_2 = store.count(filters={"_tier": "summary"})
        assert summary_count_2 == summary_count_1

    @patch("typemem.tiered._llm_call")
    def test_too_few_items_skips_consolidation(self, mock_llm, store):
        system = make_tiered_llm(store, min_group_size=3)

        # Only 2 items -- below min_group_size
        store.add("Lobby: check 1", metadata={"_tier": "raw"})
        store.add("Lobby: check 2", metadata={"_tier": "raw"})

        created = system.consolidate()
        summaries = store.get_all(filters={"_tier": "summary"})
        assert len(summaries) == 0
        assert not mock_llm.called

    def test_prune_deletes_old_processed_raw(self, store):
        system = make_tiered_llm(store, retention_secs=0.0, min_group_size=3)

        # Add old raw entries
        now = time.time()
        for i in range(4):
            store.add(
                f"Old observation {i}",
                metadata={"_tier": "raw", "_timestamp": now - 1000},
            )

        # Mock LLM for the summarization step
        with patch("typemem.tiered._llm_call", return_value=MOCK_SUMMARY_RESPONSE):
            system.consolidate()

        # Old raw entries should be pruned (retention_secs=0)
        remaining_raw = store.get_all(filters={"_tier": "raw"})
        assert len(remaining_raw) == 0

        # Summary should still exist
        summaries = store.get_all(filters={"_tier": "summary"})
        assert len(summaries) >= 1


class TestTieredInjection:
    def test_injection_returns_context(self, store):
        system = make_tiered_llm(store)

        store.add("Lobby entrance is secure and locked", metadata={"_tier": "raw"})
        store.add("All zones clear, building secure", metadata={"_tier": "summary"})
        store.add(
            "Building security is consistently maintained across all shifts",
            metadata={"_tier": "knowledge"},
        )

        context = system.inject("tiered", "is the building secure?", token_budget=500)
        assert len(context) > 0
        assert "secure" in context.lower()

    def test_injection_respects_token_budget(self, store):
        system = make_tiered_llm(store)

        for i in range(30):
            store.add(
                f"Patrol observation number {i} with some padding text here for tokens",
                metadata={"_tier": "raw"},
            )

        context = system.inject("tiered", "what happened?", token_budget=30)
        lines = [l for l in context.strip().split("\n") if l.strip()]
        assert len(lines) < 30


class TestTieredEndToEnd:
    @patch("typemem.tiered._llm_call")
    def test_observe_consolidate_inject_cycle(self, mock_llm, store):
        """Full cycle: observe raw data -> consolidate with LLM -> inject returns consolidated content."""
        mock_llm.side_effect = [MOCK_SUMMARY_RESPONSE, MOCK_LESSON_RESPONSE]
        system = make_tiered_llm(store, min_group_size=3)

        # Observe several events
        for text in [
            "Lobby: entrance secure",
            "Lobby: reception clear",
            "Lobby: front door locked",
            "Lobby: no visitors",
            "Lobby: cameras operational",
        ]:
            system.observe({"text": text})

        # Consolidate (M1->M2)
        system.consolidate()
        summaries = store.get_all(filters={"_tier": "summary"})
        assert len(summaries) >= 1

        # Add more summaries to trigger M2->M3
        for i in range(3):
            store.add(f"Summary {i}: patrol clear", metadata={"_tier": "summary"})
        system.consolidate()
        knowledge = store.get_all(filters={"_tier": "knowledge"})
        assert len(knowledge) >= 1

        # Inject should return content
        context = system.inject("tiered", "is the lobby secure?", token_budget=500)
        assert len(context) > 0
