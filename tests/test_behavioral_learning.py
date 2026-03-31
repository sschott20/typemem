import pytest
from unittest.mock import MagicMock
from typemem.memory_manager import MemoryManager
from typemem.memory_item import MemoryItem, MemoryTier, MemoryType
from typemem.processed_index import ProcessedIndex
from typemem.plugins.consolidation.behavioral_learning import BehavioralLearningPlugin, _parse_behavioral_response


@pytest.fixture
def manager(tmp_path):
    return MemoryManager(persist_dir=str(tmp_path / "chroma"), robot_id="test")

@pytest.fixture
def processed(tmp_path):
    return ProcessedIndex(str(tmp_path / "processed.json"))


def _add_chat_archive(manager, doc):
    return manager.add(MemoryItem(
        document=doc, tier=MemoryTier.M2, memory_type=MemoryType.INSTRUCTION,
        robot_id="test", source="chat_history",
    ))


class TestParseBehavioralResponse:
    def test_parses_all_fields(self):
        response = "Rule: Go to hallway when person detected\nTrigger: person in room\nKeywords: person, hallway"
        rule, trigger, keywords = _parse_behavioral_response(response)
        assert rule == "Go to hallway when person detected"
        assert trigger == "person in room"
        assert keywords == "person, hallway"

    def test_missing_trigger(self):
        response = "Rule: Do something\nKeywords: test"
        rule, trigger, keywords = _parse_behavioral_response(response)
        assert rule == "Do something"
        assert trigger == ""
        assert keywords == "test"

    def test_empty_response(self):
        rule, trigger, keywords = _parse_behavioral_response("")
        assert rule == ""


class TestBehavioralLearningPlugin:
    def test_requires_two_conversations(self, manager, processed):
        _add_chat_archive(manager, "Conversation: [User] go to hallway [Robot] ok")
        mock_llm = MagicMock()
        plugin = BehavioralLearningPlugin()
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed)
        assert new_ids == []
        mock_llm.assert_not_called()

    def test_creates_lesson_from_two_conversations(self, manager, processed):
        _add_chat_archive(manager, "Morning conversation: The user asked the robot to navigate to the hallway because a visitor arrived at the front door.")
        _add_chat_archive(manager, "Evening conversation: The user instructed the robot to fetch a glass of water from the kitchen counter and bring it back.")

        mock_llm = MagicMock(return_value="Rule: Go to hallway when person detected\nTrigger: person in room\nKeywords: person,hallway")
        plugin = BehavioralLearningPlugin()
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed)

        assert len(new_ids) == 1
        lesson = manager.get(new_ids[0])
        assert lesson.tier == MemoryTier.M3
        assert lesson.memory_type == MemoryType.LESSON
        assert lesson.source == "behavioral_learning"
        assert "Go to hallway" in lesson.document
        assert "Trigger: person in room" in lesson.document

    def test_skips_without_llm(self, manager, processed):
        _add_chat_archive(manager, "conv1")
        _add_chat_archive(manager, "conv2")
        plugin = BehavioralLearningPlugin()
        assert plugin.run(manager, llm=None, processed_index=processed) == []

    def test_processed_index_prevents_reprocessing(self, manager, processed):
        _add_chat_archive(manager, "conv1")
        _add_chat_archive(manager, "conv2")

        mock_llm = MagicMock(return_value="Rule: test rule\nTrigger: test trigger\nKeywords: test")
        plugin = BehavioralLearningPlugin()
        plugin.run(manager, llm=mock_llm, processed_index=processed)
        mock_llm.reset_mock()

        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed)
        assert new_ids == []
        mock_llm.assert_not_called()

    def test_only_reads_chat_history_source(self, manager, processed):
        manager.add(MemoryItem(document="other m2", tier=MemoryTier.M2, memory_type=MemoryType.SUMMARY, robot_id="test", source="M1ToM2"))
        _add_chat_archive(manager, "conv1")
        _add_chat_archive(manager, "conv2")

        mock_llm = MagicMock(return_value="Rule: test\nTrigger: test\nKeywords: test")
        plugin = BehavioralLearningPlugin()
        plugin.run(manager, llm=mock_llm, processed_index=processed)

        prompt = mock_llm.call_args[0][0]
        assert "other m2" not in prompt

    def test_no_pattern_marks_done(self, manager, processed):
        _add_chat_archive(manager, "conv1")
        _add_chat_archive(manager, "conv2")

        mock_llm = MagicMock(return_value="NO_PATTERN")
        plugin = BehavioralLearningPlugin()
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed)
        assert new_ids == []

        # Should not reprocess
        mock_llm.reset_mock()
        _add_chat_archive(manager, "conv3")  # need fresh items to trigger
        # But conv1 and conv2 are already processed, only conv3 exists = 1 < 2
        new_ids = plugin.run(manager, llm=mock_llm, processed_index=processed)
        assert new_ids == []
