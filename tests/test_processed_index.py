import pytest
from typemem.processed_index import ProcessedIndex


class TestProcessedIndex:
    def test_mark_and_check(self, tmp_path):
        idx = ProcessedIndex(str(tmp_path / "processed.json"))
        idx.mark_processed("plugin_a", ["id1", "id2"])
        assert idx.is_processed("plugin_a", "id1")
        assert idx.is_processed("plugin_a", "id2")
        assert not idx.is_processed("plugin_a", "id3")

    def test_filter_unprocessed(self, tmp_path):
        idx = ProcessedIndex(str(tmp_path / "processed.json"))
        idx.mark_processed("plugin_a", ["id1", "id2"])
        result = idx.filter_unprocessed("plugin_a", ["id1", "id2", "id3", "id4"])
        assert result == ["id3", "id4"]

    def test_cross_plugin_isolation(self, tmp_path):
        idx = ProcessedIndex(str(tmp_path / "processed.json"))
        idx.mark_processed("plugin_a", ["id1"])
        assert not idx.is_processed("plugin_b", "id1")

    def test_prune(self, tmp_path):
        idx = ProcessedIndex(str(tmp_path / "processed.json"))
        idx.mark_processed("plugin_a", ["id1", "id2", "id3"])
        idx.prune(live_ids={"id1", "id3"})
        assert idx.is_processed("plugin_a", "id1")
        assert not idx.is_processed("plugin_a", "id2")
        assert idx.is_processed("plugin_a", "id3")

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "processed.json")
        idx = ProcessedIndex(path)
        idx.mark_processed("p", ["a", "b"])
        idx.save()
        idx2 = ProcessedIndex(path)
        assert idx2.is_processed("p", "a")
        assert idx2.is_processed("p", "b")

    def test_empty_plugin(self, tmp_path):
        idx = ProcessedIndex(str(tmp_path / "processed.json"))
        assert idx.filter_unprocessed("nonexistent", ["id1"]) == ["id1"]
