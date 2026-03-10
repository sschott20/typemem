import json
import pytest
from typemem.link_index import LinkIndex


class TestLinkIndex:
    def test_add_and_get_links(self, tmp_path):
        idx = LinkIndex(str(tmp_path / "links.json"))
        idx.add_link("a", "b")
        assert "b" in idx.get_links("a")
        assert "a" in idx.get_links("b")

    def test_self_link_ignored(self, tmp_path):
        idx = LinkIndex(str(tmp_path / "links.json"))
        idx.add_link("a", "a")
        assert idx.link_count("a") == 0

    def test_remove_link(self, tmp_path):
        idx = LinkIndex(str(tmp_path / "links.json"))
        idx.add_link("a", "b")
        idx.remove_link("a", "b")
        assert idx.link_count("a") == 0
        assert idx.link_count("b") == 0

    def test_remove_node(self, tmp_path):
        idx = LinkIndex(str(tmp_path / "links.json"))
        idx.add_link("a", "b")
        idx.add_link("a", "c")
        idx.remove_node("a")
        assert idx.link_count("a") == 0
        assert "a" not in idx.get_links("b")
        assert "a" not in idx.get_links("c")

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "links.json")
        idx = LinkIndex(path)
        idx.add_link("a", "b")
        idx.add_link("b", "c")
        idx.save()
        idx2 = LinkIndex(path)
        assert "b" in idx2.get_links("a")
        assert "c" in idx2.get_links("b")

    def test_get_links_empty(self, tmp_path):
        idx = LinkIndex(str(tmp_path / "links.json"))
        assert idx.get_links("nonexistent") == set()

    def test_link_count(self, tmp_path):
        idx = LinkIndex(str(tmp_path / "links.json"))
        idx.add_link("a", "b")
        idx.add_link("a", "c")
        idx.add_link("a", "d")
        assert idx.link_count("a") == 3
