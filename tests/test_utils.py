import pytest
from typemem.utils import parse_summary_keywords


class TestParseSummaryKeywords:
    def test_standard_format(self):
        response = "Summary: A chair was seen near the kitchen.\nKeywords: chair, kitchen, observation"
        content, keywords = parse_summary_keywords(response)
        assert content == "A chair was seen near the kitchen."
        assert "chair" in keywords
        assert "kitchen" in keywords

    def test_no_keywords_line(self):
        response = "Just a plain summary without keywords."
        content, keywords = parse_summary_keywords(response)
        assert content == "Just a plain summary without keywords."
        assert keywords == ""

    def test_lesson_prefix(self):
        response = "Lesson: People gather at lunch.\nKeywords: people, lunch"
        content, keywords = parse_summary_keywords(response)
        assert content == "People gather at lunch."

    def test_deduplicates_keywords(self):
        response = "Summary: test\nKeywords: a, b, a, c"
        _, keywords = parse_summary_keywords(response)
        assert keywords == "a,b,c"
