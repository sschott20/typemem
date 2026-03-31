"""Shared utilities for memory processing."""

import re
from typing import Tuple


def parse_summary_keywords(response: str) -> Tuple[str, str]:
    """Parse LLM response containing 'Summary/Lesson:' and 'Keywords:' lines.

    Returns (content, comma_separated_keywords). Falls back gracefully
    if the LLM doesn't follow the format exactly.
    """
    # Try to split on "Keywords:" line
    kw_match = re.search(r'(?:^|\n)\s*Keywords?\s*:\s*(.+)', response, re.IGNORECASE)
    if kw_match:
        raw_keywords = kw_match.group(1).strip().rstrip(".")
        # Everything before the Keywords line is the content
        content = response[:kw_match.start()].strip()
    else:
        content = response.strip()
        raw_keywords = ""

    # Strip leading "Summary:" or "Lesson:" prefix from content
    content = re.sub(r'^(?:Summary|Lesson)\s*:\s*', '', content, flags=re.IGNORECASE).strip()

    # Normalize keywords: split, strip, lowercase, deduplicate
    keywords = []
    for kw in raw_keywords.split(","):
        kw = kw.strip().lower()
        if kw and kw not in keywords:
            keywords.append(kw)

    return content, ",".join(keywords)
