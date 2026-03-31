"""Shared utilities for consolidation plugins."""

from collections import Counter
from typing import List, Optional

import numpy as np

from typemem.memory_item import MemoryItem, MemoryTier
from typemem.memory_manager import MemoryManager


def most_common_waypoint(items: List[MemoryItem]) -> Optional[int]:
    """Return the most frequent non-None waypoint across items, or None."""
    waypoints = [item.waypoint for item in items if item.waypoint is not None]
    if not waypoints:
        return None
    return Counter(waypoints).most_common(1)[0][0]


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix. Returns NxN matrix in [0, 2]."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    normalized = embeddings / norms
    return 1.0 - normalized @ normalized.T


def group_by_similarity(
    manager: MemoryManager,
    items: List[MemoryItem],
    tier: MemoryTier,
    distance_threshold: Optional[float] = None,
) -> List[List[MemoryItem]]:
    """Greedy clustering of items by semantic similarity.

    Retrieves stored embeddings in one batch call and computes exact
    pairwise cosine distances in-memory, avoiding N sequential ChromaDB
    semantic queries.

    Args:
        manager: MemoryManager for embedding retrieval.
        items: Items to cluster.
        tier: Tier (unused, kept for API compatibility).
        distance_threshold: If set, only include items below this cosine distance.
            If None, include all items in each seed's group.

    Returns:
        List of groups (each group is a list of MemoryItems).
    """
    if not items:
        return []

    item_ids = [item.id for item in items]
    raw_embeddings = manager.get_embeddings(item_ids)

    # Fallback to search-based clustering if embeddings unavailable
    if raw_embeddings is None:
        return _group_by_search_fallback(manager, items, tier, distance_threshold)

    embeddings = np.array(raw_embeddings, dtype=np.float32)
    dist_matrix = _cosine_distance_matrix(embeddings)

    used = set()
    groups = []

    for i, item in enumerate(items):
        if item.id in used:
            continue

        group = [item]
        used.add(item.id)

        for j, other in enumerate(items):
            if other.id in used:
                continue
            dist = dist_matrix[i, j]
            if distance_threshold is None or dist < distance_threshold:
                group.append(other)
                used.add(other.id)

        groups.append(group)

    return groups


def _group_by_search_fallback(
    manager: MemoryManager,
    items: List[MemoryItem],
    tier: MemoryTier,
    distance_threshold: Optional[float],
) -> List[List[MemoryItem]]:
    """Fallback: greedy clustering via N semantic searches (used if embeddings unavailable)."""
    used = set()
    groups = []
    item_map = {item.id: item for item in items}

    for item in items:
        if item.id in used:
            continue

        group = [item]
        used.add(item.id)

        if distance_threshold is not None:
            similar, distances = manager.search_with_distances(
                query=item.document, tiers=[tier],
                n_results=min(20, len(items)),
            )
            for sim_item, dist in zip(similar, distances):
                if sim_item.id in used or sim_item.id not in item_map:
                    continue
                if dist < distance_threshold:
                    group.append(item_map[sim_item.id])
                    used.add(sim_item.id)
        else:
            similar = manager.search(
                query=item.document, tiers=[tier],
                n_results=min(20, len(items)),
            )
            for sim_item in similar:
                if sim_item.id in used or sim_item.id not in item_map:
                    continue
                group.append(sim_item)
                used.add(sim_item.id)

        groups.append(group)

    return groups
