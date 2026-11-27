"""Shortest paths on next_call edges. No trials and no scores."""

from collections import deque
from pathlib import Path

from common.fileStore import fileSha256


def provenance(path):
    path = Path(path)
    resolved = path.resolve()
    root = Path(__file__).resolve().parents[1]
    display = resolved.relative_to(root) if resolved.is_relative_to(root) else resolved
    return {"path": display.as_posix(), "sha256": fileSha256(path)}


def nextCallAdjacency(nodes, edges):
    adjacent = {node: set() for node in nodes}
    for source, target, _, kind in edges:
        if source not in adjacent or target not in adjacent:
            raise ValueError(f"Edge has an undeclared endpoint: {source} → {target}")
        if kind == "next_call":
            adjacent[source].add(target)
    return adjacent


def shortestPaths(adjacent, start, omitted=None):
    """Return shortest positive-length paths, including a cycle to start if present."""
    witnesses = {}
    pending = deque([(start, (start,))])
    while pending:
        current, prefix = pending.popleft()
        for neighbor in sorted(adjacent[current]):
            if (current, neighbor) == omitted or neighbor in witnesses:
                continue
            path = (*prefix, neighbor)
            witnesses[neighbor] = path
            if neighbor != start:
                pending.append((neighbor, path))
    return witnesses


def reachablePairs(adjacent, omitted=None):
    return {(start, target) for start in adjacent for target in shortestPaths(adjacent, start, omitted)}


def edgeDropChecks(adjacent, pairs):
    checks = []
    for source, targets in sorted(adjacent.items()):
        for target in sorted(targets):
            removed = (source, target)
            reachable_after = reachablePairs(adjacent, removed)
            alternative = shortestPaths(adjacent, source, removed).get(target)
            checks.append({
                "edge": removed,
                "alternative_path": alternative,
                "lost_pairs": sorted(pairs - reachable_after),
            })
    return checks
