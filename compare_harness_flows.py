"""Audit next-call adjacency and structural reachability across saved LLM flow graphs."""

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path

from agenteve import metric, read_json


ROOT = Path(__file__).resolve().parent
GRAPH_ROOT = ROOT / "results/stage1-llm-flow"
CASES = ("open-deep-research", "stirrup", "opencode")
TRACE_FILES = tuple(ROOT / "results/stage2" / batch / "runs.jsonl"
                    for batch in ("deepseek-pilot-2048", "deepseek-holdout-v1"))


def provenance(path):
    resolved = path.resolve()
    display = resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved
    return {"path": display.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def next_call_adjacency(nodes, edges):
    adjacent = {node: set() for node in nodes}
    for source, target, _, kind in edges:
        if source not in adjacent or target not in adjacent:
            raise ValueError(f"Edge has an undeclared endpoint: {source} → {target}")
        if kind == "next_call":
            adjacent[source].add(target)
    return adjacent


def paths_from(adjacent, start, omitted=None):
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


def reachable_pairs(adjacent, omitted=None):
    return {(start, target) for start in adjacent for target in paths_from(adjacent, start, omitted)}


def edge_drop_checks(adjacent, pairs):
    checks = []
    for source, targets in sorted(adjacent.items()):
        for target in sorted(targets):
            removed = (source, target)
            reachable_after = reachable_pairs(adjacent, removed)
            alternative = paths_from(adjacent, source, removed).get(target)
            checks.append({
                "edge": removed,
                "alternative_path": alternative,
                "lost_pairs": sorted(pairs - reachable_after),
            })
    return checks


def compare_case(graph_path, reference_path):
    graph, reference = read_json(graph_path), read_json(reference_path)
    if graph.get("graph_kind") != "llm_call_successors":
        raise ValueError(f"{graph_path} does not contain an LLM-call projection")
    observed_nodes = {(node["id"], node["kind"]) for node in graph["nodes"]}
    reference_nodes = {tuple(node) for node in reference["nodes"]}
    observed_edges = {(edge["source"], edge["target"], edge["route"], edge["kind"]): edge
                      for edge in graph["edges"]}
    reference_edges = {tuple(edge) for edge in reference["edges"]}
    observed = next_call_adjacency({node for node, _ in observed_nodes}, observed_edges)
    expected = next_call_adjacency({node for node, _ in reference_nodes}, reference_edges)
    observed_pairs, expected_pairs = reachable_pairs(observed), reachable_pairs(expected)
    edge_score = metric(set(observed_edges), reference_edges)
    return {
        "graph": provenance(graph_path),
        "reference": provenance(reference_path),
        "scope": reference["scope"],
        "entry_matches": graph["entry"] == reference["entry"],
        "nodes": metric(observed_nodes, reference_nodes),
        "typed_direct_edges": edge_score,
        "sequential_reachability": metric(observed_pairs, expected_pairs),
        "async_launch_count": {
            "predicted": sum(kind == "async_launch" for _, _, _, kind in observed_edges),
            "reference": sum(kind == "async_launch" for _, _, _, kind in reference_edges),
        },
        "extra_edges": [
            {
                "edge": edge,
                "raw_condition": observed_edges[edge].get("condition"),
                "first_source_anchor": next(iter(observed_edges[edge].get("evidence", [])), None),
                "reference_alternative_path":
                    paths_from(expected, edge[0]).get(edge[1]) if edge[3] == "next_call" and edge[0] in expected else None,
            }
            for edge in edge_score["extra"]
        ],
        "reference_edge_drop_checks": edge_drop_checks(expected, expected_pairs),
    }


def trace_audit(graph_path, trace_paths):
    graph = read_json(graph_path)
    allowed = {(edge["source"], edge["target"]) for edge in graph["edges"] if edge["kind"] == "next_call"}
    result = []
    for path in trace_paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").split("\n") if line.strip()]
        if not rows:
            raise ValueError(f"{path} contains no recorded runs")
        counts = Counter()
        missing = []
        for row in rows:
            for pair in zip(row["path"], row["path"][1:]):
                counts[pair] += 1
                if pair not in allowed:
                    missing.append({"run_id": row["run_id"], "transition": pair})
        result.append({
            "source": provenance(path),
            "runs": len(rows),
            "adjacent_transitions": sum(counts.values()),
            "matched": sum(count for pair, count in counts.items() if pair in allowed),
            "observed": [{"edge": pair, "count": count} for pair, count in sorted(counts.items())],
            "missing": missing,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New JSON audit output")
    args = parser.parse_args()
    try:
        cases = {
            name: compare_case(GRAPH_ROOT / name / "extracted/flow.json", GRAPH_ROOT / name / "reference.json")
            for name in CASES
        }
        result = {
            "protocol": {
                "direct_edges": "(source, target, route, kind)",
                "reachability": "Positive-length structural paths over next_call edges; async launch is separate.",
                "edge_drop": "Drop one reference next_call adjacency at a time, with no mutation of source files.",
                "trace_scope": "Audits recorded Open Deep Research node transitions within the Stage 2 pilot segment.",
            },
            "cases": cases,
            "open_deep_research_trace_check": trace_audit(
                GRAPH_ROOT / "open-deep-research/extracted/flow.json", TRACE_FILES),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for name, audit in cases.items():
        direct, reach = audit["typed_direct_edges"], audit["sequential_reachability"]
        print(f"{name}: direct {direct['matched']}/{direct['reference']} (+{len(direct['extra'])}), "
              f"sequential reachability {reach['matched']}/{reach['reference']} (+{len(reach['extra'])})")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
