"""Build a call graph from a fixed source snapshot and check it against a reference."""

import json
from pathlib import Path

from common.callPaths import edgeDropChecks, nextCallAdjacency, reachablePairs
from common.fileStore import readJson, writeJson
from src.core.declaredTopology import readDeclaredTopology
from src.core.graphModel import Source
from src.core.graphOutput import writeGraph
from src.core.projectFromApi import projectFromApi
from src.core.projectNextCall import projectNextCall
from src.core.projectStirrup import projectStirrup

ADAPTERS = ("locateApi", "projectNextCall", "projectStirrup", "readDeclaredTopology")


def matchSets(actual, expected):
    hit = len(actual & expected)
    precision = hit / len(actual) if actual else None
    recall = hit / len(expected) if expected else None
    return {"matched": hit, "predicted": len(actual), "reference": len(expected),
            "precision": precision, "recall": recall,
            "f1": 2 * hit / (len(actual) + len(expected)) if actual or expected else None,
            "missing": sorted(expected - actual, key=str), "extra": sorted(actual - expected, key=str)}


def scoreGraph(actual, expected):
    nodes = {(node["id"], node["kind"]) for node in actual["nodes"]}
    edges = {(edge["source"], edge["target"], edge["route"], edge["kind"]) for edge in actual["edges"]}
    result = {"nodes": matchSets(nodes, {tuple(node) for node in expected["nodes"]}),
              "edges": matchSets(edges, {tuple(edge) for edge in expected["edges"]}),
              "entry_matches": actual["entry"] == expected["entry"],
              "cycle_nodes_match": actual["structure"]["cycle_nodes"] == sorted(expected["cycle_nodes"], key=str)}
    if "call_sites" in expected:
        observed_calls = set()
        for node in actual["nodes"]:
            if node["kind"] == "llm_call":
                anchor = next(iter(node.get("evidence", [])), {})
                caller = node.get("caller", {})
                observed_calls.add((node["id"], anchor.get("file"), anchor.get("line"),
                                    caller.get("file"), caller.get("line")))
        annotated_calls = {(name, anchor["file"], anchor["line"], anchor["caller_file"], anchor["caller_line"])
                           for name, anchor in expected["call_sites"].items()}
        result["call_sites"] = matchSets(observed_calls, annotated_calls)
    return result


def _listed(value):
    if isinstance(value, tuple):
        return [_listed(item) for item in value]
    if isinstance(value, list):
        return [_listed(item) for item in value]
    if isinstance(value, dict):
        return {key: _listed(item) for key, item in value.items()}
    return value


def structuralCheck(actual, expected):
    observed_nodes = {(node["id"], node["kind"]) for node in actual["nodes"]}
    reference_nodes = {tuple(node) for node in expected["nodes"]}
    observed_edges = {(edge["source"], edge["target"], edge["route"], edge["kind"]): edge
                      for edge in actual["edges"]}
    reference_edges = {tuple(edge) for edge in expected["edges"]}
    observed = nextCallAdjacency({node for node, _ in observed_nodes}, observed_edges)
    reference = nextCallAdjacency({node for node, _ in reference_nodes}, reference_edges)
    observed_pairs, reference_pairs = reachablePairs(observed), reachablePairs(reference)
    return {
        "reachable_pairs": {
            "predicted": len(observed_pairs),
            "reference": len(reference_pairs),
            "matched": len(observed_pairs & reference_pairs),
        },
        "unresolved": actual.get("structure", {}).get("contains_unresolved_transitions"),
        "reference_edge_drops": _listed(edgeDropChecks(reference, reference_pairs)),
    }


def checkGraph(graphPath, referencePath):
    """Compare an extracted graph with a reference. Return 0 when the reference matches."""
    actual = readJson(graphPath)
    expected = readJson(referencePath)
    result = scoreGraph(actual, expected)
    if actual.get("graph_kind") == "llm_call_successors":
        result["structural"] = structuralCheck(actual, expected)
    writeJson(Path(graphPath).with_name("graphCheck.json"), result)
    printable = {key: value for key, value in result.items() if key != "structural"}
    print(json.dumps(printable, ensure_ascii=True, indent=2))
    checks = [result["nodes"], result["edges"]]
    if "call_sites" in result:
        checks.append(result["call_sites"])
    failed = any(item["extra"] or item["missing"] for item in checks)
    failed = failed or not result["entry_matches"] or not result["cycle_nodes_match"]
    if actual.get("structure", {}).get("contains_unresolved_transitions"):
        failed = True
    basis = {"present": True, "basis": "reference", "reference": Path(referencePath).as_posix(), "matched": not failed}
    writeJson(Path(graphPath).with_name("groundTruth.json"), basis)
    return int(failed)


def buildGraph(source, adapter, outDir, *, builder=None, reference=None, sourceUrl=None,
               clientProtocol=None, clientProtocolUrl=None, coordinates="file"):
    """Extract one call graph, write every view of it, and check it when a reference is given."""
    outDir = Path(outDir)
    source = Path(source)
    if adapter not in ADAPTERS:
        raise ValueError(f"Unknown adapter {adapter}. Choose one of {', '.join(ADAPTERS)}")
    if adapter == "locateApi":
        graph = projectFromApi(source, builder=builder, protocol=clientProtocol, sourceUrl=sourceUrl,
                               protocolUrl=clientProtocolUrl, coordinates=coordinates)
    elif adapter == "projectStirrup":
        if clientProtocol is None:
            raise ValueError("projectStirrup requires clientProtocol")
        graph = projectStirrup(Source(source, sourceUrl, coordinates),
                               Source(clientProtocol, clientProtocolUrl, coordinates))
    elif adapter == "projectNextCall":
        graph = projectNextCall(Source(source, sourceUrl, coordinates), builder)
    else:
        graph = readDeclaredTopology(Source(source, sourceUrl, coordinates), builder)
    writeGraph(outDir, graph, reference=reference)
    if reference:
        return checkGraph(outDir / "callGraph.json", reference)
    return 0
