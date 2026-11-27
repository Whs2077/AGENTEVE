"""Write one extracted call graph as JSON, ground-truth basis, Mermaid, DOT, and HTML."""

import json
from pathlib import Path

from common.fileStore import writeJson
from src.core.flowView import writeFlowHtml


def writeGraph(directory, graph, *, reference=None):
    """Write every graph view into ``directory``. ``reference`` is the instance's reference flow graph."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    graph["ground_truth"] = groundTruth(reference)
    writeJson(directory / "callGraph.json", graph)
    writeJson(directory / "groundTruth.json", graph["ground_truth"])
    (directory / "callGraph.mmd").write_text(mermaid(graph), encoding="utf-8")
    (directory / "callGraph.dot").write_text(dot(graph), encoding="utf-8")
    writeFlowHtml(directory / "callGraph.html", graph)
    print(f"{graph['extractor']}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges -> {directory}")


def groundTruth(reference):
    """Say whether a reference flow graph exists. Absence means there is no ground truth."""
    if reference is None:
        return {"present": False, "basis": "project_analysis",
                "note": "没有参考流图。这张图完全由本项目分析得到，没有真值依据。"}
    return {"present": True, "basis": "reference", "reference": Path(reference).as_posix()}


def mermaid(graph):
    ids = {n["id"]: f"n{i}" for i, n in enumerate(graph["nodes"])}
    output = ["flowchart TD"]
    for node in graph["nodes"]:
        label = node["label"].replace("&", "&amp;").replace('"', "&quot;")
        shape = ('{"', '"}') if node["kind"] == "branch" else ('["', '"]')
        output.append(f'  {ids[node["id"]]}{shape[0]}{label}{shape[1]}')
    for edge in graph["edges"]:
        arrow = "-.->" if edge["kind"] in {"possible_exception", "unresolved", "async_launch"} else "-->"
        condition = "" if edge["route"] == "next" else f'|"{edge["route"].replace(chr(34), "&quot;")}"|'
        output.append(f'  {ids[edge["source"]]} {arrow}{condition} {ids[edge["target"]]}')
    return "\n".join(output) + "\n"


def dot(graph):
    q = lambda value: json.dumps(value, ensure_ascii=False)
    lines = [
        "digraph llm_flow {",
        "  graph [bgcolor=\"#f6f9fd\", labelloc=t, label=\"AGENTEVE | Extracted LLM call flow\", "
        "fontsize=18, fontname=\"Arial\", ranksep=.6, pad=.3];",
        "  node [shape=box, style=\"filled,rounded\", fillcolor=white, color=\"#6398bf\", "
        "fontcolor=\"#183e59\", fontname=\"Arial\", fontsize=14, margin=\".2,.12\"];",
        "  edge [color=\"#397ba7\", arrowsize=.75, penwidth=1.3];",
    ]
    for node in graph["nodes"]:
        boundary = ", shape=oval, fillcolor=\"#397ba7\", fontcolor=white" if node["kind"] != "llm_call" else ""
        lines.append(f"  {q(node['id'])} [label={q(node['label'])}{boundary}];")
    for edge in graph["edges"]:
        style = " [style=dashed]" if edge.get("condition") else ""
        lines.append(f"  {q(edge['source'])} -> {q(edge['target'])}{style};")
    lines.append("}")
    return "\n".join(lines) + "\n"
