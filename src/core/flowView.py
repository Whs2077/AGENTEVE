"""Write the call graph as a self-contained HTML page beside callGraph.json."""

import html
import json
from collections import defaultdict
from pathlib import Path


def writeFlowHtml(path, graph):
    """Write one HTML flow graph. The page embeds the same JSON written beside it."""
    Path(path).write_text(renderFlowHtml(graph), encoding="utf-8")


def renderFlowHtml(graph):
    """Return an HTML document with an SVG flow graph and the graph JSON."""
    payload = json.dumps(graph, ensure_ascii=False).replace("<", "\\u003c")
    basis = graph.get("ground_truth") or {}
    if basis.get("present"):
        banner = "本图将与实例中的参考流图对照。对照是否一致写在同目录的 groundTruth.json。"
    else:
        banner = "没有参考流图。这张图完全由本项目分析得到，没有真值依据。"
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>AGENTEVE 流图</title>
<style>
body {{ margin: 0; font: 14px Arial, sans-serif; color: #183e59; background: #f6f9fd; }}
header {{ padding: 16px 20px 8px; }}
h1 {{ margin: 0 0 6px; font-size: 20px; }}
p {{ margin: 0; }}
#layout {{ display: flex; gap: 16px; align-items: stretch; padding: 8px 16px 20px; }}
#stage {{ overflow: auto; flex: 1; background: white; border: 1px solid #c5d7e6; border-radius: 8px; }}
aside {{ width: 320px; background: white; border: 1px solid #c5d7e6; border-radius: 8px; padding: 12px; }}
svg {{ display: block; }}
.node {{ cursor: pointer; }}
.node.selected rect {{ stroke: #183e59; stroke-width: 3; }}
</style>
</head>
<body>
<header>
<h1>AGENTEVE 流图</h1>
<p>{html.escape(banner)}</p>
</header>
<div id="layout">
<div id="stage">{_svg(graph)}</div>
<aside id="detail"><strong>节点</strong><p>点击图中的节点查看源码位置。</p></aside>
</div>
<script type="application/json" id="call-graph">{payload}</script>
<script>
const graph = JSON.parse(document.getElementById("call-graph").textContent);
const detail = document.getElementById("detail");
document.querySelectorAll(".node").forEach((item) => {{
  item.addEventListener("click", () => {{
    document.querySelectorAll(".node").forEach((other) => other.classList.remove("selected"));
    item.classList.add("selected");
    const node = graph.nodes[Number(item.dataset.index)];
    const evidence = (node.evidence || []).map((anchor) =>
      `${{anchor.file || ""}}:${{anchor.line || ""}}`).join("<br>");
    detail.innerHTML = `<strong>${{node.label}}</strong><p>${{node.kind}}</p><p>${{node.id}}</p><p>${{evidence}}</p>`;
  }});
}});
</script>
</body>
</html>
"""


def _svg(graph):
    rank = _ranks(graph)
    columns = defaultdict(list)
    for node in graph["nodes"]:
        columns[rank[node["id"]]].append(node)
    width, height, origin, column_of = 240, 76, {}, {}
    for column, depth in enumerate(sorted(columns)):
        for row, node in enumerate(columns[depth]):
            origin[node["id"]] = (48 + column * 300, 48 + row * 120, width, height)
            column_of[node["id"]] = column
    base_y = max((item[1] + height for item in origin.values()), default=height) + 36
    backwards = [edge for edge in graph["edges"]
                 if edge["source"] != edge["target"] and column_of.get(edge["target"], 0) <= column_of.get(edge["source"], 0)]
    gutter = {id(edge): base_y + index * 22 for index, edge in enumerate(backwards)}
    max_x = max((item[0] + width for item in origin.values()), default=width) + 64
    max_y = max([base_y, *gutter.values()], default=base_y) + 28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_x}" height="{max_y}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#397ba7"/></marker></defs>',
    ]
    index = {node["id"]: number for number, node in enumerate(graph["nodes"])}
    for edge in graph["edges"]:
        if edge["source"] not in origin or edge["target"] not in origin:
            continue
        parts.append(_edge(origin[edge["source"]], origin[edge["target"]], edge,
                           column_of[edge["source"]], column_of[edge["target"]], gutter.get(id(edge))))
    for node in graph["nodes"]:
        x, y, box_width, box_height = origin[node["id"]]
        fill, ink = ("#397ba7", "#ffffff") if node["kind"] in {"entry", "terminal", "exit"} else ("#ffffff", "#183e59")
        dash = ' stroke-dasharray="5 4"' if node["kind"] == "unresolved" else ""
        parts.append(
            f'<g class="node" data-index="{index[node["id"]]}">'
            f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="10" '
            f'fill="{fill}" stroke="#6398bf"{dash}/>{_label(node, x, y, box_width, ink)}</g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _label(node, x, y, width, ink):
    text = node["label"]
    center = x + width / 2
    if " (" in text and len(text) > 24:
        name, extra = text.split(" (", 1)
        return (f'<text x="{center}" y="{y + 32}" text-anchor="middle" fill="{ink}" font-size="13">'
                f'{html.escape(name)}</text>'
                f'<text x="{center}" y="{y + 52}" text-anchor="middle" fill="{ink}" font-size="12">'
                f'{html.escape("(" + extra)}</text>')
    return (f'<text x="{center}" y="{y + 44}" text-anchor="middle" fill="{ink}" font-size="14">'
            f'{html.escape(text)}</text>')


def _edge(start, end, edge, source_column, target_column, gutter):
    x1, y1, width, height = start
    x2, y2, _, _ = end
    dashed = ' stroke-dasharray="5 4"' if edge["kind"] in {"possible_exception", "unresolved", "async_launch"} else ""
    stroke = f'fill="none" stroke="#397ba7" marker-end="url(#arrow)"{dashed}'
    if edge["source"] == edge["target"]:
        left, top = x1 + width, y1 + 18
        return f'<path d="M {left} {top} C {left + 46} {top - 28}, {left + 46} {top + 28}, {left} {y1 + 58}" {stroke}/>'
    if gutter is not None:
        leave = (x1 + width / 2, y1 + height)
        arrive = (x2 + width / 2, y2 + height)
        return (f'<path d="M {leave[0]} {leave[1]} C {leave[0]} {gutter}, {arrive[0]} {gutter}, {arrive[0]} {arrive[1]}" '
                f'{stroke}/>')
    leave = (x1 + width, y1 + height / 2)
    arrive = (x2, y2 + height / 2)
    if target_column == source_column + 1 and abs(leave[1] - arrive[1]) < 1:
        return f'<line x1="{leave[0]}" y1="{leave[1]}" x2="{arrive[0]}" y2="{arrive[1]}" {stroke}/>'
    bend = min(leave[1], arrive[1]) - 28
    return f'<path d="M {leave[0]} {leave[1]} C {leave[0] + 40} {bend}, {arrive[0] - 40} {bend}, {arrive[0]} {arrive[1]}" {stroke}/>'


def _ranks(graph):
    """Place the terminal after the longest forward path. Cycle edges bend underneath."""
    identifiers = [node["id"] for node in graph["nodes"]]
    adjacent = defaultdict(list)
    for edge in graph["edges"]:
        if edge["kind"] != "unresolved":
            adjacent[edge["source"]].append(edge["target"])
    color = {node_id: 0 for node_id in identifiers}
    back = set()

    def walk(node):
        color[node] = 1
        for target in adjacent[node]:
            if target not in color:
                continue
            if target == node or color[target] == 1:
                back.add((node, target))
            elif color[target] == 0:
                walk(target)
        color[node] = 2

    entry = graph.get("entry")
    if entry in color:
        walk(entry)
    for node_id in identifiers:
        if color[node_id] == 0:
            walk(node_id)
    rank = {entry: 0} if entry in color else {}
    for _ in identifiers:
        changed = False
        for source, targets in adjacent.items():
            if source not in rank:
                continue
            for target in targets:
                if (source, target) in back:
                    continue
                if rank.get(target, -1) < rank[source] + 1:
                    rank[target] = rank[source] + 1
                    changed = True
        if not changed:
            break
    for node_id in identifiers:
        rank.setdefault(node_id, max(rank.values(), default=-1) + 1)
    terminals = {node["id"] for node in graph["nodes"] if node["kind"] in {"terminal", "exit"}}
    if terminals:
        body = max((rank[node_id] for node_id in identifiers if node_id not in terminals), default=-1)
        for node_id in terminals:
            rank[node_id] = body + 1
    return rank
