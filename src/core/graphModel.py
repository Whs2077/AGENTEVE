"""Call graph objects and the structural fields added after extraction."""

import ast
from collections import defaultdict
import hashlib
import tokenize
from pathlib import Path


class Unsupported(ValueError):
    pass


class Source:
    def __init__(self, path, url=None, coordinates="file"):
        self.path = Path(path)
        self.url = url
        self.coordinates = coordinates
        with tokenize.open(self.path) as stream:
            self.tree = ast.parse(stream.read(), filename=str(path))

    def evidence(self, node):
        result = {"file": self.path.name, "line": node.lineno, "end_line": node.end_lineno,
                  "column": node.col_offset + 1}
        if self.url:
            result["url"] = self.url
            if self.coordinates == "file" and "/blob/" in self.url:
                result["url"] += f"#L{node.lineno}-L{node.end_lineno}"
        return result

    def metadata(self):
        # Ignores formatting so browser text snapshots and local files can be compared.
        fingerprint = hashlib.sha256(ast.dump(self.tree).encode("utf-8")).hexdigest()
        return {"file": self.path.name, "url": self.url, "coordinates": self.coordinates,
                "ast_sha256": fingerprint}


class Graph:
    def __init__(self, extractor, source, entry, scope, graph_kind):
        self.data = {"schema_version": "1.0", "extractor": extractor, "entry": entry,
                     "scope": scope, "graph_kind": graph_kind, "sources": [source.metadata()],
                     "nodes": [], "edges": [], "diagnostics": []}
        self.node_ids = set()
        self.edge_index = {}

    def node(self, node_id, kind, label, *evidence):
        if node_id in self.node_ids:
            raise Unsupported(f"Duplicate node {node_id!r}")
        self.node_ids.add(node_id)
        self.data["nodes"].append({"id": node_id, "kind": kind, "label": label,
                                   "evidence": list(evidence)})

    def edge(self, start, end, route, evidence, *, kind="flow", condition=None, additional_evidence=()):
        key = (start, end, route, kind)
        anchors = [evidence, *additional_evidence]
        if key in self.edge_index:
            edge = self.edge_index[key]
            edge["evidence"].extend(anchor for anchor in anchors if anchor not in edge["evidence"])
            return
        edge = {"source": start, "target": end, "route": route, "kind": kind,
                "condition": condition, "evidence": anchors}
        self.edge_index[key] = edge
        self.data["edges"].append(edge)

    def unknown(self, start, route, message, evidence):
        node_id = f"<unresolved:{start}:{route}:{evidence['line']}:{evidence['column']}>"
        if node_id not in self.node_ids:
            self.node(node_id, "unresolved", "Unresolved transition", evidence)
        self.edge(start, node_id, route, evidence, kind="unresolved")
        self.warn(message, evidence, unresolved=True)

    def warn(self, message, evidence, *, unresolved=False):
        self.data["diagnostics"].append({"message": message, "evidence": evidence,
                                         "unresolved": unresolved})

    def finish(self):
        return completeGraph(self.data)


def completeGraph(graph):
    ids = {node["id"] for node in graph["nodes"]}
    entry = graph["entry"]
    if len(ids) != len(graph["nodes"]) or entry not in ids:
        raise Unsupported("Duplicate nodes or missing graph entry")
    adjacency = defaultdict(set)
    predecessors = defaultdict(set)
    for edge in graph["edges"]:
        start, end = edge["source"], edge["target"]
        if start not in ids or end not in ids:
            raise Unsupported(f"Edge references an undeclared node: {start!r} → {end!r}")
        if edge["kind"] != "unresolved":
            adjacency[start].add(end)
            predecessors[end].add(start)

    def reaches_origin(origin, visible):
        visited = set()
        pending = list(adjacency[origin] & visible)
        while pending:
            target = pending.pop()
            if target == origin:
                return True
            if target not in visited:
                visited.add(target)
                pending.extend((adjacency[target] & visible) - visited)
        return False

    joins = {n["id"] for n in graph["nodes"] if n["kind"] == "join"}
    entry_reachable = {entry}
    pending = [entry]
    while pending:
        start = pending.pop()
        for end in adjacency[start] - entry_reachable:
            if end not in joins or predecessors[end] <= entry_reachable:
                entry_reachable.add(end)
                pending.append(end)
    graph["structure"] = {
        "reachable_from_entry": sorted(entry_reachable, key=str),
        "cycle_nodes": sorted((n for n in entry_reachable if reaches_origin(n, entry_reachable)), key=str),
        "contains_unresolved_transitions": any(e["kind"] == "unresolved" for e in graph["edges"])
        or any(note.get("unresolved") for note in graph["diagnostics"]),
    }
    return graph
