"""Incoming and outgoing calls for symbols that contain a located API site.

The walk follows the same two steps as Crabviz: name the functions, then ask
which function calls which. It does not decide which of those functions is a
graph node.
"""

import ast
from collections import defaultdict
from pathlib import Path

from src.core.pyScan import dotted
from src.core.tsScan import KEYWORDS, typescript_calls, typescript_functions


def hierarchy(sites, files=None):
    """Keep the callers that reach each site and the direct calls those sites make."""
    files = list(files) if files is not None else sorted({Path(site.path) for site in sites})
    symbols, edges = _index(files)
    site_ids = {site.symbol for site in sites}
    keep = set(site_ids)
    pending = list(site_ids)
    incoming = defaultdict(list)
    for edge in edges:
        incoming[edge["target"]].append(edge)
    while pending:
        current = pending.pop()
        for edge in incoming.get(current, []):
            if edge["source"] not in keep:
                keep.add(edge["source"])
                pending.append(edge["source"])
    for edge in edges:
        if edge["source"] in site_ids:
            keep.add(edge["target"])
    visible = [edge for edge in edges if edge["source"] in keep and edge["target"] in keep]
    return {"symbols": [symbols[item] for item in sorted(keep) if item in symbols],
            "edges": visible}


def _index(files):
    symbols = {}
    pending = []
    for file in files:
        if file.suffix.lower() == ".py":
            file_symbols, file_calls = _python_index(file)
        elif file.suffix.lower() in {".ts", ".tsx"}:
            file_symbols, file_calls = _typescript_index(file)
        else:
            continue
        symbols.update(file_symbols)
        pending.extend(file_calls)
    by_suffix = defaultdict(list)
    for symbol_id in symbols:
        short = symbol_id.split(":", 1)[1]
        by_suffix[short].append(symbol_id)
    edges = []
    seen = set()
    for source, callee, file, line in pending:
        matches = by_suffix.get(callee, [])
        if len(matches) != 1 or matches[0] == source:
            continue
        key = (source, matches[0], file, line)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": source, "target": matches[0], "file": file, "line": line})
    return symbols, edges


def _python_index(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {}
    raw_calls = []

    class Index(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node):
            self._function(node)

        def visit_AsyncFunctionDef(self, node):
            self._function(node)

        def _function(self, node):
            self.stack.append(node.name)
            functions[".".join(self.stack)] = node
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            if self.stack:
                raw_calls.append((".".join(self.stack), dotted(node.func), node.lineno))
            self.generic_visit(node)

    Index().visit(tree)
    qualified = {f"{path.name}:{name}": {"id": f"{path.name}:{name}", "file": path.name, "line": node.lineno}
                 for name, node in functions.items()}
    calls = []
    for owner, callee, line in raw_calls:
        resolved = _resolve_python(path.name, owner, callee, qualified)
        if resolved is not None:
            calls.append((f"{path.name}:{owner}", resolved.split(":", 1)[1], path.name, line))
    return qualified, calls


def _resolve_python(file, owner, callee, qualified):
    if not callee or callee.split(".")[-1] in KEYWORDS:
        return None
    if callee.startswith("self.") and callee.count(".") == 1 and "." in owner:
        candidate = f"{file}:{owner.rsplit('.', 1)[0]}.{callee.split('.', 1)[1]}"
        if candidate in qualified:
            return candidate
    local = f"{file}:{callee}"
    if local in qualified:
        return local
    return None


def _typescript_index(path):
    functions, _tokens = typescript_functions(path)
    qualified = {}
    for item in functions["ranges"]:
        symbol = f"{path.name}:{item['symbol']}"
        qualified[symbol] = {"id": symbol, "file": path.name, "line": item["line"]}
    calls = []
    for call in typescript_calls(path):
        source = f"{path.name}:{call['symbol']}"
        if source in qualified:
            calls.append((source, call["callee"], path.name, call["line"]))
    return qualified, calls


