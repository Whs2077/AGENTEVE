"""Project normal LangGraph source paths onto the next known chat-model call."""

import ast
from collections import defaultdict, deque
from dataclasses import dataclass
import json

from agenteve import Graph, Unsupported, argument, dotted, function_nodes, langgraph


MODEL_FACTORY = "langchain.chat_models.init_chat_model"
MODEL_WRAPPERS = {"bind", "bind_tools", "with_config", "with_retry", "with_structured_output"}
INVOKE_METHODS = {"invoke", "ainvoke", "batch", "abatch", "stream", "astream"}
ASYNC_METHODS = {"ainvoke", "abatch", "astream"}


@dataclass(frozen=True)
class Binding:
    kind: str
    value: object
    origin: ast.AST | None = None


@dataclass(frozen=True)
class Deferred:
    receiver: Binding
    call: ast.Call
    batched: bool = False


@dataclass
class Cursor:
    position: int
    bindings: dict

    def copy(self):
        return Cursor(self.position, self.bindings.copy())


@dataclass(frozen=True)
class Arc:
    target: int
    evidence: dict
    condition: str | None = None
    step: str | None = None


@dataclass
class Handler:
    name: str
    finish: int
    routes: dict
    dynamic_returns: set
    also_follows_fixed_edges: bool
    builders: tuple


def assigned_names(stmt):
    if isinstance(stmt, ast.Assign):
        return [t.id for t in stmt.targets if isinstance(t, ast.Name)]
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id]
    return []


def parameter_names(arguments):
    parameters = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                  arguments.vararg, arguments.kwarg)
    return {parameter.arg for parameter in parameters if parameter is not None}


def import_bindings(stmt):
    if isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0:
        return {alias.asname or alias.name: Binding("import", f"{stmt.module}.{alias.name}")
                for alias in stmt.names if alias.name != "*"}
    if isinstance(stmt, ast.Import):
        return {alias.asname or alias.name.split(".")[0]:
                Binding("import", alias.name if alias.asname else alias.name.split(".")[0])
                for alias in stmt.names}
    return {}


def symbol(expr, bindings):
    name = dotted(expr)
    head, sep, rest = name.partition(".")
    root = bindings.get(head)
    return f"{root.value}{sep}{rest}" if root and root.kind == "import" else ""


def model_selector(expr):
    if expr is None:
        return None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return ast.unparse(expr)


def static_binding(expr, bindings):
    """Track model provenance, selected configuration and simple constant guards."""
    if isinstance(expr, ast.Name):
        return bindings.get(expr.id)
    if isinstance(expr, ast.Constant) and type(expr.value) in {bool, int, str}:
        return Binding("constant", expr.value)
    if isinstance(expr, ast.Dict):
        for key, value in zip(expr.keys, expr.values):
            if isinstance(key, ast.Constant) and key.value == "model":
                return Binding("model_config", model_selector(value))
            if isinstance(key, ast.Constant) and key.value == "configurable":
                inner = static_binding(value, bindings)
                if inner and inner.kind == "model_config":
                    return inner
    if isinstance(expr, ast.Call):
        if symbol(expr.func, bindings) == MODEL_FACTORY:
            return Binding("model", model_selector(argument(expr, 0, "model")), expr)
        if isinstance(expr.func, ast.Attribute) and expr.func.attr in MODEL_WRAPPERS:
            receiver = static_binding(expr.func.value, bindings)
            if receiver and receiver.kind == "model":
                if expr.func.attr == "with_config":
                    cfg = argument(expr, 0, "config")
                    if cfg is None:
                        cfg = next((kw.value for kw in expr.keywords if kw.arg == "configurable"), None)
                    config = static_binding(cfg, bindings)
                    if config and config.kind == "model_config":
                        return Binding("model", config.value, receiver.origin)
                return receiver
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        value = truth(expr.operand, bindings)
        if value is not None:
            return Binding("constant", not value)
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        left = static_binding(expr.left, bindings)
        right = static_binding(expr.comparators[0], bindings)
        if left and right and left.kind == right.kind == "constant":
            lhs, rhs, op = left.value, right.value, expr.ops[0]
            if isinstance(op, ast.Eq):
                return Binding("constant", lhs == rhs)
            if isinstance(op, ast.NotEq):
                return Binding("constant", lhs != rhs)
            if type(lhs) is type(rhs):
                if isinstance(op, ast.Lt):
                    return Binding("constant", lhs < rhs)
                if isinstance(op, ast.LtE):
                    return Binding("constant", lhs <= rhs)
                if isinstance(op, ast.Gt):
                    return Binding("constant", lhs > rhs)
                if isinstance(op, ast.GtE):
                    return Binding("constant", lhs >= rhs)
    return None


def truth(expr, bindings):
    value = static_binding(expr, bindings)
    return bool(value.value) if value and value.kind == "constant" else None


def condition_text(expr, positive):
    if not positive and isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return ast.unparse(expr.operand)
    text = ast.unparse(expr)
    return text if positive else f"not ({text})"


class CallFlow:
    def __init__(self, source):
        self.source = source
        self.arcs = defaultdict(list)
        self.vertices = 0
        self.public = {}
        self.nodes = {}
        self.diagnostics = []
        self.topologies = {}
        self.launches = {}
        self.globals = {}
        self.functions = {}
        for stmt in source.tree.body:
            self.globals.update(import_bindings(stmt))
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[stmt.name] = stmt
                self.globals.pop(stmt.name, None)
            names = assigned_names(stmt)
            if not names:
                continue
            for name in names:
                self.functions.pop(name, None)
            value = static_binding(stmt.value, self.globals)
            if (isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute)
                    and stmt.value.func.attr == "compile" and isinstance(stmt.value.func.value, ast.Name)):
                value = Binding("subgraph", stmt.value.func.value.id, stmt.value)
            self.assign(self.globals, names, value)

    @staticmethod
    def assign(env, names, value):
        for name in names:
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value

    def vertex(self):
        self.vertices += 1
        return self.vertices

    def link(self, start, target, evidence, *, condition=None, step=None):
        arc = Arc(target, evidence, condition, step)
        if arc not in self.arcs[start]:
            self.arcs[start].append(arc)

    def advance(self, cursor, anchor, *, condition=None, step=None):
        target = self.vertex()
        self.link(cursor.position, target, self.source.evidence(anchor), condition=condition, step=step)
        cursor.position = target

    def warn(self, message, anchor):
        note = {"message": message, "evidence": self.source.evidence(anchor), "unresolved": True}
        if note not in self.diagnostics:
            self.diagnostics.append(note)

    def topology(self, builder):
        if builder not in self.topologies:
            self.topologies[builder] = langgraph(self.source, builder, with_actions=True)
        return self.topologies[builder]

    def build(self, builder, ancestors=()):
        topology, actions = self.topology(builder)
        selected = topology["scope"]
        if selected in ancestors:
            raise Unsupported(f"Recursive compiled subgraph: {selected}")
        stack = (*ancestors, selected)
        begin = {node["id"]: self.vertex() for node in topology["nodes"]}
        finish = {node_id: self.vertex() for node_id in actions}
        by_source = defaultdict(list)
        for edge in topology["edges"]:
            by_source[edge["source"]].append(edge)
        self.diagnostics.extend(note for note in topology["diagnostics"]
                                if note["unresolved"] and note not in self.diagnostics)
        entry, end = begin["__start__"], begin["__end__"]
        for node_id in ("__start__", *actions):
            origin = begin[node_id] if node_id == "__start__" else finish[node_id]
            fixed = []
            routes = defaultdict(list)
            dynamic_returns = set()
            for edge in by_source[node_id]:
                evidence = edge["evidence"][0]
                if edge["kind"] == "command":
                    for anchor in edge["evidence"]:
                        location = anchor["line"], anchor["column"]
                        routes[location].append((begin[edge["target"]], edge["target"], anchor))
                elif edge["kind"] == "unresolved":
                    dynamic_returns.add((evidence["line"], evidence["column"]))
                elif edge["kind"] in {"send", "join"}:
                    raise Unsupported("LLM flow does not yet expand LangGraph Send or declared joins")
                else:
                    fixed.append(edge)
            for edge in fixed:
                self.link(origin, begin[edge["target"]], edge["evidence"][0], condition=edge["condition"])
            if sum(edge["kind"] == "flow" for edge in fixed) > 1 or (
                fixed and routes or any(len(items) > 1 for items in routes.values())
            ):
                raise Unsupported("LLM flow does not yet expand parallel LangGraph destinations")
            if node_id == "__start__":
                continue
            action = actions[node_id]
            if not fixed and not dynamic_returns:
                self.link(origin, end, self.source.evidence(action), step="graph returns")
            if isinstance(action, ast.Lambda):
                cursor = Cursor(begin[node_id], self.globals.copy())
                self.assign(cursor.bindings, parameter_names(action.args), None)
                handler = Handler(f"lambda_{node_id}", origin, {}, set(), False, stack)
                self.expression(action.body, cursor, handler)
                self.link(cursor.position, origin, self.source.evidence(action))
                continue
            action_name = dotted(action)
            value = static_binding(action, self.globals)
            if value and value.kind == "subgraph":
                child_entry, child_end, _ = self.build(value.value, stack)
                evidence = self.source.evidence(action)
                self.link(begin[node_id], child_entry, evidence, step=f"enter {value.value}")
                self.link(child_end, origin, evidence, step=f"return from {value.value}")
            elif action_name in self.functions:
                function = self.functions[action_name]
                env = self.globals.copy()
                local_names = parameter_names(function.args)
                local_names.update(node.id for node in function_nodes(function)
                                   if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)))
                self.assign(env, local_names, None)
                handler = Handler(action_name, origin, routes, dynamic_returns, bool(fixed), stack)
                for cursor in self.block(function.body, [Cursor(begin[node_id], env)], handler):
                    self.link(cursor.position, origin, self.source.evidence(function))
            else:
                self.warn(f"Cannot inspect registered action {ast.unparse(action)}", action)
        return entry, end, topology

    def model_call(self, call, receiver, cursor, handler):
        evidence = self.source.evidence(call)
        node_id = f"{handler.name}:{evidence['line']}:{evidence['column']}"
        self.advance(cursor, call)
        self.public[cursor.position] = node_id
        node = self.nodes.setdefault(node_id, {"label": f"{handler.name} (L{evidence['line']})",
                                               "kind": "llm_call", "evidence": [evidence],
                                               "model_selectors": [], "method": call.func.attr})
        origin = self.source.evidence(receiver.origin)
        if origin not in node["evidence"]:
            node["evidence"].append(origin)
        if receiver.value is not None and receiver.value not in node["model_selectors"]:
            node["model_selectors"].append(receiver.value)

    def invoke(self, deferred, cursor, handler, wait=None):
        receiver, call = deferred.receiver, deferred.call
        if receiver.kind == "model":
            self.model_call(call, receiver, cursor, handler)
            return
        key = (handler.builders, handler.name, call, cursor.position)
        if key not in self.launches:
            entry, end, _ = self.build(receiver.value, handler.builders)
            self.launches[key] = (entry, end)
        entry, end = self.launches[key]
        continuation = self.vertex()
        evidence = self.source.evidence(call)
        mult = "parallel " if deferred.batched else ""
        self.link(cursor.position, entry, evidence, step=f"invoke {mult}{receiver.value}")
        return_anchor = self.source.evidence(wait) if wait is not None else evidence
        self.link(end, continuation, return_anchor, step=f"return from {mult}{receiver.value}")
        cursor.position = continuation

    def expression(self, expr, cursor, handler):
        if expr is None or isinstance(expr, ast.Lambda):
            return None
        if isinstance(expr, ast.Await):
            result = self.expression(expr.value, cursor, handler)
            if result and result.kind == "deferred":
                self.invoke(result.value, cursor, handler)
            elif result and result.kind == "gather":
                branches = result.value
                if len(branches) == 1:
                    self.advance(cursor, expr, step="await parallel gather")
                    self.invoke(branches[0], cursor, handler, wait=expr)
                elif branches:
                    self.warn("Gather has multiple distinct modeled call sites", expr)
            return None
        if isinstance(expr, ast.Call):
            receiver = None
            if isinstance(expr.func, ast.Attribute):
                receiver = self.expression(expr.func.value, cursor, handler)
            else:
                self.expression(expr.func, cursor, handler)
            args = [self.expression(arg.value if isinstance(arg, ast.Starred) else arg, cursor, handler)
                    for arg in expr.args]
            for kw in expr.keywords:
                self.expression(kw.value, cursor, handler)
            if symbol(expr.func, cursor.bindings) == "asyncio.gather":
                branches = tuple(value.value for value in args if value and value.kind in {"deferred", "batch"})
                return Binding("gather", branches)
            if isinstance(expr.func, ast.Attribute) and expr.func.attr in INVOKE_METHODS:
                if receiver and receiver.kind in {"model", "subgraph"}:
                    if expr.func.attr in {"stream", "astream"}:
                        self.warn("Streaming iteration requires separate flow analysis", expr)
                        return None
                    deferred = Deferred(receiver, expr)
                    if expr.func.attr in ASYNC_METHODS:
                        return Binding("deferred", deferred)
                    self.invoke(deferred, cursor, handler)
                    return None
                self.warn("Unclassified invoke receiver; left outside the confirmed LLM calls", expr)
            return static_binding(expr, cursor.bindings)
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            local = cursor.copy()
            for generator in expr.generators:
                for variable in ast.walk(generator.target):
                    if isinstance(variable, ast.Name) and isinstance(variable.ctx, ast.Store):
                        local.bindings.pop(variable.id, None)
            if isinstance(expr, ast.ListComp) and not expr.generators[0].ifs:
                element = self.expression(expr.elt, local, handler)
                if element and element.kind == "deferred":
                    call = element.value
                    return Binding("batch", Deferred(call.receiver, call.call, batched=True))
                if local.position != cursor.position:
                    raise Unsupported("Eager model calls inside comprehensions need iteration analysis")
            return None
        if isinstance(expr, (ast.BoolOp, ast.IfExp)):
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in INVOKE_METHODS
                   for n in ast.walk(expr)):
                self.warn("Model-call candidates in a short-circuit expression need branch analysis", expr)
            return static_binding(expr, cursor.bindings)
        for item in ast.iter_child_nodes(expr):
            self.expression(item, cursor, handler)
        return static_binding(expr, cursor.bindings)

    def loop(self, stmt, cursor, handler):
        if isinstance(stmt, ast.While):
            test = stmt.test
            self.expression(test, cursor, handler)
            known = truth(test, cursor.bindings)
            enter, leave = condition_text(test, True), condition_text(test, False)
        else:
            self.expression(stmt.iter, cursor, handler)
            known = None
            enter, leave = f"iterate {ast.unparse(stmt.iter)}", "loop exhausted"
        exhausted = []
        if known is not True:
            skip = cursor.copy()
            self.advance(skip, stmt, condition=leave)
            exhausted.append(skip)
        if known is False:
            return self.block(stmt.orelse, exhausted, handler)
        first = cursor.copy()
        self.advance(first, stmt, condition=enter)
        if isinstance(stmt, ast.For):
            targets = {n.id for n in ast.walk(stmt.target)
                       if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            self.assign(first.bindings, targets, None)
        breaks, continues = [], []
        normal = self.block(stmt.body, [first.copy()], handler, (breaks, continues))
        for tail in (*normal, *continues):
            if isinstance(stmt, ast.While):
                self.expression(stmt.test, tail, handler)
                known_again = truth(stmt.test, tail.bindings)
            else:
                known_again = None
            if known_again is not False:
                self.link(tail.position, first.position, self.source.evidence(stmt), condition=enter)
                old = {k: v for k, v in first.bindings.items() if v.kind != "constant"}
                new = {k: v for k, v in tail.bindings.items() if v.kind != "constant"}
                if old != new:
                    self.warn("Loop changes receiver bindings; only the first iteration was analyzed", stmt)
            if known_again is not True:
                self.advance(tail, stmt, condition=leave)
                exhausted.append(tail)
        return self.block(stmt.orelse, exhausted, handler) + breaks

    def block(self, statements, cursors, handler, loop_transfers=None):
        for stmt in statements:
            next_cursors = []
            for cursor in cursors:
                if isinstance(stmt, ast.If):
                    self.expression(stmt.test, cursor, handler)
                    known = truth(stmt.test, cursor.bindings)
                    for positive, body in ((True, stmt.body), (False, stmt.orelse)):
                        if known is None or known is positive:
                            branch = cursor.copy()
                            self.advance(branch, stmt.test, condition=condition_text(stmt.test, positive))
                            next_cursors.extend(self.block(body, [branch], handler, loop_transfers))
                elif isinstance(stmt, ast.Return):
                    self.expression(stmt.value, cursor, handler)
                    location = stmt.lineno, stmt.col_offset + 1
                    routes = handler.routes.get(location, ())
                    for target, name, evidence in routes:
                        self.link(cursor.position, target, evidence, step=f"goto {name}")
                    if location not in handler.dynamic_returns and (not routes or handler.also_follows_fixed_edges):
                        self.link(cursor.position, handler.finish, self.source.evidence(stmt))
                elif isinstance(stmt, ast.Raise):
                    self.expression(stmt.exc, cursor, handler)
                elif isinstance(stmt, (ast.For, ast.While)):
                    next_cursors.extend(self.loop(stmt, cursor, handler))
                elif isinstance(stmt, ast.Try) and not stmt.finalbody:
                    normal = self.block(stmt.body, [cursor], handler, loop_transfers)
                    next_cursors.extend(self.block(stmt.orelse, normal, handler, loop_transfers))
                elif isinstance(stmt, (ast.Break, ast.Continue)) and loop_transfers:
                    loop_transfers[isinstance(stmt, ast.Continue)].append(cursor)
                elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    value = self.expression(stmt.value, cursor, handler)
                    self.assign(cursor.bindings, assigned_names(stmt), None if isinstance(stmt, ast.AugAssign) else value)
                    next_cursors.append(cursor)
                elif isinstance(stmt, ast.Expr):
                    self.expression(stmt.value, cursor, handler)
                    next_cursors.append(cursor)
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    cursor.bindings.update(import_bindings(stmt))
                    next_cursors.append(cursor)
                elif isinstance(stmt, ast.Pass):
                    next_cursors.append(cursor)
                else:
                    self.warn(f"Unsupported statement: {type(stmt).__name__}", stmt)
            cursors = self.merge(next_cursors, stmt)
        return cursors

    def merge(self, cursors, anchor):
        grouped = defaultdict(list)
        for cursor in cursors:
            grouped[frozenset(cursor.bindings.items())].append(cursor)
        merged = []
        for same in grouped.values():
            if len(same) == 1:
                merged.extend(same)
                continue
            position = self.vertex()
            evidence = self.source.evidence(anchor)
            for cursor in same:
                self.link(cursor.position, position, evidence)
            merged.append(Cursor(position, same[0].bindings))
        return merged

    def reachable(self, entry):
        visited, pending = {entry}, [entry]
        while pending:
            for arc in self.arcs[pending.pop()]:
                if arc.target not in visited:
                    visited.add(arc.target)
                    pending.append(arc.target)
        return visited

    def following_calls(self, origin):
        """Keep a shortest witness per next event and final explicit graph jump."""
        queue = deque([(origin, None, ())])
        seen = {(origin, None)}
        matches = {}
        while queue:
            vertex, last_jump, path = queue.popleft()
            for arc in self.arcs[vertex]:
                jump = last_jump
                if arc.step and arc.step.startswith("goto "):
                    jump = (arc.evidence["line"], arc.evidence["column"], arc.step)
                key = (arc.target, jump)
                if arc.target in self.public:
                    matches.setdefault(key, (*path, arc))
                elif key not in seen:
                    seen.add(key)
                    queue.append((arc.target, jump, (*path, arc)))
        return matches

    def extract(self, builder):
        if builder and ":" in builder:
            raise Unsupported("LLM flow currently takes a module-level StateGraph builder")
        entry, end, topology = self.build(builder)
        boundaries = {n["id"]: n["evidence"] for n in topology["nodes"] if n["kind"] in {"entry", "terminal"}}
        return self.project(
            entry, end, topology["scope"], boundaries, extractor="llm-flow",
            analysis_scope=(
                "Static normal paths in one Python module; same-module compiled subgraphs; "
                "a homogeneous awaited parallel batch represented as one worker path. "
                "Exception handlers, provider retries, and model calls hidden inside arbitrary helpers are outside scope."
            ),
        )

    def project(self, entry, end, scope, boundaries, *, extractor, analysis_scope):
        self.public.update({entry: "__start__", end: "__end__"})
        reachable = self.reachable(entry)
        result = Graph(extractor, self.source, "__start__", scope, "llm_call_successors")
        result.node("__start__", "entry", "Start", *boundaries["__start__"])
        for node_id, node in sorted(self.nodes.items(), key=lambda item: item[1]["evidence"][0]["line"]):
            if any(public_id == node_id and vertex in reachable for vertex, public_id in self.public.items()):
                result.node(node_id, node["kind"], node["label"], *node["evidence"])
                for field in ("model_selectors", "method"):
                    result.data["nodes"][-1][field] = node[field]
        result.node("__end__", "terminal", "End", *boundaries["__end__"])
        for position, start in self.public.items():
            if position not in reachable or position == end:
                continue
            for (target, _), path in self.following_calls(position).items():
                end_id = self.public[target]
                conditions = list(dict.fromkeys(a.condition for a in path if a.condition))
                steps = list(dict.fromkeys(a.step for a in path if a.step))
                anchors = []
                for arc in path:
                    if arc.evidence not in anchors:
                        anchors.append(arc.evidence)
                result.edge(start, end_id, "next", anchors[0], kind="next_call",
                            condition=" and ".join(f"({term})" for term in conditions) or None,
                            additional_evidence=anchors[1:])
                emitted = result.edge_index[(start, end_id, "next", "next_call")]
                witness = {"conditions": conditions, "steps": steps, "evidence": anchors}
                if witness not in emitted.setdefault("witnesses", []):
                    emitted["witnesses"].append(witness)
        for edge in result.data["edges"]:
            witnesses = edge.get("witnesses", [])
            if not any(step.startswith("goto ") for witness in witnesses for step in witness["steps"]):
                continue
            alternatives = list(dict.fromkeys(
                " and ".join(f"({term})" for term in witness["conditions"])
                for witness in witnesses
            ))
            if len(alternatives) > 1:
                edge["condition"] = " or ".join(f"({guard})" for guard in alternatives) if all(alternatives) else None
        result.data["diagnostics"] = self.diagnostics
        result.data["analysis_scope"] = analysis_scope
        return result.finish()


def extract_llm_flow(source, builder):
    return CallFlow(source).extract(builder)


def write_dot(path, graph):
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
        style = " [style=dashed]" if edge["condition"] else ""
        lines.append(f"  {q(edge['source'])} -> {q(edge['target'])}{style};")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
