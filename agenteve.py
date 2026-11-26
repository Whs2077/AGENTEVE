"""Extract source-backed harness flow graphs and compare them with reference annotations."""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tokenize


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
        return complete_graph(self.data)


def complete_graph(graph):
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


def dotted(expr):
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return f"{dotted(expr.value)}.{expr.attr}"
    return ""


def string(expr, default=None):
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return default


def argument(call, index, keyword):
    if len(call.args) > index:
        return call.args[index]
    return next((kw.value for kw in call.keywords if kw.arg == keyword), None)


def builder_calls(expr, constructors):
    """Return calls in a single fluent construction chain in execution order."""
    calls = []
    while isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        if dotted(expr.func) in constructors:
            break
        calls.append(expr)
        expr = expr.func.value
    return expr, list(reversed(calls))


def literal_destinations(node, literal_names, resolve):
    if isinstance(node, ast.Subscript) and dotted(node.value) in literal_names:
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return [resolve(item) for item in items]
    return []


def direct_return_conditions(function, resolve):
    """Read straight-line guard/return routers without interpreting expressions."""
    conditions = {}
    preceding = []
    for stmt in function.body:
        if isinstance(stmt, ast.If) and not stmt.orelse and len(stmt.body) == 1:
            terminal = stmt.body[0]
            condition = ast.unparse(stmt.test)
            if isinstance(terminal, ast.Return) and resolve(terminal.value) is not None:
                key = resolve(terminal.value)
                if key in conditions:
                    return {}
                conditions[key] = (" and ".join([*preceding, condition]), terminal)
            elif not isinstance(terminal, ast.Raise):
                return {}
            preceding.append(f"not ({condition})")
        elif isinstance(stmt, ast.Return) and resolve(stmt.value) is not None:
            key = resolve(stmt.value)
            if key not in conditions:
                conditions[key] = (" and ".join(preceding) or None, stmt)
            return conditions
        elif not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Expr)):
            return {}
    return conditions


def function_nodes(function):
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            pending.extend(reversed(list(ast.iter_child_nodes(node))))


def static_routes(function, resolve):
    """Find direct return destinations and leave unsupported paths unresolved."""
    params = (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs,
              function.args.vararg, function.args.kwarg)
    shadowed = {parameter.arg for parameter in params if parameter is not None}
    shadowed.update(node.id for node in function_nodes(function)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)))

    def value_routes(expr, guards, anchor):
        if isinstance(expr, ast.IfExp):
            guard = ast.unparse(expr.test)
            return (value_routes(expr.body, (*guards, guard), expr.body)
                    + value_routes(expr.orelse, (*guards, f"not ({guard})"), expr.orelse))
        if isinstance(expr, (ast.List, ast.Tuple)):
            return [route for element in expr.elts for route in value_routes(element, guards, element)]
        name = dotted(expr)
        destination = None if name.split(".", 1)[0] in shadowed else resolve(expr)
        return [(destination, guards, anchor)]

    def block(statements, guards):
        routes = []
        for stmt in statements:
            if isinstance(stmt, ast.Return):
                return routes + value_routes(stmt.value, guards, stmt), False
            if isinstance(stmt, ast.Raise):
                return routes, False
            if isinstance(stmt, ast.If):
                condition = ast.unparse(stmt.test)
                positive = (*guards, condition)
                negative = (*guards, f"not ({condition})")
                yes_routes, yes_falls = block(stmt.body, positive)
                no_routes, no_falls = block(stmt.orelse, negative)
                routes.extend(yes_routes + no_routes)
                if not yes_falls and not no_falls:
                    return routes, False
                if yes_falls != no_falls:
                    guards = positive if yes_falls else negative
                continue
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Pass)):
                routes.append((None, guards, stmt))
                return routes, False
        return routes, True

    routes, falls_through = block(function.body, ())
    if falls_through:
        routes.append((None, (), function))
    return routes


def langgraph(source, selected, *, with_actions=False):
    # Module-level builder declarations. Imports are read as syntax, never executed.
    constructors = set()
    literal_names = set()
    commands = set()
    sends = set()
    specials = {}
    for stmt in source.tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module in {"langgraph.graph", "langgraph.types",
                                                               "typing", "typing_extensions"}:
            for alias in stmt.names:
                name = alias.asname or alias.name
                if stmt.module == "langgraph.graph" and alias.name == "StateGraph":
                    constructors.add(name)
                if stmt.module == "langgraph.graph" and alias.name in {"START", "END"}:
                    specials[name] = f"__{alias.name.lower()}__"
                if stmt.module in {"typing", "typing_extensions"} and alias.name == "Literal":
                    literal_names.add(name)
                if stmt.module == "langgraph.types" and alias.name == "Command":
                    commands.add(name)
                if stmt.module == "langgraph.types" and alias.name == "Send":
                    sends.add(name)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "langgraph.graph":
                    prefix = alias.asname or alias.name
                    constructors.add(f"{prefix}.StateGraph")
                    specials.update({f"{prefix}.START": "__start__", f"{prefix}.END": "__end__"})
                if alias.name in {"typing", "typing_extensions"}:
                    literal_names.add(f"{alias.asname or alias.name}.Literal")
                if alias.name == "langgraph.types":
                    prefix = alias.asname or alias.name
                    commands.add(f"{prefix}.Command")
                    sends.add(f"{prefix}.Send")

    def resolve(expr):
        return string(expr, specials.get(dotted(expr)))

    functions = {s.name: s for s in source.tree.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
    body = source.tree.body
    scope = ""
    if selected and ":" in selected:
        scope, _, selected = selected.rpartition(":")
        if scope not in functions or not selected:
            raise Unsupported("Choose a top-level factory and builder via --builder FACTORY:BUILDER")
        body = functions[scope].body
        functions.update({s.name: s for s in body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))})

    builders = {}
    for stmt in body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            root, calls = builder_calls(stmt.value, constructors)
            if isinstance(root, ast.Call) and dotted(root.func) in constructors:
                builders[stmt.targets[0].id] = (root, calls)
            elif dotted(root) in builders and calls:
                builders[dotted(root)][1].extend(calls)
        elif isinstance(stmt, ast.Expr):
            root, calls = builder_calls(stmt.value, constructors)
            if dotted(root) in builders:
                builders[dotted(root)][1].extend(calls)
    if selected is None and len(builders) == 1:
        selected = next(iter(builders))
    if selected not in builders:
        raise Unsupported(f"Choose a StateGraph via --builder BUILDER or FACTORY:BUILDER. "
                          f"Found in selected scope: {', '.join(builders) or 'none'}")

    root, calls = builders[selected]
    compiled_at = next((i for i, call in enumerate(calls) if call.func.attr == "compile"), len(calls))
    calls = calls[:compiled_at]
    graph = Graph("langgraph", source, "__start__", f"{scope}:{selected}" if scope else selected, "declared_topology")
    graph.node("__start__", "entry", "Start", source.evidence(root))
    graph.node("__end__", "terminal", "End", source.evidence(root))
    definitions = {}
    actions = {}
    for call in calls:
        if call.func.attr != "add_node":
            continue
        arg = argument(call, 0, "node")
        action = argument(call, 1, "action")
        arg = arg if arg is not None else action
        node_id = string(arg) or (arg.id if isinstance(arg, ast.Name) and arg.id in functions else None)
        if node_id is None:
            graph.warn("Node name could not be resolved", source.evidence(call), unresolved=True)
            continue
        action = action or arg
        actions[node_id] = action
        definition = functions.get(dotted(action))
        evidence = [source.evidence(call)]
        if definition:
            evidence.append(source.evidence(definition))
        graph.node(node_id, "step", node_id, *evidence)
        if argument(call, 2, "destinations") is not None:
            graph.warn(f"{node_id}: destinations metadata does not schedule execution", source.evidence(call))
        if definition:
            definitions[node_id] = definition

    def add(start, end, route, evidence, kind="flow", condition=None, additional_evidence=()):
        if start not in graph.node_ids:
            graph.warn("Edge source could not be resolved or is undeclared", evidence, unresolved=True)
        elif end not in graph.node_ids:
            graph.unknown(start, route, "Edge destination could not be resolved or is undeclared", evidence)
        else:
            graph.edge(start, end, route, evidence, kind=kind, condition=condition,
                       additional_evidence=additional_evidence)

    for node_id, definition in definitions.items():
        body_nodes = list(function_nodes(definition))
        directly_returned = set()
        for expr in body_nodes:
            if not isinstance(expr, ast.Return) or not isinstance(expr.value, ast.Call):
                continue
            call = expr.value
            if dotted(call.func) not in commands:
                continue
            directly_returned.add(call)
            target = next((kw.value for kw in call.keywords if kw.arg == "goto"), None)
            parent = any(kw.arg == "graph" for kw in call.keywords)
            if target is None and not any(kw.arg is None for kw in call.keywords):
                continue
            targets = target.elts if isinstance(target, (ast.List, ast.Tuple)) else [target]
            for value in targets:
                if isinstance(value, ast.Call) and dotted(value.func) in sends:
                    value = argument(value, 0, "node")
                    kind = "send"
                else:
                    kind = "command"
                end = None if parent else resolve(value)
                add(node_id, end, end or "runtime", source.evidence(expr), kind)
        for expr in body_nodes:
            if (isinstance(expr, ast.Call) and expr not in directly_returned and dotted(expr.func) in commands
                    and any(kw.arg in {"goto", None} for kw in expr.keywords)):
                graph.unknown(node_id, "runtime", "Indirect Command routing needs additional evidence",
                              source.evidence(expr))

    for call in calls:
        method = call.func.attr
        evidence = source.evidence(call)
        first = argument(call, 0, "key")
        if method == "set_entry_point":
            add("__start__", resolve(first), "next", evidence)
        elif method == "set_finish_point":
            add(resolve(first), "__end__", "next", evidence)
        elif method == "add_edge":
            start = argument(call, 0, "start_key")
            end = resolve(argument(call, 1, "end_key"))
            if isinstance(start, (ast.List, ast.Tuple)):
                upstream = [resolve(item) for item in start.elts]
                if upstream and all(item in graph.node_ids for item in upstream) and end in graph.node_ids:
                    join = f"<join:{end}:{'|'.join(upstream)}>"
                    if join not in graph.node_ids:
                        graph.node(join, "join", "Wait for all predecessors", evidence)
                        for predecessor in upstream:
                            add(predecessor, join, "join", evidence, "join")
                        add(join, end, "all", evidence, "join")
                else:
                    graph.warn("Join has unresolved predecessors or destination", evidence, unresolved=True)
            else:
                add(resolve(start), end, "next", evidence)
        elif method in {"add_conditional_edges", "set_conditional_entry_point"}:
            entry = method == "set_conditional_entry_point"
            start = "__start__" if entry else resolve(argument(call, 0, "source"))
            router = argument(call, 0 if entry else 1, "path")
            path_map = argument(call, 1 if entry else 2, "path_map")
            choices = []
            route_evidence = evidence
            fn = functions.get(dotted(router))
            conditions = direct_return_conditions(fn, resolve) if fn else {}
            if isinstance(path_map, ast.Dict):
                for key, value in zip(path_map.keys, path_map.values):
                    if isinstance(key, ast.Constant) or resolve(key) is not None:
                        route = resolve(key) if resolve(key) is not None else str(key.value)
                        choices.append((route, resolve(value)))
                    else:
                        choices.append(("dynamic", None))
            elif isinstance(path_map, ast.List):
                choices = [(resolve(item) or "dynamic", resolve(item)) for item in path_map.elts]
            elif path_map is None or isinstance(path_map, ast.Constant) and path_map.value is None:
                if fn:
                    destinations = literal_destinations(fn.returns, literal_names, resolve)
                    choices = [(d or "dynamic", d) for d in destinations]
                    route_evidence = source.evidence(fn)
                    if not choices:
                        resolved = defaultdict(list)
                        for destination, guards, anchor in static_routes(fn, resolve):
                            if destination is None:
                                add(start, None, "dynamic", source.evidence(anchor), "conditional")
                            else:
                                resolved[destination].append((guards, source.evidence(anchor)))
                        for destination, paths in resolved.items():
                            predicates = list(dict.fromkeys(" and ".join(f"({g})" for g in guards)
                                                            for guards, _ in paths))
                            condition = " or ".join(f"({p})" for p in predicates) if all(predicates) else None
                            add(start, destination, destination, evidence, "conditional",
                                condition, [anchor for _, anchor in paths])
                        continue
            if choices:
                for route, end in choices:
                    extra = [evidence] if route_evidence != evidence else []
                    condition = f"{dotted(router) or 'router'} → {route}"
                    if route in conditions:
                        predicate, return_stmt = conditions[route]
                        condition = predicate or condition
                        extra.append(source.evidence(return_stmt))
                    add(start, end, route, route_evidence, "conditional",
                        condition, extra)
            else:
                add(start, None, "dynamic", evidence, "conditional")
    graph.warn("Scope is the declared graph builder. Node internals, subgraphs and runtime plugins are not expanded.",
               source.evidence(root))
    topology = graph.finish()
    return (topology, actions) if with_actions else topology


def is_call(expr, name):
    return isinstance(expr, ast.Call) and dotted(expr.func) == name


def has_call(node, name):
    return any(is_call(expr, name) for expr in ast.walk(node))


def require(condition, description):
    if not condition:
        raise Unsupported(f"mini-swe-agent profile cannot verify {description}")


def one(items, description):
    require(len(items) == 1, description)
    return items[0]


def is_exit_message(node):
    return isinstance(node, ast.Dict) and any(
        string(key) == "role" and string(value) == "exit"
        for key, value in zip(node.keys, node.values)
    )


def inheritance_path(classes, name, ancestor, seen=frozenset()):
    if name not in classes or name in seen:
        return []
    cls = classes[name]
    if name == ancestor:
        return [cls]
    for base in cls.bases:
        path = inheritance_path(classes, dotted(base), ancestor, seen | {name})
        if path:
            return [cls, *path]
    return []


def mini(source, exceptions):
    """Recognize the control idioms in DefaultAgent from mini-SWE-agent v2.4.6."""
    cls = one([s for s in source.tree.body if isinstance(s, ast.ClassDef) and s.name == "DefaultAgent"],
              "DefaultAgent")
    methods = {s.name: s for s in cls.body if isinstance(s, ast.FunctionDef)}
    for name in ("run", "step", "query", "execute_actions", "save"):
        require(name in methods, f"method {name}")
    run, step, query, actions = (methods[name] for name in ("run", "step", "query", "execute_actions"))
    loop = one([s for s in run.body if isinstance(s, ast.While)], "run loop")
    require(isinstance(loop.test, ast.Constant) and loop.test.value is True, "unbounded loop header")
    require(not loop.orelse, "absence of a loop else clause")
    handler, exit_check = (loop.body if len(loop.body) == 2 else (None, None))
    require(isinstance(handler, ast.Try) and not handler.orelse, "try/except/finally structure")
    require(isinstance(exit_check, ast.If) and len(exit_check.body) == 1
            and isinstance(exit_check.body[0], ast.Break) and not exit_check.orelse
            and ast.unparse(exit_check.test) == "self.messages[-1].get('role') == 'exit'", "exit break")
    require(isinstance(run.body[-1], ast.Return) and run.body[-2] is loop, "return after break")
    init = one([s for s in run.body if isinstance(s, ast.Expr) and is_call(s.value, "self.add_messages")],
               "initial message construction")
    require(has_call(init, "self._render_template") and init.lineno < loop.lineno, "initial templates")
    require(len(handler.body) == 2 and isinstance(handler.body[0], ast.Expr)
            and is_call(handler.body[0].value, "self.step")
            and isinstance(handler.body[1], ast.Assign)
            and ast.unparse(handler.body[1]) == "self.n_consecutive_format_errors = 0", "step and counter reset")
    reset = handler.body[1]
    require([dotted(h.type) for h in handler.handlers] == ["FormatError", "InterruptAgentFlow", "Exception"],
            "exception handler order")
    format_handler, interrupt, uncaught = handler.handlers
    format_guard = one([s for s in format_handler.body if isinstance(s, ast.If)], "format error threshold")
    require(format_handler.body[-1] is format_guard and len(format_guard.body) == len(format_guard.orelse) == 1,
            "format branches")
    format_stop = format_guard.body[0]
    require(isinstance(format_stop, ast.Expr) and is_call(format_stop.value, "self.add_messages")
            and bool(format_stop.value.args) and is_exit_message(format_stop.value.args[-1])
            and has_call(format_guard.orelse[0], "self.add_messages"), "format recovery and exit")
    require(len(interrupt.body) == 1 and has_call(interrupt.body[0], "self.add_messages"), "interrupt message")
    require(len(uncaught.body) == 2 and has_call(uncaught.body[0], "self.handle_uncaught_exception")
            and isinstance(uncaught.body[-1], ast.Raise) and uncaught.body[-1].exc is None, "uncaught re-raise")
    save = one(handler.finalbody, "finally body")
    require(isinstance(save, ast.Expr) and is_call(save.value, "self.save"), "finally trajectory save")
    step_return = step.body[-1]
    require(isinstance(step_return, ast.Return) and is_call(step_return.value, "self.execute_actions")
            and len(step_return.value.args) == 1 and is_call(step_return.value.args[0], "self.query"),
            "query before action execution")
    guards = [s for s in query.body if isinstance(s, ast.If)]
    require(len(guards) == 2, "query limit checks")
    with_budget, with_time = guards
    exception_classes = {s.name: s for s in exceptions.tree.body if isinstance(s, ast.ClassDef)}
    handler_sources = []
    for guard, name in zip(guards, ("LimitsExceeded", "TimeExceeded")):
        limit_raise = one(guard.body, f"{name} branch")
        require(isinstance(limit_raise, ast.Raise) and is_call(limit_raise.exc, name)
                and not guard.orelse, name)
        payload = argument(limit_raise.exc, 0, "")
        require(is_exit_message(payload), f"{name} exit payload")
        path = inheritance_path(exception_classes, name, "InterruptAgentFlow")
        require(bool(path), f"{name} inheritance")
        handler_sources.append([exceptions.evidence(cls) for cls in path])
    model = one([s for s in query.body if has_call(s, "self.model.query")], "model call")
    require(query.body.index(with_budget) < query.body.index(with_time) < query.body.index(model)
            and isinstance(query.body[-1], ast.Return) and has_call(query, "self.add_messages"),
            "query order and message recording")
    batch = one([s for s in actions.body if has_call(s, "self.env.execute")], "tool execution")
    observe = actions.body[-1]
    require(isinstance(batch, ast.Assign) and isinstance(batch.value, ast.ListComp)
            and isinstance(observe, ast.Return) and has_call(observe, "self.model.format_observation_messages")
            and has_call(observe, "self.add_messages"), "batch execution and observation append")

    graph = Graph("mini", source, "entry", "DefaultAgent.run → step/query/execute_actions",
                  "projected_control_flow")
    graph.data["sources"].append(exceptions.metadata())
    for node_id, kind, label, anchors in [
        ("entry", "entry", "Task input", [run]),
        ("initialize", "step", "Initialize messages", [init]),
        ("budget", "branch", "Step or cost limit?", [with_budget]),
        ("time", "branch", "Wall-time limit?", [with_time]),
        ("model", "model", "Query model and record response", [model]),
        ("tools", "tool", "Run zero or more actions", [batch]),
        ("observe", "step", "Append observations", [observe]),
        ("reset", "step", "Reset format-error counter", [reset]),
        ("format", "branch", "Count format error and check limit", [format_handler]),
        ("format_retry", "step", "Append format feedback", [format_guard.orelse[0]]),
        ("format_stop", "step", "Append repeated-format exit", [format_guard.body[0]]),
        ("interrupt", "step", "Append interrupt messages", [interrupt]),
        ("interrupt_exit", "step", "Append known exit message", [interrupt, with_budget, with_time]),
        ("uncaught", "step", "Record unexpected exception", [uncaught]),
        ("save", "step", "Save trajectory", [save]),
        ("exit_check", "branch", "Last message is exit?", [exit_check]),
        ("save_exit", "step", "Save after a known exit", [save]),
        ("exit_check_known", "branch", "Check recorded exit", [exit_check]),
        ("return", "terminal", "Return last message metadata", [run.body[-1]]),
        ("save_error", "step", "Save while exception propagates", [save]),
        ("raise", "terminal", "Propagate unexpected exception", [uncaught.body[-1]]),
    ]:
        graph.node(node_id, kind, label, *(source.evidence(anchor) for anchor in anchors))

    def connect(start, end, anchor, route="next", kind="flow", condition=None, additional_evidence=()):
        graph.edge(start, end, route, source.evidence(anchor), kind=kind, condition=condition,
                   additional_evidence=additional_evidence)

    connect("entry", "initialize", init)
    connect("initialize", "budget", loop)
    connect("budget", "interrupt_exit", with_budget, "limit", "exception", ast.unparse(with_budget.test),
            handler_sources[0])
    connect("budget", "time", with_budget, "within_budget", "conditional", "Step/cost limit not reached")
    connect("time", "interrupt_exit", with_time, "timeout", "exception", ast.unparse(with_time.test),
            handler_sources[1])
    connect("time", "model", with_time, "within_time", "conditional", "Wall-time limit not reached")
    connect("model", "tools", step_return)
    connect("tools", "observe", observe)
    connect("observe", "reset", reset)
    connect("reset", "save", save)
    for start in ("model", "tools", "observe"):
        connect(start, "format", format_handler, "format_error", "possible_exception", "FormatError")
        connect(start, "interrupt", interrupt, "interrupt", "possible_exception", "InterruptAgentFlow")
        connect(start, "uncaught", uncaught, "unexpected", "possible_exception", "Other Exception")
    connect("format", "format_retry", format_guard, "retry", "conditional", "Format-error limit not reached")
    connect("format", "format_stop", format_guard, "limit", "conditional", ast.unparse(format_guard.test))
    connect("format_retry", "save", save)
    connect("format_stop", "save_exit", save)
    connect("interrupt", "save", save)
    connect("interrupt_exit", "save_exit", save)
    connect("uncaught", "save_error", save, "finally", "exception")
    connect("save_error", "raise", uncaught.body[-1], "reraise", "exception")
    connect("save", "exit_check", exit_check)
    connect("save_exit", "exit_check_known", exit_check)
    connect("exit_check_known", "return", exit_check, "exit", "conditional", ast.unparse(exit_check.test))
    connect("exit_check", "return", exit_check, "exit", "conditional", ast.unparse(exit_check.test))
    connect("exit_check", "budget", exit_check, "continue", "loop", "Last message is not exit")
    graph.warn("Possible exception edges reflect handler routing, not proof that each component raises each type.",
               source.evidence(handler))
    graph.warn("The save and exit check are shown separately for known exit messages to keep those paths out of the loop.",
               source.evidence(exit_check))
    graph.warn("Provider/environment internals, CLI overrides, and secondary failures in initialization or save are outside scope.",
               source.evidence(run))
    return graph.finish()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compiled_graph(path, source_url):
    payload = read_json(path)
    source = {"file": path.name, "url": source_url, "coordinates": "graph_export",
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    nodes = []
    for i, item in enumerate(payload["nodes"]):
        data = item.get("data")
        label = data.get("name", item["id"]) if isinstance(data, dict) else item["id"]
        nodes.append({"id": item["id"], "label": str(label),
                      "kind": {"__start__": "entry", "__end__": "terminal"}.get(item["id"], "step"),
                      "evidence": [{"file": path.name, "pointer": f"/nodes/{i}"}]})
    edges = [{"source": item["source"], "target": item["target"],
              "kind": "conditional" if item.get("conditional") else "flow",
              "route": str(item["data"]) if item.get("data") is not None else "next",
              "condition": None, "evidence": [{"file": path.name, "pointer": f"/edges/{i}"}]}
             for i, item in enumerate(payload["edges"])]
    return complete_graph({"schema_version": "1.0", "extractor": "langgraph-compiled", "entry": "__start__",
                           "scope": "Compiled LangGraph export", "graph_kind": "compiled_topology",
                           "sources": [source], "nodes": nodes, "edges": edges,
                           "diagnostics": [{"message": "Compiled display topology can include conditional or inferred edges. "
                                                       "Tools can invoke subgraphs outside this visualization.",
                                            "evidence": {"file": path.name}, "unresolved": False}]})


def write_graph(directory, graph):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "flow.json", graph)
    (directory / "flow.mmd").write_text(mermaid(graph), encoding="utf-8")
    print(f"{graph['extractor']}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges → {directory}")


def metric(actual, expected):
    hit = len(actual & expected)
    precision = hit / len(actual) if actual else None
    recall = hit / len(expected) if expected else None
    return {"matched": hit, "predicted": len(actual), "reference": len(expected),
            "precision": precision, "recall": recall,
            "f1": 2 * hit / (len(actual) + len(expected)) if actual or expected else None,
            "missing": sorted(expected - actual, key=str), "extra": sorted(actual - expected, key=str)}


def score(actual, expected):
    nodes = {(n["id"], n["kind"]) for n in actual["nodes"]}
    edges = {(e["source"], e["target"], e["route"], e["kind"]) for e in actual["edges"]}
    result = {"nodes": metric(nodes, {tuple(n) for n in expected["nodes"]}),
              "edges": metric(edges, {tuple(e) for e in expected["edges"]}),
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
        result["call_sites"] = metric(observed_calls, annotated_calls)
    return result


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


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="read source and write flow artifacts")
    extract.add_argument("adapter", choices=["mini", "langgraph", "llm-flow", "stirrup-llm"])
    extract.add_argument("source", type=Path)
    extract.add_argument("--source-url")
    extract.add_argument("--coordinates", choices=["file", "browser-text"], default="file")
    extract.add_argument("--exceptions", type=Path)
    extract.add_argument("--exceptions-url")
    extract.add_argument("--builder")
    extract.add_argument("--client-protocol", type=Path, help="Stirrup's matching core/models.py")
    extract.add_argument("--client-protocol-url")
    extract.add_argument("--out", type=Path, required=True)
    import_graph = commands.add_parser("import-graph", help="normalize a compiled LangGraph or AGENTEVE JSON graph")
    import_graph.add_argument("kind", choices=["compiled", "agenteve"])
    import_graph.add_argument("source", type=Path)
    import_graph.add_argument("--source-url")
    import_graph.add_argument("--out", type=Path, required=True)
    compare = commands.add_parser("score", help="compare graph with annotated reference")
    compare.add_argument("graph", type=Path)
    compare.add_argument("reference", type=Path)
    compare.add_argument("--out", type=Path)
    reachability = commands.add_parser("reachability", help="analyze each LLM node and pair recorded completions")
    reachability.add_argument("graph", type=Path)
    reachability.add_argument("--runs", type=Path, help="Saved JSONL runs with per-node completion")
    reachability.add_argument("--trace-format", choices=["node-completion", "odr-pilot"], default="node-completion")
    reachability.add_argument("--out", required=True, type=Path, help="New analysis output directory")
    args = cli.parse_args()
    try:
        if args.command == "reachability":
            from reachability import run_analysis

            return run_analysis(args)

        if args.command == "score":
            result = score(read_json(args.graph), read_json(args.reference))
            if args.out:
                write_json(args.out, result)
            else:
                print(json.dumps(result, ensure_ascii=True, indent=2))
            checks = [result["nodes"], result["edges"]]
            if "call_sites" in result:
                checks.append(result["call_sites"])
            return int(any(m["extra"] or m["missing"] for m in checks)
                       or not result["entry_matches"] or not result["cycle_nodes_match"])

        if args.command == "import-graph":
            graph = (compiled_graph(args.source, args.source_url) if args.kind == "compiled"
                     else complete_graph(read_json(args.source)))
            write_graph(args.out, graph)
            return 0

        source = Source(args.source, args.source_url, args.coordinates)
        if args.adapter == "mini":
            if args.exceptions is None:
                cli.error("mini requires --exceptions, the matching minisweagent/exceptions.py")
            graph = mini(source, Source(args.exceptions, args.exceptions_url, args.coordinates))
        elif args.adapter in {"llm-flow", "stirrup-llm"}:
            from llm_flow import extract_llm_flow, write_dot

            if args.adapter == "stirrup-llm":
                from stirrup_flow import extract_stirrup

                if args.client_protocol is None:
                    cli.error("stirrup-llm requires --client-protocol, the matching core/models.py")
                protocol = Source(args.client_protocol, args.client_protocol_url, args.coordinates)
                graph = extract_stirrup(source, protocol)
            else:
                graph = extract_llm_flow(source, args.builder)
            write_graph(args.out, graph)
            write_dot(args.out / "flow.dot", graph)
            return 0
        else:
            graph = langgraph(source, args.builder)
        write_graph(args.out, graph)
        return 0
    except (ValueError, SyntaxError, OSError, KeyError, TypeError) as error:
        print(f"agenteve: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
