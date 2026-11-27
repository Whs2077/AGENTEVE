"""Read a StateGraph builder into a declared topology."""

import ast
from collections import defaultdict

from src.core.graphModel import Graph, Unsupported
from src.core.pyScan import dotted


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


def readDeclaredTopology(source, selected, *, with_actions=False):
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

