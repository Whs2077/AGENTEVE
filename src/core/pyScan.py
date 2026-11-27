"""Python AST queries shared by the locator, the call hierarchy, and the projectors."""

import ast

FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def dotted(expr) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return f"{dotted(expr.value)}.{expr.attr}"
    return ""


def call_name(node) -> str:
    return dotted(node.func) if isinstance(node, ast.Call) else ""


def parents(tree) -> dict:
    found = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            found[child] = node
    return found


def enclosing(node, parent_of) -> str:
    """Dotted class and function names around ``node``, or ``<module>``."""
    current = parent_of.get(node)
    parts = []
    while current is not None:
        if isinstance(current, (*FUNCTION_TYPES, ast.ClassDef)):
            parts.append(current.name)
        current = parent_of.get(current)
    return ".".join(reversed(parts)) or "<module>"


def function_symbol(function, parent_of) -> str:
    owner = enclosing(function, parent_of)
    return function.name if owner == "<module>" else f"{owner}.{function.name}"


def symbol_covering(tree, parent_of, line: int) -> str:
    """The innermost function whose span covers ``line``."""
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, FUNCTION_TYPES):
            continue
        if node.lineno <= line <= (node.end_lineno or node.lineno) and (best is None or node.lineno >= best.lineno):
            best = node
    return "<module>" if best is None else function_symbol(best, parent_of)


def imports_prefix(tree, prefix: str) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefix):
            return True
        if isinstance(node, ast.Import) and any(alias.name.startswith(prefix) for alias in node.names):
            return True
    return False


def imported_aliases(tree, original: str) -> set:
    return {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
            for alias in node.names if alias.name == original}


def named_arguments(function) -> list:
    """Positional-only, ordinary, and keyword-only ``ast.arg`` nodes, in that order."""
    arguments = function.args
    return [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
