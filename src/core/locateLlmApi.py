"""Find large-language-model API sites before any call-graph node is created.

Bucket 1 reads fixed framework calls. Bucket 2 reads fixed protocol calls in
files bucket 1 did not claim. Bucket 3 asks a model only when those buckets
find nothing or disagree, and it accepts only a file, a line, and a symbol.
"""

import ast
from pathlib import Path

from src.core.apiSite import (
    INVOKE_METHODS, MODEL_WRAPPERS, ApiBucket, ApiProtocol, iter_source_files, make_site,
)
from src.core.graphModel import Unsupported
from src.core.locateByModel import choose_among, locate_by_model
from src.core.pyScan import (
    FUNCTION_TYPES, call_name, dotted, enclosing, function_symbol, imported_aliases, imports_prefix,
    named_arguments, parents,
)
from src.core.tsScan import typescript_calls

_FAMILIES = {
    ApiProtocol.LANGCHAIN_INVOKE.value: "langchain",
    ApiProtocol.LANGCHAIN_FACTORY.value: "langchain",
    ApiProtocol.DEEPAGENT_CREATE.value: "deepagent",
    ApiProtocol.AI_SDK_STREAM.value: "ai-sdk",
    ApiProtocol.CLIENT_GENERATE.value: "python-client",
    ApiProtocol.OPENAI_CHAT.value: "python-client",
}


def locate(path, ask=None):
    """Return API sites under ``path``. ``ask`` receives a prompt and returns a location object."""
    root = Path(path)
    sites = []
    for file in iter_source_files(root):
        sites.extend(_python_sites(file) if file.suffix.lower() == ".py" else _typescript_sites(file))
    families = {_FAMILIES.get(site.protocol, site.protocol) for site in sites}
    if len(families) > 1:
        if ask is None:
            raise Unsupported("LLM API buckets disagree: " + ", ".join(f"{site.file}:{site.line} {site.symbol}"
                                                                        for site in sites))
        return choose_among(ask, sites)
    if sites:
        return sites
    if ask is None:
        raise Unsupported(f"No LLM API site in {root}")
    return [locate_by_model(ask, root)]


def _python_sites(file):
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    parent_of = parents(tree)
    framework = []
    if imports_prefix(tree, "langchain"):
        framework.extend(_langchain_sites(file, tree, parent_of))
    if imports_prefix(tree, "deepagents"):
        framework.extend(_named_calls(file, tree, parent_of, {"create_deep_agent"}, ApiProtocol.DEEPAGENT_CREATE,
                                      ApiBucket.FRAMEWORK))
    if framework:
        return framework
    return [*_protocol_definitions(file, tree, parent_of), *_typed_generate_calls(file, tree, parent_of),
            *_openai_create_calls(file, tree, parent_of)]


def _langchain_sites(file, tree, parent_of):
    names = _model_names(tree)
    sites = [_site(file, node, enclosing(node, parent_of), ApiProtocol.LANGCHAIN_INVOKE, ApiBucket.FRAMEWORK)
             for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr in INVOKE_METHODS and _unwrap_model(node.func.value, names)]
    if sites:
        return sites
    return _named_calls(file, tree, parent_of, {"init_chat_model"}, ApiProtocol.LANGCHAIN_FACTORY, ApiBucket.FRAMEWORK)


def _model_names(tree):
    names = set()
    assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
    changed = True
    while changed:
        changed = False
        for node in assignments:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is None or not _unwrap_model(node.value, names):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    changed = True
    return names


def _unwrap_model(expr, names):
    while isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr in MODEL_WRAPPERS:
        expr = expr.func.value
    if isinstance(expr, ast.Name) and expr.id in names:
        return True
    return isinstance(expr, ast.Call) and call_name(expr).split(".")[-1] == "init_chat_model"


def _protocol_definitions(file, tree, parent_of):
    return [_site(file, node, function_symbol(node, parent_of), ApiProtocol.CLIENT_GENERATE, ApiBucket.PROTOCOL)
            for node in ast.walk(tree)
            if isinstance(node, FUNCTION_TYPES) and node.name == "generate"
            and {"messages", "tools"} <= {arg.arg for arg in named_arguments(node)}]


def _typed_generate_calls(file, tree, parent_of):
    aliases = imported_aliases(tree, "LLMClient")
    if not aliases:
        return []
    sites = []
    for owner in tree.body:
        if not isinstance(owner, ast.ClassDef):
            continue
        methods = [stmt for stmt in owner.body if isinstance(stmt, FUNCTION_TYPES)]
        initializer = next((method for method in reversed(methods) if method.name == "__init__"), None)
        if initializer is None:
            continue
        parameters = {arg.arg for arg in named_arguments(initializer) if dotted(arg.annotation) in aliases}
        fields = {child.target.attr for child in ast.walk(initializer)
                  if isinstance(child, ast.AnnAssign) and dotted(child.annotation) in aliases
                  and isinstance(child.value, ast.Name) and child.value.id in parameters
                  and isinstance(child.target, ast.Attribute) and dotted(child.target.value) == "self"}
        for child in ast.walk(owner):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                continue
            receiver = child.func.value
            if (child.func.attr == "generate" and isinstance(receiver, ast.Attribute)
                    and dotted(receiver.value) == "self" and receiver.attr in fields):
                sites.append(_site(file, child, enclosing(child, parent_of),
                                   ApiProtocol.CLIENT_GENERATE, ApiBucket.PROTOCOL))
    return sites


def _openai_create_calls(file, tree, parent_of):
    return [_site(file, node, enclosing(node, parent_of), ApiProtocol.OPENAI_CHAT, ApiBucket.PROTOCOL)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _attribute_chain(node.func) == ["chat", "completions", "create"]]


def _attribute_chain(expr, depth=3):
    chain = []
    while isinstance(expr, ast.Attribute) and len(chain) < depth:
        chain.append(expr.attr)
        expr = expr.value
    return chain[::-1]


def _named_calls(file, tree, parent_of, names, protocol, bucket):
    return [_site(file, node, enclosing(node, parent_of), protocol, bucket)
            for node in ast.walk(tree) if isinstance(node, ast.Call) and call_name(node).split(".")[-1] in names]


def _typescript_sites(file):
    sites = []
    for call in typescript_calls(file):
        callee = call["callee"]
        if callee == "streamText":
            protocol, bucket = ApiProtocol.AI_SDK_STREAM, ApiBucket.PROTOCOL
        elif callee.endswith("chat.completions.create"):
            protocol, bucket = ApiProtocol.OPENAI_CHAT, ApiBucket.PROTOCOL
        elif callee.split(".")[-1] == "create_deep_agent":
            protocol, bucket = ApiProtocol.DEEPAGENT_CREATE, ApiBucket.FRAMEWORK
        else:
            continue
        sites.append(make_site(file, call["line"], call["column"], call["symbol"], protocol, bucket))
    return sites


def _site(file, node, symbol, protocol, bucket):
    return make_site(file, node.lineno, node.col_offset + 1, symbol, protocol, bucket)
