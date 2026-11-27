"""Build a call graph only after an LLM API site has been located.

The located site selects the path analysis. LangGraph node names and Agent
method names are produced by that analysis, not by the locator.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile

from src.core.callHierarchy import hierarchy
from src.core.graphModel import Source, Unsupported, completeGraph
from src.core.locateLlmApi import ApiProtocol, iter_source_files, locate


def projectFromApi(source, *, builder=None, protocol=None, sourceUrl=None, protocolUrl=None,
                   coordinates="file", askModel=None):
    """Locate the API, keep its call hierarchy, then project the next model call."""
    root = Path(source)
    sites = locate(root, askModel)
    relation = hierarchy(sites, iter_source_files(root) if root.is_dir() else None)
    graph = _project(root, sites, builder=builder, protocol=protocol, sourceUrl=sourceUrl,
                     protocolUrl=protocolUrl, coordinates=coordinates)
    graph["api_location"] = {"sites": [site.as_dict() for site in sites], "hierarchy": relation}
    return graph


def _project(root, sites, *, builder, protocol, sourceUrl, protocolUrl, coordinates):
    protocols = {site.protocol for site in sites}
    if ApiProtocol.LANGCHAIN_INVOKE.value in protocols or ApiProtocol.LANGCHAIN_FACTORY.value in protocols:
        return _project_langchain(sites, builder, sourceUrl, coordinates)
    if ApiProtocol.DEEPAGENT_CREATE.value in protocols:
        site = next(site for site in sites if site.protocol == ApiProtocol.DEEPAGENT_CREATE.value)
        raise Unsupported(f"DeepAgent API is at {site.symbol}:{site.line}, and no pinned path analysis reads it yet")
    if _has_agent_generate(sites) or (
            ApiProtocol.CLIENT_GENERATE.value in protocols and _agent_file(sites) is not None):
        return _project_stirrup(sites, protocol, sourceUrl, protocolUrl, coordinates)
    if ApiProtocol.AI_SDK_STREAM.value in protocols:
        return _project_opencode(root, sites, sourceUrl)
    if protocols == {ApiProtocol.MODEL_LOCATED.value}:
        site = sites[0]
        raise Unsupported(f"Model locator found {site.symbol} at {site.file}:{site.line}, "
                          "and no framework or protocol path analysis applies")
    if protocols <= {ApiProtocol.OPENAI_CHAT.value, ApiProtocol.CLIENT_GENERATE.value}:
        site = sites[0]
        raise Unsupported(f"API site {site.symbol} at {site.file}:{site.line} has no Agent.run entry")
    raise Unsupported("Located API sites do not select a path analysis: " + ", ".join(sorted(protocols)))


def _project_langchain(sites, builder, sourceUrl, coordinates):
    files = _files_for(sites, {ApiProtocol.LANGCHAIN_INVOKE.value, ApiProtocol.LANGCHAIN_FACTORY.value})
    source = Source(files[0], sourceUrl, coordinates)
    selected = builder or _pick_builder(source.tree, sites)
    from src.core.projectNextCall import projectNextCall
    return projectNextCall(source, selected)


def _project_stirrup(sites, protocol, sourceUrl, protocolUrl, coordinates):
    agent = _agent_file(sites)
    if agent is None:
        raise Unsupported("A generate protocol was found, and no Agent class calls it")
    protocol_path = Path(protocol) if protocol else _protocol_file(sites)
    if protocol_path is None:
        sibling = Path(agent).parent / "models.py"
        if sibling.is_file():
            protocol_path = sibling
    if protocol_path is None:
        raise Unsupported(f"Agent API sites in {agent.name} need the LLMClient protocol file")
    from src.core.projectStirrup import projectStirrup
    return projectStirrup(Source(agent, sourceUrl, coordinates), Source(protocol_path, protocolUrl, coordinates))


def _project_opencode(root, sites, sourceUrl):
    session = _session_dir(root, sites)
    script = session.parent / "check" / "buildOpenCodeGraph.cjs"
    if not script.is_file():
        site = sites[0]
        raise Unsupported(f"streamText is at {site.symbol}:{site.line}, and {script} is not available for path analysis")
    typescript = _typescript_module()
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "callGraph.json"
        completed = subprocess.run(
            ["node", str(script), typescript, str(session), sourceUrl or "", str(output)],
            capture_output=True, text=True, encoding="utf-8", check=False,
        )
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout or "node failed").strip()
            raise Unsupported(f"OpenCode path analysis failed after streamText was located: {detail}")
        import json
        graph = json.loads(output.read_text(encoding="utf-8"))
    return completeGraph(graph)


def _pick_builder(tree, sites):
    import ast
    names = {_short(site.symbol) for site in sites}
    scores = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_node":
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        callables = [arg.id for arg in node.args if isinstance(arg, ast.Name)]
        callables.extend(keyword.value.id for keyword in node.keywords if isinstance(keyword.value, ast.Name))
        scores.setdefault(node.func.value.id, set()).update(name for name in callables if name in names)
    ranked = sorted(((len(found), name) for name, found in scores.items() if found), reverse=True)
    if len(ranked) == 1 or (len(ranked) > 1 and ranked[0][0] > ranked[1][0]):
        return ranked[0][1]
    found = ", ".join(name for _score, name in ranked) or "none"
    raise Unsupported(f"LangChain API sites need one StateGraph builder. Candidates: {found}")


def _has_agent_generate(sites):
    return any(site.protocol == ApiProtocol.CLIENT_GENERATE.value and _short(site.symbol).startswith("Agent.")
               for site in sites)


def _agent_file(sites):
    matches = [Path(site.path) for site in sites
               if site.protocol == ApiProtocol.CLIENT_GENERATE.value and ":Agent." in f":{site.symbol}"]
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def _protocol_file(sites):
    definition_sites = [site for site in sites if site.protocol == ApiProtocol.CLIENT_GENERATE.value
                        and site.symbol.rsplit(".", 1)[-1] == "generate"]
    files = list(dict.fromkeys(Path(site.path) for site in definition_sites))
    if len(files) == 1:
        return files[0]
    return None


def _files_for(sites, protocols):
    files = list(dict.fromkeys(Path(site.path) for site in sites if site.protocol in protocols))
    if not files:
        raise Unsupported("The selected API family has no source file")
    if len(files) == 1:
        return files
    counts = {file: sum(site.path == str(file) for site in sites) for file in files}
    best = max(counts.values())
    winners = [file for file, count in counts.items() if count == best]
    if len(winners) != 1:
        raise Unsupported("More than one file contains the same number of API sites")
    return winners


def _session_dir(root, sites):
    stream = next(site for site in sites if site.protocol == ApiProtocol.AI_SDK_STREAM.value)
    parent = Path(stream.path).parent
    if parent.name == "session":
        return parent
    if root.is_dir() and (root / "llm.ts").is_file():
        return root
    raise Unsupported(f"streamText at {stream.symbol}:{stream.line} is not in a session directory")


def _typescript_module():
    if shutil.which("node") is None:
        raise Unsupported("node is not on PATH, so the OpenCode path analysis cannot run")
    completed = subprocess.run(
        ["node", "-e", "process.stdout.write(require.resolve('typescript'))"],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = (completed.stderr or "typescript was not found").strip()
        raise Unsupported(f"The typescript package is required after streamText is located: {detail}")
    return completed.stdout.strip()


def _short(symbol):
    return symbol.split(":", 1)[-1]
