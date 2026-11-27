import json
from pathlib import Path
import tempfile
import unittest

from src.core.apiSite import iter_source_files
from src.core.callHierarchy import hierarchy
from src.core.graphModel import Unsupported
from src.core.locateLlmApi import locate
from src.core.projectFromApi import projectFromApi


ROOT = Path(__file__).resolve().parents[1]
ODR = ROOT / "instances/odr/open_deep_research/deep_researcher.py"
STIRRUP_AGENT = ROOT / "instances/stirrup/stirrup/core/agent.py"
STIRRUP_PROTOCOL = ROOT / "instances/stirrup/stirrup/core/models.py"
OPENCODE = ROOT / "instances/opencode/session"


def pairs(graph):
    return {(edge["source"], edge["target"], edge["route"], edge["kind"]) for edge in graph["edges"]}


def node_ids(graph):
    return {(node["id"], node["kind"]) for node in graph["nodes"]}


class LocateApi(unittest.TestCase):
    def test_odr_sites_are_model_invocations_inside_node_functions(self):
        sites = locate(ODR)
        found = {(site.symbol.split(":")[-1], site.line) for site in sites}
        self.assertEqual(found, {
            ("clarify_with_user", 101),
            ("write_research_brief", 154),
            ("supervisor", 214),
            ("researcher", 415),
            ("compress_research", 551),
            ("final_report_generation", 650),
        })
        self.assertTrue(all(site.bucket == "framework" and site.protocol == "langchain-invoke" for site in sites))

    def test_state_graph_without_a_model_call_is_not_a_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "agent.py"
            source.write_text("""
from langgraph.graph import END, START, StateGraph

def only(state):
    return state

builder = StateGraph(dict)
builder.add_node(only)
builder.add_edge(START, "only")
builder.add_edge("only", END)
""", encoding="utf-8")
            with self.assertRaisesRegex(Unsupported, "No LLM API site"):
                projectFromApi(source)

    def test_model_locator_returns_only_a_symbol_and_does_not_invent_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "agent.py"
            source.write_text("def run():\n    return 1\n", encoding="utf-8")

            def ask(prompt):
                self.assertIn("Do not return graph nodes", prompt)
                return {"file": "agent.py", "line": 2, "symbol": "run"}

            sites = locate(source, ask)
            self.assertEqual([(site.bucket, site.symbol, site.line) for site in sites],
                             [("model", "agent.py:run", 2)])
            with self.assertRaisesRegex(Unsupported, "no framework or protocol path analysis"):
                projectFromApi(source, askModel=ask)

    def test_disagreement_uses_the_model_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent.py").write_text("""
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
chat = init_chat_model("demo")

async def only(state):
    await chat.ainvoke(state)
    return {}

builder = StateGraph(dict)
builder.add_node(only)
builder.add_edge(START, "only")
builder.add_edge("only", END)
""", encoding="utf-8")
            (root / "hook.ts").write_text("""
export async function stream(input) {
  return streamText({input})
}
""", encoding="utf-8")
            with self.assertRaisesRegex(Unsupported, "buckets disagree"):
                locate(root)

            def ask(prompt):
                self.assertIn("only", prompt)
                self.assertIn("stream", prompt)
                return {"file": "agent.py", "line": 7, "symbol": "only"}

            graph = projectFromApi(root, askModel=ask)
            self.assertEqual({site["symbol"] for site in graph["api_location"]["sites"]}, {"agent.py:only"})
            self.assertIn(("__start__", graph["nodes"][1]["id"], "next", "next_call"), pairs(graph))

    def test_deepagent_site_does_not_become_a_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "agent.py"
            source.write_text("""
from deepagents import create_deep_agent

def build():
    return create_deep_agent(model="demo")
""", encoding="utf-8")
            sites = locate(source)
            self.assertEqual(sites[0].protocol, "deepagent-create")
            self.assertEqual(sites[0].symbol, "agent.py:build")
            with self.assertRaisesRegex(Unsupported, "no pinned path analysis"):
                projectFromApi(source)

    def test_odr_reference_follows_the_located_sites(self):
        graph = projectFromApi(ODR, builder="deep_researcher_builder")
        expected = json.loads((ROOT / "instances/odr/check/reference.json").read_text(encoding="utf-8"))
        self.assertEqual(node_ids(graph), {tuple(node) for node in expected["nodes"]})
        self.assertEqual(pairs(graph), {tuple(edge) for edge in expected["edges"]})
        self.assertEqual(graph["api_location"]["sites"][0]["bucket"], "framework")

    def test_stirrup_reference_follows_typed_generate(self):
        graph = projectFromApi(STIRRUP_AGENT, protocol=STIRRUP_PROTOCOL)
        expected = json.loads((ROOT / "instances/stirrup/check/reference.json").read_text(encoding="utf-8"))
        self.assertEqual(node_ids(graph), {tuple(node) for node in expected["nodes"]})
        reference_edges = {tuple(edge) for edge in expected["edges"]}
        self.assertTrue(reference_edges <= pairs(graph))
        self.assertEqual(pairs(graph) - reference_edges, {
            ("Agent.summarize_messages:1378:37", "__end__", "next", "next_call"),
        })
        symbols = {site["symbol"].split(":")[-1] for site in graph["api_location"]["sites"]}
        self.assertIn("Agent.step", symbols)
        self.assertIn("Agent.summarize_messages", symbols)
        callers = {edge["source"].split(":")[-1] for edge in graph["api_location"]["hierarchy"]["edges"]}
        self.assertTrue(callers)

    def test_opencode_stream_text_has_callers_and_matches_reference(self):
        sites = locate(OPENCODE)
        stream = [site for site in sites if site.protocol == "ai-sdk-stream"]
        self.assertEqual([(site.file, site.line, site.symbol.split(":")[-1]) for site in stream],
                         [("llm.ts", 172, "LLM.stream")])
        callers = {edge["source"] for edge in hierarchy(sites, iter_source_files(OPENCODE))["edges"]
                   if edge["target"].endswith("LLM.stream")}
        self.assertTrue(any(caller.endswith("ensureTitle") for caller in callers))
        self.assertTrue(any(caller.endswith("process") for caller in callers))
        try:
            graph = projectFromApi(OPENCODE)
        except Unsupported as error:
            if "typescript" in str(error) or "node" in str(error):
                self.skipTest(str(error))
            raise
        expected = json.loads((ROOT / "instances/opencode/check/reference.json").read_text(encoding="utf-8"))
        self.assertEqual(node_ids(graph), {tuple(node) for node in expected["nodes"]})
        self.assertEqual(pairs(graph), {tuple(edge) for edge in expected["edges"]})
