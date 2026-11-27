from pathlib import Path
import tempfile
import unittest

from src.core.graphModel import Source
from src.core.projectNextCall import projectNextCall


PREAMBLE = """
from langchain.chat_models import init_chat_model as make_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

chat = make_model("demo-model")
"""


class LlmFlow(unittest.TestCase):
    def extract(self, script):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "agent.py"
            source.write_text(PREAMBLE + script, encoding="utf-8")
            return projectNextCall(Source(source), "builder")

    def next_calls(self, graph):
        return {(edge["source"].split(":")[0], edge["target"].split(":")[0])
                for edge in graph["edges"]}

    def test_same_route_before_and_after_optional_model_call(self):
        graph = self.extract("""
async def first(state):
    if state["skip"]:
        return Command(goto="second")
    await chat.ainvoke(state)
    return Command(goto="second")

async def second(state):
    await chat.with_config({"model": "writer"}).ainvoke(state)
    return {}

builder = StateGraph(dict)
builder.add_node(first)
builder.add_node(second)
builder.add_edge(START, "first")
builder.add_edge("second", END)
""")
        self.assertEqual(self.next_calls(graph), {
            ("__start__", "first"), ("__start__", "second"), ("first", "second"), ("second", "__end__"),
        })
        nodes = {n["id"].split(":")[0]: n for n in graph["nodes"] if n["kind"] == "llm_call"}
        self.assertEqual(nodes["first"]["model_selectors"], ["demo-model"])
        self.assertEqual(nodes["second"]["model_selectors"], ["writer"])
        edge = next(e for e in graph["edges"] if e["source"].startswith("first:"))
        self.assertEqual(edge["witnesses"][0]["steps"], ["goto second"])
        self.assertIn(12, {a["line"] for a in edge["witnesses"][0]["evidence"]})

    def test_rebinding_model_to_tool_and_unawaited_call(self):
        graph = self.extract("""
async def one_action(state):
    chat.ainvoke(state)
    model = chat
    model.invoke(state)
    model = state["tool"]
    model.invoke(state)
    final = chat.bind_tools([])
    await final.ainvoke(state)
    return {}

builder = StateGraph(dict)
builder.add_node(one_action)
builder.add_edge(START, "one_action")
builder.add_edge("one_action", END)
""")
        nodes = [n for n in graph["nodes"] if n["kind"] == "llm_call"]
        self.assertEqual([n["method"] for n in nodes], ["invoke", "ainvoke"])
        self.assertEqual([n["evidence"][0]["line"] for n in nodes], [11, 15])
        self.assertEqual({(e["source"], e["target"]) for e in graph["edges"]}, {
            (nodes[0]["id"], nodes[1]["id"]), (nodes[1]["id"], "__end__"), ("__start__", nodes[0]["id"]),
        })
        self.assertTrue(graph["structure"]["contains_unresolved_transitions"])

    def test_distinct_command_returns_preserve_evidence_on_the_same_llm_edge(self):
        graph = self.extract("""
async def research(state):
    await chat.ainvoke(state)
    return Command(goto="dispatch")

async def dispatch(state):
    if not state["tool_calls"]:
        return Command(goto="compress")
    if state["done"] or state["limit"]:
        return Command(goto="compress")
    return Command(goto="research")

async def compress(state):
    await chat.ainvoke(state)
    return {}

builder = StateGraph(dict)
builder.add_node(research)
builder.add_node(dispatch)
builder.add_node(compress)
builder.add_edge(START, "research")
builder.add_edge("compress", END)
""")
        edges = [edge for edge in graph["edges"]
                 if edge["source"].startswith("research:") and edge["target"].startswith("compress:")]
        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(len(edge["witnesses"]), 2)
        self.assertEqual(
            {tuple(witness["conditions"]) for witness in edge["witnesses"]},
            {("not state['tool_calls']",),
             ("state['tool_calls']", "state['done'] or state['limit']")},
        )
        self.assertIn("state['done'] or state['limit']", edge["condition"])


if __name__ == "__main__":
    unittest.main()
