import tempfile
import unittest
from pathlib import Path

from agenteve import Source, langgraph


class StaticRoutesTest(unittest.TestCase):
    def extract(self, router, declaration=None):
        builder = declaration or '''
builder = StateGraph(dict)
builder.add_node("model", lambda state: state)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route)
graph = builder.compile()
'''
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "agent.py"
            source.write_text("from langgraph.graph import START, END, StateGraph\n" + router + "\n" + builder,
                              encoding="utf-8")
            return langgraph(Source(source), "build:builder" if declaration else "builder")

    def test_finds_guarded_end_and_loop(self):
        graph = self.extract('''
def route(state):
    if state["messages"]:
        return END
    return "model"
''')
        edges = {edge["target"]: edge for edge in graph["edges"] if edge["source"] == "model"}
        self.assertEqual(set(edges), {"__end__", "model"})
        self.assertIn("not", edges["model"]["condition"])
        self.assertEqual(len(edges["model"]["evidence"]), 2)
        self.assertEqual(graph["structure"]["cycle_nodes"], ["model"])

    def test_retains_unknown_branch_alongside_known_edge(self):
        graph = self.extract('''
def route(state):
    return END if state["done"] else state["next"]
''')
        self.assertTrue(graph["structure"]["contains_unresolved_transitions"])
        self.assertEqual({edge["kind"] for edge in graph["edges"] if edge["source"] == "model"},
                         {"conditional", "unresolved"})

    def test_does_not_resolve_a_shadowed_end_parameter(self):
        graph = self.extract('''
def route(END):
    return END
''')
        self.assertTrue(graph["structure"]["contains_unresolved_transitions"])
        self.assertFalse(any(edge["target"] == "__end__" for edge in graph["edges"]))

    def test_selects_factory_builder(self):
        graph = self.extract('''
def route(state):
    return ["model", END]
''', '''
def build():
    builder = StateGraph(dict)
    builder.add_node("model", lambda state: state)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route)
    return builder.compile()
''')
        self.assertEqual(graph["scope"], "build:builder")
        self.assertEqual({edge["target"] for edge in graph["edges"] if edge["source"] == "model"},
                         {"__end__", "model"})


if __name__ == "__main__":
    unittest.main()
