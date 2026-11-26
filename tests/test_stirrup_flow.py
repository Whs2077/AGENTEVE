from pathlib import Path
import tempfile
import unittest

from agenteve import Source
from stirrup_flow import extract_stirrup


class StirrupFlowTest(unittest.TestCase):
    def test_follows_typed_client_through_a_method_and_skips_tool_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Path(directory) / "agent.py"
            protocol = Path(directory) / "models.py"
            protocol.write_text("""
from typing import Protocol

class LLMClient(Protocol):
    async def generate(self, messages, tools): ...
""", encoding="utf-8")
            agent.write_text("""
from stirrup.core.models import LLMClient as Client

class Agent:
    def __init__(self, client: Client, tool):
        self.backend: Client = client
        self.tool = tool

    async def make_answer(self):
        await self.tool.generate()
        return await self.backend.generate([], {})

    async def run(self):
        await self.make_answer()
        return await self.backend.generate([], {})
""", encoding="utf-8")
            graph = extract_stirrup(Source(agent), Source(protocol))
        calls = [n for n in graph["nodes"] if n["kind"] == "llm_call"]
        self.assertEqual([n["evidence"][0]["line"] for n in calls], [11, 15])
        found = {(e["source"], e["target"]) for e in graph["edges"]}
        self.assertEqual(found, {
            ("__start__", calls[0]["id"]), (calls[0]["id"], calls[1]["id"]), (calls[1]["id"], "__end__"),
        })
        self.assertEqual([e["file"] for e in calls[0]["evidence"]], ["agent.py", "agent.py", "models.py"])


if __name__ == "__main__":
    unittest.main()

