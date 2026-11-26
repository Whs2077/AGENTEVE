"""Compile the installed Deep Agents SDK and exercise its tool loop offline."""

import argparse
import inspect
from importlib.metadata import version
from pathlib import Path

from agenteve import write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()

    from deepagents import create_deep_agent
    from deepagents.graph import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langsmith import tracing_context

    calls = []

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two integers."""
        calls.append((a, b))
        return a + b

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    responses = [
        AIMessage(content="", tool_calls=[
            {"name": "add_numbers", "args": {"a": 2, "b": 3}, "id": "sum-01"}]),
        AIMessage(content="Result: 5"),
    ]
    with tracing_context(enabled=False):
        agent = create_deep_agent(model=ScriptedModel(responses=responses), tools=[add_numbers])
        exported = agent.get_graph(xray=True).to_json()
        trace = list(agent.stream({"messages": [{"role": "user", "content": "Compute 2 + 3"}]},
                                  stream_mode="updates", subgraphs=True))

    # Verify the observed model→tool→model path using runtime message objects.
    events = []
    for namespace, updates in trace:
        for node, update in updates.items():
            events.append({"namespace": list(namespace), "node": node})
            if not isinstance(update, dict):
                continue
            messages = update.get("messages") or []
            for message in messages:
                if isinstance(message, ToolMessage) and message.tool_call_id == "sum-01":
                    events[-1]["tool_result"] = message.content
                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        events[-1]["tool_calls"] = [call["id"] for call in message.tool_calls]
                    if message.content:
                        events[-1]["assistant_content"] = message.content
    requested = next(i for i, event in enumerate(events) if event.get("tool_calls") == ["sum-01"])
    returned = next(i for i, event in enumerate(events) if event.get("tool_result") == "5")
    answered = next(i for i, event in enumerate(events) if event.get("assistant_content") == "Result: 5")
    assert calls == [(2, 3)] and requested < returned < answered
    edge_pairs = {(edge["source"], edge["target"]) for edge in exported["edges"]}
    assert (events[requested]["node"], events[returned]["node"]) in edge_pairs
    assert (events[returned]["node"], events[answered]["node"]) in edge_pairs

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "compiled.json", exported)
    packages = {name: version(name) for name in ("deepagents", "langchain", "langchain-core", "langgraph")}
    write_json(args.out / "validation.json",
               {"packages": packages, "profile": "Default SDK configuration with scripted model and one local arithmetic tool",
                "graph_source": "create_deep_agent(...).get_graph(xray=True).to_json()",
                "assembly": [{"module": fn.__module__, "function": fn.__name__,
                              "first_line": inspect.getsourcelines(fn)[1]}
                             for fn in (create_deep_agent, create_agent)],
                "checks": {"tool_called_once": True, "runtime_order_matches_compiled_edges": True},
                "events": events})
    print(f"Deep Agents {packages['deepagents']}: {len(exported['nodes'])} nodes, "
          f"{len(exported['edges'])} edges; model→tool→model verified")


if __name__ == "__main__":
    main()
