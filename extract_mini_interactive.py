"""Project the upstream InteractiveAgent control flow with inherited loop evidence."""

import argparse
import ast
from pathlib import Path

from agenteve import Graph, Source, Unsupported, dotted, mini, write_graph


def anchor(method, cls, predicate, name):
    hits = [node for node in ast.walk(cls[method]) if predicate(node)]
    if len(hits) != 1:
        raise Unsupported(f"InteractiveAgent {method}: expected one {name}, found {len(hits)}")
    return hits[0]


def extract(base, interactive, exceptions):
    inherited = mini(base, exceptions)
    cls = next((n for n in interactive.tree.body
                if isinstance(n, ast.ClassDef) and n.name == "InteractiveAgent"
                and any(dotted(b) == "DefaultAgent" for b in n.bases)), None)
    if cls is None:
        raise Unsupported("InteractiveAgent does not extend DefaultAgent")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    def find(method, node_type, match, name):
        return interactive.evidence(anchor(method, methods,
                                           lambda node: isinstance(node, node_type) and match(node), name))

    def guard(method, condition):
        return find(method, ast.If, lambda node: ast.unparse(node.test) == condition, condition)

    def handler(method, exception):
        return find(method, ast.ExceptHandler, lambda n: dotted(n.type) == exception, exception)

    def call(method, name):
        return find(method, ast.Call, lambda n: dotted(n.func) == name, name)

    upstream = {node["id"]: node["evidence"][0] for node in inherited["nodes"]}
    graph = Graph("mini-interactive", interactive, "entry", "InteractiveAgent plus inherited DefaultAgent.run",
                  "projected_control_flow")
    graph.data["sources"] += inherited["sources"]
    item = {
        "entry": ("entry", "Initialize messages", upstream["initialize"]),
        "mode": ("branch", "Human mode?", guard("query", "self.config.mode == 'human'")),
        "human": ("branch", "Read command or switch mode", find("query", ast.Match, lambda n: True, "match")),
        "limits": ("branch", "Check step and cost limits", upstream["budget"]),
        "time": ("branch", "Check time limit", upstream["time"]),
        "tty": ("branch", "Can request more budget?", guard("query", "not self._stdin_is_interactive()")),
        "budget": ("step", "Read new limits and query once", handler("query", "LimitsExceeded")),
        "model": ("model", "Query model and record reply", upstream["model"]),
        "approval": ("branch", "Any action requires approval?", guard("_ask_confirmation_or_interrupt",
                                                                  "not any((self._should_ask_confirmation(c) for c in commands))")),
        "prompt": ("branch", "Read confirmation or rejection", find("_ask_confirmation_or_interrupt",
                                                                   ast.Match, lambda n: True, "confirmation match")),
        "actions": ("tool", "Execute action batch", call("execute_actions", "self.env.execute")),
        "submission": ("branch", "Confirm submission if configured", guard("_check_for_new_task_or_submit",
                                                                           "self.config.confirm_exit")),
        "partial_end": ("step", "Append partial observations before submission", call("execute_actions",
                                                                                       "self.model.format_observation_messages")),
        "partial_continue": ("step", "Append partial observations before continuing", call("execute_actions",
                                                                                         "self.model.format_observation_messages")),
        "observe": ("step", "Append observations and reset format count", call("execute_actions",
                                                                              "self.model.format_observation_messages")),
        "interrupt": ("step", "Append user interruption", call("_interrupt", "UserInterruption")),
        "format": ("branch", "Check consecutive format errors", upstream["format"]),
        "format_retry": ("step", "Append format feedback", upstream["format_retry"]),
        "format_stop": ("step", "Record repeated format error exit", upstream["format_stop"]),
        "exit_message": ("step", "Append exit message from exception", upstream["interrupt_exit"]),
        "save_continue": ("step", "Save trajectory for live loop", upstream["save"]),
        "save_exit": ("step", "Save final trajectory", upstream["save_exit"]),
        "return": ("terminal", "Return exit metadata", upstream["return"]),
    }
    for node_id, (kind, label, proof) in item.items():
        graph.node(node_id, kind, label, proof)

    def edge(start, end, route="next", kind="flow", proof=None):
        graph.edge(start, end, route, proof or item[start][2], kind=kind,
                   condition=route if kind != "flow" else None)

    edge("entry", "mode")
    edge("mode", "human", "human", "conditional")
    edge("mode", "limits", "agent mode", "conditional")
    edge("human", "approval", "manual action", "conditional")
    edge("human", "limits", "switch to agent", "conditional")
    edge("limits", "tty", "step/cost limit", "exception")
    edge("limits", "time", "within budget", "conditional")
    edge("tty", "exit_message", "no interactive stdin", "conditional")
    edge("tty", "budget", "interactive stdin", "conditional")
    edge("budget", "time", "new budget permits request", "conditional")
    edge("budget", "exit_message", "new budget still exceeded", "exception")
    edge("time", "exit_message", "time limit", "exception", handler("query", "TimeExceeded"))
    edge("time", "model", "within time", "conditional")
    edge("model", "approval")
    edge("model", "format", "FormatError", "exception")
    edge("approval", "prompt", "approval required", "conditional")
    edge("approval", "actions", "skip approval", "conditional")
    edge("prompt", "actions", "approve", "conditional")
    edge("prompt", "partial_continue", "reject or switch to human", "exception")
    edge("actions", "observe", "batch complete", "flow")
    edge("actions", "submission", "Submitted", "exception", handler("execute_actions", "Submitted"))
    edge("submission", "partial_end", "submit", "exception")
    edge("submission", "partial_continue", "new task or human mode", "exception")
    edge("submission", "submission", "mode switch requests confirmation again", "loop")
    edge("partial_continue", "interrupt")
    edge("partial_end", "exit_message")
    edge("observe", "save_continue")
    edge("interrupt", "save_continue")
    edge("format", "format_retry", "retry", "conditional")
    edge("format", "format_stop", "limit reached", "conditional")
    edge("format_retry", "save_continue")
    edge("format_stop", "save_exit")
    edge("exit_message", "save_exit")
    edge("save_continue", "mode", "next round", "loop")
    edge("save_exit", "return")
    graph.warn("Projection follows regular tool flow and selected explicitly handled outcomes. KeyboardInterrupt, "
               "unexpected exceptions, failures in saving, and slash-command internals are outside this projection.",
               upstream["entry"])
    return graph.finish()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("agent_directory", type=Path, help="mini-swe-agent/src/minisweagent")
    cli.add_argument("--source-url", required=True, help="GitHub blob URL up to and including /src/minisweagent")
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    files = ["agents/default.py", "agents/interactive.py", "exceptions.py"]
    graph = extract(*(Source(args.agent_directory / file, f"{args.source_url.rstrip('/')}/{file}") for file in files))
    write_graph(args.out, graph)


if __name__ == "__main__":
    main()
