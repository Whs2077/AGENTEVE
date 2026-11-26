"""Exercise real Harness branches with predetermined model responses."""

import json

from agenteve import Source
from llm_flow import extract_llm_flow
from stage2.odr_pilot import MODULE, SNAPSHOT
from stage2.paired_pilot import (
    FINISH_TOOL, GRAPH_DIR, MODEL_BUDGET, ODR_CALL, ODR_COMPRESSION, READ_TOOL, STIRRUP_CALL,
    Exchange, OutputError, read, run_odr, run_stirrup, sha256, tool_feedback, write,
)


EVIDENCE = (
    {"D01": "The Alder Library opens on Sundays.", "D02": "The Maple Library opens on Sundays."},
    {"D03": "The Cedar Library is closed on Sundays.", "D04": "The Pine Library is closed on Sundays."},
)
TASK = {
    "question": "Are both the Alder Library and the Maple Library open on Sundays?",
    "documents": [{"id": doc_id, "sentence": sentence} for page in EVIDENCE for doc_id, sentence in page.items()],
}
SUBMISSION = {"answer": "yes", "citation_ids": ["D01", "D02"]}
COMPRESSION = "Scripted summary of the tool observations."
SCENARIOS = (
    ("submit-after-one-read", 4, (READ_TOOL, FINISH_TOOL)),
    ("submit-after-two-reads", 4, (READ_TOOL, READ_TOOL, FINISH_TOOL)),
    ("turn-limit-after-two-reads", 2, (READ_TOOL, READ_TOOL)),
)
# These lines are in the pinned ODR snapshot and its corresponding extracted graph.
ODR_LATE_EXIT_TEST = 498
ODR_LATE_EXIT_RETURN = 500


class ScriptedResponses:
    model_id = "agenteve-scripted"
    token_budget = MODEL_BUDGET

    def __init__(self):
        self.calls = []
        self.compression_inputs = []


class BranchExchange(Exchange):
    def __init__(self, call_limit, steps):
        super().__init__(TASK, ScriptedResponses())
        self.call_limit = call_limit
        self.steps = steps

    def read_evidence(self):
        if len(self.events) >= len(EVIDENCE):
            raise OutputError("No scripted evidence page remains")
        payload = json.dumps(EVIDENCE[len(self.events)])
        self.events.append({"tool": READ_TOOL, "result": payload})
        return payload

    def generate(self, messages):
        index = len(self.model.calls)
        if index >= self.call_limit or index >= len(self.steps):
            raise OutputError("Harness requested an unexpected model turn")
        for event in self.events:
            if tool_feedback(event["tool"], event["result"]) not in messages:
                raise OutputError("Harness omitted a prior tool result from the next model call")
        name = self.steps[index]
        args = SUBMISSION if name == FINISH_TOOL else {}
        self.model.calls.append({"tool": name, "arguments": args, "messages": messages})
        return name, args, f"branch-{index + 1}"

    def compress(self, messages):
        self.model.compression_inputs.append([{"type": message.type, "content": message.content}
                                              for message in messages])
        return COMPRESSION


async def run_branch_checks(out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(SNAPSHOT / "SOURCE.json")
    source_url = (
        f"{manifest['repository']}/blob/{manifest['commit']}/src/open_deep_research/deep_researcher.py"
    )
    graph_files = {
        "open-deep-research": out / "open-deep-research-flow.json",
        "stirrup": GRAPH_DIR / "stirrup/extracted/flow.json",
    }
    write(graph_files["open-deep-research"], extract_llm_flow(
        Source(MODULE / "deep_researcher.py", source_url), "deep_researcher_builder",
    ))
    results = {}
    for scenario, limit, steps in SCENARIOS:
        expected_stop = "submitted" if steps[-1] == FINISH_TOOL else "turn_limit"
        scenario_dir = out / scenario
        scenario_dir.mkdir()
        results[scenario] = {}
        for name, runner in (("open-deep-research", run_odr), ("stirrup", run_stirrup)):
            exchange = BranchExchange(limit, steps)
            path, details = [], {}
            odr = name == "open-deep-research"
            try:
                details = await runner(exchange, path, **({"compress": exchange.compress} if odr else {}))
            except OutputError as error:
                details["runtime_error"] = str(error)
            graph_file = graph_files[name]
            graph_edges = {(edge["source"], edge["target"]): edge for edge in read(graph_file)["edges"]
                           if edge["kind"] == "next_call"}
            observed_edges = list(zip(path, path[1:]))
            missing_edges = [edge for edge in observed_edges if edge not in graph_edges]
            stop = "unexpected"
            if exchange.submission is not None:
                stop = "submitted"
            elif len(exchange.model.calls) == limit and "runtime_error" not in details:
                stop = "turn_limit"
            expected_path = ([ODR_CALL] * len(steps) + [ODR_COMPRESSION] if odr else
                             ["__start__"] + [STIRRUP_CALL] * len(steps) + ["__end__"])
            passed = (
                "runtime_error" not in details
                and stop == expected_stop
                and [call["tool"] for call in exchange.model.calls] == list(steps)
                and [event["tool"] for event in exchange.events] == list(steps)
                and path == expected_path
                and not missing_edges
            )
            result = {
                "passed": passed,
                "stop": stop,
                "tool_loop_model_calls": len(exchange.model.calls),
                "compression_model_calls": len(exchange.model.compression_inputs),
                "matched_endpoints": len(observed_edges) - len(missing_edges),
                "observed_edges": len(observed_edges),
            }
            if odr:
                edge = graph_edges.get((ODR_CALL, ODR_COMPRESSION), {})
                result["compression_return_witnessed"] = any(
                    {ODR_LATE_EXIT_TEST, ODR_LATE_EXIT_RETURN} <= {anchor["line"] for anchor in witness["evidence"]}
                    for witness in edge.get("witnesses", [])
                )
                result["passed"] &= (
                    details.get("compressed_research") == COMPRESSION and result["compression_return_witnessed"]
                )
            elif "finish_params" in details:
                result["passed"] &= details["finish_params"] == exchange.submission
            results[scenario][name] = result
            write(scenario_dir / f"{name}.json", {
                "expected": {"stop": expected_stop, "tool_loop_limit": limit, "tools": steps, "llm_path": expected_path},
                "details": details,
                "llm_path": path,
                "missing_edges": missing_edges,
                "graph_sha256": sha256(graph_file),
                "model_requests": exchange.model.calls,
                "compression_inputs": exchange.model.compression_inputs,
                "tool_events": exchange.events,
                "output": exchange.submission,
                "result": result,
            })
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{scenario} {name}: {status}, stop={stop}, "
                  f"graph={result['matched_endpoints']}/{result['observed_edges']}", flush=True)
    write(out / "branch-summary.json", {"backend": "scripted", "scenarios": results})
    return all(result["passed"] for harnesses in results.values() for result in harnesses.values())
