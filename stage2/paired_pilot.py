"""Run a fixed-evidence HotpotQA pilot through the ODR researcher loop and Stirrup."""

import argparse
import asyncio
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Literal

from stage2.odr_pilot import (
    CACHE, CONTEXT, DATE, DEEPSEEK_MODEL, MODULE, ROOT, SNAPSHOT, TEMPERATURE,
    DeepSeekModel, InferenceError, LocalModel, OutputError, Request,
    object_schema, read, read_deepseek_key, sha256, source_slice, write,
)


TASK = ROOT / "stage2/paired-hotpotqa.json"
BATCH = ROOT / "stage2/paired-hotpotqa-batch.json"
STIRRUP = ROOT / "sources/stirrup-runtime"
MODEL_BUDGET = 192
MODEL_CALL_LIMIT = 2
SEED = 101
READ_TOOL = "read_evidence"
FINISH_TOOL = "ResearchComplete"
READ_DESCRIPTION = "Return the same ten fixed, numbered evidence sentences for this question."
FINISH_DESCRIPTION = "Submit one yes or no answer and two supporting evidence IDs."
SYSTEM = (
    "Answer the question from the fixed evidence. First call read_evidence. "
    "Then submit ResearchComplete with answer yes or no and citation_ids containing the two "
    "document IDs that prove the comparison. Read the evidence before answering."
)
GRAPH_DIR = ROOT / "results/stage1-llm-flow"
ODR_CALL = "researcher:415:22"
ODR_COMPRESSION = "compress_research:551:30"
STIRRUP_CALL = "Agent.step:1301:35"


def tool_feedback(name, content):
    return "user", f"Tool {name} returned:\n{content}"


class DeepSeekPairModel(DeepSeekModel):
    """Translate the paired response contract into DeepSeek JSON output."""

    def generate(self, messages, schema, phase):
        prepared = [{"role": role, "content": text} for role, text in messages]
        read_example = json.dumps({"tool": READ_TOOL})
        submit_example = json.dumps(
            {"answer": "<yes or no>", "citation_ids": ["<first document ID>", "<second document ID>"]}
        )
        example = {
            "request_evidence": read_example,
            "choose_next": f"{read_example} OR {submit_example}. Return exactly one object.",
            "submit_answer": submit_example,
        }[phase]
        prepared[0]["content"] += (
            "\nRespond with a JSON object using the allowed values in this schema: "
            + json.dumps(schema)
            + "\nJSON shape example (replace placeholders using the evidence): "
            + example
        )
        response = super().generate(prepared, phase, response_format={"type": "json_object"})
        try:
            result = json.loads(response.get("content"))
        except (TypeError, ValueError) as error:
            raise OutputError("The paired DeepSeek response is empty or invalid JSON") from error
        if not isinstance(result, dict):
            raise OutputError("The paired model response must be a JSON object")
        return result


class Exchange:
    call_limit = MODEL_CALL_LIMIT
    system_prompt = SYSTEM
    read_description = READ_DESCRIPTION

    def __init__(self, task, model):
        self.task = task
        self.model = model
        self.payload = json.dumps(task["documents"], ensure_ascii=False)
        self.events = []
        self.submission = None

    def read_evidence(self):
        self.events.append({"tool": READ_TOOL, "result": self.payload})
        return self.payload

    def submit(self, answer, citation_ids):
        self.submission = {"answer": answer, "citation_ids": citation_ids}
        self.events.append({"tool": FINISH_TOOL, "arguments": self.submission})
        return "Answer recorded."

    def generate(self, messages):
        if len(self.model.calls) >= MODEL_CALL_LIMIT:
            raise OutputError("Shared model-call budget reached")
        if not self.model.calls:
            # Both harnesses use the same forced evidence-read protocol for this pilot.
            schema = object_schema({"tool": {"enum": [READ_TOOL]}})
            result = self.model.generate(messages, schema, "request_evidence")
            if result != {"tool": READ_TOOL}:
                raise OutputError("Invalid evidence tool request")
            return READ_TOOL, {}, "paired-1"
        if tool_feedback(READ_TOOL, self.payload) not in messages:
            raise OutputError("The second model call did not receive the shared tool result")
        document_ids = [doc["id"] for doc in self.task["documents"]]
        schema = object_schema({
            "answer": {"enum": ["yes", "no"]},
            "citation_ids": {"type": "array", "minItems": 2, "maxItems": 2,
                             "items": {"enum": document_ids}},
        })
        submission = self.model.generate(messages, schema, "submit_answer")
        citations = submission.get("citation_ids")
        if (set(submission) != {"answer", "citation_ids"} or submission["answer"] not in ("yes", "no")
                or not isinstance(citations, list) or len(citations) != 2
                or any(citation not in document_ids for citation in citations)):
            raise OutputError("Invalid answer or evidence IDs from the paired model")
        return FINISH_TOOL, submission, "paired-2"

    def grade(self):
        output = self.submission or {}
        reference = self.task["reference"]
        return {
            "model_calls": len(self.model.calls),
            "shared_evidence_once": sum(event["tool"] == READ_TOOL for event in self.events) == 1,
            "answer_match": output.get("answer") == reference["answer"],
            "support_match": sorted(output.get("citation_ids", [])) == sorted(reference["citation_ids"]),
        }


async def run_odr(exchange, path, *, compress=None):
    sys.path.insert(0, str(SNAPSHOT / "src"))
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command
    from langsmith import tracing_context
    from open_deep_research.configuration import Configuration
    from open_deep_research.state import ResearcherState

    @tool(READ_TOOL, description=exchange.read_description)
    async def read_evidence():
        return exchange.read_evidence()

    @tool(FINISH_TOOL, description=FINISH_DESCRIPTION)
    async def submit(answer: Literal["yes", "no"], citation_ids: list[str]):
        return exchange.submit(answer, citation_ids)

    async def get_all_tools(_config):
        return [read_evidence, submit]

    class LocalResearchAdapter(Request):
        def bind_tools(self, tools):
            if {tool.name for tool in tools} != {READ_TOOL, FINISH_TOOL}:
                raise ValueError("Unexpected tools in the ODR researcher")
            return self

        async def ainvoke(self, messages):
            chat = []
            for message in messages:
                if isinstance(message, ToolMessage):
                    chat.append(tool_feedback(message.name, message.content))
                elif isinstance(message, AIMessage):
                    chat.append(("assistant", f"Tool request: {message.tool_calls[0]['name']}"))
                else:
                    chat.append(("system" if isinstance(message, SystemMessage) else "user", message.content))
            calls_before = len(exchange.model.calls)
            try:
                name, args, call_id = exchange.generate(chat)
            finally:
                path.extend([ODR_CALL] * (len(exchange.model.calls) - calls_before))
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])

    class CompressionAdapter(Request):
        async def ainvoke(self, messages):
            path.append(ODR_COMPRESSION)
            return AIMessage(content=compress(messages))

    def local_only(_message):
        return False

    env = {
        "Configuration": Configuration, "ResearcherState": ResearcherState, "RunnableConfig": RunnableConfig,
        "Command": Command, "Literal": Literal, "SystemMessage": SystemMessage, "ToolMessage": ToolMessage,
        "configurable_model": LocalResearchAdapter(exchange.model), "get_all_tools": get_all_tools,
        "get_api_key_for_model": lambda *_: None, "get_today_str": lambda: DATE,
        "research_system_prompt": exchange.system_prompt,
        "openai_websearch_called": local_only, "anthropic_websearch_called": local_only, "asyncio": asyncio,
    }
    source_slice(MODULE / "deep_researcher.py", ("researcher", "execute_tool_safely", "researcher_tools"), env)
    builder = StateGraph(ResearcherState)
    for node in ("researcher", "researcher_tools"):
        builder.add_node(node, env[node])
    if compress is None:
        builder.add_node("compress_research", lambda _state: {})
    else:
        from langchain_core.messages import filter_messages
        from open_deep_research.prompts import compress_research_simple_human_message, compress_research_system_prompt

        compression_env = {
            **env,
            "configurable_model": CompressionAdapter(exchange.model),
            "HumanMessage": HumanMessage,
            "compress_research_simple_human_message": compress_research_simple_human_message,
            "compress_research_system_prompt": compress_research_system_prompt,
            "filter_messages": filter_messages,
            "is_token_limit_exceeded": local_only,
        }
        source_slice(MODULE / "deep_researcher.py", ("compress_research",), compression_env)
        builder.add_node("compress_research", compression_env["compress_research"])
    builder.add_edge(START, "researcher")
    builder.add_edge("compress_research", END)
    graph = builder.compile(interrupt_before=["compress_research"] if compress is None else None)
    config = {"configurable": {"research_model": exchange.model.model_id, "research_model_max_tokens": MODEL_BUDGET,
                               "max_structured_output_retries": 1, "max_react_tool_calls": exchange.call_limit,
                               "compression_model": exchange.model.model_id,
                               "compression_model_max_tokens": MODEL_BUDGET}}
    state = {"researcher_messages": [HumanMessage(content=exchange.task["question"])],
             "research_topic": exchange.task["question"]}
    updates = []
    summary = None
    with tracing_context(enabled=False):
        async for step in graph.astream(state, config, stream_mode="updates"):
            for node, update in step.items():
                updates.append(node)
                if node == "compress_research":
                    summary = update["compressed_research"]
    if compress is not None:
        return {"nodes": updates, "compressed_research": summary}
    return {"nodes": updates, "boundary": "Kernel stopped before compress_research"}


async def run_stirrup(exchange, path):
    from pydantic import BaseModel
    from stirrup.core.agent import Agent
    from stirrup.core.models import AssistantMessage, TokenUsage, Tool, ToolCall, ToolMessage, ToolResult
    from stirrup.utils.logging import AgentLogger

    class Answer(BaseModel):
        answer: Literal["yes", "no"]
        citation_ids: list[str]

    class PairedClient:
        context_window_tokens = CONTEXT
        model_slug = exchange.model.model_id

        async def generate(self, messages, tools):
            if set(tools) != {READ_TOOL, FINISH_TOOL}:
                raise ValueError("Unexpected tools in the Stirrup agent")
            chat = []
            for message in messages:
                if isinstance(message, ToolMessage):
                    chat.append(tool_feedback(message.name, message.content))
                elif isinstance(message, AssistantMessage):
                    chat.append(("assistant", f"Tool request: {message.blocks[0].name}"))
                else:
                    chat.append((message.role, message.content))
            calls_before = len(exchange.model.calls)
            try:
                name, args, call_id = exchange.generate(chat)
            finally:
                path.extend([STIRRUP_CALL] * (len(exchange.model.calls) - calls_before))
            usage = exchange.model.calls[-1].get("usage") or {}
            return AssistantMessage(
                blocks=[ToolCall(name=name, arguments=json.dumps(args), tool_call_id=call_id)],
                token_usage=TokenUsage(input=usage.get("prompt_tokens") or 0, answer=usage.get("completion_tokens") or 0),
            )

    read_tool = Tool(name=READ_TOOL, description=exchange.read_description,
                     executor=lambda _: ToolResult(content=exchange.read_evidence()))
    finish_tool = Tool(name=FINISH_TOOL, description=FINISH_DESCRIPTION, parameters=Answer,
                       executor=lambda params: ToolResult(content=exchange.submit(**params.model_dump())))
    agent = Agent(PairedClient(), "agenteve-paired", max_turns=exchange.call_limit, system_prompt=exchange.system_prompt,
                  tools=[read_tool], finish_tool=finish_tool, run_sync_in_thread=False,
                  recover_from_context_overflow=False, logger=AgentLogger(show_spinner=False, level=logging.WARNING))
    path.append("__start__")
    async with agent.session(clear_cache_on_success=False, cache_on_interrupt=False) as session:
        answer, history, _ = await session.run(exchange.task["question"])
    path.append("__end__")
    return {"finish_params": answer.model_dump() if answer else None,
            "message_roles": [message.role for batch in history for message in batch]}


async def compare(task, create_model, out, exchange_type=Exchange):
    scores = {}
    for name, runner in (("open-deep-research", run_odr), ("stirrup", run_stirrup)):
        exchange = exchange_type(task, create_model())
        path, details = [], {}
        try:
            details = await runner(exchange, path)
        except (InferenceError, OutputError) as error:
            details["runtime_error"] = str(error)
            print(f"{name}: {error}", file=sys.stderr)
        source_graph = GRAPH_DIR / name / "extracted/flow.json"
        edges = {(edge["source"], edge["target"]) for edge in read(source_graph)["edges"] if edge["kind"] == "next_call"}
        observed = list(zip(path, path[1:]))
        scores[name] = {
            **exchange.grade(),
            "matched_graph_edges": sum(edge in edges for edge in observed),
            "observed_graph_edges": len(observed),
            "completion_observed": exchange.submission is not None,
        }
        write(out / f"{name}.json", {"details": details, "llm_path": path, "missing_edges": [
            edge for edge in observed if edge not in edges],
            "model_requests": exchange.model.calls, "tool_events": exchange.events, "output": exchange.submission,
            "graph_sha256": sha256(source_graph), "scores": scores[name]})
    return scores


async def run_case(task, create_model, out, experiment, fixture_sha256, *, row_index=None, exchange_type=Exchange):
    out.mkdir(parents=True, exist_ok=False)
    scores = await compare(task, create_model, out, exchange_type)
    task_info = {"id": task["row_id"], "question": task["question"], "manifest_sha256": fixture_sha256,
                 "reference": task["reference"]}
    if row_index is not None:
        task_info["row_index"] = row_index
    write(out / "comparison.json", {"task": task_info, "experiment": experiment, "scores": scores})
    for name, score in scores.items():
        pages = f", pages={score['pages_read']}" if "pages_read" in score else ""
        print(f"{name}: answer={score['answer_match']}, citations={score['support_match']}, "
              f"calls={score['model_calls']}{pages}, "
              f"graph={score['matched_graph_edges']}/{score['observed_graph_edges']}",
              flush=True)
    return scores


async def run_batch(batch, create_model, out, experiment, fixture_sha256, *, exchange_type=Exchange, summarize=None):
    cases = batch["cases"]
    out.mkdir(parents=True, exist_ok=False)
    write(out / "batch-protocol.json", {
        **{key: value for key, value in batch.items() if key != "cases"},
        "fixture_sha256": fixture_sha256,
        "experiment": experiment,
        "cases": [{"id": case["row_id"], "row_index": case["row_index"],
                   "comparison": f"{case['row_id']}/comparison.json"} for case in cases],
    })
    results = {}
    for task in cases:
        row_id = task["row_id"]
        print(f"HotpotQA row {task['row_index']}: {row_id}", flush=True)
        results[row_id] = await run_case(
            task, create_model, out / row_id, experiment, fixture_sha256,
            row_index=task["row_index"], exchange_type=exchange_type,
        )

    totals = {}
    for name in ("open-deep-research", "stirrup"):
        scores = [result[name] for result in results.values()]
        totals[name] = {
            "cases": len(cases),
            "completed": sum(score["completion_observed"] for score in scores),
            "evidence_read_once": sum(score["shared_evidence_once"] for score in scores),
            "answer_matches": sum(score["answer_match"] for score in scores),
            "support_matches": sum(score["support_match"] for score in scores),
            "joint_matches": sum(score["answer_match"] and score["support_match"] for score in scores),
            "model_calls": sum(score["model_calls"] for score in scores),
            "matched_graph_edges": sum(score["matched_graph_edges"] for score in scores),
            "observed_graph_edges": sum(score["observed_graph_edges"] for score in scores),
        }
        if summarize is not None:
            totals[name].update(summarize(scores))
        count = totals[name]
        print(f"{name}: completed={count['completed']}/{count['cases']}, "
              f"answer={count['answer_matches']}/{count['cases']}, "
              f"citations={count['support_matches']}/{count['cases']}, "
              f"joint={count['joint_matches']}/{count['cases']}, "
              f"graph={count['matched_graph_edges']}/{count['observed_graph_edges']}")
    write(out / "batch-summary.json", {"harnesses": totals, "case_scores": results})
    return all(count["completed"] == count["cases"] for count in totals.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("local", "deepseek"), default="local")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--batch", action="store_true", help="Run the six frozen HotpotQA validation cases")
    mode.add_argument("--branches", action="store_true", help="Check tool loops and exits with scripted model responses")
    mode.add_argument("--paged", action="store_true", help="Let the model decide whether to read a second evidence page")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh paired result directory")
    deps = Path(os.environ.get("AGENTEVE_PY_DEPS", ROOT.parent / "agenteve-deps"))
    sys.path[:0] = [str(path) for path in (deps, STIRRUP / "src", STIRRUP / "deps") if path.is_dir()]
    for root, manifest in ((SNAPSHOT, read(SNAPSHOT / "SOURCE.json")), (STIRRUP, read(ROOT / "sources/stirrup/SOURCE.json"))):
        for file in manifest["files"]:
            path = root / (file["file"] if "file" in file else file["path"])
            if sha256(path) != file["sha256"]:
                parser.error(f"Pinned source hash failed for {path}")
    if args.branches:
        if args.backend != "local":
            parser.error("Branch checks use scripted responses")
        from stage2.paired_branches import run_branch_checks

        complete = asyncio.run(run_branch_checks(args.out))
        print(args.out.resolve())
        return int(not complete)

    if args.paged:
        from stage2.paired_paged import PAGED_TASKS, PagedExchange, summarize_paged, validate_paged_tasks

    fixture = PAGED_TASKS if args.paged else (BATCH if args.batch else TASK)
    tasks = read(fixture)
    fixture_sha256 = sha256(fixture)
    if args.batch:
        ids = [case["row_id"] for case in tasks["cases"]]
        if len(ids) != 6 or len(set(ids)) != len(ids) or not all(re.fullmatch(r"[0-9a-f]{24}", row_id) for row_id in ids):
            parser.error("Expected six unique HotpotQA case IDs")
    if args.paged:
        try:
            validate_paged_tasks(tasks["cases"])
        except ValueError as error:
            parser.error(str(error))
    experiment = {"backend": args.backend, "same_evidence": True,
                  "call_limit": PagedExchange.call_limit if args.paged else MODEL_CALL_LIMIT,
                  "max_output_tokens_per_call": MODEL_BUDGET, "temperature": TEMPERATURE}
    if args.backend == "deepseek":
        try:
            key = read_deepseek_key()
        except ValueError as error:
            parser.error(str(error))

        def create_model():
            return DeepSeekPairModel({}, key, token_budget=MODEL_BUDGET)

        experiment.update(model=DEEPSEEK_MODEL, seed=None, thinking_mode="disabled",
                          response_protocol="json_object with a schema in the prompt")
    else:
        model_info = read(CACHE / "runtime.json")
        if sha256(CACHE / model_info["filename"]) != model_info["sha256"]:
            parser.error("Local model weight hash failed")

        def create_model():
            return LocalModel(model_info, SEED, token_budget=MODEL_BUDGET)

        experiment.update(model=model_info["model"], weight_sha256=model_info["sha256"], seed=SEED,
                          response_protocol="llama.cpp JSON-schema constrained output")
    experiment["packages"] = {package: version(package) for package in
                              ("langgraph", "langchain-core", "pydantic", "moviepy", "trafilatura")}
    if args.paged:
        experiment["evidence_mode"] = "read page one; then choose submit or read page two"
        complete = asyncio.run(run_batch(
            tasks, create_model, args.out, experiment, fixture_sha256,
            exchange_type=PagedExchange, summarize=summarize_paged,
        ))
    elif args.batch:
        complete = asyncio.run(run_batch(tasks, create_model, args.out, experiment, fixture_sha256))
    else:
        scores = asyncio.run(run_case(tasks, create_model, args.out, experiment, fixture_sha256))
        complete = all(score["completion_observed"] for score in scores.values())
    print(args.out.resolve())
    return int(not complete)


if __name__ == "__main__":
    raise SystemExit(main())
