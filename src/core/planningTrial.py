"""Run and score the Open Deep Research brief-to-supervisor pilot."""

import ast
import asyncio
from getpass import GetPassWarning, getpass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Literal
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request as HTTPRequest, build_opener
import warnings

from common.fileStore import fileSha256 as sha256, readJson as read, writeJson as write
from src.core.nodePasses import writeNodePasses
from src.core.semanticRecord import heads_found_in_text, unsupported_of

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "instances/odr"
MODULE = SNAPSHOT / "open_deep_research"
CACHE = ROOT / "stage2/model-cache"
TASKS = ROOT / "tasks/planningPilot.json"
BRIEF = "write_research_brief:154:22"
SUPERVISOR = "supervisor:214:22"
DATE = "Wed Nov 18, 2026"
TOKEN_BUDGET = 288
TEMPERATURE = 0.5
CONTEXT = 4096
THREADS = 6
BATCH_SIZE = 256
MICRO_BATCH_SIZE = 128
MAX_SUPERVISOR_CALLS = 3
MAX_DELEGATIONS = 2
LOCAL_MAX_TOOL_CALLS = 2
API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-flash"


class OutputError(ValueError):
    """Generated output cannot be used by the next operation."""


class InferenceError(RuntimeError):
    """Local model was unable to run."""


def source_slice(filename, names, env):
    """Load only the selected upstream functions, preserving source coordinates."""
    tree = ast.parse(filename.read_text(encoding="utf-8"), filename=str(filename))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"Expected functions {names} in {filename.name}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(filename), "exec"), env)
    return env


def upstream_functions(model):
    sys.path.insert(0, str(SNAPSHOT / "src"))
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import tool
    from langgraph.types import Command
    from open_deep_research.configuration import Configuration
    from open_deep_research.prompts import lead_researcher_prompt, transform_messages_into_research_topic_prompt
    from open_deep_research.state import AgentState, ConductResearch, ResearchComplete, ResearchQuestion, SupervisorState

    env = source_slice(MODULE / "utils.py", ("think_tool",), {"tool": tool, "__name__": "pilot_tools"})
    env.update(locals())
    env.update({"Literal": Literal, "configurable_model": model, "get_today_str": lambda: DATE,
                "get_api_key_for_model": lambda _name, _config: None})
    return source_slice(MODULE / "deep_researcher.py", ("write_research_brief", "supervisor"), env)


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def chatml(messages):
    return ("".join(f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in messages)
            + "<|im_start|>assistant\n")


class PilotModel:
    def __init__(self, provenance, seed, *, token_budget=TOKEN_BUDGET):
        self.provenance = provenance
        self.seed = seed
        self.token_budget = token_budget
        self.calls = []

    def with_structured_output(self, schema):
        return self.request(schema=schema)

    def bind_tools(self, tools):
        fields = {item.__name__: item for item in tools if isinstance(item, type)}
        for item in tools:
            if not isinstance(item, type):
                fields[item.name] = item.args_schema
        return self.request(tools=fields)


class LocalModel(PilotModel):
    """Use upstream prompts and translate local JSON generation into LangChain messages."""

    model_id = "local:qwen2.5-0.5b-instruct-q4_k_m"

    def request(self, **options):
        return Request(self, **options)

    def generate(self, messages, schema, phase):
        seed = self.seed + 1009 * len(self.calls)
        prompt = chatml(messages)
        event = {"phase": phase, "seed": seed, "messages": messages,
                 "max_output_tokens": self.token_budget}
        self.calls.append(event)
        with tempfile.TemporaryDirectory(prefix="inference-", dir=CACHE) as scratch:
            directory = Path(scratch)
            prompt_path = directory / "prompt.txt"
            schema_path = directory / "schema.json"
            prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
            write(schema_path, schema)
            cmd = [
                str(CACHE / "llama-b7035/llama-cli.exe"), "-m", self.provenance["filename"],
                "--no-conversation", "--no-display-prompt", "--no-warmup",
                "-t", str(THREADS), "-c", str(CONTEXT), "-ngl", "0",
                "-b", str(BATCH_SIZE), "-ub", str(MICRO_BATCH_SIZE), "-fa", "on", "--no-repack",
                "-s", str(seed), "--temp", str(TEMPERATURE), "-n", str(self.token_budget),
                "--simple-io", "-f", str(prompt_path.relative_to(CACHE)),
                "--json-schema-file", str(schema_path.relative_to(CACHE)),
            ]
            started = perf_counter()
            try:
                run = subprocess.run(cmd, cwd=CACHE, stdin=subprocess.DEVNULL,
                                     capture_output=True, encoding="utf-8", errors="replace",
                                     timeout=75, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (subprocess.TimeoutExpired, OSError) as error:
                event["elapsed_ms"] = round(1000 * (perf_counter() - started))
                event["runtime_error"] = type(error).__name__
                raise InferenceError(type(error).__name__) from error
            event["elapsed_ms"] = round(1000 * (perf_counter() - started))
        event["raw_output"] = run.stdout.strip()
        token_counts = {
            "prompt_tokens": r"prompt eval time\s*=.+?/\s*(\d+)\s+tokens",
            "decode_steps": r"(?m)^llama_perf_context_print:\s+eval time\s*=.+?/\s*(\d+)\s+runs",
        }
        for name, pattern in token_counts.items():
            matched = re.search(pattern, run.stderr)
            event[name] = int(matched.group(1)) if matched else None
        if run.returncode or not re.search(r"prompt eval time", run.stderr):
            event["runtime_error"] = run.stderr[-1800:]
            raise InferenceError(f"llama.cpp exited {run.returncode} without a valid inference result")
        try:
            data, _ = json.JSONDecoder().raw_decode(event["raw_output"])
            if not isinstance(data, dict):
                raise OutputError("Expected a JSON object")
            return data
        except json.JSONDecodeError as error:
            raise OutputError("Invalid or truncated JSON") from error


class Request:
    def __init__(self, model, *, schema=None, tools=None):
        self.model = model
        self.schema = schema
        self.tools = tools

    def with_retry(self, *, stop_after_attempt):
        if stop_after_attempt != 1:
            raise ValueError("Pilot requires one attempt per model invocation")
        return self

    def with_config(self, config):
        if config["model"] != self.model.model_id or config["max_tokens"] != self.model.token_budget:
            raise ValueError("Pilot model or token budget differs from the fixed configuration")
        return self

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage, ToolMessage

        if self.schema:
            instruction = 'Return only a JSON object with one string field named "research_brief".'
            schema = object_schema({"research_brief": {"type": "string"}})
            result = self.model.generate([("system", instruction), ("user", messages[0].content)], schema, "brief")
            try:
                return self.schema.model_validate(result)
            except ValueError as error:
                raise OutputError("Invalid research brief schema") from error

        schema = object_schema({
            "tool_calls": {
                "type": "array", "maxItems": LOCAL_MAX_TOOL_CALLS,
                "items": object_schema({"name": {"enum": list(self.tools)}, "input": {"type": "string"}}),
            },
        })
        instruction = (
            '\nReturn only JSON: {"tool_calls":[{"name":"TOOL_NAME","input":"TOOL_INPUT"}]}. '
            "For think_tool the input is a planning reflection. For ConductResearch the input is a "
            'standalone research topic. For ResearchComplete use an empty input string. '
            "If a reflection result is already present, proceed with research delegation when appropriate."
        )
        prepared = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                prepared.append(("user", f"Tool result ({msg.name}): {msg.content}"))
            elif isinstance(msg, AIMessage):
                previous = [{"name": call["name"], "input": next(iter(call["args"].values()), "")}
                            for call in msg.tool_calls]
                prepared.append(("assistant", json.dumps({"tool_calls": previous}, ensure_ascii=False)))
            else:
                prepared.append(("system" if msg.type == "system" else "user",
                                 msg.content + (instruction if msg.type == "system" else "")))
        data = self.model.generate(prepared, schema, "supervisor")
        calls = []
        try:
            raw_calls = data["tool_calls"]
            if not isinstance(raw_calls, list) or len(raw_calls) > LOCAL_MAX_TOOL_CALLS:
                raise ValueError("Invalid tool batch")
            for n, call in enumerate(raw_calls, 1):
                name, value = call["name"], call["input"]
                if not isinstance(value, str):
                    raise TypeError("Tool input must be a string")
                fields = {"think_tool": {"reflection": value}, "ConductResearch": {"research_topic": value},
                          "ResearchComplete": {}}[name]
                arguments = self.tools[name].model_validate(fields).model_dump()
                calls.append({"name": name, "args": arguments, "id": f"response-{len(self.model.calls)}-{n}"})
        except (KeyError, TypeError, ValueError) as error:
            raise OutputError("Invalid tool-call structure") from error
        return AIMessage(content="", tool_calls=calls)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DeepSeekModel(PilotModel):
    """Send the pilot prompts to DeepSeek and use its native function calls."""

    model_id = DEEPSEEK_MODEL

    def __init__(self, provenance, api_key, *, token_budget=TOKEN_BUDGET):
        super().__init__(provenance, None, token_budget=token_budget)
        self.api_key = api_key
        self.opener = build_opener(NoRedirect())

    def request(self, **options):
        return DeepSeekRequest(self, **options)

    def generate(self, messages, phase, **options):
        body = {
            "model": self.model_id, "messages": messages, "stream": False,
            "temperature": TEMPERATURE, "max_tokens": self.token_budget,
            "thinking": {"type": "disabled"}, **options,
        }
        event = {"phase": phase, "seed": None, "messages": messages,
                 "max_output_tokens": self.token_budget}
        self.calls.append(event)
        req = HTTPRequest(
            DEEPSEEK_URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        start = perf_counter()
        try:
            with self.opener.open(req, timeout=45) as response:
                result = json.loads(response.read(1024 * 1024))
        except HTTPError as error:
            advice = {
                401: "Check that the full key is valid and has not been revoked.",
                402: "Check the balance in the DeepSeek API platform.",
                429: "DeepSeek is rate-limiting requests. Try again later.",
            }.get(error.code, "Check DeepSeek API status and the account configuration.")
            event["http_status"] = error.code
            raise InferenceError(f"DeepSeek HTTP {error.code}. {advice}") from None
        except (URLError, OSError, ValueError) as error:
            raise InferenceError(f"DeepSeek connection or invalid API response ({type(error).__name__})") from None
        finally:
            event["elapsed_ms"] = round(1000 * (perf_counter() - start))

        try:
            choice = result["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("Unexpected message format")
            event.update({
                "finish_reason": choice.get("finish_reason"), "provider_model": result.get("model"),
                "request_id": result.get("id"), "usage": result.get("usage"),
                "raw_output": {"content": message.get("content"), "tool_calls": message.get("tool_calls")},
            })
        except (IndexError, KeyError, TypeError):
            raise InferenceError("DeepSeek response has no valid choice") from None
        if event["finish_reason"] in {"insufficient_system_resource", "aborted"}:
            raise InferenceError(f"DeepSeek stopped with {event['finish_reason']}")
        if event["finish_reason"] == "length":
            raise OutputError(f"DeepSeek output reached the {self.token_budget}-token limit (finish_reason=length)")
        return message


class DeepSeekRequest(Request):
    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.utils.function_calling import convert_to_openai_tool

        prepared = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                prepared.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
            elif isinstance(msg, AIMessage):
                calls = [{"id": call["id"], "type": "function",
                          "function": {"name": call["name"], "arguments": json.dumps(call["args"], ensure_ascii=False)}}
                         for call in msg.tool_calls]
                item = {"role": "assistant", "content": msg.content}
                if calls:
                    item["tool_calls"] = calls
                prepared.append(item)
            else:
                prepared.append({"role": "system" if msg.type == "system" else "user", "content": msg.content})

        if self.schema:
            prepared.insert(0, {"role": "system",
                "content": 'Return only JSON matching this example: {"research_brief":"A detailed research question"}.'})
            message = self.model.generate(prepared, "brief", response_format={"type": "json_object"})
            try:
                return self.schema.model_validate_json(message.get("content") or "")
            except ValueError:
                raise OutputError("Invalid or truncated research brief JSON") from None

        tools = []
        for name, schema in self.tools.items():
            tool = convert_to_openai_tool(schema)
            tool["function"]["name"] = name
            tools.append(tool)
        message = self.model.generate(prepared, "supervisor", tools=tools, tool_choice="auto")
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise OutputError("Expected tool_calls to be a list")
        calls = []
        try:
            for call in raw_calls:
                fn = call["function"]
                name = fn["name"]
                args = self.tools[name].model_validate_json(fn["arguments"]).model_dump()
                calls.append({"name": name, "args": args, "id": call["id"]})
            return AIMessage(content=message.get("content") or "", tool_calls=calls)
        except (KeyError, TypeError, ValueError):
            raise OutputError("Invalid tool-call structure") from None


def read_deepseek_key():
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise ValueError(f"Set {API_KEY_ENV} to the API key for the model named in config.json.")
    return key


def matchRequiredPhrases(output, required):
    value = unicodedata.normalize("NFKC", output).casefold()
    checks = {}
    for name, aliases in required.items():
        hits = [alias for alias in aliases
                if re.search(r"(?<!\w)" + re.escape(unicodedata.normalize("NFKC", alias).casefold()) + r"(?!\w)", value)]
        checks[name] = hits
    return checks


def _attach_semantic_heads(task, result):
    """Record the four section-3.1 fields. Alias match only; no entailment model."""
    required = task.get("required") or {}
    prompt = str(task.get("prompt") or "")
    brief = result.get("brief") or {}
    supervisor = result.get("supervisor") or {}
    user_heads = heads_found_in_text(required, prompt)
    brief_text = str(brief.get("output") or "")
    brief_out = heads_found_in_text(required, brief_text)
    topic_text = "\n".join(str(item) for item in (supervisor.get("topics") or []))
    supervisor_out = heads_found_in_text(required, topic_text)
    brief_inputs = {"user": user_heads}
    supervisor_inputs = {"user": user_heads, BRIEF: brief_out}
    brief_error = brief.get("output_error")
    supervisor_error = supervisor.get("output_error")
    result["semantic_heads"] = {
        BRIEF: {
            "inputs": brief_inputs, "output": brief_out,
            "unsupported": unsupported_of(brief_inputs, brief_out),
            "format_ok": brief_error is None, "format_error": brief_error,
            "prompt": prompt, "raw_output": brief_text,
        },
        SUPERVISOR: {
            "inputs": supervisor_inputs, "output": supervisor_out,
            "unsupported": unsupported_of(supervisor_inputs, supervisor_out),
            "format_ok": supervisor_error is None, "format_error": supervisor_error,
            "prompt": topic_text or brief_text, "raw_output": topic_text,
        },
    }
    errors = [text for text in (brief_error, supervisor_error) if text]
    result["format_ok"] = not errors
    if errors:
        result["format_error"] = errors[0]


async def runPlanningTrial(task, repeat, model):
    from langchain_core.messages import HumanMessage, ToolMessage, message_to_dict

    upstream = upstream_functions(model)
    config = {"configurable": {
        "research_model": model.model_id,
        "research_model_max_tokens": model.token_budget,
        "max_structured_output_retries": 1,
        "max_concurrent_research_units": MAX_DELEGATIONS,
        "max_researcher_iterations": MAX_SUPERVISOR_CALLS,
    }}
    used = upstream["Configuration"].from_runnable_config(config)
    for field, expected in config["configurable"].items():
        if getattr(used, field) != expected:
            raise ValueError(f"External configuration overrides fixed pilot field {field}")
    result = {"task_id": task["id"], "run_id": f"{task['id']}-R{repeat}", "seed": model.seed,
              "path": [], "model_calls": model.calls, "infrastructure_error": None,
              "brief": {"entered": False, "complete": False},
              "supervisor": {"entered": False, "complete": False, "rounds": []}}
    a, b = result["brief"], result["supervisor"]
    a["entered"] = True
    result["path"].append(BRIEF)
    active = a
    try:
        brief = await upstream["write_research_brief"]({"messages": [HumanMessage(content=task["prompt"])]}, config)
        if brief.goto != "research_supervisor":
            raise ValueError(f"Unexpected upstream route: {brief.goto}")
        a["output"] = brief.update["research_brief"]
        a["checks"] = matchRequiredPhrases(a["output"], task["required"])
        a["complete"] = all(a["checks"].values())
        a["route"] = brief.goto
        messages = list(brief.update["supervisor_messages"]["value"])
        b["input_messages"] = [message_to_dict(message) for message in messages]
        b["input_matches_brief"] = messages[-1].content == a["output"]
        assert b["input_matches_brief"]
        active = b
        for _ in range(MAX_SUPERVISOR_CALLS):
            b["entered"] = True
            result["path"].append(SUPERVISOR)
            route = await upstream["supervisor"](
                {"supervisor_messages": messages, "research_iterations": len(b["rounds"])}, config)
            if route.goto != "supervisor_tools":
                raise ValueError(f"Unexpected supervisor route: {route.goto}")
            response = route.update["supervisor_messages"][0]
            calls = response.tool_calls
            b["rounds"].append({"calls": calls, "route": route.goto})
            if not calls or any(call["name"] == "ResearchComplete" for call in calls):
                b["stop_reason"] = "no_tool_call_or_research_complete"
                break
            topics = [call["args"]["research_topic"] for call in calls if call["name"] == "ConductResearch"]
            if topics:
                if len(topics) > MAX_DELEGATIONS:
                    raise OutputError(f"Received {len(topics)} ConductResearch calls. Maximum is {MAX_DELEGATIONS}.")
                # Every earlier completed round ran only planning tools.
                b["planning_order_ok"] = len(b["rounds"]) > 1 and all(call["name"] != "think_tool" for call in calls)
                b["topics"] = topics
                b["checks"] = matchRequiredPhrases("\n".join(topics), task["required"])
                b["actionable"] = all(len(re.findall(r"\b\w+\b", topic)) >= 8 for topic in topics)
                b["complete"] = b["actionable"] and all(b["checks"].values())
                b["stop_reason"] = "first_delegation_batch"
                break
            messages.append(response)
            for call in calls:
                text = upstream["think_tool"].invoke(call["args"])
                messages.append(ToolMessage(content=text, name=call["name"], tool_call_id=call["id"]))
        else:
            b["stop_reason"] = "supervisor_round_budget"
    except OutputError as error:
        active["output_error"] = str(error)
        if active is b:
            b["stop_reason"] = "model_output_error"
    except InferenceError as error:
        result["infrastructure_error"] = str(error)
    result["elapsed_model_ms"] = sum(call.get("elapsed_ms", 0) for call in model.calls)
    result["node_completion"] = {BRIEF: a["complete"], SUPERVISOR: b["complete"]}
    _attach_semantic_heads(task, result)
    return result


def recordPlanningTrials(args):
    """Run the planning pilot and write protocol.json plus trials.jsonl."""
    if not 1 <= args.max_output_tokens <= CONTEXT:
        raise ValueError(f"--max-output-tokens must be from 1 to {CONTEXT}")
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh output directory to preserve previous runs")
    task_data = args.tasks.read_bytes()
    dataset = json.loads(task_data)
    tasks, seeds = dataset["tasks"], dataset["seeds"]
    if not tasks or not seeds:
        raise ValueError("The task manifest must contain at least one task and seed")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Task IDs in the manifest must be unique")
    deps = Path(os.environ.get("AGENTEVE_PY_DEPS", ROOT.parent / "agenteve-deps"))
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    manifest = SNAPSHOT / "SOURCE.json"
    snapshot = {"files": []}
    if manifest.exists():
        snapshot = read(manifest)
        for item in snapshot["files"]:
            relative = item.get("file") or item.get("path")
            if sha256(SNAPSHOT / relative) != item["sha256"]:
                raise ValueError(f"Upstream source hash failed: {relative}")
    local = args.backend == "local"
    if local:
        model = read(CACHE / "runtime.json")
        if sha256(CACHE / model["filename"]) != model["sha256"]:
            raise ValueError("Local model weight hash failed")
        model_settings = {"context_tokens": CONTEXT, "threads": THREADS, "batch_size": BATCH_SIZE,
                          "micro_batch_size": MICRO_BATCH_SIZE, "flash_attention": True, "weight_repacking": False,
                          "supervisor_json_max_tool_calls": LOCAL_MAX_TOOL_CALLS}
    else:
        api_key = read_deepseek_key()
        model = {"model": DEEPSEEK_MODEL, "endpoint": DEEPSEEK_URL, "thinking_mode": "disabled",
                 "brief_format": "json_object", "supervisor_format": "native function calls"}
        model_settings = {"sampling_seed": "not sent; server-controlled sampling"}
    graph = read(args.graph)
    if not any(edge["source"] == BRIEF and edge["target"] == SUPERVISOR for edge in graph["edges"]):
        raise ValueError("The call graph does not include the brief-to-supervisor edge")
    if args.smoke:
        tasks, seeds = tasks[:1], seeds[:1]
    if not local:
        seeds = [None] * len(seeds)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "protocol.json", {
        "experiment": dataset["id"], "sample": dataset["split"], "smoke": args.smoke, "backend": args.backend,
        "tasks_sha256": hashlib.sha256(task_data).hexdigest(), "upstream": snapshot, "model": model,
        "runner_sha256": sha256(Path(__file__)), "python": sys.version.split()[0],
        "packages": {name: version(name) for name in ("langchain-core", "langgraph", "pydantic")},
        "date_in_prompt": DATE, "selected_edge": [BRIEF, SUPERVISOR],
        "seeds": seeds, "task_ids": [task["id"] for task in tasks], "temperature": TEMPERATURE,
        "max_output_tokens_per_call": args.max_output_tokens, **model_settings,
        "max_supervisor_calls": MAX_SUPERVISOR_CALLS, "max_delegations": MAX_DELEGATIONS,
        "scope": "Upstream brief and supervisor functions; only think_tool runs. "
                 "Stop at first delegation batch. Do not execute research, retrieve documents, or write the final report.",
        "rubric": "Prespecified text markers. Supervisor must also emit a valid ConductResearch batch "
                  "with at least eight words per topic. This is a scope-retention proxy.",
        "planning_order_audit": "A prior think_tool round and no think_tool in the first valid ConductResearch batch.",
    })
    runs = []
    with asyncio.Runner() as runner, (output / "trials.jsonl").open("w", encoding="utf-8") as dst:
        for task in tasks:
            for repeat, seed in enumerate(seeds, 1):
                budget = args.max_output_tokens
                backend = (LocalModel(model, seed, token_budget=budget) if local
                           else DeepSeekModel(model, api_key, token_budget=budget))
                row = runner.run(runPlanningTrial(task, repeat, backend))
                runs.append(row)
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                dst.flush()
                print(f"{row['run_id']}: brief={row['brief']['complete']} "
                      f"supervisor={row['supervisor']['complete']} model_calls={len(row['model_calls'])}", flush=True)
                if row["infrastructure_error"]:
                    print(row["infrastructure_error"], file=sys.stderr, flush=True)
                    break
            if runs[-1]["infrastructure_error"]:
                break
    writeNodePasses(output / "nodePasses.jsonl", runs)
    failed = [row["run_id"] for row in runs if row["infrastructure_error"]]
    print(f"{len(runs) - len(failed)}/{len(tasks) * len(seeds)} scored runs -> {output}", flush=True)
    return int(bool(failed))
