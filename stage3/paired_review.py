"""Check a model's explanation against paired HotpotQA tool events."""

import argparse
import json
from os.path import relpath
from pathlib import Path

from stage2.odr_pilot import (
    CACHE, ROOT, DeepSeekModel, InferenceError, LocalModel, OutputError,
    object_schema, read, read_deepseek_key, sha256, write,
)
from stage2.paired_paged import PAGED_TASKS, PAGE_SIZE, validate_paged_tasks
from stage2.paired_pilot import FINISH_TOOL, READ_TOOL


BATCH = ROOT / "results/harness-comparison/paired-hotpotqa-paged-deepseek-v1"
REVIEW_ROWS = (59, 154)
GROUNDED_ROWS = (59, 99, 118, 154)
HARNESSES = {"odr": "open-deep-research", "stirrup": "stirrup"}
HARNESS_LABELS = {"odr": "Open Deep Research 研究员子图", "stirrup": "Stirrup"}
METRICS = ("joint_matches", "second_page_reads_on_split_tasks")
VERDICTS = ("early_submission", "full_read_mismatch", "joint_match")
MODEL_BUDGET = 640
GROUNDED_BUDGET = 1024
LOCAL_SEED = 17
SYSTEM = """你是 AGENTEVE 实验复核员。只根据当前数据判断，返回 JSON。
每题 reference 是标准答案和两个参考编号。答案相同、两个编号都相同（顺序任意）才算联合命中。
reference_pages 列出参考证据所在页。若包含两页，就是跨页题。
metrics 对每套 Harness 分别统计八题的联合命中数，以及四道跨页题中读取第二页的次数。
逐一审查 case_events 里的运行，输出 reviews。每项包含 verdict、last_read 和 submission。
两条编号均已可见且联合命中填 joint_match。提交时仍有参考证据所在页未读填 early_submission。
参考证据所在页已读齐，答案或引用有失误填 full_read_mismatch。
last_read 和 submission 分别填写最后一次 read_evidence 事件 ID 和 ResearchComplete 提交事件 ID。
explanation 用一句中文概括两套 Harness 在指定案例上的差别，并写明证据限于本批。
不要根据读页行为推测模型的内部原因。"""
GROUNDED_SYSTEM = """你是 AGENTEVE 实验复核员。只依据给出的 cases，返回包含 reviews 的 JSON 对象。
每条运行分别判断 answer_match 和 support_match。第一个字段检查答案，第二个检查两个引用编号（顺序任意）。
每条运行用 verdict 解释停止和作答情况。
提交前尚有参考证据所在页未读填 early_submission。
参考证据所在页已读齐，但答案或引用有误，填 full_read_mismatch。
答案和两条引用均命中，且参考证据所在页已读齐，填 joint_match。
last_read 填最后一次 read_evidence 的事件 ID，submission 填 ResearchComplete 的事件 ID。
按事件顺序检查两个 ID，有第三个事件时也要区分它的类型。保持每个字段与该运行的逐条证据一致。"""


def prepare(review_rows=REVIEW_ROWS):
    tasks = read(PAGED_TASKS)["cases"]
    validate_paged_tasks(tasks)
    protocol_file = BATCH / "batch-protocol.json"
    summary_file = BATCH / "batch-summary.json"
    protocol = read(protocol_file)
    summary = read(summary_file)
    if (protocol["fixture_sha256"] != sha256(PAGED_TASKS)
            or protocol["experiment"]["model"] != "deepseek-flash"
            or list(summary["case_scores"]) != [task["row_id"] for task in tasks]):
        raise ValueError("Batch provenance does not match the frozen paged tasks")
    evidence = {"rows": [], "case_events": {}}
    reference = {
        "metrics": {alias: dict.fromkeys(METRICS, 0) for alias in HARNESSES},
        "reviews": {},
        "sources": {file.name: sha256(file) for file in (PAGED_TASKS, protocol_file, summary_file)},
    }
    for task in tasks:
        row_index, row_id = task["row_index"], task["row_id"]
        gold = task["reference"]
        gold_ids = set(gold["citation_ids"])
        pages = [{doc["id"] for doc in task["documents"][start:start + PAGE_SIZE]}
                 for start in range(0, len(task["documents"]), PAGE_SIZE)]
        reference_pages = [i for i, ids in enumerate(pages, 1) if gold_ids & ids]
        row = {"row": row_index, "reference": gold, "reference_pages": reference_pages, "runs": {}}
        evidence["rows"].append(row)
        for alias, name in HARNESSES.items():
            trace_path = BATCH / row_id / f"{name}.json"
            trace = read(trace_path)
            events = trace["tool_events"]
            reads, last = events[:-1], events[-1]
            pages_read = [event["page"] for event in reads]
            output = last["arguments"]
            if (pages_read not in ([1], [1, 2]) or any(event["tool"] != READ_TOOL for event in reads)
                    or last["tool"] != FINISH_TOOL or output != trace["output"]):
                raise ValueError(f"Incomplete or inconsistent evidence trace: {name} row {row_index}")
            for event in reads:
                page = event["page"]
                payload = json.loads(event["result"])
                expected = task["documents"][(page - 1) * PAGE_SIZE:page * PAGE_SIZE]
                if payload["page"] != page or payload["documents"] != expected:
                    raise ValueError(f"Evidence page changed: {name} row {row_index} page {page}")
            scores = summary["case_scores"][row_id][name]
            answer_match = output["answer"] == gold["answer"]
            support_match = sorted(output["citation_ids"]) == sorted(gold["citation_ids"])
            if (answer_match != scores["answer_match"] or support_match != scores["support_match"]
                    or len(pages_read) != scores["pages_read"] or not scores["completion_observed"]):
                raise ValueError(f"Scores disagree with trace: {name} row {row_index}")
            joint = answer_match and support_match
            row["runs"][alias] = {"pages_read": pages_read, "submitted": output}
            reference["metrics"][alias]["joint_matches"] += joint
            reference["metrics"][alias]["second_page_reads_on_split_tasks"] += 2 in reference_pages and 2 in pages_read
            if row_index not in review_rows:
                continue
            run_id = f"{alias}_{row_index}"
            source = trace_path.relative_to(ROOT).as_posix()
            event_ids = [f"{run_id}:e{i}" for i in range(1, len(events) + 1)]
            evidence["case_events"][run_id] = [
                {"event_id": event_id, "tool": READ_TOOL, "page": event["page"]}
                for event_id, event in zip(event_ids[:-1], reads, strict=True)
            ] + [{"event_id": event_ids[-1], "tool": FINISH_TOOL, "submitted": output}]
            verdict = "early_submission"
            if set(reference_pages) <= set(pages_read):
                verdict = "joint_match" if joint else "full_read_mismatch"
            reference["reviews"][run_id] = {
                "verdict": verdict, "last_read": event_ids[-2], "submission": event_ids[-1],
                "trace": source, "trace_sha256": sha256(trace_path),
                "pointers": {event_id: f"/tool_events/{i}" for i, event_id in enumerate(event_ids)},
            }
    if len(reference["reviews"]) != len(review_rows) * len(HARNESSES):
        raise ValueError("Missing one or more preselected review cases")
    return evidence, reference


def review_fields(events):
    event_id = {"enum": [event["event_id"] for event in events]}
    return {"verdict": {"enum": list(VERDICTS)}, "last_read": event_id, "submission": event_id}


def response_schema(evidence):
    count = {"type": "integer", "minimum": 0, "maximum": len(evidence["rows"])}
    metrics = object_schema(dict.fromkeys(HARNESSES, object_schema(dict.fromkeys(METRICS, count))))
    reviews = {}
    for run_id, events in evidence["case_events"].items():
        reviews[run_id] = object_schema(review_fields(events))
    return object_schema({"metrics": metrics, "reviews": object_schema(reviews),
                          "explanation": {"type": "string", "maxLength": 200}})


def audit(answer, reference, *, grounded=False):
    metrics = answer.get("metrics")
    reviews = answer.get("reviews")
    checks = {}
    if not grounded:
        for name, expected in reference["metrics"].items():
            reported = metrics.get(name) if isinstance(metrics, dict) else None
            for key, count in expected.items():
                value = reported.get(key) if isinstance(reported, dict) else None
                checks[f"{name}:{key}"] = type(value) is int and value == count
    for run_id, expected in reference["reviews"].items():
        reported = reviews.get(run_id) if isinstance(reviews, dict) else None
        if not isinstance(reported, dict):
            reported = {}
        checks[f"{run_id}:verdict"] = reported.get("verdict") == expected["verdict"]
        checks[f"{run_id}:event_citations"] = (
            reported.get("last_read") == expected["last_read"] and reported.get("submission") == expected["submission"]
        )
        if grounded:
            for field in ("answer_match", "support_match"):
                checks[f"{run_id}:{field}"] = reported.get(field) is expected[field]
    return checks


def prepare_grounded(evidence, reference):
    rows = {row["row"]: row for row in evidence["rows"]}
    cases = {}
    schemas = {}
    for run_id, events in evidence["case_events"].items():
        alias, index = run_id.split("_")
        row = rows[int(index)]
        gold = row["reference"]
        output = row["runs"][alias]["submitted"]
        reference["reviews"][run_id].update(
            answer_match=output["answer"] == gold["answer"],
            support_match=sorted(output["citation_ids"]) == sorted(gold["citation_ids"]),
        )
        cases[run_id] = {"reference": gold, "reference_pages": row["reference_pages"], "events": events}
        schemas[run_id] = object_schema({
            **review_fields(events), "answer_match": {"type": "boolean"}, "support_match": {"type": "boolean"},
        })
    return {"cases": cases}, object_schema({"reviews": object_schema(schemas)})


def write_grounded_report(out, evidence, reference, result):
    counts = reference["metrics"]
    reviews = reference["reviews"]
    case_checks = ("verdict", "event_citations", "answer_match", "support_match")
    adopted = {
        run_id: all(result["checks"][f"{run_id}:{field}"] for field in case_checks)
        for run_id in reviews
    }
    task_count = len(evidence["rows"])
    split_count = sum(2 in row["reference_pages"] for row in evidence["rows"])
    result["verified_metrics"] = {
        name: {**values, "tasks": task_count, "split_tasks": split_count}
        for name, values in counts.items()
    }
    result["model_adopted"] = adopted

    lines = [
        "# AGENTEVE｜分批证据实验解释报告", "",
        f"本报告基于 {task_count} 道 HotpotQA 比较题和两套 Harness 的成对轨迹。"
        f"下表的数字由程序从原始轨迹重算。逐条核验 {len(reviews)} 条案例轨迹，"
        f"其中 {sum(adopted.values())} 条采用模型判定与引证，其余由程序根据轨迹生成。"
        "每题每套运行一次，案例是本批内选取的示例。", "",
        "| Harness | 答案和两条引用同时正确 | 跨页题读取第二页 |",
        "|---|---:|---:|",
    ]
    for alias, name in HARNESS_LABELS.items():
        tally = counts[alias]
        lines.append(f"| {name} | {tally['joint_matches']}/{task_count} | "
                     f"{tally['second_page_reads_on_split_tasks']}/{split_count} |")
    lines.extend(["", "## 个案解释", "",
                  "| HotpotQA 行号 | Harness | 判定 | 末次读证与提交 | 来源 |",
                  "|---|---|---|---|---|"])
    verdicts = {"early_submission": "参考证据未读齐即提交", "full_read_mismatch": "读齐参考证据页后仍失分",
                "joint_match": "答案与两条引用均命中"}
    for run_id, expected in reviews.items():
        alias, row_index = run_id.split("_")
        use_model = adopted[run_id]
        record = result["answer"]["reviews"][run_id] if use_model else expected
        finding = verdicts[record["verdict"]]
        if record["verdict"] != "joint_match":
            matches = "正确" if record["answer_match"] else "有误"
            citations = "齐全" if record["support_match"] else "未完全命中"
            finding += f"，答案{matches}，引用{citations}"
        event_ids = f"{record['last_read']} → {record['submission']}"
        trace_path = ROOT / expected["trace"]
        try:
            trace_link = Path(relpath(trace_path, out)).as_posix()
        except ValueError:
            trace_link = trace_path.as_uri()
        source = "模型复核通过" if use_model else "程序依据轨迹"
        lines.append(f"| {row_index} | {HARNESS_LABELS[alias]} | {finding} | "
                     f"[{event_ids}](<{trace_link}>) | {source} |")
    lines.extend(["", "事件在原始 JSON 的 `tool_events` 中按顺序排列。"
                  "具体索引见[事件指针](reference.json)，模型原始响应和逐项核验见[复核记录](review.json)。", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("local", "deepseek"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--grounded", action="store_true", help="Compute metrics in code and check case reviews only")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a fresh result directory")
    budget = GROUNDED_BUDGET if args.grounded else MODEL_BUDGET
    try:
        evidence, reference = prepare(GROUNDED_ROWS if args.grounded else REVIEW_ROWS)
        if args.backend == "deepseek":
            model = DeepSeekModel({}, read_deepseek_key(), token_budget=budget)
        else:
            weights = read(CACHE / "runtime.json")
            if sha256(CACHE / weights["filename"]) != weights["sha256"]:
                raise ValueError("Local model weight hash failed")
            model = LocalModel(weights, LOCAL_SEED, token_budget=budget)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    if args.grounded:
        model_evidence, schema = prepare_grounded(evidence, reference)
        system = GROUNDED_SYSTEM
    else:
        model_evidence, schema = evidence, response_schema(evidence)
        system = SYSTEM
    if args.backend == "deepseek":
        system += "\nJSON schema:\n" + json.dumps(schema, separators=(",", ":"))
    args.out.mkdir(parents=True, exist_ok=False)
    json_input = args.out / "model-input.json"
    write(json_input, {"system": system, "evidence": model_evidence, "schema": schema})
    reference_path = args.out / "reference.json"
    write(reference_path, reference)
    user = json.dumps(model_evidence, ensure_ascii=False, separators=(",", ":"))
    result = {"backend": args.backend, "model": model.model_id, "max_output_tokens": budget,
              "input_sha256": sha256(json_input), "reference_sha256": sha256(reference_path)}
    answer = {}
    try:
        messages = [("system", system), ("user", user)]
        if args.backend == "deepseek":
            response = model.generate([{"role": role, "content": text} for role, text in messages],
                                      "paired_review", response_format={"type": "json_object"})
            answer = json.loads(response.get("content") or "")
            if not isinstance(answer, dict):
                answer = {}
                raise OutputError("Expected a JSON object")
        else:
            answer = model.generate(messages, schema, "paired_review")
    except (InferenceError, OutputError, OSError, ValueError) as error:
        result["error"] = str(error)
    checks = audit(answer, reference, grounded=args.grounded)
    result.update(answer=answer, checks=checks, passed=sum(checks.values()), total=len(checks),
                  requests=[{k: v for k, v in call.items() if k != "messages"} for call in model.calls])
    destination = args.out / "review.json"
    if args.grounded:
        write_grounded_report(args.out, evidence, reference, result)
    write(destination, result)
    print(f"{result['passed']}/{result['total']} evidence checks passed -> {destination.resolve()}")
    if args.grounded:
        print(f"{sum(result['model_adopted'].values())}/{len(result['model_adopted'])} case reviews adopted"
              f" -> {(args.out / 'report.md').resolve()}")
    return int("error" in result)


if __name__ == "__main__":
    raise SystemExit(main())
