"""Analyze source-level reachability and node completion from recorded runs."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from common.callPaths import edgeDropChecks, nextCallAdjacency, provenance, reachablePairs, shortestPaths
from common.fileStore import readJson, writeJson
from common.runLog import info, warn
from src.core.nodePasses import passCounts, writeNodePasses
from src.core.semanticRecord import format_rejected, render_single_run, summarize_single_run


def fraction(count, total):
    return {"n": count, "denominator": total, "rate": count / total if total else None}


def readTrials(path):
    path = Path(path)
    runs = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not runs:
        raise ValueError(f"No recorded runs in {path}")
    return [row if "node_completion" in row else {**row, "node_completion": {}} for row in runs]


def pairStepGrades(source_grades, target_grades, n_runs):
    groups = {key: [] for key in ("both", "source_only", "target_only", "neither")}
    paired = sorted(source_grades.keys() & target_grades.keys())
    categories = {
        (True, True): "both", (True, False): "source_only",
        (False, True): "target_only", (False, False): "neither",
    }
    for run_id in paired:
        groups[categories[source_grades[run_id], target_grades[run_id]]].append(run_id)
    both = len(groups["both"])
    source_count = both + len(groups["source_only"])
    target_count = both + len(groups["target_only"])
    ratio = fraction(target_count, source_count)
    review = None
    if ratio["rate"] is not None:
        if ratio["rate"] > 1:
            review = "above_one"
        elif ratio["rate"] >= 0.95:
            review = "near_one"
    return {
        "paired_assessments": fraction(len(paired), n_runs),
        "completion_count_ratio_q": ratio,
        "target_pass_given_source_pass": fraction(both, source_count),
        "review_signal": review,
        "cancellation": bool(groups["target_only"]) and ratio["rate"] is not None and ratio["rate"] >= 0.95,
        "necessary_condition_candidates": groups["target_only"],
        "sufficient_condition_candidates": groups["source_only"],
        "same_run_groups": groups,
    }


def measureTrials(graph, runs):
    if graph.get("graph_kind") != "llm_call_successors":
        raise ValueError("Reachability requires an AGENTEVE LLM-call graph")
    nodes = {node["id"]: node for node in graph["nodes"]}
    entry = graph["entry"]
    if len(nodes) != len(graph["nodes"]) or entry not in nodes:
        raise ValueError("Duplicate nodes or missing graph entry")
    adjacent = nextCallAdjacency(nodes, (
        (edge["source"], edge["target"], edge["route"], edge["kind"]) for edge in graph["edges"]
    ))
    paths = {node: shortestPaths(adjacent, node) for node in nodes}
    ends = {node for node, spec in nodes.items() if spec["kind"] in {"exit", "terminal"}}
    entry_paths = {**paths[entry], entry: (entry,)}
    seen_runs = set()
    excluded = []
    task_runs = defaultdict(list)
    reached = defaultdict(list)
    visits = Counter()
    grades = defaultdict(dict)
    observed_edges = Counter()
    unmatched_edges = []
    format_failures = Counter()
    for row in runs:
        run_id = row["run_id"]
        if not isinstance(run_id, str) or not run_id or run_id in seen_runs:
            raise ValueError(f"Invalid or repeated run ID: {run_id!r}")
        seen_runs.add(run_id)
        if row.get("infrastructure_error"):
            excluded.append(run_id)
            continue
        path = row["path"]
        completion = row["node_completion"]
        if not isinstance(path, list) or not isinstance(completion, dict):
            raise ValueError(f"{run_id}: expected a path list and node_completion object")
        visited = set(path)
        undeclared = (visited | completion.keys()) - nodes.keys()
        if undeclared:
            raise ValueError(f"{run_id}: undeclared nodes {sorted(undeclared)}")
        task_runs[row["task_id"]].append(run_id)
        visits.update(path)
        for node in visited:
            reached[node].append(run_id)
        for node, passed in completion.items():
            if passed is None:
                continue
            if format_rejected(row, node):
                format_failures[node] += 1
                continue
            if type(passed) is not bool or (passed and node not in visited):
                raise ValueError(f"{run_id}: invalid completion for {node!r}")
            grades[node][run_id] = passed
        for source, target in zip(path, path[1:]):
            observed_edges[source, target] += 1
            if target not in adjacent[source]:
                unmatched_edges.append({"run_id": run_id, "source": source, "target": target})

    n_runs = len(seen_runs) - len(excluded)
    node_results = {}
    for node, spec in nodes.items():
        without_node = shortestPaths({
            other: targets - {node} for other, targets in adjacent.items() if other != node
        }, entry) if node != entry else {}
        assessed = grades[node]
        node_results[node] = {
            "label": spec["label"], "kind": spec["kind"], "source_evidence": spec.get("evidence", []),
            "entry_path": entry_paths.get(node),
            "direct_predecessors": sorted(other for other, targets in adjacent.items() if node in targets),
            "direct_successors": sorted(adjacent[node]),
            "reachable_paths": paths[node],
            "cycle_path": paths[node].get(node),
            "exit_paths_without_node": {end: without_node[end] for end in sorted(ends & without_node.keys())},
            "exit_paths_lost_without_node": sorted((ends & entry_paths.keys()) - without_node.keys()),
            "observed_runs": reached[node], "visit_count": visits[node],
            "completion": fraction(sum(assessed.values()), len(assessed)),
            "format_failures": format_failures[node],
            "unassessed_runs": n_runs - len(assessed),
            "completion_by_task": {
                task: {
                    **fraction(sum(assessed[run] for run in ids if run in assessed),
                               sum(run in assessed for run in ids)),
                    "scored_runs": len(ids),
                }
                for task, ids in task_runs.items()
            } if assessed else {},
        }
    edge_results = []
    for source, targets in adjacent.items():
        for target in sorted(targets):
            row = {"source": source, "target": target, "observed_transitions": observed_edges[source, target]}
            # A loop revisits a call site, but its completion is assessed once per run.
            if source != target and grades[source].keys() & grades[target].keys():
                row.update(pairStepGrades(grades[source], grades[target], n_runs))
            edge_results.append(row)
    return {
        "entry": entry,
        "graph_scope": graph.get("analysis_scope", graph.get("scope")),
        "structural_reachable_pairs": sum(map(len, paths.values())),
        "scored_runs": n_runs, "tasks": len(task_runs),
        "excluded_infrastructure_runs": excluded,
        "nodes": node_results, "edges": edge_results,
        "unmatched_observed_transitions": unmatched_edges,
        "edge_drops": _edgeDrops(adjacent),
    }


def _edgeDrops(adjacent):
    pairs = reachablePairs(adjacent)
    edges = []
    for item in edgeDropChecks(adjacent, pairs):
        edges.append({
            "edge": list(item["edge"]),
            "alternative_path": list(item["alternative_path"]) if item["alternative_path"] else None,
            "lost_pairs": [list(pair) for pair in item["lost_pairs"]],
        })
    return {"reachable_pairs": len(pairs), "edges": edges}


def show_rate(value):
    if value["rate"] is None:
        return "未定义"
    return f"{value['n']}/{value['denominator']}（{value['rate']:.1%}）"


def renderReport(result):
    nodes = result["nodes"]
    lines = [
        "# 节点可达性与完成度",
        "",
        f"顺序可达对 {result['structural_reachable_pairs']} 个。可用运行 {result['scored_runs']} 次，"
        f"覆盖 {result['tasks']} 个任务。已排除 {len(result['excluded_infrastructure_runs'])} 次基础设施故障。",
        "",
        "| 节点 | 从入口可达 | 下游可达节点数 | 回路 | 观测到达次数（运行/访问） | 节点完成率 |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in nodes.values():
        completion = row["completion"]
        rate = show_rate(completion) if completion["denominator"] else "未评估"
        if completion["denominator"] and row["unassessed_runs"]:
            rate += f"（另有 {row['unassessed_runs']} 次未评估）"
        lines.append(f"| {row['label']} | {'是' if row['entry_path'] else '否'} | {len(row['reachable_paths'])} | "
                     f"{'有' if row['cycle_path'] else '无'} | {len(row['observed_runs'])}/{row['visit_count']} | {rate} |")

    lines.extend(["", "## 相邻节点的逐次配对", ""])
    paired = [edge for edge in result["edges"] if "paired_assessments" in edge]
    if not paired:
        lines.append("本次没有相邻节点同时具备验收记录。")
    signals = {"above_one": "比值超过 100%，检查依赖与验收口径", "near_one": "比值接近 100%，回查配对结果"}
    for edge in paired:
        source, target = (nodes[edge[key]]["label"] for key in ("source", "target"))
        lines.extend([
            f"### {source} → {target}", "",
            f"- 同时有验收结果的运行 {show_rate(edge['paired_assessments'])}。",
            f"- 完成个数之比 Q 为 {show_rate(edge['completion_count_ratio_q'])}。",
            f"- 上游通过时下游也通过的比例为 {show_rate(edge['target_pass_given_source_pass'])}。",
            "- 必要性候选反例（上游未通过、下游通过） "
            + (", ".join(edge["necessary_condition_candidates"]) or "本批未观察到") + "。",
            "- 充分性候选反例（上游通过、下游未通过） "
            + (", ".join(edge["sufficient_condition_candidates"]) or "本批未观察到") + "。",
        ])
        if edge["review_signal"]:
            lines.append(f"- {signals[edge['review_signal']]}。")
        if edge.get("cancellation"):
            lines.append("- 完成个数之比接近或超过 100%，同时存在只完成下游的试验，两个差集会对消条件通过率。")
        lines.append("")
    counts = result.get("pass_counts") or {}
    lines.extend(["## 原始通过记录", "",
                  "每次到达和每次验收各写一行，保存在 nodePasses.jsonl。下表按这些行计数，不把重复访问并成一次。",
                  "",
                  "| 节点 | 访问次数 | 通过 | 未通过 | 未验收 |",
                  "|---|---:|---:|---:|---:|"])
    for node, row in nodes.items():
        count = counts.get(node, {"visits": 0, "passed": 0, "failed": 0, "unassessed": 0})
        lines.append(f"| {row['label']} | {count['visits']} | {count['passed']} | {count['failed']} | {count['unassessed']} |")
    drops = result.get("edge_drops") or {"reachable_pairs": 0, "edges": []}
    losing = [item for item in drops["edges"] if item["lost_pairs"]]
    lines.extend(["", "## 删边后的可达性", "",
                  f"顺序可达对 {drops['reachable_pairs']} 个。去掉一条边后会失去可达对的边有 {len(losing)} 条。",
                  ""])
    for item in losing:
        alternative = " → ".join(item["alternative_path"]) if item["alternative_path"] else "无替代路径"
        lines.append(f"- {' → '.join(item['edge'])}：失去 {len(item['lost_pairs'])} 对，替代路径为 {alternative}。")
    lines.extend([
        "",
        "静态可达性依据 next_call 边计算。JSON 保留最短路径、逐任务完成频次、"
        "节点移除后的终点可达性和各类运行编号。",
        "配对结果适用于该批次的验收规则。节点职责或冗余需要在调整设计后用独立任务复测。",
    ])
    return "\n".join(lines) + "\n"


def trialOutcome(row):
    if isinstance(row.get("outcome"), bool):
        return row["outcome"]
    brief, supervisor = row.get("brief"), row.get("supervisor")
    if isinstance(brief, dict) and isinstance(supervisor, dict):
        return bool(brief.get("complete") and supervisor.get("complete"))
    return None


def locateLogicFaults(result):
    """Locate nodes that break static reachability or upstream-downstream order.

    The call graph and recorded pass values are the only inputs. No model is called.
    """
    nodes = result["nodes"]
    faults = []
    for node, row in nodes.items():
        if not row["entry_path"]:
            faults.append({"node": node, "reason": "unreachable"})
    for edge in result["edges"]:
        source, target = edge["source"], edge["target"]
        source_rate = nodes[source]["completion"]["rate"]
        target_rate = nodes[target]["completion"]["rate"]
        if source_rate is not None and target_rate is not None and target_rate > source_rate:
            faults.append({"node": target, "upstream": source, "reason": "higher_completion_than_upstream"})
        if edge.get("necessary_condition_candidates"):
            faults.append({"node": target, "upstream": source, "reason": "passed_without_upstream"})
        ratio = (edge.get("completion_count_ratio_q") or {}).get("rate")
        if ratio is not None and ratio > 1:
            faults.append({"node": target, "upstream": source, "reason": "completion_ratio_above_one"})
    return faults


def logBaseline(result):
    """Print each node score and the statically located logic faults."""
    nodes = result["nodes"]
    for row in nodes.values():
        completion = row["completion"]
        rate = show_rate(completion) if completion["denominator"] else "未评估"
        reached = "从入口可达" if row["entry_path"] else "从入口不可达"
        info(f"节点 {row['label']} {reached} 完成率 {rate}")
    reasons = {
        "unreachable": "调用图上从入口不可达",
        "higher_completion_than_upstream": "完成率高于直接上游",
        "passed_without_upstream": "直接上游未通过时本节点仍通过",
        "completion_ratio_above_one": "完成个数之比高于直接上游",
    }
    for fault in locateLogicFaults(result):
        label = nodes[fault["node"]]["label"]
        if "upstream" in fault:
            upstream = nodes[fault["upstream"]]["label"]
            warn(f"逻辑异常 {label}：相对 {upstream}，{reasons[fault['reason']]}")
        else:
            warn(f"逻辑异常 {label}：{reasons[fault['reason']]}")


def postEvaluation(callGraphPath, trialsPath, outDir, *, improve=None, generator=None, node_runner=None):
    """Read a call graph and trial records. Write results, step data, trajectories, and a report."""
    if isinstance(improve, dict) and improve.get("enabled"):
        from improve.settings import parse_improve
        parse_improve(improve)
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=False)
    graph = readJson(callGraphPath)
    runs = readTrials(trialsPath)
    measured = measureTrials(graph, runs)
    measured["pass_counts"] = passCounts(writeNodePasses(outDir / "nodePasses.jsonl", runs))
    byTask = {}
    outcomes = []
    trajectories = []
    for row in runs:
        outcome = None if row.get("infrastructure_error") else trialOutcome(row)
        if outcome is not None:
            outcomes.append(outcome)
            stat = byTask.setdefault(row["task_id"], {"passed": 0, "trials": 0})
            stat["trials"] += 1
            stat["passed"] += int(bool(outcome))
        completion = row.get("node_completion") or {}
        firstFailed = next((node for node in row.get("path") or [] if completion.get(node) is False), None)
        trajectories.append({
            "run_id": row.get("run_id"),
            "task_id": row.get("task_id"),
            "path": row.get("path"),
            "first_failed_step": firstFailed,
            "outcome": outcome,
            "infrastructure_error": row.get("infrastructure_error"),
        })
    for stat in byTask.values():
        stat["any_passed"] = stat["passed"] > 0
        stat["all_passed"] = stat["passed"] == stat["trials"]
    for edge in measured["edges"]:
        if edge.get("cancellation"):
            warn(f"{edge['source']} -> {edge['target']}: completion ratio near 1 hides a lower conditional pass rate",
                 once=f"{edge['source']}->{edge['target']}")
    writeJson(outDir / "results.json", {
        "graph": provenance(callGraphPath),
        "trials": provenance(trialsPath),
        "scored_runs": measured["scored_runs"],
        "tasks": byTask,
        "outcomes_recorded": len(outcomes),
        "trial_pass_rate": fraction(sum(bool(item) for item in outcomes), len(outcomes)) if outcomes else None,
    })
    measured["single_run"] = summarize_single_run(runs, measured)
    for item, snapshot in zip(trajectories, measured["single_run"]["runs"]):
        item["first_incomplete_node"] = snapshot["first_incomplete_node"]
        item["format_ok"] = snapshot["format_ok"]
        item["semantic_retention"] = snapshot["semantic_retention"]
        item["unsupported_share"] = snapshot["unsupported_share"]
    writeJson(outDir / "stepData.json", measured)
    writeJson(outDir / "trajectories.json", {
        "runs": trajectories,
        "unmatched_transitions": measured["unmatched_observed_transitions"],
    })
    report = renderReport(measured) + render_single_run(measured["single_run"])
    report += _write_improve(improve, runs, measured, outDir, generator, node_runner)
    (outDir / "report.md").write_text(report, encoding="utf-8")
    logBaseline(measured)
    print(f"{measured['scored_runs']} scored runs, {len(measured['nodes'])} nodes -> {outDir / 'report.md'}")
    return 0


def _write_improve(improve, runs, measured, outDir, generator, node_runner):
    """Append improve only when it is enabled. A closed switch does not call a generator."""
    if not isinstance(improve, dict) or not improve.get("enabled"):
        return ""
    from improve.runImprove import run_improve
    from improve.settings import parse_improve
    result = run_improve(parse_improve(improve), runs, measured, generator=generator, node_runner=node_runner)
    writeJson(outDir / "improve.json", result["data"])
    (outDir / "improve.md").write_text(result["markdown"], encoding="utf-8")
    return result["markdown"]


