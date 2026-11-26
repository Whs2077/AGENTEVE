"""Analyze source-level reachability and node completion from recorded runs."""

import json
from collections import Counter, defaultdict

from agenteve import read_json, write_json
from compare_harness_flows import next_call_adjacency, paths_from, provenance


def fraction(count, total):
    return {"n": count, "denominator": total, "rate": count / total if total else None}


def load_runs(path, trace_format):
    runs = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not runs:
        raise ValueError(f"No recorded runs in {path}")
    inputs = {"runs": provenance(path)}
    if trace_format == "odr-pilot":
        protocol_path = path.with_name("protocol.json")
        brief, supervisor = read_json(protocol_path)["selected_edge"]
        inputs["protocol"] = provenance(protocol_path)
        runs = [
            {
                "run_id": row["run_id"],
                "task_id": row["task_id"],
                "path": row["path"],
                "infrastructure_error": row["infrastructure_error"],
                "node_completion": {brief: row["brief"]["complete"], supervisor: row["supervisor"]["complete"]},
            }
            for row in runs
        ]
    return runs, inputs


def score_pair(source_grades, target_grades, n_runs):
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
        "necessary_condition_candidates": groups["target_only"],
        "sufficient_condition_candidates": groups["source_only"],
        "same_run_groups": groups,
    }


def analyze(graph, runs):
    if graph.get("graph_kind") != "llm_call_successors":
        raise ValueError("Reachability requires an AGENTEVE LLM-call graph")
    nodes = {node["id"]: node for node in graph["nodes"]}
    entry = graph["entry"]
    if len(nodes) != len(graph["nodes"]) or entry not in nodes:
        raise ValueError("Duplicate nodes or missing graph entry")
    adjacent = next_call_adjacency(nodes, (
        (edge["source"], edge["target"], edge["route"], edge["kind"]) for edge in graph["edges"]
    ))
    paths = {node: paths_from(adjacent, node) for node in nodes}
    ends = {node for node, spec in nodes.items() if spec["kind"] == "exit"}
    entry_paths = {**paths[entry], entry: (entry,)}
    seen_runs = set()
    excluded = []
    task_runs = defaultdict(list)
    reached = defaultdict(list)
    visits = Counter()
    grades = defaultdict(dict)
    observed_edges = Counter()
    unmatched_edges = []
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
        without_node = paths_from({
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
                row.update(score_pair(grades[source], grades[target], n_runs))
            edge_results.append(row)
    return {
        "entry": entry,
        "graph_scope": graph.get("analysis_scope", graph.get("scope")),
        "structural_reachable_pairs": sum(map(len, paths.values())),
        "scored_runs": n_runs, "tasks": len(task_runs),
        "excluded_infrastructure_runs": excluded,
        "nodes": node_results, "edges": edge_results,
        "unmatched_observed_transitions": unmatched_edges,
    }


def show_rate(value):
    if value["rate"] is None:
        return "未定义"
    return f"{value['n']}/{value['denominator']}（{value['rate']:.1%}）"


def render_report(result):
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
        lines.append("")
    lines.extend([
        "静态可达性依据 next_call 边计算。JSON 保留最短路径、逐任务完成频次、"
        "节点移除后的终点可达性和各类运行编号。",
        "配对结果适用于该批次的验收规则。节点职责或冗余需要在调整设计后用独立任务复测。",
    ])
    return "\n".join(lines) + "\n"


def run_analysis(args):
    if args.trace_format == "odr-pilot" and args.runs is None:
        raise ValueError("The odr-pilot trace format requires --runs")
    runs, inputs = load_runs(args.runs, args.trace_format) if args.runs else ([], {})
    result = analyze(read_json(args.graph), runs)
    result["inputs"] = {"graph": provenance(args.graph), **inputs}
    result["trace_format"] = args.trace_format if args.runs else None
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "reachability.json", result)
    report = args.out / "report.md"
    report.write_text(render_report(result), encoding="utf-8")
    print(f"{len(result['nodes'])} nodes, {result['structural_reachable_pairs']} reachable pairs, "
          f"{result['scored_runs']} scored runs -> {report.resolve()}")
    return 0
