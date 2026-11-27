"""Semantic-head records and metrics one run can determine. No model calls."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

TOKEN_OR_LIMIT = ("token", "budget", "length", "limit")


def undefined_rate(reason: str) -> dict[str, Any]:
    """A proportion that cannot be computed. The reason is kept for the report."""
    return {
        "n": None, "denominator": None, "rate": None, "defined": False, "reason": reason,
        "interval": {"defined": False, "reason": reason},
    }


def wilson_interval(count: int, total: int, z: float = 1.96) -> dict[str, Any]:
    """Wilson score interval for a count/total proportion."""
    if total <= 0:
        return {"defined": False, "reason": "分母为 0，Wilson 区间无定义"}
    phat = count / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (phat + z2 / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * total)) / total) / denom
    return {"defined": True, "low": max(0.0, center - margin), "high": min(1.0, center + margin), "z": z}


def defined_rate(count: int, total: int, empty_reason: str) -> dict[str, Any]:
    """Numerator, denominator, rate, and a Wilson interval."""
    if total <= 0:
        return undefined_rate(empty_reason)
    return {
        "n": count, "denominator": total, "rate": count / total, "defined": True,
        "interval": wilson_interval(count, total),
    }


def format_kind(message: str) -> str:
    """Classify a format failure. Token budget, truncation, and tool structure stay separate."""
    text = (message or "").casefold()
    if "tool" in text:
        return "tool_structure"
    if "truncat" in text:
        return "truncated"
    if any(token in text for token in TOKEN_OR_LIMIT):
        return "token_budget"
    return "format_other"


def node_records(row: dict) -> dict[str, dict]:
    """Return per-node semantic records. Missing or malformed records yield an empty map."""
    raw = row.get("semantic_heads")
    if not isinstance(raw, dict):
        return {}
    return {str(node): record for node, record in raw.items() if isinstance(record, dict)}


def coerce_heads(value: Any) -> list[dict] | None:
    """Accept ``{key, value}`` heads or a one-pair ``{key: value}`` object."""
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    heads = []
    for item in value:
        head = _one_head(item)
        if head is not None:
            heads.append(head)
    return heads


def _one_head(item: Any) -> dict | None:
    if not isinstance(item, dict):
        return None
    aliases = [str(alias) for alias in item.get("aliases") or [] if str(alias)]
    if "key" in item and "value" in item:
        return {"key": str(item["key"]), "value": str(item["value"]), "aliases": aliases}
    pairs = [(str(key), value) for key, value in item.items() if key not in {"aliases", "key", "value"}]
    if len(pairs) == 1:
        key, value = pairs[0]
        return {"key": key, "value": str(value), "aliases": aliases}
    return None


def input_channels(record: dict) -> dict[str, list[dict]] | None:
    """Named input channels. A bare list is the single channel ``input``."""
    raw = record.get("inputs", record.get("input_channels"))
    if raw is None:
        return None
    if isinstance(raw, list):
        return {"input": coerce_heads(raw) or []}
    if not isinstance(raw, dict):
        return None
    return {str(name): coerce_heads(heads) or [] for name, heads in raw.items()}


def output_heads(record: dict) -> list[dict] | None:
    raw = record.get("output", record.get("output_heads"))
    if raw is None:
        return None
    return coerce_heads(raw) or []


def heads_match(left: dict, right: dict) -> bool:
    """Alias string match. Keys must agree, and the values must share a value or an alias."""
    if left["key"].casefold() != right["key"].casefold():
        return False
    left_values = {left["value"].casefold(), *(alias.casefold() for alias in left.get("aliases", []))}
    right_values = {right["value"].casefold(), *(alias.casefold() for alias in right.get("aliases", []))}
    return bool(left_values & right_values)


def unique_heads(channels: dict[str, list[dict]]) -> list[dict]:
    found: list[dict] = []
    for heads in channels.values():
        for head in heads:
            if not any(heads_match(head, item) for item in found):
                found.append(head)
    return found


def heads_found_in_text(required: dict, text: str) -> list[dict]:
    """String-match ``{key: [aliases]}`` against text. This does not call a model."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    found = []
    for key, aliases in (required or {}).items():
        names = [str(alias) for alias in (aliases or [key]) if str(alias)]
        hit = next((alias for alias in names if _alias_in(alias, folded)), None)
        if hit is not None:
            found.append({"key": str(key), "value": hit, "aliases": names})
    return found


def _alias_in(alias: str, folded: str) -> bool:
    token = unicodedata.normalize("NFKC", alias).casefold()
    if not token:
        return False
    return re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", folded) is not None


def unsupported_of(channels: dict[str, list[dict]], output: list[dict]) -> list[dict]:
    """Output heads that no input channel contains."""
    known = unique_heads(channels)
    return [head for head in output if not any(heads_match(head, item) for item in known)]


def format_ok_for(row: dict, node: str) -> bool | None:
    """Node format flag. ``None`` means the trial did not record it."""
    record = node_records(row).get(node)
    if isinstance(record, dict) and isinstance(record.get("format_ok"), bool):
        return record["format_ok"]
    if isinstance(row.get("format_ok"), bool):
        completion = row.get("node_completion") or {}
        if node in completion or node in (row.get("path") or []):
            return row["format_ok"]
    return None


def format_rejected(row: dict, node: str) -> bool:
    """Format failures are counted apart from completion and do not enter necessity."""
    return format_ok_for(row, node) is False


def retention_of(channels: dict[str, list[dict]] | None, output: list[dict] | None) -> dict[str, Any]:
    if channels is None or output is None:
        return undefined_rate("没有语义头记录")
    known = unique_heads(channels)
    if not known:
        return undefined_rate("没有输入语义头")
    kept = sum(1 for head in known if any(heads_match(head, item) for item in output))
    return defined_rate(kept, len(known), "没有输入语义头")


def unsupported_rate_of(channels: dict[str, list[dict]] | None, output: list[dict] | None) -> dict[str, Any]:
    if channels is None or output is None:
        return undefined_rate("没有语义头记录")
    if not output:
        return undefined_rate("没有输出语义头")
    missing = unsupported_of(channels, output)
    return defined_rate(len(missing), len(output), "没有输出语义头")


def information_flow(channels: dict[str, list[dict]] | None, output: list[dict] | None) -> dict[str, Any]:
    """Where each recorded head enters and whether it leaves in the output.

    The call graph is not a substitute. Missing channels stay undefined.
    """
    if channels is None or output is None:
        return {"defined": False, "reason": "没有语义头记录，调用图不代替信息流", "heads": []}
    rows = []
    for name, heads in channels.items():
        for head in heads:
            rows.append({
                "key": head["key"], "value": head["value"], "channel": name,
                "left": not any(heads_match(head, item) for item in output),
            })
    return {"defined": True, "reason": "", "heads": rows}


def _node_semantics(record: dict) -> dict[str, Any]:
    channels = input_channels(record)
    output = output_heads(record)
    return {
        "semantic_retention": retention_of(channels, output),
        "unsupported_share": unsupported_rate_of(channels, output),
        "information_flow": information_flow(channels, output),
    }


def assessed_pass(row: dict, node: str) -> bool | None:
    """Pass bit for one node. Format failures and missing checks are not passes or failures."""
    if row.get("infrastructure_error") or format_rejected(row, node):
        return None
    value = (row.get("node_completion") or {}).get(node)
    if isinstance(value, bool):
        return value
    return None


def first_incomplete(row: dict) -> str | None:
    """First path node whose recorded completion is false, skipping format failures."""
    completion = row.get("node_completion") or {}
    for node in row.get("path") or []:
        if format_rejected(row, node):
            continue
        if completion.get(node) is False:
            return node
    return None


def run_format_ok(row: dict) -> bool | None:
    flags = [record["format_ok"] for record in node_records(row).values() if isinstance(record.get("format_ok"), bool)]
    if flags:
        return all(flags)
    if isinstance(row.get("format_ok"), bool):
        return row["format_ok"]
    return None


def edge_metrics_for_run(row: dict, edges: list[dict]) -> list[dict]:
    """Q and the conditional pass rate for one run, only on edges both nodes touch."""
    rows = []
    visited = set(row.get("path") or [])
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source == target or source not in visited or target not in visited:
            continue
        source_pass = assessed_pass(row, source)
        target_pass = assessed_pass(row, target)
        if source_pass is None and target_pass is None:
            continue
        if source_pass is None or target_pass is None:
            blank = undefined_rate("该次运行这对节点没有同时验收")
            rows.append({"source": source, "target": target, "q": blank, "conditional_pass": blank})
            continue
        if source_pass:
            ratio = defined_rate(int(target_pass), 1, "上游未通过，完成个数之比的分母为 0")
            conditional = defined_rate(int(target_pass), 1, "该次运行上游未通过")
        else:
            ratio = undefined_rate("上游未通过，完成个数之比的分母为 0")
            conditional = undefined_rate("该次运行上游未通过")
        rows.append({"source": source, "target": target, "q": ratio, "conditional_pass": conditional})
    return rows


def _wrap_completion(value: dict) -> dict[str, Any]:
    if not value or not value.get("denominator"):
        return undefined_rate("没有已验收的运行")
    return defined_rate(int(value["n"]), int(value["denominator"]), "没有已验收的运行")


def _pool(snapshots: list[dict], node: str, field: str, missing: str) -> dict[str, Any]:
    count = total = used = 0
    for snapshot in snapshots:
        metrics = (snapshot.get("node_semantics") or {}).get(node, {}).get(field)
        if not metrics or not metrics.get("defined"):
            continue
        count += int(metrics["n"])
        total += int(metrics["denominator"])
        used += 1
    if used == 0:
        return undefined_rate(missing)
    return defined_rate(count, total, missing)


def _flow_for(snapshots: list[dict], node: str) -> dict[str, Any]:
    heads = []
    for snapshot in snapshots:
        flow = (snapshot.get("node_semantics") or {}).get(node, {}).get("information_flow")
        if not flow or not flow.get("defined"):
            continue
        for item in flow["heads"]:
            heads.append({"run_id": snapshot.get("run_id"), **item})
    if not heads and not any((snapshot.get("node_semantics") or {}).get(node) for snapshot in snapshots):
        return {"defined": False, "reason": "没有语义头记录，调用图不代替信息流", "heads": []}
    if not heads:
        return {"defined": False, "reason": "没有输入语义头，调用图不代替信息流", "heads": []}
    return {"defined": True, "reason": "", "heads": heads}


def format_events(runs: list[dict], node: str) -> list[str]:
    kinds = []
    for row in runs:
        if row.get("infrastructure_error") or not format_rejected(row, node):
            continue
        record = node_records(row).get(node) or {}
        message = record.get("format_error") or row.get("format_error") or ""
        kinds.append(format_kind(str(message)))
    return kinds


def run_snapshot(row: dict, measured: dict) -> dict[str, Any]:
    """Section 3.2 fields for one trial. Semantic rates stay undefined when heads are absent."""
    semantics = {node: _node_semantics(record) for node, record in node_records(row).items()}
    unmatched = [item for item in measured.get("unmatched_observed_transitions") or []
                 if item.get("run_id") == row.get("run_id")]
    if semantics:
        count = total = 0
        unsupported_count = unsupported_total = 0
        retain_used = unsupported_used = False
        for metrics in semantics.values():
            if metrics["semantic_retention"].get("defined"):
                count += metrics["semantic_retention"]["n"]
                total += metrics["semantic_retention"]["denominator"]
                retain_used = True
            if metrics["unsupported_share"].get("defined"):
                unsupported_count += metrics["unsupported_share"]["n"]
                unsupported_total += metrics["unsupported_share"]["denominator"]
                unsupported_used = True
        retained = (defined_rate(count, total, "没有输入语义头") if retain_used
                    else undefined_rate("没有输入语义头"))
        unsupported = (defined_rate(unsupported_count, unsupported_total, "没有输出语义头") if unsupported_used
                       else undefined_rate("没有输出语义头"))
    else:
        retained = undefined_rate("没有语义头记录")
        unsupported = undefined_rate("没有语义头记录")
    return {
        "run_id": row.get("run_id"),
        "task_id": row.get("task_id"),
        "path": list(row.get("path") or []),
        "excluded": bool(row.get("infrastructure_error")),
        "format_ok": run_format_ok(row),
        "first_incomplete_node": None if row.get("infrastructure_error") else first_incomplete(row),
        "unmatched_transitions": unmatched,
        "edges": [] if row.get("infrastructure_error") else edge_metrics_for_run(row, measured.get("edges") or []),
        "semantic_retention": retained,
        "unsupported_share": unsupported,
        "node_semantics": semantics,
    }


def _usage(runs: list[dict]) -> dict[str, Any]:
    calls = tokens = 0
    elapsed = 0.0
    seen_calls = seen_tokens = seen_time = False
    for row in runs:
        model_calls = row.get("model_calls")
        if isinstance(model_calls, list):
            seen_calls = True
            calls += len(model_calls)
            for call in model_calls:
                if not isinstance(call, dict):
                    continue
                usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
                if isinstance(usage.get("total_tokens"), int):
                    seen_tokens = True
                    tokens += usage["total_tokens"]
                elif isinstance(call.get("prompt_tokens"), int):
                    seen_tokens = True
                    tokens += call["prompt_tokens"]
        if isinstance(row.get("elapsed_model_ms"), (int, float)) and not isinstance(row.get("elapsed_model_ms"), bool):
            seen_time = True
            elapsed += float(row["elapsed_model_ms"])
    return {
        "calls": calls if seen_calls else None,
        "tokens": tokens if seen_tokens else None,
        "elapsed_ms": elapsed if seen_time else None,
    }


def _split_name(row: dict) -> str | None:
    split = row.get("split")
    if split in {"development", "development-pilot"}:
        return "development"
    if split == "holdout":
        return "holdout"
    return None


def split_gap(runs: list[dict], node: str) -> dict[str, Any]:
    """Pass-rate difference between development tasks and held-out tasks."""
    buckets = {"development": [], "holdout": []}
    for row in runs:
        name = _split_name(row)
        if name is None:
            continue
        flag = assessed_pass(row, node)
        if flag is not None:
            buckets[name].append(flag)
    if not buckets["development"] or not buckets["holdout"]:
        return {"defined": False, "reason": "轨迹没有同时包含开发题和留出题"}
    dev = defined_rate(sum(buckets["development"]), len(buckets["development"]), "没有开发题验收")
    hold = defined_rate(sum(buckets["holdout"]), len(buckets["holdout"]), "没有留出题验收")
    return {"defined": True, "development": dev, "holdout": hold, "difference": dev["rate"] - hold["rate"]}


def summarize_single_run(runs: list[dict], measured: dict) -> dict[str, Any]:
    """Metrics IdealGoal section 3.2 can compute without another model call."""
    snapshots = [run_snapshot(row, measured) for row in runs]
    nodes = {}
    for node, spec in measured["nodes"].items():
        kinds = format_events(runs, node)
        nodes[node] = {
            "label": spec["label"],
            "arrival": defined_rate(len(spec["observed_runs"]), measured["scored_runs"], "没有已计入的运行"),
            "completion": _wrap_completion(spec["completion"]),
            "format_failures": len(kinds),
            "format_kinds": {name: kinds.count(name) for name in sorted(set(kinds))},
            "semantic_retention": _pool(snapshots, node, "semantic_retention", "没有语义头记录"),
            "unsupported_share": _pool(snapshots, node, "unsupported_share", "没有语义头记录"),
            "information_flow": _flow_for(snapshots, node),
            "split_gap": split_gap(runs, node),
        }
    return {
        "nodes": nodes,
        "runs": snapshots,
        "usage": _usage(runs),
        "entailment": {
            "defined": False,
            "reason": "基础路径不调用模型，未做互相蕴含判定，也没有冻结人工样本上的一致率",
        },
    }


def format_rate(value: dict) -> str:
    """Render a proportion as numerator/denominator, or 无定义."""
    if not value or not value.get("defined"):
        reason = (value or {}).get("reason") or "无定义"
        return f"无定义（{reason}）"
    text = f"{value['n']}/{value['denominator']}（{value['rate']:.1%}"
    interval = value.get("interval") or {}
    if interval.get("defined"):
        text += f"，Wilson 95% [{interval['low']:.3f}, {interval['high']:.3f}]"
    return text + "）"


_KIND_LABEL = {
    "token_budget": "超出令牌预算",
    "truncated": "输出截断",
    "tool_structure": "工具调用结构错误",
    "format_other": "其他格式不合格",
}


def render_single_run(summary: dict) -> str:
    """Chinese section for the report. Logs stay limited to node scores and logic faults."""
    lines = [
        "",
        "## 一次运行可计算的指标",
        "",
        "语义头只按别名做字符串匹配。互相蕴含的判定未执行：基础路径不调用模型，也没有另一系列模型在冻结人工样本上的一致率。",
        "没有语义头记录时，语义保持率和无输入依据的语义所占比例为无定义，不另造数字。",
        "信息流只来自试验记录里的输入通道。没有这些记录时信息流无定义，调用图不代替信息流。",
        "格式不合格单独计数，不进入完成率，也不进入必要性。",
        "",
        "| 节点 | 到达 | 完成 | 格式不合格 | 语义保持率 | 无输入依据的语义 |",
        "|---|---|---|---:|---|---|",
    ]
    kind_totals: dict[str, int] = {}
    for row in summary["nodes"].values():
        for name, count in row["format_kinds"].items():
            kind_totals[name] = kind_totals.get(name, 0) + count
        lines.append(
            f"| {row['label']} | {format_rate(row['arrival'])} | {format_rate(row['completion'])} | "
            f"{row['format_failures']} | {format_rate(row['semantic_retention'])} | "
            f"{format_rate(row['unsupported_share'])} |"
        )
    if kind_totals:
        detail = "，".join(f"{_KIND_LABEL.get(name, name)} {count}" for name, count in sorted(kind_totals.items()))
        lines.extend(["", f"格式不合格按原因计数：{detail}。"])
    else:
        lines.extend(["", "这次记录没有格式不合格。"])
    lines.extend(["", "### 各次运行", ""])
    for snapshot in summary["runs"]:
        path = " → ".join(snapshot["path"]) or "（空路径）"
        if snapshot["format_ok"] is True:
            form = "合格"
        elif snapshot["format_ok"] is False:
            form = "不合格"
        else:
            form = "未记录"
        missed = snapshot["unmatched_transitions"]
        missed_text = "无" if not missed else "，".join(f"{item['source']}→{item['target']}" for item in missed)
        incomplete = snapshot["first_incomplete_node"] or "无"
        lines.append(
            f"- `{snapshot['run_id']}` 路径 {path}。第一个未完成节点：{incomplete}。"
            f"格式：{form}。语义保持率：{format_rate(snapshot['semantic_retention'])}。"
            f"无输入依据的语义：{format_rate(snapshot['unsupported_share'])}。"
            f"与 next_call 不一致的转移：{missed_text}。"
        )
        for edge in snapshot["edges"]:
            lines.append(
                f"  - Q（{edge['source']}→{edge['target']}）为 {format_rate(edge['q'])}，"
                f"条件通过率为 {format_rate(edge['conditional_pass'])}。"
            )
    usage = summary["usage"]
    lines.extend(["", "### 令牌、调用与耗时", ""])
    if usage["calls"] is None and usage["tokens"] is None and usage["elapsed_ms"] is None:
        lines.append("未记录令牌数、调用次数和耗时。这三项不并入通过率。")
    else:
        calls = "未记录" if usage["calls"] is None else str(usage["calls"])
        tokens = "未记录" if usage["tokens"] is None else str(usage["tokens"])
        elapsed = "未记录" if usage["elapsed_ms"] is None else str(usage["elapsed_ms"])
        lines.append(f"调用次数 {calls}。令牌数 {tokens}。耗时 {elapsed} 毫秒。这三项不并入通过率。")
    lines.extend(["", "### 开发题与留出题", ""])
    gaps = [row["split_gap"] for row in summary["nodes"].values()]
    if not any(item.get("defined") for item in gaps):
        lines.append("无定义。轨迹没有同时包含开发题和留出题，因此不计算两类通过率之差。")
    else:
        for row in summary["nodes"].values():
            gap = row["split_gap"]
            if not gap.get("defined"):
                continue
            lines.append(
                f"- {row['label']}：开发题 {format_rate(gap['development'])}，留出题 {format_rate(gap['holdout'])}，"
                f"差值 {gap['difference']:.3f}。"
            )
    lines.extend(["", "### 信息流", ""])
    flow_lines = []
    for row in summary["nodes"].values():
        flow = row["information_flow"]
        if not flow.get("defined"):
            continue
        for item in flow["heads"]:
            state = "离开" if item["left"] else "仍在输出中"
            flow_lines.append(f"- {row['label']} 从通道 {item['channel']} 进入 {item['key']}={item['value']}，{state}。")
    lines.extend(flow_lines or ["无定义。试验记录没有输入通道，调用图不代替信息流。"])
    return "\n".join(lines) + "\n"
