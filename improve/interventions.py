"""Semantic-density sampling, sufficiency, necessity, and checked substitutions.

The four tests do not learn a replacement table and do not edit the instance source.
Position changes are a control, not a fifth test.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Any, Callable

from src.core.semanticRecord import (
    assessed_pass, defined_rate, format_rejected, heads_match, input_channels, node_records,
    output_heads, retention_of, undefined_rate, unsupported_of, unsupported_rate_of,
)

REDUNDANT_N = 0.05
WORDING_GAP = 0.2
UNSUPPORTED_HIGH = 0.5
MAINTAIN_HIGH = 0.5
MATCH_MISS = "匹配漏检"
OTHER_INPUT = "另一路输入"
MODEL_FILL = "模型自行补全"
REDUNDANT = "冗余"
SOURCE_ORDER = (MATCH_MISS, OTHER_INPUT, MODEL_FILL, REDUNDANT)
Runner = Callable[[dict], dict]


def retain_value(value: str, retain: float) -> str:
    """Keep the leading fraction of a head. ``retain`` of 1 keeps the whole value."""
    if retain >= 1 or not value:
        return value
    kept = math.ceil(len(value) * retain - 1e-12)
    return value[:kept]


def density_sample(heads: list[dict], sample_count: int, retain: float, seed: int) -> dict[str, Any]:
    """Draw ``sample_count`` heads and keep ``retain`` of each value.

    ``retain`` of 1 together with a draw of every head is the unsampled control.
    """
    indexed = list(enumerate(heads))
    if not indexed:
        return {"heads": [], "control": False, "sampled": 0, "requested": sample_count}
    control = retain >= 1 and sample_count >= len(indexed)
    draw = min(sample_count, len(indexed))
    if draw >= len(indexed):
        chosen = indexed
    else:
        picked = random.Random(seed).sample(indexed, draw)
        chosen = sorted(picked, key=lambda item: item[0])
    kept = []
    for _, head in chosen:
        value = head.get("value", "")
        kept.append({**head, "value": value if retain >= 1 else retain_value(str(value), retain)})
    return {"heads": kept, "control": control, "sampled": len(kept), "requested": sample_count}


def split_sentences(text: str) -> list[str]:
    sentences = []
    start = 0
    for index, char in enumerate(text):
        if char in "。！？.!?\n":
            piece = text[start:index + 1].strip()
            if piece:
                sentences.append(piece)
            start = index + 1
    tail = text[start:].strip()
    if tail and not sentences:
        sentences.append(tail)
    return sentences


def is_degree_shape(text: str) -> bool:
    """A degree word has no spaces and no sentence punctuation. There is no replacement lexicon."""
    if not text or text.strip() != text:
        return False
    return not any(char.isspace() or char in "。！？.!?；;，,、" for char in text)


def is_full_sentence(prompt: str, original: str) -> bool:
    return original.strip() in split_sentences(prompt)


def match_replacement(prompt: str, item: object, relation: str) -> dict[str, Any]:
    """Keep a replacement only when the rest of the prompt is unchanged."""
    rejected = {"accepted": False, "prompt": None, "original": None, "replacement": None,
                "kind": None, "relation": relation}
    if not isinstance(item, dict):
        return {**rejected, "reason": "not_object"}
    original, replacement, kind, rel = (item.get("original"), item.get("replacement"),
                                        item.get("kind"), item.get("relation"))
    if not isinstance(original, str) or not isinstance(replacement, str) or not isinstance(kind, str):
        return {**rejected, "reason": "missing_fields"}
    if rel != relation:
        return {**rejected, "reason": "wrong_relation"}
    if kind not in {"degree", "sentence"}:
        return {**rejected, "reason": "bad_kind"}
    if not original or not replacement or original == replacement:
        return {**rejected, "reason": "unchanged"}
    if original not in prompt:
        return {**rejected, "reason": "not_in_prompt"}
    if prompt.count(original) != 1:
        return {**rejected, "reason": "ambiguous"}
    if kind == "degree" and not is_degree_shape(original):
        return {**rejected, "reason": "not_degree_shape"}
    if kind == "sentence" and not is_full_sentence(prompt, original):
        return {**rejected, "reason": "not_full_sentence"}
    prefix, _, suffix = prompt.partition(original)
    updated = prefix + replacement + suffix
    if updated[:len(prefix)] + updated[len(prefix) + len(replacement):] != prefix + suffix:
        return {**rejected, "reason": "rest_changed"}
    return {
        "accepted": True, "reason": "", "prompt": updated, "original": original,
        "replacement": replacement, "kind": kind, "relation": relation,
    }


def relocate_span(prompt: str, span: str) -> str:
    """Move one span. The characters stay the same."""
    prefix, found, suffix = prompt.partition(span)
    if not found:
        return prompt
    rest = prefix + suffix
    if prefix:
        return span + rest
    return rest + span


def _seed(row: dict, node: str) -> int:
    raw = row.get("seed")
    base = raw if isinstance(raw, int) and not isinstance(raw, bool) else 0
    digest = hashlib.sha256(f"{row.get('task_id')}\0{node}\0{base}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31)


def _collected(record: dict) -> list[dict]:
    channels = input_channels(record) or {}
    heads = []
    for name, items in channels.items():
        for head in items:
            heads.append({**head, "channel": name})
    return heads


def node_prompt(row: dict, node: str) -> str | None:
    record = node_records(row).get(node) or {}
    if isinstance(record.get("prompt"), str) and record["prompt"]:
        return record["prompt"]
    prompts = row.get("prompts")
    if isinstance(prompts, dict) and isinstance(prompts.get(node), str) and prompts[node]:
        return prompts[node]
    return None


def _ask(runner: Runner | None, case: dict) -> dict[str, Any]:
    if runner is None:
        return {"defined": False, "passed": None, "reaction": None,
                "reason": "未提供节点执行器，未重跑被测节点"}
    try:
        result = runner(case)
    except Exception as error:
        return {"defined": False, "passed": None, "reaction": None, "reason": f"节点执行器失败：{error}"}
    if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
        return {"defined": False, "passed": None, "reaction": None,
                "reason": "节点执行器没有返回 passed 布尔值"}
    return {"defined": True, "passed": result["passed"], "reaction": result.get("reaction"), "reason": ""}


def _rate_from(flags: list[bool]) -> dict[str, Any]:
    if not flags:
        return undefined_rate("没有可比较的结果")
    return defined_rate(sum(bool(flag) for flag in flags), len(flags), "没有可比较的结果")


def _first_indexes(passed_flags: list[bool | None]) -> dict[str, Any]:
    """One execution: a pass is attempt 1, and a failure does not invent a later attempt."""
    if not passed_flags or all(flag is None for flag in passed_flags):
        return {"defined": False, "attempts": [], "reason": "没有可比较的结果"}
    if not any(flag is True for flag in passed_flags):
        return {"defined": False, "attempts": [], "reason": "这一次没有通过，无法确定第几次才第一次通过"}
    attempts = [1 if flag is True else None for flag in passed_flags if flag is not None]
    return {"defined": True, "attempts": attempts, "reason": ""}


def run_sampling(settings: Any, runs: list[dict], runner: Runner | None) -> dict[str, Any]:
    """Sample collected heads once per task, seed, and node."""
    seen = set()
    cases = []
    for row in runs:
        if row.get("infrastructure_error"):
            continue
        for node, record in node_records(row).items():
            if format_rejected(row, node):
                continue
            key = (row.get("task_id"), row.get("seed"), node)
            if key in seen:
                continue
            seen.add(key)
            heads = _collected(record)
            drawn = density_sample(heads, settings.sample_count, settings.retain, _seed(row, node))
            original = assessed_pass(row, node)
            if not heads:
                outcome = {"defined": False, "passed": None, "reaction": None, "reason": "没有已收集的信息"}
            else:
                outcome = _ask(runner, {
                    "task_id": row.get("task_id"), "seed": row.get("seed"), "node": node,
                    "mode": "sample", "prompt": node_prompt(row, node) or "", "heads": drawn["heads"],
                    "control": drawn["control"],
                })
            cases.append({
                "task_id": row.get("task_id"), "seed": row.get("seed"), "node": node,
                "control": drawn["control"], "sampled": drawn["sampled"], "requested": drawn["requested"],
                "heads": drawn["heads"], "original": original, "outcome": outcome,
            })
    paired = [case for case in cases if case["original"] is not None and case["outcome"].get("defined")]
    before = _rate_from([bool(case["original"]) for case in paired])
    after = _rate_from([bool(case["outcome"]["passed"]) for case in paired])
    drop = None if not before.get("defined") or not after.get("defined") else before["rate"] - after["rate"]
    return {
        "cases": cases,
        "before": before,
        "after": after,
        "drop": drop,
        "still_passed": [case["task_id"] for case in paired if case["original"] and case["outcome"]["passed"]],
        "no_longer_passed": [case["task_id"] for case in paired if case["original"] and not case["outcome"]["passed"]],
        "control": bool([case for case in cases if case["heads"]]) and all(
            case["control"] for case in cases if case["heads"]),
    }


def _groups(runs: list[dict]) -> list[tuple[dict, str, dict]]:
    seen = set()
    groups = []
    for row in runs:
        if row.get("infrastructure_error"):
            continue
        for node, record in node_records(row).items():
            if format_rejected(row, node):
                continue
            key = (row.get("task_id"), row.get("seed"), node)
            if key in seen:
                continue
            seen.add(key)
            groups.append((row, node, record))
    return groups


def run_sufficiency(runs: list[dict], runner: Runner | None) -> dict[str, Any]:
    """Same task and seed, with only condition G provided."""
    cases = []
    for row, node, record in _groups(runs):
        if format_rejected(row, node):
            continue
        seen_keys = set()
        for head in _collected(record):
            if head["key"] in seen_keys:
                continue
            seen_keys.add(head["key"])
            outcome = _ask(runner, {
                "task_id": row.get("task_id"), "seed": row.get("seed"), "node": node, "mode": "sufficiency",
                "prompt": node_prompt(row, node) or "", "heads": [head], "condition": head,
            })
            cases.append({"task_id": row.get("task_id"), "seed": row.get("seed"), "node": node,
                          "key": head["key"], "channel": head["channel"], "outcome": outcome})
    return {"cases": cases, "by_condition": _effect_table(cases, "s-pass@1", "s-pass@k", baseline_one=False)}


def run_necessity(runs: list[dict], runner: Runner | None) -> dict[str, Any]:
    """On successful samples, remove one head. Format failures stay out."""
    cases = []
    for row in runs:
        if row.get("infrastructure_error"):
            continue
        for node, record in node_records(row).items():
            if assessed_pass(row, node) is not True:
                continue
            heads = _collected(record)
            for index, head in enumerate(heads):
                kept = [item for item_index, item in enumerate(heads) if item_index != index]
                outcome = _ask(runner, {
                    "task_id": row.get("task_id"), "seed": row.get("seed"), "node": node, "mode": "necessity",
                    "prompt": node_prompt(row, node) or "", "heads": kept, "removed": head,
                })
                cases.append({
                    "task_id": row.get("task_id"), "seed": row.get("seed"), "node": node,
                    "key": head["key"], "channel": head["channel"], "outcome": outcome,
                })
    return {"cases": cases, "by_condition": _effect_table(cases, "e-pass@1", "e-pass@k", baseline_one=True)}


def _effect_table(cases: list[dict], once_name: str, attempt_name: str, *, baseline_one: bool) -> list[dict]:
    grouped: dict[tuple, list] = {}
    for case in cases:
        grouped.setdefault((case["node"], case["channel"], case["key"]), []).append(case)
    rows = []
    for (node, channel, key), items in grouped.items():
        defined = [item for item in items if item["outcome"].get("defined")]
        flags = [bool(item["outcome"]["passed"]) for item in defined]
        once = _rate_from(flags)
        effect = None
        if once.get("defined"):
            baseline = 1.0 if baseline_one else once["rate"]
            effect = baseline - once["rate"] if baseline_one else once["rate"]
        rows.append({
            "node": node, "channel": channel, "key": key, "defined": bool(once.get("defined")),
            once_name: once,
            attempt_name: _first_indexes([item["outcome"]["passed"] if item["outcome"].get("defined") else None
                                          for item in items]),
            "S" if not baseline_one else "N": effect,
            "n": effect,
        })
    return rows


def prepare_substitutions(runs: list[dict], generator: Callable[[str, str], list[dict]] | None) -> dict[str, Any]:
    """Call the generator and string-check before any substituted input is used."""
    jobs = []
    seen = set()
    for row in runs:
        for node in node_records(row):
            if format_rejected(row, node):
                continue
            prompt = node_prompt(row, node)
            if not prompt or (node, prompt) in seen:
                continue
            seen.add((node, prompt))
            jobs.append({"row": row, "node": node, "prompt": prompt, "record": node_records(row)[node]})
    empty = {"available": True, "reason": "试验记录没有原提示词", "accepted": [], "discarded": 0}
    if not jobs:
        return empty
    if generator is None:
        return {"available": False, "reason": "没有生成函数，未调用生成模型", "accepted": [], "discarded": 0}
    accepted = []
    discarded = 0
    try:
        for job in jobs:
            for relation in ("synonym", "antonym"):
                items = generator(job["prompt"], relation)
                if not isinstance(items, list):
                    discarded += 1
                    continue
                for item in items:
                    matched = match_replacement(job["prompt"], item, relation)
                    if not matched["accepted"]:
                        discarded += 1
                        continue
                    accepted.append({
                        **matched, "node": job["node"], "task_id": job["row"].get("task_id"),
                        "seed": job["row"].get("seed"), "run_id": job["row"].get("run_id"),
                        "original_prompt": job["prompt"], "original_pass": assessed_pass(job["row"], job["node"]),
                        "only_source": len(_collected(job["record"])) == 1,
                        "contradicts_user": _contradicts_user(job["record"], matched["original"]),
                        "heads": _collected(job["record"]),
                    })
    except Exception as error:
        reason = str(error) or "生成模型失败"
        return {"available": False, "reason": reason, "accepted": [], "discarded": discarded}
    return {"available": True, "reason": "", "accepted": accepted, "discarded": discarded}


def _contradicts_user(record: dict, original: str) -> bool:
    channels = input_channels(record) or {}
    for name, heads in channels.items():
        if name not in {"user", "用户"}:
            continue
        if any(original in head["value"] or head["value"] in original for head in heads if head["value"]):
            return True
    return False


def execute_substitutions(prepared: dict, runner: Runner | None) -> dict[str, Any]:
    """Use only replacements that already passed the string check."""
    if not prepared.get("available"):
        blank = undefined_rate(prepared.get("reason") or "反例与同义代换未执行")
        return {**prepared, "cases": [], "synonym_delta": {"defined": False, "reason": blank["reason"]},
                "counterexample": {"adopt": blank, "maintain": blank, "contradict": blank},
                "position_control": {"defined": False, "reason": blank["reason"], "cases": []}}
    cases = []
    for item in prepared["accepted"]:
        mode = "synonym" if item["relation"] == "synonym" else "counterexample"
        outcome = _ask(runner, {
            "task_id": item["task_id"], "seed": item["seed"], "node": item["node"], "mode": mode,
            "prompt": item["prompt"], "heads": item["heads"], "relation": item["relation"],
        })
        cases.append({**item, "outcome": outcome})
    return {
        **prepared, "cases": cases,
        "synonym_delta": _synonym_delta(cases),
        "counterexample": _counterexample_rates(cases),
        "position_control": _position_control(prepared["accepted"], runner),
    }


def _synonym_delta(cases: list[dict]) -> dict[str, Any]:
    paired = [case for case in cases if case["relation"] == "synonym"
              and case["original_pass"] is not None and case["outcome"].get("defined")]
    if not paired:
        return {"defined": False, "reason": "没有通过字符串匹配且可比较的同义替换"}
    before = _rate_from([bool(case["original_pass"]) for case in paired])
    after = _rate_from([bool(case["outcome"]["passed"]) for case in paired])
    return {"defined": True, "before": before, "after": after, "difference": after["rate"] - before["rate"]}


def _reaction(value: object) -> str | None:
    names = {"adopt": "adopt", "采用": "adopt", "maintain": "maintain", "维持": "maintain",
             "contradict": "contradict", "矛盾": "contradict"}
    return names.get(value) if isinstance(value, str) else None


def _counterexample_rates(cases: list[dict]) -> dict[str, Any]:
    antonyms = [case for case in cases if case["relation"] == "antonym" and case["outcome"].get("defined")]
    labeled = [case for case in antonyms if _reaction(case["outcome"].get("reaction"))]
    only_source = [case for case in labeled if case["only_source"]]
    contradictions = [case for case in labeled if case["contradicts_user"]]

    def portion(subset: list[dict], name: str) -> dict[str, Any]:
        if not subset:
            return undefined_rate(f"没有适用于「{name}」的反例样本")
        return defined_rate(sum(_reaction(case["outcome"].get("reaction")) == name for case in subset),
                            len(subset), "没有反例样本")

    return {"adopt": portion(only_source, "adopt"), "maintain": portion(labeled, "maintain"),
            "contradict": portion(contradictions, "contradict")}


def _position_control(accepted: list[dict], runner: Runner | None) -> dict[str, Any]:
    """Move text without changing it. This is not one of the four tests."""
    seen = set()
    cases = []
    for item in accepted:
        key = (item["node"], item["original_prompt"], item["original"])
        if key in seen:
            continue
        seen.add(key)
        moved = relocate_span(item["original_prompt"], item["original"])
        if Counter(moved) != Counter(item["original_prompt"]) or moved == item["original_prompt"]:
            continue
        outcome = _ask(runner, {
            "task_id": item["task_id"], "seed": item["seed"], "node": item["node"], "mode": "position",
            "prompt": moved, "heads": item["heads"],
        })
        cases.append({"node": item["node"], "prompt": moved, "original_pass": item["original_pass"], "outcome": outcome})
    if not cases:
        return {"defined": False, "reason": "没有通过字符串匹配的片段，位置控制无定义", "cases": []}
    paired = [case for case in cases if case["original_pass"] is not None and case["outcome"].get("defined")]
    before = _rate_from([bool(case["original_pass"]) for case in paired])
    after = _rate_from([bool(case["outcome"]["passed"]) for case in paired])
    difference = None if not before.get("defined") or not after.get("defined") else after["rate"] - before["rate"]
    return {"defined": bool(paired), "reason": "" if paired else "未重跑被测节点",
            "cases": cases, "before": before, "after": after, "difference": difference}


def classify_fault(upstream_id: str, upstream_record: dict | None, downstream_record: dict | None,
                   effects: list[float]) -> dict[str, Any]:
    """Assign one upstream-fail / downstream-pass run to the four sources."""
    found = set()
    downstream_channels = input_channels(downstream_record) if isinstance(downstream_record, dict) else None
    upstream_output = output_heads(upstream_record) if isinstance(upstream_record, dict) else None
    raw_output = (upstream_record or {}).get("raw_output") if isinstance(upstream_record, dict) else ""
    if downstream_channels is not None:
        from_upstream = downstream_channels.get(upstream_id) or []
        for name, heads in downstream_channels.items():
            if name == upstream_id:
                continue
            if any(not any(heads_match(head, item) for item in from_upstream) for head in heads):
                found.add(OTHER_INPUT)
        if unsupported_of(downstream_channels, output_heads(downstream_record) or []):
            found.add(MODEL_FILL)
        if raw_output:
            matched = upstream_output or []
            folded = str(raw_output).casefold()
            for heads in downstream_channels.values():
                for head in heads:
                    if any(heads_match(head, item) for item in matched):
                        continue
                    if head["value"] and head["value"].casefold() in folded:
                        found.add(MATCH_MISS)
    if any(abs(value) <= REDUNDANT_N for value in effects):
        found.add(REDUNDANT)
    sources = [name for name in SOURCE_ORDER if name in found]
    return {"primary": sources[0] if sources else "未归类", "sources": sources}


def _node_effects(necessity_rows: list[dict]) -> dict[tuple, list[float]]:
    grouped: dict[tuple, list[float]] = {}
    for row in necessity_rows:
        if row.get("defined") and isinstance(row.get("n"), (int, float)):
            grouped.setdefault((row["node"], row["channel"]), []).append(float(row["n"]))
    return grouped


def fault_sources(measured: dict, runs: list[dict], necessity_rows: list[dict]) -> list[dict]:
    by_id = {row.get("run_id"): row for row in runs}
    effects = _node_effects(necessity_rows)
    found = []
    for edge in measured.get("edges") or []:
        for run_id in edge.get("necessary_condition_candidates") or []:
            row = by_id.get(run_id) or {}
            source, target = edge["source"], edge["target"]
            classified = classify_fault(source, node_records(row).get(source), node_records(row).get(target),
                                        effects.get((target, source), []))
            found.append({"run_id": run_id, "upstream": source, "downstream": target, **classified})
    return found


def upstream_fail_rates(measured: dict) -> list[dict]:
    rows = []
    for edge in measured.get("edges") or []:
        groups = edge.get("same_run_groups")
        paired = edge.get("paired_assessments") or {}
        if not groups or not paired.get("n"):
            continue
        numer = len(groups.get("target_only") or [])
        rows.append({"source": edge["source"], "target": edge["target"],
                     "r": defined_rate(numer, int(paired["n"]), "没有同时验收的运行")})
    return rows


def structural_suggestions(measured: dict, necessity_rows: list[dict]) -> list[dict]:
    """Split, merge, and delete are suggested only from N near 0."""
    defined = [row for row in necessity_rows if row.get("defined") and isinstance(row.get("n"), (int, float))]
    by_channel: dict[str, list[dict]] = {}
    for row in defined:
        by_channel.setdefault(row["channel"], []).append(row)
    suggestions = []
    nodes = measured.get("nodes") or {}
    for channel, rows in by_channel.items():
        spec = nodes.get(channel) or {}
        successors = list(spec.get("direct_successors") or sorted({row["node"] for row in rows}))
        covered = {successor: [row for row in rows if row["node"] == successor] for successor in successors}
        fully_redundant = bool(successors) and all(
            covered[successor] and all(abs(row["n"]) <= REDUNDANT_N for row in covered[successor])
            for successor in successors
        )
        if channel in nodes and fully_redundant:
            suggestions.append({
                "action": "合并或删除该节点", "node": channel,
                "evidence": f"该节点的信息对已测下游的 N 绝对值都不超过 {REDUNDANT_N}",
            })
        else:
            redundant = [row for row in rows if abs(row["n"]) <= REDUNDANT_N]
            for row in redundant:
                suggestions.append({
                    "action": "删除这条冗余信息", "node": row["node"],
                    "evidence": f"去掉 {row['key']} 后 N={row['n']}",
                })
        exclusive: dict[str, list[str]] = {}
        by_key: dict[str, list[dict]] = {}
        for row in rows:
            by_key.setdefault(row["key"], []).append(row)
        for key, items in by_key.items():
            needed = {item["node"] for item in items if item["n"] > REDUNDANT_N}
            redundant_for = {item["node"] for item in items if abs(item["n"]) <= REDUNDANT_N}
            if len(needed) == 1 and redundant_for:
                exclusive.setdefault(next(iter(needed)), []).append(key)
        if channel in nodes and len(exclusive) >= 2:
            suggestions.append({
                "action": "拆分该节点", "node": channel,
                "evidence": "信息分成两组，各自只对一个下游必要，对其余下游的 N 约为 0",
            })
    for node, spec in nodes.items():
        predecessors = list(spec.get("direct_predecessors") or [])
        for index, left in enumerate(predecessors):
            for right in predecessors[index + 1:]:
                if _reaches(nodes, left, right) or _reaches(nodes, right, left):
                    continue
                if _needed(defined, node, left) and _needed(defined, node, right):
                    suggestions.append({
                        "action": "改为并行", "node": node,
                        "evidence": f"{left} 与 {right} 互不依赖，且下游都需要它们",
                    })
    return suggestions


def _reaches(nodes: dict, source: str, target: str) -> bool:
    paths = (nodes.get(source) or {}).get("reachable_paths") or {}
    return target in paths


def _needed(rows: list[dict], node: str, channel: str) -> bool:
    return any(row["node"] == node and row["channel"] == channel and row["n"] > REDUNDANT_N for row in rows)


def wording_suggestions(runs: list[dict], substitution: dict) -> list[dict]:
    suggestions = []
    seen = set()
    for row in runs:
        for node, record in node_records(row).items():
            channels = input_channels(record)
            output = output_heads(record)
            retained = retention_of(channels, output)
            unsupported = unsupported_rate_of(channels, output)
            if retained.get("defined") and retained["rate"] < 1 and ("retain", node) not in seen:
                seen.add(("retain", node))
                suggestions.append({"action": "修改该节点的提示词或保留方式", "node": node,
                                    "evidence": f"语义保持率 {retained['n']}/{retained['denominator']}"})
            if unsupported.get("defined") and unsupported["rate"] >= UNSUPPORTED_HIGH and ("gap", node) not in seen:
                seen.add(("gap", node))
                suggestions.append({"action": "补充输入或增加约束", "node": node,
                                    "evidence": f"无输入依据的语义 {unsupported['n']}/{unsupported['denominator']}"})
    delta = substitution.get("synonym_delta") or {}
    if delta.get("defined") and abs(delta["difference"]) >= WORDING_GAP:
        suggestions.append({"action": "修改提示词", "node": "",
                            "evidence": f"同义替换前后的通过率之差为 {delta['difference']:.3f}"})
    maintain = (substitution.get("counterexample") or {}).get("maintain") or {}
    if maintain.get("defined") and maintain["rate"] >= MAINTAIN_HIGH:
        suggestions.append({"action": "检查节点是否使用了这条输入", "node": "",
                            "evidence": f"反义替换后仍维持原判断 {maintain['n']}/{maintain['denominator']}"})
    return suggestions


def run_four_tests(settings: Any, runs: list[dict], measured: dict, generator, runner: Runner | None) -> dict[str, Any]:
    """Run the four tests once. A negative repeat is not a repetition count."""
    prepared = prepare_substitutions(runs, generator)
    sampling = run_sampling(settings, runs, runner)
    sufficiency = run_sufficiency(runs, runner)
    necessity = run_necessity(runs, runner)
    substitution = execute_substitutions(prepared, runner)
    necessity_rows = necessity["by_condition"]
    return {
        "mode": "four",
        "repeat": settings.repeat,
        "tests_ran": True,
        "sampling": sampling,
        "sufficiency": sufficiency,
        "necessity": necessity,
        "substitution": substitution,
        "faults": fault_sources(measured, runs, necessity_rows),
        "upstream_fail": upstream_fail_rates(measured),
        "suggestions": structural_suggestions(measured, necessity_rows) + wording_suggestions(runs, substitution),
        "source_edited": False,
    }
