"""Dispatch improve by repeat. Disabled callers never import this module."""

from __future__ import annotations

from typing import Any, Callable

from common.runLog import info, warn
from improve.estimates import node_flags, summarize_flags
from improve.generator import model_generator
from improve.interventions import run_four_tests
from improve.settings import ImproveSettings
from src.core.semanticRecord import assessed_pass, defined_rate, format_rate, undefined_rate


def run_improve(settings: ImproveSettings, runs: list[dict], measured: dict, *,
                generator: Callable[[str, str], list[dict]] | None = None,
                node_runner: Callable[[dict], dict] | None = None) -> dict[str, Any]:
    """Run the branch selected by ``repeat``. Negative repeat is not a repetition count."""
    info(f"improve 已启用 repeat={settings.repeat} retain={settings.retain} sample_count={settings.sample_count}")
    if settings.repeat == 1:
        data = {"mode": "once", "repeat": 1, "tests_ran": False}
        return {"data": data, "markdown": _render_once()}
    if settings.repeat > 1:
        data = _multi(settings, runs, measured)
        return {"data": data, "markdown": _render_multi(data, measured)}
    generator = generator or model_generator(settings)
    data = run_four_tests(settings, runs, measured, generator, node_runner)
    if not data["substitution"].get("available"):
        warn(data["substitution"].get("reason") or "反例与同义代换未执行")
    return {"data": data, "markdown": _render_four(data, measured)}


def _ordered_groups(runs: list[dict]) -> dict[Any, list[dict]]:
    buckets: dict[Any, list[tuple[int, dict]]] = {}
    order = []
    for index, row in enumerate(runs):
        if row.get("infrastructure_error"):
            continue
        task = row.get("task_id")
        buckets.setdefault(task, []).append((index, row))
        if task not in order:
            order.append(task)
    grouped = {}
    for task in order:
        items = sorted(buckets[task], key=lambda pair: _sort_key(pair[1], pair[0]))
        grouped[task] = [row for _, row in items]
    return grouped


def _sort_key(row: dict, index: int) -> tuple:
    repeat = row.get("repeat")
    if isinstance(repeat, int) and not isinstance(repeat, bool):
        return (0, repeat, index)
    return (1, str(row.get("run_id")), index)


def _rollup(per_task: list[dict], flags: list[bool | None], k: int) -> dict[str, Any]:
    assessed = [flag for flag in flags if flag is not None]
    passed = sum(bool(flag) for flag in assessed)
    at_1 = defined_rate(passed, len(assessed), "没有已验收的运行") if assessed else undefined_rate("没有已验收的运行")
    estimates = [item["pass_at_k"] for item in per_task]
    powers = [item["pass_to_the_k"] for item in per_task]
    return {
        "pass_at_1": at_1,
        "pass_at_k": _combine(estimates, "有的题观测次数小于 k，pass@k 无定义"),
        "pass_to_the_k": _combine(powers, "有的题观测次数小于 k，pass^k 无定义"),
        "first_pass": per_task[0]["first_pass"] if len(per_task) == 1 else {
            "defined": bool(per_task) and all(item["first_pass"].get("defined") for item in per_task),
            "attempts": [item["first_pass"] for item in per_task],
            "reason": "" if per_task and all(item["first_pass"].get("defined") for item in per_task)
            else "有的题在观测次数内没有通过",
        },
        "per_task": per_task,
        "k": k,
    }


def _combine(rows: list[dict], missing: str) -> dict[str, Any]:
    if rows and len(rows) == 1:
        return rows[0]
    if not rows or not all(row.get("defined") for row in rows):
        return undefined_rate(missing)
    return {
        "defined": True,
        "estimate": sum(row["estimate"] for row in rows) / len(rows),
        "per_task": [{"text": row.get("text"), "numerator": row.get("numerator"),
                      "denominator": row.get("denominator")} for row in rows],
        "text": "各题估计的平均值",
        "interval": {"defined": False, "reason": "多题平均的组合估计，区间无定义"},
    }


def _first_sheet(flags: list[bool | None], k: int) -> dict[str, Any]:
    """Statistics from each task's first run. k>1 metrics stay undefined."""
    summary = summarize_flags(flags, 1)
    if k > 1:
        summary["pass_at_k"] = undefined_rate("第一次运行无法确定 k>1 的 pass@k")
        summary["pass_to_the_k"] = undefined_rate("第一次运行无法确定 k>1 的 pass^k")
        passed = [flag for flag in flags if flag is not None]
        if passed and all(passed):
            summary["first_pass"] = {"defined": True, "attempt": 1, "reason": ""}
        else:
            summary["first_pass"] = {
                "defined": False, "attempt": None,
                "reason": "只有第一次运行，未通过则无法知道首次通过出现在第几次",
            }
    return summary


def _multi(settings: ImproveSettings, runs: list[dict], measured: dict) -> dict[str, Any]:
    groups = _ordered_groups(runs)
    k = settings.repeat
    nodes = {}
    for node in measured.get("nodes") or {}:
        per_task = []
        all_flags: list[bool | None] = []
        first_flags: list[bool | None] = []
        for task, rows in groups.items():
            flags = node_flags(rows, node)
            all_flags.extend(flags)
            per_task.append({"task_id": task, **summarize_flags(flags, k)})
            first_flags.append(assessed_pass(rows[0], node) if rows else None)
        nodes[node] = {"all_runs": _rollup(per_task, all_flags, k), "first_run": _first_sheet(first_flags, k)}
    return {"mode": "multi", "repeat": settings.repeat, "k": k, "nodes": nodes, "tests_ran": False}


def _render_once() -> str:
    return "\n".join([
        "",
        "## improve",
        "",
        "repeat=1。只统计一次运行可确定的指标。多次运行指标没有计算。",
        "四项测试未执行，因此不写入 s-pass@1、s-pass@k、e-pass@1、e-pass@k、S 和 N。",
        "一次运行的到达、完成、格式、语义保持率、无输入依据的语义、Q、条件通过率、"
        "第一个未完成节点和与 next_call 不一致的转移见上一节。",
        "",
    ])


def _render_attempt(value: dict) -> str:
    if not value or not value.get("defined"):
        return f"无定义（{(value or {}).get('reason') or '无定义'}）"
    if value.get("attempt") is not None:
        return str(value["attempt"])
    attempts = value.get("attempts") or []
    shown = []
    for item in attempts:
        if isinstance(item, dict) and item.get("defined"):
            shown.append(str(item.get("attempt")))
        elif isinstance(item, int):
            shown.append(str(item))
        else:
            shown.append("无定义")
    return "、".join(shown) if shown else "无定义"


def _render_estimate(value: dict) -> str:
    if not value or not value.get("defined"):
        return f"无定义（{(value or {}).get('reason') or '无定义'}）"
    if value.get("text"):
        estimate = value.get("estimate")
        if isinstance(estimate, float):
            return f"{value['text']} = {estimate:.3f}"
        return str(value["text"])
    return format_rate(value)


def _render_multi(data: dict, measured: dict) -> str:
    lines = [
        "",
        "## improve",
        "",
        f"repeat={data['repeat']}。pass@k 与 pass^k 的 k 取 repeat。负号不是本模式。",
        "pass@k 的估计是 1 - C(n-c, k) / C(n, k)。pass^k 的估计是 C(c, k) / C(n, k)。",
        "比例写出分子和分母。k>1 的组合估计没有 Wilson 区间。",
        "每个节点另有一份只用第一次运行的统计。那一份里，一次运行确定不了的指标为无定义。",
        "四项测试未执行，因此不写入 s-pass@1、s-pass@k、e-pass@1、e-pass@k。",
        "",
    ]
    labels = {node: spec.get("label", node) for node, spec in (measured.get("nodes") or {}).items()}
    for node, row in data["nodes"].items():
        lines.append(f"### {labels.get(node, node)}")
        lines.append("")
        lines.append(f"- 全部运行 pass@1：{_render_estimate(row['all_runs']['pass_at_1'])}。")
        lines.append(f"- 全部运行 pass@k：{_render_estimate(row['all_runs']['pass_at_k'])}。")
        lines.append(f"- 全部运行 pass^k：{_render_estimate(row['all_runs']['pass_to_the_k'])}。")
        lines.append(f"- 全部运行首次通过所需次数：{_render_attempt(row['all_runs']['first_pass'])}。")
        lines.append(f"- 第一次运行 pass@1：{_render_estimate(row['first_run']['pass_at_1'])}。")
        lines.append(f"- 第一次运行 pass@k：{_render_estimate(row['first_run']['pass_at_k'])}。")
        lines.append(f"- 第一次运行 pass^k：{_render_estimate(row['first_run']['pass_to_the_k'])}。")
        lines.append(f"- 第一次运行首次通过所需次数：{_render_attempt(row['first_run']['first_pass'])}。")
        lines.append("")
    return "\n".join(lines)


def _render_conditions(rows: list[dict], once: str, later: str, effect: str) -> list[str]:
    if not rows:
        return ["没有可用条件。"]
    lines = []
    for row in rows:
        value = row.get(effect)
        if row.get("defined") and isinstance(value, float):
            effect_text = f"{value:.3f}"
        else:
            effect_text = "无定义"
        lines.append(
            f"- 节点 {row['node']} 通道 {row['channel']} 信息 {row['key']}："
            f"{once} {_render_estimate(row.get(once) or {})}，"
            f"{later} {_render_attempt(row.get(later) or {})}，{effect}={effect_text}。"
        )
    return lines


def _render_four(data: dict, measured: dict) -> str:
    sampling = data["sampling"]
    substitution = data["substitution"]
    lines = [
        "",
        "## improve",
        "",
        f"repeat={data['repeat']}。负号只表示执行一次四项测试，绝对值不是重复次数。",
        "同一题、同一个种子上比较，只改被考察的那一条信息。未修改被测源码。",
        "",
        "### 语义密集性采样",
        "",
    ]
    if sampling["drop"] is None:
        lines.append(f"完成率下降无定义（{sampling['before'].get('reason') or sampling['after'].get('reason') or '没有可比较的结果'}）。")
    else:
        lines.append(
            f"采样前 {format_rate(sampling['before'])}，采样后 {format_rate(sampling['after'])}，"
            f"下降 {sampling['drop']:.3f}。"
        )
    still = "、".join(sampling["still_passed"]) or "无"
    lost = "、".join(sampling["no_longer_passed"]) or "无"
    lines.append(f"仍然通过的题：{still}。不再通过的题：{lost}。")
    lines.append("retain=1 且抽取全部条数时，这一组是对照，输入与未采样时相同。"
                 if sampling["control"] else "这一组不是对照。")
    lines.extend(["", "### 充分性", ""])
    lines.extend(_render_conditions(data["sufficiency"]["by_condition"], "s-pass@1", "s-pass@k", "S"))
    lines.extend(["", "### 必要性", "", "格式不合格的样本没有进入必要性。", ""])
    lines.extend(_render_conditions(data["necessity"]["by_condition"], "e-pass@1", "e-pass@k", "N"))
    lines.extend(["", "### 反例与同义代换", ""])
    if not substitution.get("available"):
        lines.append(f"未执行：{substitution.get('reason') or '没有可用的生成模型'}。没有把这项写成通过。")
    else:
        lines.append(f"字符串匹配后保留 {len(substitution.get('accepted') or [])} 条，"
                     f"丢弃 {substitution.get('discarded', 0)} 条。")
        delta = substitution.get("synonym_delta") or {}
        if delta.get("defined"):
            lines.append(
                f"同义替换前 {format_rate(delta['before'])}，替换后 {format_rate(delta['after'])}，"
                f"通过率之差 {delta['difference']:.3f}。"
            )
        else:
            lines.append(f"同义替换前后的通过率之差无定义（{delta.get('reason') or '无定义'}）。")
        counter = substitution.get("counterexample") or {}
        lines.append(f"采用新表述的比例：{format_rate(counter.get('adopt') or {})}。")
        lines.append(f"维持原判断的比例：{format_rate(counter.get('maintain') or {})}。")
        lines.append(f"指出前后矛盾的比例：{format_rate(counter.get('contradict') or {})}。")
    position = substitution.get("position_control") or {}
    lines.extend(["", "### 位置控制", "", "文字不变，只改位置。这不是第五项测试。"])
    if position.get("defined"):
        lines.append(f"位置变化前后的通过率之差 {position.get('difference')}。")
    else:
        lines.append(f"无定义（{position.get('reason') or '无定义'}）。")
    lines.extend(["", "### R(u→v)", ""])
    if not data["upstream_fail"]:
        lines.append("没有同时验收的相邻节点，R(u→v) 无定义。")
    for item in data["upstream_fail"]:
        lines.append(f"- R({item['source']}→{item['target']}) = {format_rate(item['r'])}。")
    lines.extend(["", "### 逻辑异常的来源", ""])
    if not data["faults"]:
        lines.append("这次没有“上游未通过、下游通过”的运行。")
    for fault in data["faults"]:
        sources = "、".join(fault["sources"]) or "未归类"
        lines.append(f"- 运行 {fault['run_id']}，{fault['upstream']}→{fault['downstream']}："
                     f"主要来源 {fault['primary']}；命中的来源 {sources}。")
    lines.extend(["", "### 修改建议", "", "建议只写在报告里，不改被测源码。只有 N 约为 0 的冗余进入拆分、合并或删除。", ""])
    if not data["suggestions"]:
        lines.append("没有满足条件的修改建议。")
    for item in data["suggestions"]:
        target = item["node"] or "相关节点"
        lines.append(f"- {target}：{item['action']}。证据：{item['evidence']}。")
    lines.append("")
    return "\n".join(lines)
