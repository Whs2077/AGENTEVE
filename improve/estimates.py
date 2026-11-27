"""pass@1, pass@k, pass^k, and the first passing attempt. No model calls."""

from __future__ import annotations

import math
from typing import Any

from src.core.semanticRecord import assessed_pass, defined_rate, undefined_rate, wilson_interval


def binomial(n: int, k: int) -> int:
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> dict[str, Any]:
    """Unbiased estimate 1 - C(n-c, k) / C(n, k). The interval is defined only for k = 1."""
    if n <= 0 or k <= 0:
        return undefined_rate("没有已验收的运行")
    if k > n:
        return undefined_rate(f"观测次数 n={n} 小于 k={k}")
    numer = binomial(n - c, k)
    denom = binomial(n, k)
    estimate = 1 - numer / denom if denom else None
    row = {
        "defined": True, "k": k, "n": n, "c": c,
        "numerator": numer, "denominator": denom, "estimate": estimate,
        "text": f"1 - {numer}/{denom}",
    }
    if k == 1:
        row["interval"] = wilson_interval(c, n)
    else:
        row["interval"] = {
            "defined": False,
            "reason": "pass@k 在 k>1 时是组合估计，不是独立伯努利比例，区间无定义",
        }
    return row


def pass_to_the_k(n: int, c: int, k: int) -> dict[str, Any]:
    """Estimate that all k draws pass: C(c, k) / C(n, k)."""
    if n <= 0 or k <= 0:
        return undefined_rate("没有已验收的运行")
    if k > n:
        return undefined_rate(f"观测次数 n={n} 小于 k={k}")
    numer = binomial(c, k)
    denom = binomial(n, k)
    return {
        "defined": True, "k": k, "n": n, "c": c,
        "numerator": numer, "denominator": denom,
        "estimate": numer / denom if denom else None,
        "text": f"{numer}/{denom}",
        "interval": {"defined": False, "reason": "pass^k 是组合估计，区间无定义"} if k != 1
        else wilson_interval(c, n),
    }


def first_pass_attempt(flags: list[bool | None]) -> dict[str, Any]:
    """1-based index among assessed runs. Format failures are already None and are skipped."""
    assessed = 0
    for flag in flags:
        if flag is None:
            continue
        assessed += 1
        if flag:
            return {"defined": True, "attempt": assessed}
    if assessed == 0:
        return {"defined": False, "attempt": None, "reason": "没有已验收的运行"}
    return {"defined": False, "attempt": None, "reason": "观测次数内没有通过"}


def node_flags(runs: list[dict], node: str) -> list[bool | None]:
    return [assessed_pass(row, node) for row in runs]


def summarize_flags(flags: list[bool | None], k: int) -> dict[str, Any]:
    assessed = [flag for flag in flags if flag is not None]
    n, c = len(assessed), sum(bool(flag) for flag in assessed)
    return {
        "pass_at_1": defined_rate(c, n, "没有已验收的运行") if n else undefined_rate("没有已验收的运行"),
        "pass_at_k": pass_at_k(n, c, k),
        "pass_to_the_k": pass_to_the_k(n, c, k),
        "first_pass": first_pass_attempt(flags),
        "n": n,
        "c": c,
    }


def undefined_multi(reason: str) -> dict[str, Any]:
    """Metrics a single run cannot determine."""
    blank = undefined_rate(reason)
    return {"pass_at_1": blank, "pass_at_k": blank, "pass_to_the_k": blank,
            "first_pass": {"defined": False, "attempt": None, "reason": reason}, "n": 0, "c": 0}
