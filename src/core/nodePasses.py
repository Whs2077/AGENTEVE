"""Raw node visit and assessment records. No reachability and no summary scores."""

import json
from pathlib import Path


def nodePassRecords(runs):
    """Expand each run into visit events and assessment events without dropping either."""
    records = []
    for row in runs:
        run_id = row.get("run_id")
        task_id = row.get("task_id")
        if row.get("infrastructure_error"):
            records.append({"event": "infrastructure_error", "run_id": run_id, "task_id": task_id,
                            "message": row["infrastructure_error"]})
            continue
        for index, node in enumerate(row.get("path") or []):
            records.append({"event": "visit", "run_id": run_id, "task_id": task_id,
                            "node": node, "visit_index": index})
        for node, passed in (row.get("node_completion") or {}).items():
            records.append({"event": "assessment", "run_id": run_id, "task_id": task_id,
                            "node": node, "passed": passed if isinstance(passed, bool) else None})
    return records


def passCounts(records):
    """Count visits and assessments separately. Repeated visits stay in the visit total."""
    counts = {}
    for record in records:
        node = record.get("node")
        if not node:
            continue
        bucket = counts.setdefault(node, {"visits": 0, "passed": 0, "failed": 0, "unassessed": 0})
        if record["event"] == "visit":
            bucket["visits"] += 1
        elif record["event"] == "assessment":
            if record["passed"] is True:
                bucket["passed"] += 1
            elif record["passed"] is False:
                bucket["failed"] += 1
            else:
                bucket["unassessed"] += 1
    return counts


def writeNodePasses(path, runs):
    """Write one JSON object per line."""
    records = nodePassRecords(runs)
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    Path(path).write_text(text, encoding="utf-8")
    return records
