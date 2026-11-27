import json
from pathlib import Path
import tempfile
import unittest

from src.core.semanticRecord import heads_match, retention_of, summarize_single_run, unsupported_rate_of
from src.postEvaluation import locateLogicFaults, measureTrials, postEvaluation


GRAPH = {
    "entry": "start", "graph_kind": "llm_call_successors",
    "nodes": [
        {"id": "start", "label": "入口", "kind": "entry"},
        {"id": "brief", "label": "简报", "kind": "llm_call"},
        {"id": "supervisor", "label": "主管", "kind": "llm_call"},
    ],
    "edges": [
        {"source": "start", "target": "brief", "kind": "next_call", "route": "next"},
        {"source": "brief", "target": "supervisor", "kind": "next_call", "route": "next"},
    ],
}


def _run(run_id, brief, supervisor):
    return {
        "run_id": run_id, "task_id": "example",
        "path": ["brief", "supervisor"],
        "node_completion": {"brief": brief, "supervisor": supervisor},
    }


class RecordMetrics(unittest.TestCase):
    def test_old_trials_keep_q_and_leave_semantic_rates_undefined(self):
        runs = [_run("example-1", True, True), _run("example-2", True, False),
                _run("example-3", False, True), _run("example-4", False, False)]
        measured = measureTrials(GRAPH, runs)
        edge = next(item for item in measured["edges"] if item["source"] == "brief")
        self.assertEqual(edge["completion_count_ratio_q"], {"n": 2, "denominator": 2, "rate": 1})
        self.assertEqual(edge["target_pass_given_source_pass"]["n"], 1)
        self.assertEqual(edge["target_pass_given_source_pass"]["denominator"], 2)
        summary = summarize_single_run(runs, measured)
        self.assertFalse(summary["nodes"]["brief"]["semantic_retention"]["defined"])
        self.assertFalse(summary["nodes"]["supervisor"]["unsupported_share"]["defined"])
        self.assertFalse(summary["entailment"]["defined"])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            graph = folder / "callGraph.json"
            trials = folder / "trials.jsonl"
            graph.write_text(json.dumps(GRAPH), encoding="utf-8")
            trials.write_text("".join(json.dumps(row) + "\n" for row in runs), encoding="utf-8")
            code = postEvaluation(graph, trials, folder / "post")
            report = (folder / "post" / "report.md").read_text(encoding="utf-8")
            self.assertFalse((folder / "post" / "improve.md").exists())
        self.assertEqual(code, 0)
        self.assertIn("2/2", report)
        self.assertIn("1/2", report)
        self.assertIn("无定义", report)
        self.assertIn("语义保持率", report)

    def test_semantic_heads_use_alias_match_and_format_failures_stay_out(self):
        self.assertTrue(heads_match(
            {"key": "visits", "value": "visits", "aliases": ["visitors"]},
            {"key": "visits", "value": "visitors", "aliases": []},
        ))
        channels = {"user": [{"key": "city", "value": "Seattle", "aliases": []}]}
        output = [
            {"key": "city", "value": "Seattle", "aliases": []},
            {"key": "extra", "value": "invented", "aliases": []},
        ]
        self.assertEqual(retention_of(channels, output)["n"], 1)
        self.assertEqual(unsupported_rate_of(channels, output),
                         unsupported_rate_of(channels, output) | {"n": 1, "denominator": 2})
        graph = {
            "entry": "start", "graph_kind": "llm_call_successors",
            "nodes": [
                {"id": "start", "label": "入口", "kind": "entry"},
                {"id": "model", "label": "模型", "kind": "llm_call"},
            ],
            "edges": [{"source": "start", "target": "model", "kind": "next_call", "route": "next"}],
        }
        measured = measureTrials(graph, [
            {"run_id": "r1", "task_id": "t", "path": ["start", "model"], "node_completion": {"model": True},
             "semantic_heads": {"model": {"format_ok": False, "format_error": "reached the 288-token limit",
                                          "inputs": channels, "output": output}}},
            {"run_id": "r2", "task_id": "t", "path": ["start", "model"], "node_completion": {"model": True},
             "semantic_heads": {"model": {"format_ok": True, "inputs": channels, "output": output}}},
        ])
        self.assertEqual(measured["nodes"]["model"]["completion"]["n"], 1)
        self.assertEqual(measured["nodes"]["model"]["completion"]["denominator"], 1)
        self.assertEqual(measured["nodes"]["model"]["format_failures"], 1)
        summary = summarize_single_run([
            {"run_id": "r1", "task_id": "t", "path": ["start", "model"], "node_completion": {"model": True},
             "semantic_heads": {"model": {"format_ok": False, "format_error": "reached the 288-token limit",
                                          "inputs": channels, "output": output}}},
            {"run_id": "r2", "task_id": "t", "path": ["start", "model"], "node_completion": {"model": True},
             "semantic_heads": {"model": {"format_ok": True, "inputs": channels, "output": output}}},
        ], measured)
        self.assertEqual(summary["nodes"]["model"]["format_kinds"]["token_budget"], 1)
        self.assertEqual(summary["nodes"]["model"]["semantic_retention"]["n"], 2)
        self.assertEqual(summary["nodes"]["model"]["semantic_retention"]["denominator"], 2)
        self.assertEqual(summary["nodes"]["model"]["unsupported_share"]["n"], 2)
        self.assertEqual(summary["nodes"]["model"]["unsupported_share"]["denominator"], 4)

    def test_completion_ratio_above_one_is_a_logic_fault(self):
        graph = {
            "entry": "start", "graph_kind": "llm_call_successors",
            "nodes": [
                {"id": "start", "label": "入口", "kind": "entry"},
                {"id": "up", "label": "上游", "kind": "llm_call"},
                {"id": "down", "label": "下游", "kind": "llm_call"},
                {"id": "aside", "label": "旁支", "kind": "llm_call"},
            ],
            "edges": [
                {"source": "start", "target": "up", "kind": "next_call", "route": "next"},
                {"source": "up", "target": "down", "kind": "next_call", "route": "next"},
            ],
        }
        measured = measureTrials(graph, [
            {"run_id": "a", "task_id": "t", "path": ["start", "up", "down"],
             "node_completion": {"up": True, "down": True}},
            {"run_id": "b", "task_id": "t", "path": ["start", "up", "down"],
             "node_completion": {"up": False, "down": True}},
        ])
        reasons = {item["reason"] for item in locateLogicFaults(measured)}
        self.assertIn("completion_ratio_above_one", reasons)
        self.assertIn("unreachable", reasons)


if __name__ == "__main__":
    unittest.main()
