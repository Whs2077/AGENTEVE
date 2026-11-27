import json
from pathlib import Path
import tempfile
import unittest

from src.core.graphOutput import writeGraph
from src.core.nodePasses import nodePassRecords, passCounts
from src.postEvaluation import postEvaluation


GRAPH = {
    "schema_version": "1.0", "extractor": "test", "entry": "start", "scope": "test",
    "graph_kind": "llm_call_successors",
    "nodes": [
        {"id": "start", "kind": "entry", "label": "Start", "evidence": []},
        {"id": "model", "kind": "llm_call", "label": "Model", "evidence": [{"file": "a.py", "line": 3}]},
        {"id": "end", "kind": "terminal", "label": "End", "evidence": []},
    ],
    "edges": [
        {"source": "start", "target": "model", "route": "next", "kind": "next_call", "condition": None, "evidence": []},
        {"source": "model", "target": "end", "route": "next", "kind": "next_call", "condition": None, "evidence": []},
    ],
    "diagnostics": [],
}


class FlowOutput(unittest.TestCase):
    def test_graph_step_writes_json_and_html_without_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            writeGraph(directory, json.loads(json.dumps(GRAPH)))
            folder = Path(directory)
            stored = json.loads((folder / "callGraph.json").read_text(encoding="utf-8"))
            basis = json.loads((folder / "groundTruth.json").read_text(encoding="utf-8"))
            page = (folder / "callGraph.html").read_text(encoding="utf-8")
        self.assertFalse(stored["ground_truth"]["present"])
        self.assertEqual(basis["basis"], "project_analysis")
        self.assertIn("没有参考流图", page)
        self.assertIn(">Model<", page)
        self.assertLess(page.index('data-index="0"'), page.index('data-index="2"'))
        self.assertGreater(page.index('x="648"'), page.index('x="48"'))
        self.assertIn('"extractor": "test"', page)

    def test_reference_marks_the_flow_as_grounded(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            reference.write_text("{}", encoding="utf-8")
            writeGraph(directory, json.loads(json.dumps(GRAPH)), reference=reference)
            basis = json.loads((Path(directory) / "groundTruth.json").read_text(encoding="utf-8"))
            page = (Path(directory) / "callGraph.html").read_text(encoding="utf-8")
        self.assertTrue(basis["present"])
        self.assertEqual(basis["basis"], "reference")
        self.assertIn("参考流图", page)

    def test_pass_ledger_keeps_every_visit_and_assessment(self):
        runs = [
            {"run_id": "r1", "task_id": "t", "path": ["start", "model", "model", "end"],
             "node_completion": {"model": True}},
            {"run_id": "r2", "task_id": "t", "path": ["start", "model"],
             "node_completion": {"model": False}, "infrastructure_error": None},
        ]
        records = nodePassRecords(runs)
        visits = [item for item in records if item["event"] == "visit" and item["node"] == "model"]
        assessments = [item for item in records if item["event"] == "assessment"]
        self.assertEqual(len(visits), 3)
        self.assertEqual(assessments, [
            {"event": "assessment", "run_id": "r1", "task_id": "t", "node": "model", "passed": True},
            {"event": "assessment", "run_id": "r2", "task_id": "t", "node": "model", "passed": False},
        ])
        self.assertEqual(passCounts(records)["model"], {"visits": 3, "passed": 1, "failed": 1, "unassessed": 0})

    def test_post_evaluation_uses_the_ledger_and_edge_drops(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            graph = folder / "callGraph.json"
            trials = folder / "trials.jsonl"
            graph.write_text(json.dumps(GRAPH), encoding="utf-8")
            trials.write_text(
                '{"run_id":"r1","task_id":"t","path":["start","model","end"],"node_completion":{"model":true}}\n'
                '{"run_id":"r2","task_id":"t","path":["start","model"],"node_completion":{"model":false}}\n',
                encoding="utf-8")
            code = postEvaluation(graph, trials, folder / "post")
            post = folder / "post"
            ledger = [json.loads(line) for line in (post / "nodePasses.jsonl").read_text(encoding="utf-8").splitlines()]
            step = json.loads((post / "stepData.json").read_text(encoding="utf-8"))
            report = (post / "report.md").read_text(encoding="utf-8")
        self.assertEqual(code, 0)
        self.assertEqual(passCounts(ledger)["model"]["visits"], 2)
        self.assertTrue(any(item["lost_pairs"] for item in step["edge_drops"]["edges"]))
        self.assertIn("原始通过记录", report)
        self.assertIn("删边后的可达性", report)
