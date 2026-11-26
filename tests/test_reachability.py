import unittest

from reachability import analyze, score_pair


class ReachabilityTest(unittest.TestCase):
    def test_pdf_example_keeps_run_pairing(self):
        upstream = {str(i): i < 80 for i in range(100)}
        downstream = {str(i): i < 60 or 80 <= i < 98 for i in range(100)}
        edge = score_pair(upstream, downstream, 100)

        self.assertEqual(edge["completion_count_ratio_q"]["rate"], 0.975)
        self.assertEqual(edge["target_pass_given_source_pass"]["rate"], 0.75)
        self.assertEqual(len(edge["necessary_condition_candidates"]), 18)
        self.assertEqual(len(edge["sufficient_condition_candidates"]), 20)
        self.assertEqual(edge["review_signal"], "near_one")

    def test_zero_denominator_and_incomplete_pair_coverage(self):
        edge = score_pair({"r1": False, "r2": True}, {"r1": True}, 2)

        self.assertEqual(edge["paired_assessments"]["rate"], 0.5)
        self.assertIsNone(edge["completion_count_ratio_q"]["rate"])
        self.assertIsNone(edge["target_pass_given_source_pass"]["rate"])
        self.assertEqual(edge["necessary_condition_candidates"], ["r1"])

    def test_counts_a_loop_as_multiple_visits_but_one_run(self):
        graph = {
            "entry": "start", "graph_kind": "llm_call_successors",
            "nodes": [{"id": node, "label": node, "kind": kind}
                      for node, kind in (("start", "entry"), ("model", "llm_call"), ("end", "exit"))],
            "edges": [{"source": source, "target": target, "kind": "next_call", "route": "next"}
                      for source, target in (("start", "model"), ("model", "model"), ("model", "end"))],
        }
        result = analyze(graph, [
            {"run_id": "r1", "task_id": "t1", "path": ["start", "model", "model", "end"],
             "node_completion": {"model": True}},
            {"run_id": "r2", "task_id": "t1", "path": ["start"], "node_completion": {"model": False}},
            {"run_id": "r3", "infrastructure_error": "timeout"},
        ])

        model = result["nodes"]["model"]
        self.assertEqual(model["cycle_path"], ("model", "model"))
        self.assertEqual(model["observed_runs"], ["r1"])
        self.assertEqual(model["visit_count"], 2)
        self.assertEqual(model["completion"]["rate"], 0.5)
        self.assertEqual(model["completion_by_task"]["t1"]["rate"], 0.5)
        self.assertEqual(model["exit_paths_lost_without_node"], ["end"])
        self.assertIsNone(result["nodes"]["end"]["completion"]["rate"])
        self.assertFalse(any("completion_count_ratio_q" in edge for edge in result["edges"]))
