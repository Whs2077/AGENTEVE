import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from improve.interventions import classify_fault, density_sample, match_replacement, structural_suggestions
from improve.runImprove import run_improve
from improve.settings import parse_improve
from src.postEvaluation import postEvaluation


ENABLED = {
    "enabled": True, "repeat": 1, "retain": 1, "sample_count": 1,
    "generator": {"modelName": "test-model", "apiUrl": "https://example.invalid/v1",
                  "apiKeyEnv": "AGENTEVE_IMPROVE_API_KEY"},
}


def _settings(**overrides):
    raw = {**ENABLED, **overrides}
    return parse_improve(raw)


class ImproveConfig(unittest.TestCase):
    def test_disabled_config_does_not_reject_a_low_retain(self):
        settings = parse_improve({"enabled": False, "retain": 0.1})
        self.assertFalse(settings.enabled)

    def test_retain_below_half_and_bad_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_improve({**ENABLED, "retain": 0.49})
        with self.assertRaises(ValueError):
            parse_improve({**ENABLED, "retain": 1.01})
        parse_improve({**ENABLED, "retain": 0.5})
        with self.assertRaises(ValueError):
            parse_improve({**ENABLED, "sample_count": 0})
        with self.assertRaises(ValueError):
            parse_improve({**ENABLED, "repeat": 0})


class ImproveRepeat(unittest.TestCase):
    def test_repeat_one_writes_only_single_run_metrics(self):
        result = run_improve(_settings(repeat=1), [], {"nodes": {}})
        self.assertEqual(result["data"]["mode"], "once")
        self.assertNotIn("pass_at_k", result["data"])
        self.assertIn("四项测试未执行", result["markdown"])
        self.assertNotIn("s-pass@1：", result["markdown"])

    def test_repeat_above_one_keeps_two_copies_and_the_binomial_estimate(self):
        runs = []
        passes = [False, True, True, False]
        for index, passed in enumerate(passes, 1):
            runs.append({
                "run_id": f"late-{index}", "task_id": "t", "repeat": index,
                "path": ["model"], "node_completion": {"model": passed},
            })
        runs.append({
            "run_id": "format", "task_id": "t", "repeat": 5, "path": ["model"],
            "node_completion": {"model": True},
            "semantic_heads": {"model": {"format_ok": False, "format_error": "Invalid tool-call structure"}},
        })
        runs = [runs[2], runs[0], runs[4], runs[1], runs[3]]
        measured = {"nodes": {"model": {"label": "模型"}}, "edges": []}
        result = run_improve(_settings(repeat=2), runs, measured)
        node = result["data"]["nodes"]["model"]
        estimate = node["all_runs"]["pass_at_k"]
        self.assertEqual(estimate["numerator"], 1)
        self.assertEqual(estimate["denominator"], 6)
        self.assertAlmostEqual(estimate["estimate"], 5 / 6)
        self.assertFalse(estimate["interval"]["defined"])
        self.assertEqual(node["all_runs"]["pass_at_1"]["n"], 2)
        self.assertEqual(node["all_runs"]["pass_at_1"]["denominator"], 4)
        self.assertEqual(node["all_runs"]["pass_to_the_k"]["numerator"], 1)
        self.assertEqual(node["all_runs"]["pass_to_the_k"]["denominator"], 6)
        self.assertEqual(node["all_runs"]["first_pass"]["attempt"], 2)
        self.assertFalse(node["first_run"]["pass_at_k"]["defined"])
        self.assertEqual(node["first_run"]["pass_at_1"]["n"], 0)
        self.assertEqual(node["first_run"]["pass_at_1"]["denominator"], 1)
        self.assertIn("1 - 1/6", result["markdown"])

    def test_negative_repeat_uses_a_fake_generator_and_does_not_call_the_network(self):
        calls = []

        def fake(prompt, relation):
            calls.append((prompt, relation))
            if relation == "synonym":
                return [
                    {"original": "必须", "replacement": "务必", "kind": "degree", "relation": "synonym"},
                    {"original": "不存在", "replacement": "别的", "kind": "degree", "relation": "synonym"},
                    {"original": "整段都改了。", "replacement": "不行。", "kind": "sentence", "relation": "synonym"},
                ]
            return [{"original": "必须", "replacement": "不必", "kind": "degree", "relation": "antonym"}]

        def runner(case):
            self.assertEqual(case["seed"], 7)
            keys = {head["key"] for head in case["heads"]}
            if case["mode"] == "sufficiency":
                return {"passed": case["condition"]["key"] == "keep"}
            if case["mode"] == "necessity":
                return {"passed": "keep" in keys}
            if case["mode"] == "sample":
                return {"passed": "keep" in keys}
            if case["mode"] == "counterexample":
                return {"passed": False, "reaction": "contradict"}
            return {"passed": True}

        runs = [{
            "run_id": "r1", "task_id": "t", "seed": 7, "path": ["model"],
            "node_completion": {"model": True},
            "semantic_heads": {"model": {
                "format_ok": True,
                "prompt": "必须在今天完成。",
                "inputs": {
                    "user": [{"key": "keep", "value": "必须保留"}],
                    "upstream": [{"key": "drop", "value": "可有可无"}],
                },
                "output": [{"key": "keep", "value": "必须保留"}],
            }},
        }]
        measured = {"nodes": {
            "upstream": {"label": "上游", "direct_successors": ["model"], "direct_predecessors": [],
                         "reachable_paths": {}},
            "model": {"label": "模型", "direct_successors": [], "direct_predecessors": ["upstream"],
                      "reachable_paths": {}},
        }, "edges": []}
        os.environ.pop("AGENTEVE_IMPROVE_API_KEY", None)
        with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            once = run_improve(_settings(repeat=-1, retain=1, sample_count=2), runs, measured,
                               generator=fake, node_runner=runner)
            self.assertEqual(len(calls), 2)
            calls.clear()
            again = run_improve(_settings(repeat=-4, retain=1, sample_count=2), runs, measured,
                                generator=fake, node_runner=runner)
        self.assertEqual(len(calls), 2)
        self.assertEqual(once["data"]["repeat"], -1)
        self.assertEqual(again["data"]["repeat"], -4)
        self.assertTrue(once["data"]["sampling"]["control"])
        self.assertEqual(once["data"]["substitution"]["discarded"], 2)
        self.assertFalse(once["data"]["source_edited"])
        conditions = {row["key"]: row for row in once["data"]["sufficiency"]["by_condition"]}
        self.assertEqual(conditions["keep"]["s-pass@1"]["n"], 1)
        self.assertEqual(conditions["drop"]["s-pass@1"]["n"], 0)
        self.assertEqual(conditions["keep"]["S"], 1)
        self.assertFalse(conditions["drop"]["s-pass@k"]["defined"])
        necessity = {row["key"]: row for row in once["data"]["necessity"]["by_condition"]}
        self.assertEqual(necessity["drop"]["N"], 0)
        self.assertEqual(necessity["keep"]["N"], 1)
        self.assertEqual(necessity["drop"]["e-pass@1"]["n"], 1)
        actions = [item["action"] for item in once["data"]["suggestions"]]
        self.assertIn("合并或删除该节点", actions)
        self.assertNotIn("删除这条冗余信息", actions)
        self.assertIn("s-pass@1", once["markdown"])
        self.assertIn("不是第五项测试", once["markdown"])

    def test_missing_generator_key_does_not_pretend_the_substitution_ran(self):
        runs = [{
            "run_id": "r1", "task_id": "t", "seed": 1, "path": ["model"],
            "node_completion": {"model": True},
            "semantic_heads": {"model": {"format_ok": True, "prompt": "必须在今天完成。",
                                         "inputs": {"user": [{"key": "keep", "value": "必须"}]}}},
        }]
        measured = {"nodes": {"model": {"label": "模型", "direct_successors": [], "direct_predecessors": []}},
                    "edges": []}
        os.environ.pop("AGENTEVE_IMPROVE_API_KEY", None)
        with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            result = run_improve(_settings(repeat=-1), runs, measured)
        self.assertFalse(result["data"]["substitution"]["available"])
        self.assertIn("未调用生成模型", result["data"]["substitution"]["reason"])
        self.assertIn("未执行", result["markdown"])


class ImprovePieces(unittest.TestCase):
    def test_sampling_control_and_string_match(self):
        heads = [{"key": "a", "value": "abcd"}, {"key": "b", "value": "wxyz"}]
        control = density_sample(heads, 2, 1, 3)
        self.assertTrue(control["control"])
        self.assertEqual([item["value"] for item in control["heads"]], ["abcd", "wxyz"])
        shortened = density_sample(heads, 2, 0.5, 3)
        self.assertFalse(shortened["control"])
        self.assertEqual([item["value"] for item in shortened["heads"]], ["ab", "wx"])
        prompt = "必须在今天完成。"
        accepted = match_replacement(prompt, {
            "original": "必须", "replacement": "务必", "kind": "degree", "relation": "synonym",
        }, "synonym")
        rejected = match_replacement(prompt, {
            "original": "必须在今天完成。请再写一句。", "replacement": "另一句。",
            "kind": "sentence", "relation": "synonym",
        }, "synonym")
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["prompt"], "务必在今天完成。")
        self.assertFalse(rejected["accepted"])

    def test_only_near_zero_n_suggests_deletion_and_match_miss_does_not(self):
        measured = {"nodes": {
            "upstream": {"direct_successors": ["down"], "direct_predecessors": [], "reachable_paths": {}},
            "down": {"direct_successors": [], "direct_predecessors": ["upstream"], "reachable_paths": {}},
        }}
        mixed = structural_suggestions(measured, [
            {"node": "down", "channel": "upstream", "key": "drop", "n": 0.0, "defined": True},
            {"node": "down", "channel": "upstream", "key": "keep", "n": 1.0, "defined": True},
        ])
        self.assertIn("删除这条冗余信息", [item["action"] for item in mixed])
        self.assertNotIn("合并或删除该节点", [item["action"] for item in mixed])
        fault = classify_fault(
            "upstream",
            {"raw_output": "已经写了目标", "output": []},
            {"inputs": {"upstream": [{"key": "goal", "value": "目标"}]}, "output": [{"key": "goal", "value": "目标"}]},
            [],
        )
        self.assertEqual(fault["primary"], "匹配漏检")
        self.assertNotIn("冗余", fault["sources"])
        self.assertFalse(structural_suggestions(measured, []))

    def test_post_evaluation_enabled_improve_stays_offline(self):
        graph = {
            "entry": "start", "graph_kind": "llm_call_successors",
            "nodes": [
                {"id": "start", "label": "入口", "kind": "entry"},
                {"id": "model", "label": "模型", "kind": "llm_call"},
            ],
            "edges": [{"source": "start", "target": "model", "kind": "next_call", "route": "next"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            graph_path = folder / "callGraph.json"
            trials = folder / "trials.jsonl"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            trials.write_text(json.dumps({
                "run_id": "r1", "task_id": "t", "path": ["start", "model"],
                "node_completion": {"model": True},
            }) + "\n", encoding="utf-8")
            os.environ.pop("AGENTEVE_IMPROVE_API_KEY", None)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                code = postEvaluation(graph_path, trials, folder / "post", improve={**ENABLED, "repeat": 1})
            report = (folder / "post" / "report.md").read_text(encoding="utf-8")
            self.assertTrue((folder / "post" / "improve.md").exists())
        self.assertEqual(code, 0)
        self.assertIn("repeat=1", report)


if __name__ == "__main__":
    unittest.main()
