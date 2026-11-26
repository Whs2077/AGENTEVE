import unittest

from stage2.paired_paged import DOCUMENT_IDS, PagedExchange
from stage2.paired_pilot import FINISH_TOOL, READ_TOOL, OutputError, tool_feedback


class Responses:
    def __init__(self, *actions):
        self.actions = iter(actions)
        self.calls = []

    def generate(self, messages, schema, phase):
        self.calls.append({"schema": schema, "phase": phase})
        return next(self.actions)


class PagedEvidenceTest(unittest.TestCase):
    def exchange(self, *actions):
        task = {
            "documents": [{"id": doc_id, "sentence": doc_id} for doc_id in DOCUMENT_IDS],
            "reference": {"answer": "yes", "citation_ids": ["D01", "D10"]},
        }
        exchange = PagedExchange(task, Responses({"tool": READ_TOOL}, *actions))
        self.assertEqual(exchange.generate([])[0], READ_TOOL)
        return exchange

    def test_second_read_exposes_new_ids_and_keeps_first_page(self):
        answer = {"answer": "yes", "citation_ids": ["D01", "D10"]}
        exchange = self.exchange({"tool": READ_TOOL}, answer)
        first = tool_feedback(READ_TOOL, exchange.read_evidence())
        self.assertEqual(exchange.generate([first])[0], READ_TOOL)
        second = tool_feedback(READ_TOOL, exchange.read_evidence())
        with self.assertRaisesRegex(OutputError, "previously read page"):
            exchange.generate([second])
        tool_name, arguments, _ = exchange.generate([first, second])
        self.assertEqual((tool_name, arguments), (FINISH_TOOL, answer))
        self.assertEqual([call["phase"] for call in exchange.model.calls],
                         ["request_evidence", "choose_next", "submit_answer"])

    def test_early_submission_is_allowed_but_only_with_seen_ids(self):
        answer = {"answer": "yes", "citation_ids": ["D01", "D02"]}
        exchange = self.exchange(answer)
        first = tool_feedback(READ_TOOL, exchange.read_evidence())
        self.assertEqual(exchange.generate([first])[0], FINISH_TOOL)
        exchange.submit(**answer)
        grade = exchange.grade()
        self.assertTrue(grade["answer_match"])
        self.assertFalse(grade["support_match"])
        self.assertTrue(grade["submitted_before_all_support_read"])

        invalid = self.exchange({"answer": "yes", "citation_ids": ["D01", "D10"]})
        with self.assertRaisesRegex(OutputError, "unread page"):
            invalid.generate([tool_feedback(READ_TOOL, invalid.read_evidence())])
