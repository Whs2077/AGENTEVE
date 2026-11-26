"""Let the same model choose whether to read another page or submit."""

from collections import Counter
import json
import re

from stage2.paired_pilot import Exchange, FINISH_TOOL, READ_TOOL, ROOT, OutputError, object_schema, tool_feedback


PAGED_TASKS = ROOT / "stage2/paired-hotpotqa-paged.json"
PAGE_SIZE = 5
DOCUMENT_IDS = [f"D{number:02d}" for number in range(1, 2 * PAGE_SIZE + 1)]
READ_RESPONSE = {"tool": READ_TOOL}
READ_SCHEMA = object_schema({"tool": {"enum": [READ_TOOL]}})


def evidence_layout(task):
    support = set(task["reference"]["citation_ids"])
    return "first_page" if support <= set(DOCUMENT_IDS[:PAGE_SIZE]) else "split_pages"


def validate_paged_tasks(cases):
    counts = Counter()
    ids = set()
    last_index = 48
    for task in cases:
        row_id = task["row_id"]
        docs = task["documents"]
        reference = task["reference"]
        support = set(reference["citation_ids"])
        if (not re.fullmatch(r"[0-9a-f]{24}", row_id) or row_id in ids or task["row_index"] <= last_index
                or [doc["id"] for doc in docs] != DOCUMENT_IDS or len({doc["title"] for doc in docs}) != len(docs)
                or not all(doc["sentence"].strip() for doc in docs)
                or len(reference["citation_ids"]) != 2 or len(support) != 2
                or not support <= set(DOCUMENT_IDS) or not support.intersection(DOCUMENT_IDS[:PAGE_SIZE])
                or reference["answer"] not in ("yes", "no")):
            raise ValueError(f"Invalid paged HotpotQA sample: {row_id}")
        counts[reference["answer"], evidence_layout(task)] += 1
        ids.add(row_id)
        last_index = task["row_index"]
    if counts != {(answer, layout): 2 for answer in ("yes", "no") for layout in ("first_page", "split_pages")}:
        raise ValueError("Expected two HotpotQA cases in each answer and evidence layout")


class PagedExchange(Exchange):
    call_limit = 3
    read_description = "Read the next of two pages, five numbered evidence documents per page."
    system_prompt = (
        "Answer the yes/no question using the evidence tools. Read page 1 first with read_evidence. "
        "After page 1, submit if you have both supporting documents. If a supporting document is missing, "
        "call read_evidence again to get page 2. After page 2, submit your best answer. "
        "Cite exactly two document IDs from pages you have read. Use the read response or the answer response "
        "as separate alternatives. Do not combine them in one response."
    )

    def __init__(self, task, model):
        super().__init__(task, model)
        self.pages = [task["documents"][:PAGE_SIZE], task["documents"][PAGE_SIZE:]]
        self.pages_read = 0

    def read_evidence(self):
        if self.pages_read >= len(self.pages):
            raise OutputError("There are no more evidence pages")
        docs = self.pages[self.pages_read]
        self.pages_read += 1
        payload = json.dumps({"page": self.pages_read, "pages_total": len(self.pages), "documents": docs},
                             ensure_ascii=False)
        self.events.append({"tool": READ_TOOL, "page": self.pages_read, "result": payload})
        return payload

    def generate(self, messages):
        if len(self.model.calls) >= self.call_limit:
            raise OutputError("Shared model-call budget reached")
        for event in self.events:
            if tool_feedback(READ_TOOL, event["result"]) not in messages:
                raise OutputError("The model did not receive a previously read page")
        seen_ids = DOCUMENT_IDS[:self.pages_read * PAGE_SIZE]
        answer_schema = object_schema({
            "answer": {"enum": ["yes", "no"]},
            "citation_ids": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"enum": seen_ids}},
        })
        if self.pages_read == 0:
            schema, phase = READ_SCHEMA, "request_evidence"
        elif self.pages_read < len(self.pages):
            schema, phase = {"anyOf": [READ_SCHEMA, answer_schema]}, "choose_next"
        else:
            schema, phase = answer_schema, "submit_answer"
        action = self.model.generate(messages, schema, phase)
        call_id = f"paged-{len(self.model.calls)}"
        if action == READ_RESPONSE and self.pages_read < len(self.pages):
            return READ_TOOL, {}, call_id
        citations = action.get("citation_ids")
        if (not self.pages_read or set(action) != {"answer", "citation_ids"} or action["answer"] not in ("yes", "no")
                or not isinstance(citations, list) or len(citations) != 2
                or any(citation not in seen_ids for citation in citations)):
            raise OutputError("Invalid paged action or citation from an unread page")
        return FINISH_TOOL, action, call_id

    def grade(self):
        scores = super().grade()
        scores["evidence_layout"] = evidence_layout(self.task)
        scores["pages_read"] = self.pages_read
        scores["submitted_before_all_support_read"] = (
            self.submission is not None and self.pages_read == 1 and scores["evidence_layout"] == "split_pages"
        )
        return scores


def summarize_paged(scores):
    by_layout = {}
    for layout in ("first_page", "split_pages"):
        group = [score for score in scores if score["evidence_layout"] == layout]
        by_layout[layout] = {
            "cases": len(group),
            "second_page_reads": sum(score["pages_read"] == 2 for score in group),
            "early_submissions_without_all_reference_evidence": sum(
                score["submitted_before_all_support_read"] for score in group
            ),
            "first_page_submissions": sum(score["pages_read"] == 1 and score["completion_observed"] for score in group),
            "joint_matches": sum(score["answer_match"] and score["support_match"] for score in group),
        }
    return {"by_evidence_layout": by_layout}
