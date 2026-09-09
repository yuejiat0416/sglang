"""CPU checks for evidence attribution; these do not exercise DSpark or an NPU."""

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("collect_acceptance_trace.py")
SPEC = importlib.util.spec_from_file_location("collect_acceptance_trace", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)
RID = "ms1-trace-fixture"


def fixture(counts=(0, 2, 0), *, extra=(), rid=RID):
    """Static rows use eight proposals; accepted proposals plus bonus are output.

    The first response token comes from prefill. Worker-only overlap rows are
    recorded after the API-counted prefix and never contribute to API counters.
    """
    records, output = [], [700]
    prefix_len = 23
    for index, count in enumerate(tuple(counts) + tuple(extra)):
        drafts = list(range(1000 + 10 * index, 1008 + 10 * index))
        bonus = 2000 + index
        records.append(
            {
                "forward_ct": 40 + 2 * index,
                "reqs": [
                    {
                        "rid": rid,
                        "req_pool_index": 0,
                        "prefix_len": prefix_len,
                        "verify_len": 9,
                        "acc_len": count + 1,
                        "correct_drafts": count,
                        "cap_trim": 0,
                        "draft_tokens": drafts,
                        "bonus_token": bonus,
                    }
                ],
            }
        )
        prefix_len += count + 1
        if index < len(counts):
            output.extend(drafts[:count] + [bonus])
    payload = {
        "components": ["core", "reqs"],
        "mode": "static",
        "gamma": 8,
        "verify_num_draft_tokens": 9,
        "simulate_acc_len": None,
        "records": records,
    }
    response = {
        "id": rid,
        "choices": [
            {
                "finish_reason": "length",
                "response_token_ids": output,
                "meta_info": {"num_retractions": 0},
            }
        ],
        "usage": {"completion_tokens": len(output)},
        "sglext": {
            "spec_tokens_details": {
                "spec_verify_ct": len(counts),
                "spec_num_correct_drafts": sum(counts),
                "spec_num_proposed_drafts": 8 * len(counts),
                "spec_correct_drafts_histogram": [counts.count(i) for i in range(9)],
            }
        },
    }
    return {"internal_states": [{"dspark_info_record": payload}]}, response


def records(info):
    return info["internal_states"][0]["dspark_info_record"]["records"]


class TraceAnalysisTests(unittest.TestCase):
    def test_filters_other_requests_and_sorts_real_forward_order(self):
        info, response = fixture()
        foreign = copy.deepcopy(records(info)[0])
        foreign["forward_ct"] = 1
        foreign["reqs"][0]["rid"] = "old-request"
        records(info).append(foreign)
        records(info)[0]["reqs"].append(copy.deepcopy(foreign["reqs"][0]))
        records(info).reverse()
        summary, trace = collector.analyze_trace(info, response, RID)
        self.assertEqual(summary["status"], "TRACE_COLLECTED")
        self.assertEqual(
            summary["candidate_prefix_accepted_drafts_by_round"], [0, 2, 0]
        )
        self.assertEqual(
            [r["forward_ct"] for r in trace["candidate_api_prefix"]], [40, 42, 44]
        )
        self.assertEqual(summary["first_draft_accepted_rounds"], 1)
        self.assertEqual(summary["recorded_worker_rounds"], 3)

    def test_same_histogram_keeps_different_chronology(self):
        summaries = [
            collector.analyze_trace(*fixture(counts), RID)[0]
            for counts in ((2, 0, 0), (0, 0, 2))
        ]
        self.assertEqual(summaries[0]["api"], summaries[1]["api"])
        self.assertEqual(
            [s["candidate_prefix_trailing_zero_rounds"] for s in summaries], [2, 0]
        )
        self.assertNotEqual(
            summaries[0]["candidate_prefix_accepted_drafts_by_round"],
            summaries[1]["candidate_prefix_accepted_drafts_by_round"],
        )

    def test_finish_trimming_does_not_rewrite_worker_acceptance(self):
        for reason in ("stop", "length"):
            with self.subTest(reason=reason):
                info, response = fixture((2, 3))
                choice = response["choices"][0]
                choice["finish_reason"] = reason
                choice["response_token_ids"] = choice["response_token_ids"][:-2]
                response["usage"]["completion_tokens"] -= 2
                summary, _ = collector.analyze_trace(info, response, RID)
                self.assertEqual(summary["status"], "TRACE_COLLECTED")
                self.assertEqual(summary["api"]["accepted_drafts"], 5)
                self.assertEqual(summary["api"]["accept_rate"], 5 / 16)
                self.assertEqual(
                    summary["candidate_output_tokens_before_finish_trimming"], 8
                )
                self.assertEqual(summary["completion_tokens"], 6)

    def test_multiple_unsettled_overlap_rows_are_retained(self):
        info, response = fixture(extra=(4, 8))
        summary, trace = collector.analyze_trace(info, response, RID)
        self.assertEqual(summary["status"], "TRACE_COLLECTED")
        self.assertEqual(summary["additional_worker_rows"], 2)
        self.assertEqual(
            [r["correct_drafts"] for r in trace["additional_worker_rows"]], [4, 8]
        )
        self.assertEqual(summary["api"]["accepted_drafts"], 2)

    def test_counter_or_histogram_mismatch_requires_review(self):
        changes = {
            "spec_verify_ct": 4,
            "spec_num_correct_drafts": 1,
            "spec_num_proposed_drafts": 25,
            "spec_correct_drafts_histogram": None,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                info, response = fixture()
                response["sglext"]["spec_tokens_details"][key] = value
                summary, _ = collector.analyze_trace(info, response, RID)
                self.assertEqual(summary["status"], "TRACE_REVIEW_REQUIRED")
                self.assertFalse(summary["candidate_prefix_matches_api_counters"])

    def test_missing_round_requires_review(self):
        info, response = fixture()
        records(info).pop(1)
        summary, _ = collector.analyze_trace(info, response, RID)
        self.assertEqual(summary["status"], "TRACE_REVIEW_REQUIRED")
        self.assertFalse(summary["candidate_prefix_matches_api_counters"])

    def test_duplicate_round_or_rank_is_ambiguous(self):
        for duplicate_rank in (False, True):
            with self.subTest(duplicate_rank=duplicate_rank):
                info, response = fixture()
                if duplicate_rank:
                    info["internal_states"].append(
                        copy.deepcopy(info["internal_states"][0])
                    )
                else:
                    records(info).append(copy.deepcopy(records(info)[0]))
                with self.assertRaises(ValueError):
                    collector.analyze_trace(info, response, RID)

    def test_prefix_discontinuity_cannot_be_a_clean_trace(self):
        info, response = fixture()
        records(info)[1]["reqs"][0]["prefix_len"] += 9
        summary, _ = collector.analyze_trace(info, response, RID)
        self.assertEqual(summary["status"], "TRACE_REVIEW_REQUIRED")
        self.assertTrue(summary["prefix_discontinuities"])

    def test_retraction_or_missing_lifecycle_data_requires_review(self):
        for meta in ({"num_retractions": 1}, {}, None):
            with self.subTest(meta=meta):
                info, response = fixture()
                response["choices"][0]["meta_info"] = meta
                summary, _ = collector.analyze_trace(info, response, RID)
                self.assertEqual(summary["status"], "TRACE_REVIEW_REQUIRED")

    def test_output_disagreement_or_missing_ids_requires_review(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                info, response = fixture()
                if missing:
                    del response["choices"][0]["response_token_ids"]
                else:
                    response["choices"][0]["response_token_ids"][1] = 99999
                summary, _ = collector.analyze_trace(info, response, RID)
                self.assertEqual(summary["status"], "TRACE_REVIEW_REQUIRED")
                self.assertFalse(summary["candidate_prefix_matches_output_token_ids"])

    def test_wrong_response_id_never_falls_back_to_time(self):
        info, response = fixture()
        response["id"] = "another-request"
        with self.assertRaisesRegex(ValueError, "Response id"):
            collector.analyze_trace(info, response, RID)

    def test_incomplete_response_ids_cannot_match_an_empty_prefix(self):
        info, response = fixture()
        response["choices"][0]["response_token_ids"] = [700]
        response["usage"]["completion_tokens"] = 64
        summary, _ = collector.analyze_trace(info, response, RID)
        self.assertEqual(summary["status"], "TRACE_REVIEW_REQUIRED")
        self.assertFalse(summary["candidate_prefix_matches_output_token_ids"])

    def test_length_finish_cannot_trim_entire_prior_rounds(self):
        info, response = fixture()
        response["choices"][0]["response_token_ids"] = [700]
        response["usage"]["completion_tokens"] = 1
        summary, _ = collector.analyze_trace(info, response, RID)
        self.assertEqual(summary["status"], "TRACE_REVIEW_REQUIRED")

    def test_abort_requires_lifecycle_review(self):
        info, response = fixture()
        response["choices"][0]["finish_reason"] = "abort"
        summary, _ = collector.analyze_trace(info, response, RID)
        self.assertEqual(summary["status"], "TRACE_REVIEW_REQUIRED")


class FakeOpener:
    """In-process HTTP substitute; never opens a socket."""

    def __init__(self, before=None):
        self.before = before if before is not None else fixture(rid="history")[0]
        self.calls = []
        self.after = self.response = self.request = None

    def open(self, request, timeout):
        self.calls.append((request.get_method(), request.full_url, timeout))
        if request.get_method() == "POST":
            self.request = json.loads(request.data)
            self.after, self.response = fixture(rid=self.request["rid"])
            value = self.response
        else:
            value = self.before if self.response is None else self.after
        return io.BytesIO(json.dumps(value).encode("utf-8"))


class CollectionTests(unittest.TestCase):
    def test_missing_recording_sends_no_chat_and_saves_preflight(self):
        no_reqs = {
            "internal_states": [{"dspark_info_record": {"components": ["core"]}}]
        }
        for before in ({}, no_reqs):
            with self.subTest(before=before), tempfile.TemporaryDirectory() as temp:
                opener = FakeOpener(before)
                with self.assertRaisesRegex(ValueError, "No core,reqs"):
                    collector.collect("http://fixture", Path(temp), 64, opener=opener)
                self.assertEqual([call[0] for call in opener.calls], ["GET"])
                self.assertEqual(
                    json.loads((Path(temp) / "server_info.before.json").read_text()),
                    before,
                )
                self.assertFalse((Path(temp) / "request.json").exists())

    def test_nonstatic_or_simulated_recording_sends_no_chat(self):
        for key, value in (("mode", "compact"), ("simulate_acc_len", 3)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                before, _ = fixture()
                before["internal_states"][0]["dspark_info_record"][key] = value
                opener = FakeOpener(before)
                with self.assertRaises(ValueError):
                    collector.collect("http://fixture", Path(temp), 64, opener=opener)
                self.assertEqual([call[0] for call in opener.calls], ["GET"])

    def test_http_order_unique_rid_and_raw_evidence(self):
        rids = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temp:
                opener = FakeOpener()
                output = Path(temp)
                summary = collector.collect(
                    "http://fixture/", output, 64, timeout=19, opener=opener
                )
                self.assertEqual(
                    opener.calls,
                    [
                        ("GET", "http://fixture/server_info", 19),
                        ("POST", "http://fixture/v1/chat/completions", 19),
                        ("GET", "http://fixture/server_info", 19),
                    ],
                )
                self.assertEqual(summary["status"], "TRACE_COLLECTED")
                for field in (
                    "return_meta_info",
                    "return_token_ids",
                    "return_spec_tokens_details",
                ):
                    self.assertIs(opener.request[field], True)
                self.assertFalse(opener.request["stream"])
                self.assertEqual(opener.request["temperature"], 0)
                self.assertEqual(opener.request["max_tokens"], 64)
                self.assertTrue(opener.request["rid"].startswith("ms1-trace-"))
                rids.append(opener.request["rid"])
                for name, value in (
                    ("server_info.before", opener.before),
                    ("request", opener.request),
                    ("response", opener.response),
                    ("server_info.after", opener.after),
                    ("summary", summary),
                ):
                    self.assertEqual(
                        json.loads((output / f"{name}.json").read_text()), value
                    )
                trace = json.loads((output / "trace.json").read_text())
                self.assertEqual(len(trace["candidate_api_prefix"]), 3)
        self.assertNotEqual(*rids)

    def test_raw_evidence_survives_analysis_failure(self):
        class WrongIDOpener(FakeOpener):
            def open(self, request, timeout):
                result = super().open(request, timeout)
                if request.get_method() == "POST":
                    self.response["id"] = "incorrect-response-id"
                    return io.BytesIO(json.dumps(self.response).encode("utf-8"))
                return result

        with tempfile.TemporaryDirectory() as temp:
            opener = WrongIDOpener()
            output = Path(temp)
            with self.assertRaisesRegex(ValueError, "Response id"):
                collector.collect("http://fixture", output, 64, opener=opener)
            self.assertEqual(
                json.loads((output / "response.json").read_text()), opener.response
            )
            self.assertEqual(
                json.loads((output / "server_info.after.json").read_text()),
                opener.after,
            )
            self.assertFalse((output / "trace.json").exists())


if __name__ == "__main__":
    unittest.main()
