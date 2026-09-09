"""CPU fake-HTTP tests for fixed-suite evidence and aggregation contracts.

No server or socket is opened, and generated responses are never executed.
"""

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

import collect_acceptance_suite as suite
from test_collect_acceptance_trace import collector, fixture, records


def plan_fixture(*, repeats=1):
    return {
        "suite_id": "cpu-fixed-suite",
        "max_tokens": 512,
        "repeats": repeats,
        "expected_server_configuration": {
            "device": "npu",
            "speculative_algorithm": "DSPARK",
            "model_path": "/target",
            "speculative_draft_model_path": "/draft",
            "tp_size": 16,
            "dp_size": 1,
            "nnodes": 1,
            "disable_cuda_graph": True,
        },
        "cases": [
            {
                "id": "zh_case",
                "category": "Chinese",
                "prompt": "请解释内存与硬盘的区别。",
                "manual_review": "Check factual accuracy and completion manually.",
            },
            {
                "id": "code_case",
                "category": "Code",
                "prompt": "Write a Python function that preserves unique items.",
                "manual_review": "Read the code; do not execute it.",
            },
        ],
    }


class SuiteOpener:
    """Retain prior rid records and script faults entirely inside the process."""

    def __init__(self, plan, *, counts=None, faults=None, token_offsets=None):
        self.configuration = copy.deepcopy(plan["expected_server_configuration"])
        self.counts = counts or [(0, 2, 0)] * (len(plan["cases"]) * plan["repeats"])
        self.faults = faults or {}
        self.token_offsets = token_offsets or {}
        self.calls = []
        self.requests = []
        self.responses = []
        self.history = copy.deepcopy(records(fixture(rid="historical-request")[0]))

    def open(self, request, timeout):
        index = len(self.calls)
        method = request.get_method()
        self.calls.append((method, request.full_url, timeout))
        if method == "POST":
            payload = json.loads(request.data)
            item = len(self.requests)
            self.requests.append(payload)
            info, value = fixture(self.counts[item], rid=payload["rid"])
            offset = self.token_offsets.get(item, 0)
            if offset:
                for record in records(info):
                    row = record["reqs"][0]
                    row["draft_tokens"] = [
                        token + offset for token in row["draft_tokens"]
                    ]
                    row["bonus_token"] += offset
                choice = value["choices"][0]
                choice["response_token_ids"] = [
                    token + offset for token in choice["response_token_ids"]
                ]
            value["choices"][0]["message"] = {
                "role": "assistant",
                "content": "Identical visible answer for token-based comparisons.",
            }
            value["choices"][0]["meta_info"]["cached_tokens"] = item * 10
            value["usage"]["prompt_tokens"] = 23
            value["usage"]["prompt_tokens_details"] = {"cached_tokens": item * 10}
            self.history.extend(copy.deepcopy(records(info)))
            self.responses.append(value)
        elif method == "GET":
            value, _ = fixture(rid="unused")
            records(value)[:] = copy.deepcopy(self.history)
            value.update(self.configuration)
        else:
            raise AssertionError(f"Unexpected HTTP method: {method}")

        fault = self.faults.get(index)
        if isinstance(fault, BaseException):
            raise fault
        if callable(fault):
            fault(value)
        return io.BytesIO(json.dumps(value).encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text())


def summarized_row(counts, case_id, repeat=1):
    info, response = fixture(counts)
    summary, _ = collector.analyze_trace(info, response, response["id"])
    return {
        "case_id": case_id,
        "repeat": repeat,
        "directory": "unused",
        "status": summary["status"],
        "summary": summary,
    }


class FixedSuiteTests(unittest.TestCase):
    def setUp(self):
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.output = Path(contexts.enter_context(tempfile.TemporaryDirectory()))
        contexts.enter_context(redirect_stdout(io.StringIO()))

    def test_plan_preserves_fixed_inputs_and_rejects_ambiguous_or_empty_cases(self):
        plan = plan_fixture()
        path = self.output / "plan.json"
        path.write_text(json.dumps(plan))
        self.assertEqual(suite.load_plan(path), plan)
        changes = (
            lambda p: p.update(repeats=True),
            lambda p: p.update(max_tokens=0),
            lambda p: p.update(cases=[]),
            lambda p: p["cases"][1].update(id=p["cases"][0]["id"]),
            lambda p: p["cases"][0].update(id="../escape"),
            lambda p: p["cases"][0].update(prompt=" "),
            lambda p: p["cases"][0].update(manual_review=""),
        )
        for change in changes:
            with self.subTest(change=change):
                invalid = copy.deepcopy(plan)
                change(invalid)
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    suite.load_plan(path)

    def test_complete_suite_uses_weighted_counts_not_mean_request_percentages(self):
        plan = plan_fixture()
        opener = SuiteOpener(plan, counts=[(8,), (0,) * 9])
        report = suite.run_suite("http://fixture", self.output, plan, opener=opener)
        self.assertEqual(report["status"], "SUITE_COLLECTED")
        aggregate = report["aggregate"]
        self.assertEqual(aggregate["accepted_drafts"], 8)
        self.assertEqual(aggregate["proposed_drafts"], 80)
        self.assertEqual(aggregate["verify_rounds"], 10)
        self.assertEqual(aggregate["accept_rate"], 0.1)
        self.assertNotEqual(aggregate["accept_rate"], (1.0 + 0.0) / 2)
        self.assertEqual(aggregate["first_draft_accepted_rounds"], 1)
        self.assertIs(aggregate["strictly_above_user_reference_0_5"], False)
        self.assertEqual(report["quality_review"], "NOT_REVIEWED")
        self.assertEqual(report["length_finished_requests"], 2)
        self.assertTrue(all(row["output_budget_reached"] for row in report["cases"]))
        self.assertEqual(read_json(self.output / "suite-summary.json"), report)

    def test_reference_boundary_is_strictly_greater_than_fifty_percent(self):
        plan = plan_fixture()
        plan["cases"] = plan["cases"][:1]
        for accepted in (3, 4, 5):
            with self.subTest(accepted=accepted):
                row = summarized_row((accepted,), plan["cases"][0]["id"])
                report = suite.summarize(plan, [row])
                self.assertEqual(report["aggregate"]["accept_rate"], accepted / 8)
                self.assertIs(
                    report["aggregate"]["strictly_above_user_reference_0_5"],
                    accepted > 4,
                )
                self.assertEqual(report["quality_review"], "NOT_REVIEWED")

    def test_repeat_order_unique_rids_token_comparisons_and_cached_metadata(self):
        plan = plan_fixture(repeats=2)
        opener = SuiteOpener(plan, token_offsets={2: 10000})
        report = suite.run_suite(
            "http://fixture/", self.output, plan, timeout=17, opener=opener
        )
        self.assertEqual(report["status"], "SUITE_COLLECTED")
        expected_cases = [case for _ in range(2) for case in plan["cases"]]
        self.assertEqual(
            [request["messages"][0]["content"] for request in opener.requests],
            [case["prompt"] for case in expected_cases],
        )
        rids = [request["rid"] for request in opener.requests]
        self.assertEqual(len(set(rids)), 4)
        self.assertEqual([row["repeat"] for row in report["cases"]], [1, 1, 2, 2])
        self.assertEqual(
            [row["cached_tokens"] for row in report["cases"]], [0, 10, 20, 30]
        )
        self.assertEqual(
            [row["same_output_token_ids"] for row in report["repeat_comparisons"]],
            [False, True],
        )
        self.assertEqual(
            {
                response["choices"][0]["message"]["content"]
                for response in opener.responses
            },
            {"Identical visible answer for token-based comparisons."},
        )
        self.assertEqual([call[0] for call in opener.calls], ["GET", "POST", "GET"] * 4)
        self.assertTrue(all(call[2] == 17 for call in opener.calls))
        for row, request in zip(report["cases"], opener.requests):
            directory = self.output / row["directory"]
            self.assertEqual(request["temperature"], 0)
            self.assertEqual(request["max_tokens"], 512)
            self.assertIs(request["stream"], False)
            self.assertEqual(read_json(directory / "request.json"), request)
            self.assertEqual(row["summary"]["rid"], request["rid"])
            trace = read_json(directory / "trace.json")
            self.assertTrue(
                all(
                    record["rid"] == request["rid"]
                    for record in trace["candidate_api_prefix"]
                )
            )
            response = read_json(directory / "response.json")
            self.assertEqual(
                response["usage"]["prompt_tokens_details"]["cached_tokens"],
                row["cached_tokens"],
            )

    def test_initial_later_and_after_response_configuration_changes_stop_sending(self):
        def wrong_config(value):
            value["tp_size"] = 8

        for failing_call, expected_posts, expected_trusted in (
            (0, 0, 0),
            (3, 1, 1),
            (2, 1, 0),
        ):
            with self.subTest(failing_call=failing_call):
                plan = plan_fixture()
                opener = SuiteOpener(plan, faults={failing_call: wrong_config})
                output = self.output / str(failing_call)
                report = suite.run_suite("http://fixture", output, plan, opener=opener)
                self.assertEqual(report["status"], "SUITE_INCOMPLETE")
                self.assertIsNone(report["aggregate"])
                self.assertEqual(report["trusted_requests"], expected_trusted)
                self.assertEqual(len(opener.requests), expected_posts)
                self.assertEqual(len(opener.calls), failing_call + 1)
                failed = report["cases"][-1]
                self.assertEqual(failed["status"], "COLLECTION_FAILED")
                self.assertIn("configuration differs", failed["error"])
                directory = output / failed["directory"]
                self.assertTrue((directory / "server_info.before.json").exists())
                if failing_call == 2:
                    self.assertTrue((directory / "response.json").exists())
                    self.assertEqual(
                        read_json(directory / "server_info.after.json")["tp_size"], 8
                    )
                else:
                    self.assertFalse((directory / "request.json").exists())

    def test_wrong_gamma_is_rejected_before_post_even_when_server_args_match(self):
        def wrong_window(value):
            payload = value["internal_states"][0]["dspark_info_record"]
            payload.update(gamma=4, verify_num_draft_tokens=5)

        plan = plan_fixture()
        opener = SuiteOpener(plan, faults={0: wrong_window})
        report = suite.run_suite("http://fixture", self.output, plan, opener=opener)
        self.assertEqual(report["status"], "SUITE_INCOMPLETE")
        self.assertEqual([call[0] for call in opener.calls], ["GET"])
        self.assertIn("gamma=8", report["cases"][0]["error"])
        self.assertIsNone(report["aggregate"])

    def test_trace_review_stops_without_promoting_the_trusted_prefix(self):
        def inconsistent_count(value):
            value["sglext"]["spec_tokens_details"]["spec_num_correct_drafts"] += 1

        plan = plan_fixture(repeats=2)
        opener = SuiteOpener(plan, faults={4: inconsistent_count})
        report = suite.run_suite("http://fixture", self.output, plan, opener=opener)
        self.assertEqual(report["status"], "SUITE_INCOMPLETE")
        self.assertIsNone(report["aggregate"])
        self.assertEqual(report["attempted_requests"], 2)
        self.assertEqual(report["not_attempted_requests"], 2)
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(
            report["trusted_prefix_totals"],
            {"accepted_drafts": 2, "proposed_drafts": 24, "verify_rounds": 3},
        )
        row = report["cases"][-1]
        self.assertEqual(row["status"], "TRACE_REVIEW_REQUIRED")
        self.assertTrue((self.output / row["directory"] / "trace.json").exists())

    def test_http_failure_or_timeout_preserves_completed_raw_files_and_never_retries(
        self,
    ):
        for failing_call in (1, 2, 4):
            for failure in (
                TimeoutError("simulated timeout after possible server execution"),
                HTTPError("http://fixture", 503, "unavailable", None, None),
            ):
                with self.subTest(
                    failing_call=failing_call, failure=type(failure).__name__
                ):
                    plan = plan_fixture(repeats=2)
                    opener = SuiteOpener(plan, faults={failing_call: failure})
                    output = self.output / f"{failing_call}-{type(failure).__name__}"
                    report = suite.run_suite(
                        "http://fixture", output, plan, opener=opener
                    )
                    self.assertEqual(report["status"], "SUITE_INCOMPLETE")
                    self.assertIsNone(report["aggregate"])
                    self.assertEqual(len(opener.calls), failing_call + 1)
                    self.assertEqual(
                        len(opener.requests), 2 if failing_call == 4 else 1
                    )
                    row = report["cases"][-1]
                    directory = output / row["directory"]
                    self.assertIn(type(failure).__name__, row["error"])
                    self.assertTrue((directory / "server_info.before.json").exists())
                    self.assertTrue((directory / "request.json").exists())
                    self.assertEqual(
                        (directory / "response.json").exists(), failing_call == 2
                    )
                    self.assertEqual(read_json(output / "suite-summary.json"), report)
                    self.assertEqual(read_json(directory / "suite-case.json"), row)

    def test_wrong_response_rid_preserves_raw_evidence_and_stops(self):
        def wrong_rid(value):
            value["id"] = "a-different-request"

        plan = plan_fixture()
        opener = SuiteOpener(plan, faults={1: wrong_rid})
        report = suite.run_suite("http://fixture", self.output, plan, opener=opener)
        self.assertEqual(report["status"], "SUITE_INCOMPLETE")
        self.assertIsNone(report["aggregate"])
        self.assertEqual(len(opener.requests), 1)
        directory = self.output / report["cases"][0]["directory"]
        self.assertEqual(
            read_json(directory / "response.json")["id"], "a-different-request"
        )
        self.assertTrue((directory / "server_info.after.json").exists())

    def test_interrupt_records_the_inflight_attempt_without_sending_the_next_case(self):
        plan = plan_fixture(repeats=2)
        opener = SuiteOpener(
            plan, faults={4: KeyboardInterrupt("user stopped collection")}
        )
        report = suite.run_suite("http://fixture", self.output, plan, opener=opener)
        self.assertEqual(report["status"], "SUITE_INCOMPLETE")
        self.assertEqual(report["attempted_requests"], 2)
        self.assertEqual(report["not_attempted_requests"], 2)
        self.assertEqual(report["trusted_requests"], 1)
        self.assertIsNone(report["aggregate"])
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(len(opener.calls), 5)
        row = report["cases"][-1]
        self.assertEqual(row["status"], "INTERRUPTED")
        self.assertIn("KeyboardInterrupt", row["error"])
        directory = self.output / row["directory"]
        self.assertTrue((directory / "server_info.before.json").exists())
        self.assertTrue((directory / "request.json").exists())
        self.assertFalse((directory / "response.json").exists())
        self.assertEqual(read_json(directory / "suite-case.json"), row)
        self.assertEqual(read_json(self.output / "suite-summary.json"), report)

    def test_natural_eos_without_verify_rounds_is_not_a_zero_acceptance_rate(self):
        def natural_eos(value):
            value["choices"][0]["finish_reason"] = "stop"

        plan = plan_fixture()
        opener = SuiteOpener(plan, counts=[(), (2,)], faults={1: natural_eos})
        report = suite.run_suite("http://fixture", self.output, plan, opener=opener)
        self.assertEqual(report["status"], "SUITE_INCOMPLETE")
        self.assertIsNone(report["aggregate"])
        self.assertEqual(report["trusted_requests"], 0)
        self.assertEqual(report["cases"][0]["status"], "NO_SPECULATIVE_ROUNDS")
        self.assertNotIn("summary", report["cases"][0])
        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(report["quality_review"], "NOT_REVIEWED")

    def test_legacy_single_request_default_prompt_and_max_tokens_are_unchanged(self):
        opener = SuiteOpener(plan_fixture())
        summary = collector.collect("http://fixture", self.output, 64, opener=opener)
        self.assertEqual(summary["status"], "TRACE_COLLECTED")
        self.assertEqual(
            opener.requests[0]["messages"],
            [{"role": "user", "content": "请用三句话解释为什么天空看起来是蓝色的。"}],
        )
        self.assertEqual(opener.requests[0]["max_tokens"], 64)
        self.assertEqual([call[0] for call in opener.calls], ["GET", "POST", "GET"])

    def test_generated_code_is_only_saved_as_fenced_text_for_manual_review(self):
        generated = '```python\nraise RuntimeError("never execute generated code")\n```'

        def code_answer(value):
            value["choices"][0]["message"]["content"] = generated

        plan = plan_fixture()
        opener = SuiteOpener(plan, faults={1: code_answer})
        report = suite.run_suite("http://fixture", self.output, plan, opener=opener)
        self.assertEqual(report["status"], "SUITE_COLLECTED")
        answers = (self.output / "responses.md").read_text()
        self.assertIn("````text\n" + generated + "\n````", answers)
        self.assertIn("NOT_REVIEWED", answers)
        self.assertEqual(report["quality_review"], "NOT_REVIEWED")

    def test_cli_records_exact_input_fingerprints_without_using_a_network(self):
        plan = plan_fixture()
        plan_path = self.output / "plan.json"
        plan_path.write_text(json.dumps(plan))
        opener = SuiteOpener(plan)
        argv = [
            "collect_acceptance_suite.py",
            "--url",
            "http://fixture",
            "--evidence-root",
            str(self.output / "evidence"),
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(suite, "PLAN_PATH", plan_path),
            patch.object(
                suite.collector.urllib.request, "build_opener", return_value=opener
            ),
            patch.object(
                suite.subprocess,
                "run",
                return_value=SimpleNamespace(stdout="fixture-head\n"),
            ),
        ):
            self.assertEqual(suite.main(), 0)
        directories = list((self.output / "evidence").glob("acceptance-suite-*"))
        self.assertEqual(len(directories), 1)
        output = directories[0]
        identity = read_json(output / "collector.json")
        self.assertEqual(identity["collector_git_head"], "fixture-head")
        for path in (Path(suite.__file__), Path(suite.collector.__file__), plan_path):
            self.assertEqual(
                identity["files_sha256"][path.name],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        self.assertEqual(read_json(output / "suite-plan.json"), plan)
        self.assertEqual(
            read_json(output / "suite-summary.json")["status"], "SUITE_COLLECTED"
        )
        self.assertEqual(len(opener.requests), 2)


if __name__ == "__main__":
    unittest.main()
