"""CPU tests for replay attribution; no model execution or accuracy thresholds."""

import copy
import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import probe_verify_prefill as probe
from test_collect_acceptance_trace import fixture


def source_files(path, counts=(2, 0, 8), extra=(0,)):
    info, response = fixture(counts, extra=extra)
    info.update(
        {
            "device": "npu",
            "speculative_algorithm": "DSPARK",
            "disable_cuda_graph": True,
            "dp_size": 1,
            "pp_size": 1,
            "nnodes": 1,
            "tp_size": 16,
            "model_path": "/target",
            "speculative_draft_model_path": "/draft",
            "sampling_defaults": "openai",
        }
    )
    response["choices"][0]["prompt_token_ids"] = list(range(23))
    response["usage"]["prompt_tokens"] = 23
    request = {"rid": response["id"], "temperature": 0, "max_tokens": 64}
    for name, content in (
        ("request.json", request),
        ("response.json", response),
        ("server_info.after.json", info),
    ):
        (path / name).write_text(json.dumps(content))
    return info, response


def scoring_response(case, request):
    predictions = case["known_predictions"] + list(
        range(3000, 3000 + len(case["unknown_verify_positions"]))
    )
    tops = [[[-0.1, token, None], [-1.1, 9999, None]] for token in predictions]
    suffix = case["input_ids"][case["anchor_index"] :]
    return {
        "meta_info": {
            "id": request["rid"],
            "prompt_tokens": len(case["input_ids"]),
            "completion_tokens": 1,
            "cached_tokens": 0,
            "num_retractions": 0,
            "finish_reason": {"type": "length", "length": 1},
            "input_top_logprobs": [None] + tops[:-1],
            "output_top_logprobs": tops[-1:],
            "input_token_logprobs": [
                [None if i == 0 else -1.0, token, None]
                for i, token in enumerate(suffix)
            ],
            "output_token_logprobs": [[-0.1, predictions[-1], None]],
        }
    }


class TestSourceReconstruction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.info, self.response = source_files(self.path)

    def test_positions_and_only_accounted_rounds(self):
        plan = probe.read_source(self.path)
        self.assertEqual(plan["planned_requests"], 6)
        self.assertEqual(plan["excluded_worker_rows"], 1)
        self.assertEqual(
            [c["labels"] for c in plan["cases"]],
            [["first"], ["first_zero"], ["first_full"]],
        )
        output = self.response["choices"][0]["response_token_ids"]
        for case, offset in zip(plan["cases"], (0, 3, 4)):
            self.assertEqual(case["anchor_index"], 23 + offset)
            self.assertEqual(case["anchor_token"], output[offset])
            self.assertEqual(
                case["input_ids"][: 23 + offset + 1],
                list(range(23)) + output[: offset + 1],
            )
            self.assertEqual(len(case["input_ids"]), case["anchor_index"] + 9)
            self.assertEqual(
                len(case["known_predictions"]), case["accepted_drafts"] + 1
            )

    def test_selector_deduplicates_without_adding_requests(self):
        source_files(self.path, counts=(0, 8))
        plan = probe.read_source(self.path)
        self.assertEqual(plan["planned_requests"], 4)
        self.assertEqual(plan["cases"][0]["labels"], ["first", "first_zero"])

    def test_missing_category_does_not_trigger_new_source_generation(self):
        source_files(self.path, counts=(2, 3))
        plan = probe.read_source(self.path)
        self.assertEqual(plan["planned_requests"], 2)

    def test_tampered_api_attribution_rejected(self):
        self.response["sglext"]["spec_tokens_details"]["spec_num_correct_drafts"] += 1
        (self.path / "response.json").write_text(json.dumps(self.response))
        with self.assertRaises(ValueError):
            probe.read_source(self.path)

    def test_missing_prompt_ids_rejected(self):
        del self.response["choices"][0]["prompt_token_ids"]
        (self.path / "response.json").write_text(json.dumps(self.response))
        with self.assertRaises(ValueError):
            probe.read_source(self.path)

    def test_unsupported_source_adjustments_rejected(self):
        request = json.loads((self.path / "request.json").read_text())
        for key, value in (
            ("temperature", 1),
            ("repetition_penalty", 1.2),
            ("response_format", {"type": "json_object"}),
        ):
            with self.subTest(key=key):
                (self.path / "request.json").write_text(
                    json.dumps({**request, key: value})
                )
                with self.assertRaises(ValueError):
                    probe.read_source(self.path)

    def test_prefix_must_start_at_prompt(self):
        for record in self.info["internal_states"][0]["dspark_info_record"]["records"]:
            record["reqs"][0]["prefix_len"] += 1
        (self.path / "server_info.after.json").write_text(json.dumps(self.info))
        with self.assertRaisesRegex(ValueError, "continuous"):
            probe.read_source(self.path)

    def test_model_sampling_defaults_are_read_from_file(self):
        info = {"sampling_defaults": "model", "model_path": str(self.path)}
        result = probe.sampling_evidence({}, info)
        self.assertIsNone(result["generation_config_sha256"])
        path = self.path / "generation_config.json"
        path.write_text(json.dumps({"repetition_penalty": 1.0, "top_p": 0.95}))
        result = probe.sampling_evidence({}, info)
        self.assertEqual(result["effective_repetition_penalty"], 1)
        self.assertEqual(len(result["generation_config_sha256"]), 64)
        path.write_text(json.dumps({"repetition_penalty": 1.1}))
        with self.assertRaisesRegex(ValueError, "repetition_penalty"):
            probe.sampling_evidence({}, info)
        explicit = probe.sampling_evidence({"repetition_penalty": 1}, info)
        self.assertEqual(explicit["repetition_source"], "explicit source request")

    def test_unavailable_sampling_defaults_do_not_guess(self):
        for info in (
            {},
            {"sampling_defaults": "model", "model_path": str(self.path / "missing")},
        ):
            with self.subTest(info=info), self.assertRaises(ValueError):
                probe.sampling_evidence({}, info)

    def test_null_generation_penalty_falls_back_but_overrides_require_review(self):
        info = {"sampling_defaults": "model", "model_path": str(self.path)}
        (self.path / "generation_config.json").write_text(
            '{"repetition_penalty": null}'
        )
        self.assertEqual(
            probe.sampling_evidence({}, info)["effective_repetition_penalty"], 1
        )
        for changes in (
            {"preferred_sampling_params": {"min_new_tokens": 100}},
            {"override_config_file": "other.json"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                probe.sampling_evidence({}, {**info, **changes})

    def test_unreadable_generation_config_is_not_a_missing_file(self):
        info = {"sampling_defaults": "model", "model_path": str(self.path)}
        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                probe.sampling_evidence({}, info)

    def test_dry_run_makes_no_network_calls(self):
        with (
            patch.object(
                probe, "request_json", side_effect=AssertionError("network")
            ) as request,
            patch(
                "sys.argv",
                [
                    "probe",
                    "--source",
                    str(self.path),
                    "--evidence-root",
                    str(self.path / "out"),
                ],
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(probe.main(), 0)
        request.assert_not_called()
        report = next((self.path / "out").glob("*/report.json"))
        self.assertEqual(
            json.loads(report.read_text())["status"], "PLAN_PREPARED_NO_REQUESTS"
        )

    def test_interrupt_leaves_report(self):
        with (
            patch.object(probe, "collect", side_effect=KeyboardInterrupt),
            patch(
                "sys.argv",
                [
                    "probe",
                    "--run",
                    "--source",
                    str(self.path),
                    "--evidence-root",
                    str(self.path / "out"),
                ],
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(probe.main(), 1)
        report = next((self.path / "out").glob("*/report.json"))
        self.assertEqual(json.loads(report.read_text())["status"], "INTERRUPTED")


class TestScoringContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.info, _ = source_files(self.path)
        self.plan = probe.read_source(self.path)

    def test_known_and_unknown_positions_separate(self):
        for case in self.plan["cases"]:
            with self.subTest(accepted=case["accepted_drafts"]):
                request = probe.score_request(case)
                self.assertEqual(request["logprob_start_len"], case["anchor_index"])
                response = scoring_response(case, request)
                result = probe.compare_response(case, request, response)
                self.assertEqual(
                    len(result["comparisons"]), case["accepted_drafts"] + 1
                )
                self.assertEqual(result["known_top1_mismatches"], 0)
                self.assertEqual(len(result["prefill_top1_all_positions"]), 9)

    def test_no_match_recorded_without_inventing_a_tolerance(self):
        case = self.plan["cases"][1]  # first draft rejected: only bonus is known
        request = probe.score_request(case)
        response = scoring_response(case, request)
        response["meta_info"]["input_top_logprobs"][1][0][1] = 8888
        result = probe.compare_response(case, request, response)
        self.assertEqual(result["known_top1_mismatches"], 1)
        self.assertIsNone(result["comparisons"][0]["expected_rank_in_returned_top5"])
        self.assertNotIn("pass", result)

    def test_full_accept_bonus_uses_sampled_token_in_top_tie(self):
        case = self.plan["cases"][-1]
        request = probe.score_request(case)
        response = scoring_response(case, request)
        bonus = case["known_predictions"][-1]
        response["meta_info"]["output_top_logprobs"][0] = [
            [-0.1, 7777, None],
            [-0.1, bonus, None],
        ]
        result = probe.compare_response(case, request, response)
        self.assertEqual(result["known_top1_mismatches"], 0)
        self.assertEqual(result["comparisons"][-1]["prefill_top1"], bonus)
        self.assertEqual(result["prefill_top1_all_positions"][-1], bonus)

    def test_input_tie_is_not_outside_maximum_set(self):
        case = self.plan["cases"][1]
        request = probe.score_request(case)
        response = scoring_response(case, request)
        expected = case["known_predictions"][0]
        response["meta_info"]["input_top_logprobs"][1] = [
            [-0.1, 7777, None],
            [-0.1, expected, None],
        ]
        result = probe.compare_response(case, request, response)
        self.assertEqual(result["known_top1_mismatches"], 1)
        self.assertEqual(result["known_expected_outside_returned_max"], 0)

    def test_malformed_or_nonfresh_responses_rejected(self):
        case = self.plan["cases"][0]
        request = probe.score_request(case)
        original = scoring_response(case, request)
        for key, value in (
            ("id", "wrong"),
            ("cached_tokens", 1),
            ("num_retractions", 1),
            ("completion_tokens", 2),
            ("prompt_tokens", 2),
            ("finish_reason", {"type": "abort"}),
            ("output_token_logprobs", []),
        ):
            with self.subTest(key=key):
                response = copy.deepcopy(original)
                response["meta_info"][key] = value
                with self.assertRaises(ValueError):
                    probe.compare_response(case, request, response)

    def test_off_by_one_in_scored_tokens_rejected(self):
        case = self.plan["cases"][0]
        request = probe.score_request(case)
        response = scoring_response(case, request)
        response["meta_info"]["input_token_logprobs"][0][1] += 1
        with self.assertRaisesRegex(ValueError, "token positions"):
            probe.compare_response(case, request, response)

    def test_invalid_top_scores_rejected(self):
        for row in (
            [],
            [[float("nan"), 1], [-1, 2]],
            [[-1, 1], [-0.1, 2]],
            [[-0.1, 1], [-1, 1]],
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                probe.validate_top(row)

    def test_unique_cache_namespaces(self):
        a = probe.score_request(self.plan["cases"][0])
        b = probe.score_request(self.plan["cases"][0])
        self.assertNotEqual(a["rid"], b["rid"])
        self.assertNotEqual(a["cache_salt"], b["cache_salt"])

    def test_bounded_serial_collection_and_raw_evidence(self):
        calls = []

        def http(opener, url, payload=None, timeout=600):
            calls.append((url, payload))
            if url.endswith("/server_info"):
                return self.info
            case = next(
                case
                for case in self.plan["cases"]
                if case["input_ids"] == payload["input_ids"]
            )
            return scoring_response(case, payload)

        with (
            patch.object(probe, "request_json", side_effect=http),
            redirect_stdout(io.StringIO()),
        ):
            report = probe.collect(self.plan, self.path, "http://local", 1, object())
        self.assertEqual(report["status"], "COMPARISON_COLLECTED")
        self.assertEqual(sum(url.endswith("/generate") for url, _ in calls), 6)
        self.assertEqual(len(list(self.path.glob("case*.response.json"))), 6)
        self.assertTrue(
            all(case["prefill_repeats_top1_equal"] for case in report["results"])
        )

    def test_changed_config_stops_before_scoring(self):
        changed = {**self.info, "tp_size": 8}
        with patch.object(probe, "request_json", return_value=changed) as http:
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                probe.collect(self.plan, self.path, "http://local", 1, object())
        self.assertEqual(http.call_count, 1)

    def test_http_failure_not_retried(self):
        with (
            patch.object(
                probe, "request_json", side_effect=[self.info, OSError("transport")]
            ) as http,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(OSError):
                probe.collect(self.plan, self.path, "http://local", 1, object())
        self.assertEqual(http.call_count, 2)
        self.assertTrue((self.path / "case1-r1.request.json").exists())

    def test_http_error_body_preserved_and_bounded(self):
        error = urllib.error.HTTPError(
            "http://local/generate", 400, "Bad request", {}, io.BytesIO(b"x" * 70000)
        )
        with (
            patch.object(probe, "request_json", side_effect=[self.info, error]) as http,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(urllib.error.HTTPError):
                probe.collect(self.plan, self.path, "http://local", 1, object())
        self.assertEqual(http.call_count, 2)
        evidence = json.loads((self.path / "case1-r1.http-error.json").read_text())
        self.assertEqual(evidence["status_code"], 400)
        self.assertEqual(len(evidence["body"]), 65536)
        self.assertTrue(evidence["body_truncated"])

    def test_sampling_file_change_stops_before_scoring(self):
        path = self.path / "generation_config.json"
        info = {"sampling_defaults": "model", "model_path": str(self.path)}
        self.plan["sampling_evidence"] = probe.sampling_evidence({}, info)
        path.write_text("{}")
        with patch.object(probe, "request_json", return_value=self.info) as http:
            with self.assertRaisesRegex(ValueError, "sampling-default file changed"):
                probe.collect(self.plan, self.path, "http://local", 1, object())
        self.assertEqual(http.call_count, 1)


if __name__ == "__main__":
    unittest.main()
