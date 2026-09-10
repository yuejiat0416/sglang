"""CPU-only contracts for measured GSP cache scenarios and acceptance counts."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsp_prefix_stats import summarize_case  # noqa: E402


def response(cached=0):
    return {
        "text": "Generated response",
        "meta_info": {
            "id": "measured",
            "prompt_tokens": 131072,
            "completion_tokens": 1024,
            "cached_tokens": cached,
            "cached_tokens_details": {"device": cached, "host": 0, "storage": 0},
            "num_retractions": 0,
            "finish_reason": {"type": "length", "length": 1024},
            "spec_num_correct_drafts": 600,
            "spec_num_proposed_drafts": 1000,
            "spec_verify_ct": 125,
            "spec_accept_length": 5.8,
            "spec_accept_rate": 0.99,
        },
    }


def summarize(row, cached=0, **kwargs):
    args = dict(
        rid="measured",
        input_tokens=131072,
        output_tokens=1024,
        expected_cached_tokens=cached,
    )
    args.update(kwargs)
    return summarize_case(row, **args)


class GSPPrefixStatsTests(unittest.TestCase):
    def test_zero_half_and_page_aligned_ninety_percent_cases(self):
        for cached in (0, 65536, 117888):
            with self.subTest(cached=cached):
                report = summarize(response(cached), cached)
                self.assertEqual(report["status"], "GSP_CASE_CONDITIONS_MET")
                self.assertTrue(report["conditions_match"])
                self.assertEqual(report["actual"]["cache_hit_ratio"], cached / 131072)
                self.assertEqual(report["acceptance"]["accept_rate"], 600 / 1000)
                self.assertEqual(report["issues"], [])
                json.dumps(report, allow_nan=False)

    def test_observed_values_are_not_mutated_or_returned_by_reference(self):
        row = response()
        original = copy.deepcopy(row)
        report = summarize(row)
        report["actual"]["cached_tokens_details"]["device"] = 17
        report["actual"]["finish_reason"]["type"] = "stop"
        self.assertEqual(row, original)

    def test_identical_aliases_and_alternative_aliases(self):
        row = response()
        meta = row["meta_info"]
        meta["spec_accepted_drafts"] = 600
        meta["spec_proposed_drafts"] = 1000
        self.assertTrue(summarize(row)["conditions_match"])
        del meta["spec_num_correct_drafts"]
        del meta["spec_num_proposed_drafts"]
        self.assertTrue(summarize(row)["conditions_match"])

    def test_conflicting_aliases_rejected(self):
        for alias in ("spec_accepted_drafts", "spec_proposed_drafts"):
            with self.subTest(alias=alias):
                row = response()
                row["meta_info"][alias] = 1
                report = summarize(row)
                self.assertEqual(report["status"], "GSP_RESPONSE_INVALID")
                self.assertIsNone(report["acceptance"]["accept_rate"])
                self.assertTrue(
                    any("Conflicting" in issue for issue in report["issues"])
                )

    def test_missing_data_never_defaults_to_zero(self):
        for key in (
            "id",
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "num_retractions",
            "finish_reason",
            "spec_num_correct_drafts",
            "spec_num_proposed_drafts",
            "spec_verify_ct",
        ):
            with self.subTest(key=key):
                row = response()
                del row["meta_info"][key]
                report = summarize(row)
                self.assertFalse(report["valid_response"])
                self.assertFalse(report["conditions_match"])
                self.assertIsNone(report["acceptance"]["accept_rate"])

    def test_foreign_warmup_id_cannot_supply_measured_counts(self):
        row = response()
        row["meta_info"]["id"] = "warmup"
        report = summarize(row)
        self.assertFalse(report["valid_response"])
        self.assertIsNone(report["acceptance"]["accept_rate"])

    def test_top_level_id_if_present_must_agree(self):
        row = response()
        row["id"] = "other"
        self.assertFalse(summarize(row)["valid_response"])

    def test_conditions_mismatch_preserves_valid_counts_without_claiming_scenario(self):
        for key, value in (
            ("completion_tokens", 53),
            ("prompt_tokens", 100000),
            ("cached_tokens", 128),
            ("num_retractions", 1),
        ):
            with self.subTest(key=key):
                row = response()
                row["meta_info"][key] = value
                row["meta_info"]["finish_reason"] = {"type": "stop", "matched": 2}
                report = summarize(row)
                self.assertTrue(report["valid_response"])
                self.assertFalse(report["conditions_match"])
                self.assertEqual(report["status"], "GSP_CASE_CONDITIONS_NOT_MET")
                self.assertEqual(report["acceptance"]["accept_rate"], 0.6)
                self.assertEqual(report["acceptance"]["accepted_drafts"], 600)

    def test_short_length_finish_is_still_wrong_output_scenario(self):
        row = response()
        row["meta_info"]["completion_tokens"] = 512
        row["meta_info"]["finish_reason"] = {"type": "length", "length": 512}
        self.assertEqual(summarize(row)["status"], "GSP_CASE_CONDITIONS_NOT_MET")

    def test_abort_and_error_responses_are_invalid(self):
        row = response()
        row["meta_info"]["finish_reason"] = {"type": "abort", "message": "OOM"}
        self.assertFalse(summarize(row)["valid_response"])
        row = response()
        row["error"] = {"message": "failure"}
        self.assertFalse(summarize(row)["valid_response"])

    def test_non_integer_and_negative_fields_are_invalid(self):
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "num_retractions",
            "spec_num_correct_drafts",
            "spec_num_proposed_drafts",
            "spec_verify_ct",
        ):
            for value in (True, False, 0.0, "0", -1, None):
                with self.subTest(key=key, value=value):
                    row = response()
                    row["meta_info"][key] = value
                    self.assertFalse(summarize(row)["valid_response"])

    def test_counter_bounds(self):
        for key, value in (
            ("spec_num_correct_drafts", 1001),
            ("spec_num_proposed_drafts", 0),
            ("spec_verify_ct", 0),
        ):
            with self.subTest(key=key):
                row = response()
                row["meta_info"][key] = value
                self.assertFalse(summarize(row)["valid_response"])

    def test_zero_accepted_is_valid_not_missing(self):
        row = response()
        row["meta_info"]["spec_num_correct_drafts"] = 0
        report = summarize(row)
        self.assertTrue(report["conditions_match"])
        self.assertEqual(report["acceptance"]["accept_rate"], 0)

    def test_cached_exceeding_prompt_is_invalid(self):
        self.assertFalse(summarize(response(131073))["valid_response"])

    def test_missing_tier_details_does_not_invent_ddr_evidence(self):
        row = response()
        del row["meta_info"]["cached_tokens_details"]
        report = summarize(row)
        self.assertTrue(report["conditions_match"])
        self.assertIsNone(report["actual"]["cached_tokens_details"])

    def test_non_dict_or_missing_meta_is_invalid(self):
        for row in (None, [], "error", {}, {"meta_info": []}):
            with self.subTest(row=row):
                self.assertFalse(summarize(row)["valid_response"])

    def test_malformed_finish_reason_is_invalid_not_an_exception(self):
        for value in (None, "length", {}, {"type": []}, {"type": "unknown"}):
            with self.subTest(value=value):
                row = response()
                row["meta_info"]["finish_reason"] = value
                self.assertFalse(summarize(row)["valid_response"])

    def test_contract_accepts_other_exact_token_lengths(self):
        row = response(64)
        row["meta_info"]["prompt_tokens"] = 128
        row["meta_info"]["completion_tokens"] = 10
        row["meta_info"]["finish_reason"] = {"type": "length", "length": 10}
        report = summarize(row, 64, input_tokens=128, output_tokens=10)
        self.assertTrue(report["conditions_match"])

    def test_invalid_expected_values_raise(self):
        for kwargs in (
            {"rid": ""},
            {"input_tokens": True},
            {"input_tokens": 0},
            {"output_tokens": -1},
            {"output_tokens": 1.0},
            {"expected_cached_tokens": 131073},
            {"expected_cached_tokens": False},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                summarize(response(), **kwargs)


if __name__ == "__main__":
    unittest.main()
