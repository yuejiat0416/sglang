# SPDX-License-Identifier: Apache-2.0
"""CPU tests for routing, cache contracts and native streaming response capture."""

import asyncio
import ast
import gzip
import json
import random
import sys
import tempfile
import time
import traceback
import typing
import unittest
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_prefix as bp


def server_info():
    return {
        "page_size": 128,
        "dp_size": 8,
        "max_req_input_len": 133115,
        "max_total_num_tokens": 270336,
        "allow_auto_truncate": False,
        "disable_radix_cache": False,
        "disaggregation_mode": "null",
        "internal_states": [
            {
                "memory_usage": {"token_capacity": 270336},
                "effective_max_running_requests_per_dp": 2,
            }
            for _ in range(8)
        ],
    }


def output(tokens=bp.OUTPUT_TOKENS):
    return SimpleNamespace(
        success=True,
        error="",
        output_len=tokens,
        latency=4.0,
        ttft=1.0,
        generated_text="answer",
        itl=[0.1],
    )


def response(request, cache=0, a=5, p=8):
    return {
        "text": "answer",
        "meta_info": {
            "id": request["rid"],
            "dp_rank": request["routed_dp_rank"],
            "prompt_tokens": len(request["input_ids"]),
            "completion_tokens": request["sampling_params"]["max_new_tokens"],
            "cached_tokens": cache,
            "num_retractions": 0,
            "finish_reason": {"type": "length"},
            "spec_num_correct_drafts": a,
            "spec_num_proposed_drafts": p,
            "spec_verify_ct": 1,
        },
    }


class Tokenizer:
    all_special_ids = [0]

    def get_vocab(self):
        return {str(i): i for i in range(100)}

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()] if text else []


def gen_prompt(tokenizer, length):
    return " ".join(str(random.randrange(100)) for _ in range(length))


class PrefixContractTests(unittest.TestCase):
    def test_page_floor(self):
        self.assertEqual(bp.exact_cached_tokens(0, 128), 0)
        self.assertEqual(bp.exact_cached_tokens(50, 128), 65536)
        self.assertEqual(bp.exact_cached_tokens(90, 128), 117888)
        self.assertEqual(
            bp.exact_cached_tokens(90, 128) / bp.INPUT_TOKENS, 0.8994140625
        )

    def test_invalid_cache_geometry(self):
        for percent, page in ((75, 128), (90, 0), (50, True)):
            with (
                self.subTest(percent=percent, page=page),
                self.assertRaises(ValueError),
            ):
                bp.exact_cached_tokens(percent, page)

    def test_capacity_is_per_lane_not_global_dp_multiplication(self):
        check = bp.prefix_preflight(server_info(), {"dp_size": 8}, 8, [0, 50, 90])
        self.assertTrue(check["ready"], check)
        self.assertEqual(check["per_lane_peak_requests"], 1)
        self.assertEqual(check["conservative_required_per_dp_token_capacity"], 132225)

    def test_uses_weakest_dp_capacity(self):
        info = server_info()
        info["internal_states"][7]["memory_usage"]["token_capacity"] = 12288
        self.assertFalse(bp.prefix_preflight(info, {"dp_size": 8}, 8, [50])["ready"])

    def test_every_dp_lane_requires_its_own_positive_capacity(self):
        for invalid in (None, 0, -1, True, "270336"):
            with self.subTest(capacity=invalid):
                info = server_info()
                lane_memory = info["internal_states"][7]["memory_usage"]
                if invalid is None:
                    lane_memory.pop("token_capacity")
                else:
                    lane_memory["token_capacity"] = invalid
                result = bp.prefix_preflight(info, {"dp_size": 8}, 8, [50])
                self.assertFalse(result["ready"])
                self.assertIn(
                    "Missing positive actual KV token capacity on a DP lane",
                    result["issues"],
                )

    def test_does_not_reduce_concurrency_to_fit(self):
        check = bp.prefix_preflight(server_info(), {"dp_size": 8}, 24, [90])
        self.assertFalse(check["ready"])
        self.assertEqual(check["per_lane_peak_requests"], 3)

    def test_missing_lane_or_lengths_block(self):
        for mutate in (
            lambda i: i["internal_states"].pop(),
            lambda i: i.pop("page_size"),
            lambda i: i.update(max_req_input_len=bp.INPUT_TOKENS),
            lambda i: i.update(allow_auto_truncate=True),
            lambda i: i.update(disable_radix_cache=True),
            lambda i: i.update(enable_hierarchical_cache=True),
        ):
            info = server_info()
            mutate(info)
            self.assertFalse(
                bp.prefix_preflight(info, {"dp_size": 8}, 8, [50])["ready"]
            )

    def test_zero_case_can_check_without_radix(self):
        info = server_info()
        info["disable_radix_cache"] = True
        self.assertTrue(bp.prefix_preflight(info, {"dp_size": 8}, 8, [0])["ready"])

    def test_request_routing_and_namespace(self):
        warm = bp.request_body([2], run_id="run", percent=50, lane=7, warm=True)
        first = bp.request_body([2, 3], run_id="run", percent=50, lane=7, index=0)
        second = bp.request_body([2, 4], run_id="run", percent=50, lane=7, index=1)
        foreign = bp.request_body([2, 3], run_id="run", percent=50, lane=6, index=0)
        self.assertEqual(warm["cache_salt"], first["cache_salt"])
        self.assertEqual(first["cache_salt"], second["cache_salt"])
        self.assertNotEqual(first["cache_salt"], foreign["cache_salt"])
        self.assertEqual(first["routed_dp_rank"], 7)
        self.assertNotIn("data_parallel_rank", first)
        self.assertTrue(first["stream"])
        self.assertTrue(first["sampling_params"]["ignore_eos"])

    def test_zero_each_request_new_namespace(self):
        a = bp.request_body([2], run_id="run", percent=0, lane=0, index=0)
        b = bp.request_body([2], run_id="run", percent=0, lane=0, index=1)
        self.assertNotEqual(a["cache_salt"], b["cache_salt"])

    def test_normalisation_restores_rng_and_is_reproducible(self):
        before = random.getstate()
        a, _ = bp.normalise_component(Tokenizer(), gen_prompt, 17, 42)
        self.assertEqual(random.getstate(), before)
        b, _ = bp.normalise_component(Tokenizer(), gen_prompt, 17, 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 17)

    def test_prefix_exact_and_fresh_suffix_breaks_match(self):
        with patch.object(bp, "INPUT_TOKENS", 32):
            factory = bp.GSPInputs(Tokenizer(), gen_prompt, 50, 4, 42)
            a, _ = factory.measured(0, 0)
            b, _ = factory.measured(0, 1)
            self.assertEqual(len(a), 32)
            self.assertEqual(a[:16], b[:16])
            self.assertNotEqual(a[16], b[16])
            self.assertNotEqual(a[16], factory.guard)
            self.assertNotEqual(b[16], factory.guard)
            with self.assertRaises(ValueError):
                factory.measured(0, 100)

    def test_duration_validation_and_quick_scope(self):
        base = ["load", "--config", "unused", "--mode", "dspark-eager"]
        args = bp.parser_args(base)
        self.assertEqual((args.num_prompts, args.concurrency), (64, 8))
        args = bp.parser_args(["quick", *base[1:]])
        self.assertEqual((args.num_prompts, args.concurrency), (1, 1))
        for tail in (["--duration-seconds", "nan"], ["--num-prompts", "0"]):
            with self.assertRaises(SystemExit):
                bp.parser_args(base + tail)


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.request = bp.request_body(
            [3, 4], run_id="run", percent=50, lane=2, index=0
        )

    def test_native_response(self):
        result = bp.validate_response(
            response(self.request, 1), self.request, output(), 1, True
        )
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["accepted_drafts"], 5)
        self.assertEqual(result["actual_cache_hit_ratio"], 0.5)

    def test_foreign_lane_and_cache_counts_fail(self):
        for field_name, value in (
            ("id", "warm"),
            ("dp_rank", 0),
            ("dp_rank", None),
            ("cached_tokens", 2),
            ("num_retractions", 1),
            ("completion_tokens", 100),
            ("spec_verify_ct", True),
            ("spec_num_correct_drafts", 9),
        ):
            with self.subTest(field=field_name):
                payload = response(self.request, 1)
                payload["meta_info"][field_name] = value
                self.assertFalse(
                    bp.validate_response(payload, self.request, output(), 1, True)[
                        "valid"
                    ]
                )

    def test_counter_alias_conflict(self):
        payload = response(self.request, 1)
        payload["meta_info"]["spec_accepted_drafts"] = 4
        self.assertFalse(
            bp.validate_response(payload, self.request, output(), 1, True)["valid"]
        )

    def test_target_and_warm_do_not_require_spec_counts(self):
        payload = response(self.request, 1)
        for name in tuple(payload["meta_info"]):
            if name.startswith("spec_"):
                payload["meta_info"].pop(name)
        result = bp.validate_response(payload, self.request, output(), 1, False)
        self.assertTrue(result["valid"])
        self.assertIsNone(result["accepted_drafts"])

    def test_native_failed_or_missing_timing_invalid(self):
        for changes in (
            {"success": False, "error": "HTTP 500"},
            {"ttft": 0},
            {"latency": float("nan")},
            {"output_len": 12},
        ):
            item = output()
            item.__dict__.update(changes)
            self.assertFalse(
                bp.validate_response(
                    response(self.request, 1), self.request, item, 1, True
                )["valid"]
            )

    def test_native_single_chunk_timestamp_order_is_not_a_request_failure(self):
        request = bp.request_body([1], run_id="run", percent=50, lane=2, warm=True)
        item = output(1)
        item.latency, item.ttft = 1.0, 1.00001
        result = bp.validate_response(response(request, 0), request, item, 0, False)
        self.assertTrue(result["valid"])
        self.assertIsNotNone(result["native_timing_note"])

    def test_aggregate_weighted_and_strict_threshold(self):
        rows = [
            {
                "valid": True,
                "accepted_drafts": 1,
                "proposed_drafts": 2,
                "verify_rounds": 1,
            },
            {
                "valid": True,
                "accepted_drafts": 3,
                "proposed_drafts": 6,
                "verify_rounds": 1,
            },
        ]
        result = bp.aggregate_results(rows, speculative=True, complete=True, elapsed=3)
        self.assertEqual(result["accept_rate"], 0.5)
        self.assertFalse(result["strictly_above_0_5"])
        rows[1]["accepted_drafts"] = 5
        self.assertEqual(
            bp.aggregate_results(rows, speculative=True, complete=True, elapsed=3)[
                "accept_rate"
            ],
            0.75,
        )

    def test_incomplete_never_claims_aggregate_pass(self):
        rows = [
            {
                "valid": True,
                "accepted_drafts": 8,
                "proposed_drafts": 8,
                "verify_rounds": 1,
            }
        ]
        result = bp.aggregate_results(rows, speculative=True, complete=False, elapsed=3)
        self.assertIsNone(result["accept_rate"])
        self.assertFalse(result["complete"])

    def test_decoder_capture_returns_unchanged_objects_and_restores(self):
        decoder = SimpleNamespace(loads=lambda value: value)
        bench = SimpleNamespace(orjson=decoder)
        payload = response(self.request, 1)
        with bp.capture_streams(bench) as (captured, chunks):
            self.assertIs(bench.orjson.loads(payload), payload)
            self.assertIs(bench.orjson.loads(payload), payload)
            self.assertEqual(chunks[self.request["rid"]], 2)
            self.assertIs(captured[self.request["rid"]], payload)
        self.assertIs(bench.orjson, decoder)


@dataclass
class FakeMetrics:
    completed: int


class AsyncScenarioTests(unittest.TestCase):
    def fake_runtime(self, fail_warm=False):
        bench = SimpleNamespace(
            orjson=json, RequestFuncInput=lambda **kw: SimpleNamespace(**kw)
        )
        calls = []

        async def send(request):
            body = dict(request.extra_request_body, input_ids=request.prompt)
            calls.append(body)
            warm = body["rid"].endswith("-warm")
            payload = response(body, 0 if warm else 16)
            if fail_warm and warm:
                payload["meta_info"]["dp_rank"] = None
            await asyncio.sleep(0.002)
            bench.orjson.loads(json.dumps(payload))
            return output(request.output_len)

        bench.async_request_sglang_generate = send
        bench.calculate_metrics = lambda *a, **kw: (FakeMetrics(len(a[1])), [])
        common = SimpleNamespace(
            gen_prompt=gen_prompt, DatasetRow=lambda **kw: SimpleNamespace(**kw)
        )
        cfg = {"dp_size": 8, "base_url": "http://unused", "target_model": "unused"}
        return bench, common, cfg, calls

    def test_failed_warm_does_not_send_measurements(self):
        bench, common, cfg, calls = self.fake_runtime(fail_warm=True)
        args = SimpleNamespace(
            seed=42,
            concurrency=2,
            num_prompts=2,
            duration_seconds=None,
            mode="dspark-eager",
            action="load",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(bp, "INPUT_TOKENS", 32),
            patch.object(bp, "OUTPUT_TOKENS", 4),
        ):
            result = asyncio.run(
                bp.run_scenario(bench, common, cfg, args, Path(tmp), 50, 4, Tokenizer())
            )
        self.assertFalse(result["complete"])
        self.assertEqual(result["measured_requests"], 0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c["rid"].endswith("-warm") for c in calls))

    def test_duration_extends_load_without_reusing_suffix_guard(self):
        bench, common, cfg, calls = self.fake_runtime()
        args = SimpleNamespace(
            seed=42,
            concurrency=2,
            num_prompts=2,
            duration_seconds=0.015,
            mode="dspark-eager",
            action="load",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(bp, "INPUT_TOKENS", 32),
            patch.object(bp, "OUTPUT_TOKENS", 4),
        ):
            result = asyncio.run(
                bp.run_scenario(bench, common, cfg, args, Path(tmp), 50, 4, Tokenizer())
            )
        self.assertTrue(result["complete"], result)
        self.assertGreaterEqual(result["measurement_seconds"], 0.015)
        self.assertGreaterEqual(result["measured_requests"], 2)
        measured = [c for c in calls if not c["rid"].endswith("-warm")]
        self.assertEqual(len({c["input_ids"][16] for c in measured}), len(measured))
        self.assertTrue(all(r["issue_offset_seconds"] >= 0 for r in result["results"]))

    def test_eight_dp_warm_then_measure_native_client_and_fresh_suffix(self):

        bench = SimpleNamespace(
            orjson=json, RequestFuncInput=lambda **kw: SimpleNamespace(**kw)
        )
        calls = []

        async def send(request):
            body = dict(request.extra_request_body, input_ids=request.prompt)
            calls.append(body)
            warm = body["rid"].endswith("-warm")
            payload = response(body, 0 if warm else 16)
            await asyncio.sleep(0)
            bench.orjson.loads(json.dumps(payload))
            return output(request.output_len)

        bench.async_request_sglang_generate = send
        bench.calculate_metrics = lambda *a, **kw: (FakeMetrics(len(a[1])), [])
        common = SimpleNamespace(
            gen_prompt=gen_prompt, DatasetRow=lambda **kw: SimpleNamespace(**kw)
        )
        cfg = {"dp_size": 8, "base_url": "http://unused", "target_model": "unused"}
        args = SimpleNamespace(
            seed=42,
            concurrency=8,
            num_prompts=16,
            duration_seconds=None,
            mode="dspark-eager",
            action="load",
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(bp, "INPUT_TOKENS", 32),
            patch.object(bp, "OUTPUT_TOKENS", 4),
        ):
            result = asyncio.run(
                bp.run_scenario(bench, common, cfg, args, Path(tmp), 50, 4, Tokenizer())
            )
            self.assertTrue(result["complete"], result)
            self.assertEqual(result["warmed_dp_lanes"], list(range(8)))
            self.assertEqual(result["measured_requests"], 16)
            self.assertEqual(len(calls), 24)
            self.assertTrue(all(c["rid"].endswith("-warm") for c in calls[:8]))
            for lane in range(8):
                lane_calls = [c for c in calls if c["routed_dp_rank"] == lane]
                self.assertGreaterEqual(len(lane_calls), 2)
                prefixes = [c["input_ids"][:16] for c in lane_calls]
                self.assertTrue(all(p == prefixes[0] for p in prefixes))
                self.assertEqual(
                    len({c["input_ids"][16] for c in lane_calls}), len(lane_calls)
                )
            with gzip.open(Path(tmp) / "cache50/requests.jsonl.gz", "rt") as f:
                self.assertEqual(len(f.readlines()), 24)

    def test_actual_native_function_sends_routing_and_captures_stream(self):
        import numpy as np

        source = (
            Path(__file__).resolve().parents[3] / "python/sglang/benchmark/serving.py"
        )
        tree = ast.parse(source.read_text())
        names = {
            "RequestFuncInput",
            "RequestFuncOutput",
            "async_request_sglang_generate",
            "BenchmarkMetrics",
            "calculate_metrics",
        }
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        env = dict(
            vars(typing),
            dataclass=dataclass,
            field=field,
            time=time,
            sys=sys,
            traceback=traceback,
            tqdm=object,
            orjson=json,
            get_request_headers=lambda: {},
            np=np,
            DatasetRow=SimpleNamespace,
            PreTrainedTokenizerBase=object,
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
        posted = []
        request = bp.request_body([1, 2, 3], run_id="run", percent=50, lane=7, index=0)
        final = response(request, 1)
        first = deepcopy(final)
        first["meta_info"]["completion_tokens"] = 1
        first["meta_info"]["finish_reason"] = None

        class Content:
            def __aiter__(self):
                return self.iterate()

            async def iterate(self):
                for value in (first, final):
                    await asyncio.sleep(0.001)
                    yield b"data: " + json.dumps(value).encode()
                yield b"data: [DONE]"

        class Response:
            status = 200
            content = Content()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        class Session(Response):
            def post(self, **kwargs):
                posted.append(kwargs)
                return Response()

        env["_create_bench_client_session"] = Session
        fake = SimpleNamespace(orjson=json, RequestFuncInput=env["RequestFuncInput"])
        fake.set_global_args = lambda value: setattr(fake, "args", value)
        with bp.native_settings(fake):
            env["args"] = fake.args
            with bp.capture_streams(fake) as (captured, chunks):
                env["orjson"] = fake.orjson
                result = asyncio.run(
                    env["async_request_sglang_generate"](
                        bp.native_input(
                            fake,
                            request,
                            {"base_url": "http://unused", "target_model": "local"},
                        )
                    )
                )
        self.assertTrue(result.success, result.error)
        self.assertEqual(posted[0]["json"]["input_ids"], [1, 2, 3])
        self.assertEqual(posted[0]["json"]["routed_dp_rank"], 7)
        self.assertEqual(posted[0]["json"]["cache_salt"], request["cache_salt"])
        self.assertTrue(posted[0]["json"]["stream"])
        self.assertEqual(chunks[request["rid"]], 2)
        self.assertEqual(captured[request["rid"]], final)
        self.assertGreater(result.ttft, 0)
        metrics, lengths = env["calculate_metrics"](
            [SimpleNamespace(prompt_len=3, text_prompt_len=3, vision_prompt_len=0)],
            [result],
            1.0,
            SimpleNamespace(encode=lambda *a, **k: [1, 2]),
            "sglang",
            accept_length=None,
            plot_throughput=False,
        )
        self.assertEqual(metrics.completed, 1)
        self.assertEqual(metrics.total_input, 3)
        self.assertEqual(lengths, [bp.OUTPUT_TOKENS])
        self.assertGreater(metrics.mean_ttft_ms, 0)


if __name__ == "__main__":
    unittest.main()
