# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests including the actual community CLI and request function."""

import ast
import asyncio
import json
import sys
from argparse import ArgumentParser
from types import SimpleNamespace
from unittest import mock

import pytest

import bench_gsp_prefix as runner

TEST_TARGET = "/test/checkpoints/target"


@pytest.mark.parametrize("missing", ["--host", "--target", "--state"])
def test_cli_requires_local_connection_and_checkpoint_settings(missing):
    options = {
        "--host": "192.0.2.10",
        "--target": TEST_TARGET,
        "--state": "/test/run-state",
    }
    argv = ["check"]
    for option, value in options.items():
        if option != missing:
            argv.extend([option, value])
    with (
        mock.patch.object(runner, "execute") as execute,
        pytest.raises(SystemExit) as error,
    ):
        runner.main(argv)
    assert error.value.code == 2
    execute.assert_not_called()


def service():
    return dict(
        device="npu",
        speculative_algorithm="DSPARK",
        tp_size=16,
        dp_size=1,
        nnodes=1,
        model_path=TEST_TARGET,
        page_size=128,
        max_total_num_tokens=132352,
        max_req_input_len=132346,
        allow_auto_truncate=False,
        disable_radix_cache=False,
        internal_states=[{"memory_usage": {"token_capacity": 132352}}],
    )


def arguments(tmp_path, action="run"):
    return SimpleNamespace(
        host="127.0.0.1",
        port=8810,
        target=TEST_TARGET,
        state=tmp_path,
        cache_hit="all",
        seed=42,
        action=action,
    )


def test_minimum_capacity_page_alignment_and_ratios():
    cfg = service()
    result = runner.preflight(cfg, [0, 50, 90], TEST_TARGET)
    assert result["ready"]
    assert result["necessary_capacity_before_page_alignment"] == 132225
    assert [c["expected_cached_tokens"] for c in result["cases"]] == [0, 65536, 117888]
    assert result["cases"][2]["expected_cache_hit_ratio"] == 0.8994140625
    cfg["internal_states"][0]["memory_usage"]["token_capacity"] = 132224
    result = runner.preflight(cfg, [0], TEST_TARGET)
    assert not result["ready"]
    assert result["possible_output_by_current_scheduler_bounds"] == 1023


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_total_num_tokens", 12288),
        ("max_req_input_len", 131072),
        ("page_size", None),
        ("allow_auto_truncate", True),
        ("disable_radix_cache", True),
        ("model_path", "other"),
        ("dcp_size", 2),
        ("speculative_algorithm", "EAGLE"),
    ],
)
def test_preflight_blocks_known_invalid_case(field, value):
    cfg = service()
    cfg[field] = value
    assert not runner.preflight(cfg, [0, 50, 90], TEST_TARGET)["ready"]


def test_prefill_budget_not_mistaken_for_input_limit():
    cfg = service()
    cfg.update(max_prefill_tokens=69632, chunked_prefill_size=-1)
    assert runner.preflight(cfg, [0], TEST_TARGET)["ready"]
    cfg["disable_radix_cache"] = True
    assert runner.preflight(cfg, [0], TEST_TARGET)["ready"]


def test_blocked_run_and_check_send_only_read_requests(tmp_path):
    cfg = service()
    cfg["max_total_num_tokens"] = 12288
    with mock.patch.object(runner, "fetch_json", return_value=cfg) as fetch:
        assert runner.execute(arguments(tmp_path)) == 2
    assert len(fetch.call_args_list) == 1
    assert fetch.call_args.args[0].endswith("/server_info")
    with mock.patch.object(runner, "fetch_json", return_value=service()) as fetch:
        assert runner.execute(arguments(tmp_path, "check")) == 0
    assert len(fetch.call_args_list) == 1
    assert len(list(tmp_path.glob("evidence/*/preflight.json"))) == 2


def test_warm_and_measure_exact_prefix_and_isolation():
    ids = [10 + (i % 100) for i in range(runner.INPUT_TOKENS)]
    salts = []
    for percent in (0, 50, 90):
        count = runner.cache_tokens(percent, 128)
        warm, measured = runner.build_case(ids, percent, 128, "run", 3)
        salts.append(measured["cache_salt"])
        assert measured["input_ids"] == ids
        assert measured["sampling_params"] == {
            "temperature": 0,
            "max_new_tokens": 1024,
            "ignore_eos": True,
        }
        if count:
            assert warm["input_ids"][:-1] == ids[:count]
            assert warm["input_ids"][-1] != ids[count]
            assert len(warm["input_ids"]) == count + 1
            assert warm["cache_salt"] == measured["cache_salt"]
            assert warm["rid"] != measured["rid"]
            assert warm["sampling_params"]["max_new_tokens"] == 1
        else:
            assert warm is None
    assert len(set(salts)) == 3
    with pytest.raises(ValueError, match="guard"):
        runner.build_case(ids, 50, 128, "run", ids[65536])


def test_gsp_normalization_trim_and_extend():
    calls = []

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return list(range(text))

    def generator(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(prompt="source")]

    # Tokenizer text stub keeps test memory bounded while exercising ID lengths.
    tok = Tokenizer()
    tok.encode = lambda text, **kwargs: [7] * (131090 if text == "source" else 64)
    ids, info = runner.make_exact_gsp(tok, generator, lambda *_: "extra", 42, 128)
    assert len(ids) == 131072 and info["trimmed_tokens"] == 18
    assert calls[0]["range_ratio"] == 1.0 and calls[0]["send_routing_key"] is True
    tok.encode = lambda text, **kwargs: [7] * (131060 if text == "source" else 64)
    ids, info = runner.make_exact_gsp(tok, generator, lambda *_: "extra", 42, 128)
    assert len(ids) == 131072 and info["extension_calls"] == 1


def test_environment_restores_even_on_failure():
    with mock.patch.dict(
        runner.os.environ, {"SGLANG_IS_IN_CI": "true", "NO_PROXY": "old"}
    ):
        with pytest.raises(RuntimeError):
            with runner.client_environment():
                assert runner.os.environ["SGLANG_IS_IN_CI"] == "false"
                assert runner.os.environ["HF_HUB_OFFLINE"] == "1"
                raise RuntimeError()
        assert runner.os.environ["SGLANG_IS_IN_CI"] == "true"
        assert runner.os.environ["NO_PROXY"] == "old"


def test_actual_cli_accepts_dataset_and_has_no_hidden_warmup(tmp_path):
    source = ast.parse((runner.REPO / "python/sglang/benchmark/serving.py").read_text())
    # Run the actual parser, stopping at the validation/runner boundary.
    selected = [
        node
        for node in source.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name
        in {
            "cli_main",
            "LoRAPathAction",
            "_finite_positive_float",
            "_validate_parsed_gsp_args",
        }
    ]
    seen = []
    scope = dict(
        ArgumentParser=ArgumentParser,
        argparse=__import__("argparse"),
        ASYNC_REQUEST_FUNCS={"sglang": object()},
        _DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT=60,
        math=__import__("math"),
        run_benchmark=lambda args: seen.append(args),
    )
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "actual_cli", "exec"), scope
    )
    with mock.patch.object(
        sys,
        "argv",
        [
            "serving",
            *runner.bench_arguments(arguments(tmp_path), tmp_path / "bench.jsonl"),
        ],
    ):
        scope["cli_main"]()
    assert len(seen) == 1
    args = seen[0]
    assert args.dataset_name == "generated-shared-prefix"
    assert args.model == args.tokenizer == TEST_TARGET
    assert args.max_concurrency == 1 and args.warmup_requests == 0
    assert args.disable_ignore_eos is False and args.disable_stream is True
    assert args.flush_cache is False and args.cache_report is True


def test_real_native_request_and_capture(tmp_path):
    import time
    import traceback
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional, Union

    import aiohttp
    from aiohttp import web

    source = ast.parse((runner.REPO / "python/sglang/benchmark/serving.py").read_text())
    names = {"RequestFuncInput", "RequestFuncOutput", "async_request_sglang_generate"}
    selected = [
        node
        for node in source.body
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    scope = dict(
        Any=Any,
        Dict=Dict,
        List=List,
        Optional=Optional,
        Union=Union,
        dataclass=dataclass,
        field=field,
        time=time,
        traceback=traceback,
        tqdm=object,
        sys=sys,
        orjson=json,
        get_request_headers=lambda: {},
        _create_bench_client_session=lambda: aiohttp.ClientSession(),
    )
    scope["args"] = SimpleNamespace(
        temperature=0,
        disable_ignore_eos=False,
        top_p=1.0,
        disable_stream=True,
        return_logprob=False,
        return_routed_experts=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        cache_report=True,
    )
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "actual_native", "exec"),
        scope,
    )

    async def check():
        bodies = []
        response = {
            "text": "result",
            "meta_info": {
                "id": "measured",
                "prompt_tokens": 131072,
                "completion_tokens": 1024,
                "cached_tokens": 65536,
                "num_retractions": 0,
                "finish_reason": {"type": "length", "length": 1024},
                "spec_num_correct_drafts": 800,
                "spec_num_proposed_drafts": 1600,
                "spec_verify_ct": 200,
            },
        }

        async def reply(request):
            bodies.append(await request.json())
            return web.json_response(response)

        app = web.Application(client_max_size=4 * 1024**2)
        app.router.add_post("/generate", reply)
        server = web.AppRunner(app)
        await server.setup()
        site = web.TCPSite(server, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        bench = SimpleNamespace(orjson=json, get_dataset=lambda *_: None)
        captured = {}
        try:
            with runner.capture_native(
                bench, ["prepared"], captured, tmp_path / "responses.json"
            ):
                scope["orjson"] = bench.orjson
                req = scope["RequestFuncInput"](
                    prompt=[1] * 131072,
                    api_url=f"http://127.0.0.1:{port}/generate",
                    prompt_len=131072,
                    output_len=1024,
                    model="local",
                    lora_name=None,
                    image_data=None,
                    extra_request_body={"rid": "measured", "cache_salt": "test-salt"},
                )
                output = await scope["async_request_sglang_generate"](req)
                assert output.success, output.error
                assert output.cached_tokens == 65536 and output.output_len == 1024
            assert bench.orjson is json and captured == {"measured": response}
            assert len(bodies) == 1 and len(bodies[0]["input_ids"]) == 131072
            assert bodies[0]["cache_salt"] == "test-salt"
            assert bodies[0]["sampling_params"]["ignore_eos"] is True
            assert "text" not in bodies[0]
        finally:
            await server.cleanup()

    asyncio.run(check())
