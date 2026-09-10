# SPDX-License-Identifier: Apache-2.0
"""Client protocol and real aiohttp response capture; no model/NPU required."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import bench_gsm8k_modes as runner

TEST_TARGET = "/test/checkpoints/target"
TEST_DRAFT = "/test/checkpoints/draft"


@pytest.mark.parametrize(
    "missing", ["--host", "--target", "--state", "--draft", "--served-model-name"]
)
def test_cli_requires_local_connection_and_checkpoint_settings(missing):
    options = {
        "--host": "192.0.2.10",
        "--target": TEST_TARGET,
        "--state": "/test/run-state",
        "--draft": TEST_DRAFT,
        "--served-model-name": "test-model",
    }
    argv = ["run", "dspark-eager", "--print-command"]
    for option, value in options.items():
        if option != missing:
            argv.extend([option, value])
    with pytest.raises(SystemExit) as error:
        runner.main(argv)
    assert error.value.code == 2


def test_target_cli_does_not_require_an_unrelated_draft(capsys):
    assert (
        runner.main(
            [
                "run",
                "target-eager",
                "--print-command",
                "--host",
                "192.0.2.10",
                "--target",
                TEST_TARGET,
                "--state",
                "/test/run-state",
                "--served-model-name",
                "test-model",
            ]
        )
        == 0
    )
    assert TEST_TARGET in capsys.readouterr().out


def server_config(mode):
    dspark, nextn = mode.startswith("dspark"), mode.startswith("nextn")
    return {
        "speculative_draft_model_path": TEST_DRAFT if dspark else None,
        "cuda_graph_config": {
            "decode": {"backend": "disabled" if mode.endswith("eager") else "full"}
        },
        "device": "npu",
        "tp_size": 16,
        "dp_size": 1,
        "nnodes": 1,
        "model_path": TEST_TARGET,
        "served_model_name": "test-model",
        "speculative_algorithm": "DSPARK" if dspark else ("EAGLE" if nextn else None),
        "disable_cuda_graph": mode.endswith("eager"),
        "disable_decode_cuda_graph": False,
        "enable_metrics": True,
        "speculative_dspark_block_size": 8 if dspark else None,
        "speculative_num_draft_tokens": 9 if dspark else (5 if nextn else None),
        "speculative_num_steps": 4 if nextn else None,
        "speculative_eagle_topk": 1 if nextn else None,
    }


def response(rid):
    return {
        "id": rid,
        "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 10},
        "sglext": {
            "spec_tokens_details": {
                "spec_num_correct_drafts": 3,
                "spec_num_proposed_drafts": 8,
                "spec_verify_ct": 1,
            }
        },
    }


def test_real_bundle_matches_pinned_ids_and_answers_excluded():
    bundle = runner.cases()
    rows = runner.request_rows(bundle["cases"], "run", 1024)
    assert len(rows) == 10
    for r, item in zip(rows, bundle["cases"]):
        assert r["messages"] == [{"role": "user", "content": item["question"]}]
        assert "answer" not in r and "####" not in r["messages"][0]["content"]
        assert r["max_tokens"] == 1024 and r["temperature"] == 0
    assert len({r["cache_salt"] for r in rows}) == 10
    assert (
        rows[0]["cache_salt"]
        != runner.request_rows(bundle["cases"], "next", 1024)[0]["cache_salt"]
    )


@pytest.mark.parametrize("mode", runner.MODES)
def test_modes_validate_and_launch_isolated(mode):
    runner.validate_mode(server_config(mode), mode)
    with mock.patch.dict(
        runner.os.environ,
        {
            "SGLANG_NPU_GLM_DSPARK_QUAROT": "original",
            "SGLANG_DSPARK_DEBUG_DUMP": "core",
        },
        clear=True,
    ):
        env = runner.launch_environment(
            mode, Path("state"), "host", 8810, "target", "draft"
        )
    assert env["GRAPH"] == str(int(mode.endswith("graph")))
    assert env["ENABLE_METRICS"] == "1"
    assert "SGLANG_DSPARK_DEBUG_DUMP" not in env
    assert (env.get("SGLANG_NPU_GLM_DSPARK_QUAROT") == "original") == mode.startswith(
        "dspark"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("device", "cuda"),
        ("tp_size", 8),
        ("speculative_algorithm", "EAGLE"),
        ("disable_cuda_graph", False),
        ("enable_metrics", False),
        ("speculative_num_draft_tokens", 8),
    ],
)
def test_mismatched_server_rejected(key, value):
    cfg = server_config("dspark-eager")
    cfg[key] = value
    with pytest.raises(ValueError):
        runner.validate_mode(cfg, "dspark-eager")


def test_nested_observer_rejected():
    with mock.patch.dict(runner.os.environ, {"GLM52_PROPOSAL_SNAPSHOT_CONFIG": "old"}):
        with pytest.raises(ValueError, match="plain terminal"):
            runner.launch_environment(
                "dspark-graph", Path("state"), "host", 1, "t", "d"
            )


def test_bench_args_no_hidden_extra_requests():
    args = SimpleNamespace(
        host="host",
        port=8810,
        target="tokenizer",
        max_tokens=1024,
        served_model_name="test-model",
    )
    argv = runner.bench_arguments(args, Path("run"))
    for key, value in (
        ("--num-prompts", "10"),
        ("--warmup-requests", "0"),
        ("--max-concurrency", "1"),
        ("--dataset-name", "openai"),
        ("--backend", "sglang-oai-chat"),
    ):
        assert argv[argv.index(key) + 1] == value
    assert "--disable-ignore-eos" in argv and "--disable-stream" in argv
    assert "--flush-cache" not in argv and "--apply-chat-template" not in argv


def load_actual_request_function():
    # Execute exact current community client functions, avoiding unrelated Triton
    # imports on CPU. No rewritten request loop in this integration test.
    from copy import deepcopy
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional, Union
    import time
    import traceback

    names = {
        "RequestFuncInput",
        "RequestFuncOutput",
        "async_request_openai_chat_completions",
        "_combine_openai_chat_content",
    }
    source = ast.parse((runner.REPO / "python/sglang/benchmark/serving.py").read_text())
    selected = [
        node
        for node in source.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert len(selected) == len(names)
    scope = dict(
        Any=Any,
        Dict=Dict,
        List=List,
        Optional=Optional,
        Union=Union,
        dataclass=dataclass,
        field=field,
        deepcopy=deepcopy,
        time=time,
        traceback=traceback,
        tqdm=object,
    )
    scope.update(
        args=SimpleNamespace(disable_stream=True, disable_ignore_eos=True),
        get_request_headers=lambda: {},
        _create_bench_client_session=lambda: None,
    )
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), "actual_serving_subset", "exec"
        ),
        scope,
    )
    return scope


def test_actual_community_request_captures_unchanged_json(tmp_path):
    from aiohttp import web

    async def check():
        scope = load_actual_request_function()
        payloads, captured = [], []

        async def reply(request):
            payload = await request.json()
            payloads.append(payload)
            return web.json_response(response(payload["rid"]))

        app = web.Application()
        app.router.add_post("/v1/chat/completions", reply)
        srv = web.AppRunner(app)
        await srv.setup()
        site = web.TCPSite(srv, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        bench = SimpleNamespace(
            _create_bench_client_session=scope["_create_bench_client_session"]
        )
        previous = bench._create_bench_client_session
        try:
            rows = runner.request_rows(runner.cases()["cases"], "test", 1024)
            with runner.capture_responses(
                bench, tmp_path / "responses.jsonl", captured
            ):
                scope["_create_bench_client_session"] = (
                    bench._create_bench_client_session
                )
                for row in rows:
                    extra = {
                        k: v
                        for k, v in row.items()
                        if k not in ("messages", "max_tokens")
                    }
                    req = scope["RequestFuncInput"](
                        prompt=row["messages"],
                        api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
                        prompt_len=32,
                        output_len=1024,
                        model="test",
                        lora_name=None,
                        image_data=None,
                        extra_request_body=extra,
                    )
                    result = await scope["async_request_openai_chat_completions"](req)
                    assert result.success, result.error
                    assert result.generated_text == "42"
            assert bench._create_bench_client_session is previous
            assert len(payloads) == len(captured) == 10
            assert captured == [response(row["rid"]) for row in rows]
            for payload in payloads:
                assert payload["stream"] is False and payload["ignore_eos"] is False
                assert payload["max_completion_tokens"] == 1024
                assert payload["return_spec_tokens_details"] is True
            assert len((tmp_path / "responses.jsonl").read_text().splitlines()) == 10
        finally:
            await srv.cleanup()

    asyncio.run(check())


def test_client_retains_partial_failure_and_rejects_wrong_server(tmp_path):
    args = SimpleNamespace(
        state=tmp_path,
        mode="dspark-eager",
        max_tokens=1024,
        host="host",
        port=8810,
        target=TEST_TARGET,
        served_model_name="test-model",
        draft=TEST_DRAFT,
    )
    cfg = server_config("target-eager")
    with mock.patch.object(runner, "fetch_text", return_value=json.dumps(cfg)):
        assert runner.run_client(args) == 1
    summary = json.loads(next(tmp_path.glob("evidence/*/summary.json")).read_text())
    assert summary["complete"] is False and summary["aggregate"] is None
    assert any("algorithm" in s for s in summary["issues"])


def test_resolved_graph_backend_is_authoritative():
    cfg = server_config("dspark-graph")
    cfg["cuda_graph_config"]["decode"]["backend"] = "disabled"
    with pytest.raises(ValueError, match="resolved"):
        runner.validate_mode(cfg, "dspark-graph")


@pytest.mark.parametrize("mode", runner.MODES)
def test_community_cli_preparation_uses_local_model_and_http_alias(tmp_path, mode):
    """Exercise the previously uncovered CLI -> template -> tokenizer boundary.

    Execute current community functions verbatim, replacing model loading and
    HTTP/NPU work with recording dependencies. No external network or weights.
    """
    import argparse
    import random
    import sys

    target = tmp_path / "local-target"
    target.mkdir()
    args = SimpleNamespace(
        host="host",
        port=8810,
        target=str(target),
        max_tokens=1024,
        served_model_name="test-model",
    )
    rows = runner.request_rows(runner.cases()["cases"], mode, 1024)
    (tmp_path / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    loaded, submitted = [], []
    tokenizer = SimpleNamespace(init_kwargs={"chat_template": "local template"})

    def from_pretrained(path, **kwargs):
        loaded.append(path)
        if not Path(path).is_dir():
            raise AssertionError(f"Attempted remote model lookup: {path}")
        return tokenizer

    def dataset(parsed, actual_tokenizer, model_id):
        assert actual_tokenizer is tokenizer
        assert parsed.model == str(target)
        assert parsed.tokenizer == str(target)
        return [
            json.loads(line)
            for line in Path(parsed.dataset_path).read_text().splitlines()
        ]

    async def benchmark(**kwargs):
        submitted.append(kwargs)
        return {"completed": len(kwargs["input_requests"])}

    scope = dict(
        argparse=argparse,
        ArgumentParser=argparse.ArgumentParser,
        ASYNC_REQUEST_FUNCS={"sglang-oai-chat": None},
        _finite_positive_float=float,
        _EMBEDDING_BACKENDS=set(),
        _BACKEND_API_PATHS={"sglang-oai-chat": "/v1/chat/completions"},
        _DEFAULT_SGLANG_FLUSH_CACHE_TIMEOUT=60,
        set_ulimit=lambda: None,
        random=random,
        np=SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)),
        json=json,
        sys=sys,
        asyncio=asyncio,
        os=runner.os,
        resolve_base_url=lambda base, host, port: f"http://{host}:{port}",
        wait_for_endpoint=lambda url, timeout: True,
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained),
        get_dataset=dataset,
        benchmark=benchmark,
    )
    source = ast.parse((runner.REPO / "python/sglang/benchmark/serving.py").read_text())
    names = {
        "LoRAPathAction",
        "cli_main",
        "_validate_parsed_gsp_args",
        "run_benchmark",
        "check_chat_template",
    }
    nodes = [
        n
        for n in source.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    utils = ast.parse((runner.REPO / "python/sglang/benchmark/utils.py").read_text())
    nodes += [
        n
        for n in utils.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_tokenizer"
    ]
    assert len(nodes) == 6
    # Postponed annotations avoid importing unrelated transformers types.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(module), "actual_benchmark_preparation", "exec"
        ),
        scope,
    )
    with mock.patch.object(
        sys, "argv", ["serving", *runner.bench_arguments(args, tmp_path)]
    ):
        scope["cli_main"]()
    assert loaded == [str(target), str(target)]
    assert len(submitted) == 1
    assert submitted[0]["model_id"] == "test-model"
    assert submitted[0]["input_requests"] == rows
    assert submitted[0]["warmup_requests"] == 0
