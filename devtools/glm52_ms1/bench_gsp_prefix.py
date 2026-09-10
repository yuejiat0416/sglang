#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Temporary GSP 128k/1k prefix-cache cases using the unchanged serving client."""

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from gsp_prefix_stats import summarize_case

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
INPUT_TOKENS = 131072
OUTPUT_TOKENS = 1024


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=30 if payload is None else 3600) as response:
        return json.load(response)


def is_positive_int(value):
    return type(value) is int and value > 0


def cache_tokens(percent, page_size):
    # Integer arithmetic: 90% cannot be exact for one 131072-token request.
    return (INPUT_TOKENS * percent // (100 * page_size)) * page_size


def preflight(info, percentages, target):
    """Necessary checks only; successful requests still need length/cache audit."""
    issues = []
    if info.get("device") != "npu" or info.get("speculative_algorithm") != "DSPARK":
        issues.append("Expected the existing NPU DSPARK service")
    if info.get("model_path") != target:
        issues.append("Server model_path differs from the local tokenizer path")
    for field, expected in (("dp_size", 1), ("nnodes", 1), ("tp_size", 16)):
        if info.get(field) != expected:
            issues.append(f"This diagnostic currently expects {field}={expected}")
    # Do not apply single-pool arithmetic to distributed context sharding.
    for field in ("dcp_size", "attn_cp_size"):
        if info.get(field, 1) not in (None, 1):
            issues.append(f"Capacity calculation does not cover {field}>1")
    if info.get("allow_auto_truncate") is not False:
        issues.append("allow_auto_truncate must be explicitly false")
    if info.get("prefill_only_disable_kv_cache") is True:
        issues.append(
            "Prefill-only KV cache disabling is incompatible with these cases"
        )
    if any(percentages) and info.get("disable_radix_cache") is not False:
        issues.append("50/90% cases require disable_radix_cache=false")
    page = info.get("page_size")
    max_input = info.get("max_req_input_len")
    capacities = [info.get("max_total_num_tokens")]
    capacities += [
        state.get("memory_usage", {}).get("token_capacity")
        for state in info.get("internal_states", [])
    ]
    capacities = [v for v in capacities if is_positive_int(v)]
    capacity = min(capacities) if capacities else None
    if not is_positive_int(page):
        issues.append("Missing positive page_size")
    if not is_positive_int(max_input):
        issues.append("Missing actual max_req_input_len")
    elif INPUT_TOKENS >= max_input:
        issues.append(
            f"Input {INPUT_TOKENS} must be below max_req_input_len={max_input}"
        )
    if capacity is None:
        issues.append("Missing actual KV token capacity")
    possible_output = required_capacity = None
    if is_positive_int(page) and capacity is not None and is_positive_int(max_input):
        # Current tp_worker: max_req_input_len=max_req_len-5.
        # Current scheduler: min(max_req_len-input-1,
        # capacity-ceil_page(input)-page-1). Do not invent a DSpark reserve.
        padded_input = ((INPUT_TOKENS + page - 1) // page) * page
        possible_output = min(
            max_input + 5 - INPUT_TOKENS - 1, capacity - padded_input - page - 1
        )
        required_capacity = padded_input + page + OUTPUT_TOKENS + 1
        if possible_output < OUTPUT_TOKENS:
            issues.append(
                f"Current length/capacity bounds allow at most {possible_output} output "
                f"tokens for this input; need {OUTPUT_TOKENS}. KV capacity={capacity}; "
                f"necessary capacity>={required_capacity} before page alignment"
            )
    return {
        "status": "PREFLIGHT_READY" if not issues else "PREFLIGHT_BLOCKED",
        "ready": not issues,
        "issues": issues,
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "page_size": page,
        "actual_kv_token_capacity": capacity,
        "max_req_input_len": max_input,
        "possible_output_by_current_scheduler_bounds": possible_output,
        "necessary_capacity_before_page_alignment": required_capacity,
        "cases": [
            {
                "cache_percent_label": p,
                "expected_cached_tokens": cache_tokens(p, page),
                "expected_cache_hit_ratio": cache_tokens(p, page) / INPUT_TOKENS,
            }
            for p in percentages
        ]
        if is_positive_int(page)
        else [],
        "limits": [
            "Necessary capacity/configuration checks, not an allocation or compatibility guarantee.",
            "Server remains unchanged; a blocked result sends no generate request.",
            "Device/host/storage cache attribution comes from measured responses; no DDR/SSD claim.",
        ],
    }


@contextlib.contextmanager
def client_environment():
    values = {
        "NO_PROXY": "*",
        "no_proxy": "*",
        "SGLANG_IS_IN_CI": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_exact_gsp(tokenizer, generator, gen_prompt, seed, page):
    """Keep community GSP content; normalize lengths only in this client."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    shared = cache_tokens(90, page)
    rows = generator(
        num_groups=1,
        prompts_per_group=1,
        system_prompt_len=shared,
        question_len=INPUT_TOKENS - shared,
        output_len=OUTPUT_TOKENS,
        range_ratio=1.0,
        tokenizer=tokenizer,
        seed=seed,
        send_routing_key=True,
        num_turns=1,
        fast_prepare=False,
        ordered=True,
    )
    # send_routing_key bypasses the generator's pickle cache; routing_key is
    # deliberately not sent. The only server isolation mechanism is cache_salt.
    source_text = rows[0].prompt
    token_ids = tokenizer.encode(source_text, add_special_tokens=False)
    original_count = len(token_ids)
    extensions = 0
    while len(token_ids) < INPUT_TOKENS:
        if extensions >= 8:
            raise ValueError("GSP tokenization could not produce sufficient input")
        extra = gen_prompt(tokenizer, INPUT_TOKENS - len(token_ids) + 32)
        token_ids.extend(tokenizer.encode(extra, add_special_tokens=False))
        extensions += 1
    before_trim = len(token_ids)
    token_ids = token_ids[:INPUT_TOKENS]
    if any(type(i) is not int or i < 0 for i in token_ids):
        raise ValueError("Tokenizer did not return a flat list of token IDs")
    return token_ids, {
        "source": "community generated_shared_prefix, one group, one prompt",
        "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "nominal_shared_tokens": shared,
        "nominal_question_tokens": INPUT_TOKENS - shared,
        "range_ratio": 1.0,
        "seed": seed,
        "source_reencoded_tokens": original_count,
        "extension_calls": extensions,
        "trimmed_tokens": before_trim - INPUT_TOKENS,
        "input_tokens": len(token_ids),
        "add_special_tokens": False,
        "chat_template": False,
        "normalization": "Encode GSP text; extend if short using fresh GSP random text, trim to exact IDs",
        "notes": [
            "Random GSP content, not GSM8K or a quality dataset.",
            "Same full input IDs for all cache scenarios; no answer or task prompt is added.",
            "Special tokens follow community gen_prompt; no silent filtering.",
        ],
    }


def build_case(token_ids, percent, page, run_id, guard_token):
    count = cache_tokens(percent, page)
    salt = f"{run_id}-cache{percent}"
    common = {"cache_salt": salt, "stream": False}
    warm = None
    if count:
        if guard_token == token_ids[count]:
            raise ValueError("Warm-up guard must differ from the next measured token")
        warm = dict(
            common,
            rid=salt + "-warm",
            input_ids=token_ids[:count] + [guard_token],
            sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
        )
    measured = dict(
        common,
        rid=salt + "-measure",
        input_ids=token_ids,
        sampling_params={
            "temperature": 0,
            "max_new_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
        },
    )
    return warm, measured


@contextlib.contextmanager
def capture_native(bench, rows, responses, output):
    """Observe this client's native JSON decoder; return every object unchanged."""
    previous_json, previous_dataset = bench.orjson, bench.get_dataset

    class RecordingJSON:
        def loads(self, *args, **kwargs):
            value = previous_json.loads(*args, **kwargs)
            meta = value.get("meta_info") if isinstance(value, dict) else None
            if isinstance(meta, dict) and isinstance(meta.get("id"), str):
                responses[meta["id"]] = value
            return value

        def __getattr__(self, name):
            return getattr(previous_json, name)

    bench.orjson = RecordingJSON()
    bench.get_dataset = lambda *args, **kwargs: rows
    try:
        yield
    finally:
        bench.orjson, bench.get_dataset = previous_json, previous_dataset
        write_json(output, list(responses.values()))


def bench_arguments(args, output):
    return [
        "--backend",
        "sglang",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.target,
        "--tokenizer",
        args.target,
        "--dataset-name",
        "generated-shared-prefix",
        "--gsp-num-groups",
        "1",
        "--gsp-prompts-per-group",
        "1",
        "--gsp-output-len",
        str(OUTPUT_TOKENS),
        "--gsp-range-ratio",
        "1",
        "--num-prompts",
        "1",
        "--request-rate",
        "inf",
        "--max-concurrency",
        "1",
        "--warmup-requests",
        "0",
        "--disable-stream",
        "--temperature",
        "0",
        "--cache-report",
        "--output-details",
        "--output-file",
        str(output),
    ]


def run_bench(bench, row_type, request, args, case_dir):
    row = row_type(
        prompt=request["input_ids"],
        prompt_len=INPUT_TOKENS,
        output_len=OUTPUT_TOKENS,
        extra_request_body={k: v for k, v in request.items() if k != "input_ids"},
    )
    argv = bench_arguments(args, case_dir / "benchmark.jsonl")
    write_json(case_dir / "benchmark-arguments.json", argv)
    previous_argv = sys.argv
    captured = {}
    try:
        sys.argv = ["sglang.benchmark.serving", *argv]
        with capture_native(bench, [row], captured, case_dir / "responses.json"):
            bench.cli_main()
    finally:
        sys.argv = previous_argv
    expected_id = request["rid"]
    if set(captured) != {expected_id}:
        raise ValueError(
            f"Expected exactly one measured response; got IDs {list(captured)}"
        )
    result = json.loads((case_dir / "benchmark.jsonl").read_text().splitlines()[-1])
    if result.get("completed") != 1:
        raise ValueError("Native benchmark did not complete the measured request")
    return captured[expected_id]


def execute(args):
    percentages = [0, 50, 90] if args.cache_hit == "all" else [int(args.cache_hit)]
    run_id = (
        "gsp-prefix-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run = args.state / "evidence" / run_id
    run.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {run}", flush=True)
    summary = {
        "status": "NOT_RUN",
        "cases": [],
        "issues": [],
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "concurrency": 1,
        "measured_requests_per_case": 1,
        "quality": "NOT_SCORED",
        "performance": "NOT_QUALIFIED",
        "limits": [
            "One request per cache scenario; no sustained load qualification.",
            "Nonstream TTFT is full latency and TPOT is not measurable.",
            "Graph labels/configuration do not establish complete graph replay.",
            "Native benchmark Accept length can include service history; use each case's response A/P.",
        ],
    }
    base_url = f"http://{args.host}:{args.port}"
    try:
        before = fetch_json(base_url + "/server_info")
        write_json(run / "server_info.before.json", before)
        check = preflight(before, percentages, args.target)
        write_json(run / "preflight.json", check)
        summary["preflight"] = check
        print(check["status"], flush=True)
        for issue in check["issues"]:
            print("ISSUE:", issue, flush=True)
        if args.action == "check" or not check["ready"]:
            summary["status"] = check["status"]
            return 0 if check["ready"] else 2
        if not Path(args.target).is_dir():
            raise ValueError(
                "Local tokenizer directory is missing; do not download from the NPU server"
            )
        sys.path.insert(0, str(REPO / "python"))
        with client_environment():
            bench = importlib.import_module("sglang.benchmark.serving")
            gsp = importlib.import_module(
                "sglang.benchmark.datasets.generated_shared_prefix"
            )
            common = importlib.import_module("sglang.benchmark.datasets.common")
            if (
                Path(bench.__file__).resolve()
                != REPO / "python/sglang/benchmark/serving.py"
            ):
                raise ValueError("Benchmark import must use this checkout")
            tokenizer = bench.get_tokenizer(args.target)
            ids, protocol = make_exact_gsp(
                tokenizer,
                gsp.sample_generated_shared_prefix_requests,
                common.gen_prompt,
                args.seed,
                check["page_size"],
            )
            write_json(run / "input_ids.json", ids)
            protocol.update(
                collector_git_head=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
                ).strip(),
                collector_sha256=sha(__file__),
                stats_sha256=sha(HERE / "gsp_prefix_stats.py"),
                input_ids_sha256=sha(run / "input_ids.json"),
                benchmark_sha256=sha(bench.__file__),
                gsp_sha256=sha(gsp.__file__),
                warmup="Prefix plus a different guard token, 1 output, excluded from measurement",
                cache_isolation="New salt per case; no global flush",
                requested_cache_percentages=percentages,
            )
            write_json(run / "protocol.json", protocol)
            available = sorted(
                {i for i in tokenizer.get_vocab().values() if type(i) is int and i >= 0}
                - set(tokenizer.all_special_ids)
            )
            for percent in percentages:
                count = cache_tokens(percent, check["page_size"])
                guard = next(i for i in available if i != ids[count])
                warm, measured = build_case(
                    ids, percent, check["page_size"], run_id, guard
                )
                case_dir = run / f"cache{percent}"
                case_dir.mkdir()
                write_json(case_dir / "request.json", measured)
                if warm is not None:
                    write_json(case_dir / "warm-request.json", warm)
                    print(
                        f"WARM cache{percent}: shared prefix {count}, output 1",
                        flush=True,
                    )
                    warmed = fetch_json(base_url + "/generate", warm)
                    write_json(case_dir / "warm-response.json", warmed)
                    meta = warmed.get("meta_info", {})
                    if (
                        meta.get("id") != warm["rid"]
                        or meta.get("prompt_tokens") != count + 1
                        or meta.get("completion_tokens") != 1
                        or meta.get("cached_tokens") != 0
                        or meta.get("num_retractions") != 0
                        or (meta.get("finish_reason") or {}).get("type")
                        not in ("length", "stop")
                    ):
                        raise ValueError(
                            "Warm-up failed or was truncated; measurement not sent"
                        )
                print(
                    f"MEASURE cache{percent}: {INPUT_TOKENS}/{OUTPUT_TOKENS}, expected cached={count}",
                    flush=True,
                )
                response = run_bench(bench, common.DatasetRow, measured, args, case_dir)
                result = summarize_case(
                    response,
                    rid=measured["rid"],
                    input_tokens=INPUT_TOKENS,
                    output_tokens=OUTPUT_TOKENS,
                    expected_cached_tokens=count,
                )
                result["cache_percent_label"] = percent
                write_json(case_dir / "summary.json", result)
                summary["cases"].append(result)
                print(
                    f"cache{percent}: {result['status']}; "
                    f"acceptance={result['acceptance']['accept_rate']}; "
                    f"cached={result['actual']['cached_tokens']}/{result['actual']['prompt_tokens']}; "
                    f"output={result['actual']['completion_tokens']}",
                    flush=True,
                )
                if not result["conditions_match"]:
                    raise ValueError(
                        f"cache{percent} conditions not met; remaining cases not sent"
                    )
        after = fetch_json(base_url + "/server_info")
        write_json(run / "server_info.after.json", after)
        config_keys = (
            "model_path",
            "speculative_algorithm",
            "speculative_draft_model_path",
            "disable_cuda_graph",
            "cuda_graph_config",
            "tp_size",
            "dp_size",
            "nnodes",
            "page_size",
            "disable_radix_cache",
            "enable_hierarchical_cache",
        )
        if any(before.get(k) != after.get(k) for k in config_keys):
            raise ValueError("Server configuration changed during the cases")
        summary["status"] = "GSP_CASES_COLLECTED"
        return 0
    except (Exception, SystemExit) as exc:
        summary["status"] = "GSP_CASES_INCOMPLETE"
        summary["issues"].append(f"{type(exc).__name__}: {exc}")
        print(summary["issues"][-1], flush=True)
        return 1
    finally:
        write_json(run / "summary.json", summary)
        print(summary["status"], flush=True)
        print(
            "Send preflight.json if blocked; otherwise send summary.json. No automatic server restart.",
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--cache-hit", choices=("all", "0", "50", "90"), default="all")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8810)
    parser.add_argument("--target", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
