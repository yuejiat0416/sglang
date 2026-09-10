#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Temporary DP-aware GSP 131072/1024 cases using the native serving client.

Warm each used DP lane, then measure fresh suffixes on that exact lane. Native
stream timing and response counters are retained separately from warm-up work.
"""

import argparse
import asyncio
import contextlib
import gzip
import hashlib
import importlib
import json
import math
import random
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

INPUT_TOKENS = 131072
OUTPUT_TOKENS = 1024
_RANDOM_LOCK = threading.Lock()


def exact_cached_tokens(percent, page_size):
    if percent not in (0, 50, 90) or type(page_size) is not int or page_size < 1:
        raise ValueError("Expected cache 0/50/90 and a positive page size")
    return INPUT_TOKENS * percent // (100 * page_size) * page_size


def positive_int(value):
    return type(value) is int and value > 0


def prefix_preflight(info, config, concurrency, percentages):
    """Conservative per-lane capacity check; this never changes server settings."""
    issues = []
    page = info.get("page_size")
    dp = info.get("dp_size")
    max_input = info.get("max_req_input_len")
    states = info.get("internal_states") or []
    lane_capacities = [s.get("memory_usage", {}).get("token_capacity") for s in states]
    capacities = [info.get("max_total_num_tokens"), *lane_capacities]
    capacities = [n for n in capacities if positive_int(n)]
    capacity = min(capacities) if capacities else None
    if not positive_int(page):
        issues.append("Missing positive runtime page_size")
    if not positive_int(dp) or dp != config["dp_size"]:
        issues.append("Runtime DP size is missing or differs from the configured lanes")
    if len(states) != config["dp_size"]:
        issues.append("Expected runtime internal_states for every configured DP lane")
    if not lane_capacities or any(not positive_int(v) for v in lane_capacities):
        issues.append("Missing positive actual KV token capacity on a DP lane")
    if not positive_int(max_input) or INPUT_TOKENS >= max_input:
        issues.append("Runtime max_req_input_len does not admit 131072 input tokens")
    if capacity is None:
        issues.append("Missing actual KV token capacity")
    if info.get("allow_auto_truncate") is not False:
        issues.append("allow_auto_truncate must be false")
    if info.get("disaggregation_mode") not in (None, "null"):
        issues.append("This tool requires the colocated HTTP endpoint")
    if any(percentages) and info.get("disable_radix_cache") is not False:
        issues.append("Warm-prefix cases require radix cache")
    if info.get("prefill_only_disable_kv_cache") is True:
        issues.append("Prefill-only KV disabling is not covered")
    for field in ("attn_cp_size", "attn_dcp_size", "dcp_size"):
        if info.get(field, 1) not in (None, 1):
            issues.append(f"Capacity calculations do not cover {field}>1")
    for field in ("enable_hierarchical_cache", "enable_lmcache", "enable_lmcache_v2"):
        if info.get(field) is True:
            issues.append(
                f"{field} changes cache placement; use a separately defined tier test"
            )
    per_lane = math.ceil(concurrency / config["dp_size"])
    required = possible_output = None
    if positive_int(page):
        padded = math.ceil(INPUT_TOKENS / page) * page
        required = per_lane * (padded + OUTPUT_TOKENS + page + 1)
        if capacity is not None and capacity < required:
            issues.append(
                f"Per-DP KV capacity {capacity} < conservative need {required}"
            )
        if positive_int(max_input) and capacity is not None:
            possible_output = min(
                max_input + 5 - INPUT_TOKENS - 1, capacity - padded - page - 1
            )
            if possible_output < OUTPUT_TOKENS:
                issues.append("Scheduler bounds would shorten the 1024-token output")
    limits = [s.get("effective_max_running_requests_per_dp") for s in states]
    if not limits or any(not positive_int(v) or v < per_lane for v in limits):
        issues.append("Missing or insufficient effective request slots on a DP lane")
    return {
        "status": "PREFIX_PREFLIGHT_READY"
        if not issues
        else "PREFIX_PREFLIGHT_BLOCKED",
        "ready": not issues,
        "issues": issues,
        "page_size": page,
        "dp_size": dp,
        "per_lane_peak_requests": per_lane,
        "actual_minimum_per_dp_token_capacity": capacity,
        "conservative_required_per_dp_token_capacity": required,
        "possible_output_by_scheduler_bounds": possible_output,
        "limits": [
            "Conservative full-input reservation even for hot prefixes; no silent concurrency reduction.",
            "Passing is not an allocation, graph, model or deployment compatibility guarantee.",
        ],
    }


def normalise_component(tokenizer, gen_prompt, length, seed):
    """Use the community random-text generator, then pin actual token length."""
    if length == 0:
        return [], {"generation_calls": 0, "trimmed_tokens": 0}
    ids = []
    calls = 0
    with _RANDOM_LOCK:
        state = random.getstate()
        random.seed(seed)
        try:
            while len(ids) < length:
                if calls == 12:
                    raise ValueError(
                        "GSP component failed to reach its required token count"
                    )
                text = gen_prompt(tokenizer, length - len(ids) + 32)
                ids.extend(tokenizer.encode(text, add_special_tokens=False))
                calls += 1
        finally:
            random.setstate(state)
    if any(type(v) is not int or v < 0 for v in ids):
        raise ValueError("Tokenizer returned invalid token IDs")
    return ids[:length], {
        "generation_calls": calls,
        "trimmed_tokens": len(ids) - length,
    }


class GSPInputs:
    def __init__(self, tokenizer, gen_prompt, percent, page, seed):
        self.tokenizer = tokenizer
        self.gen_prompt = gen_prompt
        self.percent = percent
        self.shared = exact_cached_tokens(percent, page)
        self.seed = seed
        self.prefixes = {}
        self.guard_ids = sorted(
            {v for v in tokenizer.get_vocab().values() if type(v) is int and v >= 0}
            - set(getattr(tokenizer, "all_special_ids", []))
        )
        if len(self.guard_ids) < 2:
            raise ValueError("Tokenizer needs at least two non-special guard tokens")
        self.guard = self.guard_ids[0]

    def prefix(self, lane):
        if lane not in self.prefixes:
            ids, _ = normalise_component(
                self.tokenizer,
                self.gen_prompt,
                self.shared,
                self.seed + self.percent * 1000003 + lane,
            )
            self.prefixes[lane] = ids
        return self.prefixes[lane]

    def measured(self, lane, index):
        # A different first suffix token prevents a prior complete request from
        # making a later request hotter than the intended shared prefix.
        if index + 1 >= len(self.guard_ids):
            raise ValueError(
                "Fresh suffix guard IDs exhausted; do not wrap and reuse prompts"
            )
        prefix = self.prefix(lane)
        suffix, details = normalise_component(
            self.tokenizer,
            self.gen_prompt,
            INPUT_TOKENS - self.shared,
            self.seed + self.percent * 1000003 + 10000 + index,
        )
        suffix[0] = self.guard_ids[index + 1]
        ids = prefix + suffix
        assert len(ids) == INPUT_TOKENS
        return ids, details


def request_body(ids, *, run_id, percent, lane, index=None, warm=False):
    # Cold requests use independent namespaces; warm cases share only per lane.
    suffix = f"-request{index}" if percent == 0 else ""
    salt = f"{run_id}-cache{percent}-dp{lane}{suffix}"
    rid = salt + ("-warm" if warm else f"-measure{index}")
    return {
        "input_ids": ids,
        "rid": rid,
        "cache_salt": salt,
        "routed_dp_rank": lane,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1 if warm else OUTPUT_TOKENS,
            "ignore_eos": True,
        },
    }


def validate_response(response, request, output, expected_cached, speculative):
    errors = []
    meta = response.get("meta_info", {}) if isinstance(response, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    if not output.success:
        errors.append(f"Native client failure: {output.error}")
    if meta.get("id") != request["rid"]:
        errors.append("Missing or foreign response ID")
    if meta.get("dp_rank") != request["routed_dp_rank"]:
        errors.append("Actual DP lane differs from the requested lane or is absent")
    expected_output = request["sampling_params"]["max_new_tokens"]
    if output.success and (
        output.output_len != expected_output
        or not math.isfinite(output.latency)
        or output.latency <= 0
        or not math.isfinite(output.ttft)
        or output.ttft <= 0
    ):
        errors.append("Native stream length/timing is missing, inconsistent or invalid")
    for field, expected in (
        ("prompt_tokens", len(request["input_ids"])),
        ("completion_tokens", expected_output),
        ("cached_tokens", expected_cached),
        ("num_retractions", 0),
    ):
        value = meta.get(field)
        if type(value) is not int or value != expected:
            errors.append(f"{field}: expected {expected}, observed {value!r}")
    reason = meta.get("finish_reason")
    if not isinstance(reason, dict) or reason.get("type") not in ("length", "stop"):
        errors.append("Missing or unsuccessful finish reason")
    counts = {"accepted_drafts": None, "proposed_drafts": None, "verify_rounds": None}
    aliases = {
        "accepted_drafts": ("spec_num_correct_drafts", "spec_accepted_drafts"),
        "proposed_drafts": ("spec_num_proposed_drafts", "spec_proposed_drafts"),
        "verify_rounds": ("spec_verify_ct",),
    }
    if speculative:
        for key, names in aliases.items():
            values = [meta[n] for n in names if n in meta]
            if (
                not values
                or any(type(v) is not int or v < 0 for v in values)
                or len(set(values)) != 1
            ):
                errors.append(f"Missing, invalid or conflicting {key}")
            else:
                counts[key] = values[0]
        a, p, n = counts.values()
        if (p is not None and p == 0) or (n is not None and n == 0):
            errors.append("No speculative proposals or verify rounds")
        if a is not None and p is not None and a > p:
            errors.append("Accepted drafts exceeds proposed drafts")
    elif not request["rid"].endswith("-warm"):
        for names in aliases.values():
            if any(name in meta and meta[name] not in (None, 0) for name in names):
                errors.append(
                    "Target-only response unexpectedly includes speculative work"
                )
                break
    prompt = meta.get("prompt_tokens")
    cached = meta.get("cached_tokens")
    return {
        "rid": request["rid"],
        "dp_rank": meta.get("dp_rank"),
        "valid": not errors,
        "issues": errors,
        **counts,
        "prompt_tokens": prompt,
        "completion_tokens": meta.get("completion_tokens"),
        "cached_tokens": cached,
        "expected_cached_tokens": expected_cached,
        "actual_cache_hit_ratio": cached / prompt
        if type(cached) is int and positive_int(prompt)
        else None,
        "cached_tokens_details": meta.get("cached_tokens_details"),
        "num_retractions": meta.get("num_retractions"),
        "finish_reason": reason,
        "native_success": output.success,
        "native_error": output.error,
        "latency_seconds": output.latency,
        "ttft_seconds": output.ttft,
        "tpot_seconds": (
            (output.latency - output.ttft) / (output.output_len - 1)
            if output.success
            and output.output_len > 1
            and output.latency >= output.ttft
            else None
        ),
        "native_timing_note": (
            "Native parser timestamps latency before parsing and TTFT afterwards; a single response chunk can invert these by parsing time"
            if output.latency < output.ttft
            else None
        ),
    }


@contextlib.contextmanager
def capture_streams(bench):
    original = bench.orjson
    latest, chunks = {}, {}

    class Observer:
        def loads(self, *args, **kwargs):
            value = original.loads(*args, **kwargs)
            meta = value.get("meta_info") if isinstance(value, dict) else None
            if isinstance(meta, dict) and isinstance(meta.get("id"), str):
                rid = meta["id"]
                latest[rid] = value
                chunks[rid] = chunks.get(rid, 0) + 1
            return value

        def __getattr__(self, key):
            return getattr(original, key)

    bench.orjson = Observer()
    try:
        yield latest, chunks
    finally:
        bench.orjson = original


def native_input(bench, request, config):
    return bench.RequestFuncInput(
        prompt=request["input_ids"],
        api_url=config["base_url"] + "/generate",
        prompt_len=len(request["input_ids"]),
        output_len=request["sampling_params"]["max_new_tokens"],
        model=config["target_model"],
        lora_name=None,
        image_data=None,
        extra_request_body={k: v for k, v in request.items() if k != "input_ids"},
    )


@contextlib.contextmanager
def native_settings(bench):
    missing = object()
    previous = getattr(bench, "args", missing)
    bench.set_global_args(
        SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            disable_ignore_eos=False,
            disable_stream=False,
            return_logprob=False,
            return_routed_experts=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            cache_report=True,
            header=None,
        )
    )
    try:
        yield
    finally:
        if previous is missing:
            del bench.args
        else:
            bench.args = previous


def aggregate_results(records, *, speculative, complete, elapsed):
    valid = complete and bool(records) and all(r["valid"] for r in records)
    a = p = n = None
    if speculative and valid:
        a = sum(r["accepted_drafts"] for r in records)
        p = sum(r["proposed_drafts"] for r in records)
        n = sum(r["verify_rounds"] for r in records)
    rate = a / p if p else None
    return {
        "complete": valid,
        "measured_requests": len(records),
        "invalid_requests": sum(not r["valid"] for r in records),
        "accepted_drafts": a,
        "proposed_drafts": p,
        "verify_rounds": n,
        "accept_rate": rate,
        "strictly_above_0_5": rate > 0.5 if rate is not None else None,
        "acceptance_applicability": "speculative" if speculative else "target-only N/A",
        "measurement_seconds": elapsed,
        "qualification": "NOT_A_FORMAL_ACCEPTANCE_PASS",
    }


async def run_scenario(bench, common, config, args, run, percent, page, tokenizer):
    from client_common import write_json

    out_dir = run / f"cache{percent}"
    out_dir.mkdir()
    factory = GSPInputs(tokenizer, common.gen_prompt, percent, page, args.seed)
    concurrency = (
        min(args.concurrency, args.num_prompts)
        if args.duration_seconds is None
        else args.concurrency
    )
    lanes = sorted({worker % config["dp_size"] for worker in range(concurrency)})
    speculative = args.mode.startswith("dspark") or args.mode.startswith("nextn")
    records, outputs, input_rows, errors = [], [], [], []
    timer = None
    generation_seconds = 0.0
    preparation_seconds = 0.0
    next_index = 0
    stopped = False
    prepared = {}

    with (
        capture_streams(bench) as (captured, chunks),
        gzip.open(out_dir / "requests.jsonl.gz", "wt") as requests_file,
        (out_dir / "issued.jsonl").open("w") as issued_file,
    ):

        async def warm_lane(lane):
            if not factory.shared:
                return
            prefix = await asyncio.to_thread(factory.prefix, lane)
            request = request_body(
                prefix + [factory.guard],
                run_id=run.name,
                percent=percent,
                lane=lane,
                warm=True,
            )
            requests_file.write(json.dumps({"phase": "warm", "body": request}) + "\n")
            output = await bench.async_request_sglang_generate(
                native_input(bench, request, config)
            )
            response = captured.pop(request["rid"], None)
            result = validate_response(response, request, output, 0, speculative=False)
            write_json(
                out_dir / f"warm-dp{lane}.json",
                {"result": result, "response": response},
            )
            if not result["valid"]:
                raise ValueError(f"Warm DP{lane} failed: {result['issues']}")

        try:
            if args.duration_seconds is None:
                prepare_start = time.perf_counter()
                # Fixed N is generated and archived before any measured clock.
                # Assign stable worker queues so each lane respects its checked
                # peak concurrency even if other lanes complete more quickly.
                for index in range(args.num_prompts):
                    lane = (index % concurrency) % config["dp_size"]
                    ids, generation = await asyncio.to_thread(
                        factory.measured, lane, index
                    )
                    request = request_body(
                        ids, run_id=run.name, percent=percent, lane=lane, index=index
                    )
                    fingerprint = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                    prepared[index] = (request, fingerprint)
                    requests_file.write(
                        json.dumps(
                            {
                                "phase": "measure",
                                "prepared_before_measurement": True,
                                "index": index,
                                "generation": generation,
                                "body": request,
                            }
                        )
                        + "\n"
                    )
                requests_file.flush()
                preparation_seconds = time.perf_counter() - prepare_start
            # Requests on different lanes can prefill together; every result is
            # checked before any measured request is sent.
            warm_results = await asyncio.gather(
                *(warm_lane(lane) for lane in lanes), return_exceptions=True
            )
            warm_errors = [str(r) for r in warm_results if isinstance(r, BaseException)]
            if warm_errors:
                raise ValueError("; ".join(warm_errors))
            timer = time.perf_counter()

            async def worker(worker_index):
                nonlocal next_index, stopped, generation_seconds
                lane = worker_index % config["dp_size"]
                own_index = worker_index
                while not stopped:
                    elapsed = time.perf_counter() - timer
                    if args.duration_seconds is None:
                        if own_index >= args.num_prompts:
                            return
                        index = own_index
                        own_index += concurrency
                    else:
                        if (
                            next_index >= args.num_prompts
                            and elapsed >= args.duration_seconds
                        ):
                            return
                        index = next_index
                        next_index += 1
                    try:
                        if args.duration_seconds is None:
                            request, fingerprint = prepared.pop(index)
                        else:
                            before = time.perf_counter()
                            ids, generation = await asyncio.to_thread(
                                factory.measured, lane, index
                            )
                            generation_seconds += time.perf_counter() - before
                            request = request_body(
                                ids,
                                run_id=run.name,
                                percent=percent,
                                lane=lane,
                                index=index,
                            )
                            fingerprint = hashlib.sha256(
                                json.dumps(ids).encode()
                            ).hexdigest()
                            requests_file.write(
                                json.dumps(
                                    {
                                        "phase": "measure",
                                        "index": index,
                                        "generation": generation,
                                        "body": request,
                                    }
                                )
                                + "\n"
                            )
                            requests_file.flush()
                        issue_offset = time.perf_counter() - timer
                        issued_file.write(
                            json.dumps(
                                {
                                    "rid": request["rid"],
                                    "index": index,
                                    "dp_rank": lane,
                                    "input_sha256": fingerprint,
                                    "issue_offset_seconds": issue_offset,
                                }
                            )
                            + "\n"
                        )
                        issued_file.flush()
                        output = await bench.async_request_sglang_generate(
                            native_input(bench, request, config)
                        )
                        response = captured.pop(request["rid"], None)
                        record = validate_response(
                            response, request, output, factory.shared, speculative
                        )
                        record.update(
                            index=index,
                            stream_chunks=chunks.pop(request["rid"], 0),
                            input_sha256=fingerprint,
                            issue_offset_seconds=issue_offset,
                        )
                        write_json(out_dir / f"response-{index:06d}.json", response)
                        write_json(out_dir / f"result-{index:06d}.json", record)
                        records.append(record)
                        outputs.append(output)
                        input_rows.append(
                            common.DatasetRow(
                                prompt=None,
                                prompt_len=INPUT_TOKENS,
                                output_len=OUTPUT_TOKENS,
                            )
                        )
                        print(
                            f"cache{percent} request{index} DP{lane}: valid={record['valid']} cached={record['cached_tokens']} A/P={record['accepted_drafts']}/{record['proposed_drafts']}",
                            flush=True,
                        )
                        if not record["valid"]:
                            stopped = True
                            errors.extend(record["issues"])
                    except Exception as exc:
                        stopped = True
                        errors.append(
                            f"request{index} DP{lane}: {type(exc).__name__}: {exc}"
                        )

            await asyncio.gather(*(worker(i) for i in range(concurrency)))
        except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - timer if timer is not None else 0.0
    complete = not errors and len(records) >= args.num_prompts
    if args.duration_seconds is not None:
        complete = complete and elapsed >= args.duration_seconds
    result = aggregate_results(
        records, speculative=speculative, complete=complete, elapsed=elapsed
    )
    result.update(
        status="GSP_SCENARIO_COLLECTED"
        if result["complete"]
        else "GSP_SCENARIO_INCOMPLETE",
        mode=args.mode,
        stage=args.action,
        cache_percent_label=percent,
        expected_cached_tokens=factory.shared,
        expected_cache_hit_ratio=factory.shared / INPUT_TOKENS,
        requested_minimum_prompts=args.num_prompts,
        requested_minimum_duration_seconds=args.duration_seconds,
        concurrency=concurrency,
        warmed_dp_lanes=lanes if percent else [],
        used_dp_lanes=lanes,
        issues=errors,
        client_generation_wait_seconds_sum=generation_seconds,
        preparation_seconds_excluded=preparation_seconds,
        results=sorted(records, key=lambda r: r["index"]),
    )
    if outputs:
        metrics, _ = bench.calculate_metrics(
            input_rows,
            outputs,
            elapsed,
            tokenizer,
            "sglang",
            accept_length=None,
            plot_throughput=False,
        )
        result["native_bench_metrics"] = asdict(metrics)
    result["limits"] = [
        "Fixed-N input generation and all warm-up are excluded from the measurement window.",
        "Duration runs include dynamic input preparation and final drain; issue offsets reveal delivery gaps.",
        "Client concurrency is an upper bound; record CPU generation waits before claiming saturation.",
        "TTFT/TPOT are native streaming client definitions; speculative chunks are not individual kernel timestamps.",
        "Different cache scenarios use their configured shared prefix and fresh random suffixes, not identical full prompts.",
        "GSP random tokens are not a quality dataset; >0.5 is recorded separately from formal pressure qualification.",
    ]
    write_json(out_dir / "summary.json", result)
    return result


def parser_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "quick", "load"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "dspark-eager",
            "dspark-graph",
            "target-eager",
            "target-graph",
            "nextn-graph",
        ),
    )
    parser.add_argument("--cache-hit", default="all", choices=("all", "0", "50", "90"))
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="Minimum issuing duration in addition to num-prompts; outstanding requests drain afterwards",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.action == "quick":
        if (
            args.num_prompts not in (None, 1)
            or args.concurrency not in (None, 1)
            or args.duration_seconds is not None
        ):
            parser.error(
                "quick is exactly one request, concurrency one and no duration target"
            )
        args.num_prompts = args.concurrency = 1
    else:
        args.num_prompts = 64 if args.num_prompts is None else args.num_prompts
        args.concurrency = 8 if args.concurrency is None else args.concurrency
    if args.num_prompts < 1 or args.concurrency < 1:
        parser.error("num-prompts and concurrency must be positive")
    if args.duration_seconds is not None and (
        not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0
    ):
        parser.error("duration-seconds must be finite and positive")
    return args


def run(args, config=None):
    from config import load_config
    from client_common import (
        benchmark_context,
        fetch_text,
        graph_evidence,
        import_benchmark,
        prepare_run,
        validate_server,
        write_json,
    )

    config = load_config(args.config) if config is None else config
    run = prepare_run(config, "gsp-prefix", args.mode)
    percentages = [0, 50, 90] if args.cache_hit == "all" else [int(args.cache_hit)]
    summary = {
        "status": "GSP_RUN_NOT_STARTED",
        "mode": args.mode,
        "stage": args.action,
        "complete": False,
        "cases": [],
    }
    try:
        info = json.loads(fetch_text(config["base_url"] + "/server_info"))
        write_json(run / "server_info.before.json", info)
        selected = validate_server(info, config, args.mode)
        summary["server_configuration"] = selected
        preflight = prefix_preflight(info, config, args.concurrency, percentages)
        write_json(run / "preflight.json", preflight)
        summary["preflight"] = preflight
        if not preflight["ready"]:
            summary["status"] = "GSP_PREFLIGHT_BLOCKED"
            return 2
        if args.action == "check":
            summary["status"] = "GSP_PREFLIGHT_READY"
            return 0
        with benchmark_context():
            bench = import_benchmark()
            common = importlib.import_module("sglang.benchmark.datasets.common")
        with benchmark_context(bench), native_settings(bench):
            tokenizer = bench.get_tokenizer(config["tokenizer"])
            write_json(
                run / "protocol.json",
                {
                    "generator": "Community datasets.common.gen_prompt per shared prefix and per suffix; encode and pin actual lengths",
                    "generator_sha256": hashlib.sha256(
                        Path(common.__file__).read_bytes()
                    ).hexdigest(),
                    "benchmark_sha256": hashlib.sha256(
                        Path(bench.__file__).read_bytes()
                    ).hexdigest(),
                    "client_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "input_tokens": INPUT_TOKENS,
                    "output_tokens": OUTPUT_TOKENS,
                    "seed": args.seed,
                    "stream": True,
                    "ignore_eos": True,
                    "temperature": 0,
                    "routed_dp_rank": "Fixed per client worker; warm same DP lane before measurement",
                    "suffix_guard": "One non-special first suffix token unique per request index; no complete-prompt reuse",
                    "request_files": "requests.jsonl.gz contains all prepared input IDs and phase; issued.jsonl lists measured requests actually handed to the native client",
                    "formal_pressure_duration": "NOT_FROZEN",
                },
            )
            for percent in percentages:
                before_metrics = fetch_text(config["base_url"] + "/metrics")
                (run / f"metrics.cache{percent}.before.txt").write_text(before_metrics)
                result = asyncio.run(
                    run_scenario(
                        bench,
                        common,
                        config,
                        args,
                        run,
                        percent,
                        preflight["page_size"],
                        tokenizer,
                    )
                )
                after_metrics = fetch_text(config["base_url"] + "/metrics")
                (run / f"metrics.cache{percent}.after.txt").write_text(after_metrics)
                result["graph_evidence"] = graph_evidence(
                    before_metrics, after_metrics, args.mode
                )
                result["graph_evidence"]["includes_prefix_warmup"] = True
                write_json(run / f"cache{percent}/summary.json", result)
                summary["cases"].append(
                    {k: v for k, v in result.items() if k != "results"}
                )
                if not result["complete"]:
                    summary["status"] = "GSP_RUN_INCOMPLETE"
                    return 1
        after = json.loads(fetch_text(config["base_url"] + "/server_info"))
        write_json(run / "server_info.after.json", after)
        if validate_server(after, config, args.mode) != selected:
            raise ValueError("Relevant server configuration changed during measurement")
        summary["status"] = "GSP_RUN_COLLECTED"
        summary["complete"] = True
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        summary["status"] = "GSP_RUN_INCOMPLETE"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        write_json(run / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"Evidence: {run}", flush=True)


def main(argv=None):
    return run(parser_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
