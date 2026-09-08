#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicit, single-NPU experiment; does not patch or install production code."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap
import traceback


REVIEWED_KERNEL_COMMIT = "d974d3de5b7b0d6586a41f227cba93a861f07fe1"
KERNEL_AST_SCHEMA = "python-ast-fields-v1-ignore-empty-type-params"
REVIEWED_KERNEL_AST = "83898e6fdebe97b7cb0aad2164ab26c1f7e1e5ec52c6103242410e1c29da49d5"
EPS = 1e-5
# Existing split_qkv_rmsnorm_rope test's absolute tolerance, used diagnostically.
# This is not the final GLM model-quality or new-operator acceptance threshold.
DIAGNOSTIC_ATOL = 5e-2
KERNEL_PARAMETERS = (
    "input_ptr sin_ptr cos_ptr q_ptr k_ptr v_ptr q_weight_ptr q_bias_ptr "
    "k_weight_ptr k_bias_ptr batch_size q_hidden_size kv_hidden_size "
    "total_hidden_size eps Q_BLOCK_SIZE KV_BLOCK_SIZE BIAS NORMS "
    "HEAD_DIM ROPE_DIM HALF_ROPE_DIM PASS_DIM DO_PARTIAL DO_HALF"
).split()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", type=int, default=0, help="Visible logical NPU index"
    )
    parser.add_argument(
        "--out", type=Path, help="New evidence directory; never overwritten"
    )
    parser.add_argument(
        "--graph", action="store_true", help="Also check changed-input graph replay"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Also collect diagnostic event timings"
    )
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")
    return args


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_head():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def canonical_ast(value):
    """Versioned AST fields, without locations or Python 3.12's empty field."""
    if isinstance(value, ast.AST):
        return {
            "node": type(value).__name__,
            "fields": {
                name: canonical_ast(field)
                for name, field in ast.iter_fields(value)
                if not (name == "type_params" and field == [])
            },
        }
    if isinstance(value, list):
        return [canonical_ast(item) for item in value]
    # Typed representations also preserve bytes, complex numbers and Ellipsis.
    return {"type": type(value).__name__, "value": repr(value)}


def ast_fingerprint(node):
    payload = {"schema": KERNEL_AST_SCHEMA, "ast": canonical_ast(node)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_identity(function):
    source = inspect.getsource(function).replace("\r\n", "\n").replace("\r", "\n")
    source = textwrap.dedent(source).strip("\n") + "\n"
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    return source, ast_fingerprint(node)


def compiled_metadata(compiled):
    metadata = getattr(compiled, "metadata", None)
    fields = (
        "compile_mode",
        "parallel_mode",
        "force_simt_only",
        "force_simt_template",
        "num_warps",
    )
    return {key: getattr(metadata, key, None) for key in fields}


def launch_grid(vector_cores, heads):
    if vector_cores <= 0 or heads <= 0:
        raise ValueError("vector core count and local head count must be positive")
    return ((vector_cores + heads - 1) // heads, heads, 1)


def make_case(
    torch, tokens, heads, *, head_dim=192, seed=17, scale=1.0, position_shift=0
):
    """CPU inputs; each Q/K/V segment contains all heads, as in the real kernel."""
    if tokens <= 0 or heads <= 0 or head_dim not in (128, 192):
        raise ValueError("invalid diagnostic shape")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = heads * head_dim
    qkv = (torch.randn(tokens, 3 * width, generator=generator) * scale).to(
        torch.bfloat16
    )
    q_weight = torch.randn(head_dim, generator=generator).to(torch.bfloat16)
    k_weight = torch.randn(head_dim, generator=generator).to(torch.bfloat16)
    positions = torch.tensor([0, 1, 7, 127, 4096, 32767, 131071], dtype=torch.int64)
    positions = positions[torch.arange(tokens) % len(positions)] + position_shift
    inverse_frequency = 1.0 / (
        8000000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    angles = torch.outer(positions.float(), inverse_frequency)
    angles = torch.cat((angles, angles), dim=-1)
    return {
        "qkv": qkv,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "sin": angles.sin().reshape(tokens, 1, 1, head_dim).contiguous(),
        "cos": angles.cos().reshape(tokens, 1, 1, head_dim).contiguous(),
        "heads": heads,
        "head_dim": head_dim,
        "positions": positions.tolist(),
    }


def reference(torch, data):
    """FP32 norm + full NeoX RoPE; no artificial BF16 rounding between ops."""
    dim, heads = data["head_dim"], data["heads"]
    width = dim * heads
    q, k, v = data["qkv"].cpu().split(width, dim=-1)
    sin = data["sin"].cpu().reshape(-1, 1, dim)
    cos = data["cos"].cpu().reshape(-1, 1, dim)
    results = []
    for values, weight in ((q, data["q_weight"]), (k, data["k_weight"])):
        values = values.float().reshape(-1, heads, dim)
        values = values / torch.sqrt((values * values).mean(dim=-1, keepdim=True) + EPS)
        values = values * weight.cpu().float()
        rotated = torch.cat((-values[..., dim // 2 :], values[..., : dim // 2]), dim=-1)
        results.append((values * cos + rotated * sin).reshape(-1, width))
    return (*results, v.contiguous())


def compare(torch, outputs, expected):
    metrics = {}
    passed = True
    for name, actual, wanted in zip(("q", "k"), outputs[:2], expected[:2]):
        actual = actual.detach().float().cpu()
        wanted = wanted.float().cpu()
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(wanted).all())
        error = actual - wanted
        max_abs = error.abs().max().item() if finite else None
        rounded = wanted.to(torch.bfloat16).float()
        metrics[name] = {
            "finite": finite,
            "max_abs_vs_fp32": max_abs,
            "max_abs_vs_bf16_rounded": (actual - rounded).abs().max().item()
            if finite
            else None,
            "rms_error": error.square().mean().sqrt().item() if finite else None,
            "diagnostic_atol": DIAGNOSTIC_ATOL,
            "diagnostic_rtol": 0,
        }
        passed = passed and finite and max_abs <= DIAGNOSTIC_ATOL
    actual_v, wanted_v = outputs[2].detach().cpu(), expected[2].cpu()
    v_exact = actual_v.dtype == wanted_v.dtype and torch.equal(
        actual_v.contiguous().view(torch.int16), wanted_v.contiguous().view(torch.int16)
    )
    metrics["v_bitwise_equal"] = v_exact
    metrics["passed"] = bool(passed and v_exact)
    return metrics


def to_device(torch, data, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def allocate_outputs(torch, data):
    tokens = data["qkv"].shape[0]
    width = data["heads"] * data["head_dim"]
    return tuple(
        torch.full(
            (tokens, width),
            float("nan"),
            dtype=torch.bfloat16,
            device=data["qkv"].device,
        )
        for _ in range(3)
    )


def make_launcher(kernel, data, outputs, vector_cores):
    dim, heads = data["head_dim"], data["heads"]
    tokens, width = data["qkv"].shape[0], heads * dim
    grid = launch_grid(vector_cores, heads)
    for tensor in (
        data["qkv"],
        data["sin"],
        data["cos"],
        data["q_weight"],
        data["k_weight"],
        *outputs,
    ):
        if not tensor.is_contiguous():
            raise ValueError("The original fused kernel requires contiguous tensors")
    # Pointers deliberately retain the original weight, bias, weight, bias order.
    arguments = (
        data["qkv"],
        data["sin"],
        data["cos"],
        *outputs,
        data["q_weight"],
        None,
        data["k_weight"],
        None,
        tokens,
        width,
        width,
        3 * width,
        EPS,
        dim,
        dim,
        False,
        True,
        dim,
        dim,
        dim // 2,
        0,
    )

    def launch():
        # No compiler-mode override: test the installed 3.2.2 default path.
        return kernel[grid](*arguments, DO_PARTIAL=False, DO_HALF=True)

    return launch, grid


class Evidence:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.report = {
            "status": "RUNNING",
            "stages": [],
            "active_stage": None,
            "graph_status": "NOT_RUN",
            "benchmark_status": "NOT_RUN",
        }
        self.save()

    def save(self):
        temporary = self.directory / "report.json.tmp"
        temporary.write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        temporary.replace(self.directory / "report.json")

    def log(self, text):
        print(text, flush=True)
        with (self.directory / "run.log").open("a") as output:
            output.write(text + "\n")

    def step(self, name, operation):
        self.report["active_stage"] = name
        self.save()
        self.log("RUN " + name)
        result = operation()
        passed = not isinstance(result, dict) or result.get("passed", True)
        self.report["stages"].append({"name": name, "passed": passed, "result": result})
        self.save()
        if not passed:
            raise RuntimeError(name + " failed; see report.json metrics")
        self.log("PASS " + name)
        return result

    def fail(self):
        self.report["status"] = "FAILED"
        if self.report["graph_status"] == "RUNNING":
            self.report["graph_status"] = "FAILED"
        if self.report["benchmark_status"] == "RUNNING":
            self.report["benchmark_status"] = "FAILED"
        detail = traceback.format_exc()
        (self.directory / "traceback.txt").write_text(detail)
        self.report["error"] = detail
        self.save()
        self.log("FAILED " + str(self.report["active_stage"]))


def run(args, evidence):
    evidence.report["environment"] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "tool_git_head": git_head(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_ops_sha256": hashlib.sha256(
            Path(__file__).with_name("native192_probe_ops.py").read_bytes()
        ).hexdigest(),
        "versions": {
            name: distribution_version(name)
            for name in ("triton-ascend", "torch", "torch-npu", "sgl-kernel-npu")
        },
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "requested_device": args.device,
    }
    evidence.save()
    if evidence.report["environment"]["versions"]["triton-ascend"] != "3.2.2":
        raise RuntimeError(
            "This experiment targets the agreed triton-ascend 3.2.2 environment; no packages were changed"
        )
    import torch
    import torch_npu  # noqa: F401
    import triton
    import triton.language.semantic as semantic

    if not torch.npu.is_available() or args.device >= torch.npu.device_count():
        raise RuntimeError("Requested logical NPU is not available")
    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    module = importlib.import_module("sgl_kernel_npu.norm.split_qkv_rmsnorm_rope")
    kernel = module.split_qkv_rmsnorm_rope_kernel
    source, source_hash = source_identity(kernel.fn)
    (evidence.directory / "installed_kernel.py").write_text(source, encoding="utf-8")
    (evidence.directory / "installed_arange.py").write_text(
        inspect.getsource(semantic.arange)
    )
    evidence.report["environment"].update(
        {
            "triton_module_version": triton.__version__,
            "torch_module_version": torch.__version__,
            "torch_npu_module_version": torch_npu.__version__,
            "triton_path": triton.__file__,
            "semantic_path": semantic.__file__,
            "kernel_module_path": module.__file__,
            "kernel_function_ast_schema": KERNEL_AST_SCHEMA,
            "kernel_function_ast_sha256": source_hash,
            "kernel_function_source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "reviewed_kernel_commit": REVIEWED_KERNEL_COMMIT,
            "matches_reviewed_kernel_ast": source_hash == REVIEWED_KERNEL_AST,
            "device_properties": str(torch.npu.get_device_properties(device)),
            "compiler_options_override": {},
        }
    )
    evidence.save()
    if list(inspect.signature(kernel.fn).parameters) != KERNEL_PARAMETERS:
        raise RuntimeError(
            "Installed kernel signature differs from the audited interface; source has been saved"
        )
    num_aicore, vector_cores = module.get_device_properties()
    evidence.report["environment"].update(
        {"num_aicore": num_aicore, "num_vectorcore": vector_cores}
    )
    if source_hash != REVIEWED_KERNEL_AST:
        evidence.log(
            "NOTE installed kernel source differs from the local audit; results describe the saved installed source"
        )

    def check_baseline():
        data = make_case(torch, 8, 4, head_dim=128)
        on_device = to_device(torch, data, device)
        width = 4 * 128
        outputs = module.split_qkv_rmsnorm_rope(
            on_device["qkv"],
            on_device["sin"],
            on_device["cos"],
            width,
            width,
            128,
            eps=EPS,
            q_weight=on_device["q_weight"],
            k_weight=on_device["k_weight"],
        )
        torch.npu.synchronize()
        return compare(torch, outputs, reference(torch, data))

    evidence.step("existing_host_128_control", check_baseline)
    ops = importlib.import_module("native192_probe_ops")
    x_cpu = torch.linspace(-1.5, 2.5, 192, dtype=torch.float32)
    x = x_cpu.to(device)
    expected = (
        x_cpu,
        x_cpu.square().mean().reshape(1),
        torch.cat((-x_cpu[96:], x_cpu[:96])),
    )
    for operation, target in zip((ops.copy192, ops.reduce192, ops.rotate192), expected):

        def check_op(operation=operation, target=target):
            result = torch.full(
                target.shape, float("nan"), device=device, dtype=torch.float32
            )
            compiled = operation[(1,)](x, result)
            torch.npu.synchronize()
            torch.testing.assert_close(result.cpu(), target, rtol=1e-5, atol=1e-6)
            return {"passed": True, "metadata": compiled_metadata(compiled)}

        evidence.step(operation.fn.__name__, check_op)

    cases = [(tokens, heads, 1.0) for heads in (1, 4, 8) for tokens in (1, 8, 33)]
    cases.append((launch_grid(vector_cores, 4)[0] + 1, 4, 1.0))
    cases.extend(((8, 4, 0.0), (8, 4, 1e-4)))
    for tokens, heads, scale in dict.fromkeys(cases):

        def check_fused(tokens=tokens, heads=heads, scale=scale):
            data = make_case(torch, tokens, heads, seed=tokens + heads, scale=scale)
            on_device = to_device(torch, data, device)
            outputs = allocate_outputs(torch, on_device)
            launch, grid = make_launcher(kernel, on_device, outputs, vector_cores)
            compiled = launch()
            torch.npu.synchronize()
            return {
                **compare(torch, outputs, reference(torch, data)),
                "grid": grid,
                "positions": data["positions"],
                "metadata": compiled_metadata(compiled),
            }

        evidence.step(f"fused192_t{tokens}_h{heads}_scale{scale:g}", check_fused)

    # Keep addresses fixed through graph replay and timing. All setup is outside capture.
    data = make_case(torch, 8, 4)
    on_device = to_device(torch, data, device)
    outputs = allocate_outputs(torch, on_device)
    launch, _ = make_launcher(kernel, on_device, outputs, vector_cores)
    graph = None
    execution_stream = torch.npu.Stream() if (args.graph or args.benchmark) else None
    if args.graph:

        def check_graph():
            nonlocal graph
            torch.npu.synchronize()
            with torch.npu.stream(execution_stream):
                for _ in range(3):
                    launch()
            execution_stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(
                graph, stream=execution_stream, auto_dispatch_capture=True
            ):
                launch()
            torch.npu.synchronize()
            checks = []
            for seed, shift in ((73, 13), (101, 257)):
                changed = make_case(torch, 8, 4, seed=seed, position_shift=shift)
                with torch.npu.stream(execution_stream):
                    for key in ("qkv", "sin", "cos", "q_weight", "k_weight"):
                        on_device[key].copy_(changed[key])
                    for output in outputs:
                        output.fill_(float("nan"))
                    graph.replay()
                execution_stream.synchronize()
                checks.append(compare(torch, outputs, reference(torch, changed)))
            return {
                "passed": all(check["passed"] for check in checks),
                "replays": checks,
                "shape": [8, 4, 192],
                "auto_dispatch_capture": True,
                "stream_policy": "same_side_stream_for_warmup_capture_update_replay",
            }

        evidence.report["graph_status"] = "RUNNING"
        evidence.step("graph_changed_input_replay", check_graph)
        evidence.report["graph_status"] = "PASS_SINGLE_OPERATOR_ONLY"

    if args.benchmark:

        def measure():
            results = {}
            torch.npu.synchronize()
            for name, call in (
                ("eager", launch),
                ("graph", graph.replay if graph else None),
            ):
                if call is None:
                    continue
                with torch.npu.stream(execution_stream):
                    for _ in range(5):
                        call()
                    execution_stream.synchronize()
                    samples = []
                    for _ in range(5):
                        start, end = (
                            torch.npu.Event(enable_timing=True),
                            torch.npu.Event(enable_timing=True),
                        )
                        start.record()
                        for _ in range(100):
                            call()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000 / 100)
                results[name] = {
                    "us_per_call_samples": samples,
                    "median_us": statistics.median(samples),
                }
            return {
                "shape": [8, 4, 192],
                "event_stream_interval_includes_launch_gaps": True,
                "performance_acceptance": "NOT_EVALUATED",
                "timings": results,
            }

        evidence.report["benchmark_status"] = "RUNNING"
        evidence.step("diagnostic_event_timing", measure)
        evidence.report["benchmark_status"] = "RECORDED_NOT_A_SPEEDUP_RESULT"
    evidence.report["status"] = (
        "EAGER_AND_GRAPH_DIAGNOSTIC_PASS"
        if args.graph
        else "EAGER_DIAGNOSTIC_PASS_GRAPH_NOT_RUN"
    )
    evidence.report["active_stage"] = None
    evidence.save()


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = args.out or Path(
        f"/home/tyj/glm52-ms1/evidence/native192-{stamp}-{os.getpid()}"
    )
    evidence = Evidence(directory)
    evidence.log("Evidence: " + str(directory.resolve()))
    try:
        evidence.step("native192_experiment", lambda: run(args, evidence))
        evidence.report["active_stage"] = None
        evidence.save()
    except (Exception, KeyboardInterrupt):
        evidence.fail()
        return 1
    evidence.log(evidence.report["status"])
    evidence.log(
        "This result does not establish DSpark, distributed execution, or model performance."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
