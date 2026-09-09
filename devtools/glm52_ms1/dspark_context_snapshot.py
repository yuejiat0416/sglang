# SPDX-License-Identifier: Apache-2.0
"""Opt-in, one-request observation of the existing dense DSpark context path.

Loaded by a private sitecustomize; imports no framework until a matching request.
Never substitute a computed result or retry an original function after an error.
Temporary sync-branch diagnostics, excluded from the upstream feature branch.
"""

import contextvars
import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
import time
from pathlib import Path

ENV = "GLM52_CONTEXT_SNAPSHOT_CONFIG"
WORKER = "sglang.srt.speculative.dspark_components.dspark_worker_v2"
MODEL = "sglang.srt.models.dspark"
FUSED = "sglang.kernels.ops.speculative.dspark.fused_kv_write"
ACTIVE = contextvars.ContextVar("glm52_context_snapshot", default=None)
_INSTALLED = False
# Fixed diagnostic geometry; tests substitute small fixtures, never CLI overrides.
HIDDEN, FEATURES, LAYERS, HEAD_DIM, LOCAL_HEADS = 6144, 5, 5, 192, 4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_rows(count):
    require(count > 0, "Empty prefill")
    return sorted({0, count // 2, count - 1})


def cpu_copy(tensor, rows=None):
    # Blocking CPU transfer follows the producer's current stream. The clone
    # prevents even CPU fixtures/views from sharing mutable storage.
    if rows is not None:
        import torch

        index = torch.tensor(rows, dtype=torch.long, device=tensor.device)
        tensor = tensor.index_select(0, index)
    return tensor.detach().to("cpu", non_blocking=False).contiguous().clone()


class Capture:
    def __init__(self, config, worker, batch):
        self.config = config
        self.worker = worker
        self.model = worker.draft_model
        self.batch = batch
        self.root = Path(config["run_dir"])
        self.used = 0
        self.failed = False
        self.handles = []
        self.rows = []
        self.buffers = []
        self.in_write = False
        self.started = time.monotonic()
        self.report = {
            "status": "CAPTURING",
            "run_id": config["run_id"],
            "rid": config["rid"],
            "pid": os.getpid(),
            "rank": 0,
            "stage": "first_prefill",
            "arrays": {},
            "errors": [],
            "source": {},
        }

    def attempt(self, func, *args, **kwargs):
        if self.failed:
            return None
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            self.failed = True
            self.report["status"] = "SNAPSHOT_FAILED"
            self.report["errors"].append(f"{type(exc).__name__}: {exc}")
            print(f"GLM context snapshot failed: {exc}", file=sys.stderr, flush=True)
            return None

    def save(self, name, tensor, rows=None):
        import numpy as np
        import torch

        require(name not in self.report["arrays"], f"Repeated boundary: {name}")
        shape = list(tensor.shape)
        if rows is not None:
            shape[0] = len(rows)
        size = tensor.element_size()
        for dim in shape:
            size *= dim
        # NPY headers are small; reserve 4 KiB per array before copying/writing.
        require(
            self.used + size + 4096 <= self.config["max_bytes"],
            "Snapshot byte budget exceeded",
        )
        value = cpu_copy(tensor, rows)
        bf16 = value.dtype == torch.bfloat16
        array = value.view(torch.uint16).numpy() if bf16 else value.numpy()
        path = self.root / f"{name}.npy"
        with path.open("xb") as f:
            np.save(f, array, allow_pickle=False)
        self.used += path.stat().st_size
        self.report["arrays"][name] = {
            "file": path.name,
            "dtype": str(tensor.dtype),
            "encoding": "bf16_uint16" if bf16 else "numpy",
            "shape": shape,
            "original_shape": list(tensor.shape),
            "original_stride": list(tensor.stride()),
            "original_device": str(tensor.device),
            "copy": "blocking CPU transfer on current producer stream, then independent clone",
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
        return value

    def start(self):
        import torch

        args = self.worker.server_args
        require(str(args.device) == "npu", "Diagnostic requires NPU")
        require(args.speculative_algorithm == "DSPARK", "Diagnostic requires DSPARK")
        require(
            (args.tp_size, args.dp_size, args.nnodes, args.pp_size) == (16, 1, 1, 1),
            "Diagnostic scope is TP16/DP1/PP1/single node",
        )
        require(args.disable_cuda_graph, "Diagnostic requires eager")
        require(
            os.environ.get("SGLANG_RAGGED_VERIFY_MODE") == "static",
            "Diagnostic requires explicit static",
        )
        require(
            os.environ.get("SGLANG_NPU_GLM_DSPARK_QUAROT") == "original",
            "Diagnostic requires existing original mode",
        )
        require(
            type(self.model).__name__ == "DSparkDraftModel", "Unexpected draft class"
        )
        require(not self.worker._draft_is_moe, "Dense draft only")
        require(
            not self.worker._target_hidden_projection_enabled,
            "Pre-gather projection is outside this capture scope",
        )
        require(
            len(self.batch.reqs) == 1 and list(self.batch.prefix_lens) == [0],
            "Expected one cold prefill",
        )
        count = int(self.batch.extend_lens[0])
        require(count <= 512, "Expected the short fixed request")
        self.rows = selected_rows(count)
        self.report.update(
            rows=self.rows,
            token_count=count,
            forward_ct=int(self.batch.forward_iter),
            prefix_lens=list(self.batch.prefix_lens),
            model_path=args.model_path,
            draft_path=args.speculative_draft_model_path,
            device=str(args.device),
            tp_size=args.tp_size,
            config_environment={
                k: os.environ.get(k)
                for k in (
                    "SGLANG_RAGGED_VERIFY_MODE",
                    "SGLANG_NPU_GLM_DSPARK_QUAROT",
                    "SGLANG_ENABLE_FAST_INPUT_LOGPROBS",
                )
            },
        )
        require(
            type(self.worker.model_runner.model).__name__ == "GlmMoeDsaForCausalLM",
            "Unexpected target model",
        )
        self.report["observer_sha256"] = file_hash(__file__)
        self.report["scope_geometry"] = {
            "hidden": HIDDEN,
            "features": FEATURES,
            "layers": LAYERS,
            "head_dim": HEAD_DIM,
            "local_heads": LOCAL_HEADS,
        }
        self.report["torch_version"] = torch.__version__
        self.report["artifact_configs"] = {}
        for role, directory in (
            ("target", args.model_path),
            ("draft", args.speculative_draft_model_path),
        ):
            path = Path(directory) / "config.json"
            self.report["artifact_configs"][role] = {
                "path": str(path),
                "sha256": file_hash(path),
            }
        self.save("prompt_ids", self.batch.input_ids)
        for name in (
            WORKER,
            MODEL,
            "sglang.srt.models.dflash",
            FUSED,
            "sglang.srt.layers.layernorm",
            "sglang.srt.layers.rotary_embedding.base",
            "sglang.srt.hardware_backend.npu.memory_pool_npu",
        ):
            module = sys.modules.get(name)
            if module is not None and getattr(module, "__file__", None):
                self.report["source"][name] = {
                    "file": module.__file__,
                    "sha256": file_hash(module.__file__),
                }
        require(
            tuple(self.model.fc.weight.shape) == (HIDDEN, HIDDEN * FEATURES),
            "Unexpected FC shape",
        )
        require(self.model.fc.weight.dtype == torch.bfloat16, "Expected BF16 FC")
        require(self.model.fc.bias is None, "FC bias is outside diagnostic scope")
        self.report["hidden_norm_eps"] = float(self.model.hidden_norm.variance_epsilon)
        self.save("fc_weight", self.model.fc.weight)
        self.save("hidden_norm_weight", self.model.hidden_norm.weight)

        def fc_hook(module, inputs, output):
            if ACTIVE.get() is not self or not self.in_write:
                return
            self.attempt(self.save, "hidden", inputs[0], self.rows)
            self.attempt(self.save, "fc_output", output, self.rows)
            # Returning None preserves the original output object.

        def norm_hook(module, inputs, output):
            if ACTIVE.get() is not self or not self.in_write:
                return
            self.attempt(self.save, "norm_input", inputs[0], self.rows)
            self.attempt(self.save, "context", output, self.rows)

        self.handles.append(self.model.fc.register_forward_hook(fc_hook))
        self.handles.append(self.model.hidden_norm.register_forward_hook(norm_hook))

    def before_write(self, values):
        import torch

        require(not self.buffers, "More than one context write")
        require(
            not values.get("target_hidden_is_projected", False),
            "Already projected hidden",
        )
        require(
            values.get("commit_lens") is None and values.get("cache_loc_2d") is None,
            "Not a plain prefill write",
        )
        pool = values["pool"]
        require(
            type(pool).__name__ in {"MHATokenToKVPool", "NPUMHATokenToKVPool"},
            "Uncovered pool type",
        )
        self.report["pool_type"] = type(pool).__name__
        require(
            not pool.is_quantized_kv_cache and pool.store_dtype == torch.bfloat16,
            "Uncovered KV storage",
        )
        require(
            pool.layer_transfer_counter is None,
            "Layer transfer is outside diagnostic scope",
        )
        n = self.report["token_count"]
        require(
            tuple(values["target_hidden"].shape) == (n, HIDDEN * FEATURES),
            "Unexpected hidden shape",
        )
        require(
            tuple(values["positions"].shape) == (n,)
            and tuple(values["cache_loc"].shape) == (n,),
            "Unexpected position/location shape",
        )
        positions = cpu_copy(values["positions"]).tolist()
        locs = cpu_copy(values["cache_loc"]).tolist()
        require(positions == list(range(n)), "Cold prefill positions are not 0..N-1")
        require(
            len(set(locs)) == n and min(locs) >= 0, "Invalid/repeated cache addresses"
        )
        self.locs = [locs[i] for i in self.rows]
        self.save("positions", values["positions"], self.rows)
        self.save("cache_locs", values["cache_loc"], self.rows)
        require(len(self.model.layers) == LAYERS, "Expected five draft layers")
        self.report["layers"] = []
        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            require(
                (attn.head_dim, attn.num_kv_heads, attn.q_size, attn.kv_size)
                == (
                    HEAD_DIM,
                    LOCAL_HEADS,
                    HEAD_DIM * LOCAL_HEADS,
                    HEAD_DIM * LOCAL_HEADS,
                ),
                "Unexpected local head geometry",
            )
            require(
                attn.qkv_proj.bias is None
                and attn.qkv_proj.weight.dtype == torch.bfloat16,
                "Uncovered QKV weights",
            )
            require(
                tuple(attn.qkv_proj.weight.shape)
                == (3 * HEAD_DIM * LOCAL_HEADS, HIDDEN),
                "Unexpected QKV shape",
            )
            require(
                attn.attn.k_scale is None and attn.attn.v_scale is None,
                "KV scaling is outside diagnostic scope",
            )
            index = attn.attn.layer_id - pool.start_layer
            # Direct tensors of the known plain pool; no public getter waits or
            # arbitrary-pointer dereferences are introduced by the observer.
            k, v = pool.k_buffer[index], pool.v_buffer[index]
            if type(pool).__name__ == "NPUMHATokenToKVPool":
                require(
                    k.is_contiguous() and v.is_contiguous(),
                    "Noncontiguous NPU paged pool",
                )
                require(
                    tuple(k.shape[-2:]) == (LOCAL_HEADS, HEAD_DIM)
                    and tuple(v.shape[-2:]) == (LOCAL_HEADS, HEAD_DIM),
                    "Unexpected paged head dimensions",
                )
                k = k.view(-1, LOCAL_HEADS, HEAD_DIM)
                v = v.view(-1, LOCAL_HEADS, HEAD_DIM)
            for buf in (k, v):
                require(
                    buf.dtype == torch.bfloat16
                    and tuple(buf.shape[1:]) == (LOCAL_HEADS, HEAD_DIM),
                    "Uncovered pool layout",
                )
                require(max(locs) < buf.shape[0], "Cache address exceeds buffer")
            self.buffers.append((k, v))
            self.save(f"qkv_weight_{i}", attn.qkv_proj.weight)
            self.save(f"k_norm_weight_{i}", attn.k_norm.weight)
            rotary = attn.rotary_emb
            require(
                type(rotary).__name__ == "RotaryEmbedding"
                and rotary.rotary_dim == HEAD_DIM,
                "Uncovered RoPE configuration",
            )
            self.save(
                f"cos_sin_{i}", rotary.cos_sin_cache, [positions[j] for j in self.rows]
            )
            self.report["layers"].append(
                {
                    "layer_id": attn.attn.layer_id,
                    "pool_index": index,
                    "q_size": attn.q_size,
                    "kv_size": attn.kv_size,
                    "head_dim": HEAD_DIM,
                    "local_heads": LOCAL_HEADS,
                    "k_norm_eps": float(attn.k_norm.variance_epsilon),
                    "is_neox_style": bool(rotary.is_neox_style),
                    "base": float(rotary.base),
                }
            )

    def before_fused(self, values):
        require("kv_projected" not in self.report["arrays"], "Multiple fused writes")
        require(
            (values["num_layers"], values["kv_size"], values["head_dim"])
            == (LAYERS, HEAD_DIM * LOCAL_HEADS, HEAD_DIM),
            "Unexpected fused geometry",
        )
        require(values.get("commit_lens") is None, "Fused commit mask in prefill")
        self.report["source"][FUSED] = {
            "file": sys.modules[FUSED].__file__,
            "sha256": file_hash(sys.modules[FUSED].__file__),
        }
        self.report["branch"] = "fused_kv_norm_rope_write"
        self.save("kv_projected", values["kv"], self.rows)
        self.save("fused_meta", values["meta"])
        self.save("fused_knw", values["k_norm_weights"])
        self.save("fused_positions", values["positions"], self.rows)
        self.save("fused_locs", values["locs"], self.rows)
        self.save(
            "fused_cos_sin",
            values["cos_sin_cache"],
            [int(x) for x in cpu_copy(values["positions"], self.rows).tolist()],
        )
        self.report["fused_eps"] = float(values["eps"])
        self.report["fused_is_neox_style"] = bool(values.get("is_neox_style", True))
        meta = cpu_copy(values["meta"]).tolist()
        expected = [
            [k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0)]
            for k, v in self.buffers
        ]
        self.report["fused_meta_matches_pool"] = meta == expected
        # Only inspect the already-built bundle; never build/recompute it.
        cached = self.model._fused_kv_write_cache
        require(cached is not None, "Missing runtime bundle")
        self.save("fused_weight", cached[1][0])

    def before_stacked(self, values):
        require(
            sys.getprofile() is None,
            "Existing Python profiler: cannot install local observation",
        )
        self.report["branch"] = "stacked_ctx_kv"
        stacked = values["stacked"]
        require(stacked["bias"] is None, "Stacked bias outside scope")
        self.save("stacked_weight", stacked["weight"])
        self.save("stacked_knw", stacked["k_norm_weight"])
        self.report["stacked_eps"] = float(stacked["eps"])

        def rope_input(module, inputs):
            if ACTIVE.get() is not self or not self.in_write:
                return
            self.attempt(
                require,
                not hasattr(module, "sin_cos_cache"),
                "Preselected sin_cos_cache requires separate mapping analysis",
            )
            self.report["rope_forward_method"] = getattr(
                getattr(module, "_forward_method", None),
                "__qualname__",
                type(module).__name__,
            )
            self.attempt(self.save, "stacked_rope_positions", inputs[0], self.rows)
            self.attempt(self.save, "stacked_rope_input", inputs[2], self.rows)
            self.attempt(
                self.save,
                "stacked_cos_sin",
                module.cos_sin_cache,
                [int(x) for x in cpu_copy(inputs[0], self.rows).tolist()],
            )

        self.handles.append(
            self.model.layers[0].self_attn.rotary_emb.register_forward_pre_hook(
                rope_input
            )
        )

    def stacked_return(self, frame):
        # Only the original _project_ctx_kv_stacked Python frame at return is
        # inspected. No code copy/replacement or global torch F.linear wrapper.
        local = frame.f_locals
        kv = local["kv_all"]
        self.save("kv_projected", kv.reshape(kv.shape[0], -1), self.rows)
        k, v = local["k_all"], local["v_all"]
        for i in range(LAYERS):
            self.save(f"write_k_{i}", k[i], self.rows)
            self.save(f"write_v_{i}", v[i], self.rows)

    def after_write(self):
        require(
            self.report.get("branch") in {"fused_kv_norm_rope_write", "stacked_ctx_kv"},
            "Runtime selected uncovered per-layer path; it was not changed",
        )
        for i, (k, v) in enumerate(self.buffers):
            self.save(f"pool_k_{i}", k, self.locs)
            self.save(f"pool_v_{i}", v, self.locs)

    def finish(self, original_error=False):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception as exc:
                self.failed = True
                self.report["errors"].append(f"Cannot remove observer handle: {exc}")
        if original_error:
            self.failed = True
            self.report["errors"].append(
                "Original service forward raised; exception propagated unchanged"
            )
        if not self.failed:
            self.attempt(
                lambda: require(
                    f"pool_v_{LAYERS - 1}" in self.report["arrays"]
                    and "context" in self.report["arrays"],
                    "Missing observed boundaries",
                )
            )
        self.report["status"] = (
            "SNAPSHOT_FAILED" if self.failed else "CONTEXT_SNAPSHOT_COLLECTED"
        )
        self.report["saved_bytes"] = self.used
        self.report["elapsed_seconds_with_observation"] = (
            time.monotonic() - self.started
        )
        try:
            write_json(self.root / "snapshot.json", self.report)
        except Exception as exc:
            print(
                f"Cannot save context snapshot status: {exc}",
                file=sys.stderr,
                flush=True,
            )


def install_worker(module, config):
    original = module.DSparkWorkerV2._forward_prefill

    @functools.wraps(original)
    def observed(worker, batch, *args, **kwargs):
        if worker.ps.tp_rank != 0 or not any(
            r.rid == config["rid"] for r in batch.reqs
        ):
            return original(worker, batch, *args, **kwargs)
        claim = Path(config["run_dir"]) / "capture.claim"
        try:
            with claim.open("x") as f:
                f.write(str(os.getpid()))
        except OSError as exc:
            if not isinstance(exc, FileExistsError):
                print(
                    f"Cannot claim context snapshot: {exc}", file=sys.stderr, flush=True
                )
            return original(worker, batch, *args, **kwargs)
        capture = Capture(config, worker, batch)
        capture.attempt(capture.start)
        token = ACTIVE.set(capture)
        raised = True
        try:
            result = original(worker, batch, *args, **kwargs)
            raised = False
            return result
        finally:
            ACTIVE.reset(token)
            capture.finish(raised)

    module.DSparkWorkerV2._forward_prefill = observed


def install_model(module):
    original = module.DSparkDraftMixin.write_target_hidden_kv
    signature = inspect.signature(original)

    @functools.wraps(original)
    def observed(model, *args, **kwargs):
        capture = ACTIVE.get()
        if capture is None or model is not capture.model or capture.failed:
            return original(model, *args, **kwargs)
        values = signature.bind(model, *args, **kwargs).arguments
        capture.attempt(capture.before_write, values)
        capture.in_write = True
        try:
            result = original(model, *args, **kwargs)
        finally:
            capture.in_write = False
        capture.attempt(capture.after_write)
        return result

    module.DSparkDraftMixin.write_target_hidden_kv = observed
    stacked_original = getattr(module.DSparkDraftMixin, "_project_ctx_kv_stacked", None)
    if stacked_original is None:
        return
    stacked_signature = inspect.signature(stacked_original)

    @functools.wraps(stacked_original)
    def observed_stacked(model, *args, **kwargs):
        capture = ACTIVE.get()
        if (
            capture is None
            or model is not capture.model
            or not capture.in_write
            or capture.failed
        ):
            return stacked_original(model, *args, **kwargs)
        capture.attempt(
            capture.before_stacked,
            stacked_signature.bind(model, *args, **kwargs).arguments,
        )
        if capture.failed:
            return stacked_original(model, *args, **kwargs)

        def on_return(frame, event, arg):
            if (
                event == "return"
                and frame.f_code is stacked_original.__code__
                and arg is not None
            ):
                capture.attempt(capture.stacked_return, frame)

        sys.setprofile(on_return)
        try:
            return stacked_original(model, *args, **kwargs)
        finally:
            sys.setprofile(None)

    module.DSparkDraftMixin._project_ctx_kv_stacked = observed_stacked


def install_fused(module):
    original = module.fused_kv_norm_rope_write
    signature = inspect.signature(original)

    @functools.wraps(original)
    def observed(*args, **kwargs):
        capture = ACTIVE.get()
        if capture is not None and capture.in_write and not capture.failed:
            capture.attempt(
                capture.before_fused, signature.bind(*args, **kwargs).arguments
            )
        return original(*args, **kwargs)

    module.fused_kv_norm_rope_write = observed


class ObserverLoader(importlib.abc.Loader):
    def __init__(self, loader, callback):
        self.loader, self.callback = loader, callback

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.callback(module)


class ObserverFinder(importlib.abc.MetaPathFinder):
    def __init__(self, config):
        self.callbacks = {
            WORKER: lambda m: install_worker(m, config),
            MODEL: install_model,
            FUSED: install_fused,
        }

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.callbacks:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = ObserverLoader(spec.loader, self.callbacks[fullname])
        return spec


def install():
    global _INSTALLED
    config_path = os.environ.get(ENV)
    if _INSTALLED or not config_path:
        return
    config = json.loads(Path(config_path).read_text())
    finder = ObserverFinder(config)
    require(
        not any(name in sys.modules for name in finder.callbacks),
        "Observer must install before model imports",
    )
    sys.meta_path.insert(0, finder)
    _INSTALLED = True
