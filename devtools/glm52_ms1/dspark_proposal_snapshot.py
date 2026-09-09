# SPDX-License-Identifier: Apache-2.0
"""Opt-in observation of one first proposal, layer0, rank0. Sync branch only.

No framework imports at startup, replacement tensors, additional model calls or
Triton compilation. Failures belong to diagnostics; originals run exactly once.
"""

import contextvars
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
import time
from pathlib import Path

from dspark_context_snapshot import Capture as ArrayCapture
from dspark_context_snapshot import (
    ObserverLoader,
    cpu_copy,
    file_hash,
    require,
    write_json,
)

ENV = "GLM52_PROPOSAL_SNAPSHOT_CONFIG"
WORKER = "sglang.srt.speculative.dspark_components.dspark_worker_v2"
DRAFT = "sglang.srt.speculative.dspark_components.dspark_draft"
DFLASH = "sglang.srt.models.dflash"
BACKEND = "sglang.srt.hardware_backend.npu.attention.ascend_backend"
ACTIVE = contextvars.ContextVar("glm52_proposal_snapshot", default=None)
HIDDEN, HEAD_DIM, HEADS, QUERIES = 6144, 192, 4, 8
_INSTALLED = False


def source(report, name, path):
    report["source"][name] = {"file": str(path), "sha256": file_hash(path)}


class Capture(ArrayCapture):
    def __init__(self, config, worker, batch, prefill):
        super().__init__(config, worker, batch)
        self.prefill = prefill
        self.in_prepare = False
        self.finished = False
        self.report.update(stage="first_proposal_layer0", layer_id=0)

    def attempt(self, function, *args, **kwargs):
        if self.failed:
            return None
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            self.failed = True
            self.report["errors"].append(f"{type(exc).__name__}: {exc}")
            print(f"GLM proposal snapshot failed: {exc}", file=sys.stderr, flush=True)
            return None

    def start(self):
        import torch

        a = self.worker.server_args
        require(
            a.device == "npu" and a.speculative_algorithm == "DSPARK", "NPU DSPARK only"
        )
        require(
            (a.tp_size, a.dp_size, a.nnodes, a.pp_size) == (16, 1, 1, 1),
            "Scope TP16/DP1/PP1/single node",
        )
        require(a.disable_cuda_graph, "Eager only")
        require(os.environ.get("SGLANG_RAGGED_VERIFY_MODE") == "static", "Static only")
        require(
            os.environ.get("SGLANG_NPU_GLM_DSPARK_QUAROT") == "original",
            "Original mode only",
        )
        require(
            type(self.worker.model_runner.model).__name__ == "GlmMoeDsaForCausalLM",
            "Unexpected target",
        )
        require(
            type(self.model).__name__ == "DSparkDraftModel"
            and not self.worker._draft_is_moe,
            "Dense draft only",
        )
        require(len(self.batch.reqs) == 1, "One request only")
        require(
            self.prefill is not None and not self.prefill.get("error"),
            "Missing valid cold prefill observation",
        )
        self.report["prefill"] = self.prefill
        self.report["observer_sha256"] = file_hash(__file__)
        self.report["torch_version"] = torch.__version__
        self.report["draft_path"] = a.speculative_draft_model_path
        self.report["model_path"] = a.model_path
        self.report["draft_artifact_files"] = {
            path.name: {
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in Path(a.speculative_draft_model_path).glob("*.safetensors")
        }
        self.report["artifact_configs"] = {
            role: {
                "path": str(Path(path) / "config.json"),
                "sha256": file_hash(Path(path) / "config.json"),
            }
            for role, path in (
                ("draft", a.speculative_draft_model_path),
                ("target", a.model_path),
            )
        }
        self.report["geometry"] = dict(
            hidden=HIDDEN, heads=HEADS, head_dim=HEAD_DIM, queries=QUERIES
        )
        self.report["forward_ct"] = int(self.batch.forward_iter)
        self.layer = self.model.layers[0]
        self.attn = self.layer.self_attn
        require(self.attn.attn.layer_id == 0, "First layer must be layer0")
        require(
            self.layer.attention_conv is None and self.attn.attention_sink_bias is None,
            "Conv/sinks not covered",
        )
        require(
            self.attn.head_dim == HEAD_DIM
            and self.attn.q_size == self.attn.kv_size == HEADS * HEAD_DIM,
            "Unexpected local head layout",
        )
        require(
            tuple(self.attn.qkv_proj.weight.shape) == (3 * HEADS * HEAD_DIM, HIDDEN),
            "Unexpected QKV weight layout",
        )
        require(
            self.attn.qkv_proj.weight.dtype == torch.bfloat16
            and self.attn.qkv_proj.bias is None,
            "Only current BF16 bias-free projection is covered",
        )
        self.report["input_norm_eps"] = float(
            self.layer.input_layernorm.variance_epsilon
        )
        require(
            type(self.attn.rotary_emb).__name__ == "RotaryEmbedding"
            and self.attn.rotary_emb.rotary_dim == HEAD_DIM,
            "Only default full-head rotary cache covered",
        )
        self.report["rope_base"] = float(self.attn.rotary_emb.base)
        self.save("input_norm_weight", self.layer.input_layernorm.weight)
        self.save("qkv_weight", self.attn.qkv_proj.weight)
        self.save("prefix_lens", self.batch.seq_lens)
        self.save("prefix_lens_cpu", self.batch.seq_lens_cpu)
        self.save("bonus_tokens", self.batch.spec_info.bonus_tokens)
        for name in (
            WORKER,
            DRAFT,
            DFLASH,
            BACKEND,
            "sglang.srt.models.dspark",
            "sglang.srt.layers.layernorm",
            "sglang.srt.layers.rotary_embedding.base",
            "sglang.srt.layers.radix_attention",
            "sglang.srt.hardware_backend.npu.memory_pool_npu",
        ):
            module = sys.modules.get(name)
            if module is not None and getattr(module, "__file__", None):
                source(self.report, name, module.__file__)

        signature = inspect.signature(type(self.layer).forward)

        def layer_input(module, args, kwargs):
            if ACTIVE.get() is self and not self.failed and not self.finished:
                self.attempt(
                    lambda: self.layer_input(
                        signature.bind(module, *args, **kwargs).arguments
                    )
                )

        def norm(module, inputs, output):
            if ACTIVE.get() is self and not self.failed and not self.finished:
                self.attempt(self.norm_boundary, inputs, output)

        def qkv(module, inputs, output):
            if ACTIVE.get() is self and self.in_prepare and not self.failed:
                self.attempt(self.qkv_boundary, inputs, output)

        self.handles.append(
            self.layer.register_forward_pre_hook(layer_input, with_kwargs=True)
        )
        self.handles.append(self.layer.input_layernorm.register_forward_hook(norm))
        self.handles.append(self.attn.qkv_proj.register_forward_hook(qkv))

    def layer_input(self, values):
        require(values.get("residual") is None, "First layer received a residual")
        batch = values["forward_batch"]
        require(
            batch.batch_size == 1 and batch.forward_mode.is_target_verify(),
            "Unexpected draft forward mode",
        )
        require(batch.input_ids.numel() <= 16, "Outside first short block scope")
        self.forward_batch = batch
        self.report["forward_mode"] = str(batch.forward_mode)
        for name, tensor in (
            ("input_ids", batch.input_ids),
            ("positions", values["positions"]),
            ("input_embedding", values["hidden_states"]),
            ("out_cache_loc", batch.out_cache_loc),
            ("req_pool_indices", batch.req_pool_indices),
            ("forward_seq_lens", batch.seq_lens),
            ("forward_seq_lens_cpu", batch.seq_lens_cpu),
        ):
            self.save(name, tensor)
        self.report["spec_draft_token_num"] = int(batch.spec_info.draft_token_num)

    def norm_boundary(self, inputs, output):
        require(
            len(inputs) == 1 and not isinstance(output, tuple),
            "Unexpected first norm contract",
        )
        self.save("norm_input", inputs[0])
        self.save("norm_output", output)
        norm = self.layer.input_layernorm
        self.record_bias("input_norm_bias", getattr(norm, "bias", None))
        # Dispatch is resolved by the ORIGINAL call before this hook. Do not
        # force a different implementation based just on the presence of bias.
        function = getattr(norm, "_forward_method", None) or norm.forward
        function = inspect.unwrap(function)
        module_name = function.__module__
        qualname = function.__qualname__
        self.report["input_norm_dispatch"] = {
            "module": module_name,
            "qualname": qualname,
        }
        source(
            self.report, "actual_input_norm_forward", inspect.getsourcefile(function)
        )
        if (
            module_name == "sglang.srt.layers.quantization.modelslim.modelslim"
            and qualname == "npu_wrapper_rmsnorm_forward.<locals>._rmsnorm_forward_oot"
        ):
            mode = "modelslim_bias_after_norm_store"
        elif getattr(norm, "bias", None) is None or (
            module_name == "sglang.srt.layers.layernorm"
            and qualname == "RMSNorm.forward_npu"
        ):
            mode = "plain_rmsnorm"
        else:
            mode = "uncovered"
        self.report["input_norm_reference_mode"] = mode

    def record_bias(self, name, tensor):
        info = {"present": tensor is not None}
        self.report.setdefault("bias", {})[name] = info
        if tensor is None:
            return
        saved = self.save(name, tensor)
        info.update(
            shape=list(saved.shape),
            dtype=str(saved.dtype),
            nonzero_count=int((saved != 0).sum().item()),
            finite=bool(saved.isfinite().all().item()),
        )

    def qkv_boundary(self, inputs, output):
        self.save("qkv_input", inputs[0])
        self.save("qkv_output", output[0])

    def proposal_input(self, proposer, values):
        require(proposer is self.worker._proposer, "Unexpected proposer")
        require(
            values["bs"] == 1
            and proposer.gamma == QUERIES
            and proposer.sample_from_anchor,
            "Expected anchor+7 masks",
        )
        self.report["embedding_is_checkpoint_local"] = (
            values["embed_module"] is self.model.embed_tokens
        )
        require(
            self.report["embedding_is_checkpoint_local"], "Unexpected embedding source"
        )
        self.report["mask_token_id"] = int(proposer._mask_token_id)
        window = values["verify_window"]
        self.save("verify_positions", window.positions_2d)
        self.save("verify_cache_loc", window.verify_cache_loc_2d)

    def fused_input(self, values, function):
        require(
            values["head_dim"] == HEAD_DIM
            and values["q_hidden_size"] == values["kv_hidden_size"] == HEADS * HEAD_DIM,
            "Unexpected fused layout",
        )
        require(values["is_neox_style"] is True, "Only actual full NeoX path covered")
        self.record_bias("q_norm_bias", values["q_bias"])
        self.record_bias("k_norm_bias", values["k_bias"])
        self.report["fused_bias_enabled"] = values["q_bias"] is not None
        self.report["fused_eps"] = float(values["eps"])
        self.report["fused_neox"] = values["is_neox_style"]
        for name, key in (
            ("fused_input", "input"),
            ("fused_sin", "sin"),
            ("fused_cos", "cos"),
            ("q_norm_weight", "q_weight"),
            ("k_norm_weight", "k_weight"),
        ):
            self.save(name, values[key])
        source(
            self.report,
            "installed_split_qkv_rmsnorm_rope",
            inspect.getsourcefile(inspect.unwrap(function)),
        )

    def fused_output(self, result):
        require(len(result) == 3, "Unexpected fused result")
        for name, value in zip(("prepared_q", "prepared_k", "prepared_v"), result):
            self.save(name, value)

    def backend_start(self, backend, values):
        require(
            values["forward_batch"] is self.forward_batch,
            "Unexpected ForwardBatch identity",
        )
        require(
            not backend.use_mla
            and not backend.is_hybrid_swa
            and not backend.graph_mode,
            "Only ordinary eager MHA FIA path covered",
        )
        require(values["save_kv_cache"], "Expected current draft KV write")
        require(
            type(backend.token_to_kv_pool).__name__
            in ("NPUMHATokenToKVPool", "MHATokenToKVPool"),
            "Unknown pool layout",
        )
        require(
            sys.getprofile() is None,
            "Existing profiler; observer will not overwrite it",
        )
        self.report["pool_class"] = type(backend.token_to_kv_pool).__name__
        self.report["backend_class"] = type(backend).__name__
        self.report["backend_branch"] = "ordinary_mha_fia_tnd"
        self.backend = backend
        for key in ("q", "k", "v"):
            self.save("backend_" + key, values[key])

    def backend_return(self, frame, result):
        values = frame.f_locals
        b, layer = values["self"], values["layer"]
        require(
            b is self.backend and layer is self.attn.attn, "Unexpected backend frame"
        )
        lengths = [int(n) for n in values["actual_seq_lengths_kv"]]
        q_lengths = [int(n) for n in values["actual_seq_lengths"]]
        require(
            len(lengths) == len(q_lengths) == 1 and 0 < lengths[0] <= 128,
            "Outside bounded single sequence scope",
        )
        self.report["actual_seq_lengths_kv"] = lengths
        self.report["actual_seq_lengths_q"] = q_lengths
        self.report["sparse_mode"] = int(values["sparse_mode"])
        self.report["mask_is_none"] = values["mask"] is None
        self.report["mask_shape"] = (
            None if values["mask"] is None else list(values["mask"].shape)
        )
        self.report["scale"] = float(layer.scaling)
        self.report["attn_type"] = str(layer.attn_type)
        self.report["sliding_window_size"] = int(layer.sliding_window_size)
        self.report["page_size"] = int(b.page_size)
        self.save("fia_query", values["query"])
        self.save("attention_output", result)
        self.save("fia_local_output", values["attn_output"])
        # This exact table is passed to the non-hybrid op (not a guessed table).
        table = self.save("block_table", b.forward_metadata.block_tables[:1])
        page = int(b.page_size)
        k_cache, v_cache = values["k_cache"], values["v_cache"]
        require(
            k_cache.is_contiguous() and v_cache.is_contiguous(),
            "Unknown strided FIA pool",
        )
        k_flat = k_cache.view(-1, HEADS, HEAD_DIM)
        v_flat = v_cache.view(-1, HEADS, HEAD_DIM)
        require(k_flat.shape == v_flat.shape and page > 0, "Incompatible pools")
        self.report["pool_capacity"] = int(k_flat.shape[0])
        pages = (lengths[0] + page - 1) // page
        require(table.ndim == 2 and pages <= table.shape[1], "Insufficient block table")
        slots = [int(table[0, i // page]) * page + i % page for i in range(lengths[0])]
        require(all(0 <= n < k_flat.shape[0] for n in slots), "FIA slot outside pool")
        self.report["actual_slots"] = slots
        self.save("pool_k", k_flat, slots)
        self.save("pool_v", v_flat, slots)
        batch = self.forward_batch
        prefix = int(cpu_copy(batch.seq_lens).item())
        width = batch.input_ids.numel()
        require(
            0 <= prefix < 128 and prefix + width <= 128,
            "Request map exceeds diagnostic bound",
        )
        req = int(cpu_copy(batch.req_pool_indices).item())
        require(
            0 <= req < b.req_to_token_pool.req_to_token.shape[0],
            "Invalid request index",
        )
        self.save(
            "logical_slots", b.req_to_token_pool.req_to_token[req, : prefix + width]
        )
        current = cpu_copy(batch.out_cache_loc).reshape(-1).tolist()
        require(
            all(0 <= n < k_flat.shape[0] for n in current), "Current write outside pool"
        )
        self.save("current_pool_k", k_flat, current)
        self.save("current_pool_v", v_flat, current)

    def finish(self, original_error=False):
        if self.finished:
            return
        self.finished = True
        for h in self.handles:
            try:
                h.remove()
            except Exception as exc:
                self.failed = True
                self.report["errors"].append(f"Cannot remove diagnostic hook: {exc}")
        if original_error:
            self.failed = True
            self.report["errors"].append(
                "Original forward raised; exception propagated unchanged"
            )
        if not self.failed:
            self.attempt(
                lambda: require(
                    "current_pool_v" in self.report["arrays"],
                    "Missing first Attention boundaries",
                )
            )
        self.report["status"] = (
            "PROPOSAL_SNAPSHOT_FAILED" if self.failed else "PROPOSAL_SNAPSHOT_COLLECTED"
        )
        self.report["saved_bytes"] = self.used
        self.report["elapsed_seconds_with_observation"] = (
            time.monotonic() - self.started
        )
        try:
            write_json(self.root / "snapshot.json", self.report)
        except Exception as exc:
            print(f"Cannot write proposal snapshot: {exc}", file=sys.stderr, flush=True)


def bind(function, *args, **kwargs):
    values = inspect.signature(function).bind(*args, **kwargs)
    values.apply_defaults()
    return values.arguments


def install_worker(module, config):
    prefill_original = module.DSparkWorkerV2._forward_prefill
    decode_original = module.DSparkWorkerV2._forward_decode
    prefills = {}

    def matches(worker, batch):
        return worker.ps.tp_rank == 0 and any(
            r.rid == config["rid"] for r in batch.reqs
        )

    @functools.wraps(prefill_original)
    def prefill(worker, batch, *args, **kwargs):
        if matches(worker, batch) and id(worker) not in prefills:
            try:
                require(
                    len(batch.reqs) == 1 and list(batch.prefix_lens) == [0],
                    "Expected single cold prefill",
                )
                require(batch.input_ids.numel() <= 128, "Fixed short request only")
                prefills[id(worker)] = {
                    "prompt_ids": cpu_copy(batch.input_ids).tolist(),
                    "prefix_lens": list(batch.prefix_lens),
                    "rid": config["rid"],
                }
            except Exception as exc:
                prefills[id(worker)] = {"error": str(exc)}
        return prefill_original(worker, batch, *args, **kwargs)

    @functools.wraps(decode_original)
    def decode(worker, batch, *args, **kwargs):
        if not matches(worker, batch):
            return decode_original(worker, batch, *args, **kwargs)
        try:
            with (Path(config["run_dir"]) / "capture.claim").open("x") as f:
                f.write(str(os.getpid()))
        except OSError:
            return decode_original(worker, batch, *args, **kwargs)
        capture = Capture(config, worker, batch, prefills.get(id(worker)))
        token = ACTIVE.set(capture)
        raised = True
        try:
            capture.attempt(capture.start)
            result = decode_original(worker, batch, *args, **kwargs)
            raised = False
            return result
        finally:
            ACTIVE.reset(token)
            capture.finish(raised)

    module.DSparkWorkerV2._forward_prefill = prefill
    module.DSparkWorkerV2._forward_decode = decode


def install_draft(module):
    original = module.DraftBlockProposer._run_forward

    @functools.wraps(original)
    def observed(proposer, *args, **kwargs):
        c = ACTIVE.get()
        if c is None or c.finished or proposer is not c.worker._proposer:
            return original(proposer, *args, **kwargs)
        c.attempt(
            lambda: c.proposal_input(
                proposer, bind(original, proposer, *args, **kwargs)
            )
        )
        raised = True
        try:
            result = original(proposer, *args, **kwargs)
            raised = False
            return result
        finally:
            c.finish(raised)

    module.DraftBlockProposer._run_forward = observed


def install_dflash(module):
    if not hasattr(module, "split_qkv_rmsnorm_rope"):
        return
    original = module.DFlashAttention.forward_prepare_npu
    fused = module.split_qkv_rmsnorm_rope

    @functools.wraps(original)
    def prepare(attn, *args, **kwargs):
        c = ACTIVE.get()
        if c is None or c.failed or c.finished or attn is not c.attn:
            return original(attn, *args, **kwargs)
        c.in_prepare = True
        try:
            return original(attn, *args, **kwargs)
        finally:
            c.in_prepare = False

    @functools.wraps(fused)
    def observe_fused(*args, **kwargs):
        c = ACTIVE.get()
        if c is None or not c.in_prepare or c.failed:
            return fused(*args, **kwargs)
        c.attempt(lambda: c.fused_input(bind(fused, *args, **kwargs), fused))
        result = fused(*args, **kwargs)
        c.attempt(c.fused_output, result)
        return result

    module.DFlashAttention.forward_prepare_npu = prepare
    module.split_qkv_rmsnorm_rope = observe_fused


def install_backend(module):
    original = module.AscendAttnBackend.forward_mtp

    @functools.wraps(original)
    def observed(backend, *args, **kwargs):
        c = ACTIVE.get()
        if c is None or c.failed or c.finished:
            return original(backend, *args, **kwargs)
        values = c.attempt(lambda: bind(original, backend, *args, **kwargs))
        if values is None or values["layer"] is not c.attn.attn:
            return original(backend, *args, **kwargs)
        c.attempt(c.backend_start, backend, values)
        if c.failed:
            return original(backend, *args, **kwargs)

        def returned(frame, event, result):
            if (
                event == "return"
                and frame.f_code is original.__code__
                and result is not None
            ):
                c.attempt(c.backend_return, frame, result)

        sys.setprofile(returned)
        try:
            return original(backend, *args, **kwargs)
        finally:
            sys.setprofile(None)

    module.AscendAttnBackend.forward_mtp = observed


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, config):
        self.callbacks = {
            WORKER: lambda m: install_worker(m, config),
            DRAFT: install_draft,
            DFLASH: install_dflash,
            BACKEND: install_backend,
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
    if _INSTALLED or not os.environ.get(ENV):
        return
    finder = Finder(json.loads(Path(os.environ[ENV]).read_text()))
    require(
        not any(name in sys.modules for name in finder.callbacks),
        "Install observer before framework imports",
    )
    sys.meta_path.insert(0, finder)
    _INSTALLED = True
