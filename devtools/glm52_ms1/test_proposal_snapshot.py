"""CPU calibration of references and actual source observation; no NPU PASS."""

# ruff: noqa: E402 -- import this directory's standalone diagnostic tools.

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dspark_attention_reference as ref
import dspark_proposal_snapshot as obs
import probe_proposal_snapshot as probe
import with_proposal_snapshot as launch


def method(path, cls, name, namespace):
    """Execute the CURRENT source body in a small CPU harness, not its imitation."""
    tree = ast.parse(path.read_text())
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(ast.Module(body=future + [fn], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def tensorfile(path, entries):
    header, body = {}, bytearray()
    for key, tensor in entries.items():
        raw = tensor.contiguous().view(torch.uint16).numpy().tobytes()
        header[key] = dict(
            dtype="BF16",
            shape=list(tensor.shape),
            data_offsets=[len(body), len(body) + len(raw)],
        )
        body.extend(raw)
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + body)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for key, val in dict(HIDDEN=8, HEAD_DIM=4, HEADS=2, QUERIES=8).items():
        monkeypatch.setattr(obs, key, val)
    monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", "static")
    monkeypatch.setenv("SGLANG_NPU_GLM_DSPARK_QUAROT", "original")
    torch.manual_seed(42)
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    config = dict(
        run_id="fixture",
        run_dir=str(tmp_path),
        rid="fixture-rid",
        cache_salt="salt",
        max_bytes=64 * 1024**2,
        tool_hashes={
            name: obs.file_hash(HERE / name)
            for name in (
                "dspark_proposal_snapshot.py",
                "dspark_attention_reference.py",
                "dspark_local_reference.py",
                "probe_proposal_snapshot.py",
            )
        },
    )
    obs.write_json(tmp_path / "config.json", config)
    counts = dict(
        prefill=0, decode=0, proposal=0, prepare=0, qkv=0, fused=0, fia=0, o_proj=0
    )
    return_value = object()

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.linspace(0.8, 1.2, dim).bfloat16())
            self.variance_epsilon = 1e-5

        def forward(self, x):
            return F.rms_norm(
                x.float(), (x.shape[-1],), self.weight.float(), self.variance_epsilon
            ).bfloat16()

    class Projection(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight = torch.nn.Parameter(w)
            self.bias = None

        def forward(self, x):
            counts["qkv"] += 1
            return F.linear(x.float(), self.weight.float()).bfloat16(), None

    class NPUMHATokenToKVPool:
        def __init__(self):
            self.k = torch.randn(16, 4, 8).bfloat16()
            self.v = torch.randn(16, 4, 8).bfloat16()

        def set_kv_buffer(self, layer, loc, k, v):
            self.k.view(-1, 2, 4)[loc.normal] = k.reshape(-1, 2, 4)
            self.v.view(-1, 2, 4)[loc.normal] = v.reshape(-1, 2, 4)

        def get_key_buffer(self, layer):
            return self.k

        def get_value_buffer(self, layer):
            return self.v

    pool = NPUMHATokenToKVPool()
    request_pool = NS(req_to_token=torch.arange(4, 68).view(1, -1))
    backend_module = types.ModuleType(obs.BACKEND)
    backend_module.__file__ = str(HERE / "test_proposal_snapshot.py")
    for name, module in ((obs.BACKEND, backend_module),):
        monkeypatch.setitem(sys.modules, name, module)

    def fia(q, k, v, **kw):
        counts["fia"] += 1
        # Independent PyTorch SDPA oracle, with an independently expanded table.
        slots = [
            int(kw["block_table"][0, i // kw["block_size"]]) * kw["block_size"]
            + i % kw["block_size"]
            for i in range(kw["actual_seq_lengths_kv"][0])
        ]
        actual_k = k.view(-1, 2, 4)[slots].float().transpose(0, 1).unsqueeze(0)
        actual_v = v.view(-1, 2, 4)[slots].float().transpose(0, 1).unsqueeze(0)
        actual_q = q.float().transpose(0, 1).unsqueeze(0)
        output = (
            F.scaled_dot_product_attention(
                actual_q, actual_k, actual_v, scale=kw["scale"]
            )
            .squeeze(0)
            .transpose(0, 1)
            .contiguous()
            .bfloat16()
        )
        return output, None

    ns = dict(
        torch=NS(ops=NS(npu=NS(npu_fused_infer_attention_score=fia)), cat=torch.cat),
        np=np,
        KVWriteLoc=lambda normal, swa: NS(normal=normal),
        AttentionType=NS(ENCODER_ONLY="encoder_only"),
    )
    source = (
        HERE.parents[1]
        / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    )
    backend_module.AscendAttnBackend = type(
        "AscendAttnBackend",
        (),
        {"forward_mtp": method(source, "AscendAttnBackend", "forward_mtp", ns)},
    )
    backend = backend_module.AscendAttnBackend()
    backend.use_mla = backend.is_hybrid_swa = backend.graph_mode = False
    backend.page_size = 4
    backend.speculative_num_draft_tokens = 8
    backend.token_to_kv_pool = pool
    backend.req_to_token_pool = request_pool
    backend.forward_metadata = NS(
        swa_out_cache_loc=None,
        seq_lens_cpu_int=torch.tensor([11]),
        block_tables=torch.tensor([[1, 2, 3, 4]]),
    )
    obs.install_backend(backend_module)

    class Radix(torch.nn.Module):
        layer_id = 0
        tp_q_head_num = tp_k_head_num = tp_v_head_num = 2
        qk_head_dim = v_head_dim = 4
        sliding_window_size = -1
        scaling = 0.5
        attn_type = "encoder_only"

        def forward(self, q, k, v, batch):
            return backend.forward_mtp(
                q, k.reshape(-1, 2, 4), v.reshape(-1, 2, 4), self, batch, True
            )

    class RotaryEmbedding:
        base = 8000000.0
        rotary_dim = 4

        def get_cos_sin_with_position(self, positions):
            freq = 1 / (self.base ** (torch.arange(0, 4, 2).float() / 4))
            angles = positions.float()[:, None] * freq[None]
            self.position_cos = (
                torch.cat([angles.cos(), angles.cos()], -1)
                .bfloat16()
                .reshape(-1, 1, 1, 4)
            )
            self.position_sin = (
                torch.cat([angles.sin(), angles.sin()], -1)
                .bfloat16()
                .reshape(-1, 1, 1, 4)
            )

    def fused(
        input,
        sin,
        cos,
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        eps=None,
        q_weight=None,
        k_weight=None,
        q_bias=None,
        k_bias=None,
        is_neox_style=True,
    ):
        counts["fused"] += 1
        q, k, v = input.float().split(8, dim=-1)
        c, s = cos.float().view(-1, 1, 4), sin.float().view(-1, 1, 4)

        def prep(x, w):
            x = F.rms_norm(x.reshape(-1, 2, 4), (4,), w.float(), eps)
            rot = torch.cat((-x[..., 2:], x[..., :2]), -1)
            return (rot * s + x * c).flatten(1).bfloat16()

        return prep(q, q_weight), prep(k, k_weight), v.bfloat16()

    dflash_module = types.ModuleType(obs.DFLASH)
    dflash_module.__file__ = str(HERE / "test_proposal_snapshot.py")
    dflash_module.split_qkv_rmsnorm_rope = fused
    # The actual source function resolves the wrapper through this module dict.
    dflash_module.__dict__.update(torch=torch, _is_npu=True)
    source = HERE.parents[1] / "python/sglang/srt/models/dflash.py"

    class DFlashAttention(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.q_size = self.kv_size = 8
            self.head_dim = 4
            self.qkv_proj = Projection(weight)
            self.q_norm = RMSNorm(4)
            self.k_norm = RMSNorm(4)
            self.rotary_emb = RotaryEmbedding()
            self.attn = Radix()
            self.attention_sink_bias = None

        def o_proj(self, value):
            counts["o_proj"] += 1
            return value, None

        def apply_attention_output(self, out, hidden):
            return out

    DFlashAttention.forward_prepare_npu = method(
        source, "DFlashAttention", "forward_prepare_npu", dflash_module.__dict__
    )
    DFlashAttention.forward = method(
        source, "DFlashAttention", "forward", dflash_module.__dict__
    )
    dflash_module.DFlashAttention = DFlashAttention
    monkeypatch.setitem(sys.modules, obs.DFLASH, dflash_module)
    obs.install_dflash(dflash_module)
    q, k, v = [torch.randn(128, 8).bfloat16() for _ in range(3)]
    embedding = torch.randn(32, 8).bfloat16()
    attn = DFlashAttention(torch.cat([x[:8] for x in (q, k, v)]))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = attn
            self.input_layernorm = RMSNorm(8)
            self.attention_conv = None

        def forward(self, positions, hidden_states, forward_batch, residual):
            norm = self.input_layernorm(hidden_states)
            return self.self_attn(positions, norm, forward_batch)

    model = type("DSparkDraftModel", (torch.nn.Module,), {})()
    model.layers = torch.nn.ModuleList([Layer()])
    model.embed_tokens = torch.nn.Embedding.from_pretrained(embedding)
    entries = {"embed_tokens.weight": embedding}
    entries.update(
        {
            f"layers.0.self_attn.{label}_proj.weight": x
            for label, x in zip(("q", "k", "v"), (q, k, v))
        }
    )
    tensorfile(weights / "model.safetensors", entries)
    draft_module = types.ModuleType(obs.DRAFT)
    draft_module.__file__ = str(HERE / "test_proposal_snapshot.py")

    class DraftBlockProposer:
        gamma = query_token_num = 8
        sample_from_anchor = True
        _mask_token_id = 31

        def _run_forward(
            self,
            *,
            batch,
            draft_input,
            verify_window,
            bs,
            device,
            embed_module,
            draft_sampler=None,
            sampling_info=None,
        ):
            counts["proposal"] += 1
            ids = torch.full((8,), self._mask_token_id)
            ids[0] = draft_input.bonus_tokens[0]
            fb = NS(
                batch_size=1,
                forward_mode=NS(
                    is_target_verify=lambda: True, is_draft_extend_v2=lambda: False
                ),
                input_ids=ids,
                positions=verify_window.positions_2d[0, :8],
                out_cache_loc=verify_window.verify_cache_loc_2d[0, :8],
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                seq_lens_cpu=batch.seq_lens_cpu + 8,
                spec_info=NS(draft_token_num=8),
                global_num_token_non_padded_cpu=8,
            )
            self.draft_model.layers[0](fb.positions, embed_module(ids), fb, None)
            return return_value

    draft_module.DraftBlockProposer = DraftBlockProposer
    monkeypatch.setitem(sys.modules, obs.DRAFT, draft_module)
    obs.install_draft(draft_module)
    proposer = DraftBlockProposer()
    proposer.draft_model = model
    worker_module = types.ModuleType(obs.WORKER)
    worker_module.__file__ = str(HERE / "test_proposal_snapshot.py")

    class DSparkWorkerV2:
        def _forward_prefill(self, batch):
            counts["prefill"] += 1
            return return_value

        def _forward_decode(self, batch, fail=None):
            counts["decode"] += 1
            if fail:
                raise fail
            return self._proposer._run_forward(
                batch=batch,
                draft_input=batch.spec_info,
                verify_window=NS(
                    positions_2d=torch.arange(3, 12).reshape(1, -1),
                    verify_cache_loc_2d=torch.arange(7, 16).reshape(1, -1),
                ),
                bs=1,
                device="cpu",
                embed_module=model.embed_tokens,
            )

    worker_module.DSparkWorkerV2 = DSparkWorkerV2
    monkeypatch.setitem(sys.modules, obs.WORKER, worker_module)
    obs.install_worker(worker_module, config)
    worker = DSparkWorkerV2()
    worker.ps = NS(tp_rank=0)
    worker.draft_model = model
    worker._draft_is_moe = False
    worker._proposer = proposer
    worker.model_runner = NS(model=type("GlmMoeDsaForCausalLM", (), {})())
    worker.server_args = NS(
        device="npu",
        speculative_algorithm="DSPARK",
        tp_size=16,
        dp_size=1,
        nnodes=1,
        pp_size=1,
        disable_cuda_graph=True,
        speculative_draft_model_path=str(weights),
        model_path=str(weights),
    )
    batch = NS(
        reqs=[NS(rid=config["rid"])],
        input_ids=torch.tensor([1, 2, 3]),
        prefix_lens=[0],
        seq_lens=torch.tensor([3]),
        seq_lens_cpu=torch.tensor([3]),
        spec_info=NS(bonus_tokens=torch.tensor([7])),
        req_pool_indices=torch.tensor([0]),
        forward_iter=5,
    )
    return NS(
        root=tmp_path,
        config=config,
        worker=worker,
        batch=batch,
        model=model,
        backend=backend,
        pool=pool,
        counts=counts,
        output=return_value,
        weights=weights,
    )


def responses(r):
    response = dict(
        id=r.config["rid"],
        choices=[
            dict(
                finish_reason="length",
                meta_info=dict(cached_tokens=0),
                prompt_token_ids=[1, 2, 3],
                response_token_ids=[7],
            )
        ],
    )
    obs.write_json(
        r.root / "request.json",
        dict(
            rid=r.config["rid"],
            cache_salt=r.config["cache_salt"],
            messages=[dict(role="user", content=probe.PROMPT)],
            temperature=0,
            max_tokens=64,
        ),
    )
    obs.write_json(r.root / "response.json", response)
    return response


def run(r):
    assert r.worker._forward_prefill(r.batch) is r.output
    assert r.worker._forward_decode(r.batch) is r.output
    responses(r)
    return json.loads((r.root / "snapshot.json").read_text())


def test_actual_backend_and_prepare_source_roundtrip(runtime):
    r = runtime
    snap = run(r)
    assert snap["status"] == "PROPOSAL_SNAPSHOT_COLLECTED", snap["errors"]
    assert r.counts["fia"] == r.counts["fused"] == r.counts["proposal"] == 1
    assert r.counts["qkv"] == 2  # The unused outer projection is preserved.
    assert sys.getprofile() is None and obs.ACTIVE.get() is None
    assert not r.model.layers[0]._forward_pre_hooks
    result = probe.compare_snapshot(r.root)
    assert result["structural_issues"] == []
    assert result["comparisons"]["checkpoint_qkv_shard_vs_runtime_weight"][
        "exact_equal"
    ]
    assert result["comparisons"]["checkpoint_embedding_vs_layer_input"]["exact_equal"]
    assert result["attention_replay"] == "SAME_ACTUAL_INPUTS_COMPARED"
    # A second model decode still runs, but never records another observation.
    old = (r.root / "snapshot.json").read_bytes()
    assert r.worker._forward_decode(r.batch) is r.output
    assert (r.root / "snapshot.json").read_bytes() == old
    assert r.counts["decode"] == 2 and r.counts["fia"] == 2


@pytest.mark.parametrize("different", ["rid", "rank"])
def test_unmatched_scope_never_observes(runtime, different):
    r = runtime
    if different == "rid":
        r.batch.reqs[0].rid = "other"
    else:
        r.worker.ps.tp_rank = 3
    r.worker._forward_prefill(r.batch)
    assert r.worker._forward_decode(r.batch) is r.output
    assert not (r.root / "snapshot.json").exists()
    assert r.counts["fia"] == 1


@pytest.mark.parametrize(
    "bad", ["budget", "missing_prefill", "profiler", "unknown_pool"]
)
def test_diagnostic_failure_keeps_original_and_cleans_up(runtime, bad):
    r = runtime
    if bad != "missing_prefill":
        r.worker._forward_prefill(r.batch)
    if bad == "budget":
        r.config["max_bytes"] = 1
    if bad == "unknown_pool":
        r.pool.__class__ = type("OtherPool", (type(r.pool),), {})

    def existing(frame, event, arg):
        return None

    if bad == "profiler":
        sys.setprofile(existing)
    try:
        assert r.worker._forward_decode(r.batch) is r.output
        if bad == "profiler":
            assert sys.getprofile() is existing
    finally:
        sys.setprofile(None)
    snap = json.loads((r.root / "snapshot.json").read_text())
    assert snap["status"] == "PROPOSAL_SNAPSHOT_FAILED"
    assert r.counts["decode"] == r.counts["fia"] == 1
    assert obs.ACTIVE.get() is None
    assert not r.model.layers[0]._forward_pre_hooks
    assert not r.model.layers[0].self_attn.qkv_proj._forward_hooks


def test_original_exception_identity_preserved(runtime):
    r = runtime
    r.worker._forward_prefill(r.batch)
    error = RuntimeError("original error")
    with pytest.raises(RuntimeError) as exc:
        r.worker._forward_decode(r.batch, error)
    assert exc.value is error and r.counts["decode"] == 1
    assert obs.ACTIVE.get() is None
    assert not r.model.layers[0]._forward_pre_hooks


@pytest.mark.parametrize("change", ["hash", "rid", "cache", "prompt", "tools"])
def test_replay_rejects_misattributed_or_changed_data(runtime, change):
    r = runtime
    run(r)
    response = responses(r)
    if change == "hash":
        with (r.root / "fia_query.npy").open("ab") as f:
            f.write(b"bad")
    elif change == "rid":
        response["id"] = "wrong"
    elif change == "cache":
        response["choices"][0]["meta_info"]["cached_tokens"] = 1
    elif change == "prompt":
        response["choices"][0]["prompt_token_ids"][0] = 9
    else:
        r.config["tool_hashes"]["probe_proposal_snapshot.py"] = "bad"
        obs.write_json(r.root / "config.json", r.config)
    obs.write_json(r.root / "response.json", response)
    with pytest.raises(ValueError):
        probe.compare_snapshot(r.root)


def test_wrong_actual_page_order_is_reported(runtime):
    r = runtime
    r.backend.forward_metadata.block_tables[0, :3] = torch.tensor([2, 1, 3])
    run(r)
    result = probe.compare_snapshot(r.root)
    assert "same_ordered_slots" in result["structural_issues"]
    assert result["attention_replay"] == "SAME_ACTUAL_INPUTS_COMPARED"
    assert (
        "composed_preparation_actual_context_attention_bf16_storage"
        not in result["comparisons"]
    )


def test_changed_loaded_weight_is_found_from_checkpoint(runtime):
    r = runtime
    r.model.layers[0].self_attn.qkv_proj.weight.data.zero_()
    run(r)
    result = probe.compare_snapshot(r.root)
    assert not result["comparisons"]["checkpoint_qkv_shard_vs_runtime_weight"][
        "exact_equal"
    ]


def test_saved_pool_is_independent_of_later_mutation(runtime):
    r = runtime
    snap = run(r)
    before = probe.Arrays(r.root, snap, r.config["max_bytes"]).get("pool_k")
    r.pool.k.zero_()
    after = probe.Arrays(r.root, snap, r.config["max_bytes"]).get("pool_k")
    np.testing.assert_array_equal(before, after)


def test_client_failure_never_retries_request(runtime, monkeypatch):
    r = runtime
    posts = []

    def request(opener, url, payload=None, timeout=0):
        if payload is None:
            return dict(
                device="npu",
                speculative_algorithm="DSPARK",
                tp_size=16,
                dp_size=1,
                nnodes=1,
                disable_cuda_graph=True,
            )
        posts.append(payload)
        raise TimeoutError("uncertain HTTP result")

    monkeypatch.setattr(probe, "request_json", request)
    with pytest.raises(TimeoutError):
        probe.collect(r.root, "http://localhost:8810")
    with pytest.raises(FileExistsError):
        probe.collect(r.root, "http://localhost:8810")
    assert len(posts) == 1
    assert posts[0]["max_tokens"] == 64 and "return_logprob" not in posts[0]


def test_launcher_private_bootstrap_and_disabled_import(tmp_path, monkeypatch):
    for key in (obs.ENV, "GLM52_CONTEXT_SNAPSHOT_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(launch.importlib.util, "find_spec", lambda name: None)
    config, env = launch.prepare(tmp_path, HERE)
    boot = Path(config["run_dir"]) / "bootstrap"
    assert (boot / "dspark_proposal_snapshot.py").exists()
    assert (boot / "dspark_context_snapshot.py").exists()
    assert env["SGLANG_NPU_GLM_DSPARK_QUAROT"] == "original"
    assert env["SGLANG_ENABLE_FAST_INPUT_LOGPROBS"] == "0"
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(HERE)
    p = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dspark_proposal_snapshot as d; d.install(); assert not any(n.startswith(('torch','triton','sglang')) for n in sys.modules)",
        ],
        env=child_env,
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("seed", range(3))
def test_attention_independent_torch_sdpa(seed):
    rng = np.random.default_rng(seed)
    q, k, v = (rng.normal(size=s) for s in ((8, 2, 192), (31, 2, 192), (31, 2, 192)))
    computed = ref.attention(q, k, v, 192**-0.5)
    args = [torch.from_numpy(a).permute(1, 0, 2).unsqueeze(0) for a in (q, k, v)]
    expected = (
        F.scaled_dot_product_attention(*args, scale=192**-0.5)
        .squeeze(0)
        .permute(1, 0, 2)
    )
    # PyTorch's library default float64 assert_close contract, not a model gate.
    torch.testing.assert_close(torch.from_numpy(computed), expected)


def test_uniform_full_and_causal_visibility_are_distinct():
    q = np.zeros((2, 1, 2))
    k = np.zeros((3, 1, 2))
    v = np.array([[[0.0, 0.0]], [[2.0, 4.0]], [[4.0, 8.0]]])
    np.testing.assert_array_equal(ref.attention(q, k, v, 1), [[[2, 4]], [[2, 4]]])
    visible = np.array([[True, True, False], [True, True, True]])
    np.testing.assert_array_equal(
        ref.attention(q, k, v, 1, visible=visible), [[[1, 2]], [[2, 4]]]
    )


@pytest.mark.parametrize(
    "kind", ["width9", "missing_context", "duplicate", "reordered"]
)
def test_slot_contract_detects_wrong_geometry(kind):
    expected = np.arange(31)
    actual = expected.copy()
    if kind == "width9":
        actual = np.arange(32)
    elif kind == "missing_context":
        actual = actual[23:]
    elif kind == "duplicate":
        actual[4] = actual[3]
    else:
        actual[[0, 1]] = actual[[1, 0]]
    result = ref.slot_contract(expected, actual, expected[23:], 23)
    assert not result["same_ordered_slots"]


@pytest.mark.parametrize(
    "blocks,page,length,capacity",
    [([0], 4, 8, 32), ([-1], 4, 1, 32), ([8], 4, 1, 32), ([0], 0, 1, 32)],
)
def test_paged_slots_refuses_unsafe_or_incomplete_tables(
    blocks, page, length, capacity
):
    with pytest.raises(ValueError):
        ref.paged_slots(blocks, page, length, capacity)


def test_fused_native192_reference_against_independent_torch():
    generator = torch.Generator().manual_seed(43)
    x = torch.randn(8, 3 * 4 * 192, generator=generator).bfloat16()
    qw, kw = [torch.randn(192, generator=generator).bfloat16() for _ in range(2)]
    angles = torch.randn(8, 96, generator=generator)
    cos = torch.cat((angles.cos(), angles.cos()), -1).bfloat16()
    sin = torch.cat((angles.sin(), angles.sin()), -1).bfloat16()
    computed = ref.fused_prepare(
        x.float().numpy(),
        qw.float().numpy(),
        kw.float().numpy(),
        cos.float().numpy(),
        sin.float().numpy(),
        1e-5,
        4,
        192,
        bf16_storage=True,
    )
    q, k, v = x.float().split(4 * 192, dim=-1)

    def independent(value, weight):
        norm = F.rms_norm(value.reshape(8, 4, 192), (192,), weight.float(), 1e-5)
        a, b = norm.chunk(2, -1)
        c, s = cos.float()[:, None, :96], sin.float()[:, None, :96]
        return torch.cat((a * c - b * s, b * c + a * s), -1).bfloat16()

    # Library-default BF16 comparison is only calibration, never model acceptance.
    for actual, expected in zip(
        computed,
        (independent(q, qw), independent(k, kw), v.reshape(8, 4, 192).bfloat16()),
    ):
        torch.testing.assert_close(torch.from_numpy(actual).bfloat16(), expected)


def test_exception_inside_actual_backend_restores_profiler(runtime, monkeypatch):
    r = runtime
    r.worker._forward_prefill(r.batch)
    error = RuntimeError("FIA original failure")
    original = r.backend.forward_mtp.__wrapped__
    op = original.__globals__["torch"].ops.npu

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(op, "npu_fused_infer_attention_score", fail)
    with pytest.raises(RuntimeError) as exc:
        r.worker._forward_decode(r.batch)
    assert exc.value is error
    assert sys.getprofile() is None and obs.ACTIVE.get() is None
    assert not r.model.layers[0]._forward_pre_hooks
    snap = json.loads((r.root / "snapshot.json").read_text())
    assert snap["status"] == "PROPOSAL_SNAPSHOT_FAILED"


def test_disabled_startup_does_not_install_hooks(monkeypatch):
    monkeypatch.delenv(obs.ENV, raising=False)
    before = list(sys.meta_path)
    obs.install()
    assert sys.meta_path == before


def test_client_one_successful_request_and_offline_replay(runtime, monkeypatch):
    r = runtime
    calls = []

    def http(opener, url, payload=None, timeout=0):
        calls.append(url)
        if payload is None:
            return dict(
                device="npu",
                speculative_algorithm="DSPARK",
                tp_size=16,
                dp_size=1,
                nnodes=1,
                disable_cuda_graph=True,
            )
        run(r)
        return responses(r)

    monkeypatch.setattr(probe, "request_json", http)
    result = probe.collect(r.root, "http://localhost:8810")
    assert result["status"] == "PROPOSAL_COMPARISON_COLLECTED" and len(calls) == 2
    replay = probe.compare_snapshot(r.root)
    assert replay == result and len(calls) == 2


def test_budget_failure_after_hook_install_does_not_leave_hooks(runtime, monkeypatch):
    r = runtime
    original = obs.Capture.start

    def exhaust_after_hooks(capture):
        original(capture)
        assert capture.handles
        capture.config["max_bytes"] = capture.used

    monkeypatch.setattr(obs.Capture, "start", exhaust_after_hooks)
    run(r)
    snap = json.loads((r.root / "snapshot.json").read_text())
    assert snap["status"] == "PROPOSAL_SNAPSHOT_FAILED"
    assert r.counts["fia"] == 1 and r.counts["fused"] == 1
    assert not r.model.layers[0]._forward_pre_hooks
    assert not r.model.layers[0].input_layernorm._forward_hooks
