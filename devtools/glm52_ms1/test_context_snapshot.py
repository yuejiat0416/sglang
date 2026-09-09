"""CPU observer lifecycle, fixture replay and startup isolation tests; no NPU PASS."""

import ast
import importlib
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
import dspark_context_snapshot as obs
import probe_context_snapshot as probe
import with_context_snapshot as launch


class RMSNorm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.75, 1.25, width).bfloat16())
        self.variance_epsilon = 1e-5

    def forward(self, value):
        return F.rms_norm(
            value.float(),
            (value.shape[-1],),
            self.weight.float(),
            self.variance_epsilon,
        ).bfloat16()


@pytest.fixture
def runtime(tmp_path, monkeypatch, request):
    layout = getattr(request, "param", "fused")
    for name, val in dict(
        HIDDEN=8, FEATURES=3, LAYERS=2, HEAD_DIM=4, LOCAL_HEADS=2
    ).items():
        monkeypatch.setattr(obs, name, val)
    monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", "static")
    monkeypatch.setenv("SGLANG_NPU_GLM_DSPARK_QUAROT", "original")
    (tmp_path / "config.json").write_text("{}")
    config = {
        "run_dir": str(tmp_path),
        "run_id": "fixture-run",
        "rid": "fixture-rid",
        "cache_salt": "fixture-salt",
        "max_bytes": 1024**3,
        "observer_sha256": obs.file_hash(obs.__file__),
        "client_sha256": obs.file_hash(probe.__file__),
        "reference_sha256": obs.file_hash(HERE / "dspark_local_reference.py"),
    }
    obs.write_json(tmp_path / "config.json", config)
    calls = {"worker": 0, "write": 0, "fused": 0}
    output = object()
    fused_module = types.ModuleType(obs.FUSED)
    fused_module.__file__ = str(HERE / "test_context_snapshot.py")

    def fused(
        kv,
        meta,
        k_norm_weights,
        cos_sin_cache,
        positions,
        locs,
        num_layers,
        kv_size,
        head_dim,
        eps,
        commit_lens=None,
        locs_row_width=None,
        is_neox_style=True,
    ):
        calls["fused"] += 1
        assert commit_lens is None
        for i, (kb, vb) in enumerate(zip(pool.k_buffer, pool.v_buffer)):
            chunk = kv[:, i * 2 * kv_size : (i + 1) * 2 * kv_size]
            k, v = chunk.split(kv_size, dim=-1)
            k, v = k.reshape(-1, 2, 4), v.reshape(-1, 2, 4)
            norm = (
                F.rms_norm(k.float(), (4,), k_norm_weights[i].float(), eps)
                .bfloat16()
                .float()
            )
            c, s = cos_sin_cache[positions].float().chunk(2, -1)
            rotated = torch.empty_like(norm)
            for j in range(2):
                a, b = (j, j + 2) if is_neox_style else (2 * j, 2 * j + 1)
                rotated[:, :, a] = (
                    norm[:, :, a] * c[:, j : j + 1] - norm[:, :, b] * s[:, j : j + 1]
                )
                rotated[:, :, b] = (
                    norm[:, :, b] * c[:, j : j + 1] + norm[:, :, a] * s[:, j : j + 1]
                )
            kb[locs] = rotated.bfloat16()
            vb[locs] = v

    fused_module.fused_kv_norm_rope_write = fused
    monkeypatch.setitem(sys.modules, obs.FUSED, fused_module)
    obs.install_fused(fused_module)

    class DSparkDraftMixin:
        def write_target_hidden_kv(
            self,
            *,
            target_hidden,
            pool,
            positions,
            cache_loc,
            cache_loc_2d=None,
            commit_lens=None,
            target_hidden_is_projected=False,
        ):
            calls["write"] += 1
            ctx = self.hidden_norm(self.fc(target_hidden))
            layers = [layer.self_attn for layer in self.layers]
            w = torch.cat([a.qkv_proj.weight[8:] for a in layers])
            knw = torch.stack([a.k_norm.weight for a in layers])
            if layout != "fused":
                k, v = self._project_ctx_kv_stacked(
                    ctx_hidden=ctx,
                    positions=positions,
                    stacked={
                        "weight": w,
                        "bias": None,
                        "k_norm_weight": knw.float(),
                        "eps": 1e-5,
                    },
                )
                for i in range(2):
                    pool.k_buffer[i].view(-1, 2, 4)[cache_loc] = k[i]
                    pool.v_buffer[i].view(-1, 2, 4)[cache_loc] = v[i]
                return output
            meta = torch.tensor(
                [
                    [k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0)]
                    for k, v in zip(pool.k_buffer, pool.v_buffer)
                ]
            )
            cache = layers[0].rotary_emb.cos_sin_cache
            self._fused_kv_write_cache = (
                id(pool),
                (w, meta, knw, cache, 1e-5, 2, 8, 4),
            )
            kv = F.linear(ctx, w)
            fused_module.fused_kv_norm_rope_write(
                kv, meta, knw, cache, positions, cache_loc, 2, 8, 4, 1e-5
            )
            return output

    # Execute the unchanged repository method on tiny CPU fixtures. This checks
    # frame-local observation against its actual code, without importing SGLang.
    source_path = HERE.parents[1] / "python/sglang/srt/models/dspark.py"
    tree = ast.parse(source_path.read_text())
    mixin = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DSparkDraftMixin"
    )
    method = next(
        n
        for n in mixin.body
        if isinstance(n, ast.FunctionDef) and n.name == "_project_ctx_kv_stacked"
    )
    namespace = {"torch": torch, "F": F, "Tuple": tuple}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    DSparkDraftMixin._project_ctx_kv_stacked = namespace["_project_ctx_kv_stacked"]

    class RotaryEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_dim, self.base, self.is_neox_style = 4, 8000000, True
            angles = torch.arange(8).float()[:, None] / torch.tensor([1, 8000000**0.5])
            self.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()

        def forward(self, positions, query, key):
            k = key.reshape(key.shape[0], -1, 4).float()
            c, s = self.cos_sin_cache[positions].float().chunk(2, -1)
            a, b = k[..., :2], k[..., 2:]
            result = torch.cat(
                (a * c[:, None] - b * s[:, None], b * c[:, None] + a * s[:, None]), -1
            )
            return query, result.bfloat16().reshape(key.shape)

    class DSparkDraftModel(DSparkDraftMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(24, 8, bias=False, dtype=torch.bfloat16)
            self.hidden_norm = RMSNorm(8)
            self.layers = []
            for i in range(2):
                rotary = RotaryEmbedding()
                self.layers.append(
                    NS(
                        self_attn=NS(
                            head_dim=4,
                            num_kv_heads=2,
                            q_size=8,
                            kv_size=8,
                            qkv_proj=torch.nn.Linear(
                                8, 24, bias=False, dtype=torch.bfloat16
                            ),
                            k_norm=RMSNorm(4),
                            rotary_emb=rotary,
                            attn=NS(layer_id=2 + i, k_scale=None, v_scale=None),
                        )
                    )
                )

    model_module = NS(DSparkDraftMixin=DSparkDraftMixin)
    obs.install_model(model_module)
    torch.manual_seed(14)
    model = DSparkDraftModel()
    pool = type(
        "MHATokenToKVPool" if layout == "fused" else "NPUMHATokenToKVPool", (), {}
    )()
    pool.is_quantized_kv_cache, pool.store_dtype = False, torch.bfloat16
    pool.layer_transfer_counter, pool.start_layer = None, 2
    pool.k_buffer = [torch.zeros(10, 2, 4, dtype=torch.bfloat16) for _ in range(2)]
    pool.v_buffer = [torch.zeros_like(k) for k in pool.k_buffer]
    if layout != "fused":
        shape = (2, 5, 2, 2, 4) if layout == "paged" else (2, 10, 1, 2, 4)
        pool.k_buffer = torch.zeros(shape, dtype=torch.bfloat16)
        pool.v_buffer = torch.zeros_like(pool.k_buffer)
    h = torch.randn(5, 24).bfloat16()
    positions, locs = torch.arange(5), torch.tensor([3, 5, 6, 8, 9])
    batch = NS(
        reqs=[NS(rid=config["rid"])],
        prefix_lens=[0],
        extend_lens=[5],
        forward_iter=12,
        input_ids=torch.tensor([1, 17, 6, 9, 2]),
    )

    class DSparkWorkerV2:
        def _forward_prefill(self, b, callback=None):
            calls["worker"] += 1
            result = model.write_target_hidden_kv(
                target_hidden=h, pool=pool, positions=positions, cache_loc=locs
            )
            if callback:
                callback()
            assert result is output
            return output

    worker = DSparkWorkerV2()
    worker.ps = NS(tp_rank=0)
    worker.server_args = NS(
        device="npu",
        speculative_algorithm="DSPARK",
        tp_size=16,
        dp_size=1,
        nnodes=1,
        pp_size=1,
        disable_cuda_graph=True,
        model_path=str(tmp_path),
        speculative_draft_model_path=str(tmp_path),
    )
    worker.draft_model = model
    worker._draft_is_moe = worker._target_hidden_projection_enabled = False
    worker.model_runner = NS(model=type("GlmMoeDsaForCausalLM", (), {})())
    obs.install_worker(NS(DSparkWorkerV2=DSparkWorkerV2), config)
    return NS(
        root=tmp_path,
        config=config,
        worker=worker,
        batch=batch,
        h=h,
        pool=pool,
        model=model,
        calls=calls,
        output=output,
        locs=locs,
        fused=fused_module,
    )


def response_files(r):
    payload = {
        "rid": r.config["rid"],
        "cache_salt": r.config["cache_salt"],
        "messages": [{"role": "user", "content": probe.PROMPT}],
        "temperature": 0,
        "max_tokens": 64,
    }
    response = {
        "id": r.config["rid"],
        "choices": [
            {
                "finish_reason": "length",
                "prompt_token_ids": r.batch.input_ids.tolist(),
                "meta_info": {"cached_tokens": 0},
            }
        ],
    }
    obs.write_json(r.root / "request.json", payload)
    obs.write_json(r.root / "response.json", response)
    return response


def test_real_boundaries_are_copied_once_and_replayed(runtime):
    r = runtime
    assert r.worker._forward_prefill(r.batch) is r.output
    assert r.calls == {"worker": 1, "write": 1, "fused": 1}
    snapshot = json.loads((r.root / "snapshot.json").read_text())
    assert snapshot["status"] == "CONTEXT_SNAPSHOT_COLLECTED", snapshot
    assert snapshot["rows"] == [0, 2, 4]
    assert snapshot["fused_meta_matches_pool"]
    assert not r.model.fc._forward_hooks and not r.model.hidden_norm._forward_hooks
    response_files(r)
    result = probe.compare_snapshot(r.root)
    assert result["status"] == "CONTEXT_COMPARISON_COLLECTED"
    assert all(result["layout_checks"].values())
    assert all(all(x["checks"].values()) for x in result["layers"])
    assert all(x["v_copy_to_pool"]["exact_equal"] for x in result["layers"])
    old = (r.root / "hidden.npy").read_bytes()
    with torch.no_grad():
        r.h.zero_()
        r.model.fc.weight.zero_()
        r.pool.k_buffer[0].zero_()
    assert (r.root / "hidden.npy").read_bytes() == old
    assert r.worker._forward_prefill(r.batch) is r.output  # claim: never recapture
    assert (r.root / "hidden.npy").read_bytes() == old


@pytest.mark.parametrize("change", ["rid", "rank"])
def test_nonmatching_request_has_no_snapshot(runtime, change):
    r = runtime
    if change == "rank":
        r.worker.ps.tp_rank = 1
    else:
        r.batch.reqs[0].rid = "unrelated"
    assert r.worker._forward_prefill(r.batch) is r.output
    assert r.calls["worker"] == 1
    assert not (r.root / "capture.claim").exists()
    assert not r.model.fc._forward_hooks


@pytest.mark.parametrize(
    "bad", ["graph", "prefix", "model", "budget", "pool", "position"]
)
def test_diagnostic_failure_does_not_retry_or_replace_service(runtime, bad):
    r = runtime
    if bad == "graph":
        r.worker.server_args.disable_cuda_graph = False
    elif bad == "prefix":
        r.batch.prefix_lens = [1]
    elif bad == "model":
        r.worker.model_runner.model = object()
    elif bad == "budget":
        r.config["max_bytes"] = 10
    elif bad == "pool":
        r.pool.layer_transfer_counter = object()
    elif bad == "position":
        r.batch.extend_lens = [4]
    assert r.worker._forward_prefill(r.batch) is r.output
    assert r.calls["worker"] == r.calls["write"] == r.calls["fused"] == 1
    snap = json.loads((r.root / "snapshot.json").read_text())
    assert snap["status"] == "SNAPSHOT_FAILED" and snap["errors"]
    assert not r.model.fc._forward_hooks and obs.ACTIVE.get() is None


def test_original_exception_is_propagated_with_cleanup(runtime):
    r = runtime
    error = RuntimeError("original service error")

    def fail():
        raise error

    with pytest.raises(RuntimeError) as exc:
        r.worker._forward_prefill(r.batch, fail)
    assert exc.value is error
    assert r.calls["worker"] == 1
    assert obs.ACTIVE.get() is None and not r.model.fc._forward_hooks
    assert (
        json.loads((r.root / "snapshot.json").read_text())["status"]
        == "SNAPSHOT_FAILED"
    )


@pytest.mark.parametrize(
    "tamper", ["hash", "rid", "prompt", "cache", "missing", "budget"]
)
def test_replay_rejects_incomplete_or_misattributed_data(runtime, tamper):
    r = runtime
    r.worker._forward_prefill(r.batch)
    response = response_files(r)
    if tamper == "hash":
        with (r.root / "hidden.npy").open("ab") as f:
            f.write(b"changed")
    elif tamper == "rid":
        response["id"] = "wrong"
    elif tamper == "prompt":
        response["choices"][0]["prompt_token_ids"][0] = 99
    elif tamper == "cache":
        response["choices"][0]["meta_info"]["cached_tokens"] = 3
    elif tamper == "missing":
        (r.root / "pool_v_1.npy").unlink()
    elif tamper == "budget":
        r.config["max_bytes"] = 4
        obs.write_json(r.root / "config.json", r.config)
    obs.write_json(r.root / "response.json", response)
    with pytest.raises((ValueError, FileNotFoundError)):
        probe.compare_snapshot(r.root)


def test_validly_hashed_wrong_pool_values_are_reported_not_hidden(runtime):
    r = runtime
    r.worker._forward_prefill(r.batch)
    response_files(r)
    p = r.root / "pool_v_0.npy"
    data = np.load(p)
    data[:] = 0
    np.save(p, data, allow_pickle=False)
    snap = json.loads((r.root / "snapshot.json").read_text())
    snap["arrays"]["pool_v_0"]["sha256"] = obs.file_hash(p)
    obs.write_json(r.root / "snapshot.json", snap)
    result = probe.compare_snapshot(r.root)
    assert not result["layers"][0]["v_copy_to_pool"]["exact_equal"]
    assert result["layers"][0]["v_copy_to_pool"]["relative_l2"] > 0
    assert result["status"] == "CONTEXT_COMPARISON_COLLECTED"  # collection != PASS


def test_client_sends_one_fixed_request_and_does_not_retry(runtime, monkeypatch):
    r = runtime
    received = []
    info = dict(
        device="npu",
        speculative_algorithm="DSPARK",
        tp_size=16,
        dp_size=1,
        nnodes=1,
        disable_cuda_graph=True,
    )

    def http(opener, url, payload=None, timeout=0):
        if payload is None:
            return info
        received.append(payload)
        r.worker._forward_prefill(r.batch)
        return response_files(r)

    monkeypatch.setattr(probe, "request_json", http)
    assert (
        probe.collect(r.root, "http://localhost:8810")["status"]
        == "CONTEXT_COMPARISON_COLLECTED"
    )
    assert len(received) == 1 and received[0]["cache_salt"] == r.config["cache_salt"]
    assert "return_logprob" not in received[0]
    with pytest.raises(FileExistsError):
        probe.collect(r.root, "http://localhost:8810")
    assert len(received) == 1


def test_client_network_failure_is_not_retried(runtime, monkeypatch):
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
        raise TimeoutError("no response")

    monkeypatch.setattr(probe, "request_json", http)
    with pytest.raises(TimeoutError):
        probe.collect(r.root, "http://localhost:8810")
    assert len(calls) == 2 and (r.root / "client.claim").exists()


def test_bf16_bit_storage_and_budget_before_copy(tmp_path):
    cap = obs.Capture(
        dict(run_dir=str(tmp_path), run_id="x", rid="r", max_bytes=10000),
        NS(draft_model=None),
        None,
    )
    bits = torch.tensor([0, 0x8000, 0x3F80, 1, 0x7F80, 0x7FC1], dtype=torch.uint16)
    cap.save("bits", bits.view(torch.bfloat16))
    np.testing.assert_array_equal(np.load(tmp_path / "bits.npy"), bits.numpy())
    cap.config["max_bytes"] = cap.used + 1
    with pytest.raises(ValueError, match="budget"):
        cap.save("overflow", torch.ones(3))
    assert not (tmp_path / "overflow.npy").exists()


def test_disabled_install_does_not_import_framework_or_hook(tmp_path):
    code = f"import sys; sys.path.insert(0, {str(HERE)!r}); import dspark_context_snapshot as m; m.install(); assert not m._INSTALLED; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    env = os.environ.copy()
    env.pop(obs.ENV, None)
    subprocess.run([sys.executable, "-I", "-c", code], env=env, check=True)


def test_private_startup_and_spawn_and_kernel_overlay_path_order(tmp_path, monkeypatch):
    monkeypatch.delenv(obs.ENV, raising=False)
    monkeypatch.setattr(launch.importlib.util, "find_spec", lambda name: None)
    config, env = launch.prepare(tmp_path, HERE)
    boot = Path(config["run_dir"]) / "bootstrap"
    kernel = tmp_path / "kernel-overlay"
    kernel.mkdir()
    import with_kernel_checkout

    # Apply the actual existing helper to the environment prepared by the recipe.
    env["PYTHONPATH"] = str(HERE.parents[1] / "python") + os.pathsep + env["PYTHONPATH"]
    with monkeypatch.context() as m:
        m.setattr(with_kernel_checkout.os, "environ", env)
        env = with_kernel_checkout.child_environment(kernel)
    script = tmp_path / "spawn_check.py"
    script.write_text("""import multiprocessing as mp, sys
from pathlib import Path
import dspark_context_snapshot as m

def check():
    assert m._INSTALLED
    assert 'torch' not in sys.modules
    assert 'sglang' not in sys.modules
    assert type(sys.meta_path[0]).__name__ == 'ObserverFinder'
    assert Path(m.__file__).parent.name == 'bootstrap'

if __name__ == '__main__':
    check()
    p = mp.get_context('spawn').Process(target=check)
    p.start(); p.join(20)
    assert p.exitcode == 0
""")
    subprocess.run([sys.executable, str(script)], env=env, check=True, timeout=30)
    assert (boot / "dspark_context_snapshot.py").read_bytes() == (
        HERE / "dspark_context_snapshot.py"
    ).read_bytes()
    assert env["GRAPH"] == "0" and env["SGLANG_ENABLE_FAST_INPUT_LOGPROBS"] == "0"


def test_delayed_finder_calls_original_loader_then_wraps_only_selected_module(tmp_path):
    (tmp_path / "fixture_module.py").write_text("value = 17\n")
    events = []
    finder = obs.ObserverFinder({})
    finder.callbacks = {"fixture_module": lambda m: events.append(m.value)}
    sys.path.insert(0, str(tmp_path))
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module("fixture_module")
        assert module.value == 17 and events == [17]
        assert finder.find_spec("unrelated", None) is None
    finally:
        sys.meta_path.remove(finder)
        sys.path.remove(str(tmp_path))
        sys.modules.pop("fixture_module", None)


@pytest.mark.parametrize("runtime", ["paged", "fia"], indirect=True)
def test_npu_pool_layout_and_original_stacked_method_observed(runtime):
    r = runtime
    previous_profile = sys.getprofile()
    assert r.worker._forward_prefill(r.batch) is r.output
    assert sys.getprofile() is previous_profile
    assert r.calls == {"worker": 1, "write": 1, "fused": 0}
    snap = json.loads((r.root / "snapshot.json").read_text())
    assert snap["status"] == "CONTEXT_SNAPSHOT_COLLECTED", snap
    assert snap["branch"] == "stacked_ctx_kv"
    assert snap["pool_type"] == "NPUMHATokenToKVPool"
    response_files(r)
    result = probe.compare_snapshot(r.root)
    for layer in result["layers"]:
        assert all(layer["checks"].values())
        assert layer["stacked_actual_boundaries"]["k_copy_to_pool"]["exact_equal"]
        assert layer["stacked_actual_boundaries"]["v_output_to_pool"]["exact_equal"]
        assert layer["v_copy_to_pool"]["exact_equal"]
    assert not r.model.layers[0].self_attn.rotary_emb._forward_pre_hooks


@pytest.mark.parametrize("runtime", ["paged"], indirect=True)
def test_existing_profiler_preserved_and_original_runs(runtime):
    r = runtime

    def previous(frame, event, arg):
        pass

    sys.setprofile(previous)
    try:
        assert r.worker._forward_prefill(r.batch) is r.output
        assert sys.getprofile() is previous
    finally:
        sys.setprofile(None)
    snap = json.loads((r.root / "snapshot.json").read_text())
    assert snap["status"] == "SNAPSHOT_FAILED"
    assert any("profiler" in e for e in snap["errors"])
