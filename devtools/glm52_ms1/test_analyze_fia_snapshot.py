"""Synthetic CPU tests of the offline analyser, not model/NPU validation."""

# ruff: noqa: E402 -- import the standalone temporary tools.

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_fia_snapshot as tool
from dspark_attention_reference import attention
from dspark_local_reference import bf16_round, tensor_difference


def write_json(path, value):
    path.write_text(json.dumps(value))


def save_array(root, snapshot, name, value, bf16=True):
    value = np.asarray(value)
    if bf16:
        value = (bf16_round(value).view(np.uint32) >> 16).astype(np.uint16)
    file = root / (name + ".npy")
    np.save(file, value)
    snapshot["arrays"][name] = {
        "file": file.name,
        "shape": list(value.shape),
        "dtype": "torch.bfloat16" if bf16 else f"torch.int{value.dtype.itemsize * 8}",
        "encoding": "bf16_uint16" if bf16 else "numpy",
        "bytes": file.stat().st_size,
        "sha256": tool.digest(file),
    }


@pytest.fixture
def evidence(tmp_path):
    rng = np.random.default_rng(8)
    q = bf16_round(rng.normal(size=(8, 4, 192)) / 4)
    k = bf16_round(rng.normal(size=(31, 4, 192)) / 4)
    v = bf16_round(rng.normal(size=(31, 4, 192)))
    scale = 192**-0.5
    out = bf16_round(attention(q, k, v, scale))
    # Inject a known output difference without changing Q/K/V or the reference.
    out[2, 1, 17] = np.float32(2)
    snapshot = {
        "status": "PROPOSAL_SNAPSHOT_COLLECTED",
        "errors": [],
        "run_id": "fixture",
        "rid": "synthetic-rid",
        "rank": 0,
        "layer_id": 0,
        "stage": "first_proposal_layer0",
        "observer_sha256": "fixture-observer",
        "geometry": {"hidden": 6144, "heads": 4, "head_dim": 192, "queries": 8},
        "prefill": {
            "prompt_ids": list(range(23)),
            "prefix_lens": [0],
            "rid": "synthetic-rid",
        },
        "backend_branch": "ordinary_mha_fia_tnd",
        "mask_is_none": True,
        "sparse_mode": 0,
        "sliding_window_size": -1,
        "attn_type": "AttentionType.ENCODER_ONLY",
        "scale": scale,
        "actual_seq_lengths_q": [8],
        "actual_seq_lengths_kv": [31],
        "page_size": 128,
        "pool_capacity": 512,
        "actual_slots": list(range(128, 159)),
        "arrays": {},
    }
    for name, value in {
        "fia_query": q,
        "pool_k": k,
        "pool_v": v,
        "attention_output": out.reshape(8, 768),
    }.items():
        save_array(tmp_path, snapshot, name, value)
    for name, value in {
        "positions": np.arange(23, 31),
        "block_table": np.array([[1]]),
        "logical_slots": np.arange(128, 159),
        "out_cache_loc": np.arange(151, 159),
    }.items():
        dtype = np.int32 if name in ("block_table", "logical_slots") else np.int64
        save_array(tmp_path, snapshot, name, value.astype(dtype), bf16=False)
    # No checkpoint file is present. The analyser must not open this irrelevant entry.
    snapshot["arrays"]["qkv_weight"] = {"file": "not-present.npy"}
    snapshot["draft_path"] = str(tmp_path / "nonexistent-checkpoint")
    config = {
        "run_id": "fixture",
        "rid": "synthetic-rid",
        "tool_hashes": {
            "dspark_proposal_snapshot.py": "fixture-observer",
            **{
                n: tool.digest(HERE / n)
                for n in ("dspark_attention_reference.py", "dspark_local_reference.py")
            },
        },
    }
    ref = bf16_round(attention(q, k, v, scale))
    comparison = {
        "status": "PROPOSAL_COMPARISON_COLLECTED",
        "run_id": "fixture",
        "rid": "synthetic-rid",
        "scope": {
            "rank": 0,
            "layer_id": 0,
            "prefix_tokens": 23,
            "query_tokens": 8,
            "head_dim": 192,
            "local_heads": 4,
        },
        "structural_issues": [],
        "attention_replay": "SAME_ACTUAL_INPUTS_COMPARED",
        "comparisons": {
            "attention_same_actual_inputs_float64_bf16_storage": tensor_difference(
                ref, out
            )
        },
    }
    for name, obj in (
        ("config", config),
        ("snapshot", snapshot),
        ("comparison", comparison),
    ):
        write_json(tmp_path / (name + ".json"), obj)
    return tmp_path


def mutate(root, name, key, value):
    path = root / name
    obj = json.loads(path.read_text())
    obj[key] = value
    write_json(path, obj)


def hashes(root):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.is_file()
    }


def test_readonly_snapshot_replay_and_known_error(evidence):
    before = hashes(evidence)
    result = tool.analyze(evidence)
    assert hashes(evidence) == before
    assert result["status"] == "FIA_OFFLINE_ANALYSIS_COLLECTED"
    assert len(result["selected_arrays"]) == 8
    assert result["selected_array_bytes"] < 200_000
    assert result["prior_comparison"]["relative_l2_change"] == 0
    assert result["residual"]["overall"]["nonzero_count"] == 1
    largest = result["residual"]["top_abs_error"][0]
    assert (largest["query_index"], largest["head_index"], largest["dim_index"]) == (
        2,
        1,
        17,
    )
    assert largest["actual"] == 2
    assert len(result["reference_attention_profile"]["rows"]) == 32
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rid", "other"),
        ("rank", 1),
        ("errors", ["incomplete"]),
        ("mask_is_none", False),
        ("sparse_mode", 3),
        ("scale", 1.0),
        ("actual_seq_lengths_kv", [32]),
        ("actual_seq_lengths_q", [9]),
        ("observer_sha256", "other"),
        ("status", "PROPOSAL_SNAPSHOT_FAILED"),
    ],
)
def test_rejects_unmatched_evidence(evidence, field, value):
    mutate(evidence, "snapshot.json", field, value)
    with pytest.raises(ValueError):
        tool.analyze(evidence)


def test_changed_array_rejected(evidence):
    with (evidence / "pool_k.npy").open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="Array size"):
        tool.analyze(evidence)


def test_changed_reference_identity_rejected(evidence):
    config = tool.read_json(evidence / "config.json")
    config["tool_hashes"]["dspark_attention_reference.py"] = "bad"
    write_json(evidence / "config.json", config)
    with pytest.raises(ValueError, match="Reference changed"):
        tool.analyze(evidence)


def test_comparison_scope_must_match_snapshot(evidence):
    previous = tool.read_json(evidence / "comparison.json")
    previous["scope"]["prefix_tokens"] = 24
    write_json(evidence / "comparison.json", previous)
    with pytest.raises(ValueError, match="comparison scope differs"):
        tool.analyze(evidence)


@pytest.mark.parametrize(
    "name,value,bf16",
    [
        ("positions", np.arange(24, 32), False),
        ("logical_slots", np.arange(129, 160), False),
        ("block_table", np.array([[2]]), False),
        ("fia_query", np.zeros((8, 192, 4)), True),
        ("pool_v", np.full((31, 4, 192), np.nan), True),
    ],
)
def test_rejects_self_consistent_file_with_wrong_contract(evidence, name, value, bf16):
    snapshot = tool.read_json(evidence / "snapshot.json")
    save_array(evidence, snapshot, name, value, bf16=bf16)
    write_json(evidence / "snapshot.json", snapshot)
    with pytest.raises(ValueError):
        tool.analyze(evidence)


def test_uniform_reference_profile_is_not_runtime_observation():
    result = tool.reference_profile(
        np.zeros((8, 4, 192)),
        np.zeros((31, 4, 192)),
        np.ones((31, 4, 192)),
        192**-0.5,
        23,
        np.arange(23, 31),
    )
    row = result["rows"][0]
    assert row["context_probability_mass"] == pytest.approx(23 / 31)
    assert row["current_block_probability_mass"] == pytest.approx(8 / 31)
    assert row["entropy_nats"] == pytest.approx(np.log(31))
    assert "not an observation" in result["origin"]


def test_official_style_candidate_matches_independent_torch_storage_steps():
    import torch

    rng = np.random.default_rng(10)
    # Binary fractions make the small dot products exact in FP32; this test
    # checks placement of BF16 boundaries, not a tolerance for the NPU model.
    q, k, v = [
        rng.integers(-8, 9, size=shape).astype(np.float32) / 8
        for shape in ((8, 4, 192), (31, 4, 192), (31, 4, 192))
    ]
    tq, tk, tv = [torch.from_numpy(x).bfloat16().transpose(0, 1) for x in (q, k, v)]
    scale = 192**-0.5
    scores = torch.matmul(tq, tk.transpose(1, 2)) * scale
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).bfloat16()
    expected = torch.matmul(probs, tv).transpose(0, 1).float().numpy()
    actual = tool.op_plugin_style_reference(q, k, v, scale)
    np.testing.assert_array_equal(actual, expected)


def test_cli_separate_report_and_no_input_changes(evidence):
    before = hashes(evidence)
    process = subprocess.run(
        [
            sys.executable,
            str(HERE / "analyze_fia_snapshot.py"),
            "--snapshot-dir",
            str(evidence),
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    assert "FIA_OFFLINE_ANALYSIS_COLLECTED" in process.stdout
    reports = list(evidence.glob("fia-analysis-*/report.json"))
    assert len(reports) == 1
    assert hashes(evidence) == before
    report = tool.read_json(reports[0])
    assert report["run_id"] == "fixture"


def test_cli_fail_does_not_create_analysis_or_rewrite(evidence):
    mutate(evidence, "snapshot.json", "mask_is_none", False)
    before = hashes(evidence)
    process = subprocess.run(
        [
            sys.executable,
            str(HERE / "analyze_fia_snapshot.py"),
            "--snapshot-dir",
            str(evidence),
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 1
    assert "FIA_OFFLINE_ANALYSIS_FAILED" in process.stderr
    assert list(evidence.glob("fia-analysis-*")) == []
    assert hashes(evidence) == before
