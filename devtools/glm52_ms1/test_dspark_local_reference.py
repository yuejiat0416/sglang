"""Calibrate the CPU reference; these are not NPU/model precision gates.

Hand-computed cases check layout and algebra. PyTorch CPU formula comparisons
use torch.testing.assert_close's dtype defaults, not a new deployment tolerance.
"""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

SCRIPT = Path(__file__).with_name("dspark_local_reference.py")
SPEC = importlib.util.spec_from_file_location("dspark_local_reference", SCRIPT)
ref = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ref)


def close_cpu(actual, expected, *, bf16=False):
    actual = torch.from_numpy(np.array(actual, copy=True))
    if bf16:
        actual, expected = actual.bfloat16(), expected.bfloat16()
    torch.testing.assert_close(actual, expected)


def test_standalone_import_has_no_runtime_or_framework_dependency():
    code = """
import importlib.util, sys
s = importlib.util.spec_from_file_location('ref', sys.argv[1])
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
assert not any(k.split('.')[0] in {'torch', 'torch_npu', 'sglang', 'triton'} for k in sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(SCRIPT)], check=True)


def test_linear_hand_computed_non_square_and_bias():
    x = np.array([[2, -1, 3], [-3, 2, 1]], dtype=np.float32)
    w = np.array([[1, 2, 0], [-1, 0, 4]], dtype=np.float32)
    np.testing.assert_array_equal(ref.linear(x, w), [[0, 10], [1, 7]])
    np.testing.assert_array_equal(
        ref.linear(x, w, np.array([2, -3], dtype=np.float32)), [[2, 7], [3, 4]]
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_linear_matches_pytorch_cpu(dtype):
    rng = np.random.default_rng(19)
    x, w = (
        rng.normal(size=(3, 15)).astype(dtype),
        rng.normal(size=(7, 15)).astype(dtype),
    )
    close_cpu(
        ref.linear(x, w, compute_dtype=dtype),
        F.linear(torch.from_numpy(x), torch.from_numpy(w)),
    )


def test_projection_and_norm_hand_computed():
    result = ref.project_context(
        np.array([[1, 2]], dtype=np.float32),
        np.array([[2, 0], [0, 2]], dtype=np.float32),
        np.array([1, 2], dtype=np.float32),
        6.0,
    )
    np.testing.assert_array_equal(result["fc_output"], [[2, 4]])
    np.testing.assert_array_equal(result["context"], [[0.5, 2]])


@pytest.mark.parametrize("scale", [0.0, 1e-4, 0.01, 1.0, 10.0])
def test_norm_zero_small_and_regular_values_with_learned_scale(scale):
    x = np.array([[[1, -2, 3, -4], [3, 2, 1, -1]]], dtype=np.float32) * scale
    w = np.array([0.5, 2, -1, 3], dtype=np.float32)
    expected = F.rms_norm(torch.from_numpy(x), (4,), torch.from_numpy(w), eps=1e-5)
    close_cpu(ref.rms_norm(x, w, 1e-5), expected)
    if scale == 0:
        np.testing.assert_array_equal(ref.rms_norm(x, w, 1e-5), np.zeros_like(x))


def test_float64_norm_matches_scalar_definition():
    x = np.array([[1e-4, -2e-4, 3e-4], [1, 2, 5]], dtype=np.float64)
    w = np.array([1.5, 0.25, 2], dtype=np.float64)
    expected = torch.tensor(
        [
            [
                float(v)
                * float(g)
                / math.sqrt(sum(float(z) ** 2 for z in row) / 3 + 1e-5)
                for v, g in zip(row, w)
            ]
            for row in x
        ],
        dtype=torch.float64,
    )
    close_cpu(ref.rms_norm(x, w, 1e-5, compute_dtype=np.float64), expected)


def test_bf16_known_ties_sign_zero_subnormals_and_special_values():
    words = np.array(
        [
            0x3F808000,
            0x3F818000,
            0xBF808000,
            0xBF818000,
            0x00008000,
            0x00018000,
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7F800001,
        ],
        dtype=np.uint32,
    )
    expected = np.array(
        [
            0x3F800000,
            0x3F820000,
            0xBF800000,
            0xBF820000,
            0x00000000,
            0x00020000,
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(
        ref.bf16_round(words.view(np.float32)).view(np.uint32), expected
    )


def test_bf16_matches_cpu_conversion_and_preserves_readonly_input():
    x = np.random.default_rng(3).normal(size=(11, 192)).astype(np.float32)
    saved = x.copy()
    x.flags.writeable = False
    expected = torch.tensor(saved).bfloat16().float().numpy()
    actual = ref.bf16_round(x)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, saved)
    assert not np.shares_memory(actual, x)


def test_storage_projection_keeps_fc_rounding_boundary():
    rng = np.random.default_rng(41)
    x, w, g = (
        ref.bf16_round(rng.normal(size=shape)) for shape in ((2, 15), (6, 15), (6,))
    )
    z = F.linear(torch.from_numpy(x), torch.from_numpy(w)).bfloat16().float()
    c = F.rms_norm(z, (6,), torch.from_numpy(g), eps=1e-5).bfloat16().float()
    result = ref.project_context(x, w, g, 1e-5, bf16_storage=True)
    close_cpu(result["fc_output"], z, bf16=True)
    close_cpu(result["context"], c, bf16=True)
    # The mathematical and storage references are intentionally separate.
    mathematical = ref.project_context(x, w, g, 1e-5)
    assert not np.array_equal(result["fc_output"], mathematical["fc_output"])


@pytest.mark.parametrize(
    "neox, expected",
    [
        (True, [[[-3, -4, 1, 2], [-7, -8, 5, 6]]]),
        (False, [[[-2, 1, -4, 3], [-6, 5, -8, 7]]]),
    ],
)
def test_rope_quarter_turn_hand_computed(neox, expected):
    keys = np.arange(1, 9, dtype=np.float32).reshape(1, 2, 4)
    actual = ref.apply_rope(
        keys,
        np.zeros((1, 2), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        is_neox_style=neox,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(keys.ravel(), np.arange(1, 9))


@pytest.mark.parametrize("neox", [True, False])
def test_rope_position_zero_identity_192(neox):
    k = np.arange(2 * 192, dtype=np.float32).reshape(1, 2, 192)
    cos, sin = ref.rope_cos_sin(np.array([0]), 192, 8000000.0)
    np.testing.assert_array_equal(ref.apply_rope(k, cos, sin, is_neox_style=neox), k)


def test_rope_selected_positions_against_scalar_angles():
    positions = np.array([17, 1, 1000000, 17], dtype=np.int64)
    cos, sin = ref.rope_cos_sin(positions, 192, 8000000.0, compute_dtype=np.float64)
    angles = [
        [float(p) / (8000000.0 ** (2 * d / 192)) for d in range(96)] for p in positions
    ]
    for actual, func in ((cos, math.cos), (sin, math.sin)):
        expected = torch.tensor(
            [[func(a) for a in row] for row in angles], dtype=torch.float64
        )
        close_cpu(actual, expected)
    np.testing.assert_array_equal(cos[0], cos[-1])


def test_rope_cache_dtype_is_separate_from_output_storage():
    positions = np.array([2, 3])
    cos, sin = ref.rope_cos_sin(positions, 4, 10000.0)
    c_bf, s_bf = ref.rope_cos_sin(positions, 4, 10000.0, bf16_cache=True)
    np.testing.assert_array_equal(c_bf, ref.bf16_round(cos))
    np.testing.assert_array_equal(s_bf, ref.bf16_round(sin))
    k = np.arange(1, 17, dtype=np.float32).reshape(2, 2, 4)
    a = ref.apply_rope(k, cos, sin, is_neox_style=True)
    b = ref.apply_rope(k, c_bf, s_bf, is_neox_style=True)
    assert not np.array_equal(a, b)


def torch_kv_formula(context, kw, vw, nw, cos, sin, dim, eps, neox, storage):
    outputs = {k: [] for k in ("k_projected", "v_projected", "k_normed", "k_rope")}
    ctx, kw, vw, nw, c, s = [
        torch.from_numpy(x.copy()) for x in (context, kw, vw, nw, cos, sin)
    ]

    def stored(x):
        return x.bfloat16().float() if storage else x

    for layer in range(len(kw)):
        k = stored(F.linear(ctx, kw[layer])).reshape(len(ctx), -1, dim)
        v = stored(F.linear(ctx, vw[layer])).reshape_as(k)
        n = stored(F.rms_norm(k, (dim,), nw[layer], eps=eps))
        # Explicit dimension pairs, independent of the vector slice implementation.
        rotated = torch.empty_like(n)
        for j in range(dim // 2):
            a, b = (j, j + dim // 2) if neox else (2 * j, 2 * j + 1)
            rotated[..., a] = n[..., a] * c[:, j, None] - n[..., b] * s[:, j, None]
            rotated[..., b] = n[..., b] * c[:, j, None] + n[..., a] * s[:, j, None]
        for name, value in zip(outputs, (k, v, n, stored(rotated))):
            outputs[name].append(value)
    return {name: torch.stack(rows) for name, rows in outputs.items()}


@pytest.mark.parametrize("dim", [4, 192])
@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("storage", [False, True])
def test_context_kv_multiple_layers_heads_positions_cpu_formula(dim, neox, storage):
    rng = np.random.default_rng(25)
    ctx, kw, vw, nw = [
        ref.bf16_round(rng.normal(size=shape))
        for shape in ((3, 6), (2, 2 * dim, 6), (2, 2 * dim, 6), (2, dim))
    ]
    cos, sin = ref.rope_cos_sin(np.array([7, 0, 23]), dim, 8000000.0)
    result = ref.context_kv(
        ctx,
        kw,
        vw,
        nw,
        cos,
        sin,
        head_dim=dim,
        eps=1e-5,
        is_neox_style=neox,
        bf16_storage=storage,
    )
    expected = torch_kv_formula(ctx, kw, vw, nw, cos, sin, dim, 1e-5, neox, storage)
    for name in result:
        assert result[name].shape == (2, 3, 2, dim)
        close_cpu(result[name], expected[name], bf16=storage)
    assert not np.shares_memory(result["k_projected"], ctx)


def test_norm_intermediate_bf16_rounding_is_visible():
    rng = np.random.default_rng(17)
    k = ref.bf16_round(rng.normal(size=(3, 2, 192)))
    g = ref.bf16_round(rng.normal(size=192))
    cos, sin = ref.rope_cos_sin(np.array([1, 8, 31]), 192, 8000000.0)
    n = ref.rms_norm(k, g, 1e-5)
    expected = ref.apply_rope(
        ref.bf16_round(n), cos, sin, is_neox_style=True, bf16_storage=True
    )
    missing_cast = ref.apply_rope(n, cos, sin, is_neox_style=True, bf16_storage=True)
    assert ref.tensor_difference(expected, missing_cast)["max_abs_error"] > 0


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_difference_detects_layer_token_head_swaps(axis):
    original = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)
    swapped = np.roll(original, 1, axis=axis)
    report = ref.tensor_difference(original, swapped)
    assert not report["exact_equal"]
    assert report["relative_l2"] > 0
    assert report["max_abs_error"] > 0
    assert "passed" not in report


def test_difference_does_not_confuse_direction_and_scale():
    x = np.array([1, 2, 3], dtype=np.float32)
    result = ref.tensor_difference(x, 2 * x)
    assert result["relative_l2"] == 1.0
    assert result["norm_ratio"] == 2.0
    assert result["max_abs_error"] == 3.0
    assert math.isclose(result["cosine"], 1.0)
    assert not result["exact_equal"]


@pytest.mark.parametrize(
    "actual", [np.zeros(3), np.ones(3), np.full(3, np.nan), np.full(3, np.inf)]
)
def test_difference_zero_norm_nonfinite_json_and_no_pass(actual):
    report = ref.tensor_difference(np.zeros(3), actual)
    assert report["relative_l2"] is None
    assert report["cosine"] is None
    assert "pass" not in report and "status" not in report
    json.dumps(report, allow_nan=False)


def test_reference_handles_empty_token_batch_explicitly():
    x = np.empty((0, 3), dtype=np.float32)
    w = np.ones((4, 3), dtype=np.float32)
    assert ref.linear(x, w).shape == (0, 4)
    report = ref.tensor_difference(x, x)
    assert report["numel"] == 0 and report["max_abs_error"] is None


def test_difference_marks_metric_overflow_without_emitting_invalid_json():
    x = np.array([1e308, -1e308])
    report = ref.tensor_difference(x, -x)
    assert report["finite"] and not report["metrics_finite"]
    assert report["cosine"] is None and report["relative_l2"] is None
    json.dumps(report, allow_nan=False)


def test_difference_marks_ratio_overflow_without_emitting_invalid_json():
    report = ref.tensor_difference(np.array([1e-160]), np.array([1e150]))
    assert report["finite"] and not report["metrics_finite"]
    assert report["relative_l2"] is None and report["norm_ratio"] is None
    json.dumps(report, allow_nan=False)


def test_norm_local_replay_uses_actual_upstream_output():
    # FC and norm are separate boundaries. A changed FC output is not evidence
    # that norm is wrong: feed that same actual output to both norm paths.
    actual_z = np.array([[0.001, 0.03, 0.5, 2.0]], dtype=np.float32)
    g = np.array([1, 2, 0.5, 3], dtype=np.float32)
    close_cpu(
        ref.rms_norm(actual_z, g, 1e-5),
        F.rms_norm(torch.tensor(actual_z), (4,), torch.tensor(g), eps=1e-5),
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: ref.linear(np.ones((2, 3)), np.ones((4, 2))),
        lambda: ref.linear(np.ones((2, 3)), np.ones((4, 3)), np.ones((1, 4))),
        lambda: ref.rms_norm(np.ones((2, 3)), np.ones((2, 3)), 1e-5),
        lambda: ref.rms_norm(np.ones((2, 3)), np.ones(3), 0),
        lambda: ref.rms_norm(np.ones((2, 3)), np.ones(3), float("nan")),
        lambda: ref.rope_cos_sin(np.array([-1]), 192, 8000000),
        lambda: ref.rope_cos_sin(np.array([1.5]), 192, 8000000),
        lambda: ref.rope_cos_sin(np.array([0]), 191, 8000000),
        lambda: ref.rope_cos_sin(np.array([0]), 192, float("inf")),
        lambda: ref.apply_rope(
            np.ones((2, 2, 4)), np.ones((1, 2)), np.zeros((1, 2)), is_neox_style=True
        ),
        lambda: ref.tensor_difference(np.ones((2, 3)), np.ones((3, 2))),
        lambda: ref.linear(np.ones((2, 3)), np.ones((4, 3)), compute_dtype=np.float16),
        lambda: ref.context_kv(
            np.ones((1, 3)),
            np.ones((2, 5, 3)),
            np.ones((2, 5, 3)),
            np.ones((2, 4)),
            np.ones((1, 2)),
            np.zeros((1, 2)),
            head_dim=4,
            eps=1e-5,
            is_neox_style=True,
        ),
    ],
)
def test_reference_rejects_ambiguous_shapes_and_invalid_math(call):
    with pytest.raises(ValueError):
        call()
