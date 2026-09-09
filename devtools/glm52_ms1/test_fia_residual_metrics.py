"""Calibrate offline residual accounting with hand-computed BF16 examples."""

import json
import subprocess
import sys
from pathlib import Path

import fia_residual_metrics as residual
import numpy as np
import pytest


def bf16_bits(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def test_import_has_no_runtime_dependencies():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import fia_residual_metrics
assert not any(k.split('.')[0] in {'torch', 'torch_npu', 'sglang', 'triton'} for k in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(Path(__file__).parent)], check=True
    )


def test_one_boundary_neighbor_spacing_is_asymmetric():
    assert residual.bf16_neighbor_spacing(np.float32(1)) == {
        "lower_value": 1 - 2**-8,
        "upper_value": 1 + 2**-7,
        "lower_gap": 2**-8,
        "upper_gap": 2**-7,
    }
    np.testing.assert_array_equal(
        residual.bf16_steps(
            np.array([1 - 2**-8, 1, 1 + 2**-7], np.float32),
            np.array([1, 1 + 2**-7, 1 - 2**-8], np.float32),
        ),
        [1, 1, 2],
    )


def test_negative_boundary_reverses_spacing():
    assert residual.bf16_neighbor_spacing(np.float32(-1)) == {
        "lower_value": -1 - 2**-7,
        "upper_value": -1 + 2**-8,
        "lower_gap": 2**-7,
        "upper_gap": 2**-8,
    }
    np.testing.assert_array_equal(
        residual.bf16_steps(
            np.array([-1 - 2**-7, -1, -1 + 2**-8], np.float32),
            np.array([-1, -1 + 2**-8, -1 - 2**-7], np.float32),
        ),
        [1, 1, 2],
    )


def test_signed_zero_collapses_and_crossing_zero_counts_two_neighbors():
    tiny = 2**-133
    values = np.array([-tiny, -0.0, 0.0, tiny], np.float32)
    np.testing.assert_array_equal(
        residual.bf16_steps(values, np.array([tiny, 0, -0.0, -tiny], np.float32)),
        [2, 0, 0, 2],
    )
    assert residual.bf16_neighbor_spacing(np.float32(-0.0)) == {
        "lower_value": -tiny,
        "upper_value": tiny,
        "lower_gap": tiny,
        "upper_gap": tiny,
    }


def test_subnormal_to_normal_boundary_is_one_step():
    assert int(residual.bf16_steps(bf16_bits(0x007F), bf16_bits(0x0080))) == 1
    assert int(residual.bf16_steps(bf16_bits(0x8080), bf16_bits(0x807F))) == 1


def test_finite_extremes_have_null_outer_neighbor():
    positive = residual.bf16_neighbor_spacing(bf16_bits(0x7F7F))
    negative = residual.bf16_neighbor_spacing(bf16_bits(0xFF7F))
    assert positive["upper_value"] is positive["upper_gap"] is None
    assert negative["lower_value"] is negative["lower_gap"] is None
    assert positive["lower_gap"] == negative["upper_gap"] == 2**120
    assert int(residual.bf16_steps(bf16_bits(0xFF7F), bf16_bits(0x7F7F))) == 2 * 0x7F7F


def test_all_finite_bf16_neighbors_are_one_step_apart():
    # Enumerate the format independently: negatives in reverse magnitude order,
    # one zero, then positives. This includes all exponent and sign boundaries.
    bits = np.concatenate(
        [
            np.arange(0xFF7F, 0x8000, -1, dtype=np.uint32),
            np.arange(0, 0x7F80, dtype=np.uint32),
        ]
    )
    values = bf16_bits(bits)
    assert np.all(np.diff(values.astype(np.float64)) > 0)
    np.testing.assert_array_equal(
        residual.bf16_steps(values[:-1], values[1:]), np.ones(values.size - 1)
    )


def test_finite_extreme_residual_metrics_remain_json_finite():
    reference = bf16_bits([0xFF7F, 0x7F7F]).reshape(1, 1, 2)
    actual = -reference
    report = residual.analyze_residual(reference, actual, [0])
    assert report["overall"]["metrics_finite"]
    assert report["overall"]["relative_l2"] == 2
    assert report["overall"]["signed_mean_error"] == 0
    json.dumps(report, allow_nan=False)


def test_single_injected_residual_coordinates_and_group_totals():
    reference = np.ones((2, 3, 4), np.float32)
    actual = reference.copy()
    actual[1, 2, 3] = 1.125
    report = residual.analyze_residual(reference, actual, [23, 24], top_k=1)
    overall = report["overall"]
    assert overall["numel"] == 24
    assert overall["exact_count"] == 23
    assert overall["nonzero_count"] == 1
    assert overall["signed_mean_error"] == 0.125 / 24
    assert overall["rmse"] == np.sqrt(0.125**2 / 24)
    assert overall["bf16_steps"]["max"] == 16
    assert overall["bf16_steps"]["median"] == 0
    assert report["per_query"][0]["metrics"]["nonzero_count"] == 0
    assert report["per_query"][1]["metrics"]["nonzero_count"] == 1
    assert report["per_head"][2]["metrics"]["signed_mean_error"] == 0.125 / 8
    assert report["per_query_head"][5]["metrics"]["signed_mean_error"] == 0.125 / 4
    top = report["top_abs_error"][0]
    assert (
        top["query_index"],
        top["position"],
        top["head_index"],
        top["dim_index"],
    ) == (1, 24, 2, 3)
    assert top["reference"] == 1
    assert top["actual"] == 1.125
    assert top["signed_error"] == top["abs_error"] == 0.125
    assert top["bf16_step_distance"] == 16
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(reference, np.ones_like(reference))


def test_step_distribution_exact_boundaries():
    counts = np.array([0, 1, 2, 3, 4, 5, 8, 9, 16, 17], np.uint32)
    reference = np.ones((1, 1, 10), np.float32)
    actual = bf16_bits(0x3F80 + counts).reshape(reference.shape)
    report = residual.analyze_residual(reference, actual, [0], top_k=0)
    assert report["overall"]["bf16_steps"] == {
        "count": 10,
        "max": 17,
        "median": 4.5,
        "distribution": {
            "0": 1,
            "1": 1,
            "2": 1,
            "3-4": 2,
            "5-8": 2,
            "9-16": 2,
            ">16": 1,
        },
    }
    assert report["top_abs_error"] == []


def test_zero_reference_uses_null_relative_metrics_and_exact_signed_zero():
    reference = np.zeros((1, 1, 2), np.float32)
    actual = np.array([[[-0.0, 0.0]]], np.float32)
    report = residual.analyze_residual(reference, actual, [0])
    metrics = report["overall"]
    assert metrics["relative_l2"] is metrics["cosine"] is metrics["norm_ratio"] is None
    assert metrics["exact_count"] == 2
    assert metrics["rmse"] == metrics["signed_mean_error"] == 0
    json.dumps(report, allow_nan=False)


def test_largest_absolute_error_stable_ties_and_negative_sign():
    reference = np.ones((1, 2, 2), np.float32)
    actual = reference.copy()
    actual[0, 0, 1] -= 0.125
    actual[0, 1, 0] += 0.125
    report = residual.analyze_residual(reference, actual, [23], top_k=9)
    top = report["top_abs_error"]
    assert len(top) == 4
    assert (top[0]["head_index"], top[0]["dim_index"]) == (0, 1)
    assert (top[1]["head_index"], top[1]["dim_index"]) == (1, 0)
    assert top[0]["signed_error"] == -0.125
    # Same absolute error crosses twice as many BF16 values below 1 as above it.
    assert top[0]["bf16_step_distance"] == 32
    assert top[1]["bf16_step_distance"] == 16


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1.001, 1e40, 1 + 2**-30])
def test_nonfinite_or_non_bf16_values_rejected_without_rounding(value):
    reference = np.array([[[value]]], np.float64)
    actual = np.ones((1, 1, 1), np.float32)
    with pytest.raises(ValueError):
        residual.analyze_residual(reference, actual, [0])
    with pytest.raises(ValueError):
        residual.analyze_residual(actual, reference, [0])


@pytest.mark.parametrize("dtype", [np.int64, np.uint16, np.bool_, np.complex64, str])
def test_non_float_input_rejected(dtype):
    with pytest.raises(ValueError):
        residual.bf16_steps(np.ones(1, dtype=dtype), np.ones(1, np.float32))


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 1, 1, 1), (0, 2, 3), (1, 0, 3), (1, 2, 0)]
)
def test_wrong_rank_or_empty_arrays_rejected(shape):
    values = np.ones(shape, np.float32)
    with pytest.raises(ValueError):
        residual.analyze_residual(values, values, [0])


def test_broadcasting_rejected():
    reference = np.ones((2, 2, 3), np.float32)
    actual = np.ones((1, 2, 3), np.float32)
    with pytest.raises(ValueError, match="broadcasting"):
        residual.analyze_residual(reference, actual, [0, 1])
    with pytest.raises(ValueError, match="broadcasting"):
        residual.bf16_steps(reference, actual)


@pytest.mark.parametrize(
    "positions", [[1], [1, -1], [1.0, 2.0], [[1, 2]], [True, False]]
)
def test_bad_positions_rejected(positions):
    values = np.ones((2, 1, 1), np.float32)
    with pytest.raises(ValueError, match="positions"):
        residual.analyze_residual(values, values, positions)


@pytest.mark.parametrize("top_k", [-1, 0.1, True, np.bool_(False), "8"])
def test_bad_top_k_rejected(top_k):
    values = np.ones((1, 1, 1), np.float32)
    with pytest.raises(ValueError, match="top_k"):
        residual.analyze_residual(values, values, [0], top_k=top_k)


def test_spacing_rejects_vector():
    with pytest.raises(ValueError, match="scalar"):
        residual.bf16_neighbor_spacing(np.ones(2, np.float32))
