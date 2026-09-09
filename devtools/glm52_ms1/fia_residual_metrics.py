# SPDX-License-Identifier: Apache-2.0
"""Describe FIA output residuals on saved CPU arrays, without precision gates.

Both supplied tensors must already contain finite, exactly representable BF16
values. This helper neither rounds their inputs nor reconstructs FIA arithmetic.
Its ordered BF16 distance counts adjacent representable values, collapsing +0
and -0. It is not an error divided by one globally constant ULP: BF16 spacing
changes with magnitude and differs on opposite sides of a power-of-two boundary.
"""

import numpy as np
from dspark_local_reference import bf16_round, tensor_difference


def _bf16_array(values):
    values = np.asarray(values)
    if values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError("Expected finite floating point BF16 values")
    with np.errstate(over="ignore", invalid="ignore"):
        fp32 = values.astype(np.float32)
    if not np.isfinite(fp32).all() or not np.array_equal(values, fp32):
        raise ValueError("Input values must be exactly representable as BF16")
    if not np.array_equal(fp32, bf16_round(fp32)):
        raise ValueError("Input values must be exactly representable as BF16")
    return fp32


def _ordered_bf16(values):
    bits = values.view(np.uint32) >> 16
    magnitude = (bits & 0x7FFF).astype(np.int64)
    return np.where(bits & 0x8000, 0x8000 - magnitude, 0x8000 + magnitude)


def bf16_steps(reference, actual):
    """Count neighboring BF16 values between each pair; signed zeros coincide."""
    reference, actual = _bf16_array(reference), _bf16_array(actual)
    if reference.shape != actual.shape:
        raise ValueError("Reference and actual shapes differ; no broadcasting")
    return np.abs(_ordered_bf16(actual) - _ordered_bf16(reference))


def _decode_ordered(rank):
    bits = (0x8000 | (0x8000 - rank)) if rank < 0x8000 else rank - 0x8000
    value = np.array(bits << 16, dtype=np.uint32).view(np.float32).item()
    return float(value) if np.isfinite(value) else None


def bf16_neighbor_spacing(value):
    """Describe both neighbors of a scalar, using null outside finite BF16.

    At either signed zero, the neighbors are +/- the smallest BF16 subnormal.
    This reports format spacing only; it does not assume an NPU preserves every
    subnormal during computation.
    """
    value = _bf16_array(value)
    if value.ndim != 0:
        raise ValueError("Expected a scalar BF16 reference value")
    rank = int(_ordered_bf16(value))
    lower, upper = _decode_ordered(rank - 1), _decode_ordered(rank + 1)
    center = float(value)
    return {
        "lower_value": lower,
        "upper_value": upper,
        "lower_gap": center - lower if lower is not None else None,
        "upper_gap": upper - center if upper is not None else None,
    }


def _step_summary(steps):
    return {
        "count": int(steps.size),
        "max": int(steps.max()),
        "median": float(np.median(steps)),
        "distribution": {
            "0": int(np.count_nonzero(steps == 0)),
            "1": int(np.count_nonzero(steps == 1)),
            "2": int(np.count_nonzero(steps == 2)),
            "3-4": int(np.count_nonzero((steps >= 3) & (steps <= 4))),
            "5-8": int(np.count_nonzero((steps >= 5) & (steps <= 8))),
            "9-16": int(np.count_nonzero((steps >= 9) & (steps <= 16))),
            ">16": int(np.count_nonzero(steps > 16)),
        },
    }


def _metrics(reference, actual, steps):
    residual = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        **tensor_difference(reference, actual),
        "signed_mean_error": float(np.mean(residual)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "exact_count": int(np.count_nonzero(steps == 0)),
        "nonzero_count": int(np.count_nonzero(steps)),
        "bf16_steps": _step_summary(steps),
    }


def analyze_residual(reference_bf16, actual_bf16, positions, top_k=8):
    """Summarize [query, local_head, head_dim] differences; no PASS threshold.

    The signed residual is actual minus reference. Groups use precisely their
    own elements, with zero-reference relative metrics left undefined (null).
    The largest absolute residuals use flattened query/head/dimension order to
    break ties, so repeated analysis of the same snapshot is reproducible.
    """
    reference = _bf16_array(reference_bf16)
    actual = _bf16_array(actual_bf16)
    if reference.shape != actual.shape:
        raise ValueError("Reference and actual shapes differ; no broadcasting")
    if reference.ndim != 3 or min(reference.shape) <= 0:
        raise ValueError("Expected nonempty [query, local_head, head_dim] arrays")
    positions = np.asarray(positions)
    if (
        positions.shape != (reference.shape[0],)
        or positions.dtype.kind not in "iu"
        or np.any(positions < 0)
    ):
        raise ValueError("Expected nonnegative integer positions[query]")
    if (
        isinstance(top_k, (bool, np.bool_))
        or not isinstance(top_k, (int, np.integer))
        or top_k < 0
    ):
        raise ValueError("top_k must be a nonnegative integer")

    steps = bf16_steps(reference, actual)
    per_query_head, per_query, per_head = [], [], []
    for query in range(reference.shape[0]):
        per_query.append(
            {
                "query_index": query,
                "position": int(positions[query]),
                "metrics": _metrics(reference[query], actual[query], steps[query]),
            }
        )
        for head in range(reference.shape[1]):
            per_query_head.append(
                {
                    "query_index": query,
                    "position": int(positions[query]),
                    "head_index": head,
                    "metrics": _metrics(
                        reference[query, head], actual[query, head], steps[query, head]
                    ),
                }
            )
    for head in range(reference.shape[1]):
        per_head.append(
            {
                "head_index": head,
                "metrics": _metrics(
                    reference[:, head], actual[:, head], steps[:, head]
                ),
            }
        )

    residual = actual.astype(np.float64) - reference.astype(np.float64)
    sorted_indices = np.argsort(-np.abs(residual).ravel(), kind="stable")
    top = []
    for flat_index in sorted_indices[:top_k]:
        query, head, dim = (
            int(x) for x in np.unravel_index(flat_index, reference.shape)
        )
        coordinate = (query, head, dim)
        top.append(
            {
                "query_index": query,
                "position": int(positions[query]),
                "head_index": head,
                "dim_index": dim,
                "reference": float(reference[coordinate]),
                "actual": float(actual[coordinate]),
                "signed_error": float(residual[coordinate]),
                "abs_error": float(abs(residual[coordinate])),
                "bf16_step_distance": int(steps[coordinate]),
                "reference_bf16_neighbors": bf16_neighbor_spacing(
                    reference[coordinate]
                ),
            }
        )
    return {
        "overall": _metrics(reference, actual, steps),
        "per_query_head": per_query_head,
        "per_query": per_query,
        "per_head": per_head,
        "top_abs_error": top,
        "notes": [
            "Signed residual is actual minus reference; both inputs are stored BF16 values.",
            "BF16 steps count adjacent representable values, with +0 and -0 merged.",
            "Neighbor spacing is a format property, not proof of NPU subnormal behavior.",
            "No precision PASS, inferred FIA internal arithmetic, or acceptance-rate claim.",
        ],
    }
