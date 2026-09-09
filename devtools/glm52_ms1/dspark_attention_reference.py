# SPDX-License-Identifier: Apache-2.0
"""CPU-only local proposal reference; no runtime imports or acceptance threshold."""

import numpy as np
from dspark_local_reference import bf16_round, rms_norm


def paged_slots(blocks, page_size, length, capacity):
    blocks = np.asarray(blocks)
    if blocks.ndim != 1 or blocks.dtype.kind not in "iu":
        raise ValueError("Expected one integral block-table row")
    if page_size <= 0 or not 0 < length <= 128 or capacity <= 0:
        raise ValueError("Outside short-request diagnostic geometry")
    pages = (length + page_size - 1) // page_size
    if pages > blocks.size:
        raise ValueError("Block table does not cover actual KV length")
    selected = blocks[:pages].astype(np.int64)
    if np.any(selected < 0) or np.any(
        selected >= (capacity + page_size - 1) // page_size
    ):
        raise ValueError("Block outside pool")
    slots = (selected[:, None] * page_size + np.arange(page_size)).ravel()[:length]
    if np.any(slots >= capacity):
        raise ValueError("Slot outside pool")
    return slots


def slot_contract(expected, actual, current, prefix):
    expected, actual, current = (
        np.asarray(a, dtype=np.int64) for a in (expected, actual, current)
    )
    return {
        "same_length": expected.size == actual.size,
        "same_ordered_slots": bool(np.array_equal(expected, actual)),
        "actual_unique_slots": np.unique(actual).size == actual.size,
        "expected_unique_slots": np.unique(expected).size == expected.size,
        "current_matches_request_tail": bool(
            np.array_equal(current, expected[prefix:])
        ),
        "missing_slots": sorted(set(expected.tolist()) - set(actual.tolist())),
        "extra_slots": sorted(set(actual.tolist()) - set(expected.tolist())),
    }


def fused_prepare(
    qkv, q_weight, k_weight, cos, sin, eps, heads, head_dim, *, bf16_storage
):
    """Current host's full-head NeoX path: FP32 norm+RoPE, ONE output cast.

    Unlike the stacked context path, there is no BF16 store between norm and
    RoPE. Cache values are supplied as actually consumed; do not regenerate them.
    """
    x = np.asarray(qkv, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 * heads * head_dim:
        raise ValueError("Expected equal local Q/K/V heads")
    t = x.shape[0]
    c, s = (np.asarray(a, dtype=np.float32).reshape(t, -1) for a in (cos, sin))
    if c.shape != (t, head_dim) or s.shape != c.shape or head_dim % 2:
        raise ValueError("Expected full-head sin/cos")
    q, k, v = [part.reshape(t, heads, head_dim) for part in np.split(x, 3, axis=-1)]

    def prepare(a, weight):
        n = rms_norm(a, weight, eps)
        half = head_dim // 2
        rotated = np.concatenate((-n[..., half:], n[..., :half]), axis=-1)
        value = rotated * s[:, None, :] + n * c[:, None, :]
        return bf16_round(value) if bf16_storage else value

    return prepare(q, q_weight), prepare(k, k_weight), v.copy()


def attention(q, k, v, scale, *, visible=None, compute_dtype=np.float64):
    """Q[T,H,D], K/V[S,H,D]; explicit stable softmax on each head.

    Optional visibility is a boolean [T,S] mask (True = may attend), used in
    calibration to distinguish full-block from causal attention.
    """
    q, k, v = (np.asarray(a, dtype=compute_dtype) for a in (q, k, v))
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError("Incompatible Q/K/V")
    if (
        not all(np.isfinite(a).all() for a in (q, k, v))
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Non-finite attention input/invalid scale")
    scores = np.einsum("thd,shd->hts", q, k) * scale
    if visible is not None:
        visible = np.asarray(visible)
        if (
            visible.dtype != bool
            or visible.shape != (q.shape[0], k.shape[0])
            or not visible.any(axis=1).all()
        ):
            raise ValueError("Invalid visibility or fully masked query")
        scores = np.where(visible[None], scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("hts,shd->thd", probs, v)
