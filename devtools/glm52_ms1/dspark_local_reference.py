# SPDX-License-Identifier: Apache-2.0
"""Offline NumPy references for dense DSpark context features and K/V.

These functions consume supplied arrays. They do not capture activations, load
checkpoints, import SGLang/Triton/torch-npu, contact a server, or write a KV pool.
Use the same actual input at each boundary to isolate an operation; separately
compose the functions to study accumulated differences.

bf16_storage rounds OUTPUT boundaries only: inputs, weights and RoPE cache rows
must already represent the snapshot being compared. This models storage, not
NPU accumulation order. The scope is bias-free dense GLM context injection with
full-head RoPE and RMSNorm that multiplies its weight before casting the output.
No function assigns a model/kernel precision PASS or acceptance-rate threshold.
"""

import math

import numpy as np


def _dtype(compute_dtype):
    dtype = np.dtype(compute_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Reference compute dtype must be float32 or float64")
    return dtype


def _array(values, dtype):
    values = np.asarray(values)
    if values.dtype.kind != "f":
        raise ValueError("Expected floating point snapshot values")
    return values.astype(dtype, copy=False)


def bf16_round(values):
    """FP32 -> BF16, ties to even, represented as FP32; never mutate input.

    This is the same storage convention as probe_quarot_fc, kept standalone here
    so the reference has no dependency on checkpoint/HTTP diagnostic modules.
    """
    values = np.array(values, dtype=np.float32, copy=True)
    bits = values.view(np.uint32)
    with np.errstate(over="ignore"):
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(
            0xFFFF0000
        )
    rounded = np.where(np.isnan(values), np.uint32(0x7FC00000), rounded)
    return rounded.astype(np.uint32).view(np.float32)


def _stored(values, bf16_storage):
    return bf16_round(values) if bf16_storage else values


def linear(inputs, weight, bias=None, *, compute_dtype=np.float32):
    """[tokens, in] @ [out, in].T (+ [out]); no implicit input rounding."""
    dtype = _dtype(compute_dtype)
    x, w = _array(inputs, dtype), _array(weight, dtype)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("Expected matching inputs[T, in] and weight[out, in]")
    if min(w.shape) <= 0:
        raise ValueError("Weight dimensions must be positive")
    result = x @ w.T
    if bias is not None:
        b = _array(bias, dtype)
        if b.shape != (w.shape[0],):
            raise ValueError("Expected bias[out]; broadcasting is not allowed")
        result = result + b
    return result


def rms_norm(values, weight, eps, *, compute_dtype=np.float32, bf16_storage=False):
    """Normalize the last dimension; multiply learned weight, then cast.

    The caller supplies the model epsilon explicitly. Other RMSNorm conventions
    (residual addition, variance override, or casting before weight) are not used
    by this reference and must not be silently substituted at snapshot replay.
    """
    dtype = _dtype(compute_dtype)
    x, w = _array(values, dtype), _array(weight, dtype)
    if x.ndim < 2 or x.shape[-1] <= 0 or w.shape != (x.shape[-1],):
        raise ValueError("Expected values[..., width] and norm weight[width]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Expected the model's finite positive epsilon")
    variance = np.mean(x * x, axis=-1, keepdims=True)
    normalized = x / np.sqrt(variance + dtype.type(eps))
    return _stored(normalized * w, bf16_storage)


def project_context(
    hidden,
    fc_weight,
    norm_weight,
    eps,
    *,
    compute_dtype=np.float32,
    bf16_storage=False,
):
    """FC -> hidden_norm, returning both boundary outputs for inspection."""
    projected = _stored(
        linear(hidden, fc_weight, compute_dtype=compute_dtype), bf16_storage
    )
    context = rms_norm(
        projected,
        norm_weight,
        eps,
        compute_dtype=compute_dtype,
        bf16_storage=bf16_storage,
    )
    return {"fc_output": projected, "context": context}


def rope_cos_sin(
    positions, head_dim, base, *, compute_dtype=np.float32, bf16_cache=False
):
    """Default full-head RoPE rows at selected positions; no full cache build.

    This reference has no scaling, partial rotary dimension, or position offsets.
    Use actual saved cos/sin rows with apply_rope to isolate rotation arithmetic;
    use this generator separately to check cache construction and position mapping.
    """
    dtype = _dtype(compute_dtype)
    pos = np.asarray(positions)
    if pos.ndim != 1 or pos.dtype.kind not in "iu" or np.any(pos < 0):
        raise ValueError("Expected nonnegative integer positions[T]")
    if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
        raise ValueError("Expected a positive even head dimension")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("Expected a finite positive RoPE base")
    exponent = np.arange(0, head_dim, 2, dtype=dtype) / dtype.type(head_dim)
    inv_freq = dtype.type(1) / np.power(dtype.type(base), exponent)
    angles = pos.astype(dtype)[:, None] * inv_freq[None, :]
    return _stored(np.cos(angles), bf16_cache), _stored(np.sin(angles), bf16_cache)


def apply_rope(
    keys, cos, sin, *, is_neox_style, compute_dtype=np.float32, bf16_storage=False
):
    """Rotate K[T, heads, D] with cos/sin[T, D/2], without mutating K.

    NeoX pairs (i, i+D/2); interleaved pairs (2i, 2i+1). Cache dtype is a property
    of supplied arrays, independent of output storage dtype.
    """
    dtype = _dtype(compute_dtype)
    k, c, s = (_array(a, dtype) for a in (keys, cos, sin))
    if k.ndim != 3 or k.shape[-1] <= 0 or k.shape[-1] % 2:
        raise ValueError("Expected keys[T, heads, positive even D]")
    if c.shape != (k.shape[0], k.shape[2] // 2) or s.shape != c.shape:
        raise ValueError("Expected cos and sin[T, D/2]")
    if type(is_neox_style) is not bool:
        raise ValueError("RoPE pairing must be explicit")
    half = k.shape[-1] // 2
    first = slice(None, half) if is_neox_style else slice(0, None, 2)
    second = slice(half, None) if is_neox_style else slice(1, None, 2)
    a, b = k[..., first], k[..., second]
    c, s = c[:, None, :], s[:, None, :]
    result = np.empty_like(k)
    result[..., first] = a * c - b * s
    result[..., second] = b * c + a * s
    return _stored(result, bf16_storage)


def context_kv(
    context,
    k_weights,
    v_weights,
    k_norm_weights,
    cos,
    sin,
    *,
    head_dim,
    eps,
    is_neox_style,
    compute_dtype=np.float32,
    bf16_storage=False,
):
    """Per-layer bias-free K/V calculation, returning [layers,T,heads,D].

    Weights are separate K/V snapshots [layers, local_heads*D, hidden], already
    mapped to the selected TP rank. This function does not copy the runtime QKV
    slicing logic; correctness of checkpoint-to-rank mapping is a separate check.
    No pool write, index translation, layer selection or commit mask is inferred.
    """
    dtype = _dtype(compute_dtype)
    ctx, kw, vw, nw = (
        _array(a, dtype) for a in (context, k_weights, v_weights, k_norm_weights)
    )
    if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
        raise ValueError("Expected a positive even head dimension")
    if (
        ctx.ndim != 2
        or kw.ndim != 3
        or vw.shape != kw.shape
        or kw.shape[0] == 0
        or kw.shape[1] == 0
        or kw.shape[2] != ctx.shape[1]
        or kw.shape[1] % head_dim
        or nw.shape != (kw.shape[0], head_dim)
    ):
        raise ValueError("Incompatible context, K/V weights or per-layer norm weights")
    outputs = {
        name: [] for name in ("k_projected", "v_projected", "k_normed", "k_rope")
    }
    heads = kw.shape[1] // head_dim
    for layer in range(kw.shape[0]):
        k = _stored(linear(ctx, kw[layer], compute_dtype=dtype), bf16_storage)
        v = _stored(linear(ctx, vw[layer], compute_dtype=dtype), bf16_storage)
        k = k.reshape(ctx.shape[0], heads, head_dim)
        v = v.reshape(ctx.shape[0], heads, head_dim)
        normed = rms_norm(
            k, nw[layer], eps, compute_dtype=dtype, bf16_storage=bf16_storage
        )
        roped = apply_rope(
            normed,
            cos,
            sin,
            is_neox_style=is_neox_style,
            compute_dtype=dtype,
            bf16_storage=bf16_storage,
        )
        for name, value in zip(outputs, (k, v, normed, roped)):
            outputs[name].append(value)
    return {name: np.stack(rows) for name, rows in outputs.items()}


def tensor_difference(reference, actual):
    """Continuous diagnostics only; no tolerance, automatic fix or PASS.

    Shapes must match exactly: broadcasting could hide layer/head/token swaps.
    Undefined zero-norm metrics and nonfinite metrics are None, suitable for JSON.
    """
    ref, act = _array(reference, np.float64), _array(actual, np.float64)
    if ref.shape != act.shape:
        raise ValueError("Reference and actual shapes differ")
    finite = bool(np.isfinite(ref).all() and np.isfinite(act).all())
    result = {
        "shape": list(ref.shape),
        "numel": int(ref.size),
        "finite": finite,
        "metrics_finite": False,
        "exact_equal": bool(np.array_equal(ref, act)),
        "reference_norm": None,
        "actual_norm": None,
        "norm_ratio": None,
        "relative_l2": None,
        "cosine": None,
        "max_abs_error": None,
    }
    if not finite or ref.size == 0:
        return result
    r, a = ref.ravel(), act.ravel()
    with np.errstate(over="ignore", invalid="ignore"):
        rn, an = float(np.linalg.norm(r)), float(np.linalg.norm(a))
        error = float(np.linalg.norm(a - r))
        max_error = float(np.max(np.abs(a - r)))
    if not all(math.isfinite(x) for x in (rn, an, error, max_error)):
        return result
    result.update(
        metrics_finite=True,
        reference_norm=rn,
        actual_norm=an,
        norm_ratio=an / rn if rn else None,
        relative_l2=error / rn if rn else None,
        cosine=float(np.dot(r / rn, a / an)) if rn and an else None,
        max_abs_error=max_error,
    )
    for name, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            result[name] = None
            result["metrics_finite"] = False
    return result
