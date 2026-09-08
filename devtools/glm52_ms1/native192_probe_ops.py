# SPDX-License-Identifier: Apache-2.0
"""Small compiler probes, imported only by the explicitly run NPU tool."""

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def copy192(input_ptr, output_ptr):
    offsets = tl.arange(0, 192)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, values)


@triton.jit
def reduce192(input_ptr, output_ptr):
    offsets = tl.arange(0, 192)
    values = tl.load(input_ptr + offsets).reshape(1, 192)
    mean_square = tl.sum(values * values, axis=1) / 192
    tl.store(output_ptr + tl.arange(0, 1), mean_square)


@triton.jit
def rotate192(input_ptr, output_ptr):
    offsets = tl.arange(0, 192)
    values = tl.load(input_ptr + offsets).reshape(1, 192)
    left = al.extract_slice(values, offsets=(0, 0), sizes=(1, 96), strides=(1, 1))
    right = al.extract_slice(values, offsets=(0, 96), sizes=(1, 96), strides=(1, 1))
    rotated = tl.zeros((1, 192), dtype=tl.float32)
    rotated = al.insert_slice(
        rotated, -right, offsets=(0, 0), sizes=(1, 96), strides=(1, 1)
    )
    rotated = al.insert_slice(
        rotated, left, offsets=(0, 96), sizes=(1, 96), strides=(1, 1)
    )
    tl.store(output_ptr + offsets, rotated.reshape(192))
