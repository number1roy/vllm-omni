# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only integer checkpoint layout; independent of vLLM and NPU kernels."""

from __future__ import annotations

import torch
from torch import nn


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Signed two's-complement nibbles, first K value in the lowest nibble."""
    if values.device.type != "cpu" or values.dtype != torch.int8 or values.ndim != 2 or values.shape[1] % 8:
        raise ValueError("INT4 packing requires CPU int8 [N,K], K divisible by eight")
    if torch.any(values < -8) or torch.any(values > 7):
        raise ValueError("INT4 values must be in [-8,7]")
    groups = values.reshape(values.shape[0], -1, 8).to(torch.int64)
    packed = torch.zeros(groups.shape[:2], dtype=torch.int64, device="cpu")
    for nibble in range(8):
        packed |= (groups[..., nibble] & 15) << (4 * nibble)
    return packed.to(torch.int32)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError("INT4 packed checkpoint must be an int32 matrix")
    codes = torch.stack([(packed.to(torch.int64) >> (4 * i)) & 15 for i in range(8)], -1)
    return torch.where(codes >= 8, codes - 16, codes).flatten(-2).to(torch.int8)


def prepare_integer_weight(weight: torch.Tensor, scale: torch.Tensor, bits: int):
    """Preserve serialized codes/scales, padding only; all preparation on CPU."""
    if bits not in (4, 8) or weight.ndim != 2 or weight.dtype != torch.int8:
        raise ValueError("serialized integer weight requires int8 codes and bits=4/8")
    n, k = weight.shape
    if min(n, k) <= 0 or scale.dtype != torch.float32 or scale.shape not in {(n,), (n, 1)}:
        raise ValueError("integer scale must be FP32 with exactly one value per output channel")
    weight, scale = weight.cpu(), scale.detach().cpu().reshape(-1)
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("integer weight scale must be finite and positive")
    pn, pk = (n + 31) // 32 * 32, (k + 31) // 32 * 32
    padded = nn.functional.pad(weight, (0, pk - k, 0, pn - n))
    scales = nn.functional.pad(scale, (0, pn - n), value=1)
    if bits == 8:
        return padded, scales, None
    # Bound workspace for wide H3 AdaLN matrices; never materialize their
    # entire eight-times-larger int64 unpack/pack intermediate.
    packed = torch.empty((pn, pk // 8), dtype=torch.int32, device="cpu")
    bound = torch.empty(pn, dtype=torch.float32, device="cpu")
    rows = max(1, 8 * 1024 * 1024 // (pk * 16))
    for start in range(0, pn, rows):
        chunk = padded[start : start + rows]
        packed[start : start + rows] = pack_int4(chunk)
        bound[start : start + rows] = (chunk.to(torch.int32).abs().sum(1) * 8).clamp_min(1).float()
    # Dynamic INT4 can emit -8. Bound every possible integer dot product,
    # including long K, before the A2/A3 FP16-only QMM epilogue.
    qmm_scale = torch.exp2(torch.floor(torch.log2(16384.0 / bound)))
    if not torch.isfinite(scales / qmm_scale).all():
        raise ValueError("INT4 scales exceed the FP32 restoration range")
    return packed, scales, qmm_scale
