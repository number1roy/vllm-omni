# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A2/A3 real integer GEMM oracle using independently decoded actual operands."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks

torch_npu = pytest.importorskip("torch_npu")
from vllm_omni.diffusion.models.minimax_h3.integer_utils import prepare_integer_weight  # noqa: E402
from vllm_omni.diffusion.models.minimax_h3.native_int import native_int_linear  # noqa: E402
from vllm_omni.diffusion.models.minimax_h3.offline_quantization import resolve_npu_quant_family  # noqa: E402

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"npu": "A3"}, num_cards=1),
    pytest.mark.skipif(
        not torch.npu.is_available() or resolve_npu_quant_family() != "int",
        reason="requires an A2/A3 integer NPU; A5 does not validate this contract",
    ),
]


def _decode(words):
    # Deliberately independent of the production unpack helper.
    words = words.cpu().to(torch.int64)
    shifts = torch.arange(0, 32, 4, device="cpu")
    codes = (words.unsqueeze(-1) >> shifts) & 15
    return (codes - (codes >= 8).to(torch.int64) * 16).flatten(-2).float()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n,k", [(35, 65), (65568, 64), (33, 16384)])
@torch.no_grad()
def test_actual_integer_operands_scale_offsets_padding_and_float_boundaries(bits, dtype, n, k):
    torch.manual_seed(113)
    device = torch.device("npu:0")
    low, high = (-8, 8) if bits == 4 else (-127, 128)
    codes = torch.randint(low, high, (n, k), dtype=torch.int8, device="cpu")
    scales = (torch.arange(n, device="cpu") % 31 + 1).float() * 0.00010003
    w, s, qs = prepare_integer_weight(codes, scales, bits)
    # Exercise DLO's nonzero byte/element offsets and later wide chunks.
    w = torch.cat((torch.zeros_like(w[:3]), w)).to(device)[3:]
    s = torch.cat((torch.ones(3, device="cpu"), s)).to(device)[3:]
    if qs is not None:
        qs = torch.cat((torch.ones(3, device="cpu"), qs)).to(device)[3:]
    base = torch.randn(4, k * 2, device="cpu")
    base[0].zero_()
    base[1] *= 1e-5
    base[2] *= 20
    x = base.to(device=device, dtype=dtype)[:, ::2]
    bias = torch.linspace(-0.1, 0.1, n, device=device, dtype=dtype)
    y = native_int_linear(x, w, s, bits, bias, n, k, qs)
    bridge = F.pad(x.to(torch.bfloat16), (0, w.shape[1] * (8 if bits == 4 else 1) - k))
    qx, sx = torch_npu.npu_dynamic_quant(bridge, dst_type=torch.quint4x2 if bits == 4 else torch.int8)
    # Never infer hardware rounding from x/scale: use the actual emitted codes.
    operands = _decode(qx) if bits == 4 else qx.cpu().float()
    accumulator = operands[:, :k].double() @ codes.double().T
    if bits == 4:
        bounded = (accumulator * qs[:n].cpu().double()).half().float()
        reference = bounded * sx.cpu()[:, None] * (scales / qs[:n].cpu())[None, :]
    else:
        reference = (accumulator.float() * sx.cpu()[:, None] * scales[None, :]).bfloat16().float()
    reference = reference.to(dtype) + bias.cpu()
    assert y.dtype == dtype and y.shape == (4, n) and torch.isfinite(y).all()
    torch.testing.assert_close(y.cpu(), reference, rtol=0.008, atol=0.003)
    norm = F.layer_norm(y.float(), (n,))
    assert norm.dtype == torch.float32 and torch.isfinite(norm).all()
