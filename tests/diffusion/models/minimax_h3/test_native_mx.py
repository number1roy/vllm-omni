# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for delivery-driven native mixed MX loading."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod

from vllm_omni.diffusion.models.minimax_h3 import native_mx

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _checkpoint(description):
    return SimpleNamespace(
        tensors=description,
        description=description,
        execution_mode="native_mx",
        diffusers_weights=True,
    )


def test_delivery_bits_and_fused_mapping_are_preserved(monkeypatch):
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    description = {f"transformer_blocks.2.attn.to_{role}.weight": "W4A4_MXFP4_DUALSCALE" for role in "qkv"}
    description["transformer_blocks.2.ff.net.2.weight"] = "W8A8_MXFP8"
    description["proj_in.weight"] = "FLOAT"
    config = native_mx.H3NativeMXConfig.from_checkpoint(_checkpoint(description))
    assert config.targets == {"blocks.2.attn.qkv_proj": 4, "blocks.2.mlp.fc2": 8}
    for prefix, bits in (("blocks.2.attn.qkv_proj", 4), ("transformer.blocks.2.mlp.fc2", 8)):
        layer = ReplicatedLinear(64, 32, quant_config=config, prefix=prefix)
        assert isinstance(layer.quant_method, native_mx.NativeMXLinearMethod)
        assert layer.quant_method.bits == bits
    ordinary = ReplicatedLinear(64, 32, quant_config=config, prefix="video_patch_proj")
    assert isinstance(ordinary.quant_method, UnquantizedLinearMethod)


def test_fused_mixed_schemes_fail_before_loading():
    description = {
        "transformer_blocks.2.attn.to_q.weight": "W4A4_MXFP4_DUALSCALE",
        "transformer_blocks.2.attn.to_k.weight": "W8A8_MXFP8",
    }
    with pytest.raises(ValueError, match="must share one delivery scheme"):
        native_mx.H3NativeMXConfig.from_checkpoint(_checkpoint(description))


@pytest.mark.parametrize("bits", [4, 8])
@torch.no_grad()
def test_lazy_loading_registers_runtime_scale_after_complete_weight(monkeypatch, bits):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm_omni.diffusion.distributed.utils.get_local_device", lambda: torch.device("cpu"))
    calls = []

    def quantize(weight, actual_bits, device):
        calls.append((weight.clone(), actual_bits, device))
        return torch.zeros(8, 32 if bits == 4 else 64, dtype=torch.uint8), torch.ones(8, 1, 2, dtype=torch.uint8)

    monkeypatch.setattr(native_mx, "quantize_mx_weight", quantize)
    method = native_mx.NativeMXLinearMethod(bits)
    layer = nn.Module()
    method.create_weights(layer, 24, [8], 24, 8, torch.bfloat16, weight_loader=lambda p, w: p.copy_(w))
    assert layer.weight.device.type == "meta"
    weight = torch.arange(8 * 24, dtype=torch.bfloat16).reshape(8, 24)
    layer.weight.weight_loader(layer.weight, weight)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], weight)
    assert calls[0][1] == bits
    assert layer.weight.dtype == layer.weight_scale.dtype == torch.uint8
    assert layer.weight.device.type == layer.weight_scale.device.type == "cpu"
    method.process_weights_after_loading(layer)
    assert len(calls) == 1


@pytest.mark.parametrize("bits", [4, 8])
def test_vae_byte_parameters_survive_outer_float_cast(monkeypatch, bits):
    width = 32 if bits == 4 else 64
    weight = torch.arange(3 * width, dtype=torch.uint8).reshape(3, width)
    scale = torch.arange(121, 127, dtype=torch.uint8).reshape(3, 1, 2)
    monkeypatch.setattr(native_mx, "quantize_mx_weight", lambda *_: (weight.clone(), scale.clone()))
    linear = nn.Linear(24, 3)
    replacement = native_mx.NativeMXLinear(linear, bits, torch.device("cpu")).to(dtype=torch.float32)
    assert replacement.weight.dtype == replacement.weight_scale.dtype == torch.uint8
    torch.testing.assert_close(replacement.weight, weight)
    torch.testing.assert_close(replacement.weight_scale, scale)
    assert replacement.bias is linear.bias


def test_native_kernel_never_silently_falls_back_to_cpu():
    pytest.importorskip("torch_npu")
    with pytest.raises(ValueError, match="same NPU"):
        native_mx.native_mx_linear(
            torch.ones(2, 64),
            torch.zeros(3, 64, dtype=torch.uint8),
            torch.zeros(3, 1, 2, dtype=torch.uint8),
            8,
            None,
        )
