# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend native MX execution for ModelSlim H3 component deliveries.

Delivery dual-scale FP4 uses groups of 32, unlike the runtime's dual-level
kernel (512). The reader applies both scales before this adapter converts
weights once to native single-scale MX. No floating GEMM fallback is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.utils import replace_parameter

from vllm_omni.quantization.int8_config import LazyWeightMixin

if TYPE_CHECKING:
    from .offline_quantization import OfflineComponentCheckpoint

logger = init_logger(__name__)
_MX4 = "W4A4_MXFP4_DUALSCALE"
_MX8 = "W8A8_MXFP8"
_OUTPUT_CHUNK = 32768
_EXECUTED_BITS: set[int] = set()


def _target_bits(scheme: str) -> int | None:
    if scheme == _MX4:
        return 4
    return 8 if scheme == _MX8 else None


@torch.no_grad()
def quantize_mx_weight(weight: torch.Tensor, bits: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a reconstructed matrix once; return plain transportable bytes."""
    import torch_npu

    if device.type != "npu" or bits not in (4, 8):
        raise ValueError("H3 native MX requires an NPU and bits=4 or bits=8")
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("native MX weight must be a finite matrix")
    padded_k = (weight.shape[1] + 63) // 64 * 64
    dtype = torch_npu.float4_e2m1fn_x2 if bits == 4 else torch_npu.float8_e4m3fn
    weights, scales = [], []
    for chunk in weight.split(_OUTPUT_CHUNK, dim=0):
        chunk = chunk.to(device=device, dtype=torch.bfloat16)
        chunk = nn.functional.pad(chunk, (0, padded_k - weight.shape[1]))
        q, scale = torch_npu.npu_dynamic_mx_quant(
            chunk, dst_type=dtype, axis=-1, block_size=32, round_mode="rint", scale_alg=0
        )
        weights.append(q.view(torch.uint8))
        scales.append(scale.reshape(chunk.shape[0], -1, 2))
    return torch.cat(weights).contiguous(), torch.cat(scales).contiguous()


def native_mx_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bits: int, bias: torch.Tensor | None
) -> torch.Tensor:
    """Dynamic MX activations and native low-bit GEMM, including wide H3 AdaLN."""
    import torch_npu

    if x.device.type != "npu" or weight.device != x.device or scale.device != x.device:
        raise ValueError("native MX inputs, weight, and scale must reside on the same NPU")
    shape, output_dtype = x.shape, x.dtype
    padded_k = weight.shape[1] * (2 if bits == 4 else 1)
    if shape[-1] > padded_k:
        raise ValueError("native MX input exceeds the prepared weight width")
    x = x.reshape(-1, shape[-1]).to(torch.bfloat16)
    x = nn.functional.pad(x, (0, padded_k - shape[-1]))
    mx_dtype = torch_npu.float4_e2m1fn_x2 if bits == 4 else torch_npu.float8_e4m3fn
    xq, xs = torch_npu.npu_dynamic_mx_quant(
        x, dst_type=mx_dtype, axis=-1, block_size=32, round_mode="rint", scale_alg=0
    )
    outputs = []
    for offset in range(0, weight.shape[0], _OUTPUT_CHUNK):
        w = weight[offset : offset + _OUTPUT_CHUNK]
        if bits == 8:
            w = w.view(torch.float8_e4m3fn)
        # Keep BF16 output for the documented native MX kernel contract. Audio
        # VAE's public FP32 boundary is restored after the low-bit multiply.
        y = torch_npu.npu_quant_matmul(
            xq,
            w.transpose(0, 1),
            scale[offset : offset + _OUTPUT_CHUNK].transpose(0, 1),
            x1_dtype=mx_dtype,
            x2_dtype=mx_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=xs,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            output_dtype=torch.bfloat16,
            group_sizes=[1, 1, 32],
        )
        outputs.append(y)
    output = torch.cat(outputs, dim=-1).to(output_dtype)
    if bits not in _EXECUTED_BITS:
        logger.info(
            "H3 native MX executed W%dA%d: npu_dynamic_mx_quant + npu_quant_matmul; "
            "activation=%s weight_storage=%s output=%s group=32",
            bits,
            bits,
            xq.dtype,
            weight.dtype,
            output.dtype,
        )
        _EXECUTED_BITS.add(bits)
    if bias is not None:
        output = output + bias.to(output_dtype)
    return output.reshape(*shape[:-1], weight.shape[0])


class NativeMXLinear(nn.Module):
    """VAE Linear with byte parameters unaffected by outer FP32 dtype casts."""

    def __init__(self, linear: nn.Linear, bits: int, device: torch.device) -> None:
        super().__init__()
        self.in_features, self.out_features, self.bits = linear.in_features, linear.out_features, bits
        weight, scale = quantize_mx_weight(linear.weight, bits, device)
        storage_device = linear.weight.device
        self.weight = nn.Parameter(weight.to(storage_device), requires_grad=False)
        self.weight_scale = nn.Parameter(scale.to(storage_device), requires_grad=False)
        self.register_parameter("bias", linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return native_mx_linear(x, self.weight, self.weight_scale, self.bits, self.bias)


def install_native_mx_vae(model: nn.Module, checkpoint: OfflineComponentCheckpoint, device: torch.device) -> int:
    """Replace only delivery-quantized Linear modules before staging is built."""
    count = 0
    for name in checkpoint.tensors:
        if not name.endswith(".weight"):
            continue
        bits = _target_bits(checkpoint.description.get(name, "FLOAT"))
        if bits is None:
            continue
        path = name.removesuffix(".weight")
        linear = model.get_submodule(path)
        if not isinstance(linear, nn.Linear):
            raise ValueError(f"native MX VAE weight does not target Linear: {path}")
        parent_path, _, child = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child, NativeMXLinear(linear, bits, device))
        count += 1
    logger.info("Installed H3 native MX VAE mode=%s matrices=%d", checkpoint.execution_mode, count)
    return count


class NativeMXLinearMethod(LazyWeightMixin, LinearMethodBase):
    """TP-aware lazy loading followed by native MX conversion once per layer."""

    def __init__(self, bits: int) -> None:
        self.bits = bits

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer._native_mx_storage_device = layer._load_device

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        from vllm_omni.diffusion.distributed.utils import get_local_device

        storage_device = layer._native_mx_storage_device
        weight, scale = quantize_mx_weight(layer.weight, self.bits, get_local_device())
        replace_parameter(layer, "weight", weight.to(storage_device))
        layer.register_parameter("weight_scale", nn.Parameter(scale.to(storage_device), requires_grad=False))
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return native_mx_linear(x, layer.weight, layer.weight_scale, self.bits, bias)


class H3NativeMXConfig(QuantizationConfig):
    """Delivery-driven model-local MX config; no global quantization override."""

    def __init__(self, targets: dict[str, int]) -> None:
        super().__init__()
        self.targets = targets

    def get_name(self) -> str:
        return "h3_native_mx"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> H3NativeMXConfig:
        raise ValueError("H3 native MX is selected through offline_quantization.execution_mode")

    def get_quant_method(self, layer: nn.Module, prefix: str) -> LinearMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        bits = self.targets.get(prefix.removeprefix("transformer."))
        return NativeMXLinearMethod(bits) if bits else UnquantizedLinearMethod()

    def validate_model_targets(self, model: nn.Module) -> None:
        installed = {
            name: module.quant_method.bits
            for name, module in model.named_modules()
            if isinstance(getattr(module, "quant_method", None), NativeMXLinearMethod)
        }
        if installed != self.targets:
            missing = sorted(self.targets.keys() - installed.keys())
            raise ValueError(f"native MX targets must all use native methods; missing={missing}")

    @classmethod
    def from_checkpoint(cls, checkpoint: OfflineComponentCheckpoint) -> H3NativeMXConfig:
        from .fasth3 import _resolve_native_target

        schemes: dict[str, set[str]] = {}
        for name in checkpoint.tensors:
            if not name.endswith(".weight"):
                continue
            module = name.removesuffix(".weight")
            if checkpoint.diffusers_weights:
                target = _resolve_native_target(module)
                if target is None:
                    raise ValueError(f"unsupported native MX H3 projection: {module}")
                module = target[0]
            schemes.setdefault(module, set()).add(checkpoint.description.get(name, "FLOAT"))
        targets = {}
        for name, values in schemes.items():
            if len(values) != 1:
                raise ValueError(f"native MX fused projections must share one delivery scheme: {name}")
            if (bits := _target_bits(next(iter(values)))) is not None:
                targets[name] = bits
        logger.info("H3 native MX mode=%s target_modules=%d", checkpoint.execution_mode, len(targets))
        return cls(targets)
