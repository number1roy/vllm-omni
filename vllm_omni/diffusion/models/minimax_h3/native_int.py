# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 serialized dynamic INT4/INT8. No floating GEMM or weight requantization."""

from __future__ import annotations

from functools import lru_cache
from types import MethodType

import torch
from torch import nn
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.utils import replace_parameter

from vllm_omni.quantization.int8_config import BaseInt8LinearMethod, DiffusionInt8Config

from .integer_utils import pack_int4, prepare_integer_weight
from .native_mx import _OUTPUT_CHUNK, H3NativeMXConfig, NativeMXLinear, _install_native_mx_vae_callers, logger

_INT4 = "W4A4_DYNAMIC"
_INT8 = "W8A8_DYNAMIC"
_EXECUTED_BITS: set[int] = set()


@lru_cache(maxsize=16)
def _verify_pack_abi(device: str) -> None:
    import torch_npu

    codes = (torch.arange(32 * 32, device="cpu").reshape(32, 32) % 16 - 8).to(torch.int8)
    actual = torch_npu.npu_convert_weight_to_int4pack(codes.to(device=device, dtype=torch.int32)).cpu()
    if not torch.equal(actual, pack_int4(codes)):
        raise RuntimeError("NPU INT4 packing ABI differs from H3 checkpoint layout")


def native_int_linear(x, weight, scale, bits, bias, logical_n, logical_k, qmm_scale=None):
    """Native integer GEMM with floating public boundaries and current DLO views."""
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("native INT activations must be FP16, BF16, or FP32")
    if bits not in (4, 8) or x.ndim < 1 or x.shape[-1] != logical_k:
        raise ValueError("native INT activation does not match the prepared layout")
    if x.device.type != "npu" or weight.device != x.device or scale.device != x.device:
        raise ValueError("native INT inputs, weight and scale must reside on the same NPU")
    if weight.ndim != 2 or weight.dtype != (torch.int32 if bits == 4 else torch.int8) or scale.dtype != torch.float32:
        raise ValueError("native INT storage/scale dtype mismatch")
    if scale.shape != (weight.shape[0],) or not 0 < logical_n <= weight.shape[0]:
        raise ValueError("native INT output channel/scale mismatch")
    if bias is not None and (bias.shape != (logical_n,) or bias.device != x.device or not bias.is_floating_point()):
        raise ValueError("native INT bias must match floating output channels on the same NPU")
    if bits == 4 and (
        qmm_scale is None
        or qmm_scale.shape != scale.shape
        or qmm_scale.dtype != torch.float32
        or qmm_scale.device != x.device
    ):
        raise ValueError("native INT4 bounded epilogue scales are missing or incompatible")
    import torch_npu

    from .offline_quantization import require_native_family

    require_native_family("int", x.device)
    if bits == 4:
        _verify_pack_abi(str(x.device))
    shape, dtype = x.shape, x.dtype
    pk = weight.shape[1] * (8 if bits == 4 else 1)
    if pk < logical_k:
        raise ValueError("native INT packed width is smaller than the logical K")
    if x.numel() == 0:
        return x.new_empty((*shape[:-1], logical_n))
    bridge = nn.functional.pad(x.reshape(-1, logical_k).to(torch.bfloat16), (0, pk - logical_k))
    qx, sx = torch_npu.npu_dynamic_quant(bridge, dst_type=torch.quint4x2 if bits == 4 else torch.int8)
    if qx.dtype != weight.dtype or sx.dtype != torch.float32:
        raise RuntimeError("NPU dynamic integer quantization violated the storage contract")
    sx = torch.where(sx == 0, torch.ones_like(sx), sx)
    outputs = []
    for start in range(0, logical_n, _OUTPUT_CHUNK):
        end = min(start + _OUTPUT_CHUNK, weight.shape[0])
        # Rebase BEFORE transpose. Never make the packed transpose contiguous:
        # that changes which logical axis is packed. DLO offsets are not cached.
        w = weight[start:end]
        if w.storage_offset() != 0 or not w.is_contiguous():
            w = w.clone(memory_format=torch.contiguous_format)
        if bits == 4:
            qs = qmm_scale[start:end]
            y = torch_npu.npu_quant_matmul(qx, w.T, qs, output_dtype=torch.float16).float()
            # Restore in FP32 before casting back for norm/attention/residual.
            y = y * sx[:, None] * (scale[start:end] / qs)[None, :]
        else:
            y = torch_npu.npu_quant_matmul(
                qx, w.T, scale[start:end], pertoken_scale=sx, output_dtype=torch.bfloat16
            ).float()
        outputs.append(y)
    output = torch.cat(outputs, -1)[:, :logical_n].to(dtype)
    if bias is not None:
        output = output + bias.to(dtype)
    if bits not in _EXECUTED_BITS:
        logger.info("H3 native INT executed W%dA%d: npu_dynamic_quant + npu_quant_matmul; output=%s", bits, bits, dtype)
        _EXECUTED_BITS.add(bits)
    return output.reshape(*shape[:-1], logical_n)


def _apply_preserving_scales(module, original, fn, recurse=True):
    # The dtype of integer storage is not an activation dtype. Keep the exact
    # FP32 scales aside during .half()/.bfloat16()/.to(dtype), then move them
    # only to the resulting weight device. Never downcast then upcast a scale.
    scales = {
        name: module._parameters.pop(name) for name in ("weight_scale", "qmm_scale") if name in module._parameters
    }
    try:
        result = original(fn, recurse)
        for name, value in scales.items():
            if value is not None:
                value.data = value.data.to(device=module.weight.device)
            module.register_parameter(name, value)
        return result
    except BaseException:
        module._parameters.update(scales)
        raise


def _integer_layer_apply(module, fn, recurse=True):
    return _apply_preserving_scales(module, module._h3_integer_original_apply, fn, recurse)


class NativeIntLinear(NativeMXLinear):
    """Reuse the audited VAE callers, with integer storage and FP32 scales."""

    def __init__(self, linear, bits, weight, scale):
        nn.Module.__init__(self)
        self.in_features, self.out_features, self.bits = linear.in_features, linear.out_features, bits
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError("integer VAE checkpoint does not match the Linear shape")
        q, s, qs = prepare_integer_weight(weight, scale, bits)
        self.weight = nn.Parameter(q.to(linear.weight.device), requires_grad=False)
        self.weight_scale = nn.Parameter(s.to(linear.weight.device), requires_grad=False)
        self.register_parameter(
            "qmm_scale", None if qs is None else nn.Parameter(qs.to(linear.weight.device), requires_grad=False)
        )
        self.register_parameter("bias", linear.bias)

    def _apply(self, fn, recurse=True):
        return _apply_preserving_scales(self, super()._apply, fn, recurse)

    def forward(self, x):
        return native_int_linear(
            x, self.weight, self.weight_scale, self.bits, self.bias, self.out_features, self.in_features, self.qmm_scale
        )


def install_native_int_vae(model, checkpoint, device):
    count = 0
    for name in checkpoint.tensors:
        scheme = checkpoint.description.get(name, "FLOAT")
        if not name.endswith(".weight") or scheme not in {_INT4, _INT8}:
            continue
        path = name.removesuffix(".weight")
        linear = model.get_submodule(path)
        if not isinstance(linear, nn.Linear):
            raise ValueError(f"native INT VAE weight does not target Linear: {path}")
        weight, scale = checkpoint.integer_weight(name)
        parent_path, _, child = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child, NativeIntLinear(linear, 4 if scheme == _INT4 else 8, weight, scale))
        count += 1
    if count:
        _install_native_mx_vae_callers(model)
    logger.info("Installed H3 native INT VAE mode=%s matrices=%d", checkpoint.execution_mode, count)
    return count


class NativeIntLinearMethod(BaseInt8LinearMethod):
    def __init__(self, bits):
        super().__init__(DiffusionInt8Config(is_checkpoint_int8_serialized=True))
        if bits not in (4, 8):
            raise ValueError("native integer method requires bits=4/8")
        self.bits = bits

    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        # Preserve channel scales even when a parent is cast before or after
        # loading. This also covers direct VAE wrappers through the same helper.
        layer._h3_integer_original_apply = layer._apply
        layer._apply = MethodType(_integer_layer_apply, layer)

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_h3_integer_prepared", False):
            return
        q, s, qs = prepare_integer_weight(layer.weight, layer.weight_scale, self.bits)
        device = layer.weight.device
        replace_parameter(layer, "weight", q.to(device))
        replace_parameter(layer, "weight_scale", s.to(device))
        layer.register_parameter("qmm_scale", None if qs is None else nn.Parameter(qs.to(device), requires_grad=False))
        layer._h3_integer_prepared = True
        layer._already_called_process_weights_after_loading = True

    def apply(self, layer, x, bias=None):
        return native_int_linear(
            x,
            layer.weight,
            layer.weight_scale,
            self.bits,
            bias,
            layer.output_size_per_partition,
            layer.input_size_per_partition,
            layer.qmm_scale,
        )


class H3NativeIntConfig(H3NativeMXConfig):
    delivery_schemes = {_INT4, _INT8}

    def get_name(self):
        return "h3_native_int"

    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, LinearBase):
            return None
        bits = self.targets.get(prefix.removeprefix("transformer."))
        return NativeIntLinearMethod(bits) if bits else UnquantizedLinearMethod()

    def validate_model_targets(self, model):
        installed = {
            name: module.quant_method.bits
            for name, module in model.named_modules()
            if isinstance(getattr(module, "quant_method", None), NativeIntLinearMethod)
        }
        if installed != self.targets:
            raise ValueError("native INT targets must all use their integer methods")
