# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts runnable without vLLM/NPU, using the production reader/layout."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file

# The H3 package eagerly imports its pipeline. Load just these CPU modules in
# an isolated namespace so this suite also runs on a developer laptop.
_source = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3"
_namespace = "_h3_cpu_checkpoint_contracts"
_package = ModuleType(_namespace)
_package.__path__ = [str(_source)]
sys.modules[_namespace] = _package


def _load(name):
    spec = importlib.util.spec_from_file_location(_namespace + "." + name, _source / (name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils = _load("integer_utils")
reader = _load("offline_quantization")
pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def checkpoint(path, codes, scales, scheme="W4A4_DYNAMIC", extra=None):
    path.mkdir()
    tensors = {"proj.weight": codes.contiguous(), "proj.weight_scale": scales.contiguous()}
    tensors.update(extra or {})
    desc = {k: scheme if k.startswith("proj.weight") else "FLOAT" for k in tensors}
    save_file(tensors, str(path / "weights.safetensors"))
    (path / "quant_model_description.json").write_text(json.dumps(desc))
    return reader.OfflineComponentCheckpoint(path)


def test_signed_nibbles_and_nonzero_storage_offsets():
    codes = (torch.arange(9 * 32).reshape(9, 32) % 16 - 8).to(torch.int8)
    packed = utils.pack_int4(codes)
    assert packed[0, 0].item() == -19088744  # signed 0xFEDCBA98
    assert packed[0, 1].item() == 1985229328  # 0x76543210
    assert packed[3:].storage_offset() > 0
    assert torch.equal(utils.unpack_int4(packed[3:]), codes[3:])


@pytest.mark.parametrize("invalid", ["range", "dtype", "width"])
def test_pack_rejects_invalid_storage(invalid):
    q = torch.zeros(2, 32, dtype=torch.int8)
    if invalid == "range":
        q[0, 1] = 8
    elif invalid == "dtype":
        q = q.int()
    else:
        q = q[:, :31]
    with pytest.raises(ValueError):
        utils.pack_int4(q)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("k", [5, 24, 65, 4096])
def test_padding_preserves_codes_scales_and_bounds_every_integer_dot(bits, k):
    q = (torch.arange(7 * k).reshape(7, k) % 16 - 8).to(torch.int8)
    s = torch.tensor([0.100003, 0.03, 7.01, 0.000001, 0.5, 100, 0.002003])
    packed, scales, bound_scale = utils.prepare_integer_weight(q, s, bits)
    actual = utils.unpack_int4(packed) if bits == 4 else packed
    assert torch.equal(actual[:7, :k], q)
    assert not torch.count_nonzero(actual[7:]) and not torch.count_nonzero(actual[:, k:])
    assert torch.equal(scales[:7], s)
    if bits == 4:
        worst = actual.to(torch.float64).abs().sum(1) * 8
        assert torch.all(worst * bound_scale <= 16384)
        # A power-of-two epilogue and FP32 restoration must compose to the
        # same serialized scale, with no scale or byte-offset reinterpretation.
        x = (torch.arange(k) % 16 - 8).to(torch.float64)
        accumulator = x @ actual[:7, :k].double().T
        restored = (accumulator * bound_scale[:7]).half().float() * (scales[:7] / bound_scale[:7])
        expected = accumulator.float() * s
        torch.testing.assert_close(restored, expected, rtol=0.001, atol=1e-6)


@pytest.mark.parametrize("packed", [False, True])
def test_checkpoint_reads_exact_integer_codes_and_fp32_scale(tmp_path, packed):
    q = (torch.arange(3 * 32).reshape(3, 32) % 16 - 8).to(torch.int8)
    s = torch.tensor([[0.00010003], [0.31531], [1.000007]])
    cp = checkpoint(tmp_path / "int", utils.pack_int4(q) if packed else q, s)
    w, actual_scale = cp.integer_weight("proj.weight")
    assert torch.equal(w, q) and torch.equal(actual_scale, s)
    native = dict(cp.iter_weights(dequantize=False))
    assert torch.equal(native["proj.weight"], q)
    torch.testing.assert_close(dict(cp.iter_weights(dequantize=True))["proj.weight"], q.float() * s)


@pytest.mark.parametrize("issue", ["nonzero_offset", "static_correction", "nonfinite_scale", "negative_scale", "range"])
def test_invalid_integer_delivery_is_rejected(tmp_path, issue):
    q, s, extra = torch.zeros(3, 32, dtype=torch.int8), torch.ones(3, 1), {}
    if issue == "nonzero_offset":
        extra["proj.weight_offset"] = torch.ones(3, 1)
    elif issue == "static_correction":
        extra["proj.quant_bias"] = torch.zeros(3)
    elif issue == "nonfinite_scale":
        s[0] = float("inf")
    elif issue == "negative_scale":
        s[0] = -1
    else:
        q[0, 0] = 8
    with pytest.raises(ValueError):
        cp = checkpoint(tmp_path / "bad", q, s, extra=extra)
        cp.integer_weight("proj.weight")


@pytest.mark.parametrize(
    "device,family",
    [("Ascend950PR_958b", "mx"), ("Ascend910B2", "int"), ("Ascend910_9391", "int"), ("Ascend910C", "int")],
)
def test_device_family(device, family):
    assert reader.npu_quant_family(device) == family


def test_unknown_device_does_not_guess():
    with pytest.raises(ValueError, match="unsupported"):
        reader.npu_quant_family("Ascend310P")


@pytest.mark.parametrize("family", ["mx", "int"])
def test_native_auto_selects_and_validates_without_format_fallback(tmp_path, monkeypatch, family):
    int_cp = checkpoint(tmp_path / "int", torch.ones(3, 32, dtype=torch.int8), torch.ones(3, 1))
    mx_cp = checkpoint(
        tmp_path / "mx",
        torch.ones(3, 32, dtype=torch.float8_e4m3fn),
        torch.full((3, 1), 127, dtype=torch.uint8),
        scheme="W8A8_MXFP8",
    )
    monkeypatch.setattr(reader, "resolve_npu_quant_family", lambda *_: family)
    config = {"format_paths": {"int": {"path": str(int_cp.path)}, "mx": {"path": str(mx_cp.path)}}}
    resolved, reconstruct = reader.resolve_offline_components(config)
    assert resolved["transformer"].execution_mode == "native_" + family
    assert reconstruct == (family == "mx")
    with pytest.raises(ValueError, match="incompatible"):
        reader.resolve_offline_components(
            {"path": str(int_cp.path), "execution_mode": "native_" + ("int" if family == "mx" else "mx")}
        )
    wrong = mx_cp if family == "int" else int_cp
    with pytest.raises(ValueError, match="requires"):
        reader.resolve_offline_components({"path": str(wrong.path)})
