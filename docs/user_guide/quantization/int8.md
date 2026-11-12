# Int8 Quantization

## Overview

Int8 quantization supports W8A8 diffusion transformer inference on CUDA and
Ascend NPU. It can quantize BF16/FP16 weights at load time, or load serialized
Int8 checkpoints that already contain quantized weights and scales.

Only online activation scaling is currently supported.

## Hardware Support

| Device | Support |
| -------- | --------- |
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ✅ |
| AMD ROCm | ⭕ |
| Intel XPU | ⭕ |
| Ascend NPU | ✅ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this
guide.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Model | HF models | CUDA | Ascend NPU | Mode | Recommendation |
| ------- | ----------- | :----: | :----------: | ------ | ---------------- |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes | Yes | Online W8A8 | All layers |
| Wan2.2 | Wan2.2 diffusion pipelines | Not validated | Not validated | Online W8A8 | Validate before enabling in docs |
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Yes | Yes | Online W8A8 | All layers |

Other diffusion models may work if their transformer uses supported linear
layers, but they are not validated in this guide.

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Scope | Status | Notes |
| ------- | ------- | -------- | ------- |
| Qwen3-Omni | Thinker language-model stage | Not validated | Prefer checkpoint-supported ModelOpt FP8 or AutoRound paths |
| Qwen3-TTS | TTS language-model stage | Not validated | No Int8 TTS stage support is documented |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Model | Scope | Status | Notes |
| ------- | ------- | -------- | ------- |
| BAGEL | Stage-specific transformer or DiT module | Not validated | Requires explicit stage routing |
| GLM-Image | Stage-specific transformer or DiT module | Not validated | Requires quality comparison with BF16 |

## Configuration

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="<your-model>", quantization="int8")

omni_with_skips = Omni(
    model="<your-model>",
    quantization_config={
        "method": "int8",
        "ignored_layers": ["<layer-name>"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

CLI:

```bash
python text_to_image.py --model <your-model> --quantization int8
python text_to_image.py --model <your-model> --quantization int8 --ignored-layers "img_mlp"
vllm serve <your-model> --omni --quantization int8
```

## Parameters

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `method` | str | - | Quantization method (`"int8"`) |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` selects online activation scaling; static is not supported |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16/FP16 |
| `is_checkpoint_int8_serialized` | bool | `False` | Set by checkpoint config when loading serialized Int8 weights |

## Validation and Notes

Int8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="int8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```

Only add a new model to the supported table after comparing the Int8 output
against a BF16 baseline and documenting any required `ignored_layers`.

## MiniMax H3 offline component checkpoints

H3 can override the transformer's weight source with a local ModelSlim
checkpoint. Keep `model` pointing to the original, complete native H3 partition:
the quantized delivery does not replace its architecture configuration,
tokenizer, text encoder, or trusted VAE modeling code.

### Native INT8 W8A8 with dynamic activations

```python
omni = Omni(
    model="/path/to/original/FL2VA",
    trust_remote_code=True,
    quantization_config={
        "transformer": {
            "method": "int8",
            "is_checkpoint_int8_serialized": True,
        },
        "default": None,
    },
    additional_config={
        "offline_quantization": {
            "components": {"transformer": "/path/to/int8/transformer"},
        },
    },
)
```

The component directory must contain `quant_model_description.json` and
safetensors weights (optionally sharded with a safetensors index). The supported
INT8 contract is ModelSlim `W8A8_DYNAMIC`: signed INT8 `[N,K]` weights and
positive, finite FP32 per-output-channel `weight_scale`, shaped `[N]` or `[N,1]`.
An optional `weight_offset` must be zero. Static ModelSlim `W8A8` with
`quant_bias`, `input_scale`, `input_offset`, or `deq_scale` is a different
contract and is rejected. Activations are still quantized dynamically during
inference; weights are loaded directly without startup quantization.

Both native H3 names and the corresponding Diffusers names are accepted. FLOAT
layers are inferred from the description and kept unquantized. Every projection
merged into one QKV or gate/up parameter must use the same scheme. H3's fixed
FP32 input/output projections must remain FLOAT. Missing weights/scales,
unsupported tensors, and precision mismatches fail loading rather than using
the original transformer's weights. On NPU a per-rank output width above
65,535 fails early: increase TP or export that projection as FLOAT.

### Delivered MXFP8 / dual-scale MXFP4 compatibility mode

```python
omni = Omni(
    model="/path/to/original/FL2VA",
    trust_remote_code=True,
    quantization_config=None,
    additional_config={
        "offline_quantization": {
            "path": "/path/to/MiniMax-H3-Quantize",
            "execution_mode": "dequantize",
        },
    },
)
```

For this delivery layout, root resolution selects:

| Component | Relative directory |
| --- | --- |
| Transformer | `output_jiaofu/minimax_h3_fl2va` |
| Video VAE | `output_jiaofu_support_original/video_vae` |
| Audio VAE | `output_jiaofu_support_original/audio_vae` |

The original DiT delivery is used because the converted transformer directory
omits FP4 `weight_dual_scale`. The reader reconstructs E4M3 MXFP8 using E8M0
scales, and packed E2M1 FP4 using both E8M0 and FP32 dual scales. Fine groups
contain 32 K elements, including partial final groups such as the video VAE's
K=24 projection. The supplied FP4 dual scales use groups of 32; they must not be
treated as the groups of 512 expected by the existing native dual-scale kernel.

This explicit compatibility mode reconstructs floating-point weights on CPU,
then loads the model's ordinary floating-point parameters. It does **not**
provide native MX activation quantization, low-bit compute, or low-bit runtime
memory savings. Ascend 910B cannot execute the native MX kernels.

### Ascend native mixed MX execution

On hardware with native MX kernels, select `execution_mode="native_mx"` in
the same offline configuration and keep `quantization_config=None`. Delivery
metadata selects the precision: `W8A8_MXFP8` matrices use W8A8 and
`W4A4_MXFP4_DUALSCALE` matrices use W4A4. FLOAT parameters keep their original
precision. Every constituent of a fused QKV projection must share a scheme.

At load time, all delivery scales are applied before weights are encoded for
the native kernels. In particular, the supplied FP4 coarse groups of 32 are
converted to single-scale MXFP4 groups of 32; they are not passed to the
incompatible dual-level kernel with coarse groups of 512. This conversion can
introduce additional error and requires output-quality evaluation. It is not
a byte-preserving use of the original dual-scale representation.

During inference, `npu_dynamic_mx_quant` quantizes activations and
`npu_quant_matmul` multiplies low-bit operands. The GEMM returns BF16 before
restoring the caller's output dtype; this does not make the multiply BF16.
There is no floating GEMM fallback. K is padded to 64 and wide outputs are
split into native low-bit GEMMs. VAE overrides replace only the delivery's
quantized `nn.Linear` layers. Convolutions, attention, normalization, and
other FLOAT operations retain their usual precision.

The native Linear interface accepts FP16, BF16, or FP32 activations and returns
that same floating dtype. Its uint8 weight dtype describes packed storage;
it must not be used as an activation dtype. The VAE adapter handles checkpoint
dtype helpers and audio encoder QKV calls that otherwise bypass this interface.
Restoring a floating output dtype does not recover precision lost to quantization.

Runtime weights and E8M0 scales use ordinary contiguous byte parameters,
including packed FP4 bytes, so DLO can transport them without changing their
bits. The adapter rebases FP4 weight views with nonzero storage offsets before
native dispatch. This copies at most one output chunk and preserves the packed
bytes; it does not reconstruct floating weights during inference. This also
applies to later chunks of a wide projection, even without DLO.
Hardware kernel compatibility, full generation, and quality must be
validated for the specific runtime and checkpoint; selecting the mode alone
is not a validation result.

VAE overrides require both encoder and decoder enabled. VAE loading currently
constructs the original component before replacing its state, so the original
VAE checkpoint must remain available. Explicit `components` paths can select
only the desired overrides. A ModelSlim `*.weight_quantizer.bias` duplicate is
accepted only when its canonical `*.bias` exists in both the checkpoint and
model, with identical dtype, shape, and values. Conflicting or orphan aliases
and other unknown tensors are rejected.

Offline overrides currently require a standalone
native partition without HSDP, host weight runtime, LoRA/fused adapters,
replacement VAE decoders, or a combined FL2VA/Ref2VA model. Direct DLO checkpoint
mmap is bypassed so reconstruction and strict format checks cannot be skipped.

The supplied MX delivery has been checked against its tensor metadata and
floating reconstruction tests. INT8 delivery-specific output quality and full
H3 generation must be validated after the INT8 weights are provided; the
examples above are loading configurations, not an end-to-end accuracy claim.
