# PR #10253 validation evidence

Synthetic, reproducible artifacts for
`[Hardware][Ascend] Add optional MiniMax-H3 Audio VAE decoder fast paths`.

- Scope: MiniMax-H3 Audio VAE only. No external or customer audio, no private SDK,
  no model weights.
- Device: one Ascend 910B4-1, CANN 9.3.0, host driver 25.5.0.
- `round2/` is the accepted window: base `8be1f8d67`, head `7680d408e`, four inputs
  (32 / 127 / 128 / 490 latent frames) with native main before and after.
- `round1/comparison-round1.json` records the rejected earlier revision, where short
  inputs exceeded the predeclared FP32 error guardrail.

## Layout

| Path | Content |
| --- | --- |
| `inputs/` | Deterministic ffmpeg-generated WAVs and their SHA-256 |
| `round2/run-identity.txt` | UID, PID, start ticks, card, base and head SHAs |
| `round2/comparison.json` | Bit-exactness, SNR, max error, timings, peak memory |
| `round2/<arm>/<input>/` | `encode-*.f32` latent dumps and `off|fir|stage|both.wav` |
| `round2/<arm>/results.json` | Raw per-measurement records including counters |
| `round2/resources-{before,after}.log` | NPU occupancy before and after the window |
| `logs/` | pytest, run, and comparison logs |
| `acceptance.md` | Predeclared acceptance contract |
| `compare.py`, `run_npu.sh`, `run_in_nightly.sh` | Verification harness |

## How to read the result

`encode-*.f32` are raw little-endian FP32 tensors; compare bytes, not serialized
`.pt` files. Every arm and both encode-only captures must match `main-before`
byte for byte. `off` WAVs must match `main-before` byte for byte. `fir`/`stage`/`both`
are allowed to differ only within the predeclared guardrail (SNR >= 60 dB and
max absolute error <= 0.002); the accepted window reports about 76.9 dB and
7.9e-5 to 9.3e-5. Stage counters must be zero below 128 latent frames and seven
at or above it.

These are component measurements. They do not establish HTTP serving, DiT,
video/mux output, throughput, or behavior on other Ascend generations.
