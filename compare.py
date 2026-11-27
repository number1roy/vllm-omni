#!/usr/bin/env python3
"""Verify same-window, byte-level and numerical H3 Audio VAE acceptance."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf


def checksum(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "md5": hashlib.md5(data).hexdigest(), "sha256": hashlib.sha256(data).hexdigest()}


def waveform(path: Path) -> tuple[np.ndarray, int]:
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    assert np.isfinite(samples).all(), path
    return samples, rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path)
    args = parser.parse_args()
    dirs = ("main-before", "head-default", "head-ablation", "main-after")
    runs = {name: json.loads((args.window / name / "results.json").read_text()) for name in dirs}
    report = {}
    errors = []
    for case, case_info in runs["main-before"]["cases"].items():
        stem = Path(case).stem
        base_dir = args.window / "main-before" / stem
        ref, rate = waveform(base_dir / "off.wav")
        raw_encoder = (base_dir / "encode-0.f32").read_bytes()
        info = {
            "input_sha256": case_info["input_sha256"],
            "input_shape": case_info["input_shape"],
            "input_rate": case_info["input_sample_rate"],
            "encoder_shape": case_info["encode_shape"],
            "encoder_frames": case_info["encode_frames"],
            "encode_reference": checksum(base_dir / "encode-0.f32"),
            "wave_shape": list(ref.shape),
            "wave_rate": rate,
            "wave_duration_s": len(ref) / rate,
            "runs": {},
        }
        for run, data in runs.items():
            current = data["cases"][case]
            assert all(current[k] == case_info[k] for k in ("input_sha256", "encode_shape", "encode_frames"))
            current_dir = args.window / run / stem
            encodes = sorted(current_dir.glob("encode-*.f32"))
            if not all(p.read_bytes() == raw_encoder for p in encodes):
                errors.append(f"{case}/{run}: encode is not bit-exact")
            stats = {"encoder_dumps": {path.name: checksum(path) for path in encodes}, "decode": {}}
            for arm, samples in current["decode"].items():
                artifact = current_dir / f"{arm}.wav"
                out, out_rate = waveform(artifact)
                assert out.shape == ref.shape and out_rate == rate
                assert np.abs(out).max() <= 1, (run, case, arm)
                delta = out.astype(np.float64) - ref
                rmse = float(np.sqrt(np.mean(delta**2)))
                max_abs = float(np.max(np.abs(delta)))
                snr = 20 * math.log10(float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))) / rmse) if rmse else None
                exact = out.tobytes() == ref.tobytes()
                inner = ref[2:-2].astype(np.float64)
                lag_mse = {lag: float(np.mean((out[2 + lag : len(out) - 2 + lag] - inner) ** 2)) for lag in range(-2, 3)}
                best_lag = min(lag_mse, key=lag_mse.__getitem__)
                if best_lag != 0:
                    errors.append(f"{case}/{run}/{arm}: waveform alignment shifted by {best_lag} samples")
                formal = [row for row in samples if row["kind"] == "measurement"]
                times = [row["seconds"] for row in formal]
                assert len(formal) == 5
                out_hash = hashlib.sha256(out.astype("<f4", copy=False).tobytes()).hexdigest()
                if any(row["waveform_sha256"] != out_hash for row in formal):
                    errors.append(f"{case}/{run}/{arm}: decode is not repeatable")
                if arm == "off" and not exact:
                    errors.append(f"{case}/{run}: default decode is not bit-exact")
                if arm != "off" and ((snr is not None and snr < 60) or max_abs > 0.002):
                    errors.append(f"{case}/{run}/{arm}: decoder guardrail fails (SNR={snr}, max abs={max_abs})")
                stats["decode"][arm] = {
                    "exact": exact,
                    "max_abs": max_abs,
                    "rmse": rmse,
                    "snr_db": snr,
                    "best_offset_samples": best_lag,
                    "wav": checksum(artifact),
                    "fp32_sha256": out_hash,
                    "sample_s": times,
                    "median_s": statistics.median(times),
                    "mean_s": statistics.mean(times),
                    "sample_stddev_s": statistics.stdev(times),
                    "fir_calls": sorted({row["fir_calls"] for row in formal}),
                    "fir_fallbacks": sorted({row["fir_fallbacks"] for row in formal}),
                    "stage_calls": sorted({row["stage_calls"] for row in formal}),
                    "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in formal),
                    "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in formal),
                }
            info["runs"][run] = stats
        report[case] = info
    payload = {"verdict": "failed" if errors else "pass", "errors": errors, "cases": report}
    (args.window / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
