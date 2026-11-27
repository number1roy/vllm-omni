# SPDX-License-Identifier: Apache-2.0
"""Attribute MiniMax-H3 Audio VAE decode time to FIR, BigVGAN stage and the rest."""

import argparse
import csv
import json
import time
from pathlib import Path

import soundfile as sf
import torch
import torch_npu

from vllm_omni.diffusion.models.minimax_h3 import vae


def module_groups(model):
    """Select the same FIR and stage modules the production installers target."""
    decoder = model.model.decoder
    fir = [m for m in decoder.modules() if type(m).__name__ in ("UpSample1d", "LowPassFilter1d")]
    stages = [layer for stage in getattr(decoder, "ups", ()) for layer in stage]
    return fir, stages


def attach(timer, modules, group):
    handles = []
    for module in modules:

        def pre(mod, args, group=group):
            event = torch.npu.Event(enable_timing=True)
            event.record()
            mod._profile_start = event

        def post(mod, args, output, group=group):
            event = torch.npu.Event(enable_timing=True)
            event.record()
            timer.append((group, mod._profile_start, event))

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
    return handles


def kernel_table(prof_dir):
    """Aggregate the profiler kernel table by op type."""
    paths = sorted(Path(prof_dir).rglob("kernel_details.csv"))
    if not paths:
        return None, []
    rows = []
    with paths[0].open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                rows.append((row["Type"], row["Name"], float(row["Duration(us)"])))
            except (KeyError, ValueError):
                continue
    total = sum(duration for _, _, duration in rows)
    by_type = {}
    for op_type, name, duration in rows:
        entry = by_type.setdefault(op_type, {"count": 0, "total_ms": 0.0, "names": {}})
        entry["count"] += 1
        entry["total_ms"] += duration / 1e3
        entry["names"][name] = entry["names"].get(name, 0.0) + duration / 1e3
    summary = [
        {
            "type": op_type,
            "count": entry["count"],
            "total_ms": round(entry["total_ms"], 3),
            "share": round(entry["total_ms"] / (total / 1e3), 4) if total else 0.0,
            "top_names": sorted(entry["names"].items(), key=lambda item: -item[1])[:3],
        }
        for op_type, entry in by_type.items()
    ]
    return str(paths[0]), sorted(summary, key=lambda item: -item["total_ms"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--waveform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    args.output.mkdir(parents=True)
    options = {"optimize_decode": True} if args.optimized else {}
    model = vae.MiniMaxH3AudioVAE(
        str(args.model), device=device, load_device=torch.device("cpu"), trust_remote_code=True, **options
    )
    model.load_to_device()
    waveform, rate = sf.read(args.waveform, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T.copy())
    fir, stages = module_groups(model)
    installed = (
        bool(fir)
        and all(hasattr(module, "_h3_fir_original") for module in fir)
        and all(hasattr(module, "_h3_stage_original_forward") for module in stages)
    )
    if installed != args.optimized:
        raise RuntimeError(f"unexpected installation: FIR={len(fir)} stages={len(stages)} installed={installed}")
    with torch.inference_mode():
        rows, frames = model.encode_waveform(waveform, rate)
        latent = rows.reshape(2, frames, -1).transpose(1, 2).contiguous().to(device)
        for _ in range(args.warmups):
            model.decode_latent(latent)
        torch.npu.synchronize()

        timer = []
        handles = attach(timer, fir, "fir") + attach(timer, stages, "stage")
        totals = []
        try:
            for _ in range(args.runs):
                torch.npu.synchronize()
                start = time.perf_counter()
                model.decode_latent(latent)
                torch.npu.synchronize()
                totals.append(time.perf_counter() - start)
        finally:
            for handle in handles:
                handle.remove()
        torch.npu.synchronize()
        grouped = {"fir": 0.0, "stage": 0.0}
        for group, begin, end in timer:
            grouped[group] += begin.elapsed_time(end)
        grouped = {key: value / args.runs for key, value in grouped.items()}
        total = sum(totals) / len(totals) * 1e3
        attribution = {
            "total_ms": round(total, 3),
            "fir_ms": round(grouped["fir"], 3),
            "stage_ms": round(grouped["stage"], 3),
            "other_ms": round(total - grouped["fir"] - grouped["stage"], 3),
            "fir_share": round(grouped["fir"] / total, 4),
            "stage_share": round(grouped["stage"] / total, 4),
            "other_share": round((total - grouped["fir"] - grouped["stage"]) / total, 4),
            "fir_modules": len(fir),
            "stage_modules": len(stages),
            "runs_ms": [round(value * 1e3, 3) for value in totals],
        }

        prof_dir = args.output / "profiler"
        experimental = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            mstx=False,
        )
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(prof_dir)),
            experimental_config=experimental,
        ) as prof:
            for _ in range(2):
                model.decode_latent(latent)
                prof.step()
        torch.npu.synchronize()

    table, kernels = kernel_table(prof_dir)
    record = {
        "device": torch.npu.get_device_name(device),
        "optimized": args.optimized,
        "latent_frames": frames,
        "waveform": str(args.waveform),
        "attribution": attribution,
        "kernel_details_csv": table,
        "kernels_by_type": kernels,
    }
    (args.output / "profile.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
