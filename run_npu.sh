#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:?verified nightly rootfs}
base=${2:?main checkout}
head=${3:?candidate checkout}
card=${4:?physical NPU index}
window=${5:?experiment window}
shift 5
waveforms=("$@")
if ((${#waveforms[@]} == 0)); then
    waveforms=("$here/inputs/short-mono-24k.wav" "$here/inputs/long-stereo-32k.wav")
fi
base_sha=$(git -C "$base" rev-parse HEAD)
head_sha=$(git -C "$head" rev-parse HEAD)
git -C "$base" diff --quiet HEAD
git -C "$head" diff --quiet HEAD
model=/opt/tiger/models/minimax-h3-whole/Ref2VA/audio_vae
script="$head/benchmarks/diffusion/minimax_h3_audio_vae.py"
out="$here/$window"
mkdir "$out"
exec 9>"/tmp/lbw-h3-vae-npu-$card.lock"
flock -n 9
before=$(npu-smi info -t proc-mem -i "$card" -c 0)
printf '%s\n' "$before" | tee "$out/resources-before.log"
[[ $before == *'No process in device.'* ]]
trap 'npu-smi info -t proc-mem -i "$card" -c 0 | tee "$out/resources-after.log"' EXIT
printf 'uid=%s pid=%d start_ticks=%s card=%d base=%s head=%s\n' "$(id -u)" "$$" "$(awk '{print $22}' /proc/$$/stat)" "$card" "$base_sha" "$head_sha" | tee "$out/run-identity.txt"
sha256sum "$script" | tee "$out/harness.sha256"
run() {
    local label=$1 checkout=$2
    shift 2
    bash "$here/run_in_nightly.sh" "$root" "$checkout" "$card" '' -- --script "$script" \
        --model "$model" --waveform "${waveforms[@]}" \
        --output "$out/$label" --warmups 3 --repeats 5 "$@" > "$out/$label.log" 2>&1
}
run main-before "$base" --encode-only
run head-default "$head"
run head-ablation "$head" --optimized --encode-only
run main-after "$base"
printf 'NPU_WINDOW_COMPLETE\n'
