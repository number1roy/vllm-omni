#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: bash run_in_nightly.sh ROOTFS CHECKOUT VISIBLE_NPU SDK_OR_EMPTY -- <decode_ablation.py arguments>
# Use the verified image's libraries natively: the user namespace cannot see Ascend HAL devices.
set -euo pipefail
root=${1:?verified nightly rootfs}
checkout=${2:?source checkout}
card=${3:?physical NPU index}
sdk=${4-}
shift 4
[[ ${1-} == -- ]] && shift
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec env -i HOME="$here" PATH=/usr/bin:/bin bash -s -- "$root" "$checkout" "$card" "$sdk" "$here" "$@" <<'NATIVE'
set -eo pipefail
root=$1 checkout=$2 card=$3 sdk=$4 here=$5
shift 5
cd "$checkout"
python_root="$root/usr/local/python3.12.13"
cann="$root/usr/local/Ascend/cann-9.3.0"
driver=/usr/local/Ascend/driver
export ASCEND_HOME_PATH="$cann" ASCEND_TOOLKIT_HOME="$cann" ASCEND_TOOLKIT_LATEST_HOME="$cann"
export ASCEND_OPP_PATH="$cann/opp" ASCEND_AICPU_PATH="$cann" TOOLCHAIN_HOME="$cann/toolkit"
export ASCEND_RT_VISIBLE_DEVICES="$card"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export VLLM_TARGET_DEVICE=npu VLLM_OMNI_TARGET_DEVICE=npu TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=1
export HF_MODULES_CACHE="$here/hf-modules"
export PATH="$python_root/bin:$cann/bin:$cann/tools/ccec_compiler/bin:$PATH"
export LD_LIBRARY_PATH="$cann/lib64:$cann/lib64/plugin/opskernel:$cann/lib64/plugin/nnengine:$cann/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:$driver/lib64:$driver/lib64/common:$driver/lib64/driver:$python_root/lib"
export PYTHONPATH="$checkout:$root/vllm-workspace/vllm:$root/vllm-workspace/vllm-ascend:$cann/python/site-packages:$cann/opp/built-in/op_impl/ai_core/tbe"
source "$root/usr/local/Ascend/nnal/atb/set_env.sh" --cxx_abi=1
if [[ -n $sdk ]]; then
    export PYTHONPATH="$sdk/python:$PYTHONPATH"
    export ASCEND_CUSTOM_OPP_PATH="$sdk/installed/packages/vendors/extra01_vae:$sdk/installed/packages/vendors/aie_ascendc${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
fi
case ${1-} in
    --probe)
        shift
        exec "$python_root/bin/python3.12" -u "$here/runtime_probe.py" "$checkout" "$@"
        ;;
    --pytest)
        export PYTHONPATH="$PYTHONPATH:${2:?isolated test dependencies}"
        shift 2
        exec "$python_root/bin/python3.12" -m pytest "$@"
        ;;
    --script)
        shift
        exec "$python_root/bin/python3.12" -u "$@"
        ;;
esac
exit 2
NATIVE
