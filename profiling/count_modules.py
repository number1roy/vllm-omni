import json
from collections import Counter

import torch
import torch_npu

from vllm_omni.diffusion.models.minimax_h3 import vae

device = torch.device("npu", 0)
torch.npu.set_device(device)
model = vae.MiniMaxH3AudioVAE(
    "/opt/tiger/models/minimax-h3-whole/Ref2VA/audio_vae",
    device=device, load_device=torch.device("cpu"), trust_remote_code=True,
)
decoder = model.model.decoder
counts = Counter(type(m).__name__ for m in decoder.modules() if type(m).__name__ in ("UpSample1d", "LowPassFilter1d"))
stages = [layer for stage in getattr(decoder, "ups", ()) for layer in stage]
print(json.dumps({"fir_by_class": counts, "stage_count": len(stages), "stage_types": sorted({type(s).__name__ for s in stages})}))
