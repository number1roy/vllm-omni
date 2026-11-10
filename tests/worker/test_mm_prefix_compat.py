# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unrelated pipelines can import Omni on vLLM versions without the Bagel hook."""

import pytest
from vllm.v1.attention.backends import utils

from vllm_omni.utils.mm_prefix import apply_img2img_mm_prefix_patch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_missing_prefix_builder_skips_patch(monkeypatch, caplog):
    monkeypatch.delattr(utils, "compute_mm_prefix_ranges", raising=False)
    tensor_builder = getattr(utils, "compute_mm_prefix_range_tensor", None)
    apply_img2img_mm_prefix_patch()
    assert not hasattr(utils, "compute_mm_prefix_ranges")
    assert getattr(utils, "compute_mm_prefix_range_tensor", None) is tensor_builder
    assert "Bagel img2img prefix attention is unavailable" in caplog.text
