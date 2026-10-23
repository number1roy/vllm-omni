# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import msgspec
import pytest

from vllm_omni.config.omni_config import (
    VllmOmniConfig,
    _DiffusionConfigProjection,
    extract_diffusion_stage_config_kwargs,
)
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    build_stage_runtime_overrides,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.stage_engine_startup import _serialize_stage_config
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict_from_omni_stage_config,
    build_legacy_engine_args_dict,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resident_defaults():
    assert _DiffusionConfigProjection().dlo_vae_resident is False
    assert OmniDiffusionConfig().dlo_vae_resident is False
    assert StageDeployConfig(stage_id=0).dlo_vae_resident is None


@pytest.mark.parametrize("value", [0, 1, "false", "true", 1.5])
def test_resident_requires_boolean(value):
    for config_type in (_DiffusionConfigProjection, OmniDiffusionConfig, StageDeployConfig):
        with pytest.raises(ValueError, match="dlo_vae_resident"):
            kwargs = {"stage_id": 0} if config_type is StageDeployConfig else {}
            config_type(dlo_vae_resident=value, **kwargs)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_resident_stage_override_preserves_false(value, reverse):
    items = [("dlo_vae_resident", not value), ("stage_1_dlo_vae_resident", value)]
    overrides = dict(reversed(items) if reverse else items)
    assert build_stage_runtime_overrides(1, overrides)["dlo_vae_resident"] is value
    assert build_stage_runtime_overrides(0, overrides)["dlo_vae_resident"] is not value


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("cli_value,expected", [(None, True), (False, False), (True, True)])
def test_resident_reaches_worker_and_ipc(typed, cli_value, expected):
    pipeline = PipelineConfig(
        model_type="dlo-resident-test",
        stages=(
            StagePipelineConfig(
                stage_id=0, model_stage="model", execution_type=StageExecutionType.DIFFUSION, final_output=True
            ),
        ),
    )
    deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, dlo_vae_resident=True)])
    cli = {"dlo_vae_resident": cli_value}
    if typed:
        stage = VllmOmniConfig.from_pipeline_config(
            pipeline, user_deploy_config=deploy, cli_overrides=cli
        ).stage_configs[0]
        serialized = msgspec.msgpack.decode(msgspec.msgpack.encode(_serialize_stage_config(stage)))
        assert serialized["diffusion_config"]["dlo_vae_resident"] is expected
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    else:
        stage = merge_pipeline_deploy(pipeline, deploy)[0]
        stage.runtime_overrides = build_stage_runtime_overrides(0, cli)
        args = build_legacy_engine_args_dict(stage.to_omegaconf(), model="unused")
    kwargs = extract_diffusion_stage_config_kwargs(args, stage_id=0, include_engine_adapter_metadata=True)
    kwargs.pop("model")
    assert OmniDiffusionConfig.from_kwargs(**kwargs).dlo_vae_resident is expected
