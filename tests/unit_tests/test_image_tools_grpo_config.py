# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from responses_api_agents.image_tools_agent.app import ImageToolsAgentConfig


def test_standalone_bundle_resolves_without_games_or_pivot_datasets(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_TOOLS_OUTPUT_DIR", str(tmp_path / "crops"))
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": ["environments/image_tools_grpo/config.yaml"],
        },
    )
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    params = OmegaConf.to_container(
        config.image_tools_simple_agent.responses_api_agents.image_tools_agent, resolve=True
    )
    agent = ImageToolsAgentConfig.model_validate(
        {**params, "host": "localhost", "port": 10001, "name": "image_tools_simple_agent"}
    )
    assert agent.max_steps == 21
    assert agent.max_tool_calls == 20
    assert agent.max_output_tokens == 512
    assert agent.max_total_sequence_length == 32768
    assert agent.max_images == 21
    assert agent.crop_dir == str(tmp_path / "crops")
    assert {ref.name for ref in agent.resource_servers_by_agent.values()} == {
        "string_match",
        "math_with_judge",
        "mcqa",
    }
    assert config.math_with_judge.resources_servers.math_with_judge.should_use_judge is False
    assert "gym_v" not in config
    assert "visgym_agent" not in config
    assert "image_tools_pivot" not in config
    assert not params.get("datasets")
