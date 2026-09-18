# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from PIL import Image

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from responses_api_agents.image_tools_agent.app import (
    ImageToolsAgent,
    ImageToolsAgentConfig,
    ImageToolsAgentRunRequest,
)
from responses_api_agents.image_tools_agent.tests.test_app import (
    _assistant_response,
    _base_verify_response,
    _FakeClientResponse,
)


ZOOM = (
    "<tool_call><function=image_zoom_in_tool><parameter=bbox_2d>[0,0,500,500]</parameter>"
    "<parameter=img_idx>0</parameter></function></tool_call>"
)


def setup_episode(tmp_path, monkeypatch, outputs, **config):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "blue").save(image_path)
    client = MagicMock(spec=ServerClient)

    async def post(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            return _FakeClientResponse({})
        # Deliberately optimistic verifier: an exhausted tool-only episode
        # must still get zero answer reward, even if this returns one.
        return _FakeClientResponse(_base_verify_response(kwargs["json"], 1.0))

    client.post = AsyncMock(side_effect=post)
    agent = ImageToolsAgent(
        config=ImageToolsAgentConfig(
            host="localhost",
            port=10001,
            name="image_tools_simple_agent",
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            resource_servers_by_agent={
                "string_match_simple_agent": {"type": "resources_servers", "name": "string_match"}
            },
            crop_dir=str(tmp_path / "crops"),
            **config,
        ),
        server_client=client,
    )
    calls = []

    async def model(self, body, cookies):
        calls.append(body.model_copy(deep=True))
        value = outputs[len(calls) - 1]
        if isinstance(value, Exception):
            raise value
        payload = _assistant_response(
            response_id=f"r{len(calls)}",
            text=value if isinstance(value, str) else value["text"],
            prompt_token_ids=[1, 2, 3],
            generation_token_ids=[4, 5],
        )
        if isinstance(value, dict):
            payload.update(value.get("overrides", {}))
        return NeMoGymResponse.model_validate(payload), {}

    monkeypatch.setattr(ImageToolsAgent, "_call_model", model)
    monkeypatch.setenv("NEMO_GYM_COMPACT_ROLLOUT_DIR", str(tmp_path / "diagnostics"))
    body = ImageToolsAgentRunRequest.model_validate(
        {
            "image_tools_base_agent_ref": {"name": "string_match_simple_agent", "type": "responses_api_agents"},
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": str(image_path)},
                        ],
                    }
                ]
            },
        }
    )
    return agent, body, calls


@pytest.mark.asyncio
async def test_final_turn_reserved_after_tool_cap(tmp_path, monkeypatch):
    agent, body, calls = setup_episode(
        tmp_path, monkeypatch, [ZOOM, "Final answer: blue"], max_steps=2, max_tool_calls=1
    )
    result = await agent.run(SimpleNamespace(cookies={}), body)
    assert len(calls) == 2
    assert "FINAL turn" in str(calls[1].input[-1].content)
    assert result.image_tools_executed_output_ids == ["r1-message"]
    assert result.task_success is True
    record = json.loads(next((tmp_path / "diagnostics").glob("*.jsonl")).read_text())
    assert set(record) == {
        "game_name",
        "duration_seconds",
        "model_calls",
        "valid_environment_steps",
        "generated_tokens",
        "no_boxed_retries",
        "termination_reason",
    }
    assert record["model_calls"] == 2
    assert record["valid_environment_steps"] == 1
    assert record["generated_tokens"] == 4
    assert record["termination_reason"] == "final_answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [ZOOM, "<tool_call>bad", ZOOM.replace("image_zoom_in_tool", "unknown_tool")])
async def test_last_turn_tool_never_becomes_an_answer(tmp_path, monkeypatch, text):
    agent, body, calls = setup_episode(tmp_path, monkeypatch, [text], max_steps=1)
    result = await agent.run(SimpleNamespace(cookies={}), body)
    assert len(calls) == 1
    assert result.base_reward == 0
    assert result.task_success is False
    assert result.image_tools_executed_output_ids == []
    assert result.image_tools_call_count == 0
    assert result.response.metadata["termination_reason"] == "max_steps"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    ["<tool_call>bad", ZOOM.replace("image_zoom_in_tool", "unknown_tool"), ZOOM.replace("[0,0,500,500]", "[0,0,0,0]")],
)
async def test_invalid_or_failed_tool_has_no_execution_provenance(tmp_path, monkeypatch, text):
    agent, body, calls = setup_episode(tmp_path, monkeypatch, [text])
    result = await agent.run(SimpleNamespace(cookies={}), body)
    assert result.image_tools_executed_output_ids == []
    assert result.image_tools_error_count == 1
    assert result.base_reward == 0
    assert result.response.metadata["termination_reason"] == "invalid_tool_call"


@pytest.mark.asyncio
async def test_duplicate_executes_but_forces_final_and_keeps_penalty(tmp_path, monkeypatch):
    agent, body, calls = setup_episode(tmp_path, monkeypatch, [ZOOM, ZOOM, "Final answer: blue"])
    result = await agent.run(SimpleNamespace(cookies={}), body)
    assert len(calls) == 3
    assert result.image_tools_executed_output_ids == ["r1-message", "r2-message"]
    assert result.image_tools_aux_reward == pytest.approx(0.0)  # +.02 then -.02


@pytest.mark.asyncio
async def test_context_error_on_first_call_is_nonfatal_and_masked(tmp_path, monkeypatch):
    error = aiohttp.ClientResponseError(SimpleNamespace(real_url="http://model"), (), status=400)
    error.response_content = b'{"error":{"code":"context_length_exceeded"}}'
    agent, body, calls = setup_episode(tmp_path, monkeypatch, [error])
    result = await agent.run(SimpleNamespace(cookies={}), body)
    assert len(calls) == 1
    assert result.response.output == []
    assert result.model_dump()["instance_config"]["mask_sample"] is True
    assert result.model_dump()["failure_reason"] == "context_length_exceeded"
    assert result.base_reward == 0


@pytest.mark.asyncio
async def test_other_http_errors_propagate_for_infrastructure_retry(tmp_path, monkeypatch):
    error = aiohttp.ClientResponseError(SimpleNamespace(real_url="http://model"), (), status=503)
    agent, body, _ = setup_episode(tmp_path, monkeypatch, [error])
    with pytest.raises(aiohttp.ClientResponseError):
        await agent.run(SimpleNamespace(cookies={}), body)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config,reason",
    [
        ({"max_total_sequence_length": 7, "max_output_tokens": 2}, "max_total_sequence_length"),
        ({"max_images": 1}, "max_images"),
    ],
)
async def test_headroom_limits_prevent_next_call(tmp_path, monkeypatch, config, reason):
    agent, body, calls = setup_episode(tmp_path, monkeypatch, [ZOOM], **config)
    result = await agent.run(SimpleNamespace(cookies={}), body)
    assert len(calls) == 1
    assert result.response.metadata["termination_reason"] == reason
    assert result.base_reward == 0


@pytest.mark.asyncio
async def test_over_cap_warns_masks_and_does_not_execute(tmp_path, monkeypatch, caplog):
    agent, body, calls = setup_episode(tmp_path, monkeypatch, [ZOOM], max_output_tokens=1)
    result = await agent.run(SimpleNamespace(cookies={}), body)
    assert result.image_tools_call_count == 0
    assert result.model_dump()["instance_config"]["mask_sample"] is True
    assert result.response.metadata["termination_reason"] == "model_token_budget_exceeded"
    assert "exceeded token budget" in caplog.text


@pytest.mark.asyncio
async def test_truncated_call_is_not_executed(tmp_path, monkeypatch):
    agent, body, _ = setup_episode(
        tmp_path,
        monkeypatch,
        [
            {
                "text": ZOOM,
                "overrides": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            }
        ],
    )
    result = await agent.run(SimpleNamespace(cookies={}), body)
    assert result.image_tools_call_count == 0
    assert result.base_reward == 0
    assert result.response.metadata["termination_reason"] == "max_output_tokens"
