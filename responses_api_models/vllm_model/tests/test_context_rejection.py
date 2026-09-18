# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponseError
from fastapi import HTTPException, Request

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig, _context_length_error_detail


def error(status: int, body: bytes) -> ClientResponseError:
    result = ClientResponseError(SimpleNamespace(real_url="http://model/v1"), (), status=status)
    result.response_content = body
    return result


@pytest.mark.parametrize(
    "body",
    [
        b'{"error":{"code":"context_length_exceeded","message":"full"}}',
        b'{"code":"context_length_exceeded","message":"full"}',
        b"This model's maximum context length is 32768 tokens",
        b"Prompt length fills or exceeds max_model_len",
        b"max_tokens is too large for the remaining context",
    ],
)
def test_typed_and_stock_context_errors(body):
    assert _context_length_error_detail(error(400, body))["code"] == "context_length_exceeded"


@pytest.mark.parametrize(
    "status,body",
    [
        (500, b"context length"),
        (429, b"context length"),
        (400, b"max_tokens must be positive"),
        (400, b"invalid image"),
    ],
)
def test_unrelated_failures_are_not_relabelled(status, body):
    assert _context_length_error_detail(error(status, body)) is None


@pytest.mark.parametrize("completions", [False, True])
async def test_both_backends_propagate_typed_rejection_not_fake_generation(completions):
    model = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=1,
            entrypoint="",
            name="probe",
            base_url="http://localhost:9999/v1",
            api_key="dummy",
            model="test",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            use_completions_api=completions,
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    client = MagicMock()
    rejected = error(400, b'{"error":{"code":"context_length_exceeded"}}')
    client.create_chat_completion = AsyncMock(side_effect=rejected)
    client.create_completion = AsyncMock(side_effect=rejected)
    with patch.object(VLLMModel, "_resolve_client", return_value=client):
        with pytest.raises(HTTPException) as raised:
            await model.chat_completions(
                MagicMock(spec=Request),
                NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "hello"}]),
            )
    assert raised.value.status_code == 400
    assert raised.value.detail["error"]["code"] == "context_length_exceeded"
    assert raised.value.__cause__ is rejected
    assert (client.create_completion if completions else client.create_chat_completion).await_count == 1
