# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real local graders with final-answer responses, without model/judge calls."""

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    LibraryJudgeMathVerifyRequest,
)
from resources_servers.mcqa.app import MCQAResourcesServer, MCQAResourcesServerConfig, MCQAVerifyRequest
from resources_servers.string_match.app import (
    StringMatchResourcesServer,
    StringMatchResourcesServerConfig,
    StringMatchVerifyRequest,
)


def response(text):
    return NeMoGymResponse(
        id="test",
        created_at=0,
        model="test",
        object="response",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[
            {
                "id": "answer",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gold,answer,expected",
    [
        ("42", r"\boxed{42}", 1.0),
        ("42", r"\boxed{43}", 0.0),
        (r"\frac{1}{2}", r"\boxed{0.5}", 1.0),
        (r"\(42\)", r"\boxed{42}", 1.0),
        ("42", "", 0.0),
    ],
)
async def test_math_actual_verify_without_judge(gold, answer, expected):
    client = MagicMock(spec=ServerClient)
    server = LibraryJudgeMathResourcesServer(
        config=LibraryJudgeMathResourcesServerConfig(
            host="localhost",
            port=10001,
            name="math_with_judge",
            entrypoint="app.py",
            should_use_judge=False,
            judge_model_server={"type": "responses_api_models", "name": "policy_model"},
            judge_responses_create_params={"input": []},
            library_verifier_timeout_seconds=5,
        ),
        server_client=client,
    )
    result = await server.verify(
        LibraryJudgeMathVerifyRequest(
            responses_create_params={"input": []},
            question="synthetic",
            expected_answer=gold,
            response=response(answer),
        )
    )
    assert result.reward == expected
    assert result.judge_evaluations is None
    client.post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Final answer: blue", 1.0),
        (r"\boxed{blue}", 1.0),
        ("Final answer: red", 0.0),
        ("", 0.0),
    ],
)
async def test_string_actual_verify(answer, expected):
    client = MagicMock(spec=ServerClient)
    server = StringMatchResourcesServer(
        config=StringMatchResourcesServerConfig(
            host="localhost",
            port=10001,
            name="string_match",
            entrypoint="app.py",
        ),
        server_client=client,
    )
    result = await server.verify(
        StringMatchVerifyRequest(
            responses_create_params={"input": []},
            expected_answer="blue",
            response=response(answer),
            extraction_mode="final_answer",
        )
    )
    assert result.reward == expected
    client.post.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [(r"\boxed{B}", 1.0), (r"\boxed{A}", 0.0), ("", 0.0)])
async def test_mcqa_actual_verify(answer, expected):
    client = MagicMock(spec=ServerClient)
    server = MCQAResourcesServer(
        config=MCQAResourcesServerConfig(
            host="localhost",
            port=10001,
            name="mcqa",
            entrypoint="app.py",
        ),
        server_client=client,
    )
    result = await server.verify(
        MCQAVerifyRequest(
            responses_create_params={"input": []},
            expected_answer="B",
            response=response(answer),
            options=[{"A": "red"}, {"B": "blue"}],
            grading_mode="strict_single_letter_boxed",
        )
    )
    assert result.reward == expected
    client.post.assert_not_called()
