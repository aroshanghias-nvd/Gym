# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

from nemo_gym.compact_rollout_diagnostics import (
    COMPACT_ROLLOUT_DIAGNOSTIC_FIELDS,
    COMPACT_ROLLOUT_DIAGNOSTICS_ENV,
    append_compact_rollout_diagnostic,
)


def test_compact_rollout_diagnostic_is_opt_in(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(COMPACT_ROLLOUT_DIAGNOSTICS_ENV, raising=False)

    path = append_compact_rollout_diagnostic(
        component="agent",
        game_name="game/easy",
        duration_seconds=1.25,
        model_calls=2,
        valid_environment_steps=1,
        generated_tokens=321,
        no_boxed_retries=0,
        termination_reason="env_done",
    )

    assert path is None
    assert list(tmp_path.iterdir()) == []


def test_compact_rollout_diagnostic_contains_only_allowed_fields(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(COMPACT_ROLLOUT_DIAGNOSTICS_ENV, str(tmp_path))

    path = append_compact_rollout_diagnostic(
        component="visgym-agent",
        game_name="maze_2d/easy",
        duration_seconds=3.5,
        model_calls=4,
        valid_environment_steps=3,
        generated_tokens=2047,
        no_boxed_retries=1,
        termination_reason="max_steps",
    )

    assert path is not None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert tuple(rows[0]) == COMPACT_ROLLOUT_DIAGNOSTIC_FIELDS
    assert rows[0] == {
        "game_name": "maze_2d/easy",
        "duration_seconds": 3.5,
        "model_calls": 4,
        "valid_environment_steps": 3,
        "generated_tokens": 2047,
        "no_boxed_retries": 1,
        "termination_reason": "max_steps",
    }
    assert "image" not in path.read_text()
    assert "token_ids" not in path.read_text()


def test_compact_rollout_diagnostic_failure_does_not_escape(
    monkeypatch, tmp_path, caplog
) -> None:
    monkeypatch.setenv(COMPACT_ROLLOUT_DIAGNOSTICS_ENV, str(tmp_path))

    def fail_open(*args, **kwargs):
        raise OSError("diagnostic filesystem unavailable")

    monkeypatch.setattr(os, "open", fail_open)

    path = append_compact_rollout_diagnostic(
        component="gymv-agent",
        game_name="Games/Wordle-v0",
        duration_seconds=2.0,
        model_calls=1,
        valid_environment_steps=0,
        generated_tokens=128,
        no_boxed_retries=0,
        termination_reason="no_boxed_retry_cap",
    )

    assert path is None
    assert "Could not append compact rollout diagnostic" in caplog.text
