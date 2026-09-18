# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, opt-in JSONL records for rollout latency diagnosis.

The record deliberately excludes prompts, responses, images, and token arrays.
Each agent process writes its own file so concurrent services never append to
the same shared-filesystem inode.
"""

import json
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)

COMPACT_ROLLOUT_DIAGNOSTICS_ENV = "NEMO_GYM_COMPACT_ROLLOUT_DIR"
COMPACT_ROLLOUT_DIAGNOSTIC_FIELDS = (
    "game_name",
    "duration_seconds",
    "model_calls",
    "valid_environment_steps",
    "generated_tokens",
    "no_boxed_retries",
    "termination_reason",
)


def append_compact_rollout_diagnostic(
    *,
    component: str,
    game_name: str,
    duration_seconds: float,
    model_calls: int,
    valid_environment_steps: int,
    generated_tokens: int,
    no_boxed_retries: int,
    termination_reason: str,
) -> Path | None:
    """Append one seven-field record when the diagnostic directory is set."""
    root = os.environ.get(COMPACT_ROLLOUT_DIAGNOSTICS_ENV)
    if not root:
        return None

    record = {
        "game_name": game_name,
        "duration_seconds": max(0.0, float(duration_seconds)),
        "model_calls": int(model_calls),
        "valid_environment_steps": int(valid_environment_steps),
        "generated_tokens": int(generated_tokens),
        "no_boxed_retries": int(no_boxed_retries),
        "termination_reason": termination_reason,
    }
    if tuple(record) != COMPACT_ROLLOUT_DIAGNOSTIC_FIELDS:  # pragma: no cover
        raise AssertionError("Compact rollout diagnostic schema drifted")

    try:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{component}-pid{os.getpid()}.jsonl"
        payload = (json.dumps(record, separators=(",", ":")) + "\n").encode()

        # One append syscall per record, plus a per-process filename, avoids
        # both partial Python writes and cross-process contention on an inode.
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except OSError:
        # This is diagnostic-only instrumentation. A full filesystem or stale
        # mount must not turn a valid environment rollout into training loss.
        logger.warning("Could not append compact rollout diagnostic", exc_info=True)
        return None
    return path
