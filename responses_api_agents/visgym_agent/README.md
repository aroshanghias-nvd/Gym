<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VisGym rollout agent

`visgym_agent` runs a stateful, multi-turn rollout between a Responses API
policy model and the specialized VisGym resources server. It extracts the last
`\boxed{...}` item from assistant text and sends that exact string through the
server's `action_string`-only `/step` schema. Tool-call action transport is not
part of this integration.

The agent preserves the initial visual observation and per-turn token/logprob
metadata required by NeMo-RL, enforces the task and agent turn bounds, and
performs bounded idempotent cleanup after normal completion or failure.

## Dependencies and licensing

- Agent code: Apache-2.0.
- `nemo-gym`: Apache-2.0.
- `aiohttp`: Apache-2.0.
- Task data: none bundled with the agent.
