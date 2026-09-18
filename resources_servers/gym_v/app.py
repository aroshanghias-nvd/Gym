# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import isclose
from pathlib import Path
from time import monotonic
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import ConfigDict, Field

import gym_v
from gym_v import Observation
from nemo_gym.base_resources_server import SimpleResourcesServer
from resources_servers.gym_v._dataset_cache import install_dataset_cache
from resources_servers.gym_v._observation import (
    _attach_env_info,
    observation_to_user_message,
)
from resources_servers.gym_v.schemas import (
    GymVAgentVerifyRequest,
    GymVAgentVerifyResponse,
    GymVCloseRequest,
    GymVCloseResponse,
    GymVEnvStateEasyInputMessage,
    GymVResourcesServerConfig,
    GymVSeedSessionRequest,
    GymVSeedSessionResponse,
    GymVStepRequest,
    GymVStepResponse,
    GymVTaskRow,
)


logger = logging.getLogger(__name__)


async def _run_in_threadpool_to_completion(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Do not release a native-env lock while its worker thread is still running."""
    worker_task = asyncio.create_task(run_in_threadpool(func, *args, **kwargs))
    try:
        return await asyncio.shield(worker_task)
    except asyncio.CancelledError as cancellation:
        # Raw Task.cancel() bypasses AnyIO cancel-scope shielding. Wait for the
        # worker to stop touching native state, then preserve caller cancellation.
        # The request stack also uses AnyIO's level-triggered CancelScope, which
        # re-cancels every unshielded await. A nested shield plus asyncio.shield
        # covers both mechanisms; the loop also tolerates repeated raw cancels.
        with anyio.CancelScope(shield=True):
            while not worker_task.done():
                try:
                    await asyncio.shield(worker_task)
                except asyncio.CancelledError:
                    continue
        try:
            worker_task.result()
        except BaseException:
            pass
        raise cancellation


@dataclass
class _SessionOperation:
    """Per-session coordination state, published before the first await."""

    lock: asyncio.Lock = dataclass_field(default_factory=asyncio.Lock)
    seed_finished: asyncio.Event = dataclass_field(default_factory=asyncio.Event)
    close_requested: bool = False
    discard_reward_requested: bool = False


class GymVResourcesServer(SimpleResourcesServer):
    """Env-id-parametric server wrapping Gym-V single-agent envs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: GymVResourcesServerConfig

    task_rows: list[dict[str, Any]] = Field(default_factory=list)
    env_id_to_env: dict[str, gym_v.Env] = Field(default_factory=dict)
    env_id_to_total_reward: dict[str, float] = Field(default_factory=lambda: defaultdict(float))
    env_id_to_valid_action_bonus: dict[str, float] = Field(default_factory=lambda: defaultdict(float))
    env_id_to_task_success: dict[str, bool | None] = Field(default_factory=dict)
    env_id_to_task_row: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_id_to_turn_count: dict[str, int] = Field(default_factory=lambda: defaultdict(int))
    env_id_to_operation: dict[str, _SessionOperation] = Field(default_factory=dict, exclude=True)
    cleanup_tombstone_expiry: dict[str, float] = Field(default_factory=dict, exclude=True)
    background_close_tasks: set[asyncio.Task[GymVCloseResponse]] = Field(default_factory=set, exclude=True)

    def model_post_init(self, _ctx: Any) -> None:
        for jsonl_path in self.config.task_jsonl_fpaths:
            with Path(jsonl_path).open() as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    try:
                        validated = GymVTaskRow.model_validate(row)
                    except Exception:
                        logger.exception("Invalid Gym-V task row in %s:%s", jsonl_path, line_no)
                        raise
                    self.task_rows.append(validated.model_dump(mode="json"))

        install_dataset_cache()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/close")(self.close)
        return app

    async def seed_session(self, request: Request, body: GymVSeedSessionRequest) -> GymVSeedSessionResponse:
        if body.task_row is not None:
            row = body.task_row
        else:
            if body.task_idx is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either task_row or task_idx must be provided.",
                )
            if body.task_idx >= len(self.task_rows):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"task_idx={body.task_idx} out of range; "
                        f"server has {len(self.task_rows)} rows. "
                        "Did the dataset reload between training restarts?"
                    ),
                )
            row = GymVTaskRow.model_validate(self.task_rows[body.task_idx])

        success_criterion = row.success_criterion or self.config.success_criteria_by_env_id.get(row.env_id)
        if self.config.require_success_criterion and success_criterion is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"env_id={row.env_id!r} has no audited binary success criterion; "
                    "add an explicit row criterion or a server allowlist entry."
                ),
            )
        row_dict = row.model_dump(mode="json")
        if success_criterion is not None:
            row_dict["success_criterion"] = success_criterion.model_dump(mode="json")

        # Per-row env_kwargs override the server default.
        effective_env_kwargs = {
            "disable_text_feedback": self.config.disable_text_feedback,
            **row.env_kwargs,
        }
        env_id = body.env_id or str(uuid.uuid4())
        operation = self._reserve_session(env_id)
        env: gym_v.Env | None = None
        constructed_env: list[gym_v.Env | None] = [None]

        def construct_env() -> gym_v.Env:
            made_env = gym_v.make(row.env_id, **effective_env_kwargs)
            constructed_env[0] = made_env
            return made_env

        try:
            async with operation.lock:
                env = await _run_in_threadpool_to_completion(construct_env)
                obs_dict, _info_dict = await _run_in_threadpool_to_completion(env.reset, seed=row.seed)
                if set(obs_dict.keys()) != {"agent_0"}:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Multi-agent envs not supported in Stage A. "
                            f"env_id={row.env_id} returned agents={list(obs_dict.keys())}. "
                            "Filed under parent-plan Phase III."
                        ),
                    )
                obs_msg = observation_to_user_message(
                    obs_dict["agent_0"],
                    env_id=row.env_id,
                    prefix_text=self._description_for_agent_0(env),
                    image_format=self.config.image_format,
                    image_jpeg_quality=self.config.image_jpeg_quality,
                    skip_images=self.config.skip_images,
                    max_image_wh=self.config.max_image_wh,
                )

                # /close can arrive while an ambiguously completed seed request
                # is still constructing/resetting. It marks the reservation
                # before waiting for this lock, so never publish a late orphan.
                if operation.close_requested:
                    raise HTTPException(status_code=409, detail=f"env_id={env_id} was closed while being seeded.")

                self.env_id_to_env[env_id] = env
                self.env_id_to_task_row[env_id] = row_dict
                self.env_id_to_total_reward[env_id] = 0.0
                self.env_id_to_valid_action_bonus[env_id] = 0.0
                self.env_id_to_task_success[env_id] = False if success_criterion is not None else None
                self.env_id_to_turn_count[env_id] = 0
                operation.seed_finished.set()

                return GymVSeedSessionResponse(
                    env_id=env_id,
                    obs=[obs_msg],
                )
        except BaseException as exc:
            # Stop a client that knows its requested ID from racing /step
            # against partial-seed teardown after the async-with released the
            # operation lock.
            operation.close_requested = True
            if env is None:
                env = constructed_env[0]
            self._discard_session_state(env_id)
            close_failed = False
            try:
                if env is not None:
                    try:
                        await _run_in_threadpool_to_completion(env.close)
                    except BaseException:
                        # Keep the native object reachable and the ID reserved.
                        # A concurrent or agent-side abort /close can retry the
                        # idempotent teardown instead of losing an orphan after
                        # the seed request has already failed.
                        close_failed = True
                        self.env_id_to_env[env_id] = env
                        logger.warning("Failed to close partially seeded Gym-V env %s", row.env_id, exc_info=True)
            finally:
                # A concurrent /close may already be waiting on the operation
                # lock. Do not let it report completion before native teardown.
                operation.seed_finished.set()
                if not close_failed:
                    self._forget_operation(env_id, operation)
            if isinstance(exc, HTTPException):
                raise
            if not isinstance(exc, Exception):
                raise
            logger.exception("Failed to seed env %s with kwargs=%s", row.env_id, effective_env_kwargs)
            raise HTTPException(
                status_code=500,
                detail=f"env construction failed: {type(exc).__name__}: {exc}",
            ) from exc

    async def step(self, request: Request, body: GymVStepRequest) -> GymVStepResponse:
        operation = self.env_id_to_operation.get(body.env_id)
        if operation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown env_id={body.env_id}; was the session closed?",
            )
        async with operation.lock:
            if self.env_id_to_operation.get(body.env_id) is not operation or operation.close_requested:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown env_id={body.env_id}; was the session closed?",
                )
            return await self._step_session(body)

    async def _step_session(self, body: GymVStepRequest) -> GymVStepResponse:
        if body.env_id not in self.env_id_to_env:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown env_id={body.env_id}; was the session closed?",
            )

        env = self.env_id_to_env[body.env_id]
        row = self.env_id_to_task_row[body.env_id]
        env_id_str = row["env_id"]

        # Path B action transport: action_string is the canonical (and only)
        # transport since Path A was removed. The schema's Pydantic validator
        # already enforces presence.
        answer = body.action_string

        try:
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = await _run_in_threadpool_to_completion(
                env.step, {"agent_0": answer}
            )
        except Exception as exc:
            logger.warning(
                "env.step raised on env_id=%s (%s) with answer=%r: %s: %s",
                body.env_id,
                env_id_str,
                answer,
                type(exc).__name__,
                exc,
            )
            recovery = self._recovery_message(
                env_id_str,
                f"Invalid action {answer!r}: {type(exc).__name__}: {exc}",
                {"env_step_exception": str(exc), "accepted_action": False},
            )
            self.env_id_to_turn_count[body.env_id] += 1
            horizon_terminated = self._horizon_reached(body.env_id, row)
            return GymVStepResponse(
                obs=[recovery],
                reward=0.0,
                done=horizon_terminated,
                horizon_terminated=horizon_terminated,
            )

        if set(obs_dict.keys()) != {"agent_0"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Multi-agent envs not supported in Stage A. "
                    f"env_id={env_id_str} returned agents={list(obs_dict.keys())}."
                ),
            )

        reward = float(reward_dict["agent_0"])
        env_terminated = bool(terminated_dict.get("__all__", False))
        env_truncated = bool(truncated_dict.get("__all__", False))
        self._update_task_success(
            body.env_id,
            row,
            reward=reward,
            env_terminated=env_terminated,
            info=self._agent_0_info(info_dict),
        )
        self.env_id_to_total_reward[body.env_id] += reward
        remaining_bonus = max(
            0.0,
            self.config.valid_action_bonus_max - self.env_id_to_valid_action_bonus[body.env_id],
        )
        valid_action_bonus = min(self.config.valid_action_bonus, remaining_bonus)
        self.env_id_to_valid_action_bonus[body.env_id] += valid_action_bonus

        done = env_terminated or env_truncated
        self.env_id_to_turn_count[body.env_id] += 1

        horizon_terminated = self._horizon_reached(body.env_id, row)
        if horizon_terminated:
            done = True

        obs_msg = observation_to_user_message(
            obs_dict["agent_0"],
            env_id=env_id_str,
            prefix_text=None,
            image_format=self.config.image_format,
            image_jpeg_quality=self.config.image_jpeg_quality,
            skip_images=self.config.skip_images,
            max_image_wh=self.config.max_image_wh,
        )
        env_info = self._agent_0_info(info_dict)
        env_info = {
            **env_info,
            "accepted_action": True,
            "valid_action_bonus": valid_action_bonus,
        }
        obs_msg = _attach_env_info(obs_msg, env_info)

        return GymVStepResponse(
            obs=[obs_msg],
            reward=reward,
            done=done,
            horizon_terminated=horizon_terminated,
        )

    async def close(self, request: Request, body: GymVCloseRequest) -> GymVCloseResponse:
        # The cleanup request can overtake an ambiguously completed seed request
        # on a different connection. Record the abort intent before consulting
        # live state so the delayed seed cannot publish an orphan later.
        if body.discard_reward:
            self._record_cleanup_tombstone(body.env_id)
        operation = self.env_id_to_operation.get(body.env_id)
        if operation is None:
            return GymVCloseResponse(success=True, message="already closed")

        # Mark the reservation before waiting. A seed handler holding the lock
        # observes this after construction and tears down instead of publishing.
        operation.close_requested = True
        operation.discard_reward_requested |= body.discard_reward
        close_task = asyncio.create_task(self._complete_close(body.env_id, operation))
        try:
            # Request cancellation (for example, an agent-side timeout) must
            # not cancel native teardown after the server accepted /close.
            return await asyncio.shield(close_task)
        except asyncio.CancelledError:
            self._track_background_close(close_task)
            raise

    async def _complete_close(self, env_id: str, operation: _SessionOperation) -> GymVCloseResponse:
        async with operation.lock:
            await operation.seed_finished.wait()
            if self.env_id_to_operation.get(env_id) is not operation:
                return GymVCloseResponse(success=True, message="already closed")
            return await self._close_session(env_id, operation)

    async def _close_session(self, env_id: str, operation: _SessionOperation) -> GymVCloseResponse:
        env = self.env_id_to_env.get(env_id)
        if env is None:
            if operation.discard_reward_requested:
                self.env_id_to_total_reward.pop(env_id, None)
                self.env_id_to_valid_action_bonus.pop(env_id, None)
                self.env_id_to_task_success.pop(env_id, None)
                self._forget_operation(env_id, operation)
            return GymVCloseResponse(success=True, message="already closed")

        try:
            await _run_in_threadpool_to_completion(env.close)
        except Exception as exc:
            logger.warning("env.close raised on env_id=%s: %s", env_id, exc)
            return GymVCloseResponse(success=False, message=repr(exc))

        # Retire the native env only after close returns. Until then the
        # operation reservation prevents the same client ID from being reused.
        self.env_id_to_env.pop(env_id, None)
        self.env_id_to_task_row.pop(env_id, None)
        self.env_id_to_turn_count.pop(env_id, None)
        if operation.discard_reward_requested:
            self.env_id_to_total_reward.pop(env_id, None)
            self.env_id_to_valid_action_bonus.pop(env_id, None)
            self.env_id_to_task_success.pop(env_id, None)
            self._forget_operation(env_id, operation)

        return GymVCloseResponse(success=True, message="ok")

    async def verify(self, request: Request, body: GymVAgentVerifyRequest) -> GymVAgentVerifyResponse:
        env_id = body.response.env_id
        operation = self.env_id_to_operation.get(env_id)
        if operation is not None:
            async with operation.lock:
                if self.env_id_to_operation.get(env_id) is operation:
                    if env_id in self.env_id_to_env:
                        raise HTTPException(
                            status_code=409,
                            detail=f"env_id={env_id} must be closed before verification.",
                        )
                    response = self._verify_session(body)
                    self._forget_operation(env_id, operation)
                    return response
        return self._verify_session(body)

    def _verify_session(self, body: GymVAgentVerifyRequest) -> GymVAgentVerifyResponse:
        env_id = body.response.env_id
        known_env_id = env_id in self.env_id_to_total_reward
        env_base_reward = self.env_id_to_total_reward.pop(env_id, 0.0)
        valid_action_bonus = self.env_id_to_valid_action_bonus.pop(env_id, 0.0)
        reward = env_base_reward + valid_action_bonus
        task_success = self.env_id_to_task_success.pop(env_id, None)
        metadata = dict(body.response.metadata or {})
        metadata["env_base_reward"] = str(env_base_reward)
        metadata["valid_action_bonus"] = str(valid_action_bonus)
        body.response.metadata = metadata
        termination_reason = metadata.get("termination_reason")
        failure_reason_by_termination = {
            "context_length_exceeded": "context_length_exceeded",
            "model_token_budget_exceeded": "generation_token_budget_exceeded",
        }
        failure_reason = failure_reason_by_termination.get(termination_reason)
        if not known_env_id:
            logger.info("/verify drained unknown env_id=%s; returning 0.0", env_id)
        return GymVAgentVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=reward,
            failure_reason=failure_reason,
            instance_config={"mask_sample": failure_reason is not None},
            task_success=task_success,
        )

    def _session_id_in_use(self, env_id: str) -> bool:
        return any(
            env_id in state
            for state in (
                self.env_id_to_env,
                self.env_id_to_total_reward,
                self.env_id_to_valid_action_bonus,
                self.env_id_to_task_success,
                self.env_id_to_task_row,
                self.env_id_to_turn_count,
                self.env_id_to_operation,
            )
        )

    def _reserve_session(self, env_id: str) -> _SessionOperation:
        # No await in this critical section: one event-loop turn atomically
        # checks and publishes the reservation.
        if self._has_active_cleanup_tombstone(env_id):
            raise HTTPException(status_code=409, detail=f"env_id={env_id} was already closed.")
        if self._session_id_in_use(env_id):
            raise HTTPException(status_code=409, detail=f"env_id={env_id} is already in use.")
        operation = _SessionOperation()
        self.env_id_to_operation[env_id] = operation
        return operation

    def _record_cleanup_tombstone(self, env_id: str) -> None:
        now = monotonic()
        self._prune_cleanup_tombstones(now)
        self.cleanup_tombstone_expiry[env_id] = now + self.config.cleanup_tombstone_ttl_seconds

    def _has_active_cleanup_tombstone(self, env_id: str) -> bool:
        now = monotonic()
        expiry = self.cleanup_tombstone_expiry.get(env_id)
        if expiry is None:
            return False
        if expiry <= now:
            self.cleanup_tombstone_expiry.pop(env_id, None)
            return False
        return True

    def _prune_cleanup_tombstones(self, now: float) -> None:
        for env_id, expiry in list(self.cleanup_tombstone_expiry.items()):
            if expiry <= now:
                self.cleanup_tombstone_expiry.pop(env_id, None)

    def _discard_session_state(self, env_id: str) -> None:
        self.env_id_to_env.pop(env_id, None)
        self.env_id_to_task_row.pop(env_id, None)
        self.env_id_to_total_reward.pop(env_id, None)
        self.env_id_to_valid_action_bonus.pop(env_id, None)
        self.env_id_to_task_success.pop(env_id, None)
        self.env_id_to_turn_count.pop(env_id, None)

    def _forget_operation(self, env_id: str, operation: _SessionOperation) -> None:
        if self.env_id_to_operation.get(env_id) is operation:
            self.env_id_to_operation.pop(env_id, None)

    def _track_background_close(self, task: asyncio.Task[GymVCloseResponse]) -> None:
        self.background_close_tasks.add(task)
        task.add_done_callback(self._background_close_finished)

    def _background_close_finished(self, task: asyncio.Task[GymVCloseResponse]) -> None:
        self.background_close_tasks.discard(task)
        if task.cancelled():
            logger.warning("Background Gym-V session close was cancelled before cleanup completed.")
            return
        try:
            task.result()
        except BaseException:
            logger.exception("Background Gym-V session close failed unexpectedly.")

    def _horizon_reached(self, env_id: str, row: dict[str, Any]) -> bool:
        return bool(
            self.config.enforce_horizon_cap
            and row.get("horizon_cap") is not None
            and self.env_id_to_turn_count[env_id] >= row["horizon_cap"]
        )

    def _update_task_success(
        self,
        env_id: str,
        row: dict[str, Any],
        *,
        reward: float,
        env_terminated: bool,
        info: dict[str, Any],
    ) -> None:
        """Apply only the task row's audited predicate; never infer success generically."""
        criterion = row.get("success_criterion")
        if criterion is None or self.env_id_to_task_success.get(env_id) is True:
            return

        criterion_type = criterion["type"]
        if criterion_type == "terminal_step_reward":
            if not env_terminated:
                return
            threshold = float(criterion["threshold"])
            if criterion["comparison"] == "equal":
                succeeded = isclose(
                    reward,
                    threshold,
                    rel_tol=0.0,
                    abs_tol=float(criterion["absolute_tolerance"]),
                )
            else:
                succeeded = reward > threshold
            self.env_id_to_task_success[env_id] = succeeded
            return

        if criterion["require_env_terminated"] and not env_terminated:
            return
        value: Any = info
        for key in criterion["path"]:
            if not isinstance(value, dict) or key not in value:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Gym-V success criterion path {criterion['path']!r} is missing for env_id={row['env_id']!r}."
                    ),
                )
            value = value[key]
        if not isinstance(value, bool):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Gym-V success criterion path {criterion['path']!r} must resolve to bool "
                    f"for env_id={row['env_id']!r}, got {type(value).__name__}."
                ),
            )
        self.env_id_to_task_success[env_id] = value

    @staticmethod
    def _description_for_agent_0(env: gym_v.Env) -> str:
        description = env.description
        if isinstance(description, str):
            return description
        return description.get("agent_0", "")

    @staticmethod
    def _agent_0_info(info_dict: dict[str, Any]) -> dict[str, Any]:
        info = info_dict.get("agent_0", info_dict)
        return info if isinstance(info, dict) else {"info": info}

    def _recovery_message(
        self,
        env_id_str: str,
        text: str,
        env_info: dict[str, Any],
    ) -> GymVEnvStateEasyInputMessage:
        recovery = observation_to_user_message(
            Observation(image=None, text=text, metadata=env_info),
            env_id=env_id_str,
            prefix_text=None,
            image_format=self.config.image_format,
            image_jpeg_quality=self.config.image_jpeg_quality,
            max_image_wh=self.config.max_image_wh,
        )
        return _attach_env_info(recovery, env_info)


if __name__ == "__main__":
    GymVResourcesServer.run_webserver()
