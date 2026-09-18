# Gym-V NeMo-Gym resources server

Env-id-parametric NeMo-Gym resources server wrapping any
[`gym_v.Env`](https://github.com/rohitrango/gym-v) behind `/seed_session`,
`/step`, `/close`, and `/verify`. Env selection is driven entirely by JSONL
task rows (`env_id`, `seed`, `env_kwargs`, ...). Adding an environment does not
require another server implementation, but training also requires an audited
binary-success predicate for that exact environment contract.

## Layout

```
resources_servers/gym_v/
  __init__.py
  app.py                  # GymVResourcesServer + DEFAULT_SYSTEM_PROMPT
  schemas.py              # Pydantic wire shapes
  _metadata.py            # JSON-safe Observation.metadata walker
  _observation.py         # observation_to_user_message + image_to_data_url
  _dataset_cache.py       # per-class LRU cache on reasoning-gym _make_dataset
  configs/gym_v.yaml      # server defaults
  data/example.jsonl      # smoke-test task rows
  requirements.txt        # rohitrango/gym-v @ 7ccf14e... + pillow
  tests/test_server.py
```

## Running the helper-level tests

The observation, metadata, schema, and task-data checks do not require `gym_v`
to be installed (they use an `ObservationLike` protocol). The server module
skips cleanly outside its isolated runtime. Run from a venv with `pytest`,
`pydantic`, `pillow`, and `nemo_gym` on `PYTHONPATH`:

```bash
export PYTHONPATH=<repo>:<repo>/3rdparty/Gym-workspace/Gym
cd <repo>/3rdparty/Gym-workspace/Gym
python -m pytest resources_servers/gym_v/tests -v --tb=short
```

## Per-server venv (full stack)

```bash
RAY_TMPDIR=/tmp gym env test --resources-server gym_v
```

First run installs `gym-v[games,spatial,reasoning-gym]` from the
rohitrango fork pinned at `7ccf14ecb5241f37372dd530accf77f7e7ef5fc6`.

## Paired agent

This server speaks `/seed_session` + `/step` + `/close` + `/verify` and
is paired with [`responses_api_agents/gymv_agent/`](../../responses_api_agents/gymv_agent/),
which extracts `\boxed{...}` from model output and drives the step loop.

## Reward and task success

Gym-V does not expose one universal success flag. Many environments award
partial credit, successful MiniGrid episodes can return less than `1.0`, and
termination/truncation alone does not distinguish success. The server therefore
keeps the accumulated float `reward` as the learner signal and emits a separate
`task_success` only from an explicit typed predicate.

Predicates can be supplied per task row with `success_criterion` or through the
server's exact `success_criteria_by_env_id` mapping. The checked-in config keeps
the reward-only path enabled for the proven Games22 suite and contains optional
predicates audited for five example environments against the Gym-V revision
pinned in `requirements.txt`. Rows without a predicate return
`task_success=null`; their environment reward remains the learner signal, but
they are not suitable for binary accuracy, pass@k, or dynamic-sampling decisions.
