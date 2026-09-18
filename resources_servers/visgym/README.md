# VisGym integration

This package connects [VisGym](https://github.com/visgym/VIsGym) visual,
interactive environments to NeMo-Gym. VisGym is distributed as a fork of
`gymnasium`; environments are constructed through the familiar
`gymnasium.make(env_id, **env_kwargs)` interface, but observations are images
and actions are environment-specific strings.

Unlike a static visual-question-answering dataset, a VisGym sample is a
stateful episode:

1. the environment is constructed and reset;
2. the reset produces the first visual observation;
3. the policy inspects that image and emits one action;
4. the environment applies the action and returns the next image and reward;
5. policy and environment turns repeat until termination or a configured cap;
6. NeMo-Gym verifies the episode and returns its accumulated training reward.

The initial validated integration targets `maze_2d/easy`. The resource server
is parameterized by `env_id`, so the other 16 namespaced VisGym task families
can be added through configuration and task data without creating another
server implementation. Arbitrary environments from the forked Gymnasium
registry are not automatically valid training tasks.

## Components

| Component | Location | Responsibility |
| --- | --- | --- |
| Environment bundle | `environments/visgym` | Provides the canonical config and data-preparation entry point. |
| Resource server | `resources_servers/visgym` | Owns environment instances, renders observations, applies actions, and accumulates reward. |
| Rollout agent | `responses_api_agents/visgym_agent` | Alternates model calls with environment steps and extracts actions from model text. |
| Model server | `responses_api_models/vllm_model` | Generates policy responses and preserves token information needed for on-policy training. |
| Task rows | `resources_servers/visgym/data/*.jsonl` | Select the environment, seed, horizon, action grammar, and Responses API request. |

The data flow for one turn is:

```text
VisGym observation
  -> NeMo-Gym user message (text + image)
  -> Responses API policy call
  -> assistant text ending in \boxed{action}
  -> visgym_agent action extraction
  -> resources server /step
  -> next observation and reward
```

The generic `gymnasium` server and agent remain the right choice for string
observations and function-call actions. VisGym keeps a specialized pair because
it must transport multimodal image messages, parse boxed text actions, drain a
final accumulated training reward, and preserve exact prompt/generation token
metadata across every turn. Reusing the generic pair would require changing its
public request and response schemas for all existing consumers.

## Environment lifecycle

### Seed a session

`POST /seed_session` accepts either a `task_idx` loaded from configured JSONL
files or an inline `task_row`. The server:

- calls `gymnasium.make()` with `env_id` and `env_kwargs`;
- calls `env.reset(seed=..., init_state=...)` when an initial state is present;
- obtains the environment prompt with `env.get_prompt(**prompt_kwargs)` when
  available;
- converts the returned observation into a user message;
- allocates a server-side UUID used by subsequent requests.

The first image does not exist until reset, so task rows normally use an empty
Responses API input. The reset observation is returned separately as
`seed_obs`. Keeping `seed_obs` separate from generated output is important for
training: generated assistant messages carry prompt token IDs, generation token
IDs, and log probabilities, while the reset observation is environment state.

### Apply an action

`POST /step` accepts the session UUID and an `action_string`. The server calls
`env.step(action_string)`, records the turn count, accumulates reward, and
returns the next visual observation.

If an environment rejects an action, the server returns a recoverable user
observation with:

- reward `0.0`;
- `done=false`;
- a rendered image when available;
- a concise description of the invalid action.

This lets the model correct its action without losing the episode. A configured
`horizon_cap` can terminate an episode even when the underlying environment has
not terminated.

### Verify reward

`POST /verify` drains the accumulated reward for the session, attaches it to
the response metadata as `training_reward`, and returns it to NeMo-Gym. Reward
is accumulated across all turns rather than inferred from the final text.
It separately emits binary `task_success` only when an audited VisGym family
terminates with raw reward `1.0`; positive nonterminal or shaped rewards do not
count as success. Strict mode rejects registry IDs outside the 34 audited
`{easy,hard}` task IDs. Disabling it is for reward-only diagnostics and returns
`task_success=null` for those unsupported IDs.

### Close a session

`POST /close` releases the environment and removes server-side state. Closing
an already-closed session is treated as a successful no-op. The rollout agent
attempts to close sessions both on normal termination and after failures.

## Observation transport

Each environment observation becomes a user-role Responses API message. Its
content can contain:

- an `input_text` part with the environment prompt or step feedback;
- an `input_image` part encoded as a PNG or JPEG data URL;
- sanitized `env_info` metadata for inspection and debugging.

NumPy arrays, PIL images, and image-like observations are supported. If an
environment returns no image-like observation and `render_on_missing_image` is
enabled, the server calls `env.render()` as a fallback.

Server options control this conversion:

| Option | Default | Effect |
| --- | --- | --- |
| `image_format` | `PNG` | Data-URL image encoding. JPEG is also supported. |
| `image_jpeg_quality` | `90` | JPEG quality when JPEG encoding is selected. |
| `skip_images` | `false` | Omits image parts for text-only diagnostics. |
| `include_env_feedback` | `true` | Includes `info["env_feedback"]` in the next user message. |
| `render_on_missing_image` | `true` | Uses `env.render()` when reset or step does not return an image. |
| `enforce_horizon_cap` | `true` | Enforces the per-task `horizon_cap`. |

Metadata is recursively converted to JSON-safe values. Large image payloads
remain in message content rather than being copied into `env_info`.

## Action transport

The paired `visgym_agent` uses plain-text action transport. It takes the last
boxed item from the assistant response and sends it to `/step` as
`action_string`:

```text
<think>The open path is to the right.</think>
\boxed{('move', 1)}
```

For `maze_2d/easy`, legal actions are:

```text
('move', 0)
('move', 1)
('move', 2)
('move', 3)
('stop', 'stop')
```

The task's `act_grammar_regex` documents the legal grammar, while the agent's
`unboxed_action_regex` can accept an exact unboxed action for models that omit
the wrapper. Prose surrounding an unboxed action is rejected.

Important agent options include:

| Option | Purpose |
| --- | --- |
| `max_steps` | Global rollout-turn backstop in addition to the task horizon. |
| `done_if_no_boxed_answer` | Ends the rollout when no valid action can be extracted. |
| `max_no_boxed_truncation_retries` | Retries a truncated model response that omitted an action. |
| `no_boxed_truncation_retry_factor` | Increases the output-token budget on each truncation retry. |
| `re_emit_rules_each_turn` | Prepends a concise action-format reminder on later turns. |
| `rules_summary_template` | Per-environment reminder text. |
| `return_transitions` | Returns transition records instead of the compact training history. |

## Task-row format

A task row fully describes an episode. This is a shortened version of the
committed 5x5 maze fixture:

```json
{
  "agent_ref": {"type": "responses_api_agents", "name": "visgym_agent"},
  "env_id": "maze_2d/easy",
  "env_kwargs": {"maze_width": 5, "maze_height": 5},
  "seed": 1234,
  "task_id": "maze_2d_easy_seed1234_5x5",
  "act_grammar_regex": "^\\('(?:move|stop)',\\s*(?:[0-3]|'stop')\\)$",
  "horizon_cap": 8,
  "task_metadata": {
    "suite": "visgym",
    "difficulty": "easy",
    "maze_size": "5x5"
  },
  "responses_create_params": {
    "model": "policy_model",
    "input": [],
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "tools": []
  }
}
```

Additional optional fields are:

- `init_state`: explicit state passed to `env.reset()`;
- `seed_key`: constructor keyword that should also receive the seed;
- `prompt_kwargs`: keyword arguments for `env.get_prompt()`.

Task rows may be preloaded with `task_jsonl_fpaths` or passed inline to
`/seed_session`.

## Maze-size curriculum dataset

The generated curriculum increases maze size in four ordered stages:

| Stage | Maze size | Rows | Seed range | Horizon cap |
| --- | --- | ---: | --- | ---: |
| 1 | 5x5 | 1280 | 1234-2513 | 10 |
| 2 | 7x7 | 1280 | 11234-12513 | 20 |
| 3 | 9x9 | 1280 | 21234-22513 | 34 |
| 4 | 11x11 | 1280 | 31234-32513 | 52 |

Each cap covers the longest possible shortest path in a perfect maze of that
size, the required terminal `stop`, and three recovery turns. This matters even
for an optimal policy: the old size-linear caps made some generated 7x7, 9x9,
and 11x11 seeds impossible to finish. The provider tests now solve all 5120
generated mazes against the pinned VisGym runtime and enforce the recovery
cushion. Seed ranges are disjoint so the curriculum tasks are unique.

The full curriculum is ~8 MB across five files, so it is generated rather than
committed. Build it (deterministically) before training:

```bash
python environments/visgym/prepare.py maze
```

Then use the combined manifest for ordered curriculum training:

```text
data/maze_2d_easy_curriculum_5x5_7x7_9x9_11x11_1280each_t1024.jsonl
```

Generated curriculum files are gitignored. The minimal
`maze_2d_easy_example.jsonl` and `maze_2d_easy_smoke.jsonl` fixtures remain
committed so clean clones can validate the environment without preparing the
training set.

Rows 0-1279 are 5x5, rows 1280-2559 are 7x7, rows 2560-3839 are 9x9, and rows
3840-5119 are 11x11. NeMo-RL must use `data.shuffle=false`; shuffling the
combined file turns it into a mixed-size dataset rather than a curriculum. With
the recipe's default 64 prompts per step, each 1280-row stage supplies 20
prompt batches before the collector advances to the next size.

Separate stage manifests are included for experiments that need to hold,
repeat, or evaluate one stage independently. The manifest index records stage
paths, row counts, seed ranges, and horizon caps:

```text
data/maze_2d_easy_curriculum_5x5_7x7_9x9_11x11_manifest_index_t1024.json
```

Each row includes `curriculum_name`, `curriculum_stage`, `maze_size`, and
`curriculum_stage_index` in `task_metadata`, so metrics can be grouped by stage
without parsing task IDs.

Regenerate the dataset deterministically from the Gym repository root:

```bash
python environments/visgym/prepare.py maze
```

The generator accepts `--sizes`, `--samples-per-stage`, `--seed-base`,
`--seed-stride`, `--temperature`, and `--max-output-tokens`. Sizes must be
unique, odd, and strictly increasing. For example:

```bash
python environments/visgym/prepare.py maze \
  --sizes 5,7,9,11 \
  --samples-per-stage 1280
```

## Configuration

`environments/visgym/config.yaml` is the canonical runnable environment config.
The configs under `resources_servers/visgym/configs` are specialized training
variants. A model-server config must be loaded alongside them. For NeMo-RL
training, the effective NeMo-Gym configuration contains:

```yaml
config_paths:
  - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
  - resources_servers/visgym/configs/visgym_maze2d_thinking_agent.yaml
```

The maze config uses a short system prompt, re-emits compact rules on every
turn, caps the agent at 64 turns (above every default curriculum horizon), and
validates exact maze action strings.

## Dependency isolation and headless rendering

VisGym currently ships as a Gymnasium fork. NeMo-Gym installs
`requirements.txt` into this server's own venv, so VisGym's `gymnasium` does
not replace the version other servers use.

That install goes through `uv`, which **cannot parse VisGym's pyproject**: it
inherits Gymnasium's duplicate extras (`classic-control` and `classic_control`,
`mujoco-py` and `mujoco_py`, `toy-text` and `toy_text`), and PEP 685
normalization makes those collide:

```text
TOML parse error ... duplicate normalized extra name `classic-control`
```

pip still accepts them, so `requirements.txt` installs a candidate wheel built
from the pinned revision. This change proposes tracking that approximately
482-KB wheel so a clean checkout can resolve every local path in its
requirements file and preserve this server's process-level isolation. The
repository does not otherwise track wheels, so accepting the binary is an
explicit maintainer policy decision, not an existing convention. The candidate
contains only Python/package-data files and is tagged `py3-none-any`.

The adjacent
`vendor_wheels/gymnasium-1.1.1-py3-none-any.whl.provenance.json` records the
source repository and commit, SHA-256 digest, build inputs, and all embedded
license/notice paths. The build preserves VisGym's Apache-2.0 `LICENSE` and
`NOTICE` and adds the byte-for-byte MIT `LICENSE` from the official Gymnasium
v1.1.1 tag (`17eff00220d210beda933f78b0e52850022b0690`).

This notice preservation is deliberately conservative, but it does **not**
resolve the source project's licensing contradiction. The pinned VisGym repo
was squash-created, contains substantial Gymnasium v1.1.1 code, declares MIT in
`pyproject.toml`, and has only an Apache-2.0 root `LICENSE` plus its VisGym
`NOTICE`; its history has no MIT notice or authoritative dual-license
declaration. Treat redistribution or publication of the candidate wheel as
blocked until VisGym clarifies the licensing or the appropriate maintainer/legal
review approves it. A successful build or test run is not that approval.

A normal checkout, CI job, or source mirror therefore installs this one
dependency locally and does not clone VisGym. Other packages in
`requirements.txt` still need the normal package index or a populated `uv`
cache while the venv is provisioned; once provisioned, starting or running the
VisGym server does not download its implementation.

### Building the VisGym wheel

The candidate wheel should only change when the source-verified VisGym pin
changes or the build recipe is deliberately refreshed. Reproduce it with:

```bash
resources_servers/visgym/scripts/build_visgym_wheel.sh
```

That clones `https://github.com/visgym/VIsGym.git` at the revision pinned in
the script, verifies the exact clean commit, exports it to a temporary source
tree, and adds the byte-verified copy in
`vendor_licenses/gymnasium-v1.1.1-LICENSE`. It then fixes archive timestamps to
the commit time and runs `pip wheel --no-deps` with the build backend pinned in
`scripts/visgym-wheel-build-constraints.txt`. It leaves
`vendor_wheels/gymnasium-1.1.1-py3-none-any.whl`. The package is named
`gymnasium` because VisGym is a fork of it.

Two prerequisites are worth checking first, because both fail on machines that
otherwise look fine:

- **Python 3.13 or newer with pip 25.3 or newer.** `uv` cannot substitute for
  the wheel build -- it is the installer that rejects the source project. The
  script needs pip's `--build-constraint` support to pin setuptools. Point
  `PYTHON_BIN` at the appropriate interpreter if `python3` is older or lacks
  pip.
- **Outbound access to `github.com`**, unless you supply the source yourself.
  The isolated build also needs the pinned setuptools release from an index or
  pip cache. Neither is needed for ordinary installation of the candidate wheel
  if it is accepted.

To audit or rebuild without cloning inside the script, supply a clean checkout
at the pinned commit:

```bash
VISGYM_REPO_ROOT=/path/to/VIsGym \
  resources_servers/visgym/scripts/build_visgym_wheel.sh
```

`OUT_DIR` and `PYTHON_BIN` override the output directory and interpreter. The
source URL and revision are constants so the script cannot silently produce an
unpinned artifact under the fixed filename. Confirm the result with:

```bash
ls resources_servers/visgym/vendor_wheels/gymnasium-*.whl
```

Two builds from the same clean commit with the recorded Python, pip, and
setuptools versions must have the SHA-256 digest recorded in the provenance
file. The build refuses to replace the candidate when its digest differs. The
packaging test checks that digest, the ZIP integrity, the distribution
name/version, the platform-independent wheel tag, the VisGym environment
modules, and the byte-exact embedded Gymnasium MIT notice alongside VisGym's
Apache-2.0 license and NOTICE.

Once VisGym drops the duplicate extras, the wheel step can go away and
`requirements.txt` can name the git revision directly.

The `fetch_*` tasks need the Gymnasium-Robotics fork from the VisGym source
checkout, while `counting` needs the pinned public `lvis` release. They live in
`requirements-robotics.txt` and are built or downloaded by
`scripts/build_vendor_wheels.sh`, which takes a VisGym checkout as its first
argument for the robotics source:

```bash
resources_servers/visgym/scripts/build_vendor_wheels.sh /path/to/VIsGym
```

The checked-in real-runtime smoke currently covers only `maze_2d/easy`; the
other environments still need their declared assets/dependencies and bounded
reset/step validation before they are treated as supported.

For source-checkout development, set `VISGYM_REPO_ROOT` before importing the
server. The path is inserted before importing `gymnasium`.

The server defaults to headless rendering:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

`maze_2d/easy` does not require MuJoCo robotics assets or an external dataset.

## Exact token history during training

Multi-turn on-policy training must use the exact prompt token IDs sampled by
vLLM. Re-rendering previous assistant text can produce different whitespace or
multimodal placeholder tokens, making the rollout off-policy.

The training model proxy therefore promotes the latest recorded prompt and
generation token IDs to top-level `required_prefix_*` request fields. NeMo-RL
splices that exact prefix into the next rendered prompt and realigns historical
and current image placeholders independently. This behavior is training-only;
it does not change the resource-server API.

## Testing

From the Gym repository root:

```bash
pytest -q \
  resources_servers/visgym/tests \
  responses_api_agents/visgym_agent/tests/test_app.py
```

The unit tests cover task validation, observation conversion, lifecycle and
reward accumulation, invalid-action recovery, action extraction, truncation
retries, rule re-emission, and cleanup. The real-environment smoke test runs
when the pinned VisGym dependency is installed and otherwise reports a skip.

## NeMo-RL training

Training is launched from the companion NeMo-RL checkout, not from this Gym
package. The combined local integration provides this one-update smoke recipe:

```text
examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-visual-games-suite-2n8g-train-smoke.v1.yaml
```

Its workflow is documented at
`examples/nemo_gym/visual_games_suite/README.md` in that checkout. It places one
Gym-V prompt group and one VisGym prompt group into the same GRPO learner
update while NeMo-Gym starts distinct agent/resource-server pairs and distinct
dependency environments. No standalone cluster launcher is supplied here;
allocation, container, checkpoint, cache, and output locations must come from
the selected cluster's reviewed workflow.

## Adding another VisGym environment

1. Confirm the pinned VisGym package registers the desired `env_id`.
2. Add a small JSONL task fixture with deterministic seeds and a finite
   `horizon_cap`.
3. Define the legal `act_grammar_regex` and align the agent's
   `unboxed_action_regex` and prompt with that grammar.
4. Verify that reset and step observations are image-like or that
   `env.render()` provides an image.
5. Add a real reset/step smoke test and a mocked server lifecycle test.
6. Run a bounded NeMo-RL smoke before increasing rollout length or parallelism.

## Current scope and limitations

- Real VisGym execution has been validated for `maze_2d/easy`; the remaining
  rendered environments are configuration candidates, not runtime-validated
  support claims.
- Robotics and RefCOCO+ tasks require the optional forked dependencies and
  their data/assets.
- The full NeMo-Gym publication gate requires five genuine model rollouts in
  `data/example_rollouts.jsonl`; unit fixtures are not substitutes.
- The combined NeMo-RL learner smoke still requires execution on the target GPU
  cluster before this integration is training-ready.
- The vendored VisGym/Gymnasium wheel is suitable for local evaluation, but its
  license and standard package metadata need maintainer or legal clarification
  before external redistribution. Installed tooling currently identifies it as
  the official Gymnasium 1.1.1 distribution rather than an explicit VisGym fork.
