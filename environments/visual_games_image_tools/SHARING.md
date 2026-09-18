# Experimental mixed visual GRPO environment

Local sharing branch: `aroshanghias/mixed-visual-grpo`, based on Gym
`e978899b7c79f2b90c23af7bfe91a36349360699` from the Super 3.5 stack.
Use the exact Gym commit pinned by the companion NeMo-RL branch.

This includes the selectively ported Gym-V and VisGym implementations plus the
image-tools integration and local fixes. It does not require either source PR
to merge, and it is not a merge of their current tips:

- Gym-V: [Gym #2700](https://github.com/NVIDIA-NeMo/Gym/pull/2700), OPEN on
  2026-09-18; observed head `b93e7fe530e5f8b683b383eebd8478c1ba0ede0c`.
- VisGym: [Gym #2730](https://github.com/NVIDIA-NeMo/Gym/pull/2730), OPEN on
  2026-09-18; observed head `ffd7dd706ee17e1fcc2221008b7b8444f35c8eca`.

The branch captures the implementations used by the local Super 3.5 mixed-run
integration, with subsequent guards and diagnostics; the observed PR heads above
are provenance pointers, not claims that their exact complete diffs were copied.
The existing files retain their authorship/license headers. Do not replace this
pin with upstream main and assume the unmerged environment code is present.

`config.yaml` combines separate resource-server/agent pairs for Gym-V and VisGym
with `environments/image_tools_grpo/config.yaml`. Keep their dependency environments
separate: VisGym uses its own Gymnasium fork. All agents share the policy model.

The VisGym candidate wheel is deliberately excluded. Before provisioning its
resources server, build it from the pinned source using:

```bash
resources_servers/visgym/scripts/build_visgym_wheel.sh
```

Its provenance, license notices and existing review caveat remain under
`resources_servers/visgym/`. The build needs network access/build dependencies;
this branch does not establish approval to redistribute the resulting wheel.
Training datasets, model weights, rendered-game assets and credentials are not
included. See the companion RL branch's `MIXED_VISUAL_GRPO.md` for training setup.
