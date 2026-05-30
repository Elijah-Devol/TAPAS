# collect_data_rlbench.py

## Purpose
Specialised data-collection script for RLBench. Uses RLBench's built-in scripted
demo generator (`task_env.get_demos`) rather than running a TAPAS policy in the
loop.

## What it does
1. Builds the RLBench env and a `SceneDataset` replay buffer (no policy is
   instantiated here).
2. For each requested episode, calls `env.task_env.get_demos(1, live_demos=True,
   ...)` to generate a scripted demonstration.
3. Iterates over the resulting demo, converting each raw RLBench observation to
   a `SceneObservation` via `env.process_observation`, attaching the inferred
   action (`env._get_action(obs, next_obs)`) and a positive feedback flag, and
   pushing each observation into the replay buffer.
4. Saves the full trajectory and proceeds to the next episode.

## Why a separate script?
- RLBench's scripted demos drive the simulator deterministically and need no
  TAPAS policy. The generic `collect_data.py` always queries a policy and
  expects a step-by-step `env.step(action)` loop, so this variant exists to
  bypass that.
- Useful for generating large pretraining datasets with high-quality scripted
  demos.

## Key components
- `Config`: thinner than `collect_data.Config` (no policy fields).
- `main`: orchestrates demo generation and conversion.
- `complete_config`: same naming bookkeeping as in `collect_data.py`.
