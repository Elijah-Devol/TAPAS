# collect_data.py

## Purpose
Entry point `tapas-collect`. Drives an environment (RLBench / ManiSkill2 /
Franka) with a chosen policy to record demonstration trajectories into a
`SceneDataset` on disk.

## What it does
1. Builds the environment (`import_env(config.env_config)`), the policy
   (`import_policy(config.policy.value)`), and a `KeyboardObserver` for human
   feedback (success/reset).
2. Creates a `SceneDataset` (the replay buffer) under
   `data_root/<task>/<feedback_type>/`.
3. Resets the env, sleeps briefly, then repeatedly:
   - Asks the policy for an action (`policy.predict(obs)`).
   - Steps the env, attaches the chosen action and a positive feedback flag to
     the previous observation, and appends it to the replay buffer.
   - On `done` (or user-signalled success) saves the trajectory and resets;
     on user reset or horizon timeout, resets without saving.
4. Stops after `n_episodes` trajectories or on `Ctrl-C`, closing the env
   gracefully.

## Key components
- `Config`: dataclass aggregating dataset, env, policy and naming configs.
- `main`: the main collection loop.
- `complete_config`: synchronises `env_config.task`, dataset root and dataset
  name based on the `--pretraining` CLI flag.

## Notable details
- The policy is given the `KeyboardObserver` so that a manual / teleoperation
  policy can read user inputs.
- Uses `loop_sleep` to keep the loop near a fixed control frequency.
