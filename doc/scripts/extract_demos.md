# extract_demos.py

## Purpose
Converts ManiSkill2's native HDF5 demonstration files (`trajectory.h5` +
`trajectory.json`) into TAPAS' own `SceneDataset` format, optionally translating
between control modes (e.g. `pd_joint_pos` → `pd_ee_delta_pose`).

## What it does
1. Reads the HDF5 trajectory + the matching JSON metadata; pulls out the
   `env_id`, `env_kwargs`, and per-episode reset info.
2. Builds two ManiSkill envs:
   - `ori_env`: a Gym env with the original kwargs/control mode (used as a
     reference for the action-mode conversion).
   - `env`: the TAPAS-wrapped `ManiSkillEnv` configured with the target action
     and observation modes from `config`.
3. Creates a `SceneDataset` writer.
4. For each episode:
   - Resets both envs deterministically with `episode_seed`.
   - Replays the original actions either directly (no conversion) or through
     `from_pd_joint_pos` / `from_pd_joint_delta_pos` which translate the
     trajectory into the target control mode while recording the resulting
     `SceneObservation`s.
   - On success, calls `replay_memory.save_current_traj()`; otherwise retries
     up to `max_retry` times or drops the episode.
5. Closes both envs.

## Key components
- `Config`: paths, num_demos, retry settings, target control/obs modes.
- `_main`: the conversion loop.
- `complete_config`: parses the trajectory path to derive a task + variant
  name (e.g. `PickCube-v0-easy`) and updates the config.
- `main`: entry-point wrapper; uses `mp.set_start_method("spawn")` so that
  CUDA-allocating subprocesses behave.
