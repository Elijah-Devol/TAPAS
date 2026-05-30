# env/mani_skill.py - ManiSkillEnv

## Purpose
TAPAS adapter for ManiSkill2 / Sapien tasks. Builds a `gym.make`
environment with a chosen controller, exposes the standard `BaseEnvironment`
API (`reset`/`step`/`render`/`close`), and converts the raw ManiSkill
observation/action spaces to TAPAS' `SceneObservation` tensorclass and 7-DOF
action vector.

## Constants
- `ACTION_MODE = "pd_ee_delta_pose"` - default ManiSkill controller.
- `OBS_MODE = "state_dict+image"` - default observation mode.
- `default_cameras = ("hand_camera", "base_camera")`.
- `cam_name_tranlation`: TAPAS camera names <-> ManiSkill camera names
  (`hand_camera` <-> `wrist`, `base_camera` <-> `base`, etc.).

## Configs
- `ManiSkillEnvironmentConfig(BaseEnvironmentConfig)` adds:
  - `render_sapien` toggle.
  - `background` (texture choice).
  - `model_ids`, `fixed_target_link_idx` (per-task overrides).
  - `real_depth` (use the noisy depth model).
  - `seed`, `invert_xy`, `obs_mode`, `action_mode`.

## `ManiSkillEnv(BaseEnvironment)`
- Initialises with ManiSkill-tuned action scales
  (`_delta_pos_scale = 0.25`, `_delta_angle_scale = 0.5`).
- Builds the `gym` environment lazily, attaching the configured cameras.
- `reset(seed=None, options=None)`: forwards to the underlying gym env;
  returns a `SceneObservation`.
- `_step(action, ...)`: optionally negates X/Y (for the policy convention),
  post-processes, calls `gym.step`, and packs the result.
- `_get_action(raw_obs, next_raw_obs)`: utility to derive the action that
  drives the transition between two consecutive ManiSkill observations
  (used by `extract_demos.py`).
- `process_observation(raw_obs)`: converts the ManiSkill obs dict into a
  `SceneObservation`, mapping camera names through
  `inv_cam_name_tranlation`, copying intrinsics/extrinsics, joint state,
  EE pose, gripper position, object poses, and any GT masks.
- `_get_state` / `_set_state`: snapshot/restore the Sapien scene state for
  `RestoreEnvState`.
- `_get_action_mode` / `_set_action_mode`: snapshot/restore the controller
  type (used by `extract_demos.py` to switch into a different control mode
  per episode).
- `close()`: closes the gym env.

## Notes
- Imports `mani_skill2.envs` for side effects (registers the envs with gym).
- Inputs are kept in ManiSkill's native quaternion convention; the
  base-class action post-processing handles the quaternion convention used
  by TAPAS policies.
