# env/rlbench.py - RLBenchEnvironment

## Purpose
TAPAS adapter for RLBench / CoppeliaSim tasks. Wraps the standard
`RLBenchInternalEnvironment` and exposes `BaseEnvironment`'s API. Handles
RLBench's "real last" quaternion convention (TAPAS uses "real first"),
multiple per-task action modes, and the per-task camera configuration.

## Imports & registry
- Pulls a curated set of RLBench tasks
  (`CloseMicrowave`, `PickAndLift`, `OpenDrawer`, `PutItemInDrawer`, ...)
  into a module-level `task_switch` dict that maps task names to RLBench
  task classes.
- Sets `QT_QPA_PLATFORM_PLUGIN_PATH` from `COPPELIASIM_ROOT` so headless
  CoppeliaSim can find Qt plugins (matches `rlbench_mode.sh`).

## Key components

### `RLBenchEnvironmentConfig(BaseEnvironmentConfig)`
- Chooses the arm action mode (`EndEffectorPoseViaIK`,
  `EndEffectorPoseViaPlanning`, ...) and the gripper action mode
  (`Discrete`).
- Camera config: image size, render modes, point-cloud capture toggle.
- A `static` flag that disables physics (used for the dataset-collection
  variant in `collect_data_rlbench.py`).

### `RLBenchEnvironment(BaseEnvironment)`
- Builds an `RLBenchInternalEnvironment` with the configured cameras,
  action mode, and demo/headless flags; then resolves
  `task_switch[config.task]` to launch the right `TaskEnvironment`.
- `reset` returns a `SceneObservation` built from
  `process_observation`.
- `_step(action, ...)`:
  - Converts TAPAS' "real first" quaternion convention back to RLBench's
    "real last" via `quat_real_first_to_real_last`.
  - Calls `task_env.step(action)`. Catches `ConfigurationPathError` and
    `IKError`/`InvalidActionError`, signalling them through the returned
    `info` dict instead of crashing.
  - Builds the next `SceneObservation` from the raw result.
- `process_observation(raw_obs)`: maps the RLBench observation
  (`overhead_rgb`, `wrist_rgb`, depth, masks, joint pos/vel, gripper open,
  object poses, etc.) into the TAPAS tensorclass, normalising image axes,
  converting quaternions, and assembling per-camera intrinsics/extrinsics.
- `_get_action(raw_obs, next_raw_obs)`: inverse kinematics-free derivation
  of the relative pose between two RLBench observations - used by
  `collect_data_rlbench.py` when scripted demos are being recorded.
- `set_world_action_mode`, `_get_action_mode`, `_set_action_mode`: support
  the demo-collection workflow's `RestoreActionMode` context manager.
- `close`: shuts down the simulator.

## Notes
- Pulls heavily on `pyrep` (the Python bridge to CoppeliaSim). Errors
  thrown by pyrep during action execution are filtered out so the
  evaluation loop can continue.
- The list of supported tasks here matches the paper's RLBench evaluation
  set. Adding a task means importing it from `rlbench.tasks` and adding
  it to `task_switch`.
