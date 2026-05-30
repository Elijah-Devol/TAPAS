# evaluate.py

## Purpose
Entry point `tapas-eval`. Rolls out a trained policy in an environment for a
fixed number of episodes and reports the success rate.

## What it does
1. Builds the environment (real Panda / RLBench / ManiSkill), creates a
   `KeyboardObserver` if running on Panda, and loads the policy class via
   `import_policy(config.policy_type)`.
2. Loads the trained policy checkpoint and switches it to eval mode.
3. Sets up an optional `LiveKeypoints` overlay for keypoint policies.
4. Computes how often to repeat actions (when the policy was trained on
   subsampled data) and calls `run_simulation`.

`run_simulation` runs `n_episodes` episodes, logging each episode's reward and
length to W&B. `run_episode` repeatedly calls `process_step`, which:
- Applies optional observation dropout and the `disturbe_at_step` perturbation.
- Sets `obs.gripper_state` from the previous action.
- Calls `policy.predict(obs)` to get either a single action or a
  `RobotTrajectory`.
- Steps the environment `repeat_action` times (or steps through the trajectory
  if the policy returned a `RobotTrajectory`).
- Handles `done` conditions, including the keyboard observer's success/reset
  buttons on the real robot.
- Forwards policy `info` to `keypoint_viz` and to the env for visualisation.

## Key components
- `EvalConfig`, `Config`: dataclass configs (includes horizon, disturbance,
  obs-dropout, hold-until-step, etc.).
- `calculate_repeat_action`: derives the action-repeat factor from the policy's
  training sample frequency.
- `run_simulation` / `run_episode` / `process_step`: the three-level rollout
  loop.

## Notable details
- Logs the final success rate to `wandb.run.summary["success_rate"]`.
- On `KeyboardInterrupt`, still reports the success rate computed so far.
- Handles `RobotTrajectory` actions specially because the env steps through
  them in a background thread on the real robot.
