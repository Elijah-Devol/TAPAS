# env/environment.py - BaseEnvironment

## Purpose
Abstract base class for all TAPAS environments. Defines the contract every
concrete env (`FrankaEnv`, `ManiSkillEnv`, `RLBenchEnvironment`) must
fulfil: `reset`, `step`, `close`, `render`, `recover_from_errors`, etc.
Also provides cross-cutting helpers for action post-processing (scaling,
gripper delay, quaternion conventions) so the same policy can drive any
back-end.

## Key components

### `BaseEnvironmentConfig`
Common fields for every environment:
- `task`, `cameras`, `camera_pose`, `image_size`.
- `static` (no robot motion?), `headless`, `scale_action`, `delay_gripper`,
  `gripper_plot`, `postprocess_actions`.
- `env_type` (set by subclass).

### `BaseEnvironment(ABC)`
- Stores config flags; sets up the `GripperPlot` (an optional matplotlib
  widget showing the gripper open/close state for live debugging) and a
  deque of recent gripper actions.
- `step(action)`: thin wrapper around `_step` that wires the config flags.
- `_step(action, postprocess, delay_gripper, scale_action)`: abstract; the
  per-env implementation does the real work.
- `postprocess_action(action, prediction_is_quat, prediction_is_euler,
  scale_action, delay_gripper, trans_scale, rot_scale)`: the workhorse that
  splits a 7-vector action into `(delta_position, delta_rotation, gripper)`,
  converts the rotation representation, scales the deltas if requested,
  delegates the env-specific quaternion convention to
  `postprocess_quat_action`, and optionally smooths the gripper command via
  `delay_gripper`.
- `delay_gripper(action)`: only commits an open/close once the action has
  been consistent over `queue_length=4` consecutive steps. Reduces gripper
  chattering induced by noisy policy outputs.
- `update_visualization`, `propose_update_visualization`, `publish_frames`,
  `publish_path`: hooks for the Franka rviz integration.
- `_get_state`/`_set_state`, `_get_action_mode`/`_set_action_mode`: hooks
  for RLBench-style stateful demos. The companion context managers below
  use these.

### `GripperPlot`
Tiny matplotlib widget that renders a stylised gripper opening/closing. Used
during data collection and evaluation so the operator can see the
commanded gripper state at a glance.

### `RestoreEnvState`, `RestoreActionMode`
Context managers that snapshot `env._get_state`/`env._get_action_mode` on
enter and restore them on exit. Used in scripted demo generation to flip
into a different action mode temporarily without leaking it.

## Notes
- The base class defines reasonable defaults for `_delta_pos_scale` (0.01)
  and `_delta_angle_scale` (0.04); subclasses override these when running on
  controllers with different action-space normalisations.
- `squash(array, order=20)`: helper exposed at module scope; maps values to
  `[-1, 1]` with a sharper saturation than `tanh`. Used by the Franka env
  for action limiting.
