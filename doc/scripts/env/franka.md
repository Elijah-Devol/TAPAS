# env/franka.py - FrankaEnv

## Purpose
Real-robot environment for a Franka Emika Panda with two cameras (overhead
+ wrist RealSense). Wraps the lab's ROS stack:
- `rl_franka.Panda`, `RLControllerManager`, `PandaControllerManager` for
  joint/cartesian control.
- `robot_io.cams.threaded_camera.ThreadedCamera` for non-blocking RGB-D
  acquisition.
- `franka_gripper.GraspAction` for the parallel gripper.
- ROS topic publishers for joint commands, error-recovery, and rviz
  visualisation.
- `roboticstoolbox.Panda` for inverse kinematics.

## Key components

### Configs
- `IKConfig`: solver (LM/GN/NR), iteration limits, tolerance.
- `RealSenseConfig`: lab-specific camera parameters (resolution, image type,
  brightness/contrast/etc.).
- `FrankaEnvironmentConfig(BaseEnvironmentConfig)`: adds teleop and eval
  flags, optional teleop-grasp toggle, gripper open/close speeds, the IK
  solver config, and any per-task overrides.

### `FrankaEnv(BaseEnvironment)`
- Subscribes to joint-state and camera-pose topics; starts threaded
  cameras.
- Implements `_step(action, ...)` to:
  - Post-process the action through `BaseEnvironment.postprocess_action`.
  - Clamp the resulting EE translation to a per-task safety box via
    `utils.franka.clamp_translation`.
  - Solve IK (`get_inverse_kinematics`) for the target pose.
  - Send the resulting joint trajectory through
    `ThreadedJointTrajectoryFollower` so the loop doesn't block on robot
    motion.
  - Re-read the latest scene observation.
- Implements `_get_action` for converting between consecutive
  `RLBenchObservation`-style raw observations (used by
  `collect_data_rlbench.py`).
- Implements `process_observation` (raw -> `SceneObservation` tensorclass).
- Implements `recover_from_errors` (sends `ErrorRecoveryActionGoal`).
- Implements `publish_frames` and `publish_path` so keypoint frames and
  predicted trajectories show up in rviz.
- Implements `get_inverse_kinematics` using `roboticstoolbox`, with the
  solver selected from `IKConfig`.
- Holds a `RobotTrajectory` queue so the env can step through a multi-step
  trajectory predicted by the diffusion / GMM policy.

## Notes
- This file is specific to the Freiburg lab's robot setup; users with a
  different stack should fork it. The structure is portable, though.
- Helper imports near the top (`CameraOrder`, `dict_to_tensordict`, etc.)
  are used to build the per-step `SceneObservation` from the ROS messages.
