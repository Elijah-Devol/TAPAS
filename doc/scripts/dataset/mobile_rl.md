# dataset/mobile_rl.py

## Purpose
Loader for the mobile-manipulation imitation-learning dataset format
(EE positions, velocities, grasp signal) used in an external mobile-robot
project. Converts that format into TAPAS' `SceneObservation` tensorclass and
finally into a `Demos` object that the TP-GMM pipeline can fit.

## Key components
- `Imitation_Learning_Dataset(Dataset)`: loads per-trajectory numpy files
  (`robot_dof_pos.npy`, `eetip_pose_global.npy`,
  `ee_velocities_reference_frame_pos.npy`, etc.), enforces quaternion
  continuity / positive real part, and exposes joint states, EE poses, EE
  delta, and a grasp signal. `to_tensor_dict()` packs these into a
  `SceneObservation` (note: cameras are commented out, only proprioceptive
  data is used here).
- `make_demos(path)`: walks the directory, builds one
  `Imitation_Learning_Dataset` per sub-trajectory, filters trajectories by an
  initial-pose constraint (`ee_pose[0, 4] > 0`), plots the EE pose
  trajectories for sanity, converts to `SceneObservation`, and wraps the whole
  set into a `Demos` instance with `add_init_ee_pose_as_frame=True` and
  `add_world_frame=False`.

## When is this used?
This module is for a specific external dataset format and is not part of the
standard TAPAS workflow. It exists so the same TP-GMM fitting code can be
re-used on that data.
