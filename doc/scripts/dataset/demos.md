# dataset/demos.py - Demos class & TP-GMM data preparation

## Purpose
Converts a list of `SceneObservation` trajectories (each one a recorded
demonstration) into the *task-parameterised* representation that the TP-GMM and
GMR algorithms in `policy/models/tpgmm.py` and the riepybdlib-based GMM fitter
consume. This is the bridge between TAPAS' trajectory storage and the
riemannian-manifold mixture model.

## High-level idea
For each trajectory:
- Reads object frames (either from ground-truth `object_poses` or from learnt
  keypoints via `tp_from_keypoints`).
- Optionally prepends the initial end-effector pose as an extra "frame" and a
  fixed world frame.
- Homogenises frame orientation (`enforce_z_up` / `enforce_z_down`,
  `rotate_frames_180degrees`, `modulo_object_z_rotation`) so symmetrical
  objects don't induce arbitrary axis flips.
- Expresses each end-effector pose in *every* frame, producing the input to
  the TP-GMM (one trajectory in N frames).
- Maintains continuity of quaternions across time when requested.

## Key class: `Demos`
Constructor parameters select which transforms to apply
(`add_init_ee_pose_as_frame`, `add_world_frame`, `enforce_z_down`,
`enforce_z_up`, `modulo_object_z_rotation`, `frames_from_keypoints`,
`kp_indeces`, `make_quats_continuous`).

The class exposes:
- The per-frame trajectory tensors used by the TP-GMM fitter.
- Convenience accessors over the homogenous frame transforms used for GMR
  inference.
- A meta-data dictionary used as a cache-key for memoised computations of e.g.
  reference frames, action sequences, etc.

## Module-level utilities
- `rotate_frames_180degrees(quats, rot_idx, skip_idx, axis="y")`: flip a
  subset of frames.
- `configurable_rotate_frames(...)`: select rotation by `enforce_z_up/down` and
  current frame layout.
- `get_frames_from_obs(obs, frames_from_keypoints, ...)`: same logic but for
  online inference - takes one `SceneObservation` and returns the homogenous
  transforms and quaternions for all configured frames. Used by the GMM
  policy during evaluation.

The module has very few external-facing functions beyond `Demos` and
`get_frames_from_obs` - most of the file is internal helper logic for
frame-based representations on SE(3).
