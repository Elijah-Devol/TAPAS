# policy/sphere.py - SpherePolicy

## Purpose
A scripted policy used for *pretraining* data collection: drives the
wrist camera through a fixed set of "scan poses" arranged on a hemisphere
around the workspace, so that the pretraining dataset covers a wide range
of viewpoints. The dense-correspondence pretraining of `KeypointsPredictor`
benefits a lot from this kind of multi-view coverage.

## How it works
- `scan_poses` is a hard-coded array of `[x, y, z, qx, qy, qz, qw,
  gripper]` poses (5 of them in the current code).
- On each `predict(obs)`, the policy:
  - Returns `None` to signal "done" once all scan poses have been
    visited.
  - Otherwise, if no current path exists, calls
    `path_planner_func(next_pos[:3], quaternion=next_pos[3:7])` (passed in
    via kwargs from `collect_data.py`) to plan to the next pose, then
    pops joint configurations off the resulting path one chunk at a time.
- Returns `(pos, {})` where `pos` is the next joint position vector
  (with a `1.0` gripper appended).

## Notes
- This policy is specific to the RLBench setup: the path-planner produces
  arm joint configurations, and the env consumes them in joint-position
  control mode. The number of joints is derived from
  `current_path._arm.joints`.
- No config dataclass - it just expects a `path_planner_func` kwarg.
