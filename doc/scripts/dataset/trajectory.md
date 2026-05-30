# dataset/trajectory.py - Trajectory class

## Purpose
A small in-memory buffer for the trajectory currently being recorded by
`collect_data.py` (or `collect_data_rlbench.py` / `extract_demos.py`). When
the trajectory ends, it serialises itself into the `SceneDataset` directory
layout.

## Key class: `Trajectory`
- Constructed with the active camera names, the subsampling mode, and the
  filesystem/trajectory config blocks.
- `add(obs)`: append a `SceneObservation` to the in-memory list.
- `save(directory)`: write the trajectory to disk:
  - Optionally subsamples by wrist-pose-difference threshold or target length
    via `get_idx_by_pose_difference_threshold` /  `get_idx_by_target_len`
    (`downsample_traj_by_idx`).
  - For each camera, iterates over the camera-attribute map
    (`get_cam_attributes(cam, MaskTypes.GT)`) and writes per-step
    tensors/images. Image attributes (`rgb`) use `save_image` and `.png`;
    everything else uses `save_tensor` and `.dat`. Flat attributes (e.g.
    intrinsics, which don't vary per step) are stored as a single file
    `<attr_dir>.dat`.
  - Writes generic per-step attributes (`GENERIC_ATTRIBUTES`).
  - Aggregates per-camera GT object labels into a single sorted list and
    writes a per-trajectory `metadata.json` containing trajectory length and
    the label set.
- `reset()`: clear the buffer (called automatically after `save`).

## Notes
- This class is the only place that decides the on-disk *file format* of an
  individual trajectory. The directory *layout* is owned by `SceneDataset`.
- The `is_flat_attribute(attr)` predicate decides whether an attribute is
  serialised once or per-step.
