# dataset/scene.py - SceneDataset

## Purpose
The on-disk format for TAPAS demonstrations. A `SceneDataset` is rooted at one
filesystem directory and contains many trajectories, each one a directory of
per-camera tensors (`rgb/<step>.png`, `depth/<step>.dat`, `extr/...`,
`intr/...`, masks, etc.) plus generic per-step tensors (`ee_pose`, `action`,
`joint_pos`, `joint_vel`, `gripper_state`, `feedback`, `object_poses`, ...).

It is the single source of truth for everything downstream:
- The encoder training scripts (`pretrain.py`) sample image pairs from it.
- `embed_trajectories.py` and `kp_encode_trajectories.py` add per-frame
  embeddings/encodings to it.
- `tsdf_fusion.py` computes per-frame object masks and writes them back.
- `behavior_cloning.py` and the GMM notebook read from it.

## Key class: `SceneDataset(Dataset)`
Constructor takes `data_root`, `allow_creation`, and a `SceneDatasetConfig`
(only used the first time the dataset is created - afterwards the config is
persisted in `metadata.json` and reloaded).

### Persistent state
Attributes that don't start with `_` are auto-persisted into the metadata
file: `camera_names`, `image_height`, `image_width`, `object_labels`,
`smo_data` (sub-datasets), `image_crop`, encoder/encoding configs, mask labels,
etc.

### Directory layout (per `FileNameConfig`)
- `trajectories/<traj_id>/<attr_dir>/...`: one directory per trajectory, with
  one sub-directory per (camera_name, attribute) and one per generic
  attribute; image attributes are stored as `.png`, all others as `.dat`.
- `embeddings/<encoder_name>/<traj_id>/...`: per-encoder per-trajectory
  embedding tensors (one file per step per camera per attribute).
- `encodings/<encoder_name>/<selection_name>/<traj_id>/...`: per-keypoint-
  selection cached encodings.
- `fusion/`: optional TSDF mesh/pointcloud `.ply` artefacts.

### Configurable subsampling
`SubSampleTypes` (POSE, CONTENT, LENGTH, NONE) controls how scene views are
subsampled when assembling a fusion or BC trajectory. The wrist-cam is usually
subsampled by pose threshold (to keep the fused TSDF tractable), overhead
cameras by content / target length.

### Core API
- `add_observation(obs)`: append a `SceneObservation` to the in-progress
  trajectory.
- `save_current_traj()` / `reset_current_traj()`: commit or drop the
  in-progress trajectory.
- `_get_bc_traj(traj_idx, fragment_idx, fragment_length, cams, mask_type,
  encoder_name, encoding_name, encoding_attr, embedding_attr, skip_rgb,
  ...)`: the workhorse loader used by `BCDataset.__getitem__`.
  Returns a tensorclass slice of the requested length, with the requested
  cameras, optionally including pre-computed embeddings/encodings/masks.
- `get_scene(traj_idx, cams, subsample_types)`: returns the entire
  trajectory's view for TSDF fusion or visualisation.
- `sample_observation(...)`: single-observation sampler used by
  `DenseCorrespondenceDataset` and encoder initialisation.
- `add_tsdf_masks(traj_idx, cam, mask, labels)`: writes per-pixel masks
  produced by `tsdf_fusion.py`.
- `add_embedding`, `add_encoding`, `add_encoding_fig`,
  `add_embedding_config`, `add_encoding_config`: writes encoder outputs.
- `update_traj_attr(traj_idx, step, attr, value)`: in-place edit of any
  generic attribute, used by `reconstruct_actions.py`.
- `update_camera_crop(image_crop)`: re-crops every camera in the dataset and
  all sub-datasets.
- `initialize_scene_reconstruction()`: prepares the `fusion/` directory used
  by `tsdf_fusion.py`; returns `None` if it already exists (signalling a
  no-op).
- SMO datasets ("Sub-Mixture-of-Other"): the dataset can hold multiple named
  sub-datasets (used to keep multiple variants of one task, e.g. with and
  without distractors).

## Config: `SceneDatasetConfig`
- `data_root`, `camera_names`, `image_size`, `image_crop`.
- `subsample_by_difference` / `subsample_to_length`: trajectory-time
  subsampling at write time.
- `object_labels`, `ground_truth_object_pose`, `ignore_gt_labels`.
- `shorten_cam_names`: shorten camera prefixes (e.g. `wrist` -> `w`) for
  shorter on-disk attribute names.
- Nested file/traj/SMO sub-configs.

## Notable details
- Camera shortening happens at the attribute-name level
  (`_get_cam_attr_name(cam, attr)`); attribute names persist into the saved
  files so renaming cameras in-place is not supported.
- `MetadataFileNotExistsError` / `CannotCreateDatasetError` /
  `DirectoryNonEmptyError` are raised on inconsistent on-disk state; new
  datasets are only created when `allow_creation=True`.
- The encoder caches (`embeddings/`, `encodings/`) are append-only - existing
  files are not overwritten; reruns require deleting them first.
