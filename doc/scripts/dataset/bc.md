# dataset/bc.py - BCDataset

## Purpose
A PyTorch `Dataset` adapter sitting on top of a `SceneDataset` that yields
behaviour-cloning training samples (full trajectories or fixed-length
fragments).

## Key class: `BCDataset`
- Created from a `SceneDataset` plus a `BCDataConfig`.
- Supports three load modes:
  - Raw RGB/depth/etc. - used when the encoder runs end-to-end inside the
    policy.
  - Pre-computed dense embeddings (`pre_embedding=True`) - loads stored
    `descriptor` tensors from `<traj>/<step>/<cam>/descriptor` instead of RGB.
  - Pre-computed keypoint encodings (`kp_pre_encoding="<name>"`) - loads the
    3D `kp` tensor written by `kp_encode_trajectories.py`.
- Supports fixed-length fragments with optional pre- and post-padding so that
  every sample has the same length (useful for LSTM/diffusion training), or
  full-trajectory mode (`fragment_length == -1`).
- `__getitem__` translates a global integer index into `(traj_idx, obs_idx)`,
  loads the requested slice through `SceneDataset._get_bc_traj`, pads at both
  ends with zeros (with `feedback` propagated from the endpoints), and returns
  the resulting tensorclass.

## Config: `BCDataConfig`
- `fragment_length`, `pre_padding`, `post_padding`: sampling geometry.
- `cameras`, `mask_type`, `only_use_labels`: what to load per observation.
- `pre_embedding` / `kp_pre_encoding` / `encoder_name`: which cached encodings
  to read.
- `force_load_raw`, `force_skip_rgb`, `debug_encoding`: overrides for special
  cases (e.g. particle filter needs raw inputs even when embeddings exist).
- `sample_freq`, `subsample_to_common_length`: temporal subsampling controls
  consumed by the underlying `SceneDataset`.

## Other helpers
- `sample_bc` - legacy helper used by encoder init and visualisations to draw
  full-trajectory batches.
- `sample_data_point_with_object_labels` /
  `sample_data_point_with_ground_truth` - single-observation samplers used by
  pretrain.py and encoder init.
- `add_embedding`, `add_encoding`, `add_encoding_config`,
  `add_embedding_config`, `update_traj_attr` - thin wrappers over the
  underlying scene dataset that route writes to the correct on-disk location.
