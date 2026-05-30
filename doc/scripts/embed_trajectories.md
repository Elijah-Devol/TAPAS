# embed_trajectories.py

## Purpose
Entry point `tapas-embed`. Encodes every frame in every trajectory with a
pretrained representation learner (BVAE / MONet / transporter / keypoints / GT
keypoints / ViT features), and writes the per-frame encodings back into the
`BCDataset` so downstream training/eval can read pre-computed features rather
than running the encoder online.

## What it does
1. Loads the `SceneDataset` and wraps it in a `BCDataset` configured for raw
   loading (the encoder needs RGB + depth + intrinsics/extrinsics).
2. Instantiates the encoder via `encoder_switch[config.encoder_name]`, loads
   its checkpoint, and moves it to GPU.
3. For keypoint encoders, zeros out the reference descriptor (so embedding
   doesn't trigger keypoint selection); for keypoints-GT, either copies
   reference positions from another encoder or initialises them from the
   dataset.
4. Registers the encoder configuration with the dataset, then iterates over
   every trajectory and every frame:
   - Calls `encoder.encode(obs)` to obtain the embedding (and ancillary info).
   - Calls `save_descriptor` (for keypoint encoders) or `save_encoding` (for the
     others) to persist the result per-camera under
     `<traj>/<step>/<cam>/descriptor` (and `heatmap` for transporter).
5. For `keypoints_gt`, saves the encoder to disk afterwards because the
   reference selection is part of its state.

## Key components
- `Config`: encoder + naming + observation + data-loader + BC-data settings.
- `embed_trajectories`: main encoding loop.
- `save_encoding` / `save_descriptor`: per-frame writers for non-KP and KP
  encoders respectively.
- `complete_config`: aligns encoder naming with data naming.

## Notable details
- Runs inside `@torch.no_grad()` to avoid a known memory leak.
- The `--copy_selection_from` flag is for comparing different projections of the
  same GT keypoints.
