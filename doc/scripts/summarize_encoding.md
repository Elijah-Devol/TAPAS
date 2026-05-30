# summarize_encoding.py

## Purpose
Visualises and overlays the keypoint encodings produced by `kp_encode_trajectories.py`,
optionally on top of recorded scene images and depth maps, in 2D and 3D, with
optional ground-truth comparison.

## What it does
1. Loads the scene dataset and reads, for each `encoding_path =
   "<encoder>/<kp_selection_name>"`, the cached encoding tensors (`kp`,
   per-camera `2d_locations`/`post`/`heatmaps`, world coordinates, particle
   variances, ...).
2. Optionally subsamples the trajectories (`min_index`, `max_index`,
   `index_step`).
3. Computes per-camera 2D keypoint positions by unflattening the stored `kp`
   tensor and (optionally) loads a ground-truth keypoint trajectory for
   comparison.
4. Plots:
   - 2D per-camera image grids with keypoints (and optionally heatmaps)
     overlaid (`plot_2d`).
   - 3D scatter/line plots of each keypoint's trajectory, with the GT
     overlaid (`plot_3d`).
5. Supports custom markers/colors per encoder and per keypoint, legend
   rendering, annotation, and saving to PDF/PNG.

## Key components
- `Outside` enum: how to handle keypoints that project outside the image
  (`SKIP`/`PLOT`/`CLAMP`).
- `unflatten_keypoints`: stacked `xy(z)` → `(n_kp, d_kp)`.
- `plot_2d`, `plot_3d`: the two main plotting routines.
- `get_kp`, `get_gt_kp`, `get_trajs`, `subsample_trajs`: data prep helpers.

Used by paper figures and for debugging the keypoint encoder over a full
trajectory.
