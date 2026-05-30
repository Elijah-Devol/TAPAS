# visualize_demo.py

## Purpose
Diagnostic script for inspecting a single recorded demonstration alongside the
keypoint encodings/likelihoods produced for it.

## What it does
1. Loads the scene dataset and extracts one trajectory (`config.traj_idx`) for a
   single camera, requesting both raw RGB/depth and the stored encoding
   attributes (`2d_locations`, `post`, and the 3D `kp`).
2. Computes a 2D-pixel keypoint location from the stored normalised
   `2d_locations` and the image resolution.
3. Upsamples the per-channel posterior `lik` to image resolution and sums
   across channels.
4. Builds a 5-panel figure:
   - `(1)` RGB with keypoints
   - `(2)` RGB with keypoints + likelihood overlay
   - `(3)` Depth with keypoints
   - `(4)` Depth with keypoints + likelihood overlay
   - `(5)` 3D view of EE pose, camera frame, keypoints, and EE trajectory
5. Renders coordinate frames using `viz.threed.plot_coordindate_frame` and
   `quaternion_to_matrix` from `utils.geometry_torch`.
