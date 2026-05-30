# kp_encode_trajectories.py

## Purpose
Entry point `tapas-kp-encode`. Runs a pretrained keypoint encoder (with one of
three priors: none, discrete filter, particle filter) over every frame in every
trajectory and stores the resulting 3D keypoint encoding (plus per-camera
debug artefacts) back into the dataset under a named "kp selection".

## What it does
1. Loads the `SceneDataset` and wraps it in a `BCDataset` configured for raw
   image/depth/extr loading.
2. Builds an `EncoderPseudoPolicy` (a thin wrapper that owns the trained
   keypoint encoder and exposes the `encode` API). If `--copy_selection_from`
   is given, the reference keypoint positions are copied from another encoder
   checkpoint.
3. `policy.initialize_parameters_via_dataset(...)` selects reference keypoints
   from the dataset (this is the step that "picks the keypoints" from descriptor
   space).
4. Registers the encoding config with the dataset, then loops:
   - For each trajectory, resets the policy episode (so the filter prior
     restarts at the first observation).
   - For each frame, runs `policy.encoder.encode(obs)` and stores the 3D
     keypoint encoding into the dataset under
     `<encoder>/<kp_selection_name>/kp`.
   - If filter debugging is enabled, writes additional diagnostics (particles,
     prior/posterior heatmaps, likelihoods, etc.) per camera.
5. Saves the encoder back to disk with the kp-selection-name appended to its
   suffix so it can be re-used as-is in downstream training/eval.

## Key components
- `Config`: policy + naming + observation + data-loader + BC-data settings;
  notably `kp_selection_name` (required) and `constant_encode_with_first_obs`
  (encode only the first frame and replicate, for ablations).
- `encode_trajectories`: the core loop.
- `save_kp_debug`, `save_particle_debug`, `save_discrete_filter_debug`: writers
  for the three prior variants' debug artefacts.

## Notable details
- This step is mandatory in the TAPAS pipeline before the GMM is fit: the GMM
  notebooks consume the `kp` tensors that this script writes.
- Logs likelihood breakdowns (descriptor, depth, occlusion) for each camera
  when the particle filter prior is debugged.
