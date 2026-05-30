# filter/ - Temporal priors for keypoints

When the keypoint encoder is configured with a temporal prior
(`PriorTypes.DISCRETE_FILTER` or `PriorTypes.PARTICLE_FILTER`), the
filter modules here fuse the per-frame descriptor likelihood with the
predicted prior to obtain a temporally-smooth 3D keypoint posterior.

## `discrete_filter.py`
`DiscreteFilter`: a per-keypoint Bayesian filter on a dense pixel grid.
- Maintains a posterior `post[H, W]` per keypoint per camera.
- Each step:
  - Predicted prior: rigid-translate the previous posterior by the EE
    motion (when keypoints are tied to the gripper) or by a learnt
    motion model.
  - Likelihood: the per-pixel similarity (softmax over descriptors) for
    the current frame.
  - Posterior: prior * likelihood, normalised.
- The current keypoint is the argmax (or soft expectation) of the
  posterior. Optionally back-projected to 3D via depth and intrinsics.
- Config: `DiscreteFilterConfig` carrying grid size, motion variance,
  noise, descriptor-temperature, debug flag.

## `particle_filter.py` (43KB)
`ParticleFilter`: a Sequential Monte Carlo filter on 3D world
coordinates.
- `N` particles per keypoint, each `[x, y, z]`. Optionally also `vx, vy,
  vz` if motion model is enabled.
- Each step:
  - Predict: propagate particles with Gaussian noise (`process_noise`)
    and optionally the previous-step velocity.
  - Update: for each particle, project into every camera and read the
    descriptor similarity at the projected pixel. Combine likelihoods
    across cameras and (optionally) include depth and occlusion
    likelihoods (`noisy_pixel_coordinates_to_world`,
    `batchwise_project_onto_cam`).
  - Weight: importance weights = product of per-camera likelihoods.
  - Resample: systematic resampling when the effective sample size
    drops below a threshold.
- Returns: the weighted mean as the keypoint estimate plus an `info`
  dict (`particles_2d`, `keypoints_2d`, `descr_likelihood`,
  `depth_likelihood`, `occlusion_likelihood`, `particle_var`, ...)
  used by `kp_encode_trajectories.py`'s debug plots and the live
  visualiser.
- Config: `ParticleFilterConfig` - particle count, process noise per
  axis, descriptor / depth / occlusion likelihood weights, ESS resample
  threshold, debug flag, motion model toggles.

## `__init__.py`
Empty package marker.

## Where they're used
The keypoint encoder (`encoder/keypoints.py`) decides between
`NONE`/`DISCRETE_FILTER`/`PARTICLE_FILTER` based on its
`prior_type`. The filter state is reset at the start of every trajectory
via `KeypointsPredictor.reset_episode()`, which delegates to
`filter.reset()`.
