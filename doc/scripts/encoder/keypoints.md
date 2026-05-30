# encoder/keypoints.py - KeypointsPredictor (the TAPAS encoder)

## Purpose
The core keypoint encoder used throughout the TAPAS paper. It runs a
dense-descriptor backbone over the image, computes a soft argmax against a set
of reference descriptors to obtain 2D keypoint locations, optionally projects
those into 3D using depth + camera intrinsics + extrinsics, and optionally
fuses the per-frame predictions with a discrete or particle filter to obtain
temporally-smooth 3D keypoints.

## Encoder pipeline
1. **Dense descriptors** - `KeypointsModel` produces a per-pixel descriptor map
   at the backbone's stride (e.g. 32x32 for 128x128 inputs).
2. **Reference comparison** - cosine or L2 distance between the descriptor map
   and a learned set of reference descriptors (`_reference_descriptor_vec`)
   yields one similarity map per keypoint per camera.
3. **Soft argmax** - the similarity map is converted to a posterior over
   pixels with a learnable `taper`, then summed against pre-built `pos_x`,
   `pos_y` meshgrids to get a sub-pixel 2D keypoint location. If
   `use_spatial_expectation=False`, the hard argmax is used instead.
4. **Projection** to 3D, controlled by `ProjectionTypes`:
   - `NONE`: keep 2D keypoints (`(u, v)` normalised to `[-1, 1]`).
   - `UVD`: append the sampled depth as a third coordinate.
   - `LOCAL_SOFT`/`GLOBAL_SOFT`: soft 3D back-projection from the depth map.
   - `LOCAL_HARD`/`GLOBAL_HARD`: hard pixel lookups + intrinsics inversion.
   - `EGO`/`EGO_STEREO`: ego-centric coordinates (used with the particle
     filter).
5. **Temporal prior** (`PriorTypes`): `NONE`, `DISCRETE_FILTER` (via
   `tapas_gmm.filter.discrete_filter.DiscreteFilter`), or `PARTICLE_FILTER`
   (via `tapas_gmm.filter.particle_filter.ParticleFilter`). When enabled,
   the filter is reset at the start of each trajectory and consumes the raw
   per-frame likelihoods.
6. **Output** - a flat per-camera keypoint vector (`2*n_kp` or `3*n_kp`).

## Pretraining
When given a `PreTrainingConfig`:
- Loss: `PixelwiseContrastiveLoss` from
  `tapas_gmm.dense_correspondence.loss.pixelwise_contrastive_loss`.
- Optimiser: Adam with optional StepLR / CosineAnnealing / CosineAnnealingWR
  schedule.
- Reference selection (`ReferenceSelectionTypes`) at the end of pretraining:
  - `MANUAL`: open a `KeypointSelector` GUI (commented out for headless use).
  - `RANDOM`: pick random pixels.
  - `MASK_AVG`: average descriptors over the GT object mask.
  - `MASK_CENTER`: pick pixels at the centre of each object mask.

## Key state (registered buffers)
- `ref_pixels_uv`, `_reference_descriptor_vec`: the chosen reference
  positions and their descriptors.
- `norm_mean`, `norm_std`: image normalisation statistics, populated by
  `initialize_image_normalization`.

## Configs
- `KeypointsConfig`: `type` (SD/SDS/WSDS/...), `n_sample`, `n_keep`.
- `LossFunctionConfig`: descriptor and pixel margins, weighting between
  matched / masked / background non-matched losses.
- `PreTrainingConfig`: optimisation, lr schedule, reference selection.
- `EncoderConfig`: descriptor dim, prior type, projection, taper, cosine vs
  L2, soft vs hard, image dim, noise scale, threshold/overshadow keypoint
  distance.
- `KeypointsPredictorConfig`: assembles all of the above plus an optional
  `filter` config and `debug_*` toggles.

## Enums (re-exported)
- `KeypointsTypes` - re-exported from the underlying model.
- `PriorTypes`, `ProjectionTypes`, `ReferenceSelectionTypes`,
  `LRScheduleTypes` - defined here.
