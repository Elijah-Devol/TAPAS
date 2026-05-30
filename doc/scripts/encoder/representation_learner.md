# encoder/representation_learner.py - RepresentationLearner base class

## Purpose
Abstract base class shared by every TAPAS encoder. Defines the common API:
`encode`, `update_params`, `evaluate`, `reconstruct`, `to_disk`/`from_disk`,
plus utility hooks for image normalisation and dataset-driven parameter
initialisation.

## Key class: `RepresentationLearner(nn.Module)`
- Base config: `RepresentationLearnerConfig(end_to_end=False,
  disk_read_embedding=False)`.
- Class attributes:
  - `sample_type`: must be set by subclasses to one of `SampleTypes` so the
    dataset knows what shape of batches to draw.
  - `embedding_name = "descriptor"`: the on-disk attribute name where
    `embed_trajectories.py` stores embeddings for this class.

### Abstract methods (subclasses must implement)
- `update_params(batch, **kwargs)` - one training step (handles its own
  optimiser).
- `evaluate(batch, **kwargs) -> dict` - eval-mode metrics.
- `encode_single_camera(batch: SingleCamObservation) -> (embedding, info)` -
  the single-camera embedding routine. The base class' `encode` loops over
  cameras and concatenates the results.
- `reconstruct(batch)` - reconstruct an input (used by viz code).
- `get_latent_dim(config, n_cams=1, image_dim=None)` (classmethod) - returns
  the per-observation latent dim for sizing downstream policies.

### Concrete helpers
- `encode(batch: SceneObservation) -> (concatenated_embedding, info_per_cam)`:
  iterates over `batch.camera_names`/`batch.camera_obs`, calls
  `encode_single_camera` (or reads cached embeddings from `obs` if
  `disk_read_embedding=True`), and concatenates along the last dimension.
- `reset_episode()`: hook for stateful encoders (filters); default is a no-op.
- `initialize_image_normalization(replay_memory, camera_name)`: optional hook
  for setting per-channel mean/std from the dataset.
- `initialize_parameters_via_dataset(replay_memory, cam, **kwargs)`: hook for
  dataset-driven init (e.g. keypoint reference selection).
- `add_gaussian_noise(coordinates, noise_scale, skip_z=False)`: static helper
  for adding noise to a stacked `(..., 3*n)` keypoint tensor (used by
  `KeypointsPredictor` during keypoint augmentation).
- `from_disk(path)` / `to_disk(path)`: state-dict (de)serialisation.

The class doc-comments note a planned refactor to split this into a
`VisualEncoder` (inference only) and a `RepresentationLearner` (training).
