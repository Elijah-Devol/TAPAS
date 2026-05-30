# dataset/dc.py - DenseCorrespondenceDataset

## Purpose
The dataset wrapper used for *training* pretrained encoders. It samples
single-image, image-pair, or transporter-triplet batches from a `SceneDataset`,
together with the pixel correspondences and non-correspondence sets needed by
dense-correspondence training (e.g. `KeypointsPredictor`, BVAE, MONet,
transporter).

## Key class: `DenseCorrespondenceDataset`
Constructed from `(scene_dataset, DCDataConfig, sample_type, cameras)`.

- `sample_type` (from `utils.observation.SampleTypes`) decides what each
  `__getitem__` returns:
  - `CAM_SINGLE`: a single image (for autoencoder-style encoders).
  - `DC`: a pair of images plus matched/non-matched pixel sets between them.
  - `CAM_PAIR`, transporter, etc.
- Per-batch behaviour for `SampleTypes.DC`:
  - Picks one of `DcDatasetDataType` (single-object within scene, same object
    across scene, different object, multi-object, synthetic multi-object) by
    the `data_type_probabilities` mixture in the config.
  - Samples a matched pixel set using `correspondence_finder` (back-projects
    one image's pixels into 3D, projects into the other) and a non-matched set
    (either uniformly on the off-mask region or split between masked and
    background by `fraction_masked_non_matches`).
  - Applies augmentations (`correspondence_augmentation`): domain
    randomisation, random flip, random crop, both spatial transforms applied
    consistently to the correspondences.
  - Returns either a `(image_a, image_b, matches_a, matches_b,
    non_matches_a, non_matches_b, ...)` tuple or a dataclass.
- Caches image statistics with `lru_cache`.
- Built-in debug plotting hooks (`debug_plots`, `cross_debug_plot`) when
  `dc_config.debug=True`.

## Config: `DCDataConfig`
- Mixture probabilities, augmentation toggles, crop/flip settings,
  matched/non-matched counts, mask filtering, contrast set (currently
  disabled), object-pose filtering.

## Other helpers
- `get_collate_func`: returns a sample-type-specific collator. The dense
  correspondence sample shape varies per batch, so a custom collator is needed.
- `variable_sample`: legacy hand-rolled batched sampler used by
  `visualize_embedding.py`.
- `sample_images_across_trajectories`: returns N (image, mask) pairs sampled
  uniformly across trajectories for the t-SNE visualisation.

## Where it's used
- `pretrain.py` (representation learner training).
- `visualize_dense_correspondence.py` (live heatmap viz).
- `visualize_embedding.py` (t-SNE / reconstruction viz).
