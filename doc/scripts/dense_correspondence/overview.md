# dense_correspondence/ - Pixel correspondence training

Vendored & adapted from
[pytorch-dense-correspondence](https://github.com/RobotLocomotion/pytorch-dense-correspondence).
This package supports training the dense-descriptor backbone used by
`KeypointsPredictor`.

## `correspondence_finder.py` (22KB)
The "geometry layer" for matched and non-matched pixel pairs:
- `random_sample_from_masked_image_torch(mask, num_samples)`: uniformly
  sample foreground pixels.
- `pinhole_projection_image_to_world_coordinates(...)` / inverse: back- /
  re-projection using camera intrinsics & extrinsics. There are also
  batched variants in `utils.geometry_torch`.
- `find_pixel_correspondences(...)`: given image-A pixel UVs, image-A and
  image-B depth maps, intrinsics, and extrinsics, project A's pixels into
  3D and then into B; return the UV in B and a valid-mask.
- `get_mask_center(mask)`, `get_masked_avg_descriptor(...)`: utilities
  used by the encoder's reference-selection workflow.
- Helpers for "across-scene" correspondences (single-object across
  trajectories, multi-object, synthetic multi-object) used by
  `DenseCorrespondenceDataset` to mix in cross-scene pairs.

## `correspondence_augmentation.py` (16KB)
Image and pixel-coordinate augmentations that keep the two consistent:
- `domain_randomize_background`: paint a random colour/texture over the
  off-mask region of an image.
- Random flip, random crop, paired colour jitter - each with a sibling
  function that transforms a list of UV indices the same way.
- Used by `DenseCorrespondenceDataset` per-batch when
  `domain_randomize`/`random_crop`/`random_flip` are enabled.

## `loss/`
- `pixelwise_contrastive_loss.py`: the heart of dense-correspondence
  training. Computes a margin-based contrastive loss between matched
  pairs (small distance) and non-matched pairs (large distance), with
  separate margins for background vs masked non-matches, and an L2 pixel
  loss option for masked non-matches. Scales by hard-negative mining
  weights when `scale_by_hard_negatives` is enabled. Config:
  `LossFunctionConfig` from `tapas_gmm/encoder/keypoints.py`.
- `loss_composer.py`: glues `PixelwiseContrastiveLoss` to the per-batch
  `DcDatasetDataType` (single-object within scene / cross-scene /
  different-object / multi-object / synthetic-multi-object). Picks the
  right pixel sets and combines the loss terms into a single scalar.

## `__init__.py`
Empty package markers in this package and in `loss/`.
