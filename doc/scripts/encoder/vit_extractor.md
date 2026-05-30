# encoder/vit_extractor.py - VitFeatureEncoder / VitKeypointsPredictor

## Purpose
Off-the-shelf Vision-Transformer feature extractor (DINO / DINOv2) used as a
dense descriptor backbone, plus a TAPAS-style keypoint predictor built on
top.

## Classes

### `VitFeatureEncoder(RepresentationLearner)`
- Wraps a `VitEncoderModel` from `encoder.models.vit_extractor.extractor`.
- `sample_type = SampleTypes.DC` (so it can be reused inside the
  correspondence visualiser).
- `frozen=True` runs the ViT under `torch.inference_mode()`; `False` runs
  under `nullcontext()` so gradients flow.
- `_get_descriptor_resolution` derives the descriptor map size for both
  DINO and DINOv2 backbones, accounting for `stride`, `center_crop`, and
  `pad`.
- `compute_descriptor` / `compute_descriptor_batch`: preprocess the input,
  extract descriptors via the ViT, reshape into `(B, C, H_descr, W_descr)`,
  optionally bilinearly upsample to the original image size.
- `encode`: per-camera dense descriptors, concatenated along the last
  dimension.
- `generate_mask`: threshold the ViT saliency map to a foreground mask.
- `from_disk` is a no-op (the ViT is loaded from its pretrained weights via
  `VitEncoderModel`).

### `VitKeypointsPredictor`
- Has the same projection/prior/keypoint machinery as
  `KeypointsPredictor`, but uses a `VitFeatureEncoder` as its descriptor
  backbone instead of the custom `KeypointsModel`. Defined further down in
  the same module (see the encoder file for the full implementation).
- Config: `VitKeypointsEncoderConfig` + `PreTrainingConfig`,
  `VitKeypointsPredictorConfig`.

## Where it fits
This encoder is the "vit" variant used in the paper's ablations and is the
encoder type used in `conf/kp_encode_trajectories/vit/...`.
