# encoder/keypoints_gt.py - GTKeypointsPredictor

## Purpose
A drop-in keypoint encoder that uses *ground-truth* object poses (from the
simulator) instead of a learnt dense-descriptor backbone. Lets you compare
TAPAS-GMM against an oracle that has perfect per-frame keypoint locations -
useful for ablating away the encoder's contribution to overall task success.

## Key class: `GTKeypointsPredictor(KeypointsPredictor)`
- `sample_type = SampleTypes.GT` - the dataset must contain ground-truth
  object poses.
- Re-uses much of the `KeypointsPredictor` machinery (the same projection
  modes, output shape, viz hooks), but replaces the descriptor-matching step
  with `model_based_vision`-based reprojection of the reference pixel into
  the current frame using the chosen reference object pose.

## Reference selection (PreTrainingConfig)
Same `ReferenceSelectionTypes` as the learned KP predictor: `MANUAL` /
`RANDOM` / `MASK_AVG` / `MASK_CENTER`. Once a reference pixel is selected,
the encoder stores:
- `ref_pixels_uv`: 2D pixel location in the reference frame.
- `ref_pixel_world`: 3D world coordinate of that pixel.
- `ref_depth`, `ref_int`, `ref_ext`: the reference frame's depth, intrinsics
  and extrinsics.
- `ref_object_poses` (in `_extra_state`): the ground-truth object pose at the
  reference frame.

At inference time, for each new frame: take the current object poses, compute
the rigid transform from the reference frame to the current one, transform
`ref_pixel_world` accordingly, project into the current camera, sample depth,
and emit the resulting 3D keypoint.

## Notable details
- `prior_type` defaults to `NONE` but the field is kept for compatibility
  with the learned predictor's downstream code paths.
- `only_use_first_emb = True`: with 3D/ego projections, one camera is enough.
- `disk_read_embedding = False`: ground-truth lookups are cheap, so caching
  isn't useful.
- Uses `set_extra_state`/`get_extra_state` to round-trip the reference object
  pose (TensorClass buffer functionality doesn't yet support this).
