# encoder/cnn.py - CNN / CNNDepth encoders

## Purpose
A pair of small convolutional encoders, intended as a lightweight baseline
that is trained *end-to-end* alongside the policy (no pretraining stage).
- `CNN` takes RGB input.
- `CNNDepth` takes RGBD input (4 channels).

## Architecture
Three `Conv2d(stride=2) + ELU` blocks; channels `3 -> 2 -> 1 -> 1` for `CNN`
and `4 -> 2 -> 1 -> 1` for `CNNDepth`. Inputs are first downsampled to half
their original resolution.

## Notable details
- `from_disk` is a no-op: there is no pretraining checkpoint to load - this
  encoder is always trained inside the policy.
- `update_params` raises `NotImplementedError` for the same reason.
- `get_latent_dim` uses a hard-coded map from `image_dim` to flat latent size:
  `(256, 256) -> 256`, `(360, 480) -> 690`.
- `CNNDepth.forward` concatenates the depth tensor onto `batch.cam_rgb` (and
  any second camera) before delegating to the inherited `encode`.

This encoder is mainly used for end-to-end ablations; it is not a focus of the
TAPAS paper.
