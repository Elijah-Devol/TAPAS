# encoder/transporter.py - Transporter encoder

## Purpose
Wrapper around the Transporter model from
`encoder.models.transporter.transporter`. Trained as a "transport"
autoencoder: given a source image `xs` and target image `xt`, predict `xt`
from the features of `xs` modulated by the keypoint heatmaps of `xt`. The
resulting keypoints are used as the embedding (`n_keypoints * 2 * n_cams`).

## Key class: `Transporter(RepresentationLearner)`
- `sample_type = SampleTypes.CAM_PAIR` - the dataset emits paired (source,
  target) images from the same trajectory.
- Loss: MSE between reconstructed and target images.
- Optimiser: Adam.

### Architecture
Built from three sub-modules from the underlying model:
- `FeatureEncoder` (image -> feature map).
- `PoseRegressor` (image -> n_keypoints heatmaps).
- `RefineNet` (features + heatmaps -> reconstructed image).
The triple is assembled into a `transporter.Transporter` with the configured
keypoint Gaussian std.

### Methods
- `process_batch`: downsample both images to 128x128, forward, compute MSE.
- `encode_single_camera`: downsample, run `model.encode`, return per-camera
  embedding plus the heatmaps (in `info`).
- `reconstruct`: feature-encode the input and decode through the refine net
  (no transport - used by `visualize_embedding.py`).
- `get_latent_dim(config, n_cams=1, image_dim=None)`:
  `config.encoder.n_keypoints * 2 * n_cams`.
