# encoder/models/ - Backing models

Each encoder in `tapas_gmm/encoder/` is a thin training-loop wrapper around a
"model" in `tapas_gmm/encoder/models/`. The models hold the raw network
architectures and inference operations; the encoder classes provide the
`RepresentationLearner` API expected by the rest of TAPAS.

## Sub-modules

### `bvae/`
- `base.py` - the abstract `BaseVAE` interface (encode, decode, sample,
  reconstruct, forward, loss-stats slots).
- `beta_vae.py` - concrete beta-VAE encoder/decoder networks plus the
  `Config` dataclass (`in_channels`, `latent_dim`, hidden dims).
- `types_.py`, `utils.py` - small helpers.

### `keypoints/`
- `keypoints.py` - the `KeypointsModel`: a stack of `vision_net` blocks
  (ResNet / vision transformer / ...) that produces a per-pixel descriptor
  map. Selects the architecture via `KeypointsTypes`.
- `model_based_vision.py` - geometric helpers: pixel-to-3D, 3D-to-pixel
  reprojection, intrinsics inversion, sub-pixel depth sampling. Used by
  both the learnt and GT keypoint predictors.
- `networks.py` - smaller building blocks (custom conv stacks, sub-modules
  used by the descriptor backbone).
- `resnet.py` - ResNet variants adapted for descriptor output.
- `vision_transformer.py` - ViT backbone variant used by
  `KeypointsPredictor` when `vision_net` is set to a transformer name.

### `monet/`
- `monet.py` - `MONetModel` + `MONetModelConfig` (slot count, slot dims,
  encoder/decoder configs).
- `networks.py` - U-Net-style components used inside the MONet model
  (attention network, component VAE).

### `transporter/`
- `transporter.py` - `Transporter`, `FeatureEncoder`, `PoseRegressor`,
  `RefineNet`, plus `TransporterModelConfig` (architecture map, channel
  count, n_keypoints, keypoint Gaussian std).
- `utils.py` - heatmap utilities (spatial softmax, Gaussian map
  generation).

### `vit_extractor/`
- `extractor.py` - `VitEncoderModel` + `VitFeatureModelConfig` (DINO /
  DINOv2 variant, stride, load size, layers, facets, padding, center
  crop, saliency threshold). Implements `preprocess`,
  `extract_descriptors`, `extract_saliency_maps` on top of the pretrained
  ViT.

## Why the split?
The encoder classes are PyTorch `nn.Module`s that participate in TAPAS'
training loop (own optimisers, EMAs, checkpoints, etc.). The models contain
*only* the network arithmetic, so they could in principle be re-used outside
of TAPAS. This is a layering convention inherited from the upstream
projects (BASK, pytorch-dense-correspondence, MONet, Transporter, etc.).
