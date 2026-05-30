# visualize_dense_correspondence.py

## Purpose
Launches a live interactive heatmap visualiser for one or more pretrained
encoders. Lets you hover a query pixel in one camera/frame and see the
corresponding similarity heatmap (and best-match pixel) in another frame, for
each loaded encoder.

## What it does
1. Loads the scene dataset and a `DenseCorrespondenceDataset` with the
   first encoder's `sample_type`.
2. Instantiates each encoder in `config.encoder` (either a `KeypointsPredictor`
   trained by TAPAS or an off-the-shelf `VitFeatureEncoder`) and loads its
   weights.
3. Instantiates `viz.live_heatmap_visualization.HeatmapVisualization` over all
   encoders and runs the interactive loop.

## Key components
- `EncoderConfig`: a single encoder's name, config, suffix, naming, and either
  a custom path or the default `pretrain_checkpoint_name`.
- `Config`: a tuple of `EncoderConfig`s plus the visualisation config and
  shared data/observation config.
- `main`: builds encoders and the heatmap viz; the `HACK` block injects the
  per-encoder fields into a deep-copied `config` since the encoder constructors
  expect them at the top level.

Useful to qualitatively assess pretrained representations before committing to
a TAPAS-GMM run with them.
