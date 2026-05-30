# visualize_embedding.py

## Purpose
Standalone visualisation utility (predates the OmegaConf-based pipeline) for
inspecting what a pretrained encoder has learnt. Supports three modes:
- `reconstruction`: image → encoder → decoded image, shown alongside the input.
- `slots`: for MONet, decode every slot.
- `tsne`: project per-pixel descriptors of many images into 2D via t-SNE,
  coloured by GT mask label.

## What it does
1. Argparse-driven CLI; constructs a config dict (no OmegaConf here) that
   mirrors the legacy "central encoder_configs" layout.
2. Loads a replay memory (by `--path` or by `feedback_type`/`task`/`mask`),
   wraps it as `DenseCorrespondenceDataset`.
3. Instantiates the encoder via `encoder_switch` and loads its checkpoint.
4. Dispatches to one of three visualisation functions:
   - `vis_reconstruction`: grid of (reconstruction, input) pairs.
   - `vis_slots`: per-slot reconstructions for MONet.
   - `vis_tsne`: random-pixel descriptors → t-SNE → scatter coloured by GT
     mask label, with a per-class legend.

This script is older than the rest and is not exposed as an entry point in
`setup.py`. It still works for the encoders that predate the keypoint pipeline.
