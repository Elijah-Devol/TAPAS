# inspect_dataset.py

## Purpose
Developer utility that prints a summary of a loaded `SceneDataset`/replay
memory and plots the action distribution.

## What it does
- Argparse: choose a dataset by `-f`/`-t`/`-m` (feedback type, task, mask flag)
  or by absolute `-p` path.
- `load_replay_memory(config)` (or `torch.load(path)`) yields the buffer.
- For every attribute on the buffer, classifies it as `PRINTABLE`,
  `SUMMARIZABLE`, or `OTHER` (depending on whether it fits in 25 elements) and
  prints accordingly.
- Concatenates `replay_memory.action` across trajectories and calls
  `viz.action_distribution.make_all` to plot per-dimension distributions.

Useful for sanity-checking what tensors a dataset contains and seeing if the
recorded actions look reasonable.
