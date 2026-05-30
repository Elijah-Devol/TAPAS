# compare_models.py

## Purpose
Small CLI utility that loads two PyTorch checkpoints and reports differences
between their state dictionaries.

## What it does
- Reads paths `-a` and `-b` from argparse.
- `torch.load`s each checkpoint onto CPU.
- Delegates to `tapas_gmm.utils.torch.compare_state_dicts`, which surfaces
  missing keys, shape mismatches, and value differences.

Useful when debugging EMA vs. main weights, fine-tuning runs, or two checkpoints
that should be identical.
