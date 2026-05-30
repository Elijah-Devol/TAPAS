# behavior_cloning.py

## Purpose
Entry point `tapas-bc`. Trains a learning-based policy (e.g. `LSTMPolicy` or
`EncoderPolicy`) via behavior cloning on a set of recorded demonstrations.

## What it does
1. Loads the recorded `SceneDataset` for the chosen task (via `data_naming`).
2. Wraps it in a `BCDataset` that performs (optionally pre-embedded /
   pre-encoded) lookups suited for behavior-cloning training.
3. Selects a policy class through `import_policy(config.policy_type)`, loads any
   pretrained encoder checkpoint, and moves the policy to GPU.
4. Initialises policy parameters from the dataset (e.g. action statistics).
5. Runs `run_training`, an iterative train/eval loop that:
   - Samples batches via `InfiniteDataIterator`s built from `BCDataset`.
   - Updates the policy parameters and (optionally) maintains an EMA copy
     (`EMAModel`).
   - Periodically evaluates on the held-out split and logs to Weights & Biases.
   - Saves intermediate checkpoints and stops early on overfit if configured.
6. Persists the trained policy and the (now resolved) config next to the
   checkpoint.

## Key components
- `EMAConfig`, `TrainingConfig`, `Config`: dataclass configs.
- `run_training`: the core training loop (supports epoch- or step-based modes
  and EMA).
- `run_eval_epoch`: runs an evaluation pass on the validation iterator.
- `complete_config`: post-processes the OmegaConf config (e.g. fills encoder
  naming fields, marks pre-embeddings, records the git commit hash).
- `entry_point`: argparse + OmegaConf instantiation + `wandb.init` + `main`.

## Notable details
- Supports learning either an end-to-end encoder-policy (image observations
  flow through the encoder during training) or a policy that consumes
  pre-computed embeddings/keypoint encodings from disk.
- When the configured encoder uses the particle filter prior and debug mode is
  enabled, attaches a `ParticleFilterViz` visualisation.
- `complete_config` enforces that exactly one of `training.steps` and
  `training.epochs` is specified.
