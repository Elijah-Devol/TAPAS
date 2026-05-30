# pretrain.py

## Purpose
Entry point `tapas-pretrain`. Trains a representation learner (BVAE / MONet /
transporter / dense-correspondence keypoint encoder / ViT extractor) on
unlabelled or weakly-labelled image pairs sampled from a `SceneDataset`.

## What it does
1. Loads the `SceneDataset` and updates the camera crop from the observation
   config.
2. Instantiates the encoder via `encoder_switch[config.encoder_name](config)`
   and pushes it to GPU.
3. Wraps the dataset in a `DenseCorrespondenceDataset`, asking for the encoder's
   declared `sample_type` (e.g. correspondence pairs, single images,
   transporter triplets, ...). This dataset is the bridge between TAPAS' scene
   storage and the encoder's expected input format.
4. Initialises the encoder's image normalisation statistics from the dataset.
5. Calls `run_training`, which loops for `training.steps` steps:
   - Builds infinite train/val iterators with the encoder's chosen
     `collate_func`.
   - Each step: `encoder.update_params(batch, ...)` returns training metrics.
     If no valid correspondences were sampled, the step is retried.
   - Periodically runs `run_eval_step` and (if `auto_early_stopping`) halts on
     loss regression.
   - Periodically writes intermediate checkpoints.
6. Saves the final checkpoint and the config along with it.

## Key components
- `TrainingConfig`, `Config`: dataclass configs.
- `run_training`, `run_training_step`, `run_eval_step`: the training/eval
  steps. The retry loop in the latter two handles empty-correspondence batches
  for dense-correspondence training.
- `entry_point`: argparse + W&B + seeding + `main`.

## Notable details
- Asserts `encoder_config.end_to_end == False`: end-to-end encoders are trained
  inside `behavior_cloning.py`, not here.
- `config.encoder_naming = config.data_naming` means the pretrained encoder
  takes its name from the pretraining dataset.
