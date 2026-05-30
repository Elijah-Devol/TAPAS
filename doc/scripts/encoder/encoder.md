# encoder/encoder.py - ObservationEncoder

## Purpose
The bridge between a `SceneObservation` (containing camera images,
proprioception, EE poses, object poses) and the flat vector that downstream
policies consume. Combines a configurable selection of low-dim inputs with an
optional image encoder.

## Key class: `ObservationEncoder(nn.Module)`
- Built from `ObservationEncoderConfig` plus an optional
  `image_encoder_checkpoint`.
- Holds an `image_encoder` chosen via `get_image_encoder_class` and loaded from
  disk on construction.

### Config: `ObservationEncoderConfig`
- `ee_pose`, `proprio_obs`, `object_poses`: which low-dim fields to include.
- `image_encoder`: nested config of the image encoder to use (`None` for
  proprio-only policies).
- `pre_encoding`: name of a precomputed encoding attribute on the observation
  (e.g. `"kp"`) - used when training/eval on cached encodings instead of running
  the encoder online.
- `online_encoding`: a flag toggling on-the-fly encoder evaluation.
- `constant_image_encoding_per_episode`: HACK flag used during real-robot
  rollouts; computes the image encoding once at the start of the episode and
  reuses it for every step.

### Key methods
- `encode(obs) -> (low_dim, image_enc_info)`:
  - Selects EE pose / proprio / object pose tensors per the config.
  - Computes/loads the image encoding by one of four paths (constant cached,
    online, none, or pre-computed from disk).
  - Concatenates everything into a single vector along the last dim.
- `get_image_encoding(obs)`: returns just the image encoding plus its info
  dict (or `(None, {})` if no image encoder).
- `get_obs_distribution(replay_memory)`: gathers the joint distribution of all
  selected low-dim and pre-encoding fields across the dataset; used to fit
  policy normalisation statistics.
- `reset_episode()`: clears the cached "constant" image encoding.
- `get_viz_encoder_callback()`: returns `image_encoder.encode` so the
  environment can run live visualisation of the encoder output before an
  episode starts.

## Notes
- Asserts that `constant_image_encoding_per_episode` and `pre_encoding` are
  mutually exclusive.
- The image encoder is fully optional - policies can use only proprio/object
  inputs by leaving `image_encoder=None`.
