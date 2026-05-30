# policy/encoder.py - EncoderPolicy & friends

## Purpose
Three policy variants that share an `Encoder` (any `RepresentationLearner`
subclass) and an LSTM head. They differ in *how* the encoder is exercised:

| Class | Encoder runs online? | Encoder weights tunable? | Used by |
| --- | --- | --- | --- |
| `EncoderPolicy` | yes | optional (`end_to_end` flag) | `evaluate.py`, end-to-end BC |
| `DiskReadEncoderPolicy` | no - reads cached embeddings/encodings | no | BC with pre-cached embeddings |
| `EncoderPseudoPolicy` | yes - inference only | no | `kp_encode_trajectories.py` |

## Configs
- `EncoderPolicyConfig(LSTMPolicyConfig)`: adds `encoder_name`,
  `encoder_config`, `encoder_suffix`, `encoder_naming`, `observation`,
  `kp_pre_encoded` flag.
- `PseudoEncoderPolicyConfig`: lighter weight - just the fields needed to
  construct the encoder; LSTM/training fields are placeholders.

## `EncoderPolicy(LSTMPolicy)`
- Resolves the encoder class via `encoder_switch[encoder_name]`, asks it
  for its latent dim (`get_latent_dim`), and feeds that into the LSTM as
  `visual_embedding_dim`.
- Loads the encoder checkpoint if given. If `end_to_end=True`, the
  encoder's parameters join the policy's optimiser; otherwise the encoder
  is frozen and switched to eval.
- `_get_visual_input(obs)`: runs the encoder on the observation, flattens
  the embedding from `[B, T, D, ...]` to `[B, T, D]`.
- `initialize_parameters_via_dataset(replay_memory, cameras)`: delegates
  to the encoder's own init (e.g. keypoint reference selection). For the
  `ODS` keypoint variant, also adds the reference descriptor to the
  optimiser so it can be fine-tuned.
- `from_disk(...)`: loads with `strict=False` so missing/unexpected keys
  produce warnings rather than errors (handy when changing encoder/policy
  configs between checkpoints).
- `reset_episode`: clears LSTM state *and* encoder state (filter prior,
  etc.).

## `EncoderPseudoPolicy`
Not a `Policy` subclass - it doesn't implement `predict`/`update_params`.
Used by `kp_encode_trajectories.py` solely as a manager for the encoder
during keypoint pre-encoding. Responsibilities:
- Build the encoder.
- Run `initialize_parameters_via_dataset` (the costly KP-selection step).
  Skipped when `kp_pre_encoded=True`.
- Optionally copy the reference selection from another checkpoint
  (`copy_reference_from_disk`). Handles both encoder-only and policy
  checkpoints, and reconstructs the reference descriptor when copying a
  GT-keypoint selection into a learned encoder.
- Save the encoder back to disk afterwards via `encoder_to_disk`.

## `DiskReadEncoderPolicy(EncoderPseudoPolicy, EncoderPolicy)`
Multiple-inherits to combine the two: full policy functionality but the
encoder is loaded from a pre-computed checkpoint and never queried in the
training loop (the observations carry pre-computed embeddings or
keypoints). On `to_disk`, the encoder is re-loaded from its source
checkpoint (avoiding having two copies in the policy snapshot) before
saving.
