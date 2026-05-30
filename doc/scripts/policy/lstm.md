# policy/lstm.py - LSTMPolicy

## Purpose
The neural-network policy back-end used as a baseline (and as the head
inside `EncoderPolicy`). It consumes a concatenation of
`(visual_embedding, proprio_obs)` and produces a `tanh`-bounded action
through an `nn.LSTM` plus linear head.

## Configs
- `LSTMPolicyTrainingConfig`: `learning_rate`, `weight_decay`.
- `LSTMPolicyConfig(PolicyConfig)`:
  - `visual_embedding_dim`, `proprio_dim`, `action_dim`, `lstm_layers`.
  - `use_ee_pose`, `add_gripper_state`, `add_object_poses`,
    `poses_in_ee_frame`: what to include in the proprio vector.
  - `training` (nullable - `None` skips optimiser init for eval-only use).
  - `action_scaling`: whether to normalise inputs through a
    `LinearNormalizer`.

## `LSTMPolicy(Policy)`
- Builds an `nn.LSTM` with input and hidden dim
  `visual_embedding_dim + proprio_dim`, followed by `nn.Linear(D,
  action_dim)`.
- Owns a `LinearNormalizer` (from `policy/models/diffusion/normalizer.py`)
  initialised from the dataset at construction time and used to normalise
  the input vector when `action_scaling=True`.
- Per-action distribution: `Normal(mean=tanh(linear_out), std=0.1)`
  (`self.std = 0.1 * ones(action_dim)`). Predictions can be drawn or used
  deterministically.
- `forward_step(obs)`: assemble low-dim input via `_get_policy_input`,
  optionally normalise, step the LSTM, run the linear head + tanh.
- `_prep_pose_dict(object_poses, ee_poses)`: optionally rotates each
  object pose into the EE frame before concatenation.
- `reset_episode`: clears `_lstm_state`.
- `predict(obs)`: single-step inference; returns numpy action.
- `update_params(batch)`: standard supervised step with MSE / Gaussian
  log-likelihood (see method body).
- `evaluate(batch)`: validation pass with the same loss.

## Notes
- `add_object_poses=True` requires the user to set
  `visual_embedding_dim` to include the object pose dims (no autodetect).
- The LSTM is *unidirectional*; the policy state persists across steps in
  an episode and is cleared on `reset_episode`.
