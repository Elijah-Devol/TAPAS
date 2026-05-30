# policy/gmm.py - GMMPolicy

## Purpose
The headline TAPAS policy: a task-parameterised Gaussian Mixture Model
(TP-GMM) trained on the task-frame trajectory data produced by `Demos`,
combined with TOPP (Time-Optimal Path Parameterisation) for time scaling
and the option to predict either a full robot trajectory or a per-step
action.

## Configs
- `GMMPolicyConfig(PolicyConfig)`:
  - `model`: nested `AutoTPGMMConfig | TPGMMConfig`. Picks `TPGMM`
    (manually-specified component count, frames, etc.) or `AutoTPGMM`
    (auto-discovers components / per-frame importance via Riemannian
    manifold clustering).
  - `time_based`: when the model includes a time dimension, the policy
    can interpolate along time rather than recursively predicting the
    next state.
  - `predict_dx_in_xdx_models`: for models trained on `[x, dx]`, predict
    `dx` (velocity) at every step.
  - `batch_predict_in_t_models`, `batch_t_max`,
    `topp_in_t_models`, `return_full_batch`,
    `topp_supersampling`, `time_scale`: control whether the policy
    pre-samples a full trajectory in time, runs TOPP to time-scale it,
    and returns the full trajectory in one `predict` call.
  - `pos_lag_thresh`, `quat_lag_thresh`, `pos_change_thresh`,
    `quat_change_thresh`: hysteresis thresholds for the recursive
    predictor; prevent jitter when the policy is stuck near a fixed
    point.
  - `binary_gripper_action` + threshold.
  - `obs_encoder`: optional `ObservationEncoderConfig`; used so the
    policy can extract object frames from keypoint encodings online.
  - `encoder` / `encoder_path`: an optional keypoint encoder to run
    online for frame extraction.
  - `postprocess_prediction`, `invert_prediction_batch`,
    `force_overwrite_checkpoint_config`, `dbg_prediction`: misc.

## `GMMPolicy(Policy)`
- Builds the underlying model (`AutoTPGMM` or `TPGMM`).
- `initialize_parameters_via_dataset`: fits the TP-GMM to the trajectory
  data extracted by `Demos.from_replay_memory(...)`. This is what the
  `notebooks/gm/.../*.ipynb` notebooks normally invoke.
- `predict(obs)`:
  - Builds the current set of object frames via
    `get_frames_from_obs(obs, frames_from_keypoints, ...)`.
  - Runs Gaussian Mixture Regression (GMR) conditioned on the current
    state to obtain a target state distribution.
  - Either returns the single mean target action, or generates a full
    `RobotTrajectory` and (when `topp_in_t_models=True`) time-scales it
    with `tapas_gmm.utils.topp.TOPP`.
  - Applies the configured hysteresis thresholds, handles the binary
    gripper conversion, and returns the action and an `info` dict.
- `reset_episode`: clears any cached predictions.
- `update_params` is *not* used during training - GMM fitting happens in
  the notebook, not the BC loop.

## Helper constants
- `zero_pos`, `zero_quat`, `close_gripper`, `open_gripper`: pose / gripper
  defaults used when assembling prediction outputs.

## See also
- `policy/models/tpgmm.py`: the actual TP-GMM model (~160KB; the
  computational core of the paper).
- `utils/topp.py`: TOPP wrapper.
- `viz/gmm.py`: per-component / per-frame plotting helpers used by the
  notebooks.
