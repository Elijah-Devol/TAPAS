# policy/diffusion.py - DiffusionPolicy

## Purpose
A diffusion-based policy adapted from
[Diffusion Policy (Chi et al., 2023)](https://github.com/real-stanford/diffusion_policy)
and integrated into TAPAS' interfaces. Trains a 1D conditional U-Net to
denoise an action sequence conditioned on the recent observation history,
and sampled at inference via DDPM.

## Configs
- `DiffusionPolicyTrainingConfig`: AdamW hyperparameters (`lr`, `betas`,
  `eps`, `weight_decay`), `lr_scheduler` (`cosine` etc.) with warmup,
  `gradient_accumulate_every`, `use_ema`.
- `DDPMSchedulerConfig`: `num_train_timesteps`, `beta_start`, `beta_end`,
  `beta_schedule`, `variance_type`, `clip_sample`, `prediction_type`
  (`epsilon` or `sample`).
- `DiffusionPolicyConfig(PolicyConfig)`:
  - `obs_as_local_cond`, `obs_as_global_cond`, `pred_action_steps_only`,
    `oa_step_convention`: faithful copies of the upstream knobs.
  - `action_dim`, `obs_dim`, `n_action_steps`, `n_obs_steps`, `horizon`,
    `num_inference_steps`.
  - `unet`: `ConditionalUnet1DConfig` (channels, kernel size, conditioning
    type).
  - `scheduler`: DDPM scheduler config.
  - `action_scaling`: toggle for the `LinearNormalizer`.
  - `encoder_*`: optional encoder used to compute the observation embedding
    (mirrors `EncoderPolicy`).

## `DiffusionPolicy(Policy, ModuleAttrMixin)`
- Owns a `ConditionalUnet1D` denoiser and a `DDPMScheduler`.
- Uses `LowdimMaskGenerator` to mask observation tokens during conditional
  training (matches the upstream's "obs-as-local-cond" trick).
- Uses an `ObservationEncoder` (same one as `EncoderPolicy`/`GMMPolicy`) to
  flatten the observation into a low-dim conditioning vector.
- Training step (`update_params`): sample noise, scale by the scheduler,
  predict (epsilon or sample), MSE loss.
- Prediction: runs `num_inference_steps` DDPM steps to denoise an action
  sequence of length `horizon`, returns the next `n_action_steps`
  steps as a `RobotTrajectory`.
- EMA of weights is supported via `behavior_cloning.py`'s `EMAModel`.

## Notes
- The diffusion policy is gated behind the `diffusion` extra in
  `setup.py` (`einops==0.4.1`, `diffusers==0.11.1`).
- TODO comment at the top notes a planned switch to 6D action labels.
