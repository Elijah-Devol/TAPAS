# policy/models/

Backing models for the policies, mirroring the encoder/models split.

## `tpgmm.py` (~160KB)
The core mathematical machinery for the paper: Task-Parameterised
Gaussian Mixture Model on the SE(3) (Riemannian) manifold, built on
`riepybdlib`. Includes:
- `TPGMM` and `AutoTPGMM` classes (the latter auto-discovers component
  count and per-component frame importance).
- `TPGMMConfig` and `AutoTPGMMConfig` dataclasses.
- Methods for fitting (EM on the manifold), Gaussian Mixture Regression
  conditioned on a subset of dimensions, conditional sampling, and
  diagnostic plotting.
- Helpers for converting between Euclidean tangent space and the
  manifold representation.
This file is the largest in the project. Its companion notebooks in
`notebooks/gm/.../*.ipynb` drive the fitting workflow.

## `motion_planner.py`
- `MotionPlanner` class wrapping `mplib`/Pinocchio's
  `OmplPlanner`/`KdlPlanner` to solve joint-space trajectories for the
  ManiSkill `MotionPlannerPolicy`. Provides `plan_to_goal(start_q,
  goal_pose_or_q, time_step, with_screw)` that returns a dict with
  `position`, `velocity`, `time`.

## `diffusion/`
Pure-network components copied from the upstream Diffusion Policy and
adjusted minimally for TAPAS:
- `conditional_unet1d.py` - the 1D U-Net used to denoise action
  sequences.
- `conv1d_components.py` - residual blocks used inside the U-Net.
- `base_lowdim_policy.py` - the upstream's abstract base.
- `dict_of_tensor_mixin.py` - convenience for parameters/buffers stored
  as dicts.
- `lr_scheduler.py` - returns an `transformers`-style scheduler with
  warmup (`cosine`, `linear`, etc.).
- `mask_generator.py` - `LowdimMaskGenerator` for obs-as-local-cond
  training.
- `module_attr_mixin.py` - lets `DiffusionPolicy.device` reach into
  child modules.
- `normalizer.py` - `LinearNormalizer` (mean/std fit per attribute, used
  by both `LSTMPolicy` and `DiffusionPolicy`).
- `positional_embedding.py` - sinusoidal positional embedding for the
  diffusion timestep input.
- `pytorch_util.py` - tensor helpers (`dict_apply`, `optimizer_to`,
  ...).
