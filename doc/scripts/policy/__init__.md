# policy/__init__.py

## Purpose
Registry of policy back-ends and a dynamic loader. Like `env/__init__.py`,
imports each policy module lazily so a user doesn't need optional
dependencies (e.g. `mplib` for the motion planner, `diffusers` for the
diffusion policy) unless they actually use that policy.

## Contents
- `PolicyEnum(Enum)` - `RANDOM`, `MANUAL`, `SPHERE`, `MOTION_PLANNER`,
  `GMM`, `LSTM`, `DIFFUSION`. The OmegaConf configs reference these by
  value.
- `import_policy(policy_name, disk_read=False) -> Policy` - returns the
  policy class for the given name. Supports an extra `"encoder"` name that
  maps to `EncoderPolicy` (an LSTM-on-encoder backbone), which is the
  default used by `evaluate.py`.

The commented-out `policy_switch` dict shows the previous eager-import
approach; the lazy `import_policy` was preferred so the package can be
imported on machines without every back-end installed.
