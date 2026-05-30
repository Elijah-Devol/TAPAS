# env/__init__.py

## Purpose
Defines the supported simulator/robot back-ends and provides a string-to-class
dispatcher.

## Contents
- `Environment(Enum)` - `PANDA`, `MANISKILL`, `RLBENCH`. Used throughout
  TAPAS as the canonical identifier for the active environment.
- `get_env(env_str)`: case-insensitive lookup of the enum from a string.
- `import_env(config)`: returns the environment class corresponding to
  `config.env_type`. Imports lazily so a user only needs the dependencies of
  the chosen back-end installed (matches the per-environment extras in
  `setup.py`: `rlbench`, `maniskill`, `franka`).
