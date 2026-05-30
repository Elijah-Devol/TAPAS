# TAPAS Architecture

This document describes how the moving parts of the TAPAS codebase fit
together: what each layer is responsible for, how data flows between
them, and where the major extension points are.

## Layering

```text
+----------------------------+
| Entry points / scripts     |  behavior_cloning, evaluate, pretrain,
|                            |  collect_data, kp_encode_trajectories,
|                            |  embed_trajectories, tsdf_fusion, ...
+----------------------------+
                |
                v
+----------------------------+
| Configs (OmegaConf)        |  dataclass configs in conf/, instantiated
|                            |  via utils/argparse.parse_and_build_config
+----------------------------+
                |
                v
+----------------------------+
| Domain layer               |
|  +- env.BaseEnvironment    |  RLBench / ManiSkill / Panda
|  +- policy.Policy          |  GMM / LSTM / diffusion / motion planner
|  +- encoder.RepLearner     |  Keypoints / BVAE / Monet / Transporter / ViT
+----------------------------+
                |
                v
+----------------------------+
| Dataset layer              |
|  +- SceneDataset           |  on-disk trajectories + embeddings + masks
|  +- BCDataset / DCDataset  |  adapters to the policy / encoder samplers
+----------------------------+
                |
                v
+----------------------------+
| Utility layer              |
|  +- utils.observation      |  SceneObservation tensorclass
|  +- utils.geometry_*       |  SE(3) + quaternion math
|  +- viz.*                  |  matplotlib + OpenCV
|  +- filter.*               |  temporal priors for keypoints
|  +- tsdf.* / dense_corresp |  GT mask generation + DC training
+----------------------------+
```

The arrows are *write-only*: each script knows about the layers below
it but no layer ever imports from above. The single source of truth for
robot data is the `SceneDataset`; everything else flows through it.

## The on-disk dataset

`SceneDataset` is the cornerstone: every entry point reads or writes one.
It lives at `<data_root>/<task>/<feedback_type>/` and contains:
- `metadata.json` - the persisted `SceneDatasetConfig` plus
  per-trajectory metadata (length, GT label set).
- `trajectories/<traj_id>/`:
  - `<cam>/rgb/<step>.png`, `<cam>/depth/<step>.dat`, `<cam>/mask/...`,
    `<cam>/extr/<step>.dat`, `<cam>/intr.dat` (flat - constant per
    trajectory).
  - `ee_pose/<step>.dat`, `action/<step>.dat`, `gripper_state/...`,
    `joint_pos/...`, `joint_vel/...`, `feedback/...`,
    `object_poses/...`.
  - `metadata.json` - per-trajectory length + GT label set.
- `embeddings/<encoder_name>/<traj>/<step>/<cam>/descriptor.dat`
  (and `.../heatmap.dat` for the transporter).
- `encodings/<encoder_name>/<selection_name>/<traj>/<step>/kp.dat`
  plus optional debug attributes per camera.
- `fusion/` - throwaway `.ply` artefacts from `tsdf_fusion.py`.

The append-only layout has two consequences:
- Multiple encoders can be cached side-by-side; each one lives under its
  own subdirectory.
- Re-running encoding requires deleting the relevant subdirectory by
  hand.

## Pipeline stages

### 1. Demo collection
- **`collect_data.py`**: drives `BaseEnvironment.step(action)` with a
  chosen policy (often `ManualPolicy` for teleop or `MotionPlannerPolicy`
  for scripted demos). Each per-step `SceneObservation` is appended to a
  `Trajectory` buffer and committed to the `SceneDataset` upon success.
- **`collect_data_rlbench.py`**: bypasses the policy and calls
  `env.task_env.get_demos(...)` to generate scripted demos, then converts
  each raw observation to a `SceneObservation`.
- **`extract_demos.py`**: same idea but starting from a
  pre-existing ManiSkill2 HDF5 demo file, optionally translating between
  control modes (`pd_joint_pos` -> `pd_ee_delta_pose`, etc.).

### 2. Mask generation (optional)
- **`tsdf_fusion.py`** -> `tsdf/fusion.py` -> `tsdf/filter.py` ->
  `tsdf/cluster.py`: integrates wrist-camera RGB-D into a TSDF volume,
  meshes it, removes the table plane, DBSCAN-clusters the remaining mesh
  into objects, rasterises the clusters into per-pixel masks for every
  camera view, and writes them back via
  `SceneDataset.add_tsdf_masks(...)`.
- Provides "ground-truth" masks for environments that don't have them
  natively (most real-robot recordings).

### 3. Encoder pretraining
- **`pretrain.py`**: instantiates a `RepresentationLearner` (BVAE /
  MONet / transporter / `KeypointsPredictor` / `VitFeatureEncoder`) and
  wraps the dataset in a `DenseCorrespondenceDataset` to draw the right
  shape of samples (`SampleTypes.DC`, `SampleTypes.CAM_PAIR`,
  `SampleTypes.CAM_SINGLE`, ...). Trains for a fixed number of steps,
  optionally with early stopping, and saves the final checkpoint.

### 4. Keypoint selection + encoding
- **`kp_encode_trajectories.py`** -> `policy/encoder.py:EncoderPseudoPolicy`:
  loads the pretrained encoder, picks *reference keypoints* once via
  `RepresentationLearner.initialize_parameters_via_dataset(...)`, and
  then encodes every frame of every trajectory. The 3D keypoint vector
  is stored under `encodings/<encoder>/<selection_name>/<traj>/<step>/kp.dat`.
- The encoder is saved back to disk with the selection name appended to
  its suffix, so the GMM / BC pipeline can re-instantiate it identically.

### 5. Policy fitting
Two parallel options:
- **TAPAS-GMM (notebook-driven)**: the user opens one of
  `notebooks/<env>/tpgmm/<task>_tpgmm.ipynb`. The notebook reads the
  encoded `SceneDataset`, builds a `Demos` object
  (`dataset/demos.py`) that converts trajectories to the task-frame
  representation, and fits a TP-GMM via
  `policy/models/tpgmm.py:AutoTPGMM` or `TPGMM`. The result is a small
  pickled model that `GMMPolicy` will load.
- **Baseline BC (`behavior_cloning.py`)**: trains an `EncoderPolicy`
  (LSTM head on pre-cached keypoints / embeddings, or an end-to-end
  encoder), a `LSTMPolicy`, or a `DiffusionPolicy` via a standard
  stochastic-gradient loop with optional EMA.

### 6. Evaluation
- **`evaluate.py`**: rebuilds the same env, instantiates the chosen
  policy class, loads its checkpoint, optionally attaches a
  `LiveKeypoints` overlay, and rolls out `n_episodes` episodes,
  logging success rate to W&B.
- For real-robot evaluation, a `KeyboardObserver` listens for human
  reset/success signals while the robot runs.

## How a single observation flows

`SceneObservation` (in `utils/observation.py`) is the tensorclass that
flows through the entire stack. Layout:

```
SceneObservation
+- cameras: TensorDict
|    +- wrist:    SingleCamObservation(rgb, depth, extr, intr, mask, ...)
|    +- overhead: SingleCamObservation(rgb, depth, extr, intr, mask, ...)
+- ee_pose, action, feedback
+- joint_pos, joint_vel, gripper_state
+- proprio_obs (optional)
+- object_poses: TensorDict (per-object 7-pose; optional)
+- kp / descriptor (optional, only when cached encodings/embeddings exist)
```

During training:
1. `BCDataset.__getitem__(idx)` returns a fragment of one trajectory
   (with optional padding) as a `SceneObservation` time series.
2. `Policy.forward_step(obs)` -> `ObservationEncoder.encode(obs)` ->
   either the cached `obs.kp` / `obs.descriptor` or
   `image_encoder.encode(obs)`.
3. `LSTMPolicy.linear_out(lstm_out)` (or the diffusion U-Net, or the GMR
   on the TP-GMM) produces an action.
4. The action's gradient flows back through the same chain (or, for
   GMM, isn't gradient-based at all - the fit happens in the notebook).

During evaluation:
1. `BaseEnvironment.step(action)` -> `env._step` reads sensors and
   builds a fresh `SceneObservation` via `env.process_observation`.
2. `Policy.predict(obs)` runs the inverse path: ObservationEncoder ->
   network/TP-GMM -> action vector.
3. The action vector is post-processed
   (`BaseEnvironment.postprocess_action`: quaternion convention,
   `delay_gripper`, action scaling) and sent to the robot/sim.

## Cross-cutting concerns

### Configuration
- Every entry point uses `utils.argparse.parse_and_build_config(...)` to
  load a Python config module, structure it through OmegaConf, apply
  `--overwrite` dotpath edits, and instantiate the dataclass tree via
  `OmegaConf.to_container(..., structured_config_mode=SCMode.INSTANTIATE)`.
- Sentinel values (`SET_PROGRAMMATICALLY`, `COPY_FROM_MAIN_FITTING` from
  `utils/config.py`) let configs declare "the script will fill this in",
  enforced at runtime by `value_not_set`.
- `_machine.py` (untracked) holds per-machine paths so config files stay
  portable.

### Quaternion conventions
- TAPAS uses *real-first* `[qw, qx, qy, qz]` quaternions internally.
- `env/rlbench.py` converts to/from RLBench's *real-last* convention
  (`quat_real_first_to_real_last` and the inverse).
- `env/franka.py` uses real-first directly (matches `roboticstoolbox`).
- All math lives in `utils/geometry_torch.py` (autograd-friendly) and
  `utils/geometry_np.py` (numpy mirror).

### Device selection
`utils/select_gpu.py` runs `nvidia-smi` once on import and picks the GPU
with the most free memory; the resulting `device` is imported by
basically everything. Falls back to CPU when CUDA is unavailable.

### Logging
- `loguru` for application logs, with a context-managed indentation
  facility (`utils/logging.indent_logs`).
- `wandb` for training / evaluation metrics (toggled via
  `training.wandb_mode` config field; common overrides include
  `wandb_mode=disabled`).

### Checkpoints
- `to_disk` / `from_disk` on every `Policy` and `RepresentationLearner`
  is just `torch.save(state_dict)` / `torch.load(map_location=device)`.
- `save_config_along_path(config, ckpt_path)` writes
  `<ckpt>.json` + `<ckpt>.yaml` so the run that produced the checkpoint
  is fully reconstructible. The git revision is recorded automatically
  via `utils/version.get_git_revision_hash`.

### Visualisation
- `viz/` has both online (matplotlib `ParticleFilterViz`, OpenCV
  `LiveKeypoints`, `HeatmapVisualization`) and offline (`viz/gmm.py`,
  `summarize_encoding.py`, `visualize_demo.py`) tooling.
- Helpers are decorated with `@mp_wrapper` (`utils/multi_processing.py`)
  when running them inline would block the main loop.

## Extension points

| You want to ... | Touch |
| --- | --- |
| Support a new simulator | Add `env/<name>.py`, register in `env/__init__.py:Environment` & `import_env`; add an `Environment.<NAME>` case to `tsdf/filter.py:coordinate_boxes` and `gripper_dists`. |
| Add a new encoder | Subclass `RepresentationLearner`; register in `encoder/__init__.py:encoder_switch`; add a `<encoder_config_class>` to `encoder_config_map`; declare a `sample_type` so `DenseCorrespondenceDataset` knows what batches to draw. |
| Add a new policy | Subclass `policy/policy.py:Policy`; register in `policy/__init__.py:PolicyEnum` and `import_policy`. |
| Add a new task | RLBench: add to `env/rlbench.py:task_switch`; horizon to `utils/tasks.py:task_horizons`. Sim/real: write a new config in `conf/`. The TP-GMM notebook is also per-task. |
| Add a new temporal prior | Subclass the filter; expose a config and a `PriorTypes` enum entry; wire it into `KeypointsPredictor._setup_filter`. |
| Cache a new per-frame feature | Pick a name; write it via `BCDataset.add_embedding/add_encoding`; teach `SceneDataset._get_bc_traj` to load it back. |

## Performance notes

- **Encoder pretraining** is GPU-bound. The dense-correspondence loss
  scales with `num_matching_attempts * num_non_matches_per_match`; reduce
  these in `DCDataConfig` if VRAM is tight.
- **TSDF fusion** is GPU-bound (pycuda path) or CPU-bound + numba
  (fallback). The wrist camera is heavily subsampled (`SubSampleTypes.POSE`)
  before fusion to keep the volume size manageable.
- **Particle filter** evaluates the descriptor at every particle per
  camera per step. Particle count and number of cameras have the biggest
  effect on online runtime.
- **TP-GMM** fitting is CPU-bound (`riepybdlib`) and usually completes in
  seconds for a handful of demos; inference is closed-form GMR plus
  optional TOPP scheduling.

## File-by-file documentation

Per-script documentation lives under `doc/scripts/`:

```text
doc/
+- OVERVIEW.md                 # this repo overview
+- ARCHITECTURE.md             # this file
+- scripts/
   +- behavior_cloning.md      # one .md per script under tapas_gmm/
   +- collect_data.md
   +- ...
   +- dataset/                 # one .md per file in tapas_gmm/dataset/
   +- dense_correspondence/    # overview of the DC package
   +- encoder/                 # one .md per encoder
   +- encoder_models/          # overview of encoder/models/
   +- env/                     # one .md per env adapter
   +- filter/                  # overview of the filter package
   +- policy/                  # one .md per policy
   +- policy_models/           # overview of policy/models/
   +- tsdf/                    # overview of the TSDF package
   +- utils/                   # overview of the utils package
   +- viz/                     # overview of the viz package
```
