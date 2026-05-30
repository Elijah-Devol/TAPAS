# utils/ - Shared utilities

This document covers every file under `tapas_gmm/utils/`. Most utils are
small enough that a single-paragraph description suffices.

---

## `__init__.py`
Imports `trunc_normal_` from `utils.vit` so the ViT backbone's
`torch.hub.load` finds it (the upstream DINO/DINOv2 code expects this name
at module scope). Pure side-effect import.

## `argparse.py`
Glue between argparse and OmegaConf:
- `parse_args(data_load, need_task, extra_args)`: standard argparse with
  `-c/--config` (required), `-t/--task`, `-f/--feedback_type`, `--path`,
  `--overwrite key=val ...`, and any extra args the caller wants.
- `build_config(...)`: imports the Python config file from `-c`, converts
  to `OmegaConf.structured`, fills `data_naming.task/feedback_type/path`,
  applies the `--overwrite` dotlist.
- `parse_and_build_config(...)`: one-liner used by every entry point.
- `resolve_tuple`: registers an `${as_tuple:a,b,c}` OmegaConf resolver
  used in the YAML configs.

## `config.py`
Config-handling helpers:
- `_SENTINELS` (Enum): `SET_PROGRAMMATICALLY` and `COPY_FROM_MAIN_FITTING`,
  used as in-config sentinels.
- `value_not_set(val)`: `True` for `None` or `SET_PROGRAMMATICALLY`.
- `dict_to_disk` / `dict_from_disk`: jsonpickle (de)serialisation.
- `yaml_to_disk`: writes a YAML string to disk.
- `structured_config_to_dict` / `structured_config_to_yaml`: shorthand
  conversions.
- `save_config_along_path(config, path)`: drops both `.json` and `.yaml`
  copies of a config next to a checkpoint file.
- `recursive_compare_dict` / `recursive_compare_dataclass`: human-readable
  diff for nested dicts / dataclasses.

## `cuda.py`
Optional pycuda integration:
- Try-imports `pycuda.driver` and `pycuda.compiler.SourceModule`. If it
  fails, sets `FUSION_GPU_MODE = 0` so `tsdf.fusion` falls back to CPU.
- `try_empty_cuda_cache`, `try_make_context`, `try_destroy_context`,
  `try_debug_memory`: silent wrappers that no-op on CPU.

## `data_loading.py`
Dataset/loader glue:
- `InfiniteDataIterator` abstract iterator.
- `AutoResettingDataIterator`: keeps yielding until a max step count,
  re-creating the iterator when the loader is exhausted.
- `DataLoaderConfig`: `train_split`, `batch_size`, `eval_batchsize`,
  `shuffle`, `num_workers`, `persistent_workers`, etc.
- `build_data_loaders(dataset, collate, config)` /
  `build_data_iterators(...)` / `build_infinte_data_iterators(...)`:
  create the (train, eval) `DataLoader`s with proper subset samplers and
  wrap them in `InfiniteDataIterator`s for the BC / pretrain loops.
- `load_image`, `save_image`, `load_tensor`, `save_tensor`: helpers used by
  `Trajectory.save` and `SceneDataset._get_bc_traj` to read PNG images
  and `.dat` tensors.

## `debug.py`
Developer helpers:
- `nan_hook(self, inp, output)`: register on `nn.Module`s; prints input
  shapes and indices of any NaNs in the output - very handy when a loss
  goes NaN.
- `summarize_tensor`, `summarize_list_of_tensors`: pretty-print tensor
  shape/dtype/stats, used by `inspect_dataset.py`.
- `measure_runtime(func)`: decorator that wraps a function call with
  `time.perf_counter` and logs the elapsed time.
- `save_q_traj_dbg`: debug dumper for joint trajectories (commented out
  reference in `topp.py`).

## `disturbance.py`
`disturbe_at_step_no(env, current_step, disturbe_at_step)`: if the current
step matches the configured "disturb at" step, call `env.reset_joint_pose`.
Used by `evaluate.py` to test recovery from forced re-resets.

## `ema.py`
`EMAModel`: Exponential Moving Average of model weights, copied from the
`diffusers` package. Holds a `copy.deepcopy(model)` and updates it
incrementally with a warmup-aware decay schedule
(`decay = 1 - (1 + step/inv_gamma)^(-power)`, clamped to
`[min_value, max_value]`). Exposes `averaged_model` for eval.

## `franka_joint_commander.py`
Real-robot helpers for the Franka stack:
- `JointPoitionCommander`: low-level ROS publisher that takes a `q`
  target, publishes it on a command topic, and listens for the
  resulting `JointStateMsg` to confirm.
- `ThreadedJointTrajectoryFollower`: runs the joint trajectory on a
  background thread so the env's main step loop doesn't block waiting for
  motion. Consumes a `RobotTrajectory` and steps point-by-point.

## `franka.py`
Franka-specific math helpers:
- `clamp_translation(pos, box)`: clamp a 3-vector to a per-task safety
  box.
- `compute_ee_delta(ee_pose, next_ee_pose)`: convert two consecutive EE
  poses to a delta-pose action (matching the env's action space).
- `reconstruct_actions_obs(batch_obs)`: derive a per-step action from a
  sequence of `SceneObservation`s, used by `reconstruct_actions.py`.
- `subsample_image`: bilinear downsample to 256x256.
- `pad_list`, etc.

## `geometry_np.py` (16KB)
Numpy implementations of rigid-body geometry routines:
- Quaternion math: `quaternion_to_axis_angle`,
  `axis_angle_to_quaternion`, `quaternion_to_matrix`,
  `quaternion_multiply`, `quaternion_diff`, `conjugate_quat`,
  `quaternion_pose_diff`, `compute_angle_between_quaternions`,
  `quat_real_first_to_real_last` (and inverse),
  `ensure_quat_positive_real_part`, `ensure_quaternion_continuity`,
  `quaternion_is_unit`, `normalize_quaternion`.
- SE(3) helpers: `homogenous_transform_from_rot_shift`,
  `invert_homogenous_transform`, `compute_angle_between_poses`,
  `compute_distance_between_poses`, `quaternion_from_matrix`,
  `axis_and_angle_to_quaternion`, `euler_angles_to_axis_angle`,
  `rotate_vector_by_quaternion`, `overlapping_split`.
All conventions are *real-first* quaternions (the project's standard);
explicit converters exist for RLBench's "real last" convention.

## `geometry_torch.py` (39KB)
Torch equivalents of the above, plus per-batch helpers needed by the
keypoint encoder and TP-GMM:
- All of the np helpers, lifted to autograd-friendly torch ops.
- `batched_project_onto_cam`, `batched_rigid_transform`,
  `append_depth_to_uv`, `hard_pixels_to_3D_world`, `invert_intrinsics`,
  `frame_transform_pos_quat`, `homogenous_transform_from_rot_shift`,
  `set_b_in_homogenous_transforms`, `get_R_from_homogenous_transforms`,
  `get_b_from_homogenous_transforms`,
  `translation_to_direction_and_magnitude`,
  `axis_angle_to_quaternion`, `axis_angle_to_matrix`,
  `quaternion_lot_multiply`, `quaternion_is_unit`, `quaternion_to_axis_and_angle`,
  `standardize_quaternion`, `rotate_quat_y180`, `cos`, `sin`,
  `modulo_rotation_angle`, `hom_to_shift_quat`, `quarter_rot_angle`,
  `remove_quaternion_dim`, `identity_quaternions`, `identity_7_pose`,
  ...

## `gmm.py`
Helpers for working with `riepybdlib` Gaussians:
- `concat_mvn_rbd(gaussians)` (`NotImplementedError` - placeholder for a
  future block-diagonal concatenation of Riemannian Gaussians).
- Helpers for stacking and converting Riemannian GMMs.

## `human_feedback.py`
Per-step feedback labels and `correct_action(keyboard_obs, action)` that
patches in joint/gripper corrections from the `KeyboardObserver`. The
feedback-type enumeration maps to the labels stored by `collect_data.py`
in the dataset (`evaluative`, `dagger`, `iwr`, `ceiling_full`, etc.).

## `keyboard_observer.py`
A small `pynput`-based hot-key listener. Keys:
- `g` / `b`: good/bad label.
- `c` / `v` / `f`: gripper close / open / free.
- `x` / `y`: reset with / without success.
- Arrow keys / WASD: joint deltas.
Exposes `has_joints_cor()`, `has_gripper_update()`, `success`,
`reset_button`, `get_label()`. Falls back to a no-op when `pynput` isn't
installed (headless mode).

## `keypoints.py`
Tiny helpers shared between encoder and policy:
- `get_keypoint_distance(set1, set2)`: per-keypoint L2 distance.
- `unflatten_keypoints(kp, kp_dim=3)`: stacked `x,y,(z)` ->
  `(n_kp, d_kp)`.
- `tp_from_keypoints(kp, indeces)`: derive task-parameter frames from
  keypoint positions (used to build object frames when no ground-truth
  pose is available).

## `logging.py`
Loguru integration:
- `indent_logs()` context manager: indents subsequent log lines for nested
  sections.
- `indent_func_log(func)` / `log_constructor(func)`: decorators that wrap
  a call in `indent_logs` and log entry/exit.
- `DuplicateFilter`: dedupe repeated log lines.
- `patcher`: adds an indentation prefix to every log line based on the
  current context.

## `manifolds.py`
Shorthand handles for the riepybdlib manifolds used by the TP-GMM:
`Manifold_T` (1D time), `Manifold_R1`, `Manifold_R3`, `Manifold_S1`,
`Manifold_S2`, `Manifold_Quat`.

## `maniskill_replay.py`
Glue between TAPAS' replay buffer and ManiSkill2's
`replay_trajectory.py`:
- `from_pd_joint_pos(target_mode, ori_actions, ori_env, env, ...)`:
  translates a joint-position-controlled demo into another control mode,
  recording each step into the TAPAS replay memory along the way.
- `from_pd_joint_delta_pos(...)`: same but starting from joint-delta
  control.
- `PDJointPos2EETranslator`: the online translator used by
  `MotionPlannerPolicy`. Maintains a twin env that runs in
  joint-position mode; for each commanded joint position, derives the
  matching EE delta-pose action to drive the live env.

## `metrics_logger.py`
`MetricsLogger`: small bookkeeping helper for per-episode and lifetime
counts (`total_successes`, `total_episodes`, `total_steps`, etc.). Used
during real-robot evaluation; logs to wandb.

## `misc.py`
Grab-bag:
- `DataNamingConfig(feedback_type, task, data_root)`: ubiquitous nested
  dataclass used by every entry point.
- `get_dataset_name`, `get_full_task_name`,
  `policy_checkpoint_name`, `pretrain_checkpoint_name`: canonical
  filename generation given a config.
- `import_config_file(path)`: dynamic import of a Python config module
  (returns the imported module).
- `load_scene_data(naming)`: build a `SceneDataset` from a naming config.
- `load_replay_memory(config)` / `load_replay_memory_from_path(path)`:
  used by the legacy visualisation scripts.
- `apply_machine_config(config)`: layers `conf/_machine.py` overrides on
  top of the loaded config.
- `loop_sleep(start_time, target_freq=20)`: sleep just enough to keep a
  loop at the target frequency.
- `multiply_iterable`: simple `reduce(mul, ...)` helper.
- `configure_class_instance`, `get_and_log_failure`: ergonomic helpers
  for loading nested configs with fallbacks.

## `multi_processing.py`
`mp_wrapper(func)` decorator and `launch_in_mp(func, *args, **kwargs)`
helper: spawn `multiprocessing.Process(target=func, ...)` to run a
function in its own process. Used to keep heavy SimulatorScene processes
from polluting the main process.

## `np.py`
`np_cache(*lru_args, array_argument_index=0, **lru_kwargs)`: an
`functools.lru_cache` variant that hashes a numpy array argument by its
tobytes representation. Used for caching expensive numpy-keyed
computations.

## `observation.py` (22KB)
The tensorclass definitions and per-attribute machinery that everyone
else uses:
- `SceneObservation`: top-level tensorclass with `cameras` (a TensorDict
  keyed by camera name), `action`, `feedback`, `ee_pose`, `gripper_state`,
  `proprio_obs`, `joint_pos`, `joint_vel`, `object_poses` (a TensorDict),
  and optional `kp`/`descriptor` slots.
- `SingleCamObservation`: `rgb`, `depth`, `extr`, `intr`, `mask`, etc.
- `SingleCamSceneObservation`: stacked-over-cameras helper.
- `ObservationConfig`: which cameras / mask type / image_crop / image_dim
  /  pose-related fields a script should consume.
- `MaskTypes` (Enum): `NONE`, `GT`, `TSDF`.
- `SampleTypes` (Enum): `CAM_SINGLE`, `CAM_PAIR`, `DC`, `GT`, ...
- `CameraOrder`: registers the canonical camera ordering used when
  flattening a `cameras` dict into a tensor.
- `ALL_CAMERA_ATTRIBUTES`, `GENERIC_ATTRIBUTES`: lookup tables of
  per-camera and per-step attribute names that `Trajectory.save` uses to
  drive its on-disk layout.
- `get_cam_attributes(cam, mask_type)`, `make_cam_attr_name(cam, attr)`,
  `is_flat_attribute(attr)`: name-mangling helpers used by the trajectory
  writer/reader.
- Subsampling: `get_idx_by_pose_difference_threshold`,
  `get_idx_by_target_len`, `downsample_traj_by_idx`,
  `downsample_tensordict_by_idx`, `downsample_to_target_freq`,
  `get_idx_by_pose_difference_threshold_matrix`.
- Other helpers: `dict_to_tensordict`, `tensorclass_from_tensordict`,
  `tensor_dict_equal`, `random_obs_dropout`, `empty_batchsize`,
  `collate`, `get_object_labels`.

## `random.py`
Seed plumbing: `configure_seeds(seed)` and `set_seeds(seed)` (torch +
cuda + numpy + python `random`).

## `robot_trajectory.py`
- `TrajectoryPoint`: dataclass with `t`, `q`, `qd`, `qdd`, `gripper`,
  `ee`. Any subset of these can be set.
- `RobotTrajectory`: list of `TrajectoryPoint`s plus a duration; iterable.
  Used to plumb full trajectories (from GMM/diffusion) through the env's
  step loop.

## `sapien_scene.py`
A small Sapien-based "twin" scene used by the motion-planner policy.
Loads the Franka URDF, builds a `pinocchio_model` for IK, and exposes a
`step` method that mirrors the live env.

## `select_gpu.py`
On import, parses `nvidia-smi`'s free-memory column and picks the GPU
with the most free memory, exporting `device = torch.device("cuda:N")`.
Falls back to CPU if CUDA is unavailable. Pretty much every module that
needs `device` imports it from here.

## `tasks.py`
Hard-coded `task_horizons` lookup (e.g. `CloseMicrowave: 300`,
`PutRubbishInBin: 600`) wrapped in a `defaultdict(lambda: 300)`.
`get_task_horizon(config)` returns `None` for the real Panda and the
table value for simulated tasks. Used by `evaluate.py` to set the default
horizon.

## `topp.py`
Wraps `mplib.Planner.TOPP` (Time-Optimal Path Parameterisation) for the
GMM policy:
- `TOPP` class: takes a `RobotTrajectory` and re-times it to satisfy
  velocity / acceleration constraints.
- Falls back to no-op if `mplib` is not installed (logs a warning at
  import time).

## `torch.py`
Torch tensor helpers:
- `list_or_tensor(func)`, `list_or_tensor_mult_return(func)`,
  `list_or_tensor_mult_args(func)`: decorators that let a function operate
  on either a tensor, a tuple, or a list of tensors transparently.
- `cat`, `stack`, `unsqueeze`, `to_numpy`: list-or-tensor lifts of the
  obvious torch ops.
- `batched_block_diag`, `eye_like`, `single_index_any_tensor_dim`,
  `slice_any_tensor_dim`: indexing helpers used by the encoder and
  TP-GMM.
- `compare_state_dicts(a, b)`: diff helper used by `compare_models.py`.

## `typing.py`
Type aliases: `TensorOrTensorSeq`, `NDArrayOrNDArraySeq`.

## `version.py`
`get_git_revision_hash()`: returns `git rev-parse HEAD` as a string.
Recorded in every checkpoint config so a future user can identify the
exact code revision that produced it.

## `vit.py`
Re-implementation of `torch.nn.init.trunc_normal_` for compatibility with
older PyTorch releases - required by the upstream DINO/DINOv2 ViT models
loaded via `torch.hub.load`.
