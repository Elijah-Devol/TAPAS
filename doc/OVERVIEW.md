# TAPAS Repository Overview

## What is TAPAS?

This repository, **TAPAS-GMM**, is the official implementation of the IEEE
RA-L 2024 paper "The Art of Imitation: Learning Long-Horizon Manipulation
Tasks From Few Demonstrations" (von Hartz, Welschehold, Valada, Boedecker).
It learns to perform long-horizon robot manipulation from a small number
of demonstrations (typically 5-10) by combining:

1. A **learned keypoint encoder** (`tapas_gmm.encoder.KeypointsPredictor` /
   `VitKeypointsPredictor`) trained with dense-correspondence pretraining.
   The encoder turns each RGB-D image into a small set of object-centric
   3D keypoints whose locations are consistent across viewpoints, objects,
   and trajectories.
2. A **task-parameterised Gaussian Mixture Model on the SE(3) manifold**
   (`tapas_gmm.policy.GMMPolicy` / `policy/models/tpgmm.py`, built on
   `riepybdlib`). The mixture is conditioned on per-object frames derived
   from the keypoints, so the same model generalises across different
   initial object placements.
3. A **time-optimal trajectory generator** (`utils/topp.py`) that scales
   the GMM-predicted trajectory in time to satisfy
   velocity/acceleration constraints on the real robot.

The repository also ships **baselines** (LSTM-on-encoder behavioural
cloning, Diffusion Policy, motion-planner scripted policies, random and
manual policies) so the same pipeline can compare TAPAS-GMM against
alternative learning approaches on the same data.

## Supported environments
- **RLBench** (CoppeliaSim) - the paper's main simulated benchmark.
- **ManiSkill2** (Sapien) - additional simulated tasks.
- **Real Franka Emika Panda** - the lab's ROS-based real-robot setup
  (also includes a teleop workflow).

## Workflow at a glance

```text
              +-----------------+
              | collect_data.py |  manual / scripted demos
              | (or extract_demos.py from ManiSkill HDF5)
              +--------+--------+
                       |
                       v
              SceneDataset on disk (trajectories/, embeddings/, encodings/)
                       |
        +--------------+--------------+
        |                             |
        v                             v
+------------------+        +------------------+
|  tsdf_fusion.py  |        |   pretrain.py    |  Train encoder
| Generates GT mask|        |  (DC / BVAE /    |  via dense correspondence
+------------------+        |   MONet / VIT)   |  or autoencoding
                            +--------+---------+
                                     |
                                     v
                       +----------------------------+
                       |  kp_encode_trajectories.py |  Pick reference keypoints,
                       |  (--selection_name exp)    |  encode every frame, write
                       +-------------+--------------+  back to dataset
                                     |
                       +-------------+---------------------+
                       |                                   |
                       v                                   v
       +-----------------------------+      +---------------------------+
       |   notebooks/.../*_tpgmm     |      |    behavior_cloning.py    |
       |   .ipynb (fit TP-GMM)       |      |    (LSTM / Diffusion BC)  |
       +-------------+---------------+      +-------------+-------------+
                     |                                    |
                     v                                    v
                +----------------+              +---------------+
                |  evaluate.py   |  <-----------+  same script  |
                +----------------+              +---------------+
```

## Entry points (see `setup.py`)
- `tapas-collect` -> `tapas_gmm.collect_data` - record demos
  (real Franka or sim).
- `tapas-collect-rlbench` -> `tapas_gmm.collect_data_rlbench` - record
  RLBench scripted demos.
- `tapas-pretrain` -> `tapas_gmm.pretrain` - train an encoder.
- `tapas-embed` -> `tapas_gmm.embed_trajectories` - cache dense
  embeddings.
- `tapas-kp-encode` -> `tapas_gmm.kp_encode_trajectories` - pick
  reference keypoints and cache per-frame 3D keypoint encodings.
- `tapas-bc` -> `tapas_gmm.behavior_cloning` - train an LSTM /
  diffusion / encoder-policy.
- `tapas-eval` -> `tapas_gmm.evaluate` - roll out a trained policy in
  the env.

Other scripts (not entry points) are run with `python -m
tapas_gmm.<name>`:
- `compare_models`, `extract_demos`, `inspect_dataset`,
  `reconstruct_actions`, `summarize_encoding`, `tsdf_fusion`,
  `visualize_demo`, `visualize_dense_correspondence`,
  `visualize_embedding`.

## Configuration

Every entry point is configured by:
1. A Python module under `conf/<script>/.../config.py` that builds a
   `Config` dataclass instance via OmegaConf. The structure of `conf/`
   mirrors the script structure of `tapas_gmm/`.
2. Command-line arguments:
   - `-c/--config <path>` (required).
   - `-t/--task <Name>` (most scripts) - chosen task name; doesn't require
     a new config file per task.
   - `-f/--feedback_type <name>` - which dataset variant to read/write.
   - `--overwrite key1=val1 key2=val2 ...` - dot-path overrides
     (e.g. `wandb_mode=disabled`, `policy.suffix=tx`).
3. A machine-specific `conf/_machine.py` containing local paths
   (CUDA, CoppeliaSim, data root). Imported on demand by other configs.

The bundled `notebooks/` directory contains one Jupyter notebook per
paper task; the notebook is the canonical place that *fits* the TP-GMM
(it's a multi-step process with interactive choices about component count,
frame importance, etc.). The notebooks read the same on-disk
`SceneDataset` that the rest of the pipeline writes.

## Repository layout

```text
TAPAS/
+- README.md, LICENSE, setup.py, requirements.txt, justfile
+- rlbench_mode.sh                 # Coppelia/Qt env vars
+- update_display.sh               # X-forwarding helper
+- panda_joystick.lyt              # joystick layout for Franka teleop
+- gamepad_labelled.png            # documentation image
+- conf/                           # OmegaConf configs, mirroring scripts
|   +- _machine.py                 # local paths
|   +- behavior_cloning/, evaluate/, kp_encode_trajectories/,
|       pretrain/, collect_data/, embed_trajectories/, tsdf_fusion/, ...
+- notebooks/                      # TP-GMM fitting notebooks per task
|   +- franka/, rlbench/
+- tapas_gmm/                      # the Python package
    +- behavior_cloning.py         # entry point: BC training
    +- collect_data.py             # entry point: demo collection
    +- collect_data_rlbench.py     # RLBench scripted-demo collector
    +- embed_trajectories.py       # entry point: cache embeddings
    +- evaluate.py                 # entry point: policy rollout
    +- extract_demos.py            # convert ManiSkill HDF5 demos
    +- inspect_dataset.py          # CLI dataset inspector
    +- kp_encode_trajectories.py   # entry point: cache 3D keypoints
    +- pretrain.py                 # entry point: encoder pretraining
    +- reconstruct_actions.py      # rebuild action labels from obs
    +- summarize_encoding.py       # plot keypoint trajectories
    +- tsdf_fusion.py              # build per-object GT masks
    +- visualize_demo.py           # 5-panel single-demo viz
    +- visualize_dense_correspondence.py  # live heatmap GUI
    +- visualize_embedding.py      # legacy reconstr / t-SNE / slots
    +- compare_models.py           # state-dict diff CLI
    +- assets/                     # bundled MPLib config etc.
    +- dataset/                    # SceneDataset / BCDataset / DCDataset
    |   +- scene.py                # on-disk SceneDataset
    |   +- bc.py                   # BC adapter
    |   +- dc.py                   # dense-correspondence adapter
    |   +- demos.py                # TP-GMM input prep
    |   +- mobile_rl.py            # external dataset adapter
    |   +- trajectory.py           # in-memory trajectory buffer
    +- encoder/                    # representation learners + ObservationEncoder
    |   +- representation_learner.py  # base class
    |   +- encoder.py              # ObservationEncoder (low-dim + image)
    |   +- bvae.py, cnn.py, monet.py, transporter.py
    |   +- keypoints.py            # KeypointsPredictor (TAPAS encoder)
    |   +- keypoints_gt.py         # GT-keypoint oracle
    |   +- vit_extractor.py        # DINO/DINOv2 backbone + ViT keypoints
    |   +- models/                 # raw network arch (bvae, monet,
    |                                transporter, keypoints, vit_extractor)
    +- env/                        # simulator / robot adapters
    |   +- environment.py          # BaseEnvironment, action postprocess
    |   +- franka.py               # real Panda via ROS
    |   +- mani_skill.py           # ManiSkill2 / Sapien
    |   +- rlbench.py              # RLBench / CoppeliaSim
    +- policy/                     # control policies
    |   +- policy.py               # Policy base class
    |   +- encoder.py              # EncoderPolicy / PseudoPolicy / DiskRead
    |   +- lstm.py                 # LSTMPolicy (baseline)
    |   +- gmm.py                  # GMMPolicy (TAPAS)
    |   +- diffusion.py            # DiffusionPolicy (baseline)
    |   +- motion_planner.py       # scripted ManiSkill policy
    |   +- manual.py, random.py, sphere.py
    |   +- models/
    |       +- tpgmm.py            # TP-GMM math (160KB)
    |       +- motion_planner.py   # mplib wrapper
    |       +- diffusion/          # Diffusion Policy networks
    +- dense_correspondence/       # DC pretraining helpers
    |   +- correspondence_finder.py
    |   +- correspondence_augmentation.py
    |   +- loss/
    |       +- pixelwise_contrastive_loss.py
    |       +- loss_composer.py
    +- filter/                     # temporal priors for keypoints
    |   +- discrete_filter.py
    |   +- particle_filter.py
    +- tsdf/                       # TSDF fusion + clustering
    |   +- fusion.py
    |   +- filter.py
    |   +- cluster.py
    +- viz/                        # matplotlib/OpenCV plotting
    +- utils/                      # shared math / IO / logging
```

## Where to start reading the code

If you just want to understand the TAPAS-GMM pipeline:
1. `tapas_gmm/encoder/keypoints.py` and `encoder/models/keypoints/` -
   how a frame becomes 3D keypoints.
2. `tapas_gmm/dataset/demos.py` - how those keypoints become TP-GMM
   inputs.
3. `tapas_gmm/policy/models/tpgmm.py` - the math.
4. `tapas_gmm/policy/gmm.py` - how the fitted model is used online.
5. A task notebook in `notebooks/franka/tpgmm/<task>_tpgmm.ipynb` - the
   workflow tying it all together.

For the baseline comparison (Diffusion Policy / LSTM behavioural
cloning), start at `tapas_gmm/behavior_cloning.py` and
`tapas_gmm/policy/diffusion.py` (or `policy/lstm.py`).
