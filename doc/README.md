# TAPAS Documentation

This folder contains the auto-generated developer documentation for the
TAPAS-GMM repository.

## Top-level docs
- **[OVERVIEW.md](OVERVIEW.md)** - what the repository does, supported
  environments, the end-to-end workflow, and a directory map.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - how the modules fit together,
  the on-disk dataset layout, how an observation flows through the
  stack, and where to extend.

## Per-script docs (`scripts/`)

Each Python script under `tapas_gmm/` has its own markdown file.

### Top-level entry points & utilities
- [behavior_cloning.md](scripts/behavior_cloning.md) - `tapas-bc`: train a
  BC policy.
- [collect_data.md](scripts/collect_data.md) - `tapas-collect`: drive an
  env with a policy and record demos.
- [collect_data_rlbench.md](scripts/collect_data_rlbench.md) -
  RLBench-scripted demo collection.
- [compare_models.md](scripts/compare_models.md) - state-dict diff CLI.
- [embed_trajectories.md](scripts/embed_trajectories.md) - `tapas-embed`:
  cache dense embeddings.
- [evaluate.md](scripts/evaluate.md) - `tapas-eval`: roll out a trained
  policy.
- [extract_demos.md](scripts/extract_demos.md) - convert ManiSkill2 HDF5
  demos.
- [inspect_dataset.md](scripts/inspect_dataset.md) - print a dataset
  summary.
- [kp_encode_trajectories.md](scripts/kp_encode_trajectories.md) -
  `tapas-kp-encode`: cache 3D keypoint encodings.
- [pretrain.md](scripts/pretrain.md) - `tapas-pretrain`: train an
  encoder.
- [reconstruct_actions.md](scripts/reconstruct_actions.md) - rebuild
  per-step actions from observations.
- [summarize_encoding.md](scripts/summarize_encoding.md) - plot keypoint
  trajectories.
- [tsdf_fusion.md](scripts/tsdf_fusion.md) - build per-object masks via
  TSDF fusion.
- [visualize_demo.md](scripts/visualize_demo.md) - 5-panel single-demo
  viz.
- [visualize_dense_correspondence.md](scripts/visualize_dense_correspondence.md) -
  live heatmap GUI.
- [visualize_embedding.md](scripts/visualize_embedding.md) - legacy
  reconstr/t-SNE/slots viz.

### Sub-modules
- [dataset/](scripts/dataset/) - on-disk `SceneDataset`, BC/DC adapters,
  TP-GMM `Demos` prep.
- [encoder/](scripts/encoder/) - `RepresentationLearner` base class and
  each concrete encoder (BVAE, MONet, transporter, keypoints,
  keypoints_gt, CNN, ViT).
- [encoder_models/](scripts/encoder_models/overview.md) - raw network
  architectures under `encoder/models/`.
- [env/](scripts/env/) - env adapters for Panda, ManiSkill, and RLBench
  plus the shared `BaseEnvironment` API.
- [policy/](scripts/policy/) - `Policy` base, LSTM/GMM/Diffusion
  baselines, motion-planner / manual / random / sphere policies, and
  the `EncoderPolicy` variants.
- [policy_models/](scripts/policy_models/overview.md) - TP-GMM and
  diffusion-network internals.
- [dense_correspondence/](scripts/dense_correspondence/overview.md) -
  correspondence sampling, augmentation, and pixelwise contrastive loss.
- [filter/](scripts/filter/overview.md) - discrete and particle filters
  used as temporal keypoint priors.
- [tsdf/](scripts/tsdf/overview.md) - TSDF fusion, plane filtering,
  DBSCAN clustering.
- [utils/](scripts/utils/overview.md) - argparse glue, geometry, IO,
  loggers, tensorclass observation, GPU selection, etc.
- [viz/](scripts/viz/overview.md) - matplotlib + OpenCV visualisation
  helpers.
