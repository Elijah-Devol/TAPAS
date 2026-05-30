# encoder/bvae.py - BVAE encoder

## Purpose
A beta-VAE image encoder, wrapping `encoder.models.bvae.beta_vae.BetaVAE`.
Trained as a single-image autoencoder; the posterior mean `mu` is used as the
embedding.

## Key class: `BVAE(RepresentationLearner)`
- `sample_type = SampleTypes.CAM_SINGLE` - dense-correspondence training only
  needs single images for the BVAE loss.
- Loss types (`H`, `B`) follow the original beta-VAE papers (Higgins et al.
  2017 and Burgess et al. 2018). The "B" loss anneals a capacity term `C` from
  0 to `max_capacity` over `capacity_max_iter` iterations.
- KLD correction: if `kld_correction=True`, the KLD term is reweighted by
  `batch_size/dataset_size` to keep the ELBO scale invariant to batch size.

### Configs
- `BVAEConfig` -> `bvae.Config` (model arch) + `PretrainingConfig` (training
  hyperparameters: lr, beta, gamma, max capacity, KLD correction flag).

### Key methods
- `process_batch`: downsamples to 128x128, forwards through `model`, calls
  `loss`.
- `update_params`: one Adam step.
- `encode_single_camera`: downsamples the camera RGB to the model's resolution
  and returns the encoder's posterior mean as the embedding.
- `reconstruct(batch)`: decodes the input through the BVAE (used by
  `visualize_embedding.py`).
- `get_latent_dim(config, n_cams=1, image_dim=None)`:
  `config.encoder.latent_dim * n_cams`.
