# encoder/monet.py - MONet encoder

## Purpose
Wrapper around the MONet (Multi-Object Network) model from
`encoder.models.monet.monet`. Unsupervised slot-based image decomposition;
each slot encodes a different object/region. The concatenated slot latents
form the embedding.

## Key class: `Monet(RepresentationLearner)`
- `sample_type = SampleTypes.CAM_SINGLE`.
- Loss: combination of:
  - `loss_E`: VAE encoder loss inside each slot.
  - `loss_D = -logsumexp(b).sum() / batch_size`: per-slot reconstruction
    likelihood.
  - `loss_mask = KL(softmax(m_tilde_logits) || m)`: mask consistency between
    the attention masks and the model-predicted masks.
- Hyperparameters `beta` (encoder KLD weight, default 5) and `gamma` (mask KLD
  weight, default 0.5).
- Optimiser: Adam.

### Methods
- `update_params`: standard backprop step.
- `encode_single_camera`: downsample to 128x128, return `z_mu` from
  `model.encode`.
- `reconstruct` / `reconstruct_w_extras`: return either the masked
  reconstruction or the full set of per-slot outputs.
- `get_latent_dim(config, n_cams, image_dim)`: sums the slot-specific latent
  dims (`config.encoder.latent_dims[v]` for each slot value) over slots and
  cameras.

## Helpers
- `initialize_weights(m, init_type, init_gain)`: per-layer initialiser (kept
  for reference; currently called only in commented code).
