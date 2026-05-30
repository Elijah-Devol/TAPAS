# encoder/__init__.py

## Purpose
The encoder package's public registry: maps user-facing encoder names to the
`RepresentationLearner` subclass that implements them, and maps config types
back to encoder classes (used by `ObservationEncoder` to discover the right
image encoder).

## Contents
- `encoder_switch: dict[str, type[RepresentationLearner]]` - the central
  string-to-class map. Keys are the values used in OmegaConf configs and on
  the CLI: `"transporter"`, `"bvae"`, `"monet"`, `"keypoints"`,
  `"keypoints_gt"`, `"cnn"`, `"cnnd"`, `"vit_extractor"`, `"vit_keypoints"`.
- `encoder_names = list(encoder_switch.keys())` - convenience.
- `encoder_config_map: dict[type, type]` - reverse lookup from a config class
  (e.g. `KeypointsPredictorConfig`) to the encoder class. Only the encoders
  with dedicated config classes are registered.
- `get_image_encoder_class(encoder_config)` - resolves a config object to its
  encoder class (`None` if no config is given). Raises if the config type is
  unknown.
