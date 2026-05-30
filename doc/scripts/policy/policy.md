# policy/policy.py - Policy base class

## Purpose
Abstract base for every policy that participates in TAPAS' train/eval loops.
Subclasses provide `predict`, `update_params`, `evaluate`,
`reset_episode`, and optionally `initialize_parameters_via_dataset`.

## Key components
- `PolicyConfig` - base config carrying only `suffix` (used in checkpoint
  naming). Policy-specific configs subclass this.
- `Policy(nn.Module)` - abstract base class with:
  - `__init__(config, skip_module_init=False, encoder_checkpoint=None)`:
    optionally skips `nn.Module.__init__` (used by `DiskReadEncoderPolicy`
    when its encoder is assigned before init). Loads an encoder checkpoint
    if a path is given.
  - `from_disk` / `to_disk` - standard `torch.load` / `torch.save` of the
    state dict.
  - `_load_encoder_checkpoint(ckpt)`: loads the encoder; for the GT
    keypoint encoder uses `force_read=True` to overwrite the encoder's
    runtime-only attributes; freezes the encoder and switches it to eval.
  - `initialize_parameters_via_dataset(replay_memory, cameras)`: default
    no-op; overridden by GMM/encoder policies to pick reference points
    or fit normalisation statistics.
  - `reset_episode(env=None)`: hook for stateful policies.
  - `update_params` / `evaluate` / `predict`: abstract - subclasses must
    implement them. `predict` returns `(action: np.ndarray, info: dict)`.

The companion `get_viz_encoder_callback` (commented out) used to expose
the encoder for live visualisation; that hook has been moved to
`ObservationEncoder.get_viz_encoder_callback`.
