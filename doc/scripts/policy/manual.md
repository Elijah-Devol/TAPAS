# policy/manual.py - ManualPolicy

## Purpose
A no-network teleoperation policy. Reads joystick/keyboard input through a
`KeyboardObserver` and emits the corrected action. Used by
`collect_data.py` to record human demonstrations.

## How it works
- Default action: zero translation, zero rotation, gripper held at `0.9`
  (open).
- If the keyboard observer reports joint corrections or a gripper update,
  `utils.human_feedback.correct_action` patches those values into the
  default action vector.
- The gripper open/close state is sticky: once changed via the keyboard,
  it persists across calls until `reset_episode` is called.

## Methods
- `predict(obs)`: returns the current corrected action and `{}`.
- `reset_episode(env)`: resets the gripper to the open state.
- `from_disk(...)`: no-op (nothing to load).
