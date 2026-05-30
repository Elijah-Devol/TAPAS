# reconstruct_actions.py

## Purpose
Regenerates the per-step "action" tensor of an existing dataset from the
recorded end-effector observations. Useful when an old dataset was recorded
without per-step action labels, when the action representation needs to be
reinterpreted, or when the policy's expected action space changed.

## What it does
1. Loads the `SceneDataset` and wraps it in a `BCDataset` configured for raw
   loading.
2. Iterates over all trajectories; for each one, calls
   `tapas_gmm.utils.franka.reconstruct_actions_obs(batch.squeeze(0))` which
   derives an action sequence from the recorded EE/gripper observations.
3. Writes the new actions back, step-by-step, with
   `bc_data.update_traj_attr(traj_no, step, "action", ...)`.

This is run as `python -m tapas_gmm.reconstruct_actions ...` (no
console-script entry).
