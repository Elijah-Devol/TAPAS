# policy/motion_planner.py - MotionPlannerPolicy

## Purpose
A "scripted" policy that drives the robot by feeding a series of high-level
goals through a motion planner (`tapas_gmm.policy.models.motion_planner`,
backed by `mplib`/Pinocchio). Used only inside ManiSkill2 environments to
generate clean demonstrations programmatically.

## How it works
- Each ManiSkill task exposes a `get_solution_sequence()` returning a list
  of `Action(action_type, goal, ...)` items where `action_type` is one of
  `MOVE_TO`, `OPEN_GRIPPER`, `CLOSE_GRIPPER`, `NOOP`. The policy executes
  these goals one by one.
- For `MOVE_TO` goals, the policy plans a joint-position trajectory via
  `MotionPlanner.plan_to_goal` and feeds the joint positions through a
  `PDJointPos2EETranslator` (from `utils.maniskill_replay`) to convert
  them to the env's `pd_ee_delta_pose` action mode online (the translator
  maintains a twin env to stay synchronised).
- For gripper / NOOP goals it issues zero-EE-motion plans with the
  appropriate gripper command (`zero_movement` + open/close gripper) for
  a few timesteps.
- Gripper state is binarised via `_binary_gripper_state` (threshold
  `0.85` of the normalised state, default finger width 0.04m).

## Key methods
- `predict(obs)`: if the current plan is empty, pop the next goal from
  `goal_list` and re-plan; otherwise step through the current plan. If
  no goals remain, return a no-op action while waiting for the env to
  settle.
- `_make_plan(obs, goal)`: routes to the correct sub-planner per action
  type.
- `_make_gripper_open_plan` / `_make_gripper_close_plan` /
  `_make_noop_plan`: zero-motion plans of length `repeat=10` / `repeat=5`.
- `reset_episode(env)`: clears the current goal list and plan.

## Caveats
- Requires `mplib` (Pinocchio) - imported via the lazy `import_policy`.
- The translator uses a twin environment in joint-position mode; the seed
  is synchronised with the live env via `env.get_seed()`.
- `from_disk` is a no-op (no learnt parameters).
