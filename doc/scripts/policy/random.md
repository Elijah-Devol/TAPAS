# policy/random.py - RandomPolicy

## Purpose
A trivial policy that emits actions drawn uniformly from `[0, 1)` of the
configured dimensionality. Useful for sanity-checking the data-collection
pipeline (does the env step at all? are observations being recorded?) and
as a smoke test for the rest of the stack.

## Components
- `RandomPolicyConfig(PolicyConfig)`: adds `action_dim`.
- `RandomPolicy.predict(obs)`: `np.array([random.random() for _ in
  range(action_dim)]), {}`.
- `from_disk(...)`: no-op.
