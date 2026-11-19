"""Tinker integration: terminal episodes in a coding-agent harness, as tinker-cookbook RL environments.

tinker-cookbook's RL loop (`tinker_cookbook.rl.train`) samples and trains through the Tinker API,
which Thinking Machines hosts (LoRA). Rollouts run where the script runs and need no GPU: the bwrap
sandboxes and the harness (pi) are local processes.

- `harness`: the episode settings, and the env that hands pi's model calls to the cookbook's sampler.
- `dataset`: groups of episodes per question, batched into a dataset.
- `cli`: the settings of `scripts/tinker/train.py` and `scripts/tinker/eval.py`.

`scripts/tinker` has the training and evaluation entry points.
"""
