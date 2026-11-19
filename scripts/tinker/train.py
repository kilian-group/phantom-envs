#!/usr/bin/env -S uv run python
"""Train the terminal agent with RL through the Tinker API, using tinker-cookbook's training loop.

The agent, inside the pi harness, answers each question with bash over the corpus files and writes
its answer to /app/output.md. Each question gets a group of episodes, rewarded by the F1 of the
answer file and centered on the group's mean. Tinker trains a LoRA adapter; the sandboxes and
harnesses run on this machine.

A model's settings live in scripts/tinker/configs/models/<model>.yaml. As in scripts/skyrl,
`--config` reads YAML files and `key=value` arguments override them (see
phantom_envs.integrations.tinker.cli).

Usage:
    # Inkling 2 Small with pi in the bwrap sandbox (.env holds TINKER_API_KEY and SANDBOX_ROOTFS)
    uv run --extra tinker --env-file .env python scripts/tinker/train.py \\
        --config scripts/tinker/configs/models/Inkling-2-Small.yaml data_dir=data/envs/pw-rewrite \\
        wandb_project=phantom-envs

    # A smoke test: 2 steps, no evals
    ... max_steps=2 num_eval_tasks=0

Gotchas:
- Groups whose episodes all get the same reward have no advantage, so no gradient, but Tinker would
  still bill their training tokens: `remove_constant_reward_groups` drops them before training.
- Every episode of a step runs its own pi at once, each with its own open files and sockets: raise
  `ulimit -n` first.
"""

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train

from phantom_envs.integrations.tinker.cli import TrainConfig, load_config
from phantom_envs.integrations.tinker.dataset import TerminalDatasetBuilder


async def main(cli: TrainConfig) -> None:
    """Build the cookbook's RL config from the CLI settings and train."""
    env = cli.env
    dataset_builder = TerminalDatasetBuilder(
        data_dir=cli.data_dir,
        env=env,
        batch_size=cli.batch_size,
        group_size=cli.group_size,
        seed=cli.seed,
        num_eval_tasks=cli.num_eval_tasks,
    )
    run_name = (
        f"terminal_{env.harness}__{Path(cli.data_dir).name}__{env.model_name.split('/')[-1]}"
        f"__r{cli.lora_rank}__lr{cli.learning_rate:g}__{datetime.now():%m-%d-%H%M}"
    )
    log_path = cli.log_path or str(Path(os.environ.get("RUN_BASE_DIR", ".")) / "tinker" / run_name)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    config = train.Config(
        model_name=env.model_name,
        recipe_name="phantom_envs_terminal",
        renderer_name=env.renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        lora_rank=cli.lora_rank,
        loss_fn=cli.loss_fn,
        max_tokens=env.max_tokens,
        remove_constant_reward_groups=cli.remove_constant_reward_groups,
        eval_every=cli.eval_every if cli.num_eval_tasks else 0,
        save_every=cli.save_every,
        max_steps=cli.max_steps,
        base_url=cli.base_url,
        log_path=log_path,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name or run_name,
    )
    await train.main(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the terminal agent through the Tinker API.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="YAML",
        help="Model config; repeat to stack configs, later files win "
        "(e.g. scripts/tinker/configs/models/Inkling-2-Small.yaml).",
    )
    args, overrides = parser.parse_known_args()
    asyncio.run(main(load_config(TrainConfig, args.config, overrides)))
