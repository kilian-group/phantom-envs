#!/usr/bin/env -S uv run python
"""Train a search agent with SkyRL (GRPO, Megatron backend) on a PhantomEnvs or benchmark dataset.

Loads a yaml config, merges dot-notation CLI overrides on top, and runs SkyRL's stock
``BasePPOExp``. The run name, checkpoint path and export path are derived from the config:

    $RUN_BASE_DIR/<run_dir>/<env>/<model>/<estimator>/<user>/<MM-DD>__r=<reward>__max_t=<turns>__s=<seed>

where ``<env>`` comes from the train parquet's data_source. Checkpoint, HF-export and eval
intervals and the LR warmup default to 10% of the total training steps.

Usage:
    # Qwen2.5-3B-Instruct on PhantomWiki (Hops)
    uv run python scripts/skyrl/train.py --config scripts/skyrl/configs/base.yaml

    # Llama-3.2-3B-Instruct on Search-R1: stack model and env yamls (later files win)
    uv run python scripts/skyrl/train.py --config scripts/skyrl/configs/base.yaml \
        --config scripts/skyrl/configs/models/Llama-3.2-3B-Instruct.yaml \
        --config scripts/skyrl/configs/envs/searchr1.yaml

    # CLI overrides are applied before the run name is generated, so it reflects them
    uv run python scripts/skyrl/train.py --config scripts/skyrl/configs/base.yaml generator.max_turns=8

    # Explicit run name (disables auto-generation)
    uv run python scripts/skyrl/train.py --config scripts/skyrl/configs/base.yaml trainer.run_name=my-run
"""

import argparse
import os
from pathlib import Path

import ray
from common import (
    env_name,
    get_run_name,
    is_set,
    load_config,
    resolve_model_path,
    set_auto_intervals,
    set_max_model_len,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


@ray.remote(num_cpus=1)
def train_entrypoint(cfg: SkyRLTrainConfig) -> None:
    """Ray task that registers the env and runs SkyRL's stock PPO experiment."""
    from phantom_envs.integrations.skyrl import register

    register()
    BasePPOExp(cfg).run()


def main(args: argparse.Namespace) -> None:
    """Build the training config and launch SkyRL.

    Args:
        args: Parsed CLI arguments: ``config``, ``run_dir`` and the SkyRL ``overrides``.
    """
    cfg, yaml_cfg, cli_cfg = load_config(args.config, args.overrides)

    resolve_model_path(cfg)
    set_max_model_len(cfg)

    # Absolute data paths survive Ray workers running from a different working directory.
    cfg.data.train_data = [str(Path(p).resolve()) for p in cfg.data.train_data]
    cfg.data.val_data = [str(Path(p).resolve()) for p in cfg.data.val_data]

    set_auto_intervals(cfg, yaml_cfg, cli_cfg)

    if not is_set("trainer.run_name", cli_cfg):
        cfg.trainer.run_name = get_run_name(cfg, args.run_dir, env_name(cfg.data.train_data[0]))
    run_path = Path(os.environ.get("RUN_BASE_DIR", ".")) / cfg.trainer.run_name
    cfg.trainer.ckpt_path = str(run_path.resolve())
    cfg.trainer.export_path = str((run_path / "exports").resolve())
    Path(cfg.trainer.export_path).mkdir(parents=True, exist_ok=True)

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(train_entrypoint.remote(cfg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a search agent via SkyRL.")
    parser.add_argument(
        "--config",
        required=True,
        action="append",
        metavar="YAML",
        help="Yaml config; repeat to stack overrides, later files win "
        "(e.g. scripts/skyrl/configs/base.yaml, then models/<model>.yaml).",
    )
    parser.add_argument(
        "--run_dir",
        default="runs",
        help="Top-level prefix of the run name and checkpoint path under $RUN_BASE_DIR (default: runs).",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    main(args)
