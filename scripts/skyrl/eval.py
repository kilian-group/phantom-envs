#!/usr/bin/env -S uv run python
"""Evaluate a model as a search agent with SkyRL, with no training and no policy worker.

Runs SkyRL's stock ``EvalOnlyEntrypoint``: one pass of ``skyrl.train.evaluate.evaluate`` over
``data.val_data``. vLLM loads the model straight from ``trainer.policy.model.path``, so any
HuggingFace-format dir works: a base model or a ``<run>/exports/global_step_<N>/policy`` export.

Stack ``scripts/skyrl/configs/eval.yaml`` last for the paper's eval setup: one GPU, 20 turns and a
32768-token input. Without it the training defaults (two GPUs, 10 turns) apply, so the effective
turns and input length are printed at startup. Trajectories are dumped under

    <out_dir>/model=<model_slug>/dumped_evals/eval_only/

where an export path ``.../<run_name>/exports/global_step_<N>/policy`` collapses to the slug
``<run_name>__step<N>``. Pass ``--name`` to choose the directory name instead. The tracker is
always ``console`` (no W&B).

Usage (the model yaml carries the model family's sampling params):
    # Base model on a PhantomWiki universe
    uv run python scripts/skyrl/eval.py --config scripts/skyrl/configs/base.yaml \
        --config scripts/skyrl/configs/eval.yaml \
        --out_dir out__eval=pw2 \
        data.val_data=[data/pw/depth_16_size_1000_seed_2/validation.parquet]

    # Trained Llama checkpoint on a multi-hop benchmark
    uv run python scripts/skyrl/eval.py --config scripts/skyrl/configs/base.yaml \
        --config scripts/skyrl/configs/models/Llama-3.2-3B-Instruct.yaml \
        --config scripts/skyrl/configs/eval.yaml \
        --out_dir out__eval=hp500 \
        trainer.policy.model.path=/path/to/<run>/exports/global_step_160/policy \
        data.val_data=[data/hp500/validation.parquet]

To evaluate a checkpoint without an HF export, run scripts/skyrl/train.py with
``trainer.epochs=0 trainer.eval_before_train=true trainer.resume_mode=latest
trainer.ckpt_interval=-1 trainer.hf_save_interval=-1`` instead.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Any

import ray
from common import load_config, resolve_model_path, set_max_model_len
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg

from phantom_envs.runs import compact_model_slug


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: SkyRLTrainConfig) -> dict[str, Any]:
    """Ray task that registers the env and runs SkyRL's stock eval-only experiment."""
    from phantom_envs.integrations.skyrl import register

    register()
    exp = EvalOnlyEntrypoint(cfg)
    # As in SkyRL's own eval entrypoint: the client is built outside asyncio.run
    return asyncio.run(exp.run(exp.get_inference_client()))


def main(args: argparse.Namespace) -> None:
    """Build the eval config and run one eval pass.

    Args:
        args: Parsed CLI arguments: ``config``, ``out_dir``, ``name`` and the SkyRL ``overrides``.
    """
    cfg, _, _ = load_config(args.config, args.overrides)
    print(
        f"[skyrl] Eval with max_turns={cfg.generator.max_turns}, "
        f"max_input_length={cfg.generator.max_input_length} (eval.yaml sets the paper's 20 and 32768)"
    )

    resolve_model_path(cfg)
    set_max_model_len(cfg)

    # Eval needs no train data (the Search-R1 train parquet is ~170k rows).
    cfg.data.val_data = [str(Path(p).resolve()) for p in cfg.data.val_data]
    cfg.data.train_data = []

    name = args.name or f"model={compact_model_slug(cfg.trainer.policy.model.path)}"
    export_path = (Path(args.out_dir) / name).resolve()
    export_path.mkdir(parents=True, exist_ok=True)
    cfg.trainer.export_path = str(export_path)
    cfg.trainer.ckpt_path = str(export_path / "ckpts")  # unused, but BasePPOExp expects it
    cfg.trainer.run_name = export_path.name

    # Eval-only: console logger, no checkpoints, and eval_interval > 0 so the eval dataset loads.
    cfg.trainer.logger = "console"
    cfg.trainer.ckpt_interval = -1
    cfg.trainer.hf_save_interval = -1
    if cfg.trainer.eval_interval <= 0:
        cfg.trainer.eval_interval = 1

    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    print("=" * 80)
    print(f"Eval results written to: {cfg.trainer.export_path}")
    print(f"Metrics: {metrics}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval-only run of a search agent via SkyRL.")
    parser.add_argument(
        "--config",
        required=True,
        action="append",
        metavar="YAML",
        help="Yaml config; repeat to stack overrides, later files win "
        "(e.g. scripts/skyrl/configs/base.yaml, models/<model>.yaml, then eval.yaml).",
    )
    parser.add_argument("--out_dir", required=True, help="Eval output root (e.g. out__eval=hp500).")
    parser.add_argument(
        "--name",
        default=None,
        help="Output dir name under --out_dir (default: model=<slug of trainer.policy.model.path>).",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    main(args)
