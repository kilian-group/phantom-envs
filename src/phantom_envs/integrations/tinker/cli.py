"""Settings of scripts/tinker/train.py and scripts/tinker/eval.py, read as SkyRL's scripts read theirs.

Both scripts take YAML files with `--config` (repeat to stack them; later files win) and dot-notation
`key=value` overrides on top, merged with OmegaConf:

    python scripts/tinker/train.py --config scripts/tinker/configs/models/Inkling-2-Small.yaml \\
        data_dir=data/envs/pw env.max_turns=10

The episode settings (`TerminalEnvConfig`) sit under `env`, as SkyRL's sit under
`environment.skyrl_gym.phantom_terminal`. One model config serves both scripts: each reads the
settings it has and ignores the other's, as SkyRL's eval ignores trainer settings. A setting that
neither script has is an error.
"""

import os
from typing import Any

import chz
from omegaconf import OmegaConf
from tinker.types import LossFnType
from tinker_cookbook import cli_utils

from phantom_envs.integrations.tinker.harness import TerminalEnvConfig


@chz.chz
class TrainConfig:
    """Settings of a training run."""

    data_dir: str = chz.field(doc="Data dir with tasks/{train,validation}.jsonl and corpus.jsonl.")
    env: TerminalEnvConfig = chz.field(doc="Episode settings, also the model's chat format.")
    lora_rank: int = chz.field(default=32, doc="LoRA rank.")
    # The cookbook's Search-R1 recipe trains LoRA at this rate
    learning_rate: float = chz.field(default=4e-5, doc="Adam learning rate.")
    loss_fn: LossFnType = chz.field(default="importance_sampling", doc="Tinker loss function.")
    batch_size: int = chz.field(default=128, doc="Questions per step.")
    group_size: int = chz.field(default=8, doc="Episodes per question.")
    remove_constant_reward_groups: bool = chz.field(default=True, doc="Skip groups without advantage.")
    num_eval_tasks: int = chz.field(default=500, doc="Validation tasks per eval (0: no evals).")
    eval_every: int = chz.field(default=20, doc="Steps between evals.")
    save_every: int = chz.field(default=20, doc="Steps between checkpoints.")
    max_steps: int | None = chz.field(default=None, doc="Stop after this many steps (default: one epoch).")
    seed: int = chz.field(default=0, doc="Seed of the training tasks' order.")
    base_url: str | None = chz.field(default=None, doc="Tinker API URL (default: hosted Tinker).")
    log_path: str | None = chz.field(default=None, doc="Run dir (default: $RUN_BASE_DIR/tinker/<run>).")
    wandb_project: str | None = chz.field(default=None, doc="W&B project (default: no W&B).")
    wandb_name: str | None = chz.field(default=None, doc="W&B run name (default: the run name).")
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = chz.field(
        default="ask", doc="delete, resume, ask or raise."
    )


@chz.chz
class EvalConfig:
    """Settings of an eval."""

    data_dir: str = chz.field(doc="Data dir with tasks/validation.jsonl and corpus.jsonl.")
    out_dir: str = chz.field(doc="Eval output dir, e.g. out__eval=hp500__pi.")
    env: TerminalEnvConfig = chz.field(doc="Episode settings, also the model's chat format.")
    checkpoint: str | None = chz.field(
        default=None, doc="tinker:// sampler weights (default: the base model)."
    )
    name: str | None = chz.field(default=None, doc="Model dir name (default: a slug of env.model_name).")
    temperature: float = chz.field(default=1.0, doc="Sampling temperature.")
    limit: int | None = chz.field(default=None, doc="Only the first N questions.")
    concurrency: int = chz.field(default=64, doc="Episodes run at once.")
    budget_usd: float | None = chz.field(default=None, doc="Stop starting episodes at this estimated spend.")
    base_url: str | None = chz.field(default=None, doc="Tinker API URL (default: hosted Tinker).")
    # USD per 1M tokens; the defaults are Inkling 2 Small's
    price_prefill: float = chz.field(default=0.30, doc="Price of prompt tokens not in the cache.")
    price_cached_prefill: float = chz.field(default=0.03, doc="Price of prompt tokens read from the cache.")
    price_sample: float = chz.field(default=1.00, doc="Price of sampled tokens.")


SCRIPT_CONFIGS: tuple[type, ...] = (TrainConfig, EvalConfig)


def flatten(settings: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Return nested settings with dot-notation keys: `{"env": {"max_turns": 5}}` as `env.max_turns`."""
    flat = {}
    for key, value in settings.items():
        if isinstance(value, dict):
            flat |= flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def load_config[Config: (TrainConfig, EvalConfig)](
    config_cls: type[Config], config_paths: list[str], overrides: list[str]
) -> Config:
    """Merge YAML files and command-line overrides into one script's settings.

    Args:
        config_cls: `TrainConfig` or `EvalConfig`.
        config_paths: YAML files, in order; later files win.
        overrides: Dot-notation `key=value` settings, which win over the files.

    Returns:
        An instance of `config_cls`.

    Raises:
        ValueError: If a setting belongs to neither script.
        chz.blueprint.EntrypointException: If a setting under `env` doesn't exist, or a value doesn't
            parse as its field's type.
    """
    merged = OmegaConf.merge(*(OmegaConf.load(path) for path in config_paths), OmegaConf.from_cli(overrides))
    # The sandbox image comes from the environment unless a setting names it
    settings = {"env.rootfs": os.environ.get("SANDBOX_ROOTFS", "")} | flatten(OmegaConf.to_container(merged))
    known = set().union(*(cls.__chz_fields__ for cls in SCRIPT_CONFIGS))
    unknown = sorted(key for key in settings if key.split(".")[0] not in known)
    if unknown:
        raise ValueError(f"Unknown settings {unknown}; the Tinker scripts take {sorted(known)}")
    # chz checks the names under env, and turns the text into each field's type, as on its command line
    ours = {key: value for key, value in settings.items() if key.split(".")[0] in config_cls.__chz_fields__}
    return chz.Blueprint(config_cls).make_from_argv([f"{key}={value}" for key, value in ours.items()])
