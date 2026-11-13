"""Driver-side helpers shared by scripts/skyrl/train.py and scripts/skyrl/eval.py.

Nothing here runs inside Ray workers (they only import the installed ``phantom_envs`` package),
so these helpers may live next to the scripts.
"""

import os
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig

from phantom_envs.integrations.skyrl import PhantomTrainConfig

# Trainer cadence fields auto-sized to 10% of total training steps when not set in the yaml or CLI.
AUTO_INTERVAL_FIELDS = ("hf_save_interval", "ckpt_interval", "eval_interval")
WARMUP_KEY = "trainer.policy.optimizer_config.num_warmup_steps"


def load_config(
    config_paths: list[str], overrides: list[str]
) -> tuple[PhantomTrainConfig, DictConfig, DictConfig]:
    """Merge yaml configs in order (later files win), then dot-notation CLI overrides on top.

    Returns:
        The typed config, plus the merged yaml and the CLI configs (to tell which keys were set
        explicitly).
    """
    yaml_cfg = OmegaConf.merge(*(OmegaConf.load(p) for p in config_paths))
    cli_cfg = OmegaConf.from_cli(overrides)
    cfg = PhantomTrainConfig.from_dict_config(OmegaConf.merge(yaml_cfg, cli_cfg))
    return cfg, yaml_cfg, cli_cfg


def is_set(key: str, *cfgs: DictConfig) -> bool:
    """Whether ``key`` is set in any of the given raw configs."""
    return any(OmegaConf.select(c, key) is not None for c in cfgs)


def set_max_model_len(cfg: PhantomTrainConfig) -> None:
    """Set vLLM's max_model_len to max_input_length + max_generate_length, capped at the model's
    max_position_embeddings (read with AutoConfig, so HF repo ids are capped as well as local dirs)."""
    desired = cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length
    try:
        model_config = AutoConfig.from_pretrained(cfg.trainer.policy.model.path, trust_remote_code=True)
        native_max = getattr(model_config, "max_position_embeddings", None)
    except OSError as e:  # e.g. a gated repo without a token: keep the uncapped length
        print(f"[skyrl] Could not read the model config, max_model_len stays uncapped: {e}")
        native_max = None
    if native_max is not None:
        desired = min(desired, int(native_max))
    cfg.generator.inference_engine.engine_init_kwargs["max_model_len"] = desired


def resolve_model_path(cfg: PhantomTrainConfig) -> None:
    """Make a local model dir absolute so Ray / vLLM workers don't mistake it for an HF repo id."""
    model_path = Path(cfg.trainer.policy.model.path)
    if model_path.is_dir():
        cfg.trainer.policy.model.path = str(model_path.resolve())


def compute_total_training_steps(cfg: PhantomTrainConfig) -> int:
    """Total training steps, as SkyRL's train dataloader counts them (``drop_last=True``), capped at
    ``trainer.max_training_steps``.

    Reads parquet metadata only. SkyRL also drops prompts longer than ``max_prompt_length``, so
    the true count can be slightly lower; that is fine for sizing checkpoint cadences.
    """
    total_rows = sum(pq.read_metadata(p).num_rows for p in cfg.data.train_data)
    total_steps = (total_rows // cfg.trainer.train_batch_size) * cfg.trainer.epochs
    if cfg.trainer.max_training_steps is not None:
        # Megatron's LR scheduler asserts that the warmup is shorter than the run
        total_steps = min(total_steps, cfg.trainer.max_training_steps)
    return total_steps


def set_auto_intervals(cfg: PhantomTrainConfig, yaml_cfg: DictConfig, cli_cfg: DictConfig) -> None:
    """Size checkpoint / eval cadences and LR warmup to 10% of total steps unless set explicitly."""
    if cfg.trainer.epochs <= 0 or not cfg.data.train_data:
        return
    total_steps = compute_total_training_steps(cfg)
    auto = max(1, total_steps // 10)
    for name in AUTO_INTERVAL_FIELDS:
        if not is_set(f"trainer.{name}", yaml_cfg, cli_cfg):
            setattr(cfg.trainer, name, auto)
            print(f"[skyrl] Auto-set trainer.{name}={auto} (10% of {total_steps} total training steps)")
    if not is_set(WARMUP_KEY, yaml_cfg, cli_cfg):
        cfg.trainer.policy.optimizer_config.num_warmup_steps = auto
        print(f"[skyrl] Auto-set {WARMUP_KEY}={auto} (10% of {total_steps} total training steps)")


def env_name(parquet_path: str) -> str:
    """Environment name for run paths, from the first row's data_source.

    ``searchR1_*`` maps to ``searchr1``, and ``pb`` (pwcons parquets made before the rename) to ``pwcons``.
    """
    data_source = pq.read_table(parquet_path, columns=["data_source"]).column(0)[0].as_py()
    if data_source.startswith("searchR1"):
        return "searchr1"
    return "pwcons" if data_source == "pb" else data_source


def get_run_name(cfg: PhantomTrainConfig, run_dir: str, env: str) -> str:
    """Self-describing run name, used as the W&B run name and the checkpoint sub-path.

    Format: ``<run_dir>/<env>/<model>/<estimator>/<user>/<MM-DD>__r=<reward>__max_t=<turns>__s=<seed>``.
    """
    user = os.environ.get("USER", "unknown")
    date_str = date.today().strftime("%m-%d")
    reward_func = cfg.environment.skyrl_gym.phantom_search.reward_func
    suffix = f"__r={reward_func}__max_t={cfg.generator.max_turns}__s={cfg.trainer.seed}"
    model = cfg.trainer.policy.model.path.strip("/")
    return f"{run_dir}/{env}/{model}/{cfg.trainer.algorithm.advantage_estimator}/{user}/{date_str}{suffix}"
