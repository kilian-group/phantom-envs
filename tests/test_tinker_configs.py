"""Tests for phantom_envs.integrations.tinker.cli and the model configs in scripts/tinker/configs/models/."""

from pathlib import Path

import pytest

pytest.importorskip("tinker_cookbook")

from chz.blueprint import ExtraneousBlueprintArg  # noqa: E402

from phantom_envs.integrations.tinker.cli import EvalConfig, TrainConfig, load_config  # noqa: E402

MODEL_CONFIGS = sorted((Path(__file__).resolve().parents[1] / "scripts/tinker/configs/models").glob("*.yaml"))


def test_files_stack_and_overrides_win(tmp_path):
    first = tmp_path / "first.yaml"
    first.write_text("env:\n  model_name: a\n  max_tokens: 10\nlearning_rate: 1.0e-5\n")
    second = tmp_path / "second.yaml"
    second.write_text("env:\n  max_tokens: 20\n")
    overrides = ["data_dir=d", "learning_rate=3e-5", "env.max_turns=7"]
    config = load_config(TrainConfig, [str(first), str(second)], overrides)
    assert (config.env.model_name, config.env.max_tokens, config.env.max_turns) == ("a", 20, 7)
    assert config.learning_rate == 3e-5


def test_each_script_ignores_the_others_settings(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("lora_rank: 8\nprice_sample: 2.0\nenv:\n  model_name: a\n")
    assert load_config(TrainConfig, [str(path)], ["data_dir=d"]).lora_rank == 8
    assert load_config(EvalConfig, [str(path)], ["data_dir=d", "out_dir=o"]).price_sample == 2.0


def test_unknown_setting_is_an_error():
    with pytest.raises(ValueError, match="Unknown settings"):
        load_config(TrainConfig, [], ["data_dir=d", "env.model_name=a", "learning_rte=1e-5"])
    with pytest.raises(ExtraneousBlueprintArg, match="max_turn"):
        load_config(TrainConfig, [], ["data_dir=d", "env.model_name=a", "env.max_turn=3"])


def test_rootfs_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("SANDBOX_ROOTFS", "/images/a")
    settings = ["data_dir=d", "env.model_name=a"]
    assert load_config(TrainConfig, [], settings).env.rootfs == "/images/a"
    assert load_config(TrainConfig, [], [*settings, "env.rootfs=/images/b"]).env.rootfs == "/images/b"


@pytest.mark.parametrize("config", MODEL_CONFIGS, ids=lambda p: p.stem)
def test_model_configs_load_for_both_scripts(config):
    train = load_config(TrainConfig, [str(config)], ["data_dir=d"])
    evals = load_config(EvalConfig, [str(config)], ["data_dir=d", "out_dir=o"])
    assert train.env == evals.env and train.env.harness == "pi"
