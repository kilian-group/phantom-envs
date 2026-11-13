"""Tests for scripts/skyrl/common.py (run names, the env name lookup) and phantom_envs.runs (model slugs)."""

import sys
from pathlib import Path

import pytest

# scripts/skyrl/common.py imports SkyRL (the skyrl extra)
pytest.importorskip("skyrl")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "skyrl"))

from common import compute_total_training_steps, env_name, get_run_name, load_config  # noqa: E402

from phantom_envs.runs import compact_model_slug, slugify_path  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "scripts" / "skyrl" / "configs" / "base.yaml"


class TestModelSlug:
    def test_hf_repo_id(self):
        assert compact_model_slug("Qwen/Qwen2.5-3B-Instruct") == "Qwen--Qwen2.5-3B-Instruct"

    def test_export_collapses_to_step(self):
        path = "share/runs/pw/Qwen/Qwen2.5-3B-Instruct/grpo/u/05-01__r=f1__max_t=10__s=1/exports/global_step_220/policy"
        assert (
            compact_model_slug(path)
            == "share--runs--pw--Qwen--Qwen2.5-3B-Instruct--grpo--u--05-01__r=f1__max_t=10__s=1__step220"
        )

    def test_absolute_export_path(self):
        assert (
            compact_model_slug("/scratch/u/runs/pw/m/exports/global_step_3/policy")
            == "scratch--u--runs--pw--m__step3"
        )

    def test_huggingface_suffix(self):
        assert compact_model_slug("run/exports/global_step_5/policy/huggingface") == "run__step5"

    def test_parquet_suffix_dropped(self):
        assert slugify_path("data/pw/validation.parquet") == "data--pw--validation"


class TestRunName:
    def test_format(self, monkeypatch):
        monkeypatch.setenv("USER", "tester")
        cfg, _, _ = load_config([str(CONFIG)], ["trainer.seed=2"])
        name = get_run_name(cfg, "runs", "pw")
        assert name.startswith("runs/pw/Qwen/Qwen2.5-3B-Instruct/grpo/tester/")
        assert name.endswith("__r=f1__max_t=10__s=2")

    def test_absolute_model_path_has_no_double_slash(self):
        cfg, _, _ = load_config([str(CONFIG)], ["trainer.policy.model.path=/models/my-model"])
        assert "//" not in get_run_name(cfg, "runs", "pw")


class TestEnvName:
    def test_phantom_env(self, tmp_path):
        path = tmp_path / "train.parquet"
        pd.DataFrame({"data_source": ["pwcomp"]}).to_parquet(path)
        assert env_name(str(path)) == "pwcomp"

    def test_pb_is_pwcons(self, tmp_path):
        path = tmp_path / "train.parquet"
        pd.DataFrame({"data_source": ["pb"]}).to_parquet(path)
        assert env_name(str(path)) == "pwcons"

    def test_searchr1(self, tmp_path):
        path = tmp_path / "train.parquet"
        pd.DataFrame({"data_source": ["searchR1_nq"]}).to_parquet(path)
        assert env_name(str(path)) == "searchr1"


class TestTotalTrainingSteps:
    def make(self, tmp_path, *overrides: str):
        path = tmp_path / "train.parquet"
        pd.DataFrame({"data_source": ["pw"] * 1000}).to_parquet(path)
        cfg, _, _ = load_config([str(CONFIG)], [f"data.train_data=[{path}]", *overrides])
        return cfg

    def test_full_batches_per_epoch(self, tmp_path):
        cfg = self.make(tmp_path, "trainer.train_batch_size=256", "trainer.epochs=2")
        assert compute_total_training_steps(cfg) == 6

    def test_capped_at_max_training_steps(self, tmp_path):
        cfg = self.make(tmp_path, "trainer.train_batch_size=256", "trainer.max_training_steps=2")
        assert compute_total_training_steps(cfg) == 2
