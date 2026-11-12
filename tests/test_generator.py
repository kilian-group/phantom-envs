"""Tests for the PhantomEnvs generator: templates, splits, task ids and (with SWI-Prolog) generation."""

import shutil
from dataclasses import replace

import pytest

from phantom_envs._types import SplitConfig, Task
from phantom_envs.environment import PRESETS, content_hash, generate
from phantom_envs.generator.questions import samplable_templates
from phantom_envs.generator.templates import generate_templates
from phantom_envs.splits import split_tasks
from phantom_envs.tasks import load_tasks, task_id

needs_swipl = pytest.mark.skipif(shutil.which("swipl") is None, reason="needs SWI-Prolog")


class TestTemplates:
    # Template counts of the PhantomWiki fork at 839fa26 (without "How many" questions)
    @pytest.mark.parametrize(
        ("preset", "total", "samplable"), [("pw", 55, 55), ("pwcons", 187, 187), ("pwcomp", 327, 296)]
    )
    def test_preset_counts(self, preset, total, samplable):
        cfg = PRESETS[preset]
        assert len(generate_templates(cfg.max_hops, cfg.max_constraints, cfg.kinds)) == total
        assert len(samplable_templates(cfg)) == samplable

    def test_count_kind(self):
        cfg = PRESETS["pw"]
        templates = generate_templates(cfg.max_hops, cfg.max_constraints, ("base", "count"))
        assert len(templates) == 83
        assert {t.kind for t in templates if t.question[0] == "How many"} == {"count"}

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            generate_templates(2, 1, ("superlative",))


def make_task(i: int, hops: int) -> Task:
    return Task(str(i), "pw", f"q{i}", ["a"], {"hops": hops})


class TestSplits:
    def test_cap_and_ratio(self):
        tasks = [make_task(i, hops=1) for i in range(20)] + [make_task(100 + i, hops=2) for i in range(100)]
        train, val = split_tasks(tasks, SplitConfig(cells=("hops",), cap_factor=2))
        # The hops=2 cell is capped at 2 x 20; each cell puts 10% in validation
        assert (len(train), len(val)) == (18 + 36, 2 + 4)
        assert not {t.id for t in train} & {t.id for t in val}

    def test_deterministic(self):
        tasks = [make_task(i, hops=i % 3) for i in range(50)]
        config = SplitConfig(cells=("hops",))
        assert split_tasks(tasks, config) == split_tasks(tasks, config)

    def test_at_least_one_validation_task(self):
        train, val = split_tasks([make_task(i, hops=1) for i in range(3)], SplitConfig(cells=("hops",)))
        assert (len(train), len(val)) == (2, 1)


def test_task_id_ignores_answer_order():
    assert task_id("Who?", ["a", "b"]) == task_id("Who?", ["b", "a"]) != task_id("Who?", ["a"])


@needs_swipl
class TestGenerate:
    def test_deterministic_across_workers(self, tmp_path):
        cfg = replace(PRESETS["pwcomp"], size=20, questions_per_template=2)
        a = generate(tmp_path / "a", cfg, workers=1)
        b = generate(tmp_path / "b", cfg, workers=2)
        assert a.manifest["content_hash"] == b.manifest["content_hash"] == content_hash(tmp_path / "a")
        tasks = load_tasks(tmp_path / "a", "train")
        assert {t.metadata["kind"] for t in tasks} == {"base", "comparison"}
        assert {t.data_source for t in tasks} == {"pwcomp"}

    def test_refuses_existing_env(self, tmp_path):
        cfg = replace(PRESETS["pw"], size=10, questions_per_template=1)
        generate(tmp_path, cfg, workers=1)
        with pytest.raises(FileExistsError):
            generate(tmp_path, cfg, workers=1)
