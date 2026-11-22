"""Tests for the PhantomEnvs generator: templates, splits, task ids, relations, queries and generation."""

from dataclasses import replace

import pytest

from phantom_envs._types import SplitConfig, Task
from phantom_envs.environment import PRESETS, content_hash, generate
from phantom_envs.generator.query import answer_query
from phantom_envs.generator.questions import samplable_templates
from phantom_envs.generator.relations import RULES, Database, children_of_related_parents
from phantom_envs.generator.templates import generate_templates
from phantom_envs.generator.vocab import ALL_RELATIONS
from phantom_envs.splits import split_tasks
from phantom_envs.tasks import load_tasks, task_id


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


@pytest.fixture
def family() -> Database:
    """Dan and Eve's children Ann and Bob, their spouses Hal and Fay, and their children Ivy and Gus."""
    db = Database()
    genders = {"female": ["Ann", "Eve", "Fay", "Ivy"], "male": ["Bob", "Dan", "Gus", "Hal"]}
    db.add(*[("type", name, "person") for names in genders.values() for name in names])
    db.add(*[("gender", name, gender) for gender, names in genders.items() for name in names])
    parents = {"Ann": ["Dan", "Eve"], "Bob": ["Dan", "Eve"], "Gus": ["Bob", "Fay"], "Ivy": ["Ann", "Hal"]}
    db.add(*[("parent", child, parent) for child, both in parents.items() for parent in both])
    db.add(
        ("friend_", "Gus", "Ivy"),
        ("hobby", "Gus", "chess"),
        ("hobby", "Ivy", "chess"),
        ("hobby", "Bob", "golf"),
    )
    return db


class TestRelations:
    def test_repeats_are_kept(self, family):
        # Once per shared parent, per child, and per way the parents are siblings
        assert family["sibling"].of("Ann") == ["Bob", "Bob"]
        assert family["married"].of("Dan") == ["Eve", "Eve"]
        assert family["cousin"].of("Gus") == ["Ivy", "Ivy"]

    def test_lookups_from_both_sides(self, family):
        assert family["mother"].of("Gus") == ["Fay"]
        assert family["mother"].inverse_of("Eve") == ["Ann", "Bob"]
        assert family["grandfather"].of("Ivy") == ["Dan"]
        assert family["sister_in_law"].of("Fay") == ["Ann", "Ann"]
        assert family["friend"].of("Ivy") == ["Gus"] and family["friend"].of("Gus") == ["Ivy"]

    def test_new_facts_update_relations(self, family):
        assert family["daughter"].of("Bob") == []
        family.add(("type", "Joy", "person"), ("gender", "Joy", "female"), ("parent", "Joy", "Bob"))
        assert family["daughter"].of("Bob") == ["Joy"]
        assert family.names[-1] == "Joy"

    def test_unknown_relation(self, family):
        with pytest.raises(KeyError):
            family["stepmother"]

    def test_every_relation_has_a_rule(self):
        assert set(ALL_RELATIONS) - {"parent"} <= set(RULES)

    def test_cousin_shortcut_matches_plain_loop(self):
        # Four generations: Gil and Gia's children Pat and Quin, their children Al and Bea, and
        # their daughters Xia and Yue, who are second cousins (twice: Pat and Quin share two parents)
        db = Database()
        parents = {"Pat": ["Gil", "Gia"], "Quin": ["Gil", "Gia"], "Al": ["Pat"], "Bea": ["Quin"]}
        parents |= {"Xia": ["Al"], "Yue": ["Bea"]}
        db.add(*[("parent", child, parent) for child, both in parents.items() for parent in both])
        db.add(("gender", "Xia", "female"), ("gender", "Yue", "female"))
        assert db["female_second_cousin"].of("Xia") == ["Yue", "Yue"]

        parent = db["parent"]
        for related in (db["sibling"], db["cousin"]):
            plain = [(x, y) for x, a in parent for y, b in parent for _ in range(related.of(a).count(b))]
            assert children_of_related_parents(parent, related) == plain


class TestQuery:
    def test_chain(self, family):
        # The mother of the people whose hobby is chess, whichever way the atoms are listed
        query = ["mother(Y_2, Y_1)", 'hobby(Y_2, "chess")']
        assert answer_query(family, query, "Y_1") == ["Ann", "Fay"]
        assert answer_query(family, query[::-1], "Y_1") == ["Ann", "Fay"]

    def test_name_anchor_and_filter(self, family):
        query = ['hobby(Y_1, "golf")', 'child("Dan", Y_1)']
        assert answer_query(family, query, "Y_1") == ["Bob"]
        assert answer_query(family, ["hobby(Y_1, Y_2)", 'cousin("Ivy", Y_1)'], "Y_2") == ["chess"]

    def test_count(self, family):
        # Dan's children have 1 child each; Dan has 2 distinct children
        count = "aggregate_all(count, distinct(child(Y_1, Y_2)), Count_2)"
        assert answer_query(family, [count, 'father(Y_1, "Dan")'], "Count_2") == ["1"]
        assert answer_query(
            family, ['aggregate_all(count, distinct(child("Dan", Y_2)), Count_2)'], "Count_2"
        ) == ["2"]
        assert answer_query(family, [count, 'hobby(Y_1, "chess")'], "Count_2") == ["0"]

    def test_unanswerable(self, family):
        with pytest.raises(ValueError):
            answer_query(family, ["D_1 @< D_2"], "D_1")
        with pytest.raises(ValueError):
            answer_query(family, ['mother("Gus", Y_1)'], "Y_2")


class TestGenerate:
    # Content hashes of environments generated by PhantomWiki's SWI-Prolog rules, which the rules in
    # relations.py reproduce pair for pair. A changed hash means a seed no longer gives the same data.
    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            (
                replace(PRESETS["pwcomp"], size=20, questions_per_template=2),
                "1551a36716c636b23f0f365dff43b26bf9adb4b2c34b00fcb6fedbca1b4d4c30",
            ),
            (
                replace(PRESETS["pwcons"], size=50, questions_per_template=5),
                "d3f90be8e97dbff05a3bafb7025c7b7b3038ba4716e72bd7fb60304d72c9bb4e",
            ),
            (
                replace(
                    PRESETS["pw"],
                    size=100,
                    questions_per_template=5,
                    hard_mode=True,
                    kinds=("base", "count"),
                    sampler="backward",
                ),
                "4fdc85b7e7fe539ef55e037ff7c1adae96868bf9bd892874e247be6099446410",
            ),
            (
                replace(
                    PRESETS["pwcons"],
                    size=100,
                    questions_per_template=5,
                    hard_mode=True,
                    kinds=("base", "count", "comparison"),
                ),
                "768c4951cfd5a25f2c6dbdb48ba3a47ac2dccb22101836273bc0f1e76fe16722",
            ),
        ],
    )
    def test_content_hash(self, tmp_path, config, expected):
        env = generate(tmp_path, config)
        assert env.manifest["content_hash"] == content_hash(tmp_path) == expected

    def test_task_kinds(self, tmp_path):
        generate(tmp_path, replace(PRESETS["pwcomp"], size=20, questions_per_template=2))
        tasks = load_tasks(tmp_path, "train")
        assert {t.metadata["kind"] for t in tasks} == {"base", "comparison"}
        assert {t.data_source for t in tasks} == {"pwcomp"}

    def test_refuses_existing_env(self, tmp_path):
        cfg = replace(PRESETS["pw"], size=10, questions_per_template=1)
        generate(tmp_path, cfg)
        with pytest.raises(FileExistsError):
            generate(tmp_path, cfg)
