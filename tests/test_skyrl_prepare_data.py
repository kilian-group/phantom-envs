"""Tests for scripts/skyrl/prepare_data.py: tasks to SkyRL parquet rows."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")  # the skyrl extra

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "skyrl"))

from prepare_data import skyrl_row  # noqa: E402

from phantom_envs._types import Task  # noqa: E402
from phantom_envs.prompts import build_messages  # noqa: E402

PW_TASK = Task(
    id="q_001",
    data_source="pw",
    question="Who is Alice's employer?",
    answers=["Acme Corp", "ACME"],
    metadata={
        "kind": "base",
        "hops": 1,
        "constraints": 0,
        "difficulty": {"hops": 1, "constraints": 0},
        "template": 7,
        "prolog": {"query": ['job("Alice", Y_1)'], "answer": "Y_1"},
    },
)
SEARCHR1_TASK = Task("train_0", "searchR1_nq", "What is the capital of France?", ["Paris"], {})
HOTPOT_TASK = Task("5a8b", "hp500", "Who is older?", ["Ann"], {"id": "5a8b", "type": "comparison"})


class TestPhantomRow:
    def test_columns(self):
        row = skyrl_row(PW_TASK, "train", 0)
        assert list(row) == [
            "data_source",
            "prompt",
            "ability",
            "env_class",
            "reward_spec",
            "extra_info",
            "metadata",
        ]
        assert (row["data_source"], row["ability"], row["env_class"]) == (
            "pw",
            "phantom_wiki",
            "phantom_search",
        )

    def test_prompt_matches_build_messages(self):
        assert skyrl_row(PW_TASK, "train", 0)["prompt"] == build_messages("Who is Alice's employer?")

    def test_ground_truth_and_tools_kwargs(self):
        row = skyrl_row(PW_TASK, "train", 0)
        assert row["reward_spec"] == {"ground_truth": {"target": ["Acme Corp", "ACME"]}}
        create_kwargs = row["extra_info"]["tools_kwargs"]["search"]["create_kwargs"]
        assert create_kwargs == {
            "ground_truth": {"target": ["Acme Corp", "ACME"]},
            "question": "Who is Alice's employer?",
            "data_source": "pw",
        }

    def test_extra_info(self):
        extra = skyrl_row(PW_TASK, "validation", 3)["extra_info"]
        assert (extra["index"], extra["split"], extra["need_tools_kwargs"]) == (3, "validation", True)

    def test_metadata(self):
        assert skyrl_row(PW_TASK, "train", 0)["metadata"] == {
            "id": "q_001",
            "difficulty": {"hops": 1, "constraints": 0},
            "type": 7,
            "is_aggregation": False,
        }

    def test_custom_env_name(self):
        task = Task(PW_TASK.id, "pw-seed2", PW_TASK.question, PW_TASK.answers, PW_TASK.metadata)
        assert skyrl_row(task, "train", 0)["ability"] == "phantom_wiki"


class TestBenchmarkRows:
    def test_searchr1(self):
        row = skyrl_row(SEARCHR1_TASK, "train", 0)
        assert row["reward_spec"] == {"ground_truth": {"target": ["Paris"]}, "style": "rule"}
        assert (row["ability"], row["metadata"]) == ("fact-reasoning", None)

    def test_metadata_passes_through(self):
        row = skyrl_row(HOTPOT_TASK, "validation", 0)
        assert (row["ability"], row["metadata"]) == ("hotpot_qa", {"id": "5a8b", "type": "comparison"})

    def test_unknown_data_source(self):
        with pytest.raises(ValueError):
            skyrl_row(Task("1", "mystery", "Q?", ["A"], {}), "validation", 0)
