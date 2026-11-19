"""Tests for phantom_envs.terminal.evals: resumable eval dumps and their aggregates."""

import json

import pytest

from phantom_envs._types import Task
from phantom_envs.terminal.evals import EpisodeDump, aggregate, episode_row

TASKS = [Task(id=f"t{i}", data_source="pw", question=f"Question {i}?", answers=["Bea Lee"]) for i in range(3)]
SCORES = {"em": 1.0, "subem": 1.0, "f1": 1.0}


def test_dump_resumes(tmp_path):
    dump = EpisodeDump(tmp_path, "m", "pw")
    dump.add(episode_row(TASKS[0], "Bea Lee", SCORES, "stop", num_turns=3, num_commands=2))
    reopened = EpisodeDump(tmp_path, "m", "pw")
    assert [row["id"] for row in reopened.rows] == ["t0"]
    assert [task.id for task in reopened.todo(TASKS)] == ["t1", "t2"]

    reopened.write_results({"f1": 1.0})
    assert json.loads((tmp_path / "model=m" / "aggregated_results.jsonl").read_text()) == {"f1": 1.0}


def test_aggregate():
    zeros = dict.fromkeys(SCORES, 0.0)
    rows = [
        episode_row(TASKS[0], "Bea Lee", SCORES, "stop", num_turns=3, num_commands=2),
        episode_row(TASKS[1], None, zeros, "timeout", num_turns=1, num_commands=0),
    ]
    results = aggregate(rows)
    assert (results["num_episodes"], results["f1"], results["answered"]) == (2, 0.5, 0.5)
    assert (results["num_turns"], results["num_commands"]) == (2.0, 1.0)
    assert results["stop/stop"] == results["stop/timeout"] == pytest.approx(0.5)
