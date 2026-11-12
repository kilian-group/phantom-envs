"""Tasks of an environment: ``tasks/{train,validation}.jsonl`` in its data dir, one :class:`Task` per line."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from phantom_envs._types import Task

SPLITS = ("train", "validation")


def task_id(question: str, answers: list[str]) -> str:
    """A stable id for a question.

    Args:
        question: The question.
        answers: Its answers, in any order.

    Returns:
        16 hex digits of a SHA-1; equal questions with equal answers share an id.
    """
    return hashlib.sha1(json.dumps([question, sorted(answers)]).encode()).hexdigest()[:16]


def task_path(data_dir: str | Path, split: str) -> Path:
    """Path of a split's tasks in an environment's data dir."""
    return Path(data_dir) / "tasks" / f"{split}.jsonl"


def write_tasks(data_dir: str | Path, split: str, tasks: list[Task]) -> None:
    """Write a split's tasks, one JSON object per line.

    Args:
        data_dir: The environment's data dir.
        split: ``train`` or ``validation``.
        tasks: The tasks, in order.
    """
    path = task_path(data_dir, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for task in tasks:
            f.write(json.dumps(asdict(task)) + "\n")


def load_tasks(data_dir: str | Path, split: str) -> list[Task]:
    """Load the tasks of one split.

    Args:
        data_dir: The environment's data dir.
        split: ``train`` or ``validation``.

    Returns:
        The tasks, in file order.
    """
    with task_path(data_dir, split).open() as f:
        return [Task(**json.loads(line)) for line in f]
