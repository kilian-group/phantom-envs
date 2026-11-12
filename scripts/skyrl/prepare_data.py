#!/usr/bin/env -S uv run python
"""Write SkyRL parquets for a data dir: ``tasks/<split>.jsonl`` -> ``<split>.parquet``.

Works for every data dir: PhantomEnvs environments (``phantom_envs generate`` or
reproduce_paper/scripts/generate_paper_data.py) and the benchmarks (``scripts/benchmarks/``). Each task
becomes one row in the SkyRL-Gym conversation format, run by the ``phantom_search`` env
(``phantom_envs.integrations.skyrl``).

Usage:
    uv run python scripts/skyrl/prepare_data.py --data_dir data/pw/depth_16_size_1000_seed_1_50k
"""

import argparse
from pathlib import Path

import pandas as pd

from phantom_envs._types import Task
from phantom_envs.prompts import build_messages
from phantom_envs.tasks import SPLITS, load_tasks, task_path

# The parquet's ``ability`` column per data source (informational; nothing reads it)
ABILITIES = {
    "pw": "phantom_wiki",
    "pwcons": "phantom_wiki_constraints",
    "pwcomp": "phantom_wiki_comparison",
    "hp500": "hotpot_qa",
    "2wiki500": "2wiki_multihop",
    "cofca500": "cofca",
    "msq500": "musique",
    "synthrm500": "synthworlds_rm",
    "synthsm500": "synthworlds_sm",
    "frames": "frames",
    "browsecomp_plus": "browsecomp_plus",
}
SEARCHR1_PREFIX = "searchR1_"


def is_phantom_task(task: Task) -> bool:
    """Whether a task comes from a PhantomEnvs environment (its metadata holds the Prolog query)."""
    return "prolog" in task.metadata


def ability(task: Task) -> str:
    """The ``ability`` label of a task's data source.

    Args:
        task: The task.

    Returns:
        The label; Search-R1 rows are "fact-reasoning" and custom PhantomEnvs names "phantom_wiki".

    Raises:
        ValueError: For an unknown data source.
    """
    if task.data_source.startswith(SEARCHR1_PREFIX):
        return "fact-reasoning"
    if task.data_source in ABILITIES:
        return ABILITIES[task.data_source]
    if is_phantom_task(task):
        return "phantom_wiki"
    raise ValueError(f"Unknown data source {task.data_source!r}")


def row_metadata(task: Task) -> dict | None:
    """The parquet's ``metadata`` of a task.

    Args:
        task: The task.

    Returns:
        For PhantomEnvs, the id, difficulty, template (as ``type``) and whether it is a "How many"
        question; for benchmarks, their metadata as is (None when empty, as for Search-R1's NQ rows).
    """
    if is_phantom_task(task):
        m = task.metadata
        return {
            "id": task.id,
            "difficulty": m["difficulty"],
            "type": m["template"],
            "is_aggregation": m["kind"] == "count",
        }
    return task.metadata or None


def skyrl_row(task: Task, split: str, idx: int) -> dict:
    """One SkyRL-Gym parquet row.

    Args:
        task: The task.
        split: ``train`` or ``validation``.
        idx: The task's position in its split.

    Returns:
        The row: data_source, prompt, ability, env_class, reward_spec, extra_info and metadata.
    """
    ground_truth = {"target": task.answers}
    reward_spec = {"ground_truth": ground_truth}
    if task.data_source.startswith(SEARCHR1_PREFIX):
        # Search-R1's own rows carry a reward style
        reward_spec["style"] = "rule"
    # The env reads its episode arguments from tools_kwargs
    create_kwargs = {"ground_truth": ground_truth, "question": task.question, "data_source": task.data_source}
    extra_info = {
        "index": idx,
        "need_tools_kwargs": True,
        "question": task.question,
        "split": split,
        "tools_kwargs": {"search": {"create_kwargs": create_kwargs}},
    }
    return {
        "data_source": task.data_source,
        "prompt": build_messages(task.question),
        "ability": ability(task),
        "env_class": "phantom_search",
        "reward_spec": reward_spec,
        "extra_info": extra_info,
        "metadata": row_metadata(task),
    }


def main(args: argparse.Namespace) -> None:
    """Write one parquet per split that has tasks."""
    for split in SPLITS:
        if not task_path(args.data_dir, split).exists():
            continue
        rows = [skyrl_row(t, split, i) for i, t in enumerate(load_tasks(args.data_dir, split))]
        path = Path(args.data_dir) / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write SkyRL parquets for a data dir's tasks.")
    parser.add_argument("--data_dir", required=True, help="Data dir with tasks/<split>.jsonl.")
    args = parser.parse_args()
    main(args)
