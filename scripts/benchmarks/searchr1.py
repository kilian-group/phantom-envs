#!/usr/bin/env -S uv run python
"""Prepare the Search-R1 training data (NQ + HotpotQA) from HuggingFace.

Downloads ``PeterJinGo/nq_hotpotqa_train`` and writes ``tasks/{train,validation}.jsonl`` (the HF
test split is the validation split). Each task's data source is ``searchR1_<source>`` and its
metadata is the row's own, passed on as is. The corpus and index come from
``scripts/benchmarks/download_searchr1_index.sh``; ``scripts/skyrl/prepare_data.py`` writes the parquets.

Usage:
    uv run python scripts/benchmarks/searchr1.py --data_dir data/searchr1
    uv run python scripts/benchmarks/searchr1.py --data_dir data/searchr1 --max_rows 1000
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from phantom_envs._types import Task
from phantom_envs.tasks import write_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HF_REPO_ID = "PeterJinGo/nq_hotpotqa_train"
# HF split -> our split
SPLITS = {"train": "train", "test": "validation"}


def to_python(x):
    """Turn numpy arrays and scalars (as read from parquet) into lists and Python scalars.

    Args:
        x: A value, possibly nested in dicts and lists.

    Returns:
        The value with JSON-serializable containers and scalars.
    """
    if isinstance(x, np.ndarray | list):
        return [to_python(v) for v in x]
    if isinstance(x, dict):
        return {k: to_python(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


def row_task(row: pd.Series, split: str, idx: int) -> Task:
    """Turn a Search-R1 row into a task.

    Args:
        row: A row of the HF parquet (question, data_source, reward_model, metadata, ...).
        split: Our split name, for ids of rows without one.
        idx: The row's position in its split.

    Returns:
        The task, answered by the row's ground-truth targets.
    """
    metadata = row.get("metadata")
    return Task(
        id=str(row["id"]) if "id" in row else f"{split}_{idx}",
        data_source=f"searchR1_{row['data_source']}",
        question=row["question"],
        answers=to_python(row["reward_model"]["ground_truth"]["target"]),
        metadata=to_python(metadata) if metadata is not None else {},
    )


def main(args: argparse.Namespace) -> None:
    """Download both splits and write their tasks."""
    data_dir = Path(args.data_dir).expanduser()
    for hf_split, split in SPLITS.items():
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info(f"Downloading {hf_split}.parquet from {HF_REPO_ID} ...")
            path = hf_hub_download(HF_REPO_ID, f"{hf_split}.parquet", repo_type="dataset", local_dir=tmp_dir)
            df = pd.read_parquet(path)
        if args.max_rows is not None:
            # A small validation split for quick runs
            df = df.iloc[: args.max_rows if split == "train" else min(args.max_rows, 10)]
        tasks = [row_task(row, split, idx) for idx, row in df.iterrows()]
        write_tasks(data_dir, split, tasks)
        logger.info(f"Wrote {len(tasks)} {split} tasks to {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the Search-R1 training data (NQ + HotpotQA).")
    parser.add_argument("--data_dir", default="data/searchr1", help="Output dir (default: data/searchr1).")
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Cap on training rows (validation: min(max_rows, 10))."
    )
    args = parser.parse_args()
    main(args)
