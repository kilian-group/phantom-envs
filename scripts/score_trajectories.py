#!/usr/bin/env -S uv run python
"""Recompute EM / F1 (and any supported metric) from dumped eval trajectories.

The eval pipeline only logs the single configured ``reward_func``, so this
script re-scores episodes post-hoc using the dataset class's scorers. It
auto-discovers every model subdirectory under ``--eval_dir`` that has a
trajectory dump at ``<subdir>/dumped_evals/eval_only/<data_source>.jsonl``,
computes means (± stderr) per model, prints a markdown table to stdout, and
writes a CSV to ``<eval_dir>/scores/<data_source>.csv``.

Stderr convention:
- EM / subem (binary): binomial stderr ``sqrt(p*(1-p)/n)``.
- F1 and num_searches (continuous): sample-std / sqrt(n).

Usage:
    uv run python scripts/score_trajectories.py \\
        --eval_dir out__eval=hp500 --data_source hp500

    uv run python scripts/score_trajectories.py \\
        --eval_dir out__eval=hp500 --data_source hp500 --scores em
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from phantom_envs.protocol import count_searches, extract_answer
from phantom_envs.scoring import METRICS, get_scorer, mean_stderr

SUPPORTED_DATA_SOURCES = [
    "hp500",
    "2wiki500",
    "msq500",
    "cofca500",
    "synthrm500",
    "synthsm500",
    "frames",
    "browsecomp_plus",
    "pw",
    "pwcons",
    "pwcomp",
    "pb",  # pwcons eval dumps made before the rename
]


def load_records(path: Path) -> list[dict]:
    """Load all valid JSON records from a jsonl file, skipping malformed lines."""
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def score_trajectory_file(path: Path, data_source: str, metrics: list[str]) -> dict[str, list]:
    """Compute per-episode metric values and turn counts for one trajectory file."""
    scorer = get_scorer(data_source)
    per_metric: dict[str, list[float]] = {m: [] for m in metrics}
    searches: list[int] = []
    n_no_answer = 0

    for record in load_records(path):
        gts = record["env_extras"]["reward_spec"]["ground_truth"]["target"]
        pred = extract_answer(record.get("output_response", ""))
        if pred is None:
            n_no_answer += 1
            pred = ""
        preds = scorer.parse_answer(pred)
        for metric in metrics:
            per_metric[metric].append(scorer.score(preds, gts, metric))
        searches.append(count_searches(record.get("output_response", "")))

    return {"metrics": per_metric, "searches": searches, "n_no_answer": n_no_answer}


def build_row(model: str, scored: dict, metrics: list[str]) -> tuple[dict, dict]:
    """Build one row each for the CSV (numeric) and stdout (formatted) tables."""
    csv_row: dict[str, float | int | str] = {
        "model": model,
        "n": len(scored["searches"]),
        "n_no_answer": scored["n_no_answer"],
    }
    stdout_row: dict[str, str | int] = {
        "model": model,
        "n": len(scored["searches"]),
        "n_no_answer": scored["n_no_answer"],
    }

    searches_mean, searches_se = mean_stderr(scored["searches"], binary=False)
    csv_row["num_searches"] = searches_mean
    csv_row["num_searches_stderr"] = searches_se
    stdout_row["num_searches"] = f"{searches_mean:.2f} ± {searches_se:.2f}"

    for metric in metrics:
        binary = metric in {"em", "subem"}
        mean, stderr = mean_stderr(scored["metrics"][metric], binary=binary)
        csv_row[metric] = mean
        csv_row[f"{metric}_stderr"] = stderr
        stdout_row[metric] = f"{mean:.3f} ± {stderr:.3f}"
    return csv_row, stdout_row


def main(args: argparse.Namespace) -> None:
    """Score every model under --eval_dir and report means + stderr."""
    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"--eval_dir not found: {eval_dir}")

    jsonl_name = f"{args.data_source}.jsonl"
    trajectory_paths = sorted(eval_dir.glob(f"*/dumped_evals/eval_only/{jsonl_name}"))
    if not trajectory_paths:
        raise FileNotFoundError(
            f"No trajectory files matched {eval_dir}/*/dumped_evals/eval_only/{jsonl_name}"
        )

    csv_rows: list[dict] = []
    stdout_rows: list[dict] = []
    for path in trajectory_paths:
        # <eval_dir>/<model_subdir>/dumped_evals/eval_only/<data_source>.jsonl
        model = path.parents[2].name
        scored = score_trajectory_file(path, args.data_source, args.scores)
        if len(scored["searches"]) == 0:
            print(f"Skipping {model}: no valid records in {path}")
            continue
        csv_row, stdout_row = build_row(model, scored, args.scores)
        csv_rows.append(csv_row)
        stdout_rows.append(stdout_row)

    df_stdout = pd.DataFrame(stdout_rows)
    print(tabulate(df_stdout, headers="keys", tablefmt="github", showindex=False))

    scores_dir = eval_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    csv_path = scores_dir / f"{args.data_source}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--eval_dir", type=str, required=True, help="Root eval directory (e.g. out__eval=hp500)."
    )
    parser.add_argument(
        "--data_source",
        type=str,
        required=True,
        choices=SUPPORTED_DATA_SOURCES,
        help="Dataset the trajectories are from (selects the jsonl filename and scorer).",
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        default=["em", "f1"],
        choices=METRICS,
        help="Metrics to recompute (default: em f1).",
    )
    main(parser.parse_args())
