#!/usr/bin/env -S uv run python
"""Collect every evaluation run's raw scores into a single JSON, one record per training seed.

This is the one place that walks ``out__eval=*`` and ``final_ckpts.yaml``. It does no
statistics: each record holds the numbers exactly as ``score_trajectories.py`` wrote them,
so consumers can slice what they need and aggregate however they like. Seed averaging,
between-seed spread and benchmark averaging live in the consumers (see
``plotting_utils.combine_seeds`` / ``combine_benchmarks``).

Covers all 4 base models, all 7 eval benchmarks, and all 5 training regimes -- a superset
of what any one figure or table uses.

Layout::

    {
      "meta": {"retriever": ..., "models": [...], "benchmarks": [...], "train_datasets": [...]},
      "data": {
        "<model>": {
          "<train_dataset>": {
            "<benchmark>": [
              {"seed": 1, "path": ..., "step": 220, "n": 500, "f1": ..., "f1_stderr": ...,
               "num_searches": ..., "num_searches_stderr": ..., "n_no_answer": ...},
              ...                                  # one entry per training seed
            ]
          }
        }
      }
    }

``base`` rows carry ``"seed": null`` and ``"path": null`` -- there is no training run.

Reads:
    <eval_dirs_root>/<eval_dir_template>/scores/<benchmark>.csv
    reproduce_paper/final_ckpts.yaml

Writes:
    <figures_dir>/<out_name>

``--eval_dir_template`` selects which family of eval directories to walk, so the same
collector serves the per-benchmark evals and the pooled-corpus ones.

Usage:
    uv run --no-sync python reproduce_paper/scripts/save_transfer_data.py

    # Pooled-corpus evals: one directory holding every benchmark's dumps
    uv run --no-sync python reproduce_paper/scripts/save_transfer_data.py \\
        --eval_dir_template "out__eval=pooled6__ret={retriever}" \\
        --benchmarks hp500,2wiki500,msq500,synthrm500,synthsm500,frames \\
        --out_name pooled_data.json
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import plotting_utils
import yaml
from plotting_utils import find_csv_key_for_ckpt, seed_from_path

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-4-mini-instruct",
]
DEFAULT_EVAL_DIR_TEMPLATE = "out__eval={benchmark}__ret={retriever}"

# Each training dataset's section in final_ckpts.yaml
TRAIN_DATASET2SECTION = {
    "searchr1": "searchr1_own_trained",
    "pw": "pw_trained",
    "pwcons": "pwcons_trained",
    "pwcomp": "pwcomp_trained",
}

# Columns copied verbatim from the scores CSV into each record.
SCORE_COLUMNS = ["n", "n_no_answer", "num_searches", "num_searches_stderr", "f1", "f1_stderr"]

STEP_RE = re.compile(r"__step(\d+)$")


def record_from_row(row: pd.Series, seed: int | None, path: str | None, step: int | None) -> dict:
    """Build one JSON record from a scores-CSV row, tagged with its run identity."""
    record: dict = {"seed": seed, "path": path, "step": step}
    for column in SCORE_COLUMNS:
        value = row.get(column)
        record[column] = None if pd.isna(value) else float(value)
    return record


def trained_entries(final_ckpts: dict, section: str, model: str) -> list[dict]:
    """Return the e5_Flat-trained checkpoint entries for one model, in seed order."""
    return [
        entry
        for entry in final_ckpts.get(section, [])
        if entry.get("model") == model
        and entry.get("path") is not None
        and entry.get("training_retriever", "e5_Flat") == "e5_Flat"
    ]


def collect(
    eval_dirs_root: Path,
    eval_dir_template: str,
    retriever: str,
    benchmarks: list[str],
    final_ckpts: dict,
) -> dict:
    """Walk every benchmark CSV and return ``data[model][train_dataset][benchmark] -> [record]``."""
    data: dict = {model: {} for model in MODELS}

    for benchmark in benchmarks:
        eval_dir = eval_dir_template.format(benchmark=benchmark, retriever=retriever)
        csv_path = eval_dirs_root / eval_dir / "scores" / f"{benchmark}.csv"
        if not csv_path.exists():
            print(f"Warning: scores CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)

        for model in MODELS:
            alias = plotting_utils.MODEL_HF2ALIAS[model]

            base_rows = df[df["model"] == plotting_utils.model_hf_to_key(model)]
            if base_rows.empty:
                print(f"  Warning: base {alias} missing from {csv_path.name}")
            else:
                data[model].setdefault("base", {})[benchmark] = [
                    record_from_row(base_rows.iloc[0], seed=None, path=None, step=None)
                ]

            for train_dataset, section in TRAIN_DATASET2SECTION.items():
                records = []
                for entry in trained_entries(final_ckpts, section, model):
                    key = find_csv_key_for_ckpt(df, entry["path"], entry.get("ckpt_to_eval", -1))
                    if key is None:
                        print(
                            f"  Warning: {train_dataset} ckpt for {alias} missing from "
                            f"{csv_path.name}: {entry['path']}"
                        )
                        continue
                    step_match = STEP_RE.search(key)
                    records.append(
                        record_from_row(
                            df[df["model"] == key].iloc[0],
                            seed=seed_from_path(entry["path"]),
                            path=entry["path"],
                            step=int(step_match.group(1)) if step_match else None,
                        )
                    )
                if records:
                    data[model].setdefault(train_dataset, {})[benchmark] = records

    return data


def main(args: argparse.Namespace) -> None:
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(args.final_ckpts_yaml) as f:
        final_ckpts = yaml.safe_load(f)

    benchmarks = args.benchmarks.split(",")
    data = collect(Path(args.eval_dirs_root), args.eval_dir_template, args.retriever, benchmarks, final_ckpts)
    payload = {
        "meta": {
            "retriever": args.retriever,
            "eval_dir_template": args.eval_dir_template,
            "final_ckpts_yaml": args.final_ckpts_yaml,
            "models": MODELS,
            "benchmarks": benchmarks,
            "train_datasets": ["base", *TRAIN_DATASET2SECTION],
            "score_columns": SCORE_COLUMNS,
        },
        "data": data,
    }

    out_path = figures_dir / args.out_name
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")

    num_records = sum(
        len(records)
        for per_train in data.values()
        for per_benchmark in per_train.values()
        for records in per_benchmark.values()
    )
    print(f"Saved {num_records} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--final_ckpts_yaml",
        type=str,
        default="reproduce_paper/final_ckpts.yaml",
        help="Path to final_ckpts.yaml.",
    )
    parser.add_argument(
        "--eval_dirs_root",
        type=str,
        default=".",
        help="Root directory containing out__eval=*__ret=* dirs.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="qwen3-4b_Flat",
        help="Retriever suffix used in eval directory names.",
    )
    parser.add_argument(
        "--eval_dir_template",
        type=str,
        default=DEFAULT_EVAL_DIR_TEMPLATE,
        help="Eval directory name, with {benchmark} and {retriever} placeholders.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(plotting_utils.EVAL_DATASETS),
        help="Comma-separated benchmarks to collect.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="transfer_data.json",
        help="Filename to write under --figures_dir.",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="reproduce_paper/figures",
        help="Directory to write transfer_data.json.",
    )
    main(parser.parse_args())
