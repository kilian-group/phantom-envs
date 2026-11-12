#!/usr/bin/env -S uv run python
"""Generate the evaluation-setup ablation tables as LaTeX fragments, one per model.

The paper evaluates every model with a deliberately stronger setup than it trains with
(Qwen3-Embedding-4B, 20 turns, 32768 context vs e5-base-v2, 10 turns, 8192 context). This
table answers the obvious objection: that the reported gains are an artifact of that
mismatch. Columns are the six benchmarks; rows pair each training regime's paper setup
with the same checkpoint re-evaluated under its own training setup, so the effect of the
evaluation setup is read vertically and compared across regimes.

The two setups differ in exactly three ways -- retriever, turn budget and context length --
with ``topk=3`` and every other setting held fixed.

Seeds are intersected before averaging. The paper setup has two training seeds but the
matched setup was only ever run for seed 1, and averaging two seeds against one would make
the row pair unpaired; only seeds present under *both* setups are used. Adding the missing
seed-2 evals later widens the comparison automatically, with no change here.

Reads:
    <figures_dir>/transfer_data.json   (paper setup, qwen3-4b_Flat)
    <figures_dir>/e5_data.json         (matched setup, e5_Flat)

Writes:
    <figures_dir>/retriever_table_<slug>.tex
    <figures_dir>/retriever_table_<slug>.pdf   (preview)
    <figures_dir>/retriever_tables.json

Usage:
    uv run --no-sync python reproduce_paper/scripts/save_transfer_data.py
    uv run --no-sync python reproduce_paper/scripts/save_transfer_data.py \\
        --retriever e5_Flat \\
        --benchmarks hp500,2wiki500,msq500,synthrm500,synthsm500,frames \\
        --out_name e5_data.json
    uv run --no-sync python reproduce_paper/scripts/create_retriever_tables.py
"""

import argparse
import json
from pathlib import Path

import plotting_utils
from plotting_utils import (
    combine_seeds,
    latex_cell,
    load_transfer_data,
    render_table_previews,
    wrap_latex_table,
)

# Only the models re-evaluated under the matched setup.
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
BENCHMARKS = ["hp500", "2wiki500", "msq500", "synthrm500", "synthsm500", "frames"]

BENCHMARK2HEADER = plotting_utils.EVAL_DATASET_ALIAS2NAME | {
    "synthrm500": "Synth-RM",
    "synthsm500": "Synth-SM",
}

MODEL2SLUG = {
    "Qwen/Qwen2.5-3B-Instruct": "qwen3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen7b",
}

# A None label means "use the model alias", matching the pooled-corpus tables.
REGIMES = [
    {"key": "base", "label": None},
    {"key": "pw", "label": f"Trained w/ {plotting_utils.PENV_SHORT_NAME}"},
    {"key": "searchr1", "label": "Trained w/ NQ+HotpotQA"},
]

# The matched-to-training rows are the ablation, so they carry the highlight.
SETUPS = [
    {"key": "paper", "label": "Paper setup", "highlight": False},
    {"key": "matched", "label": "Training setup", "highlight": True},
]

METRIC = "f1"
SCALE = 100.0
FMT = ".1f"
TABCOLSEP = "3pt"
# Float placement, matching how the paper places these appendix tables.
FLOAT_ARGS = "[!h]"


def paired_seed_cells(per_setup: dict[str, dict], model: str, regime: str, bench: str) -> dict | None:
    """Slice one cell from every setup, restricted to the seeds all of them share.

    Returns ``{setup_key: [(mean, stderr), ...]}`` in a common seed order, or None when any
    setup is missing that run or the setups share no seed.
    """
    by_seed: dict[str, dict] = {}
    for key, data in per_setup.items():
        records = data.get(model, {}).get(regime, {}).get(bench)
        if not records:
            return None
        by_seed[key] = {record["seed"]: record for record in records}

    shared = set.intersection(*(set(seeds) for seeds in by_seed.values()))
    if not shared:
        return None
    # Base rows carry seed None; sorting keys on (is_not_none, value) keeps None first.
    order = sorted(shared, key=lambda seed: (seed is not None, seed))
    return {
        key: [
            (float(by_seed[key][seed][METRIC]), float(by_seed[key][seed][f"{METRIC}_stderr"]))
            for seed in order
        ]
        for key in by_seed
    }


def summarize(per_setup: dict[str, dict], models: list[str]) -> dict:
    """Collapse raw per-seed records into ``summary[model][regime][setup][benchmark]``."""
    summary: dict = {}
    for model in models:
        summary[model] = {}
        for regime in REGIMES:
            per_setup_stats: dict = {setup["key"]: {} for setup in SETUPS}
            for bench in BENCHMARKS:
                paired = paired_seed_cells(per_setup, model, regime["key"], bench)
                if paired is None:
                    continue
                for key, seed_cells in paired.items():
                    mean, variance = combine_seeds(seed_cells)
                    per_setup_stats[key][bench] = {
                        "mean": mean * SCALE,
                        "stderr": variance**0.5 * SCALE,
                        "num_seeds": len(seed_cells),
                    }
            summary[model][regime["key"]] = {key: cells for key, cells in per_setup_stats.items() if cells}
    return summary


def regime_rows(summary: dict, model: str) -> list[str]:
    """Emit a label row plus one row per setup for every regime measured under both."""
    lines: list[str] = []
    for regime in REGIMES:
        per_setup_stats = summary[model][regime["key"]]
        missing = [setup["key"] for setup in SETUPS if setup["key"] not in per_setup_stats]
        if missing:
            print(
                f"  Warning: {plotting_utils.MODEL_HF2ALIAS[model]}/{regime['key']} has no "
                f"{'/'.join(missing)} results; omitting that block"
            )
            continue
        if lines:
            lines.append(r"\midrule")
        label = regime["label"] or plotting_utils.MODEL_HF2ALIAS[model]
        lines.append(rf"\multicolumn{{{1 + len(BENCHMARKS)}}}{{l}}{{{label}}} \\")
        for setup in SETUPS:
            cells = per_setup_stats[setup["key"]]
            color = r"\rowcolor{lightorange} " if setup["highlight"] else ""
            rendered = " & ".join(latex_cell(cells.get(bench), FMT) for bench in BENCHMARKS)
            lines.append(rf"{color}\quad {setup['label']} & {rendered} \\")
    return lines


CAPTION = "\n".join(
    [
        r"\caption{%",
        r"    \textbf{Ablation results on retriever configuration.}",
        r"    In the main text, we evaluate every model with a deliberately stronger setup than it trains"
        r" with (\texttt{Qwen3-Embedding-4B}, 20 turns, 32768 context). Here we re-evaluate the same"
        r" checkpoints under the training setup (\texttt{e5-base-v2}, 10 turns, 8192 context).",
        r"    Other index settings remain fixed.",
        r"    The weaker training setup generally lowers every model's score, regardless of how it was"
        r" trained.",
        r"    We report results for one training seed---including the Paper setup rows, which therefore"
        r" differ slightly from \Cref{tab:f1}---with standard errors over the test sets. The other"
        r" training seed follows the same trend.",
        r"    }",
    ]
)


def write_model_table(summary: dict, model: str, figures_dir: Path) -> Path | None:
    """Write one model's setup-by-benchmark table, or None when it has no paired results."""
    body = regime_rows(summary, model)
    if not body:
        print(f"  Warning: no paired results for {plotting_utils.MODEL_HF2ALIAS[model]}; skipping table")
        return None

    slug = MODEL2SLUG[model]
    table = wrap_latex_table(
        body,
        "l" + "c" * len(BENCHMARKS),
        " & ".join(
            [r"\shortstack[l]{Evaluation\\Setup}"] + [BENCHMARK2HEADER[bench] for bench in BENCHMARKS]
        ),
        CAPTION,
        f"tab:retriever_{slug}",
        TABCOLSEP,
        float_args=FLOAT_ARGS,
    )
    path = figures_dir / f"retriever_table_{slug}.tex"
    path.write_text(table)
    print(f"Saved {path}")
    return path


def main(args: argparse.Namespace) -> None:
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    models = args.models.split(",")
    unknown = [model for model in models if model not in MODEL2SLUG]
    if unknown:
        raise SystemExit(f"--models has entries with no table slug: {', '.join(unknown)}")

    per_setup = {}
    for setup, name in (("paper", args.transfer_json), ("matched", args.e5_json)):
        path = figures_dir / name
        if not path.exists():
            raise SystemExit(f"{path} not found -- run reproduce_paper/scripts/save_transfer_data.py first")
        per_setup[setup] = load_transfer_data(path)

    summary = summarize(per_setup, models)
    cache_path = figures_dir / "retriever_tables.json"
    with open(cache_path, "w") as f:
        json.dump(summary, f, indent=4)
        f.write("\n")
    print(f"Saved data to {cache_path}")

    tex_paths = [path for model in models if (path := write_model_table(summary, model, figures_dir))]
    if tex_paths and not args.no_preview:
        render_table_previews(tex_paths, figures_dir, "retriever_tables_preview")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS),
        help="Comma-separated HuggingFace model ids to emit tables for.",
    )
    parser.add_argument(
        "--transfer_json",
        type=str,
        default="transfer_data.json",
        help="Paper-setup (qwen3-4b_Flat) data file under --figures_dir.",
    )
    parser.add_argument(
        "--e5_json",
        type=str,
        default="e5_data.json",
        help="Matched-setup (e5_Flat) data file under --figures_dir.",
    )
    parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Skip compiling the standalone preview PDFs (e.g. when pdflatex is unavailable).",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="reproduce_paper/figures",
        help="Directory holding the input JSONs and receiving the .tex fragments.",
    )
    main(parser.parse_args())
