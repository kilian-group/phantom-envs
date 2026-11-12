#!/usr/bin/env -S uv run python
"""Generate the paper's Base-vs-Synthetic result tables as LaTeX fragments.

Emits three tables, all from the same seed-averaged numbers so the main text and the
appendix can never disagree:

    main_results_table.tex        Main text. One row per model and per
                                  ``+ Synthetic`` fine-tune; columns are Avg. F1 (%)
                                  and Avg. # Searches, each averaged over the six
                                  benchmarks.
    appendix_f1_table.tex         Appendix. Per-benchmark F1.
    appendix_searches_table.tex   Appendix. Per-benchmark # search calls.

Aggregation (see ``plotting_utils.combine_seeds`` / ``combine_benchmarks``):

1. Within a benchmark, each run contributes its mean and its standard error over test
   examples, straight from the ``score_trajectories.py`` CSVs.
2. Training seeds are pooled per benchmark. The variance adds the test-example noise
   ``(1/S^2) * sum(se_i^2)`` to the between-seed spread ``var(means)/S``.
Every reported error therefore pools test-set noise with the between-seed spread. A
six-benchmark average is still computed into the JSON (via ``combine_benchmarks``, which
treats the benchmarks as fixed) but no longer appears in either table.

Reads:
    <figures_dir>/transfer_data.json   (written by save_transfer_data.py -- run it first)

Writes:
    <figures_dir>/{f1,searches}_table.tex
    <figures_dir>/{f1,searches}_table.pdf   (preview)
    <figures_dir>/result_tables.json

Usage:
    uv run --no-sync python reproduce_paper/scripts/save_transfer_data.py
    uv run --no-sync python reproduce_paper/scripts/create_result_tables.py
"""

import argparse
import json
from pathlib import Path

import plotting_utils
from plotting_utils import (
    combine_benchmarks,
    combine_seeds,
    latex_cell,
    load_transfer_data,
    render_table_previews,
    transfer_cells,
    wrap_latex_table,
)

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-4-mini-instruct",
]
BENCHMARKS = ["hp500", "2wiki500", "msq500", "synthrm500", "synthsm500", "frames"]

# Shortened headers, matching the hand-edited paper copy of the old table.
BENCHMARK2HEADER = plotting_utils.EVAL_DATASET_ALIAS2NAME | {
    "synthrm500": "Synth-RM",
    "synthsm500": "Synth-SM",
}

TRAINED_SECTION = "pw_trained"
SYNTHETIC_ROW_LABEL = rf"\quad + {plotting_utils.PENV_SHORT_NAME}"

METRICS = {
    "f1": {"col": "f1", "col_stderr": "f1_stderr", "scale": 100.0, "fmt": ".1f"},
    "num_searches": {
        "col": "num_searches",
        "col_stderr": "num_searches_stderr",
        "scale": 1.0,
        "fmt": ".2f",
    },
}

# Per-table column padding. The search-call cells ("X.XX \pm X.XX") are wider than the F1
# ones, so the seven-column searches table needs tighter padding to fit \textwidth.
TABCOLSEP = {"f1": "4pt", "num_searches": "3pt"}
# Float placement, matching how each table is placed in the paper.
FLOAT_ARGS = {"f1": "[tbp]", "num_searches": "[t]"}


def summarize(data: dict) -> dict:
    """Collapse the raw per-seed records into per-benchmark and six-benchmark statistics.

    Returns ``summary[metric][model][regime]`` with a ``per_benchmark`` map of
    ``{mean, stderr}`` and an ``average`` of ``{mean, stderr, num_seeds}``.
    """
    summary: dict = {metric: {model: {} for model in MODELS} for metric in METRICS}
    for metric, cfg in METRICS.items():
        scale = cfg["scale"]
        for model in MODELS:
            for regime in ("base", "pw"):
                per_benchmark = {}
                pooled = []
                num_seeds = 0
                for bench in BENCHMARKS:
                    seed_cells = transfer_cells(data, model, regime, bench, metric)
                    if not seed_cells:
                        continue
                    num_seeds = len(seed_cells)
                    mean, variance = combine_seeds(seed_cells)
                    per_benchmark[bench] = {
                        "mean": mean * scale,
                        "stderr": variance**0.5 * scale,
                    }
                    pooled.append((mean, variance))
                if len(pooled) != len(BENCHMARKS):
                    if per_benchmark:
                        print(
                            f"  Warning: {plotting_utils.MODEL_HF2ALIAS[model]}/{regime} has "
                            f"{len(pooled)}/{len(BENCHMARKS)} benchmarks for {metric}; skipping average"
                        )
                    continue
                mean, stderr = combine_benchmarks(pooled)
                summary[metric][model][regime] = {
                    "per_benchmark": per_benchmark,
                    "average": {"mean": mean * scale, "stderr": stderr * scale, "num_seeds": num_seeds},
                }
    return summary


def model_rows(render_row) -> list[str]:
    """Emit a base row + a highlighted ``+ Synthetic`` row for each model, in MODELS order."""
    lines = []
    for model in MODELS:
        lines.append(f"{plotting_utils.MODEL_HF2ALIAS[model]} & " + render_row(model, "base") + r" \\")
        lines.append(r"\addlinespace[2pt]")
        lines.append(rf"\rowcolor{{lightorange}} {SYNTHETIC_ROW_LABEL} & " + render_row(model, "pw") + r" \\")
        lines.append(r"\addlinespace[2pt]")
    return lines[:-1]  # trailing \addlinespace would double up with \bottomrule


# Caption wording is carried over verbatim from the paper's existing table, apart from the
# metric name and the model list (which now includes \llamathree).
CAPTIONS = {
    "f1": [
        r"    \textbf{F1 scores on real-world agentic search benchmarks after RL fine-tuning on rule-generated synthetic environments.}",
        r"    Synthetic training significantly improves performance of all LLM families and sizes.",
        r"    Improvements are largest on the newer and harder benchmarks---\synthrm, \synthsm, and \frames---\llamathree improves by up to $7.1\times$ on \synthsm.",
        r"    We report mean $\pm$ standard error over the test sets and two training seeds.",
    ],
    "num_searches": [
        r"    \textbf{Number of search calls per question on real-world multi-hop agentic search benchmarks before and after RL fine-tuning on rule-generated synthetic environment.}",
        r"    We evaluate four LLMs (\qwenthree, \qwenseven, \llamathree, \phimini) and their fine-tuned counterparts across six benchmarks.",
        r"    These benchmarks are created from different Wikipedia date cutoffs and require 2-10 hops for answering questions.",
        r"    Fine-tuned agents consistently issue more search calls per question than their base models, across all benchmarks, LLM families, and sizes.",
        r"    We report standard errors over the test sets and the training seeds.",
    ],
}
LABELS = {"f1": "tab:f1", "num_searches": "tab:searches"}
FIG_STEMS = {"f1": "f1_table", "num_searches": "searches_table"}


def write_metric_table(summary: dict, metric: str, figures_dir: Path) -> Path:
    """Write one model-by-benchmark table for a single metric."""

    def render_row(model: str, regime: str) -> str:
        stats = summary[metric][model].get(regime)
        return " & ".join(
            latex_cell(stats["per_benchmark"].get(bench) if stats else None, METRICS[metric]["fmt"])
            for bench in BENCHMARKS
        )

    caption = "\n".join([r"\caption{%", *CAPTIONS[metric], r"    }"])
    table = wrap_latex_table(
        model_rows(render_row),
        "l" + "c" * len(BENCHMARKS),
        " & ".join(["Model"] + [BENCHMARK2HEADER[bench] for bench in BENCHMARKS]),
        caption,
        LABELS[metric],
        TABCOLSEP[metric],
        float_args=FLOAT_ARGS[metric],
    )
    path = figures_dir / f"{FIG_STEMS[metric]}.tex"
    path.write_text(table)
    print(f"Saved {path}")
    return path


def main(args: argparse.Namespace) -> None:
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    transfer_path = figures_dir / "transfer_data.json"
    if not transfer_path.exists():
        raise SystemExit(
            f"{transfer_path} not found -- run reproduce_paper/scripts/save_transfer_data.py first"
        )
    summary = summarize(load_transfer_data(transfer_path))

    cache_path = figures_dir / "result_tables.json"
    with open(cache_path, "w") as f:
        json.dump(summary, f, indent=4)
        f.write("\n")
    print(f"Saved data to {cache_path}")

    tex_paths = [write_metric_table(summary, metric, figures_dir) for metric in METRICS]
    if not args.no_preview:
        render_table_previews(tex_paths, figures_dir, "result_tables_preview")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        help="Directory to write the .tex fragments and the data cache.",
    )
    main(parser.parse_args())
