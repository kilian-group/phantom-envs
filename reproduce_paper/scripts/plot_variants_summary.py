#!/usr/bin/env -S uv run python
"""Compact summary of the environment-complexity ablation for Section 5.1.

One panel per base model, three bars per panel (Hops / Hops+Comparisons /
Hops+Constraints), each bar the F1 averaged over the six real-world benchmarks and over
both independent training seeds. The base model is a dashed reference line.

CofCA is deliberately excluded here: it is the benchmark where the matched-axis effect
lives, so it is reported per question category in the category-delta figure instead of
being averaged away. This matches the benchmark set used by the other summary figures.

Error bars pool two independent sources via ``combine_seeds`` (test-example noise within
each seed, plus between-seed spread) and then propagate across benchmarks with
``combine_benchmarks``.

Reads ``transfer_data.json`` (written by ``save_transfer_data.py``) and writes
``f1_variants_summary.{pdf,png}`` plus a JSON of every plotted number.

Usage:
    uv run python reproduce_paper/scripts/plot_variants_summary.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotting_utils
from matplotlib.patches import Patch
from plotting_utils import (
    combine_benchmarks,
    combine_seeds,
    load_transfer_data,
    transfer_cells,
)

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
VARIANTS = ["pw", "pwcons", "pwcomp"]  # Hops, Hops+Constraints, Hops+Comparisons
# myYellow is too light to read on white; a dark ochre keeps the label legible.
VALUE_LABEL_COLOR_OVERRIDES = {"pwcons": "#B8860B"}
BENCHMARKS = [d for d in plotting_utils.EVAL_DATASETS if d != "cofca500"]

BAR_WIDTH = 0.85
# Clearance between the top of a bar's error bar and its value label, in F1 points.
VALUE_LABEL_PAD = 0.8
YLIM = (0, 55)
YTICKS = [0, 10, 20, 30, 40, 50]


def group_score(data: dict, model: str, train_dataset: str) -> tuple[float, float] | None:
    """Mean F1 and standard error over BENCHMARKS and training seeds, or None if missing.

    Args:
        data: the ``data`` mapping from transfer_data.json.
        model: HuggingFace model name.
        train_dataset: training regime alias (``base``, ``pw``, ``pwcomp``, ``pwcons``).

    Returns:
        ``(mean, stderr)`` as fractions in [0, 1], or None when the cell was never evaluated.
    """
    per_benchmark = []
    for benchmark in BENCHMARKS:
        seed_cells = transfer_cells(data, model, train_dataset, benchmark, "f1")
        if not seed_cells:
            return None
        per_benchmark.append(combine_seeds(seed_cells))
    return combine_benchmarks(per_benchmark)


def draw_panel(ax: plt.Axes, data: dict, model: str, show_yticks: bool) -> dict:
    """Draw one model's three variant bars plus its base reference line.

    Returns the plotted numbers for this model, for the JSON dump.
    """
    record: dict = {}

    base = group_score(data, model, "base")
    if base is not None:
        ax.axhline(
            y=base[0] * 100,
            color="black",
            linestyle=plotting_utils.BASE_LINE_STYLE,
            linewidth=plotting_utils.BASE_LINE_WIDTH,
            zorder=3,
        )
        record["base"] = {"f1": base[0] * 100, "stderr": base[1] * 100}

    for i, variant in enumerate(VARIANTS):
        cell = group_score(data, model, variant)
        if cell is None:
            print(f"Warning: no {variant} data for {model}")
            continue
        mean, stderr = cell[0] * 100, cell[1] * 100
        color = plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[variant]]
        ax.bar(
            i,
            mean,
            BAR_WIDTH,
            yerr=stderr,
            color=color,
            edgecolor=plotting_utils.EDGE_COLOR,
            linewidth=plotting_utils.LINE_WIDTH,
            error_kw={"linewidth": plotting_utils.LINE_WIDTH, "ecolor": "black", "capsize": 4},
            zorder=2,
        )
        ax.text(
            i,
            mean + stderr + VALUE_LABEL_PAD,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=plotting_utils.ANNOTATION_FONT_SIZE - 4,
            fontweight="bold",
            color=VALUE_LABEL_COLOR_OVERRIDES.get(variant, color),
            zorder=5,
        )
        record[variant] = {"f1": mean, "stderr": stderr}
        if base is not None:
            record[variant]["delta_vs_base"] = mean - base[0] * 100

    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([])
    ax.set_xlim(-0.6, len(VARIANTS) - 0.4)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.set_title(
        plotting_utils.MODEL_HF2ALIAS[model],
        fontsize=plotting_utils.LABEL_FONT_SIZE,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=plotting_utils.TICK_FONT_SIZE, labelleft=show_yticks)
    ax.grid(axis="y", linestyle="dotted", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    if show_yticks:
        ax.set_ylabel("Avg. F1 (%)", fontsize=plotting_utils.LABEL_FONT_SIZE)
    return record


def main(args: argparse.Namespace) -> None:
    figures_dir = Path(args.figures_dir)
    transfer_path = figures_dir / "transfer_data.json"
    if not transfer_path.exists():
        raise SystemExit(
            f"{transfer_path} not found -- run reproduce_paper/scripts/save_transfer_data.py first"
        )
    data = load_transfer_data(transfer_path)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.1 * len(MODELS), 3.4), sharey=True)
    axes = np.atleast_1d(axes)

    summary = {}
    for i, model in enumerate(MODELS):
        summary[model] = draw_panel(axes[i], data, model, show_yticks=(i == 0))

    handles = [
        plt.Line2D(
            [],
            [],
            color="black",
            linestyle=plotting_utils.BASE_LINE_STYLE,
            linewidth=plotting_utils.BASE_LINE_WIDTH,
            label="Base",
        )
    ] + [
        Patch(
            facecolor=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[v]],
            edgecolor=plotting_utils.EDGE_COLOR,
            label=plotting_utils.TRAIN_DATASET_ALIAS2NAME[v],
        )
        for v in VARIANTS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=plotting_utils.LEGEND_FONT_SIZE - 2,
        bbox_to_anchor=(0.5, -0.16),
    )

    fig.tight_layout()
    stem = figures_dir / "f1_variants_summary"
    for suffix in ("pdf", "png"):
        fig.savefig(f"{stem}.{suffix}", bbox_inches="tight", dpi=300)
        print(f"Saved {stem}.{suffix}")
    payload = {
        "benchmarks": BENCHMARKS,
        "note": "F1 (%) averaged over benchmarks and training seeds; stderr pools test noise and seed spread",
        "models": summary,
    }
    json_path = figures_dir / "f1_variants_summary.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="reproduce_paper/figures",
        help="Directory holding transfer_data.json, and where the figure is written.",
    )
    main(parser.parse_args())
