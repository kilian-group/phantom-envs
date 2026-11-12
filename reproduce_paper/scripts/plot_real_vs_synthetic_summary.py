#!/usr/bin/env -S uv run python
"""Three-panel summary of synthetic vs. real-world (NQ+HotpotQA) training, one figure per model.

Condenses the per-benchmark F1 scores into a single row that carries the Section 4.1
argument directly:

    (a) F1 averaged over the three benchmarks in-domain to NQ+HotpotQA
        (HotpotQA, 2Wiki, MuSiQue) -- real-world training wins.
    (b) F1 averaged over the three out-of-domain benchmarks
        (SynthWorlds-RM, SynthWorlds-SM, FRAMES) -- synthetic training matches or wins.
    (c) Knowledge Advantage (KA) gap on the SynthWorlds RM/SM pair, in the line-plot
        style of the SynthWorlds paper (arXiv:2510.24427, Fig. 1): the gap widens under
        NQ+HotpotQA training and closes under synthetic training.

Pretrained knowledge and fluency (PopQA, MMLU, Wikipedia-2018 perplexity) was a fourth
panel here; it is now ``create_nlu_table.py``.

All F1 numbers are averaged over the two independent training seeds in
``final_ckpts.yaml``. Reported standard errors pool test-example noise with the
between-seed spread (see :func:`combine_seeds`).

Reads:
    <figures_dir>/transfer_data.json   (written by save_transfer_data.py -- run it first)

Writes:
    <figures_dir>/real_vs_synthetic_summary__<model_alias>.{pdf,png}
    <figures_dir>/real_vs_synthetic_summary.json

Usage:
    uv run --no-sync python reproduce_paper/scripts/save_transfer_data.py
    uv run --no-sync python reproduce_paper/scripts/plot_real_vs_synthetic_summary.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.lines as mlines
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

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# Benchmarks sharing the Wikipedia-2018 corpus that NQ+HotpotQA is drawn from, vs. those
# sourced from later dumps or fictional referents.
IN_DOMAIN_BENCHMARKS = ["hp500", "2wiki500", "msq500"]
OOD_BENCHMARKS = ["synthrm500", "synthsm500", "frames"]
ALL_BENCHMARKS = IN_DOMAIN_BENCHMARKS + OOD_BENCHMARKS

# Trained regimes drawn as bars; "base" is drawn as a dashed reference line instead.
TRAIN_REGIME2SECTION = {"searchr1": "searchr1_own_trained", "pw": "pw_trained"}
BAR_REGIMES = ["searchr1", "pw"]
KA_REGIMES = ["base", "searchr1", "pw"]
REGIME2LABEL = plotting_utils.TRAIN_DATASET_ALIAS2NAME | {"pw": plotting_utils.PENV_SHORT_NAME}

# Panel (c) lines. Deliberately NOT myBlue/myOrange: those already mean NQ+HotpotQA and
# Synthetic in panels (a), (b) and (d), and panel (c)'s x-axis is those same regimes.
SYNTHWORLDS_VARIANTS = ["synthrm500", "synthsm500"]
SYNTHWORLDS_VARIANT2COLOR = {
    "synthrm500": plotting_utils.COLORS2HEX["myGreen"],
    "synthsm500": plotting_utils.COLORS2HEX["myPurple"],
}
SYNTHWORLDS_VARIANT2MARKER = {"synthrm500": "o", "synthsm500": "s"}
# Short keys used for the RM/SM fields of each KA summary entry.
SYNTHWORLDS_VARIANT2KEY = {"synthrm500": "rm", "synthsm500": "sm"}
KA_SHADE_COLOR = "#606060"
# Below this many F1 points the arrow is shorter than its own heads, so it is skipped.
KA_MIN_ARROW_GAP = 2.0

# Panel (d) metrics. Percent changes are reported with their natural sign -- PopQA and MMLU
# are higher-better, Wiki-18 perplexity lower-better -- and the main text states the reading.
BAR_WIDTH = 0.42
BAR_SPACING = 0.5
# Clearance between the top of a bar's error bar and its value label, in F1 points.
VALUE_LABEL_PAD = 1.0
F1_YTICKS = [0, 20, 40, 60]
# The (a)/(b) panels share this scale so their heights are directly comparable.
F1_YLIM = 70
# Relative panel widths: the two bar panels hold two bars each, the KA panel holds six.
PANEL_WIDTH_RATIOS = [0.25, 0.25, 0.50]
# Fixed KA scale: leaves room above the tallest RM point for its "KA = N" label.
KA_YLIM = 60
# Axis labels and category tick labels share one size; the defaults made x labels
# noticeably smaller than the y labels.
# This figure carries the Section 4.1 argument at column width, so every font sits four
# points above the shared defaults.
FONT_BUMP = 4
LABEL_FONT_SIZE = plotting_utils.LABEL_FONT_SIZE + FONT_BUMP
TICK_FONT_SIZE = plotting_utils.TICK_FONT_SIZE + FONT_BUMP
LEGEND_FONT_SIZE = plotting_utils.LEGEND_FONT_SIZE + FONT_BUMP
ANNOTATION_FONT_SIZE = plotting_utils.ANNOTATION_FONT_SIZE + FONT_BUMP
AXIS_FONT_SIZE = LABEL_FONT_SIZE
# RM is drawn solid, SM hatched, so the pair stays separable without relying on color.
# "NQ+HotpotQA" is too wide for one line at AXIS_FONT_SIZE; wrapping keeps every
# axis label the same size rather than shrinking this panel's alone.
KA_REGIME2XLABEL = {"base": "Base", "searchr1": "NQ+HotpotQA", "pw": plotting_utils.PENV_SHORT_NAME}
KA_VARIANT_HATCH = {"rm": "", "sm": "////"}
KA_GROUP_WIDTH = 0.8
KA_BAR_WIDTH = KA_GROUP_WIDTH / 2
FIGURE_WIDTH = 16.0


# --- Data loading -----------------------------------------------------------------


def trained_entries(final_ckpts: dict, section: str, model: str) -> list[dict]:
    """Return the e5_Flat-trained checkpoint entries for one model, in seed order."""
    return [
        entry
        for entry in final_ckpts.get(section, [])
        if entry.get("model") == model
        and entry.get("path") is not None
        and entry.get("training_retriever", "e5_Flat") == "e5_Flat"
    ]


def load_f1_cells(data: dict) -> tuple[dict, dict]:
    """Slice per-seed F1 cells out of the raw transfer data, plus each seed's checkpoint step.

    Args:
        data: the ``data`` mapping from ``transfer_data.json``.

    Returns:
        ``(cells, steps)`` where ``cells[model][benchmark][regime]`` is a list of
        ``(mean, test_stderr)`` -- one entry per training seed, or a single entry for
        ``"base"`` -- and ``steps[model][regime]`` lists the checkpoint steps, reused to
        locate the matching ``out__nlu/`` results so the two can never drift apart.
    """
    cells: dict = {model: {bench: {} for bench in ALL_BENCHMARKS} for model in MODELS}
    steps: dict = {model: {} for model in MODELS}

    for model in MODELS:
        for bench in ALL_BENCHMARKS:
            for regime in KA_REGIMES:
                seed_cells = transfer_cells(data, model, regime, bench, "f1")
                if not seed_cells:
                    continue
                cells[model][bench][regime] = seed_cells
        for regime in BAR_REGIMES:
            records = data.get(model, {}).get(regime, {}).get(ALL_BENCHMARKS[0], [])
            if records:
                steps[model][regime] = [record["step"] for record in records]

    return cells, steps


def group_score(cells: dict, model: str, benchmarks: list[str], regime: str) -> tuple[float, float] | None:
    """Mean F1 and standard error over ``benchmarks`` and training seeds, or None if missing."""
    per_benchmark = [
        combine_seeds(cells[model][bench][regime]) for bench in benchmarks if regime in cells[model][bench]
    ]
    if len(per_benchmark) != len(benchmarks):
        return None
    return combine_benchmarks(per_benchmark)


def summarize(f1_cells: dict) -> dict:
    """Collapse the raw per-seed cells into the plotted numbers, for the JSON dump."""
    summary: dict = {}
    for model in MODELS:
        per_model: dict = {"in_domain": {}, "out_of_domain": {}, "ka": {}}

        for panel, benchmarks in [("in_domain", IN_DOMAIN_BENCHMARKS), ("out_of_domain", OOD_BENCHMARKS)]:
            for regime in KA_REGIMES:
                score = group_score(f1_cells, model, benchmarks, regime)
                if score is not None:
                    per_model[panel][regime] = {"f1": score[0], "f1_stderr": score[1]}

        for regime in KA_REGIMES:
            variants = {}
            for variant in SYNTHWORLDS_VARIANTS:
                if regime in f1_cells[model][variant]:
                    mean, variance = combine_seeds(f1_cells[model][variant][regime])
                    variants[variant] = {"f1": mean, "f1_stderr": float(np.sqrt(variance)), "var": variance}
            if len(variants) == len(SYNTHWORLDS_VARIANTS):
                rm, sm = variants["synthrm500"], variants["synthsm500"]
                per_model["ka"][regime] = {
                    "rm": rm["f1"],
                    "rm_stderr": rm["f1_stderr"],
                    "sm": sm["f1"],
                    "sm_stderr": sm["f1_stderr"],
                    "gap": rm["f1"] - sm["f1"],
                    "gap_stderr": float(np.sqrt(rm["var"] + sm["var"])),
                }

        summary[model] = per_model
    return summary


# --- Drawing ----------------------------------------------------------------------


def draw_f1_panel(ax: plt.Axes, panel_summary: dict, title: str, show_ylabel: bool) -> None:
    """Draw one averaged-F1 bar panel: a bar per trained regime over a dashed base line."""
    if "base" in panel_summary:
        ax.axhline(
            y=panel_summary["base"]["f1"] * 100,
            color=plotting_utils.BASE_COLOR,
            linestyle=plotting_utils.BASE_LINE_STYLE,
            linewidth=plotting_utils.BASE_LINE_WIDTH,
            zorder=3,
        )

    for i, regime in enumerate(BAR_REGIMES):
        if regime not in panel_summary:
            continue
        mean = panel_summary[regime]["f1"] * 100
        stderr = panel_summary[regime]["f1_stderr"] * 100
        color = plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[regime]]
        ax.bar(
            i * BAR_SPACING,
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
            i * BAR_SPACING,
            mean + stderr + VALUE_LABEL_PAD,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE - 2,
            fontweight="bold",
            color=color,
            zorder=5,
        )

    ax.set_xlim(-BAR_WIDTH / 2 - 0.12, (len(BAR_REGIMES) - 1) * BAR_SPACING + BAR_WIDTH / 2 + 0.12)
    ax.set_ylim(0, F1_YLIM)
    ax.set_yticks(F1_YTICKS)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    if show_ylabel:
        ax.set_ylabel("F1 (%)", fontsize=AXIS_FONT_SIZE)
    ax.set_title(title, fontsize=LABEL_FONT_SIZE, fontweight="bold")


def draw_ka_panel(ax: plt.Axes, ka_summary: dict) -> None:
    """Draw the SynthWorlds KA panel: an RM/SM bar pair per regime, gap arrowed and annotated."""
    regimes = [regime for regime in KA_REGIMES if regime in ka_summary]

    for regime_idx, regime in enumerate(regimes):
        entry = ka_summary[regime]
        color = (
            plotting_utils.BASE_COLOR
            if regime == "base"
            else plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[regime]]
        )
        for variant_idx, key in enumerate(["rm", "sm"]):
            ax.bar(
                regime_idx + (variant_idx - 0.5) * KA_BAR_WIDTH,
                entry[key] * 100,
                KA_BAR_WIDTH,
                yerr=entry[f"{key}_stderr"] * 100,
                color=color,
                edgecolor=plotting_utils.EDGE_COLOR,
                linewidth=plotting_utils.LINE_WIDTH,
                hatch=KA_VARIANT_HATCH[key],
                error_kw={"linewidth": plotting_utils.LINE_WIDTH, "ecolor": "black", "capsize": 3},
                zorder=2,
            )

        # Sits just right of the bars' shared edge rather than on it, so the arrow is not
        # lost against the bar outlines. A gap this small would be all arrowheads.
        rm, sm = entry["rm"] * 100, entry["sm"] * 100
        if rm - sm >= KA_MIN_ARROW_GAP:
            arrow_x = regime_idx + 0.22 * KA_BAR_WIDTH
            ax.annotate(
                "",
                xy=(arrow_x, rm),
                xytext=(arrow_x, sm),
                arrowprops=dict(arrowstyle="<->", color=KA_SHADE_COLOR, lw=2.5, shrinkA=0, shrinkB=0),
                zorder=4,
            )
        y_top = max(
            entry["rm"] * 100 + entry["rm_stderr"] * 100, entry["sm"] * 100 + entry["sm_stderr"] * 100
        )
        ax.text(
            regime_idx,
            y_top + 1.5,
            f"KA$=${entry['gap'] * 100:.0f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE - 3,
            fontweight="bold",
            zorder=5,
        )

    # The shared legend already names the three regimes by color, so tick labels here
    # would repeat it and force the panel wider.
    ax.set_xticks(np.arange(len(regimes), dtype=float))
    ax.set_xticklabels([])
    ax.set_xlim(-0.6, len(regimes) - 0.4)
    ax.set_ylim(0, F1_YLIM)
    ax.set_yticks(F1_YTICKS)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_title(
        "Knowledge Advantage (KA) gap on SynthWorlds",
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
    )


def style_axes(ax: plt.Axes) -> None:
    """Apply the shared grid/spine styling used across the paper's figures."""
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=plotting_utils.LINE_WIDTH)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(plotting_utils.LINE_WIDTH)


def draw_figure(model: str, model_summary: dict, figures_dir: Path) -> None:
    """Render and save the summary row for one base model."""
    fig, axes = plt.subplots(
        1,
        len(PANEL_WIDTH_RATIOS),
        figsize=(FIGURE_WIDTH * sum(PANEL_WIDTH_RATIOS), 4.2),
        gridspec_kw={"width_ratios": PANEL_WIDTH_RATIOS},
    )

    draw_f1_panel(axes[0], model_summary["in_domain"], "In-domain benchmarks", True)
    draw_f1_panel(axes[1], model_summary["out_of_domain"], "Out-of-domain benchmarks", False)
    draw_ka_panel(axes[2], model_summary["ka"])
    for ax in axes:
        style_axes(ax)
    for ax, letter in zip(axes, ["(a)", "(b)", "(c)"], strict=True):
        ax.text(
            0.03,
            0.97,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
            zorder=6,
        )

    # Regimes (shared by all three panels), then panel (c)'s RM/SM fill distinction --
    # which now lives here rather than in a second legend inside (c).
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=plotting_utils.BASE_COLOR,
            linestyle=plotting_utils.BASE_LINE_STYLE,
            linewidth=plotting_utils.BASE_LINE_WIDTH,
            label=REGIME2LABEL["base"],
        ),
        *[
            Patch(
                facecolor=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[regime]],
                edgecolor="black",
                label=REGIME2LABEL[regime],
            )
            for regime in BAR_REGIMES
        ],
        Patch(facecolor="white", edgecolor=plotting_utils.EDGE_COLOR, label="Synth-RM"),
        Patch(
            facecolor="white",
            edgecolor=plotting_utils.EDGE_COLOR,
            hatch=KA_VARIANT_HATCH["sm"],
            label="Synth-SM",
        ),
    ]
    alias = plotting_utils.MODEL_HF2ALIAS[model]

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.30)

    # No panel carries x tick labels now, so the strip beneath the row is empty; the shared
    # legend goes there instead of costing the figure a band at the top.
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.5, axes[0].get_position().y0 - 0.01),
        ncol=len(legend_handles),
    )

    for suffix in (".pdf", ".png"):
        save_path = figures_dir / f"real_vs_synthetic_summary__{alias}{suffix}"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
    plt.close()


def main(args: argparse.Namespace) -> None:
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    transfer_path = figures_dir / "transfer_data.json"
    if not transfer_path.exists():
        raise SystemExit(
            f"{transfer_path} not found -- run reproduce_paper/scripts/save_transfer_data.py first"
        )
    f1_cells, _ = load_f1_cells(load_transfer_data(transfer_path))
    summary = summarize(f1_cells)

    cache_path = figures_dir / "real_vs_synthetic_summary.json"
    with open(cache_path, "w") as f:
        json.dump(summary, f, indent=4)
        f.write("\n")
    print(f"Saved data to {cache_path}")

    for model in MODELS:
        draw_figure(model, summary[model], figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="reproduce_paper/figures",
        help="Directory holding transfer_data.json, and where figures are written.",
    )
    main(parser.parse_args())
