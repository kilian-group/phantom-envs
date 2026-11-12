#!/usr/bin/env -S uv run python
"""Plot eval F1 (avg_score) vs training steps for all pw_trained checkpoints.

Reads per-checkpoint eval results from ``out__eval=<dataset>__all_ckpts/`` (one
``model=...__step<N>`` subdir per checkpoint) and the base-model (step 0) results
from ``out__eval=<dataset>__ret=qwen3-4b_Flat/``. Produces one subplot per eval
dataset with one line per model.

Example usage:
```bash
uv run python reproduce_paper/scripts/plot_f1_vs_training_steps.py
```
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import plotting_utils
from matplotlib import lines

EVAL_DATASETS = ["hp500", "2wiki500", "msq500", "synthrm500", "synthsm500", "frames"]

# Training runs span 0-220 steps; three round ticks keep the six panels uncluttered.
XTICKS = [0, 100, 200]

# Shortened from the shared map: six panel titles on one row leave no room for "SynthWorlds-".
DATASET2TITLE = plotting_utils.EVAL_DATASET_ALIAS2NAME | {
    "synthrm500": "Synth-RM",
    "synthsm500": "Synth-SM",
}

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
MODEL2COLOR = {
    "Qwen/Qwen2.5-3B-Instruct": plotting_utils.COLORS2HEX["myBlue"],
    "Qwen/Qwen2.5-7B-Instruct": plotting_utils.COLORS2HEX["myGreen"],
    "microsoft/Phi-4-mini-instruct": plotting_utils.COLORS2HEX["myPurple"],
    "meta-llama/Llama-3.2-3B-Instruct": plotting_utils.COLORS2HEX["myBrown"],
}
MODEL2MARKER = {
    "Qwen/Qwen2.5-3B-Instruct": "o",
    "Qwen/Qwen2.5-7B-Instruct": "s",
    "microsoft/Phi-4-mini-instruct": "^",
    "meta-llama/Llama-3.2-3B-Instruct": "D",
}

# Trained-checkpoint dirs look like model=share--runs...--pw--<Org>--<Model>--grpo--...__step<N>
CKPT_MODEL_RE = re.compile(r"--pw--(.+?)--grpo--")
CKPT_STEP_RE = re.compile(r"__step(\d+)$")


def read_mean_stderr(eval_only_dir: Path, dataset: str) -> tuple[float, float] | None:
    """Compute (mean, stderr) of per-question F1 scores from <dataset>.jsonl, or None if missing."""
    scores_path = eval_only_dir / f"{dataset}.jsonl"
    if not scores_path.exists():
        return None
    # score is the token-level reward vector; the episode F1 sits on the last token
    values = [
        record["score"][-1] if isinstance(record["score"], list) else record["score"]
        for record in plotting_utils.load_records(scores_path)
    ]
    if not values:
        return None
    return plotting_utils.mean_stderr(values, binary=False)


def collect_scores(dataset: str, base_eval_dir_template: str) -> dict[str, dict[int, tuple[float, float]]]:
    """Collect {model_hf_name: {step: (mean_f1, stderr)}} for one eval dataset.

    Step 0 comes from the base-model eval dir; steps > 0 from the all_ckpts run dir.
    """
    scores: dict[str, dict[int, tuple[float, float]]] = {model: {} for model in MODELS}

    for model in MODELS:
        base_dir = Path(base_eval_dir_template.format(dataset=dataset))
        base_score = read_mean_stderr(
            base_dir / f"model={model.replace('/', '--')}/dumped_evals/eval_only", dataset
        )
        if base_score is not None:
            scores[model][0] = base_score
        else:
            print(f"  Warning: no base-model results for {model} on {dataset}")

    run_dir = Path(f"out__eval={dataset}__all_ckpts")
    for model_dir in sorted(run_dir.glob("model=*__step*")):
        model_match = CKPT_MODEL_RE.search(model_dir.name)
        step_match = CKPT_STEP_RE.search(model_dir.name)
        if model_match is None or step_match is None:
            continue
        model = model_match.group(1).replace("--", "/")
        if model not in scores:
            continue
        score = read_mean_stderr(model_dir / "dumped_evals/eval_only", dataset)
        if score is None:
            print(f"  Warning: no results in {model_dir}")
            continue
        scores[model][int(step_match.group(1))] = score

    return scores


# MaxNLocator would need a step of ~(range/4) to land 5 ticks, which is rarely one of its
# preferred multiples, so it falls back to 3 coarse ticks. These candidates always admit one.
NICE_STEPS = [1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]


def five_ticks(lo: float, hi: float) -> tuple[list[float], tuple[float, float]]:
    """Return 5 evenly spaced round ticks spanning [lo, hi], and the axis limits to match."""
    for scale in [0.1, 1, 10, 100]:
        for step in (s * scale for s in NICE_STEPS):
            start = step * (lo // step)
            if start + 4 * step >= hi:
                return [start + i * step for i in range(5)], (start, start + 4 * step)
    raise ValueError(f"no 5-tick grid covers [{lo}, {hi}]")


def main(args: argparse.Namespace) -> None:
    LABEL_FONT_SIZE = plotting_utils.LABEL_FONT_SIZE + 6
    TICK_FONT_SIZE = plotting_utils.TICK_FONT_SIZE + 5
    LEGEND_FONT_SIZE = plotting_utils.LEGEND_FONT_SIZE + 6

    fig, axes = plt.subplots(1, len(EVAL_DATASETS), figsize=(len(EVAL_DATASETS) * 4, 5))

    for i, dataset in enumerate(EVAL_DATASETS):
        ax = axes[i]
        print(f"Collecting {dataset}")
        scores = collect_scores(dataset, args.base_eval_dir_template)

        # Llama has 5 dumped checkpoints against Qwen/Phi's 11; plot every model on the
        # steps they all share so the curves are comparable point for point.
        shared_steps = set.intersection(*(set(s) for s in scores.values() if s))
        print(f"  shared steps: {sorted(shared_steps)}")

        lo, hi = float("inf"), float("-inf")
        for model in MODELS:
            step2score = scores[model]
            if not step2score:
                continue
            steps = sorted(shared_steps)
            means = [step2score[s][0] * 100 for s in steps]
            stderrs = [step2score[s][1] * 100 for s in steps]
            ax.plot(
                steps,
                means,
                marker=MODEL2MARKER[model],
                color=MODEL2COLOR[model],
                linestyle="solid",
                linewidth=plotting_utils.LINE_WIDTH,
            )
            ax.fill_between(
                steps,
                [m - e for m, e in zip(means, stderrs, strict=True)],
                [m + e for m, e in zip(means, stderrs, strict=True)],
                alpha=0.15,
                color=MODEL2COLOR[model],
                linewidth=0,
            )
            lo = min(lo, *(m - e for m, e in zip(means, stderrs, strict=True)))
            hi = max(hi, *(m + e for m, e in zip(means, stderrs, strict=True)))

        yticks, ylim = five_ticks(lo, hi)
        ax.set_yticks(yticks)
        ax.set_ylim(ylim)

        ax.set_title(DATASET2TITLE[dataset], fontsize=LABEL_FONT_SIZE, fontweight="bold")
        ax.set_xlabel("Training steps", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(XTICKS)
        ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
        if i == 0:
            ax.set_ylabel("F1 (%)", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=plotting_utils.LINE_WIDTH)
        ax.set_axisbelow(True)

    legend_elements = [
        lines.Line2D(
            [0],
            [0],
            color=MODEL2COLOR[model],
            marker=MODEL2MARKER[model],
            label=plotting_utils.MODEL_HF2ALIAS[model],
            linewidth=plotting_utils.LINE_WIDTH,
            linestyle="solid",
        )
        for model in MODELS
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.49, 1.06),
        ncol=len(MODELS),
    )

    plt.subplots_adjust(left=0.08, right=0.9, top=0.8, bottom=0.1, wspace=0.20)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_path = figures_dir / "f1_vs_training_steps.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved plot to {save_path} and {save_path.with_suffix('.png')}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base_eval_dir_template",
        type=str,
        default="out__eval={dataset}__ret=qwen3-4b_Flat",
        help="Template for the base-model (step 0) eval run dir",
    )
    parser.add_argument("--figures_dir", type=str, default="reproduce_paper/figures")
    args = parser.parse_args()
    main(args)
