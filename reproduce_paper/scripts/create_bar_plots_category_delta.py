#!/usr/bin/env -S uv run python
"""Per-question-category Δ-F1 (vs base) bar plots for PW / PW+Comparisons / PW+Constraints.

One figure per benchmark (CofCA, 2Wiki), each a panel per model side by side.
Within each panel: x = question categories with comparison-flavored buckets first
(comparison / bridge[_comparison] shaded); 3 bars per category for
PW / PW+Comparisons / PW+Constraints. y = mean paired Δ F1 (trained_F1 − base_F1, per
question), with paired-stderr error bars in F1 percentage points. Zero line in black.
X-tick labels include each panel's per-bucket Base F1 to surface ceiling differences
between models.

Reads per-question scores from each model's dumped_evals jsonl and question categories
from the validation parquet.

Usage:
    uv run python reproduce_paper/scripts/create_bar_plots_category_delta.py \\
        --final_ckpts_yaml reproduce_paper/final_ckpts.yaml \\
        --eval_dirs_root . \\
        --retriever qwen3-4b_Flat \\
        --figures_dir reproduce_paper/figures
"""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_utils
import yaml
from matplotlib.patches import Patch
from plotting_utils import combine_seeds, load_records

# This figure is reproduced at full text width, so every font sits two points above
# the shared defaults.
FONT_BUMP = 2
LABEL_FONT_SIZE = plotting_utils.LABEL_FONT_SIZE + FONT_BUMP
TICK_FONT_SIZE = plotting_utils.TICK_FONT_SIZE + FONT_BUMP
LEGEND_FONT_SIZE = plotting_utils.LEGEND_FONT_SIZE + FONT_BUMP
ANNOTATION_FONT_SIZE = plotting_utils.ANNOTATION_FONT_SIZE + FONT_BUMP

# --- Config: models, datasets, train conditions ---

MODELS = ["Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
EVAL_DATASETS = ["cofca500", "2wiki500"]
# Bar order: Hops, then the two extended axes, with Hops+Comparisons last so it sits
# next to the shaded comparison region it targets.
TRAIN_CONDITIONS = ["pw", "pwcons", "pwcomp"]
TRAINED_SECTIONS = {
    "pw": "pw_trained",
    "pwcons": "pwcons_trained",
    "pwcomp": "pwcomp_trained",
}

# Each benchmark's own categories are collapsed to two buckets: the comparison
# questions (shaded) and everything else. The per-category breakdown lives in the
# appendix figure; here the reader only needs the comparison / non-comparison contrast.
# Membership is per-question, so the merged bucket is a genuine pooled mean rather than
# an average of category means with unequal n.
COMPARISON_BUCKETS = {
    # CofCA's comparison set includes the questions whose label was recovered by
    # build_cofca_categories.py, not just the 208 the dataset labels directly.
    "cofca500": {"comparison"},
    # NOTE: 2Wiki's ``bridge_comparison`` stays on the non-comparison side, matching what
    # the per-category figure shaded. Move it here to count it as a comparison question.
    "2wiki500": {"comparison"},
}
COMPARISON = "comparison"
NON_COMPARISON = "non-comparison"
BUCKET_ORDER = {ds: [COMPARISON, NON_COMPARISON] for ds in EVAL_DATASETS}
REGION_LABELS = {COMPARISON: "comparison QAs", NON_COMPARISON: "non-comparison QAs"}
# Each benchmark gets its own figure file so the paper can place them independently.
FIG_SUFFIX = {"cofca500": "__cofca", "2wiki500": "__2wiki"}
# Shade the leading (comparison) bucket only.
NUM_COMPARISON_FLAVORED = {ds: 1 for ds in EVAL_DATASETS}


# --- Question-category bucketing ---


COFCA_CATEGORIES_PATH = Path("reproduce_paper/figures/cofca500_categories.json")


@lru_cache(maxsize=1)
def cofca_categories() -> dict[str, str]:
    """Return {question_id: category} for CofCA, including the recovered labels.

    CofCA drops the reasoning category for 223 of its 500 questions (``type`` stored as a
    bare hop count), which previously left them in an ``uncategorized`` bucket that is in
    fact about half comparison questions. ``build_cofca_categories.py`` recovers them.
    """
    if not COFCA_CATEGORIES_PATH.exists():
        raise SystemExit(
            f"{COFCA_CATEGORIES_PATH} not found -- run reproduce_paper/scripts/build_cofca_categories.py first"
        )
    return json.loads(COFCA_CATEGORIES_PATH.read_text())["categories"]


def cofca_bucket(meta: dict) -> str:
    """Return the cofca question category, looked up by question id."""
    return cofca_categories()[meta["id"]]


def twowiki_bucket(meta: dict) -> str:
    """Return 2wiki question category from the metadata.type string."""
    return meta.get("type") if isinstance(meta, dict) else None


BUCKET_FN = {"cofca500": cofca_bucket, "2wiki500": twowiki_bucket}


def merged_bucket(eval_ds: str, meta: dict) -> str:
    """Collapse a question's native category into ``comparison`` / ``non-comparison``."""
    return COMPARISON if BUCKET_FN[eval_ds](meta) in COMPARISON_BUCKETS[eval_ds] else NON_COMPARISON


# --- Checkpoint dir resolution ---


def find_eval_dir(eval_root: Path, train_path: str, ckpt_to_eval: int) -> Path | None:
    """Locate the dumped_evals dir for a trained checkpoint under eval_root.

    eval_root is e.g. ``out__eval=cofca500__ret=qwen3-4b_Flat``. Each child looks like
    ``model=<slug>__step<N>``; ``slug`` is ``train_path`` with the leading component
    stripped and ``/`` → ``--``.
    """
    suffix = "/".join(train_path.split("/")[1:])
    slug = suffix.replace("/", "--")
    candidates = [d for d in eval_root.iterdir() if d.is_dir() and slug in d.name]
    if not candidates:
        return None
    if ckpt_to_eval == -1:

        def step(d: Path) -> int:
            m = re.search(r"__step(\d+)$", d.name)
            return int(m.group(1)) if m else -1

        return max(candidates, key=step)
    target = f"__step{ckpt_to_eval}"
    for d in candidates:
        if d.name.endswith(target):
            return d
    return None


def load_scores(jsonl_path: Path) -> list[float]:
    """Return per-question final scores (last entry of the score list) from a dumped_evals jsonl."""
    return [r["score"][-1] for r in load_records(jsonl_path)]


# --- Data assembly ---


def collect_deltas(
    final_ckpts_yaml: dict,
    eval_dirs_root: Path,
    retriever: str,
) -> dict:
    """Return nested dict per (model, eval_ds) with bucket counts, per-bucket Base F1
    means, and per-train-condition (mean Δ, stderr Δ) per bucket.

    Every training seed present for a (model, train condition) is pooled per bucket by
    :func:`combine_seeds`, so a bar shows the seed average and its error bar carries the
    between-seed spread rather than whichever seed happened to be listed last.

    Shape::

        out[model][eval_ds] = {
            "bucket_counts": {bucket: int, ...},
            "base_f1": {bucket: float, ...},                 # per-bucket Base F1 (0..1)
            "train": {train_ds: {bucket: (mean_delta, stderr_delta)}},
        }
    """
    base_path_by_model = {e["model"]: e["model"] for e in final_ckpts_yaml.get("base_models", [])}
    trained_lookup: dict[tuple[str, str], list[dict]] = {}
    for train_ds, section in TRAINED_SECTIONS.items():
        for entry in final_ckpts_yaml.get(section, []):
            if entry.get("training_retriever", "e5_Flat") != "e5_Flat":
                continue
            if entry.get("path") is None:
                continue
            trained_lookup.setdefault((entry["model"], train_ds), []).append(entry)

    out: dict = {}
    for model in MODELS:
        out[model] = {}
        for eval_ds in EVAL_DATASETS:
            eval_root = eval_dirs_root / f"out__eval={eval_ds}__ret={retriever}"

            df = pd.read_parquet(f"data/{eval_ds}/validation.parquet")
            buckets_per_q = np.array([merged_bucket(eval_ds, m) for m in df["metadata"]])

            base_dir = eval_root / plotting_utils.model_hf_to_key(base_path_by_model[model])
            base_jsonl = base_dir / "dumped_evals" / "eval_only" / f"{eval_ds}.jsonl"
            if not base_jsonl.exists():
                print(f"Warning: missing base jsonl: {base_jsonl}")
                continue
            base_scores = np.array(load_scores(base_jsonl))

            unique_buckets = list(set(buckets_per_q))
            bucket_counts = {b: int((buckets_per_q == b).sum()) for b in unique_buckets}
            base_f1 = {b: float(base_scores[buckets_per_q == b].mean()) for b in unique_buckets}

            entry_data = {"bucket_counts": bucket_counts, "base_f1": base_f1, "train": {}}

            for train_ds in TRAIN_CONDITIONS:
                entries = trained_lookup.get((model, train_ds), [])
                if not entries:
                    print(f"Warning: no {train_ds} ckpt for {model}")
                    continue
                per_seed_stats = []
                for entry in entries:
                    tdir = find_eval_dir(eval_root, entry["path"], entry.get("ckpt_to_eval", -1))
                    if tdir is None:
                        print(f"Warning: no eval dir for {model} {train_ds} {entry['path']}")
                        continue
                    tjsonl = tdir / "dumped_evals" / "eval_only" / f"{eval_ds}.jsonl"
                    if not tjsonl.exists():
                        print(f"Warning: missing trained jsonl: {tjsonl}")
                        continue
                    trained_scores = np.array(load_scores(tjsonl))
                    if len(trained_scores) != len(base_scores):
                        print(
                            f"Warning: length mismatch on {model} {train_ds} {eval_ds}: "
                            f"base={len(base_scores)} trained={len(trained_scores)}"
                        )
                        continue
                    deltas = trained_scores - base_scores
                    stats = {}
                    for bucket in unique_buckets:
                        d = deltas[buckets_per_q == bucket]
                        if len(d) == 0:
                            continue
                        mean = float(d.mean())
                        stderr = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
                        stats[bucket] = (mean, stderr)
                    per_seed_stats.append(stats)
                if not per_seed_stats:
                    continue
                # Pool the seeds per bucket: within-seed question noise plus between-seed spread.
                pooled = {}
                for bucket in unique_buckets:
                    cells = [stats[bucket] for stats in per_seed_stats if bucket in stats]
                    if not cells:
                        continue
                    mean, variance = combine_seeds(cells)
                    pooled[bucket] = (mean, variance**0.5)
                entry_data["train"][train_ds] = pooled
            out[model][eval_ds] = entry_data
    return out


# --- Plot ---


BAR_WIDTH = 0.22
BAR_SPACING = 0.28
SHADE_COLOR = "lightgray"
SHADE_ALPHA = 0.35


def draw_panel(
    ax: plt.Axes,
    eval_ds: str,
    panel_data: dict,
) -> None:
    """Draw one (model, dataset) panel with shading on comparison-flavored buckets."""
    ordered_buckets = BUCKET_ORDER[eval_ds]
    base_f1 = panel_data["base_f1"]
    train_stats = panel_data["train"]

    bucket_centers = np.arange(len(ordered_buckets))
    n_train = len(TRAIN_CONDITIONS)
    offsets = (np.arange(n_train) - (n_train - 1) / 2) * BAR_SPACING

    # Shading for comparison-flavored buckets (drawn first so bars sit on top).
    n_shaded = NUM_COMPARISON_FLAVORED[eval_ds]
    if n_shaded > 0:
        ax.axvspan(
            -0.5,
            n_shaded - 0.5,
            color=SHADE_COLOR,
            alpha=SHADE_ALPHA,
            zorder=0,
            linewidth=0,
        )

    for i, train_ds in enumerate(TRAIN_CONDITIONS):
        means, errs = [], []
        for bucket in ordered_buckets:
            mean, err = train_stats.get(train_ds, {}).get(bucket, (np.nan, 0.0))
            means.append(mean * 100.0)
            errs.append(err * 100.0)
        color = plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[train_ds]]
        ax.bar(
            bucket_centers + offsets[i],
            means,
            BAR_WIDTH,
            yerr=errs,
            color=color,
            edgecolor=plotting_utils.EDGE_COLOR,
            linewidth=plotting_utils.LINE_WIDTH,
            error_kw={"linewidth": plotting_utils.LINE_WIDTH, "ecolor": "black", "capsize": 3},
            zorder=2,
        )

    ax.axhline(0, color="black", linewidth=plotting_utils.LINE_WIDTH, zorder=3)
    ax.set_xlim(-0.5, len(ordered_buckets) - 0.5)
    ax.set_xticks(bucket_centers)
    ax.set_xticklabels(
        [f"Base F1 = {base_f1[b] * 100:.0f}%" for b in ordered_buckets],
        fontsize=TICK_FONT_SIZE - 1,
    )
    for i, bucket in enumerate(ordered_buckets):
        ax.annotate(
            REGION_LABELS[bucket],
            (i, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -6),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=TICK_FONT_SIZE,
            fontweight="bold",
        )
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=plotting_utils.LINE_WIDTH)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(plotting_utils.LINE_WIDTH)


def make_figure(data: dict, eval_ds: str, figures_dir: Path) -> None:
    """Build one benchmark's figure (a panel per model, side by side) and save it."""
    fig, axes = plt.subplots(
        1,
        len(MODELS),
        figsize=(5.4 * len(MODELS), 3.9),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    # Shared y-limits across this benchmark's model panels (bar plus its stderr).
    y_all: list[float] = []
    for model in MODELS:
        train_stats = data.get(model, {}).get(eval_ds, {}).get("train", {})
        for stats in train_stats.values():
            for mean, err in stats.values():
                y_all.extend([(mean + err) * 100, (mean - err) * 100])
    y_pad = 3.0
    y_lo = min(y_all) - y_pad if y_all else -10
    y_hi = max(y_all) + y_pad if y_all else 30
    # Reserve a band at the top for the region labels so they never sit on a bar.
    y_hi += 0.16 * (y_hi - y_lo)

    for j, model in enumerate(MODELS):
        ax = axes[j]
        entry = data.get(model, {}).get(eval_ds)
        if entry is None:
            ax.set_visible(False)
            continue
        draw_panel(ax, eval_ds, entry)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            plotting_utils.MODEL_HF2ALIAS[model],
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
        )
        if j == 0:
            ax.set_ylabel(
                r"Trained $-$ Base F1 (pp)",
                fontsize=TICK_FONT_SIZE,
                labelpad=0,
            )
            ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    legend_handles = [
        Patch(
            facecolor=plotting_utils.COLORS2HEX[plotting_utils.TRAIN_DATASET_ALIAS2COLOR[t]],
            edgecolor="black",
            label=plotting_utils.TRAIN_DATASET_ALIAS2NAME[t],
        )
        for t in TRAIN_CONDITIONS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        fontsize=LEGEND_FONT_SIZE - 2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(TRAIN_CONDITIONS),
    )

    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        save_path = figures_dir / f"f1_category_delta_pw_variants{FIG_SUFFIX[eval_ds]}{suffix}"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(args.final_ckpts_yaml) as f:
        final_ckpts_yaml = yaml.safe_load(f)

    data = collect_deltas(final_ckpts_yaml, Path(args.eval_dirs_root), args.retriever)
    for eval_ds in EVAL_DATASETS:
        make_figure(data, eval_ds, figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--final_ckpts_yaml", type=str, default="reproduce_paper/final_ckpts.yaml")
    parser.add_argument("--eval_dirs_root", type=str, default=".")
    parser.add_argument("--retriever", type=str, default="qwen3-4b_Flat")
    parser.add_argument("--figures_dir", type=str, default="reproduce_paper/figures")
    main(parser.parse_args())
