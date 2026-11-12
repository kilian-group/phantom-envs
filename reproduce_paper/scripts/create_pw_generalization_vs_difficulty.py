#!/usr/bin/env -S uv run python
"""Plot PW in-domain training evolution + out-of-domain generalization.

Produces a 2×(2*n_models) figure (saved as pdf + png). For each model, two columns:
  Col A — In-Domain training evolution: 6 lines (base step 0 gray dashed,
          plus 5 evenly-spaced training-step lines colored by white→orange gradient).
  Col B — Final-checkpoint Unseen comparison: 3 lines
          (ID size 1K orange, Unseen size 1K blue, Unseen size 10K brown).
          All three are EVAL-time runs sharing one retriever and turn budget. The ID line
          comes from out__eval=pw1__ret=<retriever>, not the training-time dumps: those use
          the training retriever (e5_Flat), and mixing retrievers makes the Unseen universes
          look better than the training universe purely from stronger retrieval.

Top row is F1, bottom row is # Search Calls. Data sourced from pw.jsonl files:
  - Evolution / ID: <run_path>/exports/dumped_evals/global_step_<N>_evals/pw.jsonl
  - Unseen-1k:         out__eval=pw2__ret=<retriever>/model=.../dumped_evals/eval_only/pw.jsonl
  - Unseen-10k:        out__eval=pwOOD__ret=<retriever>/model=.../dumped_evals/eval_only/pw.jsonl

Writes figure to <figures_dir>/pw_generalization_vs_difficulty<suffix>.{pdf,png} and a
JSON data cache alongside it.

``--models`` picks which pair of models to draw, so the main-text and appendix variants are
two invocations rather than an edit to this file. ``--seed`` picks which training run to
plot: these panels show one run's trajectory, not a seed average, so the seed is explicit.

Usage:
    # Main text: Qwen2.5-7B + Llama-3.2-3B
    uv run --no-sync python reproduce_paper/scripts/create_pw_generalization_vs_difficulty.py

    # Appendix: Qwen2.5-3B + Phi-4-mini
    uv run --no-sync python reproduce_paper/scripts/create_pw_generalization_vs_difficulty.py \\
        --models Qwen/Qwen2.5-3B-Instruct,microsoft/Phi-4-mini-instruct \\
        --fig_suffix _qwen3b_phi
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import plotting_utils
import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from plotting_utils import extract_answer, load_records, num_searches, seed_from_path

from phantom_envs.scoring import PhantomScorer

# Default is the main-text pair; the appendix pair is passed via --models.
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

CONDITIONS = [
    ("id", "Training Universe (1K)", plotting_utils.COLORS2HEX["myOrange"], "-"),
    ("ood1k", "Unseen Universe (1K)", plotting_utils.COLORS2HEX["myBlue"], "-"),
    ("ood10k", "Unseen Universe (10K)", plotting_utils.COLORS2HEX["myGreen"], "-"),
]

METRICS = [
    ("f1", "F1 (%)", [0, 20, 40, 60, 80]),
    ("searches", "# Search Calls", [1, 3, 5, 7, 9]),
]

MAX_DIFFICULTY = 7

# Number of evenly-spaced (post-base) training-step lines to draw in evolution panels.
N_EVOLUTION_STEPS = 5

# White → orange gradient for evolution lines (PW training color).
EVO_COLORMAP = LinearSegmentedColormap.from_list(
    "WhiteToOrange",
    [(0, "white"), (1, plotting_utils.COLORS2HEX["myOrange"])],
    N=256,
)

FIG_STEM = "pw_generalization_vs_difficulty"

# Local font-size overrides (shared values are too small relative to subplot size).
LABEL_FS = plotting_utils.LABEL_FONT_SIZE + 4
TICK_FS = plotting_utils.TICK_FONT_SIZE + 3
LEGEND_FS = plotting_utils.LEGEND_FONT_SIZE + 3
TITLE_FS = LABEL_FS + 2


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def find_id_jsonl(run_path: str, ckpt_to_eval: int) -> Path | None:
    """Return the pw.jsonl from the training run's last (or specified) checkpoint dump."""
    base = Path(run_path) / "exports" / "dumped_evals"
    if not base.exists():
        return None
    if ckpt_to_eval == -1:
        step_dirs = [d for d in base.iterdir() if re.match(r"global_step_\d+_evals", d.name)]
        if not step_dirs:
            return None
        best = max(step_dirs, key=lambda d: int(re.search(r"\d+", d.name).group()))
    else:
        best = base / f"global_step_{ckpt_to_eval}_evals"
    jsonl = best / "pw.jsonl"
    return jsonl if jsonl.exists() else None


def find_evolution_jsonls(run_path: str) -> dict[int, Path]:
    """Return {step: pw.jsonl} for every dumped checkpoint eval under run_path."""
    base = Path(run_path) / "exports" / "dumped_evals"
    out: dict[int, Path] = {}
    if not base.exists():
        return out
    for d in base.iterdir():
        m = re.match(r"global_step_(\d+)_evals", d.name)
        if not m:
            continue
        jsonl = d / "pw.jsonl"
        if jsonl.exists():
            out[int(m.group(1))] = jsonl
    return out


def select_evolution_steps(all_steps: list[int]) -> list[int]:
    """Pick the closest available step at each percentile in [20, 40, 60, 80, 100]% of max."""
    nonzero = sorted(s for s in all_steps if s > 0)
    if not nonzero:
        return []
    max_step = nonzero[-1]
    selected: list[int] = []
    for i in range(1, N_EVOLUTION_STEPS + 1):
        target = (i / N_EVOLUTION_STEPS) * max_step
        closest = min(nonzero, key=lambda s: abs(s - target))
        if closest not in selected:
            selected.append(closest)
    return selected


def find_ood_jsonl(eval_dir: Path, run_path: str, ckpt_to_eval: int) -> Path | None:
    """Return pw.jsonl from an out__eval dir matching the run path + checkpoint."""
    path_slug = "/".join(run_path.split("/")[1:])
    path_slug = path_slug.replace("/", "--")
    matches = [d for d in eval_dir.iterdir() if d.is_dir() and path_slug in d.name]
    if not matches:
        return None
    if ckpt_to_eval == -1:
        best = max(
            matches, key=lambda d: int(m.group(1)) if (m := re.search(r"__step(\d+)$", d.name)) else -1
        )
    else:
        matches = [d for d in matches if d.name.endswith(f"__step{ckpt_to_eval}")]
        if not matches:
            return None
        best = matches[0]
    jsonl = best / "dumped_evals" / "eval_only" / "pw.jsonl"
    return jsonl if jsonl.exists() else None


def collect_by_difficulty(jsonl_path: Path) -> dict[int, dict[str, list[float]]]:
    """Parse pw.jsonl into per-difficulty lists of f1 and search-call counts."""
    by_diff: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"f1": [], "searches": []})
    for record in load_records(jsonl_path):
        try:
            diff = record["env_extras"]["metadata"]["difficulty"]
            gts = record["env_extras"]["reward_spec"]["ground_truth"]["target"]
        except (KeyError, TypeError):
            continue
        pred_raw = extract_answer(record.get("output_response", ""))
        preds = PhantomScorer.parse_answer(pred_raw) if pred_raw else []
        by_diff[diff]["f1"].append(PhantomScorer.f1(preds, gts))
        by_diff[diff]["searches"].append(num_searches(record))
    return dict(by_diff)


def summarise(by_diff: dict[int, dict[str, list[float]]]) -> dict[str, dict[int, tuple[float, float]]]:
    """Return {metric: {difficulty: (mean, stderr)}} from collect_by_difficulty output."""
    out: dict[str, dict[int, tuple[float, float]]] = {"f1": {}, "searches": {}}
    for diff, vals in by_diff.items():
        for metric, data in vals.items():
            arr = np.asarray(data, dtype=float)
            mean = float(arr.mean())
            stderr = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            out[metric][diff] = (mean, stderr)
    return out


def load_all_data(
    final_ckpts_yaml: dict,
    eval_dirs_root: Path,
    retriever: str,
    models: list[str],
    seed: int,
    ood1k_override: Path | None = None,
    ood10k_override: Path | None = None,
    id_override: Path | None = None,
) -> dict:
    """Load per-condition summaries for each model.

    Args:
        final_ckpts_yaml: parsed contents of final_ckpts.yaml.
        models: HuggingFace model names to plot, in column order.
        seed: which training seed's run to trace.
        ood1k_override: explicit Unseen-1K eval dir, for runs whose evals were archived
            elsewhere (the e5_Flat pw2 evals live under ``out__old/``).
        ood10k_override: explicit Unseen-10K eval dir.
        id_override: explicit training-universe eval dir (``pw1``).
        eval_dirs_root: root containing out__eval=*__ret=* dirs.
        retriever: retriever suffix (e.g. "qwen3-4b_Flat").

    Returns:
        Nested dict: data[model][condition] = {metric: {difficulty: (mean, stderr)}}.
    """
    id_dir = id_override or eval_dirs_root / f"out__eval=pw1__ret={retriever}"
    ood1k_dir = ood1k_override or eval_dirs_root / f"out__eval=pw2__ret={retriever}"
    ood10k_dir = ood10k_override or eval_dirs_root / f"out__eval=pwOOD__ret={retriever}"

    data: dict = {m: {} for m in models}

    # These panels trace a single run over training, so pick one seed explicitly rather
    # than letting a later entry silently win (and never average two trajectories).
    pw_entries: dict = {}
    for e in final_ckpts_yaml.get("pw_trained", []):
        if (
            e["model"] not in models
            or e.get("path") is None
            or e.get("training_retriever", "e5_Flat") != "e5_Flat"
        ):
            continue
        if seed_from_path(e["path"]) == seed:
            pw_entries[e["model"]] = e

    for model in models:
        entry = pw_entries.get(model)
        if entry is None:
            print(f"Warning: no e5_Flat pw_trained entry for {model} at seed {seed}")
            continue

        run_path = entry["path"]
        ckpt = entry.get("ckpt_to_eval", -1)
        alias = plotting_utils.MODEL_HF2ALIAS[model]

        # Evolution: every checkpoint dump under <run_path>/exports/dumped_evals
        evo_jsonls = find_evolution_jsonls(run_path)
        if evo_jsonls:
            selected = select_evolution_steps(list(evo_jsonls))
            steps_to_load = ([0] if 0 in evo_jsonls else []) + selected
            evo_summary: dict[int, dict] = {}
            for step in steps_to_load:
                print(f"  Loading {alias} evolution step {step}: {evo_jsonls[step]}")
                evo_summary[step] = summarise(collect_by_difficulty(evo_jsonls[step]))
            data[model]["evolution"] = evo_summary
        else:
            print(f"  Warning: no evolution dumps found for {alias}")

        id_jsonl = find_ood_jsonl(id_dir, run_path, ckpt) if id_dir.exists() else None
        if id_jsonl is None:
            id_jsonl = find_id_jsonl(run_path, ckpt)
            if id_jsonl is not None:
                print(
                    f"  Warning: no {id_dir} entry for {alias}; falling back to the training-time "
                    "validation dumps, which use the TRAINING retriever (e5_Flat). The Unseen "
                    "panels then mix retrievers and are not comparable to this line."
                )

        for cond, label, jsonl_fn in [
            ("id", "ID", id_jsonl),
            ("ood1k", "OOD-1k", find_ood_jsonl(ood1k_dir, run_path, ckpt) if ood1k_dir.exists() else None),
            (
                "ood10k",
                "OOD-10k",
                find_ood_jsonl(ood10k_dir, run_path, ckpt) if ood10k_dir.exists() else None,
            ),
        ]:
            if jsonl_fn is None:
                print(f"  Warning: {label} jsonl not found for {alias}")
                continue
            print(f"  Loading {alias} {label}: {jsonl_fn}")
            data[model][cond] = summarise(collect_by_difficulty(jsonl_fn))

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def format_axis(
    ax: plt.Axes,
    metric: str,
    yticks: list,
    show_ylabel: bool,
    show_yticks: bool,
    show_xlabel: bool,
    title: str | None,
) -> None:
    """Apply shared axis formatting (ticks, labels, grid, spines, title)."""
    ax.set_xticks(range(1, MAX_DIFFICULTY + 1))
    ax.set_xlim(1, MAX_DIFFICULTY)
    ax.set_yticks(yticks)
    ax.set_ylim(0, max(yticks))

    tick_labels = [f"{v:.1f}" if isinstance(v, float) else str(v) for v in yticks]
    if show_yticks:
        ax.set_yticklabels(tick_labels, fontsize=TICK_FS)
    else:
        ax.set_yticklabels([])

    if show_ylabel:
        ylabel = next(lbl for m, lbl, _ in METRICS if m == metric)
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)

    if show_xlabel:
        ax.set_xlabel("Question Difficulty", fontsize=LABEL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS, labelbottom=show_xlabel)

    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=plotting_utils.LINE_WIDTH)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(plotting_utils.LINE_WIDTH)
    if title:
        ax.set_title(title, fontsize=LABEL_FS, fontweight="bold")


def plot_ood_subplot(
    ax: plt.Axes,
    data: dict,  # {condition: {metric: {difficulty: (mean, stderr)}}}
    metric: str,
    **fmt_kwargs,
) -> None:
    """Draw one Unseen-comparison cell with shaded stderr bands."""
    scale = 100.0 if metric == "f1" else 1.0
    for cond, label, color, ls in CONDITIONS:
        if cond not in data or metric not in data[cond]:
            continue
        metric_data = {d: v for d, v in data[cond][metric].items() if d <= MAX_DIFFICULTY}
        diffs = sorted(metric_data)
        means = [metric_data[d][0] * scale for d in diffs]
        stderrs = [metric_data[d][1] * scale for d in diffs]
        ax.plot(diffs, means, color=color, linestyle=ls, linewidth=1.5, label=label, marker=".")
        ax.fill_between(
            diffs,
            [m - s for m, s in zip(means, stderrs, strict=False)],
            [m + s for m, s in zip(means, stderrs, strict=False)],
            color=color,
            alpha=0.2,
            linewidth=0,
        )
    format_axis(ax, metric, **fmt_kwargs)


def plot_evolution_subplot(
    ax: plt.Axes,
    evolution: dict,  # {step: {metric: {difficulty: (mean, stderr)}}}
    metric: str,
    max_step: int,
    **fmt_kwargs,
) -> None:
    """Draw one in-domain training-evolution cell. Step 0 → gray dashed; rest → orange gradient."""
    scale = 100.0 if metric == "f1" else 1.0
    for step in sorted(evolution):
        if metric not in evolution[step]:
            continue
        metric_data = {d: v for d, v in evolution[step][metric].items() if d <= MAX_DIFFICULTY}
        diffs = sorted(metric_data)
        means = [metric_data[d][0] * scale for d in diffs]
        if step == 0:
            color = "darkgray"
            ls = plotting_utils.BASE_LINE_STYLE
            lw = 1.5
            label = "Base"
        else:
            color = EVO_COLORMAP(step / max_step) if max_step > 0 else EVO_COLORMAP(1.0)
            ls = "-"
            lw = 1.5
            label = None
        ax.plot(diffs, means, color=color, linestyle=ls, linewidth=lw, label=label)
    format_axis(ax, metric, **fmt_kwargs)


def draw_figure(data: dict, models: list[str], figures_dir: Path, fig_suffix: str) -> None:
    """Single gridspec with title/colorbar/body/legend rows so y-axes align across halves."""
    n_rows = len(METRICS)
    n_models = len(models)
    n_cols = 2 * n_models

    fig = plt.figure(figsize=(4.0 * n_cols, 3.3 * n_rows + 1.6))
    outer_gs = fig.add_gridspec(
        nrows=3,
        ncols=n_cols,
        height_ratios=[0.16, n_rows * 1.0, 0.55],
        hspace=0.30,
        left=0.05,
        right=0.99,
    )

    for j, title in enumerate(
        [
            "Performance evolution on Training Universe",
            "Performance on Unseen Universes",
        ]
    ):
        title_ax = fig.add_subplot(outer_gs[0, j * n_models : (j + 1) * n_models])
        title_ax.axis("off")
        title_ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=TITLE_FS, fontweight="bold")

    body_gs = outer_gs[1, :].subgridspec(n_rows, n_cols, hspace=0.30, wspace=0.05)
    left_axes = np.array([[fig.add_subplot(body_gs[r, c]) for c in range(n_models)] for r in range(n_rows)])
    right_axes = np.array(
        [[fig.add_subplot(body_gs[r, n_models + c]) for c in range(n_models)] for r in range(n_rows)]
    )

    # Push base legend / colorbar / OOD legend to the left, leaving whitespace on the right.
    bottom_gs = outer_gs[2, :].subgridspec(1, 6, width_ratios=[0.5, 0.05, 1.6, 0.3, 1.4, 0.1], wspace=0.0)
    legend_left_ax = fig.add_subplot(bottom_gs[0, 0])
    legend_left_ax.axis("off")
    cbar_holder = bottom_gs[0, 2].subgridspec(
        3,
        1,
        height_ratios=[1.0, 0.35, 1.5],
        hspace=0.0,
    )
    cbar_ax = fig.add_subplot(cbar_holder[1, 0])
    legend_right_ax = fig.add_subplot(bottom_gs[0, 4])
    legend_right_ax.axis("off")

    global_max_step = 0
    for model in models:
        evo = data.get(model, {}).get("evolution", {})
        if evo:
            global_max_step = max(global_max_step, max(evo))

    for mi, model in enumerate(models):
        model_data = data.get(model, {})
        alias = plotting_utils.MODEL_HF2ALIAS[model]
        evolution = model_data.get("evolution", {})
        evo_max_step = max(evolution) if evolution else 0

        for row, (metric, _, yticks) in enumerate(METRICS):
            show_xlabel = row == n_rows - 1
            plot_evolution_subplot(
                left_axes[row, mi],
                evolution,
                metric=metric,
                max_step=evo_max_step,
                yticks=yticks,
                show_ylabel=(mi == 0),
                show_yticks=(mi == 0),
                show_xlabel=show_xlabel,
                title=alias if row == 0 else None,
            )
            plot_ood_subplot(
                right_axes[row, mi],
                model_data,
                metric=metric,
                yticks=yticks,
                # NOTE: hide y-axis labels and ticks for OOD subplots, they are in the leftmost subplot
                show_ylabel=False,
                show_yticks=False,
                show_xlabel=show_xlabel,
                title=alias if row == 0 else None,
            )

    if global_max_step > 0:
        norm = Normalize(vmin=0, vmax=global_max_step)
        sm = ScalarMappable(cmap=EVO_COLORMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Training Steps", fontsize=LABEL_FS)
        cbar.set_ticks(np.linspace(0, global_max_step, 6))
        cbar.ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{int(round(100 * x / global_max_step))}%")
        )
        cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.ax.xaxis.set_label_position("bottom")
        cbar.ax.tick_params(labelsize=TICK_FS, labelcolor="black")
        cbar.outline.set_visible(False)
    else:
        cbar_ax.axis("off")

    evo_handles, evo_labels = left_axes[0, 0].get_legend_handles_labels()
    if evo_handles:
        legend_left_ax.legend(
            evo_handles,
            evo_labels,
            loc="center",
            fontsize=LEGEND_FS,
            frameon=False,
            ncol=len(evo_handles),
        )
    ood_handles, ood_labels = right_axes[0, 0].get_legend_handles_labels()
    if ood_handles:
        legend_right_ax.legend(
            ood_handles,
            ood_labels,
            loc="center left",
            bbox_to_anchor=(0.0, 0.35),
            fontsize=LEGEND_FS,
            frameon=False,
            ncol=1,
            handlelength=1.8,
            handletextpad=0.5,
            labelspacing=0.3,
        )

    for suffix in (".pdf", ".png"):
        path = figures_dir / f"{FIG_STEM}{fig_suffix}{suffix}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close()


def main(args: argparse.Namespace) -> None:
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(args.final_ckpts_yaml) as f:
        final_ckpts_yaml = yaml.safe_load(f)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in plotting_utils.MODEL_HF2ALIAS]
    if unknown:
        raise SystemExit(f"Unknown model name(s): {unknown}")
    data = load_all_data(
        final_ckpts_yaml,
        Path(args.eval_dirs_root),
        args.retriever,
        models,
        args.seed,
        Path(args.ood1k_dir) if args.ood1k_dir else None,
        Path(args.ood10k_dir) if args.ood10k_dir else None,
        Path(args.id_dir) if args.id_dir else None,
    )

    def serialise_metric_block(block: dict) -> dict:
        return {
            metric: {str(d): (m, s) for d, (m, s) in diff_data.items()} for metric, diff_data in block.items()
        }

    cache: dict = {}
    for model, cond_data in data.items():
        cache[model] = {}
        for cond, payload in cond_data.items():
            if cond == "evolution":
                cache[model][cond] = {str(step): serialise_metric_block(b) for step, b in payload.items()}
            else:
                cache[model][cond] = serialise_metric_block(payload)
    cache_path = figures_dir / f"{FIG_STEM}{args.fig_suffix}.json"
    cache_path.write_text(json.dumps(cache, indent=4) + "\n")
    print(f"Saved data to {cache_path}")

    draw_figure(data, models, figures_dir, args.fig_suffix)


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
        help="Retriever suffix for the pw1/pw2/pwOOD eval dirs. All three right-hand "
        "conditions must share it: they are eval-time runs, and mixing retrievers makes "
        "the Unseen universes look better than the training universe purely from stronger "
        "retrieval.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated HuggingFace model names to draw, in column order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Training seed whose run to trace. These panels show one run, not an average.",
    )
    parser.add_argument(
        "--id_dir",
        type=str,
        default=None,
        help="Explicit training-universe eval dir, overriding out__eval=pw1__ret=<retriever>.",
    )
    parser.add_argument(
        "--ood1k_dir",
        type=str,
        default=None,
        help="Explicit Unseen-1K eval dir, overriding out__eval=pw2__ret=<retriever>.",
    )
    parser.add_argument(
        "--ood10k_dir",
        type=str,
        default=None,
        help="Explicit Unseen-10K eval dir, overriding out__eval=pwOOD__ret=<retriever>.",
    )
    parser.add_argument(
        "--fig_suffix",
        type=str,
        default="_qwen7b_llama",
        help="Suffix appended to the figure/cache stem, naming the model pair drawn. "
        "Defaults to the main-text pair; pass '_qwen3b_phi' with the appendix --models.",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="reproduce_paper/figures",
        help="Directory to write figures and data cache.",
    )
    main(parser.parse_args())
