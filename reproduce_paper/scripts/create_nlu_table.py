#!/usr/bin/env -S uv run python
"""Generate the pretrained-knowledge-and-fluency table as a LaTeX fragment.

Tests the mechanism hypothesis behind the transfer result: if agentic search is largely a
procedural skill, RL fine-tuning should shape the search policy without eroding what the
base model already knows. Each model contributes three rows -- the base model and its two
fine-tunes -- measured on factual recall (PopQA SubEM), general knowledge and reasoning
(MMLU accuracy) and language-modelling fluency on Wikipedia-2018 (perplexity).

Values are absolute, not deltas, so each fine-tune is read against the base row directly
above it. Training seeds are pooled with ``combine_seeds``, which adds the between-seed
spread to each seed's own standard error; a base row has a single measurement, so its error
is that measurement's alone.

Reads:
    <nlu_dir>/{popqa,mmlu,wiki_ppl}/model=<slug>.json
    reproduce_paper/final_ckpts.yaml  (to locate each checkpoint's NLU results)

Writes:
    <figures_dir>/nlu_table.tex
    <figures_dir>/nlu_table.pdf   (preview)
    <figures_dir>/nlu_table.json

Usage:
    uv run --no-sync python reproduce_paper/scripts/create_nlu_table.py
"""

import argparse
import json
import re
from pathlib import Path

import plotting_utils
import yaml
from plotting_utils import (
    combine_seeds,
    latex_cell,
    render_table_previews,
    wrap_latex_table,
)

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-4-mini-instruct",
]
MODEL2MACRO = {
    "Qwen/Qwen2.5-3B-Instruct": r"\qwenthree",
    "Qwen/Qwen2.5-7B-Instruct": r"\qwenseven",
    "meta-llama/Llama-3.2-3B-Instruct": r"\llamathree",
    "microsoft/Phi-4-mini-instruct": r"\phimini",
}

# The two fine-tunes are highlighted; the base row above them is the reference.
TRAIN_REGIME2SECTION = {"pw": "pw_trained", "searchr1": "searchr1_own_trained"}
REGIMES = [
    {"key": "base", "label": None, "highlight": False},
    {"key": "pw", "label": rf"\quad + {plotting_utils.PENV_SHORT_NAME}", "highlight": True},
    {"key": "searchr1", "label": r"\quad + NQ+HotpotQA", "highlight": True},
]

# ``scale`` converts a stored fraction to the percentage the column header implies;
# perplexity is already in its natural units.
METRICS = [
    {
        "subdir": "popqa",
        "value_key": ("subem",),
        "stderr_key": ("subem_stderr",),
        "header": r"PopQA SubEM (\%, $\uparrow$)",
        "scale": 100.0,
        "fmt": ".2f",
    },
    {
        "subdir": "mmlu",
        "value_key": ("results", "mmlu", "acc"),
        "stderr_key": ("results", "mmlu", "acc_stderr"),
        "header": r"MMLU Acc (\%, $\uparrow$)",
        "scale": 100.0,
        "fmt": ".2f",
    },
    {
        "subdir": "wiki_ppl",
        "value_key": ("perplexity",),
        "stderr_key": ("perplexity_stderr",),
        "header": r"Wiki-18 PPL ($\downarrow$)",
        "scale": 1.0,
        "fmt": ".2f",
    },
]

TABCOLSEP = "6pt"
FLOAT_ARGS = "[!htbp]"


def nested_get(record: dict, keys: tuple[str, ...]) -> float:
    """Fetch a value from a nested dict by a tuple path, e.g. ``("results", "mmlu", "acc")``."""
    for key in keys:
        record = record[key]
    return float(record)


STEP_RE = re.compile(r"__step(\d+)$")


def base_slug(model: str) -> str:
    """Slug of an untrained model, as ``out__nlu/`` filenames spell it."""
    return model.replace("/", "--")


def ckpt_slug(metric_dir: Path, path: str, ckpt_to_eval: int) -> str | None:
    """Slug of a trained checkpoint, resolving ``ckpt_to_eval=-1`` to the latest dumped step.

    Mirrors ``phantom_envs.runs.compact_model_slug``: the run path is slugified and
    suffixed with ``__step<N>``. ``final_ckpts.yaml`` records -1 for "whatever the run
    finished at", so that case is resolved against the files actually present.
    """
    prefix = path.replace("/", "--")
    if ckpt_to_eval != -1:
        return f"{prefix}__step{ckpt_to_eval}"
    steps = sorted(
        int(match.group(1))
        for candidate in metric_dir.glob(f"model={prefix}__step*.json")
        if (match := STEP_RE.search(candidate.stem))
    )
    return f"{prefix}__step{steps[-1]}" if steps else None


def trained_entries(final_ckpts: dict, section: str, model: str) -> list[dict]:
    """Return the e5_Flat-trained checkpoint entries for one model, in seed order."""
    return [
        entry
        for entry in final_ckpts.get(section, [])
        if entry.get("model") == model
        and entry.get("path") is not None
        and entry.get("training_retriever", "e5_Flat") == "e5_Flat"
    ]


def read_metric(metric_dir: Path, slug: str, metric: dict) -> tuple[float, float] | None:
    """Read one checkpoint's ``(value, stderr)`` for one metric, or None if not evaluated."""
    path = metric_dir / f"model={slug}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return nested_get(record, metric["value_key"]), nested_get(record, metric["stderr_key"])


def collect(nlu_dir: Path, final_ckpts: dict) -> dict:
    """Return ``summary[model][regime][subdir] -> {mean, stderr, num_seeds}``.

    Missing evaluations are simply absent, which the renderer prints as an em-dash.
    """
    summary: dict = {}
    for model in MODELS:
        summary[model] = {}
        for regime in REGIMES:
            entries = (
                []
                if regime["key"] == "base"
                else trained_entries(final_ckpts, TRAIN_REGIME2SECTION[regime["key"]], model)
            )
            per_metric: dict = {}
            for metric in METRICS:
                metric_dir = nlu_dir / metric["subdir"]
                # Resolved per metric directory, since a checkpoint may be dumped for one
                # evaluation and not another.
                slugs = (
                    [base_slug(model)]
                    if regime["key"] == "base"
                    else [ckpt_slug(metric_dir, e["path"], e["ckpt_to_eval"]) for e in entries]
                )
                cells = [read_metric(metric_dir, slug, metric) if slug else None for slug in slugs]
                present = [cell for cell in cells if cell is not None]
                if not present or len(present) != len(cells):
                    if present:
                        print(
                            f"  Warning: {plotting_utils.MODEL_HF2ALIAS[model]}/{regime['key']} has "
                            f"{len(present)}/{len(cells)} seeds for {metric['subdir']}; skipping cell"
                        )
                    continue
                mean, variance = combine_seeds(present)
                per_metric[metric["subdir"]] = {
                    "mean": mean * metric["scale"],
                    "stderr": variance**0.5 * metric["scale"],
                    "num_seeds": len(present),
                }
            if per_metric:
                summary[model][regime["key"]] = per_metric
    return summary


def model_rows(summary: dict) -> list[str]:
    """Emit a base row plus the two highlighted fine-tune rows for every model."""
    lines: list[str] = []
    for model in MODELS:
        if not summary[model]:
            print(f"  Warning: no NLU results for {plotting_utils.MODEL_HF2ALIAS[model]}; skipping block")
            continue
        if lines:
            lines.append(r"\midrule")
        for regime in REGIMES:
            per_metric = summary[model].get(regime["key"], {})
            label = regime["label"] or plotting_utils.MODEL_HF2ALIAS[model]
            color = r"\rowcolor{lightorange} " if regime["highlight"] else ""
            rendered = " & ".join(
                latex_cell(per_metric.get(metric["subdir"]), metric["fmt"]) for metric in METRICS
            )
            lines.append(rf"{color}{label} & {rendered} \\")
    return lines


CAPTION = [
    r"    \textbf{RL fine-tuning leaves pretrained knowledge and fluency intact.}",
    r"    We measure parametric factual recall (PopQA SubEM metric~\citep{mallen2023popqa}),"
    r" general knowledge and reasoning (MMLU accuracy~\citep{hendrycks2020measuring}), and"
    r" language-modeling fluency on Wikipedia-2018 (perplexity, 1M passages randomly sampled"
    r" from the 21M released by~\citep{jin2025searchr1}).",
    r"    PopQA does not degrade after fine-tuning (either synthetic or NQ+HotpotQA training),"
    rf" except Phi-4-mini-instruct + {plotting_utils.PENV_SHORT_NAME} ($-2.1$ pp).",
    r"    MMLU accuracy is unchanged throughout after fine-tuning.",
    r"    Wiki-18 perplexity rises slightly after fine-tuning (1-2\% relative, and 7\% for"
    r" Phi-4-mini-instruct on synthetic).",
    r"    Learning agentic search through RL therefore largely teaches procedural skill, one"
    r" that can be learned in complement to the LLMs' memorized parametric knowledge.",
    r"    This is in line with~\citet{chen2025retaining}, who show that RL fine-tuning can"
    r" retain parametric knowledge.",
]


def main(args: argparse.Namespace) -> None:
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(args.final_ckpts_yaml) as f:
        final_ckpts = yaml.safe_load(f)
    summary = collect(Path(args.nlu_dir), final_ckpts)

    cache_path = figures_dir / "nlu_table.json"
    with open(cache_path, "w") as f:
        json.dump(summary, f, indent=4)
        f.write("\n")
    print(f"Saved data to {cache_path}")

    table = wrap_latex_table(
        model_rows(summary),
        "l" + "c" * len(METRICS),
        " & ".join(["Model"] + [metric["header"] for metric in METRICS]),
        "\n".join([r"\caption{%", *CAPTION, r"    }"]),
        "tab:nlu",
        TABCOLSEP,
        float_args=FLOAT_ARGS,
    )
    tex_path = figures_dir / "nlu_table.tex"
    tex_path.write_text(table)
    print(f"Saved {tex_path}")

    if not args.no_preview:
        render_table_previews([tex_path], figures_dir, "nlu_table_preview")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nlu_dir", type=str, default="out__nlu")
    parser.add_argument("--final_ckpts_yaml", type=str, default="reproduce_paper/final_ckpts.yaml")
    parser.add_argument(
        "--no_preview",
        action="store_true",
        help="Skip compiling the standalone preview PDF (e.g. when pdflatex is unavailable).",
    )
    parser.add_argument("--figures_dir", type=str, default="reproduce_paper/figures")
    main(parser.parse_args())
