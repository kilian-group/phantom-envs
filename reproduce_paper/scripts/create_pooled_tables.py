#!/usr/bin/env -S uv run python
"""Generate the pooled-corpus robustness tables as LaTeX fragments, one per model.

Each table asks a single question: what happens when an agent must search a benchmark's
questions against a corpus 44x larger than that benchmark's own, pooled from all six and
full of interfering near-duplicate documents? Columns are the six benchmarks; rows pair
each training regime's ``Benchmark's own`` corpus score with its ``All pooled`` score, so
the degradation is read vertically within a regime and compared across regimes.

Aggregation matches ``create_result_tables.py`` exactly (see ``plotting_utils``): within a
benchmark each run contributes its mean and test-example standard error, and training
seeds are then pooled with ``combine_seeds``, which adds the between-seed spread to the
test-example noise. Both corpora are aggregated identically, so a row pair is comparable.

The two corpora come from separate eval directories, hence two input JSONs -- both written
by ``save_transfer_data.py``, which stays the only reader of the eval CSVs.

Reads:
    <figures_dir>/transfer_data.json   (per-benchmark corpora)
    <figures_dir>/pooled_data.json     (pooled corpus)

Writes:
    <figures_dir>/pooled_table_<slug>.tex
    <figures_dir>/pooled_table_<slug>.pdf   (preview)
    <figures_dir>/pooled_tables.json

Usage:
    uv run --no-sync python reproduce_paper/scripts/save_transfer_data.py
    uv run --no-sync python reproduce_paper/scripts/save_transfer_data.py \\
        --eval_dir_template "out__eval=pooled6__ret={retriever}" \\
        --benchmarks hp500,2wiki500,msq500,synthrm500,synthsm500,frames \\
        --out_name pooled_data.json
    uv run --no-sync python reproduce_paper/scripts/create_pooled_tables.py
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

# Only the models evaluated on the pooled corpus. Phi-4-mini was never run there.
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
BENCHMARKS = ["hp500", "2wiki500", "msq500", "synthrm500", "synthsm500", "frames"]

# Shortened headers, matching the other result tables.
BENCHMARK2HEADER = plotting_utils.EVAL_DATASET_ALIAS2NAME | {
    "synthrm500": "Synth-RM",
    "synthsm500": "Synth-SM",
}

# Short, dot-free stems so the fragments are \input-able by name.
MODEL2SLUG = {
    "Qwen/Qwen2.5-3B-Instruct": "qwen3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen7b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3b",
}
MODEL2MACRO = {
    "Qwen/Qwen2.5-3B-Instruct": r"\qwenthree",
    "Qwen/Qwen2.5-7B-Instruct": r"\qwenseven",
    "meta-llama/Llama-3.2-3B-Instruct": r"\llamathree",
}

# Training regimes, one per column. A None label means "use the model alias".
REGIMES = [
    {"key": "base", "label": None},
    {"key": "pw", "label": f"+ {plotting_utils.PENV_SHORT_NAME}"},
]

# One row per corpus. The pooled row is highlighted, so the eye lands on the
# deployment-scale condition.
CORPORA = [
    {"key": "own", "label": "Benchmark's own", "highlight": False},
    {"key": "pooled", "label": "All pooled", "highlight": True},
]

METRIC = "f1"
SCALE = 100.0
FMT = ".1f"

# Per-model layout, matching how the paper uses each table. \qwenseven is a wrapped float in
# the main text, so it is narrow and its caption defers the discussion to the body text; the
# others are appendix floats carrying only a title.
MODEL2LAYOUT = {
    "Qwen/Qwen2.5-3B-Instruct": {
        "float_env": "table",
        "float_args": "[!htbp]",
        "tabcolsep": "6pt",
        "full_caption": False,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "float_env": "wraptable",
        "float_args": r"{r}{0.47\linewidth}",
        "tabcolsep": "2pt",
        "full_caption": True,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "float_env": "table",
        "float_args": "[!htbp]",
        "tabcolsep": "6pt",
        "full_caption": False,
    },
}


def summarize(per_corpus: dict[str, dict], models: list[str]) -> dict:
    """Collapse raw per-seed records into ``summary[model][regime][corpus]``.

    Each leaf holds ``per_benchmark`` stats plus the ``average`` over all six, or is absent
    when that run was never evaluated. Seeds are pooled first (``combine_seeds``), then the
    benchmarks are averaged as fixed effects (``combine_benchmarks``), matching the other
    result tables so the numbers can never disagree.
    """
    summary: dict = {}
    for model in models:
        summary[model] = {}
        for regime in REGIMES:
            per_corpus_stats: dict = {}
            for corpus in CORPORA:
                data = per_corpus[corpus["key"]]
                cells = {}
                pooled_variance = []
                for bench in BENCHMARKS:
                    seed_cells = transfer_cells(data, model, regime["key"], bench, METRIC)
                    if not seed_cells:
                        continue
                    mean, variance = combine_seeds(seed_cells)
                    cells[bench] = {
                        "mean": mean * SCALE,
                        "stderr": variance**0.5 * SCALE,
                        "num_seeds": len(seed_cells),
                    }
                    pooled_variance.append((mean, variance))
                if len(pooled_variance) != len(BENCHMARKS):
                    if cells:
                        print(
                            f"  Warning: {plotting_utils.MODEL_HF2ALIAS[model]}/{regime['key']} has "
                            f"{len(pooled_variance)}/{len(BENCHMARKS)} benchmarks; skipping average"
                        )
                    continue
                mean, stderr = combine_benchmarks(pooled_variance)
                per_corpus_stats[corpus["key"]] = {
                    "per_benchmark": cells,
                    "average": {"mean": mean * SCALE, "stderr": stderr * SCALE},
                }
            summary[model][regime["key"]] = per_corpus_stats
    return summary


def corpus_rows(summary: dict, model: str) -> list[str]:
    """Emit one row per corpus, with a column per training regime holding the 6-benchmark average."""
    lines: list[str] = []
    for corpus in CORPORA:
        color = r"\rowcolor{lightorange} " if corpus["highlight"] else ""
        rendered = " & ".join(
            latex_cell(summary[model][regime["key"]].get(corpus["key"], {}).get("average"), FMT)
            for regime in REGIMES
        )
        lines.append(rf"{color}{corpus['label']} & {rendered} \\")
    return lines


def caption_for(model: str, full: bool) -> str:
    """Build the caption: a title, plus the discussion only for the main-text table."""
    lines = [
        r"    \textbf{Average F1 scores of " + MODEL2MACRO[model] + r", ablating choice of search corpus.}",
    ]
    if full:
        lines += [
            r"    Each benchmark's questions are answered either against that benchmark's own"
            r" corpus, or against a single pooled corpus of all six.",
            r"    Expectedly, agents perform worse against the pooled corpus but the drop is"
            r" slight, with or without training.",
            r"    See main text for further discussion.",
        ]
    return "\n".join([r"\caption{%", *lines, r"    }"])


def write_model_table(summary: dict, model: str, figures_dir: Path) -> Path | None:
    """Write one model's corpus-by-regime table, or None when it has no pooled results."""
    if not any(summary[model][regime["key"]] for regime in REGIMES):
        print(f"  Warning: no pooled results for {plotting_utils.MODEL_HF2ALIAS[model]}; skipping table")
        return None

    slug = MODEL2SLUG[model]
    layout = MODEL2LAYOUT[model]
    headers = [regime["label"] or plotting_utils.MODEL_HF2ALIAS[model] for regime in REGIMES]
    table = wrap_latex_table(
        corpus_rows(summary, model),
        "l" + "c" * len(REGIMES),
        " & ".join(["Corpus"] + headers),
        caption_for(model, layout["full_caption"]),
        f"tab:pooled_{slug}",
        layout["tabcolsep"],
        float_env=layout["float_env"],
        float_args=layout["float_args"],
    )
    path = figures_dir / f"pooled_table_{slug}.tex"
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

    per_corpus = {}
    for corpus, name in (("own", args.transfer_json), ("pooled", args.pooled_json)):
        path = figures_dir / name
        if not path.exists():
            raise SystemExit(f"{path} not found -- run reproduce_paper/scripts/save_transfer_data.py first")
        per_corpus[corpus] = load_transfer_data(path)

    summary = summarize(per_corpus, models)
    cache_path = figures_dir / "pooled_tables.json"
    with open(cache_path, "w") as f:
        json.dump(summary, f, indent=4)
        f.write("\n")
    print(f"Saved data to {cache_path}")

    tex_paths = [path for model in models if (path := write_model_table(summary, model, figures_dir))]
    if tex_paths and not args.no_preview:
        render_table_previews(tex_paths, figures_dir, "pooled_tables_preview")


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
        help="Per-benchmark-corpus data file under --figures_dir.",
    )
    parser.add_argument(
        "--pooled_json",
        type=str,
        default="pooled_data.json",
        help="Pooled-corpus data file under --figures_dir.",
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
