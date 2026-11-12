"""Shared plotting styles, color palettes, name mappings and helpers for the scripts in reproduce_paper/scripts/.

The scripts import it as a local module (``import plotting_utils``), as scripts/skyrl/ does with
``common``. Answer extraction, search counts and standard errors come from ``phantom_envs``.
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phantom_envs.protocol import count_searches, extract_answer  # noqa: F401 (used by the scripts)
from phantom_envs.scoring import mean_stderr  # noqa: F401 (used by the scripts)

plt.rcParams.update({"font.family": "Lato"})

COLORS2HEX = {
    "myGreen": "#3B9B7B",
    "myOrange": "#D96831",
    "myPurple": "#8B5FA8",
    "myBlue": "#3B8FBF",
    "myYellow": "#E8A93C",
    "myBrown": "#8B5A3C",
    "myDarkGray": "#A9A9A9",
}

# --- Dataset aliases → display names ---

EVAL_DATASETS = ["hp500", "2wiki500", "msq500", "cofca500", "synthrm500", "synthsm500", "frames"]
EVAL_DATASET_ALIAS2NAME = {
    "hp500": "HotpotQA",
    "2wiki500": "2Wiki",
    "msq500": "MuSiQue",
    "cofca500": "CofCA",
    "synthrm500": "SynthWorlds-RM",
    "synthsm500": "SynthWorlds-SM",
    "frames": "FRAMES",
}

TRAIN_DATASET_ALIAS2NAME = {
    "base": "Base",
    "searchr1": "NQ+HotpotQA",
    "pw": "Hops",
    "pwcons": "Hops+Constraints",
    "pwcomp": "Hops+Comparisons",
}

# Paper-facing name for the (Hops) synthetic training regime in tables and plots.
PENV_SHORT_NAME = "PhantomEnvs"

# --- Model name mappings ---

MODEL_HF2ALIAS = {
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
}

# --- Colors per training dataset ---

TRAIN_DATASET_ALIAS2COLOR = {
    "base": "myDarkGray",
    "searchr1": "myBlue",
    "pw": "myOrange",
    "pwcons": "myYellow",
    "pwcomp": "myPurple",
}
TRAIN_DATASET_ALIAS2TEXT_COLOR = {
    "base": COLORS2HEX["myDarkGray"],
    "searchr1": COLORS2HEX["myBlue"],
    "pw": COLORS2HEX["myOrange"],
    "pwcons": COLORS2HEX["myYellow"],
    "pwcomp": COLORS2HEX["myPurple"],
}

# --- Base (untrained) bar style ---

BASE_COLOR = COLORS2HEX["myDarkGray"]
BASE_LINE_WIDTH = 2
BASE_LINE_STYLE = "dashed"

# --- General figure style ---

LINE_WIDTH = 1
EDGE_COLOR = "black"

LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 15
ANNOTATION_FONT_SIZE = 15


# --- Trajectory scoring utilities ---


def num_searches(record: dict) -> int:
    """Return the number of search calls in an episode by counting ``</search>`` tags
    in ``output_response``. Robust to length-truncated rollouts where
    ``env_extras.num_turns`` may be unset.
    """
    return count_searches(record.get("output_response", ""))


def load_records(path: Path) -> list[dict]:
    """Load all valid JSON records from a jsonl file, skipping malformed lines."""
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


SEED_RE = re.compile(r"__s=(\d+)")


def seed_from_path(path: str) -> int | None:
    """Extract the training seed from a run path's ``__s=<N>`` suffix, or None."""
    match = SEED_RE.search(path)
    return int(match.group(1)) if match else None


# --- Raw transfer data (written by reproduce_paper/scripts/save_transfer_data.py) ---


def load_transfer_data(path: Path) -> dict:
    """Load ``transfer_data.json`` and return its ``data`` mapping.

    The mapping is ``[model][train_dataset][benchmark] -> [record, ...]``, one record per
    training seed, holding raw per-run scores with no aggregation applied.
    """
    with open(path) as f:
        return json.load(f)["data"]


def transfer_cells(
    data: dict, model: str, train_dataset: str, benchmark: str, metric: str
) -> list[tuple[float, float]] | None:
    """Slice one cell out of the transfer data as per-seed ``(mean, stderr)`` pairs.

    Returns None when the model/regime/benchmark combination was never evaluated, which is
    the caller's cue to skip that bar, row or column rather than to error out.
    """
    records = data.get(model, {}).get(train_dataset, {}).get(benchmark)
    if not records:
        return None
    return [(float(record[metric]), float(record[f"{metric}_stderr"])) for record in records]


# --- Multi-level aggregation (training seeds, then benchmarks) ---


def combine_seeds(seed_cells: list[tuple[float, float]]) -> tuple[float, float]:
    """Average ``(mean, test_stderr)`` cells over independent training seeds.

    Returns ``(mean, variance)`` rather than ``(mean, stderr)`` so callers can keep
    accumulating variances across a further level (see :func:`combine_benchmarks`).

    The variance pools two independent sources: the test-example noise carried by each
    seed's own standard error, and the spread between seed means. With a single seed only
    the first term is available, so such a row's error bar understates run-to-run variance.
    """
    means = np.array([mean for mean, _ in seed_cells], dtype=float)
    stderrs = np.array([stderr for _, stderr in seed_cells], dtype=float)
    n = len(means)
    variance = float((stderrs**2).sum()) / n**2
    if n > 1:
        variance += float(means.var(ddof=1)) / n
    return float(means.mean()), variance


def combine_benchmarks(per_benchmark: list[tuple[float, float]]) -> tuple[float, float]:
    """Average ``(mean, variance)`` pairs over benchmarks, returning ``(mean, stderr)``.

    Benchmarks are treated as *fixed*, not as a sample from a population: we report the
    mean of these specific benchmarks, so each one's uncertainty is propagated
    (``Var = Sigma var_b / B^2``). Taking a variance across benchmark means instead would
    absorb the genuine difficulty spread between benchmarks into the error bar.
    """
    means = [mean for mean, _ in per_benchmark]
    variances = [variance for _, variance in per_benchmark]
    return float(np.mean(means)), float(np.sqrt(sum(variances)) / len(per_benchmark))


# --- Model / directory key helpers ---


def find_csv_key_for_ckpt(df: pd.DataFrame, path: str, ckpt_to_eval: int) -> str | None:
    """Find the CSV 'model' column value matching a training path + checkpoint.

    Strips the leading path component (share/, scratch/, etc.) before matching
    to handle symlink variations between training and eval environments.
    """
    path_suffix = "/".join(path.split("/")[1:])
    path_slug = path_suffix.replace("/", "--")
    matches = df[df["model"].str.contains(path_slug, regex=False)]
    if matches.empty:
        return None

    if ckpt_to_eval == -1:

        def extract_step(s: str) -> int:
            m = re.search(r"__step(\d+)$", s)
            return int(m.group(1)) if m else -1

        matches = matches.copy()
        matches["_step"] = matches["model"].apply(extract_step)
        return matches.loc[matches["_step"].idxmax(), "model"]

    target_suffix = f"__step{ckpt_to_eval}"
    matches = matches[matches["model"].str.endswith(target_suffix)]
    return matches.iloc[0]["model"] if not matches.empty else None


def model_hf_to_key(model_name: str) -> str:
    """Convert a HuggingFace model name to the eval directory key format.

    Args:
        model_name: HuggingFace model name, e.g. ``"Qwen/Qwen2.5-3B-Instruct"``.

    Returns:
        Directory key string, e.g. ``"model=Qwen--Qwen2.5-3B-Instruct"``.
    """
    return f"model={model_name.replace('/', '--')}"


# --- LaTeX table rendering ---

# Default float placement; individual tables override it to match how the paper uses them.
LATEX_FLOAT_PLACEMENT = "[t]"
LATEX_PREAMBLE_NOTE = [
    r"% Requires in the paper preamble:",
    r"%   \usepackage[table]{xcolor}",
    r"%   \usepackage{booktabs}",
    r"%   \definecolor{lightorange}{HTML}{FBE5DA}",
]

# Standalone preview wrapper. The geometry matches the NeurIPS body text, so an overfull
# \hbox in the log means the table really would overflow the column in the paper. Macros
# used in the captions are stubbed here because the fragments compile outside main.tex.
LATEX_PREVIEW_TEMPLATE = r"""\documentclass{article}
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=5.5in,textheight=9in]{geometry}
\usepackage[table]{xcolor}
\usepackage{booktabs}
\usepackage{xspace}
\usepackage{cleveref}
\usepackage{wrapfig}
\providecommand{\citep}[1]{[#1]}
\providecommand{\citet}[1]{#1}
\definecolor{lightorange}{HTML}{FBE5DA}
\newcommand{\qwenthree}{Qwen2.5-3B-Instruct\xspace}
\newcommand{\qwenseven}{Qwen2.5-7B-Instruct\xspace}
\newcommand{\llamathree}{Llama-3.2-3B-Instruct\xspace}
\newcommand{\phimini}{Phi-4-mini-instruct\xspace}
\newcommand{\synthrm}{SynthWorlds-RM\xspace}
\newcommand{\synthsm}{SynthWorlds-SM\xspace}
\newcommand{\frames}{FRAMES\xspace}
\pagestyle{empty}
\begin{document}
%s
\end{document}
"""


def latex_cell(stats: dict | None, fmt: str) -> str:
    """Render one ``mean +/- stderr`` math cell, or an em-dash when the run is missing."""
    if stats is None:
        return "--"
    return f"${stats['mean']:{fmt}} \\pm {stats['stderr']:{fmt}}$"


def wrap_latex_table(
    body: list[str],
    col_spec: str,
    header: str,
    caption: str,
    label: str,
    tabcolsep: str,
    float_env: str = "table",
    float_args: str = LATEX_FLOAT_PLACEMENT,
) -> str:
    """Assemble a full float around an already-rendered tabular body.

    Args:
        float_env: Float environment, e.g. ``"table"`` or ``"wraptable"``.
        float_args: Everything after ``\\begin{<env>}`` -- a placement spec like ``"[!htbp]"``
            for ``table``, or ``"{r}{0.47\\linewidth}"`` for ``wraptable``.
    """
    preamble_note = list(LATEX_PREAMBLE_NOTE)
    if float_env == "wraptable":
        preamble_note.append(r"%   \usepackage{wrapfig}")
    return "\n".join(
        [
            *preamble_note,
            rf"\begin{{{float_env}}}{float_args}",
            r"\centering",
            rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\addlinespace[2pt]",
            r"\end{tabular}",
            caption,
            rf"\label{{{label}}}",
            rf"\end{{{float_env}}}",
            "",
        ]
    )


def render_table_previews(tex_paths: list[Path], figures_dir: Path, combined_stem: str) -> None:
    """Compile each .tex fragment to a one-page preview PDF beside it.

    All fragments are compiled in a single document so that cross-references between the
    captions resolve, then split back out per table with Ghostscript. Reports overfull
    boxes, since those are exactly what a preview is meant to catch.

    Args:
        tex_paths: Table fragments to compile, one page each, in order.
        figures_dir: Where to drop the combined PDF if Ghostscript is unavailable.
        combined_stem: Filename stem for that un-split fallback PDF.
    """
    if shutil.which("pdflatex") is None:
        print("Warning: pdflatex not found; skipping previews")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for tex_path in tex_paths:
            shutil.copy(tex_path, tmp_dir / tex_path.name)
        body = "\n\\clearpage\n".join(rf"\input{{{path.stem}}}" for path in tex_paths)
        wrapper = tmp_dir / "preview.tex"
        wrapper.write_text(LATEX_PREVIEW_TEMPLATE % body)

        for _ in range(2):  # second pass resolves \Cref between captions
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", wrapper.name],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            print(f"Warning: preview compile failed:\n{result.stdout[-2000:]}")
            return

        overfull = [line for line in result.stdout.splitlines() if "Overfull" in line]
        for line in overfull:
            print(f"  Overfull box in preview: {line}")
        if not overfull:
            print("Previews compiled with no overfull boxes.")

        combined = tmp_dir / "preview.pdf"
        if shutil.which("gs") is None:
            fallback = figures_dir / f"{combined_stem}.pdf"
            shutil.copy(combined, fallback)
            print(f"Saved {fallback} (gs not found; pages not split)")
            return
        for page, tex_path in enumerate(tex_paths, start=1):
            pdf_path = tex_path.with_suffix(".pdf")
            subprocess.run(
                [
                    "gs",
                    "-dNOPAUSE",
                    "-dBATCH",
                    "-dQUIET",
                    "-sDEVICE=pdfwrite",
                    f"-dFirstPage={page}",
                    f"-dLastPage={page}",
                    f"-sOutputFile={pdf_path}",
                    str(combined),
                ],
                check=True,
                capture_output=True,
            )
            print(f"Saved {pdf_path}")
