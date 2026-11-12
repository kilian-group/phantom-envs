#!/usr/bin/env -S uv run python
"""Regenerate the paper's PhantomEnvs datasets with the exact generators that made them.

pw comes from ``phantom-wiki==1.0.3``; pwcons and pwcomp come from a PhantomWiki fork at commit
``839fa26``, whose CLI adds ``--max-hops``, ``--max-constraints`` and ``--question-types``. Both
import as ``phantom_wiki``, so each runs in its own ``uvx`` env, with dependencies resolved as of
:data:`EXCLUDE_NEWER`. Both need SWI-Prolog (``swipl``) on PATH; pwcons peaks at ~62 GB of RAM.

The generator's raw output must match the paper's :func:`raw_hash`; it is then converted into the
layout of :mod:`phantom_envs.environment`, split as in the paper (without the "How many" questions,
which the paper did not use).

Usage:
    # Writes data/pw/depth_16_size_1000_seed_1_50k, the dir the training configs read
    uv run python reproduce_paper/scripts/generate_paper_data.py pw-train

    uv run python reproduce_paper/scripts/generate_paper_data.py pwcons-train --output_dir /tmp/pwcons-train
"""

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from phantom_envs._types import SplitConfig, Task
from phantom_envs.environment import PhantomEnv, write_env
from phantom_envs.generator.difficulty import compute_difficulty
from phantom_envs.generator.templates import COMPARISON_PREFIXES
from phantom_envs.tasks import task_id

PW_1_0_3 = "phantom-wiki==1.0.3"
PW_FORK = "git+https://github.com/swathisaravananan/phantom-wiki@839fa26"
PYTHON = "3.12"
EXCLUDE_NEWER = "2026-10-18"
# Facts in the generators' facts.pl (a Prolog listing, which also holds rules and system clauses)
FACT_PREFIXES = ("type(", "gender(", "parent(", "dob(", "friend_(", "job(", "hobby(", "attribute(")

# The paper's splits: pw stratified by its integer difficulty, pwcons and pwcomp balanced
PW_SPLIT = SplitConfig(cells=("difficulty",))
PWCONS_SPLIT = SplitConfig(cells=("hops", "constraints"), cap_factor=1)
PWCOMP_SPLIT = SplitConfig(cells=("hops", "kind"), cap_factor=4)


@dataclass(frozen=True)
class PaperDataset:
    """One dataset of the paper and the generator call that reproduces it."""

    generator: str = field(metadata={"description": "uvx --from spec of the generator."})
    args: tuple[str, ...] = field(metadata={"description": "pw-generate arguments, except the output dir."})
    data_dir: str = field(metadata={"description": "The dataset's dir under the paper's data/."})
    raw_hash: str = field(metadata={"description": "raw_hash() of the generator's output for the paper."})
    split: SplitConfig = field(metadata={"description": "The paper's train / validation split."})
    data_source: str = field(metadata={"description": "The tasks' data_source: pw, pwcons or pwcomp."})


def flags(**options: int | str | bool) -> tuple[str, ...]:
    """Turn keywords into pw-generate flags.

    Args:
        **options: Flag values; True makes a bare switch.

    Returns:
        The flags, e.g. ``seed=1, easy_mode=True`` -> ``("--seed", "1", "--easy-mode")``.
    """
    args: list[str] = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        args += [flag] if value is True else [flag, str(value)]
    return tuple(args)


def pw_args(seed: int, question_depth: int, num_family_trees: int, per_type: int) -> tuple[str, ...]:
    """Flags of the paper's pw ``generate_dataset.sh`` (phantom-wiki 1.0.3).

    Args:
        seed: Universe and question seed.
        question_depth: Grammar depth (16 is at most 6 hops).
        num_family_trees: Family trees of 10 people.
        per_type: Questions (and sampling attempts) per template.

    Returns:
        The pw-generate flags.
    """
    return flags(
        seed=seed,
        question_depth=question_depth,
        num_family_trees=num_family_trees,
        max_family_tree_size=10,
        max_family_tree_depth=10,
        num_questions_per_type=per_type,
        num_sampling_attempts=per_type,
        easy_mode=True,
        article_format="json",
        question_format="json",
        skip_solution_traces=True,
        use_multithreading=True,
    )


def fork_args(seed: int, max_hops: int, max_constraints: int, **extra: str) -> tuple[str, ...]:
    """Flags of the paper's pb / pwcomp ``generate_dataset.sh`` (the fork, 1K people).

    Args:
        seed: Universe and question seed.
        max_hops: Most hops in a question.
        max_constraints: Most constraints in a question.
        **extra: More flags, e.g. ``question_types``.

    Returns:
        The pw-generate flags.
    """
    return flags(
        seed=seed,
        max_hops=max_hops,
        max_constraints=max_constraints,
        **extra,
        num_family_trees=100,
        max_family_tree_size=10,
        max_family_tree_depth=10,
        num_questions_per_type=1000,
        num_sampling_attempts=1000,
        num_multiprocesses=8,
        easy_mode=True,
        article_format="json",
        question_format="json",
        skip_solution_traces=True,
    )


# pw1 / pw2 / pwOOD are the paper's PW evals: the training universe, an unseen one, and an unseen 10K-person one.
PAPER_DATASETS: dict[str, PaperDataset] = {
    "pw-train": PaperDataset(
        PW_1_0_3,
        pw_args(seed=1, question_depth=16, num_family_trees=100, per_type=2500),
        "pw/depth_16_size_1000_seed_1_50k",
        "310cabc2531d3e5fccbe884ffac3688f707d86ebc60fd2ae2c4273163891dda6",
        PW_SPLIT,
        "pw",
    ),
    "pw1": PaperDataset(
        PW_1_0_3,
        pw_args(seed=1, question_depth=16, num_family_trees=100, per_type=500),
        "pw/depth_16_size_1000_seed_1",
        "ea1d5f598c468abd6388b9ecb82c52eed31d6d13213ad90f385831545f869ba4",
        PW_SPLIT,
        "pw",
    ),
    "pw2": PaperDataset(
        PW_1_0_3,
        pw_args(seed=2, question_depth=16, num_family_trees=100, per_type=500),
        "pw/depth_16_size_1000_seed_2",
        "a7022b7ebc2618cbd12fd8ca6e3fd786753d22b78dcafe0b56a99f86e13f0a38",
        PW_SPLIT,
        "pw",
    ),
    "pwOOD": PaperDataset(
        PW_1_0_3,
        pw_args(seed=2, question_depth=22, num_family_trees=1000, per_type=500),
        "pw/depth_22_size_10000_seed_2",
        "4912ad3c172144bde4ec8886a3342633d46bf16733f9a934cc2c2d3d145ca937",
        PW_SPLIT,
        "pw",
    ),
    "pwcons-train": PaperDataset(
        PW_FORK,
        fork_args(seed=1, max_hops=7, max_constraints=3),
        "pwcons/hops_7_cons_3_size_1000_seed_1_50k",
        "a405c195078446126e68b6a2c79a0ff1a875dffe6b94d4f2b868e44726d1093f",
        PWCONS_SPLIT,
        "pwcons",
    ),
    "pwcomp-train": PaperDataset(
        PW_FORK,
        fork_args(seed=1, max_hops=7, max_constraints=1, question_types="base,comparison_age"),
        "pwcomp/hops_7_cons_1_size_1000_seed_1_50k_balanced",
        "bb4c3dca2e3e4eaa0a31d98a55cd3c0ec9428db51ab62661749449b1c63d91ed",
        PWCOMP_SPLIT,
        "pwcomp",
    ),
}


def raw_hash(raw_dir: str | Path) -> str:
    """Hash a pinned generator's output.

    Ignores what the generators randomize between identical runs: the question ids (uuid4) and
    the order of the ``facts.pl`` lines.

    Args:
        raw_dir: Dir with the generator's ``articles.json``, ``questions.json`` and ``facts.pl``.

    Returns:
        SHA-256 hex digest.
    """
    raw_dir = Path(raw_dir)
    articles = json.loads((raw_dir / "articles.json").read_text())
    questions = json.loads((raw_dir / "questions.json").read_text())
    questions = [{k: v for k, v in q.items() if k != "id"} for q in questions]
    facts = sorted((raw_dir / "facts.pl").read_text().splitlines())
    digest = hashlib.sha256()
    for part in (articles, questions, facts):
        digest.update(json.dumps(part, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def paper_task(question: dict, data_source: str) -> Task:
    """Turn a question of the old generators into a task.

    Args:
        question: An entry of their ``questions.json``.
        data_source: The paper dataset's environment: pw, pwcons or pwcomp.

    Returns:
        The task. phantom-wiki 1.0.3 records an integer difficulty, so hops and constraints are
        computed from the query, as the fork does.
    """
    difficulty = question["difficulty"]
    prolog = question["prolog"]
    scores = (
        difficulty
        if isinstance(difficulty, dict)
        else compute_difficulty(prolog["query"], answer_var=prolog["answer"])
    )
    kind = (
        "comparison" if question["question"].startswith(tuple(p[0] for p in COMPARISON_PREFIXES)) else "base"
    )
    return Task(
        id=task_id(question["question"], question["answer"]),
        data_source=data_source,
        question=question["question"],
        answers=question["answer"],
        metadata={
            "kind": kind,
            "hops": scores["hops"],
            "constraints": scores["constraints"],
            "difficulty": difficulty,
            "template": question["type"],
            "prolog": prolog,
        },
    )


def convert_raw(raw_dir: str | Path, output_dir: str | Path, name: str) -> PhantomEnv:
    """Convert a pinned generator's output into an environment, split as in the paper.

    Args:
        raw_dir: The generator's output dir.
        output_dir: Where to write the environment.
        name: The paper dataset, one of :data:`PAPER_DATASETS`.

    Returns:
        The environment.
    """
    dataset = PAPER_DATASETS[name]
    raw_dir = Path(raw_dir)
    # Facts out of the Prolog listing, sorted (the listing's order varies between runs)
    lines = (raw_dir / "facts.pl").read_text().splitlines()
    facts = sorted(line.strip().removesuffix(".") for line in lines if line.startswith(FACT_PREFIXES))
    articles = json.loads((raw_dir / "articles.json").read_text())
    # The paper left out the "How many" questions
    questions = json.loads((raw_dir / "questions.json").read_text())
    tasks = [paper_task(q, dataset.data_source) for q in questions if not q["is_aggregation_question"]]
    manifest = {
        "paper": name,
        "generator": dataset.generator,
        "generator_args": list(dataset.args),
        "raw_hash": dataset.raw_hash,
        "split": asdict(dataset.split),
    }
    return write_env(Path(output_dir), facts, articles, tasks, dataset.split, manifest)


def generate_paper(name: str, output_dir: str | Path | None = None) -> PhantomEnv:
    """Regenerate a paper dataset, check it against the paper's, and write it as an environment.

    Args:
        name: One of :data:`PAPER_DATASETS`.
        output_dir: An empty or new dir; defaults to ``data/<data_dir>``, the dir the configs read.

    Returns:
        The environment.

    Raises:
        FileExistsError: If the output dir is not empty.
        RuntimeError: If ``swipl`` or ``uvx`` is missing, or the output differs from the paper's.
    """
    dataset = PAPER_DATASETS[name]
    output_dir = Path(output_dir or Path("data") / dataset.data_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; pick another dir or empty it")
    for tool in ("swipl", "uvx"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is not on PATH (see the README's install section)")

    # Run the pinned generator into raw/ and check its output
    raw_dir = output_dir / "raw"
    cmd = ["uvx", "--python", PYTHON, "--exclude-newer", EXCLUDE_NEWER, "--from", dataset.generator]
    subprocess.run([*cmd, "pw-generate", "-od", str(raw_dir), *dataset.args], check=True)
    if (got := raw_hash(raw_dir)) != dataset.raw_hash:
        raise RuntimeError(f"{raw_dir} differs from the paper's {name}: raw hash {got}")
    print(f"{name}: matches the paper ({got})")

    env = convert_raw(raw_dir, output_dir, name)
    shutil.rmtree(raw_dir)
    return env


def main(args: argparse.Namespace) -> None:
    """Regenerate one paper dataset and print the environment's summary."""
    env = generate_paper(args.name, args.output_dir)
    summary = {k: env.manifest[k] for k in ("content_hash", "num_people", "num_tasks")}
    print(json.dumps({"path": str(env.path)} | summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Regenerate a paper dataset with its pinned generator.")
    parser.add_argument("name", choices=list(PAPER_DATASETS), help="Paper dataset.")
    parser.add_argument(
        "--output_dir", default=None, help="An empty dir (default: data/<dir>, the dir the configs read)."
    )
    args = parser.parse_args()
    main(args)
