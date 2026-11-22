"""Generate a PhantomEnvs environment: a fictional universe, its articles, and train / validation tasks.

Output layout of `output_dir`:

    phantom_env.json            manifest: config, content hash, task counts per cell
    universe/facts.pl           the universe's facts, in Prolog syntax (the relations that follow from
                                them: phantom_envs/generator/relations.py)
    universe/articles.json      one article per person: {"title", "article", "facts"}, plus "template"
                                when rewritten with an LLM (EnvConfig.rewrite_docs_with_llm)
    corpus.jsonl                the articles as the retriever's corpus
    tasks/{train,validation}.jsonl

Usage::

    from phantom_envs import generate

    env = generate("data/envs/pwcons", "pwcons")

or from the command line, ``uv run phantom_envs generate --preset pwcons --output_dir ...``.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

from phantom_envs import __version__
from phantom_envs._types import EnvConfig, SplitConfig, Task
from phantom_envs.corpus import write_corpus
from phantom_envs.generator.parallel import parallel_map
from phantom_envs.splits import split_tasks
from phantom_envs.tasks import SPLITS, load_tasks, task_id, write_tasks

MANIFEST = "phantom_env.json"
HASHED_FILES = (
    "universe/facts.pl",
    "universe/articles.json",
    "corpus.jsonl",
    "tasks/train.jsonl",
    "tasks/validation.jsonl",
)

PRESETS: dict[str, EnvConfig] = {
    # Hops (PhantomWiki)
    "pw": EnvConfig(name="pw", max_hops=6, max_constraints=1),
    # Hops+Constraints: every (hops, constraints) cell cut to the smallest
    "pwcons": EnvConfig(name="pwcons", max_hops=7, max_constraints=3, split=SplitConfig(cap_factor=1)),
    # Hops+Comparisons: every (hops, kind) cell capped at 4x the smallest (hops-0 comparisons)
    "pwcomp": EnvConfig(
        name="pwcomp",
        max_hops=7,
        max_constraints=1,
        kinds=("base", "comparison"),
        split=SplitConfig(cells=("hops", "kind"), cap_factor=4),
    ),
}


@dataclass
class PhantomEnv:
    """A generated environment on disk."""

    path: Path = field(metadata={"description": "The environment's data dir."})
    manifest: dict = field(metadata={"description": "Contents of phantom_env.json."})

    @classmethod
    def load(cls, path: str | Path) -> "PhantomEnv":
        """Open the environment in a data dir.

        Args:
            path: The data dir, holding ``phantom_env.json``.

        Returns:
            The environment.
        """
        return cls(Path(path), json.loads((Path(path) / MANIFEST).read_text()))

    def tasks(self, split: str) -> list[Task]:
        """The tasks of one split, ``train`` or ``validation``."""
        return load_tasks(self.path, split)


def content_hash(data_dir: str | Path) -> str:
    """Hash an environment's files, which are all written deterministically (LLM rewrites aside).

    Args:
        data_dir: The environment's data dir.

    Returns:
        SHA-256 hex digest over the names and bytes of :data:`HASHED_FILES`.
    """
    digest = hashlib.sha256()
    for name in HASHED_FILES:
        digest.update(name.encode() + b"\0" + (Path(data_dir) / name).read_bytes())
    return digest.hexdigest()


def to_task(question: dict, data_source: str) -> Task:
    """Turn a sampled question into a task.

    Args:
        question: A question from :func:`phantom_envs.generator.questions.sample_questions`.
        data_source: The environment's name.

    Returns:
        The task; hops and constraints in its metadata come from the difficulty record.
    """
    difficulty = question["difficulty"]
    return Task(
        id=task_id(question["question"], question["answers"]),
        data_source=data_source,
        question=question["question"],
        answers=question["answers"],
        metadata={
            "kind": question["kind"],
            "hops": difficulty["hops"],
            "constraints": difficulty["constraints"],
            "difficulty": difficulty,
            "template": question["template"],
            "prolog": question["prolog"],
        },
    )


def write_env(
    output_dir: Path,
    facts: Iterable[str],
    articles: list[dict],
    tasks: list[Task],
    split: SplitConfig,
    manifest: dict,
    num_workers: int = 1,
) -> PhantomEnv:
    """Split the tasks and write the environment's files and manifest.

    Args:
        output_dir: The environment's data dir.
        facts: The universe's facts in Prolog syntax, without the trailing period (read once).
        articles: `{"title", "article", "facts"}` per person.
        tasks: All tasks, in generation order.
        split: How to split them.
        manifest: What made the environment (config, generator); counts and the hash are added.
        num_workers: Processes to write the files with.

    Returns:
        The written environment.
    """
    universe = output_dir / "universe"
    universe.mkdir(parents=True, exist_ok=True)
    # The tasks, split into train and validation
    splits = dict(zip(SPLITS, split_tasks(tasks, split), strict=True))

    def write_facts() -> None:
        (universe / "facts.pl").write_text("".join(f"{fact}.\n" for fact in facts))

    def write_articles() -> None:
        (universe / "articles.json").write_text(json.dumps(articles, indent=1))

    # Each file is written on its own, so workers can write them at the same time
    writers = [write_facts, write_articles, partial(write_corpus, articles, output_dir / "corpus.jsonl")]
    writers += [partial(write_tasks, output_dir, name, members) for name, members in splits.items()]
    parallel_map(lambda i: writers[i](), range(len(writers)), num_workers)

    # Task counts per (hops, constraints, kind) cell, then the manifest
    cells: dict[tuple, dict[str, int]] = {}
    for name, members in splits.items():
        for t in members:
            key = (t.metadata["hops"], t.metadata["constraints"], t.metadata["kind"])
            counts = cells.setdefault(key, dict.fromkeys(SPLITS, 0))
            counts[name] += 1
    manifest = manifest | {
        "phantom_envs": __version__,
        "content_hash": content_hash(output_dir),
        "num_people": len(articles),
        "num_tasks": {name: len(ts) for name, ts in splits.items()},
        "cells": [{"hops": h, "constraints": c, "kind": k} | n for (h, c, k), n in sorted(cells.items())],
    }
    (output_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return PhantomEnv(output_dir, manifest)


def generate(
    output_dir: str | Path, config: EnvConfig | str = "pw", *, num_workers: int = 8, overwrite: bool = False
) -> PhantomEnv:
    """Generate an environment into `output_dir`.

    Args:
        output_dir: Where to write it (see the module docstring for the layout).
        config: An `EnvConfig`, or the name of a preset in `PRESETS`.
        num_workers: Processes to write articles, sample and answer questions, and write files with.
            The environment does not depend on it.
        overwrite: Replace an existing environment in `output_dir`.

    Returns:
        The generated environment.

    Raises:
        ValueError: If `size` is not a multiple of the family tree size (10), or `friendship_degree` is negative.
        FileExistsError: If `output_dir` has an environment and `overwrite` is False.
    """
    # Imported here so that loading an environment does not import the generator's libraries (nltk, networkx)
    from phantom_envs.generator.articles import build_articles
    from phantom_envs.generator.questions import sample_questions
    from phantom_envs.generator.relations import format_fact
    from phantom_envs.generator.universe import FAMILY_TREE_SIZE, build_universe

    config = PRESETS[config] if isinstance(config, str) else config
    if config.size % FAMILY_TREE_SIZE:
        raise ValueError(f"size must be a multiple of {FAMILY_TREE_SIZE}, got {config.size}")
    if config.friendship_degree < 0:
        raise ValueError(f"friendship_degree must not be negative, got {config.friendship_degree}")
    output_dir = Path(output_dir)
    if (output_dir / MANIFEST).exists() and not overwrite:
        raise FileExistsError(f"{output_dir} already has an environment; pass overwrite=True to replace it")

    # Universe, articles, then questions about it
    db = build_universe(config.size, config.seed, config.friendship_degree)
    articles = build_articles(db, num_workers)
    manifest = {"config": asdict(config)}
    if config.rewrite_docs_with_llm:
        from phantom_envs.generator.rewrite import rewrite_articles

        articles, manifest["rewrite"] = rewrite_articles(articles, config.rewrite_docs_with_llm)
    tasks = [to_task(q, config.name) for q in sample_questions(db, config, num_workers)]
    facts = map(format_fact, db.facts)
    return write_env(output_dir, facts, articles, tasks, config.split, manifest, num_workers)
