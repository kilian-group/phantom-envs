#!/usr/bin/env -S uv run python
"""Pool the per-benchmark multi-hop evaluation sets into a single cross-domain corpus.

Each benchmark prepared by ``scripts/benchmarks/multihop.py`` ships a corpus
containing only the paragraphs its own questions need, so retrieval is easy: the
distractor pool is at most a few thousand in-domain documents. This script unions
those corpora into one index, so every question competes against the documents of
all other benchmarks (including the fictional SynthWorlds worlds and the ~35k FRAMES
chunks). Answering still requires the question's own gold documents, but the
retriever now has to find them in an 8-44x larger, cross-domain haystack.

Document embeddings are reconstructed from each benchmark's existing
``<encoder>_Flat.index`` (an exact ``IndexFlatIP`` over normalized vectors), so
pooling is a concatenation — no re-encoding, no GPU needed.

The per-benchmark tasks are concatenated too. Each keeps its benchmark as its data source, which
SkyRL keys eval metrics and trajectory dumps by, so a single pooled eval run still reports each
benchmark separately and ``scripts/score_trajectories.py`` works unchanged. The only
change is that ``metadata`` gains every benchmark's keys and ``metadata.id`` becomes a string, so
the benchmarks share one Arrow schema in the parquet; scoring does not read either field.

Usage:
    uv run python reproduce_paper/scripts/pooled.py --out_dir data/pooled6

    uv run python reproduce_paper/scripts/pooled.py \
        --datasets hp500 2wiki500 msq500 synthrm500 synthsm500 frames cofca500 \
        --out_dir data/pooled7

Output (written into --out_dir):
    corpus.jsonl                 — pooled documents: {"id", "title", "contents", "source"}
    <encoder_name>_Flat.index    — FAISS inner-product index over the pooled documents
    tasks/validation.jsonl       — the benchmarks' tasks, concatenated
"""

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np

from phantom_envs._types import Task
from phantom_envs.tasks import load_tasks, write_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# The six benchmarks reported in the paper's main table.
DEFAULT_DATASETS: list[str] = ["hp500", "2wiki500", "msq500", "synthrm500", "synthsm500", "frames"]


def load_corpus(corpus_path: Path) -> list[dict]:
    """Load a benchmark corpus JSONL, preserving file order.

    Line order matters: the retrieval server maps a FAISS result index to a document
    positionally (``self.corpus[int(i)]``), so line ``i`` must stay aligned with
    embedding row ``i``.

    Args:
        corpus_path: Path to a benchmark's ``corpus.jsonl``.

    Returns:
        List of document dicts with at least "title" and "contents" keys.
    """
    with corpus_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_embeddings(index_path: Path, n_docs: int) -> np.ndarray:
    """Reconstruct the document embeddings stored in a flat FAISS index.

    ``IndexFlatIP`` keeps the raw vectors, so reconstruction is exact and avoids
    re-encoding the corpus.

    Args:
        index_path: Path to a benchmark's ``<encoder>_Flat.index``.
        n_docs: Number of documents in the matching corpus, used as a consistency check.

    Returns:
        Float32 array of shape (n_docs, dim) with unit-norm rows.
    """
    index = faiss.read_index(str(index_path))
    if index.ntotal != n_docs:
        raise ValueError(f"{index_path}: index has {index.ntotal} vectors but corpus has {n_docs} docs")
    return index.reconstruct_n(0, index.ntotal)


def pool_corpora(datasets: list[str], data_root: Path, encoder_name: str) -> tuple[list[dict], np.ndarray]:
    """Union the benchmark corpora, dropping documents whose text is already present.

    Deduplication keys on ``contents`` alone, since that is the only field the agent
    ever sees (the retrieval server returns ``title=None``). In practice it is a no-op
    for the paper's 6 benchmarks — HotpotQA / 2Wiki / MuSiQue share a Wikipedia 2018
    source but chunk it differently, so no two paragraphs are byte-identical.

    Args:
        datasets: Benchmark names to pool, in the order their documents should appear.
        data_root: Directory holding one subdirectory per benchmark.
        encoder_name: Encoder family prefix of the index filename (e.g. "qwen3-4b").

    Returns:
        Tuple of (pooled documents with global sequential ids, matching embedding matrix).
    """
    pooled_docs: list[dict] = []
    pooled_embs: list[np.ndarray] = []
    seen: set[str] = set()

    for dataset in datasets:
        data_dir = data_root / dataset
        docs = load_corpus(data_dir / "corpus.jsonl")
        embs = load_embeddings(data_dir / f"{encoder_name}_Flat.index", len(docs))

        keep = [i for i, doc in enumerate(docs) if doc["contents"] not in seen]
        seen.update(docs[i]["contents"] for i in keep)
        for i in keep:
            pooled_docs.append(
                {
                    "id": str(len(pooled_docs)),
                    "title": docs[i]["title"],
                    "contents": docs[i]["contents"],
                    "source": dataset,
                }
            )
        pooled_embs.append(embs[keep])
        logger.info(f"{dataset}: {len(docs)} docs, kept {len(keep)}, dropped {len(docs) - len(keep)} dupes")

    embeddings = np.concatenate(pooled_embs, axis=0)
    if len(pooled_docs) != len(embeddings):
        raise ValueError(f"Doc/embedding mismatch: {len(pooled_docs)} docs vs {len(embeddings)} vectors")
    logger.info(f"Pooled corpus: {len(pooled_docs)} documents, dim={embeddings.shape[1]}")
    return pooled_docs, embeddings


def write_corpus(docs: list[dict], corpus_path: Path) -> None:
    """Write the pooled corpus to a JSONL file, one document per line."""
    with corpus_path.open("w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(docs)} documents to {corpus_path}")


def write_index(embeddings: np.ndarray, index_path: Path) -> None:
    """Build and save a flat inner-product FAISS index over the pooled embeddings."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    logger.info(f"Wrote FAISS index with {index.ntotal} vectors to {index_path}")


def unify_metadata(tasks: list[Task]) -> None:
    """Give every task's metadata the same keys, in place.

    Arrow cannot hold the benchmarks in one parquet as-is: their metadata hold different keys
    (FRAMES has ``n_hops``/``wiki_titles``, the others ``type``) and conflicting ``id`` types
    (FRAMES ints vs hex strings elsewhere). Every key is unioned, missing ones are filled with
    None, and ``id`` is cast to str.

    Args:
        tasks: The pooled tasks.
    """
    keys = sorted({key for task in tasks for key in task.metadata})
    for task in tasks:
        task.metadata = {
            key: str(task.metadata[key]) if key == "id" else task.metadata.get(key) for key in keys
        }


def write_pooled_tasks(datasets: list[str], data_root: Path, out_dir: Path) -> None:
    """Concatenate the benchmarks' tasks.

    Tasks are otherwise left untouched: ``data_source`` already distinguishes the benchmarks.
    """
    tasks = []
    for dataset in datasets:
        dataset_tasks = load_tasks(data_root / dataset, "validation")
        logger.info(f"{dataset}: {len(dataset_tasks)} tasks")
        tasks += dataset_tasks
    unify_metadata(tasks)
    write_tasks(out_dir, "validation", tasks)
    logger.info(f"Wrote {len(tasks)} tasks to {out_dir}")


def main(args: argparse.Namespace) -> None:
    """Build the pooled corpus, index, and tasks."""
    data_root = Path(args.data_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    docs, embeddings = pool_corpora(args.datasets, data_root, args.encoder_name)
    write_corpus(docs, out_dir / "corpus.jsonl")
    write_index(embeddings, out_dir / f"{args.encoder_name}_Flat.index")
    write_pooled_tasks(args.datasets, data_root, out_dir)
    logger.info(f"Pooled dataset ready in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Benchmarks to pool, in document order (default: {' '.join(DEFAULT_DATASETS)}).",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Directory containing one prepared subdirectory per benchmark (default: data).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for the pooled corpus, index, and tasks.",
    )
    parser.add_argument(
        "--encoder_name",
        default="qwen3-4b",
        help="Encoder family prefix of the per-benchmark index filename (default: qwen3-4b).",
    )
    main(parser.parse_args())
