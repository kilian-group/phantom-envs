#!/usr/bin/env -S uv run python
"""The BM25 retriever: keyword search over a data dir's corpus, with no model and no GPU.

bm25s (https://github.com/xhluca/bm25s) scores documents with numpy and scipy. Documents and queries
are tokenized the same way (see tokenize): lowercased words, without English stopwords, stemmed. A
query that shares no word with the corpus finds nothing.

Build the index once per corpus, then search it:

    retriever = BM25Retriever("data/envs/pw")
    [hits] = retriever.search(["Who is the mother of Ann Lee?"], topk=3)

Usage:
    # Writes <data_dir>/bm25_index/
    uv run python -m phantom_envs.retriever.bm25 --data_dir data/envs/pw
"""

import argparse
import logging
import shutil
import threading
from pathlib import Path

import bm25s
import Stemmer

from phantom_envs._types import DocumentResult
from phantom_envs.retriever.base import Retriever, load_corpus, to_document

logger = logging.getLogger(__name__)

INDEX_DIRNAME: str = "bm25_index"

# A PyStemmer stemmer must not be shared between threads, so each thread gets its own
_thread = threading.local()


def index_dir(data_dir: str | Path) -> Path:
    """Path of the BM25 index in data_dir."""
    return Path(data_dir) / INDEX_DIRNAME


def tokenize(texts: list[str]) -> list[list[str]]:
    """Split texts into the words BM25 matches on.

    Args:
        texts: Documents or queries.

    Returns:
        Per text, its lowercased and stemmed words, without English stopwords.
    """
    if not hasattr(_thread, "stemmer"):
        _thread.stemmer = Stemmer.Stemmer("english")
    return bm25s.tokenize(
        texts, stopwords="en", stemmer=_thread.stemmer, return_ids=False, show_progress=False
    )


def build_index(data_dir: str | Path) -> Path:
    """Index a data dir's corpus, replacing an index that is already there.

    Args:
        data_dir: The dir with corpus.jsonl.

    Returns:
        The index dir, <data_dir>/bm25_index.
    """
    out_dir = index_dir(data_dir)
    texts = load_corpus(data_dir)["contents"]
    logger.info(f"Indexing {len(texts)} documents ...")
    index = bm25s.BM25()
    index.index(tokenize(texts), show_progress=False)

    # Written next to its final place and renamed, so a reader never sees a partial index
    tmp_dir = out_dir.with_name(f".{INDEX_DIRNAME}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    index.save(tmp_dir, show_progress=False)
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    logger.info(f"Index saved to {out_dir}")
    return out_dir


class BM25Retriever(Retriever):
    """BM25 keyword search over a document corpus."""

    def __init__(self, data_dir: str | Path) -> None:
        """Load the index and the corpus.

        Args:
            data_dir: The dir with corpus.jsonl and the bm25_index that build_index wrote.

        Raises:
            FileNotFoundError: If the index has not been built.
        """
        path = index_dir(data_dir)
        if not path.is_dir():
            raise FileNotFoundError(
                f"No BM25 index at {path}; build it with "
                f"`python -m phantom_envs.retriever.bm25 --data_dir {data_dir}`"
            )
        # Memory-mapped: the index is read-only, and server processes over one corpus share its pages
        self.index = bm25s.BM25.load(path, mmap=True, show_progress=False)
        self.corpus = load_corpus(data_dir)
        logger.info(f"BM25 index loaded: {len(self.corpus)} documents.")

    def search(self, queries: list[str], topk: int) -> list[list[DocumentResult]]:
        """Find the documents that best match each query's words.

        Args:
            queries: The search queries.
            topk: The most documents to return per query.

        Returns:
            Per query, the documents that share a word with it, with their BM25 scores, best first.
        """
        # bm25s refuses to rank more documents than the corpus has
        k = min(topk, len(self.corpus))
        idxs, scores = self.index.retrieve(tokenize(queries), k=k, show_progress=False)
        return [
            [
                DocumentResult(document=to_document(self.corpus[int(idx)]), score=float(score))
                for idx, score in zip(row_idxs, row_scores, strict=True)
                if score > 0  # a score of 0 means no word matched
            ]
            for row_idxs, row_scores in zip(idxs, scores, strict=True)
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Build a BM25 index over a data dir's corpus.jsonl.")
    parser.add_argument(
        "--data_dir", required=True, help="Directory with corpus.jsonl; the index is written here."
    )
    build_index(Path(parser.parse_args().data_dir).expanduser())
