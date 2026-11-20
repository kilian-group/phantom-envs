"""The retriever interface: what every search backend implements.

A retriever ranks the documents of a data dir's corpus.jsonl for a query. The servers in this
package only use this interface, so a new backend needs no change to them, nor to the agents that
search through them.

Backends:

- bm25 (bm25.py): keyword search, with no model and no GPU.
- e5 and qwen3-4b (dense.py): a dense encoder with a flat FAISS index; the names are the presets
  of encoders.py.

Usage:

    retriever = open_retriever("bm25", "data/envs/pw")
    [hits] = retriever.search(["Who is the mother of Ann Lee?"], topk=3)
    hits[0].document.contents, hits[0].score

To add a backend, subclass Retriever and give it a name in open_retriever.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phantom_envs._types import Document, DocumentResult
from phantom_envs.retriever.encoders import ENCODERS, corpus_path

if TYPE_CHECKING:
    import datasets

BM25: str = "bm25"
RETRIEVERS: tuple[str, ...] = (BM25, *ENCODERS)


class Retriever(ABC):
    """Ranks a corpus's documents for queries. Safe to call from many threads at once."""

    @abstractmethod
    def search(self, queries: list[str], topk: int) -> list[list[DocumentResult]]:
        """Find the best documents for each query.

        Args:
            queries: The search queries.
            topk: The most documents to return per query.

        Returns:
            Per query, its documents with their scores, best first. A backend may return fewer than
            topk, e.g. BM25 for a query that shares no word with the corpus.
        """


def load_corpus(data_dir: str | Path) -> "datasets.Dataset":
    """Load a data dir's corpus, memory-mapped, so a large corpus doesn't have to fit in RAM.

    Args:
        data_dir: The dir with corpus.jsonl.

    Returns:
        The corpus, one {"id", "contents", ...} row per document, in file order.
    """
    import datasets

    return datasets.load_dataset("json", data_files=str(corpus_path(data_dir)), split="train", num_proc=4)


def to_document(row: Mapping[str, Any]) -> Document:
    """Return a corpus row as the document a search returns.

    Args:
        row: The corpus row.

    Returns:
        The document, without a title: Search-R1's Wikipedia corpus has none, and the contents of
        the other corpora start with it.
    """
    return Document(id=row["id"], title=None, contents=row["contents"])


def open_retriever(
    name: str, data_dir: str | Path, faiss_gpu: bool = False, device: str | None = None
) -> Retriever:
    """Load a retriever over a data dir's corpus and the index built for it.

    Args:
        name: One of RETRIEVERS.
        data_dir: The dir with corpus.jsonl and the retriever's index.
        faiss_gpu: For a dense retriever, whether to shard the index across all GPUs.
        device: For a dense retriever, where its encoder runs (e.g. cpu); None lets it choose.

    Returns:
        The retriever.

    Raises:
        ValueError: For an unknown name.
        FileNotFoundError: If the retriever's index has not been built.
    """
    # Imported here: each backend has dependencies the others don't need (FAISS, bm25s)
    if name == BM25:
        from phantom_envs.retriever.bm25 import BM25Retriever

        return BM25Retriever(data_dir)
    if name in ENCODERS:
        from phantom_envs.retriever.dense import DenseRetriever

        return DenseRetriever(name, data_dir, faiss_gpu=faiss_gpu, device=device)
    raise ValueError(f"Unknown retriever {name!r} (expected one of {RETRIEVERS})")
