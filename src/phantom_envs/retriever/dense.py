"""The dense retriever: an encoder embeds the query, and a flat FAISS index finds the nearest documents.

Reads <data_dir>/<encoder>_Flat.index, which build_index.py writes. FAISS is not a dependency of the
venv: run with `uv run --with faiss-cpu==1.12.0`, or put a conda faiss-gpu on the path for faiss_gpu.

Usage:

    retriever = DenseRetriever("e5", "data/envs/pw")
    [hits] = retriever.search(["Who is the mother of Ann Lee?"], topk=3)
"""

import logging
import threading
from pathlib import Path

import faiss

from phantom_envs._types import DocumentResult
from phantom_envs.retriever.base import Retriever, load_corpus, to_document
from phantom_envs.retriever.encoders import get_encoder, index_path, load_encoder

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """FAISS-backed dense retriever over a document corpus."""

    def __init__(
        self, encoder: str, data_dir: str | Path, faiss_gpu: bool = False, device: str | None = None
    ) -> None:
        """Load the encoder, the index and the corpus.

        Args:
            encoder: An encoder preset of encoders.py.
            data_dir: The dir with corpus.jsonl and <encoder>_Flat.index.
            faiss_gpu: Whether to shard the index across all GPUs in fp16 (needs faiss-gpu).
            device: Where the encoder runs, e.g. cpu, to leave the GPUs to a trainer. None keeps
                the preset's choice, or a GPU if there is one.

        Raises:
            FileNotFoundError: If the index has not been built.
        """
        self.preset = get_encoder(encoder)
        index_file = index_path(data_dir, encoder)
        if not index_file.is_file():
            raise FileNotFoundError(
                f"No index at {index_file}; build it with "
                f"`python -m phantom_envs.retriever.build_index --data_dir {data_dir} --encoder {encoder}`"
            )

        logger.info(f"Loading encoder: {self.preset.model}")
        self.model = load_encoder(encoder, device=device)

        logger.info(f"Loading FAISS index from {index_file} ...")
        self.index = faiss.read_index(str(index_file))
        if faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            logger.info("FAISS index moved to GPU(s).")

        # FAISS GPU's StackDeviceMemory allocator is not thread-safe; serialize search calls under
        # concurrent requests. CPU FAISS is thread-safe for read-only search, so skip the lock there.
        self.search_lock = threading.Lock() if faiss_gpu else None

        self.corpus = load_corpus(data_dir)
        logger.info(f"Corpus loaded: {len(self.corpus)} documents.")

    def search(self, queries: list[str], topk: int) -> list[list[DocumentResult]]:
        """Find the nearest documents of each query, with one encode and one index search.

        Args:
            queries: The search queries.
            topk: The most documents to return per query.

        Returns:
            Per query, its documents with their inner-product scores, best first.
        """
        # No progress bar: a server encodes on every search
        query_emb = self.model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False, **self.preset.query_kwargs
        )
        if self.search_lock is not None:
            with self.search_lock:
                scores, idxs = self.index.search(query_emb, k=topk)
        else:
            scores, idxs = self.index.search(query_emb, k=topk)

        return [
            [
                DocumentResult(document=to_document(self.corpus[int(idx)]), score=float(score))
                for idx, score in zip(row_idxs, row_scores, strict=True)
                if idx >= 0  # FAISS pads with -1 when the corpus has fewer than topk documents
            ]
            for row_idxs, row_scores in zip(idxs, scores, strict=True)
        ]
