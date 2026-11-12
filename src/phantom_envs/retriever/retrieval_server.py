#!/usr/bin/env -S uv run python
"""Dense retrieval server: a flat FAISS index over a document corpus, served over HTTP.

Reads ``<data_dir>/corpus.jsonl`` and ``<data_dir>/<encoder>_Flat.index`` (see
:mod:`phantom_envs.retriever.encoders`). FAISS is not a dependency of the venv; run the server
with ``uv run --with faiss-cpu==1.12.0`` (or put a conda faiss-gpu on the path for ``faiss_gpu``).

Usage:
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.retrieval_server \
        --encoder e5 --data_dir data/pw/depth_16_size_1000_seed_1_50k

    # Override any config field at runtime, e.g. a kernel-chosen port written to a file
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.retrieval_server \
        --encoder qwen3-4b --data_dir data/browsecomp-plus-c512 \
        port=0 port_file=/tmp/retriever.port snippet_max_tokens=512

API:
    GET  /health           {"status": "ok"} once the index is loaded
    POST /retrieve         {"query": "...", "topk": 3, "return_scores": true}
                           -> {"result": [{"document": {...}, "score": float}, ...]}
    POST /retrieve_batch   {"queries": ["...", ...], "topk": 3, "return_scores": true}
                           -> {"results": [{"result": [...]}, ...]}
                           (padded batch encoding: rarely ranks differently from /retrieve)
"""

import argparse
import logging
import socket
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import datasets
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from phantom_envs._types import Document, DocumentResult, RetrievalResponse
from phantom_envs.retriever.encoders import ENCODERS, corpus_path, get_encoder, index_path, load_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RetrieverServerConfig:
    """Retrieval server settings; ``encoder`` and ``data_dir`` locate the index and corpus."""

    encoder: str = field(metadata={"description": f"Encoder preset: one of {list(ENCODERS)}."})
    data_dir: str = field(metadata={"description": "Dir with corpus.jsonl and <encoder>_Flat.index."})
    topk: int = field(default=3, metadata={"description": "Default number of documents per query."})
    faiss_gpu: bool = field(
        default=False, metadata={"description": "Shard the index across all GPUs in fp16 (needs faiss-gpu)."}
    )
    port: int = field(default=8000, metadata={"description": "HTTP port; 0 lets the kernel pick a free one."})
    port_file: str | None = field(
        default=None, metadata={"description": "If set, the bound port is written here."}
    )
    snippet_max_tokens: int = field(
        default=-1,
        metadata={
            "description": "Truncate each returned document to its first N tokens (-1 = full document)."
        },
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class DenseRetriever:
    """FAISS-backed dense retriever over a document corpus."""

    def __init__(self, cfg: RetrieverServerConfig) -> None:
        self.cfg = cfg
        self.preset = get_encoder(cfg.encoder)

        logger.info(f"Loading encoder: {self.preset.model}")
        self.model = load_encoder(cfg.encoder)

        # Separate raw Rust tokenizer for snippets: its encode() never mutates truncation/padding
        # state, so it's safe under concurrent requests (sharing the encoder's raises "Already borrowed").
        self.snippet_tokenizer = Tokenizer.from_pretrained(self.preset.model)

        index_file = index_path(cfg.data_dir, cfg.encoder)
        logger.info(f"Loading FAISS index from {index_file} ...")
        self.index = faiss.read_index(str(index_file))
        if cfg.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            logger.info("FAISS index moved to GPU(s).")

        # FAISS GPU's StackDeviceMemory allocator is not thread-safe; serialize
        # search calls under concurrent /retrieve requests. CPU FAISS is
        # thread-safe for read-only search, so skip the lock there.
        self.search_lock = threading.Lock() if cfg.faiss_gpu else None

        corpus_file = corpus_path(cfg.data_dir)
        logger.info(f"Loading corpus from {corpus_file} ...")
        self.corpus = datasets.load_dataset("json", data_files=str(corpus_file), split="train", num_proc=4)
        logger.info(f"Corpus loaded: {len(self.corpus)} documents.")

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode queries into normalized embeddings of shape (len(queries), dim)."""
        return self.model.encode(queries, normalize_embeddings=True, **self.preset.query_kwargs)

    def snippet(self, text: str) -> str:
        """Truncate text to its first ``cfg.snippet_max_tokens`` tokens (BrowseComp-Plus protocol)."""
        n = self.cfg.snippet_max_tokens
        if n <= 0:
            return text
        # Cut characters first so multi-MB pages don't get fully tokenized on every hit.
        token_ids = self.snippet_tokenizer.encode(text[: 16 * n], add_special_tokens=False).ids
        if len(token_ids) <= n:
            return text[: 16 * n]
        return self.snippet_tokenizer.decode(token_ids[:n], skip_special_tokens=True)

    def search_many(
        self, queries: list[str], topk: int | None = None, return_scores: bool = False
    ) -> list[list[DocumentResult]]:
        """Search the top-k documents for each query with one encode and one index search.

        Args:
            queries: Search query strings.
            topk: Number of results per query (falls back to ``cfg.topk``).
            return_scores: If True, also return FAISS inner-product scores.

        Returns:
            One list of DocumentResult per query, ordered by descending score.
        """
        k = topk or self.cfg.topk
        query_emb = self.encode_queries(queries)
        if self.search_lock is not None:
            with self.search_lock:
                scores, idxs = self.index.search(query_emb, k=k)
        else:
            scores, idxs = self.index.search(query_emb, k=k)

        results: list[list[DocumentResult]] = []
        for row_scores, row_idxs in zip(scores, idxs, strict=True):
            # NOTE: Search-R1's Wikipedia corpus has "id" and "contents" keys but no title
            passages = [self.corpus[int(i)] for i in row_idxs]
            results.append(
                [
                    DocumentResult(
                        document=Document(id=p["id"], title=None, contents=self.snippet(p["contents"])),
                        score=(float(s) if return_scores else None),
                    )
                    for p, s in zip(passages, row_scores, strict=False)
                ]
            )
        return results


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------

app = FastAPI(title="PhantomEnvs Retrieval Server")

# Global retriever — loaded once at startup, reused for all requests
retriever: DenseRetriever | None = None


class QueryRequest(BaseModel):
    """Request body for POST /retrieve."""

    query: str = Field(description="Natural language search query.")
    topk: int | None = Field(
        default=None, description="Number of results to return; falls back to server default."
    )
    return_scores: bool = Field(
        default=False, description="Whether to include FAISS retrieval scores in the response."
    )


class BatchQueryRequest(BaseModel):
    """Request body for POST /retrieve_batch."""

    queries: list[str] = Field(description="Natural language search queries.")
    topk: int | None = Field(
        default=None, description="Number of results per query; falls back to server default."
    )
    return_scores: bool = Field(
        default=False, description="Whether to include FAISS retrieval scores in the response."
    )


class BatchRetrievalResponse(BaseModel):
    """Response body returned by POST /retrieve_batch: one RetrievalResponse per query, in order."""

    results: list[RetrievalResponse] = Field(description="Top-k documents for each query.")


@app.get("/health")
def health() -> dict[str, str]:
    """Report readiness (the server only starts answering once the index is loaded)."""
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve(request: QueryRequest) -> RetrievalResponse:
    """Retrieve the top-k documents for one query."""
    assert retriever is not None, "Retriever not initialized"
    [doc_results] = retriever.search_many(
        [request.query], topk=request.topk, return_scores=request.return_scores
    )
    return RetrievalResponse(result=doc_results)


@app.post("/retrieve_batch", response_model=BatchRetrievalResponse)
def retrieve_batch(request: BatchQueryRequest) -> BatchRetrievalResponse:
    """Retrieve the top-k documents for each of several queries."""
    assert retriever is not None, "Retriever not initialized"
    results = retriever.search_many(request.queries, topk=request.topk, return_scores=request.return_scores)
    return BatchRetrievalResponse(results=[RetrievalResponse(result=r) for r in results])


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Load the index and corpus, then serve them."""
    global retriever

    base = OmegaConf.structured(RetrieverServerConfig(encoder=args.encoder, data_dir=args.data_dir))
    cfg = RetrieverServerConfig(
        **OmegaConf.to_container(OmegaConf.merge(base, OmegaConf.from_cli(args.overrides)))
    )
    logger.info(f"Retriever config: {asdict(cfg)}")

    # Bind before loading the index (which can take minutes) so the port can be reported right away.
    # uvicorn starts listening on it once the index is loaded.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", cfg.port))
    port = sock.getsockname()[1]
    if cfg.port_file:
        Path(cfg.port_file).write_text(f"{port}\n")

    retriever = DenseRetriever(cfg)

    logger.info(f"Starting retrieval server at http://0.0.0.0:{port}")
    uvicorn.Server(uvicorn.Config(app)).run(sockets=[sock])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the FAISS dense retrieval server.")
    parser.add_argument("--encoder", required=True, choices=list(ENCODERS), help="Encoder preset.")
    parser.add_argument("--data_dir", required=True, help="Dir with corpus.jsonl and <encoder>_Flat.index.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    main(args)
