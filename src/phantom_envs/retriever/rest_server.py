#!/usr/bin/env -S uv run python
"""The retrieval REST server: a retriever over a data dir's corpus, served as JSON over HTTP.

This is the server behind the search env's <search> tool, which calls it through SearchClient
(client.py). Any retriever of base.py can answer: bm25, or a dense encoder with its FAISS index.
FAISS is not a dependency of the venv; run a dense retriever with `uv run --with faiss-cpu==1.12.0`
(or put a conda faiss-gpu on the path for faiss_gpu).

Usage:
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.rest_server \
        --retriever e5 --data_dir data/pw/depth_16_size_1000_seed_1_50k

    uv run python -m phantom_envs.retriever.rest_server --retriever bm25 --data_dir data/envs/pw

    # Override any config field at runtime, e.g. a kernel-chosen port written to a file
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.rest_server \
        --retriever qwen3-4b --data_dir data/browsecomp-plus-c512 \
        port=0 port_file=/tmp/retriever.port snippet_max_tokens=512

API:
    GET  /health           {"status": "ok"} once the retriever is loaded
    POST /retrieve         {"query": "...", "topk": 3, "return_scores": true}
                           -> {"result": [{"document": {...}, "score": float}, ...]}
    POST /retrieve_batch   {"queries": ["...", ...], "topk": 3, "return_scores": true}
                           -> {"results": [{"result": [...]}, ...]}
                           (dense: padded batch encoding rarely ranks differently from /retrieve)
"""

import argparse
import logging
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from phantom_envs._types import RetrievalResponse
from phantom_envs.retriever.base import Retriever, open_retriever
from phantom_envs.retriever.serving import RetrieverServerConfig, bind_port, parse_args, snippet_cutter
from phantom_envs.retriever.snippets import SnippetCutter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for POST /retrieve."""

    query: str = Field(description="Natural language search query.")
    topk: int | None = Field(
        default=None, description="Number of results to return; falls back to server default."
    )
    return_scores: bool = Field(
        default=False, description="Whether to include retrieval scores in the response."
    )


class BatchQueryRequest(BaseModel):
    """Request body for POST /retrieve_batch."""

    queries: list[str] = Field(description="Natural language search queries.")
    topk: int | None = Field(
        default=None, description="Number of results per query; falls back to server default."
    )
    return_scores: bool = Field(
        default=False, description="Whether to include retrieval scores in the response."
    )


class BatchRetrievalResponse(BaseModel):
    """Response body returned by POST /retrieve_batch: one RetrievalResponse per query, in order."""

    results: list[RetrievalResponse] = Field(description="Top-k documents for each query.")


def create_app(retriever: Retriever, topk: int = 3, cutter: SnippetCutter | None = None) -> FastAPI:
    """Build the server's app over a loaded retriever.

    Args:
        retriever: The retriever that answers every request.
        topk: Number of documents for a request that names none.
        cutter: Cuts each returned document to a snippet; None returns full documents.

    Returns:
        The app, with the routes listed in this module's docstring.
    """
    app = FastAPI(title="PhantomEnvs Retrieval Server")

    def search(queries: list[str], request_topk: int | None, return_scores: bool) -> list[RetrievalResponse]:
        """Search each query, and shape the results as the request asked."""
        responses = []
        for results in retriever.search(queries, request_topk or topk):
            for result in results:
                if cutter is not None:
                    result.document.contents = cutter.cut(result.document.contents)
                if not return_scores:
                    result.score = None
            responses.append(RetrievalResponse(result=results))
        return responses

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report readiness (the server only starts answering once the retriever is loaded)."""
        return {"status": "ok"}

    @app.post("/retrieve", response_model=RetrievalResponse)
    def retrieve(request: QueryRequest) -> RetrievalResponse:
        """Retrieve the top-k documents for one query."""
        [response] = search([request.query], request.topk, request.return_scores)
        return response

    @app.post("/retrieve_batch", response_model=BatchRetrievalResponse)
    def retrieve_batch(request: BatchQueryRequest) -> BatchRetrievalResponse:
        """Retrieve the top-k documents for each of several queries."""
        return BatchRetrievalResponse(results=search(request.queries, request.topk, request.return_scores))

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Load the retriever, then serve it."""
    cfg = RetrieverServerConfig.from_args(args)
    logger.info(f"Retriever config: {asdict(cfg)}")

    sock = bind_port(cfg, host="0.0.0.0")
    cutter = snippet_cutter(cfg)
    retriever = open_retriever(cfg.retriever, cfg.data_dir, faiss_gpu=cfg.faiss_gpu, device=cfg.device)
    app = create_app(retriever, topk=cfg.topk, cutter=cutter)

    logger.info(f"Starting retrieval server at http://0.0.0.0:{sock.getsockname()[1]}")
    uvicorn.Server(uvicorn.Config(app)).run(sockets=[sock])


if __name__ == "__main__":
    main(parse_args("Launch the retrieval REST server."))
