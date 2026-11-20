#!/usr/bin/env -S uv run python
"""The retrieval MCP server: a retriever behind one search tool, over the Model Context Protocol.

An agent that speaks MCP, such as the pi harness, calls the tool without knowing what answers: bm25,
a dense encoder, or any other backend of base.py. See search_tool.py for the tool itself.

One server answers every episode of a run, so it keeps no state per client: each call is one HTTP
request. Searches that arrive together are answered in one batch (batching.py), and a repeated query
from its earlier result.

The server listens on a local port and, with socket=<path>, on a Unix socket too. A sandbox without
a network reaches the socket once it is mounted into it.

Usage:
    uv run python -m phantom_envs.retriever.mcp_server --retriever bm25 --data_dir data/envs/pw

    # A dense retriever, on a kernel-chosen port and a socket for sandboxes
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.mcp_server \
        --retriever qwen3-4b --data_dir data/hp500 port=0 port_file=/tmp/mcp.port socket=/tmp/mcp.sock

API:
    GET  /health   {"status": "ok"} once the retriever is loaded (see search_tool.wait_for_server)
    POST /mcp      the MCP endpoint (streamable HTTP), with the tool search(query)
"""

import argparse
import asyncio
import logging
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated

import uvicorn
from mcp.server.mcpserver import MCPServer
from pydantic import Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from phantom_envs._types import DocumentResult
from phantom_envs.retriever.base import Retriever, open_retriever
from phantom_envs.retriever.batching import QueryBatcher
from phantom_envs.retriever.search_tool import (
    HEALTH_PATH,
    QUERY_DESCRIPTION,
    SERVER_INSTRUCTIONS,
    SERVER_NAME,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    format_results,
)
from phantom_envs.retriever.serving import RetrieverServerConfig, bind_port, parse_args, snippet_cutter
from phantom_envs.retriever.snippets import SnippetCutter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# The path of the MCP endpoint
MCP_PATH: str = "/mcp"


@dataclass
class McpServerConfig(RetrieverServerConfig):
    """Settings of the retrieval MCP server."""

    socket: str | None = field(
        default=None, metadata={"description": "If set, also listen on a Unix socket at this path."}
    )
    cache_size: int = field(
        default=10_000,
        metadata={"description": "Most queries whose results are kept: episodes often repeat a query."},
    )
    max_batch_size: int = field(
        default=64,
        metadata={
            "description": "Most concurrent searches answered by one retriever call. 1 encodes each "
            "query alone, but then searches queue up under load and can time out (see batching.py)."
        },
    )


def create_server(
    retriever: Retriever,
    topk: int = 3,
    cutter: SnippetCutter | None = None,
    cache_size: int = 10_000,
    max_batch_size: int = 64,
) -> MCPServer:
    """Build the MCP server over a loaded retriever.

    Args:
        retriever: The retriever that answers every search.
        topk: Number of documents per search. The agent doesn't choose it, so it is one setting of
            an experiment.
        cutter: Cuts each returned document to a snippet; None returns full documents.
        cache_size: Most queries whose results are kept.
        max_batch_size: Most concurrent searches answered by one retriever call (see batching.py).

    Returns:
        The server, with the search tool and a /health route.
    """
    server = MCPServer(SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    batcher = QueryBatcher(retriever, topk, max_batch_size)
    # Each query's search, running or done. Episodes on one question often make the same query, even
    # at the same moment; they share one search. Oldest first, as dicts keep insertion order.
    searches: dict[str, asyncio.Task[str]] = {}

    def cut_and_format(results: list[DocumentResult]) -> str:
        """Cut each document to a snippet, if the server serves snippets, and format the results."""
        if cutter is not None:
            for result in results:
                result.document.contents = cutter.cut(result.document.contents)
        return format_results(results)

    async def results_for(query: str) -> str:
        """Search one query in the next batch, and format its documents off the event loop."""
        results = await asyncio.wrap_future(batcher.submit(query))
        return await asyncio.to_thread(cut_and_format, results)

    @server.tool(name=TOOL_NAME, description=TOOL_DESCRIPTION, structured_output=False)
    async def search(query: Annotated[str, Field(description=QUERY_DESCRIPTION)]) -> str:
        query = query.strip()
        task = searches.get(query)
        if task is None:
            task = searches[query] = asyncio.create_task(results_for(query))
            # Forget the oldest searches, but never the one this call waits on
            while len(searches) > cache_size and next(iter(searches)) != query:
                del searches[next(iter(searches))]
        try:
            # Shielded: a caller that gives up must not cancel the search for the others
            return await asyncio.shield(task)
        except Exception:
            # A failed search is not kept, unless a newer search of the query has taken its place
            if searches.get(query) is task:
                del searches[query]
            raise

    @server.custom_route(HEALTH_PATH, methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return server


def http_app(server: MCPServer) -> Starlette:
    """Return the web app that serves an MCP server over streamable HTTP.

    Args:
        server: The MCP server.

    Returns:
        The app. It is stateless and answers in plain JSON: a tool call is one request, which any
        client can make at any time.
    """
    return server.streamable_http_app(streamable_http_path=MCP_PATH, stateless_http=True, json_response=True)


def bind_unix_socket(path: str | Path) -> socket.socket:
    """Bind a Unix socket, replacing a stale one left at the path.

    Args:
        path: Where to create the socket. Unix socket paths hold at most ~100 characters.

    Returns:
        The bound socket.
    """
    Path(path).unlink(missing_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    return sock


def main(args: argparse.Namespace) -> None:
    """Load the retriever, then serve it."""
    cfg = McpServerConfig.from_args(args)
    logger.info(f"Retriever config: {asdict(cfg)}")

    # Local only: sandboxes come in through the Unix socket
    sockets = [bind_port(cfg, host="127.0.0.1")]
    if cfg.socket:
        sockets.append(bind_unix_socket(cfg.socket))

    cutter = snippet_cutter(cfg)
    retriever = open_retriever(cfg.retriever, cfg.data_dir, faiss_gpu=cfg.faiss_gpu, device=cfg.device)
    server = create_server(
        retriever, topk=cfg.topk, cutter=cutter, cache_size=cfg.cache_size, max_batch_size=cfg.max_batch_size
    )
    app = http_app(server)

    logger.info(f"Starting MCP server at http://127.0.0.1:{sockets[0].getsockname()[1]}{MCP_PATH}")
    # No access log: a training run makes a request per search of every episode
    uvicorn.Server(uvicorn.Config(app, access_log=False)).run(sockets=sockets)


if __name__ == "__main__":
    main(parse_args("Launch the retrieval MCP server."))
