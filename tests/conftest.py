"""Fixtures shared by the terminal tests: a check for the bwrap sandbox, and a retrieval MCP server."""

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from phantom_envs.terminal.harness import McpServer


@pytest.fixture
def needs_sandbox() -> None:
    """Skip the test unless this machine can run pi in a bwrap sandbox."""
    pytest.importorskip("aiohttp")
    if shutil.which("bwrap") is None or not os.environ.get("SANDBOX_ROOTFS"):
        pytest.skip("needs bubblewrap and SANDBOX_ROOTFS")


@pytest.fixture
def search_server() -> Iterator[McpServer]:
    """The retrieval MCP server on a Unix socket, over a retriever that always finds Ann Lee's article."""
    uvicorn = pytest.importorskip("uvicorn")
    from phantom_envs._types import Document, DocumentResult
    from phantom_envs.retriever.base import Retriever
    from phantom_envs.retriever.mcp_server import bind_unix_socket, create_server, http_app

    class AnnLee(Retriever):
        def search(self, queries: list[str], topk: int) -> list[list[DocumentResult]]:
            document = Document(contents="# Ann Lee\n\nThe mother of Ann Lee is Bea Lee.")
            return [[DocumentResult(document=document, score=1.0)] for _ in queries]

    # A short dir: a Unix socket's path must stay under ~100 characters
    socket_dir = Path(tempfile.mkdtemp(prefix="mcp-", dir="/tmp"))
    socket_path = socket_dir / "retriever.sock"
    server = uvicorn.Server(uvicorn.Config(http_app(create_server(AnnLee())), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [bind_unix_socket(socket_path)]})
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            pytest.fail("The test's MCP server didn't start")
        time.sleep(0.01)
    yield McpServer(name="retriever", socket_path=str(socket_path), tools=("search",))
    server.should_exit = True
    thread.join()
    shutil.rmtree(socket_dir, ignore_errors=True)
