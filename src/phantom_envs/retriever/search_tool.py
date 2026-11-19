"""The search tool: what an agent sees of a retriever, whichever retriever answers.

The MCP server (mcp_server.py) serves this one tool. Its name, description and result format stay the
same for every backend, so a policy trained with one retriever can be evaluated with another. An MCP
client names a tool after its server, so pi declares it as mcp__retriever__search.

A client that starts with the server waits for it with `wait_for_server`: loading a dense index takes
a while, and pi drops an MCP server's tools if it doesn't answer when an episode starts.
"""

import http.client
import socket
import time

from phantom_envs._types import DocumentResult, RetrievalResponse

SERVER_NAME: str = "retriever"
TOOL_NAME: str = "search"

SERVER_INSTRUCTIONS: str = "Searches a collection of documents."
TOOL_DESCRIPTION: str = (
    "Search the document collection. Returns the documents most relevant to the query, best first."
)
QUERY_DESCRIPTION: str = "What to look for: a question or a few keywords."
NO_RESULTS: str = "No search results found."
# The server's readiness route: it answers once the retriever is loaded
HEALTH_PATH: str = "/health"


def format_results(results: list[DocumentResult]) -> str:
    """Format a search's documents as the text the agent reads.

    Args:
        results: The documents, best first.

    Returns:
        The documents, numbered from Doc 1, as the search env formats them; or a note that the
        search found nothing.
    """
    if not results:
        return NO_RESULTS
    return str(RetrievalResponse(result=results)).strip()


def server_ready(socket_path: str) -> bool:
    """Whether the retrieval MCP server on a Unix socket has loaded its retriever.

    Args:
        socket_path: The server's Unix socket.

    Returns:
        True if the server answers its readiness route.
    """
    connection = http.client.HTTPConnection("localhost", timeout=5)
    try:
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.settimeout(5)
        connection.sock.connect(socket_path)
        connection.request("GET", HEALTH_PATH)
        return connection.getresponse().status == 200
    except OSError:
        return False
    finally:
        connection.close()


def wait_for_server(socket_path: str, timeout: float = 900.0) -> None:
    """Wait until the retrieval MCP server on a Unix socket has loaded its retriever.

    Args:
        socket_path: The server's Unix socket.
        timeout: Most seconds to wait.

    Raises:
        TimeoutError: If the server isn't ready in time.
    """
    deadline = time.monotonic() + timeout
    while not server_ready(socket_path):
        if time.monotonic() > deadline:
            raise TimeoutError(f"No retrieval MCP server answered on {socket_path} within {timeout:g} s")
        time.sleep(1)
