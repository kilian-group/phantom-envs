"""A byte relay between a Unix socket and a TCP port, to reach a model endpoint without a network.

A sandbox has no network, but it shares a directory with the host. So a harness's model calls take
two hops over a Unix socket in that directory:

    harness -> 127.0.0.1:<port>  (inside)  -> socket -> (host) -> the model endpoint

The host side is `serve_socket`. The sandbox side is this file run as a script by the sandbox's
own Python, so it uses only the standard library:

    python3 relay.py <port> <socket path>
"""

import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator
from pathlib import Path


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy bytes from `reader` to `writer` until either side closes."""
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()


async def join(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    upstream: tuple[asyncio.StreamReader, asyncio.StreamWriter],
) -> None:
    """Relay one accepted connection to and from an open upstream connection."""
    upstream_reader, upstream_writer = upstream
    await asyncio.gather(pipe(reader, upstream_writer), pipe(upstream_reader, writer))


@contextlib.asynccontextmanager
async def serve_socket(socket_path: str | Path, host: str, port: int) -> AsyncIterator[None]:
    """Relay connections to a Unix socket on to a TCP endpoint, while the context is open.

    Args:
        socket_path: Where to create the socket. Keep it short: Unix socket paths have a small
            length limit.
        host: The endpoint's host.
        port: The endpoint's port.

    Yields:
        Nothing; the socket accepts connections inside the `async with` block.
    """

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            upstream = await asyncio.open_connection(host, port)
        except OSError:
            writer.close()
            return
        await join(reader, writer, upstream)

    server = await asyncio.start_unix_server(accept, path=str(socket_path))
    try:
        yield
    finally:
        server.close()
        await server.wait_closed()


async def serve_port(port: int, socket_path: str) -> None:
    """Relay connections to `127.0.0.1:port` on to a Unix socket, forever.

    Args:
        port: The local port to listen on.
        socket_path: The socket to forward to.
    """

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            upstream = await asyncio.open_unix_connection(socket_path)
        except OSError:
            writer.close()
            return
        await join(reader, writer, upstream)

    server = await asyncio.start_server(accept, "127.0.0.1", port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve_port(int(sys.argv[1]), sys.argv[2]))
