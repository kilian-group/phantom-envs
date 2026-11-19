"""Coding-agent harnesses: programs that run the agent's loop inside the sandbox.

Without a harness, a trainer samples each turn itself and `loop` decides what to run. A harness such
as pi takes that loop over: it gets the task as its prompt, calls the model through an
OpenAI-compatible endpoint, runs the model's commands itself, and exits when the agent is done. The
verifier then reads the answer file, as in every shell episode.

A harness needs real processes, so it runs in the bwrap sandbox. The sandbox has no network; the
model endpoint reaches it through a relay over a Unix socket (see `relay`).

Usage:

    harness = PiHarness(max_turns=20)
    endpoint = ModelEndpoint(base_url="http://127.0.0.1:8000/v1", model="policy")
    with open_sandbox("bwrap", corpus_path, rootfs=rootfs) as sandbox:
        run = await run_harness(harness, sandbox, build_instruction(question), endpoint)
        scores = verify(sandbox.read_answer(), gts, data_source)

To add a harness, subclass `Harness` and put its program in the sandbox image.
"""

import asyncio
import shlex
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from phantom_envs.terminal import relay
from phantom_envs.terminal.bwrap import SHARED_DIR, BwrapSandbox

# The port a harness calls inside the sandbox, and the socket (in the shared dir) behind it
ENDPOINT_PORT: int = 8080
ENDPOINT_SOCKET: str = "model.sock"


@dataclass
class ModelEndpoint:
    """An OpenAI-compatible endpoint that serves the agent's model."""

    base_url: str = field(metadata={"description": "The API's base URL, over plain HTTP."})
    model: str = field(metadata={"description": "The model name to request."})
    api_key: str = field(default="none", metadata={"description": "The API key to send."})
    context_window: int = field(
        default=32768, metadata={"description": "Most tokens of a prompt plus its completion."}
    )
    max_tokens: int = field(default=2048, metadata={"description": "Most tokens the model writes per turn."})


@dataclass
class HarnessRun:
    """What a harness reports about one episode."""

    turns: int = field(default=0, metadata={"description": "Model calls the agent made."})
    commands: int = field(default=0, metadata={"description": "Tool calls the agent made."})
    stop_reason: str = field(
        default="stop",
        metadata={
            "description": "Why the episode ended: stop (the agent finished or used up its turns), "
            "length (a turn hit its token limit), context_length (the prompt outgrew the model's "
            "context), timeout (the episode's time limit) or error."
        },
    )
    error: str | None = field(default=None, metadata={"description": "The error message, if any."})


class Harness(ABC):
    """A coding agent that runs inside a `BwrapSandbox`."""

    @abstractmethod
    def run(
        self, sandbox: BwrapSandbox, instruction: str, endpoint_url: str, endpoint: ModelEndpoint
    ) -> HarnessRun:
        """Run one episode to its end. Blocks; the agent leaves its answer in the sandbox.

        Args:
            sandbox: The episode's sandbox.
            instruction: The task, as `task.build_instruction` wrote it.
            endpoint_url: The model API's base URL as the sandbox reaches it.
            endpoint: The model endpoint's settings; its `base_url` is the host's view.

        Returns:
            What happened.
        """


def relay_command(sandbox: BwrapSandbox) -> str:
    """Return a bash snippet that starts the sandbox side of the endpoint relay and waits for it.

    The relay runs in the background of the harness's command, so the sandbox stops it when that
    command ends.

    Args:
        sandbox: The sandbox; the relay script is copied into its shared dir.

    Returns:
        The snippet, to put before the harness's own command.
    """
    shutil.copy(relay.__file__, sandbox.shared_dir / "relay.py")
    start = f"python3 {SHARED_DIR}/relay.py {ENDPOINT_PORT} {SHARED_DIR}/{ENDPOINT_SOCKET} &"
    wait = f"until (exec 3<> /dev/tcp/127.0.0.1/{ENDPOINT_PORT}) 2> /dev/null; do sleep 0.02; done"
    return f"{start}\n{wait}\n"


def env_prefix(env: dict[str, str]) -> str:
    """Return `NAME=value ...` assignments to put before a command, quoted for bash."""
    return " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())


async def run_harness(
    harness: Harness,
    sandbox: BwrapSandbox,
    instruction: str,
    endpoint: ModelEndpoint,
    executor: Executor | None = None,
) -> HarnessRun:
    """Run one episode of a harness, with the model endpoint relayed into the sandbox.

    Args:
        harness: The harness.
        sandbox: The episode's sandbox.
        instruction: The task text.
        endpoint: The model endpoint, reachable from this process.
        executor: Where the blocking run goes; the event loop's default executor if None.

    Returns:
        What the harness reported.

    Raises:
        ValueError: If the endpoint is not plain HTTP, which is all the relay carries.
    """
    url = urlsplit(endpoint.base_url)
    if url.scheme != "http":
        raise ValueError(f"The relay carries plain HTTP only, not {endpoint.base_url!r}")
    endpoint_url = f"http://127.0.0.1:{ENDPOINT_PORT}{url.path}"
    socket_path = Path(sandbox.shared_dir) / ENDPOINT_SOCKET
    async with relay.serve_socket(socket_path, url.hostname, url.port or 80):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, harness.run, sandbox, instruction, endpoint_url, endpoint)
