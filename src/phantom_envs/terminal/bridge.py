"""Harness episodes whose model calls are answered by Python code, one turn at a time.

A harness calls its model through an OpenAI-compatible endpoint. Some trainers serve one (SkyRL, via
skycap); others, like tinker-cookbook's loop, sample every turn themselves, and an eval of an API model
keeps the API key out of the sandbox. For those, `ChatBridge` turns the harness's calls around: each
chat request is handed to the caller, who answers it with the next turn. `BridgedEpisode` runs a whole
episode of an `agent.AgentSetup` this way: sandbox, harness, bridge and answer file.

Usage:

    episode = BridgedEpisode(AgentSetup(corpus_path, rootfs))
    request = await episode.start(question)                      # the first chat request
    messages = request["messages"]                               # system prompt and task
    while True:
        turn = ...                                  # the next assistant message, from any model
        new_messages = await episode.step(turn)     # the tool results that followed it
        if new_messages is None:                    # the harness exited
            break
        messages += [turn, *new_messages]
    run, answer = await episode.finish()

A harness makes one call at a time, so requests are answered in the order they arrive.
"""

import asyncio
import json
from concurrent.futures import Executor
from typing import Any

from aiohttp import web

from phantom_envs.terminal.agent import AgentSetup
from phantom_envs.terminal.bwrap import BwrapSandbox
from phantom_envs.terminal.harness import HarnessRun, ModelEndpoint, run_harness

# The model name the harness requests; the caller decides which model answers
MODEL_NAME: str = "policy"


def completion_chunks(message: dict[str, Any], include_usage: bool) -> list[dict[str, Any]]:
    """Return an assistant message as the chunks of a streamed chat completion.

    Args:
        message: The assistant message, in OpenAI's form.
        include_usage: Whether to end with a usage chunk (`stream_options.include_usage`).

    Returns:
        The chunks: the whole message as one delta, the finish reason, and optionally the usage.
    """
    delta = {key: value for key, value in message.items() if value is not None}
    if "tool_calls" in delta:
        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(delta["tool_calls"])]
    finish_reason = "tool_calls" if message.get("tool_calls") else "stop"

    def chunk(delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
        choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        return {"id": "chatcmpl-bridge", "object": "chat.completion.chunk", "choices": [choice]}

    chunks = [chunk(delta, None), chunk({}, finish_reason)]
    if include_usage:
        # The caller knows the token counts, not this endpoint
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        chunks.append(
            {"id": "chatcmpl-bridge", "object": "chat.completion.chunk", "choices": [], "usage": usage}
        )
    return chunks


class ChatBridge:
    """Serves `POST /v1/chat/completions` on a local port and lets the caller answer each request."""

    def __init__(self) -> None:
        self.requests: asyncio.Queue[tuple[dict[str, Any], asyncio.Future]] = asyncio.Queue()
        self.pending: asyncio.Future | None = None
        self.runner: web.AppRunner | None = None

    async def start(self) -> str:
        """Start serving.

        Returns:
            The endpoint's base URL, `http://127.0.0.1:<port>/v1`.
        """
        app = web.Application(client_max_size=1024**3)
        app.router.add_post("/v1/chat/completions", self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        return f"http://127.0.0.1:{self.runner.addresses[0][1]}/v1"

    async def handle(self, request: web.Request) -> web.StreamResponse:
        """Hand a chat request to the caller and send back the caller's reply."""
        body = await request.json()
        reply: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.requests.put((body, reply))
        message = await reply

        if not body.get("stream"):
            finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
            choice = {"index": 0, "message": message, "finish_reason": finish_reason}
            return web.json_response(
                {"id": "chatcmpl-bridge", "object": "chat.completion", "choices": [choice]}
            )

        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        for chunk in completion_chunks(message, include_usage):
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    async def next_request(self, unless: asyncio.Future | None = None) -> dict[str, Any] | None:
        """Wait for the harness's next chat request.

        Args:
            unless: Stop waiting when this completes, e.g. the task that runs the harness.

        Returns:
            The request's JSON body, to answer with `reply` before waiting for another; or
            None if `unless` completed first.
        """
        request = asyncio.ensure_future(self.requests.get())
        await asyncio.wait({request} | ({unless} if unless else set()), return_when=asyncio.FIRST_COMPLETED)
        if not request.done():
            request.cancel()
            return None
        body, self.pending = request.result()
        return body

    def reply(self, message: dict[str, Any]) -> None:
        """Answer the request that `next_request` returned.

        Args:
            message: The assistant message, in OpenAI's form (`content`, `tool_calls`, ...).
        """
        if self.pending is None or self.pending.done():
            raise RuntimeError("There is no chat request to answer")
        self.pending.set_result(message)

    async def close(self) -> None:
        """Stop serving; a request still waiting for its reply is dropped."""
        if self.pending is not None and not self.pending.done():
            self.pending.cancel()
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None


class BridgedEpisode:
    """One harness episode in a bwrap sandbox, whose model calls the caller answers.

    The harness replays the whole conversation in every request, and `step` returns only what is
    new. The caller keeps the conversation, with each turn exactly as its model wrote it.
    """

    def __init__(
        self,
        agent: AgentSetup,
        *,
        context_window: int = 32768,
        max_tokens: int = 2048,
        executor: Executor | None = None,
    ) -> None:
        """Set up the episode; `start` opens the sandbox and starts the harness.

        Args:
            agent: The agent: its harness, tools, sandbox and task.
            context_window: Most tokens of a prompt plus its turn, as the harness is told.
            max_tokens: Most tokens per turn, as the harness is told.
            executor: Where the sandbox and the harness block; the event loop's default if None.
        """
        self.agent = agent
        self.harness = agent.harness()
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.executor = executor
        self.bridge = ChatBridge()
        self.sandbox: BwrapSandbox | None = None
        self.run: asyncio.Task[HarnessRun] | None = None

    async def start(self, question: str) -> dict[str, Any]:
        """Open the sandbox, start the harness on the question, and wait for its first model call.

        Args:
            question: The question; the agent gets it in the task text for its tools.

        Returns:
            The first chat request's JSON body: the harness's system prompt and the task in
            `messages`, and its `tools`.

        Raises:
            RuntimeError: If the harness exits before its first model call.
        """
        loop = asyncio.get_running_loop()
        self.sandbox = await loop.run_in_executor(self.executor, self.agent.open_sandbox)
        endpoint = ModelEndpoint(
            base_url=await self.bridge.start(),
            model=MODEL_NAME,
            context_window=self.context_window,
            max_tokens=self.max_tokens,
        )
        instruction = self.agent.instruction(question)
        self.run = asyncio.create_task(
            run_harness(
                self.harness, self.sandbox, instruction, endpoint, self.executor, self.agent.mcp_servers
            )
        )
        request = await self.bridge.next_request(unless=self.run)
        if request is None:
            raise RuntimeError(f"The harness ended before its first model call: {self.run.result()}")
        return request

    async def step(self, turn: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Answer the harness's model call, and wait for its next one.

        Args:
            turn: The assistant message, in OpenAI's form.

        Returns:
            The messages the harness added after the turn (its tool results), or None if it exited.
        """
        self.bridge.reply(turn)
        request = await self.bridge.next_request(unless=self.run)
        if request is None:
            return None
        messages = request["messages"]
        last_turn = max(i for i, message in enumerate(messages) if message["role"] == "assistant")
        return messages[last_turn + 1 :]

    async def finish(self) -> tuple[HarnessRun | None, str | None]:
        """Read the answer file and close the episode.

        Returns:
            What the harness reported, or None if it was still running (the caller ended the episode
            early); and the answer file's contents.
        """
        ended = self.run is not None and self.run.done()
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(self.executor, self.sandbox.read_answer)
        await self.close()
        if not ended:
            return None, answer
        try:
            return self.run.result(), answer
        except Exception as error:  # noqa: BLE001 (reported as a failed run)
            return HarnessRun(stop_reason="error", error=f"{type(error).__name__}: {error}"), answer

    async def close(self) -> None:
        """Stop the harness and close the bridge and the sandbox. Safe to call more than once."""
        await self.bridge.close()
        if self.sandbox is not None:
            # Closing the sandbox ends the harness's command, which ends its run
            sandbox, self.sandbox = self.sandbox, None
            await asyncio.get_running_loop().run_in_executor(self.executor, sandbox.close)
        if self.run is not None and not self.run.done():
            await asyncio.wait({self.run})
