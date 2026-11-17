"""Tests for the coding-agent harness: the endpoint relay (phantom_envs.terminal.relay) and pi
(phantom_envs.terminal.pi).

The end-to-end test runs pi in a bwrap sandbox against a scripted model, so it needs bubblewrap,
SANDBOX_ROOTFS (see scripts/sandbox/build_rootfs.sh) and aiohttp (the skyrl extra).
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from phantom_envs.terminal import relay
from phantom_envs.terminal.bwrap import MAX_FILE_MIB
from phantom_envs.terminal.harness import ENDPOINT_PORT, ModelEndpoint, env_prefix, run_harness
from phantom_envs.terminal.pi import PiHarness, parse_events
from phantom_envs.terminal.task import ANSWER_PATH, build_instruction


def turn_end(stop_reason: str, tool_results: int = 0, error: str | None = None) -> str:
    """One ``turn_end`` line of pi's JSON event stream."""
    message = {"role": "assistant", "stopReason": stop_reason}
    if error:
        message["errorMessage"] = error
    return json.dumps({"type": "turn_end", "message": message, "toolResults": [{}] * tool_results})


class TestParseEvents:
    def test_counts_turns_and_commands(self):
        output = "\n".join(['{"type":"session"}', turn_end("toolUse", 2), "not json", turn_end("stop")])
        run = parse_events(output)
        assert (run.turns, run.commands, run.stop_reason, run.error) == (2, 2, "stop", None)

    def test_turn_cap_abort_is_not_a_turn(self):
        run = parse_events("\n".join([turn_end("toolUse", 1), turn_end("aborted")]))
        assert (run.turns, run.stop_reason) == (1, "stop")

    def test_context_overflow(self):
        error = '400: {"message":"prompt of 910 tokens leaves no room","code":"context_length_exceeded"}'
        run = parse_events("\n".join([turn_end("toolUse", 1), turn_end("error", error=error)]))
        assert (run.turns, run.stop_reason) == (1, "context_length") and "910 tokens" in run.error

    def test_other_error(self):
        run = parse_events(turn_end("error", error="500: upstream failed"))
        assert (run.turns, run.stop_reason, run.error) == (0, "error", "500: upstream failed")

    def test_token_limit(self):
        assert parse_events(turn_end("length")).stop_reason == "length"


def test_pi_command_carries_the_budgets():
    command = PiHarness(max_turns=7, command_timeout=30).command("Who is Ann's mother?", "policy")
    assert "PHANTOM_MAX_TURNS=7" in command and "PHANTOM_COMMAND_TIMEOUT=30" in command
    assert command.endswith("'Who is Ann'\"'\"'s mother?'")  # the instruction, quoted for bash
    assert "--system-prompt" not in command
    assert "--system-prompt 'Be brief.'" in PiHarness(system_prompt="Be brief.").command("q", "policy")


def test_env_prefix_quotes_values():
    assert env_prefix({"A": "1", "B": "two words"}) == "A=1 B='two words'"


def test_relay_round_trip(tmp_path: Path):
    """Bytes sent to the sandbox side's port reach a TCP server behind the host side's socket."""

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write((await reader.readline()).upper())
        await writer.drain()
        writer.close()

    async def round_trip() -> bytes:
        server = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        socket_path = tmp_path / "model.sock"
        async with server, relay.serve_socket(socket_path, "127.0.0.1", port):
            listener = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)  # to pick a free port
            relay_port = listener.sockets[0].getsockname()[1]
            listener.close()
            await listener.wait_closed()
            sandbox_side = asyncio.create_task(relay.serve_port(relay_port, str(socket_path)))
            try:
                for _ in range(100):  # until the relay listens
                    try:
                        reader, writer = await asyncio.open_connection("127.0.0.1", relay_port)
                        break
                    except ConnectionRefusedError:
                        await asyncio.sleep(0.01)
                writer.write(b"hello\n")
                await writer.drain()
                return await reader.readline()
            finally:
                sandbox_side.cancel()

    assert asyncio.run(round_trip()) == b"HELLO\n"


def chunk(delta: dict, finish_reason: str | None = None) -> bytes:
    """One server-sent event of a streamed chat completion."""
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    body = {"id": "c", "object": "chat.completion.chunk", "model": "policy", "choices": [choice]}
    return f"data: {json.dumps(body)}\n\n".encode()


class ScriptedModel:
    """A chat-completions server that plays a fixed episode: one bash command per turn, then text."""

    def __init__(self, commands: list[str]) -> None:
        self.commands = commands
        self.requests: list[dict] = []

    async def complete(self, request):
        from aiohttp import web

        body = await request.json()
        self.requests.append(body)
        turn = sum(message["role"] == "assistant" for message in body["messages"])
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if turn < len(self.commands):
            function = {"name": "bash", "arguments": json.dumps({"command": self.commands[turn]})}
            call = {"index": 0, "id": f"call_{turn}", "type": "function", "function": function}
            await response.write(chunk({"role": "assistant", "tool_calls": [call]}))
            await response.write(chunk({}, "tool_calls"))
        else:
            await response.write(chunk({"role": "assistant", "content": "Done."}))
            await response.write(chunk({}, "stop"))
        await response.write(b"data: [DONE]\n\n")
        return response

    async def play(self, harness: PiHarness, corpus_path: Path) -> tuple:
        """Run one pi episode against this model; return the harness's report and the answer file."""
        from aiohttp import web

        from phantom_envs.terminal.sandbox import open_sandbox

        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.complete)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        endpoint = ModelEndpoint(base_url=f"http://127.0.0.1:{port}/v1", model="policy")
        try:
            with open_sandbox("bwrap", corpus_path, rootfs=os.environ["SANDBOX_ROOTFS"]) as sandbox:
                run = await run_harness(harness, sandbox, build_instruction("Who is Ann's mother?"), endpoint)
                return run, sandbox.read_answer()
        finally:
            await runner.cleanup()


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A one-document corpus.jsonl."""
    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
    doc = {"id": "0", "title": "Ann Lee", "contents": "# Ann Lee\n\nThe mother of Ann Lee is Bea Lee.\n"}
    path.write_text(json.dumps(doc) + "\n")
    return path


@pytest.fixture
def needs_sandbox() -> None:
    """Skip the test unless this machine can run pi in a bwrap sandbox."""
    pytest.importorskip("aiohttp")
    if shutil.which("bwrap") is None or not os.environ.get("SANDBOX_ROOTFS"):
        pytest.skip("needs bubblewrap and SANDBOX_ROOTFS")


@pytest.mark.usefixtures("needs_sandbox")
class TestPiEpisode:
    def test_agent_answers_through_the_relay(self, corpus_path):
        model = ScriptedModel(["grep -h mother /corpus/*.md", f"echo 'Bea Lee' > {ANSWER_PATH}"])
        run, answer = asyncio.run(model.play(PiHarness(), corpus_path))
        assert (run.turns, run.commands, run.stop_reason) == (3, 2, "stop")
        assert answer == "Bea Lee\n"
        # The model read the first command's output, and pi offered it one tool
        tool_outputs = [m["content"] for m in model.requests[-1]["messages"] if m["role"] == "tool"]
        assert "The mother of Ann Lee is Bea Lee." in str(tool_outputs[0])
        assert [tool["function"]["name"] for tool in model.requests[0]["tools"]] == ["bash"]
        assert f":{ENDPOINT_PORT}" not in str(model.requests[0])  # pi's prompt doesn't leak the endpoint

    def test_budgets(self, corpus_path):
        model = ScriptedModel(["sleep 5; echo slept", "seq 1000", "echo third", "echo fourth"])
        harness = PiHarness(max_turns=3, command_timeout=1, max_output_chars=50)
        run, _ = asyncio.run(model.play(harness, corpus_path))
        assert (run.turns, run.stop_reason) == (3, "stop")  # the turn cap ended the episode
        assert len(model.requests) == 3
        tool_outputs = [str(m["content"]) for m in model.requests[-1]["messages"] if m["role"] == "tool"]
        assert "timed out" in tool_outputs[0] and "slept" not in tool_outputs[0]
        assert "[output truncated: showed 50 of" in tool_outputs[1]

    def test_output_larger_than_the_file_size_limit(self, corpus_path):
        # pi itself writes past the limit: its event stream carries the whole output
        too_big = MAX_FILE_MIB + 1
        model = ScriptedModel(
            [f"head -c {too_big}M /dev/zero | tr '\\0' x", f"head -c {too_big}M /dev/zero > big"]
        )
        run, _ = asyncio.run(model.play(PiHarness(), corpus_path))
        assert (run.turns, run.commands, run.stop_reason) == (3, 2, "stop")
        # A command still can't write a file past the limit
        assert "File size limit exceeded" in str(model.requests[-1]["messages"][-1]["content"])
