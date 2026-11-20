"""Tests for the coding-agent harness: the endpoint relay (phantom_envs.terminal.relay) and pi
(phantom_envs.terminal.pi).

The end-to-end test runs pi in a bwrap sandbox against a scripted model, so it needs bubblewrap,
SANDBOX_ROOTFS (see scripts/sandbox/build_rootfs.sh) and aiohttp (the skyrl extra).
"""

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from phantom_envs.terminal import relay
from phantom_envs.terminal.bwrap import MAX_FILE_MIB
from phantom_envs.terminal.harness import (
    ENDPOINT_PORT,
    MCP_PORT,
    HarnessRun,
    McpServer,
    ModelEndpoint,
    env_prefix,
    mcp_mounts,
    run_harness,
)
from phantom_envs.terminal.pi import PiHarness, parse_events
from phantom_envs.terminal.task import ANSWER_PATH, build_instruction

if TYPE_CHECKING:
    import aiohttp.web


def turn_end(stop_reason: str, tool_results: int = 0, error: str | None = None) -> str:
    """One turn_end line of pi's JSON event stream."""
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


RETRIEVER = McpServer(name="retriever", socket_path="/host/retriever.sock", tools=("search",))


class TestMcpServers:
    def test_tools_are_named_after_their_server(self):
        assert RETRIEVER.tool_names == ["mcp__retriever__search"]
        assert mcp_mounts([RETRIEVER]) == {"/run/mcp/retriever.sock": "/host/retriever.sock"}

    def test_pi_declares_only_the_tools_it_is_given(self):
        command = PiHarness().command("q", "policy", [RETRIEVER])
        assert "--tools bash,mcp__retriever__search" in command and "--extension builtin:mcp" in command
        assert "--tools write,mcp__retriever__search" in PiHarness(tools=("write",)).command(
            "q", "policy", [RETRIEVER]
        )
        assert "--tools bash " in PiHarness().command(
            "q", "policy"
        ) and "builtin:mcp" not in PiHarness().command("q", "policy")

    def test_pi_config_points_at_the_relayed_server(self, tmp_path: Path):
        endpoint = ModelEndpoint(base_url="http://127.0.0.1:1/v1", model="policy")
        PiHarness(command_timeout=30).write_config(
            tmp_path, "http://127.0.0.1:8080/v1", endpoint, "ulimit -v 1", [RETRIEVER]
        )
        servers = json.loads((tmp_path / "mcp.json").read_text())["mcpServers"]
        assert servers == {
            "retriever": {"url": f"http://127.0.0.1:{MCP_PORT}/mcp", "exposure": "direct", "timeout": 30}
        }

    def test_mcp_calls_are_counted(self):
        results = [{"toolName": "bash"}, {"toolName": "mcp__retriever__search"}]
        message = {"role": "assistant", "stopReason": "toolUse"}
        run = parse_events(json.dumps({"type": "turn_end", "message": message, "toolResults": results}))
        assert (run.commands, run.mcp_calls) == (2, 1)

    def test_a_missing_tool_fails_the_episode(self):
        missing = json.dumps({"type": "phantom_missing_tools", "tools": ["mcp__retriever__search"]})
        run = parse_events("\n".join([missing, turn_end("aborted")]))
        assert (run.turns, run.stop_reason) == (0, "error") and "mcp__retriever__search" in run.error


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


def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
    """One server-sent event of a streamed chat completion."""
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    body = {"id": "c", "object": "chat.completion.chunk", "model": "policy", "choices": [choice]}
    return f"data: {json.dumps(body)}\n\n".encode()


class ScriptedModel:
    """A chat-completions server that plays a fixed episode: one tool call per turn, then text.

    A call is a bash command, or (tool name, arguments) for another tool.
    """

    def __init__(self, calls: list[str | tuple[str, dict[str, Any]]]) -> None:
        self.calls = [("bash", {"command": call}) if isinstance(call, str) else call for call in calls]
        self.requests: list[dict[str, Any]] = []

    async def complete(self, request: "aiohttp.web.Request") -> "aiohttp.web.StreamResponse":
        """Answer a chat request with the next scripted turn, streamed."""
        from aiohttp import web

        body = await request.json()
        self.requests.append(body)
        turn = sum(message["role"] == "assistant" for message in body["messages"])
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if turn < len(self.calls):
            name, arguments = self.calls[turn]
            function = {"name": name, "arguments": json.dumps(arguments)}
            call = {"index": 0, "id": f"call_{turn}", "type": "function", "function": function}
            await response.write(chunk({"role": "assistant", "tool_calls": [call]}))
            await response.write(chunk({}, "tool_calls"))
        else:
            await response.write(chunk({"role": "assistant", "content": "Done."}))
            await response.write(chunk({}, "stop"))
        await response.write(b"data: [DONE]\n\n")
        return response

    async def play(
        self, harness: PiHarness, corpus_path: Path | None, mcp_servers: tuple[McpServer, ...] = ()
    ) -> tuple[HarnessRun, str | None]:
        """Run one pi episode against this model; return the harness's report and the answer file.

        The sandbox holds the corpus, or none and the MCP servers' sockets instead.
        """
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
            rootfs, mounts = os.environ["SANDBOX_ROOTFS"], mcp_mounts(mcp_servers)
            with open_sandbox("bwrap", corpus_path, rootfs=rootfs, mounts=mounts) as sandbox:
                instruction = build_instruction("Who is Ann's mother?")
                run = await run_harness(harness, sandbox, instruction, endpoint, mcp_servers=mcp_servers)
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


@pytest.mark.usefixtures("needs_sandbox")
class TestPiEpisodeWithSearch:
    def test_agent_searches_through_the_mounted_socket(self, search_server: McpServer):
        search = ("mcp__retriever__search", {"query": "mother of Ann Lee"})
        model = ScriptedModel([search, f"ls /corpus; echo 'Bea Lee' > {ANSWER_PATH}"])
        run, answer = asyncio.run(model.play(PiHarness(), None, (search_server,)))
        assert (run.turns, run.commands, run.mcp_calls, run.stop_reason) == (3, 2, 1, "stop")
        assert answer == "Bea Lee\n"
        # pi offered the search tool beside bash, and the model read the document the search found
        tools = [tool["function"]["name"] for tool in model.requests[0]["tools"]]
        assert tools == ["bash", "mcp__retriever__search"]
        search_output, bash_output = (
            str(m["content"]) for m in model.requests[-1]["messages"] if m["role"] == "tool"
        )
        assert "Doc 1:" in search_output and "The mother of Ann Lee is Bea Lee." in search_output
        assert "No such file" in bash_output  # the sandbox holds no corpus

    def test_search_output_is_cut_like_any_tool_output(self, search_server: McpServer):
        model = ScriptedModel([("mcp__retriever__search", {"query": "mother of Ann Lee"})])
        asyncio.run(model.play(PiHarness(max_output_chars=20), None, (search_server,)))
        [search_output] = (str(m["content"]) for m in model.requests[-1]["messages"] if m["role"] == "tool")
        assert "[output truncated: showed 20 of" in search_output

    def test_write_tool_replaces_bash_for_a_search_only_agent(self, search_server: McpServer):
        write = ("write", {"path": ANSWER_PATH, "content": "Bea Lee"})
        model = ScriptedModel([write])
        _, answer = asyncio.run(model.play(PiHarness(tools=("write",)), None, (search_server,)))
        assert answer.strip() == "Bea Lee"
        tools = [tool["function"]["name"] for tool in model.requests[0]["tools"]]
        assert tools == ["write", "mcp__retriever__search"]

    def test_episode_fails_when_the_server_does_not_answer(self, tmp_path: Path, needs_sandbox: None):
        """pi must not run an episode without the search tool it was promised."""
        import socket

        # Bound but never listening: connections are refused
        dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        dead.bind(str(tmp_path / "dead.sock"))
        server = McpServer(name="retriever", socket_path=str(tmp_path / "dead.sock"), tools=("search",))
        model = ScriptedModel([f"echo 'Bea Lee' > {ANSWER_PATH}"])
        run, answer = asyncio.run(model.play(PiHarness(), None, (server,)))
        dead.close()
        assert run.stop_reason == "error" and "mcp__retriever__search" in run.error
        assert answer is None and model.requests == []  # the model was never called
