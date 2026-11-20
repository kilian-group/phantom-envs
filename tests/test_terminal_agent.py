"""Tests for phantom_envs.terminal.agent: one setup of the terminal agent for every trainer and eval.

The sandbox and bridge tests run pi in bwrap, so they need bubblewrap, SANDBOX_ROOTFS and aiohttp.
"""

import asyncio
import json
import os

import pytest

from phantom_envs.terminal.agent import AgentSetup
from phantom_envs.terminal.harness import McpServer
from phantom_envs.terminal.task import ANSWER_PATH, CORPUS_DIR


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A one-document corpus.jsonl."""
    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
    doc = {"id": "0", "title": "Ann Lee", "contents": "# Ann Lee\n\nThe mother of Ann Lee is Bea Lee.\n"}
    path.write_text(json.dumps(doc) + "\n")
    return str(path)


class TestSetup:
    def test_a_bash_agent_has_bash_and_the_corpus(self, corpus_path: str):
        agent = AgentSetup(corpus_path, rootfs="/rootfs")
        assert agent.harness().tools == ("bash",) and agent.mcp_servers == []
        assert CORPUS_DIR in agent.instruction("Who is Ann's mother?")

    def test_a_search_agent_gets_the_retrieval_servers_tool(self, corpus_path: str):
        agent = AgentSetup(corpus_path, rootfs="/rootfs", tools="bash+search", mcp_socket="/tmp/r.sock")
        [server] = agent.mcp_servers
        assert server.tool_names == ["mcp__retriever__search"] and server.socket_path == "/tmp/r.sock"
        assert agent.harness().tools == ("bash",)
        assert "search tool" in agent.instruction("Who?") and CORPUS_DIR not in agent.instruction("Who?")

    def test_without_bash_the_agent_writes_its_answer_with_pis_write_tool(self, corpus_path: str):
        agent = AgentSetup(corpus_path, rootfs="/rootfs", tools="search", mcp_socket="/tmp/r.sock")
        assert agent.harness().tools == ("write",)

    def test_budgets_reach_the_harness(self, corpus_path: str):
        harness = AgentSetup(corpus_path, rootfs="/rootfs", max_turns=7, command_timeout=30).harness()
        assert (harness.max_turns, harness.command_timeout) == (7, 30)

    def test_the_search_tool_needs_its_server(self, corpus_path: str):
        with pytest.raises(ValueError, match="mcp_socket"):
            AgentSetup(corpus_path, rootfs="/rootfs", tools="bash+search")

    def test_unknown_tools(self, corpus_path: str):
        with pytest.raises(ValueError, match="Unknown tools 'grep'"):
            AgentSetup(corpus_path, rootfs="/rootfs", tools="grep")


@pytest.mark.usefixtures("needs_sandbox")
class TestSandbox:
    def test_a_search_agents_sandbox_has_the_server_and_no_corpus(self, corpus_path: str, search_server):
        agent = AgentSetup(
            corpus_path,
            os.environ["SANDBOX_ROOTFS"],
            tools="bash+search",
            mcp_socket=search_server.socket_path,
        )
        with agent.open_sandbox() as sandbox:
            assert "No such file" in sandbox.run(f"ls {CORPUS_DIR}")
            assert "retriever.sock" in sandbox.run("ls /run/mcp")

    def test_a_bash_agents_sandbox_has_the_corpus(self, corpus_path: str):
        with AgentSetup(corpus_path, os.environ["SANDBOX_ROOTFS"]).open_sandbox() as sandbox:
            assert "Ann_Lee.md" in sandbox.run(f"ls {CORPUS_DIR}")


@pytest.mark.usefixtures("needs_sandbox")
def test_a_bridged_episode_searches(corpus_path: str, search_server: McpServer):
    """An episode answered from Python, as scripts/eval_api.py and the Tinker env run it, with search."""
    from phantom_envs.terminal.bridge import BridgedEpisode

    agent = AgentSetup(
        corpus_path, os.environ["SANDBOX_ROOTFS"], tools="bash+search", mcp_socket=search_server.socket_path
    )

    def call(name: str, arguments: dict, turn: int) -> dict:
        function = {"name": name, "arguments": json.dumps(arguments)}
        return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{turn}", "type": "function", "function": function}]}  # fmt: skip

    async def play() -> tuple:
        episode = BridgedEpisode(agent)
        try:
            request = await episode.start("Who is the mother of Ann Lee?")
            tools = [tool["function"]["name"] for tool in request["tools"]]
            found = await episode.step(call("mcp__retriever__search", {"query": "mother of Ann Lee"}, 1))
            await episode.step(call("bash", {"command": f"echo 'Bea Lee' > {ANSWER_PATH}"}, 2))
            assert await episode.step({"role": "assistant", "content": "Done."}) is None
            run, answer = await episode.finish()
        finally:
            await episode.close()
        return tools, found, run, answer, request["messages"]

    tools, found, run, answer, messages = asyncio.run(play())
    assert tools == ["bash", "mcp__retriever__search"]
    assert "The mother of Ann Lee is Bea Lee." in str(found)
    assert "search tool" in str(messages)  # the task text for a search agent
    assert (run.mcp_calls, run.stop_reason, answer) == (1, "stop", "Bea Lee\n")
