"""The terminal agent's setup: its harness, its tools, its sandbox and its task, in one place.

SkyRL's harness generator, the Tinker env and `scripts/eval_api.py` all run the same agent: pi in a
bwrap sandbox, under the same budgets. An `AgentSetup` holds what they share, and builds each part:

- the tools: bash over the corpus as files (`bash`), or a search tool served by the retrieval MCP
  server on the host, with bash (`bash+search`) or with only pi's write tool (`search`);
- the harness: pi under the episode's budgets, with those tools;
- the sandbox: over the corpus, or, with the search tool, with no corpus and the server's socket;
- the task text the agent gets.

Usage:

    agent = AgentSetup(corpus_path, rootfs, tools="bash+search", mcp_socket="/tmp/retriever.sock")
    agent.wait_for_servers()  # once, before the first episode
    with agent.open_sandbox() as sandbox:
        run = await run_harness(
            agent.harness(), sandbox, agent.instruction(question), endpoint, mcp_servers=agent.mcp_servers
        )
"""

from dataclasses import dataclass, field
from typing import cast

from phantom_envs.retriever.search_tool import SERVER_NAME, TOOL_NAME, wait_for_server
from phantom_envs.terminal.bwrap import BwrapSandbox
from phantom_envs.terminal.harness import McpServer, mcp_mounts
from phantom_envs.terminal.pi import PiHarness
from phantom_envs.terminal.sandbox import open_sandbox
from phantom_envs.terminal.task import BASH, SEARCH, TOOLSETS, build_instruction, has_search


@dataclass(frozen=True)
class AgentSetup:
    """One terminal agent: where its documents are, the tools it gets, and the episode's budgets."""

    corpus_path: str = field(metadata={"description": "corpus.jsonl, served under /corpus to a bash agent."})
    rootfs: str = field(metadata={"description": "Root filesystem of the bwrap sandbox (SANDBOX_ROOTFS)."})
    tools: str = field(default=BASH, metadata={"description": f"The agent's tools: one of {TOOLSETS}."})
    mcp_socket: str = field(
        default="", metadata={"description": "Unix socket of the retrieval MCP server, for the search tool."}
    )
    max_turns: int = field(default=20, metadata={"description": "Most turns the agent takes."})
    command_timeout: float = field(default=60.0, metadata={"description": "Seconds a tool call may run."})
    max_output_chars: int = field(
        default=8000, metadata={"description": "Tool outputs are cut to this many characters."}
    )
    episode_timeout: float = field(
        default=1800.0, metadata={"description": "Seconds a whole episode may take."}
    )
    memory_gib: int = field(
        default=4, metadata={"description": "Address-space limit of each command, in GiB."}
    )
    system_prompt: str | None = field(
        default=None, metadata={"description": "Replaces pi's own system prompt; None keeps it."}
    )

    def __post_init__(self) -> None:
        """Check that the tools are known, and that the search tool has its server.

        Raises:
            ValueError: For an unknown tool set, or the search tool without an MCP socket.
        """
        if self.tools not in TOOLSETS:
            raise ValueError(f"Unknown tools {self.tools!r} (expected one of {TOOLSETS})")
        if has_search(self.tools) and not self.mcp_socket:
            raise ValueError(
                "The search tool needs the retrieval MCP server's Unix socket (mcp_socket): start one with "
                "`python -m phantom_envs.retriever.mcp_server ... socket=<path>`, or launch_skyrl.sh --search"
            )

    @property
    def mcp_servers(self) -> list[McpServer]:
        """The MCP servers whose tools the agent gets: the retrieval server, for the search tool."""
        if not has_search(self.tools):
            return []
        return [McpServer(name=SERVER_NAME, socket_path=self.mcp_socket, tools=(TOOL_NAME,))]

    def harness(self) -> PiHarness:
        """Return pi with the agent's tools, under the episode's budgets.

        A search-only agent gets pi's write tool in place of bash, so that it can still write its answer
        file.
        """
        return PiHarness(
            tools=("write",) if self.tools == SEARCH else ("bash",),
            max_turns=self.max_turns,
            command_timeout=self.command_timeout,
            max_output_chars=self.max_output_chars,
            episode_timeout=self.episode_timeout,
            system_prompt=self.system_prompt,
        )

    def instruction(self, question: str) -> str:
        """Return the task text for a question, worded for the agent's tools."""
        return build_instruction(question, self.tools)

    def open_sandbox(self) -> BwrapSandbox:
        """Open a bwrap sandbox: over the corpus, or, with the search tool, with its server's socket."""
        sandbox = open_sandbox(
            "bwrap",
            None if has_search(self.tools) else self.corpus_path,
            timeout=self.command_timeout,
            memory_gib=self.memory_gib,
            rootfs=self.rootfs,
            mounts=mcp_mounts(self.mcp_servers),
        )
        return cast(BwrapSandbox, sandbox)

    def wait_for_servers(self, timeout: float = 900.0) -> None:
        """Wait until the agent's MCP servers answer, so that no episode starts without their tools.

        Args:
            timeout: Most seconds to wait for each server.

        Raises:
            TimeoutError: If a server isn't ready in time.
        """
        for server in self.mcp_servers:
            wait_for_server(server.socket_path, timeout)
