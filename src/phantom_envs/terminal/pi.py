"""The pi harness: pi (https://pi.dev), a minimal open-source coding agent, as the agent's loop.

pi runs inside the sandbox with its `bash` tool (or the tools named in `PiHarness.tools`), the tools
of any MCP servers it is given, and its own system prompt. It keeps its history
append-only (compaction is off), so every model call extends the previous one's tokens. The episode's
budgets match the bare tool loop's: a turn cap, a time limit per command, and a cap on the output
the model reads. pi has settings for some; `pi_extension.js` enforces the rest. Each command pi
starts gets the sandbox's per-process limits.

The sandbox image has pi installed (`scripts/sandbox/Dockerfile` pins its version).

Gotchas:

- pi runs a turn's commands before the turn cap ends the episode; the bare loop doesn't run the
  commands of its last turn.
- pi keeps only the tail of a long output, and then `max_output_chars` keeps the head of that.
- pi starts without an MCP server's tools if the server is slow to connect; `pi_extension.js` then
  ends the episode, which reports an error.
"""

import json
import shlex
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phantom_envs.terminal.bwrap import HOME_DIR, SHARED_DIR, BwrapSandbox
from phantom_envs.terminal.harness import (
    Harness,
    HarnessRun,
    McpServer,
    ModelEndpoint,
    env_prefix,
    is_mcp_tool,
    mcp_url,
    relay_command,
)

# Enforces the budgets that pi has no setting for
EXTENSION: Path = Path(__file__).with_name("pi_extension.js")
# pi's config dir inside the sandbox, a subdir of the shared dir
CONFIG_DIR: str = "pi"
# Where pi runs from a copy of its config: pi writes lock files there, and the shared dir is read-only
STATE_DIR: str = f"{HOME_DIR}/.pi"
# The name pi's config gives the model endpoint
PROVIDER: str = "phantom"
# How skycap and vLLM word a prompt that doesn't fit the model's context
CONTEXT_ERRORS: tuple[str, ...] = ("context_length_exceeded", "max_model_len", "maximum context length")


@dataclass
class PiHarness(Harness):
    """pi with its own tools (bash by default) and any MCP servers' tools, under the episode's budgets."""

    tools: tuple[str, ...] = field(
        default=("bash",),
        metadata={"description": "pi's own tools the agent gets: bash, read, write, edit."},
    )
    max_turns: int = field(default=20, metadata={"description": "Most turns the agent takes."})
    command_timeout: float = field(default=60.0, metadata={"description": "Seconds a command may run."})
    max_output_chars: int = field(
        default=8000, metadata={"description": "Tool outputs are cut to this many characters."}
    )
    episode_timeout: float = field(
        default=1800.0, metadata={"description": "Seconds a whole episode may take."}
    )
    system_prompt: str | None = field(
        default=None, metadata={"description": "Replaces pi's own system prompt; None keeps it."}
    )

    def run(
        self,
        sandbox: BwrapSandbox,
        instruction: str,
        endpoint_url: str,
        endpoint: ModelEndpoint,
        mcp_servers: Sequence[McpServer] = (),
    ) -> HarnessRun:
        """Run pi on the task until the agent finishes or a budget ends the episode.

        Args:
            sandbox: The episode's sandbox.
            instruction: The task text, given to pi as its prompt.
            endpoint_url: The model API's base URL as the sandbox reaches it.
            endpoint: The model endpoint's settings.
            mcp_servers: MCP servers whose tools the agent gets.

        Returns:
            The turns and commands pi took, and why it stopped.
        """
        config_dir = sandbox.shared_dir / CONFIG_DIR
        self.write_config(config_dir, endpoint_url, endpoint, sandbox.command_limits, mcp_servers)
        command = relay_command(sandbox, mcp_servers) + self.command(instruction, endpoint.model, mcp_servers)
        # pi runs unlimited; its settings put the sandbox's limits on each command it starts
        output, exit_code, timed_out = sandbox.execute(command, timeout=self.episode_timeout, limited=False)
        run = parse_events(output)
        if timed_out:
            run.stop_reason = "timeout"
        elif run.turns == 0 and run.stop_reason == "stop":
            run.stop_reason = "error"
            run.error = run.error or f"pi exited with code {exit_code} before its first turn: {output[-500:]}"
        return run

    def write_config(
        self,
        config_dir: Path,
        endpoint_url: str,
        endpoint: ModelEndpoint,
        command_limits: str,
        mcp_servers: Sequence[McpServer] = (),
    ) -> None:
        """Write pi's config dir: the model endpoint, its settings, the MCP servers and the extension.

        Args:
            config_dir: The dir, inside the sandbox's shared dir.
            endpoint_url: The model API's base URL as the sandbox reaches it.
            endpoint: The model endpoint's settings.
            command_limits: The bash line that limits each command pi starts.
            mcp_servers: MCP servers whose tools the agent gets.
        """
        config_dir.mkdir(exist_ok=True)
        model = {
            "id": endpoint.model,
            "contextWindow": endpoint.context_window,
            "maxTokens": endpoint.max_tokens,
        }
        provider = {
            "baseUrl": endpoint_url,
            "api": "openai-completions",
            "apiKey": endpoint.api_key,
            "models": [model],
        }
        settings = {
            "compaction": {"enabled": False},
            "shellCommandPrefix": command_limits,
        }
        (config_dir / "models.json").write_text(json.dumps({"providers": {PROVIDER: provider}}))
        (config_dir / "settings.json").write_text(json.dumps(settings))
        # direct: pi declares the tools to the model like its own, not behind a tool-search step
        servers = {
            server.name: {
                "url": mcp_url(mcp_servers, server),
                "exposure": "direct",
                "timeout": self.command_timeout,
            }
            for server in mcp_servers
        }
        (config_dir / "mcp.json").write_text(json.dumps({"mcpServers": servers}))
        shutil.copy(EXTENSION, config_dir / EXTENSION.name)

    def command(self, instruction: str, model: str, mcp_servers: Sequence[McpServer] = ()) -> str:
        """Return the bash command that runs pi once on a task and prints its events as JSONL.

        Args:
            instruction: The task text, given to pi as its prompt.
            model: The model name pi requests.
            mcp_servers: MCP servers whose tools the agent gets.

        Returns:
            The command: copy pi's config where pi can write, then run pi with its environment.
        """
        config_dir = f"{SHARED_DIR}/{CONFIG_DIR}"
        # pi declares an MCP tool only if it is named here
        mcp_tools = [name for server in mcp_servers for name in server.tool_names]
        env = {
            "PI_CODING_AGENT_DIR": STATE_DIR,
            # No update checks, telemetry or model catalog refreshes: the sandbox has no network
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
            "PHANTOM_MAX_TURNS": str(self.max_turns),
            "PHANTOM_COMMAND_TIMEOUT": f"{self.command_timeout:g}",
            "PHANTOM_MAX_OUTPUT_CHARS": str(self.max_output_chars),
            # An MCP server that is slow to connect must fail the episode, not leave it without tools
            "PHANTOM_REQUIRED_TOOLS": ",".join(mcp_tools),
        }
        args = [
            "pi", "--print", "--mode", "json", "--no-session",
            "--provider", PROVIDER, "--model", model, "--tools", ",".join([*self.tools, *mcp_tools]),
            # Only what this module configures: nothing from the image or the working dir
            "--no-context-files", "--no-skills", "--no-prompt-templates", "--no-extensions",
            "--extension", f"{config_dir}/{EXTENSION.name}",
        ]  # fmt: skip
        if mcp_servers:
            # --no-extensions turned pi's built-in MCP support off as well
            args += ["--extension", "builtin:mcp"]
        if self.system_prompt is not None:
            args += ["--system-prompt", self.system_prompt]
        return f"cp -r {config_dir} {STATE_DIR} && {env_prefix(env)} {shlex.join([*args, instruction])}"


def parse_events(output: str) -> HarnessRun:
    """Summarize pi's JSONL event stream.

    Args:
        output: pi's stdout in `--mode json`; lines that aren't JSON events are skipped.

    Returns:
        The turns and commands so far, and the stop reason of the last assistant message:
        `context_length` when the prompt outgrew the model's context, `length` when a turn hit its
        token limit, `error` on another failure, else `stop`.
    """
    run = HarnessRun()
    last_message: dict[str, Any] = {}
    missing_tools: list[str] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "phantom_missing_tools":
            # pi_extension.js ended the episode: an MCP server's tools were not there at the first turn
            missing_tools = event.get("tools") or []
        if not isinstance(event, dict) or event.get("type") != "turn_end":
            continue
        last_message = event.get("message") or {}
        # A turn that the turn cap aborted, or whose model call failed, is not a turn the agent took
        if last_message.get("stopReason") not in ("aborted", "error"):
            run.turns += 1
            results = event.get("toolResults") or []
            run.commands += len(results)
            run.mcp_calls += sum(is_mcp_tool(str(result.get("toolName", ""))) for result in results)

    if last_message.get("stopReason") == "error":
        run.error = last_message.get("errorMessage") or "unknown error"
        overflowed = any(marker in run.error for marker in CONTEXT_ERRORS)
        run.stop_reason = "context_length" if overflowed else "error"
    elif last_message.get("stopReason") == "length":
        run.stop_reason = "length"
    if missing_tools:
        run.stop_reason = "error"
        run.error = f"pi started without the tools {missing_tools}: their MCP server did not connect in time"
    return run
