"""The bare tool loop of a terminal episode, with no I/O, for trainers that sample the agent's turns.

The agent calls a `bash` tool in Qwen3.5's tool-call format (`<tool_call><function=bash>
<parameter=command>`) and gets each command's output back as a tool response. The attempt ends on a
turn without tool calls, or after `max_turns` turns; then the verifier (`task.verify`) scores the
answer file.

Trainer adapters drive `ShellProtocol` and run the commands in a sandbox:

    protocol = ShellProtocol(max_turns=20)
    state = ShellEpisodeState()
    result = protocol.act(state, turn)  # once per agent turn
    if result.done:
        scores = verify(sandbox.read_answer(), gts, data_source)
    else:
        outputs = [tool_error(call) or sandbox.run(call.arguments["command"]) for call in result.calls]
        observations = protocol.observe(state, outputs)

A coding-agent harness such as pi replaces this module: it runs its own loop inside the sandbox.
"""

import re
from dataclasses import dataclass, field

# A Qwen3.5 tool call, and each parameter inside it
TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


@dataclass
class ToolCall:
    """One tool call of an agent turn."""

    name: str = field(metadata={"description": "Function name, e.g. bash."})
    arguments: dict[str, str] = field(metadata={"description": "Parameter values by name, as text."})


def parse_tool_calls(turn: str) -> list[ToolCall]:
    """Parse the tool calls of an agent turn (Qwen3.5's XML format).

    Args:
        turn: The text the agent wrote.

    Returns:
        The well-formed tool calls, in order; an unclosed call is ignored.
    """
    return [
        ToolCall(name=name, arguments=dict(PARAMETER_RE.findall(body)))
        for name, body in TOOL_CALL_RE.findall(turn)
    ]


def tool_error(call: ToolCall) -> str | None:
    """Explain why a tool call can't run.

    Args:
        call: The tool call.

    Returns:
        The error message the agent gets back as the tool response, or None if the call can run.
    """
    if call.name != "bash":
        return f"Error: unknown tool {call.name!r}; the only tool is bash."
    if "command" not in call.arguments:
        return "Error: bash needs a command parameter."
    return None


def truncate(text: str, max_chars: int) -> str:
    """Cut a tool output to `max_chars` characters.

    Args:
        text: The tool output.
        max_chars: The most characters to keep.

    Returns:
        The text if it fits; otherwise its start, with a note on how much was left out.
    """
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[output truncated: showed {max_chars} of {len(text)} characters]"


@dataclass
class ShellEpisodeState:
    """Mutable state of one terminal episode."""

    turns: int = field(default=0, metadata={"description": "Agent turns taken so far."})
    commands: int = field(default=0, metadata={"description": "Tool calls made so far."})


@dataclass
class ShellTurnResult:
    """Outcome of one agent turn."""

    done: bool = field(metadata={"description": "Whether the attempt has ended (time to verify)."})
    calls: list[ToolCall] = field(
        default_factory=list, metadata={"description": "Tool calls to run next; empty when done."}
    )


@dataclass
class ShellProtocol:
    """Episode logic of the bare tool loop: when the attempt ends and which commands to run."""

    max_turns: int = field(metadata={"description": "Maximum number of agent turns per episode."})
    max_output_chars: int = field(
        default=8000, metadata={"description": "Tool outputs are cut to this many characters."}
    )

    def act(self, state: ShellEpisodeState, turn: str) -> ShellTurnResult:
        """Record an agent turn and decide what happens next.

        Args:
            state: The episode's state, updated in place.
            turn: The text the agent wrote.

        Returns:
            Whether the attempt has ended and, if not, the tool calls to run.
        """
        state.turns += 1
        calls = parse_tool_calls(turn)
        if not calls or state.turns >= self.max_turns:
            return ShellTurnResult(done=True)
        state.commands += len(calls)
        return ShellTurnResult(done=False, calls=calls)

    def observe(self, state: ShellEpisodeState, outputs: list[str]) -> list[str]:
        """Turn a turn's command outputs into the tool responses the agent reads.

        Args:
            state: The episode's state.
            outputs: The output of each tool call of the turn, in order.

        Returns:
            The tool responses, each cut to `max_output_chars`.
        """
        return [truncate(output, self.max_output_chars) for output in outputs]
