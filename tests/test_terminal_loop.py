"""Tests for phantom_envs.terminal.loop: tool-call parsing and the bare tool loop's episode logic."""

from phantom_envs.terminal.loop import (
    ShellEpisodeState,
    ShellProtocol,
    ToolCall,
    parse_tool_calls,
    tool_error,
    truncate,
)


def bash_call(command: str) -> str:
    """An agent turn's tool call, as Qwen3.5 writes it."""
    return f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>\n</tool_call>"


class TestParseToolCalls:
    def test_one_call(self):
        assert parse_tool_calls("Let me look.\n" + bash_call("ls /corpus")) == [
            ToolCall(name="bash", arguments={"command": "ls /corpus"})
        ]

    def test_calls_in_order(self):
        calls = parse_tool_calls(bash_call("ls") + "\n" + bash_call("cat a.md"))
        assert [c.arguments["command"] for c in calls] == ["ls", "cat a.md"]

    def test_multiline_command(self):
        (call,) = parse_tool_calls(bash_call("grep -l x *.md |\n  head -3"))
        assert call.arguments["command"] == "grep -l x *.md |\n  head -3"

    def test_none_without_calls(self):
        assert parse_tool_calls("<answer>Bob</answer>") == []

    def test_unclosed_call_is_ignored(self):
        assert parse_tool_calls("<tool_call>\n<function=bash>\n<parameter=command>\nls") == []


class TestToolError:
    def test_valid_call(self):
        assert tool_error(ToolCall(name="bash", arguments={"command": "ls"})) is None

    def test_unknown_tool(self):
        assert "unknown tool" in tool_error(ToolCall(name="python", arguments={"code": "1"}))

    def test_missing_command(self):
        assert "command" in tool_error(ToolCall(name="bash", arguments={}))


def test_truncate():
    assert truncate("abc", 5) == "abc"
    assert truncate("abcdefgh", 3).startswith("abc\n[output truncated: showed 3 of 8 characters]")


class TestShellProtocol:
    def protocol(self, max_turns: int = 5) -> ShellProtocol:
        return ShellProtocol(max_turns=max_turns)

    def test_tool_call_continues(self):
        state = ShellEpisodeState()
        result = self.protocol().act(state, bash_call("ls"))
        assert not result.done and result.calls[0].arguments == {"command": "ls"}
        assert state.commands == 1

    def test_turn_without_calls_ends_attempt(self):
        state = ShellEpisodeState()
        protocol = self.protocol()
        protocol.act(state, bash_call("echo 'Alice, Bob' > /app/output.md"))
        assert protocol.act(state, "Done.").done and state.turns == 2

    def test_max_turns_ends_attempt(self):
        protocol, state = self.protocol(max_turns=2), ShellEpisodeState()
        assert not protocol.act(state, bash_call("ls")).done
        assert protocol.act(state, bash_call("ls")).done

    def test_observe_truncates(self):
        protocol = ShellProtocol(max_turns=5, max_output_chars=4)
        assert protocol.observe(ShellEpisodeState(), ["ok", "123456"]) == [
            "ok",
            "1234\n[output truncated: showed 4 of 6 characters]",
        ]
