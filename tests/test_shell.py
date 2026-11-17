"""Tests for phantom_envs.shell: tool-call parsing, the episode logic and the ShellSim sandbox."""

import json

import pytest

from phantom_envs.prompts import SHELL_INSTRUCTION
from phantom_envs.shell import (
    ANSWER_PATH,
    CORPUS_DIR,
    MAX_DOC_BYTES,
    ShellEpisodeState,
    ShellProtocol,
    ShellSandbox,
    ToolCall,
    build_corpus_image,
    doc_filename,
    doc_text,
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
        return ShellProtocol(gts=["Alice", "Bob"], data_source="pw", max_turns=max_turns)

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

    def test_reward_verifies_answer_file(self):
        protocol = self.protocol()
        assert protocol.reward("Alice, Bob\n") == 1.0
        assert protocol.reward("Alice") == pytest.approx(2 / 3)
        assert protocol.reward(None) == 0.0
        assert protocol.reward("  \n") == 0.0

    def test_prompt_names_answer_file(self):
        assert ANSWER_PATH in SHELL_INSTRUCTION

    def test_observe_truncates(self):
        protocol = ShellProtocol(gts=["x"], data_source="pw", max_turns=5, max_output_chars=4)
        assert protocol.observe(ShellEpisodeState(), ["ok", "123456"]) == [
            "ok",
            "1234\n[output truncated: showed 4 of 6 characters]",
        ]


def test_doc_filename_unique_and_safe():
    taken: set[str] = set()
    assert doc_filename("Alison Smock", taken) == "Alison_Smock.md"
    assert doc_filename("Alison Smock", taken) == "Alison_Smock_2.md"
    assert doc_filename("a/b: c?", taken) == "a_b_c.md"


def test_doc_text():
    assert doc_text("Ann", "# Ann\n\nText.") == "# Ann\n\nText."
    assert doc_text("Paris", "Paris is a city.") == "# Paris\n\nParis is a city."
    assert len(doc_text("Big", "x" * (MAX_DOC_BYTES + 10)).encode()) == MAX_DOC_BYTES


class TestShellSandbox:
    @pytest.fixture(scope="class")
    def image(self, tmp_path_factory):
        pytest.importorskip("shellsim")
        path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
        docs = [
            {"id": "0", "title": "Ann Lee", "contents": "# Ann Lee\n\nThe mother of Ann Lee is Bea Lee.\n"},
            {"id": "1", "title": "Bea Lee", "contents": "# Bea Lee\n\nThe hobby of Bea Lee is chess.\n"},
        ]
        path.write_text("".join(json.dumps(d) + "\n" for d in docs))
        return build_corpus_image(path)

    def test_search_and_read(self, image):
        sandbox = ShellSandbox(image)
        assert sandbox.run(f"grep -l chess {CORPUS_DIR}/*.md").strip() == f"{CORPUS_DIR}/Bea_Lee.md"
        assert "Bea Lee" in sandbox.run(f"cat {CORPUS_DIR}/Ann_Lee.md")

    def test_answer_file(self, image):
        sandbox = ShellSandbox(image)
        assert sandbox.read_answer() is None
        sandbox.run("echo 'Bea Lee' > output.md")  # the working dir is /app
        assert sandbox.read_answer() == "Bea Lee\n"

    def test_sandboxes_are_isolated(self, image):
        first, second = ShellSandbox(image), ShellSandbox(image)
        first.run("echo note > /tmp/x")
        assert "No such file" in second.run("cat /tmp/x")

    def test_failure_and_timeout_notes(self, image):
        sandbox = ShellSandbox(image, timeout=0.5)
        assert "[exit code 1]" in sandbox.run(f"grep -l nothing {CORPUS_DIR}/*.md")
        assert "timed out" in sandbox.run("while true; do :; done")
        assert sandbox.run("echo alive").strip() == "alive"
