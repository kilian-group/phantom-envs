"""Tests for phantom_envs.terminal.task: the instruction and the verifier."""

import pytest

from phantom_envs.terminal.task import (
    ANSWER_PATH,
    CORPUS_DIR,
    build_instruction,
    build_messages,
    has_answer,
    has_search,
    verify,
)


def test_instruction_names_the_paths():
    system, user = build_messages("Who is the mother of Ann Lee?")
    assert system["role"] == "system" and user["role"] == "user"
    assert ANSWER_PATH in user["content"] and CORPUS_DIR in user["content"]
    assert user["content"].endswith("Question: Who is the mother of Ann Lee?")


def test_question_with_braces():
    assert build_messages("What is {x}?")[1]["content"].endswith("Question: What is {x}?")


class TestToolSets:
    def test_bash_is_the_default_task(self):
        assert build_instruction("Who?") == build_instruction("Who?", "bash")
        assert "/corpus" in build_instruction("Who?")

    @pytest.mark.parametrize("tools", ["bash+search", "search"])
    def test_search_tasks_name_the_tool_and_no_corpus_dir(self, tools):
        instruction = build_instruction("What is {x}?", tools)
        assert "search tool" in instruction and "/corpus" not in instruction
        assert ANSWER_PATH in instruction and instruction.endswith("Question: What is {x}?")
        assert has_search(tools) and not has_search("bash")

    def test_without_bash_the_answer_is_written_with_the_write_tool(self):
        assert "write tool" in build_instruction("Who?", "search")
        assert "write tool" not in build_instruction("Who?", "bash+search")


class TestVerify:
    def test_scores_every_metric(self):
        assert verify("Alice, Bob\n", ["Alice", "Bob"], "pw") == {"em": 1.0, "subem": 1.0, "f1": 1.0}

    def test_partial_answer(self):
        assert verify("Alice", ["Alice", "Bob"], "pw")["f1"] == pytest.approx(2 / 3)

    @pytest.mark.parametrize("answer", [None, "", "  \n"])
    def test_no_answer_scores_zero(self, answer):
        assert not has_answer(answer)
        assert verify(answer, ["Alice"], "pw") == {"em": 0.0, "subem": 0.0, "f1": 0.0}
