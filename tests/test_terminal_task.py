"""Tests for phantom_envs.terminal.task: the instruction and the verifier."""

import pytest

from phantom_envs.terminal.task import ANSWER_PATH, CORPUS_DIR, build_messages, has_answer, verify


def test_instruction_names_the_paths():
    system, user = build_messages("Who is the mother of Ann Lee?")
    assert system["role"] == "system" and user["role"] == "user"
    assert ANSWER_PATH in user["content"] and CORPUS_DIR in user["content"]
    assert user["content"].endswith("Question: Who is the mother of Ann Lee?")


def test_question_with_braces():
    assert build_messages("What is {x}?")[1]["content"].endswith("Question: What is {x}?")


class TestVerify:
    def test_scores_every_metric(self):
        assert verify("Alice, Bob\n", ["Alice", "Bob"], "pw") == {"em": 1.0, "subem": 1.0, "f1": 1.0}

    def test_partial_answer(self):
        assert verify("Alice", ["Alice", "Bob"], "pw")["f1"] == pytest.approx(2 / 3)

    @pytest.mark.parametrize("answer", [None, "", "  \n"])
    def test_no_answer_scores_zero(self, answer):
        assert not has_answer(answer)
        assert verify(answer, ["Alice"], "pw") == {"em": 0.0, "subem": 0.0, "f1": 0.0}
