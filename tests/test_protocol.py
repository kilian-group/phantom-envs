"""Tests for phantom_envs.protocol: turn parsing, episode termination and the terminal reward."""

from phantom_envs.prompts import USER_PROMPT_PREFIX
from phantom_envs.protocol import (
    EpisodeState,
    SearchProtocol,
    count_searches,
    extract_answer,
    format_observation,
    is_final,
    parse_search,
)


class TestParseSearch:
    def test_query_unstripped(self):
        assert parse_search("Let me look. <search> who is Eloy </search>") == " who is Eloy "

    def test_first_query_only(self):
        assert parse_search("<search>a</search> <search>b</search>") == "a"

    def test_multiline(self):
        assert parse_search("<search>line one\nline two</search>") == "line one\nline two"

    def test_none_without_tags(self):
        assert parse_search("I will answer now.") is None

    def test_none_when_closing_tag_comes_first(self):
        assert parse_search("</search> oops <search>") is None


class TestExtractAnswer:
    def test_last_answer_stripped(self):
        assert extract_answer("<answer>Bob</answer> then <answer> Alice </answer>") == "Alice"

    def test_none_without_answer(self):
        assert extract_answer("<search>q</search>") is None


class TestIsFinal:
    def test_answer_ends(self):
        assert is_final("<answer>Alice</answer>", turns=1, max_turns=10)

    def test_search_continues(self):
        assert not is_final("<search>q</search>", turns=1, max_turns=10)

    def test_no_tags_ends(self):
        assert is_final("I am not sure.", turns=1, max_turns=10)

    def test_max_turns_ends(self):
        assert is_final("<search>q</search>", turns=10, max_turns=10)


def test_format_observation():
    assert format_observation("q", "Doc 1: x") == (
        "\n<information>For query: q\nThe search results are: Doc 1: x</information>\n"
    )


def test_count_searches():
    assert count_searches("<search>a</search> x <search>b</search>") == 2


class TestSearchProtocol:
    def make(self, max_turns: int = 10) -> SearchProtocol:
        return SearchProtocol(gts=["Alice", "Bob"], data_source="pw", max_turns=max_turns)

    def test_search_then_answer(self):
        protocol, state = self.make(), EpisodeState()
        result = protocol.act(state, "Let me check. <search> parents of Carol </search>")
        assert not result.done
        assert result.query == " parents of Carol "
        observation = protocol.observe(state, result.query, "Doc 1: Alice and Bob")
        assert observation in state.transcript

        result = protocol.act(state, "<answer>Alice, Bob</answer>")
        assert result.done
        assert result.reward == 1.0
        assert state.turns == 2

    def test_partial_answer_f1(self):
        protocol, state = self.make(), EpisodeState()
        assert protocol.act(state, "<answer>Alice</answer>").reward == 2 / 3

    def test_scores_every_metric(self):
        protocol, state = self.make(), EpisodeState()
        assert protocol.scores(state) == {"em": 0.0, "subem": 0.0, "f1": 0.0}
        protocol.act(state, "<answer>Alice</answer>")
        assert protocol.scores(state) == {"em": 1.0, "subem": 1.0, "f1": 2 / 3}

    def test_no_answer_scores_zero(self):
        protocol, state = self.make(), EpisodeState()
        result = protocol.act(state, "I don't know.")
        assert result.done
        assert result.reward == 0.0

    def test_max_turns_without_answer(self):
        protocol, state = self.make(max_turns=1), EpisodeState()
        result = protocol.act(state, "<search>q</search>")
        assert result.done
        assert result.reward == 0.0

    def test_prompt_example_not_credited(self):
        # The instructions contain <answer>Alice, Bob</answer>; the transcript never includes them.
        assert "<answer>Alice, Bob</answer>" in USER_PROMPT_PREFIX
        protocol, state = self.make(), EpisodeState()
        assert protocol.act(state, "No idea.").reward == 0.0
