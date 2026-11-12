"""Tests for phantom_envs.scoring: normalization, answer parsing and EM / subEM / F1."""

import pytest

from phantom_envs.scoring import (
    HotpotScorer,
    MultiHopScorer,
    PhantomScorer,
    SearchR1Scorer,
    get_scorer,
    score,
)

# ---------------------------------------------------------------------------
# SearchR1Scorer
# ---------------------------------------------------------------------------


class TestSearchR1Normalize:
    def test_lowercase(self):
        assert SearchR1Scorer.normalize("Beijing") == "beijing"

    def test_remove_articles(self):
        assert SearchR1Scorer.normalize("the quick brown fox") == "quick brown fox"

    def test_remove_punctuation(self):
        assert SearchR1Scorer.normalize("hello, world!") == "hello world"

    def test_whitespace_fix(self):
        assert SearchR1Scorer.normalize("  too   many   spaces  ") == "too many spaces"

    def test_combined(self):
        assert SearchR1Scorer.normalize("The Quick  Brown-Fox!") == "quick brownfox"


class TestSearchR1ParseAnswer:
    def test_single_string(self):
        assert SearchR1Scorer.parse_answer("Beijing") == ["Beijing"]

    def test_no_splitting_on_comma(self):
        assert SearchR1Scorer.parse_answer("Alice, Bob") == ["Alice, Bob"]


class TestSearchR1ScoreEM:
    def test_exact_match(self):
        assert SearchR1Scorer.em(["Beijing"], ["Beijing"]) == 1.0

    def test_case_insensitive(self):
        assert SearchR1Scorer.em(["beijing"], ["Beijing"]) == 1.0

    def test_no_match(self):
        assert SearchR1Scorer.em(["Shanghai"], ["Beijing"]) == 0.0

    def test_articles_ignored(self):
        assert SearchR1Scorer.em(["the answer"], ["answer"]) == 1.0

    def test_punctuation_ignored(self):
        assert SearchR1Scorer.em(["new york."], ["New York"]) == 1.0

    def test_multiple_gts(self):
        assert SearchR1Scorer.em(["NYC"], ["New York City", "NYC"]) == 1.0

    def test_multiple_gts_no_match(self):
        assert SearchR1Scorer.em(["LA"], ["New York City", "NYC"]) == 0.0


class TestSearchR1ScoreSubEM:
    def test_substring_match(self):
        assert SearchR1Scorer.subem(["john smith is great"], ["john smith"]) == 1.0

    def test_exact_also_matches(self):
        assert SearchR1Scorer.subem(["beijing"], ["Beijing"]) == 1.0

    def test_no_substring_match(self):
        assert SearchR1Scorer.subem(["john"], ["john smith"]) == 0.0


class TestSearchR1ScoreF1:
    def test_exact_match(self):
        assert SearchR1Scorer.f1(["Beijing"], ["Beijing"]) == 1.0

    def test_case_insensitive(self):
        assert SearchR1Scorer.f1(["beijing"], ["Beijing"]) == 1.0

    def test_no_match(self):
        assert SearchR1Scorer.f1(["Shanghai"], ["Beijing"]) == 0.0

    def test_articles_ignored(self):
        assert SearchR1Scorer.f1(["the answer"], ["answer"]) == 1.0

    def test_punctuation_ignored(self):
        assert SearchR1Scorer.f1(["new york."], ["New York"]) == 1.0

    def test_multiple_gts_one_match(self):
        # 1 pred, 2 gts → P=1/1, R=1/2, F1 = 2/3
        assert SearchR1Scorer.f1(["NYC"], ["New York City", "NYC"]) == pytest.approx(2 / 3)

    def test_multiple_gts_no_match(self):
        assert SearchR1Scorer.f1(["LA"], ["New York City", "NYC"]) == 0.0

    def test_empty_preds(self):
        assert SearchR1Scorer.f1([], ["Beijing"]) == 0.0

    def test_empty_gts(self):
        assert SearchR1Scorer.f1(["Beijing"], []) == 0.0


# ---------------------------------------------------------------------------
# PhantomScorer
# ---------------------------------------------------------------------------


class TestPWNormalize:
    def test_lowercase(self):
        assert PhantomScorer.normalize("Alice") == "alice"

    def test_strip(self):
        assert PhantomScorer.normalize("  Alice  ") == "alice"

    def test_preserves_punctuation(self):
        assert PhantomScorer.normalize("O'Brien") == "o'brien"

    def test_preserves_articles(self):
        assert PhantomScorer.normalize("The Great Wall") == "the great wall"


class TestPWParseAnswer:
    def test_single(self):
        assert PhantomScorer.parse_answer("Alice") == ["Alice"]

    def test_comma_separated(self):
        assert PhantomScorer.parse_answer("Alice, Bob") == ["Alice", "Bob"]

    def test_strips_whitespace(self):
        assert PhantomScorer.parse_answer("  Alice , Bob  ") == ["Alice", "Bob"]

    def test_drops_empty(self):
        assert PhantomScorer.parse_answer("Alice, , Bob,") == ["Alice", "Bob"]

    def test_many_items(self):
        assert PhantomScorer.parse_answer("A, B, C, D") == ["A", "B", "C", "D"]


class TestPWScoreEM:
    def test_exact_match(self):
        assert PhantomScorer.em(["Alice"], ["Alice"]) == 1.0

    def test_case_insensitive(self):
        assert PhantomScorer.em(["alice"], ["Alice"]) == 1.0

    def test_no_match(self):
        assert PhantomScorer.em(["Bob"], ["Alice"]) == 0.0

    def test_any_pred_matches_any_gt(self):
        assert PhantomScorer.em(["Bob", "Alice"], ["Alice", "Charlie"]) == 1.0

    def test_none_match(self):
        assert PhantomScorer.em(["Bob", "Dave"], ["Alice", "Charlie"]) == 0.0

    def test_whitespace_in_pred(self):
        assert PhantomScorer.em(["  Alice  "], ["Alice"]) == 1.0


class TestPWScoreSubEM:
    def test_substring_match(self):
        assert PhantomScorer.subem(["Alice Smith"], ["Alice"]) == 1.0

    def test_exact_also_matches(self):
        assert PhantomScorer.subem(["Alice"], ["Alice"]) == 1.0

    def test_no_match(self):
        assert PhantomScorer.subem(["Bob"], ["Alice"]) == 0.0

    def test_gt_not_substring_of_pred(self):
        # subem checks if GT is in pred, not the other way around
        assert PhantomScorer.subem(["Ali"], ["Alice"]) == 0.0


class TestPWScoreF1:
    def test_exact_match_single(self):
        assert PhantomScorer.f1(["Alice"], ["Alice"]) == 1.0

    def test_exact_match_set(self):
        assert PhantomScorer.f1(["Alice", "Bob"], ["Alice", "Bob"]) == 1.0

    def test_case_insensitive(self):
        assert PhantomScorer.f1(["alice"], ["Alice"]) == 1.0

    def test_no_overlap(self):
        assert PhantomScorer.f1(["Bob"], ["Alice"]) == 0.0

    def test_pred_superset(self):
        # 2 preds, 1 gt, 1 tp → P=1/2, R=1/1, F1 = 2/3
        assert PhantomScorer.f1(["Alice", "Bob"], ["Alice"]) == pytest.approx(2 / 3)

    def test_pred_subset(self):
        # 1 pred, 2 gts, 1 tp → P=1/1, R=1/2, F1 = 2/3
        assert PhantomScorer.f1(["Alice"], ["Alice", "Bob"]) == pytest.approx(2 / 3)

    def test_partial_overlap(self):
        # 2 preds, 2 gts, 1 tp → P=1/2, R=1/2, F1 = 1/2
        assert PhantomScorer.f1(["Alice", "Bob"], ["Alice", "Charlie"]) == pytest.approx(0.5)

    def test_duplicate_preds_dedupe(self):
        # Set-based: duplicates collapse → still perfect F1
        assert PhantomScorer.f1(["Alice", "Alice"], ["Alice"]) == 1.0

    def test_whitespace_normalized(self):
        assert PhantomScorer.f1(["  Alice  "], ["Alice"]) == 1.0

    def test_empty_preds(self):
        assert PhantomScorer.f1([], ["Alice"]) == 0.0

    def test_empty_gts(self):
        assert PhantomScorer.f1(["Alice"], []) == 0.0

    def test_both_empty(self):
        assert PhantomScorer.f1([], []) == 0.0


# ---------------------------------------------------------------------------
# get_scorer
# ---------------------------------------------------------------------------


class TestGetScorer:
    def test_pw(self):
        assert get_scorer("pw") is PhantomScorer

    def test_searchr1_exact(self):
        assert get_scorer("searchR1") is SearchR1Scorer

    def test_searchr1_prefix(self):
        assert get_scorer("searchR1_nq") is SearchR1Scorer
        assert get_scorer("searchR1_hotpotqa") is SearchR1Scorer

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown data_source"):
            get_scorer("unknown_source")


# ---------------------------------------------------------------------------
# Multi-hop scorers
# ---------------------------------------------------------------------------


class TestMultiHopF1:
    def test_token_overlap(self):
        assert MultiHopScorer.f1(["Barack Obama"], ["Obama"]) == pytest.approx(2 / 3)

    def test_max_over_golds(self):
        assert MultiHopScorer.f1(["Obama"], ["Barack Obama", "Obama"]) == 1.0

    def test_empty_prediction_matches_empty_gold(self):
        assert MultiHopScorer.token_f1("", "the") == 1.0

    def test_yes_no_partial_credit(self):
        assert MultiHopScorer.f1(["no way"], ["no"]) == pytest.approx(2 / 3)


class TestHotpotF1:
    def test_yes_no_guard(self):
        assert HotpotScorer.f1(["no way"], ["no"]) == 0.0

    def test_yes_no_exact(self):
        assert HotpotScorer.f1(["No."], ["no"]) == 1.0

    def test_token_overlap(self):
        assert HotpotScorer.f1(["Barack Obama"], ["Obama"]) == pytest.approx(2 / 3)


class TestScore:
    def test_pw_comma_split(self):
        assert score("Alice, Bob", ["Alice", "Bob"], data_source="pw", metric="f1") == 1.0

    def test_searchr1_no_split(self):
        assert score("Alice, Bob", ["Alice"], data_source="searchR1_nq", metric="em") == 0.0

    def test_pwcons_and_pb_alias(self):
        assert get_scorer("pwcons") is PhantomScorer
        assert get_scorer("pb") is PhantomScorer

    def test_benchmark_scorers(self):
        assert get_scorer("2wiki500") is HotpotScorer
        assert get_scorer("msq500") is MultiHopScorer

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            score("Alice", ["Alice"], data_source="pw", metric="bleu")
