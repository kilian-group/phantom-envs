"""Answer normalization and scoring (EM, substring EM, F1) for every data source.

Each data source maps to a :class:`Scorer`, which normalizes answers, splits a raw
``<answer>`` string into predictions, and scores them against the gold answers:

- PhantomEnvs (``pw``, ``pwcons``, ``pwcomp``): lowercase + strip, comma-separated answers,
  set-based EM / F1.
- Search-R1 (``searchR1_*``): SQuAD normalization, a single answer, set-based EM / F1.
- Multi-hop benchmarks and BrowseComp-Plus: SQuAD normalization, a single answer, token-level
  F1 as in the HotpotQA / MuSiQue reference scorers (max over golds).

Usage::

    from phantom_envs.scoring import score

    score("Alice, Bob", ["Alice", "Bob"], data_source="pw", metric="f1")  # 1.0
"""

import re
import string
from collections import Counter

import numpy as np

METRICS: list[str] = ["em", "subem", "f1"]


def squad_normalize(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation, drop articles, fix whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


class Scorer:
    """Set-based EM / substring EM / F1 over normalized answers. Subclasses set the normalization."""

    @classmethod
    def normalize(cls, s: str) -> str:
        """Normalize a single answer string."""
        raise NotImplementedError

    @classmethod
    def parse_answer(cls, raw: str) -> list[str]:
        """Split a raw ``<answer>`` string into individual predictions."""
        return [raw]

    @classmethod
    def em(cls, preds: list[str], gts: list[str]) -> float:
        """1.0 if any normalized prediction equals any normalized gold answer."""
        normalized_preds = {cls.normalize(p) for p in preds}
        normalized_gts = {cls.normalize(g) for g in gts}
        return 1.0 if normalized_preds & normalized_gts else 0.0

    @classmethod
    def subem(cls, preds: list[str], gts: list[str]) -> float:
        """1.0 if any normalized gold answer is a substring of any normalized prediction."""
        normalized_preds = [cls.normalize(p) for p in preds]
        normalized_gts = [cls.normalize(g) for g in gts]
        return 1.0 if any(gt in pred for gt in normalized_gts for pred in normalized_preds) else 0.0

    @classmethod
    def f1(cls, preds: list[str], gts: list[str]) -> float:
        """Set-based F1 between the normalized predictions and gold answers."""
        normalized_preds = {cls.normalize(p) for p in preds}
        normalized_gts = {cls.normalize(g) for g in gts}
        if not normalized_preds or not normalized_gts:
            return 0.0
        tp = len(normalized_preds & normalized_gts)
        if tp == 0:
            return 0.0
        precision = tp / len(normalized_preds)
        recall = tp / len(normalized_gts)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def score(cls, preds: list[str], gts: list[str], metric: str) -> float:
        """Score predictions against gold answers with ``metric`` (em, subem, f1)."""
        match metric:
            case "em":
                return cls.em(preds, gts)
            case "subem":
                return cls.subem(preds, gts)
            case "f1":
                return cls.f1(preds, gts)
            case _:
                raise ValueError(f"Unknown metric: {metric!r}")


class PhantomScorer(Scorer):
    """PhantomEnvs: comma-separated answer lists, compared after lowercase + strip."""

    ANSWER_SEP: str = ","

    @classmethod
    def normalize(cls, s: str) -> str:
        return s.strip().lower()

    @classmethod
    def parse_answer(cls, raw: str) -> list[str]:
        return [item for item in (item.strip() for item in raw.split(cls.ANSWER_SEP)) if item]


class SearchR1Scorer(Scorer):
    """Search-R1 (NQ + HotpotQA): SkyRL-Gym's ``normalize_answer``, a single answer."""

    @classmethod
    def normalize(cls, s: str) -> str:
        return squad_normalize(s)


class MultiHopScorer(Scorer):
    """Single-answer multi-hop QA with token-level F1, max over (prediction, gold) pairs.

    The token F1 mirrors ``msq_metrics/answer.compute_f1``: when either token list is empty,
    F1 is 1 only if both are empty. EM and substring EM are the set-based defaults.
    """

    @classmethod
    def normalize(cls, s: str) -> str:
        return squad_normalize(s)

    @classmethod
    def token_f1(cls, pred: str, gold: str) -> float:
        """Token-level F1 between one prediction and one gold answer, after normalization."""
        pred_tokens = cls.normalize(pred).split()
        gold_tokens = cls.normalize(gold).split()
        if not gold_tokens or not pred_tokens:
            return float(gold_tokens == pred_tokens)
        return token_overlap_f1(pred_tokens, gold_tokens)

    @classmethod
    def f1(cls, preds: list[str], gts: list[str]) -> float:
        if not preds or not gts:
            return 0.0
        return max(cls.token_f1(p, g) for p in preds for g in gts)


class HotpotScorer(MultiHopScorer):
    """HotpotQA-style token F1: a port of ``hotpot_evaluate_v1.f1_score``.

    Unlike the MuSiQue-style F1, a mismatched yes / no / noanswer answer on either side scores 0,
    so short answers get no partial credit from token overlap.
    """

    @classmethod
    def token_f1(cls, pred: str, gold: str) -> float:
        np_pred = cls.normalize(pred)
        np_gold = cls.normalize(gold)
        if np_pred in {"yes", "no", "noanswer"} and np_pred != np_gold:
            return 0.0
        if np_gold in {"yes", "no", "noanswer"} and np_pred != np_gold:
            return 0.0
        return token_overlap_f1(np_pred.split(), np_gold.split())


def token_overlap_f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    """F1 of the multiset overlap between two token lists (0 when nothing overlaps)."""
    num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


SCORERS: dict[str, type[Scorer]] = {
    "pw": PhantomScorer,
    "pwcons": PhantomScorer,
    "pb": PhantomScorer,  # pwcons parquets and eval dumps made before the rename
    "pwcomp": PhantomScorer,
    "searchR1": SearchR1Scorer,
    "browsecomp_plus": MultiHopScorer,
    "hp500": HotpotScorer,
    "2wiki500": HotpotScorer,
    "msq500": MultiHopScorer,
    "cofca500": HotpotScorer,
    "synthrm500": MultiHopScorer,
    "synthsm500": MultiHopScorer,
    "frames": MultiHopScorer,
}


def get_scorer(data_source: str) -> type[Scorer]:
    """Return the scorer for ``data_source``; prefixes match too (``searchR1_nq`` → Search-R1).

    Raises:
        ValueError: If no registered data source matches.
    """
    if data_source in SCORERS:
        return SCORERS[data_source]
    for prefix, scorer in SCORERS.items():
        if data_source.startswith(prefix):
            return scorer
    raise ValueError(f"Unknown data_source: {data_source!r}. Known prefixes: {list(SCORERS)}")


def score(answer: str, gts: list[str], data_source: str, metric: str) -> float:
    """Score a raw ``<answer>`` string against the gold answers of one question.

    Args:
        answer: Content of the final ``<answer>`` tag.
        gts: Gold answers.
        data_source: Data source of the question (selects the scorer).
        metric: One of :data:`METRICS`.

    Returns:
        The score in [0, 1].
    """
    scorer = get_scorer(data_source)
    return scorer.score(scorer.parse_answer(answer), gts, metric)


def mean_stderr(values: list[float], binary: bool) -> tuple[float, float]:
    """Return (mean, stderr): binomial stderr for binary metrics, std / sqrt(n) otherwise."""
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if binary:
        stderr = float(np.sqrt(mean * (1 - mean) / n))
    else:
        stderr = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, stderr
