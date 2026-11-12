"""The ``<search>`` / ``<information>`` / ``<answer>`` protocol of a search episode, with no I/O.

The agent writes one turn at a time. A turn with ``<search>query</search>`` gets the retrieved
documents back in an ``<information>`` block. The episode ends on a turn with
``<answer>...</answer>``, on a turn with neither tag, or after ``max_turns`` turns. The reward
scores the last answer in the episode transcript. The transcript holds agent turns and
observations only, never the prompt: its instructions contain an example
``<answer>Alice, Bob</answer>`` that must not be credited.

Trainer adapters drive :class:`SearchProtocol` and run the searches themselves::

    protocol = SearchProtocol(gts=["Alice"], data_source="pw", max_turns=10)
    state = EpisodeState()
    result = protocol.act(state, turn)  # once per agent turn
    if not result.done:
        observation = protocol.observe(state, result.query, retriever.search(result.query))
"""

import re
from dataclasses import dataclass, field

from phantom_envs.scoring import METRICS, score

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
STOP_STRINGS: list[str] = ["</search>", "</answer>"]


def parse_search(turn: str) -> str | None:
    """Return the first ``<search>`` query of an agent turn, unstripped, or None."""
    if "<search>" in turn and "</search>" in turn:
        match = SEARCH_RE.search(turn)
        if match:
            return match.group(1)
    return None


def extract_answer(text: str) -> str | None:
    """Return the last ``<answer>...</answer>`` content in ``text``, stripped, or None."""
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def is_final(turn: str, turns: int, max_turns: int) -> bool:
    """Whether the episode ends after this agent turn (``turns`` counts it)."""
    if turns >= max_turns:
        return True
    if "<answer>" in turn and "</answer>" in turn:
        return True
    # No well-formed search call and no answer: the agent has nothing left to do.
    return not ("<search>" in turn and "</search>" in turn)


def format_observation(query: str | None, results: str) -> str:
    """Wrap the search results for ``query`` in the ``<information>`` block fed back to the agent."""
    return f"\n<information>For query: {query}\nThe search results are: {results}</information>\n"


def count_searches(response: str) -> int:
    """Number of search calls in an episode response (robust to length-truncated rollouts)."""
    return response.count("</search>")


@dataclass
class EpisodeState:
    """Mutable state of one episode."""

    turns: int = field(default=0, metadata={"description": "Agent turns taken so far."})
    transcript: list[str] = field(
        default_factory=list,
        metadata={"description": "Agent turns and observations in order; never the prompt."},
    )


@dataclass
class TurnResult:
    """Outcome of one agent turn."""

    done: bool = field(metadata={"description": "Whether the episode has ended."})
    query: str | None = field(
        default=None, metadata={"description": "Search query to run next; None when done."}
    )
    reward: float = field(default=0.0, metadata={"description": "Terminal reward; 0.0 until done."})


@dataclass
class SearchProtocol:
    """Episode logic for one question: when it ends, what to search, and the reward."""

    gts: list[str] = field(metadata={"description": "Gold answers of the question."})
    data_source: str = field(metadata={"description": "Data source of the question (selects the scorer)."})
    max_turns: int = field(metadata={"description": "Maximum number of agent turns per episode."})
    metric: str = field(default="f1", metadata={"description": "Reward metric: em, subem or f1."})

    def act(self, state: EpisodeState, turn: str) -> TurnResult:
        """Record an agent turn and decide what happens next."""
        state.turns += 1
        state.transcript.append(turn)
        if is_final(turn, state.turns, self.max_turns):
            return TurnResult(done=True, reward=self.reward(state))
        return TurnResult(done=False, query=parse_search(turn))

    def observe(self, state: EpisodeState, query: str | None, results: str) -> str:
        """Record and return the observation for a search."""
        observation = format_observation(query, results)
        state.transcript.append(observation)
        return observation

    def reward(self, state: EpisodeState) -> float:
        """Score the last answer in the transcript; 0.0 when there is none."""
        answer = extract_answer("".join(state.transcript))
        if answer is None:
            return 0.0
        return score(answer, self.gts, self.data_source, self.metric)

    def scores(self, state: EpisodeState) -> dict[str, float]:
        """Score the last answer in the transcript with every metric.

        Args:
            state: The episode's state.

        Returns:
            The score per metric (em, subem, f1); all 0.0 when there is no answer.
        """
        answer = extract_answer("".join(state.transcript))
        return {m: 0.0 if answer is None else score(answer, self.gts, self.data_source, m) for m in METRICS}
