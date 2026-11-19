"""The `phantom_search` SkyRL-Gym environment: a search agent with a retriever.

`SearchProtocol` decides when the episode ends, what to search and the reward; `SearchClient` runs
the searches over HTTP.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.search.env import SearchEnvConfig

from phantom_envs.retriever.client import SearchClient
from phantom_envs.scoring import METRICS
from phantom_envs.search.protocol import EpisodeState, SearchProtocol


@dataclass
class PhantomSearchEnvConfig(SearchEnvConfig):
    """SearchEnvConfig extended with a configurable HTTP connection pool size and reward."""

    pool_size: int = field(
        default=64, metadata={"help": "HTTP connection pool size for concurrent search calls"}
    )
    reward_func: Literal[tuple(METRICS)] = field(
        default="f1",
        metadata={"help": "Reward function used to score answers (em, subem, f1)"},
    )


class PhantomSearchEnv(BaseTextEnv):
    """Drop-in replacement for SkyRL-Gym's SearchEnv with a pooled retrieval client."""

    def __init__(self, env_config: PhantomSearchEnvConfig, extras: dict[str, Any] | None = None) -> None:
        """Set up one episode.

        Args:
            env_config: The `environment.skyrl_gym.phantom_search` config.
            extras: The dataset row's fields: `reward_spec.ground_truth.target` (the gold answers),
                `data_source` and `max_turns`.
        """
        super().__init__()

        assert (
            "reward_spec" in extras
            and "ground_truth" in extras["reward_spec"]
            and "target" in extras["reward_spec"]["ground_truth"]
            and isinstance(extras["reward_spec"]["ground_truth"]["target"], list)
        ), "extras must contain reward_spec.ground_truth.target as a list"

        self.extras: dict[str, Any] = extras  # NOTE: mutate this to store metrics
        self.protocol = SearchProtocol(
            gts=extras["reward_spec"]["ground_truth"]["target"],
            data_source=extras["data_source"],  # e.g. "pw" or "searchR1_<original_source>"
            max_turns=extras["max_turns"],
            metric=env_config.reward_func,
        )
        self.state = EpisodeState()
        self.client = SearchClient(
            search_url=env_config.search_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
            pool_size=env_config.pool_size,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Return the episode's metrics, which SkyRL aggregates and logs to W&B."""
        return {"num_turns": self.state.turns}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Take one agent turn: run its search, or end the episode with the reward.

        Args:
            action: The text the agent wrote.

        Returns:
            The search results as a user message, or the final reward when the episode is done.
        """
        result = self.protocol.act(self.state, action)
        self.turns = self.state.turns

        if result.done:
            # NOTE: mutate the extras to store the number of turns, gets saved by SkyRL Gym Generator
            self.extras["num_turns"] = self.state.turns
            return BaseTextEnvStepOutput(observations=[], reward=result.reward, done=True, metadata={})

        try:
            results = self.client.search(result.query) if result.query is not None else ""
            observation = self.protocol.observe(self.state, result.query, results)
        except Exception as e:
            observation = str(e)
            if observation:
                self.state.transcript.append(observation)

        info = {"tool_group": "SearchToolGroup", "tool_name": "search", "tool_input": [result.query]}
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": observation}] if observation else [],
            reward=0.0,
            done=False,
            metadata=info,
        )
