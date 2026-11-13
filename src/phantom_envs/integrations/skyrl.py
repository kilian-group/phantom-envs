"""SkyRL integration: the ``phantom_search`` SkyRL-Gym environment and its config.

The episode logic lives in :mod:`phantom_envs.protocol` and the HTTP calls in
:mod:`phantom_envs.retriever.client`; this module adapts them to SkyRL-Gym's ``BaseTextEnv``.

Ray workers import this module when they unpickle :class:`PhantomTrainConfig` (the entrypoint),
so everything they need is here: the env and its config.

Usage (inside the Ray remote entrypoint)::

    from phantom_envs.integrations.skyrl import register

    register()  # then every dataset row with env_class "phantom_search" uses PhantomSearchEnv
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import EnvironmentConfig, SkyRLGymConfig
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.registration import register as skyrl_gym_register
from skyrl_gym.envs.registration import registry
from skyrl_gym.envs.search.env import SearchEnvConfig

from phantom_envs.protocol import EpisodeState, SearchProtocol
from phantom_envs.retriever.client import SearchClient
from phantom_envs.scoring import METRICS

ENV_ID = "phantom_search"


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


@dataclass
class PhantomSkyRLGymConfig(SkyRLGymConfig):
    """SkyRLGymConfig with a phantom_search field (SkyRL looks env configs up by env_class)."""

    phantom_search: PhantomSearchEnvConfig = field(default_factory=PhantomSearchEnvConfig)


@dataclass
class PhantomEnvironmentConfig(EnvironmentConfig):
    """EnvironmentConfig that uses PhantomSkyRLGymConfig."""

    skyrl_gym: PhantomSkyRLGymConfig = field(default_factory=PhantomSkyRLGymConfig)


@dataclass
class PhantomTrainConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig whose environment section accepts ``phantom_search``."""

    environment: PhantomEnvironmentConfig = field(default_factory=PhantomEnvironmentConfig)


class PhantomSearchEnv(BaseTextEnv):
    """Drop-in replacement for SkyRL-Gym's SearchEnv with a pooled retrieval client."""

    def __init__(self, env_config: PhantomSearchEnvConfig, extras: dict[str, Any] | None = None):
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
        # Gets aggregated to wandb
        return {"num_turns": self.state.turns}

    def step(self, action: str) -> BaseTextEnvStepOutput:
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


def register() -> None:
    """Register :class:`PhantomSearchEnv` with SkyRL-Gym as ``phantom_search`` (idempotent)."""
    if ENV_ID not in registry:
        skyrl_gym_register(id=ENV_ID, entry_point=f"{__name__}:PhantomSearchEnv")
