"""SkyRL integration: the ``phantom_search`` and ``phantom_shell`` SkyRL-Gym environments and their
configs.

The episode logic lives in :mod:`phantom_envs.protocol` (search) and :mod:`phantom_envs.shell`
(bash in a sandbox), and the HTTP calls in :mod:`phantom_envs.retriever.client`; this module
adapts them to SkyRL-Gym's ``BaseTextEnv``.

Ray workers import this module when they unpickle :class:`PhantomTrainConfig` (the entrypoint),
so everything they need is here: the envs and their configs.

Usage (inside the Ray remote entrypoint)::

    from phantom_envs.integrations.skyrl import register

    register()  # then dataset rows with env_class "phantom_search" use PhantomSearchEnv, and so on
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
from phantom_envs.shell import ShellEpisodeState, ShellProtocol, ShellSandbox, corpus_image, tool_error

ENV_ID = "phantom_search"
SHELL_ENV_ID = "phantom_shell"


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
class PhantomShellEnvConfig:
    """Config of the phantom_shell env: the corpus its sandboxes serve, and its limits and reward."""

    corpus_path: str = field(
        default="",
        metadata={"help": "corpus.jsonl that the sandbox holds under /corpus (set by the launcher)"},
    )
    timeout: float = field(default=60.0, metadata={"help": "Wall-clock limit per command, in seconds"})
    memory_gib: int = field(default=4, metadata={"help": "Memory limit of each sandbox, in GiB"})
    max_output_chars: int = field(
        default=8000, metadata={"help": "Tool outputs are cut to this many characters"}
    )
    reward_func: Literal[tuple(METRICS)] = field(
        default="f1",
        metadata={"help": "Reward function used to score answers (em, subem, f1)"},
    )


@dataclass
class PhantomSkyRLGymConfig(SkyRLGymConfig):
    """SkyRLGymConfig with phantom_search and phantom_shell fields (SkyRL looks env configs up by
    env_class)."""

    phantom_search: PhantomSearchEnvConfig = field(default_factory=PhantomSearchEnvConfig)
    phantom_shell: PhantomShellEnvConfig = field(default_factory=PhantomShellEnvConfig)


@dataclass
class PhantomEnvironmentConfig(EnvironmentConfig):
    """EnvironmentConfig that uses PhantomSkyRLGymConfig."""

    skyrl_gym: PhantomSkyRLGymConfig = field(default_factory=PhantomSkyRLGymConfig)


@dataclass
class PhantomTrainConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig whose environment section accepts ``phantom_search`` and ``phantom_shell``."""

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


class PhantomShellEnv(BaseTextEnv):
    """A coding agent with a bash tool in a sandbox over the corpus (see :mod:`phantom_envs.shell`).

    Needs ``generator.use_conversation_multi_turn=true``: tool outputs go back as ``tool`` messages,
    which the chat template turns into tool responses.
    """

    def __init__(self, env_config: PhantomShellEnvConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        assert env_config.corpus_path, "environment.skyrl_gym.phantom_shell.corpus_path must be set"
        self.env_config = env_config
        self.extras: dict[str, Any] = extras  # NOTE: mutate this to store metrics
        self.protocol = ShellProtocol(
            gts=extras["reward_spec"]["ground_truth"]["target"],
            data_source=extras["data_source"],
            max_turns=extras["max_turns"],
            metric=env_config.reward_func,
            max_output_chars=env_config.max_output_chars,
        )
        self.state = ShellEpisodeState()
        self.sandbox: ShellSandbox | None = None
        self.answered = False

    def init(self, prompt: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        # Runs in SkyRL's env executor, so a process's first (slow) corpus load doesn't block rollouts
        self.sandbox = ShellSandbox(
            corpus_image(self.env_config.corpus_path),
            timeout=self.env_config.timeout,
            memory_gib=self.env_config.memory_gib,
        )
        return prompt, {}

    def close(self) -> None:
        self.sandbox = None  # frees the machine's own files; the corpus image stays cached

    def get_metrics(self) -> dict[str, Any]:
        # Gets aggregated to wandb. "answered" (wrote a non-empty answer file) separates learning the
        # output format from learning to search.
        return {
            "num_turns": self.state.turns,
            "num_commands": self.state.commands,
            "answered": float(self.answered),
        }

    def step(self, action: str) -> BaseTextEnvStepOutput:
        result = self.protocol.act(self.state, action)
        self.turns = self.state.turns
        if result.done:
            # The verifier reads the answer file the agent left in the sandbox
            answer = self.sandbox.read_answer()
            self.answered = bool((answer or "").strip())
            # NOTE: mutate the extras to store the turns and the answer, saved by SkyRL Gym Generator
            self.extras["num_turns"] = self.state.turns
            self.extras["answer"] = answer
            return BaseTextEnvStepOutput(
                observations=[], reward=self.protocol.reward(answer), done=True, metadata={}
            )

        outputs = [tool_error(call) or self.sandbox.run(call.arguments["command"]) for call in result.calls]
        observations = self.protocol.observe(self.state, outputs)
        info = {
            "tool_group": "ShellToolGroup",
            "tool_name": "bash",
            "tool_input": [c.arguments for c in result.calls],
        }
        return BaseTextEnvStepOutput(
            observations=[{"role": "tool", "content": o} for o in observations],
            reward=0.0,
            done=False,
            metadata=info,
        )


def register() -> None:
    """Register :class:`PhantomSearchEnv` as ``phantom_search`` and :class:`PhantomShellEnv` as
    ``phantom_shell`` with SkyRL-Gym (idempotent)."""
    if ENV_ID not in registry:
        skyrl_gym_register(id=ENV_ID, entry_point=f"{__name__}:PhantomSearchEnv")
    if SHELL_ENV_ID not in registry:
        skyrl_gym_register(id=SHELL_ENV_ID, entry_point=f"{__name__}:PhantomShellEnv")
