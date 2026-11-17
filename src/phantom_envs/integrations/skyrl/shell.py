"""The ``phantom_shell`` SkyRL-Gym environment: a coding agent with bash in a sandbox over the corpus.

SkyRL's generator samples the agent's turns; :class:`~phantom_envs.terminal.loop.ShellProtocol` decides
when the attempt ends, a sandbox runs the commands, and :func:`~phantom_envs.terminal.task.verify`
scores the answer file.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from phantom_envs.scoring import METRICS
from phantom_envs.terminal.loop import ShellEpisodeState, ShellProtocol, tool_error
from phantom_envs.terminal.sandbox import BACKENDS, Sandbox, open_sandbox
from phantom_envs.terminal.task import has_answer, verify


@dataclass
class PhantomShellEnvConfig:
    """Config of the phantom_shell env: its sandboxes, the corpus they serve, and its limits and reward."""

    sandbox: Literal[BACKENDS] = field(
        default="shellsim",
        metadata={"help": "Sandbox backend: shellsim (simulated bash) or bwrap (real processes)"},
    )
    rootfs: str = field(
        default="",
        metadata={"help": "Root filesystem of the bwrap sandbox (the launcher sets it from SANDBOX_ROOTFS)"},
    )
    corpus_path: str = field(
        default="",
        metadata={"help": "corpus.jsonl that the sandbox holds under /corpus (set by the launcher)"},
    )
    timeout: float = field(default=60.0, metadata={"help": "Wall-clock limit per command, in seconds"})
    memory_gib: int = field(
        default=4,
        metadata={"help": "Memory limit in GiB: of the sandbox for shellsim, of each process for bwrap"},
    )
    max_output_chars: int = field(
        default=8000, metadata={"help": "Tool outputs are cut to this many characters"}
    )
    reward_func: Literal[tuple(METRICS)] = field(
        default="f1",
        metadata={"help": "Reward function used to score answers (em, subem, f1)"},
    )


class PhantomShellEnv(BaseTextEnv):
    """One shell episode driven by SkyRL's generator.

    Needs ``generator.use_conversation_multi_turn=true``: tool outputs go back as ``tool`` messages,
    which the chat template turns into tool responses.
    """

    def __init__(self, env_config: PhantomShellEnvConfig, extras: dict[str, Any] | None = None) -> None:
        """Set up one episode; :meth:`init` creates its sandbox.

        Args:
            env_config: The ``environment.skyrl_gym.phantom_shell`` config.
            extras: The dataset row's fields: ``reward_spec.ground_truth.target`` (the gold answers),
                ``data_source`` and ``max_turns``.
        """
        super().__init__()
        assert env_config.corpus_path, "environment.skyrl_gym.phantom_shell.corpus_path must be set"
        self.env_config = env_config
        self.extras: dict[str, Any] = extras  # NOTE: mutate this to store metrics
        self.protocol = ShellProtocol(
            max_turns=extras["max_turns"], max_output_chars=env_config.max_output_chars
        )
        self.state = ShellEpisodeState()
        self.sandbox: Sandbox | None = None
        self.answered = False

    def init(self, prompt: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Create the episode's sandbox.

        Runs in SkyRL's env executor, so a process's first (slow) use of a corpus doesn't block rollouts.

        Args:
            prompt: The messages that open the episode.

        Returns:
            The prompt, unchanged, and no extra info.
        """
        self.sandbox = open_sandbox(
            self.env_config.sandbox,
            self.env_config.corpus_path,
            timeout=self.env_config.timeout,
            memory_gib=self.env_config.memory_gib,
            rootfs=self.env_config.rootfs,
        )
        return prompt, {}

    def close(self) -> None:
        """Close the sandbox; the corpus stays prepared for the next episode."""
        if self.sandbox is not None:
            self.sandbox.close()
            self.sandbox = None

    def get_metrics(self) -> dict[str, Any]:
        """Return the episode's metrics, which SkyRL aggregates and logs to W&B.

        ``answered`` (wrote a non-empty answer file) separates learning the output format from
        learning to search.
        """
        return {
            "num_turns": self.state.turns,
            "num_commands": self.state.commands,
            "answered": float(self.answered),
        }

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Take one agent turn: run its commands, or end the attempt and verify the answer file.

        Args:
            action: The text the agent wrote.

        Returns:
            The command outputs as tool messages, or the final reward when the attempt is done.
        """
        result = self.protocol.act(self.state, action)
        self.turns = self.state.turns
        if result.done:
            # The verifier reads the answer file the agent left in the sandbox
            answer = self.sandbox.read_answer()
            self.answered = has_answer(answer)
            scores = verify(
                answer,
                gts=self.extras["reward_spec"]["ground_truth"]["target"],
                data_source=self.extras["data_source"],
            )
            # NOTE: mutate the extras to store the turns and the answer, saved by SkyRL Gym Generator
            self.extras["num_turns"] = self.state.turns
            self.extras["answer"] = answer
            return BaseTextEnvStepOutput(
                observations=[], reward=scores[self.env_config.reward_func], done=True, metadata={}
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
