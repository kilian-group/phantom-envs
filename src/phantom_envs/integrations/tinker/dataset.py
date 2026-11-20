"""Terminal episodes as tinker-cookbook RL data: groups of episodes per question, batches of groups.

- `TerminalEnvGroupBuilder`: `group_size` episodes of one task, whose rewards are centered on the
  group's mean. A harness (pi) runs each episode in its own bwrap sandbox.
- `TerminalDatasetBuilder`: batches of groups over a data dir's `tasks/<split>.jsonl`.

Usage (`scripts/tinker/train.py` runs a whole training job):

    env = TerminalEnvConfig(model_name="Qwen/Qwen3.5-9B", rootfs=os.environ["SANDBOX_ROOTFS"])
    builder = TerminalDatasetBuilder(data_dir="data/envs/pw", env=env, batch_size=128, group_size=8)
    config = tinker_cookbook.rl.train.Config(dataset_builder=builder, max_tokens=env.max_tokens, ...)

Gotchas:

- The cookbook ends some episodes early, with `cutoff_reward` and no verification: a turn that hits
  `max_tokens`, a conversation past `max_input_length`, or a turn the renderer can't parse. SkyRL
  verifies the answer file after a turn that hits its token limit.
- Every episode of a step runs its own harness at once, each with its own open files and sockets.
- Rollout workers get group builders pickled, so builders hold settings only until `make_envs`.
"""

import asyncio
import math
import random
from collections.abc import Sequence
from pathlib import Path

import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from phantom_envs._types import Task
from phantom_envs.integrations.tinker.harness import HarnessMessageEnv, TerminalEnvConfig, agent_setup
from phantom_envs.tasks import load_tasks
from phantom_envs.terminal.agent import AgentSetup


class TerminalEnvGroupBuilder(EnvGroupBuilder):
    """`group_size` episodes of one task, each in its own sandbox over the same corpus."""

    def __init__(self, task: Task, agent: AgentSetup, env: TerminalEnvConfig, group_size: int) -> None:
        """Store the settings; make_envs builds everything else, in the rollout worker.

        Args:
            task: The task.
            agent: The agent every episode runs (see `agent_setup`).
            env: Episode settings.
            group_size: Number of episodes.
        """
        self.task = task
        self.agent = agent
        self.env = env
        self.group_size = group_size
        self.message_envs: list[HarnessMessageEnv] = []

    async def make_envs(self) -> Sequence[Env]:
        """Build the group's envs, each with its own sandbox."""
        env = self.env
        renderer = renderers.get_renderer(env.renderer_name, get_tokenizer(env.model_name))
        self.message_envs = [
            HarnessMessageEnv(self.task, self.agent, env, renderer) for _ in range(self.group_size)
        ]
        return [
            EnvFromMessageEnv(
                renderer=renderer,
                message_env=message_env,
                failed_parse_reward=env.cutoff_reward,
                # The cookbook ends an episode when prompt + one turn's budget exceeds this
                max_trajectory_tokens=env.max_input_length + env.max_tokens,
                max_generation_tokens=env.max_tokens,
                context_overflow_reward=env.cutoff_reward,
            )
            for message_env in self.message_envs
        ]

    async def cleanup(self) -> None:
        """Close the episodes' sandboxes, also those of episodes the cookbook ended early."""
        await asyncio.gather(*(message_env.close() for message_env in self.message_envs))
        self.message_envs = []

    def logging_tags(self) -> list[str]:
        """Return the tags that the cookbook groups this builder's metrics by."""
        return ["terminal", self.task.data_source]


class TerminalDataset(RLDataset):
    """Batches of `batch_size` tasks, in order, each task a group of `group_size` episodes."""

    def __init__(
        self, tasks: list[Task], agent: AgentSetup, env: TerminalEnvConfig, batch_size: int, group_size: int
    ) -> None:
        """Store the tasks; get_batch makes the group builders.

        Args:
            tasks: The tasks, in batch order.
            agent: The agent every episode runs.
            env: Episode settings.
            batch_size: Tasks per batch (the last batch may be smaller).
            group_size: Episodes per task.
        """
        self.tasks = tasks
        self.agent = agent
        self.env = env
        self.batch_size = batch_size
        self.group_size = group_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Return the group builders of batch `index`."""
        tasks = self.tasks[index * self.batch_size : (index + 1) * self.batch_size]
        return [TerminalEnvGroupBuilder(t, self.agent, self.env, self.group_size) for t in tasks]

    def __len__(self) -> int:
        return math.ceil(len(self.tasks) / self.batch_size)


@chz.chz
class TerminalDatasetBuilder(RLDatasetBuilder):
    """Terminal episodes over a data dir: shuffled training tasks, and the first validation tasks for evals."""

    data_dir: str = chz.field(doc="Data dir with tasks/{train,validation}.jsonl and corpus.jsonl.")
    env: TerminalEnvConfig = chz.field(doc="Episode settings.")
    batch_size: int = chz.field(default=128, doc="Tasks per training step.")
    group_size: int = chz.field(default=8, doc="Episodes per task (GRPO's group).")
    seed: int = chz.field(default=0, doc="Seed of the training tasks' order.")
    num_eval_tasks: int = chz.field(
        default=500, doc="Validation tasks to evaluate on, one episode each (0: no evals)."
    )

    async def __call__(self) -> tuple[TerminalDataset, TerminalDataset | None]:
        """Build the training dataset and, unless `num_eval_tasks` is 0, the eval dataset."""
        # One agent for the whole run: a bad setting fails here, not in every episode
        agent = agent_setup(self.env, str(Path(self.data_dir) / "corpus.jsonl"))
        await asyncio.to_thread(agent.wait_for_servers)
        train_tasks = load_tasks(self.data_dir, "train")
        random.Random(self.seed).shuffle(train_tasks)
        train = TerminalDataset(train_tasks, agent, self.env, self.batch_size, self.group_size)
        if self.num_eval_tasks == 0:
            return train, None
        eval_tasks = load_tasks(self.data_dir, "validation")[: self.num_eval_tasks]
        return train, TerminalDataset(eval_tasks, agent, self.env, self.batch_size, group_size=1)
