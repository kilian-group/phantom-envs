"""A terminal episode run by a coding-agent harness, as a tinker-cookbook env.

The cookbook's loop samples every turn itself, while a harness such as pi runs its own loop and calls
a model endpoint. `HarnessMessageEnv` connects the two through a `BridgedEpisode`: the harness's chat
request becomes the env's observation, and the turn the cookbook samples goes back to the harness as
the chat reply.

    cookbook loop                         HarnessMessageEnv                  pi, in the bwrap sandbox
    initial_observation()  ------------>  start pi, wait for its request  <--  chat request 1
    sample a turn
    step(turn)  ----------------------->  reply to request 1  ------------>  runs the turn's commands
                                          wait for the next request  <------  chat request 2
    ...                                   pi exits: verify /app/output.md

The env keeps the conversation itself and takes only the new tool results from each request, so
every prompt extends the previous prompt and turn, and an episode trains as one sequence.

Gotchas:

- The harness's system prompt and tools open the conversation.
- An episode the cookbook ends early (a turn at `max_tokens`, a prompt past `max_input_length`)
  leaves the harness waiting; `HarnessMessageEnv.close` stops it.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import chz
from tinker_cookbook import renderers
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import STOP_METRIC_PREFIX, StopReason
from tinker_cookbook.third_party.openai_compat import (
    openai_messages_to_tinker,
    openai_tools_to_tinker,
    tinker_messages_to_openai,
)

from phantom_envs._types import Task
from phantom_envs.terminal.bridge import BridgedEpisode
from phantom_envs.terminal.harness import SCORED_STOP_REASONS, Harness
from phantom_envs.terminal.pi import PiHarness
from phantom_envs.terminal.task import build_instruction, has_answer, verify

# Roles whose messages are instructions rather than turns of the conversation
SYSTEM_ROLES: tuple[str, ...] = ("system", "developer")

# Each episode holds a thread while its harness runs; threads start only as episodes need them
EXECUTOR = ThreadPoolExecutor(max_workers=2048, thread_name_prefix="harness-episode-")


@chz.chz
class TerminalEnvConfig:
    """Settings of every terminal episode in a run: the model's chat format, the harness, the budgets."""

    model_name: str = chz.field(doc="Model whose tokenizer renders the conversation, e.g. Qwen/Qwen3.5-9B.")
    renderer_name: str = chz.field(
        default="qwen3_5_disable_thinking",
        doc="tinker-cookbook renderer (chat format and tool calls), e.g. tml_v0 for Inkling models.",
    )
    harness: str = chz.field(default="pi", doc="The coding agent that runs the episode's loop: pi.")
    rootfs: str = chz.field(default="", doc="Root filesystem of the bwrap sandbox (SANDBOX_ROOTFS).")
    system_prompt: str | None = chz.field(
        default=None, doc="Replaces the harness's own system prompt; None keeps it."
    )
    max_turns: int = chz.field(default=20, doc="Maximum number of agent turns per episode.")
    reward_metric: str = chz.field(default="f1", doc="Reward metric: em, subem or f1.")
    command_timeout: float = chz.field(default=60.0, doc="Wall-clock limit per command, in seconds.")
    episode_timeout: float = chz.field(default=1800.0, doc="Wall-clock limit per episode, in seconds.")
    memory_gib: int = chz.field(default=4, doc="Memory limit of each process in the sandbox, in GiB.")
    max_output_chars: int = chz.field(default=8000, doc="Tool outputs are cut to this many characters.")
    max_tokens: int = chz.field(default=2048, doc="Tokens sampled per turn.")
    max_input_length: int = chz.field(
        default=32768, doc="An episode ends when its conversation (the next prompt) gets longer."
    )
    cutoff_reward: float = chz.field(
        default=0.0, doc="Reward of an episode that ends early or fails (0: no format penalty, as in SkyRL)."
    )


def make_harness(env: TerminalEnvConfig) -> Harness:
    """Build the harness that `env.harness` names, under the episode's budgets."""
    if env.harness != "pi":
        raise ValueError(f"Unknown harness {env.harness!r} (expected pi)")
    return PiHarness(
        max_turns=env.max_turns,
        command_timeout=env.command_timeout,
        max_output_chars=env.max_output_chars,
        episode_timeout=env.episode_timeout,
        system_prompt=env.system_prompt,
    )


def tool_declarations(renderer: renderers.Renderer, tools: list[dict]) -> list[dict]:
    """Return tools in the form that makes `renderer` declare them as a chat template would.

    vLLM hands chat templates tools in the OpenAI form `{"type": "function", "function": ...}`, and
    renderers wrap the cookbook's `ToolSpec` (the inner dict) in it themselves, except the Qwen3.5
    family's, which writes tools as given. That one gets the OpenAI form.

    Args:
        renderer: The renderer that writes the tool declarations.
        tools: The tools, in the OpenAI form.

    Returns:
        The tool list for `renderer.create_conversation_prefix_with_tools`.
    """
    writes_as_given = (
        type(renderer).create_conversation_prefix_with_tools
        is Qwen3_5Renderer.create_conversation_prefix_with_tools
    )
    return tools if writes_as_given else openai_tools_to_tinker(tools)


def text_of(content: Any) -> str:
    """Return the text of an OpenAI message's content, which is a string or a list of parts."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [])


def with_call_ids(message: renderers.Message, turn: int) -> renderers.Message:
    """Return an assistant message whose tool calls all have ids, which the chat API requires.

    Args:
        message: The message a renderer parsed; its tool calls may lack ids.
        turn: The turn's number, which makes the new ids unique within the episode.

    Returns:
        A copy of the message with ids filled in.
    """
    calls = message.get("tool_calls")
    if not calls:
        return message
    named = [call.model_copy(update={"id": call.id or f"call_{turn}_{i}"}) for i, call in enumerate(calls)]
    return {**message, "tool_calls": named}


class HarnessMessageEnv(MessageEnv):
    """One terminal episode whose turns a harness asks for, at the message level."""

    def __init__(
        self, task: Task, corpus_path: str, env: TerminalEnvConfig, renderer: renderers.Renderer
    ) -> None:
        """Set up the episode; `initial_observation` starts the sandbox and the harness.

        Args:
            task: The task.
            corpus_path: corpus.jsonl that the sandbox holds under /corpus.
            env: Episode settings.
            renderer: The renderer that declares the harness's tools.
        """
        self.task = task
        self.env = env
        self.renderer = renderer
        self.episode = BridgedEpisode(
            make_harness(env),
            corpus_path,
            rootfs=env.rootfs,
            command_timeout=env.command_timeout,
            memory_gib=env.memory_gib,
            context_window=env.max_input_length + env.max_tokens,
            max_tokens=env.max_tokens,
            executor=EXECUTOR,
        )
        self.history: list[renderers.Message] = []
        self.turns = 0
        self.commands = 0
        # EnvFromMessageEnv puts the task id on stored trajectories
        self.example_id = task.id

    async def initial_observation(self) -> list[renderers.Message]:
        """Start the harness and return the conversation its first model call opens with."""
        request = await self.episode.start(build_instruction(self.task.question))
        system_prompt = "\n\n".join(
            text_of(m["content"]) for m in request["messages"] if m["role"] in SYSTEM_ROLES
        )
        turns = [m for m in request["messages"] if m["role"] not in SYSTEM_ROLES]
        tools = tool_declarations(self.renderer, request.get("tools") or [])
        self.history = self.renderer.create_conversation_prefix_with_tools(tools, system_prompt=system_prompt)
        self.history += openai_messages_to_tinker(turns)
        return self.history

    async def step(self, message: renderers.Message) -> MessageStepResult:
        """Give the harness the agent's turn; return its tool results, or verify when it is done."""
        self.turns += 1
        message = with_call_ids(message, self.turns)
        calls = message.get("tool_calls") or []
        self.commands += len(calls)
        self.history.append(message)
        results = await self.episode.step(tinker_messages_to_openai([message])["messages"][0])
        if results is None:
            return await self.verify()
        names = {call.id: call.function.name for call in calls}
        for result in openai_messages_to_tinker(results):
            self.history.append({**result, "name": names.get(result.get("tool_call_id"), "")})
        return MessageStepResult(reward=0.0, episode_done=False, next_messages=self.history)

    async def verify(self) -> MessageStepResult:
        """End the episode: score the answer file the agent left in the sandbox."""
        run, answer = await self.episode.finish()
        stop_reason = StopReason.MAX_TURNS if self.turns >= self.env.max_turns else StopReason.COMPLETED
        if run.stop_reason == "timeout":
            stop_reason = StopReason.ROLLOUT_TIMEOUT
        metrics = {
            "num_turns": self.turns,
            "num_commands": self.commands,
            # Separates learning the output format from learning to search
            "answered": float(has_answer(answer)),
            "harness_error": float(run.stop_reason == "error"),
            **verify(answer, self.task.answers, self.task.data_source),
            f"{STOP_METRIC_PREFIX}{stop_reason}": 1.0,
        }
        scored = run.stop_reason in SCORED_STOP_REASONS
        return MessageStepResult(
            reward=metrics[self.env.reward_metric] if scored else self.env.cutoff_reward,
            episode_done=True,
            next_messages=self.history,
            metrics=metrics,
            logs={"answer": answer or "", "harness_error": run.error or ""},
        )

    async def close(self) -> None:
        """Stop the harness and close the sandbox. Safe to call more than once."""
        await self.episode.close()
