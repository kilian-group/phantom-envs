"""Shell episodes run by a coding-agent harness, as a SkyRL generator.

With `environment.skyrl_gym.phantom_shell.harness` set (e.g. to `pi`), the harness runs the agent's
loop inside the bwrap sandbox, and SkyRL doesn't see the turns. skycap
(https://github.com/NovaSky-AI/SkyRL/tree/main/skycap) sits between the two: each episode gets its
own OpenAI-compatible URL on a capture server in this process, which renders the harness's chat
requests to tokens, samples from SkyRL's engines token-in/token-out, and hands back the episode's
exact tokens, loss mask and logprobs.

Per episode, `HarnessGenerator` opens a skycap trajectory and a sandbox, runs the harness, verifies
the answer file, and turns the captured tokens into one row of SkyRL's `GeneratorOutput`, shaped
like a row of SkyRL's own multi-turn generator.

Usage (in the entrypoints of `scripts/skyrl`):

    class Experiment(HarnessGeneratorMixin, BasePPOExp):
        pass

Gotchas:

- A harness needs the bwrap sandbox (`sandbox: bwrap`).
- An episode whose harness fails or runs out of time trains nothing and scores 0; so does one whose
  prompt outgrows the model's context, as in SkyRL's generator. The `generate/harness/*` metrics
  count them.
- Routed experts and sampler support are not captured.
"""

import asyncio
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from skycap import CapturePool, CaptureService, Sample
from skycap.tokens.engine import VLLMEngine
from skycap.tokens.renderer import RenderersRenderer
from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInterface
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import SamplingParams
from skyrl.train.generators.base import (
    TRAINING_PHASE_EVAL,
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
)
from skyrl.train.generators.utils import (
    apply_overlong_filtering,
    build_vllm_cache_salt,
    get_rollout_metrics,
)
from transformers import PreTrainedTokenizerBase

from phantom_envs.terminal.harness import Harness, HarnessRun, ModelEndpoint, run_harness
from phantom_envs.terminal.pi import PiHarness
from phantom_envs.terminal.sandbox import Sandbox, open_sandbox
from phantom_envs.terminal.task import has_answer, verify

# The model name a harness requests; skycap sends the engine its own name for the policy
MODEL_NAME: str = "policy"
# Seconds a call to the capture server may take. It shares this process with a whole batch's
# episodes, so it answers slowly while a large batch starts
CAPTURE_TIMEOUT: float = 600.0
# Stop reasons of an episode that trains nothing and scores 0
FAILED: frozenset[str] = frozenset({"error", "timeout"})


class RouterEngine(VLLMEngine):
    """SkyRL's inference router as a skycap engine.

    The router speaks vLLM's token-in/token-out wire on its own path, keeps each trajectory on one
    engine (its session), and keys vLLM's prefix cache by a salt that changes with the weights.
    """

    # skycap's engine settings: the router's name and paths
    name = "skyrl"
    generate_path = "/skyrl/v1/generate"
    release_path = "/finish_session"

    def __init__(self, cache_salt: Callable[[], str | None]) -> None:
        """Set up the engine.

        Args:
            cache_salt: Returns the prefix-cache salt of the weights the engines serve now, or None.
        """
        self.cache_salt = cache_salt

    def request(self, *, cache_salt: str | None, **kwargs: Any) -> dict[str, Any]:
        """Build a generate request, salted for the current weights unless the caller salted it."""
        return super().request(cache_salt=cache_salt or self.cache_salt(), **kwargs)


@dataclass
class Episode:
    """One episode's outcome: what the harness and the verifier said, and what skycap captured."""

    stop_reason: str = field(metadata={"description": "stop, length, context_length, timeout or error."})
    reward: float = field(default=0.0, metadata={"description": "The verifier's score."})
    answer: str | None = field(default=None, metadata={"description": "The answer file's contents."})
    sample: Sample | None = field(
        default=None, metadata={"description": "The episode's tokens; None if skycap captured none."}
    )
    unbridged_calls: int = field(
        default=0, metadata={"description": "Model calls that didn't extend the previous one's tokens."}
    )
    seconds: float = field(default=0.0, metadata={"description": "Wall-clock time of the episode."})

    @property
    def trains(self) -> bool:
        """Whether the episode's tokens enter the loss."""
        return self.sample is not None and self.stop_reason not in FAILED

    @property
    def turns(self) -> int:
        """Model calls on the episode's path."""
        return len(self.sample.targets) if self.sample is not None else 0

    @property
    def commands(self) -> int:
        """Tool calls on the episode's path."""
        if self.sample is None:
            return 0
        return sum(len(message.get("tool_calls") or []) for message in self.sample.messages)


def sampling_overrides(params: SamplingParams) -> dict[str, Any]:
    """Return what every model call is sampled with, whatever the harness asks for.

    The trainer computes logprobs under these settings, so the rollout must use them too.

    Args:
        params: The phase's sampling params (`generator.sampling_params` or its eval twin).

    Returns:
        The overrides, in vLLM's names.
    """
    return {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "min_p": params.min_p,
        "max_tokens": params.max_generate_length,
    }


def to_row(episode: Episode) -> tuple[list[int], list[int], list[int], list[float]]:
    """Split an episode's tokens into a prompt and a response, at the first token the model wrote.

    Args:
        episode: The episode.

    Returns:
        The prompt ids, the response ids, the response's loss mask (1 on tokens the model wrote, 0 on
        tool outputs) and its logprobs. An episode without tokens gives a one-token row that trains
        nothing, so the batch keeps a row per episode.
    """
    sample = episode.sample
    if sample is None or sample.input_ids is None or 1 not in sample.loss_mask:
        return [0], [0], [0], [0.0]
    first = sample.loss_mask.index(1)
    loss_mask = sample.loss_mask[first:] if episode.trains else [0] * (len(sample.input_ids) - first)
    return sample.input_ids[:first], sample.input_ids[first:], loss_mask, list(sample.logprobs[first:])


def to_generator_output(
    episodes: list[Episode], env_classes: list[str], overlong_filtering: bool = False
) -> GeneratorOutput:
    """Assemble a batch's episodes into SkyRL's `GeneratorOutput`, one row per episode.

    Args:
        episodes: The episodes, in the input batch's order.
        env_classes: The env class of each episode, for per-env metrics.
        overlong_filtering: Whether episodes that didn't end with `stop` train nothing
            (`generator.apply_overlong_filtering`).

    Returns:
        The generator output, with rollout metrics under `generate/` and `environment/`.
    """
    rows = [to_row(episode) for episode in episodes]
    prompt_ids, response_ids, loss_masks, logprobs = (list(column) for column in zip(*rows, strict=True))
    rewards = [episode.reward for episode in episodes]
    # SkyRL knows stop and length; a filled context is its "length"
    stop_reasons = ["length" if e.stop_reason == "context_length" else e.stop_reason for e in episodes]
    env_metrics = [
        {"num_turns": e.turns, "num_commands": e.commands, "answered": float(has_answer(e.answer))}
        for e in episodes
    ]
    seconds = [episode.seconds for episode in episodes]

    metrics = get_rollout_metrics(
        response_ids, rewards, env_metrics, env_classes, loss_masks, trajectory_completion_times=seconds
    )
    metrics["generate/harness/num_error_episodes"] = sum(e.stop_reason == "error" for e in episodes)
    metrics["generate/harness/num_timeout_episodes"] = sum(e.stop_reason == "timeout" for e in episodes)
    metrics["generate/harness/num_context_length_episodes"] = sum(
        e.stop_reason == "context_length" for e in episodes
    )
    # Zero for a harness that keeps its history append-only, as pi does
    metrics["generate/harness/num_unbridged_calls"] = sum(e.unbridged_calls for e in episodes)

    if overlong_filtering:
        loss_masks = apply_overlong_filtering(loss_masks, stop_reasons)
    return GeneratorOutput(
        prompt_token_ids=prompt_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        stop_reasons=stop_reasons,
        rollout_metrics=metrics,
        rollout_logprobs=logprobs,
        trajectory_ids=None,
        trajectory_generation_times=seconds,
        trajectory_time_splits=None,
        rollout_expert_indices=None,
        rollout_sample_support=None,
        is_last_step=None,
        env_metrics=env_metrics,
    )


class HarnessGenerator(GeneratorInterface):
    """Runs each prompt's episode with a harness in a bwrap sandbox, capturing its tokens with skycap."""

    def __init__(self, cfg: SkyRLTrainConfig, inference_engine_client: InferenceEngineInterface) -> None:
        """Set up the generator; capture servers start with the first batch.

        Args:
            cfg: The run's config. Its `environment.skyrl_gym.phantom_shell` picks the harness.
            inference_engine_client: SkyRL's client of the engines that serve the policy.

        Raises:
            ValueError: If the config asks for something a harness can't do.
        """
        shell_cfg = cfg.environment.skyrl_gym.phantom_shell
        engine_cfg = cfg.generator.inference_engine
        if shell_cfg.sandbox != "bwrap":
            raise ValueError(
                "A harness runs real programs: set environment.skyrl_gym.phantom_shell.sandbox=bwrap"
            )
        if cfg.generator.step_wise_trajectories:
            raise ValueError("A harness episode is one row: set generator.step_wise_trajectories=false")
        if engine_cfg.enable_return_routed_experts or engine_cfg.enable_return_sample_support_set:
            raise ValueError("Routed experts and sampler support are not captured with a harness")

        self.cfg = cfg
        self.shell_cfg = shell_cfg
        self.inference_engine_client = inference_engine_client
        self.policy_model_name = resolve_policy_model_name(cfg)
        self.max_model_len: int = (engine_cfg.engine_init_kwargs or {}).get("max_model_len") or (
            cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length
        )
        self.harness: Harness = PiHarness(
            max_turns=cfg.generator.max_turns,
            command_timeout=shell_cfg.timeout,
            max_output_chars=shell_cfg.max_output_chars,
            episode_timeout=shell_cfg.episode_timeout,
            system_prompt=shell_cfg.system_prompt or None,
        )
        # One capture server per sampling setup: training and evaluation may sample differently
        self.captures: dict[tuple, CaptureService] = {}

    def cache_salt(self) -> str | None:
        """Return vLLM's prefix-cache salt for the weights the engines serve now, as SkyRL builds it."""
        weight_version = getattr(self.inference_engine_client, "weight_version", None)
        if not self.cfg.generator.use_cache_salt or weight_version is None:
            return None
        return build_vllm_cache_salt(weight_version, self.policy_model_name)

    def capture_url(self, params: SamplingParams) -> str:
        """Return the URL of the capture server that samples with these params, starting it if needed."""
        overrides = sampling_overrides(params)
        key = tuple(sorted(overrides.items()))
        if key not in self.captures:
            # The harness sends its own tools, so the config's tool declaration is left out
            template_kwargs = {
                k: v for k, v in self.cfg.generator.chat_template_kwargs.items() if k != "tools"
            }
            # The renderer picks the chat format by base model name; a checkpoint dir has none
            chat_format_model = self.shell_cfg.chat_format_model or self.cfg.trainer.policy.model.path
            service = CaptureService(
                self.inference_engine_client.get_endpoint_url(),
                mode="tokens",
                engine=RouterEngine(self.cache_salt),
                renderer=RenderersRenderer(chat_format_model, chat_template_kwargs=template_kwargs),
                model=self.policy_model_name,
                max_model_len=self.max_model_len,
                sampling_overrides=overrides,
                host="127.0.0.1",
            )
            service.start()
            self.captures[key] = service
        return self.captures[key].url

    def close(self) -> None:
        """Stop the capture servers."""
        for service in self.captures.values():
            service.stop()
        self.captures.clear()

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        """Run the batch's episodes, all at once, and return one row per prompt in input order.

        Args:
            input_batch: SkyRL's batch: per episode a prompt (whose user message is the task) and the
                dataset row's fields (the gold answers and the data source).
            disable_tqdm: Accepted for SkyRL's callers; this generator shows no progress bar.

        Returns:
            The episodes' tokens, loss masks, logprobs and rewards.
        """
        prompts, extras = input_batch["prompts"], input_batch["env_extras"]
        metadata = input_batch.get("batch_metadata")
        evaluating = getattr(metadata, "training_phase", None) == TRAINING_PHASE_EVAL
        params = self.cfg.generator.eval_sampling_params if evaluating else self.cfg.generator.sampling_params
        url = self.capture_url(params)

        # Each episode holds a thread while its harness runs
        with ThreadPoolExecutor(max_workers=len(prompts), thread_name_prefix="harness-episode-") as executor:
            async with CapturePool([url], timeout=CAPTURE_TIMEOUT) as pool:
                episodes = await asyncio.gather(
                    *(
                        self.episode(pool, executor, prompt, extra)
                        for prompt, extra in zip(prompts, extras, strict=True)
                    )
                )
        for extra, episode in zip(extras, episodes, strict=True):
            # NOTE: mutate the extras to store the turns and the answer, saved in SkyRL's eval dumps
            extra["num_turns"] = episode.turns
            extra["answer"] = episode.answer
        return to_generator_output(
            list(episodes), input_batch["env_classes"], self.cfg.generator.apply_overlong_filtering
        )

    async def episode(
        self,
        pool: CapturePool,
        executor: Executor,
        prompt: list[Mapping[str, Any]],
        extras: Mapping[str, Any],
    ) -> Episode:
        """Run one episode. Never raises: a failed episode must not cancel its batch.

        Args:
            pool: The capture servers' client.
            executor: Where the sandbox and the harness block.
            prompt: The episode's opening messages; the user message is the task.
            extras: The dataset row's fields.

        Returns:
            The episode's outcome.
        """
        started = time.monotonic()
        try:
            episode = await self.attempt(pool, executor, prompt, extras)
        except Exception as error:  # noqa: BLE001 (reported as a failed episode)
            logger.warning(f"A harness episode failed: {type(error).__name__}: {error}")
            episode = Episode(stop_reason="error")
        episode.seconds = time.monotonic() - started
        return episode

    async def attempt(
        self,
        pool: CapturePool,
        executor: Executor,
        prompt: list[Mapping[str, Any]],
        extras: Mapping[str, Any],
    ) -> Episode:
        """Run one episode: sandbox, harness, verifier, and the captured tokens. See `episode`."""
        loop = asyncio.get_running_loop()
        instruction = next(message["content"] for message in prompt if message["role"] == "user")
        # "final": the conversation the harness ended with, which for an append-only harness is all of it
        async with pool.trajectory({"data_source": extras["data_source"]}, paths="final") as trajectory:
            endpoint = ModelEndpoint(
                base_url=trajectory.base_url,
                model=MODEL_NAME,
                api_key=trajectory.api_key or "none",
                context_window=self.max_model_len,
                max_tokens=self.cfg.generator.sampling_params.max_generate_length,
            )
            sandbox = await loop.run_in_executor(executor, self.open_sandbox)
            try:
                run: HarnessRun = await run_harness(self.harness, sandbox, instruction, endpoint, executor)
                answer = await loop.run_in_executor(executor, sandbox.read_answer)
            finally:
                await loop.run_in_executor(executor, sandbox.close)

            scores = verify(answer, extras["reward_spec"]["ground_truth"]["target"], extras["data_source"])
            # As in SkyRL's generator: a turn cut at its token limit is still verified, and an
            # episode that outgrew the context scores 0
            scored = run.stop_reason in ("stop", "length")
            reward = scores[self.shell_cfg.reward_func] if scored else 0.0
            if run.error:
                logger.warning(f"A harness episode ended with {run.stop_reason}: {run.error}")
            finished = await trajectory.finish({"reward": reward, "stop_reason": run.stop_reason})

        if finished.status != "finished" or not finished.samples:
            logger.warning(f"skycap captured no tokens to train on (status {finished.status!r})")
            return Episode(stop_reason="error", answer=answer)
        return Episode(
            stop_reason=run.stop_reason,
            reward=reward,
            answer=answer,
            sample=finished.samples[0],
            unbridged_calls=finished.unbridged_calls,
        )

    def open_sandbox(self) -> Sandbox:
        """Open a bwrap sandbox over the env's corpus."""
        return open_sandbox(
            "bwrap",
            self.shell_cfg.corpus_path,
            timeout=self.shell_cfg.timeout,
            memory_gib=self.shell_cfg.memory_gib,
            rootfs=self.shell_cfg.rootfs,
        )


class HarnessGeneratorMixin:
    """Gives a SkyRL experiment the harness generator when the config asks for a harness.

    Put it before `BasePPOExp` or `EvalOnlyEntrypoint` in a subclass's bases.
    """

    def get_generator(
        self,
        cfg: SkyRLTrainConfig,
        tokenizer: PreTrainedTokenizerBase,
        inference_engine_client: InferenceEngineInterface,
    ) -> GeneratorInterface:
        """Return `HarnessGenerator` for shell runs with a harness, else SkyRL's own generator."""
        shell = cfg.environment.env_class == "phantom_shell"
        if shell and cfg.environment.skyrl_gym.phantom_shell.harness != "none":
            return HarnessGenerator(cfg, inference_engine_client)
        return super().get_generator(cfg, tokenizer, inference_engine_client)
