"""Tests for phantom_envs.integrations.tinker: episodes through tinker-cookbook's rollout loop.

A scripted policy plays the model's turns, so no test reaches the Tinker service. Needs the tinker
extra and Qwen3.5-9B's tokenizer in the Hugging Face cache; episodes also need bubblewrap and
SANDBOX_ROOTFS (see scripts/sandbox/build_rootfs.sh).
"""

import asyncio
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("tinker_cookbook")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "tinker"))

import eval as tinker_eval  # noqa: E402 (scripts/tinker/eval.py)
import tinker  # noqa: E402
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs  # noqa: E402
from tinker_cookbook.rl.data_processing import trajectory_to_data  # noqa: E402
from tinker_cookbook.rl.rollouts import do_group_rollout  # noqa: E402
from tinker_cookbook.tokenizer_utils import register_tokenizer  # noqa: E402

from phantom_envs._types import Task  # noqa: E402
from phantom_envs.integrations.tinker.cli import EvalConfig  # noqa: E402
from phantom_envs.integrations.tinker.dataset import (  # noqa: E402
    TerminalDatasetBuilder,
    TerminalEnvGroupBuilder,
)
from phantom_envs.integrations.tinker.harness import TerminalEnvConfig, agent_setup  # noqa: E402
from phantom_envs.tasks import write_tasks  # noqa: E402

QWEN = "Qwen/Qwen3.5-9B"
# The cached tokenizer, registered under its own name so tests never reach the Hugging Face Hub
LOCAL_QWEN = "local/Qwen3.5-9B"
TASK = Task(id="t0", data_source="pw", question="Who is the mother of Ann Lee?", answers=["Bea Lee"])


def bash_call(command: str) -> str:
    """An agent turn's tool call, as Qwen3.5 writes it."""
    return f"<tool_call>\n<function=bash>\n<parameter=command>\n{command}\n</parameter>\n</function>\n</tool_call>"


class ScriptedPolicy(TokenCompleter):
    """Plays fixed turns: the n-th turn of every episode is `turns[n]` (the last one repeats).

    Stateless, so one policy serves a whole group's concurrent episodes.
    """

    def __init__(self, tokenizer, turns: list[str]):
        self.tokenizer = tokenizer
        self.turns = turns

    async def __call__(self, model_input: tinker.ModelInput, stop, *, max_tokens=None) -> TokensWithLogprobs:
        prompt = self.tokenizer.decode(model_input.to_ints())
        n = prompt.count("<|im_start|>assistant") - 1
        tokens = self.tokenizer.encode(self.turns[min(n, len(self.turns) - 1)] + "<|im_end|>")
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=[-0.5] * len(tokens))


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    except OSError:
        pytest.skip(f"{QWEN}'s tokenizer is not in the Hugging Face cache")
    register_tokenizer(LOCAL_QWEN, lambda: tokenizer)
    return tokenizer


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("pw")
    docs = [
        {"id": "0", "title": "Ann Lee", "contents": "# Ann Lee\n\nThe mother of Ann Lee is Bea Lee.\n"},
        {"id": "1", "title": "Bea Lee", "contents": "# Bea Lee\n\nThe hobby of Bea Lee is chess.\n"},
    ]
    (path / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    tasks = [
        Task(id=f"t{i}", data_source="pw", question=f"Question {i}?", answers=["Bea Lee"]) for i in range(5)
    ]
    write_tasks(path, "train", tasks)
    write_tasks(path, "validation", tasks[:3])
    return path


@pytest.fixture
def rootfs() -> str:
    """The sandbox image; skips the test if this machine can't run the bwrap sandbox."""
    rootfs = os.environ.get("SANDBOX_ROOTFS")
    if shutil.which("bwrap") is None or not rootfs:
        pytest.skip("needs bubblewrap and SANDBOX_ROOTFS")
    return rootfs


def group(data_dir, rootfs: str = "", group_size: int = 2, max_turns: int = 5) -> TerminalEnvGroupBuilder:
    env = TerminalEnvConfig(model_name=LOCAL_QWEN, rootfs=rootfs, max_turns=max_turns)
    return TerminalEnvGroupBuilder(TASK, agent_setup(env, str(data_dir / "corpus.jsonl")), env, group_size)


def test_group_builder_pickles(data_dir):
    """Rollout executors send group builders to worker processes."""
    builder = pickle.loads(pickle.dumps(group(data_dir)))
    assert builder.task == TASK and builder.env.max_turns == 5


def test_dataset_builder(data_dir):
    env = TerminalEnvConfig(model_name=LOCAL_QWEN)
    builder = TerminalDatasetBuilder(
        data_dir=str(data_dir), env=env, batch_size=2, group_size=4, num_eval_tasks=2
    )
    train, evals = asyncio.run(builder())
    assert len(train) == 3 and [len(train.get_batch(i)) for i in range(3)] == [2, 2, 1]
    assert sorted(b.task.id for i in range(3) for b in train.get_batch(i)) == [f"t{i}" for i in range(5)]
    assert train.get_batch(0)[0].group_size == 4
    (eval_batch,) = [evals.get_batch(i) for i in range(len(evals))]
    assert [b.task.id for b in eval_batch] == ["t0", "t1"] and eval_batch[0].group_size == 1


class TestPiHarness:
    """pi asks for each turn through the chat bridge; the cookbook's loop samples them."""

    TURNS = [
        "Let me search.\n\n" + bash_call("grep -h 'mother of Ann Lee' /corpus/*.md"),
        bash_call("echo 'Bea Lee' > /app/output.md"),
        "I wrote the answer.",
    ]

    def rollout(self, tokenizer, data_dir, rootfs, turns=TURNS, max_turns=5):
        builder = group(data_dir, rootfs, max_turns=max_turns)
        result = asyncio.run(do_group_rollout(builder, ScriptedPolicy(tokenizer, turns)))
        assert builder.message_envs == []  # cleanup closed the sandboxes
        return result

    def test_episode_verifies_answer_file(self, tokenizer, data_dir, rootfs):
        result = self.rollout(tokenizer, data_dir, rootfs)
        assert result.get_total_rewards() == [1.0, 1.0]
        trajectory = result.trajectories_G[0]
        assert len(trajectory.transitions) == 3 and trajectory.stop_reason == "completed"
        metrics = trajectory.transitions[-1].metrics
        assert (metrics["num_turns"], metrics["num_commands"], metrics["answered"]) == (3, 2, 1.0)
        assert metrics["harness_error"] == 0.0

        # pi's own system prompt and bash tool open the conversation, then the task
        prompt = tokenizer.decode(trajectory.transitions[0].ob.to_ints())
        assert "coding agent" in prompt and '"name": "bash"' in prompt and TASK.question in prompt
        # pi ran the command and sent its output back
        second_prompt = tokenizer.decode(trajectory.transitions[1].ob.to_ints())
        assert "The mother of Ann Lee is Bea Lee." in second_prompt

    def test_episode_trains_as_one_sequence(self, tokenizer, data_dir, rootfs):
        trajectory = self.rollout(tokenizer, data_dir, rootfs).trajectories_G[0]
        (datum,) = trajectory_to_data(trajectory, traj_advantage=1.0)
        mask = datum.loss_fn_inputs["mask"].to_torch()
        assert mask.sum().item() == sum(len(t.ac.tokens) for t in trajectory.transitions)

    def test_turn_cap(self, tokenizer, data_dir, rootfs):
        result = self.rollout(tokenizer, data_dir, rootfs, turns=[bash_call("ls /corpus")], max_turns=2)
        trajectory = result.trajectories_G[0]
        assert len(trajectory.transitions) == 2 and trajectory.stop_reason == "max_turns"
        assert result.get_total_rewards() == [0.0, 0.0]


def test_eval_rows_and_totals(tokenizer, data_dir, rootfs):
    """scripts/tinker/eval.py: an episode's dump row, and the eval's averages and token totals."""
    turns = [bash_call("echo 'Bea Lee' > /app/output.md"), "Done."]
    builder = group(data_dir, rootfs, group_size=1)
    trajectory = asyncio.run(do_group_rollout(builder, ScriptedPolicy(tokenizer, turns))).trajectories_G[0]
    row = tinker_eval.tinker_row(TASK, trajectory, tokenizer)
    assert (row["answer"], row["score"]["f1"], row["stop_reason"]) == ("Bea Lee\n", 1.0, "completed")
    assert (row["num_turns"], row["num_commands"], row["answered"]) == (2, 1, 1.0)
    assert TASK.question in row["input_prompt"] and row["output_response"].endswith("Done.<|im_end|>")
    # Nothing came from a prefix cache: both prompts and both turns are counted in full
    prompts = sum(t.ob.length for t in trajectory.transitions)
    sampled = sum(len(t.ac.tokens) for t in trajectory.transitions)
    assert row["tokens"] == {"prefill": prompts, "cached_prefill": 0, "sample": sampled}

    cli = EvalConfig(
        data_dir=str(data_dir), out_dir="unused", env=builder.env, price_prefill=1.0, price_sample=2.0
    )
    totals = tinker_eval.totals([row, row], cli)
    assert (totals["num_episodes"], totals["f1"], totals["answered"]) == (2, 1.0, 1.0)
    assert totals["cost_usd"] == pytest.approx(2 * (prompts * 1.0 + sampled * 2.0) / 1e6)
