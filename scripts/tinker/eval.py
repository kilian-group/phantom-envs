#!/usr/bin/env -S uv run python
"""Evaluate a model as the terminal agent through the Tinker API, on a data dir's validation questions.

One episode per question, sampled from a base model or a Tinker checkpoint. The agent answers with
bash over the corpus files inside the pi harness, and the verifier scores `/app/output.md`. Writes
one row per episode, with the whole conversation and its token counts (see phantom_envs.terminal.evals):

    <out_dir>/model=<name>/{<data_source>.jsonl, aggregated_results.jsonl}

Episodes already in the dump are skipped, so rerunning the command finishes an interrupted eval.

A model's settings come from the same config files as training (scripts/tinker/configs/models/);
`key=value` arguments override them, and training-only settings are ignored.

Usage:
    # Inkling 2 Small with pi in the bwrap sandbox (bills TINKER_API_KEY's account; both are in .env)
    uv run --extra tinker --env-file .env python scripts/tinker/eval.py \\
        --config scripts/tinker/configs/models/Inkling-2-Small.yaml data_dir=data/hp500 \\
        out_dir=out__eval=hp500__pi limit=10 budget_usd=5

    # A trained checkpoint
    ... checkpoint=tinker://<run>/sampler_weights/<name> name=<run>__step40

    # Another episode setting
    ... env.max_turns=10

Gotchas:
- `budget_usd` stops starting episodes once the estimated spend reaches it; running ones finish.
- The cost is an estimate from the token counts and the configured prices, not Tinker's bill.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import Trajectory
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

from phantom_envs._types import Task
from phantom_envs.integrations.tinker.cli import EvalConfig, load_config
from phantom_envs.integrations.tinker.dataset import TerminalEnvGroupBuilder
from phantom_envs.integrations.tinker.harness import agent_setup
from phantom_envs.runs import compact_model_slug
from phantom_envs.scoring import METRICS
from phantom_envs.tasks import load_tasks
from phantom_envs.terminal.evals import EpisodeDump, aggregate, episode_row

# Token counts of an episode, by how Tinker bills them
TOKEN_KINDS: tuple[str, ...] = ("prefill", "cached_prefill", "sample")


def count_tokens(trajectory: Trajectory) -> dict[str, int]:
    """Count an episode's tokens by how Tinker bills them.

    Args:
        trajectory: The episode. Each transition is one model call.

    Returns:
        The prompt tokens outside the prefix cache, those read from it, and the sampled tokens.
    """
    counts = dict.fromkeys(TOKEN_KINDS, 0)
    for transition in trajectory.transitions:
        cached = getattr(transition.ac, "prompt_cache_hit_tokens", None) or 0
        counts["prefill"] += transition.ob.length - cached
        counts["cached_prefill"] += cached
        counts["sample"] += len(transition.ac.tokens)
    return counts


def cost_usd(tokens: dict[str, float], cli: EvalConfig) -> float:
    """Estimate what token counts cost at the configured prices."""
    prices = {
        "prefill": cli.price_prefill,
        "cached_prefill": cli.price_cached_prefill,
        "sample": cli.price_sample,
    }
    return sum(tokens[kind] * prices[kind] for kind in TOKEN_KINDS) / 1e6


def tinker_row(task: Task, trajectory: Trajectory, tokenizer: Tokenizer) -> dict[str, Any]:
    """Build an episode's dump row.

    Args:
        task: The question.
        trajectory: The episode.
        tokenizer: The model's tokenizer, to write the conversation as text.

    Returns:
        The row: the task, the answer and its scores, the env's metrics, the token counts, and the
        conversation split into its first prompt and everything after it (as in SkyRL's dumps).
    """
    last = trajectory.transitions[-1]
    metrics = last.metrics
    row = episode_row(
        task,
        answer=last.logs.get("answer"),
        scores={metric: metrics.get(metric, 0.0) for metric in METRICS},
        stop_reason=trajectory.stop_reason or "unknown",
        num_turns=int(metrics.get("num_turns", len(trajectory.transitions))),
        num_commands=int(metrics.get("num_commands", 0)),
    )
    first_prompt = trajectory.transitions[0].ob.to_ints()
    # The last prompt holds every earlier turn and tool output; the last turn follows it
    conversation = last.ob.to_ints() + list(last.ac.tokens)
    return row | {
        "metrics": dict(metrics),
        "tokens": count_tokens(trajectory),
        "input_prompt": tokenizer.decode(first_prompt),
        "output_response": tokenizer.decode(conversation[len(first_prompt) :]),
    }


def totals(rows: list[dict[str, Any]], cli: EvalConfig) -> dict[str, float]:
    """Aggregate the rows, with their total tokens and estimated cost."""
    tokens = {kind: sum(row["tokens"][kind] for row in rows) for kind in TOKEN_KINDS}
    results = aggregate(rows) | {f"tokens/{kind}": tokens[kind] for kind in TOKEN_KINDS}
    results["cost_usd"] = cost_usd(tokens, cli)
    return results


async def main(cli: EvalConfig) -> None:
    """Run the eval's missing episodes and write the dump and the aggregated results."""
    env = cli.env
    tasks = load_tasks(cli.data_dir, "validation")[: cli.limit]
    # One agent for every episode, built and checked before the first one
    agent = agent_setup(env, str(Path(cli.data_dir) / "corpus.jsonl"))
    await asyncio.to_thread(agent.wait_for_servers)
    dump = EpisodeDump(cli.out_dir, cli.name or compact_model_slug(env.model_name), tasks[0].data_source)
    todo = dump.todo(tasks)
    print(f"{env.model_name} on {cli.data_dir}: {len(todo)} of {len(tasks)} episodes to run", flush=True)

    service = tinker.ServiceClient(base_url=cli.base_url)
    sampling_client = service.create_sampling_client(
        model_path=cli.checkpoint, base_model=None if cli.checkpoint else env.model_name
    )
    policy = TinkerTokenCompleter(sampling_client, max_tokens=env.max_tokens, temperature=cli.temperature)
    tokenizer = get_tokenizer(env.model_name)
    semaphore = asyncio.Semaphore(cli.concurrency)
    spent = {"usd": totals(dump.rows, cli)["cost_usd"] if dump.rows else 0.0, "skipped": 0, "failed": 0}

    async def run(task: Task) -> None:
        async with semaphore:
            if cli.budget_usd is not None and spent["usd"] >= cli.budget_usd:
                spent["skipped"] += 1
                return
            try:
                group = await do_group_rollout(TerminalEnvGroupBuilder(task, agent, env, 1), policy)
            except Exception as error:  # noqa: BLE001 (the episode stays undone for a rerun)
                spent["failed"] += 1
                print(f"episode {task.id} failed: {type(error).__name__}: {error}", flush=True)
                return
        row = tinker_row(task, group.trajectories_G[0], tokenizer)
        dump.add(row)
        spent["usd"] += cost_usd(row["tokens"], cli)
        if len(dump.rows) % 25 == 0:
            f1 = aggregate(dump.rows)["f1"]
            print(f"{len(dump.rows)}/{len(tasks)}: F1 {f1:.3f}, ~${spent['usd']:.2f}", flush=True)

    await asyncio.gather(*(run(task) for task in todo))

    if spent["skipped"] or spent["failed"]:
        print(
            f"{spent['skipped']} episodes not started (budget) and {spent['failed']} failed; "
            "rerun the same command to finish them",
            flush=True,
        )
    if dump.rows:
        results = totals(dump.rows, cli)
        dump.write_results(results)
        print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model as the terminal agent through the Tinker API."
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="YAML",
        help="Model config; repeat to stack configs, later files win "
        "(e.g. scripts/tinker/configs/models/Inkling-2-Small.yaml).",
    )
    args, overrides = parser.parse_known_args()
    asyncio.run(main(load_config(EvalConfig, args.config, overrides)))
