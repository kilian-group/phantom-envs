#!/usr/bin/env -S uv run python
"""Evaluate a chat-completions API model (e.g. on OpenRouter) as the shell agent on a data dir.

Runs the episodes of ``launch_skyrl.sh eval shell`` with the API in place of vLLM: the prompts of
``<data_dir>/shell/validation.parquet``, the bash tool through the API's native tool calling, a ShellSim
sandbox per episode, and the same budgets (turns and input tokens from eval.yaml, tokens per turn and
the tool from envs/shell.yaml, sampling from base.yaml). As in SkyRL, an episode whose input outgrows
the budget scores 0. Writes SkyRL-style dumps, one row per episode with the whole chat in ``messages``:

    <out_dir>/model=<model slug>/dumped_evals/eval_only/{<data_source>.jsonl, aggregated_results.jsonl}

Episodes already in the dump are skipped, so rerunning the command finishes an interrupted eval.

Usage:
    uv run --no-sync python scripts/eval_api.py --data_dir data/hp500 --model qwen/qwen4-max \\
        --out_dir out__eval=hp500__shell [--limit 10] [--concurrency 32] [--reasoning off]

The API defaults to OpenRouter with OPENROUTER_API_KEY from .env; --base_url and --api_key_env point it
at any OpenAI-compatible server instead.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

from phantom_envs.runs import compact_model_slug
from phantom_envs.shell import (
    ShellEpisodeState,
    ShellProtocol,
    ShellSandbox,
    ToolCall,
    corpus_image,
    tool_error,
)

CONFIGS = Path(__file__).resolve().parent / "skyrl" / "configs"


def load_budgets() -> dict[str, Any]:
    """The SkyRL shell eval's tool, sampling and budgets, read from its configs."""
    base, shell, eval_ = (
        yaml.safe_load((CONFIGS / f).read_text()) for f in ("base.yaml", "envs/shell.yaml", "eval.yaml")
    )
    return {
        "tools": shell["generator"]["chat_template_kwargs"]["tools"],
        "max_tokens": shell["generator"]["sampling_params"]["max_generate_length"],
        "temperature": base["generator"]["sampling_params"]["temperature"],
        "top_p": base["generator"]["sampling_params"]["top_p"],
        "max_turns": eval_["generator"]["max_turns"],
        "max_input_length": eval_["generator"]["max_input_length"],
    }


def parse_arguments(text: str | None) -> dict[str, str]:
    """A native tool call's JSON arguments as text values; {} if they don't parse."""
    try:
        args = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(args, dict):
        return {}
    return {k: v if isinstance(v, str) else json.dumps(v) for k, v in args.items()}


def add_usage(total: dict[str, float], usage: Any) -> None:
    """Add one response's token counts (and OpenRouter's cost, when reported) to ``total``."""
    if usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    total["prompt_tokens"] += usage.prompt_tokens or 0
    total["completion_tokens"] += usage.completion_tokens or 0
    total["cached_tokens"] += (getattr(details, "cached_tokens", None) or 0) if details else 0
    total["cost"] += getattr(usage, "cost", None) or 0.0


def run_episode(client: OpenAI, args: argparse.Namespace, budgets: dict[str, Any], row: dict) -> dict:
    """Run one question's episode and return its dump row."""
    protocol = ShellProtocol(
        gts=row["reward_spec"]["ground_truth"]["target"],
        data_source=row["data_source"],
        max_turns=budgets["max_turns"],
    )
    sandbox = ShellSandbox(corpus_image(Path(args.data_dir) / "corpus.jsonl"))
    state = ShellEpisodeState()
    messages = list(row["prompt"])
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost": 0.0}
    extra_body = {"reasoning": {"enabled": args.reasoning == "on"}} if args.reasoning != "default" else {}
    reward, answer = 0.0, None
    while True:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=budgets["tools"],
            temperature=budgets["temperature"],
            top_p=budgets["top_p"],
            max_tokens=budgets["max_tokens"],
            extra_body=extra_body,
        )
        add_usage(usage, response.usage)
        if response.usage and response.usage.prompt_tokens > budgets["max_input_length"]:
            stop_reason = "length"  # SkyRL stops before such a call, with reward 0
            break
        choice = response.choices[0]
        message = choice.message
        messages.append(message.model_dump(exclude_none=True))
        calls = [
            ToolCall(name=c.function.name, arguments=parse_arguments(c.function.arguments))
            for c in message.tool_calls or []
        ]
        result = protocol.act_on_calls(state, calls)
        if result.done:
            answer = sandbox.read_answer()
            reward = protocol.reward(answer)
            stop_reason = choice.finish_reason
            break
        outputs = [tool_error(call) or sandbox.run(call.arguments["command"]) for call in calls]
        for tool_call, output in zip(message.tool_calls, protocol.observe(state, outputs), strict=True):
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

    extras = {k: v for k, v in row.items() if k not in ("prompt", "env_class")}
    return {
        "messages": messages,
        "score": [reward],
        "stop_reason": stop_reason,
        "env_class": row["env_class"],
        "env_extras": {**extras, "num_turns": state.turns, "answer": answer},
        "data_source": row["data_source"],
        "env_metrics": {
            "num_turns": state.turns,
            "num_commands": state.commands,
            "answered": float(bool((answer or "").strip())),
        },
        "usage": usage,
    }


def aggregate(rows: list[dict]) -> dict[str, float]:
    """SkyRL-style eval metrics over the dump rows, plus API usage."""

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    groups: dict[str, list[dict]] = {"all": rows}
    for r in rows:
        groups.setdefault(r["data_source"], []).append(r)
    metrics = {}
    for name, group in groups.items():
        scores = [sum(r["score"]) for r in group]
        metrics[f"eval/{name}/avg_score"] = mean(scores)
        metrics[f"eval/{name}/pass_at_1"] = mean([float(s > 0) for s in scores])
    for key in rows[0]["env_metrics"]:
        metrics[f"eval/all/environment/{key}"] = mean([r["env_metrics"][key] for r in rows])
    metrics["eval/all/length_stops"] = mean([float(r["stop_reason"] == "length") for r in rows])
    for key in rows[0]["usage"]:
        metrics[f"eval/all/usage/{key}"] = sum(r["usage"][key] for r in rows)
    return metrics


def main(args: argparse.Namespace) -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    budgets = load_budgets()
    client = OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env], max_retries=8, timeout=600)

    rows = pq.read_table(Path(args.data_dir) / "shell" / "validation.parquet").to_pylist()[: args.limit]
    out = (
        Path(args.out_dir)
        / f"model={args.name or compact_model_slug(args.model)}"
        / "dumped_evals"
        / "eval_only"
    )
    out.mkdir(parents=True, exist_ok=True)
    dump = out / f"{rows[0]['data_source']}.jsonl"
    done = [json.loads(line) for line in dump.open()] if dump.exists() else []
    done_ids = {r["env_extras"]["extra_info"]["index"] for r in done}
    todo = [r for r in rows if r["extra_info"]["index"] not in done_ids]
    print(
        f"{args.model} on {args.data_dir}: {len(todo)} of {len(rows)} episodes to run, budgets {budgets | {'tools': '[bash]'}}",
        flush=True,
    )

    corpus_image(Path(args.data_dir) / "corpus.jsonl")  # load once before the threads start
    failures = 0
    with ThreadPoolExecutor(args.concurrency) as pool, dump.open("a") as f:
        futures = [pool.submit(run_episode, client, args, budgets, row) for row in todo]
        for i, future in enumerate(as_completed(futures), 1):  # rows are written from this thread only
            if future.cancelled():
                continue
            try:
                result = future.result()
            except Exception as e:  # an API error after retries: the episode stays undone for a rerun
                failures += 1
                print(f"episode failed: {e!r}", flush=True)
                # A bad key, used-up credits (402) or a spent key limit (403) fails every episode, each
                # after paying for its first turns
                if isinstance(e, APIStatusError) and e.status_code in (401, 402, 403):
                    print("account error: cancelling the episodes not yet started", flush=True)
                    for pending in futures:
                        pending.cancel()
                continue
            f.write(json.dumps(result) + "\n")
            f.flush()
            done.append(result)
            if i % 25 == 0 or i == len(todo):
                m = aggregate(done)
                print(
                    f"{i}/{len(todo)}: avg score {m['eval/all/avg_score']:.3f}, cost ${m['eval/all/usage/cost']:.2f}",
                    flush=True,
                )

    if failures:
        print(
            f"{failures} episodes failed; rerun the same command to finish them and the cancelled ones",
            flush=True,
        )
    if not done:
        return
    metrics = aggregate(done)
    (out / "aggregated_results.jsonl").write_text(json.dumps(metrics) + "\n")
    print(json.dumps(metrics, indent=1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_dir", required=True, help="Data dir with corpus.jsonl and shell/validation.parquet"
    )
    parser.add_argument("--model", required=True, help="API model id, e.g. qwen/qwen4-max")
    parser.add_argument("--out_dir", required=True, help="Eval output dir, e.g. out__eval=hp500__shell")
    parser.add_argument("--name", default=None, help="Model dir name (default: a slug of --model)")
    parser.add_argument("--base_url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible API")
    parser.add_argument(
        "--api_key_env", default="OPENROUTER_API_KEY", help="Env var (or .env key) of the API key"
    )
    parser.add_argument(
        "--reasoning",
        choices=["off", "on", "default"],
        default="off",
        help="OpenRouter's reasoning switch (off matches the 9B's non-thinking evals; default sends nothing)",
    )
    parser.add_argument("--concurrency", type=int, default=32, help="Episodes run at once")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke tests)")
    main(parser.parse_args())
