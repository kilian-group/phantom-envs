#!/usr/bin/env -S uv run python
"""Evaluate a chat-completions API model (e.g. on OpenRouter) as the terminal agent, inside the pi harness.

One episode per validation question: pi runs in a bwrap sandbox over the data dir's corpus (or with a
search tool over it), with the budgets of the SkyRL terminal evals, and its model calls are answered by
the API. The API key stays on
the host; the sandbox has no network. Writes one row per episode, with the whole conversation and the
API's token counts (see phantom_envs.terminal.evals):

    <out_dir>/model=<name>/{<data_source>.jsonl, aggregated_results.jsonl}

Episodes already in the dump are skipped, so rerunning the command finishes an interrupted eval.

Usage:
    uv run --extra tinker --env-file .env python scripts/eval_api.py --data_dir data/hp500 \\
        --model qwen/qwen4-max --reasoning off --out_dir out__eval=hp500__pi \\
        [--limit 10] [--concurrency 32]

    # With the search tool: start the retrieval MCP server on a Unix socket, then point the eval at it.
    # The eval waits for the server to load its index before the first episode.
    uv run --extra skyrl --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.mcp_server \\
        --retriever qwen3-4b --data_dir data/hp500 port=0 socket=/tmp/retriever.sock &
    uv run --extra tinker --env-file .env python scripts/eval_api.py --data_dir data/hp500 \\
        --model qwen/qwen4-max --out_dir out__eval=hp500__pi_search \\
        --tools bash+search --mcp_socket /tmp/retriever.sock

The API defaults to OpenRouter with OPENROUTER_API_KEY; --base_url and --api_key_env point it at any
OpenAI-compatible server. SANDBOX_ROOTFS names the sandbox image (scripts/sandbox/build_rootfs.sh).

Gotchas:
- As in SkyRL, an episode whose prompt outgrows --max_input_length scores 0.
- When the API refuses for good (a spent quota, a bad key), the episodes not yet started are left
  for a rerun.
"""

import argparse
import asyncio
import json
import os
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import aiohttp

from phantom_envs._types import Task
from phantom_envs.runs import compact_model_slug
from phantom_envs.scoring import METRICS
from phantom_envs.tasks import load_tasks
from phantom_envs.terminal.agent import AgentSetup
from phantom_envs.terminal.bridge import BridgedEpisode
from phantom_envs.terminal.evals import EpisodeDump, aggregate, episode_row
from phantom_envs.terminal.harness import SCORED_STOP_REASONS, is_mcp_tool
from phantom_envs.terminal.task import TOOLSETS, verify

# Statuses worth retrying: rate limits, timeouts and server errors
RETRY_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
USAGE_KEYS: tuple[str, ...] = (
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "cost",
)


class ApiRefused(Exception):
    """The API won't serve this run any more: a spent quota, a bad key, or retries used up."""


class RateLimiter:
    """Spaces request starts evenly, at most `per_minute` a minute."""

    def __init__(self, per_minute: float) -> None:
        self.interval = 60.0 / per_minute
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        """Sleep until this request may start."""
        async with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        await asyncio.sleep(start - now)


class ChatApi:
    """A chat-completions API client with a rate limit and retries, counting requests."""

    def __init__(self, session: aiohttp.ClientSession, args: argparse.Namespace) -> None:
        self.session = session
        self.args = args
        self.limiter = RateLimiter(args.requests_per_minute)
        self.requests = 0
        self.refused: str | None = None

    async def complete(self, messages: list[dict], tools: list[dict] | None) -> dict[str, Any]:
        """Ask the model for the next turn.

        Args:
            messages: The conversation so far, in OpenAI's form.
            tools: The tools the harness offers.

        Returns:
            The API's response body.

        Raises:
            ApiRefused: If the API refuses for good; `self.refused` then says why.
        """
        args = self.args
        body = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens}
        body |= {"temperature": args.temperature, "usage": {"include": True}}
        if tools:
            body["tools"] = tools
        if args.reasoning != "default":
            body["reasoning"] = {"enabled": args.reasoning == "on"}
        headers = {"Authorization": f"Bearer {os.environ[args.api_key_env]}"}
        error = "no attempt made"
        for attempt in range(args.max_retries + 1):
            if self.refused:
                raise ApiRefused(self.refused)
            await self.limiter.wait()
            self.requests += 1
            try:
                async with self.session.post(
                    f"{args.base_url}/chat/completions", json=body, headers=headers
                ) as r:
                    text = await r.text()
                    status = r.status
            except (TimeoutError, aiohttp.ClientError) as e:
                status, text = 0, f"{type(e).__name__}: {e}"
            if status == 200:
                response = json.loads(text)
                # Providers also report failures inside a 200 response
                if response.get("choices") and not response.get("error"):
                    return response
            error = f"HTTP {status}: {text[:300]}"
            daily_quota = status == 429 and "per-day" in text
            if daily_quota or (status and status not in RETRY_STATUSES and status != 200):
                self.refused = error
                raise ApiRefused(error)
            await asyncio.sleep(min(60.0, 2.0**attempt))
        raise ApiRefused(f"retries used up; last error: {error}")


def add_usage(total: dict[str, float], usage: dict[str, Any] | None) -> None:
    """Add one response's token counts (and cost, when the API reports it) to `total`."""
    usage = usage or {}
    total["prompt_tokens"] += usage.get("prompt_tokens") or 0
    total["completion_tokens"] += usage.get("completion_tokens") or 0
    total["cached_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    total["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    total["cost"] += usage.get("cost") or 0.0


async def run_episode(
    api: ChatApi, agent: AgentSetup, executor: Executor, args: argparse.Namespace, task: Task
) -> dict:
    """Run one question's episode and return its dump row.

    The conversation sent to the API is kept here, with each reply exactly as the API returned it
    (reasoning included); pi's requests add only its tool results.
    """
    episode = BridgedEpisode(
        agent,
        context_window=args.max_input_length + args.max_tokens,
        max_tokens=args.max_tokens,
        executor=executor,
    )
    usage = dict.fromkeys(USAGE_KEYS, 0.0)
    stop_reason = "stop"
    try:
        request = await episode.start(task.question)
        messages, tools = list(request["messages"]), request.get("tools")
        while True:
            response = await api.complete(messages, tools)
            add_usage(usage, response.get("usage"))
            if (response.get("usage") or {}).get("prompt_tokens", 0) > args.max_input_length:
                stop_reason = "context_length"  # SkyRL stops before such a call, with reward 0
                break
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                stop_reason = "length"
            messages.append(choice["message"])
            tool_results = await episode.step(choice["message"])
            if tool_results is None:
                break
            messages += tool_results
        run, answer = await episode.finish()
    finally:
        await episode.close()
    if run is not None and run.stop_reason not in SCORED_STOP_REASONS:
        stop_reason = run.stop_reason

    scored = stop_reason in SCORED_STOP_REASONS
    scores = verify(answer, task.answers, task.data_source) if scored else dict.fromkeys(METRICS, 0.0)
    turns = [m for m in messages if m["role"] == "assistant"]
    calls = [call["function"]["name"] for m in turns for call in m.get("tool_calls") or []]
    row = episode_row(task, answer, scores, stop_reason, len(turns), len(calls))
    searches = sum(is_mcp_tool(name) for name in calls)
    return row | {"num_searches": searches, "usage": usage, "messages": messages}


async def main(args: argparse.Namespace) -> None:
    """Run the eval's missing episodes and write the dump and the aggregated results."""
    tasks = load_tasks(args.data_dir, "validation")[: args.limit]
    # One agent for every episode, built and checked before the first one
    agent = AgentSetup(
        corpus_path=str(Path(args.data_dir) / "corpus.jsonl"),
        rootfs=args.rootfs,
        tools=args.tools,
        mcp_socket=args.mcp_socket,
        max_turns=args.max_turns,
        command_timeout=args.command_timeout,
        memory_gib=args.memory_gib,
    )
    await asyncio.to_thread(agent.wait_for_servers)
    dump = EpisodeDump(args.out_dir, args.name or compact_model_slug(args.model), tasks[0].data_source)
    todo = dump.todo(tasks)
    print(f"{args.model} on {args.data_dir}: {len(todo)} of {len(tasks)} episodes to run", flush=True)

    failures: list[str] = []
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        api = ChatApi(session, args)
        # Each episode holds a thread while pi runs, and another for its sandbox
        with ThreadPoolExecutor(max_workers=2 * args.concurrency) as executor:

            async def run(task: Task) -> None:
                async with semaphore:
                    if api.refused:
                        return
                    try:
                        row = await run_episode(api, agent, executor, args, task)
                    except Exception as error:  # noqa: BLE001 (the episode stays undone for a rerun)
                        failures.append(f"{task.id}: {type(error).__name__}: {str(error)[:200]}")
                        return
                dump.add(row)
                if len(dump.rows) % 10 == 0 or len(dump.rows) == len(tasks):
                    f1 = aggregate(dump.rows)["f1"]
                    print(f"{len(dump.rows)}/{len(tasks)}: F1 {f1:.3f}, {api.requests} requests", flush=True)

            await asyncio.gather(*(run(task) for task in todo))

    if api.refused:
        print(f"the API refused: {api.refused}", flush=True)
    if failures or api.refused:
        print(
            f"{len(tasks) - len(dump.rows)} episodes left; rerun the same command to finish them", flush=True
        )
        for failure in failures[:5]:
            print(f"  failed {failure}", flush=True)
    if dump.rows:
        results = aggregate(dump.rows)
        results |= {f"usage/{key}": sum(row["usage"][key] for row in dump.rows) for key in USAGE_KEYS}
        dump.write_results(results)
        print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_dir", required=True, help="Data dir with corpus.jsonl and tasks/validation.jsonl"
    )
    parser.add_argument(
        "--model", required=True, help="API model id, e.g. thinkingmachines/inkling-2-small:free"
    )
    parser.add_argument("--out_dir", required=True, help="Eval output dir, e.g. out__eval=hp500__pi")
    parser.add_argument("--name", default=None, help="Model dir name (default: a slug of --model)")
    parser.add_argument("--base_url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible API")
    parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY", help="Env var that holds the API key")
    parser.add_argument(
        "--rootfs", default=os.environ.get("SANDBOX_ROOTFS", ""), help="The bwrap sandbox's image"
    )
    parser.add_argument(
        "--tools",
        choices=TOOLSETS,
        default="bash",
        help="The agent's tools: bash over the corpus as files, or a search tool (needs --mcp_socket)",
    )
    parser.add_argument("--mcp_socket", default="", help="Unix socket of the retrieval MCP server")
    parser.add_argument(
        "--reasoning",
        choices=["off", "on", "default"],
        default="default",
        help="OpenRouter's reasoning switch (default sends nothing: the provider decides)",
    )
    # Budgets of the SkyRL terminal evals (eval.yaml, envs/shellsim.yaml, base.yaml)
    parser.add_argument("--max_turns", type=int, default=20, help="Most turns the agent takes")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Tokens per turn, reasoning included")
    parser.add_argument("--max_input_length", type=int, default=32768, help="Longest prompt that is scored")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--command_timeout", type=float, default=60.0, help="Seconds a command may run")
    parser.add_argument("--memory_gib", type=int, default=4, help="Address-space limit of each command")
    parser.add_argument("--concurrency", type=int, default=16, help="Episodes run at once")
    parser.add_argument("--requests_per_minute", type=float, default=20, help="API rate limit to keep")
    parser.add_argument("--request_timeout", type=float, default=600.0, help="Seconds an API call may take")
    parser.add_argument("--max_retries", type=int, default=8, help="Retries of a rate-limited or failed call")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    asyncio.run(main(parser.parse_args()))
