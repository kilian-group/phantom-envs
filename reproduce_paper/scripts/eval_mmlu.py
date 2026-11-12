#!/usr/bin/env -S uv run python
"""MMLU evaluation — probes general reasoning/knowledge with a format-immune metric.

Thin wrapper over EleutherAI's lm-evaluation-harness (``lm_eval.simple_evaluate``) with the
vLLM backend, pinning the flags and normalizing output into the same JSON layout as the other
NLU-preservation evals.

MMLU here is *loglikelihood-scored*: the harness compares the log-probability of each answer
letter rather than generating text. Nothing is sampled, so the score is deterministic and
temperature does not apply — and, crucially, it cannot be distorted by search-trained models
falling back on ``<search>`` formatting. That makes it the load-bearing "knowledge preserved"
number.

Chat templating is off by default so numbers match published 5-shot MMLU. Only the
within-model-family comparison (base vs synthetic vs real) is meaningful; absolute values
shift with templating, so the flag must be held fixed across the checkpoints being compared.

``--log_samples`` additionally writes a slim ``<key>_samples.jsonl`` holding one row per
document (task, doc_id, acc, doc_hash). That is what lets the analysis scripts compute
*paired* base-vs-trained differences: base and trained score the identical documents, so
differencing per document cancels the across-document variance and shrinks the standard
error several-fold versus treating the two aggregate accuracies as independent.

Usage:
    uv run --no-sync python reproduce_paper/scripts/eval_mmlu.py \\
        --model_path Qwen/Qwen2.5-3B-Instruct \\
        --out_dir out__nlu/mmlu --log_samples
"""

import argparse
import json
import logging
from pathlib import Path

from lm_eval import simple_evaluate

from phantom_envs.runs import compact_model_slug

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_model_args(args: argparse.Namespace) -> str:
    """Assemble the comma-separated model_args string for the harness' vLLM backend."""
    return ",".join(
        [
            f"pretrained={args.model_path}",
            "dtype=bfloat16",
            f"max_model_len={args.max_model_len}",
            f"gpu_memory_utilization={args.gpu_memory_utilization}",
            f"tensor_parallel_size={args.tensor_parallel_size}",
        ]
    )


def slim_samples(samples: dict[str, list[dict]]) -> list[dict]:
    """Reduce the harness' per-document logs to the fields needed for paired differences.

    The full payload carries every prompt and response; only the per-document score is
    needed downstream. Rows are sorted by ``(task, doc_id)`` so that two models' files line
    up positionally, and ``doc_hash`` lets a consumer verify that they really do.
    """
    return sorted(
        (
            {
                "task": task,
                "doc_id": example["doc_id"],
                "acc": example.get("acc"),
                "doc_hash": example.get("doc_hash"),
            }
            for task, examples in samples.items()
            for example in examples
        ),
        key=lambda row: (row["task"], row["doc_id"]),
    )


def main(args: argparse.Namespace) -> None:
    tasks = args.tasks.split(",")
    results = simple_evaluate(
        model="vllm",
        model_args=build_model_args(args),
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.apply_chat_template,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        log_samples=args.log_samples,
    )

    # Harness reports metrics as "<metric>,<filter>" keys, e.g. "acc,none".
    per_task = {
        task: {
            "acc": metrics.get("acc,none"),
            "acc_stderr": metrics.get("acc_stderr,none"),
        }
        for task, metrics in results["results"].items()
    }
    summary = {
        "eval": "mmlu",
        "model_path": args.model_path,
        "tasks": tasks,
        "num_fewshot": args.num_fewshot,
        "apply_chat_template": args.apply_chat_template,
        "seed": args.seed,
        "scoring": "loglikelihood (deterministic, no sampling)",
        "results": {task: per_task[task] for task in tasks if task in per_task},
        "results_all": per_task,
    }
    logger.info(json.dumps(summary["results"], indent=2))

    # Suffix keeps the two prompt settings from overwriting each other.
    key = f"model={compact_model_slug(args.model_path)}"
    if args.apply_chat_template:
        key += "_chat"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{key}.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote results to {out_dir}/{key}.json")

    if args.log_samples:
        rows = slim_samples(results.get("samples", {}))
        samples_path = out_dir / f"{key}_samples.jsonl"
        with samples_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        logger.info(f"Wrote {len(rows)} per-document scores to {samples_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True, help="HF repo id or local checkpoint dir.")
    parser.add_argument("--out_dir", type=str, default="out__nlu/mmlu", help="Directory for result JSON.")
    parser.add_argument("--tasks", type=str, default="mmlu", help="Comma-separated lm-eval task names.")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Few-shot exemplars (5 is standard MMLU).")
    parser.add_argument("--batch_size", type=str, default="auto", help="lm-eval batch size.")
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply the chat template (off by default to match published MMLU numbers). "
        "Must be held fixed across compared checkpoints.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Also write <key>_samples.jsonl with per-document scores, enabling paired "
        "base-vs-trained differences in the analysis scripts.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for harness RNGs.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM context length.")
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory fraction."
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size.")
    main(parser.parse_args())
