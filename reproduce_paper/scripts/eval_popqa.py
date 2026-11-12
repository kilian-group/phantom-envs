#!/usr/bin/env -S uv run python
"""Closed-book PopQA evaluation — probes parametric world knowledge without retrieval.

Tests whether RL fine-tuning for search erodes the factual knowledge a model already has.
Follows the PopQA authors' setup (Mallen et al., arXiv:2212.10511, and their run_model.py):
a ``Q: <question> A:`` completion preceded by 15 few-shot exemplars sampled one per
relationship type other than the test question's. Run as a raw completion by default — no
chat template — so the instruct/RL persona is never invoked.

The few-shot context is what makes the comparison fair. Zero-shot, these models answer at very
different lengths (~7 words for EM-reward-trained vs ~23 for F1-reward-trained), and every
containment metric is length-sensitive, so the scores partly measure verbosity rather than
knowledge. Exemplars pin the answer format for all models alike.

Search-trained models may still emit ``<search>`` here. That is format lock-in, not knowledge
loss, so ``search_rate``, ``answer_tag_rate`` and mean answer length are reported alongside the
scores to keep the two attributable.

Three metrics are recorded: ``em``, ``subem`` (both from phantom_envs.scoring, matching the
Search-R1/PopQA convention of raw substring containment) and ``subem_boundary``, a local
robustness check that requires whole-word matches.

Sampling matches ``eval_sampling_params`` in the training configs (temperature 1.0, top_p 1.0,
max 500 tokens) so numbers are comparable with the paper's evaluation runs.

Usage:
    uv run --no-sync python reproduce_paper/scripts/eval_popqa.py \\
        --model_path Qwen/Qwen2.5-3B-Instruct \\
        --out_dir out__nlu/popqa
"""

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from vllm import LLM, SamplingParams

from phantom_envs.protocol import extract_answer
from phantom_envs.runs import compact_model_slug
from phantom_envs.scoring import SearchR1Scorer, mean_stderr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Verbatim from the PopQA authors' run_model.py (AlexTMallen/adaptive-retrieval): a single
# line, no newline before "A:". Few-shot exemplars are "Q: {q} A: {obj}" joined by "\n\n".
PROMPT_TEMPLATE = "Q: {question} A:"
FEWSHOT_SEPARATOR = "\n\n"

# Not "\n": a hard newline stop scores an empty string whenever a model opens with a
# newline. We generate freely and recover the answer in parse_response() instead.
STOP_STRINGS = ["</answer>", "Q:"]


def compute_score_subem_boundary(preds: list[str], golds: list[str]) -> float:
    """subEM requiring the gold to appear as a contiguous run of whole words.

    Kept local to this script rather than added to phantom_envs.scoring: it is a
    robustness check for this eval, not a reward function, and the boundary rule is
    fragile enough that it should not be reachable from training configs.

    Plain subEM tests raw substring containment, so PopQA's abbreviated golds ("pol",
    and even bare "w"/"d") fire inside unrelated words, and a gold normalizing to ""
    ("The A" — all articles) matches everything. Those hits grow with prediction length.

    Known residue: rejects inflections, e.g. gold "disaster film" vs "disaster films".
    """
    pred_words = [SearchR1Scorer.normalize(p).split() for p in preds]
    for gold in golds:
        gold_words = SearchR1Scorer.normalize(gold).split()
        if not gold_words:
            continue
        for words in pred_words:
            for i in range(len(words) - len(gold_words) + 1):
                if words[i : i + len(gold_words)] == gold_words:
                    return 1.0
    return 0.0


def load_popqa_metadata(tsv_path: Path) -> dict[str, dict]:
    """Map question text to PopQA metadata (relationship type, object, subject popularity).

    The Search-R1 parquet drops PopQA's ``prop_id``/``obj`` columns, but stratified few-shot
    sampling needs the relationship type and the exemplar answers. Joining on question text
    against the authors' released popQA.tsv recovers them exactly.
    """
    with tsv_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    meta = {
        row["question"]: {"prop_id": row["prop_id"], "obj": row["obj"], "s_pop": row["s_pop"]} for row in rows
    }
    logger.info(f"Loaded PopQA metadata for {len(meta):,} questions from {tsv_path}")
    return meta


def build_fewshot_context(question: str, meta: dict[str, dict], num_fewshot: int, seed: int) -> str:
    """Build a stratified few-shot prefix for one question, following Mallen et al.

    For every relationship type *other* than the test question's, sample
    ``num_fewshot // (n_types - 1)`` exemplars. Excluding the question's own type is what
    stops the context from leaking the answer pattern for that template, and it guarantees
    the test question can never appear as its own exemplar.
    """
    by_prop: dict[str, list[str]] = defaultdict(list)
    for q, m in meta.items():
        by_prop[m["prop_id"]].append(q)

    own_prop = meta.get(question, {}).get("prop_id")
    other_props = sorted(p for p in by_prop if p != own_prop)
    per_prop = max(1, num_fewshot // len(other_props))
    # Seed per question so the context is reproducible and independent of evaluation order.
    rng = random.Random(f"{seed}:{question}")

    exemplars = []
    for prop in other_props:
        for q in rng.sample(by_prop[prop], min(per_prop, len(by_prop[prop]))):
            exemplars.append(f"{PROMPT_TEMPLATE.format(question=q)} {meta[q]['obj']}")
    rng.shuffle(exemplars)
    return FEWSHOT_SEPARATOR.join(exemplars) + FEWSHOT_SEPARATOR


def load_questions(parquet_path: Path, data_source: str, n: int, seed: int) -> list[dict]:
    """Load and subsample questions for one data source.

    Args:
        parquet_path: Search-R1 validation parquet.
        data_source: Value of the ``data_source`` column to keep (e.g. "searchR1_popqa").
        n: Number of questions to sample; <= 0 keeps all of them.
        seed: RNG seed for subsampling.

    Returns:
        List of dicts with "question" and "golds" (list of gold answer strings).
    """
    table = pq.read_table(parquet_path, columns=["data_source", "reward_spec", "extra_info"])
    rows = table.to_pylist()
    items = [
        {
            "question": row["extra_info"]["question"],
            "golds": list(row["reward_spec"]["ground_truth"]["target"]),
        }
        for row in rows
        if row["data_source"] == data_source
    ]
    logger.info(f"Loaded {len(items)} questions for {data_source}")
    if 0 < n < len(items):
        items = random.Random(seed).sample(items, n)
        logger.info(f"Subsampled to {len(items)} questions (seed={seed})")
    return items


def parse_response(raw: str) -> str:
    """Recover the predicted answer from a raw completion.

    Prefers ``<answer>...</answer>`` content when the model emits it (search-trained models
    often do), otherwise falls back to the first non-empty line of the completion.
    """
    tagged = extract_answer(raw)
    if tagged is not None:
        return tagged
    for line in raw.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def build_prompts(
    items: list[dict],
    tokenizer,
    chat_template: bool,
    meta: dict[str, dict] | None,
    num_fewshot: int,
    seed: int,
) -> list[str]:
    """Format questions, with optional stratified few-shot prefix and chat template.

    Few-shot matters beyond faithfulness to the paper: zero-shot, the models answer at wildly
    different lengths (7 vs 23 words), and every containment metric is length-sensitive, so
    the comparison measures verbosity as much as knowledge. Exemplars pin the answer format
    for all models alike.

    Raw completion never invokes the instruct/RL persona, so it is the default. But these are
    instruct models, and some degrade outside their chat format, so the templated variant is
    available as a comparison run.
    """
    prompts = []
    for item in items:
        prompt = PROMPT_TEMPLATE.format(question=item["question"])
        if meta is not None and num_fewshot > 0:
            prompt = build_fewshot_context(item["question"], meta, num_fewshot, seed) + prompt
        prompts.append(prompt)
    if not chat_template:
        return prompts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]


def main(args: argparse.Namespace) -> None:
    items = load_questions(Path(args.data_path), args.data_source, args.n, args.seed)
    meta = load_popqa_metadata(Path(args.popqa_tsv)) if args.num_fewshot > 0 else None
    if meta is not None:
        missing = sum(1 for i in items if i["question"] not in meta)
        if missing:
            logger.warning(f"{missing} questions absent from popQA.tsv; they get no own-type exclusion")

    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        seed=args.seed,
    )
    prompts = build_prompts(items, llm.get_tokenizer(), args.chat_template, meta, args.num_fewshot, args.seed)
    logger.info(f"Example prompt:\n{prompts[0][:1200]}")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=STOP_STRINGS,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)

    records, em_scores, subem_scores, bnd_scores = [], [], [], []
    searched, unanswered, tagged, pred_words = [], [], [], []
    for item, prompt, output in zip(items, prompts, outputs, strict=True):
        raw = output.outputs[0].text
        pred = parse_response(raw)
        preds = SearchR1Scorer.parse_answer(pred)
        em = SearchR1Scorer.score(preds, item["golds"], "em")
        subem = SearchR1Scorer.score(preds, item["golds"], "subem")
        subem_bnd = compute_score_subem_boundary(preds, item["golds"])
        em_scores.append(em)
        subem_scores.append(subem)
        bnd_scores.append(subem_bnd)
        # Diagnostics separating format lock-in from knowledge loss.
        searched.append(float("<search>" in raw))
        unanswered.append(float(not pred))
        tagged.append(float("<answer>" in raw))
        # EM punishes verbose answers and subEM rewards them, so answer length is a
        # confound whenever a rambling base model is compared against terse RL checkpoints.
        pred_words.append(float(len(pred.split())))
        records.append(
            {
                "question": item["question"],
                "golds": item["golds"],
                "prompt": prompt,
                "output_response": raw,
                "prediction": pred,
                "em": em,
                "subem": subem,
                "subem_boundary": subem_bnd,
                "pred_words": pred_words[-1],
                # PopQA stratifies by subject popularity; kept so head vs long-tail
                # knowledge can be separated at analysis time.
                "s_pop": (meta or {}).get(item["question"], {}).get("s_pop"),
            }
        )

    em_mean, em_se = mean_stderr(em_scores, binary=True)
    subem_mean, subem_se = mean_stderr(subem_scores, binary=True)
    bnd_mean, bnd_se = mean_stderr(bnd_scores, binary=True)
    search_rate, _ = mean_stderr(searched, binary=True)
    no_answer_rate, _ = mean_stderr(unanswered, binary=True)
    answer_tag_rate, _ = mean_stderr(tagged, binary=True)
    mean_pred_words, pred_words_se = mean_stderr(pred_words, binary=False)

    summary = {
        "eval": "popqa_closed_book",
        "model_path": args.model_path,
        "data_source": args.data_source,
        "n": len(items),
        "seed": args.seed,
        "chat_template": args.chat_template,
        "num_fewshot": args.num_fewshot,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "em": em_mean,
        "em_stderr": em_se,
        "subem": subem_mean,
        "subem_stderr": subem_se,
        "subem_boundary": bnd_mean,
        "subem_boundary_stderr": bnd_se,
        "search_rate": search_rate,
        "answer_tag_rate": answer_tag_rate,
        "no_answer_rate": no_answer_rate,
        "mean_pred_words": mean_pred_words,
        "mean_pred_words_stderr": pred_words_se,
    }
    logger.info(json.dumps(summary, indent=2))

    # Suffix keeps the two prompt settings from overwriting each other.
    key = f"model={compact_model_slug(args.model_path)}"
    if args.chat_template:
        key += "_chat"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{key}.json").write_text(json.dumps(summary, indent=2))
    # Keep raw completions so error modes can be read directly, not inferred from aggregates.
    with (out_dir / f"{key}_records.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info(f"Wrote results to {out_dir}/{key}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True, help="HF repo id or local checkpoint dir.")
    parser.add_argument("--out_dir", type=str, default="out__nlu/popqa", help="Directory for result JSON.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/searchr1/validation.parquet",
        help="Search-R1 validation parquet.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="searchR1_popqa",
        help="data_source to evaluate (e.g. searchR1_popqa, searchR1_triviaqa).",
    )
    parser.add_argument(
        "--chat_template",
        action="store_true",
        help="Wrap the prompt in the model's chat template. Off by default (PopQA-paper-faithful "
        "raw completion); enable to check whether an instruct model degrades outside its chat format. "
        "Results are written to model=<slug>_chat.json so both settings can coexist.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=15,
        help="Few-shot exemplars, sampled one per relationship type other than the test "
        "question's (Mallen et al.). 0 disables few-shot.",
    )
    parser.add_argument(
        "--popqa_tsv",
        type=str,
        default="data/popqa/popQA.tsv",
        help="PopQA release from AlexTMallen/adaptive-retrieval; supplies relationship type, "
        "exemplar answers, and subject popularity, which the Search-R1 parquet drops.",
    )
    parser.add_argument("--n", type=int, default=0, help="Questions to sample; <=0 uses all.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for subsampling and sampling.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Matches eval_sampling_params.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Matches eval_sampling_params.")
    parser.add_argument("--max_tokens", type=int, default=500, help="Matches max_generate_length.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM context length.")
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory fraction."
    )
    main(parser.parse_args())
