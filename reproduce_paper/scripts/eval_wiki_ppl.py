#!/usr/bin/env -S uv run python
"""Perplexity on the Wikipedia-2018 retrieval corpus — probes language modeling quality.

Measures whether RL fine-tuning for search degrades the model's ability to model the very
documents it reads at eval time. Computes per-token perplexity over fixed-length windows of
``wiki-18.jsonl`` — the same corpus the retriever serves — so a drop is directly meaningful
rather than a generic fluency proxy.

Uses HF transformers rather than vLLM: vLLM has no clean full-sequence NLL path.

IMPORTANT: perplexity is only comparable across models sharing a tokenizer. The tokenizer name
and a vocabulary hash are recorded in the output JSON so a later cross-family table (e.g. Llama
vs Qwen) cannot silently compare incomparable numbers.

Note: ``wiki-18.jsonl`` ships with a 512-byte tar header glued to the front, so its first line
is not JSON. Lines that do not start with "{" are skipped.

Usage:
    uv run --no-sync python reproduce_paper/scripts/eval_wiki_ppl.py \\
        --model_path Qwen/Qwen2.5-3B-Instruct \\
        --out_dir out__nlu/wiki_ppl
"""

import argparse
import hashlib
import json
import logging
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phantom_envs.runs import compact_model_slug

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def iter_passage_chunks(corpus_path: Path, chunk_size: int, limit: int) -> Iterator[list[str]]:
    """Stream passages from the corpus in chunks of `chunk_size`.

    Streaming is required at full-corpus scale, not an optimization: holding all 21M
    passages plus their tokenized windows in memory needs >100 GB, since each window is a
    Python list of non-cached int objects (~6 KB per ~177-token passage).

    `limit` keeps only the first N passages. That is a valid random subsample when the file
    was produced by ``sample_wiki_corpus.py``, which already shuffles; take a prefix of the
    raw ordered corpus only if you do not care about representativeness.

    Args:
        corpus_path: Path to wiki-18.jsonl or a pre-drawn sample of it.
        chunk_size: Passages per yielded chunk.
        limit: Stop after this many passages; <= 0 reads the whole file.

    Yields:
        Lists of passage text strings.
    """
    chunk: list[str] = []
    seen = 0
    with corpus_path.open("r", errors="replace") as f:
        for line in f:
            # Skips the tar header glued to the front of the file.
            if not line.startswith("{"):
                continue
            try:
                text = json.loads(line)["contents"]
            except (json.JSONDecodeError, KeyError):
                continue
            chunk.append(text)
            seen += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            if 0 < limit <= seen:
                break
    if chunk:
        yield chunk


def build_windows(texts: list[str], tokenizer, window: int) -> list[list[int]]:
    """Tokenize passages and split them into fixed-length windows of token ids.

    Windows shorter than 2 tokens are dropped since they yield no predicted position.
    """
    windows: list[list[int]] = []
    batch_size = 1000
    for start in range(0, len(texts), batch_size):
        for ids in tokenizer(texts[start : start + batch_size], add_special_tokens=False)["input_ids"]:
            for offset in range(0, len(ids), window):
                chunk = ids[offset : offset + window]
                if len(chunk) >= 2:
                    windows.append(chunk)
    logger.info(f"Built {len(windows):,} windows of up to {window} tokens")
    return windows


@torch.no_grad()
def compute_nll(
    model, windows: list[list[int]], batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-window (negative log-likelihood, predicted-token count) arrays.

    Per-window values rather than totals: scoring two models on the same passages makes the
    comparison paired, and a paired standard error on the NLL difference is far tighter than
    comparing two independent corpus-level means.

    Windows are processed longest-first so each batch pads to a similar length, which cuts
    wasted compute substantially given the ~177-token average passage.

    Memory note: bf16 logits plus their fp32 copy cost ~batch * padded_len * vocab * 6
    bytes. wiki-18 passages reach ~930 tokens (dense Wikipedia text runs ~2.1 chars/token,
    so character length badly underestimates this), putting batch_size 64 at ~54 GB — an
    OOM even on a 48 GB card. Lower it for a bigger vocabulary or longer passages.
    """
    order = sorted(range(len(windows)), key=lambda i: -len(windows[i]))
    nlls = np.zeros(len(windows), dtype=np.float64)
    counts = np.zeros(len(windows), dtype=np.int64)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        batch = [windows[i] for i in idxs]
        max_len = max(len(w) for w in batch)
        # Pad with 0 and rely on the attention mask; padded positions are excluded from loss.
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, w in enumerate(batch):
            input_ids[i, : len(w)] = torch.tensor(w, dtype=torch.long)
            attention_mask[i, : len(w)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # Predict token t+1 from token t.
        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].bool()
        token_nll = loss_fn(shift_logits.transpose(1, 2), shift_labels) * shift_mask
        per_window = token_nll.sum(dim=1).cpu().numpy()
        per_count = shift_mask.sum(dim=1).cpu().numpy()
        for i, idx in enumerate(idxs):
            nlls[idx] = per_window[i]
            counts[idx] = per_count[i]

        if (start // batch_size) % 200 == 0:
            logger.info(f"  {start + len(batch):,}/{len(windows):,} windows")
    return nlls, counts


def tokenizer_fingerprint(tokenizer) -> str:
    """Short hash of the tokenizer vocabulary — guards against cross-tokenizer PPL comparison."""
    vocab = tokenizer.get_vocab()
    payload = json.dumps(vocab, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def ratio_stderr(nlls: np.ndarray, counts: np.ndarray) -> float:
    """Standard error of per-token mean NLL, treating each window as one observation.

    Mean NLL is a ratio of sums (total NLL / total tokens), not a plain average, so its
    error is the ratio-estimator variance obtained by linearizing about the ratio.
    """
    n = len(nlls)
    ratio = nlls.sum() / counts.sum()
    residuals = nlls - ratio * counts
    return float(np.sqrt((residuals**2).sum() / (n - 1) / n) / counts.mean())


def main(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    # Chunks are processed in file order, so the concatenated per-window arrays stay aligned
    # across models — that alignment is what makes the perplexity comparison paired.
    nll_parts, count_parts = [], []
    n_documents = 0
    for chunk_num, texts in enumerate(iter_passage_chunks(Path(args.corpus_path), args.chunk_size, args.n)):
        windows = build_windows(texts, tokenizer, args.window)
        chunk_nlls, chunk_counts = compute_nll(model, windows, args.batch_size, args.device)
        nll_parts.append(chunk_nlls)
        count_parts.append(chunk_counts)
        n_documents += len(texts)
        logger.info(f"chunk {chunk_num}: {n_documents:,} passages scored")

    nlls = np.concatenate(nll_parts)
    counts = np.concatenate(count_parts)
    total_nll = float(nlls.sum())
    total_tokens = int(counts.sum())
    mean_nll = total_nll / total_tokens
    nll_stderr = ratio_stderr(nlls, counts)
    summary = {
        "eval": "wiki18_perplexity",
        "model_path": args.model_path,
        "corpus_path": args.corpus_path,
        "n_documents": n_documents,
        "n_windows": len(nlls),
        "n_tokens": total_tokens,
        "window": args.window,
        "mean_nll": mean_nll,
        "mean_nll_stderr": nll_stderr,
        "perplexity": math.exp(mean_nll),
        # Delta method: d(exp(x)) = exp(x) dx.
        "perplexity_stderr": math.exp(mean_nll) * nll_stderr,
        # Only compare perplexity across models sharing these two fields.
        "tokenizer_name_or_path": str(tokenizer.name_or_path),
        "tokenizer_vocab_hash": tokenizer_fingerprint(tokenizer),
    }
    logger.info(json.dumps(summary, indent=2))

    key = f"model={compact_model_slug(args.model_path)}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{key}.json").write_text(json.dumps(summary, indent=2))
    # Per-window values enable a paired test against another model scored on the same passages.
    np.savez_compressed(out_dir / f"{key}_windows.npz", nll=nlls, tokens=counts)
    logger.info(f"Wrote results to {out_dir}/{key}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True, help="HF repo id or local checkpoint dir.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out__nlu/wiki_ppl",
        help="Directory for result JSON.",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/searchr1/corpus.jsonl",
        help="Wikipedia 2018 retrieval corpus.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Keep only the first N passages of corpus_path; <=0 uses all of them. "
        "Use sample_wiki_corpus.py to draw a random sample first.",
    )
    parser.add_argument("--window", type=int, default=2048, help="Token window length.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Windows per forward pass. The loss materializes bf16 + fp32 logits, costing "
        "~batch * padded_len * vocab * 6 bytes; wiki-18 passages reach ~930 tokens, so 16 "
        "peaks near 14 GB at Qwen's 151936-vocab. Raise it on larger-memory GPUs.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100_000, help="Passages streamed and scored per chunk."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    main(parser.parse_args())
