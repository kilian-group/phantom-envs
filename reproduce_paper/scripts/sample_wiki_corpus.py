#!/usr/bin/env -S uv run python
"""Draw a uniform random sample of passages from the full Wikipedia-2018 corpus.

Produces the fixed passage set that ``eval_wiki_ppl.py`` scores. Sampling is done ONCE,
here, rather than inside each eval job, for two reasons:

1. Every model must be scored on the *identical* passages for the perplexity comparison to
   be paired. A shared sample file guarantees that; re-deriving it per job does not.
2. Reservoir sampling scans all ~21M passages (14 GB). Doing that once beats doing it in
   every job, and keeps NFS read load off the cluster.

The reservoir covers the WHOLE file, so the result is a uniform sample of the corpus — not
a prefix. That distinction matters: wiki-18.jsonl is ordered, so any prefix-based sample is
a biased slice of Wikipedia rather than a random one.

Note: wiki-18.jsonl ships with a 512-byte tar header glued to the front, so its first line
is not JSON. Lines that do not start with "{" are skipped.

Usage:
    uv run --no-sync python reproduce_paper/scripts/sample_wiki_corpus.py \\
        --n 200000 --out data/searchr1/wiki-18.sample200k.jsonl
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def reservoir_sample(corpus_path: Path, n: int, seed: int) -> list[str]:
    """Uniformly sample n passages from the entire corpus in a single pass.

    Args:
        corpus_path: Path to wiki-18.jsonl.
        n: Reservoir size.
        seed: RNG seed.

    Returns:
        List of passage text strings, in reservoir order.
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
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
            seen += 1
            if len(reservoir) < n:
                reservoir.append(text)
            else:
                j = rng.randrange(seen)
                if j < n:
                    reservoir[j] = text
            if seen % 2_000_000 == 0:
                logger.info(f"  scanned {seen:,} passages")
    logger.info(f"Sampled {len(reservoir):,} of {seen:,} passages ({100 * len(reservoir) / seen:.3f}%)")
    return reservoir


def main(args: argparse.Namespace) -> None:
    texts = reservoir_sample(Path(args.corpus_path), args.n, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for text in texts:
            f.write(json.dumps({"contents": text}) + "\n")
    logger.info(f"Wrote {len(texts):,} passages to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/searchr1/corpus.jsonl",
        help="Full Wikipedia 2018 retrieval corpus.",
    )
    parser.add_argument("--out", type=str, required=True, help="Output jsonl path for the sample.")
    parser.add_argument("--n", type=int, default=200_000, help="Number of passages to sample.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for the reservoir.")
    main(parser.parse_args())
