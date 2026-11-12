#!/usr/bin/env -S uv run python
"""Recover CofCA's missing question-category labels and cache them as a JSON artifact.

CofCA ships ``type`` as a list for 277 of its 500 questions (e.g. ``["4 hop",
"comparison"]``) but as a bare hop count for the other 223 -- the labelling pipeline
stored a string instead of a list and dropped the reasoning category. Those 223 land in
an ``uncategorized`` bucket that is really a mix of the existing categories: roughly
half of them are comparison questions.

This script re-derives a category for the unlabeled questions with the rule-based
classifier documented in ``<minidev_dir>/README.md`` (no model is used), and writes
``{question_id: category}`` for all 500 so the plotting scripts can bucket by a single
lookup. Categories are ``comparison``, ``bridge``, ``multi-attribute``, ``intersection``.

``minidev.json`` is the raw CofCA file in ``data/cofca500/`` (see scripts/benchmarks/multihop.py).
The map goes into the figures dir, where ``create_bar_plots_category_delta.py`` reads it.

Usage:
    uv run python reproduce_paper/scripts/build_cofca_categories.py
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

MINIDEV = Path("data/cofca500/minidev.json")
OUT_PATH = Path("reproduce_paper/figures/cofca500_categories.json")

COMPARATIVE_ADJECTIVES = r"(older|younger|earlier|later|taller|bigger|smaller|longer|shorter)\b"


def classify(item: dict) -> str:
    """Return the reasoning category for one unlabeled CofCA question.

    Rules are applied in priority order and taken verbatim from the dataset README, so
    that the counts here stay reproducible against that analysis.

    Args:
        item: a raw minidev.json record, with ``question`` and ``sub_questions``.

    Returns:
        One of ``multi-attribute``, ``intersection``, ``comparison``, ``bridge``.
    """
    question = item["question"].lower()
    sub_questions = [sq.lower() for sq in item.get("sub_questions", [])]
    n = len(sub_questions)

    # 1. Multi-attribute: several independent properties of one entity, no comparison.
    if re.search(r"(what|who|where|when).+and.+(what|who|where|when)", question):
        openings = {" ".join(sq.split()[:2]) for sq in sub_questions}
        if len(openings) == n and not re.search(r"\bor\b", question) and "same" not in question:
            return "multi-attribute"

    # 2. Intersection: one entity satisfying two simultaneous conditions.
    if re.search(r"(who|what).+(and also|who is also|that is also|who was also|also the)", question):
        return "intersection"
    if n >= 2:
        parallel = " ".join(sub_questions[0].split()[:3]) == " ".join(sub_questions[1].split()[:3])
        if parallel and re.search(r"(same person|same individual|same)", " ".join(sub_questions)):
            return "intersection"

    # 3. Comparison: two entities queried on one attribute, then compared.
    if re.search(r"\bor\b", question):
        return "comparison"
    if "between" in question and n <= 3:
        return "comparison"
    if "both" in question:
        return "comparison"
    if re.search(COMPARATIVE_ADJECTIVES, question):
        return "comparison"
    if re.search(r"(who|which).+(first|last)\b", question):
        return "comparison"
    if re.search(r"(same|differ)", question):
        return "comparison"
    for sq in sub_questions:
        if re.search(r"(which|who|is).*(more|less|first|last|same|older|younger|earlier|later|differ)", sq):
            return "comparison"
        if re.search(r"(is|are|were).*(same|both|either|neither)", sq):
            return "comparison"
    if n >= 2 and " ".join(sub_questions[0].split()[:3]) == " ".join(sub_questions[1].split()[:3]):
        return "comparison"

    # 4. Everything else chains through an intermediate entity.
    return "bridge"


def labeled_category(type_field: list[str]) -> str:
    """Return the reasoning category carried by an already-labeled ``type`` list."""
    for tag in type_field:
        if tag in ("comparison", "bridge"):
            return tag
    raise ValueError(f"no reasoning category in {type_field}")


def main(args: argparse.Namespace) -> None:
    records = json.loads(Path(args.minidev).read_text())

    categories: dict[str, str] = {}
    provenance = Counter()
    for item in records:
        if isinstance(item.get("type"), list):
            categories[item["_id"]] = labeled_category(item["type"])
            provenance["labeled"] += 1
        else:
            categories[item["_id"]] = classify(item)
            provenance["derived"] += 1

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "source": str(args.minidev),
                "counts": dict(Counter(categories.values())),
                "provenance": dict(provenance),
                "categories": categories,
            },
            indent=2,
            sort_keys=False,
        )
    )
    print(f"Saved {out_path} ({len(categories)} questions)")
    print(f"  provenance: {dict(provenance)}")
    print(f"  categories: {dict(Counter(categories.values()).most_common())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--minidev", type=str, default=str(MINIDEV), help="Path to CofCA minidev.json.")
    parser.add_argument("--out_path", type=str, default=str(OUT_PATH), help="Where to write the map.")
    main(parser.parse_args())
