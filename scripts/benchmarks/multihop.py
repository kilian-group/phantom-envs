#!/usr/bin/env -S uv run python
"""Prepare multi-hop QA evaluation sets (HotpotQA / 2Wiki / MuSiQue / CofCA / SynthWorlds-RM / SynthWorlds-SM / FRAMES).

Reads the raw file for the requested dataset (or HF for ``frames``), builds a
deduplicated corpus from the question contexts, and writes
``tasks/validation.jsonl`` + ``corpus.jsonl`` into ``--data_dir``.

- hp500: 500 questions × 10 paragraphs (2 gold + 8 distractors) each.
- 2wiki500: 500 questions × 10 paragraphs each; gold answers expanded with id_aliases.json.
- msq500: 500 questions × ~20 paragraphs (~2–4 supporting) each.
- cofca500: 500 questions × 10 paragraphs each; HotpotQA-style scoring.
- synthrm500: 500 questions × ~4 gold documents each (no distractors).
- synthsm500: 500 questions × ~4 gold documents each (no distractors).
- frames: 824 questions; per-question gold Wikipedia articles fetched at the
  FRAMES revision cutoff (2023-06-01), section-chunked, and pooled.

The corpus only contains paragraphs that appear in the questions; this is the
universe the retriever indexes against.

Usage:
    uv run python scripts/benchmarks/multihop.py --dataset hp500 --data_dir data/hp500
    uv run python scripts/benchmarks/multihop.py --dataset 2wiki500 --data_dir data/2wiki500
    uv run python scripts/benchmarks/multihop.py --dataset msq500 --data_dir data/msq500
    uv run python scripts/benchmarks/multihop.py --dataset cofca500 --data_dir data/cofca500
    uv run python scripts/benchmarks/multihop.py --dataset synthrm500 --data_dir data/synthrm500
    uv run python scripts/benchmarks/multihop.py --dataset synthsm500 --data_dir data/synthsm500
    uv run python scripts/benchmarks/multihop.py --dataset frames --data_dir data/frames

Output (written into --data_dir):
    tasks/validation.jsonl  — the eval questions; the dataset name is their data source
    corpus.jsonl            — one JSON object per line: {"id", "title", "contents"}
    wiki_cache/             — (frames only) per-article fetched wikitext chunks
"""

import argparse
import ast
import json
import logging
import re
import time
import urllib.parse
from pathlib import Path

from phantom_envs._types import Task
from phantom_envs.tasks import write_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_FILES: dict[str, str | None] = {
    "hp500": "hotpot_minidev_distractor_v1.json",
    "2wiki500": "minidev.json",
    "msq500": "musique_ans_v1.0_minidev.jsonl",
    "cofca500": "minidev.json",
    "synthrm500": "minidev.json",
    "synthsm500": "minidev.json",
    "frames": None,  # loaded directly from HF (google/frames-benchmark)
}

# Field holding each dataset's question id
ID_KEYS: dict[str, str] = {
    "hp500": "_id",
    "2wiki500": "_id",
    "msq500": "id",
    "cofca500": "_id",
    "synthrm500": "instance_id",
    "synthsm500": "instance_id",
    "frames": "Unnamed: 0",
}

# FRAMES used the TFDS wikipedia/20230601.en dump
# (https://www.tensorflow.org/datasets/catalog/wikipedia#wikipedia20230601en_default_config).
# Pin every fetched article to its latest revision <= this timestamp so the
# corpus we build matches the snapshot the questions were authored against.
FRAMES_WIKI_API_URL: str = "https://en.wikipedia.org/w/api.php"
FRAMES_REVISION_CUTOFF: str = "2023-06-01T00:00:00Z"
FRAMES_HF_REPO_ID: str = "google/frames-benchmark"


def load_raw_rows(dataset: str, data_dir: Path) -> list[dict]:
    """Load raw question rows from disk (or HF for ``frames``).

    Args:
        dataset: One of "hp500", "2wiki500", "msq500", "cofca500", "synthrm500",
            "synthsm500", "frames".
        data_dir: Directory containing the raw file (unused for "frames").

    Returns:
        List of raw question dicts, one per question.
    """
    match dataset:
        case "hp500" | "2wiki500" | "cofca500" | "synthrm500" | "synthsm500":
            path = data_dir / DATASET_FILES[dataset]
            logger.info(f"Loading raw rows from {path}")
            with path.open() as f:
                rows = json.load(f)
        case "msq500":
            path = data_dir / DATASET_FILES[dataset]
            logger.info(f"Loading raw rows from {path}")
            with path.open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
        case "frames":
            from datasets import load_dataset

            logger.info(f"Loading raw rows from HF repo {FRAMES_HF_REPO_ID} (split=test)")
            ds = load_dataset(FRAMES_HF_REPO_ID, split="test")
            rows = ds.to_list()
            for i, row in enumerate(rows):
                row["_wiki_titles"] = extract_frames_wiki_titles(row)
                if "Unnamed: 0" not in row or row["Unnamed: 0"] is None:
                    row["Unnamed: 0"] = i
        case _:
            raise ValueError(f"Unknown dataset: {dataset!r}")

    if dataset == "2wiki500":
        alias_path = data_dir / "id_aliases.json"
        logger.info(f"Expanding 2wiki gold answers with aliases from {alias_path}")
        aliases: dict[str, list[str]] = {}
        with alias_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                aliases[entry["Q_id"]] = sorted(set(entry.get("aliases", []) + entry.get("demonyms", [])))
        for row in rows:
            row["answer_aliases"] = aliases.get(row.get("answer_id"), [])

    return rows


# ---------------------------------------------------------------------------
# FRAMES: fetch + section-chunk Wikipedia articles at the pinned revision
# ---------------------------------------------------------------------------


def parse_wiki_url(url: str) -> str | None:
    """Convert a Wikipedia article URL to its page title (returns None if not an article URL)."""
    try:
        path = urllib.parse.urlparse(url).path
    except Exception:
        return None
    if not path.startswith("/wiki/"):
        return None
    return urllib.parse.unquote(path[len("/wiki/") :]).replace("_", " ")


def extract_frames_wiki_titles(row: dict) -> list[str]:
    """Pull the ordered, deduped list of gold Wikipedia titles from a FRAMES row.

    Prefers the consolidated ``wiki_links`` field (a stringified list); falls
    back to the per-link ``wikipedia_link_<i>`` columns. Some FRAMES rows
    (e.g. row 90, 199) have a single string element containing several URLs
    joined by ``", "`` — we re-split those before parsing.
    """
    raw = row.get("wiki_links")
    urls: list[str] = []
    if isinstance(raw, str) and raw.strip():
        try:
            urls = list(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            urls = []
    elif isinstance(raw, list):
        urls = list(raw)
    if not urls:
        for k in sorted(k for k in row if k.startswith("wikipedia_link_")):
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                urls.append(v)

    # Some cells are a single string with several comma-joined URLs (CSV
    # overflow into wikipedia_link_11+ that got stuffed verbatim). Split on
    # ``, https?://`` so we recover the individual links.
    flat_urls: list[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        for piece in re.split(r",\s*(?=https?://)", u):
            piece = piece.strip().rstrip(",").strip()
            if piece:
                flat_urls.append(piece)

    seen: set[str] = set()
    titles: list[str] = []
    for u in flat_urls:
        title = parse_wiki_url(u)
        if title is None or title in seen:
            continue
        seen.add(title)
        titles.append(title)
    return titles


def resolve_title_and_revid(
    title: str, session, cutoff: str = FRAMES_REVISION_CUTOFF
) -> tuple[str, int] | None:
    """Resolve redirects and return ``(canonical_title, revid)`` of the latest revision <= cutoff.

    Returns None if the page doesn't exist or has no revision before the cutoff.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "redirects": "1",
        "prop": "revisions",
        "rvstart": cutoff,
        "rvdir": "older",
        "rvlimit": "1",
        "rvprop": "ids|timestamp",
    }
    r = session.get(FRAMES_WIKI_API_URL, params=params, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return None
        revs = page.get("revisions", [])
        if not revs:
            return None
        return page["title"], int(revs[0]["revid"])
    return None


def fetch_wikitext(revid: int, session) -> str | None:
    """Fetch the wikitext source of a specific revision; returns None on API error."""
    params = {"action": "parse", "format": "json", "oldid": str(revid), "prop": "wikitext"}
    r = session.get(FRAMES_WIKI_API_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "parse" not in data or "wikitext" not in data["parse"]:
        # ``error`` key contains code/info; e.g. nosuchrevid, permissiondenied.
        logger.warning(f"MediaWiki parse failed for revid={revid}: {data.get('error', data)}")
        return None
    return data["parse"]["wikitext"]["*"]


FRAMES_SKIP_SECTIONS: frozenset[str] = frozenset(
    {
        "references",
        "external links",
        "see also",
        "further reading",
        "bibliography",
        "notes",
        "footnotes",
        "sources",
        "citations",
        "works cited",
    }
)
FRAMES_MIN_CHUNK_CHARS: int = 40


FRAMES_IMAGE_PARAM_RE: re.Pattern[str] = re.compile(
    # Greedy run of image-inclusion parameters at line start, e.g.
    # ``thumb|left|alt=foo|200px|`` — strip the parameter chain so only the
    # caption survives. Matches keywords (positional sizing/alignment),
    # ``alt=...`` / ``link=...``, and ``Npx`` width specs, separated by ``|``.
    r"^(?:(?:thumb|thumbnail|left|right|center|centre|frame|frameless|"
    r"upright(?:=[^|\n]*)?|border|alt=[^|\n]*|link=[^|\n]*|class=[^|\n]*|"
    r"lang=[^|\n]*|page=[^|\n]*|\d+\s*x?\s*\d*\s*px)\|)+",
    re.IGNORECASE | re.MULTILINE,
)


def strip_wikitext_residue(text: str) -> str:
    """Remove markup that survives ``mwparserfromhell.strip_code()``.

    Targets unclosed/malformed templates and ``<ref>`` tags that the parser
    can't pair, image-inclusion parameter chains (``thumb|left|alt=...|``)
    that get left as plain text, plus stray HTML tags and entities.
    """
    import html as html_module

    # <ref>...</ref> (including nested cite/web templates inside)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # <ref ... /> self-closing and orphan <ref...> with no closer
    text = re.sub(r"<ref[^>]*/?>", "", text, flags=re.IGNORECASE)
    # Any other stray HTML tag (e.g. <br/ >, <small>, <sup>)
    text = re.sub(r"<[^>]+>", "", text)
    # Iteratively peel off innermost {{...}} templates until none remain.
    prev: str | None = None
    while prev != text:
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # Anything still starting with {{ is an unclosed/orphaned template — drop
    # to end of line so we don't accidentally eat real prose.
    text = re.sub(r"\{\{[^\n]*", "", text)
    text = re.sub(r"\}\}", "", text)
    # Image-inclusion parameter chains left over from [[File:...|thumb|...]].
    text = FRAMES_IMAGE_PARAM_RE.sub("", text)
    text = html_module.unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def chunk_wikitext(title: str, wikitext: str) -> list[dict]:
    """Strip wikitext markup and emit one chunk per section / subsection.

    Walks the parsed wiki tree with ``mwparserfromhell.get_sections(flat=True)``
    so each heading level (lead, ``==Section==``, ``===Subsection===``) is its
    own chunk, titled by its section path. Boilerplate sections (References,
    External links, ...) are dropped, as are category/file residue lines.
    No size cap — Wikipedia editors already chose semantically coherent units.
    """
    import mwparserfromhell

    code = mwparserfromhell.parse(wikitext)
    chunks: list[dict] = []
    for sec in code.get_sections(flat=True, include_lead=True, include_headings=True):
        headings = sec.filter_headings()
        if headings:
            heading_text = headings[0].title.strip_code().strip()
            if heading_text.lower() in FRAMES_SKIP_SECTIONS:
                continue
            section_path = f"{title} / {heading_text}"
            # Remove the heading node so its text doesn't reappear in body output.
            body_code = mwparserfromhell.parse(str(sec))
            for h in body_code.filter_headings():
                body_code.remove(h)
        else:
            section_path = title
            body_code = sec

        plain = strip_wikitext_residue(body_code.strip_code())
        # Drop lines that are pure link residue (categories, file/image embeds,
        # interwiki); strip_code keeps the prefix as plain text.
        kept_lines: list[str] = []
        for line in plain.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("Category:", "File:", "Image:", "Media:")):
                continue
            kept_lines.append(line)
        # Collapse runs of blank lines and trim section as a whole.
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
        if len(body) < FRAMES_MIN_CHUNK_CHARS:
            continue
        chunks.append({"title": section_path, "contents": body})
    return chunks


def fetch_and_chunk_article(title: str, cache_dir: Path, session) -> list[dict]:
    """Fetch one article at the FRAMES revision cutoff and return section chunks (cached).

    Cache stores both the raw wikitext and the chunks; on cache hit we always
    re-chunk from the cached wikitext so chunker changes don't require
    re-fetching from MediaWiki.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", title)[:200]
    cache_path = cache_dir / f"{safe}.json"
    if cache_path.exists():
        with cache_path.open() as f:
            entry = json.load(f)
        wikitext = entry.get("wikitext")
        canonical = entry.get("title", title)
        if wikitext is None:
            return []
        return chunk_wikitext(canonical, wikitext)

    resolved = resolve_title_and_revid(title, session)
    if resolved is None:
        logger.warning(f"No revision <= {FRAMES_REVISION_CUTOFF} for title {title!r}; skipping")
        with cache_path.open("w") as f:
            json.dump({"title": title, "revid": None, "wikitext": None}, f)
        return []
    canonical, revid = resolved

    wikitext = fetch_wikitext(revid, session)
    with cache_path.open("w") as f:
        json.dump({"title": canonical, "revid": revid, "wikitext": wikitext}, f, ensure_ascii=False)
    time.sleep(0.05)  # be polite to the MediaWiki API
    return chunk_wikitext(canonical, wikitext) if wikitext is not None else []


def build_corpus(dataset: str, rows: list[dict], data_dir: Path) -> list[dict]:
    """Build a deduplicated corpus from the question contexts.

    Each question's context paragraphs (gold + distractors for hp/msq; only gold
    for synthrm; fetched Wikipedia chunks for frames) are collected across all
    questions, then deduplicated so the same paragraph isn't indexed twice.
    Docs are assigned sequential integer IDs.

    Args:
        dataset: One of "hp500", "2wiki500", "msq500", "cofca500", "synthrm500",
            "synthsm500", "frames".
        rows: Raw question rows as produced by :func:`load_raw_rows`.
        data_dir: Output directory; used by ``frames`` for the wiki cache.

    Returns:
        List of ``{"id", "title", "contents"}`` dicts, one per unique paragraph.
    """
    seen: dict[tuple[str, str], int] = {}
    docs: list[dict] = []

    def add(title: str, contents: str) -> None:
        key = (title, contents)
        if key in seen:
            return
        seen[key] = len(docs)
        docs.append({"id": str(len(docs)), "title": title, "contents": contents})

    match dataset:
        case "hp500" | "2wiki500" | "cofca500":
            for row in rows:
                for title, sentences in row["context"]:
                    add(title, "\n".join(sentences))
        case "msq500":
            for row in rows:
                for paragraph in row["paragraphs"]:
                    add(paragraph["title"], paragraph["paragraph_text"])
        case "synthrm500" | "synthsm500":
            for row in rows:
                for doc in row["gold_docs"]:
                    # SynthRM/SM docs have no titles; use the first 60 chars as a stand-in
                    # (matches the old project's convention but longer to reduce collisions).
                    add(doc[:60], doc)
        case "frames":
            import requests

            cache_dir = data_dir / "wiki_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            session = requests.Session()
            session.headers.update(
                {"User-Agent": "phantom-envs/0.0 (https://github.com/kilian-group/phantom-envs)"}
            )

            titles: list[str] = []
            seen_titles: set[str] = set()
            for row in rows:
                for t in row.get("_wiki_titles", []):
                    if t in seen_titles:
                        continue
                    seen_titles.add(t)
                    titles.append(t)
            logger.info(
                f"Fetching {len(titles)} unique Wikipedia articles at {FRAMES_REVISION_CUTOFF} "
                f"(cache: {cache_dir})"
            )
            for i, title in enumerate(titles):
                if i % 50 == 0:
                    logger.info(f"  [{i}/{len(titles)}] fetching {title!r}")
                for chunk in fetch_and_chunk_article(title, cache_dir, session):
                    add(chunk["title"], chunk["contents"])
        case _:
            raise ValueError(f"Unknown dataset: {dataset!r}")

    if dataset == "frames":
        raw_total = sum(len(r.get("_wiki_titles", [])) for r in rows)
        logger.info(f"Built corpus with {len(docs)} unique section chunks (from {raw_total} gold articles)")
    else:
        logger.info(
            f"Built corpus with {len(docs)} unique paragraphs "
            f"(from {sum(len(r.get('context', r.get('paragraphs', r.get('gold_docs', [])))) for r in rows)} raw)"
        )
    return docs


def write_corpus(corpus: list[dict], corpus_path: Path) -> None:
    """Write the corpus to a JSONL file."""
    with corpus_path.open("w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(corpus)} documents to {corpus_path}")


def row_task(dataset: str, row: dict) -> Task:
    """Turn a raw question row into a task.

    Args:
        dataset: The dataset, also the task's data source.
        row: A raw row as produced by :func:`load_raw_rows`.

    Returns:
        The task. Its metadata (passed on to trainer formats) holds the native id plus the
        question type (HotpotQA-style sets, SynthWorlds), or the gold titles and reasoning types
        (FRAMES).
    """
    match dataset:
        case "hp500" | "2wiki500" | "cofca500":
            # 2Wiki answers also accept the gold entity's aliases and demonyms
            answers = [row["answer"], *row.get("answer_aliases", [])]
            q_type = row.get("type")
            if dataset == "cofca500":
                # CofCA's type is a str or a list; always a list, for one parquet schema
                q_type = list(q_type) if isinstance(q_type, list) else [q_type] if q_type else []
            return Task(row["_id"], dataset, row["question"], answers, {"id": row["_id"], "type": q_type})
        case "msq500":
            answers = [row["answer"], *(row.get("answer_aliases") or [])]
            return Task(row["id"], dataset, row["question"], answers, {"id": row["id"]})
        case "synthrm500" | "synthsm500":
            metadata = {"id": row["instance_id"], "type": row.get("question_graph_type")}
            return Task(row["instance_id"], dataset, row["query"], list(row["gold_answers"]), metadata)
        case "frames":
            wiki_titles = list(row.get("_wiki_titles") or [])
            metadata = {
                "id": int(row["Unnamed: 0"]),
                "reasoning_types": row.get("reasoning_types"),
                "n_hops": len(wiki_titles),
                "wiki_titles": wiki_titles,
            }
            return Task(str(row["Unnamed: 0"]), dataset, row["Prompt"], [row["Answer"]], metadata)
        case _:
            raise ValueError(f"Unknown dataset: {dataset!r}")


def main(args: argparse.Namespace) -> None:
    """Build tasks/validation.jsonl and corpus.jsonl for the requested dataset."""
    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = load_raw_rows(args.dataset, data_dir)

    # Sanity check: question IDs should be unique (we don't want duplicate tasks).
    ids = [r[ID_KEYS[args.dataset]] for r in rows]
    assert len(set(ids)) == len(ids), f"Duplicate question IDs in {args.dataset}"

    write_tasks(data_dir, "validation", [row_task(args.dataset, row) for row in rows])
    write_corpus(build_corpus(args.dataset, rows, data_dir), data_dir / "corpus.jsonl")
    logger.info(f"Dataset ready in {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a multi-hop QA evaluation set "
            "(hp500 / 2wiki500 / msq500 / cofca500 / synthrm500 / synthsm500 / frames)."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_FILES),
        help="Which dataset to prepare.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory for outputs; for non-frames datasets, also where the raw file already lives.",
    )
    args = parser.parse_args()
    main(args)
