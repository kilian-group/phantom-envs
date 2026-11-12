"""The retriever's corpus: ``corpus.jsonl``, one ``{"id", "title", "contents"}`` document per line."""

import json
from pathlib import Path


def write_corpus(articles: list[dict], path: str | Path) -> None:
    """Write the retriever's corpus.

    Args:
        articles: ``{"title", "article", ...}`` per person.
        path: Where to write ``corpus.jsonl``.
    """
    # Ids are the articles' positions
    with Path(path).open("w") as f:
        for idx, article in enumerate(articles):
            f.write(
                json.dumps({"id": str(idx), "title": article["title"], "contents": article["article"]}) + "\n"
            )
