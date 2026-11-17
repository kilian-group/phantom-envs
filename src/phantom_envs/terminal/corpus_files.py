"""A corpus as files: one markdown file per document of ``corpus.jsonl``.

Every sandbox serves the same files under ``/corpus``, so an agent sees the same names and contents
whichever backend runs its commands.

Usage::

    for name, text in corpus_files("data/envs/pw/corpus.jsonl"):
        ...  # e.g. ("Ann_Lee.md", "# Ann Lee\\n\\nThe mother of Ann Lee is ...")
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path

# ShellSim's rg refuses to read a file over 4 MiB, and stops the whole search there
MAX_DOC_BYTES: int = 4_000_000


def doc_filename(title: str, taken: set[str]) -> str:
    """Return a file name for a document: its title with unsafe characters replaced.

    Args:
        title: The document's title.
        taken: File names already used. The new name is added to it.

    Returns:
        A name ending in ``.md`` that is not in ``taken``; repeated titles get a numeric suffix.
    """
    stem = re.sub(r"[^\w.,()'-]+", "_", title).strip("_")[:120] or "untitled"
    name, i = f"{stem}.md", 2
    while name in taken:
        name, i = f"{stem}_{i}.md", i + 1
    taken.add(name)
    return name


def doc_text(title: str, contents: str) -> str:
    """Return a document's file contents: a markdown title line, then the text.

    Args:
        title: The document's title.
        contents: The document's text. PhantomEnvs articles already start with a title line.

    Returns:
        The file contents, cut to :data:`MAX_DOC_BYTES`.
    """
    text = contents if contents.startswith("# ") else f"# {title}\n\n{contents}"
    data = text.encode()
    if len(data) > MAX_DOC_BYTES:
        text = data[:MAX_DOC_BYTES].decode(errors="ignore")
    return text


def corpus_files(corpus_path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield each document of a corpus as a file.

    Args:
        corpus_path: A ``corpus.jsonl`` with one ``{"title", "contents"}`` document per line.

    Yields:
        ``(file name, file contents)`` per document, in corpus order.
    """
    taken: set[str] = set()
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            yield doc_filename(doc["title"], taken), doc_text(doc["title"], doc["contents"])
