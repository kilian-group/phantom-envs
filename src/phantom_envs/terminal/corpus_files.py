"""A corpus as files: one markdown file per document of `corpus.jsonl`.

Every sandbox serves the same files under `/corpus`, so an agent sees the same names and contents
whichever backend runs its commands.

Usage:

    for name, text in corpus_files("data/envs/pw/corpus.jsonl"):
        ...  # e.g. ("Ann_Lee.md", "# Ann Lee\\n\\nThe mother of Ann Lee is ...")

    corpus_dir = write_corpus_dir("data/envs/pw/corpus.jsonl")  # data/envs/pw/shell/corpus/
"""

import json
import os
import re
import shutil
import tempfile
import threading
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
        A name ending in `.md` that is not in `taken`; repeated titles get a numeric suffix.
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
        The file contents, cut to `MAX_DOC_BYTES`.
    """
    text = contents if contents.startswith("# ") else f"# {title}\n\n{contents}"
    data = text.encode()
    if len(data) > MAX_DOC_BYTES:
        text = data[:MAX_DOC_BYTES].decode(errors="ignore")
    return text


def corpus_files(corpus_path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield each document of a corpus as a file.

    Args:
        corpus_path: A `corpus.jsonl` with one `{"title", "contents"}` document per line.

    Yields:
        (file name, file contents) per document, in corpus order.
    """
    taken: set[str] = set()
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            yield doc_filename(doc["title"], taken), doc_text(doc["title"], doc["contents"])


def corpus_dir_path(corpus_path: str | Path) -> Path:
    """Return where a corpus lives as files: `shell/corpus/` next to its `corpus.jsonl`.

    Args:
        corpus_path: The corpus's `corpus.jsonl`.

    Returns:
        The corpus dir's path, whether or not it exists yet.
    """
    return Path(corpus_path).resolve().parent / "shell" / "corpus"


# Serializes this process's writers; the rename in write_corpus_dir handles other processes
_write_lock = threading.Lock()


def write_corpus_dir(corpus_path: str | Path) -> Path:
    """Write a corpus as files next to its `corpus.jsonl`, unless they are already there.

    Safe to call from many threads and processes at once: the files are written to a temporary dir
    and renamed into place, so a reader never sees a partial corpus.

    Args:
        corpus_path: The `corpus.jsonl` to write out.

    Returns:
        The corpus dir (see `corpus_dir_path`). An existing dir is kept as is, so delete it after
        changing `corpus.jsonl`.
    """
    out_dir = corpus_dir_path(corpus_path)
    with _write_lock:
        if out_dir.is_dir():
            return out_dir
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=".corpus-", dir=out_dir.parent))
        try:
            for name, text in corpus_files(corpus_path):
                (tmp_dir / name).write_text(text)
            tmp_dir.chmod(0o755)
            os.rename(tmp_dir, out_dir)
        except OSError:
            # Another process finished first; its files are the same
            if not out_dir.is_dir():
                raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_dir
