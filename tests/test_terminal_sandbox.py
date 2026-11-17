"""Tests for the terminal sandbox: the corpus as files (phantom_envs.terminal.corpus_files) and the ShellSim
sandbox (phantom_envs.terminal.shellsim)."""

import json
from pathlib import Path

import pytest

from phantom_envs.terminal.corpus_files import MAX_DOC_BYTES, corpus_files, doc_filename, doc_text
from phantom_envs.terminal.task import CORPUS_DIR

DOCS = [
    {"id": "0", "title": "Ann Lee", "contents": "# Ann Lee\n\nThe mother of Ann Lee is Bea Lee.\n"},
    {"id": "1", "title": "Bea Lee", "contents": "# Bea Lee\n\nThe hobby of Bea Lee is chess.\n"},
]


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-document corpus.jsonl."""
    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
    path.write_text("".join(json.dumps(d) + "\n" for d in DOCS))
    return path


def test_doc_filename_unique_and_safe():
    taken: set[str] = set()
    assert doc_filename("Alison Smock", taken) == "Alison_Smock.md"
    assert doc_filename("Alison Smock", taken) == "Alison_Smock_2.md"
    assert doc_filename("a/b: c?", taken) == "a_b_c.md"


def test_doc_text():
    assert doc_text("Ann", "# Ann\n\nText.") == "# Ann\n\nText."
    assert doc_text("Paris", "Paris is a city.") == "# Paris\n\nParis is a city."
    assert len(doc_text("Big", "x" * (MAX_DOC_BYTES + 10)).encode()) == MAX_DOC_BYTES


def test_corpus_files(corpus_path):
    assert list(corpus_files(corpus_path)) == [
        ("Ann_Lee.md", DOCS[0]["contents"]),
        ("Bea_Lee.md", DOCS[1]["contents"]),
    ]


class TestShellSimSandbox:
    @pytest.fixture(scope="class")
    def image(self, corpus_path):
        pytest.importorskip("shellsim")
        from phantom_envs.terminal.shellsim import build_corpus_image

        return build_corpus_image(corpus_path)

    @pytest.fixture
    def make_sandbox(self, image):
        from phantom_envs.terminal.shellsim import ShellSimSandbox

        return lambda **kwargs: ShellSimSandbox(image, **kwargs)

    def test_search_and_read(self, make_sandbox):
        sandbox = make_sandbox()
        assert sandbox.run(f"grep -l chess {CORPUS_DIR}/*.md").strip() == f"{CORPUS_DIR}/Bea_Lee.md"
        assert "Bea Lee" in sandbox.run(f"cat {CORPUS_DIR}/Ann_Lee.md")

    def test_answer_file(self, make_sandbox):
        sandbox = make_sandbox()
        assert sandbox.read_answer() is None
        sandbox.run("echo 'Bea Lee' > output.md")  # the working dir is /app
        assert sandbox.read_answer() == "Bea Lee\n"

    def test_sandboxes_are_isolated(self, make_sandbox):
        first, second = make_sandbox(), make_sandbox()
        first.run("echo note > /tmp/x")
        assert "No such file" in second.run("cat /tmp/x")

    def test_failure_and_timeout_notes(self, make_sandbox):
        sandbox = make_sandbox(timeout=0.5)
        assert "[exit code 1]" in sandbox.run(f"grep -l nothing {CORPUS_DIR}/*.md")
        assert "timed out" in sandbox.run("while true; do :; done")
        assert sandbox.run("echo alive").strip() == "alive"
