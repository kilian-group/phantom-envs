"""Tests for reproduce_paper/scripts/generate_paper_data.py: the raw hash, the conversion of old questions,
and the generator calls."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_paper_data.py"
spec = importlib.util.spec_from_file_location("generate_paper_data", SCRIPT)
generate_paper_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_paper_data)
flags, generate_paper = generate_paper_data.flags, generate_paper_data.generate_paper
paper_task, raw_hash = generate_paper_data.paper_task, generate_paper_data.raw_hash


def write_raw(path, questions, facts):
    path.mkdir()
    (path / "articles.json").write_text(json.dumps([{"title": "Ann", "article": "Ann is 30.", "facts": []}]))
    (path / "questions.json").write_text(json.dumps(questions))
    (path / "facts.pl").write_text("\n".join(facts) + "\n")
    return path


class TestRawHash:
    def test_ignores_ids_and_fact_order(self, tmp_path):
        a = write_raw(tmp_path / "a", [{"id": "1", "answer": ["Ann"]}], ["age(ann, 30).", "female(ann)."])
        b = write_raw(tmp_path / "b", [{"id": "2", "answer": ["Ann"]}], ["female(ann).", "age(ann, 30)."])
        assert raw_hash(a) == raw_hash(b)

    def test_detects_answer_change(self, tmp_path):
        a = write_raw(tmp_path / "a", [{"id": "1", "answer": ["Ann"]}], ["female(ann)."])
        b = write_raw(tmp_path / "b", [{"id": "1", "answer": ["Bob"]}], ["female(ann)."])
        assert raw_hash(a) != raw_hash(b)


class TestPaperTask:
    def test_pw_difficulty_is_scored_from_the_query(self):
        question = {
            "question": "Who is the mother of the friend of the person whose hobby is chess?",
            "answer": ["Ann"],
            "prolog": {
                "query": ["mother(Y_2, Y_1)", "friend(Y_3, Y_2)", 'hobby(Y_3, "chess")'],
                "answer": "Y_1",
            },
            "type": 4,
            "difficulty": 5,
        }
        task = paper_task(question, "pw")
        m = task.metadata
        assert (task.data_source, m["hops"], m["constraints"], m["difficulty"], m["kind"]) == (
            "pw",
            2,
            1,
            5,
            "base",
        )

    def test_comparison_kind(self):
        question = {
            "question": "Who is younger, the son of Ann or the daughter of Bob?",
            "answer": ["Cid"],
            "prolog": {"query": [], "answer": "('Y_1', 'Y_2')"},
            "type": 300,
            "difficulty": {"hops": 1, "constraints": 0, "composite": 1, "constraint_position": None},
        }
        assert paper_task(question, "pwcomp").metadata["kind"] == "comparison"


class TestGeneratePaper:
    def test_refuses_nonempty_dir(self, tmp_path):
        (tmp_path / "train.parquet").write_text("")
        with pytest.raises(FileExistsError):
            generate_paper("pw1", tmp_path)

    def test_flags(self):
        assert flags(seed=1, question_depth=16, easy_mode=True) == (
            "--seed",
            "1",
            "--question-depth",
            "16",
            "--easy-mode",
        )
