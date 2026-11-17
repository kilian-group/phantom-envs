"""Tests of the LLM article rewrite: the fact check and the retry / fallback loop (no API calls)."""

import json
from types import SimpleNamespace

import litellm
import pytest

from phantom_envs.generator.rewrite import names_pattern, rewrite_articles, rewrite_problems

ANN = {
    "title": "Ann Smith",
    "article": "# Ann Smith\n\n## Family\nThe mother of Ann Smith is Beth Smith.\n",
    "facts": [
        'mother("Ann Smith", "Beth Smith").',
        'friend("Ann Smith", "Cal Jones").',
        'friend("Ann Smith", "Dee Park").',
        'dob("Ann Smith", "0471-09-21").',
        'job("Ann Smith", "chemist").',
        'gender("Ann Smith", "female").',
    ],
}
PEOPLE = names_pattern(["Ann Smith", "Beth Smith", "Cal Jones", "Dee Park", "Eve Moss"])
GOOD = (
    "# Ann Smith\n\nAnn Smith (born 0471-09-21) is a chemist.\n\n## Family\nHer mother is Beth Smith.\n\n"
    "## Friends\nSmith's friends are Cal Jones and Dee Park.\n"
)


@pytest.mark.parametrize(
    "text",
    [
        GOOD,
        GOOD.replace("0471-09-21", "21 September 471"),
        GOOD.replace("Her mother is Beth Smith.", "She is the daughter of Beth Smith, her mother."),
    ],
    ids=["as written", "attribute reworded", "relation reworded"],
)
def test_rewrites_that_keep_the_facts_pass(text):
    assert rewrite_problems(ANN, text, PEOPLE) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        (GOOD.replace("# Ann Smith\n", "# Smith\n"), "# Ann Smith"),
        (GOOD.replace("Her mother is Beth Smith.", "Her relative is Beth Smith."), "mother of Ann Smith"),
        (GOOD + "She knows Eve Moss.\n", "Eve Moss"),
    ],
    ids=["title", "relation dropped", "added person"],
)
def test_problems_are_found(text, expected):
    problems = rewrite_problems(ANN, text, PEOPLE)
    assert len(problems) == 1 and expected in problems[0]


def fake_batch_completion(texts_per_title):
    """A stand-in for litellm.batch_completion that answers each article with the next of its texts."""

    def batch_completion(model, messages, **kwargs):
        responses = []
        for m in messages:
            title = m[0]["content"].split('"# ')[1].split('"')[0]
            content = json.dumps({"article": texts_per_title[title].pop(0)})
            responses.append(
                SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            )
        return responses

    return batch_completion


def test_rewrite_retries_then_keeps_the_template(monkeypatch):
    bob = {
        "title": "Bob Lee",
        "article": "# Bob Lee\n\nThe hobby of Bob Lee is chess.\n",
        "facts": ['hobby("Bob Lee", "chess").'],
    }
    texts = {
        "Ann Smith": [GOOD + "She knows Bob Lee.\n", GOOD],
        "Bob Lee": ["# Lee\n\nHe plays chess.\n"] * 2,
    }
    monkeypatch.setattr(litellm, "batch_completion", fake_batch_completion(texts))

    articles, stats = rewrite_articles([ANN, bob], "some/model", attempts=2)

    assert stats == {"rewritten": 1, "kept_template": 1}
    assert articles[0]["article"] == GOOD and articles[0]["template"] == ANN["article"]
    assert articles[1]["article"] == bob["article"] == articles[1]["template"]
