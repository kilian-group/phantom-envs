"""Rewrite the template articles as realistic Wikipedia articles with an LLM, keeping their facts.

The generator states each person's facts as template sentences ("The hobby of Ann Smith is chess.").
A model on OpenRouter, called through litellm with structured output, rewrites every article as a
Wikipedia article with the same facts, in a style of its own choosing. Each rewrite is checked only for
its people: every related person keeps their exact name and shares a sentence with the relation
("mother", "friends", ...), and no one else from the universe appears. Other facts are left to spot
checks. A rewrite that fails is retried with its problems listed, then falls back to the template.

Needs OPENROUTER_API_KEY in the environment.

Usage::

    articles, stats = rewrite_articles(articles, "openai/gpt-6-luna")
"""

import logging
import re

import litellm
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROMPT = """You are a Wikipedia editor. Write the Wikipedia article about this fictional person from the \
notes below, following Wikipedia's Manual of Style for biographies. Keep every fact unchanged and add \
none: no new people, dates, places, events, numbers or traits. Write dates of birth in YYYY-MM-DD \
format. Start with the line "# {title}".

Notes:
{template}"""

FEEDBACK = "\n\nA previous rewrite had these problems; avoid them:\n{problems}"

FACT_RE = re.compile(r'^(\w+)\("(.*)", "(.*)"\)\.$')
# Words that state a relation; the others are stated by their own name ("mother", "friend", ...)
RELATION_WORDS = {"husband": ("husband", "married", "spouse"), "wife": ("wife", "married", "spouse")}
# Sentences, and lines (headings, table rows)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n")


class Rewrite(BaseModel):
    """The structured output asked of the model."""

    article: str = Field(description="The rewritten article in Markdown, starting with its title line.")


def parse_facts(facts: list[str]) -> list[tuple[str, str, str]]:
    """Split Prolog facts into their parts.

    Args:
        facts: Facts like ``mother("Ann Smith", "Beth Smith").``

    Returns:
        ``(relation, subject, value)`` per fact.
    """
    return [FACT_RE.match(fact).groups() for fact in facts]


def names_pattern(names: list[str]) -> re.Pattern:
    """Build a pattern that finds any of the names as whole words.

    Args:
        names: Everyone in the universe.

    Returns:
        The pattern; longer names come first, so a name never matches inside a longer one.
    """
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b")


def rewrite_problems(article: dict, text: str, people: re.Pattern) -> list[str]:
    """Find what a rewrite gets wrong about the people in its template article.

    Args:
        article: The template article, ``{"title", "article", "facts"}``.
        text: The rewrite.
        people: The pattern from ``names_pattern`` over everyone in the universe.

    Returns:
        One line per problem, phrased as feedback for the model. Empty if the rewrite keeps the title and
        every relation and mentions no one else.
    """
    problems = []
    title = article["title"]
    if not text.lstrip().startswith(f"# {title}\n"):
        problems.append(f'It must start with the line "# {title}".')
    sentences = SENTENCE_END.split(text)
    related = {title}
    for relation, subject, value in parse_facts(article["facts"]):
        # Relations to other people; attributes are not checked
        if not people.fullmatch(value):
            continue
        related.add(value)
        word = re.compile(rf"\b(?:{'|'.join(RELATION_WORDS.get(relation, (relation,)))})s?\b", re.IGNORECASE)
        if not any(value in s and word.search(s) for s in sentences):
            problems.append(f"It must say in one sentence that {value} is the {relation} of {subject}.")
    for name in sorted(set(people.findall(text)) - related):
        problems.append(f"It mentions {name}, who is not in the article.")
    return problems


def messages(article: dict, problems: list[str] | None) -> list[dict[str, str]]:
    """Build the chat messages that ask for one article's rewrite.

    Args:
        article: The template article, ``{"title", "article", "facts"}``.
        problems: Problems of the previous attempt, or None on the first.

    Returns:
        The messages, a single user turn.
    """
    content = PROMPT.format(title=article["title"], template=article["article"].strip())
    if problems:
        content += FEEDBACK.format(problems="\n".join(f"- {p}" for p in problems))
    return [{"role": "user", "content": content}]


def rewrite_articles(
    articles: list[dict], model: str, workers: int = 32, attempts: int = 3
) -> tuple[list[dict], dict[str, int]]:
    """Rewrite every article with a model, in concurrent batches.

    Args:
        articles: ``{"title", "article", "facts"}`` per person.
        model: OpenRouter model slug, e.g. ``openai/gpt-6-luna``.
        workers: Concurrent requests.
        attempts: Tries per article before it keeps its template.

    Returns:
        The articles, each with its rewrite in ``article`` and the original in ``template``, and the
        counts of articles rewritten and of those that kept their template.
    """
    # litellm logs every request at INFO
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    people = names_pattern([a["title"] for a in articles])
    texts: list[str | None] = [None] * len(articles)
    problems: dict[int, list[str]] = {}
    todo = list(range(len(articles)))
    for attempt in range(attempts):
        if not todo:
            break
        responses = litellm.batch_completion(
            model=f"openrouter/{model}",
            messages=[messages(articles[i], problems.get(i)) for i in todo],
            response_format=Rewrite,
            max_workers=workers,
            num_retries=3,
        )
        retry = []
        for i, response in zip(todo, responses, strict=True):
            # A request that failed after its retries comes back as the exception, in place of a response
            if isinstance(response, Exception):
                logger.warning(f"Rewrite request for {articles[i]['title']!r} failed: {response!r}")
                retry.append(i)
                continue
            try:
                text = Rewrite.model_validate_json(response.choices[0].message.content).article.strip() + "\n"
            except Exception as e:  # e.g. a truncated output, or none at all
                logger.warning(f"Rewrite of {articles[i]['title']!r} is not a valid output: {e!r}")
                retry.append(i)
                continue
            problems[i] = rewrite_problems(articles[i], text, people)
            if problems[i]:
                retry.append(i)
            else:
                texts[i] = text
        logger.info(f"Rewrite attempt {attempt + 1}: {len(todo) - len(retry)} of {len(todo)} articles passed")
        todo = retry

    stats = {"rewritten": len(articles) - len(todo), "kept_template": len(todo)}
    rewritten = [
        a | {"article": text or a["article"], "template": a["article"]}
        for a, text in zip(articles, texts, strict=True)
    ]
    return rewritten, stats
