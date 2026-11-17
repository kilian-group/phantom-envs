"""Rewrite the template articles as realistic Wikipedia articles with an LLM, keeping their facts.

:func:`~phantom_envs.generator.articles.build_articles` states each person's facts as template sentences
("The hobby of Ann Smith is chess."). :func:`rewrite_articles` asks a model on OpenRouter, through
litellm with structured output, to rewrite every article as a realistic Wikipedia article with the same
facts, leaving the style to the model. :func:`rewrite_problems` checks only the people in each rewrite:
every related person keeps their exact name and shares a sentence with the relation ("mother",
"friends", ...), and no one else from the universe appears. Other facts are left to spot checks. A
failing rewrite is retried with its problems listed, then falls back to the template.

Needs ``OPENROUTER_API_KEY`` in the environment.
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
    """The model's structured output."""

    article: str = Field(description="The rewritten article in Markdown, starting with its title line.")


def parse_facts(facts: list[str]) -> list[tuple[str, str, str]]:
    """``(relation, subject, value)`` of each Prolog fact, like ``mother("Ann Smith", "Beth Smith").``"""
    return [FACT_RE.match(fact).groups() for fact in facts]


def names_pattern(names: list[str]) -> re.Pattern:
    """A pattern that finds any of ``names`` as whole words (longest first)."""
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b")


def rewrite_problems(article: dict, text: str, people: re.Pattern) -> list[str]:
    """What a rewrite gets wrong about its template article.

    Args:
        article: The template article, ``{"title", "article", "facts"}``.
        text: The rewrite.
        people: :func:`names_pattern` over everyone in the universe.

    Returns:
        One line per problem; empty if the rewrite keeps the title and every relation and adds no one.
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
    """The chat messages asking for one article's rewrite, with a failed attempt's problems if any."""
    content = PROMPT.format(title=article["title"], template=article["article"].strip())
    if problems:
        content += FEEDBACK.format(problems="\n".join(f"- {p}" for p in problems))
    return [{"role": "user", "content": content}]


def rewrite_articles(
    articles: list[dict], model: str, workers: int = 32, attempts: int = 3
) -> tuple[list[dict], dict[str, int]]:
    """Rewrite every article with ``model``, in concurrent batches.

    Args:
        articles: ``{"title", "article", "facts"}`` per person.
        model: OpenRouter model slug, e.g. ``openai/gpt-6-luna``.
        workers: Concurrent requests.
        attempts: Tries per article before it keeps its template.

    Returns:
        The articles with ``article`` rewritten and the original in ``template``, and counts of rewritten
        articles and of those that kept their template.
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
            try:
                text = Rewrite.model_validate_json(response.choices[0].message.content).article.strip() + "\n"
            except Exception as e:  # an API error (returned in place of a response) or a broken output
                logger.warning(f"Rewrite of {articles[i]['title']!r} failed: {e!r}")
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
