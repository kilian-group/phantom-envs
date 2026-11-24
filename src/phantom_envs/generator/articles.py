"""One article per person: their family, friends and attributes, written as template sentences."""

from functools import partial

from phantom_envs.generator.parallel import parallel_map
from phantom_envs.generator.relations import Database
from phantom_envs.generator.vocab import (
    ATTRIBUTE_FACT_TEMPLATES,
    ATTRIBUTE_TYPES,
    EASY_FAMILY_RELATIONS,
    FAMILY_FACT_TEMPLATES,
    FAMILY_FACT_TEMPLATES_PL,
    FRIENDSHIP_FACT_TEMPLATES,
    FRIENDSHIP_FACT_TEMPLATES_PL,
    FRIENDSHIP_RELATION,
)

ARTICLE_TEMPLATE = """# {name}

## Family
{family_facts}

## Friends
{friend_facts}

## Attributes
{attribute_facts}
"""
# Parents, children and siblings are left out, as in PhantomWiki: mother, son, sister etc. state them
ARTICLE_FAMILY_RELATIONS = [r for r in EASY_FAMILY_RELATIONS if r not in ["parent", "child", "sibling"]]
# Gender is stated in articles, though questions leave it out
ARTICLE_ATTRIBUTES = ATTRIBUTE_TYPES + ["gender"]


def relation_sentences(
    db: Database, name: str, relations: list[str], templates: dict[str, str], plural_templates: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Relation sentences of a person, like "The sister of Ann Smith is Beth Smith.".

    Args:
        db: The universe.
        name: The person to write about.
        relations: Relations to state.
        templates: Sentence start per relation, with a `<subject>` slot, for one related person.
        plural_templates: The same for several related people.

    Returns:
        The sentences, and the facts behind them (with periods).
    """
    sentences, facts = [], []
    for relation in relations:
        # Everyone related that way, each named once
        targets = list(dict.fromkeys(db[relation].of(name)))
        facts += [f'{relation}("{name}", "{t}").' for t in targets]
        # One sentence lists them all
        if targets:
            template = plural_templates[relation] if len(targets) > 1 else templates[relation]
            sentences.append(template.replace("<subject>", name) + " " + ", ".join(targets) + ".")
    return sentences, facts


def attribute_sentences(db: Database, name: str, attributes: list[str]) -> tuple[list[str], list[str]]:
    """Attribute sentences of a person, like "The hobby of Ann Smith is chess.".

    Args:
        db: The universe.
        name: The person to write about.
        attributes: Attributes to state.

    Returns:
        The sentences, and the facts behind them (with periods).
    """
    sentences, facts = [], []
    for attr in attributes:
        targets = db[attr].of(name)
        facts += [f'{attr}("{name}", "{t}").' for t in targets]
        if targets:
            sentences.append(
                ATTRIBUTE_FACT_TEMPLATES[attr].replace("<subject>", name) + " " + ", ".join(targets) + "."
            )
    return sentences, facts


def build_article(db: Database, name: str) -> dict:
    """Write a person's article.

    Args:
        db: The universe.
        name: The person.

    Returns:
        `{"title", "article", "facts"}`. `facts` lists the facts the article states, in Prolog syntax.
    """
    # Family, friends and attributes (with gender, which questions leave out) as sentences
    family, family_facts = relation_sentences(
        db, name, ARTICLE_FAMILY_RELATIONS, FAMILY_FACT_TEMPLATES, FAMILY_FACT_TEMPLATES_PL
    )
    friends, friend_facts = relation_sentences(
        db, name, FRIENDSHIP_RELATION, FRIENDSHIP_FACT_TEMPLATES, FRIENDSHIP_FACT_TEMPLATES_PL
    )
    attributes, attribute_facts = attribute_sentences(db, name, ARTICLE_ATTRIBUTES)
    return {
        "title": name,
        "article": ARTICLE_TEMPLATE.format(
            name=name,
            family_facts="\n".join(family),
            friend_facts="\n".join(friends),
            attribute_facts="\n".join(attributes),
        ),
        "facts": family_facts + friend_facts + attribute_facts,
    }


def build_articles(db: Database, num_workers: int = 1) -> list[dict]:
    """Write an article for everyone in the universe.

    Args:
        db: The universe.
        num_workers: Processes to write with.

    Returns:
        One article per person (see `build_article`), in database order.
    """
    db.derive(ARTICLE_FAMILY_RELATIONS + FRIENDSHIP_RELATION + ARTICLE_ATTRIBUTES)
    return parallel_map(partial(build_article, db), db.names, num_workers, desc="Writing articles")
