"""One article per person: their family, friends and attributes, written as template sentences."""

from collections import defaultdict

from phantom_envs.generator.prolog import Database, decode
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


def relation_sentences(
    db: Database,
    names: list[str],
    relations: list[str],
    templates: dict[str, str],
    plural_templates: dict[str, str],
    workers: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Relation sentences of each person, like "The sister of Ann Smith is Beth Smith.".

    Args:
        db: The universe.
        names: People to write about.
        relations: Relations to state.
        templates: Sentence start per relation, with a ``<subject>`` slot, for one related person.
        plural_templates: The same for several related people.
        workers: Processes for the Prolog queries.

    Returns:
        Per person, their sentences and the facts behind them (with periods).
    """
    sents, facts = defaultdict(list), defaultdict(list)
    # One query per (person, relation) for everyone related that way
    pairs = [(name, relation) for name in names for relation in relations]
    results = db.batch_query([f'distinct({rel}("{name}", X))' for name, rel in pairs], workers)
    for (name, relation), rows in zip(pairs, results, strict=True):
        targets = [decode(row["X"]) for row in rows]
        facts[name] += [f'{relation}("{name}", "{t}").' for t in targets]
        # One sentence lists everyone related that way
        if targets:
            template = plural_templates[relation] if len(targets) > 1 else templates[relation]
            sents[name].append(template.replace("<subject>", name) + " " + ", ".join(targets) + ".")
    return sents, facts


def attribute_sentences(
    db: Database, names: list[str], attributes: list[str], workers: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Attribute sentences of each person, like "The hobby of Ann Smith is chess.".

    Args:
        db: The universe.
        names: People to write about.
        attributes: Attributes to state.
        workers: Processes for the Prolog queries.

    Returns:
        Per person, their sentences and the facts behind them (with periods).
    """
    sents, facts = defaultdict(list), defaultdict(list)
    pairs = [(name, attr) for name in names for attr in attributes]
    results = db.batch_query([f'{attr}("{name}", X)' for name, attr in pairs], workers)
    for (name, attr), rows in zip(pairs, results, strict=True):
        targets = [decode(row["X"]) for row in rows]
        facts[name] += [f'{attr}("{name}", "{t}").' for t in targets]
        if targets:
            sents[name].append(
                ATTRIBUTE_FACT_TEMPLATES[attr].replace("<subject>", name) + " " + ", ".join(targets) + "."
            )
    return sents, facts


def build_articles(db: Database, workers: int = 1) -> list[dict]:
    """Write an article for everyone in the universe.

    Args:
        db: The universe.
        workers: Processes for the Prolog queries.

    Returns:
        ``{"title", "article", "facts"}`` per person, in database order. ``facts`` lists the Prolog
        facts the article states.
    """
    names = db.person_names()
    # Family, friends and attributes (with gender, which questions leave out) as sentences
    family, family_facts = relation_sentences(
        db, names, ARTICLE_FAMILY_RELATIONS, FAMILY_FACT_TEMPLATES, FAMILY_FACT_TEMPLATES_PL, workers
    )
    friends, friend_facts = relation_sentences(
        db, names, FRIENDSHIP_RELATION, FRIENDSHIP_FACT_TEMPLATES, FRIENDSHIP_FACT_TEMPLATES_PL, workers
    )
    attributes, attribute_facts = attribute_sentences(db, names, ATTRIBUTE_TYPES + ["gender"], workers)
    return [
        {
            "title": name,
            "article": ARTICLE_TEMPLATE.format(
                name=name,
                family_facts="\n".join(family[name]),
                friend_facts="\n".join(friends[name]),
                attribute_facts="\n".join(attributes[name]),
            ),
            "facts": family_facts[name] + friend_facts[name] + attribute_facts[name],
        }
        for name in names
    ]
