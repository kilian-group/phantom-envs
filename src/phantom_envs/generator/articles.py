"""One article per person: their family, friends and attributes, written as template sentences."""

from collections import defaultdict

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


def relation_sentences(
    db: Database,
    relations: list[str],
    templates: dict[str, str],
    plural_templates: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Relation sentences of each person, like "The sister of Ann Smith is Beth Smith.".

    Args:
        db: The universe.
        relations: Relations to state.
        templates: Sentence start per relation, with a ``<subject>`` slot, for one related person.
        plural_templates: The same for several related people.

    Returns:
        Per person, their sentences and the facts behind them (with periods).
    """
    sents, facts = defaultdict(list), defaultdict(list)
    for name in db.names:
        for relation in relations:
            # Everyone related that way, each named once
            targets = list(dict.fromkeys(db[relation].of(name)))
            facts[name] += [f'{relation}("{name}", "{t}").' for t in targets]
            # One sentence lists them all
            if targets:
                template = plural_templates[relation] if len(targets) > 1 else templates[relation]
                sents[name].append(template.replace("<subject>", name) + " " + ", ".join(targets) + ".")
    return sents, facts


def attribute_sentences(
    db: Database, attributes: list[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Attribute sentences of each person, like "The hobby of Ann Smith is chess.".

    Args:
        db: The universe.
        attributes: Attributes to state.

    Returns:
        Per person, their sentences and the facts behind them (with periods).
    """
    sents, facts = defaultdict(list), defaultdict(list)
    for name in db.names:
        for attr in attributes:
            targets = db[attr].of(name)
            facts[name] += [f'{attr}("{name}", "{t}").' for t in targets]
            if targets:
                sents[name].append(
                    ATTRIBUTE_FACT_TEMPLATES[attr].replace("<subject>", name) + " " + ", ".join(targets) + "."
                )
    return sents, facts


def build_articles(db: Database) -> list[dict]:
    """Write an article for everyone in the universe.

    Args:
        db: The universe.

    Returns:
        `{"title", "article", "facts"}` per person, in database order. `facts` lists the facts
        the article states, in Prolog syntax.
    """
    # Family, friends and attributes (with gender, which questions leave out) as sentences
    family, family_facts = relation_sentences(
        db, ARTICLE_FAMILY_RELATIONS, FAMILY_FACT_TEMPLATES, FAMILY_FACT_TEMPLATES_PL
    )
    friends, friend_facts = relation_sentences(
        db, FRIENDSHIP_RELATION, FRIENDSHIP_FACT_TEMPLATES, FRIENDSHIP_FACT_TEMPLATES_PL
    )
    attributes, attribute_facts = attribute_sentences(db, ATTRIBUTE_TYPES + ["gender"])
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
        for name in db.names
    ]
