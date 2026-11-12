"""Bidirectional sampling: anchor the walk at a random person in the chain and walk both ways.

Backward sampling always starts at the chain's far end, so the answer's neighborhood is reached
only through long walks. Here the anchor is a random position of the chain: toward the answer the
walk follows relations forward (``R("known", A)``), toward the far end it follows them in
reverse (``R(A, "known")``), through the inverse predicates and caches in :class:`Lookups`.
"""

import re
from dataclasses import dataclass, field

from numpy.random import Generator

from phantom_envs.generator.sample import (
    ATTR_Y_VAL_RE,
    ATTR_Y_Y_RE,
    COUNT_NAME_Y_RE,
    COUNT_Y_Y_RE,
    RELATION_NAME_Y_RE,
    RELATION_Y_Y_RE,
    Assignments,
    Lookups,
    pick,
)
from phantom_envs.generator.templates import Template
from phantom_envs.generator.vocab import ATTRIBUTE_ALIASES, RELATION_ALIAS, RELATION_PLURAL_ALIAS

# Step kinds, matched in this order
STEP_PATTERNS = [
    ("attr_val", ATTR_Y_VAL_RE),
    ("relation_name", RELATION_NAME_Y_RE),
    ("relation_yy", RELATION_Y_Y_RE),
    ("attr_yy", ATTR_Y_Y_RE),
    ("count_name", COUNT_NAME_Y_RE),
    ("count_yy", COUNT_Y_Y_RE),
]


@dataclass
class Step:
    """One query atom as a step of the chain."""

    kind: str = field(metadata={"description": f"One of {[k for k, _ in STEP_PATTERNS]}, or 'static'."})
    match: re.Match | None = field(default=None, metadata={"description": "The atom's pattern match."})
    first: str | None = field(
        default=None, metadata={"description": "Input variable (or <name>), on the far-end side."}
    )
    second: str | None = field(
        default=None, metadata={"description": "Output variable, on the answer side (None for attr_val)."}
    )


def parse_step(atom: str) -> Step:
    """Classify a query atom.

    Args:
        atom: A query atom with placeholders.

    Returns:
        The step, ``static`` for atoms the walk does not fill (comparison plumbing).
    """
    for kind, pattern in STEP_PATTERNS:
        if m := pattern.search(atom):
            return Step(kind, m, m.group(2), None if kind == "attr_val" else m.group(3))
    return Step("static")


def variable_chain(steps: list[Step]) -> list[str]:
    """Order a query's variables along its chain.

    Args:
        steps: The query's steps, outermost atom first.

    Returns:
        The variables from the far end to the answer, e.g. ``['<name>_1', 'Y_2', 'Y_4']``.
    """
    chain = []
    for step in reversed(steps):
        for var in (step.first, step.second):
            if var is not None and var not in chain:
                chain.append(var)
    return chain


def processing_order(steps: list[Step], anchor: str) -> list[tuple[int, str]]:
    """Plan a walk from an anchor variable.

    Repeatedly takes, in template order, every step that has exactly one end known: forward if
    its input is known, inverse if its output is.

    Args:
        steps: The query's steps.
        anchor: The ``Y_i`` the walk starts from.

    Returns:
        (step index, "forward" or "inverse") for each step the walk reaches, in walk order.
    """
    # Grow the known variables from the anchor until no step can be added
    known = {anchor}
    done = set()
    order = []
    changed = True
    while changed:
        changed = False
        for idx, step in enumerate(steps):
            if idx in done:
                continue
            if step.kind == "static":
                done.add(idx)
                continue
            first_known = step.first is not None and step.first in known
            second_known = step.second is not None and step.second in known
            match step.kind:
                case "attr_val" | "attr_yy":
                    # Attributes of a known person; "What is the <an> of" also yields the answer
                    if first_known:
                        order.append((idx, "forward"))
                        if step.kind == "attr_yy" and step.second is not None:
                            known.add(step.second)
                        done.add(idx)
                        changed = True
                case "relation_yy" | "relation_name" | "count_yy":
                    # A relation is walked from whichever end is known; with both known it is done
                    if first_known and not second_known:
                        order.append((idx, "forward"))
                        if step.second is not None:
                            known.add(step.second)
                        done.add(idx)
                        changed = True
                    elif second_known and not first_known:
                        order.append((idx, "inverse"))
                        if step.first is not None:
                            known.add(step.first)
                        done.add(idx)
                        changed = True
                    elif first_known and second_known:
                        done.add(idx)
                        changed = True
                case "count_name":
                    # "How many <rp> does <name> have": from the name, or back from a related person
                    if first_known:
                        order.append((idx, "forward"))
                        if step.second:
                            known.add(step.second)
                        done.add(idx)
                        changed = True
                    elif second_known:
                        order.append((idx, "inverse"))
                        if step.first:
                            known.add(step.first)
                        done.add(idx)
                        changed = True
    return order


def walk_forward(step: Step, a: Assignments, rng: Generator, lookups: Lookups) -> bool:
    """Fill a step whose input person is known, following a relation forward (as backward sampling does).

    Args:
        step: The step.
        a: The attempt's assignments, updated in place.
        rng: Random generator.
        lookups: The universe's lookups.

    Returns:
        False on a dead end.
    """
    m = step.match
    match step.kind:
        case "relation_yy":
            # "the <relation> of" a known person: follow one of their relations
            relation, y1, y2 = m.groups()
            options = lookups.related_to(a.person(y1))
            if not options:
                return False
            rel, other = pick(rng, options)
            a.query[relation] = rel
            a.bind(y2, other)
            a.question[relation] = RELATION_ALIAS[rel]
            return True
        case "relation_name":
            # Unreachable from an anchor (a <name> is only known after an inverse step), kept for completeness
            relation, name, y = m.groups()
            person = pick(rng, lookups.names)
            a.query[name] = f'"{person}"'
            options = lookups.related_to(person)
            if not options:
                return False
            rel, other = pick(rng, options)
            a.query[relation] = rel
            a.bind(y, other)
            a.question[relation], a.question[name] = RELATION_ALIAS[rel], person
            return True
        case "attr_yy":
            # "What is the <attribute> of" a known person
            attr, y1, _ = m.groups()
            options = lookups.attributes_of(a.person(y1))
            if not options:
                return False
            name, _ = pick(rng, options)
            a.query[attr] = name
            a.question[attr] = ATTRIBUTE_ALIASES[name]
            return True
        case "attr_val":
            # A constraint on a known person, or on a new random one
            attr, y, value = m.groups()
            if y not in a.query:
                a.bind(y, pick(rng, lookups.names))
            return a.constrain(rng, lookups, attr, y, value)
        case "count_name" | "count_yy":
            # "How many <relations> does" a name or a known person "have"
            if step.kind == "count_name":
                relation_plural, name, _, _ = m.groups()
                if name not in a.query:
                    person = pick(rng, lookups.names)
                    a.query[name] = f'"{person}"'
                else:
                    person = a.atoms.get(a.query[name], a.query[name].strip('"'))
            else:
                relation_plural, y1, y2, _ = m.groups()
                person = a.person(y1)
            options = lookups.related_to(person)
            if not options:
                return False
            rel, other = pick(rng, options)
            a.query[relation_plural] = rel
            a.question[relation_plural] = RELATION_PLURAL_ALIAS[rel]
            if step.kind == "count_name":
                a.question[name] = person
            else:
                a.bind(y2, other)
            return True
    return False


def walk_inverse(step: Step, a: Assignments, rng: Generator, lookups: Lookups) -> bool:
    """Fill a step whose output person is known, by looking up who relates to them.

    Args:
        step: The step.
        a: The attempt's assignments, updated in place.
        rng: Random generator.
        lookups: The universe's lookups.

    Returns:
        False on a dead end.
    """
    m = step.match
    match step.kind:
        case "relation_yy" | "relation_name" | "count_name" | "count_yy":
            # Relation steps are (rel, first, second, ...): find a first with rel(first, second)
            rel_placeholder, first, second = m.group(1, 2, 3)
            options = lookups.inverse_of(a.person(second))
            if not options:
                return False
            rel, found = pick(rng, options)
            a.query[rel_placeholder] = rel
            alias = RELATION_PLURAL_ALIAS if step.kind.startswith("count") else RELATION_ALIAS
            if first.startswith("Y_"):
                a.bind(first, found)
                a.question[rel_placeholder] = alias[rel]
            else:
                a.query[first] = f'"{found}"'
                a.question[rel_placeholder], a.question[first] = alias[rel], found
            return True
        case "attr_val":
            # A constraint on a person the walk already reached
            attr, y, value = m.groups()
            if y not in a.query:
                return False
            return a.constrain(rng, lookups, attr, y, value)
        case "attr_yy":
            # Attributes have no inverse step; read one of the person, if reached
            attr, y1, _ = m.groups()
            if y1 not in a.query:
                return False
            options = lookups.attributes_of(a.person(y1))
            if not options:
                return False
            name, _ = pick(rng, options)
            a.query[attr] = name
            a.question[attr] = ATTRIBUTE_ALIASES[name]
            return True
    return False


def sample_bidirectional(
    template: Template, rng: Generator, lookups: Lookups, attempts: int
) -> tuple[str, list[str]] | None:
    """Sample a question from a random anchor of the chain.

    Args:
        template: The template to fill.
        rng: Random generator.
        lookups: The universe's lookups, with all three caches prewarmed.
        attempts: Walks to try from the anchor before giving up.

    Returns:
        The question and the query (outermost atom first), or None if the template has no anchor
        or every attempt hit a dead end.
    """
    # Anchors: the chain's Y variables from which a walk reaches every step
    steps = [parse_step(atom) for atom in template.query]
    chain = variable_chain(steps)
    num_steps = sum(1 for s in steps if s.kind != "static")
    anchors = [v for v in chain if v.startswith("Y_") and len(processing_order(steps, v)) >= num_steps]
    if not anchors:
        return None
    anchor = pick(rng, anchors)
    order = processing_order(steps, anchor)

    walk = {"forward": walk_forward, "inverse": walk_inverse}
    for _ in range(attempts):
        # Start at a random person; the first dead end ends the walk
        a = Assignments()
        a.bind(anchor, pick(rng, lookups.names))
        if all(walk[direction](steps[idx], a, rng, lookups) for idx, direction in order):
            return a.fill(template)
    return None
