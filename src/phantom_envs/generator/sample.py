"""Fill a template's placeholders from the universe by a random walk ("backward" sampling).

The walk starts at the end of the query (the chain's anchor) and follows sampled relations toward
the answer, so every sampled question has at least one answer. Example::

    Who is the <relation>_3 of the person whose <attribute_name>_1 is <attribute_value>_1 ?
    -> Who is the child of the person whose hobby is chess?

The lookups behind the walk are cached per person in :class:`Lookups`.
"""

import re
from dataclasses import dataclass, field

from numpy.random import Generator

from phantom_envs.generator.prolog import Database, decode
from phantom_envs.generator.templates import Template
from phantom_envs.generator.vocab import (
    ATTRIBUTE_ALIASES,
    ATTRIBUTE_TYPES,
    INVERSE_RELATIONS,
    RELATION_ALIAS,
    RELATION_PLURAL_ALIAS,
)

PLACEHOLDER_RE = re.compile(r"<\w+>_\d+")
ATTR_Y_VAL_RE = re.compile(r"(<attribute_name>_\d+)\((Y_\d+), (<attribute_value>_\d+)\)")
RELATION_NAME_Y_RE = re.compile(r"(<relation>_\d+)\((<name>_\d+), (Y_\d+)\)")
RELATION_Y_Y_RE = re.compile(r"(<relation>_\d+)\((Y_\d+), (Y_\d+)\)")
ATTR_Y_Y_RE = re.compile(r"(<attribute_name>_\d+)\((Y_\d+), (Y_\d+)\)")
COUNT_NAME_Y_RE = re.compile(
    r"aggregate_all\(count, distinct\((<relation_plural>_\d+)\((<name>_\d+), (Y_\d+)\)\), (Count_\d+)\)"
)
COUNT_Y_Y_RE = re.compile(
    r"aggregate_all\(count, distinct\((<relation_plural>_\d+)\((Y_\d+), (Y_\d+)\)\), (Count_\d+)\)"
)


@dataclass
class Lookups:
    """A universe's people and their (attribute, value), (relation, person) and inverse pairs."""

    db: Database = field(metadata={"description": "The universe."})
    relations: list[str] = field(metadata={"description": "Relations questions may use."})
    workers: int = field(default=1, metadata={"description": "Processes for batched queries."})
    names: list[str] = field(default_factory=list, metadata={"description": "Everyone, in database order."})
    attributes: dict[str, list[tuple[str, str]]] = field(
        default_factory=dict, metadata={"description": "Person -> (attribute, value) pairs."}
    )
    related: dict[str, list[tuple[str, str]]] = field(
        default_factory=dict, metadata={"description": "Person X -> (R, Y) pairs with R(X, Y)."}
    )
    inverse: dict[str, list[tuple[str, str]]] = field(
        default_factory=dict, metadata={"description": "Person Y -> (R, X) pairs with R(X, Y)."}
    )

    def __post_init__(self) -> None:
        self.names = self.names or self.db.person_names()

    def lookup(self, cache: dict, key: str, predicates: list[str], inverse: bool) -> list[tuple[str, str]]:
        """Look up a person's pairs, querying Prolog on a cache miss.

        Args:
            cache: One of :attr:`attributes`, :attr:`related` or :attr:`inverse`.
            key: The person.
            predicates: Predicates P to query, ``P(key, A)``.
            inverse: Query ``P(A, key)`` instead, through the inverse predicates.

        Returns:
            ``(P, A)`` for every solution, grouped by predicate in ``predicates`` order.
        """
        if key not in cache:
            if inverse:
                # R_inverse("key", A) uses first-argument indexing, unlike R(A, "key")
                queries = [f'{INVERSE_RELATIONS[p]}("{key}", A)' for p in predicates]
            else:
                queries = [f'{p}("{key}", A)' for p in predicates]
            results = self.db.batch_query(queries, self.workers)
            cache[key] = [
                (p, decode(r["A"])) for p, rows in zip(predicates, results, strict=True) for r in rows
            ]
        return cache[key]

    def attributes_of(self, person: str) -> list[tuple[str, str]]:
        """A person's (attribute, value) pairs."""
        return self.lookup(self.attributes, person, ATTRIBUTE_TYPES, inverse=False)

    def related_to(self, person: str) -> list[tuple[str, str]]:
        """``(R, Y)`` pairs with ``R(person, Y)``, i.e. Y is the person's R."""
        return self.lookup(self.related, person, self.relations, inverse=False)

    def inverse_of(self, person: str) -> list[tuple[str, str]]:
        """``(R, X)`` pairs with ``R(X, person)``, i.e. the person is X's R."""
        return [
            (r, p)
            for r, p in self.lookup(self.inverse, person, self.relations, inverse=True)
            if r in self.relations
        ]

    def prewarm(self, cache: dict, predicates: list[str], inverse: bool) -> None:
        """Fill a cache for everyone at once, with one ``P(X, Y)`` query per predicate.

        Much faster than querying person by person, but Prolog can order the solutions differently,
        which changes the sampled questions; :func:`phantom_envs.generator.questions.sample_questions`
        fixes when to prewarm.

        Args:
            cache: One of :attr:`attributes`, :attr:`related` or :attr:`inverse`.
            predicates: Predicates to query.
            inverse: Key the cache by Y (for :attr:`inverse`) instead of X.
        """
        results = self.db.batch_query([f"{p}(X, Y)" for p in predicates], self.workers)
        for p, rows in zip(predicates, results, strict=True):
            for r in rows:
                x, y = decode(r["X"]), decode(r["Y"])
                cache.setdefault(y if inverse else x, []).append((p, x if inverse else y))


def pick(rng: Generator, items: list):
    """Draw a uniformly random item.

    Args:
        rng: Random generator.
        items: A non-empty list.

    Returns:
        One of ``items``.
    """
    return items[rng.integers(0, len(items))]


@dataclass
class Assignments:
    """Placeholder values of one sampling attempt.

    People found along the walk get atom variables ``A_i`` (so the query keeps its ``Y_i``
    variables), while relations, attributes and starting names are written into the query.
    """

    atoms: dict[str, str] = field(default_factory=dict, metadata={"description": "A_i -> person."})
    query: dict[str, str] = field(
        default_factory=dict, metadata={"description": "Placeholder -> value or A_i."}
    )
    question: dict[str, str] = field(default_factory=dict, metadata={"description": "Placeholder -> words."})
    used_attributes: dict[str, set[tuple[str, str]]] = field(
        default_factory=dict, metadata={"description": "Y_i -> its (attribute, value) constraints so far."}
    )

    def bind(self, placeholder: str, person: str) -> None:
        """Bind a ``Y_i`` to a person through a fresh atom variable ``A_j``."""
        atom = f"A_{len(self.atoms)}"
        self.atoms[atom] = person
        self.query[placeholder] = atom

    def person(self, placeholder: str) -> str:
        """The person a bound ``Y_i`` stands for."""
        return self.atoms[self.query[placeholder]]

    def constrain(self, rng: Generator, lookups: Lookups, attr: str, y: str, value: str) -> bool:
        """Fill a constraint ``attr(y, value)`` with one of the person's attributes.

        Args:
            rng: Random generator.
            lookups: The universe's lookups.
            attr: The ``<attribute_name>_N`` placeholder.
            y: The bound ``Y_i`` it constrains.
            value: The ``<attribute_value>_N`` placeholder.

        Returns:
            False if every attribute of the person already constrains them.
        """
        # Several constraints on one person must differ
        used = self.used_attributes.setdefault(y, set())
        options = [(n, v) for n, v in lookups.attributes_of(self.person(y)) if (n, v) not in used]
        if not options:
            return False
        name, val = pick(rng, options)
        used.add((name, val))
        self.query[attr], self.query[value] = name, f'"{val}"'
        self.question[attr], self.question[value] = ATTRIBUTE_ALIASES[name], val
        return True

    def fill(self, template: Template) -> tuple[str, list[str]]:
        """Write the sampled values into a template.

        Args:
            template: The sampled template.

        Returns:
            The question and the query; ``Y_i`` stay variables so Prolog finds every answer.
        """
        joined = ",,".join(template.query)  # ",," because "," appears inside atoms
        for placeholder, value in self.query.items():
            if value not in self.atoms:
                joined = joined.replace(placeholder, value)
        # No space before the final "?"
        question = " ".join(template.question[:-1]) + template.question[-1]
        for placeholder, value in self.question.items():
            question = question.replace(placeholder, value)
        return question, joined.split(",,")

    def bindings(self) -> dict[str, str]:
        """The people each ``Y_i`` and ``<name>_N`` resolved to (for comparison answers).

        Returns:
            Placeholder -> person.
        """
        out = {}
        for placeholder, value in self.query.items():
            if value in self.atoms:
                out[placeholder] = self.atoms[value]
            elif len(value) >= 2 and value[0] == value[-1] == '"':
                out[placeholder] = value[1:-1]
        return out


def walk_atom(atom: str, a: Assignments, rng: Generator, lookups: Lookups) -> bool:
    """Fill the placeholders of one query atom, continuing the walk.

    Args:
        atom: A query atom; its input person (if any) is bound by the atoms walked before it.
        a: The attempt's assignments, updated in place.
        rng: Random generator.
        lookups: The universe's lookups.

    Returns:
        False on a dead end (a person with no matching relation or attribute left).

    Raises:
        ValueError: For an atom with placeholders the walk cannot reach.
    """
    if m := ATTR_Y_VAL_RE.search(atom):
        # "the person whose <attr> is <value>": start at a random person, or constrain the current one
        attr, y, value = m.groups()
        if y not in a.query:
            a.bind(y, pick(rng, lookups.names))
        return a.constrain(rng, lookups, attr, y, value)
    if m := RELATION_NAME_Y_RE.search(atom):
        # "the <relation> of <name>": start at a random person
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
    if m := RELATION_Y_Y_RE.search(atom):
        # "the <relation> of ...": follow a random relation of the current person
        relation, y1, y2 = m.groups()
        assert y1 in a.query and y2 not in a.query, f"{atom}: {y1} should be bound and {y2} free"
        options = lookups.related_to(a.person(y1))
        if not options:
            return False
        rel, other = pick(rng, options)
        a.query[relation] = rel
        a.bind(y2, other)
        a.question[relation] = RELATION_ALIAS[rel]
        return True
    if m := ATTR_Y_Y_RE.search(atom):
        # "What is the <attr> of ...": the last step
        attr, y1, y2 = m.groups()
        assert y1 in a.query and y2 not in a.query, f"{atom}: {y1} should be bound and {y2} free"
        options = lookups.attributes_of(a.person(y1))
        if not options:
            return False
        name, _ = pick(rng, options)
        a.query[attr] = name
        a.question[attr] = ATTRIBUTE_ALIASES[name]
        return True
    if m := COUNT_NAME_Y_RE.search(atom):
        # "How many <relations> does <name> have": the only step
        relation_plural, name, _, _ = m.groups()
        person = pick(rng, lookups.names)
        a.query[name] = f'"{person}"'
        options = lookups.related_to(person)
        if not options:
            return False
        rel, _ = pick(rng, options)
        a.query[relation_plural] = rel
        a.question[relation_plural], a.question[name] = RELATION_PLURAL_ALIAS[rel], person
        return True
    if m := COUNT_Y_Y_RE.search(atom):
        # "How many <relations> does ... have": a relation the current person has
        relation_plural, y1, y2, _ = m.groups()
        assert y1 in a.query and y2 not in a.query, f"{atom}: {y1} should be bound and {y2} free"
        options = lookups.related_to(a.person(y1))
        if not options:
            return False
        rel, other = pick(rng, options)
        a.query[relation_plural] = rel
        a.bind(y2, other)
        a.question[relation_plural] = RELATION_PLURAL_ALIAS[rel]
        return True
    if all(p in a.query for p in PLACEHOLDER_RE.findall(atom)):
        # Comparison plumbing (dob, @<) whose placeholders, if any, the chains fill
        return True
    raise ValueError(f"Cannot sample {atom!r}")


def sample_backward(
    template: Template, rng: Generator, lookups: Lookups, attempts: int
) -> tuple[str, list[str], dict[str, str]] | None:
    """Sample a question by walking the query from its last atom to its first.

    Args:
        template: The template to fill.
        rng: Random generator.
        lookups: The universe's lookups.
        attempts: Walks to try before giving up.

    Returns:
        The question, the query (outermost atom first) and the people the placeholders resolved
        to, or None if every attempt hit a dead end.
    """
    for _ in range(attempts):
        # A fresh walk; the first dead end ends it
        a = Assignments()
        if all(walk_atom(atom, a, rng, lookups) for atom in reversed(template.query)):
            return *a.fill(template), a.bindings()
    return None
