"""A universe's facts, and the family and friendship relations that follow from them.

A universe starts as *facts*: who exists, and each person's gender, parents, date of birth, friends,
job and hobby. Every other relation (sibling, aunt, mother-in-law, ...) follows from a *rule*, a
short function in the second half of this module:

    @rule
    def mother(parent: Relation, gender: Relation) -> list[Pair]:
        '''Y is the mother of X: a female parent.'''
        return [(x, y) for x, y in parent if gender.has(y, "female")]

A rule is named after the relation it derives, and its parameters name the relations it reads,
which are facts or the results of other rules. A pair `(x, y)` of a relation R always reads
"y is x's R". The `Database` runs a rule the first time its relation is asked for, and keeps
the result as a `Relation`, a list of pairs that can be looked up from either side:

    db["mother"].of("Ann Smith")            # ["Beth Smith"]
    db["mother"].inverse_of("Beth Smith")   # everyone whose mother is Beth Smith

To add a relation, write its rule here; to ask questions about it, also list it in `vocab.py`.

**The order of a relation's pairs, and their repeats, are part of the generated data.** The question
sampler picks a person's relatives by position in these lists, so:

- changing the order of a rule's loops changes the questions a seed gives;
- a pair listed twice is drawn twice as often. Full siblings are listed once per shared parent,
  a married couple once per child, and so on. Rules never remove these repeats.

Each rule lists its pairs in the order SWI-Prolog enumerates the solutions of the matching
PhantomWiki rule: the rule's conditions become nested loops, left to right, each over a relation in
its own order. A seed therefore gives the same environment as PhantomWiki's generator (with its 3
friends per person), which `tests/test_generator.py` checks against recorded content hashes.
"""

import inspect
from collections.abc import Callable, Iterator

from phantom_envs.generator.vocab import PERSON_TYPE

Pair = tuple[str, str]
# A fact is a predicate followed by its arguments, e.g. ("parent", "Ann Smith", "Beth Smith")
Fact = tuple[str, ...]
Rule = Callable[..., list[Pair]]

# Facts with two arguments, which the database serves as relations just like the derived ones:
# type(X, person), gender(X, G), parent(X, P), dob(X, D), friend_(X, F), job(X, J), hobby(X, H).
# A friend_ fact states a friendship once; the relation that holds both ways is the rule "friend".
FACT_RELATIONS = ("type", "gender", "parent", "dob", "friend_", "job", "hobby")
# Relation name -> the rule that derives it, filled by @rule
RULES: dict[str, Rule] = {}


def rule(function: Rule) -> Rule:
    """Register a function as the rule that derives the relation of the same name.

    Args:
        function: Takes the relations named by its parameters and returns the new relation's pairs.

    Returns:
        The function, unchanged.

    Raises:
        ValueError: If a rule or a fact already has that name.
    """
    name = function.__name__
    if name in RULES or name in FACT_RELATIONS:
        raise ValueError(f"A rule or fact is already named {name!r}")
    RULES[name] = function
    return function


def format_fact(fact: Fact) -> str:
    """Write a fact the way `universe/facts.pl` lists it (Prolog syntax, without the period).

    Args:
        fact: A predicate and its arguments.

    Returns:
        E.g. `parent("Ann Smith", "Beth Smith")`.
    """
    predicate, *arguments = fact
    if predicate == "type":
        # A type fact says that someone is a person, the only type there is. The type is written
        # without quotes, unlike names and values: type("Ann Smith", person)
        return f'type("{arguments[0]}", {arguments[1]})'
    # Every other fact quotes all its arguments: parent("Ann Smith", "Beth Smith"), attribute("chess")
    return f"{predicate}({', '.join(f'"{argument}"' for argument in arguments)})"


class Relation:
    """An ordered list of pairs `(x, y)`, each read "y is x's <relation>".

    Args:
        pairs: The pairs, in the order lookups should return them. Repeats are kept.
    """

    def __init__(self, pairs: list[Pair]) -> None:
        self.pairs = pairs
        # Each x with its y's, and each y with its x's, both in the order of the pairs
        self.by_first: dict[str, list[str]] = {}
        self.by_second: dict[str, list[str]] = {}
        for x, y in pairs:
            self.by_first.setdefault(x, []).append(y)
            self.by_second.setdefault(y, []).append(x)

    def __iter__(self) -> Iterator[Pair]:
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def of(self, x: str) -> list[str]:
        """Every y with `(x, y)` in the relation: for the mother relation, the mother of x.

        The list is the relation's own; do not modify it.
        """
        return self.by_first.get(x, [])

    def inverse_of(self, y: str) -> list[str]:
        """Every x with `(x, y)` in the relation: for the mother relation, the people whose mother is y.

        The list is the relation's own; do not modify it.
        """
        return self.by_second.get(y, [])

    def has(self, x: str, y: str) -> bool:
        """Whether `(x, y)` is in the relation."""
        return y in self.of(x)


class Database:
    """The facts of a universe and, on demand, every relation that follows from them.

    The facts are a list, not a set, because their order is part of the data: it sets the order of
    `names` and of every relation's pairs, which the seeded random draws index into.
    """

    def __init__(self) -> None:
        self.facts: list[Fact] = []
        # Relations looked up so far, by name
        self.relations: dict[str, Relation] = {}

    def add(self, *facts: Fact) -> None:
        """Add facts to the universe.

        Args:
            *facts: Facts such as `("parent", "Ann Smith", "Beth Smith")`.
        """
        self.facts += facts
        # Relations worked out before now would miss the new facts
        self.relations.clear()

    def __getitem__(self, name: str) -> Relation:
        """Look up a relation, working it out from its rule (or the facts) on first use.

        Args:
            name: A relation with a rule, or one of `FACT_RELATIONS`.

        Returns:
            The relation.

        Raises:
            KeyError: If no rule or fact has that name, or its rule reads a relation with no rule or fact.
        """
        if name not in self.relations:
            if name in RULES:
                # A derived relation, e.g. "aunt". Its rule is the function aunt(parent, sister),
                # whose parameter names say which relations it reads: look those up the same way
                # (deriving "sister" in turn, if this is its first use), then run the rule on them.
                rule_function = RULES[name]
                parameters = inspect.signature(rule_function).parameters
                if unknown := [p for p in parameters if p not in RULES and p not in FACT_RELATIONS]:
                    raise KeyError(f"The rule {name!r} reads {unknown}, which no rule or fact is named")
                pairs = rule_function(*[self[parameter] for parameter in parameters])
            elif name in FACT_RELATIONS:
                # A relation stated by facts, e.g. "parent": the fact
                # ("parent", "Ann Smith", "Beth Smith") gives the pair ("Ann Smith", "Beth Smith").
                pairs = [(fact[1], fact[2]) for fact in self.facts if fact[0] == name]
            else:
                # E.g. a misspelled name, or "attribute", whose facts have one argument
                raise KeyError(f"No rule or fact is named {name!r}")
            self.relations[name] = Relation(pairs)
        return self.relations[name]

    def derive(self, names: list[str]) -> None:
        """Work out relations now instead of on their first use.

        Call this before handing the database to worker processes: they then share the relations
        instead of each working them out again.

        Args:
            names: The relations to work out.
        """
        for name in names:
            self[name]

    @property
    def names(self) -> list[str]:
        """Everyone in the universe, in the order they were added."""
        return self["type"].inverse_of(PERSON_TYPE)


#
# Rules: immediate family
#
@rule
def sibling(parent: Relation) -> list[Pair]:
    """Y is a sibling of X: they share a parent. Listed once per shared parent."""
    return [(x, y) for x, a in parent for y in parent.inverse_of(a) if x != y]


@rule
def married(parent: Relation) -> list[Pair]:
    """Y is married to X: they are the two parents of a child. Listed once per child."""
    return [(x, y) for child, x in parent for y in parent.of(child) if x != y]


@rule
def sister(sibling: Relation, gender: Relation) -> list[Pair]:
    """Y is a sister of X."""
    return [(x, y) for x, y in sibling if gender.has(y, "female")]


@rule
def brother(sibling: Relation, gender: Relation) -> list[Pair]:
    """Y is a brother of X."""
    return [(x, y) for x, y in sibling if gender.has(y, "male")]


@rule
def mother(parent: Relation, gender: Relation) -> list[Pair]:
    """Y is the mother of X."""
    return [(x, y) for x, y in parent if gender.has(y, "female")]


@rule
def father(parent: Relation, gender: Relation) -> list[Pair]:
    """Y is the father of X."""
    return [(x, y) for x, y in parent if gender.has(y, "male")]


@rule
def child(parent: Relation) -> list[Pair]:
    """Y is a child of X."""
    return [(x, y) for y, x in parent]


@rule
def son(child: Relation, gender: Relation) -> list[Pair]:
    """Y is a son of X."""
    return [(x, y) for x, y in child if gender.has(y, "male")]


@rule
def daughter(child: Relation, gender: Relation) -> list[Pair]:
    """Y is a daughter of X."""
    return [(x, y) for x, y in child if gender.has(y, "female")]


@rule
def wife(married: Relation, gender: Relation) -> list[Pair]:
    """Y is the wife of X."""
    return [(x, y) for x, y in married if gender.has(y, "female")]


@rule
def husband(married: Relation, gender: Relation) -> list[Pair]:
    """Y is the husband of X."""
    return [(x, y) for x, y in married if gender.has(y, "male")]


#
# Rules: extended family
#
@rule
def niece(sibling: Relation, daughter: Relation) -> list[Pair]:
    """Y is a niece of X: a daughter of X's sibling."""
    return [(x, y) for x, a in sibling for y in daughter.of(a)]


@rule
def nephew(sibling: Relation, son: Relation) -> list[Pair]:
    """Y is a nephew of X: a son of X's sibling."""
    return [(x, y) for x, a in sibling for y in son.of(a)]


@rule
def grandparent(parent: Relation) -> list[Pair]:
    """Y is a grandparent of X: a parent of X's parent."""
    return [(x, y) for x, a in parent for y in parent.of(a)]


@rule
def grandmother(grandparent: Relation, gender: Relation) -> list[Pair]:
    """Y is a grandmother of X."""
    return [(x, y) for x, y in grandparent if gender.has(y, "female")]


@rule
def grandfather(grandparent: Relation, gender: Relation) -> list[Pair]:
    """Y is a grandfather of X."""
    return [(x, y) for x, y in grandparent if gender.has(y, "male")]


@rule
def great_aunt(grandparent: Relation, sister: Relation) -> list[Pair]:
    """Y is a great-aunt of X: a sister of X's grandparent."""
    return [(x, y) for x, a in grandparent for y in sister.of(a)]


@rule
def great_uncle(grandparent: Relation, brother: Relation) -> list[Pair]:
    """Y is a great-uncle of X: a brother of X's grandparent."""
    return [(x, y) for x, a in grandparent for y in brother.of(a)]


@rule
def grandchild(grandparent: Relation) -> list[Pair]:
    """Y is a grandchild of X."""
    return [(x, y) for y, x in grandparent]


@rule
def granddaughter(grandchild: Relation, gender: Relation) -> list[Pair]:
    """Y is a granddaughter of X."""
    return [(x, y) for x, y in grandchild if gender.has(y, "female")]


@rule
def grandson(grandchild: Relation, gender: Relation) -> list[Pair]:
    """Y is a grandson of X."""
    return [(x, y) for x, y in grandchild if gender.has(y, "male")]


@rule
def great_grandparent(grandparent: Relation, parent: Relation) -> list[Pair]:
    """Y is a great-grandparent of X: a parent of X's grandparent."""
    return [(x, y) for x, a in grandparent for y in parent.of(a)]


@rule
def great_grandmother(great_grandparent: Relation, gender: Relation) -> list[Pair]:
    """Y is a great-grandmother of X."""
    return [(x, y) for x, y in great_grandparent if gender.has(y, "female")]


@rule
def great_grandfather(great_grandparent: Relation, gender: Relation) -> list[Pair]:
    """Y is a great-grandfather of X."""
    return [(x, y) for x, y in great_grandparent if gender.has(y, "male")]


@rule
def great_grandchild(great_grandparent: Relation) -> list[Pair]:
    """Y is a great-grandchild of X."""
    return [(x, y) for y, x in great_grandparent]


@rule
def great_granddaughter(great_grandchild: Relation, gender: Relation) -> list[Pair]:
    """Y is a great-granddaughter of X."""
    return [(x, y) for x, y in great_grandchild if gender.has(y, "female")]


@rule
def great_grandson(great_grandchild: Relation, gender: Relation) -> list[Pair]:
    """Y is a great-grandson of X."""
    return [(x, y) for x, y in great_grandchild if gender.has(y, "male")]


@rule
def second_aunt(great_grandparent: Relation, sister: Relation) -> list[Pair]:
    """Y is a second aunt of X: a sister of X's great-grandparent."""
    return [(x, y) for x, a in great_grandparent for y in sister.of(a)]


@rule
def second_uncle(great_grandparent: Relation, brother: Relation) -> list[Pair]:
    """Y is a second uncle of X: a brother of X's great-grandparent."""
    return [(x, y) for x, a in great_grandparent for y in brother.of(a)]


@rule
def aunt(parent: Relation, sister: Relation) -> list[Pair]:
    """Y is an aunt of X: a sister of X's parent."""
    return [(x, y) for x, a in parent for y in sister.of(a)]


@rule
def uncle(parent: Relation, brother: Relation) -> list[Pair]:
    """Y is an uncle of X: a brother of X's parent."""
    return [(x, y) for x, a in parent for y in brother.of(a)]


#
# Rules: cousins
#
def children_of_related_parents(parent: Relation, related: Relation) -> list[Pair]:
    """Pair up people whose parents are related, for the cousin rules.

    The plain form of this rule is a loop over every two parent facts:

        [(x, y) for x, a in parent for y, b in parent if related.has(a, b)]

    with `(x, y)` repeated when `(a, b)` is repeated in `related`. That takes time quadratic
    in the number of people, so this reaches the same parent facts through `a`'s relatives, and
    sorts each x's matches back into the order of the parent facts.

    Args:
        parent: The parent facts.
        related: How the two parents are related: siblings for cousins, cousins for second cousins.

    Returns:
        The pairs `(x, y)`, in the order (and with the repeats) of the plain loop. It includes
        `(x, x)` where a person's two parents are related, which the rules filter out.
    """
    # Each person's children, with the position of the parent fact that states each
    children: dict[str, list[tuple[int, str]]] = {}
    for position, (y, b) in enumerate(parent):
        children.setdefault(b, []).append((position, y))

    pairs = []
    for x, a in parent:
        matches = sorted(child for b in related.of(a) for child in children.get(b, []))
        pairs += [(x, y) for _, y in matches]
    return pairs


@rule
def cousin(parent: Relation, sibling: Relation) -> list[Pair]:
    """Y is a cousin of X: their parents are siblings."""
    return [(x, y) for x, y in children_of_related_parents(parent, sibling) if x != y]


@rule
def female_cousin(cousin: Relation, gender: Relation) -> list[Pair]:
    """Y is a female cousin of X."""
    return [(x, y) for x, y in cousin if gender.has(y, "female")]


@rule
def male_cousin(cousin: Relation, gender: Relation) -> list[Pair]:
    """Y is a male cousin of X."""
    return [(x, y) for x, y in cousin if gender.has(y, "male")]


@rule
def second_cousin(parent: Relation, cousin: Relation) -> list[Pair]:
    """Y is a second cousin of X: their parents are cousins."""
    return [(x, y) for x, y in children_of_related_parents(parent, cousin) if x != y]


@rule
def female_second_cousin(second_cousin: Relation, gender: Relation) -> list[Pair]:
    """Y is a female second cousin of X."""
    return [(x, y) for x, y in second_cousin if gender.has(y, "female")]


@rule
def male_second_cousin(second_cousin: Relation, gender: Relation) -> list[Pair]:
    """Y is a male second cousin of X."""
    return [(x, y) for x, y in second_cousin if gender.has(y, "male")]


@rule
def female_first_cousin_once_removed(cousin: Relation, daughter: Relation) -> list[Pair]:
    """Y is a daughter of X's cousin."""
    return [(x, y) for x, a in cousin for y in daughter.of(a) if x != y]


@rule
def male_first_cousin_once_removed(cousin: Relation, son: Relation) -> list[Pair]:
    """Y is a son of X's cousin."""
    return [(x, y) for x, a in cousin for y in son.of(a) if x != y]


#
# Rules: in-laws
#
@rule
def mother_in_law(married: Relation, mother: Relation) -> list[Pair]:
    """Y is the mother-in-law of X: the mother of X's spouse."""
    return [(x, y) for x, a in married for y in mother.of(a)]


@rule
def father_in_law(married: Relation, father: Relation) -> list[Pair]:
    """Y is the father-in-law of X: the father of X's spouse."""
    return [(x, y) for x, a in married for y in father.of(a)]


@rule
def son_in_law(child: Relation, husband: Relation) -> list[Pair]:
    """Y is a son-in-law of X: the husband of X's child."""
    return [(x, y) for x, a in child for y in husband.of(a)]


@rule
def daughter_in_law(child: Relation, wife: Relation) -> list[Pair]:
    """Y is a daughter-in-law of X: the wife of X's child."""
    return [(x, y) for x, a in child for y in wife.of(a)]


@rule
def sister_in_law(married: Relation, sister: Relation) -> list[Pair]:
    """Y is a sister-in-law of X: a sister of X's spouse."""
    return [(x, y) for x, a in married for y in sister.of(a)]


@rule
def brother_in_law(married: Relation, brother: Relation) -> list[Pair]:
    """Y is a brother-in-law of X: a brother of X's spouse."""
    return [(x, y) for x, a in married for y in brother.of(a)]


#
# Rules: friends
#
@rule
def friend(friend_: Relation) -> list[Pair]:
    """Y is a friend of X.

    A `friend_` fact states a friendship once, as (X, Y) or as (Y, X). This lists it both ways, so
    that each is the other's friend.
    """
    return friend_.pairs + [(y, x) for x, y in friend_]
