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

Each rule lists its pairs in the order SWI-Prolog enumerates the solutions of the PhantomWiki rule
it comes from: the rule's conditions become nested loops, left to right, each over a relation in its
own order. A seed therefore gives the same environment as PhantomWiki's Prolog generator did, which
`tests/test_generator.py` checks against recorded content hashes.
"""

import inspect
from collections.abc import Callable, Iterator
from functools import cached_property

from phantom_envs.generator.vocab import PERSON_TYPE

Pair = tuple[str, str]
# A fact is a predicate followed by its arguments, e.g. ("parent", "Ann Smith", "Beth Smith")
Fact = tuple[str, ...]
Rule = Callable[..., list[Pair]]

# Facts with two arguments, which the database serves as relations just like the derived ones:
# type(X, person), gender(X, G), parent(X, P), dob(X, D), friend_(X, F), job(X, J), hobby(X, H)
FACT_RELATIONS = ("type", "gender", "parent", "dob", "friend_", "job", "hobby")
# Relation name -> the rule that derives it, filled by @rule
RULES: dict[str, Rule] = {}


def rule(function: Rule) -> Rule:
    """Register a function as the rule that derives the relation of the same name.

    Args:
        function: Takes the relations named by its parameters and returns the new relation's pairs.

    Returns:
        The function, unchanged.
    """
    RULES[function.__name__] = function
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
        # The type is the one argument written without quotes: type("Ann Smith", person)
        return f'type("{arguments[0]}", {arguments[1]})'
    return f"{predicate}({', '.join(f'"{argument}"' for argument in arguments)})"


class Relation:
    """An ordered list of pairs `(x, y)`, each read "y is x's <relation>".

    Args:
        pairs: The pairs, in the order lookups should return them. Repeats are kept.
    """

    def __init__(self, pairs: list[Pair]) -> None:
        self.pairs = pairs

    def __iter__(self) -> Iterator[Pair]:
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    @cached_property
    def by_first(self) -> dict[str, list[str]]:
        """Each x with its y's, in the order of `pairs`. Built on first use."""
        index: dict[str, list[str]] = {}
        for x, y in self.pairs:
            index.setdefault(x, []).append(y)
        return index

    @cached_property
    def by_second(self) -> dict[str, list[str]]:
        """Each y with its x's, in the order of `pairs`. Built on first use."""
        index: dict[str, list[str]] = {}
        for x, y in self.pairs:
            index.setdefault(y, []).append(x)
        return index

    def of(self, x: str) -> list[str]:
        """Every y with `(x, y)` in the relation: for the mother relation, the mother of x."""
        return self.by_first.get(x, [])

    def inverse_of(self, y: str) -> list[str]:
        """Every x with `(x, y)` in the relation: for the mother relation, the people whose mother is y."""
        return self.by_second.get(y, [])

    def has(self, x: str, y: str) -> bool:
        """Whether `(x, y)` is in the relation."""
        return y in self.of(x)


class Database:
    """The facts of a universe and, on demand, every relation that follows from them."""

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
            KeyError: If no rule or fact has that name.
        """
        if name not in self.relations:
            if name in RULES:
                # A rule's parameters name the relations it reads, which are looked up the same way
                derive = RULES[name]
                inputs = [self[parameter] for parameter in inspect.signature(derive).parameters]
                pairs = derive(*inputs)
            elif name in FACT_RELATIONS:
                pairs = [(fact[1], fact[2]) for fact in self.facts if fact[0] == name]
            else:
                raise KeyError(f"No rule or fact is named {name!r}")
            self.relations[name] = Relation(pairs)
        return self.relations[name]

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
    in the number of people, so this reaches the same pairs through `a`'s relatives, and sorts
    each x's matches back into the order of the parent facts.

    Args:
        parent: The parent facts.
        related: How the two parents are related: siblings for cousins, cousins for second cousins.

    Returns:
        The pairs `(x, y)`, in the order (and with the repeats) of the plain loop. It includes
        `(x, x)` where a person's two parents are related, which the rules filter out.
    """
    position = {fact: i for i, fact in enumerate(parent)}
    pairs = []
    for x, a in parent:
        matches = sorted((position[(y, b)], y) for b in related.of(a) for y in parent.inverse_of(b))
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
def female_second_cousin(parent: Relation, cousin: Relation, gender: Relation) -> list[Pair]:
    """Y is a female second cousin of X: their parents are cousins."""
    second_cousins = children_of_related_parents(parent, cousin)
    return [(x, y) for x, y in second_cousins if gender.has(y, "female") and x != y]


@rule
def male_second_cousin(parent: Relation, cousin: Relation, gender: Relation) -> list[Pair]:
    """Y is a male second cousin of X: their parents are cousins."""
    second_cousins = children_of_related_parents(parent, cousin)
    return [(x, y) for x, y in second_cousins if gender.has(y, "male") and x != y]


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
    """Y is a friend of X. A `friend_` fact states each friendship once; this lists it both ways."""
    return friend_.pairs + [(y, x) for x, y in friend_]
