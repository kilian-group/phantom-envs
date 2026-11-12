"""Generate a universe: family trees, a friendship graph, and each person's job and hobby.

The family-tree generator is adapted from Patrick Hohenecker's family-tree-data-gen
(https://github.com/phohenecker/family-tree-data-gen, BSD-2-Clause, Copyright (C) 2018).
Every random draw happens in the same order as in PhantomWiki, so a seed gives the same universe.
"""

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib.resources import files

import networkx as nx
from numpy.random import default_rng

from phantom_envs.generator.prolog import Database
from phantom_envs.generator.vocab import PERSON_TYPE

FAMILY_TREE_SIZE = 10
MAX_FAMILY_TREE_DEPTH = 10
MAX_BRANCHING_FACTOR = 5
# Average number of friends per person
FRIENDSHIP_DEGREE = 3


def load_data(name: str):
    """Load one of the JSON files in ``generator/data``.

    Args:
        name: File name without the ``.json`` extension, e.g. ``"last_names"``.

    Returns:
        The parsed JSON.
    """
    return json.loads(files("phantom_envs.generator").joinpath(f"data/{name}.json").read_text())


@dataclass
class Person:
    """A person in a family tree."""

    index: int = field(metadata={"description": "Creation order, from 1."})
    name: str = field(metadata={"description": "First name."})
    surname: str = field(metadata={"description": "Last name."})
    female: bool = field(metadata={"description": "Gender (female or male)."})
    tree_level: int = field(metadata={"description": "Generation in the tree; children are one level down."})
    date_of_birth: date = field(metadata={"description": "Date of birth."})
    children: list["Person"] = field(default_factory=list, metadata={"description": "Children."})
    parents: list["Person"] = field(default_factory=list, metadata={"description": "Mother and father."})
    married_to: "Person | None" = field(default=None, metadata={"description": "Spouse, if any."})

    def full_name(self) -> str:
        """First and last name, the person's unique id in the universe."""
        return f"{self.name} {self.surname}"


class PersonFactory:
    """Creates people with unique full names and plausible dates of birth.

    Args:
        rng: Random generator shared with the tree sampler, so draws interleave as in PhantomWiki.
    """

    MIN_PARENT_AGE = 18
    MAX_PARENT_AGE = 45
    AVG_PARENT_AGE = 27
    MAX_PARENT_AGE_DIFF = 10
    TWIN_PROBABILITY = 0.05
    AVG_TWIN_TIME_DIFF = 30
    STD_TWIN_TIME_DIFF = 10
    SPOUSE_AGE_DIFF_STD = 2
    PARENT_CHILD_AGE_DIFF_STD = 3
    MIN_DAYS_SIBLING_DIFF = 300

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.person_counter = 0
        female_names, male_names = load_data("female_names"), load_data("male_names")
        last_names = load_data("last_names")
        # Unused first names per (last name, female), so full names stay unique
        self.remaining_names = {s: {True: female_names.copy(), False: male_names.copy()} for s in last_names}
        # Last names with first names left for women (True), men (False), and both (None)
        self.remaining_last_names = {
            True: last_names.copy(),
            False: last_names.copy(),
            None: last_names.copy(),
        }

    def last_name(self, female: bool | None = None) -> str:
        """Draw a last name that still has unused first names.

        Args:
            female: Gender the name is for; None for a name with first names left for both.

        Returns:
            The last name.

        Raises:
            ValueError: If every last name is used up.
        """
        pool = self.remaining_last_names[female]
        if not pool:
            raise ValueError("Ran out of names: generate a smaller universe")
        return self.rng.choice(pool)

    def first_name(self, female: bool, surname: str) -> str:
        """Draw and use up a first name for a last name and gender.

        Args:
            female: Gender of the person.
            surname: Their last name.

        Returns:
            The first name.

        Raises:
            ValueError: If the last name has no first names left.
        """
        pool = self.remaining_names[surname][female]
        if not pool:
            raise ValueError("Ran out of names: generate a smaller universe")
        name = pool.pop(self.rng.randrange(len(pool)))
        # Last first name taken: retire the last name for this gender
        if not pool:
            self.remaining_last_names[female].remove(surname)
            self.remaining_last_names[None].remove(surname)
        return name

    def random_dob(self, year: int) -> date:
        """A random date in ``year`` (day 28 at most, valid in every month)."""
        return date(year, self.rng.randint(1, 12), self.rng.randint(1, 28))

    def create_parents(self, tree_level: int, child: Person) -> list[Person]:
        """Create the parents of a person.

        Args:
            tree_level: The parents' generation (one above the child's).
            child: The person to create parents for.

        Returns:
            The father and the mother.
        """
        # Birth years: about AVG_PARENT_AGE before the child's, within the parent age limits
        yob_child = child.date_of_birth.year
        yob_1 = int(self.rng.gauss(yob_child - self.AVG_PARENT_AGE, self.PARENT_CHILD_AGE_DIFF_STD))
        yob_2 = int(self.rng.gauss(yob_1, self.SPOUSE_AGE_DIFF_STD))
        yob_1 = min(max(yob_1, yob_child - self.MAX_PARENT_AGE), yob_child - self.MIN_PARENT_AGE)
        yob_2 = min(max(yob_2, yob_child - self.MAX_PARENT_AGE), yob_child - self.MIN_PARENT_AGE)
        yob_2 = min(max(yob_2, yob_1 - self.MAX_PARENT_AGE_DIFF), yob_1 + self.MAX_PARENT_AGE_DIFF)
        dob_1, dob_2 = self.random_dob(yob_1), self.random_dob(yob_2)

        # A married woman took her husband's name, so her parents get a new one
        surname = self.last_name() if child.female and child.married_to else child.surname
        return [
            self.create_person(tree_level, surname, dob_1, female=False),
            self.create_person(tree_level, surname, dob_2, female=True),
        ]

    def create_child(self, tree_level: int, parents: list[Person], siblings: list[Person]) -> Person:
        """Create a child, a twin of a sibling or born at least ``MIN_DAYS_SIBLING_DIFF`` days apart.

        Args:
            tree_level: The child's generation (one below the parents').
            parents: The two parents.
            siblings: The parents' existing children.

        Returns:
            The child, with the father's last name.
        """
        max_parent_dob = max(parents[0].date_of_birth, parents[1].date_of_birth)
        min_parent_dob = min(parents[0].date_of_birth, parents[1].date_of_birth)

        if siblings and self.rng.random() < self.TWIN_PROBABILITY:
            # A twin: born minutes after a sibling, so on the same date
            twin = self.rng.choice(siblings)
            delta_minutes = max(int(self.rng.gauss(self.AVG_TWIN_TIME_DIFF, self.STD_TWIN_TIME_DIFF)), 1)
            child_dob = twin.date_of_birth + timedelta(minutes=delta_minutes)
        else:
            # Born when both parents are within the parent age limits (day 28 at most, so that
            # adding years never lands on Feb 29 of a non-leap year)
            min_dob = date(
                max_parent_dob.year + self.MIN_PARENT_AGE, max_parent_dob.month, min(max_parent_dob.day, 28)
            )
            mean_dob = date(
                max_parent_dob.year + self.AVG_PARENT_AGE, max_parent_dob.month, min(max_parent_dob.day, 28)
            )
            max_dob = date(
                min_parent_dob.year + self.MAX_PARENT_AGE, min_parent_dob.month, min(min_parent_dob.day, 28)
            )
            # Days since 0001-01-01 that are out of range or too close to a sibling
            start = date(1, 1, 1)
            mean_days = (mean_dob - start).days
            invalid = [(0, (min_dob - start).days), ((max_dob - start).days, float("inf"))]
            for sibling in siblings:
                sib_days = (sibling.date_of_birth - start).days
                invalid.append((sib_days - self.MIN_DAYS_SIBLING_DIFF, sib_days + self.MIN_DAYS_SIBLING_DIFF))

            # Redraws once per invalid interval it falls in, in a single pass (as in PhantomWiki)
            child_days = int(self.rng.gauss(mean_days, 365 * self.PARENT_CHILD_AGE_DIFF_STD))
            for low, high in invalid:
                if low <= child_days <= high:
                    child_days = int(self.rng.gauss(mean_days, 365 * self.PARENT_CHILD_AGE_DIFF_STD))
            child_dob = start + timedelta(days=child_days)

        father = parents[1] if parents[0].female else parents[0]
        return self.create_person(tree_level, father.surname, child_dob)

    def create_spouse(self, tree_level: int, female: bool, spouse: Person) -> Person:
        """Create a person's spouse; a new husband gives both a new last name.

        Args:
            tree_level: The spouse's generation (the person's).
            female: Gender of the new spouse.
            spouse: The person to marry; renamed when the new spouse is a husband.

        Returns:
            The new spouse.
        """
        # Born within a few years of the person
        yob = spouse.date_of_birth.year
        spouse_yob = int(self.rng.gauss(yob, self.SPOUSE_AGE_DIFF_STD))
        spouse_yob = min(max(spouse_yob, yob - self.MAX_PARENT_AGE_DIFF), yob + self.MAX_PARENT_AGE_DIFF)
        spouse_dob = self.random_dob(spouse_yob)

        if female:
            surname = spouse.surname
        else:
            surname = self.last_name(female)
            # The wife gives back her first name under her old last name and takes one under the new
            self.remaining_names[spouse.surname][spouse.female].append(spouse.name)
            spouse.surname = surname
            spouse.name = self.first_name(spouse.female, spouse.surname)
        return self.create_person(tree_level, surname, spouse_dob, female)

    def create_person(
        self, tree_level: int, surname: str | None = None, dob: date | None = None, female: bool | None = None
    ) -> Person:
        """Create a person, drawing the fields left unset.

        Args:
            tree_level: The person's generation; sets the birth year when ``dob`` is None.
            surname: Last name, or None to draw one.
            dob: Date of birth, or None to draw one.
            female: Gender, or None to draw one.

        Returns:
            The person, with a first name unused under their last name.
        """
        if dob is None:
            year = tree_level * (self.MAX_PARENT_AGE + 1) + self.rng.randint(1, self.MAX_PARENT_AGE)
            dob = self.random_dob(year)
        if female is None:
            female = self.rng.random() > 0.5
        if surname is None:
            surname = self.last_name(female)
        name = self.first_name(female, surname)
        self.person_counter += 1
        return Person(self.person_counter, name, surname, female, tree_level, dob)


def sample_family_tree(factory: PersonFactory, rng: random.Random) -> list[Person]:
    """Grow a family tree from one person by adding parents or children at random.

    Args:
        factory: Creates the people.
        rng: Random generator shared with ``factory``.

    Returns:
        The tree's people, in creation order (about ``FAMILY_TREE_SIZE``; spouses can push it over).
    """
    tree = [factory.create_person(MAX_FAMILY_TREE_DEPTH)]
    min_level = max_level = tree[0].tree_level
    num_people = 1
    attempts = 0
    while True:
        # Pick a person; they can get parents or another child unless that deepens a full-depth tree
        person = rng.choice(tree)
        depth = max_level - min_level
        can_add_parents = not person.parents and (
            person.tree_level > min_level or depth < MAX_FAMILY_TREE_DEPTH
        )
        can_add_child = len(person.children) < MAX_BRANCHING_FACTOR and (
            person.tree_level < max_level or depth < MAX_FAMILY_TREE_DEPTH
        )
        if can_add_parents and can_add_child:
            add_parents = rng.random() > 0.5
            add_child = not add_parents
        else:
            add_parents, add_child = can_add_parents, can_add_child

        if add_child:
            # Marry the person first if needed, then add a child of the couple
            if person.married_to:
                spouse = person.married_to
            else:
                spouse = factory.create_spouse(person.tree_level, female=not person.female, spouse=person)
                spouse.married_to = person
                person.married_to = spouse
                tree.append(spouse)
                num_people += 1
            child = factory.create_child(person.tree_level + 1, [person, spouse], person.children)
            child.parents = [person, spouse]
            tree.append(child)
            person.children.append(child)
            spouse.children.append(child)
            max_level = max(max_level, child.tree_level)
            num_people += 1
        elif add_parents:
            dad, mom = factory.create_parents(person.tree_level - 1, person)
            mom.married_to, dad.married_to = dad, mom
            mom.children.append(person)
            dad.children.append(person)
            person.parents = [mom, dad]
            tree.extend([mom, dad])
            num_people += 2
            min_level = min(min_level, mom.tree_level)

        attempts += 1
        if num_people >= FAMILY_TREE_SIZE or attempts >= FAMILY_TREE_SIZE * 10:
            return tree


def family_facts(tree: list[Person]) -> list[str]:
    """Prolog facts of a family tree.

    Args:
        tree: The tree's people.

    Returns:
        ``type``, ``gender``, ``parent`` and ``dob`` facts (without periods), each group sorted.
    """
    people, genders, parents, dobs = [], [], [], []
    for p in tree:
        people.append(f'type("{p.full_name()}", {PERSON_TYPE})')
        genders.append(f'gender("{p.full_name()}", "{"female" if p.female else "male"}")')
        parents += [f'parent("{c.full_name()}", "{p.full_name()}")' for c in p.children]
        dobs.append(f'dob("{p.full_name()}", "{p.date_of_birth}")')
    return sorted(people) + sorted(genders) + sorted(parents) + sorted(dobs)


def friendship_facts(names: list[str], seed: int) -> list[str]:
    """Friendships from an Erdős–Rényi G(n, p) graph with ``FRIENDSHIP_DEGREE`` friends on average.

    Args:
        names: Everyone in the universe, in database order.
        seed: Seed for the graph.

    Returns:
        ``friend_`` facts (without periods); ``friend`` makes them symmetric.
    """
    graph = nx.fast_gnp_random_graph(n=len(names), p=FRIENDSHIP_DEGREE / len(names), seed=seed)
    # Relabeling rebuilds the graph, which can reorder its edges (and so Prolog's answers)
    graph = nx.relabel_nodes(graph, dict(enumerate(names)))
    return [f'friend_("{a}", "{b}")' for a, b in graph.edges]


def attribute_facts(names: list[str], seed: int) -> list[str]:
    """A job and a hobby for each person.

    Args:
        names: Everyone in the universe, in database order.
        seed: Seed for the draws (jobs and hobbies each start from it).

    Returns:
        ``job``, ``hobby`` and ``attribute`` facts (without periods).
    """
    jobs, hobbies = load_data("jobs"), load_data("hobbies")
    job_rng, hobby_rng = default_rng(seed), default_rng(seed)
    job_of = {name: job_rng.choice(jobs) for name in names}
    # A hobby category first, then a hobby in it
    hobby_of = {name: hobby_rng.choice(hobbies[hobby_rng.choice(list(hobbies))]) for name in names}
    facts = []
    for name in names:
        facts += [f'job("{name}", "{job_of[name]}")', f'attribute("{job_of[name]}")']
        facts += [f'hobby("{name}", "{hobby_of[name]}")', f'attribute("{hobby_of[name]}")']
    return facts


def build_universe(db: Database, size: int, seed: int) -> list[str]:
    """Generate a universe of about ``size`` people into a database.

    Args:
        db: An empty database.
        size: Target number of people, a multiple of ``FAMILY_TREE_SIZE``. Spouses push a tree past it.
        seed: Seed for the family trees, the friendships and the attributes.

    Returns:
        The facts added (without periods), in order.
    """
    # Family trees, all drawing from one generator
    rng = random.Random(seed)
    factory = PersonFactory(rng)
    family = []
    for _ in range(size // FAMILY_TREE_SIZE):
        family += family_facts(sample_family_tree(factory, rng))
    db.add(*family)

    # Friendships and attributes over everyone, in database order
    names = db.person_names()
    friends = friendship_facts(names, seed)
    db.add(*friends)
    attributes = attribute_facts(names, seed)
    db.add(*attributes)
    return family + friends + attributes
