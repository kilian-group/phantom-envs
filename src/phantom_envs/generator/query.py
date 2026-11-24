"""Answer a question's query from the database.

A query is a list of conditions ("atoms") that must all hold, written in Prolog syntax (the form
templates, difficulty scores and task metadata use). An atom `relation(first, second)` reads
"second is first's relation". Its arguments are variables (`Y_1`) or quoted constants (`"chess"`).

    Who is the mother of the friend of the person whose hobby is chess?

    query:  ['mother(Y_2, Y_1)', 'friend(Y_3, Y_2)', 'hobby(Y_3, "chess")']      answer variable: Y_1

A solution gives every variable a value so that all atoms hold. The answers are the values of the
answer variable over all solutions. `answer_query` finds them by taking one atom at a time and
keeping the solutions of the atoms taken so far, each a dict of variable -> value:

    start                     [{}]                                one solution, nothing known yet
    hobby(Y_3, "chess")       [{Y_3: Gus}, {Y_3: Ivy}]            who has chess as a hobby?
    friend(Y_3, Y_2)          [{Y_2: Ivy}, {Y_2: Gus}]            their friends (Y_3 is forgotten)
    mother(Y_2, Y_1)          [{Y_1: Ann}, {Y_1: Fay}]            those friends' mothers
    answers                   ["Ann", "Fay"]

Two choices keep this fast:

- The order of the atoms does not change the answers, so the next atom is always one that starts
  from something already known (a constant, or a variable of the solutions so far). Taking
  'mother(Y_2, Y_1)' first would list every mother and child in the universe.
- A variable that no later atom mentions is forgotten, and solutions that have become equal are
  merged. Above, two people with the same friend continue as one solution, so a long chain
  carries the people reached at each step, not every path to them.

A "How many" question starts with one atom of another kind, which sets `Count_1` to the number of
distinct sisters of `Y_2`. The rest of the query finds the people `Y_2` can be, as above, and the
answers are their counts:

    ['aggregate_all(count, distinct(sister(Y_2, Y_3)), Count_1)', 'hobby(Y_2, "chess")']
"""

from dataclasses import dataclass, field

from phantom_envs.generator.difficulty import parse_predicate
from phantom_envs.generator.relations import Database

# One solution of the atoms taken so far: the value of each variable they mention
Solution = dict[str, str]


@dataclass
class Atom:
    """One condition of a query, `relation(first, second)`: second is first's relation."""

    relation: str = field(metadata={"description": "Name of the relation."})
    first: str = field(metadata={"description": 'A variable (Y_3) or a quoted constant ("Ann Smith").'})
    second: str = field(metadata={"description": "A variable or a quoted constant."})


def is_constant(argument: str) -> bool:
    """Whether an atom's argument is a constant in double quotes, not a variable."""
    return len(argument) >= 2 and argument[0] == argument[-1] == '"'


def parse_atom(text: str) -> Atom:
    """Parse an atom `relation(first, second)`.

    Args:
        text: The atom.

    Returns:
        The parsed atom.

    Raises:
        ValueError: If the text has another form, or uses one variable for both arguments.
    """
    relation, arguments = parse_predicate(text)
    well_formed = (
        relation is not None
        and len(arguments) == 2
        and all(is_constant(argument) or argument.isidentifier() for argument in arguments)
        and arguments[0] != arguments[1]
    )
    if not well_formed:
        raise ValueError(f"Cannot answer the query atom {text!r}")
    return Atom(relation, arguments[0], arguments[1])


def parse_count(text: str) -> tuple[Atom, str]:
    """Parse a counting atom `aggregate_all(count, distinct(relation(first, second)), Count)`.

    Args:
        text: The atom.

    Returns:
        The atom whose seconds are counted, and the variable that gets the count.

    Raises:
        ValueError: If the text has another form.
    """
    aggregate, arguments = parse_predicate(text)
    if aggregate == "aggregate_all" and len(arguments) == 3 and arguments[0] == "count":
        distinct, counted = parse_predicate(arguments[1])
        if distinct == "distinct" and len(counted) == 1 and arguments[2].isidentifier():
            return parse_atom(counted[0]), arguments[2]
    raise ValueError(f"Cannot answer the query atom {text!r}")


def value_of(argument: str, solution: Solution) -> str | None:
    """Look up what an atom's argument stands for in a solution.

    Args:
        argument: A variable or a quoted constant.
        solution: The values known so far.

    Returns:
        The constant without its quotes, the variable's value, or None for a variable not known yet.
    """
    if is_constant(argument):
        return argument[1:-1]
    return solution.get(argument)


def extend(db: Database, atom: Atom, solution: Solution) -> list[Solution]:
    """Find every way to extend a solution so that one more atom holds.

    Args:
        db: The universe.
        atom: The atom to satisfy.
        solution: The values known so far.

    Returns:
        The solution with the atom's variables known too, once per way to do so. Empty if the atom
        cannot hold.
    """
    relation = db[atom.relation]
    x, y = value_of(atom.first, solution), value_of(atom.second, solution)
    if x is not None and y is not None:
        # Both known, e.g. hobby(Y_3, "chess") once Y_3 is known: keep the solution if it holds
        return [solution] if relation.has(x, y) else []
    if x is not None:
        # First known, e.g. friend(Y_3, Y_2) once Y_3 is known: one solution per friend
        return [solution | {atom.second: other} for other in relation.of(x)]
    if y is not None:
        # Second known, e.g. hobby(Y_3, "chess") at the start: one solution per person with the hobby
        return [solution | {atom.first: other} for other in relation.inverse_of(y)]
    # Neither known: every pair of the relation
    return [solution | {atom.first: first, atom.second: second} for first, second in relation]


def next_atom(atoms: list[Atom], known: set[str]) -> Atom:
    """Choose which atom to take next: the one with the most arguments already known.

    An atom with one known argument is a lookup from that person or value, and one with both known
    only filters the solutions. An atom with neither known pairs every solution with the whole
    relation, so it goes last.

    Args:
        atoms: The atoms left.
        known: The variables known so far.

    Returns:
        The atom; among equals, the first.
    """

    def is_known(argument: str) -> bool:
        return is_constant(argument) or argument in known

    return max(atoms, key=lambda atom: is_known(atom.first) + is_known(atom.second))


def values_of(db: Database, atoms: list[Atom], variable: str) -> set[str]:
    """Find every value a variable takes among the solutions of some atoms.

    Args:
        db: The universe.
        atoms: The atoms that must all hold.
        variable: The variable to report.

    Returns:
        The variable's distinct values.

    Raises:
        ValueError: If no atom mentions the variable.
    """
    if not any(variable in (atom.first, atom.second) for atom in atoms):
        raise ValueError(f"No atom gives {variable} a value")

    atoms = list(atoms)
    # All solutions share the same variables: the ones mentioned by the atoms taken so far and still needed
    solutions: list[Solution] = [{}]
    while atoms and solutions:
        # Take one more atom, and extend every solution so that it holds too
        atom = next_atom(atoms, known=set(solutions[0]))
        atoms.remove(atom)
        solutions = [extended for solution in solutions for extended in extend(db, atom, solution)]

        # Forget the variables that no atom left mentions, then merge solutions that became equal
        needed = {variable} | {argument for atom in atoms for argument in (atom.first, atom.second)}
        merged: dict[tuple, Solution] = {}
        for solution in solutions:
            kept = {v: value for v, value in solution.items() if v in needed}
            merged[tuple(kept.values())] = kept
        solutions = list(merged.values())
    return {solution[variable] for solution in solutions}


def answer_query(db: Database, query: list[str], answer: str) -> list[str]:
    """Answer a query: find every value its answer variable can take.

    Args:
        db: The universe.
        query: The query's atoms, as in a question's `prolog.query`.
        answer: The answer variable.

    Returns:
        The distinct answers as strings, sorted.

    Raises:
        ValueError: If the query has a form this cannot answer.
        KeyError: If an atom names a relation with no rule or fact.
    """
    if not query[0].startswith("aggregate_all"):
        # When next_atom has several atoms to choose from, it takes the first. A query lists the atom
        # that starts its chain (from a name or an attribute value) last, so put that one first.
        return sorted(values_of(db, [parse_atom(text) for text in reversed(query)], answer))

    # "How many": count the relatives of a named person, or of each person the other atoms find
    counted, count = parse_count(query[0])
    others = [parse_atom(text) for text in reversed(query[1:])]
    if is_constant(counted.first) and not others:
        people = {counted.first[1:-1]}
    elif not is_constant(counted.first) and others:
        people = values_of(db, others, counted.first)
    else:
        raise ValueError(f"Cannot answer the count in {query}")
    # The people counted must be a variable of the count alone, and the count must be the answer
    mentioned = {argument for atom in others for argument in (atom.first, atom.second)}
    if count != answer or is_constant(counted.second) or counted.second in mentioned:
        raise ValueError(f"Cannot answer the count in {query}")
    relation = db[counted.relation]
    return sorted({str(len(set(relation.of(person)))) for person in people})
