"""Answer a question's query from the database.

A query is a list of atoms that must all hold, written in Prolog syntax (the form templates,
difficulty scores and task metadata use). Each atom is `relation(first, second)`, read "second
is first's relation", over variables and quoted constants:

    ['mother(Y_2, Y_1)', 'friend(Y_3, Y_2)', 'hobby(Y_3, "chess")']        answer variable: Y_1

"Who is the mother of the friend of the person whose hobby is chess?" The answers are all the
values `Y_1` can take. "How many" questions count with one more kind of atom, which sets
`Count_1` to the number of distinct sisters of `Y_2`:

    'aggregate_all(count, distinct(sister(Y_2, Y_3)), Count_1)'

`answer_query` works through the atoms one at a time, keeping a table of the solutions so far
(one column per variable). Since only the set of answers matters, it is free to take the atoms in
any order and to drop the columns it no longer needs, which keeps that table small.
"""

from dataclasses import dataclass, field

from phantom_envs.generator.difficulty import is_quoted_literal, parse_predicate
from phantom_envs.generator.relations import Database, Pair, Relation


@dataclass
class Atom:
    """One condition of a query: `relation(first, second)`, or a count of the `second`s."""

    relation: str = field(metadata={"description": "Name of the relation."})
    first: str = field(metadata={"description": 'A variable (Y_3) or a quoted constant ("Ann Smith").'})
    second: str = field(metadata={"description": "A variable or a quoted constant."})
    count: str | None = field(
        default=None,
        metadata={"description": "For a count: the variable set to the number of distinct seconds."},
    )

    def variables(self) -> list[str]:
        """The variables this atom gives a value to."""
        if self.count:
            # The people counted stay inside the count
            return [self.count]
        return [term for term in (self.first, self.second) if not is_quoted_literal(term)]


def parse_atom(text: str) -> Atom:
    """Parse one atom of a query.

    Args:
        text: `relation(first, second)` or `aggregate_all(count, distinct(relation(first, second)), Count)`.

    Returns:
        The atom.

    Raises:
        ValueError: For any other form.
    """
    predicate, arguments = parse_predicate(text)
    count = None
    if predicate == "aggregate_all" and len(arguments) == 3:
        # Unwrap aggregate_all(count, distinct(<atom>), Count) down to the atom it counts
        _, distinct, count = arguments
        _, counted = parse_predicate(distinct)
        predicate, arguments = parse_predicate(counted[0]) if counted else (None, [])
    if predicate is None or len(arguments) != 2 or arguments[0] == arguments[1]:
        raise ValueError(f"Cannot answer the query atom {text!r}")
    return Atom(predicate, arguments[0], arguments[1], count)


def value_of(term: str, values: dict[str, str]) -> str | None:
    """Resolve an atom's argument in one solution.

    Args:
        term: A variable or a quoted constant.
        values: The solution's variables and their values.

    Returns:
        The constant without its quotes, the variable's value, or None for a variable with no value yet.
    """
    if is_quoted_literal(term):
        return term[1:-1]
    return values.get(term)


def matching_pairs(relation: Relation, x: str | None, y: str | None) -> list[Pair]:
    """Find the pairs of a relation that agree with what is known of x and y.

    Args:
        relation: The relation.
        x: The first element, or None if unknown.
        y: The second element, or None if unknown.

    Returns:
        The matching pairs `(x, y)`.
    """
    if x is not None and y is not None:
        return [(x, y)] if relation.has(x, y) else []
    if x is not None:
        return [(x, other) for other in relation.of(x)]
    if y is not None:
        return [(other, y) for other in relation.inverse_of(y)]
    return relation.pairs


def next_atom(atoms: list[Atom], variables: list[str]) -> Atom:
    """Choose which atom to work through next: the one with the most arguments already known.

    An atom with a known argument is a lookup from that person or value, and one with both known
    only filters the solutions. An atom with neither would pair every solution with the whole
    relation, so it goes last.

    Args:
        atoms: The atoms left.
        variables: The variables that have values so far.

    Returns:
        The atom; among equals, the first.

    Raises:
        ValueError: If only counts are left and their people have no values.
    """

    def is_known(term: str) -> bool:
        return is_quoted_literal(term) or term in variables

    # A count needs to know whose relatives it counts
    ready = [atom for atom in atoms if atom.count is None or is_known(atom.first)]
    if not ready:
        raise ValueError(f"Cannot count before {atoms[0].first} has a value")
    return max(ready, key=lambda atom: is_known(atom.first) + is_known(atom.second))


def answer_query(db: Database, query: list[str], answer: str) -> list[str]:
    """Find every value the answer variable takes among the solutions of a query.

    Args:
        db: The universe.
        query: The query's atoms, as in a question's `prolog.query`.
        answer: The answer variable.

    Returns:
        The distinct answers as strings, sorted.

    Raises:
        ValueError: If an atom cannot be parsed, or no atom gives the answer variable a value.
    """
    # The last atom is the start of the question's chain (a name or an attribute value), so put
    # it first for next_atom's ties
    atoms = [parse_atom(text) for text in reversed(query)]
    if not any(answer in atom.variables() for atom in atoms):
        raise ValueError(f"No atom of {query} gives {answer} a value")

    # The solutions so far: one row per solution, with a value for each of the variables
    variables: list[str] = []
    rows: set[tuple] = {()}
    while atoms:
        atom = next_atom(atoms, variables)
        atoms.remove(atom)
        relation = db[atom.relation]

        # Extend each solution in every way that satisfies the atom
        new_variables = variables + [v for v in atom.variables() if v not in variables]
        new_rows = set()
        for row in rows:
            values = dict(zip(variables, row, strict=True))
            x, y = value_of(atom.first, values), value_of(atom.second, values)
            if atom.count:
                extensions = [{atom.count: len(set(relation.of(x)))}]
            else:
                extensions = [{atom.first: x, atom.second: y} for x, y in matching_pairs(relation, x, y)]
            for extension in extensions:
                values.update(extension)
                new_rows.add(tuple(values[v] for v in new_variables))

        # Keep only the variables that the atoms left (or the answer) use. Solutions that differ
        # only in a dropped variable become one row, so a long chain carries one person per row
        # instead of the whole path that led to them.
        needed = {answer} | {term for atom in atoms for term in (atom.first, atom.second)}
        kept = [i for i, v in enumerate(new_variables) if v in needed]
        variables = [new_variables[i] for i in kept]
        rows = {tuple(row[i] for i in kept) for row in new_rows}
    return sorted({str(row[variables.index(answer)]) for row in rows})
