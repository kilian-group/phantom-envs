"""The difficulty of a question, from its Prolog query, on two independent axes.

- **hops**: relation steps needed to answer, each weighted by the articles it takes to resolve
  (``HOP_COUNTS``: ``mother`` is 1, ``nephew`` 2).
- **constraints**: attribute filters in the query (``hobby``, ``job`` or ``dob`` set to a value).

``composite`` (hops + constraints) gives a single number for sorting.
"""

import heapq
import re
from collections import defaultdict

from phantom_envs.generator.vocab import ATTRIBUTE_TYPES, HOP_COUNTS

# Predicates that are gender filters, not relational hops
GENDER_FILTERS = frozenset(["male", "female", "nonbinary", "gender"])
# Attribute predicates that count as constraints (not hops)
ATTRIBUTE_SET = frozenset(ATTRIBUTE_TYPES) | {"dob", "date_of_birth"}
COMPARISON_RE = re.compile(r"@[<>]|\\+|\\=|==|>|<")


def parse_predicate(query_atom: str) -> tuple[str | None, list[str]]:
    """Split a Prolog query atom into its predicate and arguments.

    Args:
        query_atom: An atom such as ``job(Y_2, "chef")``.

    Returns:
        The predicate name and its top-level arguments, or ``(None, [])`` for negations,
        comparisons and anything else that is not a predicate call.
    """
    query_atom = query_atom.strip()
    if not query_atom:
        return None, []
    if query_atom.startswith("\\+"):
        return None, []
    if COMPARISON_RE.match(query_atom):
        return None, []
    m = re.match(r"(\w+)\(", query_atom)
    if not m:
        return None, []

    name = m.group(1)
    # Extract the argument string (everything between the outermost parens)
    start = m.end()  # position right after "name("
    depth = 1
    i = start
    while i < len(query_atom) and depth > 0:
        ch = query_atom[i]
        if ch == '"':
            i += 1
            while i < len(query_atom) and query_atom[i] != '"':
                i += 1
        elif ch == "'":
            i += 1
            while i < len(query_atom) and query_atom[i] != "'":
                i += 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1

    arg_str = query_atom[start:i]

    # Split args on commas, respecting quotes and nested parens
    args: list[str] = []
    current: list[str] = []
    depth = 0
    in_quote = False
    quote_char = ""
    for ch in arg_str:
        if in_quote:
            current.append(ch)
            if ch == quote_char:
                in_quote = False
        elif ch in ('"', "'"):
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())

    return name, args


def is_quoted_literal(s: str) -> bool:
    """Whether a Prolog argument is a quoted literal (double or single quotes), not a variable."""
    s = s.strip()
    return (len(s) >= 2 and s[0] == '"' and s[-1] == '"') or (len(s) >= 2 and s[0] == "'" and s[-1] == "'")


VARIABLE_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")
QUOTED_STR_RE = re.compile(r'"[^"]*"')


def extract_variables(atom: str) -> set[str]:
    """Find the Prolog variables of an atom.

    Args:
        atom: A Prolog query atom.

    Returns:
        Its capitalized identifiers, except gender filters and words inside quoted literals (names
        like "Deane Smock" are not variables).
    """
    atom_no_strings = QUOTED_STR_RE.sub("", atom)
    return set(VARIABLE_RE.findall(atom_no_strings)) - GENDER_FILTERS


BARE_CMP_RE = re.compile(r"@[<>]|>=|<=|=:=|==|>|<")


def is_bare_comparison_atom(atom: str) -> bool:
    """Whether an atom is a top-level comparison such as ``D1 @< D2`` or ``C1 > C2``.

    Atoms with parentheses (predicates, negations) are not, so operators nested inside
    ``\\+ (... @< ...)`` do not count.

    Args:
        atom: A Prolog query atom.

    Returns:
        True for a bare comparison.
    """
    a = atom.strip()
    if a.startswith("\\+"):
        return False
    if "(" in a:
        return False
    return bool(BARE_CMP_RE.search(a))


def split_comparison_branches(query_template: list[str]) -> list[list[str]] | None:
    """Split a comparison query into the parallel branches it compares.

    A branch starts at a binder whose value flows into a top-level comparison:

    - ``dob(Entity, V)`` compared with ``@<`` / ``@>`` is plumbing: left out of the branch, whose
      content is the chain that reaches ``Entity``;
    - ``aggregate_all(count, distinct(P(Entity, _)), V)`` compared numerically is content: kept
      in the branch, adding ``P``'s hops.

    Args:
        query_template: The query's atoms.

    Returns:
        One list of atoms per binder (comparison atoms left out), or None if the query compares
        fewer than two branches.
    """
    # Top-level comparisons and the variables they compare
    cmp_atom_idx: set[int] = set()
    cmp_vars: set[str] = set()
    for i, atom in enumerate(query_template):
        if is_bare_comparison_atom(atom):
            cmp_atom_idx.add(i)
            cmp_vars.update(extract_variables(atom))
    if not cmp_vars:
        return None

    # binder_info: (atom_idx, kind, entity_token).  entity_token may be a
    # variable name (e.g. ``Y_2``) or a quoted literal (e.g. ``"Alice"``).
    binder_info: list[tuple[int, str, str]] = []
    for i, atom in enumerate(query_template):
        if i in cmp_atom_idx:
            continue
        pred, args = parse_predicate(atom)
        if pred == "dob" and len(args) >= 2 and args[1].strip() in cmp_vars:
            binder_info.append((i, "dob_plumbing", args[0].strip()))
        elif pred == "aggregate_all" and len(args) >= 3 and args[-1].strip() in cmp_vars:
            distinct_pred, distinct_args = parse_predicate(args[1])
            if distinct_pred == "distinct" and distinct_args:
                inner_pred, inner_args = parse_predicate(distinct_args[0])
                if inner_pred and inner_args:
                    binder_info.append((i, "aggregate_content", inner_args[0].strip()))

    if len(binder_info) < 2:
        return None

    # Union-find over variables, with one sentinel root per branch
    branch_count = len(binder_info)
    sentinels = [f"BRANCH_{k}" for k in range(branch_count)]
    parent: dict[str, str] = {s: s for s in sentinels}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if rb in sentinels:
                parent[ra] = rb
            else:
                parent[rb] = ra

    # Seed each branch sentinel with its binder's entity (when it's a variable).
    branches: list[list[str]] = [[] for _ in range(branch_count)]
    binder_skip_idx: set[int] = set()
    for k, (idx, kind, entity) in enumerate(binder_info):
        binder_skip_idx.add(idx)
        if not is_quoted_literal(entity):
            parent.setdefault(entity, sentinels[k])
            union(entity, sentinels[k])
        if kind == "aggregate_content":
            branches[k].append(query_template[idx])

    # Collect content atoms (excluding comparison plumbing and binders).
    content_atoms: list[tuple[str, set[str]]] = []
    for i, atom in enumerate(query_template):
        if i in cmp_atom_idx or i in binder_skip_idx:
            continue
        atom_vars = extract_variables(atom) - cmp_vars
        content_atoms.append((atom, atom_vars))

    # Atoms sharing a variable belong to the same branch
    for _, atom_vars in content_atoms:
        vlist = [v for v in atom_vars if not v.startswith("Cmp")]
        for i in range(1, len(vlist)):
            union(vlist[0], vlist[i])

    # Assign each atom to the branch its variables connect to
    for atom, atom_vars in content_atoms:
        vlist = [v for v in atom_vars if not v.startswith("Cmp")]
        if not vlist:
            continue
        root = find(vlist[0])
        if root in sentinels:
            branches[sentinels.index(root)].append(atom)

    return branches


def score_atoms(atoms: list[str], hop_counts: dict[str, int]) -> tuple[int, int]:
    """Count the hops and constraints of a list of atoms.

    Args:
        atoms: Prolog query atoms.
        hop_counts: Hops per relation.

    Returns:
        (hops, constraints). Attributes set to a value are constraints; attributes read into a
        variable count as neither; unknown predicates count one hop.
    """
    hops = 0
    constraints = 0

    for atom in atoms:
        pred, args = parse_predicate(atom)
        if pred is None:
            continue

        # A count adds the hops (or constraint) of the predicate it counts
        if pred == "aggregate_all":
            inner = re.search(r"distinct\((\w+)\(", atom)
            if inner:
                inner_pred = inner.group(1)
                if inner_pred in ATTRIBUTE_SET:
                    constraints += 1
                elif inner_pred in hop_counts:
                    hops += hop_counts[inner_pred]
                else:
                    hops += 1
            else:
                hops += 1
            continue

        if pred in ATTRIBUTE_SET:
            if len(args) >= 2 and is_quoted_literal(args[1]):
                constraints += 1
        elif pred in hop_counts:
            hops += hop_counts[pred]
        elif pred in GENDER_FILTERS:
            pass
        else:
            hops += 1

    return hops, constraints


def compute_constraint_position(
    query_template: list[str],
    answer_var: str,
    hop_counts: dict[str, int],
) -> int | None:
    """Hops from the answer to the nearest constrained person.

    Relation edges weigh their hops (``grandfather`` is 2), counts weigh their relation's, and
    attribute reads weigh 0. In a chain constrained at its anchor, the position equals the hops.

    Args:
        query_template: The query's atoms.
        answer_var: The answer variable.
        hop_counts: Hops per relation.

    Returns:
        The shortest hop-weighted distance, or None without constraints or a path to them.
    """
    if answer_var is None:
        return None

    # People with an attribute set to a value
    constrained: set[str] = set()
    for atom in query_template:
        pred, args = parse_predicate(atom)
        if pred in ATTRIBUTE_SET and len(args) >= 2 and is_quoted_literal(args[1]):
            constrained.add(args[0].strip())
    if not constrained:
        return None
    if answer_var in constrained:
        return 0

    # Graph of the query's variables, with hop-weighted edges
    adj: dict[str, list[tuple[str, int]]] = defaultdict(list)

    def add_edge(a: str, b: str, cost: int) -> None:
        if is_quoted_literal(a) or is_quoted_literal(b):
            return
        adj[a].append((b, cost))
        adj[b].append((a, cost))

    for atom in query_template:
        pred, args = parse_predicate(atom)
        if pred in hop_counts and len(args) == 2:
            add_edge(args[0].strip(), args[1].strip(), hop_counts[pred])
        elif pred in ATTRIBUTE_SET and len(args) >= 2 and not is_quoted_literal(args[1]):
            add_edge(args[0].strip(), args[1].strip(), 0)
        elif pred == "aggregate_all" and len(args) >= 3:
            inner = re.search(r"distinct\((\w+)\(([^,]+),\s*([^)]+)\)\)", atom)
            if inner and inner.group(1) in hop_counts:
                add_edge(inner.group(2).strip(), args[-1].strip(), hop_counts[inner.group(1)])

    # Dijkstra from the answer to the first constrained person
    dist: dict[str, int] = {answer_var: 0}
    heap: list[tuple[int, str]] = [(0, answer_var)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float("inf")):
            continue
        if node in constrained:
            return d
        for nb, cost in adj[node]:
            nd = d + cost
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return None


def compute_difficulty(query_template: list[str], *, answer_var: str | None = None) -> dict:
    """The difficulty of a question, from its query.

    A comparison takes the max of each axis over its two branches, not the sum: an agent can
    resolve the branches independently, so the harder one sets the difficulty.

    Args:
        query_template: The question's query atoms.
        answer_var: The answer variable (as a string); adds ``constraint_position``.

    Returns:
        ``hops``, ``constraints``, ``composite`` and, with ``answer_var``, ``constraint_position``
        (see :func:`compute_constraint_position`).
    """
    hop_counts = HOP_COUNTS

    branches = split_comparison_branches(query_template)
    if branches is not None:
        scores = [score_atoms(b, hop_counts) for b in branches]
        hops = max((s[0] for s in scores), default=0)
        constraints = max((s[1] for s in scores), default=0)
    else:
        hops, constraints = score_atoms(query_template, hop_counts)

    result = {
        "hops": hops,
        "constraints": constraints,
        "composite": hops + constraints,
    }
    if answer_var is not None:
        result["constraint_position"] = compute_constraint_position(query_template, answer_var, hop_counts)
    return result
