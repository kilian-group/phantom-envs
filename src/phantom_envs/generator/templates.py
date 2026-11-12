"""Question and Prolog query templates, from a context-free grammar plus a few direct builders.

A template is an abstract question with placeholders, and the Prolog query that answers it::

    Who is the <relation>_3 of the person whose <attribute_name>_7 is <attribute_value>_7 ?
    <relation>_3(Y_5, Y_4), <attribute_name>_7(Y_5, <attribute_value>_7)      answer: Y_4

The samplers fill the placeholders from a universe. Three kinds of templates exist:

- ``base``: "Who is ...?" and "What is ...?" questions from the grammar, plus chains with
  constraints the grammar cannot express (several constraints on one person, or a constraint
  on a person mid-chain);
- ``count``: the grammar's "How many ...?" questions;
- ``comparison``: "Who is older / younger / born first, A or B?", where A and B are chains.

The CFG expansion is adapted from ``nltk.parse.generate`` (https://www.nltk.org, Apache-2.0,
Copyright (C) 2001-2024 NLTK Project). The templates and their order match PhantomWiki's.
"""

import re
from dataclasses import dataclass, field

from nltk import CFG, Nonterminal

QA_GRAMMAR_STRING = """
    S -> 'Who is' R '?' | 'What is' A '?' | 'How many' RN_p 'does' R_c 'have' '?'
    R -> 'the' RN 'of' R_c | 'the person whose' AN 'is' AV
    R_c -> R | N
    A -> 'the' AN 'of' R
    RN -> '<relation>'
    RN_p -> '<relation_plural>'
    AN -> '<attribute_name>'
    AV -> '<attribute_value>'
    N -> '<name>'
    """
KINDS = ("base", "count", "comparison")
COMPARISON_PREFIXES = [["Who is older,"], ["Who is younger,"], ["Who was born first,"]]
# Chain fragments own disjoint subscript windows [offset, offset + STRIDE), far above the grammar's
CHAIN_OFFSET_BASE = 1000
CHAIN_OFFSET_STRIDE = 1000


@dataclass
class Template:
    """A question template, its Prolog query template, and what the query answers."""

    question: list[str] = field(metadata={"description": "Question tokens, ending with '?'."})
    query: list[str] = field(
        metadata={"description": "Prolog atoms, outermost first (they run in reverse order)."}
    )
    answer: str | tuple[str, str] = field(
        metadata={"description": "Answer variable, or for comparisons the two compared people."}
    )
    kind: str = field(metadata={"description": f"One of {KINDS}."})


@dataclass
class Fragment:
    """A piece of a question template and its query atoms.

    Empty: ``([], [], None)``. Plain terminal: ``(['the'], [], None)``. Placeholder:
    ``(['<name>_1'], ['<name>_1'], None)``. Otherwise a subquery with its answer variable.
    """

    q: list[str] = field(default_factory=list, metadata={"description": "Question tokens."})
    p: list[str] = field(default_factory=list, metadata={"description": "Prolog atoms."})
    answer: str | None = field(default=None, metadata={"description": "Answer variable of the atoms."})

    def is_empty(self) -> bool:
        """Whether this is the end of a production."""
        return not self.q

    def is_terminal(self) -> bool:
        """Whether this is plain words, with no query atoms."""
        return not self.p

    def is_placeholder(self) -> bool:
        """Whether this is a single ``<placeholder>_N`` token."""
        return len(self.p) == 1 and not self.answer

    def person(self) -> str:
        """The variable or ``<name>`` placeholder this fragment resolves to."""
        return self.answer or self.p[0]


def cfg_depth(max_hops: int) -> int:
    """Grammar depth needed to reach ``max_hops`` hops.

    Args:
        max_hops: Most relation hops in a question.

    Returns:
        2 levels per hop, plus 4 for an attribute anchor ("the person whose A is V").
    """
    return 2 * max_hops + 4


def generate_templates(max_hops: int, max_constraints: int, kinds: tuple[str, ...]) -> list[Template]:
    """All templates of the given kinds within the hop and constraint limits.

    Args:
        max_hops: Most relation hops in a question.
        max_constraints: Most attribute constraints on one person.
        kinds: Kinds to include, from :data:`KINDS`.

    Returns:
        Base and count templates in grammar order, then comparison templates.

    Raises:
        ValueError: For an unknown kind.
    """
    if unknown := set(kinds) - set(KINDS):
        raise ValueError(f"Unknown kinds {unknown}; expected some of {KINDS}")
    grammar = CFG.fromstring(QA_GRAMMAR_STRING)
    depth = cfg_depth(max_hops)
    templates: list[Template] = []

    if {"base", "count"} & set(kinds):
        # Grammar templates, then the chains it cannot express; "How many" questions are count
        chain_templates = [
            Template(f.q, f.p, f.answer, "") for f in expand(grammar, [grammar.start()], depth, depth)
        ]
        chain_templates += chain_base_templates(max_hops, max_constraints)
        for t in chain_templates:
            t.kind = "count" if t.question[0] == "How many" else "base"
        templates += [t for t in chain_templates if t.kind in kinds]

    if "comparison" in kinds:
        # Pairs of the grammar's person phrases, then long chains against n-constraint people
        rc_fragments = expand_symbol(grammar, Nonterminal("R_c"), depth, depth)
        templates += comparison_templates(rc_fragments, depth)
        deep_fragments = rc_fragments + deep_chain_fragments(max_hops)
        for n in range(2, max_constraints + 1):
            templates += comparison_mc_templates(deep_fragments, mc_fragment(n, depth), n, depth)
    return templates


#
# Grammar expansion
#
def expand(grammar: CFG, items: list, depth: int, total_depth: int) -> list[Fragment]:
    """Expand a sequence of grammar symbols.

    Args:
        grammar: The question grammar.
        items: Symbols of a production's right-hand side.
        depth: Remaining expansion depth.
        total_depth: Depth at the start symbol; subscripts are ``total_depth - depth``.

    Returns:
        Each fragment of the first symbol combined with each fragment of the rest.
    """
    if not items:
        return [Fragment()]
    return [
        combine(head, tail, depth, total_depth)
        for head in expand_symbol(grammar, items[0], depth, total_depth)
        for tail in expand(grammar, items[1:], depth, total_depth)
    ]


def expand_symbol(grammar: CFG, item, depth: int, total_depth: int) -> list[Fragment]:
    """Expand one grammar symbol.

    Args:
        grammar: The question grammar.
        item: A nonterminal, or a terminal string.
        depth: Remaining expansion depth.
        total_depth: Depth at the start symbol.

    Returns:
        The fragments of all of a nonterminal's productions, or the terminal itself (a placeholder
        gets the subscript ``total_depth - depth``). Nothing once the depth runs out.
    """
    if depth <= 0:
        return []
    if isinstance(item, Nonterminal):
        return [
            fragment
            for production in grammar.productions(lhs=item)
            for fragment in expand(grammar, production.rhs(), depth - 1, total_depth)
        ]
    if re.match(r"<.*?>", item):
        d = total_depth - depth
        return [Fragment([f"{item}_{d}"], [f"{item}_{d}"])]
    return [Fragment([item])]


def combine(f1: Fragment, f2: Fragment, depth: int, total_depth: int) -> Fragment:
    """Concatenate two fragments, turning a placeholder followed by a subquery into a query atom.

    Example: ``<relation>_2`` + ``of <name>_1`` gives the atom ``<relation>_2(<name>_1, Y_d)``,
    answered by ``Y_d``.

    Args:
        f1: The first fragment.
        f2: The fragment that follows it.
        depth: Remaining expansion depth.
        total_depth: Depth at the start symbol; new variables get the subscript ``total_depth - depth``.

    Returns:
        The combined fragment.
    """
    q = f1.q + f2.q
    # Plain words or an empty end pass the other fragment's query through
    if f1.is_empty():
        return Fragment(q, f2.p, f2.answer)
    if f1.is_terminal():
        return Fragment(q, f1.p, f1.answer) if f2.is_empty() else Fragment(q, f2.p, f2.answer)
    if f2.is_empty() or f2.is_terminal():
        return Fragment(q, f1.p, f1.answer)

    # The grammar only puts placeholders first; f2 is a placeholder or a subquery
    assert f1.is_placeholder()
    placeholder = f1.p[0]
    d = total_depth - depth
    inner, rest = (f2.p[0], []) if f2.is_placeholder() else (f2.answer, f2.p)
    if re.match(r"<relation_plural>_(\d+)", placeholder):
        # "How many <relation_plural> does ... have"
        return Fragment(
            q,
            [f"aggregate_all(count, distinct({placeholder}({inner}, Y_{d})), Count_{d})"] + rest,
            f"Count_{d}",
        )
    if re.match(r"<relation>_(\d+)", placeholder):
        # "the <relation> of ..."
        return Fragment(q, [f"{placeholder}({inner}, Y_{d})"] + rest, f"Y_{d}")
    if re.match(r"<attribute_name>_(\d+)", placeholder):
        if placeholder.replace("name", "value") in f2.p[0]:
            # "... whose hobby is running"
            return Fragment(q, [f"{placeholder}(Y_{d}, {f2.p[0]})"], f"Y_{d}")
        # "... the hobby of the mother of Alice"
        return Fragment(q, [f"{placeholder}({inner}, Y_{d})"] + rest, f"Y_{d}")
    return Fragment(q, None, None)


#
# Chains the grammar cannot express
#
def chain_fragment(n_hops: int, c_count: int, c_position: int, offset: int) -> Fragment:
    """A chain "the <rel_0> of the <rel_1> of ... <anchor>" with constraints on one person.

    The chain's people are Y_0 (the answer) .. Y_{n_hops} (the anchor). The constrained person is
    either the anchor ("the person whose A is V and whose ...") or an intermediate Y_k ("the
    <rel_k>, whose A is V, of ..."). Without constraints the anchor is a ``<name>``.

    Args:
        n_hops: Relations in the chain.
        c_count: Attribute constraints (0 for none).
        c_position: Who carries them: k in 1..n_hops-1 for Y_k, or n_hops for the anchor.
        offset: Start of the fragment's subscript window ``[offset, offset + CHAIN_OFFSET_STRIDE)``.

    Returns:
        The chain, answered by ``Y_{offset}``.
    """
    assert n_hops >= 0 and c_count >= 0
    if c_count > 0:
        assert 0 <= c_position <= n_hops
    if n_hops == 0:
        assert c_count > 0 and c_position == 0, "a chain without hops needs constraints on its anchor"

    q: list[str] = []
    p: list[str] = []

    def constraint_clause(target: str) -> str:
        """``"<an> is <av> and ..."`` on ``target``, adding the matching atoms to ``p``."""
        parts = []
        for k in range(c_count):
            an = f"<attribute_name>_{offset + n_hops + 1 + k}"
            av = f"<attribute_value>_{offset + n_hops + 1 + k}"
            parts.append(f"{an} is {av}")
            p.append(f"{an}({target}, {av})")
        return " and ".join(parts)

    # "the <rel_i> of", with the constraint clause after the constrained intermediate
    for i in range(n_hops):
        rel = f"<relation>_{offset + i}"
        if c_count > 0 and c_position == i and 1 <= c_position < n_hops:
            # One token, so joining with spaces puts none before the commas
            q.append(f"the {rel}, whose {constraint_clause(f'Y_{offset + i}')}, of")
        else:
            q.extend(["the", rel, "of"])

    # The anchor: a constrained person or a name
    if c_count > 0 and c_position == n_hops:
        anchor = f"Y_{offset + n_hops}"
        q.append(f"the person whose {constraint_clause(anchor)}")
    else:
        anchor = f"<name>_{offset + n_hops}"
        q.append(anchor)

    # Relation atoms <rel_i>(Y_{i+1}, Y_i), the last one from the anchor
    for i in range(n_hops):
        inner = anchor if i == n_hops - 1 else f"Y_{offset + i + 1}"
        p.append(f"<relation>_{offset + i}({inner}, Y_{offset + i})")
    return Fragment(q, p, f"Y_{offset}")


def chain_base_templates(max_hops: int, max_constraints: int) -> list[Template]:
    """Chains with constraints on one person, in the grammar's three question forms.

    Args:
        max_hops: Most relations in a chain (from 0, a constrained person alone).
        max_constraints: Most constraints on the person.

    Returns:
        Templates for every (hops, constraints, position), except one constraint on the anchor,
        which the grammar already makes.
    """
    templates = []
    base_offset = CHAIN_OFFSET_BASE + 1_000_000  # above deep_chain_fragments' windows
    idx = 0
    for n_hops in range(max_hops + 1):
        for c in range(1, max_constraints + 1):
            # Without hops the only person is the anchor
            for position in [0] if n_hops == 0 else range(1, n_hops + 1):
                if c == 1 and position == n_hops:
                    continue
                offset = base_offset + idx * CHAIN_OFFSET_STRIDE
                idx += 1
                templates += question_forms(chain_fragment(n_hops, c, position, offset), offset)
    return templates


def question_forms(frag: Fragment, offset: int) -> list[Template]:
    """Ask about a chain in the grammar's three question forms.

    Args:
        frag: A chain from :func:`chain_fragment`.
        offset: The chain's subscript window; the forms' own placeholders sit at ``offset + 500`` and up.

    Returns:
        "Who is <chain>?", "What is the <an> of <chain>?" and "How many <rp> does <chain> have?".
    """
    sub = offset + 500
    attr, attr_y = f"<attribute_name>_{sub}", f"Y_{sub}"
    rel_p, count_y, count = f"<relation_plural>_{sub + 1}", f"Y_{sub + 1}", f"Count_{sub + 1}"
    return [
        Template(["Who is"] + frag.q + ["?"], frag.p, frag.answer, ""),
        Template(
            ["What is", "the", attr, "of"] + frag.q + ["?"],
            [f"{attr}({frag.answer}, {attr_y})"] + frag.p,
            attr_y,
            "",
        ),
        Template(
            ["How many", rel_p, "does"] + frag.q + ["have", "?"],
            [f"aggregate_all(count, distinct({rel_p}({frag.answer}, {count_y})), {count})"] + frag.p,
            count,
            "",
        ),
    ]


def deep_chain_fragments(max_hops: int) -> list[Fragment]:
    """Chains beyond the grammar's depth, for the n-constraint comparisons.

    Args:
        max_hops: Most relations in a chain.

    Returns:
        Chains of 2..max_hops hops, each anchored on a name and on one constraint.
    """
    configs = [(n_hops, c, p) for n_hops in range(2, max_hops + 1) for c, p in [(0, 0), (1, n_hops)]]
    return [
        chain_fragment(n_hops, c, p, CHAIN_OFFSET_BASE + idx * CHAIN_OFFSET_STRIDE)
        for idx, (n_hops, c, p) in enumerate(configs)
    ]


def mc_fragment(n: int, depth: int) -> Fragment:
    """A person with ``n`` constraints: "the person whose <an_1> is <av_1> and ... whose <an_n> is <av_n>".

    Args:
        n: Number of constraints, at least 2 (the grammar makes one).
        depth: Grammar depth; the subscripts start at ``n * depth * 10``, clear of the grammar's.

    Returns:
        The fragment, answered by the constrained person.

    Raises:
        ValueError: If ``n < 2``.
    """
    if n < 2:
        raise ValueError(f"mc_fragment needs n >= 2, got {n}")
    offset = n * depth * 10
    person = f"Y_{offset}"
    q, p = ["the person"], []
    for i in range(n):
        an, av = f"<attribute_name>_{offset + i}", f"<attribute_value>_{offset + i}"
        q.extend(["whose" if i == 0 else "and whose", an, "is", av])
        p.append(f"{an}({person}, {av})")
    return Fragment(q, p, person)


#
# Comparisons
#
def renumber(fragment: Fragment, offset: int) -> Fragment:
    """Shift every ``_N`` subscript of a fragment, so two copies of it share no variables.

    Args:
        fragment: The fragment.
        offset: Amount to add to each subscript.

    Returns:
        The shifted copy.
    """

    def shift(s: str) -> str:
        return re.sub(r"_(\d+)", lambda m: f"_{int(m.group(1)) + offset}", s)

    return Fragment(
        [shift(t) for t in fragment.q],
        [shift(t) for t in fragment.p],
        shift(fragment.answer) if fragment.answer else None,
    )


def count_hops(fragment: Fragment) -> int:
    """Number of relation atoms in a fragment."""
    return sum(1 for atom in fragment.p if re.match(r"<relation>_\d+", atom))


def comparison(left: Fragment, right: Fragment, prefix: list[str], dob_l: str, dob_r: str) -> Template:
    """Compare the ages of the people two fragments resolve to.

    Args:
        left: The first person's phrase.
        right: The second person's phrase.
        prefix: The phrasing, e.g. ``["Who is older,"]``.
        dob_l: Variable for the left person's date of birth.
        dob_r: Variable for the right person's date of birth.

    Returns:
        "<prefix> <left> or <right>?", whose answer is one of the two people.
    """
    lp, rp = left.person(), right.person()
    # Runs last to first: both chains, their people's birth dates, then the comparison
    query = [f"{dob_l} @< {dob_r}", f"dob({lp}, {dob_l})", f"dob({rp}, {dob_r})"] + left.p + right.p
    return Template(prefix + left.q + ["or"] + right.q + ["?"], query, (lp, rp), "comparison")


def comparison_templates(rc_fragments: list[Fragment], depth: int) -> list[Template]:
    """Comparisons of every ordered pair of the grammar's person phrases.

    Args:
        rc_fragments: The grammar's expansions of ``R_c``.
        depth: Grammar depth; the right phrase's subscripts are shifted by it.

    Returns:
        One template per pair, cycling through the older / younger / born-first phrasings.
    """
    pairs = [(left, renumber(right, depth)) for left in rc_fragments for right in rc_fragments]
    return [
        comparison(left, right, COMPARISON_PREFIXES[i % 3], f"CmpDL_{depth}", f"CmpDR_{depth}")
        for i, (left, right) in enumerate(pairs)
    ]


def comparison_mc_templates(rc_fragments: list[Fragment], mc: Fragment, n: int, depth: int) -> list[Template]:
    """Comparisons of a chain with hops against a person with ``n`` constraints.

    Args:
        rc_fragments: Person phrases; those without hops are skipped.
        mc: The ``n``-constraint person from :func:`mc_fragment`.
        n: Its number of constraints.
        depth: Grammar depth, for the date-of-birth variable names.

    Returns:
        Two templates per chain (chain first, then constrained person first), cycling through the
        phrasings.
    """
    pairs = []
    for rc in rc_fragments:
        if count_hops(rc) > 0:
            pairs += [(rc, mc), (mc, rc)]
    return [
        comparison(left, right, COMPARISON_PREFIXES[i % 3], f"CmpDL_{depth}_{n}", f"CmpDR_{depth}_{n}")
        for i, (left, right) in enumerate(pairs)
    ]
