"""The Prolog database behind a universe: its rules, the generated facts, and batched queries.

pyswip runs one SWI-Prolog engine per process, so :class:`Database` wipes the facts and inverse
rules a previous universe asserted. Batched queries fan out over forked workers, which inherit the
engine; results keep the query order, so the output never depends on the number of workers.
"""

import logging
import multiprocessing
from importlib.resources import files

from pyswip import Prolog
from tqdm import tqdm

from phantom_envs.generator.vocab import INVERSE_RELATIONS, PERSON_TYPE

logger = logging.getLogger(__name__)

RULES = [
    files("phantom_envs.generator").joinpath(f"rules/{name}.pl")
    for name in ("family_base", "family_derived", "friends")
]
FACT_PREDICATES = ["type/2", "gender/2", "parent/2", "dob/2", "friend_/2", "job/2", "hobby/2", "attribute/1"]
# Batches smaller than this run serially: forking the workers would cost more than the queries
MIN_PARALLEL_BATCH = 32


def decode(x):
    """Decode a Prolog value, which some pyswip versions return as bytes.

    Args:
        x: A value from a query solution.

    Returns:
        ``x`` as a str if it was bytes, else unchanged.
    """
    return x.decode("utf-8") if isinstance(x, bytes) else x


class Database:
    """A SWI-Prolog database with the PhantomEnvs rules and no facts."""

    def __init__(self) -> None:
        self.prolog = Prolog()
        # Wipe facts left in the process's engine by an earlier universe
        for predicate in FACT_PREDICATES:
            self.prolog.dynamic(predicate)
            name, arity = predicate.split("/")
            self.query(f"retractall({name}({', '.join(['_'] * int(arity))}))")
        # ... and its inverse rules (symmetric relations have none)
        for inverse in set(INVERSE_RELATIONS.values()) - set(INVERSE_RELATIONS):
            self.prolog.dynamic(f"{inverse}/2")
            self.query(f"retractall({inverse}(_, _))")
        for rule in RULES:
            self.prolog.consult(str(rule))

    def add(self, *clauses: str) -> None:
        """Assert clauses at the end of the database.

        Args:
            *clauses: Facts or rules, without the trailing period.
        """
        for clause in clauses:
            self.prolog.assertz(clause)

    def add_inverse_relations(self) -> None:
        """Add ``R_inverse(X, Y) :- R(Y, X)`` for each asymmetric relation R.

        Lookups by R's second argument then use SWI-Prolog's first-argument indexing.
        """
        for relation, inverse in INVERSE_RELATIONS.items():
            if inverse != relation:
                self.add(f"({inverse}(X, Y) :- {relation}(Y, X))")

    def person_names(self) -> list[str]:
        """Names of everyone in the universe.

        Returns:
            The names, in the order their facts were added.
        """
        return [decode(r["X"]) for r in self.query(f"type(X, {PERSON_TYPE})")]

    def query(self, query: str) -> list[dict]:
        """Run one query.

        Args:
            query: A Prolog query, without the trailing period.

        Returns:
            All solutions, each a dict from variable name to value.
        """
        return list(self.prolog.query(query))

    def batch_query(self, queries: list[str], workers: int = 1) -> list[list[dict]]:
        """Run many queries, over forked processes when there are enough of them.

        Args:
            queries: Prolog queries, without trailing periods.
            workers: Number of processes; 1 runs serially.

        Returns:
            The solutions of each query (as in :meth:`query`), in the order of ``queries``.
        """
        if workers <= 1 or len(queries) < MIN_PARALLEL_BATCH:
            small = len(queries) < MIN_PARALLEL_BATCH
            return [self.query(q) for q in tqdm(queries, desc="Querying", disable=small)]
        # Forked workers inherit the engine with all facts; imap keeps the query order
        with multiprocessing.get_context("fork").Pool(processes=workers) as pool:
            return list(tqdm(pool.imap(self.query, queries), total=len(queries), desc="Querying"))
