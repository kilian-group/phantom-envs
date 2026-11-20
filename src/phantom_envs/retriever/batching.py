"""Batching of concurrent searches: one retriever call answers many queries.

A server that answers each search on its own thread runs as many retriever calls at once as it has
requests, and they slow each other down: a dense encoder is much faster on one batch of 64 queries
than on 64 batches of one. QueryBatcher lines the queries up instead. A single thread takes whatever
has arrived, up to a batch size, and searches it in one call; queries that arrive meanwhile form the
next batch. An idle server therefore answers a lone query right away.

Usage:

    batcher = QueryBatcher(retriever, topk=3)
    hits = batcher.submit("Who is the mother of Ann Lee?").result()

Gotcha: a dense encoder pads a batch to its longest query, which can change a ranking slightly. With
max_batch_size=1 every query is encoded alone, so results don't depend on the server's load, but the
server then makes one encoder call per search: a trained agent that searches often outruns it, and its
searches time out.
"""

import queue
import threading
from concurrent.futures import Future

from phantom_envs._types import DocumentResult
from phantom_envs.retriever.base import Retriever


class QueryBatcher:
    """Searches the queries of concurrent callers in batches, on one background thread."""

    def __init__(self, retriever: Retriever, topk: int, max_batch_size: int = 64) -> None:
        """Start the thread that searches.

        Args:
            retriever: The retriever that answers every batch.
            topk: Number of documents per query.
            max_batch_size: Most queries searched in one retriever call.
        """
        self.retriever = retriever
        self.topk = topk
        self.max_batch_size = max_batch_size
        self.pending: queue.Queue[tuple[str, Future[list[DocumentResult]]]] = queue.Queue()
        threading.Thread(target=self.run, name="query-batcher", daemon=True).start()

    def submit(self, query: str) -> Future[list[DocumentResult]]:
        """Queue one query.

        Args:
            query: The search query.

        Returns:
            A future of the query's documents, best first; it raises what the retriever raised.
        """
        future: Future[list[DocumentResult]] = Future()
        self.pending.put((query, future))
        return future

    def run(self) -> None:
        """Search batches forever: each is the queries waiting when the last one finished."""
        while True:
            batch = [self.pending.get()]
            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self.pending.get_nowait())
                except queue.Empty:
                    break
            try:
                results = self.retriever.search([query for query, _ in batch], self.topk)
                for (_, future), documents in zip(batch, results, strict=True):
                    future.set_result(documents)
            except Exception as error:  # noqa: BLE001 (handed to the callers)
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)
