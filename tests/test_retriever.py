"""Tests for the retrievers (phantom_envs.retriever.base and bm25) and the REST and MCP servers over them.

They need the skyrl extra (bm25s, datasets, fastapi, mcp) and no FAISS: the dense retriever is not
loaded.
"""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("bm25s")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from phantom_envs._types import Document, DocumentResult  # noqa: E402
from phantom_envs.retriever import bm25  # noqa: E402
from phantom_envs.retriever.base import Retriever, open_retriever  # noqa: E402
from phantom_envs.retriever.batching import QueryBatcher  # noqa: E402
from phantom_envs.retriever.mcp_server import create_server  # noqa: E402
from phantom_envs.retriever.rest_server import create_app  # noqa: E402
from phantom_envs.retriever.search_tool import NO_RESULTS, server_ready, wait_for_server  # noqa: E402
from phantom_envs.retriever.serving import RetrieverServerConfig, snippet_cutter  # noqa: E402
from phantom_envs.terminal.harness import McpServer  # noqa: E402

DOCS: list[str] = [
    "# Ann Lee\n\nThe mother of Ann Lee is Bea Lee.",
    "# Bob Ray\n\nBob Ray works as a baker in Orlin.",
    "# Cy Dax\n\nCy Dax is the brother of Bob Ray.",
]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A data dir with a three-document corpus and its BM25 index."""
    path = tmp_path_factory.mktemp("data")
    rows = [{"id": str(i), "title": doc.splitlines()[0][2:], "contents": doc} for i, doc in enumerate(DOCS)]
    (path / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    bm25.build_index(path)
    return path


class TestBM25:
    def test_best_match_comes_first(self, data_dir: Path):
        retriever = open_retriever("bm25", data_dir)
        [ann, bob] = retriever.search(["Who is the mother of Ann Lee?", "Bob Ray"], topk=2)
        assert ann[0].document.id == "0" and ann[0].document.contents == DOCS[0]
        assert {hit.document.id for hit in bob} == {"1", "2"}
        assert bob[0].score >= bob[1].score > 0

    def test_words_are_stemmed(self, data_dir: Path):
        [hits] = bm25.BM25Retriever(data_dir).search(["bakers"], topk=3)
        assert [hit.document.id for hit in hits] == ["1"]

    @pytest.mark.parametrize("query", ["zebra quartz", "the of", ""])
    def test_a_query_without_a_matching_word_finds_nothing(self, data_dir: Path, query: str):
        assert bm25.BM25Retriever(data_dir).search([query], topk=3) == [[]]

    def test_topk_may_exceed_the_corpus(self, data_dir: Path):
        [hits] = bm25.BM25Retriever(data_dir).search(["Lee Ray Dax"], topk=10)
        assert len(hits) == len(DOCS)

    def test_a_missing_index_names_the_build_command(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="phantom_envs.retriever.bm25 --data_dir"):
            bm25.BM25Retriever(tmp_path)


def test_open_retriever_rejects_an_unknown_name(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown retriever 'tfidf'"):
        open_retriever("tfidf", tmp_path)


class EchoRetriever(Retriever):
    """Returns topk documents per query, each holding the query, with scores 1, 1/2, 1/3, ..."""

    def search(self, queries: list[str], topk: int) -> list[list[DocumentResult]]:
        return [
            [
                DocumentResult(document=Document(id=str(i), contents=f"{query} #{i}"), score=1 / (i + 1))
                for i in range(topk)
            ]
            for query in queries
        ]


class FirstWord:
    """Stands in for a SnippetCutter, without downloading a tokenizer."""

    def cut(self, text: str) -> str:
        return text.split()[0]


class TestRestServer:
    def test_health(self):
        assert TestClient(create_app(EchoRetriever())).get("/health").json() == {"status": "ok"}

    def test_retrieve_uses_the_server_topk_and_hides_scores_by_default(self):
        client = TestClient(create_app(EchoRetriever(), topk=2))
        result = client.post("/retrieve", json={"query": "ann lee"}).json()["result"]
        assert [hit["document"]["contents"] for hit in result] == ["ann lee #0", "ann lee #1"]
        assert [hit["score"] for hit in result] == [None, None]

    def test_retrieve_takes_topk_and_scores_from_the_request(self):
        client = TestClient(create_app(EchoRetriever(), topk=2))
        body = {"query": "ann", "topk": 3, "return_scores": True}
        result = client.post("/retrieve", json=body).json()["result"]
        assert [hit["score"] for hit in result] == [1, 1 / 2, 1 / 3]

    def test_retrieve_batch_answers_each_query_in_order(self):
        client = TestClient(create_app(EchoRetriever(), topk=1))
        results = client.post("/retrieve_batch", json={"queries": ["ann", "bob"]}).json()["results"]
        assert [r["result"][0]["document"]["contents"] for r in results] == ["ann #0", "bob #0"]

    def test_documents_are_cut_to_snippets(self):
        client = TestClient(create_app(EchoRetriever(), topk=1, cutter=FirstWord()))
        result = client.post("/retrieve", json={"query": "ann lee"}).json()["result"]
        assert result[0]["document"]["contents"] == "ann"

    def test_serves_bm25(self, data_dir: Path):
        client = TestClient(create_app(open_retriever("bm25", data_dir)))
        result = client.post("/retrieve", json={"query": "mother of Ann Lee"}).json()["result"]
        assert result[0]["document"]["contents"] == DOCS[0]


class CountingRetriever(EchoRetriever):
    """An EchoRetriever that records its batches, and finds nothing for the query "nothing".

    Each search waits until `release` is set (at once by default), and raises while `failing` is set.
    """

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.release = threading.Event()
        self.release.set()
        self.failing = False

    @property
    def searches(self) -> int:
        return len(self.batches)

    def search(self, queries: list[str], topk: int) -> list[list[DocumentResult]]:
        self.batches.append(queries)
        self.release.wait(timeout=10)
        if self.failing:
            raise RuntimeError("index unavailable")
        return [
            [] if query == "nothing" else hits
            for query, hits in zip(queries, super().search(queries, topk), strict=True)
        ]


class TestMcpServer:
    """Calls the server in process, as an MCP client would over HTTP."""

    @staticmethod
    async def call(server, query: str) -> str:
        from mcp import Client

        async with Client(server) as client:
            result = await client.call_tool("search", {"query": query})
            return result.content[0].text

    def test_serves_one_search_tool(self):
        from mcp import Client

        async def tools():
            async with Client(create_server(EchoRetriever())) as client:
                return (await client.list_tools()).tools

        [tool] = asyncio.run(tools())
        assert tool.name == "search" and list(tool.input_schema["properties"]) == ["query"]
        # The name pi gives the tool, which the agent calls
        assert McpServer(name="retriever", socket_path="", tools=("search",)).tool_names == ["mcp__retriever__search"]  # fmt: skip

    def test_search_returns_topk_numbered_documents(self):
        text = asyncio.run(self.call(create_server(EchoRetriever(), topk=2), " ann lee "))
        assert text == "Doc 1:\nContents: ann lee #0\n\nDoc 2:\nContents: ann lee #1"

    def test_documents_are_cut_to_snippets(self):
        text = asyncio.run(self.call(create_server(EchoRetriever(), topk=1, cutter=FirstWord()), "ann lee"))
        assert text == "Doc 1:\nContents: ann"

    def test_a_search_that_finds_nothing_says_so(self):
        assert asyncio.run(self.call(create_server(CountingRetriever()), "nothing")) == NO_RESULTS

    def test_a_repeated_query_is_answered_from_the_cache(self):
        retriever = CountingRetriever()
        server = create_server(retriever)
        first, second = asyncio.run(self.call(server, "ann")), asyncio.run(self.call(server, "ann"))
        assert first == second and retriever.searches == 1

    def test_concurrent_calls_of_a_query_share_one_search(self):
        retriever = CountingRetriever()
        retriever.release.clear()
        server = create_server(retriever)

        async def both() -> list[str]:
            calls = [asyncio.create_task(self.call(server, query)) for query in ("ann", "ann ")]
            await asyncio.sleep(0.2)  # both calls wait on the first search
            retriever.release.set()
            return await asyncio.gather(*calls)

        first, second = asyncio.run(both())
        assert first == second and retriever.batches == [["ann"]]

    def test_without_a_cache_each_new_query_is_searched(self):
        retriever = CountingRetriever()
        server = create_server(retriever, cache_size=0)
        texts = [asyncio.run(self.call(server, query)) for query in ("ann", "bob", "ann")]
        assert texts[0] == texts[2] and retriever.searches == 3

    def test_a_failed_search_is_not_cached(self):
        from mcp import Client

        retriever = CountingRetriever()
        server = create_server(retriever)

        async def call() -> bool:
            async with Client(server) as client:
                return (await client.call_tool("search", {"query": "ann"})).is_error

        retriever.failing = True
        assert asyncio.run(call())
        retriever.failing = False
        assert not asyncio.run(call()) and retriever.searches == 2

    def test_wait_for_server(self, search_server: McpServer):
        wait_for_server(search_server.socket_path, timeout=5)
        assert server_ready(search_server.socket_path)

    def test_wait_for_server_gives_up(self, tmp_path: Path):
        with pytest.raises(TimeoutError, match="No retrieval MCP server"):
            wait_for_server(str(tmp_path / "none.sock"), timeout=0)


class TestQueryBatcher:
    def test_queries_that_arrive_during_a_search_form_the_next_batch(self):
        retriever = CountingRetriever()
        retriever.release.clear()
        batcher = QueryBatcher(retriever, topk=1, max_batch_size=2)
        futures = [batcher.submit("ann")]
        while not retriever.batches:  # the first query is being searched on its own
            time.sleep(0.01)
        futures += [batcher.submit(query) for query in ("bob", "cy", "dax")]
        retriever.release.set()
        hits = [future.result(timeout=10) for future in futures]
        assert retriever.batches == [["ann"], ["bob", "cy"], ["dax"]]
        assert [hit.document.contents for [hit] in hits] == ["ann #0", "bob #0", "cy #0", "dax #0"]

    def test_a_failed_search_fails_its_whole_batch(self):
        retriever = CountingRetriever()
        retriever.release.clear()
        retriever.failing = True
        batcher = QueryBatcher(retriever, topk=1)
        futures = [batcher.submit("ann")]
        while not retriever.batches:
            time.sleep(0.01)
        futures += [batcher.submit(query) for query in ("bob", "cy")]
        retriever.release.set()
        for future in futures:
            with pytest.raises(RuntimeError, match="index unavailable"):
                future.result(timeout=10)
        assert retriever.batches == [["ann"], ["bob", "cy"]]


class TestSnippetCutter:
    def test_off_by_default(self):
        assert snippet_cutter(RetrieverServerConfig(retriever="bm25", data_dir="data")) is None

    def test_bm25_needs_a_tokenizer_named(self):
        cfg = RetrieverServerConfig(retriever="bm25", data_dir="data", snippet_max_tokens=512)
        with pytest.raises(ValueError, match="snippet_tokenizer"):
            snippet_cutter(cfg)
