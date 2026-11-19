"""Tests for phantom_envs.terminal.bridge's ChatBridge: a harness's chat calls, answered from Python.

BridgedEpisode runs pi in the bwrap sandbox; tests/test_tinker.py and tests/test_eval_api.py cover it.
"""

import asyncio
import json

import pytest

aiohttp = pytest.importorskip("aiohttp")

from phantom_envs.terminal.bridge import ChatBridge  # noqa: E402

TOOL_CALL = {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}


async def ask(url: str, body: dict) -> dict | list[dict]:
    """Make a chat call as a harness would; return the response, or a streamed response's chunks."""
    async with aiohttp.ClientSession() as session, session.post(f"{url}/chat/completions", json=body) as r:
        if not body.get("stream"):
            return await r.json()
        lines = (await r.text()).split("\n\n")
        return [json.loads(line.removeprefix("data: ")) for line in lines if line and line != "data: [DONE]"]


@pytest.mark.parametrize("stream", [False, True])
def test_requests_get_the_callers_replies(stream):
    async def episode() -> tuple[dict, dict | list[dict]]:
        bridge = ChatBridge()
        url = await bridge.start()
        body = {"model": "policy", "messages": [{"role": "user", "content": "Hi"}], "stream": stream}
        call = asyncio.create_task(ask(url, body))
        request = await bridge.next_request(unless=call)
        bridge.reply({"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]})
        response = await call
        await bridge.close()
        return request, response

    request, response = asyncio.run(episode())
    assert request["messages"] == [{"role": "user", "content": "Hi"}]
    if stream:
        delta = response[0]["choices"][0]["delta"]
        assert delta["tool_calls"] == [{"index": 0, **TOOL_CALL}]
        assert response[1]["choices"][0]["finish_reason"] == "tool_calls"
    else:
        choice = response["choices"][0]
        assert choice["message"]["tool_calls"] == [TOOL_CALL] and choice["finish_reason"] == "tool_calls"


def test_next_request_stops_when_the_harness_ends():
    async def episode() -> dict | None:
        bridge = ChatBridge()
        await bridge.start()
        harness = asyncio.create_task(asyncio.sleep(0))
        request = await bridge.next_request(unless=harness)
        await bridge.close()
        return request

    assert asyncio.run(episode()) is None
