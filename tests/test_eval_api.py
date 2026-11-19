"""Tests for scripts/eval_api.py: pi episodes answered by a chat-completions API (a scripted local one).

Needs bubblewrap, SANDBOX_ROOTFS (see scripts/sandbox/build_rootfs.sh) and aiohttp.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web  # noqa: E402

from phantom_envs._types import Task  # noqa: E402
from phantom_envs.tasks import write_tasks  # noqa: E402

# Loaded by path: putting scripts/ on sys.path would make its skyrl/ and tinker/ dirs importable
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_api.py"
spec = importlib.util.spec_from_file_location("eval_api", SCRIPT)
eval_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_api)

COMMANDS = ["grep -h 'mother of Ann Lee' /corpus/*.md", "echo 'Bea Lee' > /app/output.md"]


async def scripted_completion(request: web.Request) -> web.Response:
    """Play a fixed episode: one bash command per turn, then a final message."""
    body = await request.json()
    turn = sum(message["role"] == "assistant" for message in body["messages"])
    if turn < len(COMMANDS):
        function = {"name": "bash", "arguments": json.dumps({"command": COMMANDS[turn]})}
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"c{turn}", "type": "function", "function": function}],
        }
    else:
        message = {"role": "assistant", "content": "Done."}
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 40}}
    return web.json_response({"choices": [{"message": message, "finish_reason": "stop"}], "usage": usage})


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A data dir with a one-document corpus and two validation questions."""
    doc = {"id": "0", "title": "Ann Lee", "contents": "# Ann Lee\n\nThe mother of Ann Lee is Bea Lee.\n"}
    (tmp_path / "corpus.jsonl").write_text(json.dumps(doc) + "\n")
    tasks = [
        Task(id=f"t{i}", data_source="pw", question="Who is the mother of Ann Lee?", answers=["Bea Lee"])
        for i in range(2)
    ]
    write_tasks(tmp_path, "validation", tasks)
    return tmp_path


def test_eval_writes_scored_rows_and_resumes(data_dir, tmp_path, monkeypatch, capsys):
    if shutil.which("bwrap") is None or not os.environ.get("SANDBOX_ROOTFS"):
        pytest.skip("needs bubblewrap and SANDBOX_ROOTFS")
    monkeypatch.setenv("TEST_API_KEY", "secret")
    out_dir = tmp_path / "out"

    async def evaluate() -> None:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", scripted_completion)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        args = argparse.Namespace(
            data_dir=str(data_dir), model="test/model", out_dir=str(out_dir), name=None,
            base_url=f"http://127.0.0.1:{runner.addresses[0][1]}/v1", api_key_env="TEST_API_KEY",
            rootfs=os.environ["SANDBOX_ROOTFS"], reasoning="default", max_turns=20, max_tokens=2048,
            max_input_length=32768, temperature=1.0, command_timeout=60.0, memory_gib=4, concurrency=2,
            requests_per_minute=6000, request_timeout=60.0, max_retries=1, limit=None,
        )  # fmt: skip
        try:
            await eval_api.main(args)
            await eval_api.main(args)  # nothing left to run
        finally:
            await runner.cleanup()

    asyncio.run(evaluate())
    rows = [json.loads(line) for line in (out_dir / "model=test--model" / "pw.jsonl").open()]
    assert sorted(row["id"] for row in rows) == ["t0", "t1"]
    row = rows[0]
    assert (row["answer"], row["score"]["f1"], row["stop_reason"]) == ("Bea Lee\n", 1.0, "stop")
    assert (row["num_turns"], row["num_commands"]) == (3, 2)
    assert row["usage"]["prompt_tokens"] == 300 and row["usage"]["cached_tokens"] == 120
    # The tool result the API saw is what pi's sandbox printed
    assert "The mother of Ann Lee is Bea Lee." in str(row["messages"])
    assert "0 of 2 episodes to run" in capsys.readouterr().out
