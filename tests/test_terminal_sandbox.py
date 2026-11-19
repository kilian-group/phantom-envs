"""Tests for the terminal sandbox: the corpus as files (phantom_envs.terminal.corpus_files) and the contract
that every backend keeps (phantom_envs.terminal.sandbox).

The bwrap tests need bubblewrap and SANDBOX_ROOTFS (see scripts/sandbox/build_rootfs.sh); the ShellSim
tests need the skyrl extra. Each backend's tests are skipped without them.
"""

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from phantom_envs.terminal.bwrap import REPLY_GRACE, SHARED_DIR, BwrapSandbox
from phantom_envs.terminal.corpus_files import (
    MAX_DOC_BYTES,
    corpus_files,
    doc_filename,
    doc_text,
    write_corpus_dir,
)
from phantom_envs.terminal.sandbox import BACKENDS, Sandbox, format_output, open_sandbox
from phantom_envs.terminal.task import CORPUS_DIR, WORK_DIR

DOCS = [
    {"id": "0", "title": "Ann Lee", "contents": "# Ann Lee\n\nThe mother of Ann Lee is Bea Lee.\n"},
    {"id": "1", "title": "Bea Lee", "contents": "# Bea Lee\n\nThe hobby of Bea Lee is chess.\n"},
]


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-document corpus.jsonl."""
    path = tmp_path_factory.mktemp("corpus") / "corpus.jsonl"
    path.write_text("".join(json.dumps(d) + "\n" for d in DOCS))
    return path


def test_doc_filename_unique_and_safe():
    taken: set[str] = set()
    assert doc_filename("Alison Smock", taken) == "Alison_Smock.md"
    assert doc_filename("Alison Smock", taken) == "Alison_Smock_2.md"
    assert doc_filename("a/b: c?", taken) == "a_b_c.md"


def test_doc_text():
    assert doc_text("Ann", "# Ann\n\nText.") == "# Ann\n\nText."
    assert doc_text("Paris", "Paris is a city.") == "# Paris\n\nParis is a city."
    assert len(doc_text("Big", "x" * (MAX_DOC_BYTES + 10)).encode()) == MAX_DOC_BYTES


def test_corpus_files(corpus_path):
    assert list(corpus_files(corpus_path)) == [
        ("Ann_Lee.md", DOCS[0]["contents"]),
        ("Bea_Lee.md", DOCS[1]["contents"]),
    ]


def test_write_corpus_dir(corpus_path):
    corpus_dir = write_corpus_dir(corpus_path)
    assert {f.name: f.read_text() for f in corpus_dir.iterdir()} == dict(corpus_files(corpus_path))
    assert write_corpus_dir(corpus_path) == corpus_dir  # kept, not rewritten


def test_format_output():
    assert format_output("hi\n", 0) == "hi\n"
    assert format_output("", 0) == "[no output]"
    assert format_output("oops\n", 2) == "oops\n\n[exit code 2]"
    assert format_output("", 137, timed_out_after=60) == "\n[the command timed out after 60 s]"


def test_unknown_backend(corpus_path):
    with pytest.raises(ValueError, match="Unknown sandbox backend"):
        open_sandbox("docker", corpus_path)


@pytest.mark.parametrize("corpus, mounts", [(None, None), ("corpus", {"/run/mcp/r.sock": "/tmp/r.sock"})])
def test_shellsim_needs_a_corpus_and_takes_no_mounts(corpus_path, corpus, mounts):
    with pytest.raises(ValueError, match="use bwrap"):
        open_sandbox("shellsim", corpus and corpus_path, mounts=mounts)


def test_bwrap_needs_rootfs(corpus_path):
    with pytest.raises(ValueError, match="root filesystem"):
        open_sandbox("bwrap", corpus_path)


def require_backend(backend: str) -> None:
    """Skip the test if this machine can't run `backend`."""
    if backend == "shellsim":
        pytest.importorskip("shellsim")
    elif shutil.which("bwrap") is None or not os.environ.get("SANDBOX_ROOTFS"):
        pytest.skip("needs bubblewrap and SANDBOX_ROOTFS")


@pytest.fixture
def make_sandbox(request: pytest.FixtureRequest, corpus_path: Path) -> Iterator[Callable[..., Sandbox]]:
    """A factory of sandboxes over the test corpus, on the backend in `request.param`."""
    require_backend(request.param)
    sandboxes: list[Sandbox] = []

    def make(**kwargs: Any) -> Sandbox:
        sandbox = open_sandbox(request.param, corpus_path, rootfs=os.environ.get("SANDBOX_ROOTFS"), **kwargs)
        sandboxes.append(sandbox)
        return sandbox

    yield make
    for sandbox in sandboxes:
        sandbox.close()


@pytest.mark.parametrize("make_sandbox", BACKENDS, indirect=True)
class TestSandboxContract:
    """What an episode relies on, whichever backend runs its commands."""

    def test_search_and_read(self, make_sandbox):
        sandbox = make_sandbox()
        assert sandbox.run(f"grep -l chess {CORPUS_DIR}/*.md").strip() == f"{CORPUS_DIR}/Bea_Lee.md"
        assert "Bea Lee" in sandbox.run(f"cat {CORPUS_DIR}/Ann_Lee.md")

    def test_answer_file(self, make_sandbox):
        sandbox = make_sandbox()
        assert sandbox.read_answer() is None
        sandbox.run("echo 'Bea Lee' > output.md")  # the working dir is /app
        assert sandbox.read_answer() == "Bea Lee\n"

    def test_commands_start_in_work_dir(self, make_sandbox):
        sandbox = make_sandbox()
        assert sandbox.run("cd /tmp && pwd").strip() == "/tmp"
        assert sandbox.run("pwd").strip() == WORK_DIR

    def test_files_persist_between_commands(self, make_sandbox):
        sandbox = make_sandbox()
        sandbox.run("echo note > /tmp/x")
        assert sandbox.run("cat /tmp/x").strip() == "note"

    def test_sandboxes_are_isolated(self, make_sandbox):
        first, second = make_sandbox(), make_sandbox()
        first.run("echo note > /tmp/x")
        assert "No such file" in second.run("cat /tmp/x")

    def test_failure_and_timeout_notes(self, make_sandbox):
        sandbox = make_sandbox(timeout=0.5)
        assert "[exit code 1]" in sandbox.run(f"grep -l nothing {CORPUS_DIR}/*.md")
        assert "timed out after 0.5 s" in sandbox.run("while true; do :; done")
        assert sandbox.run("echo alive").strip() == "alive"
        assert sandbox.run("true") == "[no output]"

    def test_close_twice(self, make_sandbox):
        sandbox = make_sandbox()
        sandbox.close()
        sandbox.close()


@pytest.mark.parametrize("make_sandbox", ["bwrap"], indirect=True)
class TestBwrapSandbox:
    """What only real processes can show."""

    def test_corpus_is_read_only(self, make_sandbox):
        sandbox = make_sandbox()
        assert "Read-only file system" in sandbox.run(f"echo x > {CORPUS_DIR}/Ann_Lee.md")

    def test_no_network_and_no_host_files(self, make_sandbox, corpus_path):
        sandbox = make_sandbox()
        with socket.create_server(("127.0.0.1", 0)) as server:  # a port open on the host
            connect = f"import socket; socket.create_connection(('127.0.0.1', {server.getsockname()[1]}), 2)"
            assert "ConnectionRefusedError" in sandbox.run(f'python3 -c "{connect}"')
        assert "No such file" in sandbox.run(f"ls {corpus_path}")

    def test_background_processes_end_with_their_command(self, make_sandbox):
        sandbox = make_sandbox()
        sandbox.run("sleep 60 & echo $! > /tmp/pid")
        assert "gone" in sandbox.run("kill -0 $(cat /tmp/pid) 2> /dev/null || echo gone")

    def test_memory_limit(self, make_sandbox):
        sandbox = make_sandbox(memory_gib=1)
        assert "MemoryError" in sandbox.run("python3 -c 'bytearray(2 * 1024**3)'")
        assert sandbox.run("python3 -c 'print(len(bytearray(1024**2)))'").strip() == str(1024**2)

    def test_file_size_limit(self, make_sandbox):
        sandbox = make_sandbox()
        too_big = sandbox.max_file_kib + 1024
        sandbox.run(f"head -c {too_big}K /dev/zero > /tmp/big")
        assert sandbox.run("stat -c %s /tmp/big").strip() == str(sandbox.max_file_kib * 1024)
        # A harness runs unlimited, and limits the commands it starts instead
        sandbox.execute(f"head -c {too_big}K /dev/zero > /tmp/big", limited=False)
        assert sandbox.run("stat -c %s /tmp/big").strip() == str(too_big * 1024)

    def test_commands_arrive_intact(self, make_sandbox):
        sandbox = make_sandbox()
        assert sandbox.run("echo 'naïve — ü'").strip() == "naïve — ü"
        assert sandbox.run("cat << 'EOF'\nline one\n  line two\nEOF").splitlines() == [
            "line one",
            "  line two",
        ]

    def test_shared_dir_is_read_only(self, make_sandbox):
        sandbox = make_sandbox()
        assert "Read-only file system" in sandbox.run(f"touch {SHARED_DIR}/x")

    def test_host_stops_waiting_on_a_stopped_loop(self, make_sandbox):
        sandbox = make_sandbox(timeout=1)
        # The command's parent is `timeout`, whose parent is the loop that answers the host
        start = time.monotonic()
        output = sandbox.run("kill -STOP $(awk '{print $4}' /proc/$PPID/stat)")
        assert output == "[the sandbox has stopped]"
        assert time.monotonic() - start < 1 + REPLY_GRACE + 5

    def test_outlives_the_thread_that_opened_it(self, make_sandbox):
        with ThreadPoolExecutor(max_workers=1) as executor:
            sandbox = executor.submit(make_sandbox).result()
        assert sandbox.run("echo alive").strip() == "alive"

    def test_closed_sandbox_reports_it(self, make_sandbox):
        sandbox = make_sandbox()
        sandbox.close()
        assert sandbox.run("echo hi") == "[the sandbox has stopped]"
        assert sandbox.read_answer() is None


def test_bwrap_harness_output_is_capped(corpus_path):
    """A harness runs without the per-process file-size limit, but its output is still capped."""
    require_backend("bwrap")
    with BwrapSandbox(
        write_corpus_dir(corpus_path), os.environ["SANDBOX_ROOTFS"], max_output_mib=1
    ) as sandbox:
        output, exit_code, _ = sandbox.execute("head -c 2M /dev/zero", limited=False)
    assert len(output) == 1024**2 and exit_code != 0


def processes_running(argv: list[str]) -> list[int]:
    """Return the pids of this machine's processes whose command line is exactly `argv`."""
    cmdline = b"\0".join(arg.encode() for arg in argv) + b"\0"
    pids = []
    for proc in Path("/proc").iterdir():
        try:
            if proc.name.isdigit() and (proc / "cmdline").read_bytes() == cmdline:
                pids.append(int(proc.name))
        except OSError:  # the process ended meanwhile
            continue
    return pids


def wait_for(condition: Callable[[], bool], seconds: float = 10.0) -> bool:
    """Poll `condition` until it holds or `seconds` pass; return whether it held."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_bwrap_sandbox_ends_with_its_host_process(corpus_path):
    """A host killed mid-command takes its sandbox with it, even a sandbox opened on another thread."""
    require_backend("bwrap")
    sleep = ["sleep", f"{os.getpid()}.{time.monotonic_ns()}"]  # a command no other process runs
    host_code = (
        "import os, sys\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "from phantom_envs.terminal.sandbox import open_sandbox\n"
        "rootfs = os.environ['SANDBOX_ROOTFS']\n"
        "with ThreadPoolExecutor(max_workers=1) as executor:\n"
        "    sandbox = executor.submit(open_sandbox, 'bwrap', sys.argv[1], timeout=600, rootfs=rootfs).result()\n"
        "sandbox.run(sys.argv[2])\n"
    )
    host = subprocess.Popen([sys.executable, "-c", host_code, str(corpus_path), shlex.join(sleep)])
    try:
        assert wait_for(lambda: processes_running(sleep)), "the sandbox's command never started"
        host.kill()
        host.wait()
        assert wait_for(lambda: not processes_running(sleep)), "the sandbox outlived its host"
    finally:
        host.kill()
        for pid in processes_running(sleep):
            os.kill(pid, signal.SIGKILL)
