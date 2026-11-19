"""The bubblewrap sandbox: real processes over a pinned root filesystem, with no network.

bubblewrap (`bwrap`, https://github.com/containers/bubblewrap) starts a process in fresh Linux
namespaces without root. Starting one is slow next to running a command, and serialized
kernel-wide, so a sandbox starts once per episode: a small bash loop inside it then runs each
command in a fresh bash.

What a command sees:

- `/usr` and `/etc` of the image that `scripts/sandbox/build_rootfs.sh` exported, read-only;
- the corpus under `/corpus`, read-only;
- `/app` (the working dir), `/tmp` and its home dir, each an empty in-memory filesystem with a
  size cap;
- no network and no host files.

Limits: a wall-clock timeout per command, and per-process caps on address space, file size and the
number of processes. These bound each process, not the sandbox's total memory; a hard cap on that
needs a cgroup. A sandbox ends with the process that started it.

Usage:

    with BwrapSandbox("data/envs/pw/terminal/corpus", rootfs="/path/to/rootfs") as sandbox:
        sandbox.run("grep -l 'Ann Lee' /corpus/*.md")
        answer = sandbox.read_answer()

A harness that runs inside the sandbox shares files and sockets with it through
`BwrapSandbox.shared_dir`, which commands see as `SHARED_DIR`. Commands can replace what is in it, so
the host writes there only before the first command runs.
"""

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from phantom_envs.terminal.sandbox import Sandbox, format_output
from phantom_envs.terminal.task import ANSWER_PATH, CORPUS_DIR, WORK_DIR

# Commands run as root of the sandbox's own user namespace, as in a container; it has no rights outside
HOME_DIR: str = "/root"
# Where commands see the sandbox's shared_dir
SHARED_DIR: str = "/run/sandbox"
# The PATH of commands; the rest of the host's environment is cleared
PATH: str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Largest file a command may write by default, in MiB
MAX_FILE_MIB: int = 64
# Most of a command's output the host reads
MAX_OUTPUT_BYTES: int = 1024**3
# Seconds past a command's timeout before the host stops waiting and kills the sandbox
REPLY_GRACE: float = 5.0

# The loop that runs inside the sandbox. For each command, the host sends one line, "<timeout in
# seconds> <address space in KiB> <file size in KiB> <command length in bytes>" (a limit may be
# "unlimited"), then the command. The loop runs it with its output on file descriptor $2, a file the
# host opened, and answers with the exit code. $1 is the limit on processes.
#
# Nothing the host reads or writes while commands run has a path inside the sandbox, so a command
# can't point the host at its own files.
READY: bytes = b"ready\n"
COMMAND_LOOP: str = r"""
ulimit -u "$1"
# read -N counts bytes in the C locale; the loop's commands keep the sandbox's locale
LC_ALL=C
echo ready
while read -r seconds memory file_size length && read -r -N "$length" command; do
    (
        ulimit -v "$memory" -f "$file_size"
        exec timeout --signal=KILL "$seconds" bash -c "$command"
    ) >&"$2" 2>&1 < /dev/null
    echo $?
    # Stop what the command left running in the background
    kill -KILL -1 2> /dev/null
done
"""
# The exit code of a command that `timeout` killed
KILLED: int = 128 + signal.SIGKILL
STOPPED: str = "[the sandbox has stopped]"

# bwrap's --die-with-parent stops the sandbox when the *thread* that started bwrap exits, not only its
# process. Sandboxes are often opened from worker threads, so all of them start from this one thread,
# which lives as long as the process.
_spawner: ThreadPoolExecutor | None = None
_spawner_pid: int | None = None
_spawner_lock = threading.Lock()


def start_process(argv: list[str], **popen_kwargs: Any) -> subprocess.Popen:
    """Start a process from a thread that lives as long as this process.

    Args:
        argv: The command line.
        **popen_kwargs: Passed to `subprocess.Popen`.

    Returns:
        The started process.
    """
    global _spawner, _spawner_pid
    with _spawner_lock:
        # A forked child inherits the executor but not its thread, so it starts its own
        if _spawner_pid != os.getpid():
            _spawner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bwrap-start")
            _spawner_pid = os.getpid()
    return _spawner.submit(subprocess.Popen, argv, **popen_kwargs).result()


class BwrapSandbox(Sandbox):
    """One episode's bubblewrap sandbox over a corpus dir."""

    def __init__(
        self,
        corpus_dir: str | Path,
        rootfs: str | Path,
        timeout: float = 60.0,
        memory_gib: int = 4,
        scratch_mib: int = 256,
        max_file_mib: int = MAX_FILE_MIB,
        max_processes: int = 256,
    ) -> None:
        """Start the sandbox.

        Args:
            corpus_dir: The corpus as files (see `corpus_files.write_corpus_dir`).
            rootfs: The root filesystem that `scripts/sandbox/build_rootfs.sh` exported.
            timeout: Wall-clock limit per command, in seconds.
            memory_gib: Address-space limit of each process, in GiB.
            scratch_mib: Size of each of `/app`, `/tmp` and the home dir, in MiB.
            max_file_mib: Largest file a command may write, in MiB. A command's output counts.
            max_processes: Most processes the sandbox may hold at once.

        Raises:
            FileNotFoundError: If `bwrap` is not installed, `rootfs` is not an exported image, or
                `corpus_dir` does not exist.
            RuntimeError: If the sandbox fails to start.
        """
        rootfs = Path(rootfs)
        if shutil.which("bwrap") is None:
            raise FileNotFoundError("bubblewrap is not installed: no `bwrap` on PATH")
        if not Path(corpus_dir).is_dir():
            raise FileNotFoundError(f"No corpus dir at {corpus_dir}")
        if not (rootfs / "usr").is_dir() or not (rootfs / "etc").is_dir():
            raise FileNotFoundError(
                f"{rootfs} is not a sandbox root filesystem; build one with scripts/sandbox/build_rootfs.sh"
            )

        self.timeout = timeout
        self.memory_kib = memory_gib * 1024**2
        self.max_file_kib = max_file_mib * 1024
        self.lock = threading.Lock()  # one command at a time
        self.closed = False

        self.host_dir = Path(tempfile.mkdtemp(prefix="phantom-sandbox-"))
        self.shared_dir = self.host_dir / "shared"
        self.shared_dir.mkdir()
        # Where commands write their output: a file with no name, which only the host can open
        self.output = tempfile.TemporaryFile(dir=self.host_dir)

        argv = [
            "bwrap", "--unshare-all", "--die-with-parent", "--uid", "0", "--gid", "0", "--new-session",
            "--clearenv",
            "--setenv", "PATH", PATH, "--setenv", "HOME", HOME_DIR, "--setenv", "LANG", "C.UTF-8",
            "--ro-bind", str(rootfs / "usr"), "/usr", "--ro-bind", str(rootfs / "etc"), "/etc",
            "--proc", "/proc", "--dev", "/dev",
            "--ro-bind", str(corpus_dir), CORPUS_DIR,
            "--bind", str(self.shared_dir), SHARED_DIR,
            "--chdir", WORK_DIR,
        ]  # fmt: skip
        # /bin, /lib and so on are symlinks into /usr in the image
        for entry in sorted(os.scandir(rootfs), key=lambda e: e.name):
            if entry.is_symlink():
                argv += ["--symlink", os.readlink(entry.path), f"/{entry.name}"]
        for scratch_dir in (WORK_DIR, "/tmp", HOME_DIR):
            argv += ["--size", str(scratch_mib * 1024**2), "--tmpfs", scratch_dir]

        # bwrap reports the sandbox's first process on this pipe; close() stops the sandbox through it
        status_read, status_write = os.pipe()
        argv += ["--json-status-fd", str(status_write)]
        output_fd = self.output.fileno()
        argv += ["bash", "-c", COMMAND_LOOP, "loop", str(max_processes), str(output_fd)]
        with open(self.host_dir / "stderr", "wb") as stderr:
            self.process = start_process(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                pass_fds=[status_write, output_fd],
            )
        os.close(status_write)
        with os.fdopen(status_read) as status:
            line = status.readline()
        self.init_pid: int | None = json.loads(line)["child-pid"] if line else None
        if self.process.stdout.readline() != READY:
            error = (self.host_dir / "stderr").read_text().strip()
            self.close()
            raise RuntimeError(f"The bwrap sandbox failed to start: {error}")

    @property
    def command_limits(self) -> str:
        """The bash line that puts a command under the sandbox's per-process limits.

        A harness runs unlimited (see `execute`) and prefixes the commands it starts with this.
        """
        return f"ulimit -v {self.memory_kib} -f {self.max_file_kib}"

    def execute(
        self, command: str, timeout: float | None = None, limited: bool = True
    ) -> tuple[str, int, bool]:
        """Run one bash command in /app and return its raw result.

        Args:
            command: The command.
            timeout: Wall-clock limit in seconds; the sandbox's own by default.
            limited: Whether the command's processes get the address-space and file-size limits. A
                harness runs without them, since Node.js can't start under an address-space limit and
                the harness's own log carries every output it reads. It limits the commands it starts
                with `command_limits` instead.

        Returns:
            The command's stdout and stderr, its exit code, and whether it timed out. A sandbox that
            has stopped gives a note in place of the output and exit code -1.
        """
        timeout = self.timeout if timeout is None else timeout
        memory, file_size = (self.memory_kib, self.max_file_kib) if limited else ("unlimited", "unlimited")
        # bash can't hold NUL bytes, and the loop counts the bytes it reads
        script = command.replace("\0", "").encode()
        with self.lock:
            if self.closed or self.process.poll() is not None:
                return STOPPED, -1, False
            output_fd = self.output.fileno()
            os.ftruncate(output_fd, 0)
            os.lseek(output_fd, 0, os.SEEK_SET)
            # `timeout` bounds the command, but a command can also stop the loop that waits on it
            watchdog = threading.Timer(timeout + REPLY_GRACE, self.kill)
            watchdog.start()
            start = time.monotonic()
            try:
                self.process.stdin.write(
                    f"{timeout:g} {memory} {file_size} {len(script)}\n".encode() + script
                )
                self.process.stdin.flush()
                reply = self.process.stdout.readline()
            except (BrokenPipeError, ValueError):  # the sandbox died, or close() ran meanwhile
                reply = b""
            finally:
                watchdog.cancel()
            try:
                exit_code = int(reply)
            except ValueError:  # the sandbox stopped, or a command wrote to the loop's own stdout
                self.kill()
                return STOPPED, -1, False
            size = min(os.fstat(output_fd).st_size, MAX_OUTPUT_BYTES)
            output = os.pread(output_fd, size, 0).decode(errors="replace")
            timed_out = exit_code == KILLED and time.monotonic() - start >= timeout
            return output, exit_code, timed_out

    def run(self, command: str) -> str:
        """Run one bash command in `/app`.

        Args:
            command: The command.

        Returns:
            Its stdout and stderr, with a note on a timeout or a non-zero exit code.
        """
        output, exit_code, timed_out = self.execute(command)
        if exit_code == -1:
            return output
        return format_output(output, exit_code, self.timeout if timed_out else None)

    def read_answer(self) -> str | None:
        """Read the answer file, for the verifier.

        Returns:
            The contents of `/app/output.md`, or None if the agent wrote none.
        """
        output, exit_code, _ = self.execute(f"cat {ANSWER_PATH}")
        return output if exit_code == 0 else None

    def kill(self) -> None:
        """Kill every process in the sandbox. Its pending command, if any, reports that it stopped."""
        # Killing the first process of a PID namespace kills every process in it. bwrap, its parent,
        # is still running, so the pid can't have been reused.
        if self.process.poll() is None and self.init_pid is not None:
            try:
                os.kill(self.init_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        """Stop the sandbox's processes and delete its files. Safe to call more than once."""
        if self.closed:
            return
        self.closed = True
        self.kill()
        self.process.wait()
        self.output.close()
        for pipe in (self.process.stdin, self.process.stdout):
            pipe.close()
        shutil.rmtree(self.host_dir, ignore_errors=True)
