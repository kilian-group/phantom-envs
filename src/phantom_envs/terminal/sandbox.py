"""The sandbox of a shell episode: where the agent's commands run.

Every backend keeps the same contract, so an episode behaves the same on any of them:

- The corpus is under `/corpus`, one file per document.
- Each command runs in a fresh bash in `/app`. Files persist between commands; the working
  directory and shell variables don't.
- A command's output is its stdout and stderr, with a note when it timed out or failed.
- The agent's answer is the file `/app/output.md`.

Backends:

- `shellsim` (see `shellsim.py`): an in-process simulated bash. No setup, and no real programs:
  only ShellSim's own grep, sed, awk and so on.
- `bwrap` (see `bwrap.py`): real processes in a bubblewrap sandbox over a pinned root filesystem.
  Runs anything the image has, including a coding-agent harness.

Usage:

    with open_sandbox("shellsim", "data/envs/pw/corpus.jsonl") as sandbox:
        sandbox.run("grep -l 'Ann Lee' /corpus/*.md")
        answer = sandbox.read_answer()  # the contents of /app/output.md, or None
"""

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Self

# The backends open_sandbox can create
BACKENDS: tuple[str, ...] = ("shellsim", "bwrap")


class Sandbox(ABC):
    """One episode's sandbox. Close it when the episode ends."""

    @abstractmethod
    def run(self, command: str) -> str:
        """Run one bash command in `/app`.

        Args:
            command: The command.

        Returns:
            Its stdout and stderr, with a note on a timeout or a non-zero exit code.
        """

    @abstractmethod
    def read_answer(self) -> str | None:
        """Read the answer file, for the verifier.

        Returns:
            The contents of `/app/output.md`, or None if the agent wrote none.
        """

    def close(self) -> None:  # noqa: B027 (optional hook)
        """Free the sandbox's resources. Safe to call more than once."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def format_output(output: str, exit_code: int, timed_out_after: float | None = None) -> str:
    """Format a command's result as the text the agent reads.

    Args:
        output: The command's stdout and stderr.
        exit_code: Its exit code.
        timed_out_after: The time limit in seconds, if the command was stopped for exceeding it.

    Returns:
        The output, followed by a note when the command timed out or failed; never empty.
    """
    if timed_out_after is not None:
        output += f"\n[the command timed out after {timed_out_after:g} s]"
    elif exit_code != 0:
        output += f"\n[exit code {exit_code}]"
    return output or "[no output]"


def open_sandbox(
    backend: str,
    corpus_path: str | Path,
    timeout: float = 60.0,
    memory_gib: int = 4,
    rootfs: str | Path | None = None,
) -> Sandbox:
    """Create a sandbox over a corpus.

    A process's first sandbox over a corpus prepares it (ShellSim loads it into memory; bwrap writes
    it as files next to `corpus.jsonl`, once), which takes a while for a large corpus.

    Args:
        backend: `shellsim` or `bwrap`.
        corpus_path: The `corpus.jsonl` to serve under `/corpus`.
        timeout: Wall-clock limit per command, in seconds.
        memory_gib: Memory limit in GiB: of the whole machine for ShellSim, of each process for bwrap.
        rootfs: For bwrap, the root filesystem that `scripts/sandbox/build_rootfs.sh` exported.

    Returns:
        The sandbox.

    Raises:
        ValueError: For an unknown backend, or bwrap without a root filesystem.
    """
    if backend == "shellsim":
        from phantom_envs.terminal.shellsim import ShellSimSandbox, corpus_image

        return ShellSimSandbox(corpus_image(corpus_path), timeout=timeout, memory_gib=memory_gib)
    if backend == "bwrap":
        from phantom_envs.terminal.bwrap import BwrapSandbox
        from phantom_envs.terminal.corpus_files import write_corpus_dir

        if not rootfs:
            raise ValueError(
                "The bwrap sandbox needs a root filesystem: build one with "
                "scripts/sandbox/build_rootfs.sh and set SANDBOX_ROOTFS in .env"
            )
        return BwrapSandbox(write_corpus_dir(corpus_path), rootfs, timeout=timeout, memory_gib=memory_gib)
    raise ValueError(f"Unknown sandbox backend {backend!r} (expected one of {BACKENDS})")
