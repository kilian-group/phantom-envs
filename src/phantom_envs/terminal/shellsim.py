"""The ShellSim sandbox: an in-process simulated bash over the corpus.

A ShellSim machine has an in-memory filesystem, its own grep / rg / sed / awk / find, and no Python,
network or host access. Each process loads a corpus once into a frozen image, which every episode's
machine shares copy-on-write, so a new sandbox costs milliseconds.

Usage::

    sandbox = ShellSimSandbox(corpus_image("data/envs/pw/corpus.jsonl"))
    sandbox.run("grep -l 'Ann Lee' /corpus/*.md")
    sandbox.read_answer()  # the contents of /app/output.md, or None
"""

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phantom_envs.terminal.corpus_files import corpus_files
from phantom_envs.terminal.task import ANSWER_PATH, CORPUS_DIR, WORK_DIR

# Room for the agent's own files on top of the corpus
SCRATCH_BYTES: int = 256 * 1024**2


@dataclass
class CorpusImage:
    """A corpus frozen into a ShellSim image, for many sandboxes to share."""

    native: Any = field(metadata={"description": "The frozen ShellSim image (a NativeImage)."})
    num_docs: int = field(metadata={"description": "Number of documents under CORPUS_DIR."})
    num_bytes: int = field(metadata={"description": "Total size of the document files."})


def build_corpus_image(corpus_path: str | Path) -> CorpusImage:
    """Load a corpus into a frozen ShellSim image.

    ShellSim has no public API for a filesystem shared between machines (``Environment.mount``
    copies into each machine and stops at 10,000 files), so this freezes a machine through its
    native module, as ``shellsim.Context`` does for its own installations.

    Args:
        corpus_path: The ``corpus.jsonl`` to load.

    Returns:
        The image: one file per document under CORPUS_DIR, and an empty WORK_DIR.
    """
    from shellsim import Limits, _native
    from shellsim._api import Environment

    limits = Limits(cpu=10**15, memory=64 * 1024**3, disk=1024**4)
    builder = Environment._from_native(
        _native.NativeEnvironment(limits.cpu, limits.memory, limits.disk, limits.output, False), limits
    )
    builder.mkdir(CORPUS_DIR)
    builder.mkdir(WORK_DIR)
    num_docs = num_bytes = 0
    for name, text in corpus_files(corpus_path):
        builder.write_file(f"{CORPUS_DIR}/{name}", text)
        num_docs += 1
        num_bytes += len(text.encode())
    return CorpusImage(native=builder._native.freeze(), num_docs=num_docs, num_bytes=num_bytes)


_images: dict[str, CorpusImage] = {}
_images_lock = threading.Lock()


def corpus_image(corpus_path: str | Path) -> CorpusImage:
    """Return this process's frozen image of a corpus, building it on first use.

    Args:
        corpus_path: The ``corpus.jsonl`` to load (BrowseComp-Plus takes ~20 s).

    Returns:
        The image, shared by every sandbox of this process.
    """
    key = str(Path(corpus_path).resolve())
    with _images_lock:
        if key not in _images:
            _images[key] = build_corpus_image(key)
        return _images[key]


class ShellSimSandbox:
    """One episode's ShellSim machine over a shared corpus image, starting in WORK_DIR."""

    def __init__(self, image: CorpusImage, timeout: float = 60.0, memory_gib: int = 4) -> None:
        """Create the machine.

        Args:
            image: The corpus image.
            timeout: Wall-clock limit per command, in seconds.
            memory_gib: Memory limit of the machine, in GiB.
        """
        from shellsim import Limits
        from shellsim._api import Environment

        # CPU and output limits would end the machine for good, so commands are bounded by wall time
        limits = Limits(
            cpu=10**15, memory=memory_gib * 1024**3, disk=image.num_bytes + SCRATCH_BYTES, output=1024**5
        )
        self.machine = Environment._from_native(
            image.native.environment(limits.cpu, limits.memory, limits.disk, limits.output, False), limits
        )
        self.timeout = timeout
        self.machine.run(f"cd {WORK_DIR}")

    def read_answer(self) -> str | None:
        """Read the answer file, for the verifier.

        Returns:
            The file's contents, or None if the agent wrote none.
        """
        from shellsim import SimulationError

        try:
            return self.machine.read_file(ANSWER_PATH).decode(errors="replace")
        except SimulationError:  # no such file
            return None

    def run(self, command: str) -> str:
        """Run one command.

        Args:
            command: The bash command.

        Returns:
            Its stdout and stderr, with a note on a timeout or a non-zero exit code.
        """
        from shellsim import CancellationToken

        if self.machine.terminated:
            return "[the sandbox has stopped: it ran out of resources]"
        token = CancellationToken()
        timer = threading.Timer(self.timeout, token.cancel)
        timer.start()
        try:
            result = self.machine.run(command, cancellation=token)
        finally:
            timer.cancel()
        text = (result.stdout + result.stderr).decode(errors="replace")
        if result.interrupted:
            text += f"\n[the command timed out after {self.timeout:g} s]"
        elif result.returncode != 0:
            text += f"\n[exit code {result.returncode}]"
        return text or "[no output]"
