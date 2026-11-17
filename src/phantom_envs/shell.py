"""Shell episodes: a coding agent answers a question by running bash over the corpus as files.

As in Terminal-Bench, the environment, the agent's attempt and the verifier are separate:

- **Environment:** a sandbox that holds the data dir's ``corpus.jsonl`` as one file per document
  under ``/corpus``, with ``/app`` as the working dir.
- **Attempt:** the agent calls a ``bash`` tool in Qwen3.5's tool-call format (``<tool_call>
  <function=bash><parameter=command>``) and gets the outputs back as tool responses. It ends on a
  turn without tool calls, or after ``max_turns`` turns.
- **Verifier:** reads the answer file the agent wrote, ``/app/output.md`` (one line, answers
  separated by commas), and scores it against the gold answers. No file scores 0.

The sandbox is a ShellSim machine: an in-process simulated bash with an in-memory filesystem, its own
grep / rg / sed / awk / find, and no Python, network or host access. Each process loads a corpus
once into a frozen image, which every episode's machine shares copy-on-write.

Trainer adapters drive :class:`ShellProtocol` and run the commands in a :class:`ShellSandbox`::

    protocol = ShellProtocol(gts=["Alice"], data_source="pw", max_turns=20)
    sandbox = ShellSandbox(corpus_image("data/envs/pw/corpus.jsonl"))
    state = ShellEpisodeState()
    result = protocol.act(state, turn)  # once per agent turn
    if result.done:
        reward = protocol.reward(sandbox.read_answer())
    else:
        outputs = [tool_error(call) or sandbox.run(call.arguments["command"]) for call in result.calls]
        observations = protocol.observe(state, outputs)
"""

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phantom_envs.scoring import METRICS, score

CORPUS_DIR = "/corpus"
WORK_DIR = "/app"
ANSWER_PATH = "/app/output.md"
# ShellSim's rg refuses to read a file over 4 MiB, and stops the whole search there
MAX_DOC_BYTES = 4_000_000
# Room for the agent's own files on top of the corpus
SCRATCH_BYTES = 256 * 1024**2

TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


@dataclass
class ToolCall:
    """One tool call of an agent turn."""

    name: str = field(metadata={"description": "Function name, e.g. bash."})
    arguments: dict[str, str] = field(metadata={"description": "Parameter values by name, as text."})


def parse_tool_calls(turn: str) -> list[ToolCall]:
    """Return the tool calls of an agent turn, in order (Qwen3.5's XML format)."""
    return [
        ToolCall(name=name, arguments=dict(PARAMETER_RE.findall(body)))
        for name, body in TOOL_CALL_RE.findall(turn)
    ]


def tool_error(call: ToolCall) -> str | None:
    """Why a tool call can't run, as the message the agent gets back, or None if it can."""
    if call.name != "bash":
        return f"Error: unknown tool {call.name!r}; the only tool is bash."
    if "command" not in call.arguments:
        return "Error: bash needs a command parameter."
    return None


def truncate(text: str, max_chars: int) -> str:
    """Cut a tool output to ``max_chars`` characters, saying how much was left out."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[output truncated: showed {max_chars} of {len(text)} characters]"


@dataclass
class ShellEpisodeState:
    """Mutable state of one shell episode."""

    turns: int = field(default=0, metadata={"description": "Agent turns taken so far."})
    commands: int = field(default=0, metadata={"description": "Tool calls made so far."})


@dataclass
class ShellTurnResult:
    """Outcome of one agent turn."""

    done: bool = field(metadata={"description": "Whether the attempt has ended (time to verify)."})
    calls: list[ToolCall] = field(
        default_factory=list, metadata={"description": "Tool calls to run next; empty when done."}
    )


@dataclass
class ShellProtocol:
    """Episode logic for one question: when the attempt ends, which commands to run, and the verifier."""

    gts: list[str] = field(metadata={"description": "Gold answers of the question."})
    data_source: str = field(metadata={"description": "Data source of the question (selects the scorer)."})
    max_turns: int = field(metadata={"description": "Maximum number of agent turns per episode."})
    metric: str = field(default="f1", metadata={"description": "Reward metric: em, subem or f1."})
    max_output_chars: int = field(
        default=8000, metadata={"description": "Tool outputs are cut to this many characters."}
    )

    def act(self, state: ShellEpisodeState, turn: str) -> ShellTurnResult:
        """Record an agent turn and decide what happens next."""
        state.turns += 1
        calls = parse_tool_calls(turn)
        if not calls or state.turns >= self.max_turns:
            return ShellTurnResult(done=True)
        state.commands += len(calls)
        return ShellTurnResult(done=False, calls=calls)

    def observe(self, state: ShellEpisodeState, outputs: list[str]) -> list[str]:
        """Return the tool responses for a turn's outputs, each cut to ``max_output_chars``."""
        return [truncate(output, self.max_output_chars) for output in outputs]

    def reward(self, answer: str | None) -> float:
        """Verify the answer file's contents (None: no file); 0.0 without an answer."""
        return self.scores(answer)[self.metric]

    def scores(self, answer: str | None) -> dict[str, float]:
        """Score the answer file's contents with every metric (all 0.0 without an answer)."""
        answer = (answer or "").strip()
        return {m: score(answer, self.gts, self.data_source, m) if answer else 0.0 for m in METRICS}


def doc_filename(title: str, taken: set[str]) -> str:
    """A file name for a document: its title with unsafe characters replaced, unique in ``taken``."""
    stem = re.sub(r"[^\w.,()'-]+", "_", title).strip("_")[:120] or "untitled"
    name, i = f"{stem}.md", 2
    while name in taken:
        name, i = f"{stem}_{i}.md", i + 1
    taken.add(name)
    return name


def doc_text(title: str, contents: str) -> str:
    """A document's file contents: a markdown title line (PhantomEnvs articles have one), then the text."""
    text = contents if contents.startswith("# ") else f"# {title}\n\n{contents}"
    data = text.encode()
    if len(data) > MAX_DOC_BYTES:
        text = data[:MAX_DOC_BYTES].decode(errors="ignore")
    return text


@dataclass
class CorpusImage:
    """A corpus frozen into a ShellSim image, for many sandboxes to share."""

    native: Any = field(metadata={"description": "The frozen ShellSim image (a NativeImage)."})
    num_docs: int = field(metadata={"description": "Number of documents under CORPUS_DIR."})
    num_bytes: int = field(metadata={"description": "Total size of the document files."})


def build_corpus_image(corpus_path: str | Path) -> CorpusImage:
    """Load ``corpus.jsonl`` into a frozen ShellSim image, one file per document under CORPUS_DIR,
    with an empty WORK_DIR.

    ShellSim has no public API for a filesystem shared between machines (``Environment.mount``
    copies into each machine and stops at 10,000 files), so this freezes a machine through its
    native module, as ``shellsim.Context`` does for its own installations.
    """
    from shellsim import Limits, _native
    from shellsim._api import Environment

    limits = Limits(cpu=10**15, memory=64 * 1024**3, disk=1024**4)
    builder = Environment._from_native(
        _native.NativeEnvironment(limits.cpu, limits.memory, limits.disk, limits.output, False), limits
    )
    builder.mkdir(CORPUS_DIR)
    builder.mkdir(WORK_DIR)
    taken: set[str] = set()
    num_docs = num_bytes = 0
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            text = doc_text(doc["title"], doc["contents"])
            builder.write_file(f"{CORPUS_DIR}/{doc_filename(doc['title'], taken)}", text)
            num_docs += 1
            num_bytes += len(text.encode())
    return CorpusImage(native=builder._native.freeze(), num_docs=num_docs, num_bytes=num_bytes)


_images: dict[str, CorpusImage] = {}
_images_lock = threading.Lock()


def corpus_image(corpus_path: str | Path) -> CorpusImage:
    """The process's frozen image of ``corpus_path``, built on first use (BrowseComp-Plus: ~20 s)."""
    key = str(Path(corpus_path).resolve())
    with _images_lock:
        if key not in _images:
            _images[key] = build_corpus_image(key)
        return _images[key]


class ShellSandbox:
    """One episode's ShellSim machine over a shared corpus image, starting in WORK_DIR."""

    def __init__(self, image: CorpusImage, timeout: float = 60.0, memory_gib: int = 4):
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
        """The answer file's contents, for the verifier; None if the agent wrote none."""
        from shellsim import SimulationError

        try:
            return self.machine.read_file(ANSWER_PATH).decode(errors="replace")
        except SimulationError:  # no such file
            return None

    def run(self, command: str) -> str:
        """Run one command; return its stdout and stderr, with a note on a timeout or failure."""
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
