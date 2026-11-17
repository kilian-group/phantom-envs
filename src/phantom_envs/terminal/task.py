"""The shell task: where the agent works, what it is asked, and how its answer is verified.

These are the parts of a shell episode that every sandbox and every agent loop share. The agent
finds the documents under :data:`CORPUS_DIR`, works in :data:`WORK_DIR`, and writes its final answer
to :data:`ANSWER_PATH`; :func:`verify` scores what it wrote.

Usage::

    messages = build_messages("Who is the mother of Ann Lee?")
    ...  # the agent runs commands in a sandbox
    scores = verify(sandbox.read_answer(), gts=["Bea Lee"], data_source="pw")  # {"em": 1.0, ...}
"""

from phantom_envs.scoring import METRICS, score

CORPUS_DIR: str = "/corpus"
WORK_DIR: str = "/app"
ANSWER_FILE: str = "output.md"
ANSWER_PATH: str = f"{WORK_DIR}/{ANSWER_FILE}"

SYSTEM_PROMPT: str = "You are a helpful assistant."

# As in a Harbor task's instruction.md, the user message holds the whole task. In the system message,
# base Qwen3.5-9B wrote the answer file in 2 of 16 runs; in the user message, in 15 of 16.
INSTRUCTION: str = (
    f"Answer the question below using the documents in {CORPUS_DIR}, one file per document. "
    "Search and read them with the bash tool, and base your answer only on what they say.\n\n"
    f"Write your final answer to {ANSWER_PATH} as a single line, with multiple answers separated by "
    f"commas (for example: Alice, Bob). Only {ANSWER_PATH} is graded.\n\n"
    "Question: {question}"
)


def build_instruction(question: str) -> str:
    """Return the task text for ``question``: what to do, and where to write the answer.

    Args:
        question: The question to answer.

    Returns:
        The instruction, as the user message of an episode or a harness's prompt.
    """
    return INSTRUCTION.format(question=question)


def build_messages(question: str) -> list[dict[str, str]]:
    """Return the system and user chat messages that open a shell episode.

    Args:
        question: The question to answer.

    Returns:
        The system message and the user message, in order.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_instruction(question)},
    ]


def has_answer(answer: str | None) -> bool:
    """Whether the agent wrote a non-empty answer file.

    Args:
        answer: The answer file's contents, or None if there is no file.

    Returns:
        True if the file exists and holds more than whitespace.
    """
    return bool((answer or "").strip())


def verify(answer: str | None, gts: list[str], data_source: str) -> dict[str, float]:
    """Score the answer file against the gold answers with every metric.

    Args:
        answer: The answer file's contents (one line, answers separated by commas), or None if the
            agent wrote no file.
        gts: Gold answers of the question.
        data_source: Data source of the question, which selects the scorer.

    Returns:
        The score per metric (em, subem, f1); all 0.0 without an answer.
    """
    if not has_answer(answer):
        return dict.fromkeys(METRICS, 0.0)
    return {metric: score(answer.strip(), gts, data_source, metric) for metric in METRICS}
