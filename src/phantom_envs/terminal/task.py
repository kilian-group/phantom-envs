"""The terminal task: where the agent works, what it is asked, and how its answer is verified.

These are the parts of a terminal episode that every sandbox and every agent loop share. The agent
finds the documents under `CORPUS_DIR`, works in `WORK_DIR`, and writes its final answer to
`ANSWER_PATH`; `verify` scores what it wrote.

Usage:

    messages = build_messages("Who is the mother of Ann Lee?")
    ...  # the agent runs commands in a sandbox
    scores = verify(sandbox.read_answer(), gts=["Bea Lee"], data_source="pw")  # {"em": 1.0, ...}
"""

from phantom_envs.scoring import METRICS, score

# The corpus, one read-only file per document
CORPUS_DIR: str = "/corpus"
# The agent's working dir
WORK_DIR: str = "/app"
# The file the verifier grades
ANSWER_FILE: str = "output.md"
ANSWER_PATH: str = f"{WORK_DIR}/{ANSWER_FILE}"

SYSTEM_PROMPT: str = "You are a helpful assistant."

# As in a Harbor task's instruction.md, the user message holds the whole task: models follow the
# answer-file instruction far more reliably there than in the system message.
INSTRUCTION: str = (
    f"Answer the question below using the documents in {CORPUS_DIR}, one file per document. "
    "Search and read them with the bash tool, and base your answer only on what they say.\n\n"
    f"Write your final answer to {ANSWER_PATH} as a single line, with multiple answers separated by "
    f"commas (for example: Alice, Bob). Only {ANSWER_PATH} is graded.\n\n"
    "Question: {question}"
)

# The tools an agent can be given, by the name of the tool set. With a search tool the corpus is not
# in the sandbox: the agent sees the documents only through the tool, whatever retriever answers it.
BASH: str = "bash"
BASH_AND_SEARCH: str = "bash+search"
SEARCH: str = "search"
TOOLSETS: tuple[str, ...] = (BASH, BASH_AND_SEARCH, SEARCH)

# The instructions name "the search tool", not its full name, which depends on who declares it
SEARCH_INSTRUCTIONS: dict[str, str] = {
    BASH_AND_SEARCH: (
        "Answer the question below using a collection of documents. The documents are not on disk: "
        "find and read them with the search tool, which returns the documents most relevant to a "
        "query, and base your answer only on what they say. The bash tool is for everything else, "
        "such as keeping notes or computing.\n\n"
        f"Write your final answer to {ANSWER_PATH} as a single line, with multiple answers separated by "
        f"commas (for example: Alice, Bob). Only {ANSWER_PATH} is graded.\n\n"
        "Question: {question}"
    ),
    SEARCH: (
        "Answer the question below using a collection of documents. Find and read them with the "
        "search tool, which returns the documents most relevant to a query, and base your answer only "
        "on what they say.\n\n"
        f"Write your final answer to {ANSWER_PATH} with the write tool, as a single line, with multiple "
        f"answers separated by commas (for example: Alice, Bob). Only {ANSWER_PATH} is graded.\n\n"
        "Question: {question}"
    ),
}


def has_search(tools: str) -> bool:
    """Whether a tool set includes the search tool (so the sandbox holds no corpus)."""
    return tools in SEARCH_INSTRUCTIONS


def build_instruction(question: str, tools: str = BASH) -> str:
    """Return the task text for a question: what to do, and where to write the answer.

    Args:
        question: The question to answer.
        tools: The agent's tool set, one of TOOLSETS.

    Returns:
        The instruction, as the user message of an episode or a harness's prompt.
    """
    template = SEARCH_INSTRUCTIONS[tools] if has_search(tools) else INSTRUCTION
    return template.format(question=question)


def build_messages(question: str) -> list[dict[str, str]]:
    """Return the system and user chat messages that open a terminal episode.

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
