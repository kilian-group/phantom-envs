"""Prompts for the search agent and the shell agent, shared by training and evaluation on every
data source."""

SYSTEM_PROMPT: str = "You are a helpful and harmless assistant."

USER_PROMPT_PREFIX: str = (
    "Answer the given question using the search tool to retrieve relevant information. "
    "Follow this process:\n"
    "1. Reason about what you know and what you still need to find out with the search tool. "
    "For complex questions, break them into sub-questions and search for each part.\n"
    "2. If you need more information, call the search engine: <search> query </search>. "
    "Write focused, specific queries. If a query returns unhelpful results, try a different "
    "phrasing or a more specific sub-question.\n"
    "3. Base your final answer only on information from the search results, not on prior knowledge.\n"
    "4. Once you have enough information, provide your answer inside <answer> and </answer>. "
    "For multiple answers, use a comma-separated list: <answer>Alice, Bob</answer>. "
    "Do not include explanations inside the answer tags.\n\n"
    "Question: "
)


def build_messages(question: str) -> list[dict[str, str]]:
    """Return the system and user chat messages that open an episode for ``question``."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_PREFIX + question},
    ]


# Shell episodes (phantom_envs.shell). As in a Harbor task's instruction.md, the user message holds the
# whole task; the chat template adds the bash tool's description to the system message.
SHELL_SYSTEM_PROMPT: str = "You are a helpful assistant."

SHELL_INSTRUCTION: str = (
    "Answer the question below using the documents in /corpus, one file per document. "
    "Search and read them with the bash tool, and base your answer only on what they say.\n\n"
    "Write your final answer to /app/output.md as a single line, with multiple answers separated by "
    "commas (for example: Alice, Bob). Only /app/output.md is graded.\n\n"
    "Question: {question}"
)


def build_shell_messages(question: str) -> list[dict[str, str]]:
    """Return the system and user chat messages that open a shell episode for ``question``."""
    return [
        {"role": "system", "content": SHELL_SYSTEM_PROMPT},
        {"role": "user", "content": SHELL_INSTRUCTION.format(question=question)},
    ]
