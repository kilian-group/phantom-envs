"""Search episodes: the agent queries a retriever with `<search>` tags and ends with `<answer>`.

- `prompts`: the messages that open an episode.
- `protocol`: the episode logic (turn parsing, termination, reward), with no I/O.

The retriever behind the `<search>` tool is `phantom_envs.retriever`.
"""
