"""Search episodes: the agent queries a retriever with ``<search>`` tags and ends with ``<answer>``.

- :mod:`phantom_envs.search.prompts`: the messages that open an episode.
- :mod:`phantom_envs.search.protocol`: the episode logic (turn parsing, termination, reward), with no I/O.

The retriever behind the ``<search>`` tool is :mod:`phantom_envs.retriever`.
"""
