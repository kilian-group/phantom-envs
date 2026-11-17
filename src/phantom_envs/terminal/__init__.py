"""Terminal episodes: a coding agent answers a question by running bash over the corpus as files.

As in Terminal-Bench, the environment, the agent's attempt and the verifier are separate:

- **Environment:** a sandbox that holds the data dir's ``corpus.jsonl`` as one file per document
  under ``/corpus``, with ``/app`` as the working dir.
- **Attempt:** the agent runs bash commands in the sandbox, turn by turn, and writes its answer to
  ``/app/output.md``.
- **Verifier:** reads the answer file and scores it against the gold answers. No file scores 0.

Modules:

- :mod:`phantom_envs.terminal.task`: the sandbox paths, the instruction and the verifier. Everything
  else builds on these.
- :mod:`phantom_envs.terminal.corpus_files`: how a corpus becomes files.
- :mod:`phantom_envs.terminal.sandbox`: the sandbox interface and :func:`~phantom_envs.terminal.sandbox.open_sandbox`,
  with two backends: :mod:`~phantom_envs.terminal.shellsim` (an in-process simulated bash) and
  :mod:`~phantom_envs.terminal.bwrap` (real processes in a bubblewrap sandbox).
- :mod:`phantom_envs.terminal.loop`: the bare tool loop, for trainers that sample the agent's turns
  themselves: tool-call parsing, when the attempt ends, and the tool responses.

Nothing is imported here, so a module's optional dependencies load only when it is used.
"""
