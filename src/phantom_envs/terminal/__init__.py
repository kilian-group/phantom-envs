"""Terminal episodes: a coding agent answers a question with bash, over the corpus as files or through
a search tool.

As in Terminal-Bench, the environment, the agent's attempt and the verifier are separate:

- **Environment:** a sandbox that holds the data dir's `corpus.jsonl` as one file per document under
  `/corpus`, with `/app` as the working dir. With a search tool (`task.TOOLSETS`), the sandbox holds
  no corpus: the agent searches it through an MCP server on the host (`retriever.mcp_server`).
- **Attempt:** the agent runs bash commands (and searches) in the sandbox, turn by turn, and writes its
  answer to `/app/output.md`.
- **Verifier:** reads the answer file and scores it against the gold answers. No file scores 0.

Modules:

- `task`: the sandbox paths, the instruction and the verifier. Everything else builds on these.
- `corpus_files`: how a corpus becomes files.
- `sandbox`: the sandbox interface, with two backends: `shellsim` (an in-process simulated bash) and
  `bwrap` (real processes in a bubblewrap sandbox).
- `agent`: one setup of the agent (its harness, tools, sandbox and task), which every trainer and eval
  that runs a harness builds.

Two ways to run the agent's loop:

- `loop`: the bare tool loop, for trainers that sample the agent's turns themselves: tool-call
  parsing, when the attempt ends, and the tool responses.
- `harness`: a coding agent that runs its own loop inside the bwrap sandbox and calls the model over
  an OpenAI-compatible endpoint. `pi` is the only harness.

Nothing is imported here, so a module's optional dependencies load only when it is used.
"""
