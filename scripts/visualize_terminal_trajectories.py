#!/usr/bin/env -S uv run python
"""Browse terminal-agent trajectories in the browser, as a chat.

Reads the trajectory dumps of terminal evals and trainings: one episode per line, in which an agent
works on a question with tools such as bash and search. The page shows an episode like a messaging
app: the task on the right, the agent's turns on the left, each tool call as its own bubble (search
and bash in different colors) and each tool output across from it. A strip of colored squares above
the chat shows the order of the episode's tool calls at a glance.

Pick a trajectory file in the dropdown, then an episode in the list; filter the list by score, by
tool use or by text.

Two dump formats are read, whatever the dataset:

- SkyRL's dumps (dumped_evals/.../<data_source>.jsonl): the episode as one rendered chat, in the
  input_prompt and output_response fields.
- scripts/eval_api.py's dumps (<data_source>.jsonl): the episode as a list of chat messages.

Usage:
    # Serve every trajectory file under the given dirs, then open http://127.0.0.1:8765
    uv run python scripts/visualize_terminal_trajectories.py out__eval=*__shell_pi_search

    # On a remote machine, forward the port first: ssh -L 8765:127.0.0.1:8765 <host>

    # Or write one HTML file that holds the trajectories, to open or send without a server
    uv run python scripts/visualize_terminal_trajectories.py out__eval=hp500__shell_pi_search \\
        --export hp500_search.html

Gotcha: a rendered chat is split on its model's chat markers. Qwen's (<|im_start|>role ... <|im_end|>)
and Inkling's (<|message_role|> ... <|end_message|>) are known; an episode in another chat format
shows as one block of text.
"""

import argparse
import functools
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

# Files in a dump dir that hold metrics, not episodes
NOT_TRAJECTORIES: frozenset[str] = frozenset({"aggregated_results.jsonl"})

# One message of a rendered chat: <|im_start|>role, its text, and <|im_end|> unless generation was cut
MESSAGE_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|\Z)", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\n?(.*?)\n?</tool_response>", re.DOTALL)
# A tool call in Qwen3.5's form: <function=name><parameter=key>value</parameter></function>
FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)

# One message of a chat rendered for Inkling: <|message_role|>, a tool name if any, <|content_kind|>,
# its text, and <|end_message|>
INKLING_MESSAGE_RE = re.compile(
    r"<\|message_(\w+)\|>(.*?)<\|content_(\w+)\|>(.*?)(?:<\|end_message\|>|\Z)", re.DOTALL
)

# Stop reasons of an episode that the agent ended itself
STOPPED: tuple[str, ...] = ("stop", "completed")

Item = dict[str, Any]


def tool_kind(name: str) -> str:
    """Sort a tool into the kinds the page colors: search, bash or other.

    Args:
        name: The tool's name, e.g. bash or mcp__retriever__search.

    Returns:
        The kind.
    """
    if "search" in name.lower():
        return "search"
    return "bash" if name == "bash" else "other"


def parse_tool_call(body: str) -> tuple[str, dict[str, Any]]:
    """Read the name and arguments of one tool call of a rendered chat.

    Args:
        body: The text between <tool_call> and </tool_call>.

    Returns:
        The tool's name and its arguments. A call in neither known form (Qwen3.5's tags, or a JSON
        object with name and arguments) is named "unknown" and keeps its text as one argument.
    """
    function = FUNCTION_RE.search(body)
    if function:
        return function.group(1), dict(PARAMETER_RE.findall(function.group(2)))
    try:
        call = json.loads(body)
        return call["name"], dict(call.get("arguments") or {})
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return "unknown", {"text": body.strip()}


def call_item(name: str, arguments: dict[str, Any]) -> Item:
    """Build the chat item of a tool call."""
    return {"type": "call", "kind": tool_kind(name), "name": name, "args": arguments}


def assistant_items(text: str) -> list[Item]:
    """Split an assistant turn of a rendered chat into what the agent thought, said and called.

    Args:
        text: The turn's text.

    Returns:
        The turn's items in order: thinking, text and call.
    """
    items: list[Item] = []
    for thought in THINK_RE.findall(text):
        if thought.strip():
            items.append({"type": "thinking", "text": thought.strip()})
    text = THINK_RE.sub("", text)

    # Text and tool calls, in the order the agent wrote them
    position = 0
    for match in TOOL_CALL_RE.finditer(text):
        said = text[position : match.start()].strip()
        if said:
            items.append({"type": "assistant", "text": said})
        items.append(call_item(*parse_tool_call(match.group(1))))
        position = match.end()
    said = text[position:].strip()
    if said:
        items.append({"type": "assistant", "text": said})
    return items


def inkling_chat_items(chat: str) -> list[Item]:
    """Split a chat rendered for Inkling into the items the page shows.

    Args:
        chat: The whole episode as text.

    Returns:
        The items in order, as rendered_chat_items returns them.
    """
    items: list[Item] = []
    for role, name, kind, text in INKLING_MESSAGE_RE.findall(chat):
        name, text = name.strip(), text.strip()
        if kind == "invoke_tool_json":
            try:
                call = json.loads(text)
                items.append(call_item(call.get("name", name), dict(call.get("args") or {})))
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                items.append(call_item(name or "unknown", {"text": text}))
        elif role == "tool":
            items.append({"type": "result", "kind": tool_kind(name), "name": name or "tool", "text": text})
        elif kind == "thinking":
            items.append({"type": "thinking", "text": text})
        elif text:
            items.append({"type": role if role in ("system", "user") else "assistant", "text": text})
    return items


def rendered_chat_items(chat: str) -> list[Item]:
    """Split a rendered chat into the items the page shows.

    Args:
        chat: The whole episode as text: the prompt and everything after it.

    Returns:
        The items in order. Each has a type: system, user, thinking, assistant, call (with kind, name
        and args) or result (with kind and name of the tool that answered). A chat in an unknown
        format is one assistant item.
    """
    if "<|message_" in chat:
        return inkling_chat_items(chat)
    messages = MESSAGE_RE.findall(chat)
    if not messages:
        return [{"type": "assistant", "text": chat}]

    items: list[Item] = []
    # Tool calls still waiting for their output; outputs come back in the order of the calls
    waiting: list[str] = []
    for role, text in messages:
        if role == "assistant":
            turn = assistant_items(text)
            waiting = [item["name"] for item in turn if item["type"] == "call"]
            items += turn
        elif role in ("user", "tool") and TOOL_RESPONSE_RE.search(text):
            for output in TOOL_RESPONSE_RE.findall(text):
                name = waiting.pop(0) if waiting else "tool"
                items.append({"type": "result", "kind": tool_kind(name), "name": name, "text": output})
        elif text.strip():
            items.append({"type": role if role in ("system", "user") else "assistant", "text": text.strip()})
    return items


def message_text(content: Any) -> str:
    """Return a chat message's content as text, whether it is a string, a list of parts or missing."""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def message_items(messages: list[dict[str, Any]]) -> list[Item]:
    """Turn a list of chat messages (the OpenAI form) into the items the page shows.

    Args:
        messages: The episode's messages, with the assistant's tool_calls and the tool messages.

    Returns:
        The items in order, as rendered_chat_items returns them.
    """
    items: list[Item] = []
    names: dict[str, str] = {}  # tool name per call id, to label the outputs
    for message in messages:
        role, text = message.get("role"), message_text(message.get("content")).strip()
        if role == "assistant":
            thought = message_text(message.get("reasoning") or message.get("reasoning_content")).strip()
            if thought:
                items.append({"type": "thinking", "text": thought})
            if text:
                items.append({"type": "assistant", "text": text})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = function.get("name", "unknown")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {"arguments": function.get("arguments")}
                names[call.get("id", "")] = name
                items.append(
                    call_item(name, arguments if isinstance(arguments, dict) else {"arguments": arguments})
                )
        elif role == "tool":
            name = names.get(message.get("tool_call_id", ""), "tool")
            items.append({"type": "result", "kind": tool_kind(name), "name": name, "text": text})
        elif text:
            items.append({"type": role if role in ("system", "user") else "assistant", "text": text})
    return items


def scalar_score(score: Any) -> float | None:
    """Reduce a dump's score to one number.

    Args:
        score: A number; a dict of metrics (its f1 is used); or SkyRL's reward per token, which is zero
            except on the episode's last token.

    Returns:
        The score, or None if there is none.
    """
    if isinstance(score, dict):
        score = score.get("f1", next(iter(score.values()), None))
    if isinstance(score, list):
        score = max(score, default=None)
    return float(score) if isinstance(score, int | float) else None


def parse_episode(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one line of a dump into the episode the page shows.

    Args:
        row: The line, in either dump format.

    Returns:
        The episode: its question, gold answers, answer, score, stop reason, chat items, and the
        number of tool calls per kind.
    """
    extras = row.get("env_extras") or {}
    if "messages" in row:
        items = message_items(row["messages"])
    else:
        items = rendered_chat_items(row.get("input_prompt", "") + row.get("output_response", ""))

    counts = {"search": 0, "bash": 0, "other": 0}
    for item in items:
        if item["type"] == "call":
            counts[item["kind"]] += 1
    gold = (
        row.get("gold") or ((extras.get("reward_spec") or {}).get("ground_truth") or {}).get("target") or []
    )
    return {
        "question": row.get("question") or (extras.get("extra_info") or {}).get("question") or "",
        "gold": list(gold),
        "answer": row.get("answer") if "answer" in row else extras.get("answer"),
        "score": scalar_score(row.get("score")),
        "stop_reason": row.get("stop_reason") or "",
        "stopped": (row.get("stop_reason") or "") in STOPPED,
        "data_source": row.get("data_source") or "",
        "counts": counts,
        "items": items,
    }


@functools.lru_cache(maxsize=8)
def load_episodes(path: Path) -> list[dict[str, Any]]:
    """Read and normalize every episode of a trajectory file (the last few files stay in memory)."""
    with path.open() as f:
        return [parse_episode(json.loads(line)) for line in f if line.strip()]


def episode_index(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the list the page shows beside the chat: every episode without its chat items."""
    return [{key: value for key, value in episode.items() if key != "items"} for episode in episodes]


def is_trajectory_file(path: Path) -> bool:
    """Whether a .jsonl file is a trajectory dump: its first line holds an episode in a known format."""
    if path.name in NOT_TRAJECTORIES:
        return False
    try:
        with path.open() as f:
            row = json.loads(f.readline())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(row, dict) and ("messages" in row or "input_prompt" in row)


def find_trajectory_files(paths: list[str]) -> list[Path]:
    """Find the trajectory files under the given paths.

    Args:
        paths: Files, or dirs to search for .jsonl dumps.

    Returns:
        The trajectory files, sorted, each once.
    """
    found: set[Path] = set()
    for path in map(Path, paths):
        candidates = [path] if path.is_file() else path.rglob("*.jsonl")
        found.update(candidate.resolve() for candidate in candidates if is_trajectory_file(candidate))
    return sorted(found)


def file_labels(files: list[Path]) -> list[str]:
    """Name each file for the dropdown: its path below the files' common dir, without filler dirs."""
    if not files:
        return []
    common = Path(os.path.commonpath([file.parent for file in files]))
    labels = []
    for file in files:
        parts = [part for part in file.relative_to(common).parts if part not in ("dumped_evals", "eval_only")]
        labels.append(" / ".join(parts))
    return labels


def embedded_data(files: list[Path]) -> dict[str, Any]:
    """Return everything the page needs, to put inside an exported HTML file.

    Args:
        files: The trajectory files to export.

    Returns:
        The file labels, and per file its episode list and episodes.
    """
    episodes = [load_episodes(file) for file in files]
    return {
        "files": file_labels(files),
        "index": [episode_index(file_episodes) for file_episodes in episodes],
        "episodes": episodes,
    }


def render_page(embedded: dict[str, Any] | None = None) -> str:
    """Return the page's HTML.

    Args:
        embedded: The data of an exported page; None for a page that asks this script's server.

    Returns:
        The HTML, with no outside resources.
    """
    data = json.dumps(embedded)
    # Keep the data from ending the script tag or breaking a JavaScript string
    data = data.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return PAGE.replace("/*EMBEDDED*/null", data)


def make_handler(files: list[Path]) -> type[BaseHTTPRequestHandler]:
    """Build the request handler of the server over the given trajectory files.

    Routes:
        /                               the page
        /api/files                      the file labels
        /api/index?file=<i>             the episode list of file i
        /api/episode?file=<i>&episode=<j>   episode j of file i

    Args:
        files: The trajectory files; requests name them by position, never by path.

    Returns:
        The handler class.
    """
    labels = file_labels(files)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            url = urlsplit(self.path)
            query = {key: values[0] for key, values in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    return self.reply(render_page().encode(), "text/html; charset=utf-8")
                if url.path == "/api/files":
                    return self.reply_json(labels)
                episodes = load_episodes(files[int(query["file"])])
                if url.path == "/api/index":
                    return self.reply_json(episode_index(episodes))
                if url.path == "/api/episode":
                    return self.reply_json(episodes[int(query["episode"])])
            except (KeyError, ValueError, IndexError):
                return self.send_error(400, "Bad request")
            self.send_error(404, "Not found")

        def reply_json(self, value: Any) -> None:
            self.reply(json.dumps(value).encode(), "application/json")

        def reply(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (http.server's name)
            pass  # no line per request

    return Handler


def main(args: argparse.Namespace) -> None:
    """Find the trajectory files, then export them or serve them."""
    files = find_trajectory_files(args.paths)
    if not files:
        raise SystemExit(f"No trajectory files under {args.paths}")
    print(f"Found {len(files)} trajectory file(s)")

    if args.export:
        Path(args.export).write_text(render_page(embedded_data(files)))
        print(f"Wrote {args.export} ({Path(args.export).stat().st_size / 1e6:.1f} MB)")
        return

    server = ThreadingHTTPServer((args.host, args.port), make_handler(files))
    print(f"Serving at http://{args.host}:{args.port} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# The page: its styles and script are inline, so it needs nothing but this file's server or, when
# exported, nothing at all. /*EMBEDDED*/null is replaced by the data of an exported page.
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Terminal trajectories</title>
<style>
  :root {
    --bg: #ffffff; --panel: #f5f5f7; --line: #dcdce0; --text: #1c1c1e; --muted: #6e6e73;
    --user: #0b84ff; --user-text: #ffffff; --assistant: #e9e9eb; --assistant-text: #1c1c1e;
    --search: #0f8a7e; --search-soft: #dff3f0; --bash: #1f2430; --bash-soft: #eceef2; --bash-text: #d6f5c8;
    --other: #b26a00; --other-soft: #fbefd9; --good: #1e8e3e; --mid: #b26a00; --bad: #c5221f;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #111113; --panel: #1b1b1e; --line: #323236; --text: #f2f2f4; --muted: #9a9aa1;
      --assistant: #2c2c30; --assistant-text: #f2f2f4; --search: #2bb3a4; --search-soft: #12302d;
      --bash: #0c0e13; --bash-soft: #23262d; --other: #e0a040; --other-soft: #33280f;
      --good: #5bd17d; --mid: #e0a040; --bad: #ff7b72;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column;
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }
  header { display: flex; gap: 12px; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line);
    background: var(--panel); flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; white-space: nowrap; }
  select, input { font: inherit; color: var(--text); background: var(--bg); border: 1px solid var(--line);
    border-radius: 8px; padding: 6px 8px; }
  #file { flex: 1 1 380px; min-width: 240px; }
  #summary { color: var(--muted); font-size: 13px; }
  main { flex: 1; display: flex; min-height: 0; }
  aside { width: 340px; flex: none; border-right: 1px solid var(--line); display: flex; flex-direction: column; }
  .filters { display: flex; gap: 8px; padding: 10px; border-bottom: 1px solid var(--line); }
  .filters input { flex: 1; min-width: 0; }
  #shown { color: var(--muted); font-size: 12px; padding: 6px 12px; }
  #episodes { overflow-y: auto; flex: 1; }
  .episode { padding: 9px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }
  .episode:hover { background: var(--panel); }
  .episode.selected { background: var(--panel); box-shadow: inset 3px 0 0 var(--user); }
  .episode .question { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .meta { display: flex; gap: 8px; align-items: center; color: var(--muted); font-size: 12px; margin-top: 4px; flex-wrap: wrap; }
  .chip { border-radius: 999px; padding: 1px 8px; font-size: 12px; font-weight: 600; color: #fff; }
  .chip.good { background: var(--good); } .chip.mid { background: var(--mid); } .chip.bad { background: var(--bad); }
  .chip.none { background: var(--muted); }
  .count.search { color: var(--search); font-weight: 600; } .count.bash { font-weight: 600; }
  #chat { flex: 1; overflow-y: auto; padding: 18px 22px 60px; min-width: 0; }
  #chat .empty { color: var(--muted); margin-top: 20vh; text-align: center; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; margin-bottom: 14px; }
  .card h2 { font-size: 15px; margin: 0 0 8px; font-weight: 600; }
  .facts { display: grid; grid-template-columns: max-content 1fr; gap: 3px 12px; font-size: 13px; }
  .facts dt { color: var(--muted); } .facts dd { margin: 0; white-space: pre-wrap; word-break: break-word; }
  .timeline { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 10px; align-items: center; }
  .timeline .label { color: var(--muted); font-size: 12px; margin-right: 6px; }
  .tick { width: 16px; height: 16px; border-radius: 4px; cursor: pointer; border: 0; padding: 0; }
  .tick.search { background: var(--search); } .tick.bash { background: var(--bash); outline: 1px solid var(--line); }
  .tick.other { background: var(--other); }
  .legend { display: flex; gap: 14px; color: var(--muted); font-size: 12px; margin-top: 8px; }
  .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px;
    margin-right: 5px; vertical-align: -1px; background: var(--swatch); }
  .row { display: flex; flex-direction: column; margin: 8px 0; align-items: flex-start; }
  .row.right { align-items: flex-end; }
  .who { font-size: 11px; color: var(--muted); margin: 0 10px 2px; }
  .bubble { max-width: min(78%, 900px); padding: 8px 13px; border-radius: 18px; white-space: pre-wrap;
    word-break: break-word; background: var(--assistant); color: var(--assistant-text); }
  .row.user .bubble { background: var(--user); color: var(--user-text); border-bottom-right-radius: 5px; }
  .row.assistant .bubble { border-bottom-left-radius: 5px; }
  .row.thinking .bubble { background: transparent; border: 1px dashed var(--line); color: var(--muted); font-style: italic; }
  .row.call .bubble { border-bottom-left-radius: 5px; }
  .row.call.search .bubble { background: var(--search); color: #fff; }
  .row.call.bash .bubble { background: var(--bash); color: var(--bash-text); }
  .row.call.other .bubble { background: var(--other); color: #fff; }
  .row.result .bubble { border-bottom-right-radius: 5px; border: 1px solid var(--line); }
  .row.result.search .bubble { background: var(--search-soft); color: var(--text); }
  .row.result.bash .bubble { background: var(--bash-soft); color: var(--text); }
  .row.result.other .bubble { background: var(--other-soft); color: var(--text); }
  .row.system { align-items: center; }
  .row.system .bubble { background: transparent; border: 1px solid var(--line); color: var(--muted); font-size: 12px;
    max-width: 100%; border-radius: 12px; }
  .clipped { max-height: 190px; overflow: hidden; -webkit-mask-image: linear-gradient(#000 70%, transparent);
    mask-image: linear-gradient(#000 70%, transparent); }
  .more { font: inherit; font-size: 12px; color: var(--user); background: none; border: 0; cursor: pointer; padding: 2px 10px; }
  .flash { animation: flash 1.2s ease-out; }
  @keyframes flash { from { box-shadow: 0 0 0 4px var(--user); } to { box-shadow: 0 0 0 4px transparent; } }
  @media (max-width: 800px) { main { flex-direction: column; } aside { width: auto; max-height: 38vh; border-right: 0;
    border-bottom: 1px solid var(--line); } .bubble { max-width: 92%; } }
</style>
</head>
<body>
<header>
  <h1>Terminal trajectories</h1>
  <select id="file" aria-label="Trajectory file"></select>
  <span id="summary"></span>
</header>
<main>
  <aside>
    <div class="filters">
      <input id="text" type="search" placeholder="Filter by question or answer">
      <select id="filter" aria-label="Filter episodes">
        <option value="all">All</option>
        <option value="right">Score 1</option>
        <option value="partial">Score between 0 and 1</option>
        <option value="wrong">Score 0</option>
        <option value="nosearch">No search call</option>
        <option value="unanswered">No answer written</option>
        <option value="cut">Did not stop on its own</option>
      </select>
    </div>
    <div id="shown"></div>
    <div id="episodes"></div>
  </aside>
  <section id="chat"><p class="empty">Pick a trajectory file, then an episode.</p></section>
</main>
<script>
const EMBEDDED = /*EMBEDDED*/null;
const $ = (id) => document.getElementById(id);
const state = { file: 0, index: [], shown: [], episode: null };

// The data comes from the page itself when it was exported, else from the server
async function load(kind, params = {}) {
  if (EMBEDDED) {
    if (kind === "files") return EMBEDDED.files;
    if (kind === "index") return EMBEDDED.index[params.file];
    return EMBEDDED.episodes[params.file][params.episode];
  }
  const response = await fetch(`/api/${kind}?${new URLSearchParams(params)}`);
  return response.json();
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scoreChip(score) {
  if (score === null || score === undefined) return element("span", "chip none", "no score");
  const level = score >= 1 ? "good" : score > 0 ? "mid" : "bad";
  return element("span", `chip ${level}`, score.toFixed(2));
}

const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);
const answered = (episode) => Boolean((episode.answer || "").trim());

const FILTERS = {
  all: () => true,
  right: (e) => e.score >= 1,
  partial: (e) => e.score > 0 && e.score < 1,
  wrong: (e) => e.score === 0,
  nosearch: (e) => e.counts.search === 0,
  unanswered: (e) => !answered(e),
  cut: (e) => !e.stopped,
};

async function selectFile(file, episode = null) {
  state.file = file;
  $("file").value = file;
  state.index = await load("index", { file });
  const scores = state.index.map((e) => e.score).filter((s) => s !== null);
  $("summary").textContent =
    `${state.index.length} episodes · mean score ${mean(scores).toFixed(3)} · ` +
    `${mean(state.index.map((e) => e.counts.search)).toFixed(2)} search and ` +
    `${mean(state.index.map((e) => e.counts.bash)).toFixed(2)} bash calls per episode`;
  renderList();
  if (episode !== null && episode < state.index.length) selectEpisode(episode);
  else $("chat").replaceChildren(element("p", "empty", "Pick an episode."));
}

function renderList() {
  const text = $("text").value.toLowerCase();
  const keep = FILTERS[$("filter").value];
  state.shown = state.index
    .map((episode, i) => ({ episode, i }))
    .filter(({ episode }) => keep(episode))
    .filter(({ episode }) => !text || `${episode.question} ${episode.answer || ""}`.toLowerCase().includes(text));
  $("shown").textContent = `Showing ${state.shown.length} of ${state.index.length}`;
  $("episodes").replaceChildren(
    ...state.shown.map(({ episode, i }) => {
      const row = element("div", "episode" + (i === state.episode ? " selected" : ""));
      row.dataset.i = i;
      row.append(element("div", "question", `${i + 1}. ${episode.question || "(no question recorded)"}`));
      const meta = element("div", "meta");
      meta.append(
        scoreChip(episode.score),
        element("span", "count search", `${episode.counts.search} search`),
        element("span", "count bash", `${episode.counts.bash} bash`),
      );
      if (episode.counts.other) meta.append(element("span", "count", `${episode.counts.other} other`));
      if (!episode.stopped) meta.append(element("span", "", episode.stop_reason || "no stop reason"));
      row.append(meta);
      row.onclick = () => selectEpisode(i);
      return row;
    }),
  );
}

// A long bubble shows its start, with a button for the rest
function bubble(text, mono) {
  const node = element("div", "bubble" + (mono ? " mono" : ""), text);
  const lines = text.split("\n").length;
  if (lines <= 12 && text.length <= 1200) return [node];
  node.classList.add("clipped");
  const more = element("button", "more", `Show all (${lines} lines)`);
  more.onclick = () => {
    const clipped = node.classList.toggle("clipped");
    more.textContent = clipped ? `Show all (${lines} lines)` : "Show less";
  };
  return [node, more];
}

function callText(item) {
  if (item.kind === "bash" && "command" in item.args) return `$ ${item.args.command}`;
  if (item.kind === "search" && "query" in item.args) return item.args.query;
  return JSON.stringify(item.args, null, 2);
}

function renderItem(item, callNumber) {
  if (item.type === "system") {
    const row = element("div", "row system");
    const pill = element("button", "more", `System prompt (${item.text.length.toLocaleString()} characters) · show`);
    const body = element("div", "bubble mono", item.text);
    body.hidden = true;
    pill.onclick = () => { body.hidden = !body.hidden; };
    row.append(pill, body);
    return row;
  }
  const side = item.type === "user" || item.type === "result" ? " right" : "";
  const row = element("div", `row ${item.type} ${item.kind || ""}${side}`);
  const who = {
    user: "Task",
    thinking: "Agent · thinking",
    assistant: "Agent",
    call: `Agent calls ${item.name} · call ${callNumber}`,
    result: `${item.name} returns`,
  }[item.type];
  row.append(element("div", "who", who));
  const text = item.type === "call" ? callText(item) : item.text || "(empty)";
  row.append(...bubble(text, item.type === "call" || item.type === "result"));
  return row;
}

async function selectEpisode(i) {
  state.episode = i;
  location.hash = `file=${state.file}&episode=${i}`;
  for (const row of $("episodes").children) row.classList.toggle("selected", Number(row.dataset.i) === i);
  const episode = await load("episode", { file: state.file, episode: i });

  const card = element("div", "card");
  card.append(element("h2", "", `${i + 1}. ${episode.question || "(no question recorded)"}`));
  const facts = element("dl", "facts");
  const fact = (name, value) => facts.append(element("dt", "", name), element("dd", "", value));
  fact("Gold answer", episode.gold.join(" | ") || "(none recorded)");
  fact("Agent's answer", answered(episode) ? episode.answer.trim() : "(no answer written)");
  fact("Score", episode.score === null ? "(none)" : episode.score.toFixed(3));
  fact("Stopped", episode.stop_reason || "(not recorded)");
  fact("Tool calls", `${episode.counts.search} search · ${episode.counts.bash} bash` +
    (episode.counts.other ? ` · ${episode.counts.other} other` : ""));
  card.append(facts);

  // One square per tool call, in order: the shape of the episode at a glance
  const calls = episode.items.filter((item) => item.type === "call");
  const timeline = element("div", "timeline");
  timeline.append(element("span", "label", "Tool calls in order:"));
  const rows = [];
  let callNumber = 0;
  const chat = episode.items.map((item) => {
    const row = renderItem(item, item.type === "call" ? ++callNumber : null);
    if (item.type === "call") rows.push(row);
    return row;
  });
  calls.forEach((call, n) => {
    const tick = element("button", `tick ${call.kind}`);
    tick.title = `${n + 1}. ${call.name}: ${callText(call).slice(0, 300)}`;
    tick.onclick = () => {
      rows[n].scrollIntoView({ behavior: "smooth", block: "center" });
      const target = rows[n].querySelector(".bubble");
      target.classList.remove("flash");
      void target.offsetWidth;
      target.classList.add("flash");
    };
    timeline.append(tick);
  });
  if (calls.length) {
    const legend = element("div", "legend");
    for (const [kind, name] of [["search", "search"], ["bash", "bash"], ["other", "other tool"]]) {
      const entry = element("span", "", name);
      entry.style.setProperty("--swatch", `var(--${kind})`);
      legend.append(entry);
    }
    card.append(timeline, legend);
  }
  $("chat").replaceChildren(card, ...chat);
  $("chat").scrollTop = 0;
}

// Arrow keys or j / k move through the shown episodes
document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  const step = { ArrowDown: 1, j: 1, ArrowUp: -1, k: -1 }[event.key];
  if (!step || !state.shown.length) return;
  event.preventDefault();
  const position = state.shown.findIndex(({ i }) => i === state.episode);
  const next = state.shown[Math.max(0, Math.min(state.shown.length - 1, position + step))];
  selectEpisode(next.i);
  document.querySelector(`.episode[data-i="${next.i}"]`)?.scrollIntoView({ block: "nearest" });
});

async function start() {
  const files = await load("files");
  $("file").replaceChildren(...files.map((label, i) => new Option(label, i)));
  $("file").onchange = () => selectFile(Number($("file").value));
  $("text").oninput = renderList;
  $("filter").onchange = renderList;
  // A link to the page can name a file and an episode
  const link = new URLSearchParams(location.hash.slice(1));
  const file = Number(link.get("file") || 0);
  await selectFile(file < files.length ? file : 0, link.has("episode") ? Number(link.get("episode")) : null);
}
start();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Browse terminal-agent trajectories in the browser, as a chat."
    )
    parser.add_argument("paths", nargs="+", help="Trajectory files, or dirs to search for them.")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Address to serve on (default: this machine only)."
    )
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default: 8765).")
    parser.add_argument(
        "--export",
        metavar="FILE",
        help="Write one HTML file that holds the trajectories, instead of serving.",
    )
    main(parser.parse_args())
