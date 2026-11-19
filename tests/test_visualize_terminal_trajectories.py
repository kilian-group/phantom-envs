"""Tests for scripts/visualize_terminal_trajectories.py: dump lines to the chat items the page shows."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from visualize_terminal_trajectories import (  # noqa: E402
    file_labels,
    find_trajectory_files,
    parse_episode,
    render_page,
    scalar_score,
)

QWEN_PROMPT = (
    "<|im_start|>system\n# Tools\n...<|im_end|>\n"
    "<|im_start|>user\nQuestion: Who is Ann's mother?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
QWEN_RESPONSE = (
    "Let me search.\n\n<tool_call>\n<function=mcp__retriever__search>\n<parameter=query>\nAnn Lee mother\n"
    "</parameter>\n</function>\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\nDoc 1:\nContents: The mother of Ann Lee is Bea Lee.\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n<think>\nFound it.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\n"
    "echo 'Bea Lee' > /app/output.md\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\n(no output)\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\nDone.<|im_end|>"
)
SKYRL_ROW = {
    "input_prompt": QWEN_PROMPT,
    "output_response": QWEN_RESPONSE,
    "score": 1.0,
    "stop_reason": "stop",
    "data_source": "pw",
    "env_extras": {
        "reward_spec": {"ground_truth": {"target": ["Bea Lee"]}},
        "extra_info": {"question": "Who is Ann's mother?"},
        "answer": "Bea Lee\n",
    },
}


def shape(items: list[dict]) -> list[tuple]:
    """Each item's type, tool kind and tool name, to compare chats briefly."""
    return [(item["type"], item.get("kind"), item.get("name")) for item in items]


def test_a_rendered_qwen_chat_becomes_ordered_items():
    episode = parse_episode(SKYRL_ROW)
    assert shape(episode["items"]) == [
        ("system", None, None),
        ("user", None, None),
        ("assistant", None, None),
        ("call", "search", "mcp__retriever__search"),
        ("result", "search", "mcp__retriever__search"),
        ("thinking", None, None),
        ("call", "bash", "bash"),
        ("result", "bash", "bash"),
        ("assistant", None, None),
    ]
    call, result = episode["items"][3], episode["items"][4]
    assert call["args"] == {"query": "Ann Lee mother"}
    assert result["text"] == "Doc 1:\nContents: The mother of Ann Lee is Bea Lee."
    assert episode["counts"] == {"search": 1, "bash": 1, "other": 0}
    assert (episode["question"], episode["gold"], episode["answer"]) == (
        "Who is Ann's mother?",
        ["Bea Lee"],
        "Bea Lee\n",
    )
    assert (episode["score"], episode["stopped"]) == (1.0, True)


def test_chat_messages_become_the_same_items():
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls /corpus"}'},
    }
    row = {
        "question": "Who?",
        "gold": ["Bea Lee"],
        "answer": None,
        "score": {"em": 0.0, "subem": 0.0, "f1": 0.5},
        "stop_reason": "length",
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Who?"},
            {"role": "assistant", "content": None, "reasoning": "Look first.", "tool_calls": [call]},
            {"role": "tool", "content": "Ann_Lee.md\n", "tool_call_id": "c1"},
        ],
    }
    episode = parse_episode(row)
    assert shape(episode["items"]) == [
        ("system", None, None),
        ("user", None, None),
        ("thinking", None, None),
        ("call", "bash", "bash"),
        ("result", "bash", "bash"),
    ]
    assert episode["items"][3]["args"] == {"command": "ls /corpus"}
    assert (episode["score"], episode["stopped"], episode["answer"]) == (0.5, False, None)


def test_a_rendered_inkling_chat_is_read_too():
    chat = (
        "<|message_system|><|content_text|>You are an agent.<|end_message|>"
        "<|message_user|><|content_text|>Who?<|end_message|>"
        "<|message_model|><|content_thinking|>Look first.<|end_message|>"
        '<|message_model|>bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"ls"}}<|end_message|>'
        "<|content_model_end_sampling|><|message_tool|>bash<|content_text|>Ann_Lee.md<|end_message|>"
    )
    episode = parse_episode({"input_prompt": chat, "output_response": "", "stop_reason": "completed"})
    assert shape(episode["items"])[2:] == [
        ("thinking", None, None),
        ("call", "bash", "bash"),
        ("result", "bash", "bash"),
    ]
    assert episode["items"][3]["args"] == {"command": "ls"} and episode["stopped"]


def test_an_unknown_chat_format_is_shown_whole():
    episode = parse_episode({"input_prompt": "[INST] Who? [/INST]", "output_response": " Bea Lee"})
    assert episode["items"] == [{"type": "assistant", "text": "[INST] Who? [/INST] Bea Lee"}]


def test_scores_reduce_to_one_number():
    assert scalar_score(0.5) == 0.5
    assert scalar_score({"em": 0.0, "f1": 0.75}) == 0.75
    assert scalar_score([0.0, 0.0, 1.0]) == 1.0  # SkyRL's reward per token
    assert scalar_score(None) is None


def test_trajectory_files_are_found_without_the_metrics_file(tmp_path: Path):
    dump_dir = tmp_path / "out__eval=pw" / "model=m" / "dumped_evals" / "eval_only"
    dump_dir.mkdir(parents=True)
    (dump_dir / "pw.jsonl").write_text(json.dumps(SKYRL_ROW) + "\n")
    (dump_dir / "aggregated_results.jsonl").write_text('{"eval/all/avg_score": 1.0}\n')
    (tmp_path / "notes.jsonl").write_text('{"not": "an episode"}\n')
    files = find_trajectory_files([str(tmp_path)])
    assert [file.name for file in files] == ["pw.jsonl"]
    assert file_labels(files) == ["pw.jsonl"]


def test_an_exported_page_holds_its_data_safely():
    page = render_page({"files": ["a </script> b"], "index": [[]], "episodes": [[]]})
    assert "/*EMBEDDED*/null" not in page and "a <\\/script> b" in page
    assert "/*EMBEDDED*/null" in render_page().replace("const EMBEDDED = null", "/*EMBEDDED*/null")
