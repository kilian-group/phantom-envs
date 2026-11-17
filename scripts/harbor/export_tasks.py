#!/usr/bin/env -S uv run python
"""Export a data dir's shell questions as Harbor tasks, to run Harbor's coding agents on them.

The tasks match the ShellSim episodes of ``launch_skyrl.sh eval shell``: the same instruction (the
episode's user message), the corpus as the same files under ``/corpus``, and the same verifier (F1 of
``/app/output.md``). Writes the corpus once and one task dir per question:

    <data_dir>/shell/corpus/<doc>.md                  mounted read-only at /corpus by harbor run
    <data_dir>/harbor/<data_source>-<index>/         (environment/ is empty: the image is prebuilt)
        instruction.md
        task.toml                                     the agent may reach only the model API host
        tests/{test.sh, score.py, scoring.py, gold.json}   F1 -> /logs/verifier/reward.txt

Every task uses the image of ``scripts/harbor/shell-task.Dockerfile``; build it once first.

Usage:
    uv run python scripts/harbor/export_tasks.py --data_dir data/frames
    docker build -t phantom-envs-shell-task -f scripts/harbor/shell-task.Dockerfile scripts/harbor
    OPENAI_API_KEY=... OPENAI_BASE_URL=https://openrouter.ai/api/v1 harbor run -p data/frames/harbor \\
        -e docker -a qwen-coder -m openrouter/qwen/qwen4-max -n 32 \\
        --mounts '[{"type": "bind", "source": "<abs data_dir>/shell/corpus", "target": "/corpus", "read_only": true}]'
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

import phantom_envs.scoring
from phantom_envs.shell import ANSWER_PATH, corpus_files

IMAGE = "phantom-envs-shell-task"

TASK_TOML = """\
schema_version = "1.4"

[metadata]
data_source = {data_source}
index = {index}

[agent]
timeout_sec = {agent_timeout}
# The model API only: no web lookups of the answer
network_mode = "allowlist"
allowed_hosts = {allowed_hosts}

[verifier]
timeout_sec = 120.0
network_mode = "no-network"

[environment]
docker_image = "{image}"
workdir = "/app"
"""

TEST_SH = """\
#!/bin/bash
# F1 of the agent's answer file against the gold answers, as the task's reward
python3 /tests/score.py
"""

SCORE_PY = f'''\
"""Write the F1 of {ANSWER_PATH} against /tests/gold.json to /logs/verifier/reward.txt (no file: 0)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tests")
from scoring import score  # noqa: E402

gold = json.loads(Path("/tests/gold.json").read_text())
answer_file = Path("{ANSWER_PATH}")
answer = answer_file.read_bytes().decode(errors="replace").strip() if answer_file.exists() else ""
f1 = score(answer, gold["answers"], gold["data_source"], "f1") if answer else 0.0
Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
Path("/logs/verifier/reward.txt").write_text(f"{{f1}}\\n")
print(f"answer: {{answer!r}}, gold: {{gold['answers']}}, F1: {{f1}}")
'''


def write_corpus(corpus_path: Path, out_dir: Path) -> int:
    """Write the corpus as one file per document into ``out_dir``; return the number of documents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, text in corpus_files(corpus_path):
        (out_dir / name).write_text(text)
        n += 1
    return n


def write_task(task_dir: Path, row: dict, allowed_hosts: list[str], agent_timeout: float) -> None:
    """Write one question's Harbor task dir."""
    tests = task_dir / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    # Harbor takes a dir as a task only with an environment/ dir, though the image is prebuilt
    (task_dir / "environment").mkdir(exist_ok=True)
    instruction = next(m["content"] for m in row["prompt"] if m["role"] == "user")
    (task_dir / "instruction.md").write_text(instruction + "\n")
    (task_dir / "task.toml").write_text(
        TASK_TOML.format(
            data_source=json.dumps(row["data_source"]),
            index=row["extra_info"]["index"],
            agent_timeout=float(agent_timeout),
            allowed_hosts=json.dumps(allowed_hosts),
            image=IMAGE,
        )
    )
    (tests / "test.sh").write_text(TEST_SH)
    (tests / "test.sh").chmod(0o755)
    (tests / "score.py").write_text(SCORE_PY)
    shutil.copy(phantom_envs.scoring.__file__, tests / "scoring.py")
    gold = {"answers": list(row["reward_spec"]["ground_truth"]["target"]), "data_source": row["data_source"]}
    (tests / "gold.json").write_text(json.dumps(gold) + "\n")


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    corpus_dir = data_dir / "shell" / "corpus"
    if corpus_dir.exists():
        print(f"{corpus_dir} exists; keeping it")
    else:
        print(f"wrote {write_corpus(data_dir / 'corpus.jsonl', corpus_dir)} documents to {corpus_dir}")

    rows = pq.read_table(data_dir / "shell" / "validation.parquet").to_pylist()
    out = data_dir / "harbor"
    for row in rows:
        name = f"{row['data_source']}-{row['extra_info']['index']:04d}"
        write_task(out / name, row, args.allowed_hosts, args.agent_timeout)
    print(f"wrote {len(rows)} tasks to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data_dir", required=True, help="Data dir with corpus.jsonl and shell/validation.parquet"
    )
    parser.add_argument(
        "--allowed_hosts",
        nargs="+",
        default=["openrouter.ai"],
        help="Hosts the agent may reach (the model API)",
    )
    parser.add_argument(
        "--agent_timeout", type=float, default=1800.0, help="Agent time limit per task, in seconds"
    )
    main(parser.parse_args())
