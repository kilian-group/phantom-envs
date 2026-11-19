"""Eval dumps of terminal episodes run outside SkyRL, e.g. by scripts/tinker/eval.py and scripts/eval_api.py.

An eval writes one JSON row per episode, then the averages over all of them:

    <out_dir>/model=<name>/{<data_source>.jsonl, aggregated_results.jsonl}

Episodes already in the dump are skipped, so rerunning an interrupted eval finishes it.

Usage:

    dump = EpisodeDump(out_dir, name, tasks[0].data_source)
    for task in dump.todo(tasks):
        dump.add(episode_row(task, answer, scores, stop_reason, num_turns, num_commands) | extra_fields)
    dump.write_results(aggregate(dump.rows))
"""

import json
from pathlib import Path
from typing import Any

from phantom_envs._types import Task
from phantom_envs.scoring import METRICS
from phantom_envs.terminal.task import has_answer


class EpisodeDump:
    """An eval's episode rows on disk: those already done, and new ones as they finish."""

    def __init__(self, out_dir: str | Path, name: str, data_source: str) -> None:
        """Open the dump, reading the rows it has.

        Args:
            out_dir: The eval's output dir, e.g. out__eval=hp500__pi.
            name: The model's dir name.
            data_source: The questions' data source, which names the dump file.
        """
        self.dir = Path(out_dir) / f"model={name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{data_source}.jsonl"
        self.rows: list[dict[str, Any]] = []
        if self.path.exists():
            self.rows = [json.loads(line) for line in self.path.open()]

    def todo(self, tasks: list[Task]) -> list[Task]:
        """Return the tasks that have no row yet."""
        done = {row["id"] for row in self.rows}
        return [task for task in tasks if task.id not in done]

    def add(self, row: dict[str, Any]) -> None:
        """Append an episode's row."""
        self.rows.append(row)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def write_results(self, results: dict[str, float]) -> None:
        """Write the eval's aggregated results next to the dump."""
        (self.dir / "aggregated_results.jsonl").write_text(json.dumps(results) + "\n")


def episode_row(
    task: Task,
    answer: str | None,
    scores: dict[str, float],
    stop_reason: str,
    num_turns: int,
    num_commands: int,
) -> dict[str, Any]:
    """Return the fields every eval's row has; an eval adds its own (tokens, the conversation).

    Args:
        task: The question.
        answer: The answer file's contents.
        scores: The answer's score per metric (0 for an episode that wasn't scored).
        stop_reason: Why the episode ended.
        num_turns: Model calls the agent made.
        num_commands: Tool calls the agent made.

    Returns:
        The row.
    """
    return {
        "id": task.id,
        "data_source": task.data_source,
        "question": task.question,
        "gold": task.answers,
        "answer": answer,
        "score": scores,
        "stop_reason": stop_reason,
        "num_turns": num_turns,
        "num_commands": num_commands,
        # Separates learning the output format from learning to search
        "answered": float(has_answer(answer)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average the rows' scores and counts, and give the share of each stop reason."""

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    results: dict[str, float] = {"num_episodes": len(rows)}
    results |= {metric: mean([row["score"][metric] for row in rows]) for metric in METRICS}
    results |= {key: mean([row[key] for row in rows]) for key in ("answered", "num_turns", "num_commands")}
    for stop in sorted({row["stop_reason"] for row in rows}):
        results[f"stop/{stop}"] = mean([float(row["stop_reason"] == stop) for row in rows])
    return results
