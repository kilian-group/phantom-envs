"""The pi harness: `pi <https://pi.dev>`_, a minimal open-source coding agent, as the agent's loop.

pi runs inside the sandbox with one tool, ``bash``, and its own system prompt. It keeps its history
append-only (compaction is off), so every model call extends the previous one's tokens. The episode's
budgets match the bare tool loop's: a turn cap, a time limit per command, and a cap on the output
the model reads. pi has settings for some; ``pi_extension.js`` enforces the rest.

The sandbox image has pi installed (``scripts/sandbox/Dockerfile`` pins its version).

Gotchas:

- pi runs a turn's commands before the turn cap ends the episode; the bare loop doesn't run the
  commands of its last turn.
- pi cuts a long output to its last 50 KB before ``max_output_chars`` keeps the head of that.
"""

import json
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from phantom_envs.terminal.bwrap import SHARED_DIR, BwrapSandbox
from phantom_envs.terminal.harness import Harness, HarnessRun, ModelEndpoint, env_prefix, relay_command

EXTENSION: Path = Path(__file__).with_name("pi_extension.js")
# pi's config dir inside the sandbox, a subdir of the shared dir
CONFIG_DIR: str = "pi"
PROVIDER: str = "phantom"
# How skycap and vLLM word a prompt that doesn't fit the model's context
CONTEXT_ERRORS: tuple[str, ...] = ("context_length_exceeded", "max_model_len", "maximum context length")


@dataclass
class PiHarness(Harness):
    """pi with a bash tool, under the episode's budgets."""

    max_turns: int = field(default=20, metadata={"description": "Most turns the agent takes."})
    command_timeout: float = field(default=60.0, metadata={"description": "Seconds a command may run."})
    max_output_chars: int = field(
        default=8000, metadata={"description": "Tool outputs are cut to this many characters."}
    )
    episode_timeout: float = field(
        default=1800.0, metadata={"description": "Seconds a whole episode may take."}
    )
    system_prompt: str | None = field(
        default=None, metadata={"description": "Replaces pi's own system prompt; None keeps it."}
    )

    def run(
        self, sandbox: BwrapSandbox, instruction: str, endpoint_url: str, endpoint: ModelEndpoint
    ) -> HarnessRun:
        """Run pi on the task until the agent finishes or a budget ends the episode.

        Args:
            sandbox: The episode's sandbox.
            instruction: The task text, given to pi as its prompt.
            endpoint_url: The model API's base URL as the sandbox reaches it.
            endpoint: The model endpoint's settings.

        Returns:
            The turns and commands pi took, and why it stopped.
        """
        self.write_config(sandbox.shared_dir / CONFIG_DIR, endpoint_url, endpoint, sandbox.command_limits)
        command = relay_command(sandbox) + self.command(instruction, endpoint.model)
        # pi runs unlimited; its settings put the sandbox's limits on each command it starts
        output, exit_code, timed_out = sandbox.execute(command, timeout=self.episode_timeout, limited=False)
        run = parse_events(output)
        if timed_out:
            run.stop_reason = "timeout"
        elif run.turns == 0 and run.stop_reason == "stop":
            run.stop_reason = "error"
            run.error = run.error or f"pi exited with code {exit_code} before its first turn: {output[-500:]}"
        return run

    def write_config(
        self, config_dir: Path, endpoint_url: str, endpoint: ModelEndpoint, command_limits: str
    ) -> None:
        """Write pi's config dir: the model endpoint, its settings and the extension.

        Args:
            config_dir: The dir, inside the sandbox's shared dir.
            endpoint_url: The model API's base URL as the sandbox reaches it.
            endpoint: The model endpoint's settings.
            command_limits: The bash line that limits each command pi starts.
        """
        config_dir.mkdir(exist_ok=True)
        model = {
            "id": endpoint.model,
            "contextWindow": endpoint.context_window,
            "maxTokens": endpoint.max_tokens,
        }
        provider = {
            "baseUrl": endpoint_url,
            "api": "openai-completions",
            "apiKey": endpoint.api_key,
            "models": [model],
        }
        settings = {
            "compaction": {"enabled": False},
            "shellCommandPrefix": command_limits,
        }
        (config_dir / "models.json").write_text(json.dumps({"providers": {PROVIDER: provider}}))
        (config_dir / "settings.json").write_text(json.dumps(settings))
        shutil.copy(EXTENSION, config_dir / EXTENSION.name)

    def command(self, instruction: str, model: str) -> str:
        """Return the bash command that runs pi once on ``instruction`` and prints its events as JSONL."""
        config_dir = f"{SHARED_DIR}/{CONFIG_DIR}"
        env = {
            "PI_CODING_AGENT_DIR": config_dir,
            # No update checks, telemetry or model catalog refreshes: the sandbox has no network
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
            "PHANTOM_MAX_TURNS": str(self.max_turns),
            "PHANTOM_COMMAND_TIMEOUT": f"{self.command_timeout:g}",
            "PHANTOM_MAX_OUTPUT_CHARS": str(self.max_output_chars),
        }
        args = [
            "pi", "--print", "--mode", "json", "--no-session",
            "--provider", PROVIDER, "--model", model, "--tools", "bash",
            # Only what this module configures: nothing from the image or the working dir
            "--no-context-files", "--no-skills", "--no-prompt-templates", "--no-extensions",
            "--extension", f"{config_dir}/{EXTENSION.name}",
        ]  # fmt: skip
        if self.system_prompt is not None:
            args += ["--system-prompt", self.system_prompt]
        return f"{env_prefix(env)} {shlex.join([*args, instruction])}"


def parse_events(output: str) -> HarnessRun:
    """Summarize pi's JSONL event stream.

    Args:
        output: pi's stdout in ``--mode json``; lines that aren't JSON events are skipped.

    Returns:
        The turns and commands so far, and the stop reason of the last assistant message:
        ``context_length`` when the prompt outgrew the model's context, ``length`` when a turn hit
        its token limit, ``error`` on another failure, else ``stop``.
    """
    run = HarnessRun()
    last_message: dict = {}
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "turn_end":
            continue
        last_message = event.get("message") or {}
        # A turn that the turn cap aborted, or whose model call failed, is not a turn the agent took
        if last_message.get("stopReason") not in ("aborted", "error"):
            run.turns += 1
            run.commands += len(event.get("toolResults") or [])

    if last_message.get("stopReason") == "error":
        run.error = last_message.get("errorMessage") or "unknown error"
        overflowed = any(marker in run.error for marker in CONTEXT_ERRORS)
        run.stop_reason = "context_length" if overflowed else "error"
    elif last_message.get("stopReason") == "length":
        run.stop_reason = "length"
    return run
