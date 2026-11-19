"""Tests for phantom_envs.integrations.skyrl.harness: captured episodes as SkyRL generator output."""

import pytest

pytest.importorskip("skyrl")  # the skyrl extra
pytest.importorskip("skycap")

from skycap import Sample  # noqa: E402
from skyrl.train.utils.trainer_utils import validate_generator_output  # noqa: E402

from phantom_envs.integrations.skyrl import register  # noqa: E402
from phantom_envs.integrations.skyrl.harness import Episode, to_generator_output, to_row  # noqa: E402

BASH_CALL = {"type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}


def captured(prompt_tokens: int = 3) -> Sample:
    """A two-turn episode as skycap captures it: a prompt, a tool call, its output, and a final reply."""
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [BASH_CALL]},
        {"role": "tool", "content": "a.md"},
        {"role": "assistant", "content": "Done."},
    ]
    turn_one, tool_output, turn_two = [11, 12], [21, 22, 23], [31]
    input_ids = list(range(prompt_tokens)) + turn_one + tool_output + turn_two
    loss_mask = [0] * prompt_tokens + [1, 1] + [0, 0, 0] + [1]
    logprobs = [0.0] * prompt_tokens + [-0.1, -0.2] + [0.0, 0.0, 0.0] + [-0.3]
    return Sample(
        leaf=3, path=[0, 1, 2, 3], messages=messages, targets=[1, 3],
        input_ids=input_ids, loss_mask=loss_mask, logprobs=logprobs,
    )  # fmt: skip


def test_row_splits_at_the_first_sampled_token():
    prompt, response, loss_mask, logprobs = to_row(Episode(stop_reason="stop", sample=captured()))
    assert prompt == [0, 1, 2]
    assert response == [11, 12, 21, 22, 23, 31]
    assert loss_mask == [1, 1, 0, 0, 0, 1]  # tool outputs don't train
    assert logprobs == [-0.1, -0.2, 0.0, 0.0, 0.0, -0.3]


@pytest.mark.parametrize("stop_reason", ["error", "timeout"])
def test_failed_episode_trains_nothing(stop_reason):
    _, response, loss_mask, _ = to_row(Episode(stop_reason=stop_reason, sample=captured()))
    assert len(response) == len(loss_mask) and not any(loss_mask)


def test_episode_without_tokens_keeps_its_row():
    prompt, response, loss_mask, logprobs = to_row(Episode(stop_reason="error"))
    assert len(prompt) == len(response) == len(loss_mask) == len(logprobs) == 1 and loss_mask == [0]


def test_episode_counts_turns_and_commands():
    episode = Episode(stop_reason="stop", sample=captured())
    assert (episode.turns, episode.commands) == (2, 1)
    assert (Episode(stop_reason="error").turns, Episode(stop_reason="error").commands) == (0, 0)


class TestGeneratorOutput:
    @pytest.fixture(scope="class", autouse=True)
    def registered(self) -> None:
        register()  # SkyRL aggregates env metrics per registered env class

    def episodes(self) -> list[Episode]:
        return [
            Episode(stop_reason="stop", reward=1.0, answer="Bea Lee\n", sample=captured(), seconds=2.0),
            Episode(stop_reason="context_length", answer=None, sample=captured(5), seconds=4.0),
            Episode(stop_reason="error", seconds=1.0),
        ]

    def test_one_valid_row_per_episode(self):
        output = to_generator_output(self.episodes(), ["phantom_terminal"] * 3)
        validate_generator_output(3, output, step_wise=False)
        assert output["rewards"] == [1.0, 0.0, 0.0]
        assert output["stop_reasons"] == ["stop", "length", "error"]
        assert [len(p) for p in output["prompt_token_ids"]] == [3, 5, 1]
        assert output["env_metrics"][0] == {"num_turns": 2, "num_commands": 1, "answered": 1.0}

    def test_metrics_count_what_went_wrong(self):
        metrics = to_generator_output(self.episodes(), ["phantom_terminal"] * 3)["rollout_metrics"]
        assert metrics["generate/harness/num_error_episodes"] == 1
        assert metrics["generate/harness/num_context_length_episodes"] == 1
        assert metrics["generate/harness/num_timeout_episodes"] == 0
        assert metrics["environment/answered"] == pytest.approx(1 / 3)

    def test_overlong_filtering_masks_unfinished_episodes(self):
        kept = to_generator_output(self.episodes(), ["phantom_terminal"] * 3)["loss_masks"]
        filtered = to_generator_output(self.episodes(), ["phantom_terminal"] * 3, overlong_filtering=True)
        assert any(kept[1]) and not any(filtered["loss_masks"][1])
        assert filtered["loss_masks"][0] == kept[0]
