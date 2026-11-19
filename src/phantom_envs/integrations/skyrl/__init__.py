"""SkyRL integration: the `phantom_search` and `phantom_terminal` SkyRL-Gym environments and their
configs.

The episode logic lives in `phantom_envs.search` and `phantom_envs.terminal`; the modules here adapt it
to SkyRL:

- `search`: `phantom_search`, an agent with a retriever.
- `terminal`: `phantom_terminal`, an agent with a terminal (a bash tool) in a sandbox; SkyRL's
  generator samples each turn and the env runs its commands.
- `harness`: terminal episodes run by a coding-agent harness such as pi, as a SkyRL generator.

Ray workers import this package when they unpickle `PhantomTrainConfig` (the entrypoint's argument),
so the config of every env is composed here.

Usage (inside the Ray remote entrypoint):

    from phantom_envs.integrations.skyrl import register

    register()  # then dataset rows with env_class "phantom_search" use PhantomSearchEnv, and so on
"""

from dataclasses import dataclass, field

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import EnvironmentConfig, SkyRLGymConfig
from skyrl_gym.envs.registration import register as skyrl_gym_register
from skyrl_gym.envs.registration import registry

from phantom_envs.integrations.skyrl.search import PhantomSearchEnv, PhantomSearchEnvConfig
from phantom_envs.integrations.skyrl.terminal import PhantomTerminalEnv, PhantomTerminalEnvConfig

# SkyRL-Gym env ids, which dataset rows name as their env_class
SEARCH_ENV_ID: str = "phantom_search"
TERMINAL_ENV_ID: str = "phantom_terminal"

# Env class per SkyRL-Gym env id (a dataset row's env_class)
ENVS: dict[str, type] = {SEARCH_ENV_ID: PhantomSearchEnv, TERMINAL_ENV_ID: PhantomTerminalEnv}


@dataclass
class PhantomSkyRLGymConfig(SkyRLGymConfig):
    """SkyRLGymConfig with phantom_search and phantom_terminal fields.

    SkyRL looks an env's config up by its env_class.
    """

    phantom_search: PhantomSearchEnvConfig = field(default_factory=PhantomSearchEnvConfig)
    phantom_terminal: PhantomTerminalEnvConfig = field(default_factory=PhantomTerminalEnvConfig)


@dataclass
class PhantomEnvironmentConfig(EnvironmentConfig):
    """EnvironmentConfig that uses PhantomSkyRLGymConfig."""

    skyrl_gym: PhantomSkyRLGymConfig = field(default_factory=PhantomSkyRLGymConfig)


@dataclass
class PhantomTrainConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig whose environment section accepts `phantom_search` and `phantom_terminal`."""

    environment: PhantomEnvironmentConfig = field(default_factory=PhantomEnvironmentConfig)


def register() -> None:
    """Register the envs of `ENVS` with SkyRL-Gym (idempotent)."""
    for env_id, env_class in ENVS.items():
        if env_id not in registry:
            skyrl_gym_register(id=env_id, entry_point=f"{env_class.__module__}:{env_class.__name__}")
