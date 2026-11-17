"""SkyRL integration: the ``phantom_search`` and ``phantom_shell`` SkyRL-Gym environments and their
configs.

The episode logic lives in :mod:`phantom_envs.search` and :mod:`phantom_envs.terminal`; the modules here
adapt it to SkyRL:

- :mod:`phantom_envs.integrations.skyrl.search`: ``phantom_search``, an agent with a retriever.
- :mod:`phantom_envs.integrations.skyrl.terminal`: ``phantom_shell``, an agent with a terminal (a bash tool)
  in a sandbox; SkyRL's generator samples each turn and the env runs its commands.
- :mod:`phantom_envs.integrations.skyrl.harness`: shell episodes run by a coding-agent harness such
  as pi, as a SkyRL generator.

Ray workers import this package when they unpickle :class:`PhantomTrainConfig` (the entrypoint's
argument), so the config of every env is composed here.

Usage (inside the Ray remote entrypoint)::

    from phantom_envs.integrations.skyrl import register

    register()  # then dataset rows with env_class "phantom_search" use PhantomSearchEnv, and so on
"""

from dataclasses import dataclass, field

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import EnvironmentConfig, SkyRLGymConfig
from skyrl_gym.envs.registration import register as skyrl_gym_register
from skyrl_gym.envs.registration import registry

from phantom_envs.integrations.skyrl.search import PhantomSearchEnv, PhantomSearchEnvConfig
from phantom_envs.integrations.skyrl.terminal import PhantomShellEnv, PhantomShellEnvConfig

SEARCH_ENV_ID: str = "phantom_search"
SHELL_ENV_ID: str = "phantom_shell"

# Env class per SkyRL-Gym env id (a dataset row's env_class)
ENVS: dict[str, type] = {SEARCH_ENV_ID: PhantomSearchEnv, SHELL_ENV_ID: PhantomShellEnv}


@dataclass
class PhantomSkyRLGymConfig(SkyRLGymConfig):
    """SkyRLGymConfig with phantom_search and phantom_shell fields (SkyRL looks env configs up by
    env_class)."""

    phantom_search: PhantomSearchEnvConfig = field(default_factory=PhantomSearchEnvConfig)
    phantom_shell: PhantomShellEnvConfig = field(default_factory=PhantomShellEnvConfig)


@dataclass
class PhantomEnvironmentConfig(EnvironmentConfig):
    """EnvironmentConfig that uses PhantomSkyRLGymConfig."""

    skyrl_gym: PhantomSkyRLGymConfig = field(default_factory=PhantomSkyRLGymConfig)


@dataclass
class PhantomTrainConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig whose environment section accepts ``phantom_search`` and ``phantom_shell``."""

    environment: PhantomEnvironmentConfig = field(default_factory=PhantomEnvironmentConfig)


def register() -> None:
    """Register the envs of :data:`ENVS` with SkyRL-Gym (idempotent)."""
    for env_id, env_class in ENVS.items():
        if env_id not in registry:
            skyrl_gym_register(id=env_id, entry_point=f"{env_class.__module__}:{env_class.__name__}")
