"""PhantomEnvironments — fictional-world environments for training LLM search agents."""

__version__ = "0.1.0"

# Loaded on first use, so trainer workers that only need the env code skip the generation modules
LAZY_ATTRIBUTES = {
    "EnvConfig": "phantom_envs._types",
    "SplitConfig": "phantom_envs._types",
    "Task": "phantom_envs._types",
    "PRESETS": "phantom_envs.environment",
    "PhantomEnv": "phantom_envs.environment",
    "generate": "phantom_envs.environment",
    "load_tasks": "phantom_envs.tasks",
}

__all__ = ["__version__", *LAZY_ATTRIBUTES]


def __getattr__(name: str):
    """Import the public API lazily (PEP 562)."""
    if name in LAZY_ATTRIBUTES:
        import importlib

        return getattr(importlib.import_module(LAZY_ATTRIBUTES[name]), name)
    raise AttributeError(f"module 'phantom_envs' has no attribute {name!r}")
